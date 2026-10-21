###
# common.py
#
# Stages shared by all experiment types.
# Includes serving model on vLLM Ascend (benchmark_points),
# ingesting of profilier data (trace_analysis) including optional
# holistic trace analysis (hta), and visualization (figures).
#
# Dylan Everingham
# 16.09.2026
###

from __future__ import annotations

import asyncio
import gc
import json
import random
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Sequence

from moe_reliability_results import schema
from moe_reliability_results.metrics import summarize_requests, trace_scalars

from .. import eplb_gate, expert_counter
from ..config import ExperimentConfig
from ..environment import contention_warning, host_snapshot
from ..logs import log
from ..runs import RunContext, utcnow

__all__ = [
    "STAGE_BENCHMARK",
    "STAGE_TRACE_ANALYSIS",
    "STAGE_HTA",
    "STAGE_FIGURES",
    "seed_everything",
    "mmlu_prompts",
    "free_accelerator_memory",
    "should_run",
    "serve_and_measure",
    "benchmark_points",
    "post_processing_stages",
    "trace_analysis_stage",
    "npu_trace_views",
    "parse_npu_profiler_data",
    "hta_stage",
    "figures_stage",
]

STAGE_BENCHMARK = "benchmark"
STAGE_TRACE_ANALYSIS = "trace_analysis"
STAGE_HTA = "hta"
STAGE_FIGURES = "figures"

def seed_everything(seed: int) -> None:
    import torch

    torch.manual_seed(seed)


def mmlu_prompts(n_samples: int, seed: int) -> tuple[list, list, list]:
    from ..core.data import format_prompts_mmlu, get_data_mmlu

    dataset = get_data_mmlu(n_samples=n_samples, shuffle_seed=seed)
    return format_prompts_mmlu(dataset)


def free_accelerator_memory() -> None:
    import torch

    gc.collect()
    torch.npu.empty_cache()


def should_run(ctx: RunContext, stage: str, force: bool = False) -> bool:
    if force:
        return True
    if ctx.stage_status(stage) == schema.STATUS_COMPLETED:
        log(f"stage '{stage}' already completed - skipping")
        return False
    return True

def eplb_settings(cfg: ExperimentConfig, record_path: str | None = None) -> dict[str, Any] | None:
    """The vllm-ascend ``eplb_config`` block, or None when EPLB is off.

    Kept here rather than in the serving layer because the record path is a
    property of the run directory, which only the pipeline knows.
    """
    if not cfg.server.enable_eplb:
        return None
    settings: dict[str, Any] = {
        "dynamic_eplb": True,
        "eplb_policy_type": cfg.server.eplb_policy_type,
        "num_redundant_experts": cfg.server.eplb_num_redundant_experts,
        "expert_heat_collection_interval": cfg.server.eplb_heat_collection_interval,
        "algorithm_execution_interval": cfg.server.eplb_algorithm_execution_interval,
    }
    if record_path:
        settings["expert_map_record_path"] = record_path
    return settings


def eplb_cycle_iterations(cfg: ExperimentConfig, n_moe_layers: int = 0) -> int:
    """Forward iterations in one full collect-plan-apply cycle.

    A run that never reaches this many iterations never rearranges, and the
    counters are cleared at the end of each cycle rather than decayed. The
    weight transfer adds one iteration per MoE layer, so with n_moe_layers left
    at 0 this is a lower bound.
    """
    return (cfg.server.eplb_heat_collection_interval
            + cfg.server.eplb_algorithm_execution_interval
            + n_moe_layers)


def counting_config(cfg: ExperimentConfig, out: str | None) -> expert_counter.CountConfig | None:
    """The screen's counting knobs, or None when the arm does not count."""
    if not cfg.server.count_experts:
        return None
    return expert_counter.CountConfig(layers=cfg.server.count_layers,
                                      token_fraction=cfg.server.count_token_fraction,
                                      window_steps=cfg.server.count_window_steps,
                                      min_assignments=cfg.server.count_min_assignments,
                                      out=out)


def counting_summary(path: Path) -> dict | None:
    """What the counting hook actually wrote: windows, layers drawn, tokens counted.

    A counting arm that wrote nothing measured nothing, the same way an EPLB arm
    that applied nothing did, so this is read back rather than assumed.
    """
    if not path.exists():
        return None
    windows = 0
    layers: set[int] = set()
    counted = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        windows += 1
        layers.update(record.get("layers") or ())
        counted += sum(sum(row) for row in record.get("counts") or ())
    return {"windows": windows, "distinct_layers": len(layers), "assignments_counted": counted}


def replay_lengths(path: str | Path, prompts: Sequence[Any]) -> list[int]:
    """Each prompt's output length in the point recorded at ``path``.

    Matched by prompt content rather than position, so warm-up (a prefix of the
    benchmark prompts) and the benchmark itself both find theirs, and a prompt
    list that differs from the recorded one is refused rather than paired with
    another prompt's length.
    """
    import gzip

    records = json.load(gzip.open(path, "rt"))["requests"]
    lengths = {json.dumps(r["prompt"], sort_keys=True): r["num_output_tokens"] for r in records}
    out, missing = [], 0
    for prompt in prompts:
        n = lengths.get(json.dumps(prompt, sort_keys=True))
        if n is None:
            missing += 1
        else:
            out.append(max(1, int(n)))
    if missing:
        raise ValueError(f"client.lengths_from: {missing} of {len(prompts)} prompts have no recorded length "
                         f"in {path}; it must come from a point that served these prompts")
    return out


def serve_and_measure(cfg: ExperimentConfig, model_path: str, prompts: Sequence[Any],
                       trace_dir: str | None, enable_expert_capture: bool = False,
                       eplb_record_path: str | None = None,
                       server_env: dict[str, str] | None = None) -> list[dict] | None:
    from ..core.vllm_serving import measure_vllm_throughput

    max_new_tokens: int | list[int] = cfg.client.max_new_tokens
    if cfg.client.lengths_from:
        max_new_tokens = replay_lengths(cfg.client.lengths_from, prompts)
        if max(max_new_tokens) > cfg.client.max_new_tokens:
            raise ValueError(f"client.lengths_from replays up to {max(max_new_tokens)} tokens, above "
                             f"client.max_new_tokens = {cfg.client.max_new_tokens}")

    return asyncio.run(measure_vllm_throughput(
        model_path,
        list(prompts),
        seed=cfg.experiment.seed,
        max_new_tokens=max_new_tokens,
        max_model_len=cfg.server.max_model_len,
        batch_size=cfg.server.batch_size,
        max_num_batched_tokens=cfg.server.max_num_batched_tokens,
        enforce_eager=cfg.server.enforce_eager,
        concurrency_limit=cfg.client.concurrency_limit,
        gpu_memory_utilization=cfg.server.gpu_memory_utilization,
        n_gpus=cfg.hardware.n_npus,  # tensor-parallel size
        n_warmup_samples=cfg.client.n_warmup_samples,
        print_output=False,
        enable_expert_parallel=cfg.server.enable_expert_parallel,
        enable_prefix_caching=cfg.server.enable_prefix_caching,
        eplb=eplb_settings(cfg, eplb_record_path),
        enable_bnb=cfg.model.enable_bnb,
        enable_expert_capture=enable_expert_capture,
        trace_dir=trace_dir,
        trace_active_iterations=cfg.benchmark.trace_active_iterations,
        trace_start_iteration=cfg.benchmark.trace_start_iteration,
        port=cfg.server.port,
        extra_env=server_env,
        collect_itl=cfg.benchmark.save_itl and cfg.client.stream,
        stream=cfg.client.stream,
        ignore_eos=cfg.client.ignore_eos,
    ))

# Runs and records metrics for all pending sweep points
def _execution_order(ctx: RunContext, cfg: ExperimentConfig) -> list[dict[str, Any]]:
    """The order the sweep points are served in.

    Points are stored and plotted in parameter order, but serving them in that
    order aliases anything that drifts during a run - a neighbouring job, thermal
    state, a cache filling - onto the swept parameter itself.

    Points are served in rounds, one per repeat, each holding every value once
    in a random order. A single shuffle of all points can fall into blocks (all
    of one value first), and with a shared seed every run of a grid falls into
    the same blocks: on 2026-10-05 every run served all three bias-100 repeats
    before any bias-0 one. Rounds keep each value spread evenly over the run.
    The seed folds in the experiment name so runs of a grid are ordered
    differently, and stays reproducible for a resume.
    """
    points = list(ctx.points)
    if not cfg.benchmark.shuffle_points or len(points) < 3:
        return points
    rng = random.Random(f"{cfg.experiment.seed}:{cfg.experiment.name}")
    rounds: dict[int, list[dict[str, Any]]] = {}
    for p in points:
        rounds.setdefault(int(p.get("repeat") or 1), []).append(p)
    order: list[dict[str, Any]] = []
    for r in sorted(rounds):
        batch = rounds[r]
        rng.shuffle(batch)
        order.extend(batch)
    return order


def _warn_if_eplb_cannot_fire(cfg: ExperimentConfig) -> None:
    """Warn when the point is too short for EPLB to rearrange even once.

    Rearrangement is on a fixed iteration counter, not a timer, and the counter
    is cleared at the end of each cycle. A point that generates fewer forward
    iterations than one cycle therefore collects load, never acts on it, and
    looks exactly like an EPLB run that found nothing to fix. The decode phase
    is about one iteration per generated token, so max_new_tokens is the bound
    worth checking - at the default interval of 600 a 100-token generation is
    an order of magnitude short.
    """
    if not cfg.server.enable_eplb:
        return
    cycle = eplb_cycle_iterations(cfg)
    if cfg.client.max_new_tokens < cycle:
        log(f"warning: EPLB needs at least {cycle} forward iterations per rearrangement "
            f"(collect {cfg.server.eplb_heat_collection_interval} + plan "
            f"{cfg.server.eplb_algorithm_execution_interval}, plus one per MoE layer) but each point "
            f"generates about {cfg.client.max_new_tokens} decode iterations. EPLB will collect expert load "
            f"and never rearrange, which is indistinguishable from finding nothing to fix. Lower "
            f"server.eplb_heat_collection_interval or raise client.max_new_tokens.")


def _require_eplb_gate_patch(cfg: ExperimentConfig) -> None:
    """Refuse to start a gated arm against an unpatched vllm-ascend.

    The gate is a source patch on the installed package (see ``eplb_gate``), and
    ``uv sync`` reverts it. An unpatched server ignores the environment variable
    and serves plain EPLB, which would read as "gating changes nothing" - the
    one answer this run must not be able to produce by accident.
    """
    if not (cfg.server.eplb_min_imbalance or cfg.server.eplb_honor_changed):
        return
    which = "eplb_min_imbalance" if cfg.server.eplb_min_imbalance else "eplb_honor_changed"
    try:
        import vllm_ascend
    except ImportError as e:  # pragma: no cover - only reachable off the node
        raise RuntimeError(f"server.{which} needs vllm_ascend installed; this runs on the node") from e
    path = Path(vllm_ascend.__file__).parent / eplb_gate.TARGET
    if not eplb_gate.is_patched(path.read_text()):
        raise RuntimeError(f"server.{which} is set but {path} does not carry the gate patch. "
                           f"Install it with: ./nrun 'uv run python scripts/apply_node_patch.py' "
                           f"(re-run it after every uv sync).")


def benchmark_points(ctx: RunContext, cfg: ExperimentConfig,
                     point_inputs: Callable[[dict[str, Any]], tuple[str, Sequence[Any]]],
                     retry_failed: bool = False) -> None:
    
    bench = cfg.benchmark
    for arm in [None] + [name for name, _ in cfg.arms]:
        _warn_if_eplb_cannot_fire(cfg.for_arm(arm))
        _require_eplb_gate_patch(cfg.for_arm(arm))
    base_cfg = cfg
    for order, p in enumerate(_execution_order(ctx, base_cfg)):
        # Everything below serves this point's arm; the prompts and validation are shared.
        cfg = base_cfg.for_arm(p.get("arm"))
        label = p["label"]
        status = p.get("status", schema.STATUS_PENDING)
        if status == schema.STATUS_COMPLETED:
            log(f"{label}: already benchmarked - skipping")
            continue
        if status == schema.STATUS_FAILED and not retry_failed:
            log(f"{label}: failed in a previous attempt - skipping (use --retry-failed)")
            continue

        model_path, prompts = point_inputs(p)
        trace_rel = None
        if bench.enable_profiling:
            trace_rel = schema.trace_dir(label)
            trace_abs = ctx.abspath(trace_rel)
            if trace_abs.exists():
                shutil.rmtree(trace_abs)  # traces of an interrupted attempt
            trace_abs.mkdir(parents=True)

        # The profiler perturbs latency, so timings and traces cannot come from
        # the same pass. With separate_profiling_run the point is served twice:
        # unprofiled for the measurements, then profiled for the traces.
        separate = bool(bench.enable_profiling and bench.separate_profiling_run)

        host_before = host_snapshot(cfg.hardware.visible_devices)
        ctx.update_point(label, status=schema.STATUS_RUNNING, started_at=utcnow(), finished_at=None, error=None,
                         model_path=model_path, n_prompts=len(prompts), trace_dir=trace_rel,
                         metrics_file=None, request_summary=None, trace_metrics_file=None, trace_summary=None,
                         host_before=host_before, host_after=None, exec_order=order)
        log(f"{label}: benchmarking {model_path} with {len(prompts)} prompts"
            f"{' (measurement pass, unprofiled)' if separate else ''}")
        warning = contention_warning(host_before)
        if warning:
            log(f"{label}: warning: {warning}")

        trace_path = str(ctx.abspath(trace_rel)) if trace_rel else None
        # Records what EPLB actually did, which is the only way to tell a
        # rearrangement that never fired from one that fired and changed nothing.
        eplb_rel = f"{label}_eplb_expert_map.json" if cfg.server.eplb_record_map else None
        eplb_path = str(ctx.abspath(eplb_rel)) if eplb_rel else None
        # Per-point server environment, e.g. an injected router bias.
        server_env = dict(p.get("server_env") or {})
        # The screen's counting runs inside the server, so its output is a server
        # artefact like the EPLB map, not something the client can see.
        counts_rel = f"{label}_expert_counts.jsonl" if cfg.server.count_experts else None
        if counts_rel:
            server_env.update(expert_counter.server_env(counting_config(cfg, str(ctx.abspath(counts_rel)))))
        # Arms the node patch for this point only, which is what lets an EPLB-on
        # and an EPLB-gated arm share one run's shuffled rounds.
        server_env.update(eplb_gate.server_env(cfg.server.eplb_min_imbalance,
                                               cfg.server.eplb_honor_changed))
        server_env = server_env or None
        log_path = ctx.abspath(schema.LOG_FILE)
        log_offset = log_path.stat().st_size if log_path.exists() else None
        results = serve_and_measure(cfg, model_path, prompts,
                                    trace_dir=None if separate else trace_path,
                                    enable_expert_capture=cfg.benchmark.capture_experts,
                                    eplb_record_path=eplb_path, server_env=server_env)

        if results is None:
            ctx.update_point(label, status=schema.STATUS_FAILED, finished_at=utcnow(),
                             error=f"inference failed for {model_path} (see {schema.LOG_FILE})")
            log(f"{label}: FAILED")
            continue

        profiling_error = None
        profiled_summary = None
        if separate:
            log(f"{label}: profiling pass")
            profiled = serve_and_measure(cfg, model_path, prompts, trace_dir=trace_path,
                                         enable_expert_capture=cfg.benchmark.capture_experts,
                                         eplb_record_path=eplb_path, server_env=server_env)
            if profiled is None:
                # The measurements stand on their own; only the traces are lost.
                profiling_error = f"profiling pass failed for {model_path} (see {schema.LOG_FILE})"
                log(f"{label}: profiling pass FAILED - keeping the unprofiled measurements")
            else:
                # The two passes are the same configuration on the same point, so their
                # difference is what the profiler costs - the only twinned measurement
                # of it available, and it costs nothing extra to keep.
                profiled_summary = summarize_requests(profiled)

        fields: dict[str, Any] = {"request_summary": summarize_requests(results),
                                  "host_after": host_snapshot(cfg.hardware.visible_devices)}
        if profiled_summary is not None:
            fields["profiled_request_summary"] = profiled_summary
        if eplb_rel:
            # The *planned* placement: in record mode vllm-ascend never applies it
            # (see server.eplb_record_map), so it is not the layout that served.
            fields["eplb_expert_map"] = eplb_rel if ctx.abspath(eplb_rel).exists() else None
        if counts_rel:
            summary = counting_summary(ctx.abspath(counts_rel))
            fields["expert_counts"] = counts_rel if summary else None
            fields["counting_summary"] = summary
            if summary:
                log(f"{label}: counted {summary['assignments_counted']} expert assignments over "
                    f"{summary['windows']} windows, {summary['distinct_layers']} distinct layers drawn")
            else:
                log(f"{label}: warning: counting was on but no counts were written - the hook did not run")
        if cfg.server.enable_eplb and log_offset is not None:
            with open(log_path, "rb") as f:
                f.seek(log_offset)
                activity = eplb_activity(f.read().decode(errors="replace"), cfg.hardware.n_npus)
            fields["eplb_activity"] = activity
            log(f"{label}: {describe_eplb_activity(activity)}")
        if profiling_error:
            fields["profiling_error"] = profiling_error
        if bench.save_request_metrics:
            fields["metrics_file"] = ctx.write_json(schema.metrics_file(label), {
                "run_id": ctx.run_id,
                "label": label,
                "sweep_parameter": ctx.manifest["sweep_parameter"],
                "sweep_value": p["value"],
                "model_path": model_path,
                "profiled": bool(bench.enable_profiling) and not separate,
                "requests": results,
            })
        ctx.update_point(label, status=schema.STATUS_COMPLETED, finished_at=utcnow(), **fields)
        s = fields["request_summary"]
        cost = ""
        if profiled_summary and s.get("tpot_ms_mean") and profiled_summary.get("tpot_ms_mean"):
            cost = (f", profiled pass {_fmt(profiled_summary['tpot_ms_mean'])} ms "
                    f"({100 * (profiled_summary['tpot_ms_mean'] / s['tpot_ms_mean'] - 1):+.1f}%)")
        log(f"{label}: completed ({s.get('n_requests')} requests, "
            f"mean TTFT {_fmt(s.get('ttft_ms_mean'))} ms, mean TPOT {_fmt(s.get('tpot_ms_mean'))} ms{cost})")


_EPLB_PLAN = re.compile(r"Expert hotness imbalance, current: mean=([\d.]+) max=([\d.]+), "
                        r"updated: mean=([\d.]+) max=([\d.]+)")


def eplb_activity(log_text: str, n_ranks: int) -> dict[str, Any]:
    """What EPLB did while serving one point, read from the server's log.

    A placement map says what EPLB *planned*. Whether it was applied shows only in
    the log: every rank reports each completed weight-update cycle, and the
    planner reports the imbalance it measured (``current``) and the imbalance its
    plan would leave (``updated``). Once plans are applied, ``current`` falls
    towards the previous plan's ``updated``; when it stays put, nothing moved.
    Both are per-layer busiest-rank load over the mean - ``mean`` averaged over
    MoE layers, ``max`` the worst layer - which is the statistic our screen uses.

    ``stalled`` is vllm-ascend refusing a transfer because the previous one never
    finished, which is what record-only mode looks like.
    """
    plans = [[float(x) for x in m.groups()] for m in _EPLB_PLAN.finditer(log_text)]
    completed = log_text.count("Full expert weight update cycle completed")
    # Every rank's planner logs its own verdict, so the raw counts come in
    # multiples of n_ranks; a remainder means the ranks disagreed, which would be
    # a divergent placement rather than a gate.
    shut, opened = log_text.count(eplb_gate.LOG_SHUT), log_text.count(eplb_gate.LOG_OPEN)
    activity = {
        "plans": len(plans),
        "applied_cycles": -(-completed // max(1, n_ranks)),
        "stalled": "D2D weight update is on-going" in log_text,
        # [current mean, current max, updated mean, updated max] per plan, in order
        "hotness": plans,
    }
    if shut or opened:
        activity.update({
            "gate_shut": -(-shut // max(1, n_ranks)),
            "gate_open": -(-opened // max(1, n_ranks)),
            "gate_consistent": shut % max(1, n_ranks) == 0 and opened % max(1, n_ranks) == 0,
        })
    return activity


def describe_eplb_activity(activity: dict[str, Any]) -> str:
    plans, applied, hot = activity["plans"], activity["applied_cycles"], activity["hotness"]
    if not plans:
        return "EPLB never planned - the point is shorter than one collection interval"
    trend = f"busiest rank {hot[0][0]:.2f}x -> {hot[-1][0]:.2f}x by EPLB's count"
    gate = ""
    if "gate_shut" in activity:
        gate = f", gate shut {activity['gate_shut']}/{activity['gate_shut'] + activity['gate_open']} times"
        if not activity["gate_consistent"]:
            gate += " (warning: the ranks did not all reach the same verdict)"
    if not applied:
        stall = ", transfer stalled (record-only mode?)" if activity["stalled"] else ""
        return f"warning: EPLB planned {plans} times and applied none{stall}{gate}; {trend}"
    return f"EPLB planned {plans} times and applied {applied}{gate}; {trend}"


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def profiling_note(bench) -> str:
    """How each point is served, for the plan printed by `validate`."""
    if not bench.enable_profiling:
        return ""
    if bench.separate_profiling_run:
        return " (served twice per point: unprofiled for timings, profiled for traces)"
    return " (profiled)"


def post_processing_stages(ctx: RunContext, cfg: ExperimentConfig, force: bool = False) -> None:
    analysis = cfg.analysis
    traced = cfg.benchmark.enable_profiling

    if not traced:
        ctx.skip_stage(STAGE_TRACE_ANALYSIS, "profiling disabled (benchmark.enable_profiling = false)")
        ctx.skip_stage(STAGE_HTA, "profiling disabled (benchmark.enable_profiling = false)")
    else:
        if not (analysis.trace_summary or analysis.parse_npu_traces):
            ctx.skip_stage(STAGE_TRACE_ANALYSIS, "disabled (analysis.parse_npu_traces and analysis.trace_summary "
                                                 "are false)")
        elif should_run(ctx, STAGE_TRACE_ANALYSIS, force):
            with ctx.stage(STAGE_TRACE_ANALYSIS):
                trace_analysis_stage(ctx, cfg, force=force)
        if not analysis.hta:
            ctx.skip_stage(STAGE_HTA, "disabled (analysis.hta = false)")
        elif should_run(ctx, STAGE_HTA, force):
            with ctx.stage(STAGE_HTA):
                hta_stage(ctx, cfg)

    if not cfg.output.save_figures:
        ctx.skip_stage(STAGE_FIGURES, "disabled (output.save_figures = false)")
    elif should_run(ctx, STAGE_FIGURES, force):
        figures_stage(ctx)


NPU_TRACE_VIEW = "ASCEND_PROFILER_OUTPUT/trace_view.json"


def npu_trace_views(trace_dir: str | Path) -> list[Path]:
    return sorted(Path(trace_dir).rglob(NPU_TRACE_VIEW))

# Raw NPU profilier data to timeline files
def parse_npu_profiler_data(trace_dir: str | Path) -> list[Path]:
    trace_dir = Path(trace_dir)
    views = npu_trace_views(trace_dir)
    if views:
        return views
    if not trace_dir.is_dir() or not any(trace_dir.iterdir()):
        raise RuntimeError(f"no profiler data in {trace_dir}")
    from torch_npu.profiler.profiler import analyse

    analyse(profiler_path=str(trace_dir))
    views = npu_trace_views(trace_dir)
    if not views:  # data recorded per worker: parse each worker directory
        for worker_dir in sorted(p for p in trace_dir.iterdir() if p.is_dir()):
            analyse(profiler_path=str(worker_dir))
        views = npu_trace_views(trace_dir)
    if not views:
        raise RuntimeError(f"torch_npu offline parsing produced no {NPU_TRACE_VIEW} in {trace_dir}")
    return views


def trace_analysis_stage(ctx: RunContext, cfg: ExperimentConfig, force: bool = False) -> None:
    from ..core.trace_analysis import summarize, summarize_ascend

    for p in ctx.points:
        label = p["label"]
        if not p.get("trace_dir") or p.get("status") != schema.STATUS_COMPLETED:
            continue
        trace_abs = ctx.abspath(p["trace_dir"])

        if cfg.analysis.parse_npu_traces and (force or not p.get("npu_trace_views")):
            log(f"{label}: parsing NPU profiler data in {trace_abs}")
            try:
                views = parse_npu_profiler_data(trace_abs)
            except Exception as exc:  # noqa: BLE001
                ctx.update_point(label, trace_parse_error=f"{type(exc).__name__}: {exc}")
                log(f"{label}: NPU profiler parsing failed: {exc!r}")
            else:
                ctx.update_point(label, npu_trace_views=[ctx.relpath(v) for v in views], trace_parse_error=None)

        if cfg.analysis.trace_summary and (force or not p.get("trace_metrics_file")):
            log(f"{label}: extracting kernel metrics from {trace_abs}")
            try:
                summary = summarize(str(trace_abs))
            except FileNotFoundError:
                # No PyTorch-format rank traces. The Ascend profiler is what this
                # stack actually writes, so analyse its output instead.
                try:
                    summary = summarize_ascend(str(trace_abs))
                except Exception as exc:  # noqa: BLE001
                    message = (f"{exc} (neither PyTorch rank traces named *rank*.pt.trace.json.gz nor parsed "
                               f"Ascend profiler output were found)")
                    ctx.update_point(label, trace_error=message)
                    log(f"{label}: {message}")
                    continue
                d = summary.get("decomposition") or {}
                moe = (d.get("compute_pct") or {}).get("moe")
                strag = (summary.get("stragglers") or {}).get("GroupedMatmul") or {}
                log(f"{label}: {len(summary['ranks'])} ranks, MoE {moe:.1f}% of compute, "
                    f"straggler {strag.get('straggler', float('nan')):.3f}x"
                    if moe is not None else f"{label}: analysed {len(summary['ranks'])} ranks")
            summary["trace_dir"] = p["trace_dir"]
            rel = ctx.write_json(schema.trace_metrics_file(label), summary)
            ctx.update_point(label, trace_metrics_file=rel, trace_summary=trace_scalars(summary), trace_error=None)


def hta_stage(ctx: RunContext, cfg: ExperimentConfig) -> None:
    try:
        from ..core.hta_analysis import extract_metrics, load_trace_analysis
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("HTA analysis requires the 'hta' extra: uv sync --extra hta") from exc

    analyses = {}
    for p in ctx.points:
        if p.get("trace_dir") and p.get("status") == schema.STATUS_COMPLETED:
            key = (cfg.model.model_name, cfg.server.batch_size, p["value"])
            analyses[key] = load_trace_analysis(str(ctx.abspath(p["trace_dir"])))
    if not analyses:
        raise RuntimeError("no traced sweep points to analyse")

    rank_df, idle_cat_df, kern_df, run_df = extract_metrics(analyses)
    rel = ctx.write_json(schema.HTA_FILE, {
        "run_id": ctx.run_id,
        "sweep_parameter": ctx.manifest["sweep_parameter"],
        "key_columns": {"model": "model.model_name", "batch": "server.batch_size",
                        "imbalance": ctx.manifest["sweep_parameter"]},
        "tables": {
            "rank": rank_df.to_dict(orient="records"),
            "idle_categories": idle_cat_df.to_dict(orient="records"),
            "kernel_types": kern_df.to_dict(orient="records"),
            "runs": run_df.to_dict(orient="records"),
        },
    })
    ctx.manifest["hta_file"] = rel
    ctx.save()


def figures_stage(ctx: RunContext) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from moe_reliability_results import Run, plots

    error: Exception | None = None
    written = []
    with ctx.stage(STAGE_FIGURES):
        try:
            figs = plots.plot_run(Run(ctx.path))
            written = plots.save_figures(figs, ctx.abspath(schema.FIGURES_DIR))
        except Exception as exc:  # noqa: BLE001
            error = exc
    if error is not None:
        ctx.skip_stage(STAGE_FIGURES, f"figure rendering failed: {error!r} "
                                      f"(re-render with: moe-reliability-results plot {ctx.run_id})")
        log(f"figure rendering failed: {error!r}")
        return
    if not written:
        # Nothing to plot, normally because the benchmark produced no usable
        # points. Leaving the stage completed would make resume skip it for the
        # life of the run, so there would be no figures even once the failed
        # points have been retried successfully.
        ctx.skip_stage(STAGE_FIGURES, "no figures rendered (no benchmark results to plot); "
                                      "retried on the next resume")
        log("rendered 0 figures - leaving the stage open for a later resume")
        return
    ctx.manifest["figures"] = [ctx.relpath(p) for p in written]
    ctx.save()
    log(f"rendered {len(written)} figures")
