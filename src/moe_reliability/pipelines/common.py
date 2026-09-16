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
import shutil
from pathlib import Path
from typing import Any, Callable, Sequence

from moe_results import schema
from moe_results.metrics import summarize_requests, trace_scalars

from ..config import ExperimentConfig
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

def serve_and_measure(cfg: ExperimentConfig, model_path: str, prompts: Sequence[Any],
                       trace_dir: str | None, enable_expert_capture: bool = False) -> list[dict] | None:
    from ..core.vllm_serving import measure_vllm_throughput

    return asyncio.run(measure_vllm_throughput(
        model_path,
        list(prompts),
        seed=cfg.experiment.seed,
        max_new_tokens=cfg.client.max_new_tokens,
        max_model_len=cfg.server.max_model_len,
        batch_size=cfg.server.batch_size,
        concurrency_limit=cfg.client.concurrency_limit,
        gpu_memory_utilization=cfg.server.gpu_memory_utilization,
        n_gpus=cfg.hardware.n_npus,  # tensor-parallel size
        n_warmup_samples=cfg.client.n_warmup_samples,
        print_output=False,
        enable_expert_parallel=cfg.server.enable_expert_parallel,
        enable_prefix_caching=cfg.server.enable_prefix_caching,
        enable_eplb=cfg.server.enable_eplb,
        enable_bnb=cfg.model.enable_bnb,
        enable_expert_capture=enable_expert_capture,
        trace_dir=trace_dir,
        trace_active_iterations=cfg.benchmark.trace_active_iterations,
        port=cfg.server.port,
    ))

# Runs and records metrics for all pending sweep points
def benchmark_points(ctx: RunContext, cfg: ExperimentConfig,
                     point_inputs: Callable[[dict[str, Any]], tuple[str, Sequence[Any]]],
                     retry_failed: bool = False) -> None:
    
    bench = cfg.benchmark
    for p in ctx.points:
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

        ctx.update_point(label, status=schema.STATUS_RUNNING, started_at=utcnow(), finished_at=None, error=None,
                         model_path=model_path, n_prompts=len(prompts), trace_dir=trace_rel,
                         metrics_file=None, request_summary=None, trace_metrics_file=None, trace_summary=None)
        log(f"{label}: benchmarking {model_path} with {len(prompts)} prompts")

        results = serve_and_measure(cfg, model_path, prompts,
                                     trace_dir=str(ctx.abspath(trace_rel)) if trace_rel else None)

        if results is None:
            ctx.update_point(label, status=schema.STATUS_FAILED, finished_at=utcnow(),
                             error=f"inference failed for {model_path} (see {schema.LOG_FILE})")
            log(f"{label}: FAILED")
            continue

        fields: dict[str, Any] = {"request_summary": summarize_requests(results)}
        if bench.save_request_metrics:
            fields["metrics_file"] = ctx.write_json(schema.metrics_file(label), {
                "run_id": ctx.run_id,
                "label": label,
                "sweep_parameter": ctx.manifest["sweep_parameter"],
                "sweep_value": p["value"],
                "model_path": model_path,
                "profiled": bool(bench.enable_profiling),
                "requests": results,
            })
        ctx.update_point(label, status=schema.STATUS_COMPLETED, finished_at=utcnow(), **fields)
        s = fields["request_summary"]
        log(f"{label}: completed ({s.get('n_requests')} requests, "
            f"mean TTFT {_fmt(s.get('ttft_ms_mean'))} ms, mean TPOT {_fmt(s.get('tpot_ms_mean'))} ms)")


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


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
    from ..core.trace_analysis import summarize

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
            log(f"{label}: extracting fused-MoE kernel metrics from {trace_abs}")
            try:
                summary = summarize(str(trace_abs))
            except SystemExit as exc:  # raised when no rank traces exist
                message = (f"{exc} (the fused-MoE kernel analysis reads PyTorch profiler traces named "
                           f"*rank*.pt.trace.json.gz)")
                ctx.update_point(label, trace_error=message)
                log(f"{label}: {message}")
                continue
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
    from moe_results import Run, plots

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
                                      f"(re-render with: moe-results plot {ctx.run_id})")
        log(f"figure rendering failed: {error!r}")
        return
    ctx.manifest["figures"] = [ctx.relpath(p) for p in written]
    ctx.save()
    log(f"rendered {len(written)} figures")
