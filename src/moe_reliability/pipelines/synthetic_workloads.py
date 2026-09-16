###
# synthetic workloads.py
#
# Experimental stages uniquely defined for synthetic workload experiments.
# Includes recording expert activations over MMLU via vLLM router replay
# (activations), constructing synthetic workloads from these prompts (workloads),
# and running inference over multiple workloads (benchmark).
#
# Dylan Everingham
# 16.09.2026
###

from __future__ import annotations

from typing import Any

from moe_results import Run, io, schema
from moe_results.metrics import workload_point_stats

from ..config import ExperimentConfig
from ..logs import log
from ..runs import RunContext, RunError, resolve_run_dir
from . import common

__all__ = ["STAGES", "run", "plan", "sweep_values"]

STAGE_ACTIVATIONS = "activations"
STAGE_WORKLOADS = "workloads"
STAGES = (STAGE_ACTIVATIONS, STAGE_WORKLOADS, common.STAGE_BENCHMARK, common.STAGE_TRACE_ANALYSIS,
          common.STAGE_HTA, common.STAGE_FIGURES)

def _owner(run: Run, kind: str) -> Run:
    """Follow reuse links until the run that actually stores ``kind``."""
    seen = set()
    while True:
        link = (run.manifest.get("inputs") or {}).get(kind)
        if not link or run.id in seen:
            return run
        seen.add(run.id)
        sibling = run.path.parent / link["run_id"]
        run = Run(sibling if (sibling / schema.MANIFEST_FILE).exists() else link["path"])


def _link_input(ctx: RunContext, kind: str, ref: str, results_dir: str) -> Run:
    source = _owner(Run(resolve_run_dir(ref, results_dir)), kind)
    if kind == STAGE_ACTIVATIONS and not source.manifest.get("activations"):
        raise RunError(f"run {source.id} has no activation records")
    if kind == STAGE_WORKLOADS and not source.manifest.get("workloads"):
        raise RunError(f"run {source.id} has no synthetic workloads")
    ctx.manifest["inputs"][kind] = {"run_id": source.id, "path": str(source.path)}
    ctx.save()
    return source


def _source(ctx: RunContext, kind: str) -> Run:
    return _owner(Run(ctx.path), kind)

def record_activations(ctx: RunContext, cfg: ExperimentConfig) -> None:
    common.seed_everything(cfg.experiment.seed)
    n_samples = cfg.activations.n_samples
    prompts, subjects, questions = common.mmlu_prompts(n_samples, cfg.experiment.seed)

    log(f"capturing routed experts for {len(prompts)} MMLU prompts with {cfg.model.model_id}")
    results = common.serve_and_measure(cfg, cfg.model.model_id, prompts, trace_dir=None,
                                        enable_expert_capture=True)
    if results is None:
        raise RuntimeError(f"routed-expert capture failed (see {schema.LOG_FILE})")

    def records():
        for r in results:
            yield {**r, "subject": subjects[r["prompt_id"]]}

    rel, n = ctx.write_jsonl(schema.ACTIVATIONS_FILE, records())
    ctx.manifest["activations"] = {
        "file": rel,
        "n_records": n,
        "model_id": cfg.model.model_id,
        "n_samples": n_samples,
        "seed": cfg.experiment.seed,
        "max_new_tokens": cfg.client.max_new_tokens,
    }
    ctx.save()
    log(f"stored {n} activation records in {rel}")

def build_workloads(ctx: RunContext, cfg: ExperimentConfig) -> None:
    from .. import models
    from ..core.synthetic_workloads import get_qs, workload_sweep_cvs

    common.seed_everything(cfg.experiment.seed)

    source = _source(ctx, STAGE_ACTIVATIONS)
    entry = source.manifest["activations"]
    log(f"loading activation records of run {source.id}")
    results = list(source.activations())

    # Parameters
    family = models.resolve_probe_family(cfg.model.model_id, cfg.model.probe)
    n_experts, n_layers, k = models.moe_dimensions(cfg.model.model_id, family)
    model_id_simple = cfg.model.model_name

    # For Deepseek, remove first layer as it is not routed.
    if "deepseek" in model_id_simple:
        for i in range(len(results)):
            results[i]['routed_experts'] = results[i]['routed_experts'][1:]
            results[i]['prompt_routed_experts'] = results[i]['prompt_routed_experts'][1:]
        n_layers = results[0]['routed_experts'].shape[1]

    # Get total number of tokens in activation results
    token_counts = [r['routed_experts'].shape[0] + r['prompt_routed_experts'].shape[0] for r in results]
    n_total_tokens = sum(token_counts)
    n_prompts = len(results)
    avg_tokens_per_prompt = n_total_tokens / n_prompts

    # Get the "naturally occuring" CV in the data
    qs = get_qs(results, n_experts, n_layers, k, weighted_by_token_count=True)
    qs_mean = qs.mean(dim=0)
    cv_nat = qs_mean.std(dim=0) / qs_mean.mean(dim=0)  # (n_layers)

    # Now search for workloads which fit a range of CVs
    # We sweep a parameter alpha which scales cv_nat
    target_alphas = cfg.workloads.target_alphas
    target_prompt_ls = cfg.workloads.target_prompt_lengths
    target_ls = [int(avg_tokens_per_prompt * p) for p in target_prompt_ls]

    # Get full MMLU formatted prompts for each workload
    formatted_prompts_mmlu, subjects, questions = common.mmlu_prompts(entry["n_samples"], entry["seed"])

    ctx.manifest.setdefault("workloads", {})
    for max_repeats in cfg.workloads.max_repeats:
        key = str(int(max_repeats))
        if key in ctx.manifest["workloads"]:
            log(f"workloads with max_repeats={max_repeats} already built - skipping")
            continue
        log(f"constructing workloads with max_repeats={max_repeats}")
        workloads = workload_sweep_cvs(results, qs, n_experts, n_layers, k, target_alphas, target_ls, cv_nat,
                                       max_repeats=max_repeats, verbose=True)

        # Put formatted prompts into the workload data structures
        for l in target_ls:
            for a in target_alphas:
                w = workloads[l][a]
                ids = w['indices']
                prompts_formatted = [formatted_prompts_mmlu[i] for i in ids]
                w['prompts_formatted'] = prompts_formatted

        doc = {
            "run_id": ctx.run_id,
            "activations_run_id": source.id,
            "model_id": cfg.model.model_id,
            "model_name": model_id_simple,
            "seed": cfg.experiment.seed,
            "max_repeats": int(max_repeats),
            "n_experts": n_experts,
            "n_layers": n_layers,
            "k": k,
            "n_prompts": n_prompts,
            "n_total_tokens": n_total_tokens,
            "avg_tokens_per_prompt": avg_tokens_per_prompt,
            "cv_nat": cv_nat,
            "target_alphas": target_alphas,
            "target_prompt_lengths": target_prompt_ls,
            "target_ls": target_ls,
            "workloads": workloads,
        }
        rel = ctx.write_json(schema.workloads_file(max_repeats), doc)
        ctx.manifest["workloads"][key] = {
            "file": rel,
            "max_repeats": int(max_repeats),
            "target_alphas": target_alphas,
            "target_prompt_lengths": target_prompt_ls,
            "target_ls": target_ls,
        }
        ctx.save()
        log(f"stored workloads in {rel}")

def _benchmark_workloads(ctx: RunContext, cfg: ExperimentConfig) -> tuple[dict[str, Any], int]:
    source = _source(ctx, STAGE_WORKLOADS)
    sets = source.manifest.get("workloads") or {}
    max_repeats = cfg.benchmark.workload_max_repeats
    entry = sets.get(str(int(max_repeats)))
    if entry is None:
        raise RunError(f"run {source.id} has no workloads with max_repeats={max_repeats}; "
                       f"available: {sorted(int(k) for k in sets)}")
    doc = io.read_json(source.path / entry["file"])
    prompt_length = cfg.benchmark.workload_prompt_length
    if prompt_length not in doc["target_prompt_lengths"]:
        raise RunError(f"workloads of run {source.id} have no target prompt length {prompt_length}; "
                       f"available: {doc['target_prompt_lengths']}")
    l = doc["target_ls"][doc["target_prompt_lengths"].index(prompt_length)]
    return doc, l


def benchmark(ctx: RunContext, cfg: ExperimentConfig, retry_failed: bool = False) -> None:
    doc, l = _benchmark_workloads(ctx, cfg)
    by_alpha = doc["workloads"][str(l)]
    workload_alphas = [float(a) for a in by_alpha]
    ctx.ensure_points(workload_alphas)

    workload_prompts = {}
    for alpha_key, p in zip(by_alpha, ctx.points):
        w = by_alpha[alpha_key]
        workload_prompts[p["label"]] = w["prompts_formatted"]
        ctx.update_point(p["label"], workload={
            **workload_point_stats(w, doc["cv_nat"]),
            "max_repeats": doc["max_repeats"],
            "target_prompt_length": cfg.benchmark.workload_prompt_length,
            "target_tokens": l,
        })
    log(f"benchmarking {len(workload_alphas)} workloads (max_repeats={doc['max_repeats']}, "
        f"{cfg.benchmark.workload_prompt_length} prompts / {l} tokens): alphas {workload_alphas}")

    common.seed_everything(cfg.experiment.seed)
    common.benchmark_points(ctx, cfg, lambda p: (cfg.model.model_id, workload_prompts[p["label"]]),
                            retry_failed=retry_failed)

def run(ctx: RunContext, cfg: ExperimentConfig, retry_failed: bool = False) -> None:
    ctx.init_stages(STAGES)
    results_dir = cfg.output.results_dir
    reuse_workloads = cfg.workloads.reuse_workloads_from
    reuse_activations = cfg.activations.reuse_activations_from

    if reuse_workloads:
        source = _link_input(ctx, STAGE_WORKLOADS, reuse_workloads, results_dir)
        if source.manifest.get("activations") or (source.manifest.get("inputs") or {}).get(STAGE_ACTIVATIONS):
            _link_input(ctx, STAGE_ACTIVATIONS, source.id, str(source.path.parent))
        ctx.skip_stage(STAGE_ACTIVATIONS, f"workloads reused from run {source.id}")
        ctx.skip_stage(STAGE_WORKLOADS, f"workloads reused from run {source.id}")
    else:
        if reuse_activations:
            source = _link_input(ctx, STAGE_ACTIVATIONS, reuse_activations, results_dir)
            ctx.skip_stage(STAGE_ACTIVATIONS, f"activation records reused from run {source.id}")
        elif common.should_run(ctx, STAGE_ACTIVATIONS):
            with ctx.stage(STAGE_ACTIVATIONS):
                record_activations(ctx, cfg)
        if common.should_run(ctx, STAGE_WORKLOADS):
            with ctx.stage(STAGE_WORKLOADS):
                build_workloads(ctx, cfg)

    has_failed = any(p.get("status") == schema.STATUS_FAILED for p in ctx.points)
    if common.should_run(ctx, common.STAGE_BENCHMARK, force=retry_failed and has_failed):
        with ctx.stage(common.STAGE_BENCHMARK):
            benchmark(ctx, cfg, retry_failed=retry_failed)

    common.post_processing_stages(ctx, cfg)


def plan(cfg: ExperimentConfig) -> list[str]:
    wl, bench = cfg.workloads, cfg.benchmark
    lines = []
    if wl.reuse_workloads_from:
        lines.append(f"reuse workloads of run {wl.reuse_workloads_from!r}")
    else:
        if cfg.activations.reuse_activations_from:
            lines.append(f"reuse activation records of run {cfg.activations.reuse_activations_from!r}")
        else:
            lines.append(f"capture routed experts for {cfg.activations.n_samples} MMLU prompts")
        lines.append(f"build workloads: alphas {wl.target_alphas} x prompt lengths {wl.target_prompt_lengths} "
                     f"x max_repeats {wl.max_repeats}")
    lines.append(f"benchmark workload set max_repeats={bench.workload_max_repeats}, "
                 f"{bench.workload_prompt_length} prompts: "
                 f"{len(wl.target_alphas) if not wl.reuse_workloads_from else 'all'} sweep points"
                 f"{' (profiled)' if bench.enable_profiling else ''}")
    return lines


def sweep_values(cfg: ExperimentConfig) -> list[float] | None:
    if cfg.workloads.reuse_workloads_from:
        return None
    return [float(a) for a in cfg.workloads.target_alphas]

