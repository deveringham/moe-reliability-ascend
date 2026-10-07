###
# forced_imbalance.py
#
# Experiment stages defined uniquely for forced imbalance experiments.
# Includes generating biased models (checkpoints), validating imbalance 
# of biased models (validation), and running inference over same prompts for
# all biased models and the baseline (benchmark).
# Dylan Everingham
# 16.09.2026
###

from __future__ import annotations

import os
from collections import Counter

from moe_reliability_results import schema

from .. import models
from ..config import ExperimentConfig
from ..logs import log
from ..runs import RunContext
from . import common

__all__ = ["STAGES", "run", "plan", "sweep_values", "checkpoint_path", "VALIDATION_PROMPTS"]

STAGE_CHECKPOINTS = "checkpoints"
STAGE_VALIDATION = "validation"
STAGES = (STAGE_CHECKPOINTS, STAGE_VALIDATION, common.STAGE_BENCHMARK, common.STAGE_TRACE_ANALYSIS,
          common.STAGE_HTA, common.STAGE_FIGURES)

# Arbitrary prompts used to confirm imbalance
VALIDATION_PROMPTS = ["Describe the concept of Mixture in Experts in detail.",
                      "Translate 'Good morning, how are you today?' into Indonesian.",
                      "What is 45 multiplied by 12?",
                      "Write a one-sentence definition of photosynthesis.",
                      "Sort these words alphabetically: banana, apple, cherry, date.",
                      "Complete the phrase: To be, or not to be, that is the..."]


def checkpoint_path(cfg: ExperimentConfig, imbalance_level: float | int) -> str:
    # Imabalance 0 gives baseline model
    if imbalance_level == 0:
        return cfg.model.model_id
    name = f"{cfg.model.model_name}-imbalance{schema.format_value(imbalance_level)}"
    return os.path.join(cfg.imbalance.model_dir, name)

def _expert_count(cfg: ExperimentConfig) -> int:
    family = models.resolve_probe_family(cfg.model.model_id, cfg.model.probe)
    n_experts, _, _ = models.moe_dimensions(cfg.model.model_id, family)
    return n_experts


def prepare_router_bias(ctx: RunContext, cfg: ExperimentConfig) -> None:
    """Every point serves the unmodified model; the level travels as a server environment."""
    from ..router_bias import server_env, target_experts

    n_experts, n_ranks = _expert_count(cfg), cfg.hardware.n_npus
    targeted = target_experts(cfg.imbalance.bias_target, n_experts, n_ranks)
    layers = f", layers {cfg.imbalance.bias_layers}" if cfg.imbalance.bias_layers else ""
    log(f"router bias on {cfg.imbalance.bias_target}: experts {targeted} of {n_experts}, {n_ranks} ranks{layers}"
        + (", plugin installed at level 0 too (all-zero vector)" if cfg.imbalance.bias_plugin_at_zero else ""))
    for p in ctx.points:
        ctx.update_point(p["label"], model_path=cfg.model.model_id, checkpoint_created=False,
                         server_env=server_env(cfg.imbalance.bias_target, p["value"], n_experts, n_ranks,
                                               at_zero=cfg.imbalance.bias_plugin_at_zero,
                                               layers=cfg.imbalance.bias_layers),
                         bias_experts=targeted)


def create_checkpoints(ctx: RunContext, cfg: ExperimentConfig) -> None:
    if cfg.imbalance.method == "router_bias":
        prepare_router_bias(ctx, cfg)
        return
    from ..core.forced_imbalance import imbalance_pretrained_moe

    common.seed_everything(cfg.experiment.seed)
    for p in ctx.points:
        level = p["value"]
        model_path = checkpoint_path(cfg, level)
        created = False
        if level != 0 and not os.path.isdir(model_path):
            log(f"{p['label']}: writing imbalanced checkpoint {model_path}")
            imbalance_pretrained_moe(cfg.model.model_id, level, model_path)
            common.free_accelerator_memory()
            created = True
        elif level != 0:
            log(f"{p['label']}: reusing existing checkpoint {model_path}")
        ctx.update_point(p["label"], model_path=model_path, checkpoint_created=created)

def _moe_layers(records) -> list[int]:
    """Captured layers that route. Dense layers appear in the capture as all-zero ids."""
    import numpy as np

    seen: np.ndarray | None = None
    for r in records:
        for key in ("prompt_routed_experts", "routed_experts"):
            experts = r.get(key)
            if experts is None or np.asarray(experts).size == 0:
                continue
            nz = (np.asarray(experts) != 0).any(axis=(0, 2))
            seen = nz if seen is None else (seen | nz)
    return [] if seen is None else [int(i) for i in np.flatnonzero(seen)]


def rank_load(per_layer_frequencies, n_ranks: int) -> dict:
    """Per-rank share of token-expert assignments under contiguous placement."""
    import numpy as np

    f = np.asarray(per_layer_frequencies, dtype=float)  # (layers, experts)
    per_rank = f.reshape(f.shape[0], n_ranks, -1).sum(axis=2)  # (layers, ranks)
    mean_share = per_rank.mean(axis=0)
    return {
        "rank_share_per_layer": per_rank.tolist(),
        "rank_share_mean": mean_share.tolist(),
        "rank_max_over_mean": float(mean_share.max() * n_ranks),
        "rank_max_over_mean_per_layer": (per_rank.max(axis=1) * n_ranks).tolist(),
    }


def _expert_load(records, n_experts: int, layer: int) -> dict:
    """Expert selection frequencies in one layer, over every captured token."""
    import numpy as np

    counts = Counter()
    for r in records:
        for key in ("prompt_routed_experts", "routed_experts"):
            experts = r.get(key)
            if experts is None:
                continue
            counts.update(np.asarray(experts)[:, layer, :].flatten().tolist())
    count_per_expert = np.array([counts[i] for i in range(n_experts)])
    tokens = int(count_per_expert.sum())
    return {"counts": count_per_expert, "n_assignments": tokens,
            "frequencies": count_per_expert / tokens if tokens else count_per_expert.astype(float)}


def validate_checkpoints(ctx: RunContext, cfg: ExperimentConfig) -> None:
    """Measure which experts each checkpoint actually routes to.

    The experts are read from the serving stack itself, with vLLM's routed-expert
    capture, rather than from a separate Hugging Face forward pass. That measures
    the deployment under test instead of a second implementation of it, and it
    does not depend on the router being reachable as a module: recent
    transformers compute the router logits functionally from ``gate.weight``, so
    a forward hook on the gate never fires.
    """
    n = cfg.imbalance.validation_samples
    subjects: list | None = None
    if cfg.imbalance.validation_workload:
        from ..core.data import workload_prompts

        prompts, subjects = workload_prompts(cfg.imbalance.validation_workload, n, cfg.experiment.seed)
    elif n:
        prompts, subjects, _ = common.mmlu_prompts(n, cfg.experiment.seed)
    else:
        prompts = VALIDATION_PROMPTS
    validated: set = set()
    for p in ctx.points:
        label, level = p["label"], p["value"]
        if p.get("validation_file"):
            continue
        if level in validated:
            continue  # repeats of a level share its validation
        validated.add(level)
        model_path = p.get("model_path") or checkpoint_path(cfg, level)
        bias_note = f" with router bias {level} on {cfg.imbalance.bias_target}" if p.get("server_env") else ""
        log(f"{label}: validating router load of {model_path}{bias_note} ({len(prompts)} prompts)")

        records = common.serve_and_measure(cfg, model_path, prompts, trace_dir=None,
                                           enable_expert_capture=True, server_env=p.get("server_env") or None)
        if not records:
            raise RuntimeError(f"routed-expert capture failed for {model_path} (see {schema.LOG_FILE})")

        # From the configuration, not from the capture: a strongly biased
        # checkpoint may never select the highest-numbered expert.
        family = models.resolve_probe_family(cfg.model.model_id, cfg.model.probe)
        n_experts, n_layers, k = models.moe_dimensions(model_path, family)
        if not _has_capture(records):
            raise RuntimeError(f"no routed experts captured for {model_path}; the server must run with "
                               f"--enable-return-routed-experts")

        layers = _moe_layers(records) or list(range(n_layers))
        layer0 = _expert_load(records, n_experts, layer=layers[0])
        per_layer = [_expert_load(records, n_experts, layer=i)["frequencies"] for i in layers]
        n_ranks = cfg.hardware.n_npus
        ranks = rank_load(per_layer, n_ranks) if n_experts % n_ranks == 0 else {}
        records_rel = None
        if cfg.imbalance.validation_save_records:
            records_rel, _ = ctx.write_jsonl(
                schema.validation_records_file(label),
                ({**r, "subject": subjects[r["prompt_id"]] if subjects else None} for r in records))
        rel = ctx.write_json(schema.validation_file(label), {
            "run_id": ctx.run_id,
            "label": label,
            "imbalance_level": level,
            "model_path": model_path,
            "router_id": layers[0],
            "moe_layers": layers,
            "n_experts": n_experts,
            "n_routers": len(layers),
            "k": k,
            "n_ranks": n_ranks,
            "server_env": p.get("server_env") or {},
            "bias_layers": list(cfg.imbalance.bias_layers),
            "validation_workload": cfg.imbalance.validation_workload or "mmlu (legacy order)",
            "records_file": records_rel,
            **layer0,
            "per_router_frequencies": per_layer,
            **ranks,
            "prompts": prompts,
            "responses": [r.get("response") for r in records],
        })
        import numpy as np

        mean_freq = np.asarray(per_layer).mean(axis=0)
        summary = {
            "n_assignments": layer0["n_assignments"],
            "max_expert_frequency": float(mean_freq.max()),
            "expected_frequency": 1 / n_experts,
            "active_experts": int((mean_freq > 0.1 / n_experts).sum()),
        }
        if ranks:
            summary["rank_share_mean"] = ranks["rank_share_mean"]
            summary["rank_max_over_mean"] = ranks["rank_max_over_mean"]
        # Every repeat of a level records the same validation.
        for q in ctx.points:
            if q["value"] == level:
                ctx.update_point(q["label"], validation_file=rel, validation_summary=summary)
        log(f"{label}: rank shares {[round(x, 3) for x in ranks.get('rank_share_mean', [])]}, "
            f"busiest rank {ranks.get('rank_max_over_mean', float('nan')):.2f}x mean, "
            f"{summary['active_experts']} of {n_experts} experts active")


def _has_capture(records) -> bool:
    import numpy as np

    return any(np.asarray(r[key]).size
               for r in records
               for key in ("routed_experts", "prompt_routed_experts")
               if r.get(key) is not None)

def run(ctx: RunContext, cfg: ExperimentConfig, retry_failed: bool = False) -> None:
    ctx.init_stages(STAGES)
    ctx.ensure_points(cfg.imbalance.imbalance_levels, repeats=cfg.benchmark.repeats)

    if common.should_run(ctx, STAGE_CHECKPOINTS):
        with ctx.stage(STAGE_CHECKPOINTS):
            create_checkpoints(ctx, cfg)

    if not cfg.imbalance.validate_imbalance:
        ctx.skip_stage(STAGE_VALIDATION, "disabled (imbalance.validate_imbalance = false)")
    elif common.should_run(ctx, STAGE_VALIDATION):
        with ctx.stage(STAGE_VALIDATION):
            validate_checkpoints(ctx, cfg)

    has_failed = any(p.get("status") == schema.STATUS_FAILED for p in ctx.points)
    if common.should_run(ctx, common.STAGE_BENCHMARK, force=retry_failed and has_failed):
        with ctx.stage(common.STAGE_BENCHMARK):
            common.seed_everything(cfg.experiment.seed)
            cache: dict[str, list] = {}

            def point_inputs(p):
                # Use MMLU questions (loaded once, only if a point still needs benchmarking)
                if "prompts" not in cache:
                    cache["prompts"], _, _ = common.mmlu_prompts(cfg.benchmark.n_samples, cfg.experiment.seed)
                return p.get("model_path") or checkpoint_path(cfg, p["value"]), cache["prompts"]

            common.benchmark_points(ctx, cfg, point_inputs, retry_failed=retry_failed)

    common.post_processing_stages(ctx, cfg)


def plan(cfg: ExperimentConfig) -> list[str]:
    levels = cfg.imbalance.imbalance_levels
    if cfg.imbalance.method == "router_bias":
        where = f", layers {cfg.imbalance.bias_layers}" if cfg.imbalance.bias_layers else ""
        lines = [f"router bias on {cfg.imbalance.bias_target}{where} at logit offsets {levels}, serving "
                 f"{cfg.model.model_id} unmodified (no checkpoints)"]
    else:
        lines = [f"checkpoints: {checkpoint_path(cfg, level)}" for level in levels]
    if cfg.imbalance.validate_imbalance:
        n = cfg.imbalance.validation_samples
        source = cfg.imbalance.validation_workload or ("MMLU" if n else "")
        records = ", per-request records kept" if cfg.imbalance.validation_save_records else ""
        lines.append(f"validate router load of every level ({n or 'six fixed'} {source} prompts, routed-expert "
                     f"capture{records})".replace("  ", " "))
    lines.append(f"benchmark {len(levels)} imbalance levels {levels} with {cfg.benchmark.n_samples} MMLU prompts"
                 f"{common.profiling_note(cfg.benchmark)}")
    return lines


def sweep_values(cfg: ExperimentConfig) -> list[float | int]:
    return list(cfg.imbalance.imbalance_levels)

