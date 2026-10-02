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

def create_checkpoints(ctx: RunContext, cfg: ExperimentConfig) -> None:
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
    for p in ctx.points:
        label, level = p["label"], p["value"]
        if p.get("validation_file"):
            continue
        model_path = p.get("model_path") or checkpoint_path(cfg, level)
        log(f"{label}: validating router load of {model_path}")

        records = common.serve_and_measure(cfg, model_path, VALIDATION_PROMPTS, trace_dir=None,
                                           enable_expert_capture=True)
        if not records:
            raise RuntimeError(f"routed-expert capture failed for {model_path} (see {schema.LOG_FILE})")

        # From the configuration, not from the capture: a strongly biased
        # checkpoint may never select the highest-numbered expert.
        family = models.resolve_probe_family(cfg.model.model_id, cfg.model.probe)
        n_experts, n_layers, k = models.moe_dimensions(model_path, family)
        if not _has_capture(records):
            raise RuntimeError(f"no routed experts captured for {model_path}; the server must run with "
                               f"--enable-return-routed-experts")

        layer0 = _expert_load(records, n_experts, layer=0)
        per_layer = [_expert_load(records, n_experts, layer=i)["frequencies"] for i in range(n_layers)]
        rel = ctx.write_json(schema.validation_file(label), {
            "run_id": ctx.run_id,
            "label": label,
            "imbalance_level": level,
            "model_path": model_path,
            "router_id": 0,
            "n_experts": n_experts,
            "n_routers": n_layers,
            "k": k,
            **layer0,
            "per_router_frequencies": per_layer,
            "prompts": VALIDATION_PROMPTS,
            "responses": [r.get("response") for r in records],
        })
        max_share = float(max(layer0["frequencies"])) if layer0["n_assignments"] else None
        ctx.update_point(label, validation_file=rel, validation_summary={
            "n_assignments": layer0["n_assignments"],
            "max_expert_frequency": max_share,
            "expected_frequency": 1 / n_experts,
        })


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
    lines = [f"checkpoints: {checkpoint_path(cfg, level)}" for level in levels]
    if cfg.imbalance.validate_imbalance:
        lines.append("validate router load of every checkpoint")
    lines.append(f"benchmark {len(levels)} imbalance levels {levels} with {cfg.benchmark.n_samples} MMLU prompts"
                 f"{common.profiling_note(cfg.benchmark)}")
    return lines


def sweep_values(cfg: ExperimentConfig) -> list[float | int]:
    return list(cfg.imbalance.imbalance_levels)

