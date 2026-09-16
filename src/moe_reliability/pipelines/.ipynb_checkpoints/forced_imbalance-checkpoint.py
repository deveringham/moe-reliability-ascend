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

from moe_results import schema

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

def _expert_load(probe, router_id: int) -> dict:
    import numpy as np

    active_experts = probe.get_active_experts()  # [batch, padded_seq_len, k, n_routers]
    active_experts = active_experts[:, :, :, router_id].flatten().cpu().tolist()

    counts = Counter(active_experts)
    expert_ids = np.array(range(probe.n_experts))
    count_per_expert = np.array([counts[i] for i in expert_ids])
    tokens = sum(counts.values())
    freqs = count_per_expert / tokens
    return {"counts": count_per_expert, "n_assignments": tokens, "frequencies": freqs}


def validate_checkpoints(ctx: RunContext, cfg: ExperimentConfig) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from ..core.hf_models import chat_generate, load_model
    from ..models import probe_class, resolve_probe_family

    family = resolve_probe_family(cfg.model.model_id, cfg.model.probe)
    enable_bnb = cfg.model.enable_bnb
    max_new_tokens = cfg.client.max_new_tokens
    prompts = VALIDATION_PROMPTS

    for p in ctx.points:
        label, level = p["label"], p["value"]
        if p.get("validation_file"):
            continue
        model_path = p.get("model_path") or checkpoint_path(cfg, level)
        log(f"{label}: validating router load of {model_path}")

        if level == 0:
            model, tokenizer = load_model(model_path, enable_bnb=enable_bnb)
        else:
            if enable_bnb:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                model = AutoModelForCausalLM.from_pretrained(model_path,
                                                             device_map="auto",
                                                             quantization_config=quantization_config)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
            tokenizer = AutoTokenizer.from_pretrained(model_path)

        probe = probe_class(family)(model)
        responses = []
        for prompt in prompts:
            response, probs, active_experts = chat_generate(model, tokenizer, probe,
                                                            prompt=prompt, max_new_tokens=max_new_tokens,
                                                            clear_probe=False,
                                                            prompt_formatted=False)
            responses.append(response)

        # Clean up memory
        del model
        common.free_accelerator_memory()

        router0 = _expert_load(probe, router_id=0)
        per_router = [_expert_load(probe, router_id=r)["frequencies"] for r in range(probe.n_routers)]
        rel = ctx.write_json(schema.validation_file(label), {
            "run_id": ctx.run_id,
            "label": label,
            "imbalance_level": level,
            "model_path": model_path,
            "router_id": 0,
            "n_experts": probe.n_experts,
            "n_routers": probe.n_routers,
            "k": probe.k,
            **router0,
            "per_router_frequencies": per_router,
            "prompts": prompts,
            "responses": responses,
        })
        max_share = float(max(router0["frequencies"])) if router0["n_assignments"] else None
        ctx.update_point(label, validation_file=rel, validation_summary={
            "n_assignments": router0["n_assignments"],
            "max_expert_frequency": max_share,
            "expected_frequency": 1 / probe.n_experts,
        })
        del probe
        torch.npu.empty_cache()

def run(ctx: RunContext, cfg: ExperimentConfig, retry_failed: bool = False) -> None:
    ctx.init_stages(STAGES)
    ctx.ensure_points(cfg.imbalance.imbalance_levels)

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
                 f"{' (profiled)' if cfg.benchmark.enable_profiling else ''}")
    return lines


def sweep_values(cfg: ExperimentConfig) -> list[float | int]:
    return list(cfg.imbalance.imbalance_levels)

