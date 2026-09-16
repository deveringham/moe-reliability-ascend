###
# __init__.py
#
# Experiment orchestration. Each type exposes a list of stages,
# run(), plan() (gives a human-readable experiment plan),
# and sweep_values()
#
# Dylan Everingham
# 16.09.2026
###

from __future__ import annotations

from types import ModuleType

from moe_results import schema

from ..config import ExperimentConfig
from ..runs import RunContext

__all__ = ["pipeline_for", "run_pipeline", "analyze"]


def pipeline_for(experiment_type: str) -> ModuleType:
    if experiment_type == schema.EXPERIMENT_SYNTHETIC_WORKLOADS:
        from . import synthetic_workloads
        return synthetic_workloads
    if experiment_type == schema.EXPERIMENT_FORCED_IMBALANCE:
        from . import forced_imbalance
        return forced_imbalance
    raise ValueError(f"unknown experiment type {experiment_type!r}")


def run_pipeline(ctx: RunContext, cfg: ExperimentConfig, retry_failed: bool = False) -> str:
    """Run (or resume) all stages of a run. Returns the final run status."""
    try:
        pipeline_for(cfg.experiment_type).run(ctx, cfg, retry_failed=retry_failed)
    except BaseException as exc:
        ctx.finalize(error=exc)
        raise
    return ctx.finalize()


def analyze(ctx: RunContext, cfg: ExperimentConfig) -> str:
    """Re-run trace analysis, HTA and figure rendering on an existing run."""
    from .common import post_processing_stages

    try:
        post_processing_stages(ctx, cfg, force=True)
    except BaseException as exc:
        ctx.finalize(error=exc)
        raise
    return ctx.finalize()
