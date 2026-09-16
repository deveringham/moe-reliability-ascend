###
# grid.py
#
# Defines grid files which define parameter sweeps.
# A grid file names a base configuration and the values to sweep::
#
# [grid]
# name = "infra-matrix" # required; prefixes experiment.name of every run
# base = "../examples/forced_imbalance.toml" # relative to the grid file
# share_workloads = true # synthetic workloads: capture/build once per model
#
# [set] # applied to every run
# "benchmark.n_samples" = 2000
#
# [matrix] # Cartesian product, in file order
# "hardware.n_npus" = [2, 4, 8]
# "server.batch_size" = [128, 256, 512]
# comma-separated keys are varied together (zipped):
# "model.model_id,model.model_name" = [["deepseek-ai/DeepSeek-V2-Lite-Chat", "deepseek-v2"],
#                                         ["Qwen/Qwen1.5-MoE-A2.7B-Chat", "qwen"]]
#
# Dylan Everingham
# 16.09.2026
###

from __future__ import annotations

import copy
import itertools
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from moe_reliability_results import Run, schema
from moe_reliability_results.store import ResultsStore

from .config import ConfigError, ExperimentConfig, apply_overrides

__all__ = ["Grid", "GridEntry", "load_grid", "find_run_by_name", "workload_share_key"]


@dataclass
class GridEntry:
    index: int
    name: str
    assignments: dict[str, Any]
    config: ExperimentConfig


@dataclass
class Grid:
    name: str
    path: Path | None
    base: dict[str, Any]
    fixed: dict[str, Any]
    matrix: list[tuple[list[str], list[Any]]]
    share_workloads: bool = False
    entries: list[GridEntry] = field(default_factory=list)

    @property
    def n_runs(self) -> int:
        return len(self.entries)

    def n_configurations(self) -> int | None:
        from .pipelines import pipeline_for

        total = 0
        for e in self.entries:
            values = pipeline_for(e.config.experiment_type).sweep_values(e.config)
            if values is None:
                return None
            total += len(values)
        return total


def _set_dotted(data: dict[str, Any], dotted: str, value: Any) -> None:
    if "." not in dotted:
        raise ConfigError(f"grid key {dotted!r} must have the form section.key")
    section, key = dotted.split(".", 1)
    table = data.setdefault(section, {})
    if not isinstance(table, dict):
        raise ConfigError(f"grid key {dotted!r}: [{section}] is not a table")
    table[key] = copy.deepcopy(value)


def _parse_matrix(raw: dict[str, Any]) -> list[tuple[list[str], list[Any]]]:
    matrix = []
    for spec, values in raw.items():
        keys = [k.strip() for k in spec.split(",") if k.strip()]
        if not isinstance(values, list) or not values:
            raise ConfigError(f"[matrix] {spec!r} must be a non-empty list")
        if len(keys) > 1:
            for v in values:
                if not isinstance(v, list) or len(v) != len(keys):
                    raise ConfigError(f"[matrix] {spec!r}: every value must be a list of {len(keys)} items")
        matrix.append((keys, values))
    return matrix

# Parse grid file, applies override values, and validates
def load_grid(path: str | Path, overrides: Iterable[str] = ()) -> Grid:
    path = Path(path)
    try:
        with open(path, "rb") as f:
            doc = tomllib.load(f)
    except FileNotFoundError:
        raise ConfigError(f"grid file not found: {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from None

    unknown = sorted(set(doc) - {"grid", "set", "matrix"})
    if unknown:
        raise ConfigError(f"{path}: unknown top-level table(s) {unknown}; expected [grid], [set], [matrix]")
    meta = doc.get("grid") or {}
    name = meta.get("name")
    if not name or not isinstance(name, str):
        raise ConfigError(f"{path}: grid.name is required")
    base_ref = meta.get("base")
    if not base_ref:
        raise ConfigError(f"{path}: grid.base (path of the base configuration) is required")
    base_path = (path.parent / base_ref) if not Path(base_ref).is_absolute() else Path(base_ref)
    try:
        with open(base_path, "rb") as f:
            base = tomllib.load(f)
    except FileNotFoundError:
        raise ConfigError(f"{path}: base configuration not found: {base_path}") from None

    grid = Grid(name=name, path=path, base=base, fixed=dict(doc.get("set") or {}),
                matrix=_parse_matrix(doc.get("matrix") or {}),
                share_workloads=bool(meta.get("share_workloads", False)))
    grid.entries = expand(grid, overrides)
    return grid


def expand(grid: Grid, overrides: Iterable[str] = ()) -> list[GridEntry]:
    overrides = list(overrides)
    entries = []
    axes = [values for _, values in grid.matrix] or [[None]]
    width = max(3, len(str(max(1, len(list(itertools.product(*axes)))))))
    for index, combo in enumerate(itertools.product(*axes)):
        data = copy.deepcopy(grid.base)
        for dotted, value in grid.fixed.items():
            _set_dotted(data, dotted, value)
        assignments: dict[str, Any] = {}
        if grid.matrix:
            for (keys, _), value in zip(grid.matrix, combo):
                items = zip(keys, value) if len(keys) > 1 else [(keys[0], value)]
                for key, v in items:
                    _set_dotted(data, key, v)
                    assignments[key] = v
        run_name = f"{grid.name}-{index:0{width}d}"
        _set_dotted(data, "experiment.name", run_name)
        data = apply_overrides(data, overrides)
        try:
            cfg = ExperimentConfig(data)
        except ConfigError as exc:
            raise ConfigError(f"grid entry {index} ({assignments}): {exc}") from None
        entries.append(GridEntry(index=index, name=run_name, assignments=assignments, config=cfg))
    names = [e.name for e in entries]
    if len(set(names)) != len(names):  # pragma: no cover - names are index based
        raise ConfigError("duplicate run names in grid")
    return entries

# Get most recent run of experiment_type with a given name
def find_run_by_name(results_dir: str | Path, experiment_type: str, name: str) -> Run | None:
    runs = ResultsStore(results_dir).runs(experiment=experiment_type, name=name)
    return runs[-1] if runs else None


def workload_share_key(cfg: ExperimentConfig) -> tuple | None:
    if cfg.experiment_type != schema.EXPERIMENT_SYNTHETIC_WORKLOADS:
        return None
    if cfg.workloads.reuse_workloads_from or cfg.activations.reuse_activations_from:
        return None
    return (cfg.model.model_id, cfg.model.enable_bnb, cfg.experiment.seed, cfg.client.max_new_tokens,
            cfg.activations.n_samples, tuple(cfg.workloads.target_alphas),
            tuple(cfg.workloads.target_prompt_lengths), tuple(cfg.workloads.max_repeats))
