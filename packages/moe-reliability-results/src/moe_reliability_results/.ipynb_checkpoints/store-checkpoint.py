###
# Interface for queries over runs in a results directory.
#
# Methods accepting **filters select runs by top-level attributes
# (ex. run_id, experiment, status) or by any scalar configuration key
# (ex. model_name, n_npus, batch_size, enable_eplb).
# A filter value may be
#  a scalar - equality,
#  a list / tuple / set - membership,
#  a callable - predicate, e.g. batch_size=lambda b: b >= 256.
# Runs that do not define a filtered key are excluded.
#
# Examples:
# store = ResultsStore("results")
# store.summary(experiment="forced_imbalance", n_npus=8)
# store.query("sweep_value > 1.0 and tpot_ms_mean > 40", model_name="deepseek-v2")
# 
# Dylan Everingham
# 16.09.2026
###

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Iterator

import pandas as pd

from . import schema
from .run import Run

__all__ = ["ResultsStore", "default_results_dir"]


def default_results_dir() -> Path:
    return Path(os.environ.get("MOE_RESULTS_DIR", "results"))


def _matches(actual: Any, expected: Any) -> bool:
    if callable(expected):
        return bool(expected(actual))
    if isinstance(expected, (list, tuple, set, frozenset)):
        return actual in expected
    return actual == expected

# All runs stored in a results directory
class ResultsStore:

    def __init__(self, root: str | os.PathLike | None = None):
        self.root = Path(root) if root is not None else default_results_dir()
        
    def run_dirs(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        return sorted(p for p in self.root.iterdir() if (p / schema.MANIFEST_FILE).is_file())

    def __iter__(self) -> Iterator[Run]:
        return iter(self.runs())

    def __len__(self) -> int:
        return len(self.run_dirs())

    def __repr__(self) -> str:
        return f"ResultsStore(root={str(self.root)!r}, runs={len(self)})"

    def runs(self, **filters: Any) -> list[Run]:
        selected = []
        for path in self.run_dirs():
            try:
                run = Run(path)
                run.manifest  # parse now so that unreadable manifests are skipped
            except (OSError, ValueError, KeyError):
                continue  # unreadable or partially written manifest
            attrs = {"run_id": run.id, "experiment": run.experiment, "status": run.status}
            attrs.update(run.flat_config)
            if all(key in attrs and _matches(attrs[key], value) for key, value in filters.items()):
                selected.append(run)
        return sorted(selected, key=lambda r: (r.created_at, r.id))

    def get(self, ref: str | os.PathLike) -> Run:
        path = Path(ref)
        if (path / schema.MANIFEST_FILE).is_file():
            return Run(path)
        exact = self.root / str(ref)
        if (exact / schema.MANIFEST_FILE).is_file():
            return Run(exact)
        candidates = [p for p in self.run_dirs() if p.name.startswith(str(ref))]
        if len(candidates) == 1:
            return Run(candidates[0])
        if not candidates:
            raise KeyError(f"No run matching {ref!r} in {self.root}")
        raise KeyError(f"Ambiguous run reference {ref!r}: {[p.name for p in candidates]}")

    def latest(self, **filters: Any) -> Run:
        runs = self.runs(**filters)
        if not runs:
            raise KeyError(f"No runs matching {filters}")
        return runs[-1]

    def summary(self, **filters: Any) -> pd.DataFrame:
        frames = [run.summary() for run in self.runs(**filters)]
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame(columns=["run_id", "experiment", "sweep_parameter", "sweep_value"])
        return pd.concat(frames, ignore_index=True)

    def query(self, expr: str, **filters: Any) -> pd.DataFrame:
        return self.summary(**filters).query(expr).reset_index(drop=True)

    def requests(self, **filters: Any) -> pd.DataFrame:
        frames = [run.requests() for run in self.runs(**filters)]
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame(columns=["run_id", "experiment", "sweep_parameter", "sweep_value"])
        return pd.concat(frames, ignore_index=True)

    def configurations(self, **filters: Any) -> pd.DataFrame:
        rows = []
        for run in self.runs(**filters):
            rows.append({"run_id": run.id, "experiment": run.experiment, "status": run.status,
                         "created_at": run.created_at, "n_points": len(run.points), **run.flat_config})
        return pd.DataFrame.from_records(rows)

    def infrastructure_configurations(self, completed_only: bool = True, **filters: Any) -> pd.DataFrame:
        
        df = self.summary(**filters)
        if df.empty:
            return df
        if completed_only:
            df = df[df["point_status"] == schema.STATUS_COMPLETED]
        keys = ["experiment", *[k for k in schema.INFRASTRUCTURE_KEYS if k in df.columns],
                "sweep_parameter", "sweep_value"]
        return (df.groupby(keys, dropna=False)
                  .agg(n_runs=("run_id", "nunique"))
                  .reset_index())

    def group_metric(self, metric: str, by: list[str], agg: str | Callable = "mean", **filters: Any) -> pd.DataFrame:
        
        df = self.summary(**filters)
        if df.empty:
            return df
        return df.groupby(by, dropna=False)[metric].agg(agg).reset_index()
