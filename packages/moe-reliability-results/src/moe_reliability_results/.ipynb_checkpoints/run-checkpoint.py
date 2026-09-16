###
# run.py
#
# Access to info on one run: manifest, config, requests, traces,
# HTA, workloads, activations, validation, summary.
#
# Dylan Everingham
# 16.09.2026
###

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd

from . import io, schema
from .metrics import summarize_requests, trace_scalars

__all__ = ["Run", "flatten_config", "decode_workloads", "decode_activation_record"]

_NON_TABULAR_SECTIONS = ("environment",)


def flatten_config(config: Mapping[str, Mapping[str, Any]], scalars_only: bool = True) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for section, values in config.items():
        if section in _NON_TABULAR_SECTIONS or not isinstance(values, Mapping):
            continue
        for key, value in values.items():
            if scalars_only and isinstance(value, (list, tuple, dict)):
                continue
            flat[key] = value
    return flat


def decode_workloads(doc: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(doc)
    out["cv_nat"] = np.asarray(doc["cv_nat"], dtype=float)
    workloads: dict[int, dict[float, dict[str, Any]]] = {}
    for length, by_alpha in doc["workloads"].items():
        workloads[int(length)] = {
            float(alpha): {**w, "obtained_cvs": np.asarray(w["obtained_cvs"], dtype=float)}
            for alpha, w in by_alpha.items()
        }
    out["workloads"] = workloads
    return out


def decode_activation_record(record: dict[str, Any]) -> dict[str, Any]:
    for key in ("routed_experts", "prompt_routed_experts"):
        if record.get(key) is not None:
            record[key] = np.asarray(record[key], dtype=np.int16)
    return record


# A single stored run: manifest, configuration, and all artefacts.
class Run:

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not (self.path / schema.MANIFEST_FILE).exists():
            raise FileNotFoundError(f"Not a run directory (no {schema.MANIFEST_FILE}): {self.path}")
        self._manifest: dict[str, Any] | None = None

    @property
    def manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            self._manifest = io.read_json(self.path / schema.MANIFEST_FILE)
        return self._manifest

    def reload(self) -> "Run":
        self._manifest = None
        return self

    @property
    def id(self) -> str:
        return self.manifest["run_id"]

    @property
    def experiment(self) -> str:
        return self.manifest["experiment"]

    @property
    def status(self) -> str:
        return self.manifest.get("status", schema.STATUS_PENDING)

    @property
    def created_at(self) -> str:
        return self.manifest.get("created_at", "")

    @property
    def config(self) -> dict[str, dict[str, Any]]:
        return self.manifest["config"]

    @property
    def flat_config(self) -> dict[str, Any]:
        return flatten_config(self.config)

    @property
    def sweep_parameter(self) -> str:
        return self.manifest.get("sweep_parameter", schema.SWEEP_PARAMETERS.get(self.experiment, "value"))

    @property
    def stages(self) -> dict[str, dict[str, Any]]:
        return self.manifest.get("stages", {})

    @property
    def points(self) -> list[dict[str, Any]]:
        return self.manifest.get("points", [])

    @property
    def sweep_values(self) -> list[float | int]:
        return [p["value"] for p in self.points]

    def point(self, key: float | int | str) -> dict[str, Any]:
        """Look up a sweep point by value (``0.8``) or label (``"alpha_0.8"``)."""
        for p in self.points:
            if p["label"] == key or (not isinstance(key, str) and p["value"] == key):
                return p
        raise KeyError(f"Run {self.id} has no sweep point {key!r}")

    def file(self, relative: str) -> Path:
        return self.path / relative

    def __repr__(self) -> str:
        return f"Run(id={self.id!r}, experiment={self.experiment!r}, status={self.status!r}, points={len(self.points)})"

    def request_records(self, key: float | int | str) -> list[dict[str, Any]] | None:
        p = self.point(key)
        if not p.get("metrics_file"):
            return None
        return io.read_json(self.path / p["metrics_file"])["requests"]

    def results_by_point(self, include_failed: bool = False) -> dict[float | int, list[dict[str, Any]] | None]:
        out: dict[float | int, list[dict[str, Any]] | None] = {}
        for p in self.points:
            if p.get("metrics_file"):
                out[p["value"]] = self.request_records(p["label"])
            elif include_failed and p.get("status") == schema.STATUS_FAILED:
                out[p["value"]] = None
        return out

    def requests(self, values: list[float | int] | None = None, include_prompts: bool = False) -> pd.DataFrame:
        frames = []
        for p in self.points:
            if values is not None and p["value"] not in values:
                continue
            records = self.request_records(p["label"]) if p.get("metrics_file") else None
            if not records:
                continue
            df = pd.DataFrame.from_records(records)
            if not include_prompts:
                df = df.drop(columns=[c for c in ("prompt", "response") if c in df.columns])
            df = df.rename(columns={"ttft": "ttft_s", "tpot": "tpot_s", "total_time": "total_time_s"})
            df["ttft_ms"] = df["ttft_s"] * 1000.0
            df["tpot_ms"] = df["tpot_s"] * 1000.0
            df.insert(0, "sweep_value", p["value"])
            df.insert(0, "sweep_parameter", self.sweep_parameter)
            df.insert(0, "experiment", self.experiment)
            df.insert(0, "run_id", self.id)
            frames.append(df)
        if not frames:
            return pd.DataFrame(columns=["run_id", "experiment", "sweep_parameter", "sweep_value"])
        return pd.concat(frames, ignore_index=True)

    def trace_summary(self, key: float | int | str) -> dict[str, Any] | None:
        """Fused-MoE kernel metrics for one sweep point (``None`` if not analysed)."""
        p = self.point(key)
        if not p.get("trace_metrics_file"):
            return None
        return io.read_json(self.path / p["trace_metrics_file"])

    def trace_summaries(self) -> dict[float | int, dict[str, Any]]:
        return {p["value"]: self.trace_summary(p["label"]) for p in self.points if p.get("trace_metrics_file")}

    def hta_frames(self) -> dict[str, pd.DataFrame] | None:
        """Holistic Trace Analysis tables (``rank``, ``idle_categories``, ``kernel_types``, ``runs``)."""
        rel = self.manifest.get("hta_file")
        if not rel:
            return None
        doc = io.read_json(self.path / rel)
        return {name: pd.DataFrame.from_records(rows) for name, rows in doc["tables"].items()}

    def _source_run(self, kind: str) -> "Run":
        source = (self.manifest.get("inputs") or {}).get(kind)
        if not source:
            return self
        sibling = self.path.parent / source["run_id"]
        if (sibling / schema.MANIFEST_FILE).exists():
            return Run(sibling)
        return Run(source["path"])

    def available_workloads(self) -> list[int]:
        src = self._source_run("workloads")
        return sorted(int(k) for k in (src.manifest.get("workloads") or {}))

    def workloads(self, max_repeats: int | None = None) -> dict[str, Any]:
        src = self._source_run("workloads")
        sets = src.manifest.get("workloads") or {}
        if not sets:
            raise KeyError(f"Run {self.id} has no synthetic workloads")
        if max_repeats is None:
            max_repeats = int((self.config.get("benchmark") or {}).get("workload_max_repeats", sorted(sets, key=int)[0]))
        entry = sets.get(str(int(max_repeats)))
        if entry is None:
            raise KeyError(f"No workloads with max_repeats={max_repeats}; available: {sorted(int(k) for k in sets)}")
        return decode_workloads(io.read_json(src.path / entry["file"]))

    def activations(self, limit: int | None = None) -> Iterator[dict[str, Any]]:
        src = self._source_run("activations")
        entry = src.manifest.get("activations")
        if not entry:
            raise KeyError(f"Run {self.id} has no activation records")
        for i, record in enumerate(io.iter_jsonl(src.path / entry["file"])):
            if limit is not None and i >= limit:
                break
            yield decode_activation_record(record)

    def validation(self, key: float | int | str) -> dict[str, Any] | None:
        p = self.point(key)
        if not p.get("validation_file"):
            return None
        return io.read_json(self.path / p["validation_file"])

    def summary(self) -> pd.DataFrame:
        base = {
            "run_id": self.id,
            "experiment": self.experiment,
            "run_status": self.status,
            "created_at": self.created_at,
            "sweep_parameter": self.sweep_parameter,
        }
        flat = self.flat_config
        rows = []
        for p in self.points:
            row: dict[str, Any] = dict(base)
            row.update({"sweep_value": p["value"], "label": p["label"], "point_status": p.get("status")})
            row.update(flat)
            req = p.get("request_summary")
            if req is None and p.get("metrics_file"):
                req = summarize_requests(self.request_records(p["label"]))
            row.update(req or {})
            trace = p.get("trace_summary")
            if trace is None and p.get("trace_metrics_file"):
                trace = trace_scalars(self.trace_summary(p["label"]))
            row.update({f"trace_{k}": v for k, v in (trace or {}).items()})
            row.update({f"workload_{k}": v for k, v in (p.get("workload") or {}).items()})
            rows.append(row)
        return pd.DataFrame.from_records(rows)

    def recompute_point_summaries(self) -> None:
        for p in self.points:
            if p.get("metrics_file"):
                p["request_summary"] = summarize_requests(self.request_records(p["label"]))
            if p.get("trace_metrics_file"):
                p["trace_summary"] = trace_scalars(self.trace_summary(p["label"]))

