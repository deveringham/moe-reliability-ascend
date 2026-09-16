###
# runs.py
#
# Creation of run directories and artefact management.
# manifest.json rewritten atomically after every state change so
# interrupted runs can be resumed.
# Dylan Everingham
# 16.09.2026
###

from __future__ import annotations

import datetime as _dt
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from moe_reliability_results import io, schema

from .config import ExperimentConfig

__all__ = ["RunContext", "RunError", "make_run_id", "resolve_run_dir", "utcnow"]


class RunError(RuntimeError):
    """Raised for invalid run references or inconsistent run state."""


def utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def make_run_id(cfg: ExperimentConfig, when: _dt.datetime | None = None) -> str:
    when = when or _dt.datetime.now(_dt.timezone.utc)
    short = {schema.EXPERIMENT_SYNTHETIC_WORKLOADS: "synthetic",
             schema.EXPERIMENT_FORCED_IMBALANCE: "imbalance"}[cfg.experiment_type]
    parts = [when.strftime("%Y%m%d-%H%M%S"), short, cfg.model.model_name,
             f"npu{cfg.hardware.n_npus}", f"bs{cfg.server.batch_size}"]
    if cfg.experiment.name:
        parts.append(cfg.experiment.name)
    return "_".join(parts)


def resolve_run_dir(ref: str | Path, results_dir: str | Path) -> Path:
    """Resolve a run id, unique run id prefix or run directory path."""
    path = Path(ref)
    if (path / schema.MANIFEST_FILE).is_file():
        return path.resolve()
    root = Path(results_dir)
    if (root / str(ref) / schema.MANIFEST_FILE).is_file():
        return (root / str(ref)).resolve()
    matches = sorted(p for p in root.glob(f"{ref}*") if (p / schema.MANIFEST_FILE).is_file()) if root.is_dir() else []
    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        raise RunError(f"no run matching {str(ref)!r} (looked in {root.resolve()})")
    raise RunError(f"ambiguous run reference {str(ref)!r}: {[m.name for m in matches]}")

# State of one run directory
class RunContext:

    def __init__(self, path: Path, manifest: dict[str, Any]):
        self.path = Path(path)
        self.manifest = manifest

    @classmethod
    def create(cls, cfg: ExperimentConfig, run_id: str | None = None,
               environment: dict[str, Any] | None = None) -> "RunContext":
        results_dir = Path(cfg.output.results_dir)
        run_id = run_id or make_run_id(cfg)
        path = results_dir / run_id
        suffix = 1
        while path.exists():
            suffix += 1
            path = results_dir / f"{run_id}-{suffix}"
        path.mkdir(parents=True)
        now = utcnow()
        manifest: dict[str, Any] = {
            "schema_version": schema.SCHEMA_VERSION,
            "run_id": path.name,
            "experiment": cfg.experiment_type,
            "sweep_parameter": schema.SWEEP_PARAMETERS[cfg.experiment_type],
            "status": schema.STATUS_PENDING,
            "created_at": now,
            "updated_at": now,
            "description": cfg.experiment.description,
            "config": cfg.to_dict(),
            "environment": environment or {},
            "inputs": {},
            "stages": {},
            "points": [],
        }
        (path / schema.CONFIG_FILE).write_text(
            cfg.to_toml(header=f"Resolved configuration of run {path.name}"), encoding="utf-8")
        ctx = cls(path, manifest)
        ctx.save()
        return ctx

    @classmethod
    def open(cls, ref: str | Path, results_dir: str | Path = "results") -> "RunContext":
        path = resolve_run_dir(ref, results_dir)
        return cls(path, io.read_json(path / schema.MANIFEST_FILE))

    @property
    def run_id(self) -> str:
        return self.manifest["run_id"]

    def config(self) -> ExperimentConfig:
        return ExperimentConfig.from_dict(self.manifest["config"])

    def save(self) -> None:
        self.manifest["updated_at"] = utcnow()
        io.write_json(self.path / schema.MANIFEST_FILE, self.manifest)
        
    @property
    def compress(self) -> bool:
        return bool(self.manifest["config"]["output"]["compress"])

    def abspath(self, relative: str) -> Path:
        return self.path / relative

    def relpath(self, path: Path) -> str:
        return Path(path).resolve().relative_to(self.path.resolve()).as_posix()

    def write_json(self, relative: str, obj: Any, compress: bool | None = None) -> str:
        written = io.write_json(self.abspath(relative), obj, compress=self.compress if compress is None else compress)
        return self.relpath(written)

    def write_jsonl(self, relative: str, records: Iterable[Any], compress: bool | None = None) -> tuple[str, int]:
        written, n = io.write_jsonl(self.abspath(relative), records,
                                    compress=self.compress if compress is None else compress)
        return self.relpath(written), n
        
    def stage_status(self, name: str) -> str:
        return self.manifest["stages"].get(name, {}).get("status", schema.STATUS_PENDING)

    def init_stages(self, names: Iterable[str]) -> None:
        for name in names:
            self.manifest["stages"].setdefault(name, {"status": schema.STATUS_PENDING})
        self.save()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        entry = self.manifest["stages"].setdefault(name, {})
        entry.update(status=schema.STATUS_RUNNING, started_at=utcnow(), finished_at=None, error=None, note=None)
        self.manifest["status"] = schema.STATUS_RUNNING
        self.save()
        try:
            yield
        except KeyboardInterrupt:
            entry.update(status=schema.STATUS_INTERRUPTED, finished_at=utcnow(), error="interrupted by user")
            self.save()
            raise
        except BaseException:
            entry.update(status=schema.STATUS_FAILED, finished_at=utcnow(), error=traceback.format_exc())
            self.save()
            raise
        else:
            entry.update(status=schema.STATUS_COMPLETED, finished_at=utcnow())
            self.save()

    def skip_stage(self, name: str, reason: str) -> None:
        entry = self.manifest["stages"].setdefault(name, {})
        entry.update(status=schema.STATUS_SKIPPED, note=reason, started_at=None, finished_at=utcnow(), error=None)
        self.save()
        
    def ensure_points(self, values: list[float | int]) -> None:
        """Create the sweep points, or check them against an existing manifest."""
        parameter = self.manifest["sweep_parameter"]
        labels = [schema.point_label(parameter, v) for v in values]
        existing = [p["label"] for p in self.manifest["points"]]
        if existing:
            if existing != labels:
                raise RunError(f"sweep points of run {self.run_id} ({existing}) differ from the requested "
                               f"points ({labels})")
            return
        self.manifest["points"] = [
            {"index": i, "value": v, "label": label, "status": schema.STATUS_PENDING}
            for i, (v, label) in enumerate(zip(values, labels))
        ]
        self.save()

    @property
    def points(self) -> list[dict[str, Any]]:
        return self.manifest["points"]

    def point(self, label: str) -> dict[str, Any]:
        for p in self.manifest["points"]:
            if p["label"] == label:
                return p
        raise RunError(f"run {self.run_id} has no sweep point {label!r}")

    def update_point(self, label: str, **fields: Any) -> dict[str, Any]:
        p = self.point(label)
        p.update(fields)
        self.save()
        return p
        
    def finalize(self, error: BaseException | None = None) -> str:
        """Derive and store the overall run status from its stages and points."""
        stages = [s.get("status") for s in self.manifest["stages"].values()]
        if isinstance(error, KeyboardInterrupt) or schema.STATUS_INTERRUPTED in stages:
            status = schema.STATUS_INTERRUPTED
        elif error is not None or schema.STATUS_FAILED in stages:
            status = schema.STATUS_FAILED
        elif all(s in (schema.STATUS_COMPLETED, schema.STATUS_SKIPPED) for s in stages):
            failed_points = any(p.get("status") == schema.STATUS_FAILED for p in self.manifest["points"])
            status = schema.STATUS_PARTIAL if failed_points else schema.STATUS_COMPLETED
        else:
            status = schema.STATUS_PARTIAL
        if error is not None:
            self.manifest["error"] = "".join(traceback.format_exception_only(type(error), error)).strip()
        elif status in (schema.STATUS_COMPLETED, schema.STATUS_PARTIAL):
            self.manifest.pop("error", None)
        self.manifest["status"] = status
        self.manifest["finished_at"] = utcnow()
        self.save()
        return status
