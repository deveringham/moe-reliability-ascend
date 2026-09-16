###
# io.py
#
# Dylan Everingham
# 16.09.2026
###

from __future__ import annotations

import gzip
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

__all__ = [
    "to_jsonable",
    "write_json",
    "read_json",
    "write_jsonl",
    "iter_jsonl",
    "resolve_path",
    "exists",
]


def to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {_key_to_str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    # NumPy scalars expose item(); arrays and torch tensors expose tolist().
    if hasattr(obj, "tolist") and hasattr(obj, "shape"):
        if callable(getattr(obj, "detach", None)) and callable(getattr(obj, "cpu", None)):
            obj = obj.detach().cpu()  # torch.Tensor on any device
        return to_jsonable(obj.tolist())
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return to_jsonable(obj.item())
        except (TypeError, ValueError):
            pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def _key_to_str(key: Any) -> str:
    if isinstance(key, str):
        return key
    if hasattr(key, "item"):
        key = key.item()
    if isinstance(key, (tuple, list)):
        return ",".join(str(k) for k in key)
    return str(key)


def resolve_path(path: str | os.PathLike) -> Path:
    p = Path(path)
    if p.exists():
        return p
    if p.suffix == ".gz":
        plain = p.with_suffix("")
        if plain.exists():
            return plain
    else:
        gz = p.with_name(p.name + ".gz")
        if gz.exists():
            return gz
    raise FileNotFoundError(f"No such file (plain or gzip): {p}")


def exists(path: str | os.PathLike) -> bool:
    try:
        resolve_path(path)
        return True
    except FileNotFoundError:
        return False


def _open_text(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def _target(path: str | os.PathLike, compress: bool) -> Path:
    p = Path(path)
    if p.suffix == ".gz":
        p = p.with_suffix("")
    return p.with_name(p.name + ".gz") if compress else p


def write_json(path: str | os.PathLike, obj: Any, compress: bool = False, indent: int | None = 2) -> Path:
    target = _target(path, compress)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    payload = to_jsonable(obj)
    if compress:
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(payload, f, allow_nan=False, separators=(",", ":"))
    else:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, allow_nan=False, indent=indent)
            f.write("\n")
    os.replace(tmp, target)
    # Remove a stale sibling with the other compression setting.
    other = _target(path, not compress)
    if other.exists():
        other.unlink()
    return target


def read_json(path: str | os.PathLike) -> Any:
    with _open_text(resolve_path(path), "r") as f:
        return json.load(f)


def write_jsonl(path: str | os.PathLike, records: Iterable[Any], compress: bool = False) -> tuple[Path, int]:
    target = _target(path, compress)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    n = 0
    opener = (lambda: gzip.open(tmp, "wt", encoding="utf-8")) if compress else (lambda: open(tmp, "w", encoding="utf-8"))
    with opener() as f:
        for record in records:
            f.write(json.dumps(to_jsonable(record), allow_nan=False, separators=(",", ":")))
            f.write("\n")
            n += 1
    os.replace(tmp, target)
    other = _target(path, not compress)
    if other.exists():
        other.unlink()
    return target, n


def iter_jsonl(path: str | os.PathLike) -> Iterator[Any]:
    with _open_text(resolve_path(path), "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
