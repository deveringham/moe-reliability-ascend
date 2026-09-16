###
# logs.py
#
# Captures stdout/stderr, including vLLM server output, and
# writes to a log file in the run directory.
#
# Dylan Everingham
# 16.09.2026
###

from __future__ import annotations

import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

__all__ = ["tee_output", "log"]


def log(message: str) -> None:
    print(f"[moe-experiments] {message}", flush=True)


@contextmanager
def tee_output(path: str | os.PathLike, enabled: bool = True) -> Iterator[Path | None]:
    if not enabled:
        yield None
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        tee = subprocess.Popen(["tee", "-a", str(path)], stdin=subprocess.PIPE)
    except OSError:
        yield None  # `tee` unavailable: keep console output only
        return
    saved_out, saved_err = os.dup(1), os.dup(2)
    os.dup2(tee.stdin.fileno(), 1)
    os.dup2(tee.stdin.fileno(), 2)
    try:
        yield path
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        tee.stdin.close()
        tee.wait()
