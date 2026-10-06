"""Split profiled traces into engine steps and write one row per step.

    uv run python scripts/step_profile.py <run_dir> [<run_dir> ...]

Runs on the node, where the raw traces live. For every profiled point of a run
it writes ``<run_dir>/step_profile/<point>.csv.gz``: one row per step and rank,
small enough to pull and analyse anywhere.

A step starts at its ``_compute_slot_mapping_kernel`` launch, whose input
shapes carry the step's request and token counts (``"<reqs>;<tokens>;..."``).
A step with as many tokens as requests is decode-only; one with more carries
prefill chunks. Window averages mix the two kinds in different proportions
from point to point, which is why they disagree with TPOT; comparing like
steps does not.

Columns: step, rank, reqs, tokens, start_us, wall_us (to the next step's
start on the same rank), busy_us (summed kernel time), gemm_us
(GroupedMatmul), collective_us (hcom_allReduce_), idle_us (wall_us minus
time covered by kernels).
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

from moe_reliability.core.trace_analysis import ascend_rank_outputs

STEP_KERNEL = "_compute_slot_mapping_kernel"
GEMM = "GroupedMatmul"
COLLECTIVE = "hcom_allReduce_"
FIELDS = ("step", "rank", "reqs", "tokens", "start_us", "wall_us", "busy_us", "gemm_us",
          "collective_us", "idle_us")


def _step_counts(shapes: str) -> tuple[int, int]:
    parts = shapes.strip('"').split(";")
    return int(parts[0]), int(parts[1])


def rank_steps(path: str) -> list[dict]:
    rows = []
    with open(os.path.join(path, "kernel_details.csv"), newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            try:
                begin = float((row.get("Start Time(us)") or "").strip())
                dur = float(row.get("Duration(us)") or 0.0)
            except ValueError:
                continue
            rows.append((begin, dur, (row.get("Type") or "").strip(), row.get("Input Shapes") or ""))
    rows.sort()
    starts = [i for i, r in enumerate(rows) if r[2] == STEP_KERNEL]
    steps = []
    # The last boundary has no next start, so it closes the window rather than opening a step.
    for n, (a, b) in enumerate(zip(starts, starts[1:])):
        reqs, tokens = _step_counts(rows[a][3])
        t0, t1 = rows[a][0], rows[b][0]
        busy = gemm = coll = covered = 0.0
        hi = t0
        for begin, dur, kind, _ in rows[a:b]:
            busy += dur
            if kind == GEMM:
                gemm += dur
            elif kind == COLLECTIVE:
                coll += dur
            end = min(begin + dur, t1)
            if end > hi:
                covered += end - max(begin, hi)
                hi = end
        steps.append({"step": n, "reqs": reqs, "tokens": tokens, "start_us": round(t0, 3),
                      "wall_us": round(t1 - t0, 3), "busy_us": round(busy, 3), "gemm_us": round(gemm, 3),
                      "collective_us": round(coll, 3), "idle_us": round(t1 - t0 - covered, 3)})
    return steps


def point(args: tuple[str, str, str]) -> str:
    run_dir, trace_dir, name = args
    out_dir = os.path.join(run_dir, "step_profile")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{name}.csv.gz")
    with gzip.open(out, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        counts = []
        for rank, parsed in ascend_rank_outputs(trace_dir).items():
            steps = rank_steps(parsed)
            counts.append(len(steps))
            for s in steps:
                w.writerow({"rank": rank, **s})
    return f"{out}: steps per rank {counts}"


def main(run_dirs: list[str]) -> None:
    jobs = []
    for run_dir in run_dirs:
        m = json.load(open(os.path.join(run_dir, "manifest.json")))
        for p in m["points"]:
            if p.get("trace_dir"):
                name = os.path.basename(p["trace_dir"].rstrip("/"))
                jobs.append((run_dir, os.path.join(run_dir, p["trace_dir"]), name))
    with ProcessPoolExecutor(max_workers=min(8, len(jobs) or 1)) as pool:
        for line in pool.map(point, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
