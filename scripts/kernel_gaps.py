"""Where a rank sits idle between kernels, per step, by the transition it sits in.

    uv run python scripts/kernel_gaps.py <trace_dir> [<trace_dir> ...]

A trace_dir is one point's ``traces/<point>`` directory (one ``*ascend_pt``
directory per rank). For each rank, kernels from every stream are merged in
start order, and each stretch with no kernel running is attributed to the pair
(last kernel type to finish, next kernel type to start). Totals are divided by
the number of steps, counted as ``_compute_slot_mapping_kernel`` launches.

A device gap is time the host has not yet launched the next kernel: launch
overhead, Python, or a host synchronisation waiting on a device result. A rank
that is consistently last into a collective while its kernels are no slower
than the others' is late because of these gaps.
"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

from moe_reliability.core.trace_analysis import ascend_rank_outputs

STEP_KERNEL = "_compute_slot_mapping_kernel"


def rank_gaps(path: str, min_gap_us: float = 1.0):
    rows = []
    with open(os.path.join(path, "kernel_details.csv"), newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            try:
                begin = float((row.get("Start Time(us)") or "").strip())
                dur = float(row.get("Duration(us)") or 0.0)
            except ValueError:
                continue
            rows.append((begin, begin + dur, (row.get("Type") or "").strip()))
    rows.sort()
    steps = sum(1 for r in rows if r[2] == STEP_KERNEL) or 1
    gaps, total = defaultdict(float), 0.0
    hi, hi_type = rows[0][1], rows[0][2]
    for begin, end, kind in rows[1:]:
        if begin - hi >= min_gap_us:
            gaps[(hi_type, kind)] += begin - hi
            total += begin - hi
        if end > hi:
            hi, hi_type = end, kind
    span = hi - rows[0][0]
    return steps, span, total, gaps


def main(dirs: list[str], top: int = 8) -> None:
    for trace_dir in dirs:
        print(f"\n== {trace_dir}")
        for rank, parsed in ascend_rank_outputs(trace_dir).items():
            steps, span, total, gaps = rank_gaps(parsed)
            print(f"  rank {rank}: {steps} steps, span {span / steps / 1e3:.1f} ms/step, "
                  f"idle {total / steps / 1e3:.1f} ms/step")
            for (a, b), us in sorted(gaps.items(), key=lambda kv: -kv[1])[:top]:
                print(f"    {us / steps / 1e3:6.2f} ms/step  {a[:32]:>32} -> {b[:32]}")


if __name__ == "__main__":
    main(sys.argv[1:])
