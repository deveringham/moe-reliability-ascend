"""Bias 0 vs 100 (or alpha) per run of a grid, with each point's contention state.

    uv run python scripts/regime_summary.py <grid-name> [--metric makespan_s]

For each run in the grid: the swept value, repeat, execution order, the metric,
and whether another process held NPUs before or after the point. The ratio of
group means is reported twice, over all points and over the points with no
neighbour at either snapshot, because a neighbouring job inflates timings
(see docs/imbalance-findings.md) and contention that differs between arms is a
confound, not an effect.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
METRICS = ("makespan_s", "input_tokens_per_s", "output_tokens_per_s", "ttft_ms_mean", "tpot_ms_mean",
           "tpot_ms_p99", "e2e_s_mean")


def contended(point: dict) -> bool:
    return any((point.get(k) or {}).get("foreign_npu_processes") for k in ("host_before", "host_after"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("grid")
    ap.add_argument("--metric", default="makespan_s", choices=METRICS)
    args = ap.parse_args()

    runs = []
    for d in sorted(RESULTS.iterdir()):
        mf = d / "manifest.json"
        if not mf.exists():
            continue
        m = json.loads(mf.read_text())
        if (m.get("grid") or {}).get("name") == args.grid:
            runs.append((d.name, m))
    if not runs:
        raise SystemExit(f"no runs of grid {args.grid!r} in {RESULTS}")

    for name, m in runs:
        print(f"\n== {name}  {m['grid'].get('assignments')}  status {m.get('status')}")
        groups: dict[str, list[tuple[float, bool]]] = defaultdict(list)
        for p in m.get("points", []):
            s = p.get("request_summary") or {}
            v = s.get(args.metric)
            c = contended(p)
            flag = "NEIGHBOUR" if c else ""
            print(f"  {p['label']:<22} order {str(p.get('exec_order', '?')):>2}  {p.get('status'):<10} "
                  f"{args.metric} {v if v is None else round(v, 3)!s:>10}  {flag}")
            if v is not None:
                groups[str(p["value"])].append((v, c))
        keys = sorted(groups, key=float)
        for subset, label in ((lambda c: True, "all points"), (lambda c: not c, "quiet points only")):
            stats = {k: [v for v, c in groups[k] if subset(c)] for k in keys}
            line = ", ".join(f"{k}: {np.mean(x):.3f} +/- {np.std(x, ddof=1) if len(x) > 1 else float('nan'):.3f} (n={len(x)})"
                             for k, x in stats.items() if x)
            if len(keys) >= 2 and stats[keys[0]] and stats[keys[-1]]:
                line += f"  ->  {keys[-1]}/{keys[0]} = {np.mean(stats[keys[-1]]) / np.mean(stats[keys[0]]):.3f}"
            print(f"  {label}: {line}")


if __name__ == "__main__":
    main()
