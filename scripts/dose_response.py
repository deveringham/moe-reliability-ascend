"""Latency against injected rank skew, per run of a router-bias grid.

    uv run python scripts/dose_response.py <name> [<name> ...]

A name is a grid name, or the experiment name of a run made outside a grid.

Each bias level is mapped to the busiest-rank load measured for it by the
model's calibration run (experiment name ``rbias-calibration``). For every
metric the script prints the per-level mean +/- sd and a fit of the metric on
that skew with execution order as a covariate, so drift during a run is not
read as an effect. Sustained throughput (90th percentile of 2.5 s completion
bins) is reported alongside makespan because external slowdown episodes halve
throughput for part of a point, which inflates makespan but not the sustained
rate.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"


def manifests():
    for d in sorted(RESULTS.iterdir()):
        mf = d / "manifest.json"
        if mf.exists():
            yield d, json.loads(mf.read_text())


def calibration(model_name: str) -> tuple[dict[float, float], dict[float, int]]:
    """Busiest-rank load and live expert count per bias level."""
    best, live = {}, {}
    for d, m in manifests():
        if d.name.split("_")[2] != model_name or not d.name.endswith("rbias-calibration"):
            continue
        if m.get("status") != "completed":
            continue
        # Runs are visited oldest first, so a later run's measurement of a level wins.
        best.update({float(p["value"]): p["validation_summary"]["rank_max_over_mean"] for p in m["points"]
                     if p.get("validation_summary", {}).get("rank_max_over_mean") is not None})
        live.update({float(p["value"]): p["validation_summary"]["active_experts"] for p in m["points"]
                     if p.get("validation_summary", {}).get("active_experts") is not None})
    return best, live


def sustained(d: Path, p: dict) -> float | None:
    if not p.get("metrics_file"):
        return None
    reqs = json.load(gzip.open(d / p["metrics_file"]))["requests"]
    if not reqs or reqs[0].get("end_s") is None:
        return None
    reqs = sorted(reqs, key=lambda r: r["end_s"])
    ends = np.array([r["end_s"] for r in reqs])
    toks = np.array([r["num_input_tokens"] + (r["num_output_tokens"] if r["num_output_tokens"] > 1 else 0)
                     for r in reqs])
    cuts = np.searchsorted(ends, np.arange(0, ends[-1] + 2.5, 2.5))
    rate = np.array([toks[a:b].sum() / 2.5 for a, b in zip(cuts[:-1], cuts[1:])])
    return float(np.percentile(rate[2:-1], 90)) if len(rate) > 4 else None


def fit(x, order, y):
    X = np.column_stack([np.ones(len(y)), x, order]).astype(float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    dof = len(y) - 3
    s2 = ((y - X @ b) ** 2).sum() / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return b, se


def main(grids: list[str]) -> None:
    for d, m in manifests():
        g = (m.get("grid") or {}).get("name") or m["config"]["experiment"]["name"]
        if g not in grids:
            continue
        model = d.name.split("_")[2]
        cal, live = calibration(model)
        # A level where experts drop out varies the active expert count as well as the skew
        # (a memory-bound decode GEMM gets cheaper with fewer experts), so it is left out of fits.
        collapsed = {lv for lv, n in live.items() if n < max(live.values())}
        # Levels without a calibration point are interpolated in level, and flagged.
        lv_all = sorted({float(p["value"]) for p in m["points"]})
        xs = sorted(cal)
        skew = {lv: cal[lv] if lv in cal else float(np.interp(lv, xs, [cal[x] for x in xs])) for lv in lv_all} \
            if cal else {}
        interpolated = [lv for lv in lv_all if lv not in cal]
        print(f"\n== {d.name}  {(m.get('grid') or {}).get('assignments', '')}  {m.get('status')}")
        pts = [p for p in m["points"] if p.get("status") == "completed"]
        if not pts:
            continue
        rows = []
        for p in pts:
            s = p["request_summary"]
            rows.append({"level": float(p["value"]), "skew": skew.get(float(p["value"]), np.nan),
                         "order": p.get("exec_order", 0),
                         "neighbour": any((p.get(k) or {}).get("foreign_npu_processes")
                                          for k in ("host_before", "host_after")),
                         "makespan_s": s.get("makespan_s"), "sustained_tok_s": sustained(d, p),
                         "ttft_ms_mean": s.get("ttft_ms_mean"), "tpot_ms_mean": s.get("tpot_ms_mean") or None,
                         "tpot_ms_p99": s.get("tpot_ms_p99") or None})
        levels = sorted({r["level"] for r in rows})
        print("  level -> busiest rank: " + ", ".join(f"{lv:g}: {skew.get(lv, float('nan')):.2f}x"
                                                    f"{'*' if lv in interpolated else ''}" for lv in levels)
              + ("   (* interpolated from calibration)" if interpolated else ""))
        print(f"  {sum(r['neighbour'] for r in rows)} of {len(rows)} points had a neighbour at a snapshot")
        if collapsed & set(levels):
            print("  excluded from fits, experts drop out: " + ", ".join(
                f"{lv:g} ({live[lv]} of {max(live.values())} live)" for lv in sorted(collapsed & set(levels))))
        for metric in ("makespan_s", "sustained_tok_s", "ttft_ms_mean", "tpot_ms_mean", "tpot_ms_p99"):
            use = [r for r in rows if r[metric] is not None and np.isfinite(r["skew"])]
            if len(use) < 5:
                continue
            per = []
            for lv in levels:
                v = np.array([r[metric] for r in use if r["level"] == lv])
                if v.size:
                    per.append(f"{lv:g}: {v.mean():.1f}±{v.std(ddof=1) if v.size > 1 else 0:.1f}")
            use = [r for r in use if r["level"] not in collapsed]
            y = np.array([r[metric] for r in use])
            b, se = fit([r["skew"] - 1 for r in use], [r["order"] for r in use], y)
            base = b[0]
            print(f"  {metric:<16} {' | '.join(per)}")
            print(f"  {'':<16} per +1x busiest-rank load: {b[1]:+.2f} ({100 * b[1] / base:+.1f}% of the balanced "
                  f"fit), t={b[1] / se[1]:.2f}; order t={b[2] / se[2]:.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
