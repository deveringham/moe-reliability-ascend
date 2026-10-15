"""Is it worth screening before letting EPLB act?

    uv run python scripts/gated_eplb.py [--eplb <run-dir>] [--overhead <run-dir>]

This is an *evaluation built from measured components*, not a measured end-to-end
system. Nothing here was served with a live gate in the loop. Three measurements
are combined, each from a run where its own arms shared rounds:

  delta(level)   what EPLB costs or saves at that skew, EPLB-on against EPLB-off
                 (the arms run: eplb-fp2, gate-closure or eplb-mixtral)
  p_open(level)  the share of windows in which the calibrated screen's rule would
                 let EPLB act, scored on that level's captured routing
  s              what running the screen costs, always, whether it opens or not
                 (the counting arm of screen-counting / gate-closure)

against three policies, as a percentage of the EPLB-off latency at that level:

  never    0                    EPLB disabled: pay the skew, never the rearranging
  always   delta                EPLB always on, as deployments ship it
  gated    s + p_open * delta   screen always, EPLB only when the gate opens

So gating wins exactly where EPLB costs something (delta > 0) and the gate is
shut (p_open small), and loses s everywhere else. The level-by-level table says
where each holds; the mixes at the end say what that is worth to a deployment,
because the answer depends entirely on how much of its time is spent in the band
that is skewed enough to look alarming and not skewed enough to cost anything.

The mixes are illustrative weightings, not measured traffic. They are the one
unmeasured input here and are labelled as such in the output.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

#: Illustrative time-at-level weightings. "benign" is a deployment that mostly
#: sits in natural routing; "mixed" gives real weight to the alarming-but-free
#: band; "skewed" is one that is genuinely imbalanced much of the time.
MIXES = {
    "benign": "0.90 at the lowest level, the rest spread over the others",
    "mixed": "even time at every measured level",
    "skewed": "0.50 at the highest level, the rest spread over the others",
}


def weights(name: str, n: int) -> np.ndarray:
    if n == 1:
        return np.ones(1)
    if name == "benign":
        w = np.full(n, 0.10 / (n - 1))
        w[0] = 0.90
    elif name == "skewed":
        w = np.full(n, 0.50 / (n - 1))
        w[-1] = 0.50
    else:
        w = np.full(n, 1 / n)
    return w / w.sum()


def arms_run(pattern: str) -> str | None:
    found = sorted(d for d in glob.glob(os.path.join(ROOT, "results", pattern))
                   if os.path.exists(os.path.join(d, "manifest.json")))
    return found[-1] if found else None


def eplb_deltas(run: str, off: str, on: str, metric: str) -> list[dict]:
    """Per level: the paired EPLB-on minus EPLB-off difference, in percent."""
    m = json.load(open(os.path.join(run, "manifest.json")))
    points = [p for p in m["points"] if p.get("status") == "completed"]
    by = {(float(p["value"]), p["arm"], p["repeat"]): p for p in points}
    rows = []
    for level in sorted({float(p["value"]) for p in points}):
        pairs = []
        for r in sorted({p["repeat"] for p in points if float(p["value"]) == level}):
            a, b = by.get((level, off, r)), by.get((level, on, r))
            if a and b:
                pairs.append((a["request_summary"][metric], b["request_summary"][metric]))
        if not pairs:
            continue
        base = np.array([x for x, _ in pairs])
        delta = np.array([100 * (y / x - 1) for x, y in pairs])
        applied = [(by[(level, on, r)].get("eplb_activity") or {}).get("applied_cycles", 0)
                   for r in sorted({p["repeat"] for p in points if float(p["value"]) == level})
                   if (level, on, r) in by]
        rows.append({"level": level, "base": float(base.mean()), "delta_pct": float(delta.mean()),
                     "delta_sd": float(delta.std(ddof=1)) if len(delta) > 1 else float("nan"),
                     "pairs": len(pairs), "applied_cycles": applied})
    return rows


def screen_cost(run: str | None, arm: str = "count-on", baseline: str = "count-off") -> tuple[float, float]:
    """What running the screen costs, and the baseline's own spread, in percent."""
    if not run:
        return float("nan"), float("nan")
    m = json.load(open(os.path.join(run, "manifest.json")))
    points = [p for p in m["points"] if p.get("status") == "completed"]
    e2e = {(p["arm"], p["repeat"]): p["request_summary"]["e2e_s_mean"] for p in points}
    d = [100 * (e2e[(arm, r)] / e2e[(baseline, r)] - 1)
         for r in sorted({k[1] for k in e2e}) if (arm, r) in e2e and (baseline, r) in e2e]
    base = [v for (a, _), v in e2e.items() if a == baseline]
    spread = 100 * float(np.std(base, ddof=1) / np.mean(base)) if len(base) > 1 else float("nan")
    return (float(np.mean(d)) if d else float("nan")), spread


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eplb", default="*_eplb-fp2", help="arms run holding the EPLB arms")
    ap.add_argument("--overhead", default="*_screen-counting-*", help="run holding the counting arm")
    ap.add_argument("--off", default="eplb-off")
    ap.add_argument("--on", default="eplb-on")
    ap.add_argument("--metric", default="tpot_ms_mean")
    ap.add_argument("--screen-cost", type=float, default=None,
                    help="override the measured screen cost, in percent")
    args = ap.parse_args()

    import plot_findings as P

    run = arms_run(args.eplb)
    if not run:
        raise SystemExit(f"no run matching {args.eplb} in results/")
    rows = eplb_deltas(run, args.off, args.on, args.metric)
    if not rows:
        raise SystemExit(f"{os.path.basename(run)} has no completed {args.off}/{args.on} pairs")

    model = json.load(open(os.path.join(run, "manifest.json")))["config"]["model"]["model_name"]
    if model not in P.GATE_SOURCES:
        raise SystemExit(f"no calibration and traffic known for model {model!r}")
    # Scored on the same model's captures: the arms run may carry none of its own
    # (eplb-fp2 and eplb-mixtral both borrow them), but borrowing across models
    # would score the gate with the wrong alarm and the wrong expert count.
    gate = P.eplb_gate({r["level"] for r in rows}, model=model)

    s = args.screen_cost
    spread = float("nan")
    if s is None:
        over = arms_run(args.overhead)
        s, spread = screen_cost(over)
        if not np.isfinite(s):
            raise SystemExit("no screen cost available - pass --screen-cost")
    print(f"EPLB arms:    {os.path.basename(run)} ({args.metric}, {rows[0]['pairs']} pairs/level)")
    print(f"gate:         {model}'s screen, scored on {P.GATE_SOURCES[model]['traffic']}")
    print(f"screen costs: {s:+.2f}% of serving time, always on"
          + (f" (baseline spread {spread:.2f}%)" if np.isfinite(spread) else ""))
    print("an evaluation combining measured parts; no gate was ever in the serving loop\n")

    print(f"{'load':>6} {'applied':>8} {'EPLB delta':>12} {'gate open':>10} | "
          f"{'never':>7} {'always':>8} {'gated':>8}  best")
    table = []
    for r in rows:
        load, p_open = gate.get(r["level"], (r["level"], float("nan")))
        never, always = 0.0, r["delta_pct"]
        gated = s + (p_open * r["delta_pct"] if np.isfinite(p_open) else float("nan"))
        options = {"never": never, "always": always, "gated": gated}
        best = min((v, k) for k, v in options.items() if np.isfinite(v))[1]
        applied = r["applied_cycles"]
        print(f"{load:6.2f}x {min(applied) if applied else 0:>3}-{max(applied) if applied else 0:<4} "
              f"{r['delta_pct']:+7.2f}% ±{r['delta_sd']:4.2f} {p_open:9.0%} | "
              f"{never:+6.2f}% {always:+7.2f}% {gated:+7.2f}%  {best}")
        table.append({"load": load, "p_open": p_open, **{k: v for k, v in options.items()},
                      "delta_pct": r["delta_pct"], "delta_sd": r["delta_sd"]})

    print("\nexpected cost over a deployment's time at each level "
          "(illustrative weightings, not measured traffic):")
    print(f"{'mix':>8} {'never':>8} {'always':>8} {'gated':>8}   {'gated vs always':>16}   weighting")
    for name, description in MIXES.items():
        w = weights(name, len(table))
        ev = {k: float(np.sum(w * np.array([t[k] for t in table]))) for k in ("never", "always", "gated")}
        print(f"{name:>8} {ev['never']:+7.2f}% {ev['always']:+7.2f}% {ev['gated']:+7.2f}%   "
              f"{ev['gated'] - ev['always']:+15.2f}pp   {description}")

    print("\nGating beats leaving EPLB on wherever EPLB costs something and the gate is shut;\n"
          f"it gives back the screen's {s:+.2f}% everywhere else. Whether that trade is worth\n"
          "making is a property of the traffic, which is why the mixes are the input to vary.")


if __name__ == "__main__":
    main()
