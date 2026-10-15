"""What the detection screen costs to run, against the capture ceiling it replaces.

    uv run python scripts/screen_overhead.py [<run-dir> ...]

Reads the screen-counting runs (configs/grids/screen_counting_overhead.toml), one
per model, each serving three arms in shared rounds:

  count-off    nothing collected - the baseline
  count-on     the screen as calibrated: 4 random layers redrawn per window,
               every tenth token, bincounted on device
  capture-on   --enable-return-routed-experts: every token's expert ids returned
               in the response body, which is the ceiling and the bridge to
               screen-overhead2

Arms share each round, so a pair differs in the instrument and in nothing that
drifted. The headline is the MEDIAN paired difference, declared in the grid
config before the run because DeepSeek's off arm has a heavy tail at this
concurrency: screen-overhead2 had one point in four 15% slow with no foreign
process and a lower loadavg than its siblings. The mean is printed beside it, and
when the two disagree the tail is doing the talking.

The comparison that answers the gating question is the last column: the screen's
cost against the EPLB false positive it would avert, which eplb-fp2 put at
+4.73% +-4.84 TPOT at 1.73x (free skew) and about +1% elsewhere.

`gate-closure` puts all of it on one model in one regime, with its own arm names
and a second skew level, so the arms and the baseline are options:

    uv run python scripts/screen_overhead.py <gate-closure-run> \
        --base base --arms screen,capture,eplb

Pairs are formed within a level as well as a round, because a run that sweeps
more than one level has several points per (arm, round) and pairing across them
would compare different skews.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
#: What a false positive costs, at 1.73x where it is worst, as (mean, sd) of the
#: paired rounds. Two measurements of the same quantity in the same regime, and
#: they disagree by more than either one resolves: eplb-fp2 read +4.73% on 4
#: pairs, gate-closure +2.33% on 5. Neither excludes zero, so the bar the screen
#: has to come under is a range, and the verdict below says so rather than
#: picking the flattering end.
EPLB_FALSE_POSITIVE = (4.73, 4.84)
EPLB_FALSE_POSITIVE_REPLICATION = (2.33, 2.55)


def runs(paths: list[Path]) -> list[Path]:
    if paths:
        return paths
    found = sorted(p for p in (ROOT / "results").glob("*_screen-counting-*")
                   if (p / "manifest.json").exists())
    if not found:
        raise SystemExit("no screen-counting run in results/ - pass run directories")
    return found


def contended(point: dict) -> bool:
    """Whether another process held an NPU at either of the point's host snapshots.

    Read from the snapshots, which every point records; there is no ``warnings``
    key on a point, so checking for one reported every run as clean.
    """
    return any((point.get(k) or {}).get("foreign_npu_processes") for k in ("host_before", "host_after"))


def paired(points: list[dict], arm: str, baseline: str, key: str) -> np.ndarray:
    """Per-round (baseline, arm) differences in percent, for one summary key.

    Points are paired within a level as well as a round: arms share each round at
    each level, so the pair differs in the instrument and in nothing else.
    """
    by = {(p["arm"], p["repeat"]): p for p in points}
    out = []
    for r in sorted({p["repeat"] for p in points}):
        a, b = by.get((baseline, r)), by.get((arm, r))
        if a and b:
            out.append(100 * (b["request_summary"][key] / a["request_summary"][key] - 1))
    return np.array(out)


def counting_fraction(run: Path, points: list[dict], screen_arm: str = "count-on") -> float | None:
    """Share of all expert assignments the screen actually counted.

    The counting arm records what it counted; the capture arm's records are the
    whole routing, so the two together give the realised sampling fraction rather
    than the nominal layers x tokens product.
    """
    counted = [p["counting_summary"]["assignments_counted"] for p in points
               if p["arm"] == screen_arm and (p.get("counting_summary") or {}).get("assignments_counted")]
    if not counted:
        return None
    totals = [p["request_summary"]["input_tokens_total"] + p["request_summary"]["output_tokens_total"]
              for p in points if p["arm"] == screen_arm]
    manifest = json.loads((run / "manifest.json").read_text())
    probe = manifest["config"]["model"].get("probe")
    # top_k x MoE layers, the assignments a model makes per token.
    per_token = {"deepseek": 6 * 26, "mistral": 2 * 32}.get(probe)
    if not (totals and per_token):
        return None
    return float(np.mean(counted) / (np.mean(totals) * per_token))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--base", default="count-off", help="the arm every other arm is read against")
    ap.add_argument("--arms", default="count-on,capture-on", help="comma-separated arms to price")
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    # The bar to come under is the SMALLER of the two readings: a screen that only
    # beats the generous one has not been shown to be worth running.
    bar = min(EPLB_FALSE_POSITIVE[0], EPLB_FALSE_POSITIVE_REPLICATION[0])
    hi = EPLB_FALSE_POSITIVE[0] + EPLB_FALSE_POSITIVE[1]
    print(f"EPLB's false positive, for comparison: +{EPLB_FALSE_POSITIVE[0]:.2f}% "
          f"+-{EPLB_FALSE_POSITIVE[1]:.2f} TPOT at 1.73x (eplb-fp2, free skew), which did not "
          f"replicate:\n  gate-closure read +{EPLB_FALSE_POSITIVE_REPLICATION[0]:.2f}% "
          f"+-{EPLB_FALSE_POSITIVE_REPLICATION[1]:.2f} on 5 fresh pairs in the same regime. "
          f"Neither excludes zero,\n  so the bar below is the smaller of the two "
          f"(+{bar:.2f}%) and 'cheaper' means cheaper than that.\n")
    print(f"{'model':>10} {'level':>6} {'arm':>11} {'e2e median':>11} {'mean':>8} {'makespan':>9} "
          f"{'off spread':>10} {'pairs':>5}  verdict")
    for run in runs(args.runs):
        manifest = json.loads((run / "manifest.json").read_text())
        model = manifest["config"]["model"]["model_name"]
        completed = [p for p in manifest["points"] if p.get("status") == "completed"]
        if not completed:
            print(f"{model:>10}  no completed points in {run.name}")
            continue
        for level in sorted({p["value"] for p in completed}):
            points = [p for p in completed if p["value"] == level]
            base = [p for p in points if p["arm"] == args.base]
            spread = float(np.std([p["request_summary"]["e2e_s_mean"] for p in base], ddof=1) /
                           np.mean([p["request_summary"]["e2e_s_mean"] for p in base]) * 100) if len(base) > 1 else float("nan")
            for arm in arms:
                d = paired(points, arm, args.base, "e2e_s_mean")
                m = paired(points, arm, args.base, "makespan_s")
                if not len(d):
                    print(f"{model:>10} {level:>6} {arm:>11}  no pairs")
                    continue
                median = float(np.median(d))
                # A cost below the baseline's own spread is not a measured cost.
                if abs(median) < spread:
                    verdict = f"under the {spread:.2f}% noise floor"
                elif median < bar:
                    verdict = "cheaper than the false positive it averts"
                elif median > hi:
                    verdict = "costs more than the false positive"
                else:
                    verdict = "same size as the false positive"
                print(f"{model:>10} {level:>6} {arm:>11} {median:+10.2f}% {np.mean(d):+7.2f}% "
                      f"{np.median(m):+8.2f}% {spread:9.2f}% {len(d):5}  {verdict}")
        screen_arm = arms[0]
        fraction = counting_fraction(run, completed, screen_arm)
        if fraction is not None:
            print(f"{'':>10} {'':>6} {'':>11} counted {fraction:.2%} of all expert assignments")
        warned = [p for p in completed if contended(p)]
        if warned:
            rounds = sorted({p["repeat"] for p in warned})
            uneven = [r for r in rounds
                      if len({contended(p) for p in completed if p["repeat"] == r}) > 1]
            print(f"{'':>10} warning: another process on the NPUs at {len(warned)} of "
                  f"{len(completed)} point(s), rounds {rounds}"
                  + (f"; arms differ within rounds {uneven}, so those pairs are not clean"
                     if uneven else "; every arm alike within each round"))
        missing = [p["label"] for p in completed
                   if p["arm"] == screen_arm and not (p.get("counting_summary") or {}).get("windows")]
        if missing:
            print(f"{'':>10} warning: {len(missing)} counting point(s) wrote nothing - "
                  f"the hook did not run, so they measured no screen")


if __name__ == "__main__":
    main()
