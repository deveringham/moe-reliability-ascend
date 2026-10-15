"""What EPLB costs where it has nothing to win, and whether a screen would have stopped it.

    uv run python scripts/eplb_false_positive.py [<run-dir>] [--loads <run-dir>]

Reads a forced-imbalance run served with two arms, `eplb-off` and `eplb-on`
(configs/examples/eplb_false_positive.toml), and prints per level:

  - what EPLB did in the on arm, from the server log: plans, applied cycles, and
    its own busiest-rank reading at the first and last plan. A level where
    nothing was applied measured nothing, and says so.
  - the arm difference in TPOT and e2e latency, paired by round. Arms share each
    round, so the pair differs in EPLB and in nothing that drifted.
  - whether the calibrated screen would have let EPLB act there, scored on that
    level's captured traffic (`--loads`, a run with validation records at the
    same levels: routing does not depend on placement).

The noise floor is the spread of the off arm's repeats at each level, which is
what a single pair difference has to clear.

The screen is calibrated per model and the gate is scored on per-request routing
records, which only a run with `validation_records` has. Both the calibration and
the loads run are therefore chosen by the run's model, and a mismatch is refused
rather than scored: a DeepSeek gate column against Mixtral latencies reads as a
result and is not one.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages/moe-reliability-results/src"))
from moe_reliability_results.detection import Screen, busiest_rank, request_counts, stream_loads  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
#: Per model: the calibration the screen is read from, and a run whose validation
#: records hold per-request routing at the same levels. Routing does not depend on
#: placement, so the records may come from another run of the same skew - but not
#: from another model, whose expert count and alarm are both different.
CALIBRATION = {
    "deepseek-v2": ROOT / "docs/calibrations/20261006-155755_imbalance_deepseek-v2_npu4_bs512_rbias-controlled-000.json",
    "mixtral": ROOT / "docs/calibrations/20261008-223820_imbalance_mixtral_npu4_bs512_calibration-sweep.json",
}
LOADS = {
    "deepseek-v2": ROOT / "results/20261009-115011_imbalance_deepseek-v2_npu4_bs512_eplb-fp-000",
    "mixtral": ROOT / "results/20261008-223820_imbalance_mixtral_npu4_bs512_calibration-sweep",
}


def latest_run() -> Path:
    runs = sorted(p for p in (ROOT / "results").glob("*_eplb-fp2") if (p / "manifest.json").exists())
    if not runs:
        raise SystemExit("no eplb-fp2 run in results/ - pass a run directory")
    return runs[-1]


def expert_count(loads_run: Path, level: float) -> int | None:
    """The model's expert count, as the loads run itself recorded it.

    Read rather than assumed: 64 on DeepSeek-V2-Lite and 8 on Mixtral, and
    scoring one model's records against the other's count silently rewrites which
    experts belong to which rank.
    """
    path = loads_run / f"validation/imbalance_{level:g}.json.gz"
    if not path.exists():
        return None
    return int(json.load(gzip.open(path, "rt"))["n_experts"])


def gate(level: float, loads_run: Path, screen: Screen, n_ranks: int, windows: int = 2000) -> dict:
    """Share of windows in which the screen's rule would let EPLB act."""
    path = loads_run / f"validation/imbalance_{level:g}.records.jsonl.gz"
    n_experts = expert_count(loads_run, level)
    if not path.exists() or n_experts is None:
        return {}
    counts = request_counts((json.loads(line) for line in gzip.open(path, "rt")), n_experts)
    loads = stream_loads([(counts, windows * screen.sampler.window)], n_ranks, screen.sampler,
                         np.random.default_rng(0))
    k = screen.consecutive
    raised = np.convolve(loads >= screen.alarm, np.ones(k, int), "valid") == k
    return {"load": busiest_rank(counts.sum(axis=0), n_ranks), "gate_open": float(raised.mean())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", type=Path)
    ap.add_argument("--loads", type=Path, help="run whose validation records the gate is scored on")
    ap.add_argument("--calibration", type=Path, help="the screen's calibration JSON")
    args = ap.parse_args()
    run = args.run or latest_run()
    manifest = json.loads((run / "manifest.json").read_text())
    n_ranks = manifest["config"]["hardware"]["n_npus"]
    model = manifest["config"]["model"]["model_name"]

    calibration_path = args.calibration or CALIBRATION.get(model)
    loads_run = args.loads or LOADS.get(model)
    if calibration_path is None or loads_run is None:
        raise SystemExit(f"no calibration and loads run known for model {model!r} - "
                         f"pass --calibration and --loads")
    calibration = json.loads(Path(calibration_path).read_text())
    if calibration.get("model") not in (None, model):
        raise SystemExit(f"calibration {Path(calibration_path).name} is for "
                         f"{calibration['model']!r}, not {model!r}")
    loads_model = json.loads((Path(loads_run) / "manifest.json").read_text())["config"]["model"]["model_name"]
    if loads_model != model:
        raise SystemExit(f"loads run {Path(loads_run).name} is {loads_model!r} routing, not {model!r} - "
                         f"the gate would be scored on the wrong model")
    screen = Screen.from_calibration(calibration)

    points = [p for p in manifest["points"] if p.get("status") == "completed"]
    by = {(p["value"], p["arm"], p["repeat"]): p for p in points}
    levels = sorted({p["value"] for p in points})
    print(f"run {run.name}: {len(points)} completed points; {model} screen alarm {screen.alarm}x "
          f"({screen.consecutive} x {screen.sampler.window} requests)")
    print(f"gate scored on {Path(loads_run).name}, calibrated by {Path(calibration_path).name}\n")
    print(f"{'level':>5} {'load':>6} {'gate':>6} | {'applied':>7} {'EPLB load':>13} | "
          f"{'TPOT off':>8} {'on-off':>14} {'e2e on-off':>11} {'off spread':>10} | pairs")
    for level in levels:
        g = gate(level, Path(loads_run), screen, n_ranks)
        tpot, e2e, applied, hot = [], [], [], []
        for r in sorted({p["repeat"] for p in points if p["value"] == level}):
            off, on = by.get((level, "eplb-off", r)), by.get((level, "eplb-on", r))
            if not (off and on):
                continue
            so, sn = off["request_summary"], on["request_summary"]
            tpot.append((so["tpot_ms_mean"], sn["tpot_ms_mean"]))
            e2e.append((so["e2e_s_mean"], sn["e2e_s_mean"]))
            act = on.get("eplb_activity") or {}
            applied.append(act.get("applied_cycles", 0))
            if act.get("hotness"):
                hot.append((act["hotness"][0][0], act["hotness"][-1][0]))
        if not tpot:
            continue
        off_t = np.array([a for a, _ in tpot])
        d_t = np.array([100 * (b / a - 1) for a, b in tpot])
        d_e = np.array([100 * (b / a - 1) for a, b in e2e])
        spread = 100 * off_t.std(ddof=1) / off_t.mean() if len(off_t) > 1 else float("nan")
        h = f"{np.mean([a for a, _ in hot]):.2f}->{np.mean([b for _, b in hot]):.2f}x" if hot else "n/a"
        print(f"{level:>5g} {g.get('load', float('nan')):6.3f} {g.get('gate_open', float('nan')):6.0%} | "
              f"{min(applied):>3}-{max(applied):<3} {h:>13} | {off_t.mean():8.1f} "
              f"{d_t.mean():+6.2f}% ±{d_t.std(ddof=1) if len(d_t) > 1 else float('nan'):4.2f} "
              f"{d_e.mean():+10.2f}% {spread:9.2f}% | {len(tpot)}")
        if min(applied) == 0:
            print(f"      warning: an eplb-on point at level {level:g} applied no placement - "
                  f"that pair measured collect-and-plan only")


if __name__ == "__main__":
    main()
