"""How short can a calibration be and still place the alarm in the right place?

    uv run python scripts/calibration_cost.py [<run-dir>] [--reference <run-dir>]
                                              [--out docs/figures/tables]

A calibration costs NPU time: the sweep behind docs/calibrations is 7 levels x 3
repeats, 21 points, 70 minutes of a 4-NPU server, and it has to be redone for
every deployment whose batch size, model or parallelism differs. If that is the
price of admission the method is impractical, so the question is what a cheaper
one gives up.

Three axes, each cut from the full sweep rather than re-run, so the comparison is
against the same traffic and the same hardware state:

  levels    how many offsets the sweep visits. The onset is found by
            interpolation between the two levels that bracket it, so coarseness
            shows up as placement error, not as failure.
  repeats   how many times each level is served. This is the noise floor, and
            the noise floor *is* the impact threshold's definition - with one
            repeat there is no floor to clear and the threshold is wherever the
            first stray point landed.
  prompts   how many requests the routing capture covers. This sets the benign
            ceiling and the per-level distributions, not the latency.

Each subset is scored against the full sweep's calibration on the quantity that
matters operationally - where the alarm ends up, what it catches, and how often
it fires on benign traffic - not on whether the numbers match.

With --reference, the scoring is against a *different* run of the same
deployment instead of against the same sweep's own full calibration. That is the
harder and more honest question: a calibration that only reproduces itself has
shown nothing about the next session.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages",
                                "moe-reliability-results", "src"))

import calibrate as C  # noqa: E402
from moe_reliability_results import detection as D  # noqa: E402

DEFAULT_RUN = "results/*calibration-sweep"


def point_minutes(run_dir):
    """{level: mean minutes a point at that level took}, from the manifest's timestamps."""
    manifest = json.load(open(os.path.join(run_dir, "manifest.json")))
    out: dict[float, list[float]] = {}
    for p in manifest["points"]:
        if not (p.get("started_at") and p.get("finished_at")):
            continue
        started = dt.datetime.fromisoformat(p["started_at"])
        finished = dt.datetime.fromisoformat(p["finished_at"])
        out.setdefault(float(p["value"]), []).append((finished - started).total_seconds() / 60)
    return {level: float(np.mean(v)) for level, v in out.items()}


def load_sweep(run_dir, sampler):
    """Everything a calibration needs from one run, loaded once and sliced after.

    Loading is the expensive part - the records are tens of megabytes per level -
    so the subsets share one read rather than re-reading per combination.
    """
    manifest, points = C.sweep_points(run_dir)
    config = manifest["config"]
    n_ranks = config["hardware"]["n_npus"]
    width = C.expert_width(run_dir, points)
    records = {}
    for level, entry in points.items():
        path = os.path.join(run_dir, entry.get("records") or "")
        if entry.get("records") and os.path.exists(path):
            import gzip
            rows = [json.loads(line) for line in gzip.open(path, "rt")]
            records[level] = D.request_counts(rows, width)
    return {"manifest": manifest, "points": points, "n_ranks": n_ranks,
            "records": records, "minutes": point_minutes(run_dir),
            "run": os.path.basename(run_dir)}


def calibration_from(sweep, levels, repeats, prompt_fraction, sampler, margin, fpr):
    """Calibrate from a subset of the sweep: these levels, this many repeats, this much capture."""
    points, records = sweep["points"], sweep["records"]
    if 0.0 not in levels or len(levels) < 2:
        return None
    tpot = {lv: points[lv]["tpot"][:repeats] for lv in levels}
    if any(len(v) == 0 for v in tpot.values()):
        return None
    base = float(np.mean(tpot[0.0]))
    loads = [points[lv]["load"] for lv in levels]
    costs = [100 * (float(np.mean(tpot[lv])) / base - 1) for lv in levels]

    cache = sweep.setdefault("_windows", {})

    def windows(level):
        counts = records.get(level)
        if counts is None:
            return None
        key = (level, prompt_fraction)
        if key not in cache:                     # the search reuses these thousands of times
            keep = max(sampler.window, int(len(counts) * prompt_fraction))
            cache[key] = D.estimate_loads(counts[:keep], sweep["n_ranks"], sampler, seed=1)
        return cache[key]

    benign = windows(0.0)
    if benign is None:
        return None
    sampling = {"window": sampler.window, "layers": sampler.layers,
                "token_fraction": sampler.token_fraction, "layer_choice": sampler.layer_choice}
    cal = D.calibrate(loads, costs, tpot[0.0], benign, sampling, alarm_margin=margin, fpr=fpr)
    cal["level_loads"] = {points[lv]["load"]: windows(lv) for lv in levels if windows(lv) is not None}
    cal["benign"] = benign
    cal["minutes"] = round(sum(sweep["minutes"].get(lv, 0.0) * repeats for lv in levels), 1)
    cal["points"] = len(levels) * repeats
    return cal


def score(cal, reference):
    """What a cheaper calibration gives up, in the terms an operator would ask.

    Against the reference's impact threshold, not its own: a subset that misplaces
    the threshold and then measures its alarm against that same misplacement looks
    perfect, which is the trap this whole script exists to avoid.
    """
    if cal is None or not cal.get("alarm"):
        return {"calibrated": False}
    truth = reference["impact_threshold"]
    alarm = cal["alarm"]["level"]
    costly = [load for load in reference["level_loads"] if load >= truth]
    harmless = [load for load in reference["level_loads"] if load < truth]
    caught = {load: float((np.asarray(v) >= alarm).mean())
              for load, v in reference["level_loads"].items()}
    return {
        "calibrated": True,
        "points": cal["points"],
        "npu_minutes": cal["minutes"],
        "noise_floor_pct": cal["noise_floor_pct"],
        "impact_threshold": cal["impact_threshold"],
        "threshold_error": round(cal["impact_threshold"] - truth, 3),
        "alarm": alarm,
        # Measured on the reference's own traffic, so every row is scored against
        # the same windows whatever the subset's capture was.
        "false_alarm_rate": round(float((np.asarray(reference["benign"]) >= alarm).mean()), 4),
        "catches_costly": (round(float(np.mean([caught[c] for c in costly])), 3) if costly else None),
        "fires_on_harmless": (round(float(np.mean([caught[h] for h in harmless])), 3)
                              if harmless else None),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", default=DEFAULT_RUN)
    ap.add_argument("--reference", default=None,
                    help="score against this run's calibration instead of the sweep's own "
                         "(a different session on the same deployment is the real test)")
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--tokens", type=float, default=0.1)
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--out", default="docs/figures/tables")
    args = ap.parse_args(argv)

    import glob
    paths = sorted(glob.glob(args.run)) or [args.run]
    sampler = D.WindowSampler(window=args.window, layers=args.layers, token_fraction=args.tokens,
                              draws=600, layer_choice="random")
    sweep = load_sweep(paths[0], sampler)
    all_levels = sorted(lv for lv, p in sweep["points"].items()
                        if p["load"] is not None and p["tpot"])
    max_repeats = max(len(sweep["points"][lv]["tpot"]) for lv in all_levels)
    full = calibration_from(sweep, all_levels, max_repeats, 1.0, sampler, args.margin, args.fpr)
    if full is None or not full.get("impact_threshold"):
        print(f"{sweep['run']}: the full sweep does not calibrate; nothing to compare against")
        return
    reference = full
    if args.reference:
        ref_paths = sorted(glob.glob(args.reference)) or [args.reference]
        ref_sweep = load_sweep(ref_paths[0], sampler)
        ref_levels = sorted(lv for lv, p in ref_sweep["points"].items()
                            if p["load"] is not None and p["tpot"])
        reference = calibration_from(ref_sweep, ref_levels,
                                     max(len(ref_sweep["points"][lv]["tpot"]) for lv in ref_levels),
                                     1.0, sampler, args.margin, args.fpr)
        if reference is None or not reference.get("impact_threshold"):
            print(f"{args.reference}: does not calibrate; cannot be the reference")
            return

    rows = []

    def add(axis, setting, levels, repeats, fraction):
        cal = calibration_from(sweep, levels, repeats, fraction, sampler, args.margin, args.fpr)
        rows.append({"axis": axis, "setting": setting, "levels": len(levels), "repeats": repeats,
                     "capture_fraction": fraction, **score(cal, reference)})

    def spread_pick(candidates, k):
        """The k levels a person would choose: as evenly spread as the grid allows."""
        best = None
        for combo in itertools.combinations(candidates, k):
            spread = float(np.std(np.diff(combo))) if k > 2 else 0.0
            if best is None or spread < best[0]:
                best = (spread, list(combo))
        return best[1] if best else []

    # Resolution, with the range held fixed. The unbiased level is the baseline
    # every cost is measured against and the top level is what makes the sweep
    # reach past the onset, so both stay in; only the levels between them are
    # dropped. Confounding the two is easy and makes coarseness look fatal when
    # the subset simply stopped short - see the "range" rows below.
    inner = [lv for lv in all_levels if lv not in (all_levels[0], all_levels[-1])]
    for k in range(0, len(inner) + 1):
        levels = sorted({all_levels[0], *spread_pick(inner, k), all_levels[-1]})
        add("resolution", f"{len(levels)} levels: {','.join(format(x, 'g') for x in levels)}",
            levels, max_repeats, 1.0)

    # Range: every level the sweep has, up to a ceiling. This is the axis that
    # actually breaks a calibration, because a sweep that never reaches past the
    # onset has nothing to interpolate between and places it wherever the noise
    # happened to fall.
    for top in all_levels[1:]:
        levels = [lv for lv in all_levels if lv <= top]
        add("range", f"stops at offset {top:g} ({sweep['points'][top]['load']:.2f}x)",
            levels, max_repeats, 1.0)

    for repeats in range(1, max_repeats + 1):
        add("repeats", f"{repeats} repeat(s)", all_levels, repeats, 1.0)

    for fraction in (0.02, 0.05, 0.1, 0.25, 0.5, 1.0):
        add("capture", f"{fraction:.0%} of the capture", all_levels, max_repeats, fraction)

    # The question the axes above only answer one slice at a time: of every sweep
    # that could have been run, which is the cheapest that still lands the alarm
    # where the full one did? Tolerance is the reference's own noise floor
    # expressed as load - a calibration cannot be asked to beat the measurement
    # it is made of.
    tolerance = 0.05
    cheapest = None
    for k in range(1, len(all_levels)):
        for combo in itertools.combinations(all_levels[1:], k):
            levels = [0.0, *combo]
            for repeats in range(1, max_repeats + 1):
                cal = calibration_from(sweep, levels, repeats, 1.0, sampler, args.margin, args.fpr)
                s = score(cal, reference)
                if not s.get("calibrated") or abs(s["threshold_error"]) > tolerance:
                    continue
                if s["false_alarm_rate"] > args.fpr * 2 or (s["catches_costly"] or 0) < 1.0:
                    continue
                if cheapest is None or s["npu_minutes"] < cheapest[0]:
                    cheapest = (s["npu_minutes"], levels, repeats, s)
    if cheapest:
        _m, levels, repeats, s = cheapest
        rows.append({"axis": "cheapest", "levels": len(levels), "repeats": repeats,
                     "capture_fraction": 1.0,
                     "setting": f"{','.join(format(x, 'g') for x in levels)} x {repeats}", **s})

    os.makedirs(args.out, exist_ok=True)
    import csv
    path = os.path.join(args.out, "calibration_cost.csv")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    ref_name = reference is not full and args.reference or "its own full sweep"
    print(f"{sweep['run']}, scored against {ref_name}")
    print(f"  reference: onset {reference['impact_threshold']}x, alarm "
          f"{reference['alarm']['level']}x, {full['points']} points, {full['minutes']} NPU-minutes")
    print(f"  {'axis':<9} {'setting':<30} {'min':>6} {'onset':>7} {'err':>7} {'alarm':>7} "
          f"{'FA':>6} {'costly':>7} {'harmless':>9}")
    for r in rows:
        if not r.get("calibrated"):
            print(f"  {r['axis']:<9} {r['setting']:<30} {'-':>6}   did not calibrate")
            continue
        print(f"  {r['axis']:<9} {r['setting']:<30} {r['npu_minutes']:>6.0f} "
              f"{r['impact_threshold']:>7.3f} {r['threshold_error']:>+7.3f} {r['alarm']:>7.3f} "
              f"{r['false_alarm_rate']:>6.1%} "
              f"{('-' if r['catches_costly'] is None else format(r['catches_costly'], '.0%')):>7} "
              f"{('-' if r['fires_on_harmless'] is None else format(r['fires_on_harmless'], '.0%')):>9}")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
