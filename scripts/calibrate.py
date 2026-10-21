"""Turn one deployment's skew sweep into the calibration a screen runs on.

    uv run python scripts/calibrate.py <run-dir> [<run-dir> ...] [--margin 0.5]
                                       [--window 8] [--layers 4] [--tokens 0.1]
                                       [--out docs/calibrations]

A calibration belongs to a configuration, not to a model: the load at which
imbalance starts to cost depends on the model, the batch size and the shape of
the skew, so it has to be measured where it will be used. This reads a sweep that
already ran on that deployment - graded skew, repeats, routing captured at each
level - and writes what the detector needs:

  impact threshold   the load at which cost rises above this deployment's own
                     noise floor, measured here rather than assumed
  noise floor        from the repeats of the balanced arm; it differs between
                     models (2.4% on DeepSeek against 0.8% on Mixtral), so a
                     shared constant would misreport one of them
  benign ceiling     what ordinary traffic on this deployment reaches, at the
                     sampling the screen will actually use
  headroom           the gap between those two, which is how much warning is
                     available before the cost arrives - and can be negative
  alarm level        placed inside that gap by --margin

The sweep itself is an ordinary run (configs/examples/router_bias_sweep.toml and
its grids). Separating the two means the expensive part happens once and the
calibration can be re-derived - at a different margin, or a different sampling -
without touching an NPU.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages",
                                "moe-reliability-results", "src"))

from moe_reliability_results import detection as D  # noqa: E402

N_RANKS_KEY = "n_npus"


def sweep_points(run_dir):
    """{level: (realised per-layer load, [tpot per repeat], live experts)} for one run."""
    manifest = json.load(open(os.path.join(run_dir, "manifest.json")))
    out: dict[float, dict] = {}
    for point in manifest["points"]:
        level = float(point["value"])
        entry = out.setdefault(level, {"tpot": [], "load": None, "live": None})
        if point.get("request_summary", {}).get("tpot_ms_mean"):
            entry["tpot"].append(point["request_summary"]["tpot_ms_mean"])
        if point.get("validation_file") and entry["load"] is None:
            f = json.load(gzip.open(os.path.join(run_dir, point["validation_file"])))
            entry["load"] = float(np.mean(f["rank_max_over_mean_per_layer"]))
            entry["live"] = (point.get("validation_summary") or {}).get("active_experts")
            entry["records"] = f.get("records_file")
    return manifest, out


def borrowed_loads(model, results_dir):
    """Realised load per level from a separate calibration run of the same model.

    A sweep that captures its own routing is the better arrangement, because load
    is a property of the traffic as well as the knob - the same offset reads
    1.35x on a six-corpus mix where it reads 1.30x on MMLU alone. This is the
    fallback for sweeps that ran before that was wired up, and it is recorded in
    the output so a calibration never silently rests on another run's traffic.
    """
    best: dict[float, tuple[float, int]] = {}
    for d in sorted(glob.glob(os.path.join(results_dir, "*rbias-calibration"))):
        if os.path.basename(d).split("_")[2] != model:
            continue
        manifest = json.load(open(os.path.join(d, "manifest.json")))
        for point in manifest["points"]:
            if not point.get("validation_file"):
                continue
            f = json.load(gzip.open(os.path.join(d, point["validation_file"])))
            best[float(point["value"])] = (float(np.mean(f["rank_max_over_mean_per_layer"])),
                                           (point.get("validation_summary") or {}).get("active_experts"))
    return best


def benign_distribution(run_dir, points, manifest, sampler, n_ranks):
    """What ordinary traffic on this deployment reads, at the screen's own sampling.

    Taken from the unbiased level's routing capture, so it is this deployment's
    traffic rather than a number carried over from another one.
    """
    balanced = points.get(0.0) or {}
    path = os.path.join(run_dir, balanced.get("records") or "")
    if not balanced.get("records") or not os.path.exists(path):
        return None
    n_experts = (manifest.get("points") or [{}])[0].get("bias_experts")
    records = [json.loads(line) for line in gzip.open(path, "rt")]
    width = max(int(np.max(r["routed_experts"] if r.get("routed_experts") is not None
                           else r["prompt_routed_experts"])) for r in records) + 1
    counts = D.request_counts(records, max(width, len(n_experts or []) or width))
    return D.estimate_loads(counts, n_ranks, sampler, seed=0)


def expert_width(run_dir, points):
    """How many experts the router scores, from whichever capture is at hand."""
    for entry in points.values():
        path = os.path.join(run_dir, entry.get("records") or "")
        if entry.get("records") and os.path.exists(path):
            for line in gzip.open(path, "rt"):
                r = json.loads(line)
                ids = r.get("routed_experts")
                if ids is None:
                    ids = r.get("prompt_routed_experts")
                return int(np.max(ids)) + 1
    return 0


def level_distributions(run_dir, points, sampler, n_ranks, width):
    """{realised load: window estimates} per level, from each level's own capture.

    Needed to measure what a margin actually catches rather than infer it: a
    level's pooled load says where it sits, its distribution says how often a
    window of it clears a given alarm.
    """
    out = {}
    for level, entry in points.items():
        path = os.path.join(run_dir, entry.get("records") or "")
        if not entry.get("records") or not os.path.exists(path) or entry["load"] is None:
            continue
        records = [json.loads(line) for line in gzip.open(path, "rt")]
        counts = D.request_counts(records, width)
        out[entry["load"]] = D.estimate_loads(counts, n_ranks, sampler, seed=1)
    return out


def calibrate_run(run_dir, sampler, margin, fpr):
    manifest, points = sweep_points(run_dir)
    config = manifest["config"]
    n_ranks = config["hardware"][N_RANKS_KEY]
    borrowed = None
    if not any(p["load"] is not None for p in points.values()):
        borrowed = borrowed_loads(config["model"]["model_name"], os.path.dirname(run_dir) or "results")
        for level, (load, live) in borrowed.items():
            if level in points:
                points[level]["load"], points[level]["live"] = load, live
    live = [p["live"] for p in points.values() if p["live"]]
    full = max(live) if live else None
    levels = sorted(lv for lv, p in points.items()
                    if p["load"] is not None and p["tpot"] and (full is None or p["live"] == full))
    if len(levels) < 2 or 0.0 not in levels:
        return {"run": os.path.basename(run_dir),
                "skipped": "needs an unbiased level and at least one more, with routing captured"}
    base = float(np.mean(points[0.0]["tpot"]))
    loads = [points[lv]["load"] for lv in levels]
    costs = [100 * (float(np.mean(points[lv]["tpot"])) / base - 1) for lv in levels]

    benign = benign_distribution(run_dir, points, manifest, sampler, n_ranks)
    if benign is None:
        # Without a capture there is no deployment-specific benign distribution;
        # the unbiased level's own realised load is the honest stand-in.
        benign = np.full(200, points[0.0]["load"])
    sampling = {"window": sampler.window, "layers": sampler.layers,
                "token_fraction": sampler.token_fraction, "layer_choice": sampler.layer_choice}
    cal = D.calibrate(loads, costs, points[0.0]["tpot"], benign, sampling,
                      alarm_margin=margin, fpr=fpr)
    cal["run"] = os.path.basename(run_dir)
    cal["model"] = config["model"]["model_name"]
    cal["deployment"] = {"n_npus": n_ranks, "batch_size": config["server"]["batch_size"],
                         "max_num_batched_tokens": config["server"]["max_num_batched_tokens"],
                         "enforce_eager": config["server"]["enforce_eager"],
                         "max_new_tokens": config["client"]["max_new_tokens"],
                         "bias_target": config["imbalance"]["bias_target"]}
    cal["balanced_tpot_ms"] = round(base, 1)
    cal["loads_from"] = "this sweep" if borrowed is None else "a separate calibration run of this model"
    # What each margin would buy, measured on this deployment's own levels rather
    # than inferred from the curve.
    cal["margin_sweep"] = None
    if cal.get("impact_threshold") and any(p.get("records") for p in points.values()):
        per_level = level_distributions(run_dir, points, sampler, n_ranks, expert_width(run_dir, points))
        if len(per_level) > 1:
            cal["margin_sweep"] = D.margin_sweep(cal["impact_threshold"], benign, per_level, fpr=fpr)
    if borrowed is not None:
        cal.setdefault("note", "")
        cal["note"] = ((cal["note"] + "; ") if cal["note"] else "") + (
            "realised loads were borrowed from another run's traffic, so the thresholds are only as "
            "good as that traffic matching this one")
    return cal


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories, or globs under results/")
    ap.add_argument("--margin", type=float, default=0.5,
                    help="how far back from the impact threshold to alarm: 0 waits until it costs, "
                         "1 alarms at the benign ceiling (default 0.5)")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--tokens", type=float, default=0.1)
    ap.add_argument("--layer-choice", default="random", choices=("random", "spaced"))
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--out", default="docs/calibrations")
    args = ap.parse_args(argv)

    sampler = D.WindowSampler(window=args.window, layers=args.layers, token_fraction=args.tokens,
                              draws=600, layer_choice=args.layer_choice)
    paths = [p for pattern in args.runs for p in sorted(glob.glob(pattern)) or [pattern]]
    os.makedirs(args.out, exist_ok=True)
    for path in paths:
        if not os.path.exists(os.path.join(path, "manifest.json")):
            print(f"{path}: no manifest, skipping")
            continue
        cal = calibrate_run(path, sampler, args.margin, args.fpr)
        name = os.path.join(args.out, f"{os.path.basename(path)}.json")
        json.dump(cal, open(name, "w"), indent=1, default=float)
        if "skipped" in cal:
            print(f"{cal['run']}: {cal['skipped']}")
            continue
        d = cal["deployment"]
        print(f"{cal['model']} @ {d['n_npus']} NPUs, batch {d['batch_size']}, "
              f"{'eager' if d['enforce_eager'] else 'graph'}")
        print(f"   noise floor      {cal['noise_floor_pct']:.2f}% of TPOT")
        print(f"   costs above      {cal['impact_threshold']}x busiest rank"
              if cal["impact_threshold"] else "   costs above      not reached by this sweep")
        print(f"   benign reaches   {cal['benign_ceiling']}x")
        if cal.get("headroom") is not None:
            print(f"   headroom         {cal['headroom']:+.3f}x")
        if cal.get("alarm"):
            a = cal["alarm"]
            print(f"   alarm at         {a['level']}x (margin {a['margin']}), "
                  f"false alarms {a['false_alarm_rate']:.1%}")
        print(f"   loads from       {cal['loads_from']}")
        for r in cal.get("margin_sweep") or []:
            print(f"   margin {r['margin']:<4} alarm {r['alarm']:>6.3f}x  false alarms "
                  f"{r['false_alarm_rate']:>6.1%}  warns before cost "
                  f"{('-' if r['caught_before_cost'] is None else format(r['caught_before_cost'], '.0%')):>5}"
                  f"  catches costly "
                  f"{('-' if r['caught_once_costly'] is None else format(r['caught_once_costly'], '.0%')):>5}")
        if cal.get("note"):
            print(f"   note: {cal['note']}")
        print(f"   -> {name}")


if __name__ == "__main__":
    main()
