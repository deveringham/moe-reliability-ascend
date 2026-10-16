"""Two detection knobs that were fixed by assumption rather than measured.

    uv run python scripts/detection_params.py [--out docs/figures/tables]

`detection_eval.py` swept the window length, the counted fraction, the layer
choice and the confirmation length k. It held two things constant, and this
script sweeps them against the same captures:

  fpr    Every alarm in docs/detection-findings.md is set at a 1% per-window
         false-alarm rate (`threshold_at_fpr(benign, 0.01)`, hard-coded in four
         places). Every alarm level, detection rate and requests-to-false-alarm
         figure in that document is conditional on that one number, and the
         trade it makes against k was never drawn. The operator's knob is the
         *pair*: a looser per-window alarm with a longer confirmation may be
         both quieter and more sensitive than the tuned k at 1%.

  split  The screen counts 4 layers at a tenth of their tokens - 1.2-1.5% of
         assignments. Since the layer axis decides whether a layer-local skew is
         visible at all (4 spaced layers catch it in 1% of windows, 4 redrawn in
         66%), how a fixed budget is divided between layers and tokens could buy
         accuracy for nothing. 4 x 10% was never compared against 8 x 5%,
         2 x 20% or every layer at 1.5% at equal cost.

Both axes are re-analysis of routing captures already under results/, so neither
needs the node. Impact labelling is per model and deliberately asymmetric:
Mixtral has a calibrated impact threshold, and DeepSeek's was withdrawn on
2026-10-14, so no DeepSeek level here is called costly or free on latency
grounds - its rows measure separation from benign traffic only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "packages",
                                "moe-reliability-results", "src"))

import detection_eval as E  # noqa: E402 - the capture loaders and the screen's own setting
from moe_reliability_results import detection as D  # noqa: E402

N_RANKS = E.N_RANKS

#: Per-window false-alarm targets. 0.1% is left out: the alarm is a quantile of
#: windows bootstrapped from ~1200 captured requests, so a 1-in-1000 level is an
#: extrapolation of that tail rather than a measurement of it.
FPRS = (0.002, 0.005, 0.01, 0.02, 0.05, 0.10)
#: Confirmation lengths, as in detection-findings C5.
CONSECUTIVE = (1, 2, 3)
#: The calibrated impact threshold, in layer-averaged busiest-rank load.
#: Mixtral: docs/calibrations/20261008-223820_..._calibration-sweep.json, at 100
#: generated tokens. DeepSeek: 2.434x was withdrawn on 2026-10-14 - the point it
#: was interpolated to cost latency through early-ending answers, not skew - and
#: nothing up to 3.6x costs anything with length held, so there is no threshold
#: to score against.
IMPACT = {"mixtral": 1.42, "deepseek-v2": None}

BENIGN_DRAWS = 8000
LEVEL_DRAWS = 3000
STREAM_DRAWS = 200


def screen(layers=4, token_fraction=0.1, window=8, draws=LEVEL_DRAWS, layer_choice="random"):
    """The recommended screen, with one knob moved."""
    return D.WindowSampler(window=window, layers=layers, token_fraction=token_fraction,
                           draws=draws, layer_choice=layer_choice)


def true_loads(levels):
    """{level: the load the screen is trying to read}, per-layer busiest-rank form."""
    return {lv: float(D.busiest_rank(c.sum(axis=0), N_RANKS)) for lv, c in levels.items()}


# --- 1. The false-alarm target, jointly with k --------------------------------

def fpr_sweep(model, levels):
    """What the 1% choice buys and costs, against looser and tighter alarms.

    The per-level draws are computed once and re-thresholded for each target, so
    every row scores the same windows: differences between rows are the alarm
    moving, not sampling noise moving with it.
    """
    sampler = screen()
    truth = true_loads(levels)
    threshold = IMPACT[model]
    benign = D.estimate_loads(levels[0.0], N_RANKS, screen(draws=BENIGN_DRAWS), seed=0)
    loads = {lv: D.estimate_loads(c, N_RANKS, sampler, seed=1)
             for lv, c in sorted(levels.items()) if lv != 0.0}

    # The level the screen has to catch: the lowest one that costs latency where
    # that is known, and otherwise the lowest captured skew, which is the hardest
    # to see and carries no impact claim.
    costly = [lv for lv in sorted(loads) if threshold and truth[lv] >= threshold]
    onset = costly[0] if costly else min(loads)

    rows = []
    for fpr in FPRS:
        alarm = D.threshold_at_fpr(benign, fpr)
        row = {"model": model, "target_fpr": fpr, "alarm": round(alarm, 4),
               "benign_median": round(float(np.median(benign)), 4),
               "realised_fpr": round(float((benign >= alarm).mean()), 4),
               "impact_threshold": threshold,
               # Positive headroom means the alarm can fire before the cost does.
               "headroom": None if not threshold else round(threshold - alarm, 4)}
        for lv in sorted(loads):
            row[f"detect_{truth[lv]:.2f}x"] = round(float((loads[lv] >= alarm).mean()), 3)
        if threshold:
            free = [lv for lv in loads if truth[lv] < threshold]
            row["fires_on_free_levels"] = (round(float(np.mean([(loads[lv] >= alarm).mean()
                                                               for lv in free])), 3) if free else None)
            row["catches_costly_levels"] = (round(float(np.mean([(loads[lv] >= alarm).mean()
                                                                 for lv in costly])), 3) if costly else None)
        rows.append(row)

    # The joint knob: each alarm with each confirmation length, scored on the
    # stream rather than on independent windows, so the quiet and the delay are
    # the ones an operator would see.
    joint = []
    for fpr in FPRS:
        alarm = D.threshold_at_fpr(benign, fpr)
        for k in CONSECUTIVE:
            delay = D.onset_delay(levels[0.0], levels[onset], N_RANKS, sampler, alarm,
                                  consecutive=k, draws=STREAM_DRAWS, seed=2)
            quiet = D.mean_windows_to_false_alarm(levels[0.0], N_RANKS, sampler, alarm,
                                                  consecutive=k, draws=STREAM_DRAWS, seed=3)
            joint.append({"model": model, "target_fpr": fpr, "consecutive": k,
                          "alarm": round(alarm, 4), "onset_load": round(truth[onset], 3),
                          "detected": delay["detected_fraction"],
                          "false_start": delay["false_start_fraction"],
                          "delay_requests_p50": delay["delay_requests_p50"],
                          "delay_requests_p90": delay["delay_requests_p90"],
                          "requests_to_false_alarm_p50": quiet["requests_to_false_alarm_p50"],
                          "quiet_over_horizon": quiet["quiet_over_horizon"]})
    return {"onset_level": onset, "onset_load": round(truth[onset], 3),
            "onset_is_costly": bool(costly), "loads": {str(k): round(v, 3) for k, v in truth.items()},
            "n_requests": int(len(levels[0.0])), "benign_draws": BENIGN_DRAWS,
            "per_window": rows, "with_confirmation": joint}


# --- 2. How to spend the counting budget --------------------------------------

#: Layers to count, against which the token fraction is set to hold the budget.
SPLITS = (1, 2, 4, 8, 13, None)
#: Budget multiples of the screen's own 4 layers x 10% of tokens.
BUDGETS = (1, 4)


def splits_for(n_moe, multiplier):
    """(layers, token_fraction) pairs that count the same share of assignments."""
    budget = 4 / n_moe * 0.1 * multiplier
    out = []
    for layers in SPLITS:
        n = n_moe if layers is None else layers
        tf = budget * n_moe / n
        if tf <= 1.0:
            out.append((layers, tf))
    return budget, out


def budget_sweep(model, levels, label="all-layer skew"):
    """Does it matter how a fixed counting budget is divided?"""
    n_moe = int(levels[0.0].shape[1])
    truth = true_loads(levels)
    threshold = IMPACT[model]
    costly = [lv for lv in sorted(levels) if lv != 0.0 and threshold and truth[lv] >= threshold]
    rows = []
    for multiplier in BUDGETS:
        budget, pairs = splits_for(n_moe, multiplier)
        for layers, tf in pairs:
            sampler = screen(layers=layers, token_fraction=tf)
            benign = D.estimate_loads(levels[0.0], N_RANKS,
                                      screen(layers=layers, token_fraction=tf, draws=BENIGN_DRAWS),
                                      seed=0)
            alarm = D.threshold_at_fpr(benign, 0.01)
            row = {"model": model, "skew": label, "budget": round(budget, 4),
                   "budget_x_screen": multiplier,
                   "layers": layers or n_moe, "of_layers": n_moe, "token_fraction": round(tf, 4),
                   "assignment_fraction": round(sampler.cost(n_moe).assignment_fraction, 4),
                   "benign_median": round(float(np.median(benign)), 4),
                   "benign_p99": round(float(np.quantile(benign, 0.99)), 4),
                   "alarm": round(alarm, 4)}
            for lv, counts in sorted(levels.items()):
                if lv == 0.0:
                    continue
                loads = D.estimate_loads(counts, N_RANKS, sampler, seed=1)
                row[f"detect_{truth[lv]:.2f}x"] = round(float((loads >= alarm).mean()), 3)
            if costly:
                row["catches_costly_levels"] = round(float(np.mean(
                    [row[f"detect_{truth[lv]:.2f}x"] for lv in costly])), 3)
            rows.append(row)
    return rows


def localised_budget_sweep():
    """The same splits against a skew confined to 6 of 26 layers.

    This is where the layer axis mattered most - a spaced 4-layer screen is blind
    to it - so if any split beats 4 x 10%, it should show here.
    """
    out = []
    for d, m in E.manifests("*detection-localisation"):
        model = os.path.basename(d).split("_")[2]
        n_experts = E.MODELS[model]["n_experts"]
        biased = set(m["config"]["imbalance"].get("bias_layers") or [])
        pools, n_moe, hot_layers = {}, 0, 0
        import gzip
        for p in m["points"]:
            f = json.load(gzip.open(os.path.join(d, p["validation_file"]))) if p.get("validation_file") else {}
            path = os.path.join(d, f.get("records_file") or "")
            if not f.get("records_file") or not os.path.exists(path):
                continue
            pools[float(p["value"])] = D.request_counts(
                [json.loads(line) for line in gzip.open(path, "rt")], n_experts)
            moe_layers = f.get("moe_layers") or list(range(pools[float(p["value"])].shape[1]))
            n_moe = len(moe_layers)
            hot_layers = len([i for i, layer in enumerate(moe_layers) if layer in biased])
        if 0.0 not in pools or len(pools) < 2:
            continue
        rows = budget_sweep(model, pools, label=f"{hot_layers} of {n_moe} layers")
        for r in rows:
            r["run"] = os.path.basename(d)
            r["hot_layers"] = hot_layers
        out.extend(rows)
    return out


# --- output -------------------------------------------------------------------

def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("docs", "figures", "tables"))
    ap.add_argument("--models", nargs="*", default=list(E.MODELS))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    store, fpr_rows, joint_rows, budget_rows = {}, [], [], []
    for model in args.models:
        levels, _ = E.capture_levels(model)
        if 0.0 not in levels or len(levels) < 2:
            print(f"{model}: no capture with a balanced arm, skipped")
            continue
        sweep = fpr_sweep(model, levels)
        store[model] = sweep
        fpr_rows.extend(sweep["per_window"])
        joint_rows.extend(sweep["with_confirmation"])
        budget_rows.extend(budget_sweep(model, levels))

        print(f"\n{model}: {sweep['n_requests']} requests per level, loads {sweep['loads']}")
        print(f"  impact threshold {IMPACT[model] or 'withdrawn - no impact labelling'}; "
              f"onset level {sweep['onset_level']} at {sweep['onset_load']}x"
              f"{'' if sweep['onset_is_costly'] else ' (lowest captured skew, not a cost)'}")
        detect_cols = [k for k in sweep["per_window"][0] if k.startswith("detect_")]
        print(f"  {'fpr':>6} {'alarm':>7} {'head':>7} " + " ".join(f"{c[7:]:>7}" for c in detect_cols))
        for r in sweep["per_window"]:
            head = "-" if r["headroom"] is None else format(r["headroom"], "+.3f")
            print(f"  {r['target_fpr']:>6.1%} {r['alarm']:>7.3f} {head:>7} "
                  + " ".join(f"{r[c]:>7.2f}" for c in detect_cols))
        print(f"  {'fpr':>6} {'k':>2} {'detect':>7} {'early':>6} {'delay':>6} {'reqs/FA':>9}")
        for r in sweep["with_confirmation"]:
            fa = r["requests_to_false_alarm_p50"]
            print(f"  {r['target_fpr']:>6.1%} {r['consecutive']:>2} {r['detected']:>7.2f} "
                  f"{r['false_start']:>6.2f} "
                  f"{('-' if r['delay_requests_p50'] is None else format(r['delay_requests_p50'], '.0f')):>6} "
                  f"{('none' if fa is None else format(fa, '.0f')):>9}")

    budget_rows.extend(localised_budget_sweep())
    if budget_rows:
        print("\nbudget splits at equal assignment cost")
        for r in budget_rows:
            detect = {k[7:]: v for k, v in r.items() if k.startswith("detect_")}
            print(f"  {r['model']:<12} {r['skew']:<16} {r['budget_x_screen']}x "
                  f"{r['layers']:>3}/{r['of_layers']} layers x {r['token_fraction']:>6.3f} tokens "
                  f"({r['assignment_fraction']:.4f})  alarm {r['alarm']:>6.3f}  "
                  + " ".join(f"{k} {v:.2f}" for k, v in detect.items()))

    paths = [write_csv(os.path.join(args.out, "detection_fpr.csv"), fpr_rows),
             write_csv(os.path.join(args.out, "detection_fpr_confirmation.csv"), joint_rows),
             write_csv(os.path.join(args.out, "detection_budget_split.csv"), budget_rows)]
    with open(os.path.join(args.out, "detection_params.json"), "w") as f:
        json.dump({"fpr": store, "budget_splits": budget_rows}, f, indent=1)
    for p in paths:
        print(f"  -> {p}")


if __name__ == "__main__":
    main()
