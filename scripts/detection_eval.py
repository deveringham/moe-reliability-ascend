"""Phase 0 of the imbalance-detection evaluation: what is measurable offline.

    uv run python scripts/detection_eval.py [<out_dir>]

Writes <out_dir>/detection.json (every number) and prints a summary. Everything
is read from results/ under this checkout; sections whose inputs are missing are
reported as skipped rather than guessed.

The pipeline under test screens on busiest-rank load (stage 1), confirms against
step time (stage 2), and attributes the step to a rank (stage 3). Phase 0 asks
only what existing data can answer:

  A. Where does a cost begin?      Impact thresholds from the controlled sweeps.
  B. What does benign traffic do?  Load distributions from unbiased captures,
                                   across workload families and window sizes.
  C. How cheap can the screen be?  Window, layer and token sampling against the
                                   resolution needed to separate A from B.
  D. Does stage 2 work offline?    Step-time fits at matched batch size.
  E. Does stage 3 work offline?    Pace-setter attribution from trace summaries.
  F. What does confirming cost?    Profiled points against their unprofiled twins.

What is missing, and why it needs the node: positives for stage 1. The
calibration runs recorded pooled expert counts per level, not per-request
routing, so the biased load *distribution* - which decides the detection rate at
a given window - cannot be formed here. ``configs/examples/detection_capture.toml``
captures it; until then the positive side is reported as the single pooled load
per level, and no ROC is claimed.
"""

from __future__ import annotations

import csv
import functools
import glob
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages",
                                "moe-reliability-results", "src"))

from moe_reliability_results import detection as D  # noqa: E402

RESULTS = "results"
N_RANKS = 4
# Largest run-to-run drift between identical points, from the zero-control's
# repeated level-2 arms. A cost below this is not distinguishable from noise.
DRIFT_PCT = 2.2
MODELS = {"deepseek-v2": {"n_experts": 64, "top_k": 6, "capture": "*alpha-sweep"},
          "mixtral": {"n_experts": 8, "top_k": 2, "capture": "*mixtral-alpha"}}
CONTROLLED = {"deepseek-v2": "*rbias-controlled-000", "mixtral": "*rbias-controlled-001"}
PROFILED = {"deepseek-v2": "*rbias-profiled-deepseek-001", "mixtral": "*rbias-profiled-mixtral-001"}


def manifests(pattern):
    for d in sorted(glob.glob(os.path.join(RESULTS, pattern))):
        path = os.path.join(d, "manifest.json")
        if os.path.exists(path):
            yield d, json.load(open(path))


def one(pattern):
    found = list(manifests(pattern))
    return found[0] if found else (None, None)


def calibration():
    """{model: {level: (busiest-rank load, live experts)}}, per-layer form."""
    out: dict[str, dict[float, tuple[float, int]]] = {}
    for d, m in manifests("*rbias-calibration"):
        if m.get("status") != "completed":
            continue
        model = os.path.basename(d).split("_")[2]
        for p in m["points"]:
            summary = p.get("validation_summary") or {}
            if summary.get("rank_max_over_mean") is None:
                continue
            f = json.load(gzip.open(os.path.join(d, p["validation_file"])))
            out.setdefault(model, {})[float(p["value"])] = (
                float(np.mean(f["rank_max_over_mean_per_layer"])), summary["active_experts"])
    return out


# --- A. Where a cost begins --------------------------------------------------

def impact(cal):
    rows = {}
    for model, pattern in CONTROLLED.items():
        d, m = one(pattern)
        if m is None:
            continue
        by_level: dict[float, list[float]] = {}
        for p in m["points"]:
            if p.get("status") == "completed":
                by_level.setdefault(float(p["value"]), []).append(p["request_summary"]["tpot_ms_mean"])
        full = max(n for _, n in cal[model].values())
        levels = [lv for lv in sorted(by_level) if cal[model][lv][1] == full]
        base = float(np.mean(by_level[0.0]))
        loads = [cal[model][lv][0] for lv in levels]
        costs = [100 * (float(np.mean(by_level[lv])) / base - 1) for lv in levels]
        threshold = D.impact_threshold(loads, costs, DRIFT_PCT)
        rows[model] = {"run": os.path.basename(d), "levels": levels,
                       "loads": [round(x, 3) for x in loads], "tpot_pct": [round(y, 2) for y in costs],
                       "balanced_ms": round(base, 1), "impact_threshold": threshold,
                       "note": None if threshold else "no swept level clears the drift floor"}
    return rows


# --- B/C. Benign traffic, and how cheaply it can be measured -----------------

def captures(model):
    """Per-request (layers, experts) counts and labels from a routed-expert capture."""
    spec = MODELS[model]
    found = glob.glob(os.path.join(RESULTS, spec["capture"], "activations", "records.jsonl.gz"))
    if not found:
        return None, None
    records, labels = [], []
    for line in gzip.open(found[0], "rt"):
        r = json.loads(line)
        records.append(r)
        labels.append(r.get("subject") or "unknown")
    return D.request_counts(records, spec["n_experts"]), np.array(labels)


SAMPLERS = [D.WindowSampler(window=w, layers=lay, token_fraction=tf, draws=400)
            for w in (1, 8, 64, 512)
            for lay, tf in ((None, 1.0), (4, 1.0), (None, 0.1), (4, 0.1))]


def benign(model, counts, labels, threshold):
    """What benign traffic reads, per window definition and per sampling setting."""
    truth = D.busiest_rank(counts.sum(axis=0), N_RANKS)
    rows = {"model": model, "n_requests": int(len(counts)), "n_moe_layers": int(counts.shape[1]),
            "pooled_load": round(truth, 4),
            "pooled_load_layerwise_vs_pooled_form": round(D.pooled_busiest_rank(counts.sum(axis=0), N_RANKS), 4)}

    # Coherent windows: one topic at a time is the worst case natural traffic offers.
    groups = {label: np.flatnonzero(labels == label) for label in np.unique(labels)}
    groups = {k: v for k, v in groups.items() if len(v) >= 30}
    rows["by_topic"] = {"n_topics": len(groups),
                        "loads": sorted(round(D.busiest_rank(counts[idx].sum(axis=0), N_RANKS), 4)
                                        for idx in groups.values())}

    # Where the skew sits, and whether one rank holds it: what a screen could
    # report beyond a single number, at no extra counting cost.
    located = D.localise(counts.sum(axis=0), N_RANKS, threshold or 1.3)
    rows["localisation"] = {k: located[k] for k in ("flagged_layers", "flagged_fraction", "mean_load",
                                                    "max_layer_load", "consistent_rank",
                                                    "consistent_rank_share", "flagged_rank")}
    rows["localisation"]["per_layer_load_range"] = [round(min(located["per_layer_load"]), 4),
                                                    round(max(located["per_layer_load"]), 4)]

    rows["sampling"] = []
    for sampler in SAMPLERS:
        loads = D.estimate_loads(counts, N_RANKS, sampler, seed=0)
        score = D.screen_scores(sampler, sampler.cost(counts.shape[1]), loads, truth=truth)
        summary = score.summary()
        summary["p99"] = round(float(np.quantile(loads, 0.99)), 4)
        if threshold:
            # A benign window that reads at or above the impact threshold is a
            # false alarm - the quantity the screen has to keep small.
            summary["false_alarm_rate"] = round(float((loads >= threshold).mean()), 4)
            summary["margin_to_threshold"] = round(threshold - summary["p99"], 4)
        rows["sampling"].append(summary)

    # The cheapest window that resolves benign traffic from the impact threshold.
    if threshold:
        needed = (threshold - truth) / 2
        rows["required_window"] = {
            "resolution": round(needed, 4),
            "all_layers": D.required_window(counts, N_RANKS, needed, draws=200),
            "four_layers": D.required_window(counts, N_RANKS, needed, layers=4, draws=200),
            "four_layers_tenth_of_tokens": D.required_window(counts, N_RANKS, needed, layers=4,
                                                             token_fraction=0.1, draws=200)}
    return rows


# --- B2. Positives: what a window reads at a known skew -----------------------

CAPTURES = {"deepseek-v2": "*detection-capture", "mixtral": "*detection-capture"}
SCREEN = D.WindowSampler(window=8, layers=4, token_fraction=0.1, draws=600)


@functools.lru_cache(maxsize=None)
def capture_levels(model):
    """{level: (requests, layers, experts) counts} from a detection capture.

    Cached: a capture is ~1200 records per level of nested routing arrays, so
    parsing it twice costs minutes.
    """
    spec = MODELS[model]
    out, labels = {}, None
    for d, m in manifests(CAPTURES[model]):
        # Completed only: a run still in flight, or one that failed partway, would
        # contribute a few levels and silently change what the rates are over.
        if os.path.basename(d).split("_")[2] != model or m.get("status") != "completed":
            continue
        for p in m["points"]:
            f = json.load(gzip.open(os.path.join(d, p["validation_file"]))) if p.get("validation_file") else {}
            path = os.path.join(d, f.get("records_file") or "")
            if not f.get("records_file") or not os.path.exists(path):
                continue
            records = [json.loads(line) for line in gzip.open(path, "rt")]
            out[float(p["value"])] = D.request_counts(records, spec["n_experts"])
            labels = np.array([r.get("subject") or "unknown" for r in records])
    return out, labels


def detection(model, levels, threshold):
    """Detection rate per level and sampling setting, against the capture's own balanced arm."""
    if 0.0 not in levels or len(levels) < 2:
        return {"skipped": "capture has no balanced arm to score against"}
    rows, by_setting = [], []
    for sampler in SAMPLERS:
        benign = D.estimate_loads(levels[0.0], N_RANKS, sampler, seed=0)
        # The alarm level is set on benign traffic, not on the impact threshold:
        # a screen is calibrated where it runs, and the impact curve only says
        # which levels it *should* catch.
        alarm = D.threshold_at_fpr(benign, 0.01)
        per_level = []
        for level, counts in sorted(levels.items()):
            if level == 0.0:
                continue
            loads = D.estimate_loads(counts, N_RANKS, sampler, seed=1)
            truth = D.busiest_rank(counts.sum(axis=0), N_RANKS)
            per_level.append({"level": level, "true_load": round(truth, 3),
                              "median_estimate": round(float(np.median(loads)), 3),
                              "detection_rate": round(float((loads >= alarm).mean()), 3),
                              "auc": round(D.roc(loads, benign)["auc"], 4),
                              "costly": None if threshold is None else bool(truth >= threshold)})
        by_setting.append({"window": sampler.window, "layers": sampler.layers,
                           "token_fraction": sampler.token_fraction,
                           "assignment_fraction": round(sampler.cost(levels[0.0].shape[1]).assignment_fraction, 4),
                           "alarm_at_1pct_fpr": round(alarm, 4), "levels": per_level})
    rows = by_setting
    return {"n_requests": int(len(levels[0.0])), "settings": rows}


def localisation_run():
    """A skew confined to a few layers on a rank other than 0: is it found?"""
    out = {}
    for d, m in manifests("*detection-localisation"):
        cfg = m["config"]["imbalance"]
        biased = set(cfg.get("bias_layers") or [])
        model = os.path.basename(d).split("_")[2]
        n_experts = MODELS[model]["n_experts"]
        points = []
        for p in m["points"]:
            f = json.load(gzip.open(os.path.join(d, p["validation_file"]))) if p.get("validation_file") else {}
            path = os.path.join(d, f.get("records_file") or "")
            if not f.get("records_file") or not os.path.exists(path):
                continue
            counts = D.request_counts([json.loads(line) for line in gzip.open(path, "rt")], n_experts)
            # The capture's MoE-layer index differs from the model layer index the
            # bias names; moe_layers maps one to the other.
            moe_layers = f.get("moe_layers") or list(range(counts.shape[1]))
            expected = {i for i, layer in enumerate(moe_layers) if layer in biased}
            found = D.localise(counts.sum(axis=0), N_RANKS, threshold=1.3)
            flagged = set(found["flagged_layers"])
            points.append({"level": float(p["value"]), "mean_load": round(found["mean_load"], 3),
                           "max_layer_load": round(found["max_layer_load"], 3),
                           "expected_layers": sorted(expected), "flagged_layers": sorted(flagged),
                           "recall": round(len(flagged & expected) / len(expected), 3) if expected else None,
                           "precision": round(len(flagged & expected) / len(flagged), 3) if flagged else None,
                           "flagged_rank": found["flagged_rank"],
                           "consistent_rank_share": round(found["consistent_rank_share"], 3)})
        out[os.path.basename(d)] = {"bias_target": cfg.get("bias_target"), "bias_layers": sorted(biased),
                                    "points": points}
    return out


# --- B3. How quickly, and how often it cries wolf -----------------------------

DELAY_SETTINGS = [D.WindowSampler(window=w, layers=4, token_fraction=0.1) for w in (8, 32, 128)]


def timing(model, levels, threshold):
    """Detection delay and time to a false alarm, for the cheap screen settings."""
    if 0.0 not in levels:
        return {"skipped": "capture has no balanced arm"}
    # The onset to detect is the lowest level that actually costs latency.
    costly = [lv for lv, c in sorted(levels.items())
              if lv != 0.0 and threshold and D.busiest_rank(c.sum(axis=0), N_RANKS) >= threshold]
    if not costly:
        return {"skipped": "no captured level reaches the impact threshold"}
    onset = levels[costly[0]]
    rows = []
    for sampler in DELAY_SETTINGS:
        alarm = D.threshold_at_fpr(D.estimate_loads(levels[0.0], N_RANKS, sampler, seed=0), 0.01)
        for consecutive in (1, 2, 3):
            delay = D.onset_delay(levels[0.0], onset, N_RANKS, sampler, alarm,
                                  consecutive=consecutive, draws=200, seed=2)
            quiet = D.mean_windows_to_false_alarm(levels[0.0], N_RANKS, sampler, alarm,
                                                  consecutive=consecutive, draws=200, seed=3)
            rows.append({**delay, "alarm": round(alarm, 3),
                         "requests_to_false_alarm_p50": quiet["requests_to_false_alarm_p50"],
                         "quiet_over_horizon": quiet["quiet_over_horizon"]})
    return {"onset_level": costly[0], "onset_load": round(float(D.busiest_rank(onset.sum(axis=0), N_RANKS)), 3),
            "settings": rows}


# --- D. Stage 2: step time at matched batch size ------------------------------

def step_table(run_dir, label):
    path = os.path.join(run_dir, "step_profile", f"{label}.csv.gz")
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(gzip.open(path, "rt")))
    n = max(int(r["step"]) for r in rows) + 1
    ranks = sorted({int(r["rank"]) for r in rows})
    wall = np.zeros((len(ranks), n))
    reqs, tokens = np.zeros(n, int), np.zeros(n, int)
    for r in rows:
        i, s = ranks.index(int(r["rank"])), int(r["step"])
        wall[i, s] = float(r["wall_us"])
        if i == 0:
            reqs[s], tokens[s] = int(r["reqs"]) - 1, int(r["tokens"])
    return {"reqs": reqs, "tokens": tokens, "wall_us": wall.mean(axis=0)}


def stage2(cal):
    out = {}
    for model, pattern in PROFILED.items():
        d, m = one(pattern)
        if m is None:
            continue
        steps, loads = {}, {}
        for p in m["points"]:
            if not p.get("trace_dir"):
                continue
            table = step_table(d, os.path.basename(p["trace_dir"]))
            level = float(p["value"])
            if table is not None and level in cal[model]:
                steps[level], loads[level] = table, cal[model][level][0]
        out[model] = ({"run": os.path.basename(d), "levels": sorted(steps), **D.step_cost_fit(steps, loads)}
                      if len(steps) >= 3 else {"skipped": "fewer than three profiled levels with step tables"})
    return out


# --- E. Stage 3: which rank paces the step ------------------------------------

def stage3():
    out = {}
    for model, pattern in PROFILED.items():
        d, m = one(pattern)
        if m is None:
            continue
        summaries = {}
        for p in m["points"]:
            f = p.get("trace_metrics_file")
            if f and os.path.exists(os.path.join(d, f)):
                summaries[float(p["value"])] = json.load(gzip.open(os.path.join(d, f)))
        out[model] = ({"run": os.path.basename(d), "points": D.pace_setter_agreement(summaries, hot_rank=0)}
                      if summaries else {"skipped": "no trace metrics"})
    return out


# --- F. What confirmation costs ----------------------------------------------

def profiling_cost():
    """What the profiled pass costs, from points served twice in one run.

    ``separate_profiling_run`` serves each point unprofiled for the timings and
    again profiled for the traces, so the pair is one configuration measured
    with and without the profiler. Runs before 2026-10-07 discarded the profiled
    pass's timings, so only later runs can answer this.
    """
    out = {}
    for model, pattern in PROFILED.items():
        d, m = one(pattern)
        if m is None:
            continue
        pairs = []
        for p in m["points"]:
            plain, profiled = p.get("request_summary"), p.get("profiled_request_summary")
            if plain and profiled and plain.get("tpot_ms_mean") and profiled.get("tpot_ms_mean"):
                pairs.append({"level": float(p["value"]),
                              "unprofiled_ms": round(plain["tpot_ms_mean"], 1),
                              "profiled_ms": round(profiled["tpot_ms_mean"], 1),
                              "overhead_pct": round(100 * (profiled["tpot_ms_mean"] / plain["tpot_ms_mean"] - 1), 1)})
        out[model] = ({"run": os.path.basename(d), "pairs": pairs,
                       "median_overhead_pct": round(float(np.median([p["overhead_pct"] for p in pairs])), 1)}
                      if pairs else {"skipped": "this run discarded the profiled pass's timings; "
                                                "re-profile to measure it"})
    return out


# --- The always-on cost of counting -------------------------------------------

def screen_cost():
    """What the screen would cost per step, for a device-side counter.

    ``per_assignment_ns`` is a placeholder until it is measured on the device;
    the structure is what matters here, so the overhead of any sampling setting
    follows from one measured number.
    """
    rows = []
    for model, spec in MODELS.items():
        for fraction in (1.0, 0.25, 0.025):
            rows.append({"model": model, "assignment_fraction": fraction,
                         **D.monitoring_overhead(fraction, per_assignment_ns=2.0, tokens_per_step=512,
                                                 step_ms=160, n_layers=26 if model == "deepseek-v2" else 32,
                                                 top_k=spec["top_k"])})
    return {"assumed_per_assignment_ns": 2.0, "unmeasured": True, "rows": rows}


def main(out_dir="docs"):
    cal = calibration()
    store = {"drift_pct": DRIFT_PCT, "calibration": {m: {str(k): v for k, v in levels.items()}
                                                     for m, levels in cal.items()}}
    store["impact"] = impact(cal)
    store["benign"] = {}
    for model in MODELS:
        counts, labels = captures(model)
        if counts is None:
            store["benign"][model] = {"skipped": "no routed-expert capture pulled (npull excludes activations/)"}
            continue
        threshold = (store["impact"].get(model) or {}).get("impact_threshold")
        store["benign"][model] = benign(model, counts, labels, threshold)
    store["detection"] = {}
    for model in MODELS:
        levels, _ = capture_levels(model)
        if not levels:
            store["detection"][model] = {"skipped": "no detection capture pulled"}
            continue
        store["detection"][model] = detection(model, levels,
                                              (store["impact"].get(model) or {}).get("impact_threshold"))
    store["timing"] = {}
    for model in MODELS:
        lv, _ = capture_levels(model)
        store["timing"][model] = (timing(model, lv, (store["impact"].get(model) or {}).get("impact_threshold"))
                                  if lv else {"skipped": "no detection capture pulled"})
    store["localisation"] = localisation_run()
    store["stage2_step_time"] = stage2(cal)
    store["stage3_pace_setter"] = stage3()
    store["profiling_cost"] = profiling_cost()
    store["screen_cost"] = screen_cost()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "detection.json")
    json.dump(store, open(path, "w"), indent=1, default=float)

    print(f"drift floor: {DRIFT_PCT}% TPOT\n")
    for model, row in store["impact"].items():
        t = row["impact_threshold"]
        print(f"{model}: cost begins at {t:.2f}x busiest-rank load" if t else f"{model}: {row['note']}")
        print(f"   loads {row['loads']}  tpot% {row['tpot_pct']}")
    print()
    for model, row in store["benign"].items():
        if "skipped" in row:
            print(f"{model}: benign skipped - {row['skipped']}")
            continue
        loc = row["localisation"]
        topics = row["by_topic"]["loads"]
        print(f"{model}: benign pooled {row['pooled_load']:.3f}x "
              f"(pooled-layer form would read {row['pooled_load_layerwise_vs_pooled_form']:.3f}x), "
              f"{row['by_topic']['n_topics']} topics span {topics[0]:.3f}-{topics[-1]:.3f}x")
        print(f"   per-layer load {loc['per_layer_load_range'][0]:.3f}-{loc['per_layer_load_range'][1]:.3f}x, "
              f"busiest rank leads {loc['consistent_rank_share']:.0%} of layers "
              f"(a persistent hot rank would lead ~100%)")
        for s in row["sampling"]:
            if s["window"] in (8, 512) and s["token_fraction"] == 1.0:
                print(f"   window {s['window']:>4}  layers {str(s['layers']):>4}  "
                      f"counts {s['assignment_fraction']:>6.3f} of assignments  "
                      f"p99 {s['p99']:.3f}  spread +-{s['spread']:.3f}"
                      + (f"  false alarms {s['false_alarm_rate']:.3f}" if "false_alarm_rate" in s else ""))
        if "required_window" in row:
            r = row["required_window"]
            print(f"   to resolve {r['resolution']:.3f}x: {r['all_layers']} requests all layers, "
                  f"{r['four_layers']} with 4 layers, {r['four_layers_tenth_of_tokens']} with 4 layers "
                  f"and a tenth of tokens")
    print()
    for model, row in store["detection"].items():
        if "skipped" in row:
            print(f"{model}: detection skipped - {row['skipped']}")
            continue
        print(f"{model}: detection rate at 1% FPR, {row['n_requests']} captured requests per level")
        for s in row["settings"]:
            if s["window"] not in (1, 8, 64):
                continue
            print(f"   window {s['window']:>3} layers {str(s['layers']):>4} tok {s['token_fraction']:<4} "
                  f"counts {s['assignment_fraction']:.3f}  alarm {s['alarm_at_1pct_fpr']:.2f}  "
                  + "  ".join(f"{lv['true_load']:.2f}x:{lv['detection_rate']:.2f}"
                              + ("*" if lv["costly"] else "") for lv in s["levels"]))
        print("   (* = load at or above this model's impact threshold)")
    print()
    for model, row in store["timing"].items():
        if "skipped" in row:
            print(f"{model}: timing skipped - {row['skipped']}")
            continue
        print(f"{model}: onset at {row['onset_load']:.2f}x (the lowest costly level), "
              f"4 layers and a tenth of tokens")
        print(f"   {'window':>6} {'k':>2} {'detected':>9} {'false start':>12} {'delay p50':>10} "
              f"{'delay p90':>10} {'quiet for':>10}")
        for r in row["settings"]:
            quiet = r["requests_to_false_alarm_p50"]
            print(f"   {r['window']:>6} {r['consecutive']:>2} {r['detected_fraction']:>9.2f} "
                  f"{r['false_start_fraction']:>12.2f} "
                  f"{(r['delay_requests_p50'] or float('nan')):>10.0f} "
                  f"{(r['delay_requests_p90'] or float('nan')):>10.0f} "
                  f"{(f'{quiet:.0f} reqs' if quiet else '>horizon'):>10}")
    print()
    for run, row in store["localisation"].items():
        print(f"localisation, bias on {row['bias_target']} layers {row['bias_layers']}:")
        for p in row["points"]:
            print(f"   level {p['level']:g}: mean {p['mean_load']:.2f}x, worst layer {p['max_layer_load']:.2f}x, "
                  f"flagged {len(p['flagged_layers'])} layers (recall {p['recall']}, precision {p['precision']}), "
                  f"rank {p['flagged_rank']}")
    print()
    for model, row in store["stage2_step_time"].items():
        if "skipped" in row:
            print(f"{model}: stage 2 skipped - {row['skipped']}")
            continue
        print(f"{model}: stage 2, ms per +1x load by step kind")
        for b in row["by_bin"]:
            print(f"   batch {b['batch']:>8} {b['kind']:<8} {b['ms_per_load']:+8.2f} ms "
                  f"({b['pct_per_load']:+.1f}%, {b['steps']} steps)")
    print()
    for model, row in store["stage3_pace_setter"].items():
        if "skipped" in row:
            continue
        hits = [p for p in row["points"] if p["is_hot_rank"]]
        print(f"{model}: stage 3, hot rank paced {len(hits)} of {len(row['points'])} profiled levels "
              + ", ".join(f"{p['level']:g}:rank{p['pace_setter_rank']}" for p in row["points"]))
    print()
    for model, row in store["profiling_cost"].items():
        if "skipped" in row:
            print(f"{model}: confirmation cost unmeasured - {row['skipped']}")
        else:
            print(f"{model}: profiling costs {row['median_overhead_pct']:+.1f}% TPOT (median of "
                  f"{len(row['pairs'])} twinned points)")
    print(f"\nwrote {path}")
    return store


if __name__ == "__main__":
    main(*sys.argv[1:2])
