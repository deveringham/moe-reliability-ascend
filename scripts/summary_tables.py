"""The two answers as tables: where imbalance costs latency, and what detects it.

    uv run python scripts/summary_tables.py [<out_dir>]

Writes CSV into <out_dir> (default docs/figures/tables) and prints the same
rows. Every number is read from a run under results/ or from detection.json; the
run each row came from is a column, so any of them can be traced back.

  regimes.csv      one row per parameter setting that was varied, with the cost
                   it produced and the spread it was measured against
  detection.csv    what the screen catches, misses and costs, per setting
  calibration.csv  what a deployment's own calibration says about it
"""

from __future__ import annotations

import collections
import csv
import glob
import gzip
import json
import os
import re
import sys

import numpy as np

RESULTS = "results"
N_RANKS = 4


def manifests(pattern):
    for d in sorted(glob.glob(os.path.join(RESULTS, pattern))):
        path = os.path.join(d, "manifest.json")
        if os.path.exists(path):
            yield d, json.load(open(path))


def tpot_by_level(manifest):
    out = collections.defaultdict(list)
    for p in manifest["points"]:
        if (p.get("request_summary") or {}).get("tpot_ms_mean"):
            out[float(p["value"])].append(p["request_summary"]["tpot_ms_mean"])
    return out


def calibration_loads(pattern="*rbias-calibration"):
    """{model: {level: (per-layer load, live experts)}} from the calibration runs."""
    out: dict = {}
    for d, m in manifests(pattern):
        model = os.path.basename(d).split("_")[2]
        for p in m["points"]:
            if not p.get("validation_file"):
                continue
            f = json.load(gzip.open(os.path.join(d, p["validation_file"])))
            out.setdefault(model, {})[float(p["value"])] = (
                float(np.mean(f["rank_max_over_mean_per_layer"])),
                (p.get("validation_summary") or {}).get("active_experts"))
    return out


def own_loads(run_dir, manifest):
    out = {}
    for p in manifest["points"]:
        if p.get("validation_file"):
            f = json.load(gzip.open(os.path.join(run_dir, p["validation_file"])))
            out[float(p["value"])] = (float(np.mean(f["rank_max_over_mean_per_layer"])),
                                      (p.get("validation_summary") or {}).get("active_experts"))
    return out


def achieved_batch(run_dir):
    path = os.path.join(run_dir, "logs", "run.log")
    if not os.path.exists(path):
        return None
    seen = [int(n) for n in re.findall(r"Running: (\d+) reqs", open(path, errors="replace").read())]
    return max(seen) if seen else None


def cost_rows(pattern, group, condition, loads=None, model=None, note="", live_only=True):
    """One row per level of a sweep: its load, its TPOT, and the cost against the balanced arm."""
    rows = []
    for d, m in manifests(pattern):
        name = os.path.basename(d)
        mdl = model or name.split("_")[2]
        by = tpot_by_level(m)
        lv_loads = loads if loads is not None else own_loads(d, m) or calibration_loads().get(mdl, {})
        if 0.0 not in by or not lv_loads:
            continue
        full = max((v[1] for v in lv_loads.values() if v[1]), default=None)
        base = float(np.mean(by[0.0]))
        base_sd = float(np.std(by[0.0], ddof=1)) if len(by[0.0]) > 1 else 0.0
        for level in sorted(by):
            if level not in lv_loads:
                continue
            load, live = lv_loads[level]
            if live_only and full and live != full:
                continue
            vals = by[level]
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            mean = float(np.mean(vals))
            rel = np.hypot(sd / mean, base_sd / base) * (mean / base) * 100 if base else 0.0
            rows.append({
                "group": group, "condition": condition, "model": mdl,
                "npus": m["config"]["hardware"]["n_npus"],
                "batch_configured": m["config"]["server"]["batch_size"],
                "batch_achieved": achieved_batch(d) or "",
                "mode": "eager" if m["config"]["server"]["enforce_eager"] else "graph",
                "busiest_rank": round(load, 3), "repeats": len(vals),
                "tpot_ms": round(mean, 1), "tpot_sd_ms": round(sd, 1),
                "cost_pct": round(100 * (mean / base - 1), 2),
                "cost_sd_pct": round(rel, 2),
                "note": note, "run": name,
            })
    return rows


def regimes():
    rows = []
    cal = calibration_loads()
    rows += cost_rows("*rbias-controlled-000", "skew strength", "one rank hot, every layer",
                      loads=cal.get("deepseek-v2"))
    rows += cost_rows("*rbias-controlled-001", "skew strength", "one rank hot, every layer",
                      loads=cal.get("mixtral"))
    rows += cost_rows("*rbias-graph-threshold-000", "execution mode", "graph mode",
                      loads=cal.get("deepseek-v2"))
    rows += cost_rows("*rbias-graph-threshold-001", "execution mode", "graph mode",
                      loads=cal.get("mixtral"))
    # The same deployment as rbias-controlled-001, swept finely around the onset
    # and capturing its own routing, so its loads are this traffic's rather than
    # another run's.
    for d, m in manifests("*calibration-sweep"):
        rows += cost_rows(os.path.basename(d), "skew strength",
                          "one rank hot, every layer, fine levels", loads=own_loads(d, m))
    rows += cost_rows("*rotating-skew-000", "skew shape", "a different rank hot each layer")
    rows += cost_rows("*rotating-skew-001", "skew shape", "a different rank hot each layer")
    rows += cost_rows("*rbias-npu8-000", "expert parallelism", "8 ranks, one expert each",
                      loads=own_loads(*next(manifests("*rbias-npu8-calibration"), (None, None)))
                      if next(manifests("*rbias-npu8-calibration"), None) else None)
    for d, _m in manifests("*decode-ridge2-*"):
        rows += cost_rows(os.path.basename(d), "decode batch, strong skew",
                          f"batch {achieved_batch(d)}", loads=calibration_loads().get("mixtral"))
    for d, _m in manifests("*natural-batch-*"):
        rows += cost_rows(os.path.basename(d), "decode batch, natural strength",
                          f"batch {achieved_batch(d)}", loads=calibration_loads().get("mixtral"))
    # Traffic: no injected skew at all, so each locale is its own condition.
    locale_load = {"mmlu": 1.149, "mmmlu:DE_DE": 1.136, "mmmlu:ZH_CN": 1.210, "mmmlu:JA_JP": 1.225,
                   "mmmlu:AR_XY": 1.261, "mmmlu:HI_IN": 1.309, "mmmlu:SW_KE": 1.346,
                   "mmmlu:YO_NG": 1.469}
    english = None
    traffic = []
    for d, m in manifests("*language-latency-*"):
        w = m["config"]["benchmark"]["workload"]
        vals = [p["request_summary"]["tpot_ms_mean"] for p in m["points"]
                if (p.get("request_summary") or {}).get("tpot_ms_mean")]
        inp = [p["request_summary"]["input_tokens_mean"] for p in m["points"]
               if (p.get("request_summary") or {}).get("tpot_ms_mean")]
        if not vals or w not in locale_load:
            continue
        traffic.append({"group": "traffic (no injected skew)", "condition": w, "model": "mixtral",
                        "npus": 4, "batch_configured": m["config"]["server"]["batch_size"],
                        "batch_achieved": achieved_batch(d) or "", "mode": "eager",
                        "busiest_rank": locale_load[w], "repeats": len(vals),
                        "tpot_ms": round(float(np.mean(vals)), 1),
                        "tpot_sd_ms": round(float(np.std(vals, ddof=1)), 1),
                        "input_tokens": round(float(np.mean(inp))),
                        "run": os.path.basename(d)})
        if w == "mmlu":
            english = traffic[-1]
    for r in traffic:
        if english:
            r["cost_pct"] = round(100 * (r["tpot_ms"] / english["tpot_ms"] - 1), 2)
            r["note"] = "vs English; prompt length confounds, see language.png"
    rows += traffic
    return rows


def detection_rows():
    path = os.path.join("docs", "detection.json")
    if not os.path.exists(path):
        return []
    data = json.load(open(path))
    rows = []
    for model, row in (data.get("detection") or {}).items():
        if "settings" not in row:
            continue
        for s in row["settings"]:
            for lv in s["levels"]:
                rows.append({"group": "screen", "model": model, "window_requests": s["window"],
                             "layers": s["layers"] or "all", "layer_choice": s.get("layer_choice", "spaced"),
                             "token_fraction": s["token_fraction"],
                             "routing_counted_frac": s["assignment_fraction"],
                             "alarm": s["alarm_at_1pct_fpr"], "busiest_rank": lv["true_load"],
                             "costs_latency": lv["costly"], "detection_rate": lv["detection_rate"],
                             "auc": lv["auc"]})
    for model, row in (data.get("timing") or {}).items():
        if "settings" not in row:
            continue
        for s in row["settings"]:
            rows.append({"group": "alarm rule", "model": model, "window_requests": s["window"],
                         "consecutive_windows": s["consecutive"],
                         "detection_rate": s["detected_fraction"],
                         "fires_before_onset": s["false_start_fraction"],
                         "delay_requests_p50": s["delay_requests_p50"],
                         "requests_to_false_alarm": s["requests_to_false_alarm_p50"]})
    for run, row in (data.get("localisation") or {}).items():
        for s in row.get("sampling", []):
            rows.append({"group": "layer sampling", "model": "deepseek-v2",
                         "layers": s["layers_counted"], "layer_choice": s["layer_choice"],
                         "routing_counted_frac": s["assignment_fraction"],
                         "biased_layers_reachable": f"{s['biased_layers_reachable']}/{s['biased_layers']}",
                         "detection_rate": s["detection_rate"],
                         "note": "skew confined to 6 of 26 layers", "run": run})
        for s in row.get("accumulation", []):
            rows.append({"group": "localisation", "model": "deepseek-v2", "layers": 4,
                         "layer_choice": "random", "windows": s["windows"],
                         "requests": s["requests"], "layers_seen": f"{s['layers_seen']}/{s['of_layers']}",
                         "recall": s["recall"], "precision": s["precision"], "run": run})
    return rows


def calibration_rows():
    rows = []
    for path in sorted(glob.glob(os.path.join("docs", "calibrations", "*.json"))):
        cal = json.load(open(path))
        if "skipped" in cal:
            continue
        d = cal.get("deployment", {})
        base = {"model": cal.get("model"), "npus": d.get("n_npus"), "batch": d.get("batch_size"),
                "mode": "eager" if d.get("enforce_eager") else "graph",
                "noise_floor_pct": cal.get("noise_floor_pct"),
                "impact_threshold": cal.get("impact_threshold"),
                "benign_ceiling": cal.get("benign_ceiling"), "headroom": cal.get("headroom"),
                "loads_from": cal.get("loads_from"), "run": cal.get("run")}
        for m in cal.get("margin_sweep") or [{}]:
            rows.append({**base, "margin": m.get("margin"), "alarm": m.get("alarm"),
                         "false_alarm_rate": m.get("false_alarm_rate"),
                         "warns_before_cost": m.get("caught_before_cost"),
                         "catches_once_costly": m.get("caught_once_costly")})
    return rows


def write(rows, path):
    if not rows:
        print(f"  (no rows for {os.path.basename(path)})")
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  {os.path.basename(path)}: {len(rows)} rows")


def markdown(rows, columns, path, title):
    """A readable rendering of the same rows, for pasting into notes."""
    if not rows:
        return
    lines = [f"## {title}", "", "| " + " | ".join(columns) + " |",
             "|" + "|".join("---" for _ in columns) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in columns) + " |")
    open(path, "w").write("\n".join(lines) + "\n")
    print(f"  {os.path.basename(path)}")


def main(out="docs/figures/tables"):
    os.makedirs(out, exist_ok=True)
    print("writing tables:")
    reg, det, cal = regimes(), detection_rows(), calibration_rows()
    write(reg, os.path.join(out, "regimes.csv"))
    write(det, os.path.join(out, "detection.csv"))
    write(cal, os.path.join(out, "calibration.csv"))

    # One digest per question, with the columns that answer it.
    markdown(reg, ["group", "condition", "model", "npus", "batch_achieved", "mode",
                   "busiest_rank", "repeats", "tpot_ms", "tpot_sd_ms", "cost_pct", "cost_sd_pct"],
             os.path.join(out, "regimes.md"),
             "Where imbalance costs latency (cost is against each run's own balanced arm)")
    screen = [r for r in det if r["group"] in ("screen", "layer sampling")]
    markdown(screen, ["group", "model", "window_requests", "layers", "layer_choice",
                      "routing_counted_frac", "busiest_rank", "costs_latency", "detection_rate",
                      "biased_layers_reachable"],
             os.path.join(out, "detection_screen.md"), "What the screen catches, and at what sampling")
    rule = [r for r in det if r["group"] in ("alarm rule", "localisation")]
    markdown(rule, ["group", "model", "window_requests", "consecutive_windows", "detection_rate",
                    "fires_before_onset", "delay_requests_p50", "requests_to_false_alarm",
                    "windows", "requests", "layers_seen", "recall", "precision"],
             os.path.join(out, "detection_rule.md"), "The alarm rule, and localising by accumulation")
    markdown(cal, ["model", "npus", "batch", "mode", "noise_floor_pct", "impact_threshold",
                   "benign_ceiling", "headroom", "margin", "alarm", "false_alarm_rate",
                   "warns_before_cost", "catches_once_costly"],
             os.path.join(out, "calibration.md"),
             "What a deployment's own calibration says, and what each alarm margin buys")


if __name__ == "__main__":
    main(*sys.argv[1:2])
