"""How long after a skew arrives does the screen raise it, on hardware?

    uv run python scripts/delay_eval.py [<run-dir>] [--out docs/figures/tables]

D2 in docs/action-items.md. Every onset figure in docs/detection-findings.md is
a splice of two static captures - a stream drawn from the balanced pool for 40
windows and from the skewed pool thereafter. `detection-delay` changes the bias
inside one running server at a known engine step
(`imbalance.bias_schedule`), with the screen counting live, so the delay is
measured rather than modelled.

Everything here is in engine steps, which is the only unit both sides share: the
schedule fires at a step and `server.count_experts` stamps every window with the
step it closed at. Requests are reported too, converted at the point's own
measured completion rate, because a delay in requests is what an operator cares
about - but the conversion is a rate, not a timestamp, so treat it as indicative.

The alarm is set on each point's own pre-onset windows at a chosen false-alarm
rate, not carried over from docs/calibrations. A step-cadence window holds about
twenty requests where the stored calibration describes eight, and B2 showed what
carrying an alarm across a change of window costs; the pre-onset stretch is the
benign calibration this run provides for itself.

Rank consistency is checked, not assumed: every rank logs its own schedule
transition, and a schedule that fired at different steps on different ranks would
mean the ranks briefly routed differently, which is the hazard the step-counted
design exists to avoid.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "packages",
                                "moe-reliability-results", "src"))

from moe_reliability_results import detection as D  # noqa: E402

N_RANKS = 4
#: Per-window false-alarm rates to set the alarm at. 0.005 is what
#: detection_params.py (2026-10-16) found better than the 1% used before.
FPRS = (0.005, 0.01, 0.02)
CONSECUTIVE = (1, 2, 3)
#: "router bias (pid 12345): step 1200 -> strength 0.5"
TRANSITION = re.compile(r"router bias \(pid (\d+)\): step (\d+) -> strength ([0-9.]+)")


def window_loads(path):
    """[(step, load, assignments)] from one point's counting file, in order."""
    out = []
    for line in open(path):
        rec = json.loads(line)
        rows = np.asarray(rec["counts"], dtype=float)
        if rows.ndim != 2 or not rows.size:
            continue
        per_layer = []
        for row in rows:
            if row.sum() <= 0:
                continue
            per_rank = row.reshape(N_RANKS, -1).sum(axis=1)
            per_layer.append(per_rank.max() / per_rank.mean())
        if per_layer:
            out.append((int(rec["step"]), float(np.mean(per_layer)), float(rows.sum())))
    return out


#: A window is a fixed number of steps, not a fixed amount of evidence: at
#: window_steps = 8 the counted assignments ranged 128 to 14240 within one point
#: of detection-delay, because the batch is tiny while the server fills and the
#: prefill/decode mix moves afterwards. A 128-assignment window reads 1.34-2.96x
#: on benign traffic - the same maximum-of-noisy-shares inflation as C6, in
#: extreme form - so scoring one is meaningless. Windows below this fraction of
#: the point's median are dropped, and the count is reported.
MIN_EVIDENCE = 0.5


def usable(loads, min_evidence=MIN_EVIDENCE):
    """Windows carrying at least a fraction of the point's median evidence."""
    counts = np.array([a for _, _, a in loads])
    if not len(counts):
        return [], 0
    floor = np.median(counts) * min_evidence
    kept = [w for w in loads if w[2] >= floor]
    return kept, len(loads) - len(kept)


def schedule_audit(run_dir, n_ranks=N_RANKS):
    """What the servers actually did with the schedule, across the whole run.

    The run keeps one log, and each point gets its own server, so the lines are
    not attributable to a point without matching pids to points. They do not need
    to be: every point in a run shares one schedule, so the audit is global - how
    many transitions fired, at which steps, and whether every rank of every
    server agreed. A step that differed between ranks would mean the ranks
    briefly routed differently, which is the hazard the step-counted trigger
    exists to prevent, and it would invalidate the run rather than one point.
    """
    steps, by_strength, pids = [], {}, set()
    for path in sorted(glob.glob(os.path.join(run_dir, "logs", "*"))
                       + glob.glob(os.path.join(run_dir, "*.log"))):
        if not os.path.isfile(path):
            continue
        try:
            text = open(path, errors="replace").read()
        except OSError:
            continue
        for pid, step, strength in TRANSITION.findall(text):
            steps.append(int(step))
            pids.add(pid)
            by_strength.setdefault(float(strength), []).append(int(step))
    return {"transitions": len(steps), "distinct_steps": sorted(set(steps)),
            "processes": len(pids), "by_strength": {s: sorted(set(v)) for s, v in by_strength.items()},
            "consistent": len(set(steps)) <= 1,
            "per_strength_ranks": {s: len(v) for s, v in by_strength.items()}}


def first_run(flags, consecutive):
    run = 0
    for i, flag in enumerate(flags):
        run = run + 1 if flag else 0
        if run >= consecutive:
            return i
    return None


def score(label, loads, onset_step, requests, point_steps, dropped=0):
    """Delay and out-of-sample false alarms for one point, per alarm and per k.

    The alarm is set on the first 60% of the clean pre-onset windows and judged
    on the rest. Setting and judging it on the same windows makes a k = 1 false
    alarm certain by construction - the quantile is attained in the sample it
    came from - which is what the first version of this script reported.
    """
    steps = np.array([s for s, _, _ in loads])
    values = np.array([v for _, v, _ in loads])
    pre_mask = steps < onset_step
    pre, post = values[pre_mask], values[~pre_mask]
    if len(pre) < 40 or not len(post):
        return []
    split = int(len(pre) * 0.6)
    fit, held = pre[:split], pre[split:]
    rate = requests / point_steps if point_steps else None
    rows = []
    for fpr in FPRS:
        alarm = float(np.quantile(fit, 1 - fpr))
        for k in CONSECUTIVE:
            early = first_run(held >= alarm, k)
            hit = first_run(post >= alarm, k)
            delay_steps = (int(steps[len(pre) + hit] - onset_step) if hit is not None else None)
            rows.append({
                "point": label, "onset_step": onset_step, "windows": len(loads),
                "windows_dropped_low_evidence": dropped,
                "pre_onset_windows": int(len(pre)), "calibrated_on": split,
                "judged_on": int(len(held)), "target_fpr": fpr, "consecutive": k,
                "alarm": round(alarm, 4),
                "pre_onset_median": round(float(np.median(pre)), 4),
                "post_onset_median": round(float(np.median(post)), 4),
                "pre_onset_alarming": round(float((held >= alarm).mean()), 4),
                "false_alarm_before_onset": early is not None,
                "detected": hit is not None,
                "delay_steps": delay_steps,
                "delay_windows": hit if hit is None else int(hit) + 1,
                "delay_requests": (round(delay_steps * rate, 1)
                                   if delay_steps is not None and rate else None),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default=None)
    ap.add_argument("--out", default=os.path.join("docs", "figures", "tables"))
    args = ap.parse_args()
    run_dir = args.run or sorted(glob.glob(os.path.join("results", "*detection-delay*")))[-1]
    manifest = json.load(open(os.path.join(run_dir, "manifest.json")))
    schedule = manifest.get("config", {}).get("imbalance", {}).get("bias_schedule") or []
    print(f"{os.path.basename(run_dir)}: schedule {schedule}")

    audit = schedule_audit(run_dir)
    print(f"  schedule fired {audit['transitions']} times across {audit['processes']} worker processes, "
          f"at step(s) {audit['distinct_steps']}, strengths {audit['per_strength_ranks']}")
    print(f"  every rank of every server agreed on the step: {audit['consistent']}")
    if not audit["consistent"]:
        print("  !! ranks disagreed on when the skew arrived; the points below are not comparable")

    rows = []
    for point in manifest["points"]:
        label = point["label"]
        path = os.path.join(run_dir, f"{label}_expert_counts.jsonl")
        if not os.path.exists(path):
            print(f"  {label}: no counting file, skipped")
            continue
        loads, dropped = usable(window_loads(path))
        if not loads:
            print(f"  {label}: no usable windows")
            continue
        onset_step = (audit["distinct_steps"][0] if audit["consistent"] and audit["distinct_steps"]
                      else (int(str(schedule[-1]).split(":")[0]) if schedule else None))
        consistent = audit["consistent"]
        metrics = point.get("metrics") or {}
        requests = next((metrics[k] for k in ("completed", "n_requests", "requests", "n_samples")
                         if isinstance(metrics.get(k), (int, float))),
                        manifest.get("config", {}).get("benchmark", {}).get("n_samples", 0))
        point_steps = max(s for s, _, _ in loads)
        print(f"  {label}: {len(loads)} windows to step {point_steps} ({dropped} dropped for thin "
              f"evidence), level {point['value']}, {requests} requests")
        if onset_step is None:
            continue
        for row in score(label, loads, onset_step, requests, point_steps, dropped):
            row["level"] = point["value"]
            row["ranks_agree"] = consistent
            rows.append(row)

    if not rows:
        raise SystemExit("no point had both a pre-onset stretch and a counting file")

    print(f"\n  {'point':<22} {'level':>5} {'fpr':>6} {'k':>2} {'alarm':>7} {'pre':>6} {'post':>6} "
          f"{'early':>6} {'delay steps':>12} {'reqs':>7}")
    for r in rows:
        if r["target_fpr"] != 0.005:
            continue
        print(f"  {r['point']:<22} {r['level']:>5} {r['target_fpr']:>6.1%} {r['consecutive']:>2} "
              f"{r['alarm']:>7.3f} {r['pre_onset_median']:>6.3f} {r['post_onset_median']:>6.3f} "
              f"{str(r['false_alarm_before_onset']):>6} "
              f"{('miss' if r['delay_steps'] is None else r['delay_steps']):>12} "
              f"{('-' if r['delay_requests'] is None else format(r['delay_requests'], '.0f')):>7}")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "detection_delay.csv")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n  -> {path}")


if __name__ == "__main__":
    main()
