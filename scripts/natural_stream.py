"""What the screen does on a long stream of traffic nobody skewed.

    uv run python scripts/natural_stream.py [<run-dir>...] [--out docs/figures/tables]

D1 in docs/action-items.md. Every alarm rate in docs/detection-findings.md is a
property of windows *bootstrapped* from 1200-1600 captured requests and scored
against an alarm set on those same requests. That leaves two things unmeasured:
whether the screen ever fires on traffic nobody skewed over a horizon long enough
to contain a false alarm at the rate it is calibrated to - one confirmed alarm
per 1200-2544 requests, so 1200 requests cannot contain one - and whether a
consecutive stream behaves like a bootstrap of it.

This scores tumbling windows over the order the requests were served, against the
alarm the deployment was calibrated to, and counts what actually fires.

Two caveats it prints rather than hides:

- With 700 requests in flight the recorded order is completion order, not
  submission order, so a window of 8 here is 8 requests that finished together.
  That is closer to what a monitor inside the server sees than a bootstrap is,
  and it is not the same window either.
- The alarm transfers but the impact threshold does not. An alarm is a quantile
  of benign traffic, so carrying one over to new traffic is a statement about
  the same quantity; an impact threshold is a latency measurement on a
  particular traffic mix and output length, and Mixtral's is the only one this
  project still supports.
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "packages",
                                "moe-reliability-results", "src"))

from moe_reliability_results import detection as D  # noqa: E402

N_RANKS = 4
MODELS = {"deepseek-v2": 64, "mixtral": 8}
#: The deployable alarm each model was calibrated to, read from the stored
#: calibration rather than retyped. It is *not* a false-alarm quantile: the
#: calibration places it a margin of the way from the benign ceiling to the
#: impact threshold (0.5, i.e. halfway), which is why DeepSeek's sits so far
#: above its benign traffic - 1.142x ceiling against a 2.434x onset - while
#: Mixtral's 1.349x nearly touches its 1.42x onset. Scoring against the 1%
#: quantile instead, as detection-findings' tables do, is a different and much
#: lower bar, and on DeepSeek the two disagree by 0.5x.
CALIBRATION = {
    "mixtral": "docs/calibrations/20261008-223820_imbalance_mixtral_npu4_bs512_calibration-sweep.json",
    "deepseek-v2": "docs/calibrations/20261006-155755_imbalance_deepseek-v2_npu4_bs512_rbias-controlled-000.json",
}
#: Mixtral's impact threshold, for context only. DeepSeek's was withdrawn on
#: 2026-10-14, so no DeepSeek row here may be called costly or free.
IMPACT = {"mixtral": 1.42, "deepseek-v2": None}


def stored_alarm(model):
    """(alarm level, how it was placed) from the model's own calibration file."""
    path = CALIBRATION.get(model)
    if not path or not os.path.exists(path):
        return None, None
    data = json.load(open(path))
    alarm = data.get("alarm") or {}
    return alarm.get("level"), f"margin {alarm.get('margin')} of ceiling->onset"
#: (layers, token_fraction): full counting, the screen as calibrated, and the
#: split detection_params.py found better at the same cost on 2026-10-16.
SAMPLING = ((None, 1.0), (4, 0.1), (8, 0.05))
WINDOWS = (8, 32)
#: Per-window false-alarm targets to self-calibrate at, for comparison with the
#: carried-over alarm.
SELF_FPR = (0.005, 0.01)


def load_capture(run_dir):
    """({token source: counts}, labels, model, meta) for a capture, in served order.

    When the capture generated more than one token per request, the same records
    are reduced twice: over prompt tokens only, as every capture behind the
    detection work did, and over prompt plus generated tokens, which is what a
    served window actually holds. The comparison is then within one capture - same
    requests, same server, same order - rather than across two runs.
    """
    manifest = json.load(open(os.path.join(run_dir, "manifest.json")))
    model = os.path.basename(run_dir.rstrip("/")).split("_")[2]
    n_experts = MODELS[model]
    generated = int(manifest.get("config", {}).get("client", {}).get("max_new_tokens", 1))
    for point in manifest["points"]:
        if not point.get("validation_file"):
            continue
        meta = json.load(gzip.open(os.path.join(run_dir, point["validation_file"])))
        path = os.path.join(run_dir, meta.get("records_file") or "")
        if not meta.get("records_file") or not os.path.exists(path):
            continue
        records = [json.loads(line) for line in gzip.open(path, "rt")]
        counts = {"prompt+decode" if generated > 1 else "prompt":
                  D.request_counts(records, n_experts, include_generated=True)}
        if generated > 1:
            counts["prompt"] = D.request_counts(records, n_experts, include_generated=False)
        labels = np.array([r.get("subject") or "unknown" for r in records])
        return counts, labels, model, meta
    return None, None, model, None


def stream_loads(counts, window, layers, token_fraction, seed=0):
    """Busiest-rank load of consecutive tumbling windows, in order.

    Not a bootstrap: window i is requests [i*w, (i+1)*w) as they were served.
    The layers are redrawn for each window, which is what the screen does and
    what makes a layer-local skew visible at all.
    """
    rng = np.random.default_rng(seed)
    n_layers = counts.shape[1]
    out = []
    for start in range(0, len(counts) - window + 1, window):
        block = counts[start:start + window]
        if layers is not None and layers < n_layers:
            block = block[:, np.sort(rng.choice(n_layers, layers, replace=False))]
        pooled = block.sum(axis=0)
        if token_fraction < 1.0:
            pooled = rng.binomial(pooled.astype(np.int64), token_fraction).astype(float)
        out.append(float(D.batch_busiest_rank(pooled[None, ...], N_RANKS)[0]))
    return np.array(out)


def confirmed_alarms(flags, consecutive):
    """Indices of windows completing a run of ``consecutive`` alarms."""
    out, run = [], 0
    for i, flag in enumerate(flags):
        run = run + 1 if flag else 0
        if run == consecutive:
            out.append(i)
    return out


def score(counts, label, model, window, layers, token_fraction, tokens="prompt",
          alarm=None, alarm_source="stored"):
    """One stream at one sampling, against one alarm.

    ``alarm`` is passed in rather than looked up, because the two alarms worth
    scoring are not interchangeable: the stored one is the deployable threshold
    and the self-calibrated one is what a deployment would get if it calibrated
    on its own mixed traffic and then served a single corpus - which is the
    failure this run is looking for.
    """
    loads = stream_loads(counts, window, layers, token_fraction)
    if not len(loads) or alarm is None:
        return None
    flags = loads >= alarm
    requests = len(loads) * window
    row = {"model": model, "stream": label, "tokens": tokens,
           "alarm_source": alarm_source,
           "requests": requests, "windows": len(loads),
           "window": window, "layers": layers or counts.shape[1], "of_layers": int(counts.shape[1]),
           "token_fraction": token_fraction,
           "pooled_load": round(float(D.busiest_rank(counts.sum(axis=0), N_RANKS)), 4),
           "window_median": round(float(np.median(loads)), 4),
           "window_p99": round(float(np.quantile(loads, 0.99)), 4),
           "window_max": round(float(loads.max()), 4),
           "alarm": round(float(alarm), 4),
           "impact_threshold": IMPACT.get(model),
           "windows_alarming": int(flags.sum()),
           "per_window_rate": round(float(flags.mean()), 4)}
    for k in (1, 2, 3):
        fired = confirmed_alarms(flags, k)
        row[f"confirmed_k{k}"] = len(fired)
        # The number this run exists to measure: how much benign traffic passes
        # per confirmed alarm, where the simulation said 1200-2544 requests.
        row[f"requests_per_alarm_k{k}"] = round(requests / len(fired), 1) if fired else None
    for fpr in SELF_FPR:
        self_alarm = float(np.quantile(loads, 1 - fpr))
        row[f"self_alarm_{fpr}"] = round(self_alarm, 4)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="run directories (default: every natural-traffic capture)")
    ap.add_argument("--out", default=os.path.join("docs", "figures", "tables"))
    args = ap.parse_args()
    runs = args.runs or sorted(glob.glob(os.path.join("results", "*_natural-*")))
    if not runs:
        raise SystemExit("no natural-long run found; ./npull first")

    rows = []
    for run_dir in runs:
        sources, labels, model, meta = load_capture(run_dir)
        if sources is None:
            print(f"{os.path.basename(run_dir)}: no capture records, skipped")
            continue
        any_counts = next(iter(sources.values()))
        print(f"\n{os.path.basename(run_dir)}: {len(any_counts)} requests, {any_counts.shape[1]} MoE "
              f"layers, {meta.get('validation_workload')}, token sources {sorted(sources)}")
        # Families, not subjects: the corpus is what moves the benign floor.
        families = np.array([s.split("/")[0] for s in labels])
        deployed, how = stored_alarm(model)
        print(f"  stored alarm {deployed} ({how}); impact threshold {IMPACT.get(model) or 'withdrawn'}")
        for tokens, counts in sorted(sources.items()):
            streams = [("mixed", counts)]
            for family in sorted(set(families.tolist())):
                idx = np.flatnonzero(families == family)
                if len(idx) >= 8 * 20:
                    streams.append((family, counts[idx]))
            for window in WINDOWS:
                for layers, token_fraction in SAMPLING:
                    # The mixed stream at this sampling is the benign reference.
                    mixed_loads = stream_loads(counts, window, layers, token_fraction)
                    self_alarm = (float(np.quantile(mixed_loads, 0.99)) if len(mixed_loads) else None)
                    for label, block in streams:
                        for alarm, source in ((deployed, "stored"), (self_alarm, "self, mixed at 1%")):
                            row = score(block, label, model, window, layers, token_fraction, tokens,
                                        alarm=alarm, alarm_source=source)
                            if row:
                                row["run"] = os.path.basename(run_dir)
                                rows.append(row)

        head = [r for r in rows if r["run"] == os.path.basename(run_dir) and r["window"] == 8]
        print(f"  {'stream':<13} {'tokens':<14} {'count':>8} {'alarm from':<18} {'level':>6} "
              f"{'pooled':>7} {'p99':>7} {'fire%':>6} {'k2':>4} {'reqs/k2':>8}")
        for r in head:
            counted = f"{r['layers']}x{r['token_fraction']:g}"
            per = r["requests_per_alarm_k2"]
            print(f"  {r['stream']:<13} {r['tokens']:<14} {counted:>8} {r['alarm_source']:<18} "
                  f"{r['alarm']:>6.3f} {r['pooled_load']:>7.3f} {r['window_p99']:>7.3f} "
                  f"{r['per_window_rate']:>6.1%} {r['confirmed_k2']:>4} "
                  f"{('-' if per is None else format(per, '.0f')):>8}")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "natural_stream.csv")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n  -> {path}")


if __name__ == "__main__":
    main()
