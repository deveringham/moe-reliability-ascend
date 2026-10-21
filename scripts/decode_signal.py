"""Does including decode tokens dilute the signal as well as the floor?

    uv run python scripts/decode_signal.py [<run-dir>] [--out docs/figures/tables]

D1c in docs/action-items.md. B3 measured prompt and decode routing to be
independent skews on *unbiased* traffic: prompt-only 1.0780x, decode-only
1.0638x, pooled 1.0517x - below either component, because their busiest ranks
agree on 7 of 26 layers and their per-rank shares correlate +0.011. Pooling
cancels them.

The open question was what that does to a screen, and it has a mechanistic
answer worth testing rather than assuming. Natural skew is whatever each token
happens to route to, so prefill and decode land independently and cancel. An
*injected* router bias is common-mode by construction: the plugin adds the same
per-expert offset to every selection call, prefill and decode alike. If that
holds, then including decode tokens lowers the benign floor while leaving the
injected signal intact - and the screen gets *more* sensitive with decode in the
window, not less, which is the opposite of what B3's floor measurement suggests
on its own.

So this scores, per level and per token source: the true pooled load, the window
distribution at the screen's sampling, and the detection rate against an alarm
set on that source's own level-0 windows. Equal footing per source, which is the
only way the comparison means anything - an alarm from one source applied to
another's readings measures the units mismatch instead (B2).

No latency is measured here and DeepSeek's impact threshold is withdrawn, so no
level below is costly or free; these are detection rates against a skew of known
size, not against a cost.
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
#: The screen as calibrated, and full counting as the noise-free reference.
SAMPLING = ((4, 0.1), (None, 1.0))
WINDOW = 8
SOURCES = {"prompt": ("prompt_routed_experts",),
           "decode": ("routed_experts",),
           "both": ("prompt_routed_experts", "routed_experts")}


def counts_for(records, n_experts, keys):
    """(requests, layers, experts) counts from one token source."""
    out = []
    for r in records:
        arrays = [np.asarray(r[k]) for k in keys if r.get(k) is not None]
        arrays = [a for a in arrays if a.size]
        if not arrays:
            raise ValueError(f"record {r.get('prompt_id')} has nothing for {keys}")
        ids = np.concatenate(arrays)
        out.append([np.bincount(ids[:, layer].ravel(), minlength=n_experts)
                    for layer in range(ids.shape[1])])
    c = np.asarray(out, dtype=float)
    return c[:, c[:, :, 1:].sum(axis=(0, 2)) > 0]


def rank_shares(pooled):
    """Per-layer per-rank share of a pooled (layers, experts) table."""
    total = pooled.sum(axis=1, keepdims=True)
    shares = np.divide(pooled, total, out=np.zeros_like(pooled), where=total > 0)
    return shares.reshape(pooled.shape[0], N_RANKS, -1).sum(axis=2)


def load_levels(run_dir):
    """{level: records} and the model, from a capture with one point per level."""
    manifest = json.load(open(os.path.join(run_dir, "manifest.json")))
    model = os.path.basename(run_dir.rstrip("/")).split("_")[2]
    levels = {}
    for point in manifest["points"]:
        if not point.get("validation_file"):
            continue
        meta = json.load(gzip.open(os.path.join(run_dir, point["validation_file"])))
        path = os.path.join(run_dir, meta.get("records_file") or "")
        if not meta.get("records_file") or not os.path.exists(path):
            continue
        levels[float(point["value"])] = [json.loads(line) for line in gzip.open(path, "rt")]
    return levels, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default=None)
    ap.add_argument("--out", default=os.path.join("docs", "figures", "tables"))
    args = ap.parse_args()
    run_dir = args.run or sorted(glob.glob(os.path.join("results", "*detection-capture-decode")))[-1]
    levels, model = load_levels(run_dir)
    if 0.0 not in levels:
        raise SystemExit(f"{run_dir}: no balanced level to score against")
    n_experts = MODELS[model]
    print(f"{os.path.basename(run_dir)}: levels {sorted(levels)}, {len(levels[0.0])} requests each")

    rows = []
    for source, keys in SOURCES.items():
        counts = {lv: counts_for(recs, n_experts, keys) for lv, recs in sorted(levels.items())}
        for layers, token_fraction in SAMPLING:
            sampler = D.WindowSampler(window=WINDOW, layers=layers, token_fraction=token_fraction,
                                      draws=2000, layer_choice="random")
            benign = D.estimate_loads(counts[0.0], N_RANKS, sampler, seed=0)
            alarm = D.threshold_at_fpr(benign, 0.01)
            for lv in sorted(counts):
                loads = D.estimate_loads(counts[lv], N_RANKS, sampler, seed=1)
                rows.append({
                    "model": model, "source": source, "level": lv,
                    "layers": layers or counts[lv].shape[1], "token_fraction": token_fraction,
                    "true_load": round(float(D.busiest_rank(counts[lv].sum(axis=0), N_RANKS)), 4),
                    "alarm": round(alarm, 4),
                    "window_median": round(float(np.median(loads)), 4),
                    "detection_rate": round(float((loads >= alarm).mean()), 3),
                    "auc": round(D.roc(loads, benign)["auc"], 4) if lv else None,
                })

    # Is the injected skew common-mode across the two sources, where natural skew
    # is independent? This is the mechanism the detection rates should follow.
    print(f"\n  {'level':>6} {'prompt':>8} {'decode':>8} {'both':>8}  "
          f"{'busiest rank agrees':>20} {'share corr':>11}")
    agreement = []
    for lv, recs in sorted(levels.items()):
        pooled = {s: counts_for(recs, n_experts, k).sum(axis=0) for s, k in SOURCES.items()}
        shares = {s: rank_shares(p) for s, p in pooled.items()}
        agree = int((shares["prompt"].argmax(1) == shares["decode"].argmax(1)).sum())
        n_layers = shares["prompt"].shape[0]
        corr = float(np.corrcoef(shares["prompt"].ravel(), shares["decode"].ravel())[0, 1])
        loads = {s: float(D.busiest_rank(p, N_RANKS)) for s, p in pooled.items()}
        print(f"  {lv:>6} {loads['prompt']:>8.4f} {loads['decode']:>8.4f} {loads['both']:>8.4f}  "
              f"{agree:>13} of {n_layers:<4} {corr:>+11.3f}")
        agreement.append({"model": model, "level": lv, "layers": n_layers,
                          "busiest_rank_agrees": agree, "share_correlation": round(corr, 4),
                          **{f"pooled_{s}": round(v, 4) for s, v in loads.items()}})

    print(f"\n  detection at a 1% alarm set on each source's own level 0, {WINDOW}-request windows")
    for layers, token_fraction in SAMPLING:
        counted = f"{layers or 'all'}x{token_fraction:g}"
        print(f"  {counted:>8} {'level':>6} {'true load':>10} " +
              "".join(f"{s:>9}" for s in SOURCES))
        for lv in sorted(levels):
            cells = []
            for source in SOURCES:
                match = [r for r in rows if r["source"] == source and r["level"] == lv
                         and r["token_fraction"] == token_fraction]
                cells.append(f"{match[0]['detection_rate']:>9.2f}" if match else f"{'-':>9}")
            true_both = [r for r in rows if r["source"] == "both" and r["level"] == lv
                         and r["token_fraction"] == token_fraction][0]["true_load"]
            print(f"  {'':>8} {lv:>6} {true_both:>10.3f} " + "".join(cells))

    # One table holds every model: replace this model's rows, keep the others'.
    os.makedirs(args.out, exist_ok=True)
    for name, data in (("decode_signal.csv", rows), ("decode_signal_sources.csv", agreement)):
        path = os.path.join(args.out, name)
        if os.path.exists(path):
            with open(path, newline="") as f:
                data = [r for r in csv.DictReader(f) if r["model"] != model] + data
        fields = list(dict.fromkeys(k for r in data for k in r))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(data)
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
