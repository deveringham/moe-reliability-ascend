"""Does the same input produce the same routing? Measured on recorded captures.

Two comparisons, both on DeepSeek-V2-Lite activation records:

- same server: prompts the alpha-sweep capture happened to serve twice
  (3000 draws, 2910 unique), so identical config, different batch company;
- across configs: the 300 smoke-capture prompts (8 NPUs, batch 128), all of
  which recur in the alpha-sweep capture (4 NPUs, batch 512).

Records from before 2026-10-05 split prompt and generated routing at the wrong
index (see docs/data-format.md), so both arrays are concatenated and re-split
at num_input_tokens here. Layer 0 is dense in DeepSeek-V2 and is dropped.

    uv run python scripts/routing_determinism.py

Results are written up in docs/drift-findings.md.
"""

from __future__ import annotations

import collections
import gzip
import json
from pathlib import Path

import numpy as np

from moe_reliability_results import drift as D

RESULTS = Path(__file__).resolve().parents[1] / "results"
SWEEP = "20261001-133042_synthetic_deepseek-v2_npu4_bs512_alpha-sweep"
SMOKE = "20261001-131615_synthetic_deepseek-v2_npu8_bs128_sw-smoke"
N_EXPERTS, TOP_K = 64, 6


def load(run: str) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Prompt and generated routing per prompt text, in capture order."""
    out: dict[str, list[tuple[np.ndarray, np.ndarray]]] = collections.defaultdict(list)
    with gzip.open(RESULTS / run / "activations" / "records.jsonl.gz", "rt") as f:
        for line in f:
            rec = json.loads(line)
            ids = np.concatenate([np.asarray(rec["prompt_routed_experts"]),
                                  np.asarray(rec["routed_experts"])])[:, 1:, :]
            n = rec["num_input_tokens"]
            out[json.dumps(rec["prompt"], sort_keys=True)].append((ids[:n], ids[n:]))
    return out


def shared_experts(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Experts two captures agree on, per (token, layer), over their common length."""
    n = min(len(a), len(b))
    ma = np.zeros(a[:n].shape[:2] + (N_EXPERTS,), bool)
    mb = ma.copy()
    np.put_along_axis(ma, a[:n].astype(int), True, -1)
    np.put_along_axis(mb, b[:n].astype(int), True, -1)
    return (ma & mb).sum(-1)


def changed(overlap: np.ndarray) -> float:
    return float((TOP_K - overlap).sum() / (overlap.size * TOP_K))


def histogram(arrays: list[np.ndarray]) -> np.ndarray:
    n_layers = arrays[0].shape[1]
    h = np.zeros((n_layers, N_EXPERTS))
    for a in arrays:
        for layer in range(n_layers):
            h[layer] += np.bincount(a[:, layer, :].ravel(), minlength=N_EXPERTS)
    return h


def js(a: np.ndarray, b: np.ndarray) -> float:
    return float(D.js_divergence(D.distribution(a), D.distribution(b)).mean())


def report_pairs(label: str, pairs: list[tuple[tuple, tuple]]) -> None:
    prefill = [shared_experts(x[0], y[0]) for x, y in pairs]
    decode = [shared_experts(x[1], y[1]) for x, y in pairs]
    rates = np.array([changed(o) for o in prefill])
    slots = np.bincount(np.concatenate([o.ravel() for o in prefill]), minlength=TOP_K + 1)
    slots = slots / slots.sum()
    print(f"\n== {label}: {len(pairs)} prompts")
    print(f"prefill assignments changed: {changed(np.concatenate([o.ravel() for o in prefill])):.2%} "
          f"(per prompt {rates.min():.2%} to {rates.max():.2%}); "
          f"bit-identical prompts {sum((o == TOP_K).all() for o in prefill)}")
    print(f"prefill (token, layer) slots: identical {slots[TOP_K]:.1%}, one expert swapped "
          f"{slots[TOP_K - 1]:.1%}, two or more {slots[:TOP_K - 1].sum():.2%}")
    per_layer = sum((TOP_K - o).sum(0) for o in prefill) / (sum(o.shape[0] for o in prefill) * TOP_K)
    print("prefill changed by MoE layer:", " ".join(f"{x:.3f}" for x in per_layer))
    by_pos = np.zeros(200)
    n_pos = np.zeros(200)
    for o in decode:
        by_pos[:len(o)] += (TOP_K - o).sum(1)
        n_pos[:len(o)] += o.shape[1] * TOP_K
    rate = by_pos / np.maximum(n_pos, 1)
    print(f"decode assignments changed: {by_pos.sum() / n_pos.sum():.1%}; at tokens "
          + ", ".join(f"{i}: {rate[i]:.1%}" for i in (0, 1, 5, 10, 20, 50, 98)))
    print(f"pooled prefill JS between the two copies: {js(histogram([x[0] for x, _ in pairs]), histogram([y[0] for _, y in pairs])):.5f} bits")


def main() -> None:
    sweep = load(SWEEP)
    smoke = load(SMOKE)

    repeated = [g for g in sweep.values() if len(g) > 1]
    report_pairs("same server (prompts the sweep capture served twice)",
                 [(g[0], g[1]) for g in repeated])
    report_pairs("across configs (8 NPUs/bs128 vs 4 NPUs/bs512)",
                 [(smoke[k][0], sweep[k][0]) for k in smoke if k in sweep])

    # The comparison a canary replaces: as many *different* prompts each side.
    singles = [g[0][0] for g in sweep.values() if len(g) == 1]
    rng = np.random.default_rng(0)
    n = len(repeated)
    scores = []
    for _ in range(50):
        idx = rng.choice(len(singles), 2 * n, replace=False)
        scores.append(js(histogram([singles[i] for i in idx[:n]]),
                         histogram([singles[i] for i in idx[n:]])))
    print(f"\npooled prefill JS, {n} random prompts vs {n} other random prompts: "
          f"mean {np.mean(scores):.5f}, min {np.min(scores):.5f} bits")


if __name__ == "__main__":
    main()
