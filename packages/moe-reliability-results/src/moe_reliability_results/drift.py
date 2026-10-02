"""Detecting that a router has changed, as opposed to its workload.

Both a router change and a workload change move the marginal distribution of
expert load, which is the only thing a load counter sees. They need different
responses - one is a model fault, the other is Tuesday - so a detector that
cannot separate them is not useful. The separation is possible because the two
act differently: a workload change alters which prompts arrive, while a router
change alters the mapping from prompt to expert. Conditioning the comparison on
an observable property of the prompt therefore removes the first and keeps the
second, which is what :func:`stratified_drift` does and
:func:`marginal_drift` does not.

Nothing here claims a drift score predicts a latency cost. On this stack it does
not: expert token counts overstate fused-MoE time imbalance by about six times
(see docs/imbalance-findings.md), so drift is a statement about the router, not
about serving cost.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

__all__ = [
    "record_histograms",
    "distribution",
    "js_divergence",
    "marginal_drift",
    "stratified_drift",
    "length_stratum",
    "roc_auc",
    "threshold_at_fpr",
]


def record_histograms(records: Iterable[dict[str, Any]], n_experts: int,
                      include_prefill: bool = True) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Per-record expert counts, shaped ``(n_records, n_layers, n_experts)``.

    Counting once up front is what makes sweeping windows and thresholds cheap:
    any window is then a sum over rows.
    """
    hists: list[np.ndarray] = []
    kept: list[dict[str, Any]] = []
    for rec in records:
        parts = [rec["routed_experts"]]
        if include_prefill and rec.get("prompt_routed_experts") is not None:
            parts.append(rec["prompt_routed_experts"])
        ids = np.concatenate([np.asarray(p) for p in parts], axis=0)  # (tokens, layers, k)
        if ids.size == 0:
            continue
        n_layers = ids.shape[1]
        flat = ids.reshape(-1, n_layers, ids.shape[2]).transpose(1, 0, 2).reshape(n_layers, -1)
        h = np.zeros((n_layers, n_experts), dtype=np.float64)
        for layer in range(n_layers):
            h[layer] = np.bincount(flat[layer], minlength=n_experts)[:n_experts]
        hists.append(h)
        kept.append(rec)
    if not hists:
        raise ValueError("no usable activation records")
    return np.stack(hists), kept


def distribution(hist: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """Normalise expert counts to a per-layer probability vector.

    The floor keeps an expert that a window never selected from making a
    divergence infinite; with 64 experts and a few thousand tokens that happens
    whenever routing is concentrated, which is exactly the case of interest.
    """
    totals = hist.sum(axis=-1, keepdims=True)
    out = np.divide(hist, totals, out=np.zeros_like(hist), where=totals > 0)
    return np.clip(out, floor, None)


def js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence in bits, per layer.

    Symmetric and bounded in [0, 1], unlike KL, so layers and windows are
    comparable and a single unseen expert cannot dominate the score.
    """
    m = 0.5 * (p + q)
    def kl(a, b):
        return np.sum(a * (np.log2(a) - np.log2(b)), axis=-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _score(window_hist: np.ndarray, reference_hist: np.ndarray, layers: Sequence[int] | None) -> float:
    p = distribution(window_hist)
    q = distribution(reference_hist)
    per_layer = js_divergence(p, q)
    if layers is not None:
        per_layer = per_layer[list(layers)]
    return float(np.mean(per_layer))


def marginal_drift(window: np.ndarray, reference: np.ndarray,
                   layers: Sequence[int] | None = None) -> float:
    """Drift of the pooled expert distribution: what a load counter would see.

    Rises for a router change and equally for a change in the prompt mix, which
    is why it is reported as the baseline to beat rather than as the detector.
    """
    return _score(window.sum(axis=0), reference.sum(axis=0), layers)


def stratified_drift(window: np.ndarray, reference: np.ndarray,
                     window_strata: Sequence[Any], reference_strata: Sequence[Any],
                     layers: Sequence[int] | None = None,
                     min_per_stratum: int = 2) -> float:
    """Drift within strata of the prompts, averaged over the reference's mix.

    A shift in which prompts arrive changes the weight of each stratum but not
    the routing inside one, so averaging per-stratum divergences against the
    reference's own weights cancels it. A router change alters routing inside
    every stratum and survives.

    Strata too thin to estimate on either side are dropped, so the score is
    always over the strata both sides actually cover.
    """
    w_idx: dict[Any, list[int]] = {}
    r_idx: dict[Any, list[int]] = {}
    for i, s in enumerate(window_strata):
        w_idx.setdefault(s, []).append(i)
    for i, s in enumerate(reference_strata):
        r_idx.setdefault(s, []).append(i)
    shared = [s for s in r_idx if s in w_idx
              and len(w_idx[s]) >= min_per_stratum and len(r_idx[s]) >= min_per_stratum]
    if not shared:
        return float("nan")
    weights = np.array([len(r_idx[s]) for s in shared], dtype=float)
    weights /= weights.sum()
    scores = np.array([_score(window[w_idx[s]].sum(axis=0),
                              reference[r_idx[s]].sum(axis=0), layers) for s in shared])
    return float(np.sum(weights * scores))


def length_stratum(rec: dict[str, Any], edges: Sequence[float] = (100, 150, 200, 300, 500)) -> int:
    """Bucket a record by its token count.

    Prompt length is the stratifier available in production - it needs no labels
    and the server already knows it - where a subject label does not exist
    outside a benchmark.
    """
    n = int(rec.get("num_input_tokens") or 0) + int(rec.get("num_output_tokens") or 0)
    if not n:
        ids = np.asarray(rec["routed_experts"])
        n = ids.shape[0]
    return int(np.searchsorted(np.asarray(edges), n))


def roc_auc(negatives: Sequence[float], positives: Sequence[float]) -> float:
    """Area under the ROC curve, as the rank statistic (ties counted as half).

    Equivalently the chance that a positive scores above a negative, so 0.5 is a
    detector that has learned nothing.
    """
    neg = np.asarray([x for x in negatives if np.isfinite(x)], dtype=float)
    pos = np.asarray([x for x in positives if np.isfinite(x)], dtype=float)
    if neg.size == 0 or pos.size == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    equal = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * equal) / (pos.size * neg.size))


def threshold_at_fpr(negatives: Sequence[float], fpr: float = 0.01) -> float:
    """Alarm threshold admitting at most ``fpr`` false alarms on the negatives.

    Calibrated empirically rather than from a chi-square null: tokens within a
    prompt route alike, so the effective sample size is far below the token
    count and an analytic null would reject almost everything.
    """
    neg = np.asarray([x for x in negatives if np.isfinite(x)], dtype=float)
    if neg.size == 0:
        return float("nan")
    return float(np.quantile(neg, 1.0 - fpr))
