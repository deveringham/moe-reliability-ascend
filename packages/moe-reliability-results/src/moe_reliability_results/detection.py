"""Detecting imbalance that costs latency, from what a server could count online.

The pipeline this supports has three stages, cheapest first:

1. **Screen.** Count token-expert assignments per layer and rank, estimate the
   busiest rank's load over a window, and flag a window whose load clears a
   threshold calibrated on benign traffic.
2. **Confirm, cheap.** Relate the flagged load to engine step time, conditioned
   on batch size and prefill content, which is what actually decides whether the
   skew costs anything.
3. **Confirm, expensive.** Profile a short window and attribute the step to a
   rank.

Stage 1 is the only part that has to run always, so its accuracy and its cost
are the design question: how many tokens and layers must be counted, over how
long a window, to separate costly skew from benign traffic.

Why a per-layer statistic. ``busiest_rank`` takes the busiest rank *in each
layer* and averages over layers. Pooling layers first, as vLLM's EPLB
balancedness did before it was fixed upstream, lets a rank that is hot in one
layer and cold in another cancel: it reads 1.02x on DeepSeek traffic where the
per-layer form reads 1.14x, and 1.00x ("perfectly balanced") for a rank hot in
every layer. The two agree only when one rank is hot everywhere.

What this module does not know. A load estimate is not a cost: on DeepSeek the
measured TPOT is flat to 2.34x busiest-rank load and only then rises, so a
screen calibrated on load alone would fire on skew that costs nothing. The
thresholds here therefore come from an impact curve (:func:`impact_threshold`),
measured per model, not from the load distribution alone.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = ["request_counts", "busiest_rank", "pooled_busiest_rank", "per_layer_load", "localise",
           "WindowSampler", "SampleCost", "estimate_loads", "threshold_at_fpr", "impact_threshold",
           "roc", "screen_scores"]


def request_counts(records: Iterable[Mapping], n_experts: int, include_generated: bool = True) -> np.ndarray:
    """Token-expert assignment counts per request and layer: ``(requests, layers, experts)``.

    Routed-expert captures store prompt and generated tokens separately
    (``prompt_routed_experts`` and ``routed_experts``, each
    ``[tokens][layers][top_k]``). Both are pooled by default, as a serving
    window holds both. Layers that never route - DeepSeek's dense first layers
    appear in a capture as all-zero ids - are dropped, so layer indices here are
    MoE-layer indices, as in ``rank_max_over_mean_per_layer``.
    """
    counts = []
    for r in records:
        # Either field may be absent: a capture with max_new_tokens = 1 leaves the
        # split undone and returns every token in routed_experts, while a capture
        # that split them fills both.
        keys = ("prompt_routed_experts", "routed_experts") if include_generated else ("prompt_routed_experts",)
        arrays = [np.asarray(r[k]) for k in keys if r.get(k) is not None]
        if not arrays and not include_generated and r.get("routed_experts") is not None:
            arrays = [np.asarray(r["routed_experts"])]  # unsplit capture: prompt tokens are all there is
        if not arrays:
            raise ValueError(f"record {r.get('prompt_id')} has no routed experts")
        ids = np.concatenate([a for a in arrays if a.size])
        counts.append([np.bincount(ids[:, layer].ravel(), minlength=n_experts) for layer in range(ids.shape[1])])
    c = np.asarray(counts, dtype=float)
    if not c.size:
        raise ValueError("no routed-expert records")
    return c[:, c[:, :, 1:].sum(axis=(0, 2)) > 0]


def _per_rank(counts: np.ndarray, n_ranks: int) -> np.ndarray:
    """Share of each layer's assignments on each rank: ``(layers, ranks)``.

    Contiguous placement, which is what vLLM does without EPLB.
    """
    totals = counts.sum(axis=1, keepdims=True)
    shares = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    return shares.reshape(shares.shape[0], n_ranks, -1).sum(axis=2)


def busiest_rank(counts: np.ndarray, n_ranks: int) -> float:
    """Busiest rank's load over the mean, per layer, averaged over layers.

    ``counts`` is ``(layers, experts)``. 1.0 is perfectly balanced; 2.0 means the
    busiest rank carries twice the mean rank's assignments in the average layer.
    """
    return float((_per_rank(counts, n_ranks).max(axis=1) * n_ranks).mean())


def pooled_busiest_rank(counts: np.ndarray, n_ranks: int) -> float:
    """The same quantity with layers pooled first - the form that hides a rotating hot rank.

    Kept to quantify the difference, not because any analysis should prefer it.
    """
    per_rank = _per_rank(counts, n_ranks).mean(axis=0)
    return float(per_rank.max() * n_ranks)


def per_layer_load(counts: np.ndarray, n_ranks: int) -> tuple[np.ndarray, np.ndarray]:
    """Each layer's busiest-rank load and which rank that is: ``(layers,)``, ``(layers,)``.

    The same counts the screen already holds, reduced one step less far. A mean
    over the loads is :func:`busiest_rank`.
    """
    per_rank = _per_rank(counts, n_ranks)
    return per_rank.max(axis=1) * n_ranks, per_rank.argmax(axis=1)


def localise(counts: np.ndarray, n_ranks: int, threshold: float) -> dict:
    """Which layers carry a skew, and whether one rank carries it throughout.

    Natural imbalance need not sit on one rank in every layer, and a skew in a
    few layers is diluted by the average that :func:`busiest_rank` takes: a
    detector that reports only the mean cannot say where to look. The per-layer
    loads cost nothing extra, since the counts are already per layer.

    ``consistent_rank`` is the rank busiest in most layers and the share of
    layers it leads. A persistent hot rank (what injected router bias makes, and
    what expert placement can fix) has a share near 1; a rotating one, which is
    what natural traffic shows, sits near 1/n_ranks.
    """
    loads, hot = per_layer_load(counts, n_ranks)
    flagged = np.flatnonzero(loads >= threshold)
    ranks, tally = np.unique(hot, return_counts=True)
    leader = int(ranks[np.argmax(tally)])
    return {"per_layer_load": loads.tolist(), "hot_rank_per_layer": hot.tolist(),
            "flagged_layers": flagged.tolist(), "flagged_fraction": float(len(flagged) / len(loads)),
            "mean_load": float(loads.mean()), "max_layer_load": float(loads.max()),
            "consistent_rank": leader, "consistent_rank_share": float(tally.max() / len(hot)),
            "flagged_rank": int(np.bincount(hot[flagged], minlength=n_ranks).argmax()) if len(flagged) else None}


@dataclass(frozen=True)
class SampleCost:
    """What one sampling setting counts, as a share of every assignment."""

    layer_fraction: float
    token_fraction: float

    @property
    def assignment_fraction(self) -> float:
        return self.layer_fraction * self.token_fraction


@dataclass(frozen=True)
class WindowSampler:
    """How a monitor would sample: a window of requests, some layers, some tokens.

    ``window`` requests are the unit here because an offline capture records
    requests, not engine steps; a serving window is a time interval holding
    whatever tokens the batch carried. ``layers`` counts how many MoE layers are
    instrumented (evenly spaced, as a monitor would pick them); ``token_fraction``
    is the share of each request's tokens counted.
    """

    window: int
    layers: int | None = None
    token_fraction: float = 1.0
    draws: int = 200

    def cost(self, n_layers: int) -> SampleCost:
        layers = n_layers if self.layers is None else min(self.layers, n_layers)
        return SampleCost(layer_fraction=layers / n_layers, token_fraction=self.token_fraction)

    def _layer_index(self, n_layers: int) -> np.ndarray:
        if self.layers is None or self.layers >= n_layers:
            return np.arange(n_layers)
        return np.unique(np.linspace(0, n_layers - 1, self.layers).round().astype(int))


def estimate_loads(counts: np.ndarray, n_ranks: int, sampler: WindowSampler,
                   seed: int = 0) -> np.ndarray:
    """Busiest-rank load as a monitor with this sampling would estimate it, one value per draw.

    Each draw takes ``sampler.window`` requests at random, keeps the sampled
    layers, thins each window's counts to ``token_fraction`` by binomial
    sampling, and reduces. Thinning is applied to the pooled window rather than
    per token, which is the same multinomial and far cheaper.
    """
    rng = np.random.default_rng(seed)
    layers = sampler._layer_index(counts.shape[1])
    window = min(sampler.window, len(counts))
    out = np.empty(sampler.draws)
    for i in range(sampler.draws):
        chosen = rng.choice(len(counts), window, replace=False)
        pooled = counts[np.ix_(chosen, layers)].sum(axis=0)
        if sampler.token_fraction < 1.0:
            pooled = rng.binomial(pooled.astype(int), sampler.token_fraction).astype(float)
        out[i] = busiest_rank(pooled, n_ranks)
    return out


def threshold_at_fpr(benign: Sequence[float], fpr: float) -> float:
    """The alarm level that benign traffic clears at most ``fpr`` of the time."""
    if not 0 < fpr < 1:
        raise ValueError("fpr must be in (0, 1)")
    return float(np.quantile(np.asarray(benign, dtype=float), 1 - fpr))


def impact_threshold(loads: Sequence[float], costs: Sequence[float], noise_pct: float) -> float | None:
    """The lowest measured load whose cost clears ``noise_pct``, by linear interpolation.

    ``costs`` are percentage changes in latency against the balanced point and
    ``noise_pct`` is the run-to-run drift between identical points - the floor
    below which a cost is not distinguishable. Returns None when no measured
    point reaches it, which is the honest answer for a model swept only below
    its threshold.
    """
    order = np.argsort(np.asarray(loads, dtype=float))
    x, y = np.asarray(loads, dtype=float)[order], np.asarray(costs, dtype=float)[order]
    for i in range(1, len(x)):
        if y[i] >= noise_pct > y[i - 1]:
            span = y[i] - y[i - 1]
            # Interpolate so a coarse sweep does not round the threshold up to its
            # next measured level.
            return float(x[i - 1] + (x[i] - x[i - 1]) * ((noise_pct - y[i - 1]) / span)) if span else float(x[i])
    return float(x[0]) if len(y) and y[0] >= noise_pct else None


def roc(positive: Sequence[float], negative: Sequence[float]) -> dict:
    """AUC and the separation of two score sets, plus the detection rate at 1% FPR."""
    pos, neg = np.asarray(positive, dtype=float), np.asarray(negative, dtype=float)
    if not pos.size or not neg.size:
        raise ValueError("roc needs both positive and negative scores")
    # Mann-Whitney U, which is the AUC, with ties counted as half.
    comparisons = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    at_1pct = threshold_at_fpr(neg, 0.01)
    return {"auc": float(comparisons / (pos.size * neg.size)),
            "threshold_at_1pct_fpr": at_1pct,
            "detection_rate_at_1pct_fpr": float((pos >= at_1pct).mean()),
            "separation": float(pos.mean() - neg.mean()),
            "positive_median": float(np.median(pos)), "negative_p99": float(np.quantile(neg, 0.99))}


@dataclass
class screen_scores:  # noqa: N801 - a result record, named for what it holds
    """Stage-1 outcome for one sampling setting against one condition."""

    sampler: WindowSampler
    cost: SampleCost
    loads: np.ndarray
    truth: float | None = None
    extra: dict = field(default_factory=dict)

    @property
    def bias(self) -> float | None:
        """Estimate minus the load measured over the whole capture, when known."""
        return None if self.truth is None else float(np.median(self.loads) - self.truth)

    @property
    def spread(self) -> float:
        """Half the central 98% interval: the resolution this setting buys."""
        lo, hi = np.quantile(self.loads, [0.01, 0.99])
        return float((hi - lo) / 2)

    def summary(self) -> dict:
        return {"window": self.sampler.window, "layers": self.sampler.layers,
                "token_fraction": self.sampler.token_fraction,
                "assignment_fraction": round(self.cost.assignment_fraction, 4),
                "median": round(float(np.median(self.loads)), 4), "spread": round(self.spread, 4),
                "bias": None if self.bias is None else round(self.bias, 4), **self.extra}


def required_window(counts: np.ndarray, n_ranks: int, resolution: float,
                    windows: Sequence[int] = (1, 4, 16, 64, 256, 1024),
                    layers: int | None = None, token_fraction: float = 1.0,
                    draws: int = 200, seed: int = 0) -> int | None:
    """The smallest listed window whose estimate lands within ``resolution`` of the truth.

    "Within" means the central 98% of draws, so this is the window at which the
    screen can tell two loads that far apart from each other.
    """
    truth = busiest_rank(counts.sum(axis=0), n_ranks)
    for window in sorted(windows):
        sampler = WindowSampler(window=window, layers=layers, token_fraction=token_fraction, draws=draws)
        loads = estimate_loads(counts, n_ranks, sampler, seed=seed)
        lo, hi = np.quantile(loads, [0.01, 0.99])
        if max(abs(hi - truth), abs(truth - lo)) <= resolution:
            return int(window)
    return None


def step_cost_fit(steps: Mapping[str, np.ndarray], loads: Mapping[float, float],
                  min_steps: int = 8) -> dict:
    """Stage 2: does step wall time rise with load, at matched batch size?

    ``steps`` maps a sweep level to a record array with ``reqs``, ``tokens`` and
    ``wall_us`` per step; ``loads`` maps the same levels to busiest-rank load.
    Steps are compared within a batch-size bin and separated into decode-only
    and prefill-carrying, because the step mix moves with the swept parameter:
    averaging a profiled window compares different workloads, which is what made
    window averages disagree with TPOT by 1.5-3x.
    """
    bins = ((1, 200), (200, 350), (350, 480), (480, 100000))
    rows = []
    for lo, hi in bins:
        for kind in ("decode", "prefill"):
            points = []
            for level, table in steps.items():
                reqs, tokens, wall = table["reqs"], table["tokens"], table["wall_us"]
                decode = tokens == reqs
                sel = (decode if kind == "decode" else ~decode) & (reqs >= lo) & (reqs < hi)
                if sel.sum() >= min_steps:
                    points.append((loads[level], float(np.median(wall[sel])) / 1e3, int(sel.sum())))
            if len(points) >= 3:
                x = np.array([p[0] for p in points])
                y = np.array([p[1] for p in points])
                slope = float(np.polyfit(x, y, 1)[0])
                rows.append({"batch": f"{lo}-{hi if hi < 100000 else 'max'}", "kind": kind,
                             "levels": len(points), "steps": sum(p[2] for p in points),
                             "ms_per_load": round(slope, 2),
                             "pct_per_load": round(100 * slope / y[int(np.argmin(x))], 1)})
    return {"by_bin": rows}


def pace_setter_agreement(trace_metrics: Mapping[float, Mapping], hot_rank: int = 0) -> list[dict]:
    """Stage 3: does the rank that paces each step match the biased rank?

    Reads the ``collective_wait`` block a trace summary already carries. A rank
    that arrives early waits *inside* the collective and so counts as busy, so
    the pace setter is the rank that arrives last, not the idlest-looking one.
    In balanced runs some rank still paces each server instance - a different one
    every run - so agreement is only evidence against that baseline.
    """
    out = []
    for level, summary in sorted(trace_metrics.items()):
        wait = summary.get("collective_wait") or {}
        out.append({"level": level, "pace_setter_rank": wait.get("pace_setter_rank"),
                    "pace_setter_share": wait.get("pace_setter_share"),
                    "is_hot_rank": wait.get("pace_setter_rank") == hot_rank,
                    "wait_pct": wait.get("wait_pct")})
    return out


def monitoring_overhead(assignment_fraction: float, per_assignment_ns: float,
                        tokens_per_step: int, step_ms: float, n_layers: int, top_k: int) -> dict:
    """Share of a step a counting monitor would cost, given a per-assignment cost.

    A scatter-add into a resident counter is the cheap implementation; the point
    of expressing it this way is that ``per_assignment_ns`` is the one number
    that has to be measured on the device, and everything else follows from the
    deployment. Host-side counting is a different and much worse story: the
    router-bias plugin's per-call tensor add cost 10.0 ms per token on eager
    DeepSeek, where decode is host-bound.
    """
    assignments = tokens_per_step * n_layers * top_k * assignment_fraction
    cost_ms = assignments * per_assignment_ns / 1e6
    return {"assignments_per_step": assignments, "cost_ms_per_step": cost_ms,
            "pct_of_step": 100 * cost_ms / step_ms if step_ms else math.inf}
