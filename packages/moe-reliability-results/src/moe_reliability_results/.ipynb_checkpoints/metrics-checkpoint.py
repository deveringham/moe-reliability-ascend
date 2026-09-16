###
# Preprocessing of raw measurements into aggregated metrics.
#
# desc
# Dylan Everingham
# 16.09.2026
###

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "PERCENTILES",
    "summarize_requests",
    "trace_scalars",
    "workload_point_stats",
    "TRACE_SCALAR_KEYS",
]

# Percentiles reported for latency distributions
PERCENTILES = (50, 90, 99)

# Scalar fields copied from a trace summary into run summaries
TRACE_SCALAR_KEYS = (
    "mean_over_ranks_us",
    "max_over_mean",
    "max_over_min",
    "hottest_rank",
    "total_over_ranks_ms",
    "calls_per_rank",
    "steps",
    "dom_mean_over_ranks_us",
    "dom_max_over_mean",
)


def _stats(prefix: str, values: Sequence[float]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        out[f"{prefix}_mean"] = None
        for p in PERCENTILES:
            out[f"{prefix}_p{p}"] = None
        out[f"{prefix}_max"] = None
        return out
    out[f"{prefix}_mean"] = float(arr.mean())
    for p, v in zip(PERCENTILES, np.percentile(arr, PERCENTILES)):
        out[f"{prefix}_p{p}"] = float(v)
    out[f"{prefix}_max"] = float(arr.max())
    return out

# Gets request counts, token totals, mean/percentile/max for TTFT, TPOT, and
# end-to-end request latency.
def summarize_requests(requests: Iterable[Mapping[str, Any]] | None) -> dict[str, Any]:
    if requests is None:
        return {"n_requests": 0}
    requests = list(requests)
    timed = [r for r in requests if r.get("ttft") is not None]
    summary: dict[str, Any] = {
        "n_requests": len(requests),
        "n_timed_requests": len(timed),
        "input_tokens_total": int(sum(r.get("num_input_tokens") or 0 for r in requests)),
        "output_tokens_total": int(sum(r.get("num_output_tokens") or 0 for r in requests)),
    }
    summary["input_tokens_mean"] = summary["input_tokens_total"] / len(requests) if requests else None
    summary["output_tokens_mean"] = summary["output_tokens_total"] / len(requests) if requests else None
    summary.update(_stats("ttft_ms", [r["ttft"] * 1000 for r in timed]))
    summary.update(_stats("tpot_ms", [r["tpot"] * 1000 for r in timed if r.get("tpot") is not None]))
    summary.update(_stats("e2e_s", [r.get("total_time") for r in requests]))
    return summary


def trace_scalars(trace_summary: Mapping[str, Any] | None) -> dict[str, Any]:
    if not trace_summary:
        return {}
    return {k: trace_summary.get(k) for k in TRACE_SCALAR_KEYS}

# Quality of workloads, i.e. error vs requested expert activation variance
# effective alpha is the per-layer ratio between obtained variance and natural variance
# from activation record
def workload_point_stats(workload: Mapping[str, Any], cv_nat: Sequence[float]) -> dict[str, Any]:
    obtained = np.asarray(workload["obtained_cvs"], dtype=float)
    nat = np.asarray(cv_nat, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        effective = obtained / nat
    effective = effective[np.isfinite(effective)]
    lo, mid, hi = (np.percentile(effective, [10, 50, 90]) if effective.size else (np.nan,) * 3)
    return {
        "n_prompts": len(workload.get("indices", [])),
        "mae": workload.get("mae"),
        "percent_unique_prompts": workload.get("percent_unique_prompts"),
        "effective_alpha_p10": float(lo),
        "effective_alpha_median": float(mid),
        "effective_alpha_p90": float(hi),
    }
