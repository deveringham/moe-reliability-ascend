"""Graded router imbalance, injected at serving time.

A per-expert bias is added to the router logits before top-k selection, so a
chosen set of experts - all of one expert-parallel rank's, or an explicit list -
wins more often, by an amount set by the bias strength. The router is otherwise
untouched, so routing keeps its input dependence; this is what the checkpoint
recipe in ``core/forced_imbalance.py`` cannot do. That recipe zero-centres every
router, which ties the other experts' logits and collapses routing onto the
lowest-numbered experts at any bias, so it has one useful setting: total
collapse.

A constant offset cannot be written into either model's checkpoint: the
routers of DeepSeek-V2 and Mixtral have no bias term, and their input (an
RMS-normalised hidden state) has no constant component to carry one. So the
offset is applied inside the server, as a vLLM general plugin:

- the pipeline computes the full bias vector (it knows the expert count) and
  puts it, as JSON, in the server's ``MOE_ROUTER_BIAS`` environment variable;
- vLLM loads general plugins in every engine and worker process, and
  :func:`register` wraps vllm-ascend's two expert-selection kernels to add the
  vector to the router logits. Every MoE path reaches one of the two, including
  DeepSeek's internal-router path, which computes its logits inside the MoE
  layer and then selects;
- with the variable unset, :func:`register` does nothing, so normal runs are
  unaffected.

The bias also shifts the combine weights the top-k softmax produces, not only
the selection. That changes the model's outputs but not what the experiment
measures: expert load and the time it costs.

Placement assumption: without EPLB, vLLM places experts contiguously, so with
R ranks and E experts rank r holds experts [r*E/R, (r+1)*E/R). Validation
reports the realised per-rank shares, so a different placement would show.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence

ENV_VAR = "MOE_ROUTER_BIAS"

logger = logging.getLogger(__name__)

__all__ = ["ENV_VAR", "parse_target", "target_experts", "bias_vector", "server_env", "register"]


def parse_target(spec: str) -> tuple[str, list[int]]:
    """``"rank:0"`` -> ``("rank", [0])``; ``"experts:0,1"`` -> ``("experts", [0, 1])``."""
    kind, _, rest = spec.partition(":")
    kind = kind.strip()
    if kind not in ("rank", "experts") or not rest.strip():
        raise ValueError(f"bias target {spec!r}: expected 'rank:<r>' or 'experts:<i>,<j>,...'")
    try:
        ids = [int(x) for x in rest.split(",") if x.strip()]
    except ValueError:
        raise ValueError(f"bias target {spec!r}: ids must be integers") from None
    if not ids or any(i < 0 for i in ids):
        raise ValueError(f"bias target {spec!r}: ids must be non-negative")
    if kind == "rank" and len(ids) != 1:
        raise ValueError(f"bias target {spec!r}: name exactly one rank")
    return kind, ids


def target_experts(spec: str, n_experts: int, n_ranks: int) -> list[int]:
    """The experts a target spec biases, under contiguous expert placement."""
    kind, ids = parse_target(spec)
    if kind == "rank":
        if n_experts % n_ranks:
            raise ValueError(f"{n_experts} experts do not divide evenly over {n_ranks} ranks")
        (rank,) = ids
        if rank >= n_ranks:
            raise ValueError(f"bias target {spec!r}: rank {rank} out of range for {n_ranks} ranks")
        per_rank = n_experts // n_ranks
        return list(range(rank * per_rank, (rank + 1) * per_rank))
    if max(ids) >= n_experts:
        raise ValueError(f"bias target {spec!r}: expert {max(ids)} out of range for {n_experts} experts")
    return sorted(set(ids))


def bias_vector(spec: str, strength: float, n_experts: int, n_ranks: int) -> list[float]:
    """Per-expert logit offsets: ``strength`` on the targeted experts, 0 elsewhere."""
    targeted = set(target_experts(spec, n_experts, n_ranks))
    return [float(strength) if e in targeted else 0.0 for e in range(n_experts)]


def server_env(spec: str, strength: float, n_experts: int, n_ranks: int) -> dict[str, str]:
    """Environment for a server that should route with this bias; empty at strength 0."""
    if strength == 0:
        return {}
    return {ENV_VAR: json.dumps(bias_vector(spec, strength, n_experts, n_ranks))}


# --- inside the vLLM server -------------------------------------------------

def _wrap(fn, bias: Sequence[float]):
    cache: dict = {}

    def biased(*args, **kwargs):
        logits = kwargs.get("router_logits")
        if logits is None:
            raise TypeError(f"{fn.__name__} called without router_logits as a keyword; "
                            "the router-bias patch does not match this vllm-ascend version")
        if logits.shape[-1] != len(bias):
            raise ValueError(f"router-bias vector has {len(bias)} entries but the router scores "
                             f"{logits.shape[-1]} experts")
        key = (logits.device, logits.dtype)
        b = cache.get(key)
        if b is None:
            import torch

            b = cache[key] = torch.tensor(bias, dtype=logits.dtype, device=logits.device)
        kwargs["router_logits"] = logits + b
        return fn(*args, **kwargs)

    biased.__wrapped__ = fn
    return biased


def register() -> None:
    """vLLM general-plugin entry point: patch expert selection if a bias is set."""
    raw = os.environ.get(ENV_VAR)
    if not raw:
        return
    bias = [float(x) for x in json.loads(raw)]
    from vllm_ascend.ops.fused_moe import experts_selector as sel

    names = ("_select_experts_with_fusion_ops", "_native_select_experts")
    for name in names:
        fn = getattr(sel, name)
        if getattr(fn, "__wrapped__", None) is not None:
            continue  # plugins can load more than once per process
        setattr(sel, name, _wrap(fn, bias))
    targeted = [i for i, v in enumerate(bias) if v]
    logger.warning("router bias active (pid %d): +%s on experts %s of %d", os.getpid(),
                   sorted({v for v in bias if v}), targeted, len(bias))
