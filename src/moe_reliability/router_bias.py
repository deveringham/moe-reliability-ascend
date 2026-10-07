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

The wrapper is not free: it runs a tensor add on every expert-selection call, on
every rank, including ranks whose entries are all zero. A level-0 point with the
variable unset therefore serves without that cost, and comparing it against a
biased point measures the skew *and* the instrument. ``bias_plugin_at_zero``
makes level 0 install the plugin with an all-zero vector, so every arm of a sweep
pays the same per-call cost and the comparison isolates the skew.

Layer targeting: by default every MoE layer is biased. Naming ``layers`` (model
layer indices, as in ``model.layers.<i>``) restricts it to those, which lets a
skew sit in a few layers the way natural imbalance does, rather than on one rank
throughout. Expert selection is not told which layer called it, so with layers
named the plugin also wraps ``AscendMoERunner.forward_impl``, which receives the
layer and encloses selection on both the plain and the shared-expert path
(DeepSeek's), to record the layer in progress from its ``layer_name``. Untargeted layers then skip the add entirely, and the zero-bias control
adds its zero vector on the same layers, so both arms still pay alike. Under graph
capture the Python runs once per layer at capture time, so each captured layer
keeps its own decision.

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
import re
from collections.abc import Sequence

ENV_VAR = "MOE_ROUTER_BIAS"

logger = logging.getLogger(__name__)

__all__ = ["ENV_VAR", "parse_target", "target_experts", "bias_vector", "server_env", "parse_env", "register"]


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


def server_env(spec: str, strength: float, n_experts: int, n_ranks: int,
               at_zero: bool = False, layers: Sequence[int] | None = None) -> dict[str, str]:
    """Environment for a server that should route with this bias.

    Empty at strength 0, which serves without the plugin and so without its
    per-call cost. With ``at_zero``, strength 0 instead installs an all-zero
    vector: routing is untouched but the wrapper runs, which is the control arm
    for everything the plugin itself costs. ``layers`` limits the bias (and the
    control's zero vector) to those model layers; empty or None means all.
    """
    if strength == 0 and not at_zero:
        return {}
    vector = bias_vector(spec, strength, n_experts, n_ranks)
    if not layers:
        return {ENV_VAR: json.dumps(vector)}
    return {ENV_VAR: json.dumps({"vector": vector, "layers": sorted({int(i) for i in layers})})}


def parse_env(raw: str) -> tuple[list[float], set[int] | None]:
    """The bias vector and targeted layers (None: every layer) from the variable's value."""
    value = json.loads(raw)
    if isinstance(value, list):
        return [float(x) for x in value], None
    return [float(x) for x in value["vector"]], {int(i) for i in value["layers"]}


# --- inside the vLLM server -------------------------------------------------

_LAYER = re.compile(r"layers\.(\d+)\.")
_current_layer: list[int | None] = [None]  # the MoE layer whose forward is in progress


def layer_index(layer_name: str) -> int | None:
    """``"model.layers.3.mlp.experts"`` -> 3."""
    m = _LAYER.search(layer_name or "")
    return int(m.group(1)) if m else None


def _track_layer(forward_impl):
    """Wrap ``AscendMoERunner.forward_impl(self, layer, ...)`` to record the layer in progress."""
    def tracked(self, layer, *args, **kwargs):
        previous = _current_layer[0]
        _current_layer[0] = layer_index(getattr(layer, "layer_name", ""))
        try:
            return forward_impl(self, layer, *args, **kwargs)
        finally:
            _current_layer[0] = previous

    tracked.__wrapped__ = forward_impl
    return tracked


def _wrap(fn, bias: Sequence[float], layers: set[int] | None = None):
    cache: dict = {}
    seen: set[int] = set()

    def biased(*args, **kwargs):
        if layers is not None:
            layer = _current_layer[0]
            if layer is None:
                # A layer-targeted bias that cannot tell layers apart would silently
                # bias all of them or none; either is a different experiment.
                raise RuntimeError("layer-targeted router bias: expert selection ran outside a tracked MoE "
                                   "layer; the forward_impl patch does not match this vllm-ascend version")
            if layer not in seen:
                seen.add(layer)
                logger.info("router bias (pid %d): layer %d %s", os.getpid(), layer,
                            "biased" if layer in layers else "untouched")
            if layer not in layers:
                return fn(*args, **kwargs)
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
    bias, layers = parse_env(raw)
    from vllm_ascend.ops.fused_moe import experts_selector as sel

    names = ("_select_experts_with_fusion_ops", "_native_select_experts")
    for name in names:
        fn = getattr(sel, name)
        if getattr(fn, "__wrapped__", None) is not None:
            continue  # plugins can load more than once per process
        setattr(sel, name, _wrap(fn, bias, layers))
    if layers is not None:
        from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner

        if getattr(AscendMoERunner.forward_impl, "__wrapped__", None) is None:
            AscendMoERunner.forward_impl = _track_layer(AscendMoERunner.forward_impl)
    targeted = [i for i, v in enumerate(bias) if v]
    where = "every layer" if layers is None else f"layers {sorted(layers)}"
    if not targeted:
        logger.warning("router bias active (pid %d): all-zero vector over %d experts on %s, the zero-bias "
                       "control - routing is unchanged and the wrapper's cost is paid", os.getpid(), len(bias),
                       where)
    else:
        logger.warning("router bias active (pid %d): +%s on experts %s of %d, %s", os.getpid(),
                       sorted({v for v in bias if v}), targeted, len(bias), where)
