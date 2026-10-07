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
from dataclasses import dataclass

ENV_VAR = "MOE_ROUTER_BIAS"
#: bias_target value that biases a different rank in each layer.
ROTATE = "rotate"

logger = logging.getLogger(__name__)

__all__ = ["ENV_VAR", "parse_target", "target_experts", "bias_vector", "server_env", "parse_env", "register"]


def parse_target(spec: str) -> tuple[str, list[int]]:
    """``"rank:0"`` -> ``("rank", [0])``; ``"experts:0,1"`` -> ``("experts", [0, 1])``."""
    kind, _, rest = spec.partition(":")
    kind = kind.strip()
    if kind == ROTATE and not rest.strip():
        return ROTATE, []
    if kind not in ("rank", "experts") or not rest.strip():
        raise ValueError(f"bias target {spec!r}: expected 'rank:<r>', 'experts:<i>,<j>,...' or "
                         f"'{ROTATE}'")
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
    if kind == ROTATE:
        # Which experts a rotating bias touches depends on the layer, so there is
        # no single answer; the per-layer vectors come from Bias.vector.
        raise ValueError(f"bias target {spec!r} is per layer; ask Bias.vector(layer, n_experts)")
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


@dataclass(frozen=True)
class Bias:
    """What the plugin applies, decoded from the environment.

    ``rotate`` biases a different rank in each layer - layer i gets rank
    i mod n_ranks - so every layer is skewed by the same amount but no rank is
    hot throughout. That is the shape natural imbalance has (the busiest rank
    leads 34-35% of layers, against ~100% for a fixed rank target), and it is the
    case where per-rank totals cancel: summed over layers the ranks come out
    even, while each layer is as skewed as a fixed target would make it.
    """

    strength: float
    n_ranks: int
    target: str | None = None          # a fixed target spec, or None when rotating
    layers: frozenset[int] | None = None   # None: every layer

    def applies_to(self, layer: int | None) -> bool:
        return self.layers is None or (layer is not None and layer in self.layers)

    def vector(self, layer: int | None, n_experts: int) -> list[float]:
        """Per-expert logit offsets for one layer."""
        target = self.target if self.target is not None else f"rank:{(layer or 0) % self.n_ranks}"
        return bias_vector(target, self.strength, n_experts, self.n_ranks)

    @property
    def rotating(self) -> bool:
        return self.target is None


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
    payload: dict = {"strength": float(strength), "n_ranks": int(n_ranks)}
    if spec != ROTATE:
        payload["target"] = spec
    if layers:
        payload["layers"] = sorted({int(i) for i in layers})
    return {ENV_VAR: json.dumps(payload)}


def parse_env(raw: str) -> Bias:
    """Decode the variable's value. A bare list is the pre-2026-10-08 form."""
    value = json.loads(raw)
    if isinstance(value, list):
        # A literal per-expert vector, from before the policy moved into the payload.
        nonzero = {v for v in value if v}
        strength = nonzero.pop() if len(nonzero) == 1 else 0.0
        experts = [i for i, v in enumerate(value) if v]
        return Bias(strength=strength, n_ranks=1,
                    target=f"experts:{','.join(map(str, experts))}" if experts else "experts:0",
                    layers=None)
    layers = value.get("layers")
    return Bias(strength=float(value["strength"]), n_ranks=int(value["n_ranks"]),
                target=value.get("target"),
                layers=frozenset(int(i) for i in layers) if layers is not None else None)


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


def _wrap(fn, bias: Bias):
    cache: dict = {}
    seen: set = set()

    def biased(*args, **kwargs):
        layer = _current_layer[0]
        if bias.layers is not None or bias.rotating:
            if layer is None:
                # Guessing here would bias every layer or none, or every layer the
                # same way; each is a different experiment.
                raise RuntimeError("per-layer router bias: expert selection ran outside a tracked MoE "
                                   "layer; the forward_impl patch does not match this vllm-ascend version")
            if not bias.applies_to(layer):
                if layer not in seen:
                    seen.add(layer)
                    logger.info("router bias (pid %d): layer %d untouched", os.getpid(), layer)
                return fn(*args, **kwargs)
        logits = kwargs.get("router_logits")
        if logits is None:
            raise TypeError(f"{fn.__name__} called without router_logits as a keyword; "
                            "the router-bias patch does not match this vllm-ascend version")
        n_experts = logits.shape[-1]
        # n_experts belongs in the key: a cached vector is only valid for the router
        # width it was built for.
        key = (logits.device, logits.dtype, n_experts, layer if bias.rotating else None)
        b = cache.get(key)
        if b is None:
            import torch

            values = bias.vector(layer, n_experts)
            if layer not in seen:
                seen.add(layer)
                logger.info("router bias (pid %d): layer %s biases experts %s of %d", os.getpid(), layer,
                            [i for i, v in enumerate(values) if v] or "none", n_experts)
            b = cache[key] = torch.tensor(values, dtype=logits.dtype, device=logits.device)
        kwargs["router_logits"] = logits + b
        return fn(*args, **kwargs)

    biased.__wrapped__ = fn
    return biased


def register() -> None:
    """vLLM general-plugin entry point: patch expert selection if a bias is set."""
    raw = os.environ.get(ENV_VAR)
    if not raw:
        return
    bias = parse_env(raw)
    from vllm_ascend.ops.fused_moe import experts_selector as sel

    names = ("_select_experts_with_fusion_ops", "_native_select_experts")
    for name in names:
        fn = getattr(sel, name)
        if getattr(fn, "__wrapped__", None) is not None:
            continue  # plugins can load more than once per process
        setattr(sel, name, _wrap(fn, bias))
    if bias.layers is not None or bias.rotating:
        from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner

        if getattr(AscendMoERunner.forward_impl, "__wrapped__", None) is None:
            AscendMoERunner.forward_impl = _track_layer(AscendMoERunner.forward_impl)
    where = "every layer" if bias.layers is None else f"layers {sorted(bias.layers)}"
    how = f"rotating over {bias.n_ranks} ranks" if bias.rotating else bias.target
    if not bias.strength:
        logger.warning("router bias active (pid %d): strength 0 on %s, the zero-bias control - routing is "
                       "unchanged and the wrapper's cost is paid", os.getpid(), where)
    else:
        logger.warning("router bias active (pid %d): +%s, %s, on %s", os.getpid(), bias.strength, how, where)
