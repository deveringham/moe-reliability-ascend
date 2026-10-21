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

Changing the skew mid-server: ``imbalance.bias_schedule`` lists engine steps at
which the strength changes, as fractions of the point's level, so one sweep gets
the same onset step at each of its skews. Two properties make it correct rather
than convenient. The trigger is a *step count*, not a clock: expert selection is
replicated, so two ranks holding different biases on the same forward would
select different experts and the dispatch would disagree - and every rank runs
every forward, so a step count is identical on all of them where a clock is not.
And the change is written into the existing bias tensor with ``copy_`` rather
than replacing it, because a captured graph holds a reference to the tensor it
was captured with; replacing it would leave the graph adding the old values. Step
boundaries are read off the layer index the same way ``expert_counter`` reads
them, and that plugin records the step per window, so an onset and an alarm are
in the same units. Without a schedule the strength is fixed for the server's
life, which is what every run before 2026-10-16 did - and why detection delay
could only be simulated by splicing two captures.

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

__all__ = ["ENV_VAR", "parse_target", "target_experts", "bias_vector", "server_env", "parse_env",
           "parse_schedule_spec", "normalise_schedule", "register"]


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
    #: ``(step, strength)`` pairs in step order: the strength in force from that
    #: engine step until the next pair. ``None`` holds ``strength`` for the
    #: server's life, which is what every run before 2026-10-16 did.
    schedule: tuple[tuple[int, float], ...] | None = None

    def applies_to(self, layer: int | None) -> bool:
        return self.layers is None or (layer is not None and layer in self.layers)

    def vector(self, layer: int | None, n_experts: int, strength: float | None = None) -> list[float]:
        """Per-expert logit offsets for one layer, at ``strength`` or the baseline."""
        target = self.target if self.target is not None else f"rank:{(layer or 0) % self.n_ranks}"
        return bias_vector(target, self.strength if strength is None else strength,
                           n_experts, self.n_ranks)

    @property
    def initial_strength(self) -> float:
        """The strength to install before any scheduled change."""
        return self.schedule[0][1] if self.schedule else self.strength

    @property
    def strengths(self) -> list[float]:
        """Every strength the server will serve at, for logging and for the manifest."""
        return [s for _, s in self.schedule] if self.schedule else [self.strength]

    @property
    def rotating(self) -> bool:
        return self.target is None


def server_env(spec: str, strength: float, n_experts: int, n_ranks: int,
               at_zero: bool = False, layers: Sequence[int] | None = None,
               schedule: Sequence[Sequence[float]] | None = None) -> dict[str, str]:
    """Environment for a server that should route with this bias.

    Empty at strength 0, which serves without the plugin and so without its
    per-call cost. With ``at_zero``, strength 0 instead installs an all-zero
    vector: routing is untouched but the wrapper runs, which is the control arm
    for everything the plugin itself costs. ``layers`` limits the bias (and the
    control's zero vector) to those model layers; empty or None means all.
    """
    if strength == 0 and not at_zero and not schedule:
        return {}
    payload: dict = {"strength": float(strength), "n_ranks": int(n_ranks)}
    if spec != ROTATE:
        payload["target"] = spec
    if layers:
        payload["layers"] = sorted({int(i) for i in layers})
    if schedule:
        payload["schedule"] = [[int(step), float(value)] for step, value in normalise_schedule(schedule)]
    return {ENV_VAR: json.dumps(payload)}


def parse_schedule_spec(specs: Sequence[str]) -> list[tuple[int, float]]:
    """``["0:0", "600:1"]`` -> ``[(0, 0.0), (600, 1.0)]``.

    The config grammar, kept as strings because the schema's list types are flat.
    The second field is a *fraction of the point's level*, so one schedule gives
    a whole sweep the same onset step at each of its skews.
    """
    pairs = []
    for spec in specs:
        step, sep, fraction = str(spec).partition(":")
        if not sep:
            raise ValueError(f"schedule entry {spec!r}: expected '<engine step>:<fraction of the level>'")
        try:
            pairs.append((int(step.strip()), float(fraction.strip())))
        except ValueError:
            raise ValueError(f"schedule entry {spec!r}: step must be an integer and fraction a number") from None
    return normalise_schedule(pairs)


def normalise_schedule(schedule: Sequence[Sequence[float]]) -> list[tuple[int, float]]:
    """Validate and sort ``(step, strength)`` pairs, which must start at step 0.

    A schedule that does not say what to serve from the first forward would leave
    the opening stretch at the baseline strength, which is a second, silent
    setting - the whole point of a schedule is that the onset instant is known.
    """
    pairs = []
    for entry in schedule:
        step, value = entry
        if int(step) < 0:
            raise ValueError(f"router bias schedule: steps must be non-negative, got {step}")
        pairs.append((int(step), float(value)))
    pairs.sort()
    if not pairs or pairs[0][0] != 0:
        raise ValueError("router bias schedule: the first pair must be at step 0, so that what the "
                         f"server serves before the first change is explicit; got {pairs}")
    steps = [s for s, _ in pairs]
    if len(set(steps)) != len(steps):
        raise ValueError(f"router bias schedule: two strengths at the same step, {pairs}")
    return pairs


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
    schedule = value.get("schedule")
    return Bias(strength=float(value["strength"]), n_ranks=int(value["n_ranks"]),
                target=value.get("target"),
                layers=frozenset(int(i) for i in layers) if layers is not None else None,
                schedule=tuple(normalise_schedule(schedule)) if schedule else None)


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


class _Schedule:
    """Advances the bias strength at known engine steps, in place.

    Steps rather than seconds, because expert selection is replicated: if two
    ranks hold different biases on the same forward they select different
    experts and the dispatch disagrees. Every rank runs every forward, so a step
    count is identical on all of them by construction, where a clock is not.

    The vector is never replaced, only overwritten with ``copy_``. A captured
    graph holds a reference to the tensor it was captured with, so mutating that
    tensor is what lets a schedule change the skew under graph capture as well
    as in eager mode - replacing it would leave the graph adding the old values.

    Step boundaries come from the layer index, as in ``expert_counter``: layers
    run in increasing order within a forward, so an index that does not advance
    is the next step.
    """

    def __init__(self, bias: Bias):
        self.bias = bias
        self.index = 0                       # the schedule entry in force
        self.step = 0                        # completed forwards
        self.last_layer = -1
        self.live: list = []                 # (tensor, layer, n_experts) to refresh

    @property
    def strength(self) -> float:
        return self.bias.schedule[self.index][1]

    def register(self, tensor, layer: int | None, n_experts: int) -> None:
        self.live.append((tensor, layer, n_experts))

    def saw_layer(self, layer: int | None) -> None:
        if layer is None:
            return
        if layer <= self.last_layer:
            self.step += 1
            self._advance()
        self.last_layer = layer

    def _advance(self) -> None:
        schedule = self.bias.schedule or ()
        moved = False
        while self.index + 1 < len(schedule) and self.step >= schedule[self.index + 1][0]:
            self.index += 1
            moved = True
        if not moved:
            return
        import torch

        for tensor, layer, n_experts in self.live:
            values = self.bias.vector(layer, n_experts, strength=self.strength)
            tensor.copy_(torch.tensor(values, dtype=tensor.dtype, device=tensor.device))
        # Every rank logs its own line: a run can then check they agree, the way
        # the EPLB gate's decisions are checked, rather than trusting that they do.
        logger.warning("router bias (pid %d): step %d -> strength %s", os.getpid(),
                       self.step, self.strength)


def _wrap(fn, bias: Bias, schedule: _Schedule | None = None):
    cache: dict = {}
    seen: set = set()

    def biased(*args, **kwargs):
        layer = _current_layer[0]
        if schedule is not None:
            schedule.saw_layer(layer)
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

            values = bias.vector(layer, n_experts,
                                 strength=schedule.strength if schedule is not None else None)
            if layer not in seen:
                seen.add(layer)
                logger.info("router bias (pid %d): layer %s biases experts %s of %d", os.getpid(), layer,
                            [i for i, v in enumerate(values) if v] or "none", n_experts)
            b = cache[key] = torch.tensor(values, dtype=logits.dtype, device=logits.device)
            if schedule is not None:
                # Registered, not re-created: the next scheduled strength is
                # written into this tensor so a captured graph keeps working.
                schedule.register(b, layer if bias.rotating else None, n_experts)
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
    # One driver for both selection paths, so a scheduled change moves every
    # vector the server has built, whichever kernel built it.
    schedule = _Schedule(bias) if bias.schedule else None
    from vllm_ascend.ops.fused_moe import experts_selector as sel

    names = ("_select_experts_with_fusion_ops", "_native_select_experts")
    for name in names:
        fn = getattr(sel, name)
        # Our own sentinel, not ``__wrapped__``: ``expert_counter`` sets that too,
        # and entry points load sorted by name, so the counter always loads first.
        # Skipping on ``__wrapped__`` therefore skipped the bias entirely whenever
        # the screen was on - gate-closure's count-on arm recorded 1.13x where its
        # validation reported 1.73x (2026-10-16).
        if getattr(fn, "_biasing", False):
            continue  # plugins can load more than once per process
        wrapped = _wrap(fn, bias, schedule)
        wrapped._biasing = True
        setattr(sel, name, wrapped)
    # A schedule needs the layer tracker too: step boundaries are read off the
    # layer index, so without it the strength would never advance.
    if bias.layers is not None or bias.rotating or schedule is not None:
        from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner

        impl = AscendMoERunner.forward_impl
        # Either tracker maintains _current_layer, so one is enough - but the test
        # is for a tracker specifically, not for "something wrapped this".
        if not getattr(impl, "_tracking_layer", False) and not getattr(impl, "_counting_steps", False):
            tracked = _track_layer(impl)
            tracked._tracking_layer = True
            AscendMoERunner.forward_impl = tracked
    where = "every layer" if bias.layers is None else f"layers {sorted(bias.layers)}"
    how = f"rotating over {bias.n_ranks} ranks" if bias.rotating else bias.target
    if bias.schedule:
        logger.warning("router bias active (pid %d): %s, on %s, scheduled %s - strength changes at the "
                       "listed engine step, counted from this process's first forward", os.getpid(),
                       how, where, [list(p) for p in bias.schedule])
    elif not bias.strength:
        logger.warning("router bias active (pid %d): strength 0 on %s, the zero-bias control - routing is "
                       "unchanged and the wrapper's cost is paid", os.getpid(), where)
    else:
        logger.warning("router bias active (pid %d): +%s, %s, on %s", os.getpid(), bias.strength, how, where)
