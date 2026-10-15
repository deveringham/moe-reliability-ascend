"""The detection screen's counting, as a deployment would actually run it.

``benchmark.capture_experts`` measures routing by asking vLLM to return every
token's expert ids in the HTTP response body (``--enable-return-routed-experts``):
about 41k ids per request on DeepSeek-V2-Lite, base64-encoded through JSON and
parsed by the client. That is the right instrument for building a calibration -
it keeps the whole trace - and the wrong one for costing the screen, because a
screen does not need the trace. It needs per-expert counts over a window.

So this plugin does what the screen does and nothing else:

- it wraps the same two expert-selection kernels as :mod:`router_bias`, but reads
  the *output* (``topk_ids``) rather than adjusting the input;
- it counts a random sample of layers, redrawn every window, because evenly
  spaced layers are blind to a skew confined to a few (``docs/detection-findings.md``);
- within those layers it counts a strided sample of tokens;
- it accumulates into a device-side ``[layer, expert]`` tensor, so the per-call
  path is one ``bincount`` and one add, with no device-to-host copy and no sync;
- once per window it copies that tensor out - ``n_experts`` ints per sampled
  layer - and appends it to a JSONL file.

Only one rank counts. Expert selection is replicated: every rank computes the
full ``topk_ids`` for every token, so one rank already holds the whole routing
decision and the other three would be doing the same arithmetic again. A real
screen would count in one place for the same reason.

Configuration arrives as JSON in ``MOE_EXPERT_COUNT``, set by the pipeline:

    {"layers": 4, "token_fraction": 0.1, "window_steps": 200, "out": "<path>"}

With the variable unset, :func:`register` does nothing.

What this does not reproduce: the screen's window is 8 *requests*, and a plugin
inside the engine sees steps, not requests. Under continuous batching the two do
not correspond, so ``window_steps`` sets the emit cadence rather than a window
the detector would score. That affects how often the copy-out happens - once per
window, against a per-call cost paid thousands of times - not the per-call path
this exists to measure.
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass

from . import router_bias

ENV_VAR = "MOE_EXPERT_COUNT"

logger = logging.getLogger(__name__)

__all__ = ["ENV_VAR", "CountConfig", "server_env", "parse_env", "register"]


@dataclass(frozen=True)
class CountConfig:
    """How much of the routing to count, and where to put it."""

    layers: int = 4
    token_fraction: float = 0.1
    window_steps: int = 200
    out: str | None = None

    def __post_init__(self) -> None:
        if self.layers < 1:
            raise ValueError(f"detection counting: layers must be at least 1, got {self.layers}")
        if not 0 < self.token_fraction <= 1:
            raise ValueError(f"detection counting: token_fraction must be in (0, 1], got {self.token_fraction}")
        if self.window_steps < 1:
            raise ValueError(f"detection counting: window_steps must be at least 1, got {self.window_steps}")

    @property
    def stride(self) -> int:
        """Token sampling as a stride. 0.1 counts every tenth token."""
        return max(1, int(round(1 / self.token_fraction)))


def server_env(cfg: CountConfig | None) -> dict[str, str]:
    """Environment for a server that should count routing. Empty when counting is off."""
    if cfg is None:
        return {}
    payload: dict = {"layers": cfg.layers, "token_fraction": cfg.token_fraction,
                     "window_steps": cfg.window_steps}
    if cfg.out:
        payload["out"] = cfg.out
    return {ENV_VAR: json.dumps(payload)}


def parse_env(raw: str) -> CountConfig:
    value = json.loads(raw)
    return CountConfig(layers=int(value.get("layers", 4)),
                       token_fraction=float(value.get("token_fraction", 0.1)),
                       window_steps=int(value.get("window_steps", 200)),
                       out=value.get("out"))


# --- inside the vLLM server -------------------------------------------------


class _Counter:
    """Per-layer expert counts for the current window, accumulated on device."""

    def __init__(self, cfg: CountConfig) -> None:
        self.cfg = cfg
        self.rng = random.Random(0)
        self.known: set[int] = set()
        self.sampled: frozenset[int] = frozenset()
        self.counts = None  # [max_layer + 1, n_experts], device-side
        self.step = 0
        self.last_layer = -1
        self.windows = 0
        self._active: bool | None = None

    def active(self) -> bool:
        """Whether this process counts. Resolved once, at the first call."""
        if self._active is None:
            self._active = _counting_rank()
            logger.warning("detection counting (pid %d): %s", os.getpid(),
                           "this is the counting rank" if self._active
                           else "not the counting rank, nothing counted here")
        return self._active

    def redraw(self) -> None:
        """Choose this window's layers. Redrawn per window, never sliced once."""
        if not self.known:
            return
        layers = sorted(self.known)
        k = min(self.cfg.layers, len(layers))
        self.sampled = frozenset(self.rng.sample(layers, k))

    def flush(self) -> None:
        """Copy the window's counts out and start the next one."""
        self.windows += 1
        if self.counts is not None and self.sampled:
            rows = sorted(self.sampled)
            counts = self.counts[rows].cpu().tolist()
            if self.cfg.out:
                record = {"window": self.windows, "step": self.step,
                          "layers": rows, "counts": counts}
                with open(self.cfg.out, "a") as f:
                    f.write(json.dumps(record) + "\n")
            self.counts.zero_()
        self.redraw()

    def on_step(self) -> None:
        self.step += 1
        if self.step % self.cfg.window_steps == 0:
            self.flush()

    def observe(self, layer: int, topk_ids, n_experts: int) -> None:
        """Count a sampled layer's expert ids. One bincount, one add, no sync."""
        import torch

        if self.counts is None or self.counts.shape[0] <= layer or self.counts.shape[1] != n_experts:
            width = max(layer + 1, (self.counts.shape[0] if self.counts is not None else 0))
            counts = torch.zeros((width, n_experts), dtype=torch.int32, device=topk_ids.device)
            if self.counts is not None and self.counts.shape[1] == n_experts:
                counts[: self.counts.shape[0]] = self.counts
            self.counts = counts
        ids = topk_ids[:: self.cfg.stride].reshape(-1)
        self.counts[layer] += torch.bincount(ids.to(torch.int64), minlength=n_experts).to(torch.int32)


def _counting_rank() -> bool:
    """True in the one process that should count.

    Asked at the first selection call, not at plugin load. vLLM loads general
    plugins in every worker before the process group exists and without setting
    RANK, so a check at load time cannot tell the ranks apart: it answered True
    in all four workers, which had them counting the same routing four times and
    appending to one file (2026-10-12 smoke run, 140 windows written four times
    over). By the first forward the group is up and the rank is knowable.
    """
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    for var in ("RANK", "LOCAL_RANK", "VLLM_DP_RANK"):
        value = os.environ.get(var)
        if value is not None:
            return int(value) == 0
    return True


def _track_steps(forward_impl, counter: _Counter):
    """Wrap ``AscendMoERunner.forward_impl`` to notice where one engine step ends.

    Layers run in increasing order within a forward, so an index that does not
    advance is the next step. This does not need the layer set to be complete,
    which it is not until the first pass has run.
    """

    def tracked(self, layer, *args, **kwargs):
        index = router_bias.layer_index(getattr(layer, "layer_name", ""))
        if index is not None and counter.active():
            if index <= counter.last_layer:
                counter.on_step()
            counter.last_layer = index
            counter.known.add(index)
            if not counter.sampled:
                counter.redraw()
        previous = router_bias._current_layer[0]
        router_bias._current_layer[0] = index
        try:
            return forward_impl(self, layer, *args, **kwargs)
        finally:
            router_bias._current_layer[0] = previous

    tracked.__wrapped__ = forward_impl
    return tracked


def _wrap(fn, counter: _Counter):
    def counting(*args, **kwargs):
        result = fn(*args, **kwargs)
        if not counter.active():
            return result
        layer = router_bias._current_layer[0]
        if layer is None or layer not in counter.sampled:
            return result
        logits = kwargs.get("router_logits")
        if logits is None:
            raise TypeError(f"{fn.__name__} called without router_logits as a keyword; "
                            "the expert-counting patch does not match this vllm-ascend version")
        counter.observe(layer, result[1], logits.shape[-1])
        return result

    counting.__wrapped__ = fn
    return counting


def register() -> None:
    """vLLM general-plugin entry point: count routing if counting is configured."""
    raw = os.environ.get(ENV_VAR)
    if not raw:
        return
    cfg = parse_env(raw)
    counter = _Counter(cfg)

    from vllm_ascend.ops.fused_moe import experts_selector as sel

    for name in ("_select_experts_with_fusion_ops", "_native_select_experts"):
        fn = getattr(sel, name)
        if getattr(fn, "_counting", False):
            continue  # plugins can load more than once per process
        wrapped = _wrap(fn, counter)
        wrapped._counting = True
        setattr(sel, name, wrapped)

    from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner

    if not getattr(AscendMoERunner.forward_impl, "_counting_steps", False):
        tracked = _track_steps(AscendMoERunner.forward_impl, counter)
        tracked._counting_steps = True
        AscendMoERunner.forward_impl = tracked

    logger.warning("detection counting installed (pid %d): %d random layers, every %dth token, "
                   "window %d steps -> %s; whether this process is the one that counts is decided "
                   "at the first forward", os.getpid(), cfg.layers, cfg.stride, cfg.window_steps,
                   cfg.out or "counts discarded")
