"""Let EPLB ask whether the imbalance it found is worth moving experts for.

vllm-ascend applies a planned placement on a fixed iteration counter, and vLLM's
own core EPLB rearranges every ``step_interval`` steps. Neither asks whether the
imbalance costs anything. On DeepSeek-V2-Lite at 1.09x and 1.73x busiest-rank
load - both free, neither costing any latency - EPLB applied 11-12 cycles per
point and cost up to +4.73% +-4.84 TPOT rearranging toward a balance that bought
nothing (``docs/findings.md``, run ``eplb-fp2``).

This module holds a patch against the installed ``vllm_ascend`` adding two
independent, default-off conditions to ``EplbWorker.do_update``.

``MOE_EPLB_MIN_IMBALANCE`` (the one our measurements support) applies a plan only
once the measured per-layer imbalance reaches a threshold. The statistic is the
one ``_compute_imbalance`` already computes and logs - busiest rank over the
mean, worst MoE layer - so nothing new is measured; EPLB's heat collection is
already near-free, reusing the ``expert_tokens`` group list the fused MoE op
produces. The threshold is the load at which skew starts costing latency, which
is a property of the deployment: 2.43x on DeepSeek-V2-Lite, 1.42x on Mixtral
8x7B, calibrated in about 30 NPU-minutes (``docs/detection-findings.md``). Per
layer, not pooled: a rotating hot rank reads 1.05-1.08x pooled while per-layer
load is 3.5x, and upstream's statistic is already the per-layer one.

``MOE_EPLB_HONOR_CHANGED`` honours the policy's own verdict instead. Policies 1
and 2 end with ``change = 1 if npu_heat_all_after < 0.95 * npu_heat_all_origin``
and the abstract policy documents the flag (``1 -- table_changed``), but
``do_update`` unpacks the result as ``_, _, new_placement`` and discards it. That
is worth fixing upstream on its own terms, but it is not the gate this project
argues for and it is not the one to expect a latency result from: the flag asks
whether the plan improves *balance*, not whether the imbalance cost anything. At
the four levels measured in ``eplb-fp2`` the plan improved EPLB's own metric by
9%, 39%, 57% and 69%, so ``changed`` is 1 at every one of them - including the
pure false positive at 1.09x. A balance-improvement threshold cannot see a
false positive, which is the whole point.

Deliberately not included: the two-consecutive-windows rule that
``docs/detection-findings.md`` recommends for the screen. It was calibrated on
8-request windows, where a single window's p99 is noisy; one EPLB cycle here is
126 forward iterations of a 512-way batch, orders of magnitude more traffic, so
carrying the rule over would be extrapolation rather than a measured choice.

Why a source patch and not a plugin, as ``expert_counter`` is: the planner runs
in a subprocess launched by ``EplbProcess``, where vLLM's general plugins are
never loaded, and whether a monkeypatch in the parent survives depends on the
multiprocessing start method (vLLM sets it in several places, and
``api_server`` asks for ``forkserver``). A source patch does not care. It is also
the artefact to send upstream, so evaluating the change and proposing it are the
same thing rather than two things that have to be argued to be equivalent.

Both switches are read from the environment per decision, so one run can compare
an EPLB-on arm against an EPLB-gated arm inside shared shuffled rounds. Without
that they would have to be separate runs, and any drift would alias onto the arm.
"""

from __future__ import annotations

#: Act only at or above this per-layer imbalance. Unset or 0 disables the gate.
ENV_MIN_IMBALANCE = "MOE_EPLB_MIN_IMBALANCE"

#: Honour the policy's own ``changed`` flag. Ignored when a threshold is set.
ENV_HONOR_CHANGED = "MOE_EPLB_HONOR_CHANGED"

#: Present in the patched source, and in the server log once per decision per
#: rank. Both get checked: the first says the patch is installed, the second
#: that it ran.
MARKER = "_eplb_gate_keeps_placement"

LOG_SHUT = "EPLB gate shut"
LOG_OPEN = "EPLB gate open"

#: The file the patch applies to, relative to the ``vllm_ascend`` package.
TARGET = "eplb/core/eplb_worker.py"

_HELPERS = '''def _eplb_gate_threshold() -> float:
    """Per-layer imbalance at which rearranging becomes worth its cost, or 0.

    Read per decision rather than cached, so the switch is a property of the
    server process's environment and not of when this module was imported.
    """
    try:
        return max(0.0, float(os.environ.get("%(min)s", "") or 0))
    except ValueError:
        logger.warning("[eplb/worker] %(min)s is not a number; the imbalance gate is off")
        return 0.0


def _eplb_gate_honours_policy() -> bool:
    """Whether to honour the policy's own ``changed`` verdict."""
    return os.environ.get("%(honor)s", "0").strip().lower() not in ("", "0", "false", "no")


''' % {"min": ENV_MIN_IMBALANCE, "honor": ENV_HONOR_CHANGED}

_METHOD = '''    def _eplb_gate_keeps_placement(self, changed, old_placement, load_info):
        """Whether to keep the current placement instead of applying the plan.

        Two independent conditions, both off by default. A threshold on the
        measured imbalance takes precedence over the policy's verdict, because it
        asks the question the verdict cannot: the verdict compares the plan
        against the current layout, so it is 1 whenever the layout is improvable,
        including when the imbalance costs nothing to begin with.

        The decision has to be identical on every rank - a divergent placement
        deadlocks the collectives that follow - and it is: ``load_info`` is the
        all-gathered load, and both the policy and this arithmetic are
        deterministic. Keeping the placement is a no-op the pipeline already
        handles, since every layer then compares equal in
        ``compose_expert_update_info_greedy``, which yields empty send/recv sets.
        """
        threshold = _eplb_gate_threshold()
        if threshold:
            hotness = self._calculate_hotness(
                old_placement, load_info.sum(0) if self.multi_stage else load_info
            )
            mean_val, max_val = self._compute_imbalance(old_placement, hotness)
            where = f"imbalance {max_val:.3f}x on the worst layer, {mean_val:.3f}x on average"
            if max_val < threshold:
                return True, f"{where}, under the {threshold:.3f}x threshold"
            return False, f"{where}, at or over the {threshold:.3f}x threshold"
        if not isinstance(changed, (bool, int, np.integer)):
            return False, f"policy returned a non-numeric changed flag ({changed!r})"
        if not changed:
            return True, "policy reports no useful change"
        return False, "policy reports a useful change"

'''

_CALL = '''        changed, _, new_placement = self.calculate_rebalance_experts(load_info, old_placement)
        if _eplb_gate_threshold() or _eplb_gate_honours_policy():
            keep, reason = self._eplb_gate_keeps_placement(changed, old_placement, load_info)
            if keep:
                logger.info("[eplb/worker] %(shut)s: %%s", reason)
                new_placement = old_placement.clone()
            else:
                logger.info("[eplb/worker] %(open)s: %%s", reason)
''' % {"shut": LOG_SHUT, "open": LOG_OPEN}

#: ``(anchor, replacement)`` pairs, applied in order. Each anchor must appear
#: exactly once in the unpatched source; a missing or repeated anchor is an error
#: rather than a silent no-op, because the point of the exercise is to know
#: whether the thing under test is installed.
PATCHES: tuple[tuple[str, str], ...] = (
    (
        "from multiprocessing import Process, Queue\nfrom typing import Any\n",
        "import os\nfrom multiprocessing import Process, Queue\nfrom typing import Any\n",
    ),
    (
        "class EplbWorker:\n",
        _HELPERS + "class EplbWorker:\n",
    ),
    (
        "    def do_update(self):\n",
        _METHOD + "    def do_update(self):\n",
    ),
    (
        "        old_placement = self.global2local(self.old_expert_maps, self.num_local_experts)\n"
        "        _, _, new_placement = self.calculate_rebalance_experts(load_info, old_placement)\n",
        "        old_placement = self.global2local(self.old_expert_maps, self.num_local_experts)\n" + _CALL,
    ),
)


class PatchError(RuntimeError):
    """The installed source is neither patched nor patchable."""


def is_patched(text: str) -> bool:
    return MARKER in text


def apply_to(text: str) -> str:
    """Return the patched source, or ``text`` unchanged if it is already patched.

    Raises ``PatchError`` when the source is unpatched but an anchor is missing or
    ambiguous, which is what a vllm-ascend version bump looks like.
    """
    if is_patched(text):
        return text
    for anchor, replacement in PATCHES:
        found = text.count(anchor)
        if found != 1:
            raise PatchError(
                f"anchor appears {found} times, expected once; vllm_ascend/{TARGET} is not the "
                f"version this patch was written against:\n{anchor!r}"
            )
        text = text.replace(anchor, replacement)
    return text


def server_env(min_imbalance: float = 0.0, honor_changed: bool = False) -> dict[str, str]:
    """The server environment that arms the patch, or ``{}`` when it should not.

    Emitted on every point, including the ungated ones, so that an arm is defined
    by the environment it runs with rather than by what a previous point left set.
    """
    env = {}
    if min_imbalance:
        env[ENV_MIN_IMBALANCE] = repr(float(min_imbalance))
    if honor_changed:
        env[ENV_HONOR_CHANGED] = "1"
    return env
