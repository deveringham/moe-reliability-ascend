# What to contribute to vllm-ascend

Read against vllm-ascend 0.23.0 as installed on the node, and checked against
`main` on 2026-10-14. Line references are to
`vllm_ascend/eplb/core/eplb_worker.py` and `vllm_ascend/eplb/eplb_updator.py`.

## The short version

Porting this project's detection screen into EPLB is the wrong contribution.
EPLB already measures what the screen measures, and measures it more cheaply.
What EPLB does not do is *ask* - it rearranges on an iteration counter, whatever
the measurement says. That is the gap, it is three small patches wide, and one of
them is a return value that the code already computes and throws away.

**Read the honest limits in "What the measurements support" before pitching the
gate.** As of 2026-10-14 the measured case for patch 1 is weaker than it looked:
the live gated run shows the patch deciding correctly on every rank, but with
the gate shut EPLB still cost +3.23% of the ~5% false positive, so the gate
averts the transfers and not most of the cost. And a threshold on measured
imbalance is only as good as the calibration behind it, which we have now
watched go stale on a change in output length alone. Patches 2 and 3 are unaffected - they are bug fixes, and neither
needs a latency result to justify it.

## Why not the screen

**EPLB's heat collection is cheaper than ours.** It does not count routing: it
reuses the `expert_tokens` group list the fused MoE op has already produced, one
device-side `add_` of a tiny tensor per layer per step
(`ops/fused_moe/fused_moe_0_23_0.py:534`). Our sampled `bincount` screen costs
+0.28% +-0.71 on Mixtral and exists because we wanted the signal *without* EPLB.
Inside EPLB it would be a regression.

**EPLB already computes our statistic, per layer.** `_compute_imbalance`
(`eplb_worker.py:303`) takes `stage_load.max() / stage_load.mean()` per MoE
layer, for the current placement and for the planned one, and logs both. That is
the per-layer busiest-rank form this project argues for over the pooled one - so
the "pooled statistics are blind to a rotating hot rank" finding, which looked
like a contribution, does not land here. Upstream is already right about it.

The screen's value is therefore for deployments that do *not* run EPLB, which is
a paper result, not a patch.

## Three patches

### 1. A gate on the measured imbalance (`eplb_min_imbalance`)

Neither implementation asks whether the imbalance costs anything. vllm-ascend
fires on `cur_iterations == expert_heat_collection_interval - 1`
(`eplb_updator.py:88`); vLLM's own core EPLB fires on
`expert_rearrangement_step >= expert_rearrangement_step_interval`
(`vllm/distributed/eplb/eplb_state.py`), and the word "imbalance" does not appear
in that module. What that costs has been measured twice and is small: on
DeepSeek-V2-Lite at 1.09x and 1.73x busiest-rank load - both free - EPLB applies
11-12 cycles per point and costs **+4.73% +-4.84 TPOT** (`eplb-fp2`, 4 repeats)
or **+2.33% +-2.55** (`gate-closure`, 5 repeats, same regime) rearranging toward
a balance that bought nothing. Neither reading excludes zero, so the claim the PR
can carry is that applying a placement is not free and nothing asks whether it is
worth it - not a figure for what it saves. The same applies on Mixtral 8x7B,
+1.85% +-1.88 at 1.76x.

The patch adds one condition: apply a plan only once the imbalance EPLB already
measured reaches a threshold. `patches/vllm-ascend-0.23.0-eplb-gate.patch`, 58
lines, no new measurement and nothing new on the serving path.

The threshold belongs to the deployment - 1.42x on Mixtral 8x7B, about 30
NPU-minutes to calibrate; DeepSeek-V2-Lite's 2.43x was withdrawn on 2026-10-14
(below), and at 300 tokens it has no threshold below 3.64x - so upstream gets a knob and
documentation, never a baked constant. That the two models differ by a full 1.0x
is itself the argument against a default, and the regime finding below is a
second: both of those numbers were calibrated on 100-token generations and are
too low for the same models serving 300-token ones. The documentation has to say
that the threshold describes a traffic mix, not a model.

Three details that make it correct rather than obvious, all verified in the
source:

- **It must keep the placement, not skip the cycle.** `block_update_q` is a
  blocking handshake: `forward_before` does `self.update_info_all.pop(0)`, so
  returning `None` raises `AttributeError`. (The existing `load_info is None`
  early return at `eplb_worker.py:64` is therefore already a latent crash,
  reachable only because `warm_up_eplb` always seeds `moe_load` first - worth a
  drive-by fix.) Setting `new_placement = old_placement.clone()` needs no new
  control flow: every layer then compares equal in
  `compose_expert_update_info_greedy`, which yields empty send/recv sets, and the
  loader walks `WAITING -> READY -> TRANSFERRING -> WAITING` transferring
  nothing. That path is exercised today by any layer whose map does not change.
- **The decision must be identical on every rank.** It is: `shared_dict["moe_load"]`
  is the all-gathered tensor and the arithmetic is deterministic numpy. This is
  the hazard to name in the PR, because vLLM core carries an explicit comment
  that a rank-divergent `rebalanced` flag hangs an `all_reduce`. Verified on
  hardware rather than only in the source: across every decision of the gated run
  all four ranks log the same reading to three decimals (`imbalance 1.200x on the
  worst layer, 1.107x on average`), and the run records `gate_consistent` by
  checking that the gate lines divide evenly by the rank count - a remainder
  would mean the ranks had disagreed.
- **Default off.** A load balancer that silently stops balancing is not a patch
  anyone merges.

### 2. Honour the `changed` flag the worker discards

Policies 1 (`DefaultEplb`) and 2 (`SwiftBalanceEplb`) both end with

```python
change = 0
if npu_heat_all_after < 0.95 * npu_heat_all_origin:
    change = 1
return change, per_layer_priority, new_deployment.tolist()
```

and `policy_abstract.py:17` documents the flag (`1 -- table_changed`,
`0 -- not_changed`). The caller:

```python
# eplb_worker.py:71, in EplbWorker.do_update
_, _, new_placement = self.calculate_rebalance_experts(load_info, old_placement)
```

Both `changed` and `per_layer_priority` are dropped, and the placement is applied
regardless. Confirmed on `main`, so it is not a fix we are behind on.

**Do not expect a latency result from this.** The flag asks whether the plan
improves *balance*, not whether the imbalance cost anything. Across the four
levels in `eplb-fp2` the plans improved EPLB's own metric by 9%, 39%, 57% and
69%, so `changed` reads 1 at every one of them - including the pure false
positive at 1.09x, where the layout is near-balanced and rearranging is pure
loss. A balance-improvement threshold cannot see a false positive. It is a real
bug worth its own small PR, and it is not the gate.

`per_layer_priority` is dropped with it. It orders the layers by how much the
plan improves them, which matters if the number of layers moved per cycle is ever
capped; today the greedy composer walks layers in index order.

### 3. Fix `expert_map_record_path`, and give it an observe-only mode

Setting a record path does not make EPLB record-only. It wedges it:

- `forward_before` always generates the D2D task and calls
  `asyn_expert_weight_transfer`, which sets the loader to `TRANSFERRING`.
- `forward_end` skips `update_expert_map_and_weight` when the record path is set
  (`eplb_updator.py:138`) - and that method is the only thing that waits the P2P
  requests and returns the state to `WAITING`.

So the loader stays in `TRANSFERRING` for the life of the server: the first
cycle's `isend`/`irecv` are never waited, and every later cycle hits
`"Current D2D weight update is on-going, cannot accept new update task"` and
bails. The file it writes is a plan that was never applied. Every EPLB run in
this project before 2026-10-12 had this set and measured no rearrangement at all.

The fix is to not start the transfer when recording. Combined with exposing what
`_compute_imbalance` already computes - `latest_expert_hotness` is written on
rank 0 and consumed only by an optional `ms_service_metric` adapter, so everyone
else scrapes the log, as we do - that gives a genuine **observe-only EPLB**: the
detection screen, upstream, at roughly zero cost, for deployments that want to
know whether they need EPLB before enabling it.

## What the measurements support

Three results, all from runs whose arms shared shuffled rounds, and all of them
constraining the pitch rather than supporting it.

**EPLB has no measured benefit on either model.** On DeepSeek-V2-Lite it drives
its own busiest-rank reading to ~1.0x from a 3.63x skew, and TPOT moves +1.07%
+-2.47. That skew first read as costing the unbalanced server +31%; it was an
artefact of how we injected it - strong router bias makes some of the model's
answers end early (205 of 300 tokens on average), and that costs latency on its
own. With every request forced to 300 tokens (`fixed-length`, 2026-10-14) the skew
costs -0.27% +-1.84 and EPLB there costs +3.83% +-1.02, all four pairs positive;
at 100 tokens likewise (-2.13% +-3.91 at 3.53x, `length-held-100`). So on DeepSeek there was never a cost for it to recover at
any level tested, and every EPLB reading on it is a false positive. On Mixtral 8x7B
(`eplb-mixtral`, 2026-10-13, 24 points) it recovers nothing at any of three
levels, and it cannot balance the model at all: 2.62x to 1.58x, because eight
experts over four ranks leaves two per rank and no placement is much better. So
the argument for the gate is not "the balancer wins less than it costs" but the
weaker "it does work whose benefit we cannot measure".

**What rearranging for nothing costs is resolved on DeepSeek, and most of it is
not the rearranging.** In TPOT, EPLB at a free 1.73x costs +4.73% (eplb-fp2) and
+5.52% (eplb-gated); pooled over eight pairs, +5.12%, interval +1.7% to +8.5%.
(`gate-closure`'s +1.52% was end-to-end latency with streaming off, a different
measure.) With the gate shut and nothing moved, EPLB still costs +3.23% at that
level. On Mixtral at 1.76x it is +1.85% +-1.88, unresolved. A PR may say
"applying a placement is not free and vllm-ascend never asks whether it is worth
it"; it should not imply that the gate recovers a measured saving.

**A threshold on measured imbalance goes stale on the traffic, not just the
model.** This is the one that matters for the knob. Mixtral's impact threshold
(1.42x) was calibrated at 100 generated tokens per request. Re-measured at 300 -
same model, same hardware, same batch size, same bias, same realised load to
three decimals - the cost of a 1.76x skew moves from +5.83% to -0.98% TPOT, and
2.62x from +14.45% to +2.95%. The skew's cost lives in prefill-heavy steps, and a
decode-dominated workload dilutes it about fivefold. A gate calibrated at 100
tokens therefore opens on 100% of windows at 1.76x in a 300-token deployment,
exactly where EPLB loses, and gating comes out 0.3-0.6pp worse than leaving EPLB
alone. DeepSeek's 2.434x threshold is withdrawn: the 100-token point it was
interpolated to cost latency through early-ending answers, not skew, and with
length held nothing up to 3.5-3.6x costs anything at 100 or 300 tokens.

For the PR this is a documentation requirement, not a blocker, and it is an
argument for a knob with no default rather than against the knob: a threshold
copied from another deployment's traffic mix is worse than no threshold, because
it opens the gate at the wrong skew while looking calibrated. It is also why the
patch logs the reading and the threshold on every decision.

## Feasibility

`eplb_updator.py` and `policy_factory.py` both carry
`Todo: remove once vllm issue #22246 / PR #24069 lands`: this code is slated for
replacement by vLLM core's EPLB. So send 2 and 3 to vllm-ascend now - small,
concrete, bug-shaped - and open an RFC for the gate against vLLM core, where
async rearrangement and the rank-consistency requirement raise the bar.

Policy 3 (`FlashLB`) already gates itself, on `average_to_peak_ratio` against
0.85/0.9 (`policy_flashlb.py:528`), and policy 2 skips a layer under imbalance
1.01. So the idea is not foreign to the codebase; it is missing from the two
default paths and from vLLM core.

## What the gated run found

`configs/examples/eplb_gated.toml` (`eplb-gated`, 2026-10-14): DeepSeek, three
levels, three arms (`eplb-off`, `eplb-on`, `eplb-gated` at 2.43x) in shared
shuffled rounds, 4 repeats, 36 of 36 points. Every point ran beside another
user's jobs on NPUs 4-5, equally in every arm. TPOT against `eplb-off`, ± the
standard deviation of four paired differences:

| level | busiest rank | EPLB on | EPLB gated | gate decisions |
|---|---|---|---|---|
| 0 | 1.09x | +1.86% +-9.39 | +0.00% +-4.55 | shut 48 of 48 |
| 0.5 | 1.73x | +5.52% +-3.80 | +3.23% +-0.42 | shut 48 of 48 |
| 2 | 3.63x | +4.63% +-4.29 | +1.54% +-0.63 | open 8 of 32 |

*(2026-10-14: "the costly one" no longer holds. With output length held, 3.63x
is free too, so the gate's opening there was a correct reading of the threshold
it was given and the wrong decision for the deployment. The mechanics below
stand; the threshold does not.)*

**The patch behaves as designed, on hardware.** Shut on every decision at both
free levels, open at the costly one, all four ranks logging the same reading to
three decimals on every decision. At level 2 it opened twice per point (worst
layer 3.67x, then 3.18x) and shut from then on with the average at ~1.73x. A
threshold gate stops at the threshold rather than at balance - EPLB-on reaches
1.08x, the gated arm 1.72x - and latency was no worse for it.

**`applied_cycles` stops measuring rearrangement once a gate is in play.**
Keeping the placement still walks the loader's state machine and logs a
completed cycle having transferred nothing: level 0.5 reports 11-12 applied
cycles with the gate shut 12 of 12 times. Count `gate_open`; `gate_consistent`
is the one that must hold.

**But it averts less than the pitch assumed.** EPLB-on's false positive at 1.73x
replicated in TPOT (+5.52%, 4 of 4 pairs positive; pooled with eplb-fp2,
+5.12%, interval +1.7% to +8.5%). The gated arm, which moved nothing at that
level, still cost **+3.23%** (+2.6% to +3.9%) - about 60% of it. So most of the
false positive is the cost of having EPLB enabled, not of applying plans, and no
condition in `do_update` can reach it. The reading is from one level (1.09x is
too noisy to confirm or contradict it) and which part of the machinery carries
it is not established.

**What this does to the PR.** The gate is still correct and still cheap to
review, and it removes the transfers; it is no longer evidence that a gated EPLB
is worth running where an ungated one is not. A PR can claim "does not apply a
plan the deployment has said is not worth applying, verified rank-consistent on
hardware", and nothing about latency saved. The latency claim this data supports
is about the feature, not the patch: on these two models EPLB has no measured
benefit and a standing cost of a few percent.

**A units caveat to fix before sending.** The patch compares
`_compute_imbalance`'s *worst-layer* value against a threshold calibrated on the
*layer-averaged* busiest-rank load. Worst-layer readings run well above the
average (2.19x worst against 1.76x average at level 0.5; 1.96-2.28x against
~1.73x after convergence at level 2), so a gate on the worst layer opens at a
lower average load than the calibration intends. Here the levels were far enough
from 2.43x for it not to matter. Upstream, either gate on the average or say
that the threshold is in worst-layer units and calibrate it that way.

Level 1 (2.49x) was left out: it sits within the calibration's error of the
threshold, and 2.49x is free in this run's 300-token traffic while the threshold
was calibrated at 100 tokens, so the gate would open there and be wrong to. It
remains a known blind spot.

## Installing the patch

```
./nrun 'uv run python scripts/apply_node_patch.py'          # install
./nrun 'uv run python scripts/apply_node_patch.py --check'  # verify
```

It patches the installed `vllm_ascend`, so **`uv sync` reverts it**; re-run it
afterwards. It is inert unless `MOE_EPLB_MIN_IMBALANCE` or
`MOE_EPLB_HONOR_CHANGED` is in the server's environment, which
`server.eplb_min_imbalance` and `server.eplb_honor_changed` set per arm - that is
what lets an EPLB-on and an EPLB-gated arm share one run's shuffled rounds
instead of being two runs with drift between them. A gated run refuses to start
against an unpatched install, because an unpatched server ignores the variable
and serves plain EPLB, which would read as "gating changes nothing".

A source patch rather than a vLLM plugin (how `expert_counter` installs itself)
because the planner runs in a subprocess launched by `EplbProcess`, where general
plugins are never loaded, and whether a monkeypatch in the parent survives
depends on the multiprocessing start method - which vLLM sets in several places,
and `api_server.py:87` asks for `forkserver`. It is also the artefact to send
upstream, so evaluating the change and proposing it are the same thing.
