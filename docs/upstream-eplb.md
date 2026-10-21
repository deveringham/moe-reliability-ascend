# What to contribute to vllm-ascend

Read against vllm-ascend 0.23.0 as installed on the node, and checked against
`main` on 2026-10-19, where `eplb_updator.py` and `eplb_worker.py` are
byte-identical to 0.23.0 - so both bugs below are live on `main` and a patch
written against the node's install applies unchanged. Line references are to
`vllm_ascend/eplb/core/eplb_worker.py` and `vllm_ascend/eplb/eplb_updator.py`.

**Patches 2 and 3 are written, tested and committed** (2026-10-19), on two
branches of a `main` clone at
`<scratchpad>/vllm-ascend`: `fix-eplb-honour-changed` and
`fix-eplb-record-mode`. Submitted 2026-10-20 as vllm-ascend #19288 (record mode) and #19290 (`changed` flag). Each carries unit tests that fail
against unpatched sources; see "The two bugfix branches" below.

## The short version

Porting this project's detection screen into EPLB is the wrong contribution.
EPLB already measures what the screen measures, and measures it more cheaply.
What EPLB does not do is *ask* - it rearranges on an iteration counter, whatever
the measurement says. That is the gap, it is three small patches wide, and one of
them is a return value that the code already computes and throws away.

**Read the honest limits in "What the measurements support" before pitching the
gate.** As of 2026-10-14 the measured case for patch 1 is weaker than it looked:
the live gated run shows the patch deciding correctly on every rank, but with
the gate shut EPLB still cost most of what it cost when rearranging. Pooled over
four runs (2026-10-20) the gate averts about a third of a ~3% standing cost -
the transfers - and not the rest. And a threshold on measured
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
improves *balance*, not whether the imbalance cost anything, so it cannot see a
false positive: at 1.09x, where the layout is near-balanced and rearranging is
pure loss, a plan that cuts the summed peak by 5% still reads `changed`. It is a
real bug worth its own small PR, and it is not the gate.

**The sharpest demonstration is a balanced load** (2026-10-19, from the unit
tests, against the real policies rather than a stub). On four ranks of two
experts with every expert equally loaded, `DefaultEplb` returns `changed = 0`
*and a different placement* (`[[0,1],[2,3],[4,5],[6,7]]` becomes a permutation),
so today vllm-ascend moves expert weights between NPUs to execute a plan the
policy itself scored as not worth 5%. `SwiftBalanceEplb` happens to return the
identical placement there, so it is harmless by luck rather than by design. Give
both policies a split worth making - two hot experts on one rank - and both
return `changed = 1` and still rearrange after the fix. Note also that with two
experts per rank *no* load makes either policy return 1 for a single hot expert:
the hot expert cannot be split, which is the Mixtral coarse-MoE bound showing up
in a unit test.

**The submitted branch, on hardware** (2026-10-19, `eplb-honour-pr`,
`configs/examples/eplb_honour_pr.toml`). The branch's `eplb_worker.py` replaced
the installed one outright - the gate patch removed for the duration, restored
after - so this is the text that would go upstream, not the prototype. It
honours the flag unconditionally, so a plain EPLB-on arm is the test. Read from
the server log, because the `eplb_activity` parser keys on the prototype's
strings:

| point | plans | declined | skip log lines | rank-consistent | busiest rank |
|---|---|---|---|---|---|
| 1 | 12 | 7 | 28 | yes (7 x 4) | 1.676 -> 1.016 |
| 2 | 12 | 5 | 20 | yes (5 x 4) | 1.676 -> 1.040 |

Every rank logs its own verdict, so a count divisible by the world size with no
remainder means all four declined the same plans - on every plan, in both
points. Control (unmodified EPLB, same settings): 1.676 -> 1.024. So the
submitted code decides consistently and still balances.

**The same, through the prototype** (2026-10-19, `eplb-honour-changed`,
`configs/examples/eplb_honour_changed.toml`). `server.eplb_honor_changed` is the
same decision in our node patch - same guard, same `old_placement.clone()` -
so this is evidence for the mechanism, not for the PR's exact diff (the patch
has no missing-load path). DeepSeek at 1.72x, 1200 requests at 300 tokens, two
arms x 2 repeats:

| arm | plans | declined | rank-consistent | busiest rank by EPLB's count |
|---|---|---|---|---|
| honour r1 | 12 | 5 | yes | 1.676 -> 1.044 |
| honour r2 | 12 | 9 | yes | 1.676 -> 1.035 |
| eplb-on r1 | 12 | - | - | 1.676 -> 1.024 |
| eplb-on r2 | 12 | - | - | 1.675 -> 1.024 |

Two things the unit tests could not show. **Every rank reached the same verdict
on every plan** (`gate_consistent` true in both points), which is the claim the
PR makes and the one whose failure would deadlock a collective. And **declining
the marginal plans does not stop EPLB balancing**: both honour points follow the
same trajectory as the control (1.68 -> 1.44 -> 1.09 -> ...) and end at
1.035-1.044x against 1.024x. The plans that matter are the early ones, and they
all clear the 5% bar. Count `gate_shut`/`gate_open`, not `applied_cycles`:
keeping the placement still walks the loader's state machine and logs a
completed cycle having moved nothing. No latency claim - the run has no
`eplb-off` baseline and was not designed as one.

**How often it would actually skip** (2026-10-19, replayed from
`eplb_activity.hotness`, where a plan is skipped when `update_max > 0.95 *
current_max`): 44-73% of plans on DeepSeek and 80-92% on Mixtral. The earlier
note here that the flag "reads 1 at every level" came from the first two or
three plans of each run, which are the ones with something to fix; once EPLB has
converged most plans fall under the 5% bar and would be skipped. So the fix does
have a cost side - at the measured 1.73x roughly half of the plans would be
skipped, worth at most ~1.2 of the ~2.3 points the gate averts - but it is still
a correctness fix and should be sent as one, without a latency claim.

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

**The fix is now measured** (2026-10-19, `eplb-record-fix`,
`configs/examples/eplb_record_fix.toml`, the branch applied to the node's
install). Two arms at 1.72x, 1200 requests at 300 tokens:

| arm | plans | applied | stalled | busiest rank by EPLB's count |
|---|---|---|---|---|
| record path set | 12 | 0 | **false** | 1.68x -> 1.81x, never falls |
| no record path | 12 | 12 | false | 1.68x -> 1.02x |

So the loader no longer wedges, nothing moves, and the worker keeps planning
from the installed placement - its `current_mean` sits at 1.68-1.83x for all
twelve plans while `updated_mean` stays at ~1.22-1.34x, i.e. it proposes the
same improvement every cycle because the layout never changes. The control arm
shows the patched package still applies normally (1.676 -> 1.439 -> 1.094 ->
... -> 1.020). The recorded file is a genuine plan: 26 layers, 4 devices x 16
experts, layer 0 non-contiguous.

`expert_map_record_path` is **MRv1-only**: it appears in `ascend_config.py`, the
MRv1 updator and adaptor, and `rfork_loader.py`, and nowhere in the V2 path.
`tests/ut/ops/test_eplb_v2_recording.py` is about recording expert *load* into
vLLM core's `expert_load_view`, not about recording a map. So the feature exists
only on the path being deprecated, and only there is it broken - worth asking
the maintainers whether it is meant to survive MRv2 before sending the fix.

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

**What rearranging for nothing costs is resolved on DeepSeek only in total, as a
ladder whose rungs are about a point each.** In TPOT, EPLB at a free 1.73x costs
+4.73% (eplb-fp2), +5.52% (eplb-gated), +1.04% (screen-vs-eplb) and +2.00%
(eplb-standing, 2026-10-20); pooled over nineteen pairs, **+3.06%, interval
+0.75% to +5.38%**, 14 of 19 positive. (Earlier poolings gave +5.12% [+1.7,
+8.5] over two runs and +3.56% [+1.0, +6.1] over three; quote +3.06%.)
(`gate-closure`'s +1.52% was end-to-end latency with streaming off, a different
measure.) With the gate shut and nothing moved, EPLB costs +3.23% (eplb-gated),
+3.18% (screen-vs-eplb) and +0.46% (eplb-standing) - pooled **+2.10%
[+0.01, +4.20]**, 11 of 15 positive. Enabled but wholly dormant it is
**+1.28% [-1.25, +3.81]**, 7 of 11, which does not separate from zero. So the
shape is off -> idle ~+1.3 -> gate shut ~+2.1 -> on ~+3.1: the transfers are
about a point, the gather/planner/empty-loader cycles about a point, and merely
enabling dynamic EPLB about a point. Individual rungs are beyond reach - the
per-pair sd is 3.8-4.8%, so one point needs ~70 paired rounds - so cite the
total and the shape, not a rung. eplb-gated's famous +3.23% +-0.42 was the
tightest run in the set and did not reproduce. On Mixtral at 1.76x the on-arm is
+1.85% +-1.88, unresolved. A PR may say
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

**There is a clock on 2 and 3.** RFC #18202 deprecates the Model Runner V1 EPLB
path before v0.31 and removes it before v0.32; MRv2 already routes through vLLM
core's controller. Both bugfixes are in MRv1 only, so they are worth sending
soon or not at all - and the gate belongs in core regardless. Note also that
core is not blameless on the gate: `vllm/distributed/eplb/eplb_state.py` skips a
rearrangement below 5% relative improvement, but only on ROCm (PR #47206,
2026-07-24), and it does so on an imbalance pooled across layers rather than per
layer.

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
replicated in TPOT (+5.52%, 4 of 4 pairs positive; pooled over four runs,
**+3.06%**, interval +0.75% to +5.38%). The gated arm, which moved nothing at
that level, still cost **+2.10%** [+0.01, +4.20] pooled over three - about two
thirds of it. So most of the false positive is the cost of having EPLB enabled,
not of applying plans, and no condition in `do_update` can reach it. Which part
of the machinery carries it is now partly established: an enabled-but-dormant
arm reads +1.28% [-1.25, +3.81] (`eplb-standing`), so roughly a point each for
being enabled, for the gather and planner, and for the transfers. The reading is
from one level (1.09x is too noisy to confirm or contradict it).

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

## What the screen-vs-eplb run found

`configs/examples/screen_vs_eplb.toml` (`screen-vs-eplb`, 2026-10-19): DeepSeek,
two free levels (1.05x and 1.72x), five arms in shared shuffled rounds, 5
repeats, 50 of 50 points after a resume. Two failed on the first pass -
`imbalance_0_eplb-off_r4` and `imbalance_0_eplb-gated_r4` - and the cause was
external: another user's job spilled onto NPU 0 and the server could not reserve
`gpu_memory_utilization = 0.85` (`Free memory on device (36.19/60.96 GiB)`).
`resume --retry-failed` recovered both, but hours later, so **round 4 at level 0
is not a like-for-like round** for the three arms that ran on the first pass
(only the re-run `eplb-off`/`eplb-gated` pair shares its conditions). The table
below is therefore the 9 rounds whose arms genuinely shared a node; adding the
tenth moves every arm by under 0.7 points, all downward, because the re-run
baseline was slow (167.60 ms against 160.5-163.0 for the other four).

TPOT against the `eplb-off` arm of the same round, pooled over both levels:

| arm | n | mean | sd | positive |
|---|---|---|---|---|
| screen | 9 | -0.47% | 4.89 | 4/9 |
| eplb-idle | 9 | +1.31% | 3.56 | 7/9 |
| eplb-gated | 9 | +2.76% | 3.74 | 8/9 |
| eplb-on | 9 | +0.40% | 3.25 | 4/9 |

**The run is too noisy to separate them.** Repeating one arm varies by 2.0-3.8%,
which is the size of the effect. The node was loaded 24-40 by the neighbour's
6-8 processes on NPUs 4-7 throughout; it never shared NPUs 0-3 with us, but
DeepSeek's eager decode is host-bound, so its CPU use lands on our step time
anyway. What the run supports is a bound on the screen's standing cost (near
zero, on DeepSeek and in TPOT at last, where before it rested on Mixtral and
end-to-end) and the two pooled figures above.

**E3 is not answered by this run.** `eplb-idle` - EPLB enabled with a
heat-collection interval longer than the point, so it never planned, gathered or
ran a loader cycle - sits at +1.31% with both EPLB-off and the shut gate inside
its interval. See `eplb-standing` below, which answers it as far as it can be
answered; the diagnosis that it needed a quieter node was wrong.

**The arms behaved as configured**, which is worth recording: `eplb-idle` never
planned in any of its 10 points, `eplb-gated` shut 12 of 12 times in all 9, and
`eplb-on` took 1.68x to 1.02x.

## E3, and why a quiet node was the wrong prescription (`eplb-standing`, 2026-10-20)

Four arms - off, idle, gate shut, on - at the single free 1.73x level, 6 repeats
in shuffled rounds, 24/24 points, output lengths matched to 0.2%. One level
rather than two, because both quoted figures live at 1.73x and splitting power
across 1.09x had resolved nothing at either.

| arm | n | mean | sd | 95% interval | positive |
|---|---|---|---|---|---|
| eplb-idle | 6 | +0.25% | 4.10 | [-4.06, +4.55] | 3/6 |
| eplb-gated | 6 | +0.46% | 3.75 | [-3.48, +4.39] | 3/6 |
| eplb-on | 6 | +2.00% | 6.27 | [-4.58, +8.58] | 4/6 |

Within the run, nothing separates from anything. Pooled with the earlier runs it
does: see the ladder above (on +3.06%, shut +2.10%, idle +1.28%). **The useful
conclusion is methodological.** The rungs are ~1 point apart and a single pair
carries an sd of 3.8-4.8%, so separating adjacent rungs needs on the order of 70
paired rounds - roughly 28 NPU-hours per arm. No amount of node hygiene changes
that; pooling runs does.

**The neighbour was never the explanation.** `screen-vs-eplb`'s non-separation
was attributed to the neighbour's 6-8 processes holding loadavg at 24-40 while
DeepSeek's eager decode is host-bound. This run saw loadavg 31-38 with 6-8
foreign processes - and *never* one on NPUs 0-3 in any of the 24 points - and
reproduced the same per-arm repeat spread (1.7-4.5%, min-max to 12.9%). The node
has **192 cores** and the neighbour's processes were using about 9 of them, so
loadavg in the twenties and thirties there is 11-20% utilisation, not
starvation. What contention does move is the absolute level: `eplb-off`'s own
TPOT rose about 6% in the two rounds where the neighbour grew to 8 processes,
which is the case for paired rounds, not for waiting.

**The arms behaved as configured here too:** `eplb-idle` planned 0 times in all
6 points, `eplb-gated` shut 12 of 12 in all 6 (with 11 loader cycles completing
and moving nothing - count `gate_shut`, never `applied_cycles`), and `eplb-on`
took 1.91x to 1.01x in every round. With the gate shut the imbalance drifted
*up*, 1.91x to 2.15x, which is what not rearranging looks like.

## What MRv2 does instead (2026-10-19)

Checked because of the obvious question: should the two bugfixes also cover
Model Runner V2? **Neither has an MRv2 analogue, and the reasons are worth
recording.**

MRv2 on Ascend is not "core's controller" as stated here earlier. It is core's
`EplbState` under an 886-line adapter (`vllm_ascend/patch/platform/patch_eplb.py`)
plus a 615-line `AscendEplbState` (`vllm_ascend/distributed/eplb/eplb_state.py`),
its own communicators, and its own policy tree
(`vllm_ascend/distributed/eplb/policy/`).

- **Patch 3 (record mode) has nothing to port.** `expert_map_record_path`
  appears nowhere in the V2 stack. An "MRv2 version" would be a new feature -
  add plan recording to V2 - not a bugfix, and it should not ride along with
  one.
- **Patch 2 (the `changed` flag) has nothing to port either.** The flag is
  returned by *vllm-ascend's own MRv1 policies*, which V2 does not use. V2's
  policies implement core's `AbstractEplbPolicy`, whose `rebalance_experts`
  returns a placement tensor and no flag, so there is nothing to discard. A
  policy that does not want to move a layer returns that layer's current
  placement - which is a no-op by the same mechanism patch 2 introduces for
  MRv1.
- **Patch 1 (the imbalance gate) is largely already there**, which is the find
  that matters. `patch_eplb.py` sets `policy_field.default = "stair"`, so the
  MRv2 default on Ascend is `StairEplbPolicy`, and it gates per layer on
  measured imbalance: `relative_balance_threshold = 0.95` against the balance
  at the last commit (hysteresis) and `absolute_balance_threshold = 0.90`,
  with a covariance-aware risk model behind the estimate.

So the RFC's motivation as drafted - "nothing asks whether the imbalance is
worth fixing" - is **false for Ascend MRv2** and has to be rewritten. What
survives is sharper: both stair's gate and core's ROCm skip are in units of
*balance*, and stair's absolute threshold fires at a mean busiest-rank ratio of
about 1.11x, far below anything we have measured a latency cost for. The
argument is not "add a threshold" but "put the threshold in units of measured
impact, calibrated per deployment" - and stair becomes the precedent to cite
rather than the gap to fill.

A second consequence: both bugfixes are repairs to a path that MRv2 does not
inherit and that #18202 removes before v0.32. That is an argument for asking
upstream first, not for skipping them - a broken `expert_map_record_path` is
still shipped today, and record mode has no V2 replacement at all.

## The two bugfix branches

Both are committed on a `main` clone at `~/src/vllm-ascend`, with DCO
sign-off, ruff-clean, and unit tests verified to fail against unpatched sources.
`fix-eplb-record-mode` is #19288 and `fix-eplb-honour-changed` is #19290; see `docs/pr_drafts.md` for submission state.

| branch | fixes | files | suite | hardware |
|---|---|---|---|---|
| `fix-eplb-record-mode` | patch 3 | `eplb_updator.py`, `eplb_worker.py`, `model_runner_v1.py`, 2 test files | 24 passed (19 baseline) | `eplb-record-fix` |
| `fix-eplb-honour-changed` | patch 2, and the latent `None`-load crash | `eplb_worker.py`, `tests/ut/eplb/core/test_eplb_worker.py` | 29 passed (19 baseline) | `eplb-honour-pr` (the branch itself) |

Both were tested on `main` at `ee29cc6fc`, are rebased onto `9a885476a` (no
change to the EPLB sources or tests between the two), and are mypy-clean on the changed files (1.11.1, the repo's `mypy.ini`, with
`--follow-imports=skip --ignore-missing-imports`, so this checks our code and
not vLLM's types).

`fix-eplb-record-mode` adds a `record_only` flag through `EplbProcess` to
`EplbWorker`, set from `expert_map_record_path`, so the worker keeps planning
from the installed placement; and returns from `forward_before` after popping
the update info, so no transfer starts.

`fix-eplb-honour-changed` reads the flag, keeps `old_placement.clone()` when it
is 0, and routes the missing-load case through the same path instead of
returning `None`. It guards the flag with `isinstance(changed, bool | int |
np.bool_ | np.integer)`, so a policy returning something unexpected keeps
today's behaviour.

Both branch from `main`, and both touch `do_update`, so whichever goes second
needs a trivial rebase - they each add `tests/ut/eplb/core/test_eplb_worker.py`,
which is a new file in both.

Tests were run on the node against a copy of the installed package with the
branch's EPLB files dropped in (`~/pr-test23`, `PYTHONPATH`-first), using
**`v0.23.0.post1`'s `tests/ut/eplb`** rather than `main`'s. That matters: with
`main`'s tests the directory shows 11 failures, all of them from `main`'s tests
referencing symbols the node's 0.23.0 package lacks
(`platform._fix_incompatible_config`, `QuantType.W8A8MXFP`) and identical with
and without our changes. Against the version-matched tests the baseline is
clean, which is the only way to read "no new failures" honestly. The EPLB
sources are byte-identical between `v0.23.0.post1` and `main`, so the branch
code under test is the code being proposed.

The node's installed package was patched for the two hardware runs and restored
afterwards from `~/prA_backup`; `apply_node_patch.py --check` confirms the gate
patch survived.

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
