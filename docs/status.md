# Where this project stands

**For the whole story in one place, written without assuming the codebase, start
with [findings.md](findings.md).** This document is the working orientation note.

Orientation document, 2026-10-02, updated 2026-10-08. One pass, no detail: the
numbers and their derivations are in [imbalance-findings.md](imbalance-findings.md),
[regime-findings.md](regime-findings.md) and
[router-bias-findings.md](router-bias-findings.md).

## The question, and the answer so far

We set out to measure what expert load imbalance costs a MoE deployment on
Ascend NPUs. **The answer is that on this stack it costs nothing measurable, and
we now understand why.** That has held against every attempt to break it.

Two independent mechanisms make imbalance invisible, and they compose:

1. **Concentrating tokens makes the kernel cheaper.** Skewing load gives
   `GroupedMatmul` fewer, larger GEMMs: the average call drops from 112 us to
   66 us at the extreme. The busiest rank does more work per step but each unit
   of work costs less, and the two very nearly cancel. A ~20% imbalance in
   expert *token counts* becomes only a ~2.6% imbalance in fused-MoE *time*.
2. **That 2.6% is then buried.** About 95% of collective kernel time is ranks
   blocked waiting for each other, and the resulting asymmetry in per-rank busy
   time is ~33% - an order of magnitude larger than anything imbalance does, and
   unrelated to it. The step is gated by collective sync, and the sync is paced
   by an essentially arbitrary rank.

So imbalance is not merely hard to see here; there are two separate reasons it
cannot surface.

**Update 2026-10-06/07: it does surface once the skew is strong enough.** A
graded, injected router bias that keeps every expert live raises 100-token TPOT
on 4 NPUs, measured with the instrument in every arm. Mixtral is linear from the
first offset, +12% per +1x of busiest-rank load (t = 52), and reproduces in graph
mode. DeepSeek's cost is large and real by 3.2-3.5x (*confounded with shorter
answers - see the 2026-10-14 update below*) but its shape below that is
unresolved: two eager sweeps disagree at 2.34x (+1.0% against +6.5%), by more
than either one's repeat spread. **The mechanism is not established on either
model**: a saturated decode step is the same length at every eager offset, which
withdrew the GEMM account, and step wall still misses TPOT by -45% to +19%.
Natural traffic reaches 1.09-1.25x even for a single prompt, below the lowest
skew that costs anything (Mixtral, +3.3% at 1.38x), so the null above is a
statement about realistic skew, not a property of the stack.
`docs/figures/impact_map.png` puts both on one axis.

**Update 2026-10-08: whether the hot rank persists decides the cost, and the two
models differ.** A rotating skew - a different rank hot in each layer, which is
the shape natural traffic has - costs Mixtral exactly what a fixed one costs at
matched per-layer load, and costs DeepSeek nothing where a fixed one costs +22%.
So DeepSeek pays only for a persistently hot rank while Mixtral pays per layer.
The rotating case is also where the pooled load statistic fails completely: it
reads 1.05-1.08x, i.e. benign, for a skew that costs Mixtral +17%
(`docs/figures/rotating_skew.png`).

**Update 2026-10-14: DeepSeek's cost of skew did not survive holding output
length fixed.** Strong bias makes DeepSeek write shorter answers (205 of 300
tokens, 84 of 100), and every DeepSeek point that ever cost heavily is one of
those. `fixed-length` (24/24 points) forced exactly 300 tokens per request: at
3.64x the skew costs **-0.27% +-1.84** TPOT, where eplb-fp2 at natural length read
+30.86%. Withdrawn at 300 tokens: the +31%, and the EPLB null at 3.63x (nothing
to recover; EPLB there is a false positive, +3.83% +-1.02).

**Update 2026-10-15: the same at 100 tokens, and it is not the mean length.**
`length-held-100` (36/36, three arms in shared rounds): natural length reproduces
+20.58% at 3.53x; held at 100 it is -2.13% +-3.91. Held at 84 - offset 2's
average - is free too, while natural is +17.70% above it at the same skew and the
same mean length. What differs is the spread: 28% of requests stop early at
scattered points, and every request in the server slows. So DeepSeek has no
measured cost of skew at any length; the router-bias knob degraded the model and
measured that. Withdrawn: the +22%, the 2.43x threshold, DeepSeek's half of the
rotating contrast. Mixtral's lengths never moved and its results stand, now the
only demonstrated cost of skew. `length-replay` (18/18) then served offset 2's
per-request lengths at offset 0, with no skew: **+19.55% +-5.64** TPOT against
every request held at 85, the same mean (6 of 6 rounds; makespan +8%). The
scattered stops are the whole cost, and every uniform-length baseline in the
project is ~20% optimistic about traffic with a spread of answer lengths.
`length-replay-profiled` (4/4) then located the cost: 63% of it is the step mix.
With stops scattered a slot frees almost every step, so the same prompt tokens
are admitted in 2.4x as many chunks (1027 per prefill-carrying step against
2457), and a step costs nearly the same whether it carries 300 prompt tokens or
4000. It is scheduling, not imbalance - the straggler reading is 1.05x in both
arms.
Detail in [router-bias-findings.md](router-bias-findings.md#fixed-length-output-deepseeks-cost-was-its-early-stops-not-the-skew).

Scope of the claim: DeepSeek-V2-Lite-Chat (64 experts, top-6) at 2 to 8-way
expert parallelism, and Mixtral 8x7B (8 experts, top-2) at 8-way, over router
bias 0-100 and effective alpha 0.75-1.44. Single node, 8x Ascend 910B3.

## What is established

**The null, on means and on tails.** TPOT p99 and max track the mean, and their
correlation with alpha changes sign across runs. The apparent effects all came
from runs served in alpha order or under a neighbouring job.

**Three apparent effects dissolved under replication** during this work (level
25 at 136.2 ms; alpha 1.2 at 149.6 ms; alpha 1.6 at 146.03 ms on 8 NPUs). This
is the methodological lesson of the project: `benchmark.repeats` above 1 is the
only noise floor available, and single points are not evidence.

**EPLB: retracted and replaced 2026-10-12.** The "rearranges correctly and buys
nothing" result measured EPLB in record-only mode: `eplb_record_map` makes
vllm-ascend plan and never apply, so its 84 "cycles" were 84 plans and no expert
moved. Re-measured with placements applied (`eplb-fp2`, 7-12 completed cycles per
point, 4 repeats): the conclusion holds for sounder reasons. EPLB takes a 3.63x
skew that costs the unbalanced server +31% TPOT down to 1.09x by its own reading
and recovers none of the latency (+1.07%); rearranging where there is nothing to
win costs up to +4.73% +-4.84 at 1.73x. Caveat: at 3.63x convergence took ~85 s
of a 127 s batch, so the steady state is not what was measured.
**Corrected 2026-10-14:** with length held (`fixed-length`, 1800-request warm-up
so convergence falls before the measured window) the 3.64x skew costs nothing,
so there was nothing to recover, and EPLB there costs +3.83% +-1.02. Every level
EPLB has been measured at on DeepSeek is now one where it can only lose.

**The null holds on Mixtral, where there was something to win** (2026-10-13,
`eplb-mixtral`, 24/24 points). Mixtral was the clean test because its cost curve
has no threshold; EPLB recovers nothing at 1.11x, 1.76x or 2.62x, every interval
spanning zero, and it cannot balance the model either - 2.62x to 1.58x, because
8 experts over 4 ranks is 2 per rank. Coarse MoE bounds the best possible plan
before any implementation gets a turn.

**An impact threshold goes stale on output length alone** (2026-10-14), and this
is the finding that most constrains the gating argument. At matched load, model,
hardware, batch size and bias, generating 300 tokens per request instead of 100
cuts the cost of Mixtral's 2.62x skew from +14.45% to +2.95% TPOT and 1.76x to
nothing: the cost lives in prefill-heavy steps, and a decode-dominated workload
dilutes it ~5x. Both calibrations were made at 100 tokens, so both thresholds are
too low for the 300-token EPLB runs - on Mixtral the gate opens on 100% of
windows at 1.76x, exactly where rearranging costs most, and gating comes out
0.3-0.6pp worse than leaving EPLB on. "Calibrate per deployment" now includes the
traffic mix.

**The gating sum does not close on DeepSeek, and the control says why not**
(2026-10-13, `gate-closure`, 40/40 points, 4 arms in shared rounds). Full capture
reads +6.38% and +9.51% against baseline spreads of 3.41% and 2.39%, so the
positive control fires and the regime can see ~6%; the screen's +1.65%/+1.71% is
a real bound. EPLB read only +1.52% there, but in end-to-end latency with
streaming off; in TPOT it replicated (below).

**A live gate decides correctly and averts less than half of the false
positive** (2026-10-14, `eplb-gated`, 36/36 points, off/on/gated in shared
rounds). The patched vllm-ascend shut on all 96 decisions at 1.09x and 1.73x and
opened at 3.63x, identically on all four ranks. At 1.73x EPLB-on costs +5.52%
TPOT (pooled with eplb-fp2, +5.12%, +1.7% to +8.5% - the first reading to
exclude zero), and the gated arm, which moved nothing, still costs +3.23% (+2.6%
to +3.9%). Most of the false positive is the cost of having EPLB on, which no
gate in the planner reaches. One level only, and the source is unattributed.

**Correction, 2026-10-14: four runs were not quiet.** eplb-mixtral, gate-closure,
eplb-gated and Mixtral's screen-counting all ran beside another user's jobs on
NPUs 4-5; `screen_overhead.py` read a key points never carry and reported them
clean. The neighbour was present alike in every arm of each round, so the paired
comparisons stand. Only eplb-fp2 ran on a quiet node.

**The screen costs nothing to run, and capture is not its proxy.** Measured
2026-10-12/13 (`screen-counting`, 30 points): counting 4 random layers and a
tenth of their tokens on device costs +0.28% +-0.71 e2e on Mixtral against a
0.44% baseline spread, while `--enable-return-routed-experts` costs +6.38% on the
same rounds. The sampled screen is a different, cheaper mechanism - no
per-request payload, one counting rank - not 1.5% of capture's cost. DeepSeek is
unmeasurable in this prefill-bound regime (baseline spread 3.75%, se 4.11%).

**The idle rank is the bottleneck, not a victim** - in 30 of 30 profiled points.
Ranks that arrive early at a collective wait *inside* the kernel and so count as
busy, which inverts the obvious reading of an occupancy figure. One rank paces
each server instance, and it is a different rank in every run of an identical
configuration.

**Expert granularity is not the lever.** Mixtral puts one expert per rank
instead of eight and behaves the same, because alpha is relative to each model's
natural CV (0.45 against 0.14). Equal alpha is equal *relative* imbalance, not
equal rank load.

**Forced imbalance is not a clean instrument.** The checkpoint recipe collapses
64 experts onto 6, so it varies the number of live experts as well as the skew.
Synthetic workloads with `max_repeats` keep every expert live and are the right
tool.

## What is built

Everything below runs in the local simulated suite (116 tests) except where it
touches hardware, which goes through `./nrun`.

- **Synthetic workload construction** targeting a per-layer CV, with
  `length_in_requests` and `prompt_length_tolerance` so workload size is not
  confounded with the imbalance it is meant to isolate.
- **Ascend trace analysis**: paired-call straggler, step decomposition, and
  `collective_wait`, which splits a collective into transfer and blocked waiting.
- **Router drift detection** (`moe_reliability_results.drift`): Jensen-Shannon
  divergence over per-layer expert distributions, with marginal and stratified
  variants and ROC tooling.
- **EPLB evaluation pipeline**, including the config path that actually reaches
  vllm-ascend (vLLM's own `--enable-eplb` does not).
- **Provenance and contention capture**: every point records a host snapshot and
  warns when another process shares the node.

## Open, in the order I would take them

0. **Can the scattered-stop cost be scheduled away** (2026-10-15).
   `length-replay` reproduced the cost with no bias (+19.55% +-5.64 against
   uniform 85) and `length-replay-profiled` put 63% of it in the step mix: the
   same prompt tokens admitted in 2.4x as many chunks, each chunk costing a step
   nearly the same whether it carries 300 tokens or 4000. Next: does raising
   `max_num_batched_tokens`, which sets the chunk, shrink the gap, and does the
   remaining 14.9 ms of per-step rate survive at matched chunk count? Real
   traffic always has a spread of answer lengths, so this outlives the
   router-bias question.
1. **The pace-setter.** Partly explained 2026-10-06: in DeepSeek decode (eager)
   the pace-setting rank's device idles 130-150 ms of a ~175-230 ms step,
   spread over every kernel transition. It is host-launch-bound. Host CANN API
   totals match across ranks, so the time goes between launches (Python or CPU
   scheduling). Injected skew makes the hot rank the pace-setter. Graph mode
   removes the in-step gaps but not the skew cost, so this explains eager-mode
   variance rather than what imbalance costs. In graph mode the largest idle is
   between steps: 45 ms of a 102 ms balanced DeepSeek step is the host preparing
   the next one.
2. **Drift detection: the canary set.** The detector catches gross router failure
   (a collapse to 6 of 64 experts scores 37x above the benign ceiling, AUC 1.000)
   but is blind to subtle failure. The limiting noise is *workload*, not
   statistics: ordinary topical variation produces 7x more drift than sampling
   error, and a router must send 10-20% of token-expert assignments somewhere new
   before it clears that. Stratifying by prompt length does not help, because
   topic drives routing content and length does not. Replaying a fixed set of
   prompts compares routing on identical inputs and removes the confound by
   construction rather than statistically.
   Measured since ([drift-findings.md](drift-findings.md)): identical prompts
   do not route identically, as ~2% of prompt-token assignments flip, but the
   flips are symmetric, and pooled drift between replays is ~80x below
   different prompts. Decode routing diverges and must be excluded.
3. **Serving regime: answered for 4 NPUs** ([regime-findings.md](regime-findings.md),
   2026-10-05). Prefill is the one regime where imbalance costs kernel time (the hot
   rank's GEMMs run 1.6x slower at bias 100, above the GEMM ridge point), but GEMMs
   are 10-16% of a step and the step lengthens by only 1.5-5%. Decode stays a null
   because its GEMMs are memory-bound. Graph mode does not amplify it.
4. **Wider expert parallelism.** The one lever that has raised the straggler:
   1.04x at 2 ranks, 1.13x at 4, 1.31x at 8, extrapolating to 1.63x at 16. Tail
   inflation also grows with rank count, independent of alpha - p99 sits 3.5%
   above the mean at 4 ranks and 12.6% at 8.
5. **EPLB at a realistic cadence, and past convergence.** Our intervals were cut
   from 600+50 to 50+10 to make it fire inside a point at all. (Realised
   imbalance never approaching predicted was not the cadence: no plan was ever
   applied - see above.) `eplb-fp2` now applies placements, but at the costly
   level convergence eats two thirds of the batch, so EPLB's *benefit* has never
   been measured in steady state. A batch long enough to discard the first ~80 s
   is the missing measurement.
6. **The screen's cost under saturation.** Measured on Mixtral and not on
   DeepSeek, whose engine queue sits empty 83-90% of the time in this
   prefill-heavy regime, so host-side work hides in the idle time. A decode-heavy
   workload would settle it, and is the regime a deployment screens in.

## What not to trust

- **Any DeepSeek cost of skew measured at natural output length** (raised and
  tested 2026-10-14). Every costly DeepSeek point is one where the answers
  shortened (205 of 300, 84 of 100, 92 of 100 at offset 1.75), and the rotating
  arm that "cost nothing" kept its length. Tested at both lengths: held to 300,
  3.64x costs -0.27% +-1.84; held to 100, 3.53x costs -2.13% +-3.91. Holding to
  the shortened *mean* (84) is free as well, so do not explain the old cost as
  "more prompt per generated token". Mixtral's lengths never move and are
  unaffected. Check
  `output_tokens_mean` per level, and set `client.ignore_eos` on any sweep that
  compares levels.

- **Prefill-only makespan and TTFT.** The engine's queue sits empty for 83-90% of
  a DeepSeek prefill-only run, so these measure the client and API server. That
  explains the noise once blamed on the neighbouring job.
- **Window-averaged trace figures.** A profiled window's step mix moves with the
  swept parameter, so averages over it compare different workloads. Split steps
  and match them on batch and prefill content (`scripts/step_profile.py`).
- **Graph-mode per-level numbers below ~2x busiest-rank load.** Their repeat
  spread is largest at the *low* levels (Mixtral balanced sd 10.3 ms against 0.4
  at its most skewed), which fits frontend starvation rather than contention: a
  fast engine the client cannot feed measures the client. The slopes and the
  high-skew points are sound.
- **An instrument that is absent in the control arm.** The router-bias plugin is
  not installed at strength 0, so its own cost rides on every effect measured
  against a balanced point: 10.0 ms per token on eager DeepSeek, nothing on
  Mixtral. `imbalance.bias_plugin_at_zero` fixes it for future runs; affected
  sweeps cannot be refitted, only re-run.
- **Client-side ITL distributions under heavy concurrency.** The pooled mean is
  within 3% of TPOT, but individual gaps arrive in bursts, so percentiles measure
  the client. Within-request spikes remain unmeasured; the engine-side histogram
  `vllm:inter_token_latency_seconds` is the way to close it.

- **TTFT is a queueing number, not a latency number.** ~2386 ms at the standard
  4-NPU/300-request config, but all requests are submitted at once, so it
  measures prefill throughput for the whole batch. It scales nearly linearly with
  request count (50 -> 668 ms, 300 -> 2386 ms, 1000 -> 9237 ms). A single-request
  TTFT has never been measured. Also: 8 NPUs is *worse* than 4 here (3815 ms),
  which fits the sync-overhead picture.
- **Per-rank totals hide stragglers.** `trace_max_over_mean` reads 1.004x where
  the paired-call straggler reads 1.088x, because the busiest rank changes from
  layer to layer and the imbalance cancels when summed. Any figure drawn from the
  totals form will say the ranks are balanced.
- **Summed collective duration is not communication cost.** It is ~95% waiting.
  Reading the 71-78% "communication" in a step decomposition as communication
  overstates it roughly twentyfold.
- **Per-request p-values are meaningless.** One sweep point is one server
  instance, so n = 1 per point, not one per request.
- **Within-request latency spikes are unmeasured.** TPOT averages over a
  request's decode steps and we capture no inter-token latencies, so a single
  slow step is invisible. Closing this needs ITL capture in the client.
- **`trace_active_iterations` does not bound the profiler window.** Spans are
  ~34 s, essentially the whole serving period, whatever the option is set to.
- **A neighbouring job contaminates timings**, costing ~3% TPOT and ~31% TTFT
  even on devices the run does not use. It invalidated the latency half of the
  EPLB comparison, where contention differed between the two arms.

## Honest assessment

The strongest results are negative or mechanistic: imbalance does not cost
anything here, and we can say precisely why. (The matching EPLB result was
withdrawn on 2026-10-12: it never rearranged.) That is a real contribution for anyone planning MoE serving on this
hardware, and the measurement pitfalls above are reusable.

What we do not yet have is a positive result. The detection work is honestly
characterised but not yet useful, and the most interesting thing in the data -
the arbitrary pace-setter - is unexplained. Of the two, the pace-setter is the
better bet: it is a large effect, it is cheap to investigate, and it is the
mechanism that would decide whether imbalance could ever matter at wider
parallelism.
