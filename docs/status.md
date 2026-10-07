# Where this project stands

Orientation document, 2026-10-02, updated 2026-10-07. One pass, no detail: the
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
first offset, +12% per +1x of busiest-rank load (t = 52). DeepSeek has a
threshold instead: flat to 2.34x, +22% at 3.45x. Graph mode shows the cost too,
though its balanced points are not yet controlled. **The mechanism is not
established on either model**: a saturated decode step is the same length at
every eager offset, which withdrew the GEMM account, and step wall still misses
TPOT by -45% to +19%. Natural traffic reaches 1.09-1.25x even for a single
prompt, below the lowest skew that costs anything (Mixtral, +3.3% at 1.38x), so
the null above is a statement about realistic skew, not a property of the stack.
`docs/figures/impact_map.png` puts both on one axis.

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

**EPLB rearranges correctly and buys nothing.** 84 cycles, a placement recorded
per point; by its own accounting it cuts rank imbalance from 1.139 to 1.007. The
measured straggler reads 1.026 off and 1.027 on. It balances token counts, which
overstate time imbalance by about six times, so there is nothing left to win.
Its one measurable effect is overhead: MoE share of compute falls 0.77pp while
absolute MoE time holds.

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
5. **EPLB at a realistic cadence.** Our intervals were cut from 600+50 to 50+10
   to make it fire inside a point at all, which leaves the weight transfer
   occupying about a third of every cycle. That overstates its overhead and may
   explain why realised imbalance (1.139) never approaches predicted (1.007).

## What not to trust

- **Prefill-only makespan and TTFT.** The engine's queue sits empty for 83-90% of
  a DeepSeek prefill-only run, so these measure the client and API server. That
  explains the noise once blamed on the neighbouring job.
- **Window-averaged trace figures.** A profiled window's step mix moves with the
  swept parameter, so averages over it compare different workloads. Split steps
  and match them on batch and prefill content (`scripts/step_profile.py`).
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
anything here, EPLB does not help, and we can now say precisely why in both
cases. That is a real contribution for anyone planning MoE serving on this
hardware, and the measurement pitfalls above are reusable.

What we do not yet have is a positive result. The detection work is honestly
characterised but not yet useful, and the most interesting thing in the data -
the arbitrary pace-setter - is unexplained. Of the two, the pace-setter is the
better bet: it is a large effect, it is cheap to investigate, and it is the
mechanism that would decide whether imbalance could ever matter at wider
parallelism.
