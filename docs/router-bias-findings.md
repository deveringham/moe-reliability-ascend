# Graded Rank Skew: Router-Bias Sweeps on 4 NPUs

2026-10-05/06. DeepSeek-V2-Lite-Chat and Mixtral 8x7B, 4x Ascend 910B3
(NPUs 0-3), 4-way expert parallelism, batch 512, 4096-token budget. Eager mode
unless marked; the graph-mode replication is at the end.
Rank 0's experts get a logit offset before top-k selection
(`imbalance.method = "router_bias"`, `bias_target = "rank:0"`). That grades the
skew while every expert stays live and routing stays input-dependent, which the
checkpoint recipe could not do.

## Answer

**Strong rank skew costs measurable latency on this stack. That is the first
positive result in the project, and it holds on both models.** In the 100-token
regime, with the instrument present in every arm (the controlled sweep,
2026-10-06, below), TPOT rises with the busiest rank's load:

- **Mixtral:** +12.0% per +1x of busiest-rank load (+19.9 ms, t = 52), linear
  from the first offset: 165.5 ms balanced to 209.8 ms at 3.36x.
- **DeepSeek: rising somewhere between 2.3x and 3.4x, shape unresolved.** The
  controlled sweep reads flat to 2.34x (174.0, 173.9, 171.8, 175.8 ms; +1.4 ms
  per +1x, t = 0.7) and +22% at 3.45x. A second sweep filling that gap
  (2026-10-07, below) reads +6.5% already at 2.34x and is non-monotone above it.
  The two disagree by more than either one's spread, so "a threshold, not a
  slope" is withdrawn pending replication. The gradual rise first reported
  *below* 2.34x was the instrument, and that still stands.

Natural traffic reaches 1.09-1.25x on the same load measure, even for a single
prompt (`docs/figures/impact_map.png`). Mixtral's first cost beyond run-to-run
drift is +3.3% at 1.38x; DeepSeek's lies somewhere in 2.3-3.4x.

**Graph mode also shows the cost** (+13.5 ms per +1x on DeepSeek, +18.1 on
Mixtral, against superseded eager fits of +13.0 and +19.2). Graph mode is how a
production deployment would run, so the result does not depend on eager mode's
launch overhead. Its balanced points ran without the plugin, so the graph-mode
figures are not yet controlled, and the DeepSeek comparison was made against
the eager slope the controlled sweep withdrew.

**Mechanism: not established on either model.** Step-resolved analysis
(2026-10-06, below) withdrew the per-step accounts this document first recorded.
A saturated decode step is the same length at every eager offset on both models
(DeepSeek 166.0 → 170.0 ms, Mixtral 153.2 → 153.1 ms, 150-220 steps per cell),
so the hot rank's GEMMs do not lengthen the step that most decode tokens wait
in. The eager cost sits in below-full-batch decode steps and in
prefill-carrying steps; in graph mode it sits in full-batch decode steps
instead. Token-weighted step wall still misses TPOT by -45% to +19%.

**The instrument inflated the first eager DeepSeek figures.** Their balanced
point ran without the router-bias plugin installed, and the plugin costs
10.0 ms per token on DeepSeek and nothing on Mixtral. The controlled sweep
re-measured both models with `imbalance.bias_plugin_at_zero`; the Latency
section's first table is the superseded form, kept for the record. The graph-mode
sweeps were not re-run, so their balanced points still lack the plugin.

**The 2026-10-02 null still holds where it was measured.** Natural traffic gives
a busiest rank of 1.09-1.25x (below). The lowest offsets (1.24-1.30x) cost
0-1%, within run-to-run drift. Only an injected skew has produced a cost.

**Prefill-only serving measured the frontend, not the model.** In the DeepSeek
prefill-only runs the engine's queue was empty for 83-90% of the time: requests
reached it at ~6k tokens/s against the ~22k it serves. Its makespan and TTFT
are not evidence either way, and the earlier "external slowdown episodes" are
probably the same thing.

## Calibration

Busiest-rank load (max over mean) and live experts, routed-expert capture on
200 MMLU prompts per level:

| Offset | 0 | 0.25 | 0.5 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|---|---|
| DeepSeek busiest rank | 1.02x | 1.30x | 1.66x | 2.34x | 3.45x | 3.96x | 4.00x |
| DeepSeek live experts | 64 | 64 | 64 | 64 | 64 | **16** | **16** |
| Mixtral busiest rank | 1.01x | 1.18x | 1.37x | 1.76x | 2.62x | 3.36x | 3.74x |
| Mixtral live experts | 8 | 8 | 8 | 8 | 8 | 8 | **3** |

On DeepSeek, offset 3 collapses routing onto rank 0's 16 experts. It varies the
active expert count as well as the skew, so it is excluded from the fits. Its
TPOT drops back to 183 ms, which is the memory-bound decode GEMM getting cheaper
with fewer live experts, as in the 2026-10-02 forced-imbalance runs.
`scripts/dose_response.py` excludes any level whose calibration shows experts
dropping out.

## Latency

**Superseded for eager mode by the controlled sweep below**: these runs'
balanced points lacked the plugin. Mixtral's figures agree with the controlled
ones within drift; DeepSeek's do not.

6 offsets x 3 repeats per run, served in shuffled rounds. Effect per +1x of
busiest-rank load from a fit with execution order as a covariate (order was
insignificant in every 100-token fit).

| Metric | Mixtral, 100 tokens | DeepSeek, 100 tokens | Mixtral, prefill only |
|---|---|---|---|
| TPOT mean | **+11.8%** (t = 36) | **+7.2%** (t = 7.4) | |
| TPOT p99 | +10.6% (t = 19) | +10.5% (t = 6.2) | |
| Makespan | +6.0% (t = 7.9) | -1.7% (t = -1.5) | +14% (t = 2.8)* |
| Sustained throughput | -8.6% (t = -2.6) | -14% (t = -4.5) | -16% (t = -5.1)* |

\* Partly frontend-limited: the queue was empty for 38% of the run (below).

Per level, TPOT mean in ms (± sd over 3 repeats):

| Offset | 0 | 0.25 | 0.5 | 1 | 2 | 3 |
|---|---|---|---|---|---|---|
| Mixtral | 163.5±0.6 | 166.5±2.0 | 168.4±1.5 | 175.8±2.7 | 192.1±1.8 | 208.7±1.1 |
| DeepSeek | 179.1±2.3 | 185.7±4.3 | 187.4±5.1 | 192.0±5.5 | 213.1±9.3 | (183.2, collapsed) |

- **Mixtral is the cleanest dose response in the project:** monotone, repeat
  spread under 2%, and the same slope on mean and p99.
- **DeepSeek's makespan is flat while TPOT rises.** Prefill throughput falls by
  a similar fraction, so a batch with all requests submitted at once finishes
  at about the same time. TPOT is the measure that sees the skew.

## Kernel level: where the time goes

Profiled passes at offsets 0 / 1 / 2, 100-token regime. Per step, from GEMM
counts (2 GroupedMatmul calls per MoE layer: 64 per Mixtral step, 52 per
DeepSeek step), change from offset 0 in brackets. "GEMM excess" is the
paired-call straggler: the sum of per-call maxima minus the mean rank's total.

| ms per step | Mixtral, offset 1 | Mixtral, offset 2 | DeepSeek, offset 1 | DeepSeek, offset 2 |
|---|---|---|---|---|
| Step span | 187.6 (+15.6) | 203.2 (+31.3) | 204.4 (+28.8) | 233.2 (+57.5) |
| GEMM excess | 20.4 (+14.8) | 44.1 (+38.5) | 3.5 (+3.1) | 7.2 (+6.8) |
| Collective transfer | 20.1 (+1.7) | 20.7 (+2.3) | 9.8 (+0.8) | 9.8 (+0.8) |
| Collective time, rank mean | 64.9 (+12.6) | 76.3 (+24.0) | 96.4 (+19.8) | 124.3 (+47.6) |
| Sweep TPOT change | +12.3 | +28.6 | +12.9 | +34.0 |

Rank 0's GEMM per call rises from 686 to 1,439 us on Mixtral while the other
ranks' falls from ~680 to ~520 us. On DeepSeek it rises from 151 to 279 us while
the others' falls to 96 us. Transfer stays flat throughout: all the added
collective time is ranks waiting.

### Mixtral: the GEMM straggler passes straight through

**Superseded** by the step-resolved section: at matched full batch, a Mixtral
decode step does not lengthen at all. The per-call figures below stand; the
inference from them to the step does not.

The GEMM excess (+14.8, +38.5 ms) accounts for the step growth (+15.6, +31.3)
and for the sweep's TPOT growth (+12.3, +28.6). At offset 2 the hot rank has
the *least* idle time between kernels (24.5 ms per step against 40-47 ms on the
other ranks, `scripts/kernel_gaps.py`). It is device-bound, and every other
rank waits for it inside the all-reduce. This is the mechanism the 2026-10-05
regime work predicted once GEMMs dominate the step: on Mixtral they do, and it
only needed a skew that natural traffic does not supply.

### DeepSeek, eager: rank 0 idles between launches

**Partly superseded**: the idle time is real and concentrated in below-full-batch
decode steps, but part of it is the instrument's host cost, and graph mode shows
a skew cost without it.

On DeepSeek the GEMM straggler is a tenth of the step growth. Idle time between
kernels, by rank (ms per step, span 176 at offset 0 and 233 at offset 2):

| Offset | Rank 0 | Rank 1 | Rank 2 | Rank 3 |
|---|---|---|---|---|
| 0 | 34.9 | 41.7 | 64.0 | **128.7** |
| 2 | **148.0** | 45.1 | 64.2 | 46.2 |

- **One rank's device sits idle for most of every step,** and that rank paces
  the collectives: rank 3 at offset 0, rank 0 at offsets 1 and 2. Its idle time
  is spread over every kernel transition (attention, dense matmuls, MoE
  dispatch and combine), not concentrated in the MoE path. Its device is
  waiting for its host to launch the next kernel.
- **This looks like the 2026-10-02 pace-setter,** the arbitrary low-occupancy
  rank: in eager mode one host is slowest and sets the step. Skew makes rank 0
  that rank, and slower than the arbitrary pace-setter was (148 against 129 ms
  idle). The graph-mode traces do not support this as the *cause* of the skew
  cost: with the in-step gaps gone, the cost in TPOT stays the same.
- **It is not more launch work.** The host-side CANN API totals
  (`api_statistic.csv`) for rank 0 and rank 3 are the same at both offsets: the
  same launch count, the same launch time, no extra synchronisation calls. The
  extra time is spent between launches, in Python or in CPU scheduling. Which of
  the two is open.

A ~19 ms step growth would follow from rank 0's idle exceeding the old
pace-setter's, plus 6.8 ms from the GEMM straggler. The measured +57.5 ms is
larger. The profiled window also holds fewer steps at offset 2 (379 against
440), so part of the gap may be a different mix of prefill and decode steps.

### A step-boundary gap

Every rank of both models has a gap between `Cast` and `Fill` once per step,
when no NPU runs anything while the host prepares the next step. In eager mode
it is ~16-18 ms (about 10% of a step) at every offset. In graph mode it is
**45.5 ms of a 102 ms DeepSeek step** at offset 0, shrinking to 14 ms at offset
2 as the slower device step overlaps more of the host's preparation. Once
graph mode removes launch overhead, host-side scheduling is the largest single
cost in a balanced step, larger than anything imbalance does.

## Graph mode

`rbias-graph` (offsets 0 / 0.5 / 1 / 2 x 3 repeats, 100 tokens, otherwise as
the eager runs) and `rbias-graph-profiled` (offsets 0 / 2). No neighbour on the
DeepSeek run; 3 of 12 Mixtral points had one.

| TPOT mean, ms | Offset 0 | 0.5 | 1 | 2 | Per +1x load |
|---|---|---|---|---|---|
| DeepSeek, eager | 179.1 | 187.4 | 192.0 | 213.1 | +13.0 (t = 7.4) |
| DeepSeek, graph | 139.5±1.8 | 144.9±6.5 | 143.1±2.6 | 174.1±5.6 | +13.5 (t = 5.0) |
| Mixtral, eager | 163.5 | 168.4 | 175.8 | 192.1 | +19.2 (t = 36) |
| Mixtral, graph | 156.0±5.4 | 156.7±7.8 | 151.0±0.6 | 184.5±5.0 | +18.1 (t = 3.8) |

- **The cost survives graph mode in absolute terms on both models,** so it is
  not eager-mode launch overhead. As a share it grows on DeepSeek (+10.1% per
  +1x against +7.2%) because the baseline is 40 ms faster.
- **In graph mode the cost appears as a step at offset 2,** not a steady rise:
  offsets 0.5 and 1 are within noise of balanced on both models. In eager mode
  Mixtral rose at every level. With three repeats this shape is suggestive only.
- **The frontend starts to bind.** Graph mode serves fast enough that the
  DeepSeek run's engine queue was empty for 28% of stats intervals (eager 100-
  token: 1%). Makespan and TTFT are affected. TPOT probably less so, but a
  starved engine runs smaller decode batches, so this is a caveat on the graph
  TPOT figures too.

Traces, per call (100 tokens, offset 0 → 2):

| | DeepSeek graph | Mixtral graph |
|---|---|---|
| Rank 0 GEMM per call | 144 → 213 us | 615 → 1,402 us |
| Other ranks' GEMM per call | 144 → 73 us | ~610 → ~555 us |
| Rank 0 compute (busy - wait) per call, offset 2 | 724 us (others ~562) | |
| Pace-setter at offset 2 | rank 0 (60%) | rank 0 (49%) |
| Device step growth | +9.6 ms | +65 ms |
| TPOT growth, unprofiled twin run | +20 ms | +44 ms |

- **DeepSeek graph mode looks like Mixtral eager:** rank 0's in-step idle is no
  larger than the other ranks', and its extra compute (~8 ms per step) matches
  the device step growth. The hot rank is late because of its GEMMs.
- **Device step growth and TPOT growth disagree,** in both directions (DeepSeek
  graph +9.6 against +20; Mixtral graph +65 against +44; DeepSeek eager +57.5
  against +34). Each comes from one profiled point whose window holds a
  different mix of prefill and decode steps (Mixtral graph: 671 steps at offset
  0 against 430 at offset 2). Decomposing TPOT needs step-resolved analysis
  (classify steps by token count, compare decode-only steps across offsets),
  not window averages.

## Step-resolved analysis (2026-10-06)

`scripts/step_profile.py` splits each profiled trace into engine steps and
writes one row per step and rank. A step's `_compute_slot_mapping_kernel`
launch carries its batch size and token count, so steps can be matched across
points instead of averaged over a window. `scripts/host_ops.py` reports
host-side torch and CANN time per step for one rank over a step range.

This resolves why window averages disagreed with TPOT, and it overturns the
per-step account given above.

### The step mix differs between points

There are almost no pure decode steps: chunked prefill puts prompt tokens into
most steps, and the proportion moves with the offset. On DeepSeek in graph mode,
prefill-carrying steps hold 45% of all token-time at offset 0 and 87% at offset
2. Averaging a profiled window therefore compares different workloads, which is
the whole of the earlier 1.5-3x disagreement.

### Decode steps at full batch do not slow down at all

Median wall per step, decode-only steps, by batch size (step count in
brackets):

| | Batch 1-200 | 200-350 | 350-480 | 480-513 |
|---|---|---|---|---|
| DeepSeek eager, offset 0 | 152.1 (38) | 157.7 (37) | 163.1 (57) | 166.0 (195) |
| DeepSeek eager, offset 1 | 225.2 (30) | 237.8 (31) | 244.3 (51) | **170.0 (157)** |
| DeepSeek eager, offset 2 | 236.3 (50) | 240.2 (28) | 253.7 (35) | 169.2 (15) |
| Mixtral eager, offset 0 | 135.1 (42) | 141.9 (54) | 150.5 (48) | 153.2 (187) |
| Mixtral eager, offset 2 | 141.4 (9) | - | 150.6 (82) | **153.1 (220)** |
| DeepSeek graph, offset 0 | 33.4 (251) | 44.1 (29) | 66.6 (16) | 55.5 (213) |
| DeepSeek graph, offset 2 | 24.0 (270) | 45.4 (73) | 55.8 (59) | 62.6 (23) |
| Mixtral graph, offset 0 | 44.7 (238) | 65.3 (71) | 92.8 (39) | 105.3 (152) |
| Mixtral graph, offset 2 | 69.8 (29) | 87.6 (23) | 118.0 (43) | 125.8 (117) |

- **At full batch (480-513) a decode step is the same length at every eager
  offset,** on both models and with hundreds of steps per cell: DeepSeek 166.0
  → 170.0, Mixtral 153.2 → 153.1 ms. Whatever raises TPOT in eager mode, it is
  not the length of a saturated decode step. That contradicts the "hot rank's
  GEMMs lengthen the step" reading recorded above for Mixtral eager.
- **The two models put the eager cost in different steps.** DeepSeek's
  below-full-batch decode steps slow by +48 to +56% at offsets 1 and 2, in every
  batch bin; Mixtral's are flat (150.5 → 150.6 ms at 350-480). Mixtral's cost is
  in prefill-carrying steps instead: those holding 2500-4096 prefill tokens go
  303.5 → 353.8 → 437.2 ms across offsets 0 / 1 / 2, the right scale to explain
  its TPOT slope given such steps hold ~40% of token-time. DeepSeek's equivalent
  steps move only 211 → 227 ms.
- **Graph mode behaves differently again.** Full-batch decode steps do slow
  (DeepSeek 55.5 → 62.6, Mixtral 105.3 → 125.8 ms) while DeepSeek's small-batch
  decode steps get *faster* (33.4 → 24.0 ms). The eager small-batch blowup is
  eager-specific.

### A confound: offset 0 runs without the instrument

`server_env` returns an empty environment at strength 0, so the router-bias
plugin is not installed at all in a balanced point, while at any nonzero offset
it wraps both expert-selection kernels and runs `logits + bias` on **every**
rank, including ranks whose bias vector is all zeros.

Host-side evidence that this costs real time in eager mode: inclusive host time
in `vllm::moe_forward_shared` on an unbiased rank is 65.8 ms per step at offset
0 (rank 3) and 100.2 ms per step at offset 2 (rank 1). An unbiased rank should
see no skew effect at all, so most of that +34 ms is the wrapper, not the
imbalance. The windows differ in step mix, so this is indicative, not measured.

**Every offset-0 comparison in this document is therefore skew plus instrument
overhead.** The eager small-batch decode slowdown is the result most exposed to
it, since those steps are host-bound. Graph mode captures the add into the
graph, which is the likely reason its small-batch steps do not show the effect,
and the graph-mode cost at full batch is the cleanest surviving evidence that
skew itself costs time. The fix is a zero-bias control: set the environment
variable with an all-zero vector so the wrapper runs and changes no routing.

### TPOT is still not reconstructed

Token-weighted mean step wall over all steps, against the twin unprofiled
point's TPOT:

| | DeepSeek eager | Mixtral eager | DeepSeek graph | Mixtral graph |
|---|---|---|---|---|
| Offset 0 | +1% | +6% | **-45%** | -10% |
| Offset 1 | +19% | +9% | | |
| Offset 2 | +16% | +8% | +4% | +0% |

The proxy is within a few percent at some points and 45% out at others, so step
wall does not yet account for TPOT. The profiled point is also a separate server
instance from the twin, and the proxy ignores queueing. Reconciling them needs
inter-token latencies from the client, which we do not capture.

### Per-rank picture, and what it does not settle

In eager mode at offset 2, in the small-batch decode steps that carry the
slowdown, rank 0's device is idle 192.6 ms of a ~230 ms step with only 14.5 ms
inside the collective, while ranks 1-3 sit 166-175 ms inside it. Rank 0 is not
compute-bound there: its device is waiting on its host, as the earlier eager
section described. In the few full-batch steps at the same offset the roles
differ again (rank 0 idle 56.5 ms, rank 2 idle 110.5 ms).

So the eager host-idle observation survives, but it is partly the instrument's
own host cost, and graph mode shows a cost without it. No single mechanism is
established. What is established is the latency effect itself and that the
earlier per-step explanations do not hold.

## The zero-bias control (2026-10-06)

`imbalance.bias_plugin_at_zero` installs the plugin at level 0 with an all-zero
vector, so both arms pay its per-call tensor add and the comparison isolates the
skew. `rbias-zero-control` runs level 0 and 2 at three repeats per model, once
with the flag and once without, on a quiet node (0 to 2 neighbour snapshots of 6
per run). TPOT mean, ms:

| | Level 0, no plugin | Level 0, plugin (zero vector) | Level 2 |
|---|---|---|---|
| DeepSeek, flag off | 165.1 ± 4.0 | | 214.2 ± 2.9 |
| DeepSeek, flag on | | 175.1 ± 3.2 | 209.5 ± 3.8 |
| Mixtral, flag off | 166.5 ± 1.4 | | 194.1 ± 3.0 |
| Mixtral, flag on | | 165.9 ± 0.6 | 191.3 ± 1.8 |

- **The instrument costs 10.0 ms on DeepSeek** (+6.1%, t = 3.4) and nothing
  measurable on Mixtral (-0.6 ms, t = -0.6). The level-2 arms are identical
  configurations in both runs, so their difference is run-to-run drift: -2.2% on
  DeepSeek and -1.5% on Mixtral. Drift and the instrument cost have opposite
  signs here, so correcting for it puts DeepSeek's instrument cost nearer
  13.6 ms (+8%).
- **The skew cost at level 2, with the instrument held constant: +34.3 ms on
  DeepSeek and +25.3 ms on Mixtral.** Measured against a plugin-free baseline in
  the same grid the same quantity reads +49.1 and +27.6 ms, so the earlier form
  of the comparison overstates DeepSeek by 43% and Mixtral by 9%, the latter
  within drift.
- **Why only DeepSeek.** Eager DeepSeek decode is host-bound here, which the
  step-resolved section shows directly, so extra host work per call lengthens the
  step. Mixtral's step is dominated by device time, which hides it. DeepSeek also
  adds a 64-wide vector where Mixtral adds an 8-wide one.

**The earlier sweeps cannot be corrected after the fact.** The plugin was present
at exactly the nonzero levels, so its offset is collinear with "level > 0": no
refit can separate a constant instrument cost from a genuine jump at the first
nonzero level. Every eager DeepSeek figure in this document is an upper bound
until the sweep is re-run with the flag on. Mixtral's figures stand.

## The controlled sweep (2026-10-06)

`rbias-controlled-000/001` (`configs/grids/router_bias_controlled.toml`) repeat
the eager sweeps with `bias_plugin_at_zero`, dropping the levels that collapse
routing. 3 repeats in shuffled rounds. TPOT mean, ms (± sd):

| Offset | 0 | 0.25 | 0.5 | 1 | 2 | 3 |
|---|---|---|---|---|---|---|
| DeepSeek | 174.0±4.2 | 173.9±4.4 | 171.8±1.6 | 175.8±2.3 | 212.6±3.7 | |
| Mixtral | 165.5±1.3 | 167.1±1.4 | 171.0±2.0 | 176.6±0.5 | 193.9±1.1 | 209.8±0.8 |

- **Mixtral is linear from the first offset:** +19.9 ms (+12.0%) per +1x of
  busiest-rank load, t = 52, and +18.7 ms per +1x on offsets 0-1 alone. Its
  earlier figures stand.
- **DeepSeek is flat until somewhere between 2.34x and 3.45x.** Offsets 0-1 fit
  +1.4 ms per +1x (t = 0.7); offset 2 is +38.6 ms. A linear fit over all
  levels (+16.4 ms, t = 6.5) is the wrong model for it. The +7.2% per +1x and
  the gradual rise in the superseded table were the plugin's cost riding on
  every nonzero level.
- **A neighbouring job ran throughout** (an unrelated vLLM server on NPUs 4-7,
  load average 21-28; 25 of 30 and 31 of 36 snapshots flag it), and no point ran
  without it. It is not aliased onto level: within-level residuals do not differ
  between points with the neighbour at both snapshots and at one (-1.0 against
  +2.0 ms on DeepSeek, -0.1 against +0.2 on Mixtral), and the level-0 and
  level-2 means match the quiet-node zero-control within 1.5% (DeepSeek 174.0 /
  212.6 against 175.1 / 209.5; Mixtral 165.5 / 193.9 against 165.9 / 191.3).

**Where natural traffic sits.** Busiest-rank load of unbiased MMLU traffic, from
the routed-expert captures of `alpha-sweep` (DeepSeek, 3000 prompts) and
`mixtral-alpha` (1500), prompt and generated tokens pooled, 4-way contiguous
placement:

| Window | DeepSeek | Mixtral |
|---|---|---|
| Single prompt (p1 / median / p99) | 1.107 / 1.142 / 1.174 | 1.140 / 1.182 / 1.249 |
| One MMLU subject (min / median / max) | 1.097 / 1.119 / 1.134 | 1.101 / 1.125 / 1.156 |
| Random 200-prompt mix (p1 / median / p99) | 1.093 / 1.097 / 1.101 | 1.110 / 1.117 / 1.122 |

A single prompt is the most coherent window natural traffic offers, roughly a
prefill chunk of one long prompt, and it stays below the lowest offset either
model was swept at. Load here is the busiest rank per MoE layer, averaged over
layers. The calibration table above takes the busiest rank of the
layer-averaged shares instead, which lets the hot rank cancel across layers in
natural traffic: it reads 1.02x where this reads 1.14x on DeepSeek's balanced
point, and the two agree at every biased level. The figures use the per-layer
form throughout. Calibration routed prompt tokens only (`max_new_tokens = 1`);
the natural captures include 100 generated tokens.

## Graph mode, controlled, and the threshold sweep (2026-10-07)

`rbias-graph-threshold` repeats the graph-mode sweeps with
`bias_plugin_at_zero` - the graph runs in the Graph mode section above measured
against a plugin-free balanced point - and adds a third run filling the gap in
DeepSeek's eager curve. TPOT mean, ms (± sd over 3 repeats):

| Busiest rank | DeepSeek graph | Mixtral graph |
|---|---|---|
| balanced | 142.3 ± 8.0 | 146.1 ± 10.3 |
| 1.24-1.30x | 140.5 ± 3.8 | 143.5 ± 7.2 |
| 1.38-1.66x | 140.1 ± 4.2 | 157.4 ± 7.4 |
| 1.76-2.34x | 160.3 ± 18.7 | 152.1 ± 10.7 |
| 2.62-3.45x | 191.2 ± 13.1 | 187.7 ± 1.9 |
| 3.36x | | 208.9 ± 0.4 |

- **The cost survives the control on both models**, +22.6 ms per +1x of
  busiest-rank load on DeepSeek (t = 6.7, and +22.2 with execution order as a
  covariate).
- **The noise sits at the *low* levels, not the high ones.** Mixtral's balanced
  point has sd 10.3 ms while its two most-skewed points have sd 1.9 and 0.4, and
  its low end is non-monotone (143.5 -> 157.4 -> 152.1). That is the opposite of
  a contention story, which would scale with the work, and it fits the frontend
  starvation already recorded for graph mode: when the engine serves faster than
  the client feeds it, TPOT partly measures the client. **Per-level graph-mode
  numbers below ~2x should not be quoted**; the slope and the high-skew points
  are what this run supports.
- **DeepSeek's graph run also drifted.** TPOT rose with execution order
  (+1.2 ms per position, t = 2.1), and order correlates 0.68 with the host load
  average, which climbed from ~23 to ~29 as the neighbouring job got busier. The
  shuffled rounds kept it off the slope but not out of the variance.

### The eager threshold sweep

`rbias-graph-threshold-002`, eager, offsets 0 / 1 / 1.25 / 1.5 / 1.75 / 2, on a
**quiet node** - zero foreign NPU processes in all 36 snapshots. The new offsets
calibrate to 2.67x, 2.95x and 3.21x, inside the gap the controlled sweep left:

| Busiest rank | TPOT (ms) | vs balanced | controlled sweep |
|---|---|---|---|
| 1.14x | 176.1 ± 13.7 | - | - |
| 2.35x | 187.5 ± 9.0 | +6.5% | **+1.0%** |
| 2.67x | 194.8 ± 2.3 | +10.6% | not swept |
| 2.95x | 184.8 ± 12.4 | +4.9% | not swept |
| 3.21x | 205.3 ± 4.8 | +16.6% | not swept |
| 3.45x | 209.3 ± 2.3 | +18.9% | +22.2% |

- **The two sweeps disagree at 2.34x**, +6.5% against +1.0%, by more than either
  one's repeat spread. The endpoints agree (+18.9% against +22.2%).
- **Execution order explains none of it** (-0.01 ms per position, t = 0.0), so
  the balanced point's wide spread (188.3, 178.7, 161.3) is noise rather than
  drift, and the quiet node did not make this run tighter than the contended one.
- **A straight fit gives +13.1 ms per +1x (t = 4.2) with a residual sd of
  9.9 ms**, and the sequence is non-monotone (2.95x reads below 2.67x). Three
  repeats are not enough to resolve the shape at this noise level.
- **What is settled:** DeepSeek's cost is real and large by 3.2-3.5x, and absent
  at natural load. **What is not:** whether it begins near 2.3x or only above
  3x, and therefore whether it is a threshold or a slope. Resolving it needs more
  repeats in 2.3-3.5x, and an explanation for why eager DeepSeek's repeat spread
  is 2-5 ms in one run and 9-14 ms in another of the same configuration.

## Rotating skew: the two models part company (2026-10-08)

`rotating-skew` biases rank *(layer mod 4)* instead of one fixed rank, so every
layer is skewed as hard as before but no rank is hot throughout. Its calibration
matches the fixed target's per-layer load to within 3% (1.73 / 2.46 / 3.59x
against 1.66 / 2.35 / 3.45x), which makes the two sweeps a matched pair: same
per-layer skew, opposite rank consistency. TPOT mean, ms, aligned on per-layer
load:

| | Rotating | | Fixed | |
|---|---|---|---|---|
| **Mixtral** | 1.09x | **171.7 ± 6.8** | 1.18x | **165.5 ± 1.3** |
| | 1.35x | 167.7 ± 2.0 | 1.38x | 171.0 ± 2.0 |
| | 1.75x | **176.4 ± 3.5** | 1.76x | **176.6 ± 0.5** |
| | 2.63x | **193.2 ± 2.4** | 2.62x | **193.9 ± 1.1** |
| **DeepSeek** | 1.06x | 165.7 ± 12.9 | 1.14x | 174.0 ± 4.2 |
| | 1.73x | 165.4 ± 9.3 | 1.66x | 171.8 ± 1.6 |
| | 2.46x | 177.3 ± 10.4 | 2.34x | 175.8 ± 2.3 |
| | 3.59x | **174.8 ± 5.5** | 3.45x | **212.6 ± 3.7** |

- **On Mixtral, rotation changes nothing.** At matched load the two sweeps land
  on the same millisecond: 176.4 against 176.6 at ~1.75x, and 193.2 against
  193.9 at ~2.6x. Mixtral's cost is a property of the skew in each layer, and
  which rank carries it does not matter.
- **On DeepSeek, rotation removes the cost.** A rotating skew is flat across the
  whole range (165.7, 165.4, 177.3, 174.8; slope +4.4 ms per +1x, t = 1.5) where
  a fixed one reaches +22% at the same load. At 3.59x rotating costs 174.8 ms
  against the fixed sweep's 212.6 ms at 3.45x, with comparable balanced points.
- **So DeepSeek's cost needs a persistently hot rank and Mixtral's does not.**
  That fits where each model's cost was already localised: DeepSeek's in
  below-full-batch decode steps, where one rank being late at every layer
  accumulates down the stack, and Mixtral's in prefill-carrying steps, where each
  layer's own GEMM is on the critical path whatever rank it sits on.
- **Mixtral's slope comparison is the weaker form of this.** Fitted over all
  levels it reads +15.9 ms per +1x rotating against +19.9 fixed, which looks like
  a reduction; the whole difference is the rotating run's noisy balanced point
  (171.7 ± 6.8 against 165.5 ± 1.3). The matched-load comparison above is the one
  to trust.
- **DeepSeek's rotating run is noisy** (repeat sd 5-13 ms, 22 of 24 snapshots
  flag a neighbour) and its balanced point carries one outlier at 150.9 ms.
  Against medians the result is unchanged: 172.4 ms balanced, 172.9 ms at 3.59x.

**What this means for the null.** Natural traffic rotates - the busiest rank
leads 34-35% of layers. On DeepSeek that is a second reason its natural imbalance
is free: not only is the skew small, it is also the kind that costs nothing even
when large. On Mixtral that reasoning does not hold, and only the smallness
protects it.

## Inter-token latency capture: usable in aggregate, not in distribution

`benchmark.save_itl` records the gap between consecutive streamed chunks of each
request, as `itl_ms` in the metrics file. It behaves as intended at low load and
not at the load these sweeps use.

| | 60 prompts, 32 tokens | 3000 prompts, 100 tokens |
|---|---|---|
| Pooled mean ITL vs mean TPOT | 122.4 vs 122.6 ms | 157.6 vs 162.5 ms |
| ITL p50 | 118.8 ms | 0.3 ms |
| Gaps under 1 ms | none | 62-70% |
| `itl_ms_spike_share` | 0.00 | 0.999 |

- **The mean is sound**: it sits within 3% of TPOT at every point, which is the
  consistency the series has to satisfy.
- **The distribution is the client's, not the engine's.** A typical series runs
  `[..., 616.0, 0.3, 0.2, 194.9, ...]`: one long stall, then several chunks
  within a millisecond. With 3000 streams on one asyncio loop the client cannot
  service sockets promptly, so chunks are read in bursts. The percentiles and
  `itl_ms_spike_share` therefore measure client delivery, and are not evidence
  about decode steps at this concurrency. They are meaningful at 60 prompts.
- **The fix is server-side.** vLLM exposes `vllm:inter_token_latency_seconds` as
  a histogram on `/metrics`, measured in the engine. Scraping it before and after
  each point gives an ITL distribution immune to client batching. Until that
  exists, TPOT still cannot be attributed to particular steps.

## Prefill-only serving was frontend-starved

vLLM's periodic stats line records the engine's waiting and running queues.
Share of stats intervals with no request waiting and at most one running:

| Run | Empty queue |
|---|---|
| DeepSeek prefill only (with neighbour) | 90% |
| DeepSeek prefill only, quiet re-run | 83% |
| Mixtral prefill only | 38% |
| DeepSeek 100 tokens | 1% |
| Mixtral 100 tokens | 0% |

A point runs either at ~22k tokens/s with hundreds of requests queued, or at
~5-6k tokens/s with an empty queue. It switches mid-batch, and later points in
a run spend more time in the slow mode. The quiet re-run
(`20261006-083710`, no neighbour at any snapshot) was *more* variable than the
original (makespan sd up to 32 s on ~60 s, order t = 3.2), so the neighbouring
job was not the cause. The client and API server cannot submit and tokenise
3000 prompts fast enough to keep DeepSeek's prefill steps fed. The node's
load average stays at 12-15 with no NPU job running (other users' processes),
which is the likely reason, but it is unconfirmed.

This affects earlier work. The 2026-10-05 regime findings put prefill-only
makespan variance down to the neighbouring job, and their Mixtral section calls
throughput collapses to 6-9k tokens/s "external". Both fit frontend starvation
better. The prefill *traces* in that document are unaffected: they measure
device time per step.

## Contention

Every point of the 2026-10-05 sweeps had another user's vLLM job on NPUs 4-7
(a chain of servers restarting every ~20-25 minutes, through the gaps between
our runs). Its presence was balanced across offsets (3-6 snapshots per level),
and adding it to the fit leaves every 100-token coefficient unchanged (Mixtral
TPOT: +12.3% per +1x with it, t = 27).

## Runs

| Run | What |
|---|---|
| `20261005-164753`, `-170842` rbias-calibration | Offsets 0-2 and 4, DeepSeek and Mixtral |
| `20261006-082937`, `-083325` rbias-calibration | Offset 3, DeepSeek and Mixtral |
| `20261005-173114`, `-181940` rbias-deepseek-000, -001 | DeepSeek, prefill only / 100 tokens, 6 offsets x 3 |
| `20261005-193016`, `-201135` rbias-mixtral-000, -001 | Mixtral, prefill only / 100 tokens, 6 offsets x 3 |
| `20261005-211031` ... `-224930` rbias-profiled-* | Traces at offsets 0 / 1 / 2, both models, both regimes |
| `20261006-083710` rbias-deepseek-quiet | DeepSeek prefill-only re-run, no neighbour |
| `20261006-094351`, `-103651` rbias-graph-000, -001 | Graph mode, 100 tokens, offsets 0-2 x 3, DeepSeek and Mixtral |
| `20261006-112553`, `-115820` rbias-graph-profiled-* | Graph-mode traces at offsets 0 / 2 |
| `20261006-135100` itl-smoke | ITL capture and zero-bias control, 60 prompts |
| `20261006-135720` ... `-150633` rbias-zero-control-000..003 | Zero-bias control: 2 models x plugin on/off x levels 0 / 2 x 3 |
| `20261006-155755`, `-165818` rbias-controlled-000, -001 | Controlled eager sweep: plugin at level 0, DeepSeek 5 / Mixtral 6 offsets x 3 |
| `20261007-100316` rbias-calibration | Offsets 1.25 / 1.5 / 1.75 on DeepSeek |
| `20261007-101356`, `-112132` rbias-graph-threshold-000, -001 | Controlled graph sweep, DeepSeek and Mixtral |
| `20261007-123441` rbias-graph-threshold-002 | Eager DeepSeek threshold sweep, quiet node |
| `20261007-202156` rotate-calibration | Rotating bias: per-layer load against the fixed target's |
| `20261007-203534`, `-214459` rotating-skew-000, -001 | Rotating skew, DeepSeek and Mixtral, 4 offsets x 3 |

Analysis: `scripts/dose_response.py <grid or experiment name>` for the latency
fits; `scripts/plot_findings.py` for the figures, which needs the two
routed-expert captures pulled with `npull -a` or copied directly. On the node, where the raw traces live: `scripts/step_profile.py <run_dir>`
writes per-step tables (small enough to pull), `scripts/kernel_gaps.py
<trace_dir>` attributes idle time to kernel transitions, and
`scripts/host_ops.py <trace_dir> <rank> <first> <last>` reports host time per
step. The per-step normalisation in the
kernel table was done inline and is superseded by `step_profile.py`. The
profiled prefill-only points are not used: their profiler windows caught 56 to
202 steps of very different phases.

## Open

1. **Replicate DeepSeek's eager curve between 2.3x and 3.5x.** Two sweeps
   disagree there and neither resolves the shape. The prior question is why the
   same configuration gives a repeat spread of 2-5 ms in one run and 9-14 ms in
   another; without that, more repeats may not converge.
2. **Scrape `vllm:inter_token_latency_seconds` per point.** Client-side ITLs are
   sound in the mean but their distribution is client delivery at this
   concurrency, so within-request spikes are still unmeasured. The server-side
   histogram is immune to it; a `/metrics` read before and after each point is
   the whole change.
3. **Why DeepSeek's eager small-batch decode steps blow up.** +48 to +56% at
   offsets 1-2 (Mixtral's are flat),
   absent in graph mode, with the pacing rank's device idle. Candidates: the
   instrument (item 1), host-launch boundedness of small steps, CPU contention
   from a node whose load average sits at 12-15. Host events are in
   `ascend_pytorch_profiler_*.db`; `trace_view.json` is 3.7 GB per rank.
4. **Fix the frontend before any more prefill-only latency work.** Pre-tokenise
   prompts, run more API server processes, or drive the engine directly. Until then,
   prefill-only makespan and TTFT are frontend measurements.
   Graph mode makes this more pressing: even the 100-token regime starts to
   starve.
5. **The step-boundary gap.** 45% of a balanced graph-mode DeepSeek step is the
   host preparing the next step. Async scheduling, if vllm-ascend supports it
   here, is the biggest available win on this deployment, independent of
   imbalance.
6. **Where DeepSeek's threshold sits.** Between 2.34x and 3.45x; no level was
   swept in that interval, and above 3.45x routing collapses onto rank 0's
   experts. Offsets 1.25 and 1.5 would locate it. Whatever sets it is also the
   mechanism question, since Mixtral shows no threshold.
