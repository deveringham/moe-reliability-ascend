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
regime TPOT rises with the busiest rank's load:

- **Mixtral:** +11.8% per +1x of busiest-rank load (t = 36), monotone from
  163.5 ms balanced to 208.7 ms at 3.36x.
- **DeepSeek:** +7.2% per +1x (t = 7.4), 179.1 ms to 213.1 ms at 3.45x.

**Graph mode leaves the cost unchanged in milliseconds** (+13.5 against +13.0 ms
per +1x on DeepSeek, +18.1 against +19.2 on Mixtral). Graph mode is how a
production deployment would run, so the result does not depend on eager mode's
launch overhead.

**Mechanism: not established on either model.** Step-resolved analysis
(2026-10-06, below) withdrew the per-step accounts this document first recorded.
A saturated decode step is the same length at every eager offset on both models
(DeepSeek 166.0 → 170.0 ms, Mixtral 153.2 → 153.1 ms, 150-220 steps per cell),
so the hot rank's GEMMs do not lengthen the step that most decode tokens wait
in. The eager cost sits in below-full-batch decode steps and in
prefill-carrying steps; in graph mode it sits in full-batch decode steps
instead. Token-weighted step wall still misses TPOT by -45% to +19%.

**Two caveats on the size of the effect.** A balanced point runs without the
router-bias plugin installed at all, so every comparison against offset 0
includes the instrument's own host cost, which is visible on ranks that carry no
bias. And graph mode serves fast enough that the engine starts to starve. The
graph-mode slope at full batch is the cleanest evidence that skew itself costs
time; the eager numbers are upper bounds.

**The 2026-10-02 null still holds where it was measured.** Natural traffic
gives a busiest rank of about 1.1x. The lowest offset (1.18-1.30x) costs 1-4%,
within a repeat's spread. The cost needs a busiest rank at 1.8x or more, which
only an injected skew has produced.

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
- **The eager cost sits in decode steps below full batch** (+43 to +55%) and in
  prefill-carrying steps. Mixtral's mixed steps carrying 2500-4096 prefill
  tokens go 303.5 → 353.8 → 437.2 ms across offsets 0 / 1 / 2, which is the
  right scale to explain its TPOT slope given those steps hold ~40% of
  token-time. DeepSeek's equivalent steps move only 211 → 227 ms.
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

Analysis: `scripts/dose_response.py <grid or experiment name>` for the latency
fits. On the node, where the raw traces live: `scripts/step_profile.py <run_dir>`
writes per-step tables (small enough to pull), `scripts/kernel_gaps.py
<trace_dir>` attributes idle time to kernel transitions, and
`scripts/host_ops.py <trace_dir> <rank> <first> <last>` reports host time per
step. The per-step normalisation in the
kernel table was done inline and is superseded by `step_profile.py`. The
profiled prefill-only points are not used: their profiler windows caught 56 to
202 steps of very different phases.

## Open

1. **A zero-bias control.** Serve a point with `MOE_ROUTER_BIAS` set to an
   all-zero vector, so the plugin wraps and adds as usual but routing is
   unchanged. That separates the instrument's cost from the skew's and is the
   single cheapest thing left: two points per model, no new code beyond letting
   `server_env` emit a zero vector. Every number in this document that compares
   against offset 0 depends on it.
2. **Inter-token latency capture in the client.** Step wall does not account for
   TPOT (-45% to +19%), and without ITLs there is no way to attribute a TPOT
   change to particular steps. This also closes the long-standing gap noted in
   status.md.
3. **Why eager small-batch decode steps blow up.** +43 to +55% at offset 1-2,
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
6. **Where natural traffic sits.** The cost scale is ~12-19 ms TPOT per +1x
   busiest-rank load, pending item 1. A workload shift that pushed a rank to
   1.5x would cost a few percent, so detection needs to resolve rank load near
   that point, not the 1.1x natural level.
