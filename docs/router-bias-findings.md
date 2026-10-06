# Graded Rank Skew: Router-Bias Sweeps on 4 NPUs

2026-10-05/06. DeepSeek-V2-Lite-Chat and Mixtral 8x7B, 4x Ascend 910B3
(NPUs 0-3), 4-way expert parallelism, batch 512, eager mode, 4096-token budget.
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

The mechanisms differ:

- **Mixtral: the hot rank's GEMMs.** MoE is ~63% of compute. Nearly
  all of rank 0's extra GEMM time reaches the step, and per-step trace growth
  matches the measured TPOT growth within a few ms.
- **DeepSeek: the hot rank's host.** The GEMM straggler accounts for 7 of the
  57 ms per step that offset 2 adds. The rest is rank 0's device sitting idle
  between kernel launches. Skew makes rank 0 the host-bound rank that every
  collective waits for.

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

The GEMM excess (+14.8, +38.5 ms) accounts for the step growth (+15.6, +31.3)
and for the sweep's TPOT growth (+12.3, +28.6). At offset 2 the hot rank has
the *least* idle time between kernels (24.5 ms per step against 40-47 ms on the
other ranks, `scripts/kernel_gaps.py`). It is device-bound, and every other
rank waits for it inside the all-reduce. This is the mechanism the 2026-10-05
regime work predicted once GEMMs dominate the step: on Mixtral they do, and it
only needed a skew that natural traffic does not supply.

### DeepSeek: the skew moves the host-bound pace-setter onto rank 0

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
- **This is the 2026-10-02 pace-setter,** the arbitrary low-occupancy rank, now
  explained mechanistically: DeepSeek decode in eager mode is host-launch-bound,
  and the slowest host sets the step. Skew makes rank 0 that rank and makes it
  slower than the arbitrary pace-setter was (148 against 129 ms idle).
- **It is not more launch work.** The host-side CANN API totals
  (`api_statistic.csv`) for rank 0 and rank 3 are the same at both offsets: the
  same launch count, the same launch time, no extra synchronisation calls. The
  extra time is spent between launches, in Python or in CPU scheduling. Which of
  the two is open.

A ~19 ms step growth would follow from rank 0's idle exceeding the old
pace-setter's, plus 6.8 ms from the GEMM straggler. The measured +57.5 ms is
larger. The profiled window also holds fewer steps at offset 2 (379 against
440), so part of the gap may be a different mix of prefill and decode steps.

### A constant step-boundary gap

Every rank of both models has a ~16-18 ms gap between `Cast` and `Fill` once
per step: about 10% of a decode step during which no NPU runs anything. It does
not depend on skew. It is the scheduler and input preparation between steps,
and graph mode or async scheduling is the lever for it, not balancing.

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

Analysis: `scripts/dose_response.py <grid or experiment name>` for the latency
fits; `scripts/kernel_gaps.py <trace_dir>` (on the node, where the raw traces
live) for idle time by kernel transition. The per-step normalisation in the
kernel table was done inline and should move into the results library. The
profiled prefill-only points are not used: their profiler windows caught 56 to
202 steps of very different phases.

## Open

1. **Why the hot rank's host is slower on DeepSeek.** The host-side Python
   events in `trace_view.json` (3.7 GB per rank, too large for `json.load`; use
   `ascend_pytorch_profiler_*.db`) would show where rank 0's time between
   launches goes. CPU affinity is the other candidate: four workers sharing a
   host with a load average of 15 is the natural way to get one arbitrarily
   slow rank.
2. **Fix the frontend before any more prefill-only latency work.** Pre-tokenise
   prompts, run more API server processes, or drive the engine directly. Until then,
   prefill-only makespan and TTFT are frontend measurements.
3. **Graph mode on DeepSeek.** If the DeepSeek effect is host-launch-bound,
   graph mode should mostly remove it while leaving Mixtral's GEMM effect
   intact. That is a cheap test of the mechanism.
4. **Where natural traffic sits.** The cost scale is now calibrated: ~12% TPOT
   per +1x on Mixtral. A workload shift that pushed a rank to 1.5x would cost
   ~6%, so detection needs to resolve rank load near that point, not the
   1.1x natural level.
