# MoE Imbalance Findings

2026-10-02 (updated with the Mixtral results)

Local copy of the shared doc at
https://claude.ai/code/artifact/3a7b75ec-2d4e-462f-b021-2babaa95ec70 — edits made
there are not reflected here.

Expert load imbalance did not measurably change serving latency under any
manipulation tried, on either model: router bias 0 to 100 and workload alpha 0.75
to 1.44 on DeepSeek-V2-Lite at 4 and 8-way expert parallelism, and alpha 0.78 to
1.43 on Mixtral 8x7B at 8-way, where one expert sits on each rank and MoE is 70%
of compute. The traces show the reason is not that imbalance
fails to form. Stragglers do form, and a kernel efficiency gain from coarser expert
grouping cancels them.

## Headline result

TPOT is flat across every imbalance level tested. The cleanest run, six workload
alphas on 8 NPUs with no competing work on the node, spans 130.04 to 130.87 ms — a
0.64% spread with no trend, while the predicted rank load rises from 1.18x to 1.31x
of the mean.

The forced-imbalance positive control is equally flat. A router bias of 100, the
maximum the infrastructure grid uses, gives 129.79 ms against 130.39 ms for the
unmodified model: -0.5%, p = 0.74, with three repeats per level. A bias of 25 is not
intermediate.

This is not a failure to create imbalance. At bias 100, vLLM's routed-expert capture
shows routing collapsed onto 6 of 64 experts, each taking exactly 1/6 of
assignments, with the coefficient of variation across experts at 3.109 against 0.265
for the base model. All six experts sit on one rank of four.

Mixtral 8x7B is the harder test and gives the same answer. It has 8 experts and
top-2, so on 8 NPUs each rank holds exactly one expert and rank load is expert load
with no averaging, and MoE is about 70% of compute against 34% on DeepSeek. TPOT
across six alphas spans 124.91 to 129.71 ms, 3.8%, with no ordering by imbalance;
the straggler runs 1.021x to 1.067x and is not monotone in alpha (rho 0.71,
p = 0.11). All 8 experts stay live at every alpha.

Scope of the claim: DeepSeek-V2-Lite-Chat (64 routed experts, top-6, 26 MoE layers)
at 2 to 8-way expert parallelism, and Mixtral 8x7B-Instruct (8 experts, top-2, 32
layers) at 8-way; vLLM Ascend 0.23.0, batch 512, 300-request workloads served in a
single wave. It does not extend to expert parallelism wider than 8, or to
deployments where the decode step is bound differently.

## Experiments

All on DeepSeek-V2-Lite-Chat, batch 512, expert parallelism on. Run ids are under
`results/` on the node.

| Run | Varied | NPUs | Result |
| --- | --- | --- | --- |
| `20261001-123944` fine-sweep | router bias 0 – 0.1, 6 levels, 1000 requests | 8 | No effect. TPOT 153.6–160.7 ms, non-monotone; apparent trend was batch size |
| `20261001-133042` alpha-sweep | workload alpha, 14 targets, token budget | 4 | Spearman +0.679 (p=0.008) on TPOT, but alpha and request count 98% collinear — unattributable |
| `20261001-151002` alpha-fixedn | alpha at fixed 300 requests, prompt length band | 4 | No effect. rho -0.143 (p=0.626). Removing the confound removed the trend |
| `20261001-161108` poscontrol | router bias 0 / 25 / 100, 3 repeats each | 4 | No effect. 100 vs 0 is -0.5%, p=0.74 |
| `20261001-164706` capture-imb100 | routed-expert capture at bias 100 | 4 | Routing collapsed to 6 of 64 experts, CV 3.109 vs 0.265 |
| `20261001-170127` trace-check | profiled bias 0 vs 100 | 4 | Rank 0 at 6.7x the GEMM work of others, yet still half the balanced case's per-rank time |
| `20261001-172617` alpha-repeats | alpha with prompt repeats, all 64 experts live | 4 | No effect. Straggler 1.036x → 1.055x |
| `20261001-222102` alpha-repeats-npu8 | the same workloads at 8-way EP | 8 | No effect. Straggler 1.052x → 1.088x |
| `20261001-233545` mixtral-alpha | the same sweep on Mixtral 8x7B, one expert per rank | 8 | No effect. TPOT 124.9–129.7 ms, straggler 1.021x–1.067x and not monotone, MoE 70% of compute |

Two 8-NPU runs before the last one were discarded: another user's job took NPUs 4–7
mid-run, both hung, and the contended readings were inflated throughout — one point
by 12%, which looked exactly like the predicted effect.

## The straggler, measured

The shared doc carries this as a chart; the underlying values are below. Source: NPU
profiler `kernel_details.csv`, 6 workload alphas x 2 expert-parallel widths,
identical workloads.

DeepSeek, by expert-parallel width:

| Effective alpha | Straggler, 4 ranks | Straggler, 8 ranks |
| --- | --- | --- |
| 0.745 | 1.044x | 1.052x |
| 0.800 | 1.036x | 1.054x |
| 1.000 | 1.037x | 1.057x |
| 1.301 | 1.037x | 1.063x |
| 1.441 | 1.055x | 1.088x |
| 1.443 | 1.051x | 1.087x |

Mixtral at 8 ranks, one expert each:

| Effective alpha | Straggler | TPOT ms | MoE % of compute |
| --- | --- | --- | --- |
| 0.778 | 1.032x | 124.91 | 70.0 |
| 0.813 | 1.021x | 128.00 | 66.5 |
| 1.000 | 1.056x | 125.90 | 70.6 |
| 1.221 | 1.055x | 126.58 | 70.1 |
| 1.339 | 1.033x | 129.71 | 66.9 |
| 1.428 | 1.067x | 127.27 | 70.1 |

The metric pairs each GroupedMatmul call across ranks and sums the per-call maxima,
divided by the mean rank total: what the layers actually waited for. Per-rank totals
on the same data read 1.004x to 1.015x and show none of this. The two rightmost
DeepSeek rows are near-replicates — effective alpha 1.441 and 1.443 — so the gap
between them is the measurement's own scatter.

Doubling expert-parallel width raises the straggler and its sensitivity to alpha.
Halving the experts per rank does not: Mixtral puts one expert on each rank against
DeepSeek's eight, and lands in the same band. Alpha scales the *natural* CV, and
Mixtral's is 0.141 against DeepSeek's 0.45, so asking for the same alpha asks for
the same relative imbalance — the finer granularity and the lower baseline cancel.
Rank load reaches 1.175x to 1.308x of the mean on Mixtral against 1.181x to 1.311x
on DeepSeek.

## Why latency never moves

Concentrating tokens onto fewer experts does two opposing things, and they cancel.
It creates a straggler, because the experts that get the work sit on fewer ranks. It
also makes `GroupedMatmul` cheaper per call, because the kernel runs fewer, larger
groups instead of many small ones. Total MoE GEMM time goes down, not up.

| Manipulation | Straggler | Mean GEMM call | Total GEMM time |
| --- | --- | --- | --- |
| alpha 0.75 → 1.44, 8 ranks | 1.052x → 1.088x | 64.4 → 58.8 us (-8.8%) | -15% |
| alpha 0.75 → 1.44, 4 ranks | 1.044x → 1.055x | ~112 us, flat | -24% |
| router bias 0 → 100, 4 ranks | 1.00x → 1.88x | 112 → 65.6 us (-41%) | -81% |

The bias-100 row is the clearest case. Rank 0 carries 6.7x the GEMM work of the
other three (105,710 us against ~15,800 us), yet its 105,710 us is still half the
198,812 us that *every* rank spends in the balanced case. The busiest rank under
maximum imbalance does less work than any rank under none.

MoE is a large enough share of the work for this to have mattered, and the Mixtral
run settles that objection. Of compute kernel time it is 34% on DeepSeek at 8 ranks
and about 70% on Mixtral, whose top-2 of 8 routing puts most of the model's
arithmetic in the experts. Latency is flat on both. The effect is absent because it
is compensated, not because MoE is too small a part of the step.

Those shares exclude communication. Measured as a share of *summed* kernel duration,
HCCL work is 80 to 88%, but a collective's duration is mostly the time it sat
blocked waiting for the other ranks, and kernels on different streams overlap, so
that sum is not wall time. By wall time — merging each rank's kernel intervals — a
profiled window runs at about 74% occupancy.

One caveat on the bias-100 figures. The checkpoint recipe zero-centres every gate
row and adds the bias to row 0, so the non-biased experts end up with near-identical
logits and `topk` breaks the ties by index. The manipulation therefore selects the
six lowest-numbered experts rather than skewing load toward expert 0, and it reduces
the active expert count from 64 to 6. That is where the efficiency gain comes from,
and it is why forced imbalance cannot isolate the straggler.

## Measurement pitfalls

Four of these produced a result that looked real and was not. Each is worth stating
in a methods section.

**Per-rank totals hide stragglers.** Summing each rank's kernel time and taking max
over mean gives 1.004x to 1.015x — near-perfect balance. Pairing the same calls
across ranks and taking the sum of per-call maxima gives 1.088x. The busiest rank
differs from layer to layer, so the totals equalise even though every layer waited.
A decode step runs its layers in sequence, so the cost is the sum over calls of the
max over ranks, not the max over ranks of the sum. `trace_max_over_mean` as the
results library computes it is the first form, and will report balance that is not
there.

**A token budget confounds imbalance with batch size.** Workloads built to a fixed
token budget reach a higher CV most cheaply by taking more, shorter prompts: request
count ran 237 at alpha 0.67 to 340 at 1.34. Alpha and batch size were 98%
collinear, and the resulting Spearman +0.679 (p=0.008) on TPOT was unattributable.
At a fixed 300 requests the trend disappears (rho -0.143, p=0.626).

**Per-request p-values are not evidence.** Mann-Whitney over 1000 requests per point
gave p < 1e-90 for differences that replication showed to be noise. The requests in
one sweep point share a server instance, a warmup and a scheduling pattern; the unit
of replication is the run, so n = 1 per point, not 1000.

**A single elevated point is usually the machine.** Three times a point looked like
an effect and dissolved under replication or on a quiet node: level 25 at 136.2 ms
against repeats of 129.7 and 130.1; alpha 1.2 at 149.6 ms while host load spiked;
alpha 1.6 at 146.0 ms on 8 NPUs, which read 130.7 ms once the node was free.
Measured contention cost is about 3% on TPOT and 31% on TTFT at load 16.

**The standard figure plots the statistic that hides it.** `kernel_sweep.png` draws
`max_over_mean`, the per-rank totals form, which reads 1.003x to 1.012x on the
Mixtral traces. Read at face value it says the ranks are balanced to within about
1%, while the paired-call straggler on the same traces is 1.067x. Anyone working
from that figure alone will conclude there is nothing to find.

Two smaller ones. Sweep points ran in alpha order until this was fixed, so anything
drifting during a run aliased onto the swept parameter — that is what manufactured
an apparent threshold at alpha 1.2. And the first MoE layer reported `cv_nat` of 7.94, which is sqrt(63):
layer 0 of DeepSeek-V2-Lite is dense, routes everything to one expert, and inflates
the reported workload MAE by roughly half at the extreme alphas. Effective alpha
should be computed excluding it.

## Tooling fixed

Fifteen commits on branch `claude-fixes`, 98 tests passing (from 79). `nrun`, `npull`
and `.rsync-exclude` are in `.git/info/exclude`, so those three fixes are
working-tree only and are not on the branch.

| Fix | What it was costing |
| --- | --- |
| `nrun` builds a job script; `-t` runs it in tmux | A tmux server started earlier carries its own environment, so the documented long-run recipe silently lost NNAL, the newer libstdc++ and the custom operator paths. Three failures at increasing depth, all looking like driver faults |
| Process-group teardown in `vllm_serving` | Only the API server was signalled; engine and workers kept the HBM, so the next sweep point started with 31 of 61 GiB free and failed. Being a race, it passed the first point and failed the second |
| `doctor` loads what it checks | Reported 14/14 on a node where nothing could run. Now 14/16, naming libatb.so and the GLIBCXX_3.4.29 failure |
| Figures stage no longer completes empty | A run that rendered 0 figures recorded the stage completed, and resume skips completed stages permanently |
| `separate_profiling_run` | Timings and traces came from one profiled pass, so the config discarded the latency data rather than report perturbed numbers |
| `length_in_requests`, `prompt_length_tolerance` | The token-budget confound above |
| `benchmark.repeats` | No noise floor. It caught a 136.2 ms reading that its own repeats put at 129.7 and 130.1 |
| Host contention snapshot per point | Another user's job inflated points invisibly. Now flagged live, by pid |
| Routed experts read from vLLM | The Hugging Face probe hooked the router module, which transformers 5.5.4 never calls — it computes the logits functionally from `gate.weight`. `validate_imbalance` raised KeyError and could not have worked on this stack. Removed 1199 lines |
| Profiler calls bounded | `start_profiling` had no timeout where `stop_profiling` had 600s. One profiled point hung 55 minutes holding 8 NPUs |
| `benchmark.shuffle_points` | Points were served in parameter order, so drift during a run aliased onto the parameter. Seeded, so it stays reproducible; the manifest keeps parameter order so plots are unaffected |
| Bounded server startup | The readiness poll was a `while True` with no deadline: a server that came up but never answered held all eight NPUs for 50 minutes |
| Ascend trace analysis | `trace_summary` read PyTorch-format traces this stack never writes, so it had only ever recorded an error. The new path reports wall time, compute shares, the straggler and the largest operators |
| `npull` excludes raw traces | A profiled run writes a few hundred MB of `trace_view.json` per rank per point; pulling one filled a 90 GB workstation disk. Use `-a` to include them |

Two of these were found by the tooling itself within minutes of being committed: the
contention check fired on 3 of 9 points in the positive control, and repeats exposed
the false 136.2 ms reading.

## Idle time: the ranks wait far more than they differ

Prompted by a question about whether imbalance shows up as device idle time or
transient spikes rather than mean latency. Neither does, but looking turned up
something larger. Computed offline over 30 profiled points already on disk -
DeepSeek-V2-Lite at 4 and 8 ranks, Mixtral at 8, alphas 0.5 to 2.0.

**About 95% of collective time is waiting, not transfer.** `hcom_allReduce_` is
66.6 s of the 87 s of summed kernel duration in a typical point. Splitting it by
`collective_wait` - a collective ends for every rank together, so the last rank
to arrive waits least and its duration bounds the transfer, while every other
rank's excess is blocked time:

```
wait share of collective duration: mean 94.9%   range 92.7% - 96.8%   n = 30
```

So the 71-78% "communication" in the step decomposition is almost entirely one
rank waiting for another; actual data movement is around 3-4% of summed kernel
time. Any reading of an MoE profile that treats summed collective duration as
communication cost is wrong by roughly twenty times.

**One rank paces each server instance, and it is a different rank every time.**
In most points a single rank is the last to arrive for 60-98% of all collectives
(`pace_setter_share` 0.54 to 0.98). Which rank it is changes between runs of an
identical configuration, so it is not topology: rank 0 in one point, rank 3 in
the next.

**The near-idle rank is the bottleneck, not a victim.** Occupancy per rank spans
0.18 to 0.90 within a single point, and the rank with the lowest occupancy is the
pace-setter in **30 of 30 points**. The inversion is the whole point: a rank that
looks idle is the one everyone else is blocked on, because the ranks that arrive
early spend the wait inside a collective kernel and therefore count as busy.
`busy_max_over_mean` reads about 1.33 at 4 ranks and 1.16 at 8, against a
fused-MoE straggler of 1.026.

**It has nothing to do with imbalance.** `busy_max_over_mean` shows no alpha
dependence (r = -0.02, +0.05, -0.28 across the three sweeps with enough points),
the wait share is flat across alpha, and fused-MoE time per rank matches to 0.5%
in every point. Whatever makes one rank late, it is not its share of expert work.

This gives the headline null a mechanism it did not have. Imbalance is invisible
twice over: a ~20% skew in expert tokens becomes only a ~2.6% skew in fused-MoE
time, and that 2.6% is then buried under a ~33% asymmetry in waiting caused by
something else entirely. The step is gated by collective sync, and the sync is
paced by an arbitrary rank.

Tails were checked at the same time and show nothing either. TPOT p99 and max
track the mean, and their correlation with alpha changes sign across runs
(+0.54, +0.17, -0.09, +0.58, -0.40, +0.52); the positive cases are exactly the
early runs served in alpha order or under a neighbouring job. One structural
effect did appear, independent of alpha: p99 sits 3.5% above the mean at 4 ranks
and 12.6% at 8, so tail inflation grows with expert-parallel degree.

Caveats. The split assumes a collective ends for all ranks at once, so with a
pipelined ring implementation the per-call minimum is an upper bound on wait and
a lower bound on transfer - the ratio is extreme enough that the conclusion holds,
but 94.9% is not a precise figure. Within-request spikes remain unmeasured: TPOT
averages over a request's decode steps and we record no inter-token latencies, so
a single slow step is smeared out. And `trace_active_iterations = 2` does not
bound the profiler window as its name suggests; the spans are ~34 s, essentially
the whole serving period, which is what made this analysis possible but should be
understood before relying on the option.

## EPLB: it works, and it buys nothing here

Run 2026-10-02, grid `eplb-eval`, DeepSeek-V2-Lite at 4-way expert parallelism,
alphas 0.8/1.0/1.6, two repeats per point, EPLB off and on with shared workloads
so the arms differ only in the rearrangement.

| | EPLB off | EPLB on |
| --- | --- | --- |
| paired-call straggler | 1.026 +/- 0.002 | 1.027 +/- 0.002 |
| fused-MoE time per rank | 1.349 ms +/- 0.025 | 1.351 ms +/- 0.021 |
| MoE share of compute | 38.30% +/- 0.15 | 37.53% +/- 0.27 |

EPLB ran properly: `SwiftBalanceEplb`, 84 rearrangement cycles across the 12
server instances, a placement map recorded for every point. By its own
accounting it does its job, over those 84 cycles:

```
current   mean 1.139 +/- 0.044   max 1.308 +/- 0.132
predicted mean 1.007 +/- 0.002   max 1.026 +/- 0.013
```

The measured straggler did not move by one standard deviation.

**Why: EPLB balances token counts, and what costs time is kernel time.** Its own
metric sees a 1.14x to 1.31x imbalance in expert hotness where the traces measure
1.026x in fused-MoE time. That is the cancellation of the headline result seen
from the other side - concentrating tokens makes `GroupedMatmul` cheaper per
token, so a ~20% imbalance in tokens is a ~2.6% imbalance in time. EPLB is
removing a 20% skew in a quantity that was already only 2.6% skewed in the
quantity that matters, so there is nothing left to win.

The consequence for monitoring is the useful part: **expert token counts are the
wrong signal**, and a `max/mean` of hotness overstates the time imbalance by
roughly a factor of six. Any detector built on router counters inherits that
error.

The one real cost is overhead. MoE share of compute falls 0.77pp (3-5 sigma, and
a within-run ratio so robust to host load) while absolute MoE time is unchanged -
EPLB added non-MoE work rather than saving MoE work. Total kernel time is about
9% higher, but with 10% scatter that is ~1 sigma and should not be quoted.

Two caveats:

- **The latency comparison from this run is unreadable.** A neighbouring job
  occupied NPUs 4-7 during 5 of the 6 off-arm points and only 1 of 6 on-arm
  points, so contention is confounded with the arm; TTFT scatters 1927-2915 ms.
  The straggler and MoE-share figures above are per-rank ratios within a run and
  survive this; the latency numbers do not. Interleaving the arms, or running
  both on a quiet node, is the fix.
- **The intervals were cut hard to make EPLB fire at all.** At the vllm-ascend
  defaults one cycle is 600 + 50 + one iteration per MoE layer, about 676 forward
  iterations, against roughly 100 decode iterations in a 100-token generation -
  EPLB would collect load for an entire run and never act on it. These runs used
  50 + 10 with 300 generated tokens, which means the 26-layer weight transfer
  occupies about a third of every cycle and the placement is rarely settled.
  Across all 84 cycles the realised imbalance (1.139) never approaches the
  predicted (1.007), which is either genuine non-stationarity of expert load at
  this timescale or an artefact of that cadence. One run at
  `eplb_heat_collection_interval = 200` separates them and would also measure the
  overhead fairly.

Also established, by reading vllm-ascend rather than measuring: rearrangement is
unconditional. `_compute_imbalance` is called every cycle and used only for a log
line - there is no threshold, no hysteresis, no gate. EPLB pays the weight
transfer on a fixed counter whether or not anything is wrong, and the counters
are cleared at the end of each cycle rather than decayed.

## Open questions

1. **Wider expert parallelism.** The one lever still untested, and the only one
   that raised the straggler so far: for the same workloads the busiest rank carries
   1.04x the mean at 2 ranks, 1.13x at 4, 1.31x at 8 and a predicted 1.63x at 16.
   Sixteen ranks would reach bias-100 levels of rank imbalance with every expert
   still live.
2. **Coarser expert granularity: answered, and it is not the lever.** Mixtral puts
   one expert on each rank instead of eight and changes nothing, because alpha is
   relative to a natural CV that is three times lower. To make granularity bite you
   would have to drive absolute CV, not alpha.
3. **Does EPLB change anything? Answered: no, and for an instructive reason.** It
   rearranges as designed and removes the token imbalance it measures, but token
   imbalance is six times larger than the time imbalance, so there is nothing left
   to win. See the EPLB section above. What remains open is its overhead at a
   realistic cadence, which these aggressive intervals overstate.
4. **Where the step time actually goes.** The traces now report it: about 74%
   occupancy by wall time, with collectives dominating summed kernel duration
   because they block. Worth separating transfer from wait inside the collectives,
   which is what would say whether a straggler can ever surface as latency here.
