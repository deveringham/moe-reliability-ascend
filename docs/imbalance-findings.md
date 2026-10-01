# MoE Imbalance Findings

2026-10-02

Local copy of the shared doc at
https://claude.ai/code/artifact/3a7b75ec-2d4e-462f-b021-2babaa95ec70 — edits made
there are not reflected here.

Expert load imbalance did not measurably change serving latency on DeepSeek-V2-Lite
under any manipulation tried: router bias 0 to 100, workload alpha 0.75 to 1.44, at
4 and 8-way expert parallelism. The traces show the reason is not that imbalance
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

Scope of the claim: DeepSeek-V2-Lite-Chat (64 routed experts, top-6, 26 MoE layers),
vLLM Ascend 0.23.0, 2 to 8-way expert parallelism, batch 512, 300-request workloads
served in a single wave. It does not extend to wider expert parallelism, to models
with coarser expert granularity, or to deployments where MoE is a larger share of
step time.

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

Two 8-NPU runs before the last one were discarded: another user's job took NPUs 4–7
mid-run, both hung, and the contended readings were inflated throughout — one point
by 12%, which looked exactly like the predicted effect.

## The straggler, measured

The shared doc carries this as a chart; the underlying values are below. Source: NPU
profiler `kernel_details.csv`, 6 workload alphas x 2 expert-parallel widths,
identical workloads.

| Effective alpha | Straggler, 4 ranks | Straggler, 8 ranks |
| --- | --- | --- |
| 0.745 | 1.044x | 1.052x |
| 0.800 | 1.036x | 1.054x |
| 1.000 | 1.037x | 1.057x |
| 1.301 | 1.037x | 1.063x |
| 1.441 | 1.055x | 1.088x |
| 1.443 | 1.051x | 1.087x |

The metric pairs each GroupedMatmul call across ranks and sums the per-call maxima,
divided by the mean rank total: what the layers actually waited for. Per-rank totals
on the same data read 1.004x to 1.015x and show none of this. The two rightmost
rows are near-replicates — effective alpha 1.441 and 1.443 — so the gap between them
is the measurement's own scatter.

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

MoE is a large enough share of kernel time for this to have mattered: 44% at 4
ranks, 34% at 8. The effect is absent because it is compensated, not because it is
negligible.

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

Two smaller ones. Sweep points run in alpha order, so anything drifting during a run
aliases onto the swept parameter — that is what manufactured an apparent threshold
at alpha 1.2. And the first MoE layer reported `cv_nat` of 7.94, which is sqrt(63):
layer 0 of DeepSeek-V2-Lite is dense, routes everything to one expert, and inflates
the reported workload MAE by roughly half at the extreme alphas. Effective alpha
should be computed excluding it.

## Tooling fixed

Twelve commits on branch `claude-fixes`, 90 tests passing (from 79). `nrun`, `npull`
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

Two of these were found by the tooling itself within minutes of being committed: the
contention check fired on 3 of 9 points in the positive control, and repeats exposed
the false 136.2 ms reading.

## Open questions

1. **Wider expert parallelism.** The straggler scales with it: for the same
   workloads the busiest rank carries 1.04x the mean at 2 ranks, 1.13x at 4, 1.31x
   at 8 and a predicted 1.63x at 16. Sixteen ranks would reach bias-100 levels of
   rank imbalance with all 64 experts still live, which is the one regime where the
   straggler might outrun the efficiency gain.
2. **Coarser expert granularity.** DeepSeek-V2-Lite is close to the least favourable
   case — 64 experts, top-6, so 16 per rank at 4-way EP and heavy averaging. Mixtral
   8x7B is 8 experts, top-2: 2 per rank, where averaging suppresses imbalance by
   about 1.4x instead of 5.6x. `forced_imbalance.toml` already defaults to Mixtral.
3. **Does EPLB change anything?** `server.enable_eplb` exists, so somebody expected
   imbalance to cost something. Running high imbalance with it on and off is a
   direct test, and a null there would be a strong result in itself.
4. **Where the step time actually goes.** MoE is 34–44% of kernel time, and an 8.8%
   straggler in it does not surface in TPOT. Worth decomposing the decode step —
   attention, communication, scheduler — to establish what the binding constraint
   is.

Still unfixed in the pipeline: sweep points run in alpha order (randomising would
decorrelate time from the swept parameter), and `start_vllm_server`'s readiness poll
is a `while True` with no deadline, which is what hung one 8-NPU run for 50 minutes.
