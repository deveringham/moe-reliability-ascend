# Where Imbalance Costs Time: Regime Sweeps on 4 NPUs

2026-10-05. DeepSeek-V2-Lite-Chat, 4x Ascend 910B3 (NPUs 0-3), 4-way expert
parallelism, batch 512. Every point ran with another user's vLLM job on NPUs
4-7, so contention was present throughout.

The question: is there any serving regime on this deployment where expert
imbalance produces a measurable latency cost? Forced imbalance (router bias 0
against 100, which routes every token to experts 0-5, all on rank 0) was used
as the strongest available positive control.

## Answer

**Prefill is the one regime where imbalance costs kernel time, but the cost
barely reaches latency.** In prefill-only serving at bias 100, the hot rank
takes 1.6x a balanced rank's MoE GEMM time per call (straggler 3.7x). That is
the first condition on this stack where concentrating tokens makes the hot rank
slower. In decode it makes the hot rank faster. Expert GEMMs are only 10-16% of
a step's wall time, though, and the other ranks absorb most of the delay. The
step lengthens by 4.6% in eager mode and 1.5% in graph mode. End-to-end latency
shows a cost of roughly 3-5% in the 100-token regime with a 16k budget, and in
prefill-only serving the noise from the neighbouring job hides it. Graph mode
does not amplify the effect.

Decode, the regime of every earlier sweep, remains a null, now explained by the
GEMM regime rather than only observed.

**On Mixtral 8x7B the structure would pass imbalance through, but natural
traffic does not create it.** Expert GEMMs are 40-45% of a Mixtral prefill step,
four times DeepSeek's share. Natural-routing workloads, though, move the rank
straggler only from 1.11x to 1.14x, and prefill throughput stays flat to within
1%.

## Why prefill and not decode: the GEMM regime

For a bf16 DeepSeek-V2-Lite expert, arithmetic intensity is roughly the number
of tokens per call, and the 910B3's ridge point is ~200 (313 TFLOP/s over
1.6 TB/s). Below it, an expert's cost is loading its weights, which depends on
how many experts are active, not how many tokens they get.

| | Tokens per expert per call (mean) | Regime | Hot rank at bias 100 |
|---|---|---|---|
| Decode, batch 512 | ~48 | Memory-bound | Half the GEMM time of a balanced rank (2026-10-02) |
| Prefill, 4096-token step | ~384 | Compute-bound | 1.6x the GEMM time of a balanced rank |

This is the mechanism behind the 2026-10-02 headline. "Concentrating tokens
makes the GEMM cheaper" holds below the ridge point and stops holding above it.

## Kernel level: prefill traces

`prefill_profiled.toml`: prefill-only, 4096-token budget, 6000 prompts, profiled
pass.

| | Eager, bias 0 | Eager, bias 100 | Graph, bias 0 | Graph, bias 100 |
|---|---|---|---|---|
| MoE share of compute | 47% | 40% | 47% | 41% |
| GroupedMatmul straggler (paired calls) | 1.07x | 3.69x | 1.06x | 3.73x |
| Critical-path GEMM per call | 307 us | 489 us | 377 us | 587 us |
| Critical-path GEMM / wall | 9.5% | 14.5% | 10.7% | 16.5% |
| Wall time per GEMM call | 3,223 us | 3,372 us | 3,505 us | 3,559 us |
| Share of the GEMM delay reaching wall | | 82% | | 26% |
| Rank occupancy (mean) | 68% | 57% | 78% | 65% |
| Collective time spent waiting | 64% | 71% | 63% | 72% |
| Pace-setting rank | 2 (47%) | 0 (41%) | 0 (33%) | 0 (51%) |

- **MoE is not a small part of prefill.** At 47% of compute, it is a larger
  share than in decode (34-38%).
- **The straggler reaches the pace-setter.** At bias 100 the hot rank arrives
  last at collectives and sets the pace, where at bias 0 the pace-setter is
  arbitrary.
- **What caps the effect is how small GEMMs are in wall time.** A step
  costs ~3.5 ms per GEMM call against 0.3-0.6 ms of critical-path GEMM. The
  rest is collectives, mostly waiting, plus dense compute and idle time.
- **Graph mode cuts idle time but absorbs more of the straggler.** Occupancy
  rises from 68% to 78%. Only 26% of the extra GEMM time reaches the wall,
  against 82% in eager mode.

Comparisons are per call because the profiler window covers a different number
of calls per point (16,640 to 19,968). The unprofiled makespans from these runs
swung from 85 to 154 s with the neighbouring job and are not used.

## Latency level

Makespan is the batch wall time; `input_tokens_per_s` is its inverse in prefill.
Effects are order-adjusted (bias and execution order fitted together).

| Regime | Mode | Budget | Repeats/arm | Makespan, bias 100 vs 0 | TPOT, bias 100 vs 0 |
|---|---|---|---|---|---|
| Prefill only | Eager | 4096 | 3 | +9% (n.s., spread ~10%) | |
| Prefill only | Eager | 16384 | 3 | flat (one neighbour outlier per arm) | |
| 100 tokens | Eager | 4096 | 3 | 0.998 | 0.976 |
| 100 tokens | Eager | 16384 | 3 + 5 | +2.5% (t = 1.3, pooled) | +6.2% (t = 2.4, 5-repeat run) |
| Prefill only | Graph | 4096 | 4 | -19% (t = -1.4, neighbour outliers) | |
| 100 tokens | Graph | 4096 | 4 | -0.6% (t = -0.1) | +0.0% (t = 0.0) |

- **The 100-token regime with a 16k budget is the only latency signal.** With
  a large budget, decode steps carry big prefill chunks, which puts the
  compute-bound GEMMs on the steps every decoding request waits for. The signal
  is marginal: consistent across measures and with the traces, but not
  established under this contention.
- **Prefill-only makespan varies by ~10-20% between repeats** because of the
  neighbour. A 2-5% effect would need dozens of repeats per arm there. The
  traces resolve it directly.
- **Raising the budget from 4096 to 16384 left prefill throughput at ~20k
  tokens/s.** That fits a step whose time grows with tokens throughout, not
  only in the GEMMs.


## Mixtral 8x7B: the cap lifts, but natural routing does not skew enough

Same deployment shape (4 NPUs, eager, prefill-only, 4096-token budget), two
experts per rank, natural-routing workloads of 2000 requests built from the
1500 recorded MMLU prompts of the 8-NPU run. The workload pool reached
effective alphas of only 0.85, 1.00 and 1.20 against targets of 0.5-2.0.

| Prefill, 4096 budget, eager | DeepSeek bias 0 | Mixtral alpha 0.85 | alpha 1.00 | alpha 1.20 |
|---|---|---|---|---|
| MoE share of compute | 47% | 80% | 80% | 79% |
| Critical-path GEMM / wall | 9.5% | 41.9% | 39.8% | 45.1% |
| Rank occupancy | 68% | 88% | 83% | 88% |
| GroupedMatmul straggler (paired calls) | 1.07x | 1.111x | 1.106x | 1.143x |
| Per-rank totals, max / mean | 1.013x | 1.014x | 1.016x | 1.022x |

- **The structural cap lifts.** Expert GEMMs are 40-45% of a Mixtral prefill
  step, four times DeepSeek's share, and the ranks are 83-88% occupied. A
  rank carrying substantially more tokens would lengthen the step by a large
  fraction of its GEMM increase.
- **Natural routing does not create that skew.** From alpha 0.85 to 1.20 the
  straggler moves from 1.11x to 1.14x, at most ~1.3% on the step. Two experts
  per rank and top-2 routing average most of the expert-level skew away at
  rank level.
- **Throughput is flat.** Sustained prefill throughput (90th percentile of
  2.5 s bins) is ~16.3k, 16.3k and 16.2k tokens/s at the three alphas
  (`20261005-144149`, 4 repeats each, mostly without a neighbour).
- **Slow episodes are external.** Five of 15 points dropped from ~15k to 6-9k
  tokens/s partway through and sometimes recovered. They are not tied to a
  workload (they hit every alpha across the two runs) and host load stayed low;
  before/after snapshots are too coarse to catch their cause. A per-point
  throughput timeline (`end_s` binned) is the way to spot them.

This changes the question for detection on 4 NPUs. The deployment can turn
rank imbalance into latency once expert GEMMs dominate the step, which they do
on Mixtral. The limiting factor is whether traffic ever skews rank load enough.
Natural MMLU traffic does not.

## Method notes

- **Blocked shuffles fixed.** Every run here put all bias-100 repeats before any
  bias-0 one: a single seeded shuffle, shared by every run in a grid. Points are
  now served in rounds, one per repeat, each holding every value once, and the
  seed includes the experiment name. In the graph-mode 100-token run both arms
  slowed by ~15% from first half to second as the neighbour ramped up. The
  interleaved rounds cancelled that drift; the old blocked order would have
  turned it into a fake 15% effect.
- **New measurements.** Per-request start/end offsets give `makespan_s` and
  throughput per point (TTFT under all-at-once submission is mostly queueing).
  New options: `server.max_num_batched_tokens` and `server.enforce_eager`.
- **Graph mode runs on this stack.** vllm-ascend captures piecewise graphs for
  mixed steps up to the token budget and full graphs for decode, in 32 s and
  0.72 GiB. It serves ~11% faster (TPOT ~144 ms against 158-163 ms in the
  matched 100-token, 4096-budget regime).
- **Contention.** All points ran with another user's vLLM job on NPUs 4-7
  (15-54 GB per device, restarting every few minutes). Per-rank ratios within a
  profiled pass hold up under it; absolute timings do not.

## What this means for detection

- **The impact of imbalance is a product of three factors:** how far the hot
  rank's critical-path GEMM time rises, which depends on the GEMM regime and so
  on tokens per expert per step; GEMMs' share of the step; and how much of the
  delay reaches the wall clock (26-82% here).
- **Token counts are the wrong input on their own.** A detector should combine
  routing counts with tokens per step (decode vs prefill, batch, budget). Below
  the ridge point, skew is free or beneficial; above it, it costs time.
- **On this deployment the ceiling is a few percent.** Even total collapse onto
  one rank moves a prefill step by 1.5-5%. A mitigation such as EPLB has at most
  that much to win, minus its own overhead.

## Runs

| Run | What |
|---|---|
| `20261005-094638` ... `-103923` prefill-regime-000..003 | Grid: prefill-only / 100 tokens x 4096 / 16384, bias 0 / 100, 3 repeats |
| `20261005-110415` prefill-profiled | Eager traces, prefill, bias 0 / 100 |
| `20261005-111916` mixed16k-replicate | 100 tokens, 16384 budget, 5 repeats |
| `20261005-120755`, `-123138` graph-regime-000, -001 | Graph mode, prefill-only / 100 tokens, 4 repeats |
| `20261005-140350` graph-prefill-profiled | Graph traces, prefill, bias 0 / 100 |
| `20261005-144149` mixtral-prefill-alpha | Mixtral, prefill-only, alpha 0.5 / 1.0 / 2.0 (effective 0.85-1.20), 4 repeats |
| `20261005-151011` mixtral-prefill-profiled | Mixtral traces, same workloads |

Analysis: `scripts/regime_summary.py <grid>`; trace normalisation is inline in
this session and should move into the results library.

## Open

1. **A controlled rank skew on Mixtral.** The structure now passes rank
   imbalance through; what is missing is an instrument that creates it. The
   forced-imbalance recipe collapses routing onto experts 0 and 1, both on rank 0,
   which is a valid extreme positive control (rank 0 takes ~4x its share of
   expert work) but cannot grade the skew. A graded control needs a recipe that
   adds a logit bias toward one rank's experts without zero-centring the routers.
2. **What fills the rest of a DeepSeek step.** At ~3.5 ms per GEMM call against
   0.3-0.6 ms of GEMM, most of a DeepSeek prefill step is collective waiting and
   idle. The pace-setter question from 2026-10-02 applies here too.
3. **Natural imbalance in DeepSeek prefill.** `configs/examples/prefill_alpha.toml`
   is ready but, given a 1.5-5% ceiling under total collapse, is not expected to
   be measurable at the latency level.
