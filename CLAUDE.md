# moe-reliability-ascend

## Execution environment

This machine has **no Ascend NPUs**. The experiment runtime (vLLM Ascend,
`torch_npu`, the NPU profiler) cannot run here and will fail with misleading
import or device errors if you try. Do not attempt to install `torch_npu`,
`vllm-ascend` or the CANN toolkit locally, and do not "work around" an NPU
error by stubbing hardware out — that hides the bug we are trying to see.

### What must run on the node

Anything that touches real hardware. Use the `./nrun` wrapper, which rsyncs
the working tree to the node, sources CANN and NNAL, and runs the command:

```
./nrun 'npu-smi info'
./nrun 'uv run moe-reliability doctor'
./nrun 'uv run moe-reliability validate my_run.toml'
./nrun 'uv run moe-reliability run my_run.toml --set hardware.n_npus=4'
./nrun 'uv run moe-reliability grid configs/grids/forced_imbalance_infrastructure.toml --dry-run'
```

Then `./npull` brings `results/` back for analysis.

Notes on the wrapper:

- Every `nrun` syncs first, so local edits are always reflected. The sync
  excludes `.venv/` (the node's venv is aarch64 with a different dependency
  set), `.git/`, `results/` and `models/` (checkpoints are built on the node
  and must survive the sync's `--delete`).
- After changing `pyproject.toml` or `uv.lock`, run `./nrun 'uv sync'` before
  the next experiment.
- Long runs: `./nrun -t run 'uv run moe-reliability run ...'` starts the job in
  a detached tmux session so it survives a dropped connection. Follow it with
  `./nrun -t run -l`, or attach with `ssh ascend -t tmux attach -t run`.
  Session logs are kept in `~/.cache/moe-reliability/<session>.log` on the node.
- Never launch a job with a bare `tmux new -d -s run '<command>'`. A tmux
  server that is already running carries its own, older environment and does
  not inherit the one `nrun` sources, so the command silently loses NNAL
  (`libatb.so`), the newer `libstdc++` and the custom operator paths. The
  failures surface deep inside vLLM and look like driver or CANN faults.
  `nrun` sets the environment up inside the shell that runs the command, which
  is why `-t` is safe and hand-rolled tmux is not.
- Interrupted runs resume rather than restart:
  `./nrun 'uv run moe-reliability resume <run-id> --retry-failed'`.
  A stage recorded as `completed` is never re-run, so check the manifest before
  assuming a resume will reproduce a missing artefact.

### What can run locally

Anything not in the above category.

## Ascend conventions

- Device selection is `ASCEND_RT_VISIBLE_DEVICES`, **not** `CUDA_VISIBLE_DEVICES`.
- Device placement is `.to("npu")` / `torch.npu`, not `.cuda()`. CUDA idioms
  are a common wrong guess here — check against `src/moe_reliability/core/`
  for how the codebase actually does it.
- `npu-smi info` is the nvidia-smi equivalent for device state and utilization.
- Host requirements: Ascend driver/firmware HDK 26.0.RC1, CANN Toolkit, Ops
  and NNAL 9.1.0. `moe-reliability doctor` verifies all of this and is the
  right first command when anything environmental looks wrong.
- vLLM Ascend ships some operators (`AddRmsNormBias` and friends) as a CANN
  custom operator vendor package bundled in the wheel, not in CANN itself. They
  are dlopened at run time and need `ASCEND_CUSTOM_OPP_PATH` plus a libstdc++
  providing `GLIBCXX_3.4.29`, which is newer than the host's. `nrun` sets both.
  When they fail to load, CANN falls back to the stock operator library and the
  model reports `aclnnXxx ... not in libopapi.so` — that is a loader problem,
  not a reason to change the CANN version.
- `src/moe_reliability/environment.py` holds the environment checks and
  provenance capture — read it before adding new version or device assertions.

## Layout

- `src/moe_reliability/` — experiment runtime, Ascend-only. `cli.py`,
  `config.py` (TOML schema), `grid.py`, `runs.py`, `pipelines/` (stages),
  `core/` (research code: serving, workload construction, forced imbalance,
  trace analysis).
- `packages/moe-reliability-results/` — analysis library, runs anywhere.
- `configs/examples/`, `configs/grids/` — TOML configurations.
- `tests/` — simulated-NPU suite.
- `docs/` — setup, usage, experiments, configuration reference, data format,
  results API. Consult these before inferring behaviour from code alone.
  `imbalance-findings.md` is the 2026-10-02 session's results and caveats.

## Working style

- Reproduce in the local simulated suite first where the bug admits it. Go to
  the node when the failure is genuinely device-, driver- or CANN-specific.
- When a remote command fails, report the actual stderr. Do not paraphrase or
  guess at Ascend error codes — paste what the node said.
- Experiments are expensive and occupy NPUs. Before launching a run or grid,
  validate the config and say what it will cost (`validate`, then `--dry-run`
  for grids).

## Measuring imbalance

Established 2026-10-02; see `docs/imbalance-findings.md` for the numbers.

- **The headline is a null**, on both models tested: DeepSeek-V2-Lite at 2–8 way
  expert parallelism over router bias 0–100 and alpha 0.75–1.44, and Mixtral
  8x7B at 8-way (one expert per rank, MoE 70% of compute) over alpha 0.78–1.43.
  Stragglers do form, but concentrating tokens makes the fused-MoE GEMM enough
  cheaper per call to cancel them. Do not re-run that ground without a reason.
- **Strong injected skew does cost latency** (2026-10-06/07,
  `docs/router-bias-findings.md`). With `imbalance.method = "router_bias"` on
  4 NPUs and the plugin in every arm, 100-token TPOT on Mixtral rises linearly,
  +19.9 ms (+12%) per +1x busiest-rank load, in eager and graph mode alike.
  DeepSeek's cost is large by 3.2-3.5x but its shape below that is unresolved:
  two eager sweeps disagree at 2.34x (+1.0% against +6.5%), by more than either
  one's spread, so do not quote a DeepSeek threshold. Natural traffic reaches 1.09-1.25x
  even per single prompt, where the null above still holds. The *mechanism* is
  not established on either model, and two accounts have already been retracted.
- **The detection screen works at ~1% of routing counted** (2026-10-07,
  `docs/detection-findings.md`). Busiest-rank load over a window of 8 requests,
  counting 4 MoE layers and a tenth of their tokens, flags every skew level that
  costs latency on both models at a 1% false-alarm rate. Window length matters
  more than how much is counted: one request at 100% counted is worse than 8 at
  1.3%, because a single prompt is itself skewed (benign p99 1.41-1.54x). Scoring
  a one-request window is the mistake to avoid.
- **Whether a hot rank persists decides the cost, and the models differ**
  (2026-10-08, `docs/router-bias-findings.md`). With `bias_target = "rotate"`
  (rank = layer mod n_npus) Mixtral lands on the same millisecond as a fixed
  target at matched per-layer load (176.4 against 176.6; 193.2 against 193.9),
  while DeepSeek stays flat (174.8 ms at 3.59x) where a fixed target costs +22%
  (212.6 ms at 3.45x). DeepSeek pays only for a persistently hot rank; Mixtral
  pays per layer. Natural traffic rotates, so this is a second reason DeepSeek's
  null holds and Mixtral's rests only on the skew being small.
- **A rotating skew is invisible to the pooled statistic** (2026-10-08).
  `imbalance.bias_target = "rotate"` biases rank (layer mod n_npus): the per-layer
  load matches a fixed rank target to within 3% (1.71 / 2.40 / 3.53x at offsets
  0.5 / 1 / 2) while the pooled form reads 1.05-1.08x, i.e. benign. It is the
  matched pair for every fixed-target sweep, and the case where EPLB's
  balancedness would report a perfectly balanced server. On Mixtral that skew
  costs full price and a pooled screen flags 5-10% of windows against the
  per-layer screen's 100%.
- **One alarming window is not an alarm.** At a 1% per-window false-alarm rate
  the screen trips on benign traffic every 324-656 requests. Requiring two
  consecutive alarming windows gives 98-100% detection, 0-1% early firing and one
  false alarm per 1200-1750 requests, for 16 requests of delay.
- **Flag layers individually, not just the mean.** With a bias on 6 of 26 layers
  the model-wide mean reads 1.40x while the biased layers read 2.58x; per-layer
  flagging recovers exactly those 6 and names the rank. `detection.localise`.
- **Sample the counted layers randomly, redrawn every window** (2026-10-09).
  Four evenly spaced layers of 26 catch a skew confined to six layers in 1% of
  windows - blind, not merely unable to localise. Four redrawn at random catch
  66% at the same per-window cost, and accumulating across windows names the
  right six (recall 1.00 after 100 windows). On an all-layer skew the two are
  within 0.04. `WindowSampler(layer_choice="random")`, and anything that slices
  the layers once before a stream starts has silently turned it back into fixed
  sampling - `onset_delay` and `mean_windows_to_false_alarm` both did.
- **The screen measures skew, not impact, and the gap is model-specific.**
  DeepSeek's 1%-false-alarm level is 1.25x where nothing costs until 2.43x, so
  it alarms on 94-100% of windows at 1.35x and 1.72x, both free. Mixtral's alarm
  (1.36x) nearly coincides with its onset. An impact claim comes from the
  calibration, never from the routing.
- **Calibrate per deployment, and budget 30 NPU-minutes, not 70**
  (2026-10-09, `scripts/calibration_cost.py`, `docs/figures/tables/calibration_cost.csv`).
  Cutting the 21-point sweep every way: 3 levels {0, 0.5, 1} at 3 repeats
  reproduces the full calibration exactly (onset 1.420x, alarm 1.339x, no false
  alarms) for 9 points. What cannot be cut is **replication** - at 1 repeat the
  onset falls 0.27x and the alarm fires on 15% of benign windows, because the
  noise floor *is* the threshold's definition - and **range**: a sweep that
  stops below the onset either fails to calibrate or misplaces it badly. The
  routing capture, by contrast, is nearly free to shorten: 2% of it (about 40
  requests) gives the same answer as all of it.
- **Take the onset where the crossing holds, not where it first happens.** A
  fine sweep spends several levels near the noise floor and one of them clearing
  it by chance is the likeliest reading there. On the mixtral calibration sweep
  the first crossing sat at 1.24x with the next level back at +0.15%, which read
  as negative headroom and "the screen can never warn early"; requiring every
  higher level to stay above the floor puts it at 1.42x with +0.14x of headroom.
  Coarse sweeps are unaffected. `detection.impact_threshold`.
- **Measure natural rank load per layer.** The validation summary's
  `rank_max_over_mean` takes the busiest rank of layer-averaged shares, which
  lets the hot rank cancel across layers: 1.02x where the per-layer form
  (`rank_max_over_mean_per_layer`, averaged) reads 1.14x. They agree under router
  bias, where one rank is hot in every layer. `scripts/plot_findings.py` uses the
  per-layer form.
- **Set `imbalance.bias_plugin_at_zero` on every router-bias sweep.** Without it
  `server_env` returns `{}` at level 0, so the balanced arm runs without the
  plugin while every other point pays its per-call tensor add. Measured: 10.0 ms
  per token on DeepSeek (+6.1%, eager, where decode is host-bound) and nothing on
  Mixtral. It overstates DeepSeek's level-0-to-2 cost by 43%. The offset is
  collinear with "level > 0", so an affected sweep cannot be fixed by refitting -
  it has to be re-run.
- **Compare like steps, never window averages.** `scripts/step_profile.py` splits
  a trace into engine steps with their batch and token counts. Chunked prefill
  puts prompt tokens in most steps and the mix moves with the swept parameter
  (45% to 87% of token-time), which made window averages disagree with TPOT by
  1.5-3x. A saturated decode step is the same length at every eager offset, and
  the two models put their cost in different kinds of step.
- **Step wall does not reconstruct TPOT** (-45% to +19% against the twin run).
- **Client-side ITLs (`benchmark.save_itl`) are sound in the mean only.** Pooled
  mean sits within 3% of TPOT, but at 3000-way concurrency 62-70% of gaps are
  under 1 ms and arrive in bursts: the percentiles and `itl_ms_spike_share`
  measure the client's event loop, not decode steps. They are meaningful at ~60
  prompts. For distributions use vLLM's own `vllm:inter_token_latency_seconds`.
- **Router-bias offsets are calibrated per model** in the `rbias-calibration`
  runs. DeepSeek collapses to 16 live experts at offset 3 and Mixtral to 3 at
  offset 4. Those levels vary the active expert count, not just skew.
- **Prefill-only latency is frontend-bound.** With 3000 prompts submitted at once
  the engine's queue sits empty 83-90% of the time on DeepSeek (38% on
  Mixtral), so makespan and TTFT measure the client and API server. Check the
  log's `Waiting:` counts before trusting a prefill-only latency. Traces are
  unaffected.
- **Traffic sets the benign baseline, and language moves it** (2026-10-09,
  `docs/figures/tables/corpus.csv`). With nothing injected, a single-locale
  corpus concentrates far more than a mix, which averages domains out: on
  Mixtral, Yoruba reads 1.47x, Swahili 1.35x, Hindi 1.31x against English and
  German at 1.14-1.15x, and the first three sit above its 1.31x onset in 100%,
  98% and 55% of 8-request windows - alarms on ordinary traffic. The same
  corpora on DeepSeek reach 1.38x and 1.32x for Yoruba and Swahili but leave
  Hindi, Chinese and Arabic near English, and nothing approaches its 2.40x. So
  the magnitude is a property of the routing topology, not of the language, and
  a calibration made on English traffic does not describe a deployment serving
  something else.
- **A single prompt is not a window.** Across corpora the p99 of one request is
  1.22-1.55x where the pooled value is 1.07-1.47x; eight requests recover most of
  the way back. Window length matters more than how much of each window is
  counted.
- **To pair router bias with EPLB, target experts, not a rank.** A rank target
  names the experts a rank holds under contiguous placement, which EPLB is free
  to change, so the skew wanders off the rank it was aimed at; the config layer
  refuses that pairing (and `rotate` with it). Rank r holds experts
  [r*n/ranks, (r+1)*n/ranks), so `experts:0,...,15` is the same skew as
  `rank:0` on 64 experts at 4 ranks and survives rearrangement. Note that
  `rank_load` assumes contiguous placement, so an EPLB-on arm's
  `rank_max_over_mean_per_layer` is not valid - recompute it from the records
  against the recorded placement.
- **Alpha is relative to each model's natural CV**, so equal alpha on two models
  is equal *relative* imbalance, not equal rank load. That is why Mixtral's one
  expert per rank behaves like DeepSeek's eight.
- **Set `benchmark.repeats` above 1.** It is the only noise floor. Three single
  points looked like effects this session and dissolved under replication.
- **Summed collective duration is ~95% waiting, not transfer.** `collective_wait`
  splits it: a collective ends for all ranks together, so the last arriver's
  duration bounds the transfer and every other rank's excess is blocked time.
  Treating the 71-78% "communication" in a step decomposition as communication
  cost overstates it by roughly twenty times.
- **The lowest-occupancy rank is the bottleneck, not a victim** (30 of 30 points).
  Ranks that arrive early wait *inside* a collective kernel and so count as busy,
  which inverts the obvious reading. One rank paces each server instance, it is a
  different rank every run, and it is unrelated to alpha.
- **Per-rank totals hide stragglers.** `trace_max_over_mean` sums each rank's
  kernel time, which equalises when the busiest rank differs per layer: it read
  1.004x where per-call pairing read 1.088x. Pair calls across ranks and sum the
  per-call maxima.
- **Workload size confounds alpha.** A token budget lets the request count grow
  with imbalance (237 → 340, 98% collinear). Use
  `workloads.length_in_requests` and `prompt_length_tolerance`.
- **Forced imbalance is not a clean instrument.** The checkpoint recipe
  collapses 64 experts to 6, so it varies active expert count as well as skew.
  Synthetic workloads with `max_repeats` keep every expert live.
- **Treat per-request p-values as meaningless.** One sweep point is one server
  instance, so n = 1 per point, not one per request.
- Points record a `host_before`/`host_after` snapshot and warn when another
  process shares the NPUs. Check it before trusting a comparison: a neighbouring
  job costs ~3% TPOT and ~31% TTFT, and inflated a whole 8-NPU sweep.
- Sweep points are served in seeded random rounds, one per repeat (`benchmark.shuffle_points`,
  on by default), so anything drifting during a run no longer aliases onto the
  swept parameter. The recorded `execution_order` says what ran when. Turn it
  off only to reproduce an older run's ordering.
- `start_vllm_server`'s readiness poll is bounded by `startup_timeout` (1800s)
  and raises rather than hanging on to every NPU.
