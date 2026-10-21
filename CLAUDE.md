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
- When queueing a job behind another tmux session on the node, wait on an
  exact target (`tmux has-session -t =name`). A bare `-t name` falls back to
  prefix matching once that session is gone, so a waiter in a session whose
  name extends it matches itself and waits forever.
- Local `results/` only holds what was last pulled. List the node's
  `results/` (or `./npull`) before saying a run hasn't happened.
- Interrupted runs resume rather than restart:
  `./nrun 'uv run moe-reliability resume <run-id> --retry-failed'`.
  A stage recorded as `completed` is never re-run, so check the manifest before
  assuming a resume will reproduce a missing artefact.

### What can run locally

Anything not in the above category. The local venv is analysis-only: the
results package, numpy, pandas, matplotlib (see `docs/handoff-2026-10-21.md`).
Run scripts with `.venv/bin/python` or `uv run --no-sync`. A plain `uv run` or
`uv sync` here resolves the whole project and starts installing vllm-ascend and
torch-npu.

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
  **DeepSeek has no measured cost of skew** (2026-10-14/15, bullet "Strong
  router bias changes the text" below): its +22% / +31% points are the ones
  whose answers ended early, and with `client.ignore_eos` the skew is free at
  100 and 300 tokens. Do not quote a DeepSeek cost or threshold. Natural traffic reaches 1.09-1.25x
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
  null holds and Mixtral's rests only on the skew being small. **DeepSeek's half
  is withdrawn** (2026-10-15): the fixed arm's 3.45x point wrote 84 of 100
  tokens, the rotating arm 99.3, and with length held a fixed 3.53x is free too,
  so there is no contrast. Mixtral's half stands.
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
- **Sweep the false-alarm target and the layers/tokens split; neither default was
  measured** (2026-10-16, `scripts/detection_params.py`,
  `docs/figures/tables/detection_fpr.csv`, `detection_budget_split.csv`). Every
  costly level is caught in 100% of windows at every per-window target from 0.2%
  to 10%, so the target only moves marginal levels and early firing: **0.5% with
  k = 2** detects in full and roughly doubles the quiet interval against the 1%
  used everywhere before (2544/3120 requests against 1704/1504), at the same
  16-request delay. At a held counting budget, **8 layers at 5% of tokens beats 4
  at 10%** - +0.21 on a six-layer skew, +0.02 on an all-layer one, every costly
  Mixtral level still 1.00 - but it spends Mixtral's early-warning headroom
  (+0.014 against +0.060). The reason the all-layer split collapses is that
  thinning tokens **biases busiest-rank-over-mean upward** (benign median 1.137
  -> 1.459 on Mixtral), so the alarm is in sampling-dependent units the impact
  threshold is not: moving `count_layers` or the token fraction invalidates the
  calibration at no change in cost.
- **Corpus shift alone sets the screen off, at the deployed setting, on traffic
  that costs nothing** (2026-10-16, `natural-long`, 12000 unbiased requests per
  model, scored as served; B2 in `docs/detection-findings.md`). On **Mixtral**,
  whose deployable alarm (1.349x) nearly touches its 1.42x onset, a homogeneous
  **Dolly** stream fires 16.3% of 8-request windows at the calibrated 4x10%
  sampling and confirms every **375 requests** - against a calibration whose own
  false-alarm rate was 0.0 - while its *true* tail counted in full is 1.314x,
  below both the alarm and the onset. Counted in full it fires 0.3%: what crosses
  the bar is sampling inflation, and **an alarm absorbs that only for the traffic
  it was calibrated on** (Dolly's windows are heavier-tailed than the
  calibration's). On **DeepSeek nothing fires at its deployable 1.788x alarm** -
  the highest window reading in 12000 requests is 1.435x - so quote DeepSeek's
  Dolly alarms only against the 1% bar, which sits half a turn of load lower.
  Lengthening the window to 32 requests silences the absolute bar entirely and
  takes a *self-calibrated* one to 53.8% on Dolly, because benign p99 tightens
  (1.147x) while the corpus offset does not (1.190x) - **the main caution on a
  self-calibrating gate**. And k=2 beats its simulation: one confirmed alarm per
  12000 requests on mixed traffic, none on DeepSeek, against a predicted one per
  1200-2544. Every number here is prompt routing; see the next bullet.
- **Prompt and decode routing are two independent skews, and pooling them hides
  both** (2026-10-16, `natural-decode`, DeepSeek, 2496 unbiased requests at 100
  generated tokens; B3 in `docs/detection-findings.md`). Same requests reduced
  three ways: prompt-only **1.0780x**, decode-only **1.0638x**, both
  **1.0517x** - *below either component*, so it is cancellation, not dilution.
  The busiest rank per layer agrees across the two sources on 7 of 26 layers
  (chance is 6.5) and their per-layer per-rank shares correlate **+0.011**. So
  every calibration in `docs/calibrations` (all `max_new_tokens = 1`) describes
  prompt routing alone, and a screen counting a mixed window measures the
  cancelled mixture - the alarm rate on this capture falls to 0.0% almost
  everywhere once decode is in the window, against 0.3-17.3% prompt-only. **Open
  and promising:** the cost of skew lives in prefill-heavy steps and a saturated
  decode step is offset-invariant, so a *prefill-only* statistic may be the one
  that tracks latency - the first lead connecting the screen to the impact
  mechanism.
- **The floor cancels, the signal does not - because our injected skew is
  common-mode** (2026-10-16, `detection-capture-decode`, five levels at 100
  generated tokens with `ignore_eos`; `scripts/decode_signal.py`). At level 0 the
  two sources are independent (10 of 26 layers agreeing, correlation **+0.085**)
  and pooling falls below both, reproducing the bullet above in a second capture.
  At **every** biased level all 26 layers agree and the correlation is +0.94 to
  **+1.000**, so pooling just averages: detection with each source scored against
  its own level-0 alarm is 0.95 prompt / 0.99 decode / **1.00 both** at the
  hardest level (1.35x) and 1.00 everywhere above. So decode makes the screen
  *more* sensitive, not less. **The limit this exposes is the important part:**
  the plugin adds the same offset to every selection call, so an injected skew is
  common-mode by construction - the easiest case for a pooled screen - while
  natural routing is not. A real skew arising in prefill alone, which is where
  its cost lives, would be diluted the way the natural floor is. Every detection
  rate in this project is an upper bound for a skew that is not common-mode, and
  we have never injected one.
- **Detection delay, measured on hardware at last: one window** (2026-10-16,
  `detection-delay`, Mixtral, 3 levels x 2 repeats, `scripts/delay_eval.py`; C7
  in `docs/detection-findings.md`). `imbalance.bias_schedule` moved the bias at
  step 1200 of a running server and the screen counted live. The step-counted
  trigger fired **24 of 24 times at step 1200** across six points and four
  worker processes, so it is rank-consistent on hardware. The skew is caught in
  the **first window that closes after it arrives**, 4 of 4 skewed points: 8
  steps at k=1, 16 at k=2, 24 at k=3. **In windows C4's simulation was exactly
  right; in requests it was optimistic by 2.3x** - a live window at
  `count_window_steps = 8` spans ~18 requests, not the bootstrap's 8, so k=2
  costs 36 requests rather than 16. The control (level 0) fires at k=1 and is
  clean at k=3, which is C5's recommendation with the request cost now measured.
  **And the implementation matters more than the threshold:** counted
  assignments per window ran **128 to 14240** within one point, a 128-assignment
  window reads 1.34-2.96x on benign traffic, and scored as the server writes it
  the alarm lands at 2.13x and **the screen never fires at all**. Dropping
  windows below half a point's median evidence puts it at 1.23-1.28x and
  recovers everything. **Close a window on counted assignments, not on steps.**
- **The screen measures skew, not impact, and the gap is model-specific.**
  DeepSeek's 1%-false-alarm level is 1.25x where nothing costs until 2.43x, so
  it alarms on 94-100% of windows at 1.35x and 1.72x, both free. Mixtral's alarm
  (1.36x) nearly coincides with its onset. An impact claim comes from the
  calibration, never from the routing. (DeepSeek's 2.43x is withdrawn,
  2026-10-15: it was interpolated to a point whose cost was early stops, not
  skew. DeepSeek has no impact threshold below 3.5x at either length.)
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
  the two models put their cost in different kinds of step. The table's `reqs`
  column was one too high until 2026-10-15 (the step kernel's first input shape
  is `query_start_loc`, of length `num_reqs + 1`), so `tokens == reqs` selected
  no decode-only step at all; `plot_findings.py` and `detection_eval.py`
  compensated and their figures stand, but check for `tokens < reqs` rows before
  trusting a direct read of an old CSV.
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
  `rank_max_over_mean_per_layer` is not valid. A map is only recorded in
  record-only mode (below), so for an arm that rearranges use EPLB's own
  per-plan reading in `eplb_activity.hotness`.
- **`eplb_record_map` makes EPLB record-only - never set it on an EPLB-on arm**
  (2026-10-12). vllm-ascend 0.23 applies a placement only when no record path is
  set, so every EPLB run before 2026-10-12 planned and never moved an expert
  (0 completed weight-update cycles; `D2D weight update is on-going` in every
  server). The "EPLB rearranges correctly and buys nothing" result is withdrawn.
  Trust `eplb_activity` on the point (plans, applied cycles, EPLB's own load per
  plan, read from the server log), not the map. Without record mode, EPLB takes
  DeepSeek at offset 2 from 3.49x to 1.01x in about four 50+50 cycles (~80 s).
  The mechanism, for the upstream report: `forward_before` starts the transfer
  and sets the loader to `TRANSFERRING`, but `forward_end` skips
  `update_expert_map_and_weight` when a record path is set - the only thing that
  waits the P2P requests and returns the state to `WAITING`. So the loader wedges
  for the life of the server, the first cycle's `isend`/`irecv` are never waited,
  and the recorded file is a plan that was never applied. **The worker compounds
  it**: `do_update` sets `old_expert_maps = new_expert_maps` after every plan
  whether or not it was applied (`eplb_worker.py:108-109`), so from the second
  cycle on it plans against a layout the server does not have. Both halves are
  fixed on the `fix-eplb-record-mode` branch, and the fix is **measured**
  (2026-10-19, `eplb-record-fix`): with a record path it plans 12/12, applies 0,
  never stalls, and its own busiest-rank reading stays at 1.68-1.83x across all
  twelve plans, while a no-record-path control in the same run still applies
  12/12 and converges 1.68x -> 1.02x. `expert_map_record_path` exists only on
  MRv1 - the V2 "recording" test is about expert *load*, not the map.
- **EPLB rearranges correctly and still buys nothing** (2026-10-12, `eplb-fp2`,
  4 repeats, arms in shared rounds, 7-12 applied cycles per point). It takes
  DeepSeek's 3.63x skew to 1.09x by its own reading and changes TPOT by +1.07%.
  The +31% that skew appeared to cost was early stops: in `fixed-length`
  (length held) it costs -0.27% and EPLB there costs **+3.83% +-1.02**, all four
  pairs positive - a false positive, not a failure to recover. Rearranging where there is
  nothing to win costs at most +4.73% +-4.84 (at 1.73x, free); a single pair
  there read +10.8% and dissolved under replication. The calibrated screen's gate
  is shut on 0-1% of windows at the two free levels and open on 100% at the two
  costly ones, so gating EPLB works. What is *not* measured is EPLB's benefit in
  steady state: at 3.63x convergence took ~85 s of a 127 s batch. The +4.73%
  replicated in TPOT (+5.52% in `eplb-gated`; pooled +5.12%, [+1.7, +8.5]) - see
  the live-gate bullet below.
- **The null holds on Mixtral too, and EPLB cannot even balance it** (2026-10-13,
  `eplb-mixtral`, 3 levels x 2 arms x 4 repeats, 24/24 points, neighbour on NPUs 4-5
  throughout, alike in every arm).
  Recovers nothing at 1.11x / 1.76x / 2.62x (-0.18% / +1.85% / -0.99% TPOT, every
  interval spanning zero), so the DeepSeek null is not an artefact of DeepSeek's
  threshold. It takes 2.62x only to **1.58x**, where on DeepSeek's 64 experts it
  reaches ~1.0x: 8 experts over 4 ranks is 2 per rank, so moving the hot expert
  onto another rank lands it on the second-hottest pair. Coarse MoE bounds the
  best possible plan, before any implementation gets a turn.
- **The cost of skew is a property of the prefill/decode mix, so an impact
  threshold goes stale on output length alone** (2026-10-14). Mixtral at the same
  measured loads, same model, hardware, batch size and bias: 1.76x costs +5.83%
  TPOT at `max_new_tokens=100` (1.72 prompt tokens per generated token) and
  -0.98% at 300 (0.59); 2.62x goes +14.45% to +2.95%. The cost lives in
  prefill-heavy steps - a saturated decode step is the same length at every eager
  offset - and a decode-dominated workload dilutes it ~5x. The two runs' differing
  `gpu_memory_utilization` (0.6/0.85) is not the cause: `max_num_seqs` 512 binds
  in both (the KV caches hold 1630 and 1934 requests of this length), so the
  resident batch is identical and prompts are the same 171 tokens. Both calibrations in
  `docs/calibrations/` were made at 100 tokens, so **both impact thresholds
  (DeepSeek 2.43x, Mixtral 1.42x) are too low for the 300-token EPLB runs**: on
  Mixtral the gate opens on 100% of windows at 1.76x where EPLB loses, and gating
  is 0.3-0.6pp worse than leaving EPLB on. Recalibrate when the traffic mix
  changes, not just the model or the hardware.
- **Strong router bias changes the text, and on DeepSeek that was the whole
  cost** (2026-10-14/15, `fixed-length` 24/24, `length-held-100` 36/36,
  `docs/router-bias-findings.md`). Past ~3x the bias degrades DeepSeek enough
  that some answers end early (205 of 300, 84 of 100 on average), and every
  costly DeepSeek point is one of those. With `client.ignore_eos` the skew is
  free: 300 tokens, 3.64x, +30.86% -> **-0.27% +-1.84**; 100 tokens, 3.53x,
  natural +20.58% against held **-2.13% +-3.91** in the same rounds. It is
  **not the mean length**: every request held at 84 (offset 2's average) is free,
  and natural is +17.70% above it at the same skew and mean - so do not explain
  it as "more prompt per generated token". What differs is the spread: 28% of
  requests stop early at scattered points, and the whole server slows (natural
  requests that ran the full 100 were as slow as the rest: 198.8 against 174.3 ms
  held). Not an averaging artefact (token-weighted 199.2 against 201.1).
  **The spread alone does it** (2026-10-15, `length-replay` 18/18): offset 2's
  per-request lengths replayed at offset 0, no skew, cost **+19.55% +-5.64** TPOT
  against all held at 85 (6 of 6; makespan +8%, TTFT -7%). So uniform-length
  baselines are ~20% optimistic about real traffic, and it is **scheduling, not
  imbalance** (2026-10-15, `length-replay-profiled` 4/4, reproduced at +19.42%
  +-9.72): 63% of the gap is the step mix. A ragged batch frees a slot almost
  every step, so the same prompt tokens are admitted in 2.4x as many chunks
  (1027 prefill tokens per prefill-carrying step against 2457; 71-74% of steps
  carry prefill against 30-35%), and a full-batch step costs nearly the same
  carrying 300 prompt tokens as 4000 (242.5 vs 237.8 ms held) - the overhead is
  per chunk, not per token. GEMM time *falls* while idle and collective time
  rise, and the straggler reading is 1.05x in both arms. Open: whether a coarser
  `max_num_batched_tokens` recovers the 27 ms, and what the remaining 14.9 ms of
  per-step rate is. The graph-mode slope rests on the same shortened point
  and was not re-run. The 300-token run's +2.5% at 2.48x did not recur at 100.
  Set `client.ignore_eos` on any sweep that compares levels, and check
  `output_tokens_mean`; comparisons across arms *within* a level are safe
  (output lengths match to 0.25%). The run's baseline TPOT sits ~8% below
  eplb-fp2's at the same length (longer warm-up suspected, unverified), so its
  absolute levels are not comparable to earlier runs.
- **Score a model's gate with that model's calibration and expert count.**
  `scripts/eplb_false_positive.py` and `plot_findings.eplb_gate` both defaulted to
  DeepSeek's calibration, DeepSeek's captures and 64 experts, and printed a
  plausible gate column for a Mixtral run (1.093x/2.485x/3.634x - DeepSeek's
  loads). Both now select by `model_name` and refuse a mismatch. A gate scored
  with the wrong alarm and the wrong rank-to-expert mapping is not a near miss.
- **EPLB measures what our screen measures, more cheaply, and never asks**
  (2026-10-14, read against vllm-ascend 0.23.0 and `main`; `docs/upstream-eplb.md`).
  Its heat collection reuses the `expert_tokens` group list the fused MoE op
  already produced - one device-side `add_` per layer per step - so it is cheaper
  than the +0.28% `count_experts` screen, and `_compute_imbalance` already
  computes busiest-rank-over-mean *per layer*, the form this project argues for.
  So there is nothing to contribute on the measurement side; the gap is that
  rearrangement fires on an iteration counter with no imbalance condition, in
  vllm-ascend and in vLLM core alike - core does skip below 5% relative
  improvement, but only on ROCm (PR #47206) and on an imbalance pooled across
  layers. `server.eplb_min_imbalance` is that gate,
  as a patch on the installed package (`scripts/apply_node_patch.py`, reverted by
  `uv sync`, inert without its env var so arms can differ inside one run).
  **There is a clock on the two vllm-ascend bugfixes**: RFC #18202 deprecates the
  MRv1 EPLB path before v0.31 and removes it before v0.32, and both bugs are
  MRv1-only. Both fixes are written, tested and committed on branches of a `main`
  clone, unpushed (`docs/upstream-eplb.md`, "The two bugfix branches").
- **To ask "does upstream already do this", enumerate the feature surface, do
  not diff the files you already know** (2026-10-19). We designed, patched and
  measured an imbalance gate for EPLB while vllm-ascend's MRv2 default policy
  already had one. `StairEplbPolicy` gates per layer on
  `relative_balance_threshold = 0.95` (hysteresis against the balance at the
  last commit) and `absolute_balance_threshold = 0.90`, and
  `patch_eplb.py` makes `stair` the MRv2 default. Half the reason we missed it
  is real: it is in **no release we run** - v0.23.0 (2026-08-17) and
  v0.23.0.post1 (2026-09-22) have no `vllm_ascend/distributed/eplb/` at all -
  and landed on `main` in #17511/#17512/#17514 on 2026-09-26/27. The other half
  is ours: it was on `main` for 2.5 weeks before the 2026-10-14 review, and that
  review only diffed the MRv1 files it already cared about, found them identical
  and wrote "confirmed on main". `find vllm_ascend -ipath "*eplb*"` would have
  shown the whole MRv2 tree in one command. The blind spot was self-reinforcing:
  RFC #18202 says MRv2 uses vLLM core's EPLB, so the MRv2 claim was checked
  against *core*, which really does have only the ROCm skip. **Nothing measured
  is affected** - only the contribution framing. The gate's novelty is gone; the
  finding behind it (balance is not impact, and an impact threshold needs
  per-deployment calibration) is still implemented nowhere, stair included,
  whose 0.90 absolute threshold fires at ~1.11x.
- **A balance-improvement threshold cannot see a false positive.** Policies 1 and
  2 return a `changed` flag - set when the plan improves the summed per-layer max
  heat by 5% - and `EplbWorker.do_update` discards it (`_, _, new_placement`),
  along with `per_layer_priority`. Worth an upstream PR, but not a gate: the flag
  asks whether the *plan* is good, not whether the imbalance cost anything, so at
  1.09x a plan that clears the 5% bar still reads `changed` where rearranging is
  pure loss. **How often it would skip, replayed from `eplb_activity.hotness`
  (2026-10-19): 44-73% of plans on DeepSeek, 80-92% on Mixtral** - the earlier
  note that it "reads 1 at every level" came from the first 2-3 plans of each
  run, before EPLB converges. At the measured 1.73x about half the plans would be
  skipped, worth at most ~1.2 of the ~2.3 points the gate averts. **Measured on
  hardware twice** (2026-10-19): through our patch's `server.eplb_honor_changed`
  (`eplb-honour-changed`, declines 5 and 9 of 12 plans, `gate_consistent` true)
  and with the upstream branch installed outright (`eplb-honour-pr`, declines 7
  and 5 of 12; every rank logs its own verdict and the counts are 28 and 20,
  exactly 4x, so all four ranks declined the same plans). It still balances:
  1.676 -> 1.016/1.040x against the control's 1.024x, same trajectory, because
  the plans that matter are the early ones and they all clear the 5% bar. Count
  `gate_shut`/`gate_open`, never `applied_cycles`. Gate on
  the *measured imbalance* against a calibrated impact threshold (1.42x Mixtral;
  DeepSeek's 2.43x is unsupported since 2026-10-14) - skew, not the planner's
  opinion of its own plan.
- **The screen costs under 1% to run, and capture is not its proxy** (2026-10-12/13,
  `screen-counting`). `server.count_experts` counts 4 random layers redrawn per
  window and a tenth of their tokens with a device-side `bincount`, copying out
  once per window, on one rank only: +0.28% +-0.71 e2e on Mixtral against a 0.44%
  baseline spread. `benchmark.capture_experts` costs +6.38% on the same rounds,
  because it returns every token's expert ids in the response body (~41k ids per
  request on DeepSeek). Do not scale the capture number by the sampling fraction
  to price the screen - they are different mechanisms, and that would over-count
  transport the screen never does. DeepSeek is unmeasurable in the prefill-bound
  regime (baseline spread 3.75%, se 4.11%: ~85 repeats to resolve 1%).
- **The screen and the router bias could not both be on, and nobody noticed**
  (2026-10-16). vLLM loads general plugins in entry-point order, which is sorted
  by name, so `moe_expert_counter` always loaded before `moe_router_bias` -
  and router_bias skipped any selection function already carrying `__wrapped__`,
  which the counter sets. **With the screen on, the bias was never applied.**
  Proof on the data: `gate-closure`'s count-on arm recorded a busiest rank of
  1.128x at level 0 and **1.13x at level 0.5**, whose own validation reports
  1.73x. So that run's screen cost (+1.65% / +1.71% on DeepSeek) compared
  unbiased traffic without the plugin against biased traffic with it - two
  differences, neither of them the screen - and is **withdrawn**. The capture
  control (+6.38% / +9.51%) is unaffected, because `benchmark.capture_experts`
  is not a plugin. Mixtral's +0.28% +-0.71 stands only because it was level 0
  and the zero-bias plugin costs Mixtral nothing measurable; it now rests on
  that rather than on the arms being alike. Detection rates are unaffected -
  they come from `validation_save_records` captures, which never set
  `count_experts`. Each plugin now owns its sentinel (`_biasing`,
  `_counting`), with a regression test that loads them in the real order.
- **Run the positive control, and in a quieter regime** (2026-10-13,
  `gate-closure`, 4 arms x 2 levels x 5 repeats, 40/40 points, neighbour throughout). On
  DeepSeek with `client.stream = false` in every arm and eplb-fp2's settings,
  capture reads **+6.38% / +9.51%** e2e (+9.8% / +17.5% makespan) against baseline
  spreads of 3.41% / 2.39%: the control fires, so the regime can see ~6% and the
  screen's +1.65% / +1.71% is a bound rather than a blind reading. Lengthening the
  point from 70 s to 134 s was the fix the config predicted. What the run did
  *not* close is the gating sum: EPLB read +1.52% median e2e at 1.73x - but that
  is e2e with streaming off, not the TPOT the +4.73% was, and in TPOT it did
  replicate (live-gate bullet below).
- **A shut gate is not free: most of EPLB's false positive is EPLB being on**
  (2026-10-14, `eplb-gated`, 3 arms x 3 levels x 4 repeats, 36/36, neighbour on
  NPUs 4-5 throughout, alike in every arm). The `eplb_min_imbalance=2.43` patch
  decides correctly on hardware: shut 48/48 at 1.09x and 1.73x, opens at 3.63x
  (twice per point, then shuts at ~1.73x average - a threshold gate stops at the
  threshold, not at balance), identical on all four ranks. But at 1.73x TPOT,
  EPLB-on costs +5.52% +-3.80 and the gated arm, moving nothing, still +3.23%
  +-0.42 (all 4 pairs +2.7 to +3.7). That +-0.42 is the tightest any run has
  been and did not reproduce: see the four-run pooled ladder below, where the
  gate averts about a third rather than a tenth. One level only (1.09x is too
  noisy to tell). The patch
  gates on EPLB's *worst-layer* reading against a *layer-averaged* calibration -
  worst runs ~0.4x above average - fix the units before sending upstream. Its
  open-at-3.63x arm is no longer a true positive: with length held that level
  is free too (`fixed-length`), so on DeepSeek every gate decision measured has
  been one where the right answer was "shut".
- **The screen's standing cost is near zero on DeepSeek too, and a loaded host
  can hide everything else** (2026-10-19, `screen-vs-eplb`, 5 arms x 2 free
  levels x 5 repeats, 50/50 after a resume; the two first-pass failures were
  another user's job taking NPU 0's memory, and the resume ran them hours later,
  so level 0's round 4 is not like-for-like and the figures below are the n=9
  rounds whose arms shared a node - the tenth moves every arm by <0.7pp).
  All four watchers, in TPOT against `eplb-off` in the same round, pooled over
  both levels: screen -0.47% +-4.89, `eplb-idle` +1.31% +-3.56, gate shut
  +2.76% +-3.74, EPLB on +0.40% +-3.25. **Repeating one arm varies by 2.0-3.8%,
  so the run separates none of them** - the neighbour sat on NPUs 4-7 with
  loadavg 24-40 and never touched ours, but DeepSeek's decode is host-bound, so
  its CPU use lands on our step time anyway. What it does give: the screen
  priced on DeepSeek and in TPOT at last (before, +0.28% rested on Mixtral and
  e2e), and the screen priced in TPOT. **E3 was still open after it**:
  `eplb-idle` (EPLB on, heat interval longer than the point, so it never planned
  or gathered) sat between EPLB-off and the shut gate with both inside its
  interval. Answered as far as it can be by the next bullet - and *not* by a
  quieter node, which was the wrong diagnosis.
- **EPLB's standing cost is a ladder of ~1-point rungs, and our noise is ~4
  points a pair: pool runs, do not wait for a quiet node** (2026-10-20,
  `eplb-standing`, 4 arms x 1 free level (1.73x) x 6 repeats, 24/24, nothing
  resident on NPUs 0-3 in any point, output lengths matched to 0.2%). Within the
  run nothing separates: `eplb-idle` +0.25% +-4.10, gate shut +0.46% +-3.75,
  EPLB on +2.00% +-6.27, every interval spanning zero, 3-4 of 6 pairs positive.
  **Pooled over all four DeepSeek runs at 1.73x, which is the number to quote:
  EPLB on +3.06% [+0.75, +5.38] 14/19 pairs; the shut gate +2.10% [+0.01, +4.20]
  11/15; `eplb-idle` +1.28% [-1.25, +3.81] 7/11.** That is a monotone ladder
  roughly a point per rung - enabling EPLB ~1.3 (not resolved from zero), the
  all-gather + planner + empty loader cycles ~0.8 more, the transfers ~1.0 more -
  so a threshold averts about a third of the cost, not the tenth the single
  `eplb-gated` run implied, and a dormant mode would avert about two thirds.
  **The rungs cannot be separated individually at any sane cost**: per-pair sd is
  3.8-4.8%, so resolving 1 point needs ~70 paired rounds, about 28 NPU-hours per
  arm. State the total and the ladder's shape, never a single rung's value.
- **A neighbour on the other NPUs was not what hid the EPLB arms** (2026-10-20).
  `screen-vs-eplb`'s failure to separate was put down to the neighbour holding
  loadavg at 24-40 while DeepSeek's eager decode is host-bound. `eplb-standing`
  ran at the same loadavg (31-38, 6-8 foreign processes, none ever on our 0-3)
  and saw the same per-arm repeat spread (1.7-4.5%, min-max up to 12.9%), so
  that attribution has no support. **The node has 192 cores**, and the
  neighbour's six-to-eight processes were using about 9 of them - loadavg 22-38
  there is ~11-20% utilisation, not a starved host. Absolute levels do move with
  it (eplb-off's own TPOT rose ~6% in the two rounds where the neighbour grew to
  8 processes), which is why paired rounds are mandatory; but the within-arm
  noise is intrinsic to this measurement, not borrowed from the neighbour. Do
  not spend a window waiting for a loadavg threshold on a 192-core box.
- **The contention check in `screen_overhead.py` was dead until 2026-10-14.** It
  read `p["warnings"]`, which points never carry, so every run it printed was
  "clean". Read `host_before`/`host_after.foreign_npu_processes` (as
  `regime_summary.contended` does). eplb-mixtral, gate-closure, eplb-gated and
  Mixtral screen-counting all ran beside another user's jobs on NPUs 4-5; only
  eplb-fp2 was quiet. Every affected run had the neighbour alike in every arm of
  a round, so paired comparisons stand; absolute levels and noise floors do not
  describe a quiet node.
- **A vLLM general plugin cannot tell the ranks apart at load time.** Plugins load
  in every worker before the process group exists and without `RANK` set, so a
  rank check there answers the same in all of them - `expert_counter` installed
  four counters writing one file before this was fixed. Decide at the first
  forward, when `torch.distributed` is initialised.
- **Compare arms inside one run** (`benchmark.arms`), never as separate grid
  entries. Entries run back to back, so drift aliases onto the arm: on
  2026-10-09 a neighbour grew from one process to eight during the EPLB-on entry
  and read as +10%. Arms share the shuffled rounds.
- **Capture is never streamed, so its control must not be.** A capturing arm
  finished 15% sooner than a streaming one, which is the cost of streaming. Set
  `client.stream = false` on both arms and compare e2e latency and makespan.
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
