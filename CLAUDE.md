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
- **Alpha is relative to each model's natural CV**, so equal alpha on two models
  is equal *relative* imbalance, not equal rank load. That is why Mixtral's one
  expert per rank behaves like DeepSeek's eight.
- **Set `benchmark.repeats` above 1.** It is the only noise floor. Three single
  points looked like effects this session and dissolved under replication.
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
- Sweep points are served in a seeded random order (`benchmark.shuffle_points`,
  on by default), so anything drifting during a run no longer aliases onto the
  swept parameter. The recorded `execution_order` says what ran when. Turn it
  off only to reproduce an older run's ordering.
- `start_vllm_server`'s readiness poll is bounded by `startup_timeout` (1800s)
  and raises rather than hanging on to every NPU.
