# Usage

All commands are run from the repository root with. `moe-reliability` starts and manages
runs; `moe-reliability-results` reads resulting data (see [results-api.md](results-api.md)).

## Workflow

```
init ─► validate ─► doctor ─► run ─► (resume) ─► list / show / summary / plot
                                ▲
                       grid ────┘  (one run per configuration)
```

Experiment commands (`run`, `resume`, `analyze`, `grid`) run on an Ascend NPU host with CANN and NNAL
activated (see [setup.md](setup.md)). Before any model code runs they check the CANN environment, import
torch_npu, verify that `hardware.n_npus` NPUs are visible (not needed for `analyze`), and verify that the
vLLM Ascend plugin and an intact Triton Ascend installation are present. They also warn about versions
that differ from the validated stack. When a check fails, the command stops before creating a run
directory and exits with code 3.

### 1. Write a configuration

```bash
uv run moe-reliability init synthetic_workloads -o configs/my_alpha_sweep.toml
uv run moe-reliability init forced_imbalance   -o configs/my_imbalance_sweep.toml
```

The generated file lists every key with its default value and a description. Keys that are left out
fall back to their defaults, so a configuration only needs the values that differ (see
`configs/examples/smoke_test.toml`). Every key is described in [configuration.md](configuration.md).

### 2. Validate

```bash
uv run moe-reliability validate configs/my_imbalance_sweep.toml [--show]
```

This checks types, ranges and cross-field consistency, then prints the deployment, the stages and the
sweep points. `--show` also prints the fully resolved configuration. Nothing is loaded or started.

### 3. Check the NPU environment

```bash
uv run moe-reliability doctor configs/my_imbalance_sweep.toml
```

`doctor` reports every environment check with the fix for each failure (see [setup.md](setup.md#3-verify)).

### 4. Run

```bash
uv run moe-reliability run configs/my_imbalance_sweep.toml \
    --set hardware.n_npus=4 \
    --set imbalance.imbalance_levels=[0,25,50,100] \
    --results-dir /data/moe-reliability-results
```

| option | effect |
|---|---|
| `--set SECTION.KEY=VALUE` | override any value. The value uses TOML syntax, and strings may be unquoted (repeatable) |
| `--results-dir DIR` | shortcut for `--set output.results_dir="DIR"` |
| `--run-id ID` | explicit run id instead of `<UTC timestamp>_<type>_<model>_npu<N>_bs<B>[_<name>]` |
| `--dry-run` | validate and print the plan only |
| `--no-log-file` | do not mirror console output into the run directory |

The run directory is created immediately. It contains the resolved `config.toml` and a `manifest.json`
that tracks the status of every stage and sweep point, and it is updated after every step. All console
output, including the vLLM server's, is mirrored into `logs/run.log`.

Exit codes: `0` completed, `1` failed or partially failed, `2` invalid configuration or run reference,
`3` Ascend environment not usable, `130` interrupted.

### 5. Resume

```bash
uv run moe-reliability resume <run-id-or-prefix> [--retry-failed]
```

A run stopped by an error, a crash or Ctrl-C continues with its stored configuration:

- completed stages are skipped;
- in the benchmark stage, completed sweep points are skipped, and points that were running when the
  run stopped are measured again (their partial traces are discarded);
- failed points are only retried with `--retry-failed`;
- workload sets that were already written are not rebuilt.

A run id prefix is enough if it is unique, for example `resume 20260916-1432`.

### 6. Re-analyse

```bash
uv run moe-reliability analyze <run>
```

This re-runs NPU profiler parsing, trace analysis, and figure rendering of an existing run without
touching its measurements. It does not need free NPUs. Useful if when the analysis code changes.

### 7. Inspect

```bash
uv run moe-reliability list                       # all runs (same as: moe-reliability-results list)
uv run moe-reliability show <run>                 # stages, configuration, per-point summary
uv run moe-reliability-results summary -f experiment=synthetic_workloads -f n_npus=8
uv run moe-reliability-results plot <run> --format pdf
```

## Run lifecycle and statuses

| run status | meaning |
|---|---|
| `pending` / `running` | created / in progress (or the process was killed) |
| `completed` | every stage completed or was skipped by configuration, and every point succeeded |
| `partial` | the run finished, but at least one sweep point failed |
| `failed` | a stage raised an error; the manifest holds the traceback |
| `interrupted` | stopped with Ctrl-C |

Stages are `pending`, `running`, `completed`, `skipped` (with a `note`), `failed` (with an `error`) or
`interrupted`. Figure rendering never fails a run: a rendering error marks the `figures` stage as
skipped with the error as note, and `moe-reliability-results plot <run>` can render the figures later.

## Reusing activations and workloads

Routed-expert capture and workload construction are the expensive, deployment-independent part of a
synthetic workload run. Benchmark the same workloads on another deployment with:

```bash
uv run moe-reliability run configs/examples/synthetic_workloads.toml \
    --set workloads.reuse_workloads_from=<run-id> \
    --set hardware.n_npus=4 --set server.batch_size=256
```

`activations.reuse_activations_from` reuses only the activation records and builds new workloads (for
example with other alphas). Reused artefacts stay in the source run. The new run stores a link in
`manifest.inputs`, and the results library follows it transparently.

## Grids

A grid file expands a base configuration over a parameter matrix:

```toml
[grid]
name = "fi-infra"                            # required; runs are named fi-infra-000, fi-infra-001, ...
base = "../examples/forced_imbalance.toml"   # relative to the grid file
share_workloads = false                      # synthetic workloads: capture and build once per model

[set]                                        # applied to every configuration
"benchmark.n_samples" = 2000

[matrix]                                     # Cartesian product, in file order
"hardware.n_npus" = [2, 4, 8]
"server.batch_size" = [128, 256, 512]
"model.model_id,model.model_name" = [        # comma-separated keys vary together
    ["deepseek-ai/DeepSeek-V2-Lite-Chat", "deepseek"],
    ["Qwen/Qwen1.5-MoE-A2.7B-Chat", "qwen"],
]
```

```bash
uv run moe-reliability grid configs/grids/forced_imbalance_infrastructure.toml --dry-run
uv run moe-reliability grid configs/grids/forced_imbalance_infrastructure.toml --write-configs expanded/
uv run moe-reliability grid configs/grids/forced_imbalance_infrastructure.toml --continue-on-error
```

- All expanded configurations are validated before the first run starts.
- `--set` overrides apply to every configuration.
- Re-running a grid skips runs that are already completed (matched by experiment type and
  `experiment.name`) and resumes unfinished ones. An interrupted grid therefore continues with the
  same command.
- Without `--continue-on-error`, the grid stops at the first run that does not complete.
- With `share_workloads = true`, the first synthetic workload run of a model captures activations and
  builds workloads, and later runs whose capture settings match reuse them. The matching settings are
  model, quantization, seed, `max_new_tokens`, `n_samples`, alphas, prompt lengths and `max_repeats`.
- Every run records its grid name, index and matrix assignments in `manifest.grid`.

Count what a results directory covers:

```python
from moe_reliability_results import ResultsStore
configs = ResultsStore("results").infrastructure_configurations()
print(len(configs))
```

## Logs

`<run>/logs/run.log` receives everything printed to the terminal during `run`, `resume`, `analyze` and
`grid` (appending across resumptions): stage progress, per-point latency summaries, vLLM server
start-up and shutdown, and tracebacks. When a sweep point fails, its `error` field in the manifest
points to this log.

## Troubleshooting

| symptom | cause and fix |
|---|---|
| `error: Ascend environment: the CANN environment is not activated` | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` and `source /usr/local/Ascend/nnal/atb/set_env.sh` in the shell that runs the command |
| `torch_npu could not be imported` | the environment is not installed (`uv sync`), or CANN libraries are not on the library path (activate CANN) |
| `hardware.n_npus = N but only M NPU(s) are visible` | check `npu-smi info`, the NPUs selected by `hardware.visible_devices` / `ASCEND_RT_VISIBLE_DEVICES`, and processes still holding NPUs |
| `Triton Ascend file(s) were overwritten` | `uv sync --reinstall-package triton-ascend` |
| `warning: software stack differs from the validated vLLM Ascend set` | the environment was modified outside the lock file; `uv sync --locked` restores it |
| `libatb.so` not found when the server starts | NNAL is not installed or not activated: `source /usr/local/Ascend/nnal/atb/set_env.sh` |
| `vLLM server process terminated unexpectedly` in the log | the server failed to start: not enough NPU memory (lower `server.gpu_memory_utilization`, `server.max_model_len` or `server.batch_size`), a tensor-parallel size the model does not support, a model not supported by vLLM Ascend, or a port already in use (`server.port`) |
| HCCL timeouts in the log with many NPUs | raise `HCCL_CONNECT_TIMEOUT` (and `HCCL_EXEC_TIMEOUT`) in the `[environment]` table |
| a point fails immediately after a previous point | a previous server still holds the port or NPU memory; check `npu-smi info` for stray processes |
| `trace_parse_error` on a point | torch_npu could not parse the raw profiler data, for example when the profiled window ended before any iteration was recorded; increase `benchmark.n_samples` or the workload size |
| `trace_error: no rank traces found` on a point | the fused-MoE kernel analysis found no `*rank*.pt.trace.json.gz` files in the trace directory (see [experiments.md](experiments.md#trace-analysis)) |
| `model.probe = 'auto' cannot infer the router family` | set `model.probe` to `deepseek`, `qwen` or `mistral` for local checkpoint paths |
| gated model download fails | export `HF_TOKEN` and accept the model terms on Hugging Face |
