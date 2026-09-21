# Data format

All results live in a results directory (`output.results_dir`, default `./results`) with one
sub-directory per run. Every artefact is strict JSON (UTF-8, no `NaN`; non-finite numbers are stored
as `null`). With `output.compress = true` (default), measurement files are gzip-compressed and carry
an additional `.gz` suffix. The manifest references files by their uncompressed name, and readers
accept both variants. The format is versioned by `manifest.schema_version` (currently `1`).

## Run directory

```
<results_dir>/<run_id>/
  manifest.json                          run metadata, resolved configuration, stage and point status
  config.toml                            resolved configuration (can be passed to `moe-reliability run`)
  logs/run.log                           console output including the vLLM server
  activations/records.jsonl[.gz]         synthetic workloads: routed-expert capture records
  workloads/workloads_repeats<R>.json[.gz]  synthetic workloads: one workload set per max_repeats value
  validation/imbalance_<level>.json[.gz]    forced imbalance: router load check per checkpoint
  metrics/<label>.json[.gz]              per-request measurements of one sweep point
  traces/<label>/                        NPU profiler data of every worker, parsed into
                                         <worker>/ASCEND_PROFILER_OUTPUT/trace_view.json
  trace_metrics/<label>.json[.gz]        fused-MoE kernel metrics extracted from the traces
  figures/*.png                          standard figures
```

`<run_id>` defaults to `<UTC yyyymmdd-HHMMSS>_<synthetic|imbalance>_<model_name>_npu<n_npus>_bs<batch_size>[_<experiment.name>]`.
`<label>` identifies a sweep point: `alpha_<value>` (for example `alpha_0.8`) or
`imbalance_<level>` (for example `imbalance_100`).

## `manifest.json`

| field | type | description |
|---|---|---|
| `schema_version` | int | format version |
| `run_id` | str | directory name |
| `experiment` | str | `synthetic_workloads` or `forced_imbalance` |
| `sweep_parameter` | str | `alpha` or `imbalance_level` |
| `status` | str | `pending`, `running`, `completed`, `partial`, `failed`, `interrupted` |
| `created_at`, `updated_at`, `finished_at` | str | ISO 8601 UTC timestamps |
| `resumed_at` | list[str] | timestamps of `resume` invocations |
| `description` | str | `experiment.description` |
| `error` | str | final error message if the run failed |
| `config` | object | fully resolved configuration, `{section: {key: value}}` |
| `environment` | object | provenance: `hostname`, `os`, `machine`, `python`, `command`, `packages` (versions of vLLM, vLLM Ascend, torch, torch_npu, Triton Ascend, Transformers, ...), `git_commit`, `git_dirty`, `ascend` (see below) |
| `inputs` | object | reused artefacts: `{"activations" \| "workloads": {"run_id", "path"}}` |
| `stages` | object | `{stage: {"status", "started_at", "finished_at", "error", "note"}}` |
| `points` | list | sweep points, see below |
| `activations` | object | `{"file", "n_records", "model_id", "n_samples", "seed", "max_new_tokens"}` |
| `workloads` | object | `{"<max_repeats>": {"file", "max_repeats", "target_alphas", "target_prompt_lengths", "target_ls"}}` |
| `figures` | list[str] | rendered figure paths |
| `log_file` | str | `logs/run.log` |
| `grid` | object | for grid runs: `{"name", "index", "assignments", "file"}` |

### NPU provenance (`manifest.environment.ascend`)

| field | description |
|---|---|
| `cann_toolkit_home`, `cann_version` | activated CANN toolkit and its version |
| `driver_version` | Ascend driver version (`/usr/local/Ascend/driver/version.info`) |
| `atb_home` | activated NNAL/ATB installation |
| `visible_devices` | `ASCEND_RT_VISIBLE_DEVICES` of the run |
| `npu_count`, `npu_name` | NPUs visible to the run and their model |
| `stack_mismatches` | packages whose versions differ from the validated vLLM Ascend set |
| `npu_smi` | output of `npu-smi info` (lines) |

### Sweep points (`manifest.points[]`)

| field | description |
|---|---|
| `index`, `value`, `label` | position, sweep value (alpha or imbalance level) and label |
| `status` | `pending`, `running`, `completed`, `failed` |
| `started_at`, `finished_at`, `error` | benchmarking timestamps and failure message |
| `model_path` | model id or checkpoint path that was served |
| `n_prompts` | number of measured requests sent |
| `metrics_file` | per-request measurements (`null` if `benchmark.save_request_metrics = false`) |
| `request_summary` | request statistics, always stored (see below) |
| `trace_dir` | NPU profiler data (`null` without profiling) |
| `npu_trace_views`, `trace_parse_error` | timelines produced by the offline NPU profiler parser and its error |
| `trace_metrics_file`, `trace_summary`, `trace_error` | extracted kernel metrics, their scalar subset and extraction error |
| `workload` | synthetic workloads: `n_prompts`, `mae`, `percent_unique_prompts`, `effective_alpha_p10/median/p90`, `max_repeats`, `target_prompt_length`, `target_tokens` |
| `checkpoint_created` | forced imbalance: whether this run generated the checkpoint |
| `validation_file`, `validation_summary` | forced imbalance: router load check and `{n_assignments, max_expert_frequency, expected_frequency}` |

`request_summary` fields: `n_requests`, `n_timed_requests`, `input_tokens_total`,
`output_tokens_total`, `input_tokens_mean`, `output_tokens_mean`, and `mean`, `p50`, `p90`, `p99` and
`max` of `ttft_ms`, `tpot_ms` and `e2e_s` (for example `tpot_ms_p99`).

`trace_summary` fields: `mean_over_ranks_us`, `max_over_mean`, `max_over_min`, `hottest_rank`,
`total_over_ranks_ms`, `calls_per_rank`, `steps`, `dom_mean_over_ranks_us`, `dom_max_over_mean`.

## `metrics/<label>.json`

```json
{
  "run_id": "...", "label": "imbalance_100", "sweep_parameter": "imbalance_level", "sweep_value": 100,
  "model_path": "models/mistral-imbalance100", "profiled": false,
  "requests": [
    {"prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
     "prompt_id": 0, "ttft": 0.0412, "tpot": 0.0187,
     "num_output_tokens": 100, "num_input_tokens": 143, "total_time": 1.94}
  ]
}
```

Times are in seconds. `prompt_id` is the index of the prompt in the list sent to the server.

## `activations/records.jsonl`

One JSON object per line, one line per captured prompt:

| field | description |
|---|---|
| `prompt` | chat messages sent to the server |
| `prompt_id` | index into the MMLU prompt list |
| `subject` | MMLU subject |
| `routed_experts` | `[generated_tokens][layers][top_k]` expert ids for generated tokens |
| `prompt_routed_experts` | `[prompt_tokens][layers][top_k]` expert ids for prompt tokens |
| `num_input_tokens`, `num_output_tokens`, `total_time` | usage and latency of the capture request |
| `ttft`, `tpot` | always `null` (timings are not valid during capture) |

`Run.activations()` returns the expert id arrays as `numpy.int16` arrays.

## `workloads/workloads_repeats<R>.json`

| field | description |
|---|---|
| `run_id`, `activations_run_id` | producing run and the run holding the activation records |
| `model_id`, `model_name`, `seed`, `max_repeats` | provenance |
| `n_experts`, `n_layers`, `k` | MoE dimensions used |
| `n_prompts`, `n_total_tokens`, `avg_tokens_per_prompt` | activation record statistics |
| `cv_nat` | natural per-layer CV of expert load, `[n_layers]` |
| `target_alphas`, `target_prompt_lengths`, `target_ls` | sweep definition; `target_ls[i]` is the token budget for `target_prompt_lengths[i]` |
| `workloads` | `{"<token budget>": {"<alpha>": workload}}` |

Each workload: `indices` (selected activation record indices, in selection order), `prompts` and
`prompts_formatted` (the corresponding chat prompts), `obtained_cvs` (`[n_layers]`), `mae`,
`percent_unique_prompts` (0-1). `Run.workloads()` converts keys to `int`/`float` and CVs to NumPy
arrays.

## `trace_metrics/<label>.json`

Output of the fused-MoE kernel analysis (field definitions in [experiments.md](experiments.md#trace-analysis)):
`trace_dir` (relative to the run), `ranks`, `per_rank_mean_us`, `per_rank_durs`,
`mean_over_ranks_us`, `max_over_mean`, `max_over_min`, `hottest_rank`, `total_over_ranks_ms`,
`calls_per_rank`, `steps`, `dominant_grid`, `dom_per_rank_mean_us`, `dom_mean_over_ranks_us`,
`dom_max_over_mean`.

## `validation/imbalance_<level>.json`

| field | description |
|---|---|
| `imbalance_level`, `model_path` | checkpoint |
| `router_id`, `n_experts`, `n_routers`, `k` | probe dimensions (`router_id` = 0) |
| `counts`, `frequencies`, `n_assignments` | expert selections of router 0: counts, relative frequencies, total |
| `per_router_frequencies` | `[n_routers][n_experts]` relative frequencies |
| `prompts`, `responses` | validation prompts and generated responses |

## Reading without the library

```python
import gzip, json
from pathlib import Path

def read(path):
    path = Path(path)
    if not path.exists():
        path = path.with_name(path.name + ".gz")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)

run = Path("results/<run_id>")
manifest = read(run / "manifest.json")
for point in manifest["points"]:
    if point.get("metrics_file"):
        requests = read(run / point["metrics_file"])["requests"]
```
