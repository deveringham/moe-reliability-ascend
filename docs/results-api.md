# Results API

`moe-reliability-results` (`packages/moe-reliability-results`, import name `moe_reliability_results`) reads results directories written by
`moe-reliability`. It depends only on NumPy, pandas and Matplotlib, and it can be installed on its own
(`uv pip install ./packages/moe-reliability-results`).

## Command line

```bash
moe-reliability-results [--results-dir DIR] COMMAND
```

`--results-dir` defaults to `$MOE_RESULTS_DIR`, or `./results` if that is unset.

| command | description |
|---|---|
| `list [-f KEY=VALUE ...]` | runs with status, model, NPUs, batch size, number of points |
| `show RUN` | stages, configuration, per-point summary and figures of one run (id, unique id prefix or path) |
| `summary [-f KEY=VALUE ...] [-q EXPR] [-c COLS \| --all-columns] [-o FILE]` | one row per sweep point; `-q` takes a pandas query expression; `-o` writes `.csv` or `.json` |
| `requests -o FILE [-f KEY=VALUE ...]` | export all per-request measurements to `.csv` or `.json` |
| `plot RUN [-o DIR] [--format png\|pdf\|svg]` | render the standard figures of a run (default: `<run>/figures`) |

Filter values are parsed as JSON when possible (`-f n_npus=8`, `-f enable_eplb=true`) and compared as
strings otherwise (`-f model_name=deepseek-v2`).

```bash
moe-reliability-results summary -f experiment=forced_imbalance -q "sweep_value == 100 and tpot_ms_p99 > 60" \
    -c run_id,n_npus,batch_size,tpot_ms_mean,tpot_ms_p99,trace_max_over_mean
moe-reliability-results summary -o exports/dataset.csv
moe-reliability-results requests -f model_name=deepseek-v2 -o exports/deepseek_requests.csv
```

## Python

### `ResultsStore`: all runs in a directory

```python
from moe_reliability_results import ResultsStore

store = ResultsStore("results")          # or ResultsStore() for $MOE_RESULTS_DIR / ./results
```

| method | returns |
|---|---|
| `runs(**filters)` | matching `Run` objects, oldest first |
| `get(ref)` | `Run` by id, unique id prefix or directory path |
| `latest(**filters)` | most recent matching `Run` |
| `summary(**filters)` | `DataFrame`, one row per (run, sweep point) |
| `query(expr, **filters)` | `summary()` filtered with `DataFrame.query(expr)` |
| `requests(**filters)` | `DataFrame`, one row per measured request |
| `configurations(**filters)` | `DataFrame`, one row per run with its flattened configuration |
| `infrastructure_configurations(completed_only=True, **filters)` | `DataFrame`, one row per distinct infrastructure configuration (deployment keys + sweep point) with `n_runs` |
| `group_metric(metric, by, agg="mean", **filters)` | `summary()` aggregated by columns |

**Filters** apply to `run_id`, `experiment`, `status` and every scalar configuration key (keys are
unique across sections, so no section prefix is used). A filter value can be

- a scalar, compared for equality: `n_npus=8`;
- a list, tuple or set, tested for membership: `model_name=["deepseek", "qwen"]`;
- a callable predicate: `batch_size=lambda b: b >= 256`.

Runs that lack a filtered key are excluded.

**Summary columns**:

- run identity: `run_id`, `experiment`, `run_status`, `created_at`;
- sweep point: `sweep_parameter`, `sweep_value`, `label`, `point_status`;
- every scalar configuration key: `model_name`, `n_npus`, `batch_size`, `enable_expert_parallel`, ...;
- request statistics: `n_requests`, `ttft_ms_mean|p50|p90|p99|max`, `tpot_ms_*`, `e2e_s_*`, token totals;
- kernel statistics with prefix `trace_`: `trace_max_over_mean`, `trace_mean_over_ranks_us`, ...;
- workload statistics with prefix `workload_`: `workload_mae`, `workload_effective_alpha_median`, ...

**Request columns** are `run_id`, `experiment`, `sweep_parameter`, `sweep_value`, `prompt_id`,
`ttft_s`, `tpot_s`, `total_time_s`, `ttft_ms`, `tpot_ms`, `num_input_tokens` and `num_output_tokens`.
Pass `include_prompts=True` to `Run.requests()` to add the prompts.

```python
# Straggler impact on tail latency, by deployment size
df = store.summary(experiment="forced_imbalance")
df = df[df.point_status == "completed"]
table = df.pivot_table(index=["model_name", "n_npus", "batch_size"], columns="sweep_value",
                       values="tpot_ms_p99")

# Relative slow-down against the balanced point of the same run
base = df[df.sweep_value == 0].set_index("run_id")["tpot_ms_mean"]
df["tpot_slowdown"] = df["tpot_ms_mean"] / df["run_id"].map(base)

# Mean TPOT per alpha across all synthetic workload runs on 8 NPUs
store.group_metric("tpot_ms_mean", by=["batch_size", "sweep_value"],
                   experiment="synthetic_workloads", n_npus=8)
```

Note that `point_status` is a summary column, not a run filter. Select on it in pandas as needed.

### `Run`: one run

| member | description |
|---|---|
| `id`, `experiment`, `status`, `created_at`, `sweep_parameter` | run metadata |
| `manifest`, `config`, `flat_config`, `stages`, `points`, `sweep_values` | raw metadata |
| `point(key)` | point by value (`0.8`) or label (`"alpha_0.8"`) |
| `summary()` | this run's rows of `ResultsStore.summary()` |
| `requests(values=None, include_prompts=False)` | per-request `DataFrame` |
| `request_records(key)` | raw request dictionaries of one point |
| `results_by_point()` | `{sweep_value: [request records]}` |
| `trace_summary(key)`, `trace_summaries()` | kernel metrics per point |
| `hta_frames()` | `{"rank", "idle_categories", "kernel_types", "runs"}` DataFrames |
| `available_workloads()`, `workloads(max_repeats=None)` | workload sets (follows reuse links; NumPy CVs, numeric keys) |
| `activations(limit=None)` | iterator over activation records (expert ids as `int16` arrays) |
| `validation(key)` | router load check of a forced imbalance point |
| `file(relative)` | absolute path of an artefact |

```python
run = store.latest(experiment="synthetic_workloads", model_name="deepseek-v2")
wl = run.workloads(max_repeats=0)
length = wl["target_ls"][0]
mae_by_alpha = {a: w["mae"] for a, w in wl["workloads"][length].items()}

for record in run.activations(limit=100):
    experts = record["routed_experts"]          # int16 array [tokens, layers, top_k]
```

### Visualization

```python
import matplotlib.pyplot as plt
from moe_reliability_results import plots, hta_plots
```

Per run:

| function | input |
|---|---|
| `plots.plot_run(run)` | all standard figures supported by the run, `{name: Figure}` |
| `plots.save_figures(figs, outdir, fmt="png")` | write `{name: Figure}` (nested dictionaries allowed) |
| `plots.plot_workload_sweep(run.workloads(R))` | workload quality: effective vs target alpha, MAE, unique prompts |
| `plots.plot_latency_sweep(run.results_by_point(), sweep_name, axis_label)` | TTFT/TPOT histograms and means vs the sweep value |
| `plots.plot_latency_comparison(records_a, records_b, labels)` | overlaid TTFT/TPOT histograms of two points |
| `plots.plot_trace_sweep(run.trace_summaries(), sweep_name, axis_label)` | straggler ratio and mean kernel time vs the sweep value |
| `plots.plot_kernel_histograms(summary_a, summary_b, labels)` | kernel duration histograms of two points |
| `plots.plot_rank_kernel_means(summary_a, summary_b, use_dom=False, labels)` | per-rank mean kernel time, raw and normalised |
| `plots.plot_expert_load(run.validation(level))` | expert activation frequencies of a checkpoint |

Across runs (forced imbalance, comparing levels 0 and 100 per model and batch size):

```python
K, T = plots.moe_imbalance_overview_inputs(store, n_npus=8)   # kernel and latency inputs
figs = plots.plot_moe_imbalance_overview(K, T, outdir="figures/overview")
```

The inputs are keyed by `(model_name, batch_size, imbalance_level)`. When several runs share a key,
the most recent one wins, so filter down to one deployment (for example `n_npus=8`). Kernel and
latency data may come from different runs (a profiled run and an unprofiled one).