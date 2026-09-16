# Experiments

Two experiment types are defined which each measure how expert load imbalance degrades MoE inference on a
multi-NPU vLLM Ascend deployment. They differ in how the imbalance is produced: by the **workload** (which prompts are
sent) or by the **model** (an artificially biased router).

## Deployment and measurement (both experiments)

Each sweep point starts a fresh vLLM OpenAI-compatible server on Ascend NPUs (`core/vllm_serving.py`).
The vLLM Ascend plugin registers itself with vLLM, and the server inherits the NPU selection
(`hardware.visible_devices`) and the `[environment]` variables. Collective communication between NPUs
uses HCCL.

| setting | value |
|---|---|
| parallelism | `--tensor-parallel-size hardware.n_npus`, `--data-parallel-size 1`, `--enable-expert-parallel` if `server.enable_expert_parallel` |
| batching | `--max-num-seqs server.batch_size`, `--max-num-batched-tokens 4096` |
| determinism | `--seed experiment.seed`, generation temperature 0, `--enforce-eager`, `--no-async-scheduling` |
| memory | `--gpu-memory-utilization server.gpu_memory_utilization` (fraction of each NPU's memory) |
| options | `--enable-eplb` (`server.enable_eplb`), `--enable-prefix-caching`/`--no-enable-prefix-caching` |
| profiling | vLLM's `torch` profiler, which vLLM Ascend records with `torch_npu.profiler` on every worker, starting after 100 scheduler iterations, for `benchmark.trace_active_iterations` iterations |

The client sends all prompts concurrently as chat completions, limited to `client.concurrency_limit`
in-flight requests. This limit should exceed the batch size so that the server stays saturated.
`client.n_warmup_samples` requests are sent first and not measured. Every measured request is streamed
and yields:

| metric | definition |
|---|---|
| `ttft` | seconds from sending the request to the first streamed chunk |
| `tpot` | (end time - first token time) / (output tokens - 1), in seconds |
| `total_time` | end-to-end request latency in seconds |
| `num_input_tokens`, `num_output_tokens` | token usage reported by the server |

Profiling perturbs timings. For that reason, runs with `benchmark.enable_profiling = true` do not
store per-request metrics by default (`benchmark.save_request_metrics`), but they still record a
summary of them.

A server that fails to start or a failed inference marks only that sweep point as failed. The sweep
continues, and the point can be retried with `moe-reliability resume <run> --retry-failed`.

## Synthetic workloads

Sweep parameter: `alpha`. Stages: `activations -> workloads -> benchmark -> trace_analysis -> hta -> figures`.

### 1. Routed-expert capture

`activations.n_samples` MMLU questions (all subjects, shuffled with `experiment.seed`) are formatted
as chat prompts and served with `--enable-return-routed-experts`. For every prompt, the experts chosen
for each prompt token and each generated token in every MoE layer are stored
(`routed_experts: [generated_tokens, layers, top_k]`,
`prompt_routed_experts: [prompt_tokens, layers, top_k]`). Timings are not recorded in this stage.

### 2. Workload construction

1. **Per-prompt load.** For each prompt `i`, `q_i[e, l]` is the number of times expert `e` is
   selected in layer `l` over all tokens, divided by `top_k`.
2. **Natural imbalance.** `cv_nat[l]` is the coefficient of variation (std/mean over experts) of the
   mean load `mean_i q_i[:, l]`.
3. **Greedy selection.** We control the targeted imbalance for each workload by a parameter alpha.
   For each target alpha and workload length, prompts are added one at a time.
   At every step the prompt chosen is the one that minimises the squared distance between the
   workload's per-layer CV and the target `alpha x cv_nat`. A prompt may be selected at most
   `max_repeats + 1` times. Selection stops once the workload reaches the token budget
   `L = avg_tokens_per_prompt x target_prompt_length`.
5. **Quality.** The mean absolute error between the obtained and target CVs (`mae`), the share of
   unique prompts, and the *effective alpha* (obtained CV / `cv_nat`; median and 10-90 % range over
   layers) are stored per workload.

One workload set is built per value in `workloads.max_repeats`. Every set contains all alphas and all
target prompt lengths.

### 3. Benchmarking

The workload set selected by `benchmark.workload_max_repeats` and `benchmark.workload_prompt_length`
is replayed. Each alpha is one sweep point.

Activation capture and workload construction do not depend on the deployment. Later runs can reuse
them with `activations.reuse_activations_from` or `workloads.reuse_workloads_from`, and grids can
share them automatically with `share_workloads = true`.

## Forced imbalance

Sweep parameter: `imbalance_level`. Stages: `checkpoints -> validation -> benchmark -> trace_analysis -> hta -> figures`.

### 1. Checkpoints

`core/forced_imbalance.py` loads the model (float16) and, for a level `b > 0`:

1. adds 1.0 to all token embedding weights, so that hidden-state sums are positive;
2. for every module whose name contains `gate`, zero-centres each row of its weight matrix and adds
   `b` to row 0, which raises the router logit of expert 0.

The checkpoint, configuration and tokenizer are saved to `imbalance.model_dir/<model_name>-imbalance<b>`
and reused by later runs. Level 0 serves the original `model.model_id`.

### 2. Validation (optional)

With `imbalance.validate_imbalance = true`, six fixed prompts are generated with Hugging Face
Transformers while a router probe records the selected experts. The expert activation frequencies of
router 0 (plus the frequencies of all routers) are stored and plotted against the uniform expectation
`1 / n_experts`. The Hugging Face models run on NPUs through `torch_npu.contrib.transfer_to_npu`.

### 3. Benchmarking

The same `benchmark.n_samples` MMLU prompts are replayed against every level.

## Trace analysis

With `benchmark.enable_profiling = true`, every vLLM Ascend worker records NPU profiler data into
`traces/<label>/`.

Torch_npu's offline parser (`torch_npu.profiler.profiler.analyse`) is runon the trace directory.
If the directory as a whole yields no output, it is run on each worker directory. The parser 
writes a timeline per worker, `ASCEND_PROFILER_OUTPUT/trace_view.json`, which can be opened with
MindStudio Insight or a Chrome trace viewer. The files are listed in the point's `npu_trace_views`,
and failures are recorded in `trace_parse_error`. Already parsed data is not parsed again.

| field | meaning |
|---|---|
| `per_rank_mean_us` | mean fused-MoE kernel duration per rank (microseconds) |
| `per_rank_durs` | all kernel durations per rank |
| `mean_over_ranks_us` | mean of the per-rank means |
| `max_over_mean` | slowest rank mean / mean over ranks (straggler ratio, 1.0 = balanced) |
| `max_over_min` | slowest rank mean / fastest rank mean |
| `hottest_rank` | rank with the largest mean |
| `total_over_ranks_ms` | total kernel time over all ranks (milliseconds) |
| `calls_per_rank`, `steps` | kernel calls and scheduler steps on rank 0 |
| `dominant_grid`, `dom_*` | the same statistics restricted to the kernel launch grid shared by all ranks with the most calls, which holds the batch shape constant |

Table keys are `model` (`model.model_name`), `batch` (`server.batch_size`) and `imbalance` (the sweep
value).

## Figures

At the end of a run, `figures/` receives the standard figures that the stored data supports:

- workload quality per workload set (effective vs target alpha, MAE, unique prompts);
- TTFT/TPOT histograms and mean TTFT/TPOT against the sweep value;
- lowest vs highest imbalance level latency comparison (forced imbalance);
- straggler ratio and mean kernel time against the sweep value, kernel duration histograms and
  per-rank means (profiled runs);
- expert activation frequencies (validated checkpoints).

Cross-run figures (per-model overview of MoE time per rank and HTA comparisons across models and
batch sizes) are produced with the results library, see [results-api.md](results-api.md).
