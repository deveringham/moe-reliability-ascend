# moe-reliability

Framework for reliability investigation of Mixture-of-Experts (MoE) inference under expert load imbalance.

This repository benchmarks MoE language models served with [vLLM](https://github.com/vllm-project/vllm)
and the [vLLM Ascend](https://github.com/vllm-project/vllm-ascend) plugin on Huawei Ascend NPU
deployments while injecting controlled expert load imbalance. Under expert parallelism an overloaded
expert potentially turns the NPU hosting it into a straggler, so imbalance can acts as a reproducible,
tunable slow-down of the serving infrastructure. Every run records end-to-end request latency, NPU
profiler traces and all intermediate artefacts as JSON, together with the full configuration and
software and NPU provenance.

The experiment runtime runs exclusively on Ascend NPUs (`torch_npu` + vLLM Ascend). Stored results can
be analysed anywhere with the lightweight results library.

It consists of two Python packages:

| package | purpose | dependencies |
|---|---|---|
| `moe-reliability` (`src/moe_reliability`) | runs experiments on Ascend NPUs from TOML configurations via the `moe-reliability` CLI | vLLM Ascend, torch_npu, Transformers |
| `moe-reliability-results` (`packages/moe-reliability-results`) | queries, aggregates and visualizes stored results via Python or the `moe-results` CLI | NumPy, pandas, Matplotlib |

## Experiments

**Synthetic workloads** (`synthetic_workloads`): the model is served with routed-expert capture to
record which experts every MMLU prompt activates in every layer. From these records, prompt subsets
are selected greedily so that the per-layer coefficient of variation (CV) of expert load equals
`alpha x` the natural CV of the data. Each workload is then replayed against a fresh deployment,
giving one measurement per imbalance level `alpha`.

**Forced imbalance** (`forced_imbalance`): checkpoints are derived from the model by adding a bias
towards expert 0 to every router, which concentrates routing on a single expert. Every imbalance
level is served and measured on the same MMLU prompts; level 0 is the unmodified model.

Both experiments report per-request time to first token (TTFT), time per output token (TPOT) and
end-to-end latency. With profiling enabled, NPU profiler traces of every worker are recorded and
converted offline into timeline files, followed by the per-rank kernel analysis.
See [docs/experiments.md](docs/experiments.md).

## Installation

Requirements: an Ascend NPU host on Linux aarch64 or x86_64 with the Ascend driver 
and firmware (HDK 26.0.RC1), CANN Toolkit, Ops and NNAL 9.1.0, and
[uv](https://docs.astral.sh/uv/). See [docs/setup.md](docs/setup.md).

```bash
git clone https://github.com/deveringham/moe-reliability-ascend.git
cd moe-reliability-ascend
source /usr/local/Ascend/ascend-toolkit/set_env.sh     # activate CANN
source /usr/local/Ascend/nnal/atb/set_env.sh           # activate NNAL
uv venv
uv sync                                                # vLLM Ascend 0.23.0 stack (torch_npu, triton-ascend, ...)
export HF_TOKEN=...                                    # for gated Hugging Face models
uv run moe-reliability doctor                          # verify CANN, torch_npu, NPUs and versions
```

`uv run` also installs the environment on first use. Analysis-only machines (no NPUs) install just
the results library: `uv pip install ./packages/moe-reliability-results`.

## Quick start

```bash
# 1. Create a configuration (or copy one from configs/examples/)
uv run moe-reliability init forced_imbalance -o my_run.toml

# 2. Check it, see what will run and verify the NPUs it needs
uv run moe-reliability validate my_run.toml
uv run moe-reliability doctor my_run.toml

# 3. Run it (values can be overridden without editing the file)
uv run moe-reliability run my_run.toml --set hardware.n_npus=4 --set server.batch_size=256

# 4. Inspect results
uv run moe-reliability list
uv run moe-reliability-results summary --filter experiment=forced_imbalance
uv run moe-reliability-results plot <run-id>
```

Interrupted or partially failed runs continue where they stopped:

```bash
uv run moe-reliability resume <run-id> [--retry-failed]
```

Query results from Python:

```python
from moe_reliability_results import ResultsStore, plots

store = ResultsStore("results")
table = store.summary(experiment="forced_imbalance", n_npus=8)       # one row per sweep point
slow = store.query("sweep_value >= 50 and tpot_ms_p99 > 80")          # pandas query syntax
requests = store.requests(model_name="deepseek-v2", batch_size=256)   # per-request measurements
store.infrastructure_configurations()                                 # distinct configurations measured

run = store.latest(experiment="synthetic_workloads")
figures = plots.plot_run(run)
```

## Datasets over many infrastructure configurations

A *grid* runs one experiment per point of a parameter matrix. The shipped grids cover more than 100
infrastructure configurations each (deployment parameters x imbalance level):

| grid | runs | configurations |
|---|---|---|
| [`configs/grids/forced_imbalance_infrastructure.toml`](configs/grids/forced_imbalance_infrastructure.toml) | 30 (3 NPU counts x 5 batch sizes x expert parallelism on/off) | 120 (x 4 imbalance levels) |
| [`configs/grids/synthetic_workloads_infrastructure.toml`](configs/grids/synthetic_workloads_infrastructure.toml) | 9 (3 NPU counts x 3 batch sizes) | 126 (x 14 alphas) |

```bash
uv run moe-reliability grid configs/grids/forced_imbalance_infrastructure.toml --dry-run
uv run moe-reliability grid configs/grids/forced_imbalance_infrastructure.toml --continue-on-error
```

Grid runs are idempotent: repeating the command skips completed runs and resumes unfinished ones.

## Repository layout

```
configs/
  examples/             ready-to-run configurations (full templates, smoke test, profiled run)
  grids/                parameter grids for multi-configuration datasets
docs/                   setup, usage, experiments, configuration, data format, results API
notebooks/              analysis examples using moe_results
packages/moe-reliability-results/   results library (query, aggregation, visualization)
src/moe_reliability/
  cli.py                moe-experiments command line interface
  config.py             TOML schema, validation, templates
  grid.py               grid expansion
  runs.py               run directories, manifest, stage and point bookkeeping
  environment.py        Ascend environment checks (CANN, torch_npu, NPUs, versions) and provenance
  pipelines/            experiment stages (activation capture, workloads, checkpoints,
                        benchmarking, trace analysis, figures)
  core/                 research code: vLLM Ascend serving and measurement, router probes and hooks,
                        synthetic workload construction, forced imbalance, trace analysis
tests/                  test suite (simulated NPU deployment)
```

## Documentation

- [Setup](docs/setup.md): Ascend host preparation, installation, Docker, credentials
- [Usage](docs/usage.md): CLI workflow, runs, resuming, grids, logs
- [Experiments](docs/experiments.md): methodology and metric definitions
- [Configuration reference](docs/configuration.md): every TOML key
- [Data format](docs/data-format.md): run directory and JSON schemas
- [Results API](docs/results-api.md): querying and plotting stored results

## Tests

```bash
uv run pytest
```

The suite does not occupy NPUs: it replaces the vLLM Ascend deployment, the torch_npu runtime and
profiler parser, the dataset download and the model configuration with simulated equivalents, and
exercises all pipelines, environment checks, storage, grids, both CLIs and the plotting functions.

## Author

Dylan Everingham, TU Berlin.
