# moe-reliability-results

Query and visualization library for results of the `moe-reliability` MoE inference reliability
experiments. It reads results directories (JSON run directories), exposes them as pandas tables, and
renders the experiment figures. It depends only on NumPy, pandas and Matplotlib.

## Installation

Inside the `moe-reliability` repository it is installed automatically (`uv sync`). Standalone:

```bash
pip install ./packages/moe-reliability-results        # or: uv pip install ./packages/moe-reliability-results
```

## Usage

```python
from moe_reliability_results import ResultsStore, plots

store = ResultsStore("results")
store.summary(experiment="forced_imbalance", n_npus=8)         # one row per sweep point
store.query("sweep_value > 1.0 and tpot_ms_mean > 40")          # pandas query expression
store.requests(model_name="deepseek-v2")                           # per-request measurements
store.infrastructure_configurations()                           # distinct configurations measured

run = store.latest(experiment="synthetic_workloads")
plots.save_figures(plots.plot_run(run), "figures/")
```

```bash
moe-reliability-results --results-dir results list
moe-reliability-results summary -f n_npus=8 -q "tpot_ms_p99 > 60" -o dataset.csv
moe-reliability-results plot <run-id>
```

Full reference: [docs/results-api.md](../../docs/results-api.md). File format:
[docs/data-format.md](../../docs/data-format.md).
