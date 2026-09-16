###
# __init__.py
#
# moe_reliability_results - access and visualize locally stored experiment results.
# Has no dependency on the experiment runtime (vLLM, PyTorch) and can be installed
# on its own for offline analysis.
# 
# Quick start:
#
# from moe_reliability_results import ResultsStore
#
# store = ResultsStore("results")
# summary = store.summary(experiment="forced_imbalance") # one row per sweep point
# slow = store.query("tpot_ms_mean > 40", n_npus=8) # pandas query expression
# run = store.latest(model_name="deepseek-v2")
# requests = run.requests() # per-request measurements
#
# from moe_reliability_results import plots
# figures = plots.plot_run(run)
#
# Dylan Everingham
# 16.09.2026
###

from .run import Run, decode_activation_record, decode_workloads, flatten_config
from .store import ResultsStore, default_results_dir

__version__ = "1.0.0"

__all__ = [
    "ResultsStore",
    "Run",
    "default_results_dir",
    "decode_activation_record",
    "decode_workloads",
    "flatten_config",
    "__version__",
]
