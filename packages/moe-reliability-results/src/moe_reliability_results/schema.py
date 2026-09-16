###
# schema.py
#
# Layout of local results directories with one sub-dir per run.
# Looks like:
#
# <results_dir>/
#    <run_id>/
#      manifest.json                       run metadata, config, stage and point status
#      config.toml                         fully resolved configuration used for the run
#      activations/records.jsonl[.gz]      expert-capture records (synthetic workloads)
#      workloads/workloads_repeats<R>.json[.gz]
#      metrics/<label>.json[.gz]           per-request end-to-end metrics for one sweep point
#      traces/<label>/                     NPU profiler data written by the vLLM Ascend workers
#      trace_metrics/<label>.json[.gz]     fused-MoE kernel metrics extracted from the traces
#      hta/hta_metrics.json[.gz]           Holistic Trace Analysis tables (optional)
#      validation/<label>.json[.gz]        router load check for forced-imbalance checkpoints
#      figures/*.png                       standard figures rendered at the end of a run
#      logs/run.log                        console output of the run, including the vLLM server
#
# <label> identifies a sweep point, e.g. alpha_0.8 or imbalance_100.
#
# Dylan Everingham
# 16.09.2026
###

from __future__ import annotations

SCHEMA_VERSION = 1

MANIFEST_FILE = "manifest.json"
CONFIG_FILE = "config.toml"

ACTIVATIONS_FILE = "activations/records.jsonl"
WORKLOADS_DIR = "workloads"
METRICS_DIR = "metrics"
TRACES_DIR = "traces"
TRACE_METRICS_DIR = "trace_metrics"
HTA_FILE = "hta/hta_metrics.json"
VALIDATION_DIR = "validation"
FIGURES_DIR = "figures"
LOG_FILE = "logs/run.log"

EXPERIMENT_SYNTHETIC_WORKLOADS = "synthetic_workloads"
EXPERIMENT_FORCED_IMBALANCE = "forced_imbalance"
EXPERIMENT_TYPES = (EXPERIMENT_SYNTHETIC_WORKLOADS, EXPERIMENT_FORCED_IMBALANCE)

# Name of the swept parameter for each experiment type
SWEEP_PARAMETERS = {
    EXPERIMENT_SYNTHETIC_WORKLOADS: "alpha",
    EXPERIMENT_FORCED_IMBALANCE: "imbalance_level",
}

# Prefix used in file and directory names for each sweep parameter
LABEL_PREFIXES = {
    "alpha": "alpha",
    "imbalance_level": "imbalance",
}

# Human-readable axis label for each sweep parameter
SWEEP_AXIS_LABELS = {
    "alpha": "Load Imbalance (Parameterized by Alpha)",
    "imbalance_level": "Forced Imbalance Level",
}

# Configuration keys that describe the serving infrastructure for a sweep point
# Together with the sweep value they identify one configuration
INFRASTRUCTURE_KEYS = (
    "model_name",
    "n_npus",
    "batch_size",
    "max_model_len",
    "gpu_memory_utilization",
    "enable_expert_parallel",
    "enable_eplb",
    "enable_prefix_caching",
    "concurrency_limit",
    "max_new_tokens",
)

# Run / stage / point status values
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"
STATUS_INTERRUPTED = "interrupted"
STATUS_PARTIAL = "partial"


def format_value(value: float | int) -> str:
    if isinstance(value, bool):
        raise TypeError("sweep values must be numeric")
    if isinstance(value, int):
        return str(value)
    return repr(float(value))


def point_label(parameter: str, value: float | int) -> str:
    return f"{LABEL_PREFIXES.get(parameter, parameter)}_{format_value(value)}"


def workloads_file(max_repeats: int) -> str:
    return f"{WORKLOADS_DIR}/workloads_repeats{int(max_repeats)}.json"


def metrics_file(label: str) -> str:
    return f"{METRICS_DIR}/{label}.json"


def trace_metrics_file(label: str) -> str:
    return f"{TRACE_METRICS_DIR}/{label}.json"


def trace_dir(label: str) -> str:
    return f"{TRACES_DIR}/{label}"


def validation_file(label: str) -> str:
    return f"{VALIDATION_DIR}/{label}.json"
