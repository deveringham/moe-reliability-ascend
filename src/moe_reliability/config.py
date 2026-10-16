###
# config.py
#
# Experiment configuration via TOML schema.
# A run is fully described by one TOML file. Its sections and keys are defined
# by schema; every key is unique across sections, which keeps the
# flattened configuration (used for querying results) unambiguous.
#
# Values can be overridden on the command line with --set section.key=value
# where value uses TOML syntax (ex. --set server.batch_size=256,
# --set workloads.target_alphas=[0.5,1.0]).
# Dylan Everingham
# 16.09.2026
###

from __future__ import annotations

import copy
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from moe_reliability_results import schema as rschema

__all__ = [
    "ConfigError",
    "ExperimentConfig",
    "Option",
    "SCHEMA",
    "TEMPLATES",
    "reference_markdown",
    "render_template",
    "sections_for",
    "to_toml",
]

EXPERIMENT_TYPES = rschema.EXPERIMENT_TYPES
PROBE_CHOICES = ("auto", "deepseek", "qwen", "mistral")


class ConfigError(ValueError):
    """Raised for invalid configuration files or overrides."""


class _Required:
    def __repr__(self) -> str:
        return "<required>"


class _Derived:
    def __repr__(self) -> str:
        return "<derived>"


REQUIRED = _Required()
DERIVED = _Derived()


@dataclass(frozen=True)
class Option:
    key: str
    kind: str  # str | int | float | number | bool | list[float] | list[int] | list[number]
    default: Any = REQUIRED
    help: str = ""
    choices: tuple = ()


@dataclass(frozen=True)
class Section:
    name: str
    help: str
    options: tuple[Option, ...] = ()
    free_form: bool = False  # string-to-string table (environment variables)


#  Schema

_EXPERIMENT = Section("experiment", "Experiment identity and reproducibility.", (
    Option("type", "str", REQUIRED, "Experiment type: sweep over synthetic workload imbalance (alpha) "
           "or over forced router imbalance levels.", EXPERIMENT_TYPES),
    Option("name", "str", "", "Optional label appended to the run id."),
    Option("description", "str", "", "Free-text description stored with the run."),
    Option("seed", "int", 43, "Seed for PyTorch, dataset shuffling and vLLM sampling."),
))

_MODEL = Section("model", "Model under test.", (
    Option("model_id", "str", REQUIRED, "Hugging Face model id or local checkpoint path served by vLLM."),
    Option("model_name", "str", REQUIRED, "Short model name used in run ids and file names "
           "(names containing 'deepseek' enable DeepSeek-specific activation preprocessing)."),
    Option("probe", "str", "auto", "Model family, used to read the MoE dimensions from the model configuration "
           "(auto infers it from model_id).", PROBE_CHOICES),
    Option("enable_bnb", "bool", False, "bitsandbytes quantization. Not supported by vLLM Ascend; must stay false "
           "(use a ModelSlim, LLM-Compressor or block-wise FP8 checkpoint as model_id instead)."),
))

_HARDWARE = Section("hardware", "Ascend NPUs used by the deployment.", (
    Option("n_npus", "int", 8, "Number of Ascend NPUs (tensor-parallel size)."),
    Option("visible_devices", "str", "", "Comma-separated NPU ids exposed to the run (ASCEND_RT_VISIBLE_DEVICES); "
           "empty keeps the current environment."),
))

_SERVER = Section("server", "vLLM server deployment.", (
    Option("port", "int", 8000, "Port of the local vLLM OpenAI-compatible server."),
    Option("max_model_len", "int", 2048, "Maximum model context length."),
    Option("gpu_memory_utilization", "float", 0.6, "Fraction of NPU memory vLLM may use "
           "(vLLM's --gpu-memory-utilization)."),
    Option("batch_size", "int", 512, "Maximum number of concurrently batched sequences (--max-num-seqs)."),
    Option("max_num_batched_tokens", "int", 4096, "Token budget of one forward pass (--max-num-batched-tokens). "
           "Sets how many tokens a prefill step carries, and so how many tokens each expert's GEMM sees: "
           "decode steps are bounded by batch_size instead."),
    Option("enable_expert_parallel", "bool", True, "Enable expert parallelism."),
    Option("enforce_eager", "bool", True, "Run the model eagerly (--enforce-eager). Eager mode leaves the NPUs "
           "idle between host-launched kernels, ~30% of a prefill step on 4 NPUs, which shrinks every kernel's "
           "share of the step; false lets vllm-ascend capture graphs."),
    Option("enable_prefix_caching", "bool", False, "Enable prefix caching."),
    Option("enable_eplb", "bool", False, "Enable vllm-ascend dynamic expert-parallel load balancing. "
           "Sets additional_config.eplb_config.dynamic_eplb and DYNAMIC_EPLB in the server environment; "
           "vLLM's own --enable-eplb flag does not reach the vllm-ascend implementation."),
    Option("eplb_policy_type", "int", 2, "Rearrangement policy: 0 random, 1 DefaultEplb, "
           "2 SwiftBalanceEplb, 3 FlashLB."),
    Option("eplb_num_redundant_experts", "int", 0, "Redundant expert slots EPLB may use to replicate hot "
           "experts. At 0 it can only permute experts between ranks, which cannot change the load when "
           "each rank holds exactly one expert."),
    Option("eplb_heat_collection_interval", "int", 600, "Forward iterations of expert-load collection "
           "before each rearrangement. The full cycle is this plus "
           "eplb_algorithm_execution_interval plus one iteration per MoE layer, and the load counters "
           "are cleared at the end of it, so a run shorter than one cycle never rearranges at all."),
    Option("eplb_algorithm_execution_interval", "int", 50, "Iterations the planner is given before the "
           "new placement is applied."),
    Option("eplb_record_map", "bool", False, "Record-only mode: EPLB collects load and plans, and the "
           "placement it planned is written to <label>_eplb_expert_map.json, but it is never applied. "
           "vllm-ascend 0.23 skips the weight update whenever a record path is set (EplbUpdator.forward_end), "
           "so the first layer's transfer is launched, never completed, and every later one is refused "
           "('D2D weight update is on-going'). The layout stays contiguous for the whole point. That makes "
           "this the collect-and-plan arm of an overhead decomposition, never an EPLB-on arm: every EPLB run "
           "before 2026-10-12 had it on and measured no rearrangement at all."),
    Option("eplb_min_imbalance", "float", 0.0, "Apply a planned placement only once the measured "
           "per-layer imbalance (busiest rank over the mean, worst MoE layer - EPLB's own "
           "_compute_imbalance statistic) reaches this load. 0 disables the gate, which is what vllm-ascend "
           "and vLLM both do today: they rearrange on an iteration counter without asking whether the "
           "imbalance costs anything. Set it to the deployment's calibrated impact threshold - the load at "
           "which skew starts costing latency, 2.43x on DeepSeek-V2-Lite and 1.42x on Mixtral 8x7B. Needs "
           "the node patch (scripts/apply_node_patch.py); the run refuses to start without it, since an "
           "unpatched server ignores it and quietly measures plain EPLB again."),
    Option("eplb_honor_changed", "bool", False, "Apply a planned placement only when EPLB's own policy "
           "says it is worth applying. Policies 1 and 2 return a 'changed' flag - set when the plan "
           "improves the summed per-layer max heat by at least 5% - and EplbWorker.do_update discards it, "
           "so placements are applied whatever the policy concluded. This honours it, and is worth fixing "
           "upstream, but do not expect a latency result from it: the flag asks whether the plan improves "
           "balance, not whether the imbalance cost anything, and at every level measured in eplb-fp2 "
           "(1.09x to 3.63x) the plan improved EPLB's own metric by 9-69%, so it reads 'changed' even at "
           "the pure false positive. Ignored when eplb_min_imbalance is set. Needs the same node patch."),
    Option("count_experts", "bool", False, "Run the detection screen's counting inside the server: a "
           "random sample of MoE layers, redrawn every window, bincounted over a strided sample of their "
           "tokens into a device-side per-layer table, copied out once per window. This is what a "
           "deployment would pay to screen, and it is not benchmark.capture_experts scaled down - capture "
           "returns every token's expert ids in the response body, which a screen never needs. Only one "
           "rank counts, since expert selection is replicated."),
    Option("count_layers", "int", 4, "MoE layers counted per window by server.count_experts. The "
           "calibrated screen uses 4; they are redrawn each window, because evenly spaced layers are "
           "blind to a skew confined to a few (docs/detection-findings.md)."),
    Option("count_token_fraction", "float", 0.1, "Fraction of a counted layer's tokens to count, applied "
           "as a stride. The calibrated screen uses 0.1, which with count_layers over 26 layers is about "
           "1.5% of the routing."),
    Option("count_window_steps", "int", 200, "Engine steps per counting window: how often the layer sample "
           "is redrawn and the counts copied out. The screen's window is 8 requests, which under "
           "continuous batching is not a fixed number of steps, so this sets the emit cadence rather than "
           "a window the detector would score."),
))

_CLIENT = Section("client", "Load generation against the server.", (
    Option("max_new_tokens", "int", 100, "Maximum number of generated tokens per request."),
    Option("concurrency_limit", "int", 700, "Maximum number of in-flight requests "
           "(should exceed batch_size to saturate the server)."),
    Option("n_warmup_samples", "int", 10, "Number of warm-up requests before measurement."),
    Option("stream", "bool", True, "Stream responses, which is what gives TTFT, TPOT and inter-token "
           "latencies. Off, each request is timed end to end only. Expert capture is never streamed, so an arm "
           "compared against benchmark.capture_experts has to turn this off too: streaming costs the client "
           "and API server per token, and with 2000 prompts in flight that difference is larger than the "
           "capture (the capturing arm finished 15% sooner than a streaming one on 2026-10-09)."),
    Option("ignore_eos", "bool", False, "Generate exactly max_new_tokens per request, ignoring "
           "end-of-sequence (vLLM's ignore_eos). Holds output length fixed across sweep points: a strong router "
           "bias degrades DeepSeek-V2-Lite enough that some answers end early (205 of 300 tokens on average at "
           "offset 2, 84 of 100), and with length held the skew costs nothing - the ~20-30% read as its cost "
           "came with the early stops. Applies to warm-up, benchmark and the validation capture alike."),
    Option("lengths_from", "str", "", "Replay another point's output lengths: a metrics file "
           "(<run>/metrics/<label>.json.gz) whose requests give each prompt the number of tokens it generated "
           "there, matched by prompt content. Needs ignore_eos, so each request generates exactly that many, "
           "and max_new_tokens at least the longest. Separates *how many* tokens a point generated from "
           "*which* requests stopped early: DeepSeek at router-bias offset 2 averages 84 of 100 tokens and "
           "costs +21% TPOT, while every request held at 84 costs nothing. A prompt missing from the file is "
           "an error, never a fallback to max_new_tokens."),
))

_ACTIVATIONS = Section("activations", "Stage 1 (synthetic workloads): expert activation capture on MMLU prompts.", (
    Option("n_samples", "int", 15000, "Number of MMLU prompts to capture routed experts for."),
    Option("reuse_activations_from", "str", "", "Run id or run directory whose activation records are "
           "reused instead of capturing new ones."),
))

_WORKLOADS = Section("workloads", "Stage 2 (synthetic workloads): workload construction.", (
    Option("target_alphas", "list[float]", [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 2.0],
           "Scaling factors of the natural per-layer coefficient of variation of expert load."),
    Option("target_prompt_lengths", "list[int]", [1000, 5000], "Workload sizes in prompts "
           "(converted to token budgets using the average tokens per prompt, "
           "unless length_in_requests is set)."),
    Option("length_in_requests", "bool", False, "Stop selection at a fixed number of requests instead of a "
           "fixed token budget. With a token budget the request count varies with alpha - more imbalanced "
           "workloads are built from more, shorter prompts - which confounds imbalance with batch size."),
    Option("prompt_length_tolerance", "float", 0.0, "Restrict selection to prompts whose token count is "
           "within this relative distance of the pool median (0.25 keeps 0.75x to 1.25x the median; "
           "0 disables the restriction). Keeps per-workload differences from being dominated by prompt size."),
    Option("max_repeats", "list[int]", [0, 10], "Workload sets to build, one per maximum number of "
           "times a prompt may be repeated."),
    Option("reuse_workloads_from", "str", "", "Run id or run directory whose workloads are reused "
           "(skips activation capture and workload construction)."),
))

_BENCHMARK_SYNTHETIC = Section("benchmark", "Stage 3: benchmarking of each sweep point.", (
    Option("workload_max_repeats", "int", 0, "Which workload set (max_repeats) to benchmark."),
    Option("workload_prompt_length", "int", 1000, "Which workload size (target prompt length) to benchmark."),
    Option("repeats", "int", 1, "Benchmark every sweep point this many times. Points sharing a value differ only in the state of the machine, so their spread measures the run's own noise floor."),
    Option("shuffle_points", "bool", True, "Serve the sweep points in seeded random rounds, one per repeat, each holding every value once, so values interleave over the run. In parameter order, anything that drifts during a run - a neighbouring job, thermal state - aliases onto the swept parameter."),
    Option("enable_profiling", "bool", True, "Record PyTorch profiler traces on all workers."),
    Option("separate_profiling_run", "bool", True, "Benchmark each point twice when profiling: once "
           "unprofiled for the timings and once profiled for the traces. Profiling perturbs latency, so a "
           "single profiled pass cannot provide both."),
    Option("trace_active_iterations", "int", 2, "Number of profiled scheduler iterations."),
    Option("trace_start_iteration", "int", 100, "Scheduler iterations after profiling starts before the window opens. A point with fewer forward passes than this never records a trace: a 2000-request prefill-only point at a 4096-token budget is ~80."),
    Option("save_request_metrics", "bool", DERIVED, "Store per-request TTFT/TPOT measurements "
           "(default: true, unless profiling without a separate unprofiled pass)."),
    Option("save_itl", "bool", True, "Record inter-token latencies: the gap between consecutive streamed "
           "chunks of each request, in ms. TPOT averages over a request's decode steps, so a single slow step "
           "is invisible in it; the ITL series is what shows within-request spikes and what a change in TPOT "
           "can be attributed to. Costs one timestamp per token in the client and a list per request in the "
           "metrics file."),
    Option("capture_experts", "bool", False, "Serve the benchmark with --enable-return-routed-experts, so "
           "every request returns the experts it routed to. This is what a monitor's collection path costs: "
           "the screen's own arithmetic is negligible beside getting the routing off the device, and that "
           "cost is otherwise assumed rather than measured. Benchmark it against an otherwise identical "
           "point with this off - the difference is the ceiling, at every layer and every token."),
))

_IMBALANCE = Section("imbalance", "Forced router imbalance.", (
    Option("method", "str", "checkpoint", "How imbalance is injected. checkpoint writes a modified model whose "
           "routing collapses onto the lowest-numbered experts at any nonzero level (one useful setting: total "
           "collapse). router_bias serves the unmodified model and adds the level to the router logits of the "
           "experts in bias_target, a graded skew that keeps routing input-dependent.",
           choices=("checkpoint", "router_bias")),
    Option("imbalance_levels", "list[number]", [0, 100], "checkpoint: bias added to expert 0's router row. "
           "router_bias: logit offset added to the targeted experts. 0 serves the unmodified model."),
    Option("bias_target", "str", "rank:0", "router_bias only: 'rank:<r>' biases every expert placed on "
           "expert-parallel rank r (contiguous placement, hardware.n_npus ranks); 'experts:<i>,<j>' biases "
           "those experts; 'rotate' biases rank (layer mod n_npus), so every layer is skewed as strongly as "
           "a fixed rank target would make it but no rank is hot throughout - the shape natural imbalance "
           "has, and the case where per-rank totals cancel."),
    Option("model_dir", "str", "models", "Directory for generated imbalanced checkpoints "
           "(reused across runs when present)."),
    Option("validate_imbalance", "bool", False, "Capture routed experts at every level before benchmarking "
           "and record per-expert and per-rank load."),
    Option("validation_samples", "int", 0, "Prompts for validation: 0 uses six fixed prompts, N > 0 the first "
           "N MMLU prompts. Rank shares need a few hundred to be stable."),
    Option("bias_layers", "list[int]", [], "router_bias only: model layer indices (as in model.layers.<i>) "
           "the bias applies to; empty biases every MoE layer. A few layers hold a skew the way natural "
           "imbalance does, and test whether a detector localises it. The zero-bias control uses the same "
           "layers."),
    Option("bias_schedule", "list[str]", [], "router_bias only: change the skew while one server runs, as "
           "'<engine step>:<fraction of this point's level>' entries - ['0:0', '600:1'] serves balanced for "
           "600 forwards and then steps to the point's level. The first entry must be at step 0, so what the "
           "server serves before the first change is explicit, and the fractions multiply the point's level, "
           "so a sweep gets one onset per level at the same step. Steps rather than seconds because expert "
           "selection is replicated: two ranks holding different biases on the same forward would select "
           "different experts, and a step count is identical on every rank where a clock is not. This is how "
           "detection *delay* is measured; every run before 2026-10-16 held its bias for the server's life, "
           "so an onset could only be simulated by splicing two captures. The counting screen records the "
           "step per window, so the two line up directly."),
    Option("validation_workload", "str", "", "Prompts for validation: empty keeps the first "
           "validation_samples MMLU prompts in the order every earlier run used; a workload spec draws them "
           "shuffled from one family or several: mmlu, gsm8k, mbpp, mmmlu:<LANG> (e.g. mmmlu:ZH_CN), dolly, "
           "ultrachat, or mixed:<a>,<b>,... (interleaved, equal shares)."),
    Option("validation_save_records", "bool", False, "Also write every validation request's routed experts "
           "(validation/<label>.records.jsonl), for detection analyses that need per-request or per-token "
           "routing rather than the pooled counts."),
    Option("bias_plugin_at_zero", "bool", False, "router_bias only: install the plugin at level 0 as well, with "
           "an all-zero bias vector. The plugin adds a tensor to the router logits on every expert-selection "
           "call of every rank, so a level-0 point without it is cheaper for a reason unrelated to skew, and "
           "the difference between the arms is the skew plus the instrument. Turn this on to isolate the skew; "
           "compare the two forms of level 0 to measure what the instrument costs."),
))

_BENCHMARK_FORCED = Section("benchmark", "Benchmarking of each imbalance level.", (
    Option("n_samples", "int", 15000, "Number of prompts sent to each checkpoint."),
    Option("workload", "str", "", "Prompts to benchmark with: empty keeps MMLU in the order every earlier "
           "run used; a workload spec serves another corpus or a mix of them (same grammar as "
           "imbalance.validation_workload). Routing concentration varies by corpus - low-resource "
           "languages reach a busiest rank well above English - so this is how a latency cost is measured "
           "on the traffic that produces it."),
    Option("repeats", "int", 1, "Benchmark every sweep point this many times. Points sharing a value differ only in the state of the machine, so their spread measures the run's own noise floor."),
    Option("shuffle_points", "bool", True, "Serve the sweep points in seeded random rounds, one per repeat, each holding every value once, so values interleave over the run. In parameter order, anything that drifts during a run - a neighbouring job, thermal state - aliases onto the swept parameter."),
    Option("arms", "list[str]", [], "Serve every level once per arm, inside this run: 'name' or "
           "'name: section.key=value; section.key=value', e.g. ['eplb-off', 'eplb-on: server.enable_eplb=true']. "
           "Arms are shuffled into the same rounds as the levels, so anything that drifts during the run "
           "falls on both. Comparing arms run as separate grid entries aliases drift onto the arm instead: "
           "on 2026-10-09 a neighbouring job grew from one process to eight during the second entry and "
           "read as a 10% EPLB cost. Overrides may touch [server], [client] and benchmark.capture_experts; "
           "an arm with no overrides is the configuration as written. Validation runs once per level, on "
           "the configuration as written, since no arm setting changes which experts a token routes to.", ),
    Option("enable_profiling", "bool", False, "Record PyTorch profiler traces on all workers."),
    Option("separate_profiling_run", "bool", True, "Benchmark each point twice when profiling: once "
           "unprofiled for the timings and once profiled for the traces. Profiling perturbs latency, so a "
           "single profiled pass cannot provide both."),
    Option("trace_active_iterations", "int", 2, "Number of profiled scheduler iterations."),
    Option("trace_start_iteration", "int", 100, "Scheduler iterations after profiling starts before the window opens. A point with fewer forward passes than this never records a trace: a 2000-request prefill-only point at a 4096-token budget is ~80."),
    Option("save_request_metrics", "bool", DERIVED, "Store per-request TTFT/TPOT measurements "
           "(default: true, unless profiling without a separate unprofiled pass)."),
    Option("save_itl", "bool", True, "Record inter-token latencies: the gap between consecutive streamed "
           "chunks of each request, in ms. TPOT averages over a request's decode steps, so a single slow step "
           "is invisible in it; the ITL series is what shows within-request spikes and what a change in TPOT "
           "can be attributed to. Costs one timestamp per token in the client and a list per request in the "
           "metrics file."),
    Option("capture_experts", "bool", False, "Serve the benchmark with --enable-return-routed-experts, so "
           "every request returns the experts it routed to. This is what a monitor's collection path costs: "
           "the screen's own arithmetic is negligible beside getting the routing off the device, and that "
           "cost is otherwise assumed rather than measured. Benchmark it against an otherwise identical "
           "point with this off - the difference is the ceiling, at every layer and every token."),
))

_ANALYSIS = Section("analysis", "Post-processing of profiler traces.", (
    Option("parse_npu_traces", "bool", True, "Convert raw NPU profiler data into timeline files "
           "(ASCEND_PROFILER_OUTPUT/trace_view.json) with torch_npu's offline parser."),
    Option("trace_summary", "bool", True, "Extract fused-MoE kernel metrics per rank from traces."),
    Option("hta", "bool", False, "Run Holistic Trace Analysis (requires the 'hta' extra)."),
))

_OUTPUT = Section("output", "Result storage.", (
    Option("results_dir", "str", "results", "Directory containing all runs."),
    Option("compress", "bool", True, "Gzip-compress large JSON artefacts."),
    Option("save_figures", "bool", True, "Render standard figures at the end of the run."),
))

_ENVIRONMENT = Section("environment", "Environment variables set before any work starts (inherited by the "
                       "vLLM server), e.g. HCCL_CONNECT_TIMEOUT or TASK_QUEUE_ENABLE.", free_form=True)

_DEFAULT_ENVIRONMENT: dict[str, str] = {}

SCHEMA: dict[str, tuple[Section, ...]] = {
    rschema.EXPERIMENT_SYNTHETIC_WORKLOADS: (
        _EXPERIMENT, _MODEL, _HARDWARE, _SERVER, _CLIENT, _ACTIVATIONS, _WORKLOADS,
        _BENCHMARK_SYNTHETIC, _ANALYSIS, _OUTPUT, _ENVIRONMENT),
    rschema.EXPERIMENT_FORCED_IMBALANCE: (
        _EXPERIMENT, _MODEL, _HARDWARE, _SERVER, _CLIENT, _IMBALANCE,
        _BENCHMARK_FORCED, _ANALYSIS, _OUTPUT, _ENVIRONMENT),
}

# Reference configurations
TEMPLATES: dict[str, dict[str, dict[str, Any]]] = {
    rschema.EXPERIMENT_SYNTHETIC_WORKLOADS: {
        "experiment": {"type": "synthetic_workloads", "name": "alpha-sweep", "seed": 43},
        "model": {"model_id": "deepseek-ai/DeepSeek-V2-Lite-Chat", "model_name": "deepseek-v2"},
        "hardware": {"n_npus": 8},
        "server": {"batch_size": 512},
        "client": {"concurrency_limit": 700},
        "benchmark": {"enable_profiling": True, "save_request_metrics": False},
    },
    rschema.EXPERIMENT_FORCED_IMBALANCE: {
        "experiment": {"type": "forced_imbalance", "name": "imbalance-sweep", "seed": 43},
        "model": {"model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1", "model_name": "mistral"},
        "hardware": {"n_npus": 8},
        "server": {"batch_size": 512},
        "client": {"concurrency_limit": 700},
        "imbalance": {"imbalance_levels": [0, 100]},
        "benchmark": {"enable_profiling": False, "save_request_metrics": True},
    },
}


def sections_for(experiment_type: str) -> tuple[Section, ...]:
    try:
        return SCHEMA[experiment_type]
    except KeyError:
        raise ConfigError(f"experiment.type must be one of {list(SCHEMA)}, got {experiment_type!r}") from None


#  Value coercion and validation
def _coerce(value: Any, kind: str, where: str) -> Any:
    def scalar(v: Any, k: str) -> Any:
        if k == "str":
            if isinstance(v, str):
                return v
        elif k == "bool":
            if isinstance(v, bool):
                return v
        elif k == "int":
            if isinstance(v, int) and not isinstance(v, bool):
                return v
        elif k == "float":
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
        elif k == "number":
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return int(v) if isinstance(v, float) and v.is_integer() else v
        raise ConfigError(f"{where}: expected {k}, got {type(v).__name__} ({v!r})")

    if kind.startswith("list[") and kind.endswith("]"):
        inner = kind[5:-1]
        if not isinstance(value, list):
            raise ConfigError(f"{where}: expected a list of {inner}, got {type(value).__name__}")
        return [scalar(v, inner) for v in value]
    return scalar(value, kind)


_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def infer_probe_family(model_id: str) -> str | None:
    """Router probe family inferred from a model id or path."""
    lowered = model_id.lower()
    if "deepseek" in lowered:
        return "deepseek"
    if "qwen" in lowered:
        return "qwen"
    if "mixtral" in lowered or "mistral" in lowered:
        return "mistral"
    return None


class _SectionView:
    """Read-only attribute access to one configuration section."""

    def __init__(self, name: str, values: Mapping[str, Any]):
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_values", values)

    def __getattr__(self, key: str) -> Any:
        try:
            return self._values[key]
        except KeyError:
            raise AttributeError(f"[{self._name}] has no key {key!r}") from None

    def __setattr__(self, key: str, value: Any) -> None:
        raise AttributeError("configuration sections are read-only")

    def __repr__(self) -> str:
        return f"[{self._name}] {dict(self._values)!r}"

# Full experiment configuration
class ExperimentConfig:

    def __init__(self, data: Mapping[str, Any], source: Path | None = None):
        self._data = _resolve(data)
        self.source = source

    @classmethod
    def load(cls, path: str | Path, overrides: Iterable[str] = ()) -> "ExperimentConfig":
        path = Path(path)
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
        except FileNotFoundError:
            raise ConfigError(f"configuration file not found: {path}") from None
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{path}: invalid TOML: {exc}") from None
        return cls(apply_overrides(data, overrides), source=path)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], overrides: Iterable[str] = ()) -> "ExperimentConfig":
        return cls(apply_overrides(copy.deepcopy(dict(data)), overrides))

    def __getattr__(self, name: str) -> Any:
        data = self.__dict__.get("_data")
        if data is not None and name in data:
            if isinstance(data[name], dict) and name != "environment":
                return _SectionView(name, data[name])
            return data[name]
        raise AttributeError(name)

    @property
    def experiment_type(self) -> str:
        return self._data["experiment"]["type"]

    @property
    def environment(self) -> dict[str, str]:
        return dict(self._data["environment"])

    @property
    def probe_family(self) -> str | None:
        probe = self._data["model"]["probe"]
        return infer_probe_family(self._data["model"]["model_id"]) if probe == "auto" else probe

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return copy.deepcopy(self._data)

    @property
    def arms(self) -> list[tuple[str, list[str]]]:
        """``(name, overrides)`` per benchmark arm; empty when the run has none."""
        return [parse_arm(spec) for spec in self._data["benchmark"].get("arms", [])]

    def for_arm(self, name: str | None) -> "ExperimentConfig":
        """This configuration with one arm's overrides applied (itself for ``None``)."""
        if name is None:
            return self
        for arm, overrides in self.arms:
            if arm == name:
                data = self.to_dict()
                data["benchmark"]["arms"] = []
                return ExperimentConfig(apply_overrides(data, overrides), source=self.source)
        raise ConfigError(f"no benchmark arm named {name!r}; arms are {[a for a, _ in self.arms]}")

    def flat(self) -> dict[str, Any]:
        return {k: v for name, sec in self._data.items() if name != "environment" for k, v in sec.items()}

    def to_toml(self, header: str | None = None) -> str:
        return to_toml(self._data, header=header)


_ARM_KEYS = ("server.", "client.", "benchmark.capture_experts")


def parse_arm(spec: str) -> tuple[str, list[str]]:
    """``'eplb-on: server.enable_eplb=true; ...'`` -> ``('eplb-on', ['server.enable_eplb=true', ...])``."""
    name, _, rest = spec.partition(":")
    return name.strip(), [item.strip() for item in rest.split(";") if item.strip()]


def apply_overrides(data: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    for item in overrides:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise ConfigError(f"invalid override {item!r}; expected section.key=value")
        dotted, raw = item.split("=", 1)
        section, key = dotted.strip().split(".", 1)
        try:
            value = tomllib.loads(f"v = {raw.strip()}")["v"]
        except tomllib.TOMLDecodeError:
            value = raw.strip()  # bare strings need no quotes on the command line
        data.setdefault(section, {})
        if not isinstance(data[section], dict):
            raise ConfigError(f"override {item!r}: [{section}] is not a table")
        data[section][key] = value
    return data


def _resolve(data: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(data.get("experiment"), Mapping) or "type" not in data["experiment"]:
        raise ConfigError("missing required key experiment.type")
    sections = sections_for(data["experiment"]["type"])
    known = {s.name for s in sections}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"unknown section(s) for experiment type {data['experiment']['type']!r}: {unknown}")

    out: dict[str, dict[str, Any]] = {}
    for section in sections:
        raw = data.get(section.name, {})
        if not isinstance(raw, Mapping):
            raise ConfigError(f"[{section.name}] must be a table")
        if section.free_form:
            env = dict(_DEFAULT_ENVIRONMENT)
            for k, v in raw.items():
                if not isinstance(v, (str, int, float, bool)):
                    raise ConfigError(f"environment.{k}: expected a string value")
                env[k] = str(v).lower() if isinstance(v, bool) else str(v)
            out[section.name] = env
            continue
        names = {o.key for o in section.options}
        extra = sorted(set(raw) - names)
        if extra:
            raise ConfigError(f"unknown key(s) in [{section.name}]: {extra}; valid keys: {sorted(names)}")
        values: dict[str, Any] = {}
        for opt in section.options:
            where = f"{section.name}.{opt.key}"
            if opt.key in raw:
                value = _coerce(raw[opt.key], opt.kind, where)
            elif opt.default is REQUIRED:
                raise ConfigError(f"missing required key {where}")
            elif opt.default is DERIVED:
                value = DERIVED
            else:
                value = copy.deepcopy(opt.default)
            if opt.choices and value not in opt.choices:
                raise ConfigError(f"{where}: must be one of {list(opt.choices)}, got {value!r}")
            values[opt.key] = value
        out[section.name] = values

    _derive(out)
    _validate(out)
    return out


def _derive(cfg: dict[str, dict[str, Any]]) -> None:
    bench = cfg["benchmark"]
    if bench["save_request_metrics"] is DERIVED:
        # A separate unprofiled pass measures latency without the profiler's
        # overhead, so the request metrics are worth keeping even when profiling.
        bench["save_request_metrics"] = (not bench["enable_profiling"]
                                         or bool(bench.get("separate_profiling_run", False)))


def _validate(cfg: dict[str, dict[str, Any]]) -> None:
    errors: list[str] = []

    def positive(section: str, key: str, allow_zero: bool = False) -> None:
        v = cfg[section][key]
        if v < 0 or (v == 0 and not allow_zero):
            errors.append(f"{section}.{key} must be {'>= 0' if allow_zero else '> 0'}, got {v}")

    if not _NAME_RE.match(cfg["model"]["model_name"]):
        errors.append("model.model_name may only contain letters, digits, '.', '_' and '-'")
    if cfg["experiment"]["name"] and not _NAME_RE.match(cfg["experiment"]["name"]):
        errors.append("experiment.name may only contain letters, digits, '.', '_' and '-'")
    positive("hardware", "n_npus")
    visible = cfg["hardware"]["visible_devices"].strip()
    if visible:
        ids = [d.strip() for d in visible.split(",")]
        if not all(d.isdigit() for d in ids) or len(set(ids)) != len(ids):
            errors.append('hardware.visible_devices must be distinct comma-separated NPU ids, e.g. "0,1,2,3"')
        elif len(ids) < cfg["hardware"]["n_npus"]:
            errors.append(f"hardware.visible_devices exposes {len(ids)} NPUs but hardware.n_npus = "
                          f"{cfg['hardware']['n_npus']}")
    if cfg["model"]["enable_bnb"]:
        errors.append("model.enable_bnb: bitsandbytes quantization is not supported by vLLM Ascend; serve a "
                      "ModelSlim, LLM-Compressor or block-wise FP8 checkpoint instead")
    for key in ("max_model_len", "batch_size", "max_num_batched_tokens"):
        positive("server", key)
    if not 0 < cfg["server"]["gpu_memory_utilization"] <= 1:
        errors.append("server.gpu_memory_utilization must be in (0, 1]")
    if not 1 <= cfg["server"]["port"] <= 65535:
        errors.append("server.port must be in [1, 65535]")
    if cfg["server"]["eplb_min_imbalance"] < 0:
        errors.append("server.eplb_min_imbalance is a busiest-rank-over-mean load, so it must be >= 1 "
                      "(or 0 to disable the gate)")
    elif 0 < cfg["server"]["eplb_min_imbalance"] < 1:
        errors.append(f"server.eplb_min_imbalance = {cfg['server']['eplb_min_imbalance']} is below 1, "
                      "which no imbalance can be under: the busiest rank is never under the mean. A "
                      "threshold under 1 gates nothing. Use 0 to disable the gate.")
    gated = cfg["server"]["eplb_min_imbalance"] or cfg["server"]["eplb_honor_changed"]
    if gated:
        which = "eplb_min_imbalance" if cfg["server"]["eplb_min_imbalance"] else "eplb_honor_changed"
        # The gate decides whether a *planned* placement is applied, so without
        # EPLB there is nothing to gate and the arm would silently be a second
        # EPLB-off arm. Record-only mode never applies a placement either.
        if not cfg["server"]["enable_eplb"]:
            errors.append(f"server.{which} needs server.enable_eplb: it gates whether a planned "
                          "placement is applied, so with EPLB off it measures nothing")
        if cfg["server"]["eplb_record_map"]:
            errors.append(f"server.{which} with server.eplb_record_map: record-only mode never applies a "
                          "placement, so the gate cannot change what the point measures")
        if cfg["server"]["eplb_policy_type"] not in (1, 2):
            errors.append(f"server.{which} needs eplb_policy_type 1 or 2, not "
                          f"{cfg['server']['eplb_policy_type']}: policy 0 is a shuffle, and policy 3 "
                          "(FlashLB) applies its own update trigger internally, so gating it would be "
                          "two triggers in series")
    positive("client", "max_new_tokens")
    if cfg["client"]["lengths_from"]:
        if not cfg["client"]["ignore_eos"]:
            errors.append("client.lengths_from needs client.ignore_eos: without it each replayed length is "
                          "only a cap, the model stops wherever it likes, and the lengths are not replayed")
        if not Path(cfg["client"]["lengths_from"]).is_file():
            errors.append(f"client.lengths_from: no such metrics file {cfg['client']['lengths_from']!r}")
    positive("client", "concurrency_limit")
    positive("client", "n_warmup_samples", allow_zero=True)
    positive("benchmark", "trace_active_iterations")
    positive("benchmark", "trace_start_iteration")

    arm_names = []
    for spec in cfg["benchmark"].get("arms", []):
        name, overrides = parse_arm(spec)
        arm_names.append(name)
        if not _NAME_RE.match(name):
            errors.append(f"benchmark.arms: arm name {name!r} may only contain letters, digits, '.', '_' and '-'")
        bad = [o for o in overrides if not o.split("=", 1)[0].strip().startswith(_ARM_KEYS)]
        if bad:
            errors.append(f"benchmark.arms: arm {name!r} overrides {bad}; an arm may only change "
                          f"[server], [client] and benchmark.capture_experts, since validation and the "
                          f"prompts are shared between arms")
            continue
        # Each arm is a configuration in its own right, so it gets the same checks.
        data = copy.deepcopy(cfg)
        data["benchmark"]["arms"] = []
        try:
            _resolve(apply_overrides(data, overrides))
        except ConfigError as e:
            detail = str(e).removeprefix("invalid configuration:\n  - ").replace("\n  - ", "; ")
            errors.append(f"benchmark.arms: arm {name!r}: {detail}")
    if len(set(arm_names)) != len(arm_names):
        errors.append("benchmark.arms: arm names must be distinct")

    etype = cfg["experiment"]["type"]
    probe = cfg["model"]["probe"]
    probe_family = infer_probe_family(cfg["model"]["model_id"]) if probe == "auto" else probe

    if etype == rschema.EXPERIMENT_SYNTHETIC_WORKLOADS:
        wl, bench = cfg["workloads"], cfg["benchmark"]
        positive("activations", "n_samples")
        if not wl["target_alphas"] or any(a <= 0 for a in wl["target_alphas"]):
            errors.append("workloads.target_alphas must be a non-empty list of positive numbers")
        if len(set(wl["target_alphas"])) != len(wl["target_alphas"]):
            errors.append("workloads.target_alphas contains duplicates")
        if not wl["target_prompt_lengths"] or any(p <= 0 for p in wl["target_prompt_lengths"]):
            errors.append("workloads.target_prompt_lengths must be a non-empty list of positive integers")
        if not wl["max_repeats"] or any(r < 0 for r in wl["max_repeats"]):
            errors.append("workloads.max_repeats must be a non-empty list of integers >= 0")
        if not wl["reuse_workloads_from"]:
            if bench["workload_max_repeats"] not in wl["max_repeats"]:
                errors.append("benchmark.workload_max_repeats must be one of workloads.max_repeats")
            if bench["workload_prompt_length"] not in wl["target_prompt_lengths"]:
                errors.append("benchmark.workload_prompt_length must be one of workloads.target_prompt_lengths")
            if probe_family is None:
                errors.append("model.probe = 'auto' cannot infer the router family from model.model_id; "
                              f"set it explicitly to one of {list(PROBE_CHOICES[1:])}")
    else:
        levels = cfg["imbalance"]["imbalance_levels"]
        positive("benchmark", "n_samples")
        if cfg["client"]["lengths_from"] and cfg["imbalance"]["validate_imbalance"]:
            errors.append("client.lengths_from with imbalance.validate_imbalance: the validation capture "
                          "serves its own prompts, which the replayed point never saw, so they have no "
                          "length to replay")
        if not levels or any(level < 0 for level in levels):
            errors.append("imbalance.imbalance_levels must be a non-empty list of numbers >= 0")
        if len(set(levels)) != len(levels):
            errors.append("imbalance.imbalance_levels contains duplicates")
        if cfg["imbalance"]["validation_samples"] < 0:
            errors.append("imbalance.validation_samples must be >= 0")
        if cfg["imbalance"]["method"] == "router_bias":
            from .router_bias import parse_target
            kind = None
            try:
                kind, ids = parse_target(cfg["imbalance"]["bias_target"])
                if kind == "rank" and ids[0] >= cfg["hardware"]["n_npus"]:
                    errors.append(f"imbalance.bias_target names rank {ids[0]} but hardware.n_npus = "
                                  f"{cfg['hardware']['n_npus']}")
            except ValueError as e:
                errors.append(f"imbalance.bias_target: {e}")
            # 'rotate' needs the experts to divide over the ranks, like a rank target;
            # the expert count is not known until the model is read, so
            # target_experts raises for it at run time.
            if any(i < 0 for i in cfg["imbalance"]["bias_layers"]):
                errors.append("imbalance.bias_layers must be layer indices >= 0")
            if len(set(cfg["imbalance"]["bias_layers"])) != len(cfg["imbalance"]["bias_layers"]):
                errors.append("imbalance.bias_layers contains duplicates")
            if cfg["imbalance"]["bias_schedule"]:
                from .router_bias import parse_schedule_spec
                try:
                    parse_schedule_spec(cfg["imbalance"]["bias_schedule"])
                except ValueError as e:
                    errors.append(f"imbalance.bias_schedule: {e}")
            # A rank target names the experts a rank holds *under contiguous
            # placement*, which EPLB is free to change, so the skew would wander
            # off the rank it was aimed at and the point would measure nothing
            # definite. An expert target names experts directly: the skew stays
            # on them wherever EPLB puts them, which is what makes an EPLB
            # comparison possible at all. 'rotate' is a rank target per layer.
            if cfg["server"]["enable_eplb"] and kind not in (None, "experts"):
                errors.append(f"imbalance.bias_target = {cfg['imbalance']['bias_target']!r} is a rank "
                              "target, which assumes contiguous expert placement; EPLB moves experts "
                              "between ranks, so the skew would not stay on the rank it names. Use an "
                              "'experts:<i>,<j>,...' target with EPLB: under contiguous placement rank r "
                              "holds experts [r*n/ranks, (r+1)*n/ranks), so naming those experts is the "
                              "same skew and survives rearrangement")
            if probe_family is None:
                errors.append("imbalance.method = 'router_bias' needs the expert count: set model.probe "
                              f"explicitly to one of {list(PROBE_CHOICES[1:])} for this model_id")
        if cfg["imbalance"]["bias_layers"] and cfg["imbalance"]["method"] != "router_bias":
            errors.append("imbalance.bias_layers applies to method = 'router_bias' only")
        if cfg["imbalance"]["bias_schedule"] and cfg["imbalance"]["method"] != "router_bias":
            errors.append("imbalance.bias_schedule applies to method = 'router_bias' only")
        if cfg["benchmark"]["workload"]:
            from .core.data import parse_workload
            try:
                parse_workload(cfg["benchmark"]["workload"])
            except ValueError as e:
                errors.append(f"benchmark.workload: {e}")
        if cfg["imbalance"]["validation_workload"]:
            from .core.data import parse_workload
            try:
                parse_workload(cfg["imbalance"]["validation_workload"])
            except ValueError as e:
                errors.append(f"imbalance.validation_workload: {e}")
            if cfg["imbalance"]["validation_samples"] == 0:
                errors.append("imbalance.validation_workload needs validation_samples > 0")
        if cfg["imbalance"]["validate_imbalance"] and probe_family is None:
            errors.append("imbalance.validate_imbalance requires model.probe to be set explicitly "
                          f"(one of {list(PROBE_CHOICES[1:])}) for this model_id")

    if errors:
        raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(errors))


#  TOML rendering
def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'"{escaped}"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"cannot render {type(value).__name__} as TOML")


def _toml_key(key: str) -> str:
    return key if re.match(r"^[A-Za-z0-9_-]+$", key) else _toml_value(key)


def to_toml(data: Mapping[str, Mapping[str, Any]], header: str | None = None) -> str:
    lines: list[str] = []
    if header:
        lines.extend(f"# {line}".rstrip() for line in header.splitlines())
        lines.append("")
    for section, values in data.items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            lines.append(f"{_toml_key(key)} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines)


def render_template(experiment_type: str) -> str:
    sections = sections_for(experiment_type)
    values = TEMPLATES[experiment_type]
    title = {
        rschema.EXPERIMENT_SYNTHETIC_WORKLOADS: "Synthetic workload sweep over alpha: expert activation capture ->\n"
        "workload construction -> benchmarking of each workload -> trace analysis.",
        rschema.EXPERIMENT_FORCED_IMBALANCE: "Forced router imbalance sweep: imbalanced checkpoint generation ->\n"
        "(optional) load validation -> benchmarking of each level -> trace analysis.",
    }[experiment_type]
    lines = [f"# {line}" for line in title.splitlines()]
    lines += ["#", "# Run with:  uv run moe-reliability run <this file>", ""]
    for section in sections:
        lines.append(f"# {section.help}")
        lines.append(f"[{section.name}]")
        if section.free_form:
            entries = {**_DEFAULT_ENVIRONMENT, **values.get(section.name, {})}
            for k, v in entries.items():
                lines.append(f"{_toml_key(k)} = {_toml_value(v)}")
            if not entries:
                lines.append('# HCCL_CONNECT_TIMEOUT = "1200"')
                lines.append('# TASK_QUEUE_ENABLE = "1"')
            lines.append("")
            continue
        for opt in section.options:
            value = values.get(section.name, {}).get(opt.key, opt.default)
            comment = opt.help
            if opt.choices:
                comment += f" Choices: {', '.join(map(str, opt.choices))}."
            lines.append(f"# {comment}")
            if value is REQUIRED:
                lines.append(f'{opt.key} = ""  # required')
            elif value is DERIVED:
                lines.append(f"# {opt.key} = true")
            else:
                lines.append(f"{opt.key} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines)


def _default_repr(opt: Option) -> str:
    if opt.default is REQUIRED:
        return "*required*"
    if opt.default is DERIVED:
        return "*derived*"
    return f"`{_toml_value(opt.default)}`"


# Markdown reference generation
def reference_markdown() -> str:
    lines = [
        "# Configuration reference",
        "",
        "A run is described by a single TOML file. Generate a commented template with",
        "`uv run moe-reliability init <experiment-type> -o my_run.toml`. Unknown sections or keys are",
        "rejected. Any value can be overridden on the command line with",
        "`--set section.key=value` (TOML value syntax).",
        "",
        "This page is generated by `uv run moe-reliability reference`.",
        "",
    ]
    for etype, sections in SCHEMA.items():
        lines += [f"## Experiment type `{etype}`", ""]
        for section in sections:
            lines += [f"### `[{section.name}]`", "", section.help, ""]
            if section.free_form:
                lines += ["Free-form table of environment variables (string values) set before the run starts and",
                          "inherited by the vLLM server, for example `HCCL_CONNECT_TIMEOUT`, `HCCL_BUFFSIZE` or",
                          "`TASK_QUEUE_ENABLE`. Select NPUs with `hardware.visible_devices` rather than",
                          "`ASCEND_RT_VISIBLE_DEVICES`. The table is stored in the run manifest, so do not put",
                          "credentials here.", ""]
                continue
            lines += ["| key | type | default | description |", "|---|---|---|---|"]
            for opt in section.options:
                desc = opt.help
                if opt.choices:
                    desc += " Choices: " + ", ".join(f"`{c}`" for c in opt.choices) + "."
                lines.append(f"| `{opt.key}` | {opt.kind} | {_default_repr(opt)} | {desc} |")
            lines.append("")
    lines += [
        "## Derived values",
        "",
        "`benchmark.save_request_metrics` defaults to `true` unless `benchmark.enable_profiling` is enabled,",
        "because the profiler perturbs request timings.",
        "",
    ]
    return "\n".join(lines)
