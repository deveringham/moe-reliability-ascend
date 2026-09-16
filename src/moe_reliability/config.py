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

from moe_results import schema as rschema

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
    Option("probe", "str", "auto", "Router probe family used to read MoE dimensions "
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
    Option("enable_expert_parallel", "bool", True, "Enable expert parallelism."),
    Option("enable_prefix_caching", "bool", False, "Enable prefix caching."),
    Option("enable_eplb", "bool", False, "Enable expert-parallel load balancing (EPLB)."),
))

_CLIENT = Section("client", "Load generation against the server.", (
    Option("max_new_tokens", "int", 100, "Maximum number of generated tokens per request."),
    Option("concurrency_limit", "int", 700, "Maximum number of in-flight requests "
           "(should exceed batch_size to saturate the server)."),
    Option("n_warmup_samples", "int", 10, "Number of warm-up requests before measurement."),
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
           "(converted to token budgets using the average tokens per prompt)."),
    Option("max_repeats", "list[int]", [0, 10], "Workload sets to build, one per maximum number of "
           "times a prompt may be repeated."),
    Option("reuse_workloads_from", "str", "", "Run id or run directory whose workloads are reused "
           "(skips activation capture and workload construction)."),
))

_BENCHMARK_SYNTHETIC = Section("benchmark", "Stage 3: benchmarking of each sweep point.", (
    Option("workload_max_repeats", "int", 0, "Which workload set (max_repeats) to benchmark."),
    Option("workload_prompt_length", "int", 1000, "Which workload size (target prompt length) to benchmark."),
    Option("enable_profiling", "bool", True, "Record PyTorch profiler traces on all workers."),
    Option("trace_active_iterations", "int", 2, "Number of profiled scheduler iterations."),
    Option("save_request_metrics", "bool", DERIVED, "Store per-request TTFT/TPOT measurements "
           "(default: true unless profiling, which perturbs timings)."),
))

_IMBALANCE = Section("imbalance", "Forced router imbalance.", (
    Option("imbalance_levels", "list[number]", [0, 100], "Router bias added to expert 0 in every layer; "
           "0 serves the unmodified model."),
    Option("model_dir", "str", "models", "Directory for generated imbalanced checkpoints "
           "(reused across runs when present)."),
    Option("validate_imbalance", "bool", False, "Measure expert load with Hugging Face inference before "
           "benchmarking each checkpoint."),
))

_BENCHMARK_FORCED = Section("benchmark", "Benchmarking of each imbalance level.", (
    Option("n_samples", "int", 15000, "Number of MMLU prompts sent to each checkpoint."),
    Option("enable_profiling", "bool", False, "Record PyTorch profiler traces on all workers."),
    Option("trace_active_iterations", "int", 2, "Number of profiled scheduler iterations."),
    Option("save_request_metrics", "bool", DERIVED, "Store per-request TTFT/TPOT measurements "
           "(default: true unless profiling, which perturbs timings)."),
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

    def flat(self) -> dict[str, Any]:
        return {k: v for name, sec in self._data.items() if name != "environment" for k, v in sec.items()}

    def to_toml(self, header: str | None = None) -> str:
        return to_toml(self._data, header=header)


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
        bench["save_request_metrics"] = not bench["enable_profiling"]


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
    for key in ("max_model_len", "batch_size"):
        positive("server", key)
    if not 0 < cfg["server"]["gpu_memory_utilization"] <= 1:
        errors.append("server.gpu_memory_utilization must be in (0, 1]")
    if not 1 <= cfg["server"]["port"] <= 65535:
        errors.append("server.port must be in [1, 65535]")
    positive("client", "max_new_tokens")
    positive("client", "concurrency_limit")
    positive("client", "n_warmup_samples", allow_zero=True)
    positive("benchmark", "trace_active_iterations")

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
        if not levels or any(level < 0 for level in levels):
            errors.append("imbalance.imbalance_levels must be a non-empty list of numbers >= 0")
        if len(set(levels)) != len(levels):
            errors.append("imbalance.imbalance_levels contains duplicates")
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
    lines += ["#", "# Run with:  uv run moe-experiments run <this file>", ""]
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
        "`uv run moe-experiments init <experiment-type> -o my_run.toml`. Unknown sections or keys are",
        "rejected. Any value can be overridden on the command line with",
        "`--set section.key=value` (TOML value syntax).",
        "",
        "This page is generated by `uv run moe-experiments reference`.",
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
