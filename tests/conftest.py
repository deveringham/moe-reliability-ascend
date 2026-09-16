# Exercises the pipelines end-to-end without any NPUs necessary.

from __future__ import annotations

import gzip
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "packages" / "moe-results" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

N_EXPERTS, N_LAYERS, TOP_K = 8, 4, 2
SUBJECTS = ["algebra", "biology", "chemistry", "law"]

def fake_mmlu_prompts(n_samples: int, seed: int):
    prompts = [[{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Question {i}?"}] for i in range(n_samples)]
    subjects = [SUBJECTS[i % len(SUBJECTS)] for i in range(n_samples)]
    questions = [f"Question {i}?" for i in range(n_samples)]
    return prompts, subjects, questions


def write_rank_trace(trace_dir: Path, rank: int, n_calls: int, mean_us: float, rng: np.random.Generator) -> None:
    events = []
    t = 0
    for step in range(n_calls):
        events.append({"ph": "X", "cat": "kernel", "name": "ncclDevKernel_AllGather_RING_LL", "ts": t, "dur": 50})
        t += 100
        dur = int(max(1, rng.normal(mean_us, mean_us * 0.05)))
        events.append({"ph": "X", "cat": "kernel", "name": "fused_moe_kernel", "ts": t, "dur": dur,
                       "args": {"grid": [16 if step % 3 else 32, 1, 1]}})
        t += dur + 10
    doc = {"distributedInfo": {"rank": rank}, "deviceProperties": [{"id": rank}], "traceEvents": events}
    with gzip.open(trace_dir / f"host_worker_rank{rank}.1.pt.trace.json.gz", "wt") as f:
        json.dump(doc, f)


def write_npu_profiler_data(trace_dir: Path, rank: int) -> None:
    worker = trace_dir / f"worker_{rank}" / "FRAMEWORK"
    worker.mkdir(parents=True, exist_ok=True)
    (worker / "torch.op_range").write_bytes(b"\x00raw")


def fake_npu_analyse(profiler_path: str, max_process_number: int = 1, export_type=None) -> None:
    root = Path(profiler_path)
    if (root / "FRAMEWORK").is_dir():
        workers = [root]
    else:
        workers = [p for p in root.iterdir() if (p / "FRAMEWORK").is_dir()]
    for worker in workers:
        out = worker / "ASCEND_PROFILER_OUTPUT"
        out.mkdir(exist_ok=True)
        (out / "trace_view.json").write_text(json.dumps([{"name": "aclnnMoeInitRouting", "ts": 0, "dur": 10}]))


class FakeDeployment:

    def __init__(self, n_ranks: int = 2, fail_models: tuple[str, ...] = (), trace_format: str = "npu"):
        self.calls: list[dict] = []
        self.n_ranks = n_ranks
        self.fail_models = set(fail_models)
        self.trace_format = trace_format

    def __call__(self, cfg, model_path, prompts, trace_dir, enable_expert_capture=False):
        rng = np.random.default_rng(len(self.calls) + 1)
        self.calls.append({"model_path": model_path, "n_prompts": len(prompts), "trace_dir": trace_dir,
                           "capture": enable_expert_capture, "batch_size": cfg.server.batch_size})
        if model_path in self.fail_models:
            return None  # measure_vllm_throughput returns None when inference fails
        slowdown = 1.0 + (3.0 if "imbalance" in str(model_path) else 0.0)
        results = []
        for i, prompt in enumerate(prompts):
            n_in, n_out = int(rng.integers(8, 30)), int(rng.integers(4, 12))
            record = {"prompt": prompt, "prompt_id": i, "num_output_tokens": n_out, "num_input_tokens": n_in,
                      "total_time": float(rng.uniform(0.5, 1.5) * slowdown)}
            if enable_expert_capture:
                # routing skewed towards an expert that depends on the prompt
                hot = i % N_EXPERTS
                probs = np.full(N_EXPERTS, 1.0)
                probs[hot] = 6.0
                probs /= probs.sum()

                def draw(n_tokens):
                    return np.stack([np.stack([rng.choice(N_EXPERTS, TOP_K, replace=False, p=probs)
                                               for _ in range(N_LAYERS)]) for _ in range(n_tokens)]).astype(np.int16)

                record.update(ttft=None, tpot=None, routed_experts=draw(n_out), prompt_routed_experts=draw(n_in))
            else:
                record.update(ttft=float(rng.uniform(0.02, 0.05) * slowdown),
                              tpot=float(rng.uniform(0.01, 0.02) * slowdown))
            results.append(record)
        if trace_dir:
            for rank in range(self.n_ranks):
                if self.trace_format == "npu":
                    write_npu_profiler_data(Path(trace_dir), rank)
                else:
                    hot_rank = 400.0 * slowdown if rank == 0 else 400.0
                    write_rank_trace(Path(trace_dir), rank, n_calls=12, mean_us=hot_rank, rng=rng)
        return results


@pytest.fixture
def deployment(monkeypatch):
    from moe_experiments import models
    from moe_experiments.pipelines import common

    fake = FakeDeployment()
    monkeypatch.setattr(common, "serve_and_measure", fake)
    monkeypatch.setattr(common, "mmlu_prompts", fake_mmlu_prompts)
    monkeypatch.setattr(common, "free_accelerator_memory", lambda: None)
    monkeypatch.setattr(models, "moe_dimensions", lambda model_id, family: (N_EXPERTS, N_LAYERS, TOP_K))

    # Checkpoint generation: write a marker directory instead of a model.
    module = types.ModuleType("moe_experiments.core.forced_imbalance")
    module.created = []

    def imbalance_pretrained_moe(model_id, imbalance_level, save_path):
        Path(save_path).mkdir(parents=True)
        (Path(save_path) / "config.json").write_text(json.dumps({"base": model_id, "level": imbalance_level}))
        module.created.append(save_path)

    module.imbalance_pretrained_moe = imbalance_pretrained_moe
    monkeypatch.setitem(sys.modules, "moe_experiments.core.forced_imbalance", module)
    fake.checkpoints = module
    install_fake_npu_profiler(monkeypatch)
    return fake


def install_fake_npu_profiler(monkeypatch, analyse=fake_npu_analyse):
    """Register ``torch_npu.profiler.profiler.analyse`` without an Ascend runtime."""
    for name in ("torch_npu", "torch_npu.profiler"):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    profiler = types.ModuleType("torch_npu.profiler.profiler")
    profiler.analyse = analyse
    monkeypatch.setitem(sys.modules, "torch_npu.profiler.profiler", profiler)
    return profiler

@pytest.fixture
def results_dir(tmp_path):
    return tmp_path / "results"


@pytest.fixture
def forced_config_data(tmp_path, results_dir):
    return {
        "experiment": {"type": "forced_imbalance", "name": "test"},
        "model": {"model_id": "org/Mixtral-test", "model_name": "mixtral"},
        "hardware": {"n_npus": 2},
        "server": {"batch_size": 16},
        "client": {"concurrency_limit": 32, "n_warmup_samples": 2},
        "imbalance": {"imbalance_levels": [0, 100], "model_dir": str(tmp_path / "models")},
        "benchmark": {"n_samples": 24},
        "output": {"results_dir": str(results_dir)},
    }


@pytest.fixture
def synthetic_config_data(results_dir):
    return {
        "experiment": {"type": "synthetic_workloads", "name": "test"},
        "model": {"model_id": "org/deepseek-test", "model_name": "deepseek-test"},
        "hardware": {"n_npus": 2},
        "server": {"batch_size": 16},
        "client": {"concurrency_limit": 32, "n_warmup_samples": 2},
        "activations": {"n_samples": 60},
        "workloads": {"target_alphas": [0.5, 1.0, 1.5], "target_prompt_lengths": [8, 16], "max_repeats": [0, 2]},
        "benchmark": {"workload_max_repeats": 0, "workload_prompt_length": 8, "enable_profiling": True},
        "output": {"results_dir": str(results_dir)},
    }


def write_toml(path: Path, data: dict) -> Path:
    from moe_experiments.config import to_toml

    path.write_text(to_toml(data), encoding="utf-8")
    return path
