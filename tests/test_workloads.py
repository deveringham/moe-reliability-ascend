"""Workload families: the traffic a detector's benign floor has to survive.

The loaders themselves stream from Hugging Face, so the tests here drive them
through a stub stream and check the parsing, the labelling and the interleaving.
"""

from __future__ import annotations

import gzip
import json

import pytest

from moe_reliability.config import ConfigError, ExperimentConfig
from moe_reliability.core import data
from moe_reliability.pipelines import run_pipeline
from moe_reliability.runs import RunContext
from moe_reliability_results import schema

# One row shape per dataset; MMMLU serves every language config from the same shape.
ROWS = {
    "cais/mmlu": {"question": "q", "choices": ["a", "b", "c", "d"], "subject": "law"},
    "openai/gsm8k": {"question": "q", "answer": "a"},
    "google-research-datasets/mbpp": {"text": "write f", "test_list": ["assert f()"]},
    "openai/MMMLU": {"Question": "q", "A": "a", "B": "b", "C": "c", "D": "d", "Subject": "law"},
    "databricks/databricks-dolly-15k": {"instruction": "do it", "context": "", "category": "open_qa"},
    "HuggingFaceH4/ultrachat_200k": {"prompt": "hello"},
}


@pytest.fixture
def stub_stream(monkeypatch):
    seen = []

    def stream(dataset_id, config, split, n, seed):
        seen.append((dataset_id, config, split, n, seed))
        return [dict(ROWS[dataset_id], prompt_id=i) for i in range(n)]

    monkeypatch.setattr(data, "_stream", stream)
    return seen


@pytest.mark.parametrize("spec, expected", [
    ("gsm8k", [("gsm8k", None)]),
    ("mmmlu:ZH_CN", [("mmmlu", "ZH_CN")]),
    ("mixed:mmlu,gsm8k", [("mmlu", None), ("gsm8k", None)]),
    ("mixed:mmlu,mmmlu:DE_DE", [("mmlu", None), ("mmmlu", "DE_DE")]),
])
def test_workload_specs_parse(spec, expected):
    assert data.parse_workload(spec) == expected


@pytest.mark.parametrize("spec, message", [
    ("nosuch", "unknown family"),
    ("mixed:mmlu,nosuch", "unknown family"),
    ("gsm8k:main", "only mmmlu takes an argument"),
])
def test_bad_workload_specs_are_rejected(spec, message):
    with pytest.raises(ValueError, match=message):
        data.parse_workload(spec)


@pytest.mark.parametrize("family", data.WORKLOAD_FAMILIES)
def test_every_family_yields_chat_messages_and_a_label(family, stub_stream):
    spec = "mmmlu:ZH_CN" if family == "mmmlu" else family
    prompts, labels = data.workload_prompts(spec, 3, seed=1)
    assert len(prompts) == len(labels) == 3
    for messages in prompts:
        assert [m["role"] for m in messages] == ["system", "user"]
        assert messages[-1]["content"].strip()
    # The label names the family, so a mixed capture can be split by family later.
    assert all(label.startswith(spec + "/") for label in labels)


def test_mixed_workloads_interleave_so_any_window_holds_every_family(stub_stream):
    prompts, labels = data.workload_prompts("mixed:mmlu,gsm8k,ultrachat", 9, seed=1)
    assert len(prompts) == 9
    families = [label.split("/")[0] for label in labels]
    assert families[:3] == ["mmlu", "gsm8k", "ultrachat"]
    assert sorted(families) == sorted(["mmlu", "gsm8k", "ultrachat"] * 3)


def test_mixed_workloads_split_an_uneven_count_without_losing_prompts(stub_stream):
    prompts, labels = data.workload_prompts("mixed:mmlu,gsm8k", 7, seed=1)
    assert len(prompts) == len(labels) == 7
    counts = {f: sum(label.split("/")[0] == f for label in labels) for f in ("mmlu", "gsm8k")}
    assert sorted(counts.values()) == [3, 4]


def test_each_family_is_streamed_shuffled_from_its_own_split(stub_stream):
    data.workload_prompts("mixed:gsm8k,mmmlu:DE_DE", 4, seed=7)
    assert stub_stream == [("openai/gsm8k", "main", "test", 2, 7),
                           ("openai/MMMLU", "DE_DE", "test", 2, 7)]


# --- Validation on a chosen workload, with per-request routing kept ----------

@pytest.fixture
def workload_config(forced_config_data):
    data_ = forced_config_data
    data_["model"]["probe"] = "deepseek"
    data_["hardware"]["n_npus"] = 4
    data_["imbalance"].update(method="router_bias", bias_target="rank:0", imbalance_levels=[0, 4],
                              validate_imbalance=True, validation_samples=8,
                              validation_workload="mixed:mmlu,gsm8k", validation_save_records=True)
    return data_


def test_validation_can_draw_from_a_workload_and_keep_per_request_routing(
        deployment, workload_config, stub_stream, tmp_path):
    cfg = ExperimentConfig.from_dict(workload_config)
    ctx = RunContext.create(cfg)
    assert run_pipeline(ctx, cfg) == schema.STATUS_COMPLETED

    point = next(p for p in ctx.points if p["value"] == 4)
    summary = json.load(gzip.open(ctx.abspath(point["validation_file"])))
    assert summary["validation_workload"] == "mixed:mmlu,gsm8k"
    assert summary["records_file"] == schema.validation_records_file(point["label"]) + ".gz"

    records = [json.loads(line) for line in gzip.open(ctx.abspath(summary["records_file"]), "rt")]
    assert len(records) == 8
    # Per-request routing is what a detector windows over; the pooled counts cannot
    # give a sampling distribution.
    assert all(r["prompt_routed_experts"] and r["routed_experts"] for r in records)
    assert {r["subject"].split("/")[0] for r in records} == {"mmlu", "gsm8k"}


def test_validation_workload_is_checked(workload_config):
    workload_config["imbalance"]["validation_workload"] = "nosuch"
    with pytest.raises(ConfigError, match="unknown family"):
        ExperimentConfig.from_dict(workload_config)

    workload_config["imbalance"]["validation_workload"] = "gsm8k"
    workload_config["imbalance"]["validation_samples"] = 0
    with pytest.raises(ConfigError, match="needs validation_samples"):
        ExperimentConfig.from_dict(workload_config)


def test_bias_layers_are_checked(workload_config):
    workload_config["imbalance"]["bias_layers"] = [1, 1]
    with pytest.raises(ConfigError, match="duplicates"):
        ExperimentConfig.from_dict(workload_config)

    workload_config["imbalance"]["bias_layers"] = [2]
    workload_config["imbalance"]["method"] = "checkpoint"
    with pytest.raises(ConfigError, match="router_bias"):
        ExperimentConfig.from_dict(workload_config)
