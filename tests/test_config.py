from __future__ import annotations

import tomllib

import pytest

from moe_experiments.config import (DERIVED, SCHEMA, TEMPLATES, ConfigError, ExperimentConfig, apply_overrides,
                                    reference_markdown, render_template)
from conftest import write_toml


def test_defaults_and_attribute_access(forced_config_data):
    cfg = ExperimentConfig.from_dict(forced_config_data)
    assert cfg.experiment_type == "forced_imbalance"
    assert cfg.server.batch_size == 16
    assert cfg.server.max_model_len == 2048 and cfg.server.gpu_memory_utilization == 0.6
    assert cfg.experiment.seed == 43
    assert cfg.environment == {}
    assert cfg.hardware.n_npus == 2 and cfg.hardware.visible_devices == ""
    assert cfg.probe_family == "mistral"
    assert cfg.benchmark.save_request_metrics is True
    with pytest.raises(AttributeError):
        cfg.server.batch_size = 1


def test_keys_are_unique_within_each_experiment_type():
    for sections in SCHEMA.values():
        keys = [o.key for s in sections for o in s.options]
        assert len(keys) == len(set(keys))


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d["server"].update(batch_size="big"), "expected int"),
    (lambda d: d["server"].update(batch_sizes=4), "unknown key"),
    (lambda d: d.update(workloads={}), "unknown section"),
    (lambda d: d["model"].pop("model_id"), "missing required key model.model_id"),
    (lambda d: d["imbalance"].update(imbalance_levels=[0, 0]), "duplicates"),
    (lambda d: d["server"].update(gpu_memory_utilization=1.5), "gpu_memory_utilization"),
    (lambda d: d["model"].update(model_name="bad name"), "model_name"),
    (lambda d: d["hardware"].update(platform="npu"), "unknown key"),
    (lambda d: d["hardware"].update(n_gpus=8), "unknown key"),
    (lambda d: d["hardware"].update(visible_devices="0,1,a"), "visible_devices"),
    (lambda d: d["hardware"].update(visible_devices="0"), "exposes 1 NPUs"),
    (lambda d: d["model"].update(enable_bnb=True), "not supported by vLLM Ascend"),
    (lambda d: d["model"].update(probe="native"), "must be one of"),
    (lambda d: d["experiment"].update(type="other"), "experiment.type"),
])
def test_validation_errors(forced_config_data, mutate, message):
    mutate(forced_config_data)
    with pytest.raises(ConfigError, match=message):
        ExperimentConfig.from_dict(forced_config_data)


def test_synthetic_validation(synthetic_config_data):
    synthetic_config_data["benchmark"]["workload_prompt_length"] = 999
    with pytest.raises(ConfigError, match="workload_prompt_length"):
        ExperimentConfig.from_dict(synthetic_config_data)
    synthetic_config_data["workloads"]["reuse_workloads_from"] = "some-run"
    ExperimentConfig.from_dict(synthetic_config_data)  # checked against the reused workloads at run time


def test_unknown_probe_family_requires_explicit_probe(synthetic_config_data):
    synthetic_config_data["model"]["model_id"] = "org/unknown-moe"
    with pytest.raises(ConfigError, match="model.probe"):
        ExperimentConfig.from_dict(synthetic_config_data)
    synthetic_config_data["model"]["probe"] = "qwen"
    assert ExperimentConfig.from_dict(synthetic_config_data).probe_family == "qwen"


def test_overrides(forced_config_data):
    cfg = ExperimentConfig.from_dict(forced_config_data, [
        "server.batch_size=256", "imbalance.imbalance_levels=[0, 50, 100]", "model.model_name=qwen",
        "benchmark.enable_profiling=true",
    ])
    assert cfg.server.batch_size == 256
    assert cfg.imbalance.imbalance_levels == [0, 50, 100]
    assert cfg.model.model_name == "qwen"
    assert cfg.benchmark.save_request_metrics is False  # derived from profiling
    with pytest.raises(ConfigError, match="section.key=value"):
        apply_overrides({}, ["batch_size=3"])


def test_toml_round_trip(tmp_path, synthetic_config_data):
    cfg = ExperimentConfig.from_dict(synthetic_config_data)
    path = tmp_path / "resolved.toml"
    path.write_text(cfg.to_toml(header="resolved"))
    again = ExperimentConfig.load(path)
    assert again.to_dict() == cfg.to_dict()
    assert DERIVED not in again.flat().values()


def test_load_errors(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        ExperimentConfig.load(tmp_path / "missing.toml")
    bad = tmp_path / "bad.toml"
    bad.write_text("[server\n")
    with pytest.raises(ConfigError, match="invalid TOML"):
        ExperimentConfig.load(bad)


@pytest.mark.parametrize("etype", list(TEMPLATES))
def test_templates_are_valid(tmp_path, etype):
    path = tmp_path / "template.toml"
    path.write_text(render_template(etype))
    cfg = ExperimentConfig.load(path)
    assert cfg.experiment_type == etype
    assert set(tomllib.loads(render_template(etype))) == {s.name for s in SCHEMA[etype]}


def test_reference_mentions_every_key():
    text = reference_markdown()
    for sections in SCHEMA.values():
        for section in sections:
            assert f"`[{section.name}]`" in text
            for opt in section.options:
                assert f"`{opt.key}`" in text


def test_write_toml_helper(tmp_path, forced_config_data):
    path = write_toml(tmp_path / "c.toml", forced_config_data)
    assert ExperimentConfig.load(path).server.batch_size == 16
