from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

from moe_reliability import router_bias as RB
from moe_reliability.config import ConfigError, ExperimentConfig
from moe_reliability.pipelines import run_pipeline
from moe_reliability.runs import RunContext
from moe_reliability_results import schema


def test_targets_resolve_to_contiguous_rank_blocks_or_explicit_experts():
    assert RB.target_experts("rank:0", 64, 4) == list(range(16))
    assert RB.target_experts("rank:3", 8, 4) == [6, 7]  # Mixtral at 4-way EP
    assert RB.target_experts("experts:5,1,1", 8, 4) == [1, 5]


@pytest.mark.parametrize("spec, message", [
    ("rank:4", "out of range"),
    ("experts:8", "out of range"),
    ("rank:0,1", "exactly one rank"),
    ("ranks:0", "expected"),
    ("experts:a", "integers"),
])
def test_bad_targets_are_rejected(spec, message):
    with pytest.raises(ValueError, match=message):
        RB.target_experts(spec, 8, 4)


def test_rank_target_needs_experts_to_divide_over_ranks():
    with pytest.raises(ValueError, match="divide"):
        RB.target_experts("rank:0", 10, 4)


def test_bias_vector_and_server_env():
    assert RB.bias_vector("rank:1", 2.5, 8, 4) == [0, 0, 2.5, 2.5, 0, 0, 0, 0]
    assert RB.server_env("rank:1", 0, 8, 4) == {}  # level 0 serves the model untouched
    env = RB.server_env("rank:1", 2.5, 8, 4)
    assert json.loads(env[RB.ENV_VAR]) == [0, 0, 2.5, 2.5, 0, 0, 0, 0]


def test_wrapper_adds_the_bias_to_router_logits_only():
    torch = pytest.importorskip("torch")
    seen = {}

    def select(hidden_states, router_logits, top_k):
        seen["logits"] = router_logits
        return router_logits.topk(top_k, dim=-1).indices

    biased = RB._wrap(select, [0.0, 0.0, 5.0, 5.0])
    logits = torch.tensor([[1.0, 2.0, 0.0, -1.0]])
    ids = biased(hidden_states=None, router_logits=logits, top_k=2)
    assert sorted(ids[0].tolist()) == [2, 3]
    assert torch.equal(seen["logits"], logits + torch.tensor([0.0, 0.0, 5.0, 5.0]))
    with pytest.raises(ValueError, match="4 entries"):
        biased(hidden_states=None, router_logits=torch.zeros(1, 8), top_k=2)
    with pytest.raises(TypeError, match="keyword"):
        biased(None, logits, 2)  # positional logits would bypass the bias silently


@pytest.fixture
def fake_selector(monkeypatch):
    """A stand-in for vllm_ascend.ops.fused_moe.experts_selector."""
    sel = types.ModuleType("vllm_ascend.ops.fused_moe.experts_selector")
    sel._select_experts_with_fusion_ops = lambda **kw: "fused"
    sel._native_select_experts = lambda **kw: "native"
    pkg = types.ModuleType("vllm_ascend.ops.fused_moe")
    pkg.experts_selector = sel
    for name, mod in {"vllm_ascend": types.ModuleType("vllm_ascend"),
                      "vllm_ascend.ops": types.ModuleType("vllm_ascend.ops"),
                      "vllm_ascend.ops.fused_moe": pkg,
                      "vllm_ascend.ops.fused_moe.experts_selector": sel}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return sel


def test_register_is_a_no_op_without_a_bias(fake_selector, monkeypatch):
    monkeypatch.delenv(RB.ENV_VAR, raising=False)
    before = fake_selector._native_select_experts
    RB.register()
    assert fake_selector._native_select_experts is before


def test_register_patches_both_selection_paths_once(fake_selector, monkeypatch):
    monkeypatch.setenv(RB.ENV_VAR, json.dumps([0.0, 1.0]))
    RB.register()
    RB.register()  # vLLM may load a plugin more than once per process
    for name in ("_select_experts_with_fusion_ops", "_native_select_experts"):
        fn = getattr(fake_selector, name)
        assert fn.__wrapped__ is not None
        assert getattr(fn.__wrapped__, "__wrapped__", None) is None  # not wrapped twice


@pytest.fixture
def router_bias_config(forced_config_data):
    data = forced_config_data
    data["model"]["probe"] = "deepseek"
    data["hardware"]["n_npus"] = 4
    data["imbalance"].update(method="router_bias", bias_target="rank:0", imbalance_levels=[0, 1.5, 4],
                             validate_imbalance=True, validation_samples=40)
    return data


def test_router_bias_sweep_serves_the_unmodified_model_with_a_graded_skew(deployment, router_bias_config, tmp_path):
    cfg = ExperimentConfig.from_dict(router_bias_config)
    ctx = RunContext.create(cfg)
    assert run_pipeline(ctx, cfg) == schema.STATUS_COMPLETED

    # no checkpoints: every server ran the original model
    assert {c["model_path"] for c in deployment.calls} == {cfg.model.model_id}
    assert not any("imbalance" in str(c["model_path"]) for c in deployment.calls)

    by_level = {p["value"]: p for p in ctx.points}
    assert by_level[0]["server_env"] == {}
    biased = json.loads(by_level[4]["server_env"][RB.ENV_VAR])
    assert biased[:2] == [4.0, 4.0] and biased[2:] == [0.0] * 6  # 8 experts, rank 0 holds 0-1

    # the env reaches the server in both validation and benchmarking
    envs = [c["server_env"] for c in deployment.calls]
    assert envs.count(by_level[4]["server_env"]) == 2

    # validation measures a share on rank 0 that rises with the level
    share = {v: p["validation_summary"]["rank_share_mean"][0] for v, p in by_level.items()}
    assert share[0] < share[1.5] < share[4]
    assert by_level[4]["validation_summary"]["rank_max_over_mean"] > 2.0


def test_router_bias_config_is_checked(router_bias_config):
    router_bias_config["imbalance"]["bias_target"] = "rank:4"
    with pytest.raises(ConfigError, match="rank 4"):
        ExperimentConfig.from_dict(router_bias_config)
    router_bias_config["imbalance"]["bias_target"] = "rank:0"
    router_bias_config["server"]["enable_eplb"] = True
    with pytest.raises(ConfigError, match="contiguous expert placement"):
        ExperimentConfig.from_dict(router_bias_config)


def test_moe_layers_skip_dense_placeholders():
    from moe_reliability.pipelines.forced_imbalance import _moe_layers, rank_load

    ids = np.zeros((5, 3, 2), dtype=np.int16)
    ids[:, 1:, :] = [[1, 3]]
    assert _moe_layers([{"routed_experts": ids, "prompt_routed_experts": None}]) == [1, 2]
    r = rank_load([[0.5, 0.5, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]], n_ranks=2)
    assert r["rank_share_mean"] == [0.75, 0.25] and r["rank_max_over_mean"] == 1.5
