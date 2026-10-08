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
    assert RB.parse_env(env[RB.ENV_VAR]).vector(None, 8) == [0, 0, 2.5, 2.5, 0, 0, 0, 0]


def test_zero_bias_control_installs_the_plugin_with_an_all_zero_vector():
    # The instrument is not free: without this, a level-0 point serves without the
    # wrapper and so pays none of its per-call cost, which lands in the comparison.
    assert RB.parse_env(RB.server_env("rank:1", 0, 8, 4, at_zero=True)[RB.ENV_VAR]).vector(None, 8) == [0.0] * 8
    assert RB.server_env("rank:1", 0, 8, 4) == {}
    # A nonzero level is unaffected by the flag.
    assert RB.server_env("rank:1", 2.5, 8, 4, at_zero=True) == RB.server_env("rank:1", 2.5, 8, 4)


def test_zero_bias_control_wraps_and_leaves_routing_unchanged():
    torch = pytest.importorskip("torch")
    seen = {}

    def select(hidden_states, router_logits, top_k):
        seen["logits"] = router_logits
        return router_logits.topk(top_k, dim=-1).indices

    biased = RB._wrap(select, RB.Bias(strength=0.0, n_ranks=2, target="rank:0"))
    logits = torch.tensor([[1.0, 2.0, 0.0, -1.0]])
    ids = biased(hidden_states=None, router_logits=logits, top_k=2)
    assert sorted(ids[0].tolist()) == [0, 1]  # the unbiased choice
    assert torch.equal(seen["logits"], logits)  # same values...
    assert seen["logits"] is not logits  # ...but the add still ran, which is the point


def test_wrapper_adds_the_bias_to_router_logits_only():
    torch = pytest.importorskip("torch")
    seen = {}

    def select(hidden_states, router_logits, top_k):
        seen["logits"] = router_logits
        return router_logits.topk(top_k, dim=-1).indices

    biased = RB._wrap(select, RB.Bias(strength=5.0, n_ranks=2, target="rank:1"))
    logits = torch.tensor([[1.0, 2.0, 0.0, -1.0]])
    ids = biased(hidden_states=None, router_logits=logits, top_k=2)
    assert sorted(ids[0].tolist()) == [2, 3]
    assert torch.equal(seen["logits"], logits + torch.tensor([0.0, 0.0, 5.0, 5.0]))
    # The vector is built from the logits' own width, so a router with a different
    # expert count gets a correctly sized bias rather than a shape error.
    wide = biased(hidden_states=None, router_logits=torch.zeros(1, 8), top_k=2)
    # Rank 1 of 2 holds experts 4-7; which two of them win a tie is torch's business.
    assert set(wide[0].tolist()) <= {4, 5, 6, 7}
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
    monkeypatch.setenv(RB.ENV_VAR, json.dumps({"strength": 1.0, "n_ranks": 2, "target": "rank:1"}))
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
    biased = RB.parse_env(by_level[4]["server_env"][RB.ENV_VAR]).vector(None, 8)
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


# --- Layer-targeted bias -----------------------------------------------------

def test_layer_target_travels_with_the_layers_it_applies_to():
    plain = RB.parse_env(RB.server_env("rank:1", 2.5, 8, 4)[RB.ENV_VAR])
    assert plain.layers is None and plain.vector(None, 8) == [0, 0, 2.5, 2.5, 0, 0, 0, 0]

    env = RB.parse_env(RB.server_env("rank:1", 2.5, 8, 4, layers=[5, 1, 1])[RB.ENV_VAR])
    assert env.layers == {1, 5}
    assert env.applies_to(1) and not env.applies_to(2)

    # The control pays the same per-call cost on the same layers.
    zero = RB.parse_env(RB.server_env("rank:1", 0, 8, 4, at_zero=True, layers=[1, 5])[RB.ENV_VAR])
    assert zero.vector(1, 8) == [0.0] * 8 and zero.layers == {1, 5}
    assert RB.server_env("rank:1", 0, 8, 4, layers=[1, 5]) == {}  # still no plugin without at_zero


def test_layer_index_is_read_from_the_layer_name():
    assert RB.layer_index("model.layers.3.mlp.experts") == 3
    assert RB.layer_index("model.layers.27.mlp") == 27
    assert RB.layer_index("model.embed_tokens") is None


def test_biased_layers_get_the_offset_and_the_others_are_untouched(monkeypatch):
    torch = pytest.importorskip("torch")
    seen = []

    def select(hidden_states, router_logits, top_k):
        seen.append(router_logits)
        return router_logits.topk(top_k, dim=-1).indices

    biased = RB._wrap(select, RB.Bias(strength=5.0, n_ranks=2, target="rank:1", layers=frozenset({1})))
    logits = torch.tensor([[1.0, 2.0, 0.0, -1.0]])

    monkeypatch.setattr(RB, "_current_layer", [1])
    ids = biased(hidden_states=None, router_logits=logits, top_k=2)
    assert sorted(ids[0].tolist()) == [2, 3]  # the bias won

    monkeypatch.setattr(RB, "_current_layer", [0])
    ids = biased(hidden_states=None, router_logits=logits, top_k=2)
    assert sorted(ids[0].tolist()) == [0, 1]  # untouched layer routes as it would unbiased
    assert seen[-1] is logits  # and pays nothing: no add ran


def test_layer_targeting_refuses_to_guess_when_the_layer_is_unknown(monkeypatch):
    torch = pytest.importorskip("torch")
    biased = RB._wrap(lambda **kw: None, RB.Bias(strength=5.0, n_ranks=2, target="rank:1",
                                                 layers=frozenset({1})))
    monkeypatch.setattr(RB, "_current_layer", [None])
    with pytest.raises(RuntimeError, match="outside a tracked MoE layer"):
        biased(hidden_states=None, router_logits=torch.zeros(1, 2), top_k=1)


def test_tracker_records_the_layer_in_progress_and_restores_it():
    layers = []

    class Runner:
        def forward_impl(self, layer, hidden_states):
            layers.append(RB._current_layer[0])
            return hidden_states

    class Layer:
        def __init__(self, name):
            self.layer_name = name

    Runner.forward_impl = RB._track_layer(Runner.forward_impl)
    runner = Runner()
    runner.forward_impl(Layer("model.layers.2.mlp.experts"), "x")
    runner.forward_impl(Layer("model.layers.7.mlp.experts"), "x")
    assert layers == [2, 7]
    assert RB._current_layer[0] is None  # restored, so a failed forward cannot leak a layer


def test_register_patches_the_runner_only_for_a_layer_target(fake_selector, monkeypatch):
    runner = types.SimpleNamespace(forward_impl=lambda self, layer: None)
    module = types.ModuleType("vllm_ascend.ops.fused_moe.fused_moe")
    module.AscendMoERunner = runner
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.fused_moe.fused_moe", module)

    monkeypatch.setenv(RB.ENV_VAR, json.dumps({"strength": 1.0, "n_ranks": 2, "target": "rank:1"}))
    RB.register()
    assert getattr(runner.forward_impl, "__wrapped__", None) is None  # whole-model bias needs no tracking

    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.fused_moe.experts_selector", fake_selector)
    fake_selector._select_experts_with_fusion_ops = lambda **kw: "fused"
    fake_selector._native_select_experts = lambda **kw: "native"
    monkeypatch.setenv(RB.ENV_VAR, json.dumps({"strength": 1.0, "n_ranks": 2, "target": "rank:1",
                                               "layers": [3]}))
    RB.register()
    RB.register()  # idempotent
    assert runner.forward_impl.__wrapped__ is not None
    assert getattr(runner.forward_impl.__wrapped__, "__wrapped__", None) is None


def test_layer_targeted_sweep_skews_only_the_named_layers(deployment, router_bias_config, tmp_path):
    router_bias_config["imbalance"]["bias_layers"] = [1, 2]
    router_bias_config["imbalance"]["imbalance_levels"] = [0, 4]
    cfg = ExperimentConfig.from_dict(router_bias_config)
    ctx = RunContext.create(cfg)
    assert run_pipeline(ctx, cfg) == schema.STATUS_COMPLETED

    by_level = {p["value"]: p for p in ctx.points}
    assert RB.parse_env(by_level[4]["server_env"][RB.ENV_VAR]).layers == {1, 2}

    import gzip as _gzip

    load = {}
    for value, point in by_level.items():
        f = json.load(_gzip.open(ctx.abspath(point["validation_file"])))
        load[value] = f["rank_max_over_mean_per_layer"]
    # Layers 1 and 2 carry the skew; the layers either side of them do not.
    assert load[4][1] > load[0][1] * 1.3 and load[4][2] > load[0][2] * 1.3
    assert load[4][0] == pytest.approx(load[0][0], rel=0.25)
    assert load[4][3] == pytest.approx(load[0][3], rel=0.25)


# --- A rotating skew: every layer hot, no rank hot throughout ----------------

def test_rotate_biases_a_different_rank_in_each_layer():
    bias = RB.parse_env(RB.server_env(RB.ROTATE, 2.0, 8, 4)[RB.ENV_VAR])
    assert bias.rotating and bias.n_ranks == 4
    # 8 experts over 4 ranks: rank r holds experts 2r, 2r+1.
    assert bias.vector(0, 8) == [2.0, 2.0, 0, 0, 0, 0, 0, 0]
    assert bias.vector(1, 8) == [0, 0, 2.0, 2.0, 0, 0, 0, 0]
    assert bias.vector(4, 8) == bias.vector(0, 8)   # wraps with the rank count
    # Every layer is skewed by the same amount, which is the point: the skew is
    # as strong as a fixed target's, and no rank carries it throughout.
    assert all(sum(bias.vector(layer, 8)) == 4.0 for layer in range(8))


def test_a_fixed_target_is_the_same_in_every_layer():
    bias = RB.parse_env(RB.server_env("rank:2", 2.0, 8, 4)[RB.ENV_VAR])
    assert not bias.rotating
    assert bias.vector(0, 8) == bias.vector(7, 8) == [0, 0, 0, 0, 2.0, 2.0, 0, 0]


def test_rotate_has_no_single_expert_set():
    with pytest.raises(ValueError, match="per layer"):
        RB.target_experts(RB.ROTATE, 8, 4)


def test_rotate_needs_to_know_its_layer():
    torch = pytest.importorskip("torch")
    biased = RB._wrap(lambda **kw: None, RB.Bias(strength=2.0, n_ranks=4, target=None))
    RB._current_layer[0] = None
    with pytest.raises(RuntimeError, match="per-layer router bias"):
        biased(hidden_states=None, router_logits=torch.zeros(1, 8), top_k=2)


def test_rotating_sweep_skews_every_layer_without_a_persistent_hot_rank(deployment, router_bias_config):
    router_bias_config["imbalance"]["bias_target"] = RB.ROTATE
    router_bias_config["imbalance"]["imbalance_levels"] = [0, 4]
    cfg = ExperimentConfig.from_dict(router_bias_config)
    ctx = RunContext.create(cfg)
    assert run_pipeline(ctx, cfg) == schema.STATUS_COMPLETED

    import gzip as _gzip

    by_level = {p["value"]: p for p in ctx.points}
    loads = {}
    for value, point in by_level.items():
        f = json.load(_gzip.open(ctx.abspath(point["validation_file"])))
        loads[value] = f["rank_max_over_mean_per_layer"]
    # Every layer is more skewed than when balanced...
    assert all(a > b for a, b in zip(loads[4], loads[0]))
    # ...but summed over layers the ranks come out close to even, which is exactly
    # what per-rank totals hide.
    shares = json.load(_gzip.open(ctx.abspath(by_level[4]["validation_file"])))["rank_share_mean"]
    assert max(shares) / min(shares) < 1.5
