"""The screen's counting hook: what it counts, how little of it, and what it writes."""

from __future__ import annotations

import json
import sys
import types

import pytest
import torch

from moe_reliability import expert_counter as EC


def ids(rows):
    return torch.tensor(rows, dtype=torch.int32)


def counter(**kw):
    cfg = EC.CountConfig(**{"layers": 2, "token_fraction": 1.0, "window_steps": 2, **kw})
    return EC._Counter(cfg)


# --- configuration ----------------------------------------------------------


def test_token_fraction_becomes_a_stride():
    assert EC.CountConfig(token_fraction=1.0).stride == 1
    assert EC.CountConfig(token_fraction=0.1).stride == 10
    assert EC.CountConfig(token_fraction=0.02).stride == 50


@pytest.mark.parametrize("kw, match", [
    ({"layers": 0}, "at least 1"),
    ({"token_fraction": 0}, "token_fraction"),
    ({"token_fraction": 1.5}, "token_fraction"),
    ({"window_steps": 0}, "at least 1"),
])
def test_impossible_sampling_is_refused(kw, match):
    with pytest.raises(ValueError, match=match):
        EC.CountConfig(**kw)


def test_server_env_is_empty_when_counting_is_off():
    assert EC.server_env(None) == {}


def test_server_env_round_trips():
    cfg = EC.CountConfig(layers=4, token_fraction=0.1, window_steps=200, out="/tmp/c.jsonl")
    assert EC.parse_env(EC.server_env(cfg)[EC.ENV_VAR]) == cfg


# --- counting ---------------------------------------------------------------


def test_counts_land_on_the_calling_layer():
    c = counter()
    c.observe(3, ids([[0, 1], [1, 2]]), n_experts=4)
    assert c.counts[3].tolist() == [1, 2, 1, 0]
    assert c.counts[0].tolist() == [0, 0, 0, 0]  # a layer that did not call is untouched


def test_the_table_grows_to_a_later_layer_without_losing_earlier_counts():
    c = counter()
    c.observe(1, ids([[0, 0]]), n_experts=3)
    c.observe(9, ids([[2, 2]]), n_experts=3)
    assert c.counts[1].tolist() == [2, 0, 0]
    assert c.counts[9].tolist() == [0, 0, 2]


def test_the_stride_is_what_makes_counting_cheap():
    """A tenth of the tokens means a tenth of the assignments counted."""
    rows = ids([[0, 1]] * 100)
    full, sampled = counter(token_fraction=1.0), counter(token_fraction=0.1)
    full.observe(0, rows, n_experts=2)
    sampled.observe(0, rows, n_experts=2)
    assert int(full.counts[0].sum()) == 200
    assert int(sampled.counts[0].sum()) == 20


# --- layer sampling ---------------------------------------------------------


def test_the_layer_sample_is_redrawn_every_window():
    """Slicing the layers once is the mistake this exists to avoid."""
    c = counter(layers=2)
    c.known = {0, 1, 2, 3, 4, 5}
    draws = set()
    for _ in range(40):
        c.redraw()
        assert len(c.sampled) == 2
        draws.add(c.sampled)
    assert len(draws) > 1, "the same two layers every window is fixed sampling"
    # Over enough windows every layer is eligible, which is what lets counts
    # accumulated across windows name a skew confined to a few layers.
    assert set().union(*draws) == c.known


def test_a_model_with_fewer_layers_than_asked_for_draws_them_all():
    c = counter(layers=8)
    c.known = {0, 1, 2}
    c.redraw()
    assert c.sampled == frozenset({0, 1, 2})


def test_redraw_before_any_layer_is_known_does_nothing():
    c = counter()
    c.redraw()
    assert c.sampled == frozenset()


# --- windows ----------------------------------------------------------------


def test_a_window_closes_when_the_layer_index_stops_advancing(tmp_path):
    """Layers run in increasing order within a forward, so a reset is the next step."""
    out = tmp_path / "counts.jsonl"
    c = counter(layers=1, window_steps=2, out=str(out))

    class Layer:
        def __init__(self, i):
            self.layer_name = f"model.layers.{i}.mlp.experts"

    runner = types.SimpleNamespace()
    calls = []
    runner.forward_impl = lambda self, layer, *a: calls.append(layer)
    tracked = EC._track_steps(runner.forward_impl, c)

    for _ in range(3):  # three passes over two MoE layers
        for i in (0, 1):
            tracked(runner, Layer(i))
            if i in c.sampled:
                c.observe(i, ids([[0]]), n_experts=2)

    assert c.step == 2, "two resets after the first pass"
    assert c.windows == 1 and out.exists()
    record = json.loads(out.read_text().splitlines()[0])
    assert record["window"] == 1 and len(record["layers"]) == 1
    assert len(record["counts"]) == len(record["layers"])
    assert c.known == {0, 1}


def test_a_flushed_window_starts_from_zero(tmp_path):
    out = tmp_path / "counts.jsonl"
    c = counter(layers=1, out=str(out))
    c.known = {0}
    c.redraw()
    c.observe(0, ids([[0, 0]]), n_experts=2)
    c.flush()
    assert int(c.counts[0].sum()) == 0, "counts carried into the next window would double-count"
    first = json.loads(out.read_text().splitlines()[0])
    assert first["counts"] == [[2, 0]]


def test_a_window_with_nothing_counted_still_advances(tmp_path):
    out = tmp_path / "counts.jsonl"
    c = counter(layers=1, out=str(out))
    c.flush()
    assert c.windows == 1
    assert not out.exists(), "no layer was ever drawn, so there is nothing to write"


# --- registration -----------------------------------------------------------


@pytest.fixture
def fake_ascend(monkeypatch):
    """Stand-ins for the two vllm-ascend modules the hook patches."""
    sel = types.ModuleType("vllm_ascend.ops.fused_moe.experts_selector")
    # The real functions take positional arguments too, which is what lets a
    # positional router_logits bypass the hook unnoticed.
    sel._select_experts_with_fusion_ops = lambda *a, **kw: ("w", ids([[0]]))
    sel._native_select_experts = lambda *a, **kw: ("w", ids([[1]]))
    fused = types.ModuleType("vllm_ascend.ops.fused_moe.fused_moe")
    fused.AscendMoERunner = types.SimpleNamespace(forward_impl=lambda self, layer: None)
    pkg = types.ModuleType("vllm_ascend.ops.fused_moe")
    pkg.experts_selector = sel
    for name, mod in {"vllm_ascend": types.ModuleType("vllm_ascend"),
                      "vllm_ascend.ops": types.ModuleType("vllm_ascend.ops"),
                      "vllm_ascend.ops.fused_moe": pkg,
                      "vllm_ascend.ops.fused_moe.experts_selector": sel,
                      "vllm_ascend.ops.fused_moe.fused_moe": fused}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return types.SimpleNamespace(sel=sel, fused=fused)


def test_register_is_a_no_op_without_the_variable(fake_ascend, monkeypatch):
    monkeypatch.delenv(EC.ENV_VAR, raising=False)
    before = fake_ascend.sel._native_select_experts
    EC.register()
    assert fake_ascend.sel._native_select_experts is before


def test_register_patches_both_paths_once(fake_ascend, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv(EC.ENV_VAR, json.dumps({"layers": 1}))
    EC.register()
    EC.register()  # vLLM may load a plugin more than once per process
    for name in ("_select_experts_with_fusion_ops", "_native_select_experts"):
        fn = getattr(fake_ascend.sel, name)
        assert getattr(fn.__wrapped__, "__wrapped__", None) is None
    assert getattr(fake_ascend.fused.AscendMoERunner.forward_impl, "_counting_steps", False)


def test_only_one_rank_counts(fake_ascend, monkeypatch):
    """Selection is replicated, so the other ranks would recount the same routing."""
    monkeypatch.setenv("RANK", "2")
    c = counter(layers=1)
    assert c.active() is False
    monkeypatch.setenv("RANK", "0")
    assert counter(layers=1).active() is True


def test_the_counting_rank_is_decided_at_the_first_call_not_at_load(fake_ascend, monkeypatch):
    """vLLM loads plugins in every worker before the process group exists and
    without setting RANK, so a check at load time cannot tell the ranks apart."""
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("VLLM_DP_RANK", raising=False)
    monkeypatch.setenv(EC.ENV_VAR, json.dumps({"layers": 1}))
    EC.register()  # installs regardless of rank
    assert getattr(fake_ascend.sel._native_select_experts, "__wrapped__", None) is not None

    # By the first forward the group is up, and a non-zero rank counts nothing.
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 3)
    c = counter(layers=1)
    c.sampled = frozenset({0})
    from moe_reliability import router_bias

    wrapped = EC._wrap(fake_ascend.sel.__dict__["_native_select_experts"].__wrapped__, c)
    router_bias._current_layer[0] = 0
    try:
        wrapped(hidden_states=None, router_logits=torch.zeros(1, 2), top_k=1)
    finally:
        router_bias._current_layer[0] = None
    assert c.counts is None, "a non-zero rank must not count"


def test_an_unsampled_layer_passes_straight_through(fake_ascend):
    from moe_reliability import router_bias

    c = counter(layers=1)
    c.known = {0, 1}
    c.sampled = frozenset({1})
    wrapped = EC._wrap(fake_ascend.sel._native_select_experts, c)
    router_bias._current_layer[0] = 0
    try:
        assert wrapped(hidden_states=None, router_logits=torch.zeros(1, 2), top_k=1)[0] == "w"
        assert c.counts is None, "an unsampled layer must not even allocate"
        router_bias._current_layer[0] = 1
        wrapped(hidden_states=None, router_logits=torch.zeros(1, 2), top_k=1)
        assert c.counts[1].tolist() == [0, 1]
    finally:
        router_bias._current_layer[0] = None


def test_positional_logits_are_refused(fake_ascend):
    """The hook reads the router width from the logits; silently guessing it is worse."""
    from moe_reliability import router_bias

    c = counter(layers=1)
    c.sampled = frozenset({0})
    wrapped = EC._wrap(fake_ascend.sel._native_select_experts, c)
    router_bias._current_layer[0] = 0
    try:
        with pytest.raises(TypeError, match="keyword"):
            wrapped(None, torch.zeros(1, 2), 1)
    finally:
        router_bias._current_layer[0] = None
