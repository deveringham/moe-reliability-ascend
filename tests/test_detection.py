"""The detection pipeline's statistics, on routing whose answer is known by construction."""

from __future__ import annotations

import numpy as np
import pytest

from moe_reliability_results import detection as D

N_RANKS = 4


def counts(per_layer_experts):
    """(layers, experts) float counts from a list of lists."""
    return np.asarray(per_layer_experts, dtype=float)


def balanced(n_layers=8, n_experts=8, per_expert=100.0):
    return np.full((n_layers, n_experts), per_expert)


def test_busiest_rank_reads_one_when_balanced_and_the_ratio_when_not():
    assert D.busiest_rank(balanced(), N_RANKS) == pytest.approx(1.0)
    # Rank 0 holds experts 0-1 of 8 over 4 ranks; give it half of every layer.
    c = balanced()
    c[:, :2] = 200.0
    load = D.busiest_rank(c, N_RANKS)
    assert load == pytest.approx(4 * (400 / 1000), rel=1e-9)  # 1.6x


def test_pooling_layers_first_hides_a_hot_rank_that_moves_between_layers():
    # Each layer has one rank carrying everything, but a different rank each time.
    c = np.zeros((4, 8))
    for layer in range(4):
        c[layer, 2 * layer:2 * layer + 2] = 100.0
    assert D.busiest_rank(c, N_RANKS) == pytest.approx(4.0)  # every layer is maximally skewed
    assert D.pooled_busiest_rank(c, N_RANKS) == pytest.approx(1.0)  # ...and pooling calls it balanced

    # With one rank hot in every layer the two agree, which is the router-bias case.
    hot = balanced()
    hot[:, :2] = 300.0
    assert D.pooled_busiest_rank(hot, N_RANKS) == pytest.approx(D.busiest_rank(hot, N_RANKS))


def test_request_counts_pool_prompt_and_generated_tokens_and_drop_dense_layers():
    # Layer 0 is dense: every id is 0. Layers 1 and 2 route.
    prompt = np.zeros((3, 3, 2), dtype=int)
    prompt[:, 1:, :] = [[1, 2]]
    generated = np.zeros((2, 3, 2), dtype=int)
    generated[:, 1:, :] = [[3, 4]]
    records = [{"prompt_routed_experts": prompt, "routed_experts": generated}]

    c = D.request_counts(records, n_experts=8)
    assert c.shape == (1, 2, 8)  # one request, two MoE layers
    assert c[0, 0, 1] == 3 and c[0, 0, 3] == 2  # prompt and generated both counted

    prompt_only = D.request_counts(records, n_experts=8, include_generated=False)
    assert prompt_only[0, 0, 1] == 3 and prompt_only[0, 0, 3] == 0


def test_a_longer_window_estimates_the_same_load_more_tightly():
    rng = np.random.default_rng(0)
    # 400 requests, each routing a little differently around a 1.6x skew.
    probs = np.array([0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    per_request = np.stack([rng.multinomial(200, probs, size=6) for _ in range(400)]).astype(float)

    truth = D.busiest_rank(per_request.sum(axis=0), N_RANKS)
    narrow = D.screen_scores(D.WindowSampler(window=64, draws=300), D.SampleCost(1.0, 1.0),
                             D.estimate_loads(per_request, N_RANKS, D.WindowSampler(window=64, draws=300)),
                             truth=truth)
    wide = D.screen_scores(D.WindowSampler(window=1, draws=300), D.SampleCost(1.0, 1.0),
                           D.estimate_loads(per_request, N_RANKS, D.WindowSampler(window=1, draws=300)),
                           truth=truth)
    assert narrow.spread < wide.spread
    assert abs(narrow.bias) < 0.02  # and stays centred on the truth


def test_sampling_fewer_layers_and_tokens_costs_resolution_and_is_priced():
    rng = np.random.default_rng(1)
    probs = np.array([0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    per_request = np.stack([rng.multinomial(200, probs, size=12) for _ in range(200)]).astype(float)

    full = D.WindowSampler(window=32, draws=300)
    thin = D.WindowSampler(window=32, layers=3, token_fraction=0.1, draws=300)
    assert thin.cost(12).assignment_fraction == pytest.approx(0.25 * 0.1)
    assert full.cost(12).assignment_fraction == 1.0

    spread_full = D.screen_scores(full, full.cost(12), D.estimate_loads(per_request, N_RANKS, full)).spread
    spread_thin = D.screen_scores(thin, thin.cost(12), D.estimate_loads(per_request, N_RANKS, thin)).spread
    assert spread_thin > spread_full


def test_required_window_finds_the_cheapest_window_that_resolves_a_difference():
    rng = np.random.default_rng(2)
    probs = np.array([0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    per_request = np.stack([rng.multinomial(200, probs, size=6) for _ in range(600)]).astype(float)

    coarse = D.required_window(per_request, N_RANKS, resolution=0.2)
    fine = D.required_window(per_request, N_RANKS, resolution=0.02)
    assert coarse is not None and fine is not None and coarse <= fine
    assert D.required_window(per_request, N_RANKS, resolution=1e-6, windows=(1, 4)) is None


def test_threshold_and_roc_separate_a_skewed_condition_from_benign():
    benign = np.random.default_rng(3).normal(1.10, 0.02, 500)
    skewed = np.random.default_rng(4).normal(1.60, 0.02, 200)
    threshold = D.threshold_at_fpr(benign, 0.01)
    assert 1.10 < threshold < 1.25

    scores = D.roc(skewed, benign)
    assert scores["auc"] == pytest.approx(1.0)
    assert scores["detection_rate_at_1pct_fpr"] == pytest.approx(1.0)

    # A condition that overlaps benign traffic must not score as separable.
    marginal = D.roc(np.random.default_rng(5).normal(1.12, 0.02, 200), benign)
    assert 0.5 < marginal["auc"] < 0.95
    assert marginal["detection_rate_at_1pct_fpr"] < 0.5


def test_roc_needs_both_classes():
    with pytest.raises(ValueError, match="both positive and negative"):
        D.roc([], [1.0])


def test_impact_threshold_interpolates_between_measured_levels():
    # Mixtral's shape: cost rises from the first offset.
    loads = [1.18, 1.24, 1.38, 1.76, 2.62, 3.36]
    costs = [0.0, 0.96, 3.32, 6.71, 17.16, 26.77]
    threshold = D.impact_threshold(loads, costs, noise_pct=2.2)
    assert 1.24 < threshold < 1.38

    # DeepSeek's shape: flat, then a jump. The threshold lies in the unswept gap.
    ds_loads = [1.14, 1.30, 1.66, 2.34, 3.45]
    ds_costs = [0.0, -0.03, -1.25, 1.03, 22.22]
    ds = D.impact_threshold(ds_loads, ds_costs, noise_pct=2.2)
    assert 2.34 < ds < 3.45

    # A sweep that never leaves the noise floor has no measured threshold.
    assert D.impact_threshold(ds_loads[:4], ds_costs[:4], noise_pct=2.2) is None


def test_step_cost_fit_compares_like_steps_and_reports_per_bin():
    rng = np.random.default_rng(6)

    def table(n, batch, prefill, wall):
        reqs = np.full(n, batch)
        return {"reqs": reqs, "tokens": reqs + prefill,
                "wall_us": rng.normal(wall, wall * 0.01, n)}

    # Full-batch decode steps do not lengthen; small-batch decode steps do.
    steps = {}
    for level, (small, full) in {0.0: (150e3, 160e3), 1.0: (220e3, 161e3), 2.0: (240e3, 162e3)}.items():
        small_table, full_table = table(30, 100, 0, small), table(30, 500, 0, full)
        steps[level] = {k: np.concatenate([small_table[k], full_table[k]]) for k in small_table}
    loads = {0.0: 1.14, 1.0: 2.34, 2.0: 3.45}

    fit = D.step_cost_fit(steps, loads)
    by_bin = {(r["batch"], r["kind"]): r for r in fit["by_bin"]}
    assert by_bin[("1-200", "decode")]["ms_per_load"] > 20
    assert abs(by_bin[("480-max", "decode")]["ms_per_load"]) < 2


def test_pace_setter_agreement_reads_the_trace_summary():
    rows = D.pace_setter_agreement({0.0: {"collective_wait": {"pace_setter_rank": 2, "pace_setter_share": 0.4,
                                                              "wait_pct": 89.0}},
                                    2.0: {"collective_wait": {"pace_setter_rank": 0, "pace_setter_share": 0.9,
                                                              "wait_pct": 91.0}}})
    assert [r["is_hot_rank"] for r in rows] == [False, True]
    assert rows[1]["pace_setter_share"] == 0.9


def test_monitoring_overhead_scales_with_what_is_counted():
    full = D.monitoring_overhead(1.0, per_assignment_ns=2.0, tokens_per_step=512, step_ms=160,
                                 n_layers=26, top_k=6)
    tenth = D.monitoring_overhead(0.1, per_assignment_ns=2.0, tokens_per_step=512, step_ms=160,
                                  n_layers=26, top_k=6)
    assert full["assignments_per_step"] == 512 * 26 * 6
    assert tenth["cost_ms_per_step"] == pytest.approx(full["cost_ms_per_step"] / 10)
    assert full["pct_of_step"] < 1.0  # a device-side counter is cheap at this scale


def test_localise_names_the_layers_that_carry_a_skew_and_the_rank_that_holds_it():
    c = balanced(n_layers=10)
    c[3:6, :2] = 400.0  # rank 0 hot in layers 3-5 only
    loads, hot = D.per_layer_load(c, N_RANKS)
    assert D.busiest_rank(c, N_RANKS) == pytest.approx(loads.mean())

    found = D.localise(c, N_RANKS, threshold=1.3)
    assert found["flagged_layers"] == [3, 4, 5]
    assert found["flagged_rank"] == 0
    assert found["mean_load"] < found["max_layer_load"]  # the average dilutes a local skew
    assert list(hot[3:6]) == [0, 0, 0]


def test_a_persistent_hot_rank_is_told_apart_from_a_rotating_one():
    persistent = balanced(n_layers=8)
    persistent[:, :2] = 300.0
    assert D.localise(persistent, N_RANKS, 1.3)["consistent_rank_share"] == pytest.approx(1.0)

    rotating = np.full((8, 8), 100.0)
    for layer in range(8):
        rank = layer % N_RANKS
        rotating[layer, 2 * rank:2 * rank + 2] = 300.0
    found = D.localise(rotating, N_RANKS, 1.3)
    # Every layer is skewed, but no rank leads more than its share, so moving
    # experts between ranks cannot help.
    assert len(found["flagged_layers"]) == 8
    assert found["consistent_rank_share"] == pytest.approx(0.25)


def test_request_counts_handle_a_capture_that_never_split_prompt_from_generated():
    # max_new_tokens = 1 leaves everything in routed_experts and the prompt field None.
    ids = np.zeros((4, 3, 2), dtype=int)
    ids[:, 1:, :] = [[2, 5]]
    unsplit = [{"prompt_routed_experts": None, "routed_experts": ids}]
    c = D.request_counts(unsplit, n_experts=8)
    assert c.shape == (1, 2, 8) and c[0, 0, 2] == 4

    with pytest.raises(ValueError, match="no routed experts"):
        D.request_counts([{"prompt_routed_experts": None, "routed_experts": None}], n_experts=8)
