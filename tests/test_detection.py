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


def test_impact_threshold_ignores_a_crossing_that_falls_back():
    # The mixtral calibration sweep's shape: fine levels around the floor, one of
    # which clears it, then a higher level back below. The onset is the crossing
    # that holds, not the first one.
    loads = [1.11, 1.13, 1.20, 1.27, 1.37, 1.76, 2.62]
    costs = [0.0, -1.09, -0.25, 1.95, 0.15, 5.83, 14.45]
    threshold = D.impact_threshold(loads, costs, noise_pct=0.95)
    assert 1.37 < threshold < 1.76

    # With the fall-back level removed the two rules agree: nothing above the
    # first crossing dips back under the floor.
    assert D.impact_threshold(loads[:4] + loads[5:], costs[:4] + costs[5:],
                              noise_pct=0.95) == pytest.approx(1.246, abs=0.01)


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


# --- Detection delay, and how often the screen cries wolf --------------------

def _pool(hot_share, n=400, n_layers=6, n_experts=8, seed=0):
    """Requests routing with a given share on rank 0's experts (0 and 1 of 8)."""
    rng = np.random.default_rng(seed)
    probs = np.full(n_experts, (1 - hot_share) / (n_experts - 2))
    probs[:2] = hot_share / 2
    return np.stack([rng.multinomial(200, probs, size=n_layers) for _ in range(n)]).astype(float)


def test_the_alarm_fires_promptly_once_the_skew_begins():
    balanced, skewed = _pool(0.25), _pool(0.55, seed=1)
    sampler = D.WindowSampler(window=8)
    alarm = D.threshold_at_fpr(D.estimate_loads(balanced, N_RANKS, sampler), 0.01)

    found = D.onset_delay(balanced, skewed, N_RANKS, sampler, alarm, pre_windows=5, draws=200)
    assert found["detected_fraction"] > 0.9
    assert found["missed_fraction"] == 0        # a skew this large is never missed
    # The window the onset lands in has to finish first, so a delay below one
    # window would mean the clock is wrong.
    assert found["delay_requests_p50"] >= sampler.window


def test_a_long_benign_stretch_fires_before_the_skew_arrives():
    # A 1% per-window alarm rate is not small over many windows: 40 windows carry
    # a 1 - 0.99^40 = 33% chance of firing first. This is why the screen needs
    # either consecutive-window confirmation or stage 2 behind it.
    balanced, skewed = _pool(0.25), _pool(0.55, seed=1)
    sampler = D.WindowSampler(window=8)
    alarm = D.threshold_at_fpr(D.estimate_loads(balanced, N_RANKS, sampler), 0.01)

    short = D.onset_delay(balanced, skewed, N_RANKS, sampler, alarm, pre_windows=5, draws=200)
    long = D.onset_delay(balanced, skewed, N_RANKS, sampler, alarm, pre_windows=40, draws=200)
    assert long["false_start_fraction"] > short["false_start_fraction"] + 0.1


def test_a_skew_that_never_comes_is_never_detected():
    balanced = _pool(0.25)
    sampler = D.WindowSampler(window=8)
    alarm = D.threshold_at_fpr(D.estimate_loads(balanced, N_RANKS, sampler), 0.01)
    # "Skewed" is the same distribution: anything reported is a false start.
    found = D.onset_delay(balanced, _pool(0.25, seed=9), N_RANKS, sampler, alarm, draws=100)
    assert found["detected_fraction"] < 0.5


def test_requiring_consecutive_alarms_trades_delay_for_quiet():
    balanced, skewed = _pool(0.25), _pool(0.45, seed=2)
    sampler = D.WindowSampler(window=8)
    alarm = D.threshold_at_fpr(D.estimate_loads(balanced, N_RANKS, sampler), 0.01)

    one = D.mean_windows_to_false_alarm(balanced, N_RANKS, sampler, alarm, consecutive=1, draws=80)
    three = D.mean_windows_to_false_alarm(balanced, N_RANKS, sampler, alarm, consecutive=3, draws=80)
    # Three in a row is far rarer by chance...
    assert three["quiet_over_horizon"] > one["quiet_over_horizon"]

    fast = D.onset_delay(balanced, skewed, N_RANKS, sampler, alarm, consecutive=1, draws=80)
    slow = D.onset_delay(balanced, skewed, N_RANKS, sampler, alarm, consecutive=3, draws=80)
    # ...and costs delay when the skew is real.
    assert slow["delay_requests_p50"] >= fast["delay_requests_p50"]


def test_alarm_stream_windows_tumble_over_the_whole_stream():
    pool = _pool(0.25, n=50)
    sampler = D.WindowSampler(window=8)
    flags = D.alarm_stream([(pool, 40), (pool, 40)], N_RANKS, sampler, alarm=0.0,
                           rng=np.random.default_rng(0))
    assert len(flags) == 80 // 8        # one decision per tumbling window
    assert flags.all()                  # an alarm at 0.0 fires on every window


def test_batch_reduction_matches_the_single_window_form():
    rng = np.random.default_rng(11)
    windows = rng.integers(1, 500, size=(7, 5, 8)).astype(float)
    one_at_a_time = np.array([D.busiest_rank(w, N_RANKS) for w in windows])
    assert np.allclose(D.batch_busiest_rank(windows, N_RANKS), one_at_a_time)


def test_pipeline_overhead_is_dominated_by_how_often_confirmation_fires():
    # The screen is the same in both; only how often it cries wolf differs.
    noisy = D.pipeline_overhead(screen_pct=0.2, requests_to_false_alarm=324,
                                confirm_pct=19.0, confirm_requests=100, stage2_rejects=0.0)
    quiet = D.pipeline_overhead(screen_pct=0.2, requests_to_false_alarm=1204,
                                confirm_pct=19.0, confirm_requests=100, stage2_rejects=0.0)
    assert noisy["total_pct"] > 3 * quiet["screen_pct"]      # confirmation dominates
    assert noisy["total_pct"] > quiet["total_pct"]

    # A screen that never fires on benign traffic pays only for counting.
    silent = D.pipeline_overhead(0.2, None, 19.0, 100, stage2_rejects=0.0)
    assert silent["total_pct"] == 0.2


def test_a_cheap_confirmation_that_rejects_removes_the_expensive_one():
    both = [D.pipeline_overhead(0.2, 324, 19.0, 100, stage2_rejects=r) for r in (0.0, 1.0)]
    assert both[1]["confirm_pct"] == 0.0 and both[0]["confirm_pct"] > 0


# --- Random layer sampling, and localising by accumulation -------------------

def test_evenly_spaced_layers_can_miss_a_block_entirely():
    sampler = D.WindowSampler(window=8, layers=4)
    spaced = set(sampler._layer_index(26).tolist())
    # The block the localisation experiment biases sits between the spaced picks.
    assert spaced.isdisjoint({9, 10, 11, 12, 13, 14})


def test_random_layer_sampling_redraws_every_window():
    sampler = D.WindowSampler(window=8, layers=4, layer_choice="random")
    rng = np.random.default_rng(0)
    draws = {tuple(sampler._layer_index(26, rng)) for _ in range(20)}
    assert len(draws) > 10                       # not the same four every time
    assert all(len(set(d)) == 4 for d in draws)  # and four distinct layers each time
    with pytest.raises(ValueError, match="needs a generator"):
        sampler._layer_index(26)


def _layered(n=300, n_layers=26, n_experts=8, hot=(), hot_share=0.5, seed=0):
    """Requests routing evenly, except on ``hot`` layers where rank 0 takes a share."""
    rng = np.random.default_rng(seed)
    out = np.empty((n, n_layers, n_experts))
    even = np.full(n_experts, 1 / n_experts)
    skew = np.full(n_experts, (1 - hot_share) / (n_experts - 2))
    skew[:2] = hot_share / 2
    for layer in range(n_layers):
        p = skew if layer in hot else even
        out[:, layer] = rng.multinomial(400, p, size=n)
    return out


def test_random_sampling_sees_a_layer_local_skew_that_spaced_sampling_misses():
    # A skew in layers 9-14 only, which the evenly spaced four do not touch.
    balanced = _layered(seed=0)
    skewed = _layered(hot=set(range(9, 15)), seed=1)

    def detection(choice):
        s = D.WindowSampler(window=8, layers=4, draws=300, layer_choice=choice)
        alarm = D.threshold_at_fpr(D.estimate_loads(balanced, N_RANKS, s, seed=0), 0.01)
        return float((D.estimate_loads(skewed, N_RANKS, s, seed=1) >= alarm).mean())

    assert detection("spaced") < 0.1     # blind to it
    assert detection("random") > 0.5     # sees it, at the same per-window cost


def test_streaming_timers_redraw_random_layers_rather_than_fixing_them_once():
    # onset_delay and mean_windows_to_false_alarm used to pick the layers before
    # the stream started, which is fixed sampling wearing the random label - and
    # on a layer-local skew it makes them blind in exactly the way above.
    balanced = _layered(seed=0)
    skewed = _layered(hot=set(range(9, 15)), seed=1)
    sampler = D.WindowSampler(window=8, layers=4, layer_choice="random")
    alarm = D.threshold_at_fpr(D.estimate_loads(balanced, N_RANKS, sampler, seed=0), 0.01)

    found = D.onset_delay(balanced, skewed, N_RANKS, sampler, alarm, pre_windows=5, draws=200)
    assert found["detected_fraction"] > 0.8
    quiet = D.mean_windows_to_false_alarm(balanced, N_RANKS, sampler, alarm, draws=80)
    assert quiet["windows_to_false_alarm_p50"] is None or quiet["windows_to_false_alarm_p50"] > 10


def test_accumulation_places_the_skew_without_counting_every_layer_at_once():
    skewed = _layered(hot=set(range(9, 15)), seed=1)
    sampler = D.WindowSampler(window=8, layers=4, layer_choice="random")

    mean_load, seen = D.accumulate_layers(skewed, N_RANKS, sampler, windows=200, seed=0)
    assert seen.min() > 0                       # every layer eventually counted
    assert np.nanmax(mean_load[9:15]) > np.nanmax(np.delete(mean_load, range(9, 15))) + 0.2
    # Four layers a window, 200 windows: each layer seen a fraction of the time.
    assert seen.sum() == 200 * 4


def test_accumulation_reports_layers_it_never_saw_as_unknown():
    counts = _layered(n=50, seed=2)
    few = D.WindowSampler(window=8, layers=4, layer_choice="random")
    mean_load, seen = D.accumulate_layers(counts, N_RANKS, few, windows=2, seed=0)
    assert np.isnan(mean_load[seen == 0]).all()  # not a reading of 0
    assert seen.sum() == 8


# --- Calibrating one deployment ---------------------------------------------

LOADS = [1.18, 1.24, 1.38, 1.76, 2.62]
COSTS = [0.0, 0.96, 3.32, 6.71, 17.16]        # Mixtral's measured eager curve
BALANCED = [165.5, 164.1, 166.5]
SAMPLING = {"window": 8, "layers": 4, "token_fraction": 0.1, "layer_choice": "random"}


def test_calibration_separates_where_it_costs_from_where_to_alarm():
    benign = np.random.default_rng(0).normal(1.15, 0.03, 500)
    cal = D.calibrate(LOADS, COSTS, BALANCED, benign, SAMPLING, alarm_margin=0.5)

    # This deployment's own noise floor is 0.77%, which the 1.24x level already
    # clears, so the threshold interpolates between the first two measured loads.
    assert cal["noise_floor_pct"] == pytest.approx(0.77, abs=0.02)
    assert LOADS[0] < cal["impact_threshold"] < LOADS[2]
    # The alarm sits below the cost, so there is warning rather than confirmation.
    assert cal["alarm"]["level"] < cal["impact_threshold"]
    assert cal["alarm"]["warns_before_cost"]
    assert cal["benign_ceiling"] < cal["alarm"]["level"]


def test_the_margin_trades_warning_against_false_alarms():
    benign = np.random.default_rng(1).normal(1.15, 0.03, 500)
    late = D.calibrate(LOADS, COSTS, BALANCED, benign, SAMPLING, alarm_margin=0.0)
    early = D.calibrate(LOADS, COSTS, BALANCED, benign, SAMPLING, alarm_margin=1.0)

    assert late["alarm"]["level"] == pytest.approx(late["impact_threshold"])
    assert early["alarm"]["level"] < late["alarm"]["level"]
    assert early["alarm"]["false_alarm_rate"] >= late["alarm"]["false_alarm_rate"]


def test_a_deployment_whose_traffic_already_reaches_the_cost_has_no_headroom():
    # Benign traffic sitting on top of the impact threshold: no alarm level
    # separates them, and the calibration has to say so rather than pick one.
    benign = np.random.default_rng(2).normal(1.45, 0.03, 500)
    cal = D.calibrate(LOADS, COSTS, BALANCED, benign, SAMPLING, alarm_margin=0.5)
    assert cal["headroom"] < 0
    assert "cannot warn before" in cal["note"]


def test_the_noise_floor_comes_from_this_deployment_not_a_constant():
    quiet = D.calibrate(LOADS, COSTS, [165.5, 165.4, 165.6], [1.1] * 50, SAMPLING)
    noisy = D.calibrate(LOADS, COSTS, [176.7, 176.1, 169.2], [1.1] * 50, SAMPLING)
    assert noisy["noise_floor_pct"] > quiet["noise_floor_pct"] * 3
    # A noisier deployment needs a larger cost before it can claim one.
    assert noisy["impact_threshold"] >= quiet["impact_threshold"]


def test_a_sweep_that_never_leaves_the_noise_says_so():
    # A noisy balanced arm (floor 2.8%) against levels that cost at most 1%.
    cal = D.calibrate(LOADS[:2], COSTS[:2], [176.7, 176.1, 169.2], [1.1] * 50, SAMPLING)
    assert cal["impact_threshold"] is None and cal["alarm"] is None
    assert "nothing to calibrate" in cal["note"]


# --- A screen built from a calibration --------------------------------------

def _calibration(margin=0.5, benign_at=1.15):
    benign = np.random.default_rng(0).normal(benign_at, 0.03, 500)
    cal = D.calibrate(LOADS, COSTS, BALANCED, benign, SAMPLING, alarm_margin=margin)
    cal["run"] = "a-deployment"
    return cal


def test_a_screen_carries_its_deployment_s_numbers():
    screen = D.Screen.from_calibration(_calibration())
    assert screen.alarm == pytest.approx(_calibration()["alarm"]["level"])
    assert screen.sampler.window == 8 and screen.sampler.layer_choice == "random"
    assert screen.source == "a-deployment"


def test_one_window_over_the_level_is_not_an_alarm():
    screen = D.Screen.from_calibration(_calibration(), consecutive=3)
    high = screen.alarm + 0.1
    assert screen.decide([high, 1.0, high, 1.0])["raised"] is None
    out = screen.decide([1.0, high, high, high, 1.0])
    assert out["raised"] == 3                       # the window that completed the run
    assert out["raised_after_requests"] == 32       # four windows of eight


def test_the_screen_separates_alarming_from_costing():
    screen = D.Screen.from_calibration(_calibration(margin=1.0))
    assert screen.alarm < screen.impact_threshold   # it warns before the cost
    mild = screen.decide([screen.alarm + 0.01] * 5)
    assert mild["raised"] is not None               # worth looking at
    assert mild["above_impact_threshold"] is False  # ...but not yet costing

    # Nothing raised means no verdict on cost, rather than a verdict of False.
    assert screen.decide([1.0] * 5)["above_impact_threshold"] is None


def test_a_single_high_window_does_not_make_traffic_look_costly():
    # Benign traffic throws occasional high windows; judging cost on the peak
    # would call ordinary traffic costly whenever one of them lands.
    screen = D.Screen.from_calibration(_calibration())
    quiet = [1.05] * 20 + [screen.impact_threshold + 0.5] + [1.05] * 20
    out = screen.decide(quiet)
    assert out["raised"] is None
    assert out["peak_load"] > screen.impact_threshold
    assert out["above_impact_threshold"] is None


def test_a_calibration_with_no_measurable_cost_cannot_arm_a_screen():
    cal = D.calibrate(LOADS[:2], COSTS[:2], [176.7, 176.1, 169.2], [1.1] * 50, SAMPLING)
    with pytest.raises(ValueError, match="no alarm level"):
        D.Screen.from_calibration(cal)


def test_a_stream_redraws_its_layers_per_window():
    # Picking the layers once for the whole stream would be fixed sampling under
    # another name, and would carry the same blind spot.
    counts = _layered(n=200, hot=set(range(9, 15)), seed=3)
    sampler = D.WindowSampler(window=8, layers=4, layer_choice="random")
    rng = np.random.default_rng(0)
    loads = D.stream_loads([(counts, 8 * 60)], N_RANKS, sampler, rng)
    assert len(loads) == 60
    # With four of 26 layers redrawn each window, some windows catch the hot
    # block and some miss it, so the readings must not all be the same.
    assert loads.std() > 0.02
    assert loads.max() > loads.min() + 0.1


def test_the_margin_trade_is_visible_end_to_end():
    rng = np.random.default_rng(0)
    benign = rng.normal(1.15, 0.03, 600)
    # Two levels below the cost and two above it.
    levels = {1.22: rng.normal(1.22, 0.03, 400), 1.28: rng.normal(1.28, 0.03, 400),
              1.45: rng.normal(1.45, 0.03, 400), 1.90: rng.normal(1.90, 0.03, 400)}
    rows = D.margin_sweep(impact=1.35, benign_loads=benign, level_loads=levels)

    late, early = rows[0], rows[-1]
    assert late["margin"] == 0.0 and early["margin"] == 1.0
    assert late["alarm"] > early["alarm"]
    # Warning earlier catches more of the harmless levels, and costs false alarms.
    assert early["caught_before_cost"] > late["caught_before_cost"]
    assert early["false_alarm_rate"] >= late["false_alarm_rate"]
    # Anything already costing is caught at every margin.
    assert all(r["caught_once_costly"] == 1.0 for r in rows)


def test_margin_sweep_reports_nothing_to_warn_about_when_every_level_costs():
    rng = np.random.default_rng(1)
    levels = {1.90: rng.normal(1.90, 0.03, 200)}
    rows = D.margin_sweep(1.35, rng.normal(1.15, 0.03, 200), levels)
    assert all(r["caught_before_cost"] is None for r in rows)


def test_placement_from_an_eplb_map_moves_the_load_with_the_experts():
    # Four experts, two ranks. All the load is on experts 0 and 1, which is
    # rank 0's half under contiguous placement: maximally skewed.
    counts = np.array([[10.0, 10.0, 0.0, 0.0]])
    assert D.busiest_rank(counts, n_ranks=2) == pytest.approx(2.0)

    # EPLB splits them: expert 1 moves to rank 1 and expert 3 takes its slot.
    expert_map = {"moe_layer_count": 1, "layer_list": [{
        "layer_id": 0, "device_count": 2, "device_list": [
            {"device_id": 0, "device_expert": [0, 3]},
            {"device_id": 1, "device_expert": [2, 1]}]}]}
    placement = D.placement_from_expert_map(expert_map, n_experts=4)
    assert placement.tolist() == [[0, 1, 1, 0]]
    # The same counts now read as balanced, because they are - against the layout
    # the server is actually using.
    assert D.busiest_rank(counts, n_ranks=2, placement=placement) == pytest.approx(1.0)

    # Read without the map, an EPLB run's capture measures a layout nobody used.
    assert D.busiest_rank(counts, n_ranks=2) == pytest.approx(2.0)


def test_a_replicated_expert_is_attributed_to_the_first_rank_holding_it():
    # With redundant slots EPLB may hold one expert on several ranks. The capture
    # cannot say which copy served a token, so the first is charged the whole lot -
    # which understates the spreading, and is why the docstring says so.
    expert_map = {"moe_layer_count": 1, "layer_list": [{
        "layer_id": 0, "device_count": 2, "device_list": [
            {"device_id": 0, "device_expert": [0, 1]},
            {"device_id": 1, "device_expert": [1, 2]}]}]}
    assert D.placement_from_expert_map(expert_map, n_experts=3).tolist() == [[0, 0, 1]]
