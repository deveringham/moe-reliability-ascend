from __future__ import annotations

import numpy as np
import pytest

from moe_reliability_results import drift as D

N_LAYERS, N_EXPERTS, TOP_K = 3, 8, 2


def _record(expert_ids, n_tokens=20, n_prompt=10):
    """A record whose every token routes to ``expert_ids`` in every layer."""
    ids = np.tile(np.asarray(expert_ids, dtype=np.int16), (n_tokens, N_LAYERS, 1))
    pids = np.tile(np.asarray(expert_ids, dtype=np.int16), (n_prompt, N_LAYERS, 1))
    return {"routed_experts": ids, "prompt_routed_experts": pids,
            "num_input_tokens": n_prompt, "num_output_tokens": n_tokens}


def test_histograms_count_prefill_and_decode_tokens():
    H, kept = D.record_histograms([_record([0, 1])], n_experts=N_EXPERTS)
    assert H.shape == (1, N_LAYERS, N_EXPERTS) and len(kept) == 1
    # 30 tokens x top-2, split evenly between experts 0 and 1, in every layer
    assert H[0, 0, 0] == 30 and H[0, 0, 1] == 30 and H[0, 0, 2:].sum() == 0
    decode_only, _ = D.record_histograms([_record([0, 1])], n_experts=N_EXPERTS,
                                         include_prefill=False)
    assert decode_only[0, 0, 0] == 20


def test_js_divergence_is_zero_for_equal_and_one_for_disjoint():
    p = D.distribution(np.array([[4.0, 4.0, 0, 0, 0, 0, 0, 0]]))
    q = D.distribution(np.array([[0, 0, 4.0, 4.0, 0, 0, 0, 0]]))
    assert D.js_divergence(p, p) == pytest.approx(0.0, abs=1e-9)
    # disjoint support saturates the bound at 1 bit
    assert D.js_divergence(p, q) == pytest.approx(1.0, abs=1e-6)


def test_marginal_drift_rises_when_routing_changes():
    ref, _ = D.record_histograms([_record([0, 1])] * 4, n_experts=N_EXPERTS)
    same, _ = D.record_histograms([_record([0, 1])] * 2, n_experts=N_EXPERTS)
    moved, _ = D.record_histograms([_record([2, 3])] * 2, n_experts=N_EXPERTS)
    assert D.marginal_drift(same, ref) == pytest.approx(0.0, abs=1e-9)
    assert D.marginal_drift(moved, ref) > 0.9


def test_stratified_drift_ignores_a_change_in_stratum_mix():
    """Reweighting by the reference's own mix is what cancels a workload shift.

    Two strata that route differently, and a window made entirely of the second:
    the pooled comparison sees a large change, the stratified one sees none,
    because routing *within* each stratum is unchanged.
    """
    # Equal token counts per stratum, so the pooled reference is an even mix and
    # a shift in the mix is visible rather than swamped by the longer records.
    short = [_record([0, 1], n_tokens=10, n_prompt=0) for _ in range(16)]
    long = [_record([2, 3], n_tokens=40, n_prompt=0) for _ in range(4)]
    H_ref, _ = D.record_histograms(short + long, n_experts=N_EXPERTS)
    ref_strata = [0] * 16 + [1] * 4

    # The window holds both strata, just in a different proportion.
    H_win, _ = D.record_histograms(short[:2] + long * 2 + long[:2], n_experts=N_EXPERTS)
    win_strata = [0] * 2 + [1] * 10

    assert D.marginal_drift(H_win, H_ref) > 0.2      # pooled: looks like drift
    # Weighted by the reference's own mix, with routing unchanged inside each
    # stratum, the mix shift cancels exactly.
    assert D.stratified_drift(H_win, H_ref, win_strata, ref_strata,
                              min_per_stratum=2) == pytest.approx(0.0, abs=1e-9)


def test_stratified_drift_is_nan_without_a_shared_stratum():
    H, _ = D.record_histograms([_record([0, 1])] * 2, n_experts=N_EXPERTS)
    assert np.isnan(D.stratified_drift(H, H, ["a", "a"], ["b", "b"]))


def test_length_stratum_buckets_by_token_count():
    assert D.length_stratum({"num_input_tokens": 10, "num_output_tokens": 20,
                             "routed_experts": np.zeros((1, 1, 1))}) == 0
    assert D.length_stratum({"num_input_tokens": 200, "num_output_tokens": 200,
                             "routed_experts": np.zeros((1, 1, 1))}) == 4


def test_roc_auc_and_threshold():
    assert D.roc_auc([0.0, 0.1], [0.5, 0.6]) == pytest.approx(1.0)
    assert D.roc_auc([0.5, 0.6], [0.0, 0.1]) == pytest.approx(0.0)
    assert D.roc_auc([0.1, 0.2], [0.1, 0.2]) == pytest.approx(0.5)  # ties count half
    assert np.isnan(D.roc_auc([], [1.0]))
    # the threshold admits at most the requested false-alarm rate
    neg = list(np.linspace(0, 1, 101))
    assert D.threshold_at_fpr(neg, 0.01) == pytest.approx(0.99, abs=0.02)
