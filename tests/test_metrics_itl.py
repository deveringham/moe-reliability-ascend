from __future__ import annotations

from moe_reliability_results.metrics import summarize_requests


def _req(itl, **kw):
    base = {"ttft": 0.1, "tpot": 0.15, "num_output_tokens": len(itl) + 1,
            "num_input_tokens": 10, "total_time": 1.0, "start_s": 0.0, "end_s": 1.0,
            "itl_ms": itl, "n_chunks": len(itl) + 1}
    return base | kw


def test_itl_stats_pool_every_gap_in_the_point():
    s = summarize_requests([_req([100.0, 100.0, 100.0]), _req([100.0, 100.0, 100.0])])
    assert s["n_itl"] == 6
    assert s["itl_ms_mean"] == 100.0
    assert s["itl_ms_median"] == 100.0
    assert s["itl_spike_count"] == 0
    assert s["itl_ms_spike_share"] == 0.0
    assert s["itl_chunks_are_tokens"] is True


def test_a_single_stalled_step_shows_in_spike_share_but_barely_in_tpot():
    # One 1 s gap among 99 x 100 ms: the request's mean per-token time rises 9%,
    # which is why TPOT alone cannot see a step that stalls once.
    stall = summarize_requests([_req([100.0] * 99 + [1000.0])])
    assert stall["itl_ms_max"] == 1000.0
    assert stall["itl_spike_count"] == 1
    assert 0.08 < stall["itl_ms_spike_share"] < 0.12


def test_coalesced_chunks_are_flagged_rather_than_reported_as_per_token():
    s = summarize_requests([_req([100.0], num_output_tokens=10)])
    assert s["itl_chunks_are_tokens"] is False
    assert s["itl_max_chunk_shortfall"] == 8


def test_a_request_ending_on_eos_is_not_mistaken_for_coalescing():
    # The EOS token carries no content, so it produces no chunk: every request that
    # stops early is one chunk short, which is not the server batching tokens.
    s = summarize_requests([_req([100.0] * 10, num_output_tokens=12)])
    assert s["itl_max_chunk_shortfall"] == 1
    assert s["itl_chunks_are_tokens"] is True


def test_points_without_itl_capture_report_none_not_zero():
    s = summarize_requests([_req([], itl_ms=None, n_chunks=None)])
    assert s["n_itl"] == 0
    assert s["itl_ms_mean"] is None
    assert s["itl_chunks_are_tokens"] is None
