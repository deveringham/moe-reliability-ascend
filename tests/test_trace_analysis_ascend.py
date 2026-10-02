from __future__ import annotations

from pathlib import Path

import pytest

from moe_reliability.core import trace_analysis as ta

OP_HEADER = "Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)\n"
KERNEL_HEADER = "Device_id,Model ID,Task ID,Stream ID,Name,Type,OP State,Accelerator Core,Start Time(us),Duration(us)\n"


def _rank_dir(root: Path, rank: int, ops: list[tuple[str, int, float]],
              kernels: list[tuple[str, float, float]]) -> None:
    """One rank's parsed profiler output. ops: (type, count, total_us)."""
    out = root / f"dp0_pp0_tp{rank}_rank{rank}_123_20261002_ascend_pt" / ta.ASCEND_OUTPUT
    out.mkdir(parents=True)
    with open(out / "op_statistic.csv", "w") as f:
        f.write(OP_HEADER)
        for name, count, total in ops:
            f.write(f"{rank},{name},AI_CORE,{count},{total},0,0,0,0\n")
    with open(out / "kernel_details.csv", "w") as f:
        f.write(KERNEL_HEADER)
        for name, start, dur in kernels:
            # the profiler writes the start time with a trailing tab
            f.write(f"{rank},0,0,0,aclnn{name},{name},dynamic,AI_CORE,{start}\t,{dur}\n")


@pytest.fixture
def traces(tmp_path):
    """Two ranks whose GEMM totals match but whose per-call maxima do not.

    Rank 0 is slow on the first call, rank 1 on the second. Summed per rank the
    imbalance cancels; paired per call it does not.
    """
    # op_statistic.csv holds only the AI-core operators, as the profiler writes
    # it: no communication kernel. kernel_details.csv carries everything.
    ops = [("GroupedMatmul", 2, 30.0), ("FusedInferAttentionScore", 1, 40.0), ("RmsNorm", 1, 10.0)]
    _rank_dir(tmp_path, 0, ops=ops,
              kernels=[("GroupedMatmul", 100.0, 20.0), ("GroupedMatmul", 200.0, 10.0),
                       ("FusedInferAttentionScore", 300.0, 40.0),
                       ("hcom_allReduce_", 400.0, 20.0), ("RmsNorm", 500.0, 10.0)])
    _rank_dir(tmp_path, 1, ops=ops,
              kernels=[("GroupedMatmul", 100.0, 10.0), ("GroupedMatmul", 200.0, 20.0),
                       ("FusedInferAttentionScore", 300.0, 40.0),
                       ("hcom_allReduce_", 400.0, 20.0), ("RmsNorm", 500.0, 10.0)])
    return tmp_path


def test_rank_outputs_and_op_statistics(traces):
    assert sorted(ta.ascend_rank_outputs(traces)) == [0, 1]
    stats = ta.op_statistics(traces)
    assert stats[0]["GroupedMatmul"] == {"total_us": 30.0, "count": 2}


def test_straggler_survives_what_per_rank_totals_cancel(traces):
    s = ta.straggler(traces, "GroupedMatmul")
    assert s["per_rank_total_us"] == [30.0, 30.0]
    # totals are identical, so the conventional statistic sees perfect balance
    assert s["totals_max_over_mean"] == pytest.approx(1.0)
    # each call still waited for the slower rank: max(20,10) + max(10,20) = 40
    assert s["sum_of_per_call_maxima_us"] == pytest.approx(40.0)
    assert s["straggler"] == pytest.approx(40.0 / 30.0)


def test_step_decomposition_counts_communication(traces):
    """Built from kernel_details, so the HCCL kernels op_statistic omits are counted."""
    d = ta.step_decomposition(traces)
    # per rank: 30 moe + 40 attention + 20 comm + 10 norm = 100, over two ranks
    assert d["summed_kernel_us"] == pytest.approx(200.0)
    assert "communication" in d["by_category_pct"], "communication kernels were dropped"
    assert d["by_category_pct"]["communication"] == pytest.approx(20.0)
    # compute shares exclude communication, whose duration is mostly blocking wait
    assert d["compute_us"] == pytest.approx(160.0)
    assert d["compute_pct"]["moe"] == pytest.approx(100.0 * 60 / 160)
    assert d["compute_pct"]["attention"] == pytest.approx(100.0 * 80 / 160)
    # kernels here do not overlap, so busy time is the sum of durations
    assert d["busy_us"] == [pytest.approx(100.0), pytest.approx(100.0)]
    assert d["busy_max_over_mean"] == pytest.approx(1.0)


def test_summarize_ascend_exposes_flat_scalars(traces):
    s = ta.summarize_ascend(traces)
    assert s["ranks"] == [0, 1]
    assert s["moe_pct"] == pytest.approx(100.0 * 60 / 160)
    assert s["communication_pct"] == pytest.approx(20.0)
    assert s["straggler_op"] == "GroupedMatmul"
    assert s["straggler"] == pytest.approx(40.0 / 30.0)
    assert s["totals_max_over_mean"] == pytest.approx(1.0)
    assert s["top_ops"][0]["op_type"] == "FusedInferAttentionScore"   # largest summed time

    from moe_reliability_results.metrics import trace_scalars

    assert trace_scalars(s)["straggler"] == pytest.approx(40.0 / 30.0)


def test_summarize_ascend_without_profiler_output(tmp_path):
    with pytest.raises(FileNotFoundError, match="no parsed Ascend profiler output"):
        ta.summarize_ascend(tmp_path)


def test_summary_carries_the_pytorch_trace_key_names(traces):
    """Figures and the results library read either kind of trace summary."""
    s = ta.summarize_ascend(traces)
    assert s["max_over_mean"] == pytest.approx(s["totals_max_over_mean"])
    for key in ("mean_over_ranks_us", "per_rank_mean_us", "calls_per_rank",
                "total_over_ranks_ms", "max_over_min", "hottest_rank"):
        assert key in s, key


@pytest.fixture
def waiting_traces(tmp_path):
    """Three ranks on one collective, with rank 2 always arriving last.

    Every rank leaves the collective together, so rank 2's short duration is the
    transfer and the other two are blocked for the difference. Compute is equal
    across ranks, which is the case worth separating: all of the spread in busy
    time is waiting.
    """
    ops = [("GroupedMatmul", 2, 20.0)]
    for rank, waits in ((0, (40.0, 60.0)), (1, (50.0, 50.0)), (2, (10.0, 10.0))):
        _rank_dir(tmp_path, rank, ops=ops,
                  kernels=[("GroupedMatmul", 100.0, 10.0), ("hcom_allReduce_", 200.0, waits[0]),
                           ("GroupedMatmul", 400.0, 10.0), ("hcom_allReduce_", 500.0, waits[1])])
    return tmp_path


def test_collective_wait_splits_transfer_from_blocking(waiting_traces):
    w = ta.collective_wait(waiting_traces, "hcom_allReduce_")
    assert w["ranks"] == [0, 1, 2] and w["calls_per_rank"] == 2
    # the last arriver's duration is the transfer: 10 + 10
    assert w["transfer_us"] == pytest.approx(20.0)
    # everything above that floor is blocked time, per rank
    assert w["per_rank_wait_us"] == pytest.approx([80.0, 80.0, 0.0])
    # 160 of the 220us summed across ranks (100 + 100 + 20) is waiting, not transfer
    assert w["wait_pct"] == pytest.approx(100.0 * 160.0 / 220.0)


def test_collective_wait_names_the_rank_the_others_wait_for(waiting_traces):
    w = ta.collective_wait(waiting_traces, "hcom_allReduce_")
    assert w["last_arriver_counts"] == [0, 0, 2]
    assert w["pace_setter_rank"] == 2
    assert w["pace_setter_share"] == pytest.approx(1.0)  # paces every call


def test_summary_finds_the_dominant_collective_without_being_told(waiting_traces):
    out = ta.summarize_ascend(waiting_traces)
    assert out["collective_op"] == "hcom_allReduce_"
    assert out["pace_setter_rank"] == 2
    assert out["collective_wait_pct"] == pytest.approx(100.0 * 160.0 / 220.0)
    # idle is reported too, as the complement of occupancy
    assert 0.0 < out["occupancy_min"] <= out["occupancy_mean"] <= 1.0
