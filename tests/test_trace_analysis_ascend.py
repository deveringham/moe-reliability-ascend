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
