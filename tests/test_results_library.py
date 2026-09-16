from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from moe_experiments.config import ExperimentConfig
from moe_experiments.pipelines import run_pipeline
from moe_experiments.runs import RunContext
from moe_results import ResultsStore, Run, plots, schema
from moe_results import hta_plots


@pytest.fixture
def store(deployment, forced_config_data, results_dir):
    for n_npus in (2, 4):
        for batch in (16, 32):
            cfg = ExperimentConfig.from_dict(forced_config_data, [f"hardware.n_npus={n_npus}",
                                                                   f"server.batch_size={batch}"])
            run_pipeline(RunContext.create(cfg), cfg)
    return ResultsStore(results_dir)


def test_filters_and_tables(store):
    assert len(store) == 4
    assert len(store.runs(n_npus=4)) == 2
    assert len(store.runs(batch_size=[16, 64], n_npus=lambda g: g > 2)) == 1
    assert store.runs(model_name="unknown") == []
    assert store.runs(not_a_key=1) == []

    summary = store.summary()
    assert len(summary) == 8
    assert {"run_id", "sweep_value", "n_npus", "batch_size", "ttft_ms_mean", "tpot_ms_p99", "e2e_s_mean"} \
        <= set(summary.columns)
    slow = store.query("sweep_value == 100 and n_npus == 2")
    assert len(slow) == 2 and (slow["tpot_ms_mean"] > 0).all()

    grouped = store.group_metric("tpot_ms_mean", by=["sweep_value"])
    assert grouped.loc[1, "tpot_ms_mean"] > grouped.loc[0, "tpot_ms_mean"]

    requests = store.requests(n_npus=2)
    assert len(requests) == 2 * 2 * 24
    assert {"ttft_ms", "tpot_ms", "num_output_tokens"} <= set(requests.columns)
    assert "prompt" not in requests.columns

    infra = store.infrastructure_configurations()
    assert len(infra) == 8
    assert set(schema.INFRASTRUCTURE_KEYS) <= set(infra.columns)
    assert (infra["n_runs"] == 1).all()
    assert store.configurations()["n_points"].tolist() == [2, 2, 2, 2]


def test_run_accessors(store):
    run = store.latest(n_npus=4, batch_size=32)
    assert isinstance(run, Run) and run.status == "completed"
    assert run.sweep_values == [0, 100]
    assert run.point(100)["label"] == run.point("imbalance_100")["label"]
    with pytest.raises(KeyError):
        run.point(50)
    assert len(run.request_records(0)) == 24
    assert set(run.results_by_point()) == {0, 100}
    assert run.trace_summary(0) is None and run.hta_frames() is None
    with pytest.raises(KeyError):
        run.workloads()
    with pytest.raises(KeyError):
        store.get("does-not-exist")


def test_standard_figures(store):
    run = store.latest()
    figs = plots.plot_run(run)
    assert {"latency_sweep", "latency_comparison"} <= set(figs)
    for fig in figs.values():
        plt.close(fig)


def test_overview_inputs(store):
    # keyed by (model, batch size, level): select one NPU count
    K, T = plots.moe_imbalance_overview_inputs(store, n_npus=2)
    assert K == {} and len(T) == 4  # no traces stored in these runs
    assert set(T[("mixtral", 16, 100)]) == {"tpot", "ttft"}


class FakeTraceAnalysis:

    def __init__(self, trace_dir):
        self.slow = "imbalance_100" in str(trace_dir)

    def get_temporal_breakdown(self, visualize=False):
        rows = []
        for rank in (0, 1):
            idle = 30.0 if (self.slow and rank == 1) else 10.0
            rows.append({"rank": rank, "idle_time(us)": idle * 10, "compute_time(us)": (90 - idle) * 10,
                         "non_compute_time(us)": 100.0, "kernel_time(us)": 1000.0, "idle_time_pctg": idle,
                         "compute_time_pctg": 90 - idle, "non_compute_time_pctg": 10.0})
        return pd.DataFrame(rows)

    def get_comm_comp_overlap(self, visualize=False):
        return pd.DataFrame({"rank": [0, 1], "comp_comm_overlap_pctg": [12.0, 8.0]})

    def get_idle_time_breakdown(self, ranks, visualize=False):
        return pd.DataFrame([{"rank": r, "idle_category": c, "idle_time": 5.0}
                             for r in ranks for c in ("host_wait", "kernel_wait", "other")]), None

    def get_gpu_kernel_breakdown(self, visualize=False):
        return pd.DataFrame({"kernel_type": ["COMPUTATION", "COMMUNICATION", "MEMORY"],
                             "sum": [70.0, 25.0, 5.0], "percentage": [70.0, 25.0, 5.0]}), None


def test_hta_stage_and_plots(deployment, forced_config_data, results_dir, monkeypatch):
    from moe_experiments.core import hta_analysis

    monkeypatch.setattr(hta_analysis, "load_trace_analysis", FakeTraceAnalysis)
    deployment.trace_format = "pytorch"  # input format of the kernel analysis and HTA
    for batch in (16, 32):
        cfg = ExperimentConfig.from_dict(forced_config_data, [
            f"server.batch_size={batch}", "benchmark.enable_profiling=true", "analysis.hta=true",
            "analysis.parse_npu_traces=false"])
        assert run_pipeline(RunContext.create(cfg), cfg) == "completed"

    store = ResultsStore(results_dir)
    run = store.latest()
    frames = run.hta_frames()
    assert set(frames) == {"rank", "idle_categories", "kernel_types", "runs"}
    assert len(frames["rank"]) == 4 and set(frames["runs"]["imbalance"]) == {0, 100}

    rank_df, idle_df, kern_df, run_df = hta_plots.hta_frames(store)
    assert len(run_df) == 4 and (run_df.loc[run_df.imbalance == 0, "throughput_rel"] == 100).all()
    figures = [
        hta_plots.plot_idle_comparison(run_df, rank_df),
        hta_plots.plot_temporal_stacked(rank_df),
        hta_plots.plot_idle_category_stacked(idle_df),
        hta_plots.plot_per_rank_load(rank_df),
        hta_plots.plot_load_imbalance(run_df),
        hta_plots.plot_overlap(run_df),
        hta_plots.plot_kernel_types(kern_df),
        hta_plots.plot_throughput(run_df),
    ]
    assert all(f is not None for f in figures)
    plt.close("all")
    table = hta_plots.comparison_table(run_df)
    assert len(table) == 2 and "idle_pctg_mean__delta" in table.columns

    K, T = plots.moe_imbalance_overview_inputs(store)
    assert len(K) == 4 and T == {}
    plots.plot_moe_imbalance_overview(K, {k: {"tpot": pd.Series([10.0, 12.0]).to_numpy(),
                                              "ttft": pd.Series([30.0, 31.0]).to_numpy()} for k in K})
    plt.close("all")
