from __future__ import annotations

import numpy as np

from moe_reliability.config import ExperimentConfig
from moe_reliability.pipelines import run_pipeline
from moe_reliability.runs import RunContext
from moe_reliability_results import ResultsStore, io, schema

from conftest import N_LAYERS


def _run(data):
    cfg = ExperimentConfig.from_dict(data)
    ctx = RunContext.create(cfg)
    status = run_pipeline(ctx, cfg)
    return ctx, status


def test_full_pipeline(deployment, synthetic_config_data, results_dir):
    ctx, status = _run(synthetic_config_data)
    assert status == schema.STATUS_COMPLETED

    capture, *bench = deployment.calls
    assert capture["capture"] and capture["n_prompts"] == 60 and capture["trace_dir"] is None
    assert len(bench) == 3 and all(not c["capture"] and c["trace_dir"] for c in bench)

    run = ResultsStore(results_dir).get(ctx.run_id)
    records = list(run.activations())
    assert len(records) == 60
    assert records[0]["routed_experts"].dtype == np.int16
    assert records[0]["routed_experts"].shape[1:] == (N_LAYERS, 2)
    assert records[5]["subject"] == "biology"

    assert run.available_workloads() == [0, 2]
    wl = run.workloads(0)
    assert wl["target_alphas"] == [0.5, 1.0, 1.5] and len(wl["target_ls"]) == 2
    assert wl["cv_nat"].shape == (N_LAYERS,)
    length = wl["target_ls"][0]
    w = wl["workloads"][length][1.0]
    assert w["prompts_formatted"] == w["prompts"]  # prompts are the recorded chat messages
    assert len(set(w["indices"])) == len(w["indices"])  # max_repeats = 0

    # one sweep point per alpha, with the selected workload's prompts
    assert [p["label"] for p in run.points] == ["alpha_0.5", "alpha_1.0", "alpha_1.5"]
    assert [c["n_prompts"] for c in bench] == [len(wl["workloads"][length][a]["indices"]) for a in (0.5, 1.0, 1.5)]
    summary = run.summary()
    assert {"workload_mae", "workload_effective_alpha_median"} <= set(summary.columns)
    assert all(len(p["npu_trace_views"]) == 2 for p in run.points)
    assert summary["point_status"].tolist() == ["completed"] * 3
    assert run.manifest["stages"]["figures"]["status"] == "completed"
    assert any("workloads_repeats0" in f for f in run.manifest["figures"])


def test_reuse_workloads_skips_capture(deployment, synthetic_config_data, results_dir):
    first, _ = _run(synthetic_config_data)
    n_calls = len(deployment.calls)

    data = dict(synthetic_config_data)
    data["workloads"] = {**data["workloads"], "reuse_workloads_from": first.run_id}
    data["benchmark"] = {**data["benchmark"], "enable_profiling": False, "workload_max_repeats": 2,
                         "workload_prompt_length": 16}
    data["server"] = {"batch_size": 8}
    second, status = _run(data)
    assert status == schema.STATUS_COMPLETED
    new_calls = deployment.calls[n_calls:]
    assert len(new_calls) == 3 and not any(c["capture"] for c in new_calls)
    assert all(c["batch_size"] == 8 for c in new_calls)

    run = ResultsStore(results_dir).get(second.run_id)
    assert run.manifest["inputs"]["workloads"]["run_id"] == first.run_id
    assert run.stages["activations"]["status"] == "skipped"
    assert run.workloads()["max_repeats"] == 2  # follows the reuse link
    assert len(list(run.activations(limit=3))) == 3
    assert len(run.requests()) == sum(c["n_prompts"] for c in new_calls)


def test_reuse_activations_rebuilds_workloads(deployment, synthetic_config_data):
    first, _ = _run(synthetic_config_data)
    n_calls = len(deployment.calls)
    data = dict(synthetic_config_data)
    data["activations"] = {"reuse_activations_from": first.run_id}
    data["workloads"] = {**data["workloads"], "max_repeats": [0]}
    second, status = _run(data)
    assert status == schema.STATUS_COMPLETED
    assert not any(c["capture"] for c in deployment.calls[n_calls:])
    assert list(second.manifest["workloads"]) == ["0"]
    assert io.read_json(second.path / second.manifest["workloads"]["0"]["file"])["activations_run_id"] == first.run_id


def test_resume_skips_completed_stages(deployment, synthetic_config_data):
    ctx, _ = _run(synthetic_config_data)
    n_calls = len(deployment.calls)
    reopened = RunContext.open(ctx.path)
    assert run_pipeline(reopened, reopened.config()) == schema.STATUS_COMPLETED
    assert len(deployment.calls) == n_calls
