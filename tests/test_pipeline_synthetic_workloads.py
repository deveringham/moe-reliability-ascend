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
    # Profiling is on, so every point is served twice: unprofiled then profiled.
    assert len(bench) == 6 and not any(c["capture"] for c in bench)
    assert [c["trace_dir"] is None for c in bench] == [True, False] * 3

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
    expected = [len(wl["workloads"][length][a]["indices"]) for a in (0.5, 1.0, 1.5)]
    measured = [c["n_prompts"] for c in bench if c["trace_dir"] is None]
    profiled = [c["n_prompts"] for c in bench if c["trace_dir"] is not None]
    assert measured == expected and profiled == expected  # both passes replay the same workload
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


def test_length_in_requests_fixes_the_request_count(deployment, synthetic_config_data, results_dir):
    """Every alpha must serve the same number of requests.

    With a token budget the request count grows with alpha, because a higher CV is
    reached most cheaply from more, shorter prompts - so the served batch size
    tracks the imbalance it is meant to isolate.
    """
    synthetic_config_data["workloads"]["length_in_requests"] = True
    synthetic_config_data["workloads"]["target_prompt_lengths"] = [12]
    synthetic_config_data["benchmark"]["workload_prompt_length"] = 12
    ctx, status = _run(synthetic_config_data)
    assert status == schema.STATUS_COMPLETED

    run = ResultsStore(results_dir).get(ctx.run_id)
    wl = run.workloads(0)
    assert wl["limit_unit"] == "requests" and wl["target_ls"] == [12]
    for a in (0.5, 1.0, 1.5):
        assert len(wl["workloads"][12][a]["indices"]) == 12
    assert {p["n_prompts"] for p in run.points} == {12}


def test_prompt_length_tolerance_restricts_the_pool(deployment, synthetic_config_data, results_dir):
    synthetic_config_data["workloads"]["prompt_length_tolerance"] = 0.1
    ctx, status = _run(synthetic_config_data)
    assert status == schema.STATUS_COMPLETED

    run = ResultsStore(results_dir).get(ctx.run_id)
    wl = run.workloads(0)
    assert wl["prompt_length_tolerance"] == 0.1

    counts = np.array([r["routed_experts"].shape[0] + r["prompt_routed_experts"].shape[0]
                       for r in run.activations()])
    median = np.median(counts)
    # The band has to exclude part of the pool, or the test proves nothing.
    in_band = (counts >= median * 0.9) & (counts <= median * 1.1)
    assert 0 < in_band.sum() < len(counts)
    length = wl["target_ls"][0]
    for a in (0.5, 1.0, 1.5):
        chosen = counts[list(wl["workloads"][length][a]["indices"])]
        assert chosen.min() >= median * 0.9 and chosen.max() <= median * 1.1


def test_repeats_give_every_point_its_own_alphas_workload(deployment, synthetic_config_data, results_dir):
    """Each repeat must get the workload of ITS alpha.

    The workloads are keyed by alpha and the points are not: with repeats there
    are several points per alpha, so pairing the two by position hands each
    repeat the next alpha's workload and leaves the later points with none.
    """
    synthetic_config_data["benchmark"]["repeats"] = 2
    synthetic_config_data["benchmark"]["enable_profiling"] = False
    ctx, status = _run(synthetic_config_data)
    assert status == schema.STATUS_COMPLETED

    run = ResultsStore(results_dir).get(ctx.run_id)
    wl = run.workloads(0)
    length = wl["target_ls"][0]

    assert [p["label"] for p in ctx.points] == [
        "alpha_0.5", "alpha_0.5_r2", "alpha_1.0", "alpha_1.0_r2", "alpha_1.5", "alpha_1.5_r2"]
    for p in ctx.points:
        assert p["status"] == schema.STATUS_COMPLETED
        expected = len(wl["workloads"][length][p["value"]]["indices"])
        assert p["n_prompts"] == expected, f"{p['label']} was served another alpha's workload"
