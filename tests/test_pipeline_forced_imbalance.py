from __future__ import annotations

import numpy as np
import pytest

from moe_reliability.config import ExperimentConfig
from moe_reliability.pipelines import forced_imbalance, run_pipeline
from moe_reliability.runs import RunContext
from moe_reliability_results import ResultsStore, io, schema
from conftest import N_EXPERTS, N_LAYERS


def test_run_stores_metrics_and_figures(deployment, forced_config_data, results_dir, tmp_path):
    cfg = ExperimentConfig.from_dict(forced_config_data)
    ctx = RunContext.create(cfg)
    assert run_pipeline(ctx, cfg) == schema.STATUS_COMPLETED

    # level 0 serves the original model, level 100 a generated checkpoint
    served = [c["model_path"] for c in deployment.calls]
    assert served == ["org/Mixtral-test", str(tmp_path / "models" / "mixtral-imbalance100")]
    assert deployment.checkpoints.created == [served[1]]
    assert all(c["n_prompts"] == 24 and c["trace_dir"] is None for c in deployment.calls)

    manifest = io.read_json(ctx.path / schema.MANIFEST_FILE)
    assert [p["label"] for p in manifest["points"]] == ["imbalance_0", "imbalance_100"]
    assert manifest["stages"]["validation"]["status"] == schema.STATUS_SKIPPED
    assert manifest["stages"]["trace_analysis"]["status"] == schema.STATUS_SKIPPED
    assert manifest["stages"]["figures"]["status"] == schema.STATUS_COMPLETED
    for p in manifest["points"]:
        assert p["status"] == schema.STATUS_COMPLETED
        doc = io.read_json(ctx.path / p["metrics_file"])
        assert len(doc["requests"]) == 24 and doc["sweep_value"] == p["value"]
        assert p["metrics_file"].endswith(".json.gz")  # output.compress defaults to true
    assert any(f.endswith("latency_sweep.png") for f in manifest["figures"])
    assert (ctx.path / schema.CONFIG_FILE).is_file()

    run = ResultsStore(results_dir).get(ctx.run_id)
    summary = run.summary()
    assert list(summary["sweep_value"]) == [0, 100]
    assert summary.loc[1, "tpot_ms_mean"] > summary.loc[0, "tpot_ms_mean"]


def test_existing_checkpoints_are_reused(deployment, forced_config_data, tmp_path):
    (tmp_path / "models" / "mixtral-imbalance100").mkdir(parents=True)
    cfg = ExperimentConfig.from_dict(forced_config_data)
    ctx = RunContext.create(cfg)
    run_pipeline(ctx, cfg)
    assert deployment.checkpoints.created == []
    assert ctx.point("imbalance_100")["checkpoint_created"] is False


def test_failed_point_gives_partial_run_and_retry(deployment, forced_config_data, tmp_path):
    failing = str(tmp_path / "models" / "mixtral-imbalance100")
    deployment.fail_models.add(failing)
    cfg = ExperimentConfig.from_dict(forced_config_data)
    ctx = RunContext.create(cfg)
    assert run_pipeline(ctx, cfg) == schema.STATUS_PARTIAL
    assert ctx.point("imbalance_100")["status"] == schema.STATUS_FAILED
    assert ctx.point("imbalance_0")["status"] == schema.STATUS_COMPLETED

    # plain resume does not retry the failed point
    n_calls = len(deployment.calls)
    reopened = RunContext.open(ctx.run_id, cfg.output.results_dir)
    assert run_pipeline(reopened, reopened.config()) == schema.STATUS_PARTIAL
    assert len(deployment.calls) == n_calls

    # --retry-failed benchmarks only the failed point again
    deployment.fail_models.clear()
    reopened = RunContext.open(ctx.run_id, cfg.output.results_dir)
    assert run_pipeline(reopened, reopened.config(), retry_failed=True) == schema.STATUS_COMPLETED
    assert [c["model_path"] for c in deployment.calls[n_calls:]] == [failing]


def test_profiled_run_parses_npu_profiler_data(deployment, forced_config_data):
    forced_config_data["benchmark"]["enable_profiling"] = True
    cfg = ExperimentConfig.from_dict(forced_config_data)
    # The separate unprofiled pass measures latency without the profiler, so the
    # request metrics are kept even though profiling is on.
    assert cfg.benchmark.separate_profiling_run is True
    assert cfg.benchmark.save_request_metrics is True
    ctx = RunContext.create(cfg)
    assert run_pipeline(ctx, cfg) == schema.STATUS_COMPLETED

    for label in ("imbalance_0", "imbalance_100"):
        p = ctx.point(label)
        assert p["metrics_file"] is not None and p["request_summary"]["n_requests"] == 24
        assert p["npu_trace_views"] == [f"traces/{label}/worker_{r}/ASCEND_PROFILER_OUTPUT/trace_view.json"
                                        for r in (0, 1)]
        assert p["trace_parse_error"] is None
        # raw NPU profiler data is not in the input format of the fused-MoE kernel analysis
        assert "*rank*.pt.trace.json.gz" in p["trace_error"] and p["trace_metrics_file"] is None
    assert ctx.stage_status("trace_analysis") == schema.STATUS_COMPLETED
    assert ctx.stage_status("hta") == schema.STATUS_SKIPPED


def test_benchmark_capture_reaches_the_benchmark_pass_only_when_asked(deployment, forced_config_data):
    """The capture is what a monitor's collection path costs, so it has to be switchable.

    Validation always captures - that is where the routing comes from - so the
    benchmark pass is the only one whose timings can be compared with and
    without it. Leaving it off by default keeps every existing sweep's TPOT
    measured on an uninstrumented server.
    """
    cfg = ExperimentConfig.from_dict(forced_config_data)
    ctx = RunContext.create(cfg)
    assert run_pipeline(ctx, cfg) == schema.STATUS_COMPLETED
    benchmark = [c for c in deployment.calls if not c["capture"]]
    assert benchmark and all(not c["capture"] for c in benchmark)

    forced_config_data["benchmark"]["capture_experts"] = True
    cfg = ExperimentConfig.from_dict(forced_config_data)
    ctx = RunContext.create(cfg)
    assert run_pipeline(ctx, cfg) == schema.STATUS_COMPLETED
    assert all(c["capture"] for c in deployment.calls[len(deployment.calls) - len(benchmark):])


def test_separate_profiling_run_serves_each_point_twice(deployment, forced_config_data):
    """Timings must come from an unprofiled pass, traces from a profiled one."""
    forced_config_data["benchmark"]["enable_profiling"] = True
    cfg = ExperimentConfig.from_dict(forced_config_data)
    ctx = RunContext.create(cfg)
    assert run_pipeline(ctx, cfg) == schema.STATUS_COMPLETED

    calls = [c for c in deployment.calls if not c["capture"]]
    assert len(calls) == 4                                   # 2 points x 2 passes
    assert [c["trace_dir"] is None for c in calls] == [True, False, True, False]


def test_profiling_pass_timings_are_kept_so_its_overhead_is_measurable(deployment, forced_config_data):
    """The two passes are one configuration with and without the profiler."""
    forced_config_data["benchmark"]["enable_profiling"] = True
    cfg = ExperimentConfig.from_dict(forced_config_data)
    ctx = RunContext.create(cfg)
    assert run_pipeline(ctx, cfg) == schema.STATUS_COMPLETED

    for point in ctx.points:
        assert point["profiled_request_summary"]["tpot_ms_mean"] > 0
        # The recorded timings stay the unprofiled ones.
        assert point["request_summary"]["tpot_ms_mean"] != point["profiled_request_summary"]["tpot_ms_mean"]


def test_single_pass_leaves_no_profiled_twin(deployment, forced_config_data):
    forced_config_data["benchmark"]["enable_profiling"] = True
    forced_config_data["benchmark"]["separate_profiling_run"] = False
    cfg = ExperimentConfig.from_dict(forced_config_data)
    ctx = RunContext.create(cfg)
    assert run_pipeline(ctx, cfg) == schema.STATUS_COMPLETED
    assert all(p.get("profiled_request_summary") is None for p in ctx.points)


def test_single_pass_when_separate_profiling_run_disabled(deployment, forced_config_data):
    forced_config_data["benchmark"]["enable_profiling"] = True
    forced_config_data["benchmark"]["separate_profiling_run"] = False
    cfg = ExperimentConfig.from_dict(forced_config_data)
    # Without a clean pass the only timings available are the perturbed ones.
    assert cfg.benchmark.save_request_metrics is False
    ctx = RunContext.create(cfg)
    assert run_pipeline(ctx, cfg) == schema.STATUS_COMPLETED

    calls = [c for c in deployment.calls if not c["capture"]]
    assert len(calls) == 2 and all(c["trace_dir"] for c in calls)


def test_kernel_metrics_from_rank_traces(deployment, forced_config_data):
    deployment.trace_format = "pytorch"
    forced_config_data["benchmark"]["enable_profiling"] = True
    forced_config_data["analysis"] = {"parse_npu_traces": False}
    cfg = ExperimentConfig.from_dict(forced_config_data)
    ctx = RunContext.create(cfg)
    assert run_pipeline(ctx, cfg) == schema.STATUS_COMPLETED

    lo, hi = ctx.point("imbalance_0"), ctx.point("imbalance_100")
    assert "npu_trace_views" not in hi
    assert hi["trace_summary"]["hottest_rank"] == 0
    assert hi["trace_summary"]["max_over_mean"] > lo["trace_summary"]["max_over_mean"]
    summary = io.read_json(ctx.path / hi["trace_metrics_file"])
    assert summary["ranks"] == [0, 1] and summary["trace_dir"] == "traces/imbalance_100"
    assert any("kernel_sweep" in f for f in ctx.manifest["figures"])


def test_npu_parsing_failure_is_recorded(deployment, forced_config_data, monkeypatch):
    from conftest import install_fake_npu_profiler

    def broken(profiler_path, **kwargs):
        raise RuntimeError("msprof data incomplete")

    install_fake_npu_profiler(monkeypatch, analyse=broken)
    forced_config_data["benchmark"]["enable_profiling"] = True
    forced_config_data["analysis"] = {"trace_summary": False}
    cfg = ExperimentConfig.from_dict(forced_config_data)
    ctx = RunContext.create(cfg)
    assert run_pipeline(ctx, cfg) == schema.STATUS_COMPLETED
    assert "msprof data incomplete" in ctx.point("imbalance_100")["trace_parse_error"]


def test_exception_marks_run_failed(deployment, forced_config_data, monkeypatch):
    def boom(ctx, cfg):
        raise RuntimeError("disk full")

    monkeypatch.setattr(forced_imbalance, "create_checkpoints", boom)
    cfg = ExperimentConfig.from_dict(forced_config_data)
    ctx = RunContext.create(cfg)
    with pytest.raises(RuntimeError):
        run_pipeline(ctx, cfg)
    manifest = io.read_json(ctx.path / schema.MANIFEST_FILE)
    assert manifest["status"] == schema.STATUS_FAILED
    assert "disk full" in manifest["stages"]["checkpoints"]["error"]


def test_figures_stage_stays_open_when_nothing_is_rendered(deployment, forced_config_data, monkeypatch):
    """A run with no usable points must not record figures as completed.

    The stage would otherwise be skipped for the life of the run, so retrying the
    failed points with `resume --retry-failed` could never produce figures.
    """
    from moe_reliability.pipelines import common

    monkeypatch.setattr(common, "serve_and_measure", lambda *args, **kwargs: None)
    cfg = ExperimentConfig.from_dict(forced_config_data)
    ctx = RunContext.create(cfg)
    run_pipeline(ctx, cfg)

    assert ctx.stage_status(common.STAGE_FIGURES) == schema.STATUS_SKIPPED
    assert common.should_run(ctx, common.STAGE_FIGURES)


def test_expert_load_counts_a_routed_expert_capture():
    # [tokens, layers, k]: layer 0 always picks experts 0 and 1, layer 1 only 3
    prompt = np.zeros((3, 2, 2), dtype=np.int16)
    prompt[:, 0, 1] = 1
    prompt[:, 1, :] = 3
    generated = np.zeros((2, 2, 2), dtype=np.int16)
    generated[:, 0, 1] = 1
    generated[:, 1, :] = 3
    records = [{"prompt_routed_experts": prompt, "routed_experts": generated}]

    load = forced_imbalance._expert_load(records, n_experts=4, layer=0)
    assert load["n_assignments"] == 10          # 5 tokens x k=2, prompt and generated
    np.testing.assert_allclose(load["frequencies"], [0.5, 0.5, 0.0, 0.0])
    np.testing.assert_allclose(
        forced_imbalance._expert_load(records, n_experts=4, layer=1)["frequencies"], [0, 0, 0, 1])
    assert forced_imbalance._has_capture(records)
    assert not forced_imbalance._has_capture([{"prompt_routed_experts": np.zeros((0, 2, 2))}])


def test_checkpoint_path_naming(forced_config_data):
    forced_config_data["imbalance"]["imbalance_levels"] = [0, 12.5]
    cfg = ExperimentConfig.from_dict(forced_config_data)
    assert forced_imbalance.checkpoint_path(cfg, 0) == "org/Mixtral-test"
    assert forced_imbalance.checkpoint_path(cfg, 12.5).endswith("mixtral-imbalance12.5")


def test_repeats_measure_each_level_more_than_once(deployment, forced_config_data):
    """Points sharing a value give the run its own noise floor."""
    forced_config_data["benchmark"]["repeats"] = 3
    cfg = ExperimentConfig.from_dict(forced_config_data)
    ctx = RunContext.create(cfg)
    assert run_pipeline(ctx, cfg) == schema.STATUS_COMPLETED

    labels = [p["label"] for p in ctx.points]
    assert labels == ["imbalance_0", "imbalance_0_r2", "imbalance_0_r3",
                      "imbalance_100", "imbalance_100_r2", "imbalance_100_r3"]
    assert [p["value"] for p in ctx.points] == [0, 0, 0, 100, 100, 100]
    assert [p["repeat"] for p in ctx.points] == [1, 2, 3, 1, 2, 3]
    # every repeat is served and measured separately
    assert all(p["status"] == schema.STATUS_COMPLETED for p in ctx.points)
    assert len({p["metrics_file"] for p in ctx.points}) == 6
    assert len([c for c in deployment.calls if not c["capture"]]) == 6


def test_points_record_competing_work_on_other_npus(deployment, forced_config_data, monkeypatch):
    """A job on NPUs this run does not own is recorded with every point."""
    from moe_reliability.pipelines import common

    busy = {"npu_processes": [{"npu": 4, "pid": 99, "name": "python3"}],
            "foreign_npu_processes": [{"npu": 4, "pid": 99, "name": "python3"}],
            "foreign_npus": [4], "loadavg": [40.0, 30.0, 20.0]}
    monkeypatch.setattr(common, "host_snapshot", lambda visible_devices: busy)

    forced_config_data["hardware"] = {"n_npus": 2, "visible_devices": "0,1"}
    cfg = ExperimentConfig.from_dict(forced_config_data)
    ctx = RunContext.create(cfg)
    assert run_pipeline(ctx, cfg) == schema.STATUS_COMPLETED

    for label in ("imbalance_0", "imbalance_100"):
        p = ctx.point(label)
        assert p["host_before"]["foreign_npus"] == [4]
        assert p["host_after"]["foreign_npus"] == [4]


def test_validation_reads_experts_from_the_serving_stack(deployment, forced_config_data):
    """Router load comes from vLLM's routed-expert capture, not a Hugging Face pass."""
    forced_config_data["imbalance"]["validate_imbalance"] = True
    cfg = ExperimentConfig.from_dict(forced_config_data)
    ctx = RunContext.create(cfg)
    assert run_pipeline(ctx, cfg) == schema.STATUS_COMPLETED
    assert ctx.stage_status("validation") == schema.STATUS_COMPLETED

    # one capture per level, served with expert capture enabled
    captures = [c for c in deployment.calls if c["capture"]]
    assert len(captures) == 2 and all(c["trace_dir"] is None for c in captures)

    for label in ("imbalance_0", "imbalance_100"):
        p = ctx.point(label)
        s = p["validation_summary"]
        assert s["n_assignments"] > 0
        assert 0.0 <= s["max_expert_frequency"] <= 1.0
        assert s["expected_frequency"] == 1 / N_EXPERTS
        doc = io.read_json(ctx.path / p["validation_file"])
        assert doc["n_experts"] == N_EXPERTS and len(doc["per_router_frequencies"]) == N_LAYERS
        np.testing.assert_allclose(sum(doc["frequencies"]), 1.0, atol=1e-6)


def test_points_are_served_in_a_shuffled_but_reproducible_order(deployment, forced_config_data):
    """Serving in parameter order aliases drift during a run onto the parameter."""
    forced_config_data["imbalance"]["imbalance_levels"] = [0, 25, 50, 100]
    forced_config_data["experiment"]["seed"] = 7
    cfg = ExperimentConfig.from_dict(forced_config_data)
    assert cfg.benchmark.shuffle_points is True
    ctx = RunContext.create(cfg)
    assert run_pipeline(ctx, cfg) == schema.STATUS_COMPLETED

    # stored in parameter order, whatever the serving order was
    assert [p["value"] for p in ctx.points] == [0, 25, 50, 100]
    served = [p["value"] for p in sorted(ctx.points, key=lambda p: p["exec_order"])]
    assert sorted(served) == [0, 25, 50, 100]
    assert served != [0, 25, 50, 100], "a 4-point sweep served in parameter order is not shuffled"

    # the same seed gives the same order again
    again = RunContext.create(ExperimentConfig.from_dict(forced_config_data))
    assert run_pipeline(again, cfg) == schema.STATUS_COMPLETED
    assert [p["value"] for p in sorted(again.points, key=lambda p: p["exec_order"])] == served


def test_shuffle_can_be_turned_off(deployment, forced_config_data):
    forced_config_data["imbalance"]["imbalance_levels"] = [0, 25, 50, 100]
    forced_config_data["benchmark"]["shuffle_points"] = False
    cfg = ExperimentConfig.from_dict(forced_config_data)
    ctx = RunContext.create(cfg)
    assert run_pipeline(ctx, cfg) == schema.STATUS_COMPLETED
    assert [p["value"] for p in sorted(ctx.points, key=lambda p: p["exec_order"])] == [0, 25, 50, 100]


def test_repeats_are_served_in_rounds_so_values_interleave(deployment, forced_config_data):
    """A plain shuffle can serve every repeat of one value first; rounds cannot."""
    forced_config_data["imbalance"]["imbalance_levels"] = [0, 100]
    forced_config_data["benchmark"]["repeats"] = 5
    cfg = ExperimentConfig.from_dict(forced_config_data)
    ctx = RunContext.create(cfg)
    assert run_pipeline(ctx, cfg) == schema.STATUS_COMPLETED
    served = sorted(ctx.points, key=lambda p: p["exec_order"])
    for r in range(5):
        # each consecutive pair is one round holding both values once
        assert sorted(p["value"] for p in served[2 * r:2 * r + 2]) == [0, 100]
        assert {p["repeat"] for p in served[2 * r:2 * r + 2]} == {r + 1}


def test_runs_differing_only_in_name_are_ordered_independently(deployment, forced_config_data):
    """Grid runs share a seed; folding in the name keeps them from sharing an order."""
    forced_config_data["imbalance"]["imbalance_levels"] = [0, 25, 50, 100]
    forced_config_data["benchmark"]["repeats"] = 3
    orders = set()
    for name in ("grid-000", "grid-001", "grid-002", "grid-003"):
        forced_config_data["experiment"]["name"] = name
        cfg = ExperimentConfig.from_dict(forced_config_data)
        ctx = RunContext.create(cfg)
        assert run_pipeline(ctx, cfg) == schema.STATUS_COMPLETED
        orders.add(tuple(p["label"] for p in sorted(ctx.points, key=lambda p: p["exec_order"])))
    assert len(orders) > 1


ARMS = ["eplb-off", "eplb-on: server.enable_eplb=true; server.eplb_heat_collection_interval=1"]


def test_arms_are_served_inside_one_run_in_shared_rounds(deployment, forced_config_data):
    """Arms run as separate grid entries let drift alias onto the arm, as on 2026-10-09."""
    forced_config_data["benchmark"].update(repeats=2, arms=ARMS)
    forced_config_data["imbalance"]["validate_imbalance"] = True
    cfg = ExperimentConfig.from_dict(forced_config_data)
    ctx = RunContext.create(cfg)
    assert run_pipeline(ctx, cfg) == schema.STATUS_COMPLETED

    labels = [p["label"] for p in ctx.points]
    assert labels == ["imbalance_0_eplb-off", "imbalance_0_eplb-off_r2", "imbalance_0_eplb-on",
                      "imbalance_0_eplb-on_r2", "imbalance_100_eplb-off", "imbalance_100_eplb-off_r2",
                      "imbalance_100_eplb-on", "imbalance_100_eplb-on_r2"]
    assert [p["arm"] for p in ctx.points] == ["eplb-off"] * 2 + ["eplb-on"] * 2 + ["eplb-off"] * 2 + ["eplb-on"] * 2
    # validation once per level, on the configuration as written
    validation = [c for c in deployment.calls if c["capture"]]
    assert len(validation) == 2 and all(c["eplb"] is None for c in validation)
    # each benchmark call is served with its own arm's settings
    served = [c for c in deployment.calls if not c["capture"]]
    by_order = sorted(ctx.points, key=lambda p: p["exec_order"])
    for p, call in zip(by_order, served):
        assert (call["eplb"] is not None) == (p["arm"] == "eplb-on")
    # every round holds both arms at every level
    for r in (1, 2):
        rnd = [p for p in by_order if p["repeat"] == r]
        assert {(p["value"], p["arm"]) for p in rnd} == {(v, a) for v in (0, 100) for a in ("eplb-off", "eplb-on")}

    summary = ResultsStore(cfg.output.results_dir).get(ctx.run_id).summary()
    on = summary[summary["arm"] == "eplb-on"]
    assert on["enable_eplb"].all() and not summary[summary["arm"] == "eplb-off"]["enable_eplb"].any()


def test_an_arm_may_not_change_what_validation_shares(forced_config_data):
    from moe_reliability.config import ConfigError

    forced_config_data["benchmark"]["arms"] = ["a", "b: imbalance.imbalance_levels=[0]"]
    with pytest.raises(ConfigError, match="may only change"):
        ExperimentConfig.from_dict(forced_config_data)


def test_eplb_activity_is_read_from_the_point_s_own_log(deployment, forced_config_data, monkeypatch):
    """Only the log says whether a plan was applied; the map says what was planned."""
    from moe_reliability.pipelines import common

    forced_config_data["benchmark"]["arms"] = ARMS
    cfg = ExperimentConfig.from_dict(forced_config_data)
    ctx = RunContext.create(cfg)
    log_path = ctx.abspath(schema.LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("Expert hotness imbalance, current: mean=9.000 max=9.0, updated: mean=1.0 max=1.0\n")

    def serve(cfg, *args, **kwargs):
        if cfg.server.enable_eplb:
            with open(log_path, "a") as f:
                f.write("Expert hotness imbalance, current: mean=2.000 max=2.5, updated: mean=1.000 max=1.1\n")
                f.write("Full expert weight update cycle completed, total_layers=26\n" * cfg.hardware.n_npus)
        return deployment(cfg, *args, **kwargs)

    monkeypatch.setattr(common, "serve_and_measure", serve)
    assert run_pipeline(ctx, cfg) == schema.STATUS_COMPLETED
    for p in ctx.points:
        if p["arm"] == "eplb-on":
            # only this point's lines, not the one written before it
            assert p["eplb_activity"]["plans"] == 1 and p["eplb_activity"]["applied_cycles"] == 1
            assert p["eplb_activity"]["hotness"][0][0] == 2.0
        else:
            assert "eplb_activity" not in p


def test_ignore_eos_reaches_the_client_and_is_off_by_default(monkeypatch, forced_config_data):
    """client.ignore_eos is what holds the prefill/decode mix fixed across levels.

    It has to reach the call that serves every pass - warm-up, benchmark and the
    validation capture all go through it - and stay off unless asked for, so an
    existing config keeps sending the requests it always sent.
    """
    from moe_reliability.core import vllm_serving
    from moe_reliability.pipelines.common import serve_and_measure

    seen = []

    async def fake(*args, **kwargs):
        seen.append(kwargs.get("ignore_eos"))
        return []

    monkeypatch.setattr(vllm_serving, "measure_vllm_throughput", fake)
    cfg = ExperimentConfig.from_dict(forced_config_data)
    serve_and_measure(cfg, "m", ["p"], trace_dir=None)
    forced_config_data["client"] = {**forced_config_data.get("client", {}), "ignore_eos": True}
    cfg = ExperimentConfig.from_dict(forced_config_data)
    serve_and_measure(cfg, "m", ["p"], trace_dir=None)
    assert seen == [False, True]


def _metrics_file(path, prompts, lengths):
    import gzip
    import json

    with gzip.open(path, "wt") as f:
        json.dump({"requests": [{"prompt": p, "num_output_tokens": n} for p, n in zip(prompts, lengths)]}, f)
    return str(path)


def test_lengths_from_replays_each_prompts_length_by_content(monkeypatch, forced_config_data, tmp_path):
    """client.lengths_from gives each prompt the length it had in the recorded point.

    Matched by content, not position: warm-up serves a prefix and the recorded
    point may have served the prompts in another order, and either way a prompt
    must get its own length, not its neighbour's.
    """
    from moe_reliability.core import vllm_serving
    from moe_reliability.pipelines.common import serve_and_measure

    prompts = [[{"role": "user", "content": f"q{i}"}] for i in range(4)]
    path = _metrics_file(tmp_path / "m.json.gz", list(reversed(prompts)), [40, 30, 20, 10])
    seen = []

    async def fake(*args, **kwargs):
        seen.append((kwargs["max_new_tokens"], kwargs["ignore_eos"]))
        return []

    monkeypatch.setattr(vllm_serving, "measure_vllm_throughput", fake)
    forced_config_data["client"] = {**forced_config_data["client"], "lengths_from": path, "ignore_eos": True}
    cfg = ExperimentConfig.from_dict(forced_config_data)
    serve_and_measure(cfg, "m", prompts, trace_dir=None)
    serve_and_measure(cfg, "m", prompts[:2], trace_dir=None)
    assert seen == [([10, 20, 30, 40], True), ([10, 20], True)]

    with pytest.raises(ValueError, match="1 of 1 prompts have no recorded length"):
        serve_and_measure(cfg, "m", [[{"role": "user", "content": "unseen"}]], trace_dir=None)


def test_lengths_from_is_refused_without_ignore_eos_or_with_validation(forced_config_data, tmp_path):
    """Without ignore_eos a replayed length is only a cap; validation serves prompts it never saw."""
    from moe_reliability.config import ConfigError

    path = _metrics_file(tmp_path / "m.json.gz", [["p"]], [5])
    forced_config_data["client"] = {**forced_config_data["client"], "lengths_from": path}
    with pytest.raises(ConfigError, match="needs client.ignore_eos"):
        ExperimentConfig.from_dict(forced_config_data)

    forced_config_data["client"]["ignore_eos"] = True
    forced_config_data["imbalance"] = {**forced_config_data["imbalance"], "validate_imbalance": True}
    with pytest.raises(ConfigError, match="validation capture"):
        ExperimentConfig.from_dict(forced_config_data)

    forced_config_data["imbalance"]["validate_imbalance"] = False
    forced_config_data["client"]["lengths_from"] = str(tmp_path / "missing.json.gz")
    with pytest.raises(ConfigError, match="no such metrics file"):
        ExperimentConfig.from_dict(forced_config_data)


def test_run_batch_sends_a_per_prompt_length_to_each_request():
    """A list of lengths reaches each request as its own max_tokens."""
    import asyncio
    import types

    from moe_reliability.core.vllm_serving import run_batch

    sent = {}

    async def create(**kwargs):
        sent[kwargs["messages"][0]["content"]] = kwargs["max_tokens"]
        usage = types.SimpleNamespace(completion_tokens=kwargs["max_tokens"], prompt_tokens=3)
        msg = types.SimpleNamespace(content="x")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=usage)

    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    prompts = [[{"role": "user", "content": c}] for c in "abc"]
    asyncio.run(run_batch(client, "m", prompts, max_new_tokens=[7, 8, 9, 10], stream=False))
    assert sent == {"a": 7, "b": 8, "c": 9}
