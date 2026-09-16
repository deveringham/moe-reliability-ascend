from __future__ import annotations

import numpy as np
import pytest
import torch

from moe_reliability.config import ExperimentConfig
from moe_reliability.pipelines import forced_imbalance, run_pipeline
from moe_reliability.runs import RunContext
from moe_reliability_results import ResultsStore, io, schema


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
    assert cfg.benchmark.save_request_metrics is False
    ctx = RunContext.create(cfg)
    assert run_pipeline(ctx, cfg) == schema.STATUS_COMPLETED

    for label in ("imbalance_0", "imbalance_100"):
        p = ctx.point(label)
        assert p["metrics_file"] is None and p["request_summary"]["n_requests"] == 24
        assert p["npu_trace_views"] == [f"traces/{label}/worker_{r}/ASCEND_PROFILER_OUTPUT/trace_view.json"
                                        for r in (0, 1)]
        assert p["trace_parse_error"] is None
        # raw NPU profiler data is not in the input format of the fused-MoE kernel analysis
        assert "*rank*.pt.trace.json.gz" in p["trace_error"] and p["trace_metrics_file"] is None
    assert ctx.stage_status("trace_analysis") == schema.STATUS_COMPLETED
    assert ctx.stage_status("hta") == schema.STATUS_SKIPPED


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


def test_expert_load_matches_probe_collation():
    class Probe:
        n_experts = 4

        def get_active_experts(self):
            # [batch, seq, k, n_routers]: router 0 always picks experts 0 and 1
            t = torch.zeros((1, 5, 2, 3), dtype=torch.int64)
            t[:, :, 1, 0] = 1
            t[:, :, :, 1] = 3
            return t

    load = forced_imbalance._expert_load(Probe(), router_id=0)
    assert load["n_assignments"] == 10
    np.testing.assert_allclose(load["frequencies"], [0.5, 0.5, 0.0, 0.0])
    np.testing.assert_allclose(forced_imbalance._expert_load(Probe(), router_id=1)["frequencies"], [0, 0, 0, 1])


def test_checkpoint_path_naming(forced_config_data):
    forced_config_data["imbalance"]["imbalance_levels"] = [0, 12.5]
    cfg = ExperimentConfig.from_dict(forced_config_data)
    assert forced_imbalance.checkpoint_path(cfg, 0) == "org/Mixtral-test"
    assert forced_imbalance.checkpoint_path(cfg, 12.5).endswith("mixtral-imbalance12.5")
