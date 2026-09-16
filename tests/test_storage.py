from __future__ import annotations

import gzip
import json
import math
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

from moe_reliability.config import ExperimentConfig
from moe_reliability.runs import RunContext, RunError, make_run_id, resolve_run_dir
from moe_reliability_results import io, schema


def test_to_jsonable_handles_arrays_tensors_and_keys():
    obj = {1.5: np.int16(3), (1, 2): np.array([[1, 2]], dtype=np.int16), "t": torch.tensor([0.5, 1.0]),
           "nan": float("nan"), "tuple": (1, 2), "np": np.float32(2.5)}
    out = io.to_jsonable(obj)
    assert out == {"1.5": 3, "1,2": [[1, 2]], "t": [0.5, 1.0], "nan": None, "tuple": [1, 2], "np": 2.5}
    json.dumps(out, allow_nan=False)
    with pytest.raises(TypeError):
        io.to_jsonable(object())


@pytest.mark.parametrize("compress", [False, True])
def test_json_round_trip(tmp_path, compress):
    target = io.write_json(tmp_path / "a" / "doc.json", {"x": [1, 2, math.inf]}, compress=compress)
    assert target.name == ("doc.json.gz" if compress else "doc.json")
    assert io.read_json(tmp_path / "a" / "doc.json") == {"x": [1, 2, None]}
    assert io.exists(tmp_path / "a" / "doc.json.gz")
    # switching compression removes the stale variant
    io.write_json(tmp_path / "a" / "doc.json", {"y": 1}, compress=not compress)
    assert len(list((tmp_path / "a").iterdir())) == 1
    assert io.read_json(tmp_path / "a" / "doc.json") == {"y": 1}


def test_jsonl_round_trip(tmp_path):
    path, n = io.write_jsonl(tmp_path / "r.jsonl", ({"i": i, "a": np.arange(i)} for i in range(5)), compress=True)
    assert n == 5 and path.suffix == ".gz"
    with gzip.open(path, "rt") as f:
        assert len(f.readlines()) == 5
    assert [r["a"] for r in io.iter_jsonl(tmp_path / "r.jsonl")][3] == [0, 1, 2]
    with pytest.raises(FileNotFoundError):
        io.read_json(tmp_path / "missing.json")


def test_point_labels():
    assert schema.point_label("alpha", 0.8) == "alpha_0.8"
    assert schema.point_label("alpha", 1.0) == "alpha_1.0"
    assert schema.point_label("imbalance_level", 100) == "imbalance_100"
    with pytest.raises(TypeError):
        schema.format_value(True)


def test_run_context_lifecycle(forced_config_data):
    cfg = ExperimentConfig.from_dict(forced_config_data)
    ctx = RunContext.create(cfg, environment={"hostname": "test"})
    assert ctx.run_id.split("_")[1:] == ["imbalance", "mixtral", "npu2", "bs16", "test"]
    second = RunContext.create(cfg, run_id=ctx.run_id)
    assert second.run_id == f"{ctx.run_id}-2"

    ctx.init_stages(["a", "b"])
    with ctx.stage("a"):
        assert io.read_json(ctx.path / schema.MANIFEST_FILE)["stages"]["a"]["status"] == "running"
    with pytest.raises(ValueError):
        with ctx.stage("b"):
            raise ValueError("bad")
    assert ctx.manifest["stages"]["b"]["status"] == "failed"
    assert "ValueError: bad" in ctx.manifest["stages"]["b"]["error"]
    assert ctx.finalize() == "failed"

    ctx.ensure_points([0, 100])
    ctx.ensure_points([0, 100])
    with pytest.raises(RunError):
        ctx.ensure_points([0, 50])
    ctx.update_point("imbalance_100", status="completed")
    reopened = RunContext.open(ctx.run_id, cfg.output.results_dir)
    assert reopened.point("imbalance_100")["status"] == "completed"
    assert reopened.config().to_dict() == cfg.to_dict()


def test_resolve_run_dir(forced_config_data, results_dir):
    cfg = ExperimentConfig.from_dict(forced_config_data)
    ctx = RunContext.create(cfg, run_id="20260101-000000_example")
    assert resolve_run_dir("20260101", results_dir) == ctx.path.resolve()
    assert resolve_run_dir(ctx.path, "elsewhere") == ctx.path.resolve()
    RunContext.create(cfg, run_id="20260101-000001_other")
    with pytest.raises(RunError, match="ambiguous"):
        resolve_run_dir("20260101", results_dir)
    with pytest.raises(RunError, match="no run"):
        resolve_run_dir("nope", results_dir)
    assert make_run_id(cfg).endswith("_imbalance_mixtral_npu2_bs16_test")


def test_tee_output_captures_subprocess_output(tmp_path):
    log_file = tmp_path / "logs" / "run.log"
    script = textwrap.dedent(f"""
        import subprocess, sys
        sys.path[:0] = {sys.path!r}
        from moe_reliability.logs import tee_output, log
        with tee_output({str(log_file)!r}):
            log("parent line")
            subprocess.run([sys.executable, "-c", "import sys; print('child out'); print('child err', file=sys.stderr)"])
        print("after")
    """)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    text = log_file.read_text()
    assert "parent line" in text and "child out" in text and "child err" in text
    assert "after" not in text and "after" in out.stdout
