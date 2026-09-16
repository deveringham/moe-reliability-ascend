"""Command line interfaces of both packages."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from moe_reliability import cli
from moe_reliability_results import ResultsStore
from moe_reliability_results import cli as results_cli
from conftest import write_toml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_environment_changes(monkeypatch):
    calls = []

    def fake_environment(cfg, require_devices=True):
        calls.append(require_devices)
        return {"hostname": "test"}

    monkeypatch.setattr(cli, "_environment", fake_environment)
    return calls


def test_init_validate_dry_run(tmp_path, capsys):
    out = tmp_path / "c.toml"
    assert cli.main(["init", "forced_imbalance", "-o", str(out)]) == 0
    assert cli.main(["init", "forced_imbalance", "-o", str(out)]) == 2
    assert cli.main(["validate", str(out), "--set", "server.batch_size=128"]) == 0
    assert "batch size 128" in capsys.readouterr().out
    assert cli.main(["run", str(out), "--dry-run", "--results-dir", str(tmp_path / "r")]) == 0
    assert not (tmp_path / "r").exists()
    assert cli.main(["validate", str(out), "--set", "server.batch_size=-1"]) == 2
    assert "batch_size must be > 0" in capsys.readouterr().err


def test_shipped_configs_validate():
    for path in sorted((ROOT / "configs" / "examples").glob("*.toml")):
        assert cli.main(["validate", str(path)]) == 0, path


def test_run_resume_analyze_and_browse(deployment, tmp_path, forced_config_data, results_dir, capsys,
                                       no_environment_changes):
    config = write_toml(tmp_path / "run.toml", forced_config_data)
    assert cli.main(["run", str(config), "--no-log-file", "--run-id", "cli-run"]) == 0
    assert (results_dir / "cli-run" / "manifest.json").is_file()
    assert cli.main(["resume", "cli-run", "--results-dir", str(results_dir), "--no-log-file"]) == 0
    assert "already completed" in capsys.readouterr().out
    assert cli.main(["analyze", "cli-run", "--results-dir", str(results_dir), "--no-log-file"]) == 0
    assert no_environment_changes == [True, False]  # run needs NPUs, re-analysis does not

    assert cli.main(["list", "--results-dir", str(results_dir)]) == 0
    assert "cli-run" in capsys.readouterr().out
    assert cli.main(["show", "cli", "--results-dir", str(results_dir)]) == 0
    assert "imbalance_100" in capsys.readouterr().out

    rd = ["--results-dir", str(results_dir)]
    assert results_cli.main(rd + ["summary", "-f", "n_npus=2", "-q", "sweep_value > 0"]) == 0
    assert "imbalance" in capsys.readouterr().out
    csv = tmp_path / "summary.csv"
    assert results_cli.main(rd + ["summary", "-o", str(csv)]) == 0
    assert len(csv.read_text().splitlines()) == 3
    out = tmp_path / "requests.json"
    assert results_cli.main(rd + ["requests", "-o", str(out)]) == 0
    assert len(json.loads(out.read_text())) == 48
    assert results_cli.main(rd + ["plot", "cli-run", "-o", str(tmp_path / "figs")]) == 0
    assert list((tmp_path / "figs").glob("*.png"))
    assert results_cli.main(rd + ["show", "missing"]) == 2


def test_failed_run_exit_code(deployment, tmp_path, forced_config_data):
    deployment.fail_models.add("org/Mixtral-test")
    config = write_toml(tmp_path / "run.toml", forced_config_data)
    assert cli.main(["run", str(config), "--no-log-file", "--run-id", "failing"]) == 1
    assert ResultsStore(forced_config_data["output"]["results_dir"]).get("failing").status == "partial"


def test_grid_runs_are_idempotent_and_share_workloads(deployment, tmp_path, synthetic_config_data, results_dir):
    write_toml(tmp_path / "base.toml", synthetic_config_data)
    (tmp_path / "grid.toml").write_text("""
[grid]
name = "sw"
base = "base.toml"
share_workloads = true
[set]
"benchmark.enable_profiling" = false
[matrix]
"server.batch_size" = [8, 16]
""")
    args = ["grid", str(tmp_path / "grid.toml"), "--no-log-file"]
    assert cli.main(args + ["--dry-run"]) == 0
    assert cli.main(args) == 0
    assert sum(c["capture"] for c in deployment.calls) == 1  # activations captured once
    store = ResultsStore(results_dir)
    runs = store.runs(experiment="synthetic_workloads")
    assert [r.config["experiment"]["name"] for r in runs] == ["sw-000", "sw-001"]
    assert runs[1].manifest["inputs"]["workloads"]["run_id"] == runs[0].id
    assert runs[1].manifest["grid"]["assignments"] == {"server.batch_size": 16}

    n_calls = len(deployment.calls)
    assert cli.main(args) == 0  # everything completed: nothing to do
    assert len(deployment.calls) == n_calls

    configs = tmp_path / "expanded"
    assert cli.main(args + ["--write-configs", str(configs)]) == 0
    assert sorted(p.name for p in configs.iterdir()) == ["sw-000.toml", "sw-001.toml"]


def test_environment_errors_exit_with_code_3(monkeypatch, tmp_path, forced_config_data, capsys):
    from moe_reliability.environment import AscendEnvironmentError

    def unavailable(cfg, require_devices=True):
        raise AscendEnvironmentError("the CANN environment is not activated")

    monkeypatch.setattr(cli, "_environment", unavailable)
    config = write_toml(tmp_path / "run.toml", forced_config_data)
    assert cli.main(["run", str(config), "--no-log-file"]) == 3
    err = capsys.readouterr().err
    assert "CANN environment is not activated" in err and "moe-reliability doctor" in err
    assert not (tmp_path / "results").exists()  # nothing is created before the environment is usable


def test_doctor(monkeypatch, tmp_path, forced_config_data, capsys):
    from moe_reliability import environment

    seen = {}

    def fake_diagnose(n_npus=None, visible_devices=""):
        seen.update(n_npus=n_npus, visible_devices=visible_devices)
        return [("CANN environment", True, "/usr/local/Ascend/ascend-toolkit/latest"),
                ("NPUs visible", n_npus <= 4, "4 NPUs")]

    monkeypatch.setattr(environment, "diagnose", fake_diagnose)
    assert cli.main(["doctor", "--n-npus", "4"]) == 0
    assert "2/2 checks passed" in capsys.readouterr().out
    config = write_toml(tmp_path / "run.toml", forced_config_data)
    assert cli.main(["doctor", str(config), "--set", "hardware.n_npus=8"]) == 3
    assert seen == {"n_npus": 8, "visible_devices": ""}
    assert "[FAIL] NPUs visible" in capsys.readouterr().out
