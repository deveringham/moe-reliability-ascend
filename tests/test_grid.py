from __future__ import annotations

from pathlib import Path

import pytest

from moe_experiments.config import ConfigError
from moe_experiments.grid import load_grid, workload_share_key

ROOT = Path(__file__).resolve().parents[1]


def _grid(tmp_path, base, body):
    from conftest import write_toml

    write_toml(tmp_path / "base.toml", base)
    path = tmp_path / "grid.toml"
    path.write_text(body)
    return path


def test_cartesian_and_zipped_axes(tmp_path, forced_config_data):
    path = _grid(tmp_path, forced_config_data, """
[grid]
name = "g"
base = "base.toml"
[set]
"benchmark.n_samples" = 10
[matrix]
"model.model_id,model.model_name" = [["org/a", "a"], ["org/b", "b"]]
"hardware.n_npus" = [2, 4, 8]
""")
    grid = load_grid(path, overrides=["server.batch_size=64"])
    assert grid.n_runs == 6 and grid.n_configurations() == 12
    first, last = grid.entries[0], grid.entries[-1]
    assert first.name == "g-000" and first.config.experiment.name == "g-000"
    assert (first.config.model.model_id, first.config.hardware.n_npus) == ("org/a", 2)
    assert (last.config.model.model_name, last.config.hardware.n_npus) == ("b", 8)
    assert all(e.config.benchmark.n_samples == 10 and e.config.server.batch_size == 64 for e in grid.entries)
    assert last.assignments == {"model.model_id": "org/b", "model.model_name": "b", "hardware.n_npus": 8}


@pytest.mark.parametrize("body, message", [
    ('[grid]\nbase = "base.toml"\n', "grid.name"),
    ('[grid]\nname = "g"\n', "grid.base"),
    ('[grid]\nname = "g"\nbase = "base.toml"\n[matrix]\n"server.batch_size" = []\n', "non-empty"),
    ('[grid]\nname = "g"\nbase = "base.toml"\n[matrix]\n"a.b,c.d" = [[1]]\n', "list of 2"),
    ('[grid]\nname = "g"\nbase = "base.toml"\n[matrix]\n"server.batch_size" = [0]\n', "grid entry 0"),
    ('[grid]\nname = "g"\nbase = "base.toml"\n[other]\n', "unknown top-level"),
])
def test_grid_errors(tmp_path, forced_config_data, body, message):
    with pytest.raises(ConfigError, match=message):
        load_grid(_grid(tmp_path, forced_config_data, body))


@pytest.mark.parametrize("name, runs, points", [
    ("forced_imbalance_infrastructure.toml", 30, 120),
    ("synthetic_workloads_infrastructure.toml", 9, 126),
])
def test_shipped_grids_exceed_100_configurations(name, runs, points):
    grid = load_grid(ROOT / "configs" / "grids" / name)
    assert grid.n_runs == runs
    assert grid.n_configurations() == points > 100


def test_workload_share_key(synthetic_config_data, forced_config_data):
    from moe_experiments.config import ExperimentConfig

    a = ExperimentConfig.from_dict(synthetic_config_data)
    b = ExperimentConfig.from_dict(synthetic_config_data, ["hardware.n_npus=8", "server.batch_size=512"])
    c = ExperimentConfig.from_dict(synthetic_config_data, ["client.max_new_tokens=50"])
    assert workload_share_key(a) == workload_share_key(b) != workload_share_key(c)
    assert workload_share_key(ExperimentConfig.from_dict(forced_config_data)) is None
