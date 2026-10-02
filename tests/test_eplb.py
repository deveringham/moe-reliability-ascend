"""EPLB wiring.

vLLM's own ``--enable-eplb`` does not reach the vllm-ascend implementation: the
Ascend subsystem is gated on ``additional_config.eplb_config.dynamic_eplb`` plus
a ``DYNAMIC_EPLB`` environment variable. Passing the wrong one fails silently -
the server starts, serves, and never rebalances - so the flag that is actually
sent is worth asserting on.
"""

from __future__ import annotations

import json
import types

import pytest

from moe_reliability.config import ExperimentConfig
from moe_reliability.core import vllm_serving
from moe_reliability.pipelines.common import eplb_cycle_iterations, eplb_settings


def _cfg(**server):
    data = {
        "experiment": {"type": "synthetic_workloads", "name": "eplb"},
        "model": {"model_id": "deepseek-ai/DeepSeek-V2-Lite-Chat", "model_name": "deepseek-v2"},
        "server": server,
    }
    return ExperimentConfig.from_dict(data)


def test_eplb_settings_is_none_when_disabled():
    assert eplb_settings(_cfg(enable_eplb=False)) is None


def test_eplb_settings_uses_the_vllm_ascend_key_names():
    cfg = _cfg(enable_eplb=True, eplb_policy_type=1, eplb_num_redundant_experts=2,
               eplb_heat_collection_interval=50, eplb_algorithm_execution_interval=10)
    assert eplb_settings(cfg) == {
        "dynamic_eplb": True,
        "eplb_policy_type": 1,
        "num_redundant_experts": 2,
        "expert_heat_collection_interval": 50,
        "algorithm_execution_interval": 10,
    }


def test_eplb_settings_records_the_placement_when_asked():
    cfg = _cfg(enable_eplb=True)
    assert "expert_map_record_path" not in eplb_settings(cfg)
    assert eplb_settings(cfg, "/runs/p/map.json")["expert_map_record_path"] == "/runs/p/map.json"


def test_eplb_cycle_counts_collection_planning_and_one_pass_per_layer():
    cfg = _cfg(enable_eplb=True, eplb_heat_collection_interval=600,
               eplb_algorithm_execution_interval=50)
    assert eplb_cycle_iterations(cfg) == 650          # lower bound, layers unknown
    assert eplb_cycle_iterations(cfg, n_moe_layers=26) == 676


class _StubProcess:
    def poll(self):
        return None


@pytest.fixture
def launched(monkeypatch):
    """Capture the argv and environment start_vllm_server would launch with."""
    captured = {}

    def fake_popen(cmd, start_new_session=False, env=None):
        captured["cmd"] = cmd
        captured["env"] = env
        return _StubProcess()

    monkeypatch.setattr(vllm_serving.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(vllm_serving.urllib.request, "urlopen",
                        lambda url, timeout=0: types.SimpleNamespace(getcode=lambda: 200))
    return captured


def test_server_launches_without_eplb_by_default(launched):
    vllm_serving.start_vllm_server("some/model", eplb=None)
    assert "--additional-config" not in launched["cmd"]
    # The flag that does not work must not be sent either.
    assert "--enable-eplb" not in launched["cmd"]
    assert "DYNAMIC_EPLB" not in launched["env"]


def test_server_passes_eplb_through_additional_config_and_env(launched):
    settings = {"dynamic_eplb": True, "num_redundant_experts": 1}
    vllm_serving.start_vllm_server("some/model", eplb=settings)

    cmd = launched["cmd"]
    assert "--enable-eplb" not in cmd  # would not reach vllm-ascend
    payload = json.loads(cmd[cmd.index("--additional-config") + 1])
    assert payload == {"eplb_config": settings}
    # vllm-ascend asserts on this variable and refuses to start without it.
    assert launched["env"]["DYNAMIC_EPLB"] == "true"
    assert "EXPERT_MAP_RECORD" not in launched["env"]


def test_recording_the_placement_sets_its_own_env_flag(launched):
    vllm_serving.start_vllm_server(
        "some/model", eplb={"dynamic_eplb": True, "expert_map_record_path": "/runs/p/map.json"})
    assert launched["env"]["EXPERT_MAP_RECORD"] == "true"
