"""The EPLB gate patch: that it applies, that it is loud when it cannot, and
that a run cannot quietly measure an unpatched server."""

from __future__ import annotations

import pytest

from moe_reliability import eplb_gate
from moe_reliability.config import ConfigError, ExperimentConfig
from moe_reliability.pipelines import common

# The anchors the patch needs, in the order and indentation vllm-ascend 0.23 has
# them. This is an excerpt, not the whole file: enough that a change to any
# anchor fails here rather than on the node at the start of a 3.6 NPU-hour run.
UPSTREAM = '''from multiprocessing import Process, Queue
from typing import Any

import numpy as np
import torch
from vllm.distributed import get_ep_group
from vllm.logger import logger


class EplbWorker:
    def __init__(self, shared_dict, policy_type):
        self.policy_type = policy_type

    def do_update(self):
        torch.set_num_threads(1)
        load_info = self.fetch_and_sum_load_info()
        if load_info is None:
            return

        # Get the updated expert table based on the workload information
        old_placement = self.global2local(self.old_expert_maps, self.num_local_experts)
        _, _, new_placement = self.calculate_rebalance_experts(load_info, old_placement)

        if self.rank_id == 0:
            pass
'''


def test_patch_applies_to_the_upstream_anchors():
    patched = eplb_gate.apply_to(UPSTREAM)
    assert eplb_gate.is_patched(patched)
    assert not eplb_gate.is_patched(UPSTREAM)
    # The discarded verdict is now read, and the gate runs before the placement
    # is used for anything.
    assert "changed, _, new_placement = self.calculate_rebalance_experts" in patched
    assert "_, _, new_placement = self.calculate_rebalance_experts" not in patched
    assert patched.index("_eplb_gate_keeps_placement(changed") > patched.index("def do_update")
    # It stays valid Python, which a string-replacement patch does not get for free.
    compile(patched, "eplb_worker.py", "exec")


def test_patch_is_idempotent():
    once = eplb_gate.apply_to(UPSTREAM)
    assert eplb_gate.apply_to(once) == once


def test_patch_refuses_a_source_it_does_not_recognise():
    # A version bump that rewrites the call site must fail loudly, not silently
    # leave the gate uninstalled.
    moved_on = UPSTREAM.replace("_, _, new_placement = self.calculate_rebalance_experts(load_info, old_placement)",
                                "new_placement = self.plan(load_info, old_placement)")
    with pytest.raises(eplb_gate.PatchError, match="anchor appears 0 times"):
        eplb_gate.apply_to(moved_on)


def test_patch_refuses_an_ambiguous_anchor():
    with pytest.raises(eplb_gate.PatchError, match="anchor appears 2 times"):
        eplb_gate.apply_to(UPSTREAM + "\nclass EplbWorker:\n")


@pytest.mark.parametrize("threshold, honor, expected", [
    (0.0, False, {}),
    (2.43, False, {eplb_gate.ENV_MIN_IMBALANCE: "2.43"}),
    (0.0, True, {eplb_gate.ENV_HONOR_CHANGED: "1"}),
    (2.43, True, {eplb_gate.ENV_MIN_IMBALANCE: "2.43", eplb_gate.ENV_HONOR_CHANGED: "1"}),
])
def test_server_env(threshold, honor, expected):
    assert eplb_gate.server_env(threshold, honor) == expected


def test_gate_counts_come_back_out_of_the_server_log():
    # Every rank's planner logs its own verdict, so a four-rank point logs four
    # lines per decision.
    log = "".join([f"[eplb/worker] {eplb_gate.LOG_SHUT}: imbalance 1.120x, under the 2.430x threshold\n"] * 8
                  + [f"[eplb/worker] {eplb_gate.LOG_OPEN}: imbalance 3.490x, over the 2.430x threshold\n"] * 4)
    activity = common.eplb_activity(log, n_ranks=4)
    assert (activity["gate_shut"], activity["gate_open"]) == (2, 1)
    assert activity["gate_consistent"]
    assert "gate shut 2/3 times" in common.describe_eplb_activity(
        {**activity, "plans": 3, "applied_cycles": 1, "hotness": [[1.12, 1.12, 1.02, 1.02]], "stalled": False})


def test_ranks_disagreeing_is_reported():
    log = f"{eplb_gate.LOG_SHUT}: x\n" * 3 + f"{eplb_gate.LOG_OPEN}: x\n" * 4
    activity = common.eplb_activity(log, n_ranks=4)
    assert not activity["gate_consistent"]
    assert "the ranks did not all reach the same verdict" in common.describe_eplb_activity(
        {**activity, "plans": 2, "applied_cycles": 1, "hotness": [[1.0, 1.0, 1.0, 1.0]], "stalled": False})


def test_an_ungated_point_records_no_gate_fields():
    activity = common.eplb_activity("Full expert weight update cycle completed\n" * 4, n_ranks=4)
    assert "gate_shut" not in activity


def _cfg(**server):
    return ExperimentConfig.from_dict({
        "experiment": {"type": "synthetic_workloads", "name": "gate"},
        "model": {"model_id": "deepseek-ai/DeepSeek-V2-Lite-Chat", "model_name": "deepseek-v2"},
        "server": server,
    })


def test_gate_needs_eplb_enabled():
    with pytest.raises(ConfigError, match="needs server.enable_eplb"):
        _cfg(eplb_min_imbalance=2.43)


def test_gate_refuses_record_only_mode():
    with pytest.raises(ConfigError, match="record-only mode never applies a placement"):
        _cfg(enable_eplb=True, eplb_min_imbalance=2.43, eplb_record_map=True)


def test_gate_refuses_a_policy_without_a_verdict():
    with pytest.raises(ConfigError, match="needs eplb_policy_type 1 or 2"):
        _cfg(enable_eplb=True, eplb_min_imbalance=2.43, eplb_policy_type=3)


def test_a_threshold_under_one_is_refused():
    # Busiest-rank-over-mean is never under 1, so such a threshold gates nothing.
    with pytest.raises(ConfigError, match="is below 1"):
        _cfg(enable_eplb=True, eplb_min_imbalance=0.9)


def test_threshold_zero_is_plain_eplb():
    cfg = _cfg(enable_eplb=True, eplb_min_imbalance=0.0)
    assert cfg.server.eplb_min_imbalance == 0.0
    assert eplb_gate.server_env(cfg.server.eplb_min_imbalance, cfg.server.eplb_honor_changed) == {}


def test_a_gated_arm_refuses_to_run_against_an_unpatched_install(tmp_path, monkeypatch):
    """The failure this guard exists for: an unpatched server ignores the
    environment variable and serves plain EPLB, which would read as 'gating
    changes nothing' - the one answer the run must not reach by accident."""
    cfg = _cfg(enable_eplb=True, eplb_min_imbalance=2.43)
    fake = tmp_path / "vllm_ascend"
    (fake / eplb_gate.TARGET).parent.mkdir(parents=True)
    (fake / eplb_gate.TARGET).write_text(UPSTREAM)
    monkeypatch.setitem(__import__("sys").modules, "vllm_ascend",
                        type("m", (), {"__file__": str(fake / "__init__.py")}))
    with pytest.raises(RuntimeError, match="does not carry the gate patch"):
        common._require_eplb_gate_patch(cfg)
    (fake / eplb_gate.TARGET).write_text(eplb_gate.apply_to(UPSTREAM))
    common._require_eplb_gate_patch(cfg)  # patched: no complaint


def test_an_ungated_config_does_not_look_for_the_patch(tmp_path, monkeypatch):
    cfg = _cfg(enable_eplb=True)
    monkeypatch.setitem(__import__("sys").modules, "vllm_ascend", None)
    common._require_eplb_gate_patch(cfg)
