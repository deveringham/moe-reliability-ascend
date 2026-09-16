from __future__ import annotations

import base64
import hashlib
import os
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from moe_reliability import environment as env
from moe_reliability.environment import AscendEnvironmentError


@pytest.fixture
def clean_env(monkeypatch):
    for key in ("ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH", "ATB_HOME_PATH", "ASCEND_RT_VISIBLE_DEVICES"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))
    return monkeypatch


def fake_npu_runtime(monkeypatch, available=True, count=8):
    for name in ("torch_npu", "torch_npu.contrib", "torch_npu.contrib.transfer_to_npu"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["torch_npu"].contrib = sys.modules["torch_npu.contrib"]
    sys.modules["torch_npu.contrib"].transfer_to_npu = sys.modules["torch_npu.contrib.transfer_to_npu"]
    npu = SimpleNamespace(is_available=lambda: available, device_count=lambda: count,
                          get_device_name=lambda i: "Ascend910B1", empty_cache=lambda: None)
    monkeypatch.setattr(torch, "npu", npu, raising=False)


def test_apply_environment(clean_env):
    env.apply_environment({"HCCL_CONNECT_TIMEOUT": "1200"}, visible_devices="4, 5,6,7")
    assert os.environ["HCCL_CONNECT_TIMEOUT"] == "1200"
    assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "4,5,6,7"
    assert os.path.dirname(sys.executable) in os.environ["PATH"].split(os.pathsep)


def test_cann_must_be_activated(clean_env, tmp_path):
    with pytest.raises(AscendEnvironmentError, match="set_env.sh"):
        env.check_cann()
    clean_env.setenv("ASCEND_TOOLKIT_HOME", str(tmp_path / "missing"))
    with pytest.raises(AscendEnvironmentError, match="does not exist"):
        env.check_cann()
    clean_env.setenv("ASCEND_TOOLKIT_HOME", str(tmp_path))
    assert env.check_cann() == str(tmp_path)


def test_torch_npu_import_failure(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_npu", None)  # makes `import torch_npu` raise ImportError
    with pytest.raises(AscendEnvironmentError, match="torch_npu could not be imported"):
        env.import_torch_npu()


def test_device_checks(monkeypatch):
    fake_npu_runtime(monkeypatch, count=4)
    assert env.check_devices(4) == {"npu_count": 4, "npu_name": "Ascend910B1"}
    with pytest.raises(AscendEnvironmentError, match="only 4 NPU"):
        env.check_devices(8)
    fake_npu_runtime(monkeypatch, available=False)
    with pytest.raises(AscendEnvironmentError, match="no Ascend NPU"):
        env.check_devices(1)


def test_stack_mismatches():
    installed = dict(env.VALIDATED_STACK, torch="2.10.0+cpu")
    assert env.stack_mismatches(installed) == {}
    installed.update(transformers="4.57.3", **{"triton-ascend": None})
    assert env.stack_mismatches(installed) == {
        "transformers": {"expected": "5.5.4", "installed": "4.57.3"},
        "triton-ascend": {"expected": "3.2.2", "installed": None},
    }


def _dist(site, name, version, files):
    info = site / f"{name.replace('-', '_')}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    rows = []
    for rel, content in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
        rows.append(f"{rel},sha256={digest},{len(content)}")
    (info / "RECORD").write_text("\n".join(rows) + "\n")


def test_triton_ascend_integrity(tmp_path):
    site = tmp_path / "site"
    (site / "triton" / "backends").mkdir(parents=True)
    ascend_impl, community_impl = b"# ascend backend\n", b"# cuda backend\n"
    _dist(site, "triton", "3.2.0", {"triton/backends/driver.py": community_impl})
    _dist(site, "triton-ascend", "3.2.2", {"triton/backends/driver.py": ascend_impl, "triton/npu.py": b"x"})
    (site / "triton" / "npu.py").write_bytes(b"x")

    (site / "triton" / "backends" / "driver.py").write_bytes(ascend_impl)
    assert env.triton_ascend_conflicts(path=[str(site)]) == []

    (site / "triton" / "backends" / "driver.py").write_bytes(community_impl)  # community Triton unpacked last
    assert env.triton_ascend_conflicts(path=[str(site)]) == ["triton/backends/driver.py"]
    assert env.triton_ascend_conflicts(path=[str(tmp_path / "empty")]) == []


def test_ascend_versions(tmp_path, clean_env):
    home = tmp_path / "cann-9.1.0"
    (home / "aarch64-linux").mkdir(parents=True)
    (home / "aarch64-linux" / "ascend_toolkit_install.info").write_text("package_name=Ascend-cann-toolkit\nversion=9.1.0\n")
    driver = tmp_path / "version.info"
    driver.write_text("Version=26.0.rc1\nascendhal_version=7.35.23\n")
    info = env.ascend_versions(toolkit_home=str(home), driver_info=driver)
    assert info["cann_version"] == "9.1.0" and info["driver_version"] == "26.0.rc1"
    assert env.ascend_versions(toolkit_home=None, driver_info=tmp_path / "none")["driver_version"] is None


def test_configure_environment(clean_env, tmp_path, monkeypatch):
    clean_env.setenv("ASCEND_TOOLKIT_HOME", str(tmp_path))
    clean_env.setenv("ATB_HOME_PATH", str(tmp_path))
    fake_npu_runtime(monkeypatch, count=8)
    monkeypatch.setattr(env, "check_vllm_ascend", lambda: None)
    monkeypatch.setattr(env, "triton_ascend_conflicts", lambda path=None: [])
    monkeypatch.setattr(env, "package_versions", lambda names: dict(env.VALIDATED_STACK))

    runtime = env.configure_environment({"TASK_QUEUE_ENABLE": "1"}, visible_devices="0,1,2,3,4,5,6,7", n_npus=8)
    assert runtime["npu_count"] == 8 and runtime["stack_mismatches"] == {}
    assert os.environ["TASK_QUEUE_ENABLE"] == "1"

    # re-analysis does not require free NPUs
    fake_npu_runtime(monkeypatch, available=False)
    assert "npu_count" not in env.configure_environment({}, require_devices=False)

    monkeypatch.setattr(env, "triton_ascend_conflicts", lambda path=None: ["triton/backends/driver.py"])
    with pytest.raises(AscendEnvironmentError, match="uv sync --reinstall-package triton-ascend"):
        env.configure_environment({}, require_devices=False)


def test_diagnose_reports_without_raising(clean_env):
    results = dict((name, (ok, detail)) for name, ok, detail in env.diagnose(n_npus=2))
    assert results["CANN environment"][0] is False
    assert "vllm-ascend==0.23.0" in results and "Triton Ascend integrity" in results


def test_parse_npu_profiler_data(tmp_path, monkeypatch):
    from conftest import fake_npu_analyse, install_fake_npu_profiler, write_npu_profiler_data
    from moe_reliability.pipelines.common import parse_npu_profiler_data

    calls = []

    def analyse(profiler_path, **kwargs):
        calls.append(profiler_path)
        fake_npu_analyse(profiler_path)

    install_fake_npu_profiler(monkeypatch, analyse=analyse)
    for rank in (0, 1):
        write_npu_profiler_data(tmp_path / "traces", rank)
    views = parse_npu_profiler_data(tmp_path / "traces")
    assert [v.relative_to(tmp_path).as_posix() for v in views] == [
        "traces/worker_0/ASCEND_PROFILER_OUTPUT/trace_view.json",
        "traces/worker_1/ASCEND_PROFILER_OUTPUT/trace_view.json"]
    assert parse_npu_profiler_data(tmp_path / "traces") == views and len(calls) == 1  # parsed once

    (tmp_path / "empty").mkdir()
    with pytest.raises(RuntimeError, match="no profiler data"):
        parse_npu_profiler_data(tmp_path / "empty")

    install_fake_npu_profiler(monkeypatch, analyse=lambda profiler_path, **kw: None)
    write_npu_profiler_data(tmp_path / "other", 0)
    with pytest.raises(RuntimeError, match="produced no ASCEND_PROFILER_OUTPUT/trace_view.json"):
        parse_npu_profiler_data(tmp_path / "other")


def test_parse_npu_profiler_data_per_worker_fallback(tmp_path, monkeypatch):
    from conftest import fake_npu_analyse, install_fake_npu_profiler, write_npu_profiler_data
    from moe_reliability.pipelines.common import parse_npu_profiler_data

    calls = []

    def worker_only(profiler_path, **kwargs):  # parser that only accepts a single worker directory
        calls.append(profiler_path)
        if (tmp_path / "traces" / "worker_0").as_posix() in profiler_path or profiler_path.endswith("worker_1"):
            fake_npu_analyse(profiler_path)

    install_fake_npu_profiler(monkeypatch, analyse=worker_only)
    for rank in (0, 1):
        write_npu_profiler_data(tmp_path / "traces", rank)
    assert len(parse_npu_profiler_data(tmp_path / "traces")) == 2
    assert [p.rsplit("/", 1)[-1] for p in calls] == ["traces", "worker_0", "worker_1"]
