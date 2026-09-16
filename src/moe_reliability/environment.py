###
# environment.py
#
# Ascend NPU runtime environment setup.
#
# Experiments run exclusively on Ascend NPUs through torch_npu and the vLLM
# Ascend plugin. Before any model code runs, configure_environment
# applies the configured environment variables and NPU visibility,
# checks that the CANN environment has been activated (set_env.sh),
# imports torch_npu and enables torch_npu.contrib.transfer_to_npu, which
# maps device-generic PyTorch calls in the research code onto NPUs,
# checks that enough NPUs are visible, that the vLLM Ascend plugin is installed,
# and that Triton Ascend has not been overwritten by community Triton,
# and reports deviations from the validated software stack.
# Dylan Everingham
# 16.09.2026
###

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import importlib.util
import os
import platform as _platform
import re
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from .logs import log

__all__ = [
    "AscendEnvironmentError",
    "VALIDATED_STACK",
    "VALIDATED_CANN",
    "apply_environment",
    "check_cann",
    "check_devices",
    "configure_environment",
    "collect_provenance",
    "diagnose",
    "package_versions",
    "stack_mismatches",
    "triton_ascend_conflicts",
    "ascend_versions",
]

#: Python packages of the vLLM Ascend 0.23.0 validated compatibility set.
VALIDATED_STACK: dict[str, str] = {
    "vllm": "0.23.0",
    "vllm-ascend": "0.23.0",
    "torch": "2.10.0",
    "torch-npu": "2.10.0.post4",
    "triton-ascend": "3.2.2",
    "transformers": "5.5.4",
}
#: CANN Toolkit / NNAL version of the validated set.
VALIDATED_CANN = "9.1.0"

CANN_ACTIVATE_HINT = ("source /usr/local/Ascend/ascend-toolkit/set_env.sh && "
                      "source /usr/local/Ascend/nnal/atb/set_env.sh")
TRITON_REPAIR_HINT = "uv sync --reinstall-package triton-ascend"

_PROVENANCE_PACKAGES = (
    "moe-reliability", "moe-reliability-results", "vllm", "vllm-ascend", "torch", "torch-npu", "torchvision", "torchaudio",
    "triton-ascend", "triton", "transformers", "accelerate", "datasets", "openai", "HolisticTraceAnalysis",
)


class AscendEnvironmentError(RuntimeError):
    """The process cannot run experiments on Ascend NPUs."""


def apply_environment(env: Mapping[str, str], visible_devices: str = "") -> None:
    """Set environment variables inherited by the vLLM server and put this interpreter's bin on PATH."""
    for key, value in env.items():
        os.environ[key] = str(value)
    if visible_devices.strip():
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(d.strip() for d in visible_devices.split(","))
    venv_bin = os.path.dirname(sys.executable)
    if venv_bin not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")


def cann_toolkit_home() -> str | None:
    return os.environ.get("ASCEND_TOOLKIT_HOME") or os.environ.get("ASCEND_HOME_PATH")


def check_cann() -> str:
    """Return the CANN toolkit directory of the activated environment."""
    home = cann_toolkit_home()
    if not home:
        raise AscendEnvironmentError(
            "the CANN environment is not activated (ASCEND_TOOLKIT_HOME is not set). Run:\n"
            f"  {CANN_ACTIVATE_HINT}")
    if not Path(home).is_dir():
        raise AscendEnvironmentError(f"ASCEND_TOOLKIT_HOME={home} does not exist; activate a valid CANN "
                                     f"installation:\n  {CANN_ACTIVATE_HINT}")
    if not os.environ.get("ATB_HOME_PATH"):
        log("warning: NNAL/ATB environment not activated (ATB_HOME_PATH unset); vLLM Ascend needs libatb.so: "
            "source /usr/local/Ascend/nnal/atb/set_env.sh")
    return home


def import_torch_npu() -> Any:
    """Import torch_npu and redirect device-generic PyTorch calls to NPUs."""
    try:
        import torch_npu  # noqa: F401
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
    except Exception as exc:  # ImportError or OSError from missing CANN libraries
        raise AscendEnvironmentError(
            f"torch_npu could not be imported ({type(exc).__name__}: {exc}). Install the project environment "
            f"with `uv sync` and activate CANN:\n  {CANN_ACTIVATE_HINT}") from exc
    return torch_npu


def check_devices(n_npus: int) -> dict[str, Any]:
    """Require at least ``n_npus`` visible NPUs."""
    import torch

    if not torch.npu.is_available():
        raise AscendEnvironmentError("no Ascend NPU is available to this process (check `npu-smi info`, the "
                                     "driver installation and ASCEND_RT_VISIBLE_DEVICES)")
    count = int(torch.npu.device_count())
    if count < n_npus:
        raise AscendEnvironmentError(f"hardware.n_npus = {n_npus} but only {count} NPU(s) are visible "
                                     f"(ASCEND_RT_VISIBLE_DEVICES={os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '')!r})")
    return {"npu_count": count, "npu_name": str(torch.npu.get_device_name(0))}


def check_vllm_ascend() -> None:
    if importlib.util.find_spec("vllm_ascend") is None:
        raise AscendEnvironmentError("the vLLM Ascend plugin (vllm_ascend) is not installed; run `uv sync`")


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _public(version: str) -> str:
    return version.split("+", 1)[0]  # drop local labels such as "+cpu"


def stack_mismatches(installed: Mapping[str, str | None],
                     expected: Mapping[str, str] = VALIDATED_STACK) -> dict[str, dict[str, str | None]]:
    """Packages whose installed version differs from the validated set."""
    out = {}
    for name, want in expected.items():
        have = installed.get(name)
        if have is None or _public(have) != want:
            out[name] = {"expected": want, "installed": have}
    return out


def _record_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode()


def triton_ascend_conflicts(path: list[str] | None = None) -> list[str]:
    """Files installed by triton-ascend that another distribution has overwritten.

    Triton Ascend installs its implementation over files shared with community
    Triton. When both are unpacked in parallel the community files can win,
    which silently disables the Ascend implementation.
    """
    dists = {}
    for dist in importlib.metadata.distributions(path=path) if path is not None else importlib.metadata.distributions():
        name = (dist.metadata["Name"] or "").lower().replace("_", "-")
        dists.setdefault(name, dist)
    ascend = dists.get("triton-ascend")
    if ascend is None or not ascend.files:
        return []
    shared: set[str] = set()
    for name, dist in dists.items():
        if name != "triton-ascend" and dist.files:
            shared.update(str(f) for f in dist.files)
    conflicts = []
    for f in ascend.files:
        if f.hash is None or str(f) not in shared:
            continue
        located = Path(f.locate())
        if not located.is_file() or _record_hash(located, f.hash.mode) != f.hash.value:
            conflicts.append(str(f))
    return conflicts


# --------------------------------------------------------------------------- #
#  Activation
# --------------------------------------------------------------------------- #
def configure_environment(env: Mapping[str, str], visible_devices: str = "", n_npus: int | None = None,
                          require_devices: bool = True) -> dict[str, Any]:
    """Prepare this process for Ascend experiments. Must run before any model code.

    Raises :class:`AscendEnvironmentError` with remediation steps when the
    environment is unusable. Returns runtime information for provenance.
    """
    apply_environment(env, visible_devices)
    runtime: dict[str, Any] = {"cann_toolkit_home": check_cann()}
    import_torch_npu()
    if require_devices:
        runtime.update(check_devices(n_npus or 1))
    check_vllm_ascend()
    conflicts = triton_ascend_conflicts()
    if conflicts:
        raise AscendEnvironmentError(
            f"{len(conflicts)} Triton Ascend file(s) were overwritten by another package (e.g. {conflicts[0]}); "
            f"repair with:\n  {TRITON_REPAIR_HINT}")
    mismatches = stack_mismatches(package_versions(VALIDATED_STACK))
    if mismatches:
        details = ", ".join(f"{k} {v['installed']} (validated {v['expected']})" for k, v in mismatches.items())
        log(f"warning: software stack differs from the validated vLLM Ascend set: {details}")
    runtime["stack_mismatches"] = mismatches
    return runtime


def _run(cmd: list[str], timeout: float = 20.0) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _read_version(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = re.search(r"^\s*(?:version|Version|ascend_toolkit_version)\s*=\s*([^\s#]+)", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def ascend_versions(toolkit_home: str | None = None,
                    driver_info: str | os.PathLike = "/usr/local/Ascend/driver/version.info") -> dict[str, Any]:
    """CANN toolkit and Ascend driver versions from their install records."""
    info: dict[str, Any] = {"cann_toolkit_home": toolkit_home or cann_toolkit_home(),
                            "atb_home": os.environ.get("ATB_HOME_PATH")}
    home = info["cann_toolkit_home"]
    if home and Path(home).is_dir():
        root = Path(home)
        candidates = [root / "version.cfg", *sorted(root.glob("*/ascend_toolkit_install.info")),
                      *sorted(root.glob("ascend_toolkit_install.info")), *sorted(root.glob("*/ascend_ops_install.info"))]
        for candidate in candidates:
            version = _read_version(candidate)
            if version:
                info["cann_version"] = version
                break
    info["driver_version"] = _read_version(Path(driver_info))
    return info


def collect_provenance(runtime: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Host, software and NPU information stored with every run."""
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "os": _platform.platform(),
        "machine": _platform.machine(),
        "python": sys.version.split()[0],
        "command": " ".join(sys.argv),
        "packages": {k: v for k, v in package_versions(_PROVENANCE_PACKAGES).items() if v is not None},
        "ascend": {**ascend_versions(), "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES")},
    }
    if runtime:
        info["ascend"].update({k: v for k, v in runtime.items() if k != "cann_toolkit_home"})
    npu_smi = _run(["npu-smi", "info"])
    if npu_smi:
        info["ascend"]["npu_smi"] = npu_smi.splitlines()[:80]
    commit = _run(["git", "rev-parse", "HEAD"])
    if commit:
        info["git_commit"] = commit
        info["git_dirty"] = bool(_run(["git", "status", "--porcelain", "--untracked-files=no"]))
    return info


def diagnose(n_npus: int | None = None, visible_devices: str = "") -> list[tuple[str, bool, str]]:
    """Run every environment check without raising: ``[(check, ok, detail)]``."""
    results: list[tuple[str, bool, str]] = []

    def record(name: str, fn) -> Any:
        try:
            value = fn()
        except Exception as exc:  # noqa: BLE001
            results.append((name, False, str(exc)))
            return None
        results.append((name, True, "" if value is None else str(value)))
        return value

    apply_environment({}, visible_devices)
    record("CANN environment", check_cann)
    versions = ascend_versions()
    results.append(("CANN version", versions.get("cann_version") == VALIDATED_CANN,
                    f"{versions.get('cann_version')} (validated {VALIDATED_CANN})"))
    results.append(("Ascend driver", versions.get("driver_version") is not None, str(versions.get("driver_version"))))
    npu_smi = _run(["npu-smi", "info"])
    results.append(("npu-smi", npu_smi is not None, "available" if npu_smi else "npu-smi info failed or not found"))
    record("torch_npu import", lambda: (import_torch_npu(), None)[1])
    if results[-1][1]:
        record("NPUs visible", lambda: check_devices(n_npus or 1))
    record("vLLM Ascend plugin", check_vllm_ascend)
    installed = package_versions(VALIDATED_STACK)
    for name, want in VALIDATED_STACK.items():
        have = installed.get(name)
        results.append((f"{name}=={want}", have is not None and _public(have) == want, str(have)))
    try:
        conflicts = triton_ascend_conflicts()
        results.append(("Triton Ascend integrity", not conflicts,
                        "ok" if not conflicts else f"{len(conflicts)} overwritten file(s); repair with: "
                                                   f"{TRITON_REPAIR_HINT}"))
    except Exception as exc:  # noqa: BLE001
        results.append(("Triton Ascend integrity", False, str(exc)))
    return results
