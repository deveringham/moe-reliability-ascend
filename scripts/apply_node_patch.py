"""Install (or check) the EPLB gate patch in the node's vllm-ascend.

    ./nrun 'uv run python scripts/apply_node_patch.py'            # install
    ./nrun 'uv run python scripts/apply_node_patch.py --check'    # report only
    ./nrun 'uv run python scripts/apply_node_patch.py --emit-patch patches/x.patch'

The patch itself lives in ``moe_reliability.eplb_gate``; see that module for why
it is a source patch rather than a plugin. It is inert until a server runs with
``MOE_EPLB_HONOR_CHANGED=1``, which ``server.eplb_honor_changed`` sets, so
installing it does not change what any existing config measures.

This has to be re-run after ``uv sync``, which reinstalls the wheel and reverts
it. ``--check`` is what the run itself uses to fail fast rather than silently
measuring an unpatched server, and it exits non-zero when the patch is missing.
"""

from __future__ import annotations

import argparse
import difflib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from moe_reliability import eplb_gate  # noqa: E402


def target_path() -> str:
    """The installed file the patch applies to."""
    try:
        import vllm_ascend
    except ImportError as e:  # pragma: no cover - the node always has it
        raise SystemExit(f"vllm_ascend is not importable here ({e}); this script runs on the node") from e
    return os.path.join(os.path.dirname(vllm_ascend.__file__), eplb_gate.TARGET)


def clear_pycache(path: str) -> list[str]:
    """Remove the compiled form, so a patched source cannot be shadowed."""
    cache = os.path.join(os.path.dirname(path), "__pycache__")
    stem = os.path.basename(path).removesuffix(".py")
    removed = []
    if os.path.isdir(cache):
        for name in os.listdir(cache):
            if name.startswith(stem + ".") and name.endswith(".pyc"):
                os.remove(os.path.join(cache, name))
                removed.append(name)
    return removed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report whether the patch is installed, change nothing")
    ap.add_argument("--revert", action="store_true", help="restore the file from the installed wheel's source")
    ap.add_argument("--emit-patch", metavar="PATH", help="write a unified diff of the change and exit")
    args = ap.parse_args()

    path = target_path()
    with open(path) as f:
        original = f.read()
    patched = eplb_gate.is_patched(original)

    if args.emit_patch:
        if patched:
            raise SystemExit(f"{path} is already patched; revert it before emitting a diff")
        new = eplb_gate.apply_to(original)
        rel = f"vllm_ascend/{eplb_gate.TARGET}"
        diff = difflib.unified_diff(original.splitlines(True), new.splitlines(True),
                                    fromfile=f"a/{rel}", tofile=f"b/{rel}")
        with open(args.emit_patch, "w") as f:
            f.writelines(diff)
        print(f"wrote {args.emit_patch}")
        return 0

    if args.check:
        print(f"eplb gate patch: {'installed' if patched else 'MISSING'} in {path}")
        return 0 if patched else 1

    if args.revert:
        if not patched:
            print(f"eplb gate patch: not installed, nothing to revert ({path})")
            return 0
        raise SystemExit("no backup is kept; revert with `uv sync --reinstall-package vllm-ascend`")

    if patched:
        print(f"eplb gate patch: already installed in {path}")
        return 0
    new = eplb_gate.apply_to(original)
    with open(path, "w") as f:
        f.write(new)
    removed = clear_pycache(path)
    print(f"eplb gate patch: installed in {path}"
          + (f" (cleared {len(removed)} stale .pyc)" if removed else ""))
    # Read it back: a patch that did not land is the failure this whole script
    # exists to make loud.
    with open(path) as f:
        if not eplb_gate.is_patched(f.read()):
            raise SystemExit("wrote the patch but the marker is absent on re-read")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
