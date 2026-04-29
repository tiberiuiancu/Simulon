#!/usr/bin/env python3
"""Build ATLAHS simulators locally and stage pre-built binaries.

Usage:
    python scripts/build_atlahs.py

This script:
    1. Detects the current platform (macOS/Linux, x86_64/arm64).
    2. Runs ``make`` in each ATLAHS simulator source directory.
    3. Copies the resulting binaries to
       ``src/simulon/backend/atlahs_binaries/<platform>/``.

The binaries are then picked up by the wheel build via
``pyproject.toml`` and discovered at runtime by
:func:`simulon.backend.atlahs_binary_finder.find_binaries`.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

_BUILD_TARGETS: list[tuple[str, str]] = [
    ("vendor/atlahs/sim/LogGOPSim", "LogGOPSim"),
    ("vendor/atlahs/sim/LogGOPSim", "txt2bin"),
    ("vendor/atlahs/sim/htsim-backend/sim/datacenter", "htsim_uec"),
    ("vendor/atlahs/goal_gen/hpc/Schedgen", "schedgen"),
]


def _get_platform_tag() -> str:
    """Return a ``<system>_<machine>`` tag for the current platform."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin":
        if machine in ("arm64", "aarch64"):
            return "darwin_arm64"
        if machine in ("x86_64", "amd64"):
            return "darwin_x86_64"
    elif system == "linux":
        if machine in ("x86_64", "amd64"):
            return "linux_x86_64"

    raise RuntimeError(
        f"Unsupported platform: {system=}, {machine=}. "
        + "Supported: darwin_arm64, darwin_x86_64, linux_x86_64."
    )


def _run_make(src_dir: Path, binary_name: str) -> Path:
    """Run ``make`` in *src_dir* and return the path to the built binary."""
    if not src_dir.exists():
        raise RuntimeError(f"Source directory does not exist: {src_dir}")

    makefile = src_dir / "Makefile"
    if not makefile.exists():
        raise RuntimeError(f"No Makefile found in {src_dir}")

    print(f"Building {binary_name} in {src_dir} ...")
    result = subprocess.run(
        ["make", "-C", str(src_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"make failed in {src_dir} (rc={result.returncode})")

    binary_path = src_dir / binary_name
    if not binary_path.exists():
        for candidate in src_dir.rglob(binary_name):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                binary_path = candidate
                break
        else:
            raise RuntimeError(
                f"Binary '{binary_name}' not found after make in {src_dir}"
            )

    return binary_path.resolve()


def main() -> int:
    """Build ATLAHS and stage binaries."""
    repo_root = Path(__file__).resolve().parent.parent
    platform_tag = _get_platform_tag()
    output_dir = repo_root / "src" / "simulon" / "backend" / "atlahs_binaries" / platform_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    unique_dirs: dict[Path, list[str]] = {}
    for rel_dir, binary_name in _BUILD_TARGETS:
        src = repo_root / rel_dir
        unique_dirs.setdefault(src, []).append(binary_name)

    for src_dir, binaries in unique_dirs.items():
        for binary_name in binaries:
            built = _run_make(src_dir, binary_name)
            dest = output_dir / binary_name
            _ = shutil.copy2(built, dest)
            dest.chmod(dest.stat().st_mode | 0o111)
            print(f"  -> {dest}")

    print(f"\nAll ATLAHS binaries staged in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
