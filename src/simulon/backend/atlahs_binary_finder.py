"""ATLAHS binary discovery layer.

Finds pre-built ATLAHS simulator binaries using the following resolution order:

1. ``SIMULON_ATLAHS_BIN_DIR`` environment variable
2. Package-relative path (``src/simulon/backend/atlahs_binaries/<platform>/``)
3. ``PATH`` search

Raises :exc:`RuntimeError` if any required binary is missing.
"""

from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path

_REQUIRED_BINARIES: dict[str, str] = {
    "LogGOPSim": "LogGOPSim",
    "txt2bin": "txt2bin",
    "htsim_uec": "htsim_uec",
    "schedgen": "schedgen",
}


def _get_platform_tag() -> str:
    """Return a ``<system>_<machine>`` tag for the current platform.

    Supported tags:
        * ``darwin_arm64``
        * ``darwin_x86_64``
        * ``linux_x86_64``

    Raises:
        RuntimeError: If the platform is not macOS or Linux.
    """
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
        + "ATLAHS binaries are only available for macOS (arm64, x86_64) and Linux (x86_64)."
    )


def _find_binary(name: str, search_dirs: list[Path]) -> Path:
    """Locate *name* in *search_dirs* or on ``PATH``.

    Returns the absolute path if found, otherwise raises :exc:`RuntimeError`.
    """
    for directory in search_dirs:
        candidate = directory / name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate.resolve()

    path_exec = shutil.which(name)
    if path_exec:
        return Path(path_exec).resolve()

    raise RuntimeError(
        f"ATLAHS binary '{name}' not found. "
        + f"Searched directories: {[str(d) for d in search_dirs]} and PATH."
    )


def find_binaries() -> dict[str, str]:
    """Discover ATLAHS simulator binaries.

    Resolution order per binary:
        1. ``SIMULON_ATLAHS_BIN_DIR`` (if set)
        2. Package-relative ``atlahs_binaries/<platform>/`` directory
        3. ``PATH``

    Returns:
        Mapping from canonical binary name to absolute filesystem path.

    Raises:
        RuntimeError: If the current platform is unsupported or any required
            binary cannot be located.
    """
    platform_tag = _get_platform_tag()

    search_dirs: list[Path] = []

    env_dir = os.environ.get("SIMULON_ATLAHS_BIN_DIR")
    if env_dir:
        search_dirs.append(Path(env_dir))

    package_dir = Path(__file__).resolve().parent
    bundled_dir = package_dir / "atlahs_binaries" / platform_tag
    search_dirs.append(bundled_dir)

    result: dict[str, str] = {}
    errors: list[str] = []

    for key, binary_name in _REQUIRED_BINARIES.items():
        try:
            path = _find_binary(binary_name, search_dirs)
            result[key] = str(path)
        except RuntimeError as exc:
            errors.append(str(exc))

    if errors:
        raise RuntimeError(
            f"Missing ATLAHS binaries on platform '{platform_tag}':\n" + "\n".join(errors)
        )

    return result
