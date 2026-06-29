#!/usr/bin/env python3
"""Stage all workload-tuning traces and scenario files in git.

Usage (from repo root):
    uv run python experiments/usecase_workload_tuning/add_traces_to_git.py
    uv run python experiments/usecase_workload_tuning/add_traces_to_git.py --dry-run
"""

from __future__ import annotations

import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from grid_search import _default_trace_dir, _generate_configs


def _git_add(path: Path, dry_run: bool) -> None:
    if not path.exists():
        return
    cmd = ["git", "add", str(path)]
    if dry_run:
        print(f"Would run: {' '.join(cmd)}")
        return
    subprocess.run(cmd, check=True)
    print(f"Staged {path}")


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print what would be staged without running git add."
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent

    for path in _generate_configs():
        trace_dir = _default_trace_dir(path)
        _git_add(trace_dir, args.dry_run)
        _git_add(path, args.dry_run)

    for extra in (
        base_dir / "results.csv",
        base_dir / "README.md",
        base_dir / "scenarios" / "base_workload.yaml",
    ):
        if extra.exists():
            _git_add(extra, args.dry_run)

    if not args.dry_run:
        print("Done. Run `git status` to review.")


if __name__ == "__main__":
    main()
