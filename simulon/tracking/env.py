from __future__ import annotations

import os
from pathlib import Path

ENV_FILE_NAME = ".tracking.env"


def load_tracking_env_file(path: Path) -> None:
    """Set variables from a .tracking.env file into os.environ.

    Each non-empty, non-comment line must contain exactly one ``=``:
        KEY=value
    Existing environment variables are **not** overwritten (standard dotenv
    behaviour).  Later files in the cascade also cannot override values
    that were set by earlier files or by the shell.
    """
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and (key not in os.environ or os.environ.get(key, "").strip() == ""):
                os.environ[key] = val


def load_cascading_tracking_env(scenario_path: str | Path) -> None:
    """Load ``.tracking.env`` files from CWD down to the scenario directory.

    For ``./experiments/validate_e2e/deepseekv3/scenario.yaml`` we load, in order:
        ./.tracking.env
        ./experiments/.tracking.env
        ./experiments/validate_e2e/.tracking.env
        ./experiments/validate_e2e/deepseekv3/.tracking.env
    """
    scenario = Path(scenario_path).resolve()
    cwd = Path.cwd().resolve()

    try:
        rel = scenario.parent.relative_to(cwd)
    except ValueError:
        dirs = [scenario.parent]
    else:
        dirs = [cwd]
        cur = cwd
        for part in rel.parts:
            cur = cur / part
            dirs.append(cur)

    for d in dirs:
        env_file = d / ENV_FILE_NAME
        if env_file.is_file():
            load_tracking_env_file(env_file)
