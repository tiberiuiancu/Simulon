# Learnings — Validation Experiment Code Quality Review

## Patterns That Worked Well

1. **Relative path resolution via `SCRIPT_DIR`** — All Python and shell scripts use `$(dirname "${BASH_SOURCE[0]}")` or `Path(__file__).resolve().parent` to anchor themselves. This makes the experiment directory relocatable.

2. **Graceful degradation for missing profile** — `sim_training.py` checks `PROFILE_PATH.exists()` and falls back to `templates/gpu/h100.yaml` with a warning. The `ignore_missing` flag in `backend.simulate()` provides a second safety net.

3. **Type hints** — All new Python files use `from __future__ import annotations` and include return-type annotations (`-> None`, `-> dict[str, object]`, etc.).

4. **`set -euo pipefail` in shell scripts** — Both `install_megatron.sh` and `profile_h100.sh` use strict mode, catching unset variables and pipeline failures early.

5. **Module-loading guard in `install_megatron.sh`** — The inline Python block verifies MoE support exists before declaring success, preventing silent misconfiguration.

6. **Subprocess safety** — `profile_h100.py` uses `subprocess.run(cmd, check=True, cwd=SCRIPT_DIR)`, ensuring errors propagate and the working directory is explicit.

## Conventions Observed

- Experiment configs are named `{model}_{mode}.yaml` (synthetic vs real).
- Results are consistently written to `./results/`.
- SLURM scripts live alongside their Python counterparts, not in a separate `jobs/` directory.
- `pyright` suppression comment (`# pyright: reportUnknownVariableType=false`) used pragmatically in `sim_training.py` where YAML parsing produces broad types.

## Things to Remember for Future Reviews

- `bash -n` only checks syntax, not semantics (e.g., it won't catch missing referenced files).
- `python -m py_compile` catches syntax errors but not import errors or runtime logic issues.
- YAML validation with the project's Pydantic models is the strongest correctness check — always run it when a venv is available.
- Git diff scope checks are essential to ensure experiments don't leak changes into core source code.

## Cleanup Follow-up

- When aligning experiment scripts with repo conventions, prefer direct CLI usage (`simulon profile gpu`) and keep wrappers thin.
- For Megatron runs, installing runtime deps inline and exporting `PYTHONPATH` to the submodule is enough; extra bootstrap scripts are unnecessary.
