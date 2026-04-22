# Code Quality Review Findings — Validation Experiment (GPT-OSS-1B-MoE)

Date: 2026-04-22
Reviewer: Sisyphus-Junior
Scope: All new files in `experiments/validation/gpt_oss_1b_training/`

---

## Summary

All new files pass basic syntax checks. No hardcoded absolute paths found in experiment-owned files. No unintended modifications outside `experiments/validation/` or `templates/model/`. Several functional and documentation issues identified.

---

## Syntax / Compilation Results

| File | bash -n | python -m py_compile | YAML valid | Schema valid |
|---|---|---|---|---|
| `install_megatron.sh` | OK | N/A | N/A | N/A |
| `profile_h100.sh` | OK | N/A | N/A | N/A |
| `profile_h100.py` | N/A | OK | N/A | N/A |
| `run_megatron.py` | N/A | OK | N/A | N/A |
| `sim_training.py` | N/A | OK | N/A | N/A |
| `gpt_oss_1b_synthetic.yaml` | N/A | N/A | OK | N/A |
| `gpt_oss_1b_real.yaml` | N/A | N/A | OK | N/A |
| `sim_training.yaml` | N/A | N/A | OK | OK |

---

## Issues Found

### 1. MISSING SBATCH SCRIPTS (High)
**Location:** `results/README.md` lines 33, 37
**Problem:** README references `run_megatron_synthetic.sh` and `run_megatron_real.sh`, but these files do **not exist** in the experiment directory. Only `profile_h100.sh` exists.
**Impact:** Users following the README will get "file not found" errors.
**Recommendation:** Create the missing SLURM wrapper scripts, or update README to remove references and document the direct `python run_megatron.py --mode synthetic/real` approach instead.

### 2. UNPINNED DEPENDENCIES (Medium)
**Location:** `requirements.txt`
**Problem:** All packages are unpinned (e.g., `torch`, `datasets`, `transformers`). On Snellius, this could install versions incompatible with CUDA 12.8 or with each other.
**Impact:** Non-deterministic environment; potential runtime failures during validation.
**Recommendation:** Pin to versions known to work on Snellius (e.g., `torch==2.5.1`, `transformers==4.46.0`).

### 3. MEGATRON INTERNAL IMPORTS (Medium)
**Location:** `run_megatron.py` lines 82–84
**Problem:** `_run_real()` imports `gpt_builders`, `model_provider`, and `pretrain_gpt` without any import-error handling. These are Megatron-LM internal modules that may not exist or may be renamed in different submodule commits.
**Impact:** `run_megatron.py --mode real` will crash with `ModuleNotFoundError` if the submodule layout changes.
**Recommendation:** Add a guard clause that checks for these modules and prints a helpful error message if they are missing.

### 4. `.venv` ASSUMPTION (Low)
**Location:** `profile_h100.sh` line 16
**Problem:** Hardcodes `source "$REPO_ROOT/.venv/bin/activate"`. If the venv is missing or located elsewhere, the job will fail.
**Impact:** Low — standard for this repo, but brittle.
**Recommendation:** Add a check: `if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then source ...; else echo "venv not found"; exit 1; fi`.

### 5. `run_megatron.py` — `sys.argv` MUTATION (Low)
**Location:** `run_megatron.py` lines 132–133, 162–163
**Problem:** `_run_real()` mutates `sys.argv` globally. While wrapped in `try/finally`, concurrent code (e.g., threaded tests) could be affected.
**Impact:** Low for the intended single-process usage.
**Recommendation:** Document the mutation in a comment, or use `argparse` on a copy of `sys.argv` instead.

### 6. `run_megatron.py` — `parse_and_validate_args` FALLBACK (Low)
**Location:** `run_megatron.py` line 143
**Problem:** `pg.add_modelopt_args` is used conditionally via `getattr(pg, "has_nvidia_modelopt", False)`. If `pg` lacks `has_nvidia_modelopt` but *does* have `add_modelopt_args`, the extra args provider will be `None`.
**Impact:** Low — likely correct for most submodule states.
**Recommendation:** Use `hasattr(pg, "add_modelopt_args")` directly for clarity.

---

## Hardcoded Paths Check

- **Experiment files:** No hardcoded absolute paths (e.g., `/home/...`, `/opt/...`, `/tmp/...`) found in `.sh`, `.py`, `.yaml`, `.md` files under `experiments/validation/gpt_oss_1b_training/`.
- **Submodule (`megatron-lm`):** Contains expected hardcoded paths in examples/docs/tests — acceptable as third-party code.

---

## File Modification Scope Verification

Files changed in the last 3 commits (from `git diff --name-only HEAD~3 HEAD`):
- `.gitmodules`
- `templates/model/gpt-oss-1b-moe.yaml`
- `experiments/validation/gpt_oss_1b_training/*` (all experiment files)
- `experiments/validation/gpt_oss_1b_training/megatron-lm` (submodule)

**Conclusion:** No source code modifications outside `experiments/validation/` or `templates/model/`. ✅

---

## Anti-Patterns Observed

| Anti-Pattern | Location | Severity | Note |
|---|---|---|---|
| Missing referenced files | `results/README.md` | High | Scripts mentioned but absent |
| Unpinned dependencies | `requirements.txt` | Medium | Reproducibility risk |
| Bare except-not-really | `sim_training.py` lines 85–89 | Low | Specific `RuntimeError` catch is fine |
| Global `sys.argv` mutation | `run_megatron.py` | Low | Contained in try/finally |

---

## Recommendations (Priority Order)

1. **Create or remove references** to `run_megatron_synthetic.sh` / `run_megatron_real.sh` in README.
## Fix Applied (2026-04-22)

- **Missing SLURM scripts (Issue #1)**: FIXED. Added `run_megatron_synthetic.sh` and `run_megatron_real.sh` following the `profile_h100.sh` pattern. Both are executable and pass `bash -n`.

## Remaining Recommendations (Optional)

1. **Pin `requirements.txt`** to known-good versions.
2. **Add import guards** for Megatron internal modules in `run_megatron.py`.
3. **Add `.venv` existence check** in `profile_h100.sh`.
4. **Clean up `has_nvidia_modelopt` conditional** in `run_megatron.py`.

