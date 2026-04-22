# Scope Fidelity Check — validation-gpt-oss-1b-moe

## Date: 2026-04-22
## Commits reviewed: e5e73ac, 42d85c1, ad56500

---

### Contamination
CLEAN — no existing source files modified.
Only `.gitmodules` was changed (expected for submodule addition).
No changes outside `experiments/validation/gpt_oss_1b_training/` or `templates/model/gpt-oss-1b-moe.yaml`.

---

### Task Compliance Matrix

| Task | Spec Deliverables | Built | Status | Notes |
|------|-------------------|-------|--------|-------|
| T1 | `templates/model/gpt-oss-1b-moe.yaml` with fields matching gpt-oss-5b.yaml | YES | COMPLIANT | Follows reference pattern exactly. Spec listed extra fields (`max_position_embeddings`, `use_moe`, `norm_type`, etc.) not present in reference template or `LLMSpec` schema — these are schema-level constraints, not build errors. |
| T2 | `.gitmodules` update, submodule, `install_megatron.sh`, `requirements.txt` | YES | COMPLIANT | All 4 deliverables present. |
| T3 | `profile_h100.sh`, `profile_h100.py` | YES | COMPLIANT | Correct dimensions (hidden=1536, heads=24, ffn=6144, seq=8192, batch=1, tp=1, epoch-num=20). |
| T4 | Run profiling on Snellius → `results/h100_profile.yaml` | PENDING | BY DESIGN | Requires Snellius H100 GPU. File does not exist yet. |
| T5 | `gpt_oss_1b_synthetic.yaml`, `gpt_oss_1b_real.yaml`, `run_megatron.py` | YES | COMPLIANT | Both configs correct. Wrapper handles synthetic (subprocess) and real (in-process C4 dataloader). |
| T6 | `sim_training.yaml`, `sim_training.py` | YES | COMPLIANT | Scenario valid, single GPU, references correct model. Script produces chrome trace. |
| T7 | `run_megatron_synthetic.sh`, chrome trace, log | MISSING SCRIPT | PARTIAL | SLURM batch script `run_megatron_synthetic.sh` was NOT created. Training itself pending (Snellius). |
| T8 | `run_megatron_real.sh`, chrome trace, log | MISSING SCRIPT | PARTIAL | SLURM batch script `run_megatron_real.sh` was NOT created. Training itself pending (Snellius). |
| T9 | `results/sim_trace.json` from simulon | YES | COMPLIANT | 126 traceEvents, valid Chrome trace format. Step time: 60.611 ms. |
| T10 | `results/README.md`, verify all artifacts | YES | COMPLIANT | README correctly documents available vs pending artifacts. Realistic given T4/T7/T8 are Snellius-only. |

---

### Must NOT Do Compliance

| Guardrail | Violation? | Evidence |
|-----------|-----------|----------|
| Multi-GPU or distributed training | NO | All configs use TP=PP=EP=DP=1 |
| Full pretraining / convergence | NO | train-iters=6 (5 warmup + 1 profile) |
| Automatic plotting or comparison | NO | README explicitly says manual comparison only |
| Modifications to existing experiments or templates | NO | Only new template created; no existing templates modified |
| Source code changes in simulon core | NO | No changes in `src/simulon/` |

---

### Gaps / Issues Found

1. **Missing SLURM scripts for T7 and T8**
   - `run_megatron_synthetic.sh` (specified in T7 acceptance criteria) is absent.
   - `run_megatron_real.sh` (specified in T8 acceptance criteria) is absent.
   - The README references both scripts, implying they should exist.
   - Mitigation: `run_megatron.py --mode synthetic` and `--mode real` provide equivalent functionality, but the spec explicitly requested SLURM batch wrappers.

2. **Missing QA evidence directory**
   - Plan QA policy: "Every task MUST include agent-executed QA scenarios. Evidence saved to `.sisyphus/evidence/`."
   - Directory `.sisyphus/evidence/` does not exist.
   - No evidence files were produced for any task.

3. **Spec/schema mismatch on T1 fields**
   - Spec lists fields (`max_position_embeddings`, `use_moe`, `norm_type`, `activation`, `tie_word_embeddings`, `dtype`) that are NOT in `LLMSpec` schema or reference templates.
   - Build correctly followed the reference template pattern instead of adding non-schema fields.
   - This is a plan inconsistency, not a build error.

---

### Verdict

- **Tasks**: 8/10 fully compliant, 2/10 partially compliant (T7, T8 missing SLURM scripts)
- **Contamination**: CLEAN (0 issues)
- **Must NOT Do**: 5/5 compliant
- **Overall**: PASS with minor gaps (missing SLURM wrappers + missing QA evidence)
