# Validation: GPT-OSS-1B-MoE Training Compute Model

## TL;DR

> Validate simulon's compute model by running a real Megatron-LM training iteration on a small MoE model (GPT-OSS-1B-MoE, ~1B params, 6 layers, 8 experts) on a single Snellius H100 GPU, and compare the per-kernel chrome trace + end-to-end step time against simulon's prediction.
>
> **Deliverables**:
> - `templates/model/gpt-oss-1b-moe.yaml` — new model profile
> - `experiments/validation/gpt_oss_1b_training/` — complete experiment directory
> - Real chrome traces from Megatron-LM (synthetic + real data)
> - Simulon chrome trace prediction
> - H100 kernel profile YAML for the new model
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES — 4 waves
> **Critical Path**: T1 → T2 → T3 → T4/T5/T6 → T7/T8/T9 → T10

---

## Context

### Original Request
Create a validation experiment under `experiments/validation/` that runs a smaller GPT-OSS MoE model (~1B params) with Megatron-LM on Snellius H100, and create the equivalent scenario in simulon for compute model validation.

### Interview Summary
**Key Discussions**:
- **Validation target**: End-to-end step time + chrome trace (torch profiler per-kernel breakdown)
- **Data**: Both synthetic (padded fixed-length) and real variable-length data
- **Duration**: 5 warmup iterations + profile 1 iteration (no convergence needed)
- **Parallelism**: Single GPU (TP=PP=EP=DP=1) to isolate network effects
- **Simulon**: Use MegatronDAGTracer (workload) + AnalyticalBackend (network)
- **Comparison**: Manual (user will compare traces themselves, plotting out of scope)
- **Architecture**: User delegated to planner — 6 layers, hidden=1536, 8 experts, top-2 chosen for good H100 kernel utilization

**Research Findings**:
- `experiments/validation/simccl/` is the only existing validation experiment (pattern: sim_*.py + run_real.sh + plot.py + results/)
- No 1B model template exists — must create from scratch
- MegatronDAGTracer at `src/simulon/backend/dag/megatron_tracer.py` produces ExecutionDAG
- Chrome trace export exists in `src/simulon/backend/dag/chrome_trace.py`
- H100 profiling via `simulon profile gpu` CLI exists in `src/simulon/profiling/kernels.py`
- Snellius H100 node template at `templates/node/snellius-h100-4g.yaml`
- Megatron-LM not currently installed — will add as git submodule in experiment dir

### Metis Review
**Identified Gaps** (addressed):
- **Precision**: Default to bf16 (standard for H100 training)
- **Kernel profile reuse**: Simulon simulation MUST use the SAME kernel profile YAML collected during the real profiling run
- **MoE kernel coverage**: Verify simulon profiling CLI covers moe_norm/moe_route/moe_expert kernels
- **Megatron-LM MoE support**: NVIDIA/Megatron-LM main branch has MoE support
- **Batch size**: micro_batch_size=1 for single GPU

---

## Work Objectives

### Core Objective
Run a real Megatron-LM training iteration on a single H100 GPU for a small MoE model, collect per-kernel timing via torch profiler chrome trace, and produce a matching simulon prediction trace for manual validation.

### Concrete Deliverables
1. `templates/model/gpt-oss-1b-moe.yaml` — Model architecture profile
2. `experiments/validation/gpt_oss_1b_training/profile_h100.sh` — SLURM job to profile H100 kernels
3. `experiments/validation/gpt_oss_1b_training/profile_h100.py` — Python profiling script
4. `experiments/validation/gpt_oss_1b_training/run_megatron.sh` — SLURM job for real training
5. `experiments/validation/gpt_oss_1b_training/run_megatron.py` — Megatron-LM pretrain wrapper
6. `experiments/validation/gpt_oss_1b_training/gpt_oss_1b_synthetic.yaml` — Megatron config (synthetic)
7. `experiments/validation/gpt_oss_1b_training/gpt_oss_1b_real.yaml` — Megatron config (real/C4)
8. `experiments/validation/gpt_oss_1b_training/sim_training.py` — Simulon simulation script
9. `experiments/validation/gpt_oss_1b_training/sim_training.yaml` — Simulon scenario config
10. Chrome trace JSONs in `experiments/validation/gpt_oss_1b_training/results/`

### Definition of Done
- [ ] Real Megatron-LM training runs successfully on Snellius (both synthetic and real data)
- [ ] Chrome traces are exported and can be loaded in chrome://tracing
- [ ] Simulon produces a matching chrome trace prediction
- [ ] H100 kernel profile YAML exists and is referenced by the simulon scenario
- [ ] All artifacts are organized in the experiment results directory

### Must Have
- Single GPU execution (all parallelism = 1)
- MoE architecture (not dense) to validate MoE kernel modeling
- Both synthetic and real data runs
- Chrome trace export from real run (torch profiler)
- Chrome trace export from simulon
- H100 kernel profiling for the exact model dimensions

### Must NOT Have (Guardrails)
- Multi-GPU or distributed training
- Full pretraining / convergence
- Automatic plotting or comparison (manual only)
- Modifications to existing experiments or templates
- Source code changes in simulon core (unless absolutely necessary)

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: NO — no test framework for experiment scripts
- **Automated tests**: None (this is an experiment, not a library module)
- **Agent-Executed QA**: MANDATORY for every task — verify files exist, scripts are executable, outputs are produced

### QA Policy
Every task MUST include agent-executed QA scenarios. Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **SLURM scripts**: Use Bash to check syntax (`bash -n`), verify paths
- **Python scripts**: Use Bash to check syntax (`python -m py_compile`)
- **Real training**: Evidence = SLURM job completion, log files, chrome trace existence
- **Simulon simulation**: Evidence = trace.json existence, can load with Python

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation — all can start immediately):
├── T1: Create model template gpt-oss-1b-moe.yaml
├── T2: Add Megatron-LM submodule + install helper
└── T3: Create SLURM profiling script (profile_h100.sh + .py)

Wave 2 (After Wave 1 — profile + configs):
├── T4: Run H100 kernel profiling on Snellius
├── T5: Create Megatron-LM training configs (synthetic + real)
└── T6: Create simulon scenario YAML + sim_training.py

Wave 3 (After Wave 2 — actual runs):
├── T7: Run Megatron-LM training (synthetic data) + chrome trace
├── T8: Run Megatron-LM training (real/C4 data) + chrome trace
└── T9: Run simulon simulation + export chrome trace

Wave 4 (After Wave 3 — collection):
└── T10: Collect results and verify all artifacts

Wave FINAL (After ALL tasks — 4 parallel reviews, then user okay):
├── F1: Plan compliance audit (oracle)
├── F2: Code quality review (unspecified-high)
├── F3: Real manual QA (unspecified-high)
└── F4: Scope fidelity check (deep)
-> Present results -> Get explicit user okay

Critical Path: T1 → T2 → T3 → (T4, T5, T6) → (T7, T8, T9) → T10 → F1-F4 → user okay
Parallel Speedup: ~50% faster than sequential
Max Concurrent: 3 (Waves 2 & 3)
```

### Dependency Matrix

| Task | Depends On | Blocks |
|---|---|---|
| T1 | — | T4, T5, T6, T9 |
| T2 | — | T5, T7, T8 |
| T3 | — | T4 |
| T4 | T1, T3 | T9 |
| T5 | T1, T2 | T7, T8 |
| T6 | T1 | T9 |
| T7 | T5 | T10 |
| T8 | T5 | T10 |
| T9 | T4, T6 | T10 |
| T10 | T7, T8, T9 | F1-F4 |

### Agent Dispatch Summary

- **Wave 1**: T1 → `quick`, T2 → `quick`, T3 → `quick`
- **Wave 2**: T4 → `unspecified-high`, T5 → `unspecified-high`, T6 → `quick`
- **Wave 3**: T7 → `unspecified-high`, T8 → `unspecified-high`, T9 → `quick`
- **Wave 4**: T10 → `quick`
- **FINAL**: F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [x] **T1. Create Model Template `gpt-oss-1b-moe.yaml`**

  **What to do**:
  Create `templates/model/gpt-oss-1b-moe.yaml` defining a small MoE model architecture:
  - `name: gpt-oss-1b-moe`
  - `num_layers: 6`
  - `hidden_size: 1536`
  - `num_heads: 24`
  - `ffn_hidden_size: 6144` (4x hidden)
  - `vocab_size: 32000`
  - `num_experts: 8`
  - `top_k: 2`
  - `max_position_embeddings: 8192`
  - `use_moe: true`
  - `norm_type: rmsnorm`
  - `activation: swiglu`
  - `tie_word_embeddings: false`
  - `dtype: bf16`
  
  Reference existing `templates/model/gpt-oss-5b.yaml` for exact field names and structure. Must be valid YAML matching `LLMSpec` schema from `src/simulon/config/workload.py`.

  **Must NOT do**:
  - Do NOT modify any existing model template
  - Do NOT add fields not present in gpt-oss-5b.yaml

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple file creation from existing pattern
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T2, T3)
  - **Blocks**: T4, T5, T6, T9
  - **Blocked By**: None

  **References**:
  - `templates/model/gpt-oss-5b.yaml` — Field names and structure to copy
  - `templates/model/gpt-oss-20b.yaml` — Alternative reference for MoE fields
  - `src/simulon/config/workload.py:LLMSpec` — Schema validation

  **Acceptance Criteria**:
  - [ ] File exists at `templates/model/gpt-oss-1b-moe.yaml`
  - [ ] `python -c "import yaml; yaml.safe_load(open('templates/model/gpt-oss-1b-moe.yaml'))"` succeeds
  - [ ] Contains all required fields matching gpt-oss-5b.yaml structure

  **QA Scenarios**:
  ```
  Scenario: Model template is valid YAML
    Tool: Bash
    Steps:
      1. python -c "import yaml; d=yaml.safe_load(open('templates/model/gpt-oss-1b-moe.yaml')); print(d['name'])"
    Expected Result: Output is "gpt-oss-1b-moe"
    Evidence: .sisyphus/evidence/task-t1-yaml-valid.txt
  ```

  **Commit**: YES
  - Message: `feat(validation): add gpt-oss-1b-moe model template`
  - Files: `templates/model/gpt-oss-1b-moe.yaml`

---

- [x] **T2. Add Megatron-LM Submodule + Install Helper**

  **What to do**:
  1. Add NVIDIA/Megatron-LM as a git submodule in the experiment directory:
     ```bash
     git submodule add https://github.com/NVIDIA/Megatron-LM.git experiments/validation/gpt_oss_1b_training/megatron-lm
     ```
  2. Create `experiments/validation/gpt_oss_1b_training/install_megatron.sh`:
     - Install dependencies (torch, apex, etc.)
     - Set up Megatron-LM from submodule
     - Verify MoE support is available
     - Handle Snellius environment (modules, CUDA 12.8, etc.)
  3. Update `.gitmodules` if needed
  4. Create a minimal `requirements.txt` in the experiment dir listing needed packages

  **Must NOT do**:
  - Do NOT install Megatron-LM globally — keep it scoped to the experiment
  - Do NOT modify existing source code

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Git operations and shell script creation
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T1, T3)
  - **Blocks**: T5, T7, T8
  - **Blocked By**: None

  **References**:
  - `scripts/profile_h100.sh` — Snellius SLURM environment setup (modules, CUDA)
  - NVIDIA/Megatron-LM README — Installation instructions

  **Acceptance Criteria**:
  - [ ] `.gitmodules` updated with megatron-lm submodule path
  - [ ] `experiments/validation/gpt_oss_1b_training/megatron-lm/` directory exists (can be empty until submodule init)
  - [ ] `install_megatron.sh` exists and is executable
  - [ ] `requirements.txt` exists

  **QA Scenarios**:
  ```
  Scenario: Submodule is registered
    Tool: Bash
    Steps:
      1. git config --file .gitmodules --get-regexp path | grep megatron-lm
    Expected Result: Output contains the submodule path
    Evidence: .sisyphus/evidence/task-t2-submodule.txt
  ```

  **Commit**: YES (groups with T1-T3)
  - Message: `feat(validation): add gpt-oss-1b-moe experiment foundation`
  - Files: `.gitmodules`, `experiments/validation/gpt_oss_1b_training/install_megatron.sh`, `experiments/validation/gpt_oss_1b_training/requirements.txt`

---

- [x] **T3. Create SLURM Profiling Script**

  **What to do**:
  Create two files for H100 kernel profiling:
  
  1. `experiments/validation/gpt_oss_1b_training/profile_h100.sh` — SLURM batch script:
     - `#SBATCH --partition=gpu`
     - `#SBATCH --gpus=1`
     - `#SBATCH --time=01:00:00`
     - Load modules (CUDA 12.8, etc.) matching `scripts/profile_h100.sh`
     - Run `profile_h100.py`
     - Output to `results/h100_profile.yaml`
  
  2. `experiments/validation/gpt_oss_1b_training/profile_h100.py` — Python profiling script:
     - Uses `simulon profile gpu` CLI programmatically (or via subprocess)
     - Profiles kernels for the gpt-oss-1b-moe dimensions:
       - hidden_size=1536, num_heads=24, ffn_hidden_size=6144
       - seq_len=8192, batch_size=1, vocab_size=32000
       - tp=1 (single GPU)
     - Runs 20 epochs for stable timing
     - Appends results to `results/h100_profile.yaml`
     - Also profiles MoE-specific dimensions if supported by CLI

  **Must NOT do**:
  - Do NOT hardcode absolute paths that won't work on Snellius
  - Do NOT modify `scripts/profile_h100.sh`

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Script creation from existing pattern
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T1, T2)
  - **Blocks**: T4
  - **Blocked By**: None

  **References**:
  - `scripts/profile_h100.sh` — Snellius SLURM environment (modules, CUDA 12.8)
  - `src/simulon/cli/__init__.py` — `simulon profile gpu` CLI arguments
  - `src/simulon/profiling/kernels.py` — `benchmark_kernels()` function signature

  **Acceptance Criteria**:
  - [ ] `profile_h100.sh` exists and `bash -n profile_h100.sh` succeeds
  - [ ] `profile_h100.py` exists and `python -m py_compile profile_h100.py` succeeds
  - [ ] Script references correct model dimensions (hidden=1536, heads=24, ffn=6144)

  **QA Scenarios**:
  ```
  Scenario: Profiling scripts are syntactically valid
    Tool: Bash
    Steps:
      1. bash -n experiments/validation/gpt_oss_1b_training/profile_h100.sh
      2. python -m py_compile experiments/validation/gpt_oss_1b_training/profile_h100.py
    Expected Result: Both commands exit 0
    Evidence: .sisyphus/evidence/task-t3-syntax-valid.txt
  ```

  **Commit**: YES (groups with T1-T3)
  - Message: `feat(validation): add gpt-oss-1b-moe experiment foundation`
  - Files: `experiments/validation/gpt_oss_1b_training/profile_h100.sh`, `experiments/validation/gpt_oss_1b_training/profile_h100.py`

---

- [ ] **T4. Run H100 Kernel Profiling on Snellius** (REQUIRES MANUAL EXECUTION — not on Snellius)

  **What to do**:
  1. Submit `profile_h100.sh` to SLURM: `sbatch experiments/validation/gpt_oss_1b_training/profile_h100.sh`
  2. Wait for job completion (`squeue` polling or ` sacct`)
  3. Verify output file `results/h100_profile.yaml` exists and is non-empty
  4. Check that key kernel types are present: `layernorm`, `attn_qkv`, `attn_flash`, `attn_proj`, `mlp_linear1`, `mlp_linear2`, `moe_norm`, `moe_route`, `moe_expert`
  5. Copy the generated profile YAML to `templates/gpu/h100.gpt_oss_1b_moe.yaml` (or append to existing if the CLI already appends)

  **Must NOT do**:
  - Do NOT run profiling on the login node
  - Do NOT overwrite existing `templates/gpu/h100.yaml` or `h100.profile.yaml`

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires running SLURM job and interacting with cluster
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T5, T6)
  - **Parallel Group**: Wave 2
  - **Blocks**: T9
  - **Blocked By**: T1, T3

  **References**:
  - `scripts/profile_h100.sh` — SLURM submission pattern
  - `templates/gpu/h100.profile.yaml` — Expected output format

  **Acceptance Criteria**:
  - [ ] SLURM job completes successfully
  - [ ] `results/h100_profile.yaml` exists and size > 0
  - [ ] Contains kernel_runs entries for the profiled dimensions
  - [ ] Key MoE kernels present (if supported by profiler)

  **QA Scenarios**:
  ```
  Scenario: Profiling job completes and produces output
    Tool: Bash
    Preconditions: T1 and T3 complete
    Steps:
      1. sbatch experiments/validation/gpt_oss_1b_training/profile_h100.sh
      2. Wait for job completion (poll squeue or use --wait)
      3. ls -lh experiments/validation/gpt_oss_1b_training/results/h100_profile.yaml
      4. grep -c "kernel_runs" experiments/validation/gpt_oss_1b_training/results/h100_profile.yaml
    Expected Result: File exists, size > 1KB, grep count >= 1
    Evidence: .sisyphus/evidence/task-t4-profile-output.txt
  ```

  **Commit**: YES (groups with T4-T6)
  - Message: `feat(validation): add training configs and profiling run`
  - Files: `experiments/validation/gpt_oss_1b_training/results/h100_profile.yaml`, `templates/gpu/h100.gpt_oss_1b_moe.yaml`

---

- [x] **T5. Create Megatron-LM Training Configs**

  **What to do**:
  Create two Megatron-LM training configuration files and a wrapper script:
  
  1. `experiments/validation/gpt_oss_1b_training/gpt_oss_1b_synthetic.yaml`:
     - Megatron-LM pretrain arguments for synthetic data
     - Model: 6 layers, 1536 hidden, 24 heads, 6144 ffn, vocab=32000
     - MoE: 8 experts, top-2
     - Data: `--data-path synthetic`, `--split 100,0,0`
     - Training: micro-batch=1, global-batch=1, seq-length=8192
     - 5 warmup + 1 profile iteration
     - Enable torch profiler chrome trace export
     - TP=PP=EP=DP=1 (single GPU)
     - bf16 precision
  
  2. `experiments/validation/gpt_oss_1b_training/gpt_oss_1b_real.yaml`:
     - Same model config as synthetic
     - Data: C4 dataset from HuggingFace datasets
     - Variable sequence length (packing or padding to max 8192)
     - Same training params otherwise
  
  3. `experiments/validation/gpt_oss_1b_training/run_megatron.py`:
     - Python wrapper that calls Megatron-LM pretrain.py with the YAML config
     - Sets up torch profiler to export chrome trace
     - Captures step time logging
     - Handles both synthetic and real data modes
     - Saves outputs to `results/` directory

  **Must NOT do**:
  - Do NOT create a full data preprocessing pipeline — use HuggingFace datasets directly
  - Do NOT modify Megatron-LM source code

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires understanding Megatron-LM CLI and config format
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T4, T6)
  - **Parallel Group**: Wave 2
  - **Blocks**: T7, T8
  - **Blocked By**: T1, T2

  **References**:
  - `examples/gpt_oss_5b_training.yaml` — simulon workload config (model dims reference)
  - NVIDIA/Megatron-LM README — Pretrain arguments and MoE config
  - `tests/test_moe.py` — MoE test patterns in simulon
  - `tests/test_e2e.py` — MegatronDAGTracer integration patterns

  **Acceptance Criteria**:
  - [ ] Both YAML configs exist and are valid
  - [ ] `run_megatron.py` compiles without syntax errors
  - [ ] Configs specify single GPU (TP=PP=EP=DP=1)
  - [ ] Real data config references C4 dataset
  - [ ] Both configs include torch profiler chrome trace export settings

  **QA Scenarios**:
  ```
  Scenario: Training configs are valid
    Tool: Bash
    Steps:
      1. python -m py_compile experiments/validation/gpt_oss_1b_training/run_megatron.py
      2. python -c "import yaml; yaml.safe_load(open('experiments/validation/gpt_oss_1b_training/gpt_oss_1b_synthetic.yaml'))"
      3. python -c "import yaml; yaml.safe_load(open('experiments/validation/gpt_oss_1b_training/gpt_oss_1b_real.yaml'))"
    Expected Result: All three commands exit 0
    Evidence: .sisyphus/evidence/task-t5-configs-valid.txt
  ```

  **Commit**: YES (groups with T4-T6)
  - Message: `feat(validation): add training configs and profiling run`
  - Files: `experiments/validation/gpt_oss_1b_training/gpt_oss_1b_synthetic.yaml`, `experiments/validation/gpt_oss_1b_training/gpt_oss_1b_real.yaml`, `experiments/validation/gpt_oss_1b_training/run_megatron.py`

---

- [x] **T6. Create Simulon Scenario YAML + Simulation Script**

  **What to do**:
  Create two files for simulon simulation:
  
  1. `experiments/validation/gpt_oss_1b_training/sim_training.yaml`:
     - `datacenter`: Use `from: snellius-h100-4g` node template
       - Override to single GPU: `gpus_per_node: 1`, `num_nodes: 1`
       - GPU: reference `h100` template, but point kernel profile to the experiment-specific profile YAML
     - `workload`: `framework: megatron`
       - `model: gpt-oss-1b-moe`
       - `parallelism`: tp=1, pp=1, ep=1, dp=1, num_microbatches=1
       - `training`: num_gpus=1, global_batch_size=1, micro_batch_size=1, sequence_length=8192
     - `collective`: library=nccl, algorithm=ring, num_channels=1
  
  2. `experiments/validation/gpt_oss_1b_training/sim_training.py`:
     - Load `sim_training.yaml` as `ScenarioConfig`
     - Instantiate `AnalyticalBackend()`
     - Call `backend.simulate(scenario)` to get DAG + result
     - Export chrome trace to `results/sim_trace.json`
     - Print total step time for comparison
     - Optionally print per-kernel breakdown

  **Must NOT do**:
  - Do NOT modify existing scenario examples
  - Do NOT use a multi-GPU config

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: YAML + small Python script from existing patterns
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T4, T5)
  - **Parallel Group**: Wave 2
  - **Blocks**: T9
  - **Blocked By**: T1

  **References**:
  - `examples/gpt_oss_5b_training.yaml` — Full scenario structure
  - `experiments/validation/simccl/sim_ccl.py` — How to build ScenarioConfig programmatically
  - `src/simulon/backend/analytical.py` — AnalyticalBackend usage
  - `src/simulon/config/scenario.py` — ScenarioConfig schema

  **Acceptance Criteria**:
  - [ ] `sim_training.yaml` is valid YAML matching ScenarioConfig schema
  - [ ] `sim_training.py` compiles without errors
  - [ ] Script references the correct model template and node template
  - [ ] Single GPU configuration (1 node, 1 GPU)

  **QA Scenarios**:
  ```
  Scenario: Simulon scenario is valid and simulatable
    Tool: Bash
    Steps:
      1. python -m py_compile experiments/validation/gpt_oss_1b_training/sim_training.py
      2. python -c "from simulon.config.scenario import ScenarioConfig; import yaml; ScenarioConfig.model_validate(yaml.safe_load(open('experiments/validation/gpt_oss_1b_training/sim_training.yaml')))"
    Expected Result: Both commands exit 0
    Evidence: .sisyphus/evidence/task-t6-scenario-valid.txt
  ```

  **Commit**: YES (groups with T4-T6)
  - Message: `feat(validation): add training configs and profiling run`
  - Files: `experiments/validation/gpt_oss_1b_training/sim_training.yaml`, `experiments/validation/gpt_oss_1b_training/sim_training.py`

---

- [ ] **T7. Run Megatron-LM Training (Synthetic Data) + Chrome Trace** (REQUIRES MANUAL EXECUTION — not on Snellius)

  **What to do**:
  1. Create `experiments/validation/gpt_oss_1b_training/run_megatron_synthetic.sh` — SLURM batch script:
     - `#SBATCH --partition=gpu`, `#SBATCH --gpus=1`, `#SBATCH --time=02:00:00`
     - Set up environment (modules, activate env, install megatron-lm if needed)
     - Run `python run_megatron.py --config gpt_oss_1b_synthetic.yaml`
     - Save logs to `results/megatron_synthetic.log`
  2. Submit job and wait for completion
  3. Verify outputs:
     - Chrome trace JSON exists at `results/chrome_trace_synthetic.json`
     - Log file contains step time measurements
     - No OOM or crash errors in logs
  4. Capture key metrics from logs (step time, tokens/sec, memory usage)

  **Must NOT do**:
  - Do NOT run on login node
  - Do NOT modify Megatron-LM source to fix issues — report and workaround in wrapper script

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires running real training on SLURM cluster
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T8, T9)
  - **Parallel Group**: Wave 3
  - **Blocks**: T10
  - **Blocked By**: T5

  **References**:
  - `experiments/validation/gpt_oss_1b_training/run_megatron.py` — Created in T5
  - `experiments/validation/gpt_oss_1b_training/gpt_oss_1b_synthetic.yaml` — Created in T5
  - `scripts/profile_h100.sh` — SLURM environment setup pattern

  **Acceptance Criteria**:
  - [ ] SLURM job completes successfully
  - [ ] `results/chrome_trace_synthetic.json` exists and size > 10KB
  - [ ] Log file shows 5 warmup + 1 profile iteration completed
  - [ ] Chrome trace contains traceEvents array with compute kernels

  **QA Scenarios**:
  ```
  Scenario: Synthetic training produces chrome trace
    Tool: Bash
    Preconditions: T5 complete
    Steps:
      1. sbatch experiments/validation/gpt_oss_1b_training/run_megatron_synthetic.sh
      2. Wait for completion
      3. ls -lh experiments/validation/gpt_oss_1b_training/results/chrome_trace_synthetic.json
      4. python -c "import json; d=json.load(open('experiments/validation/gpt_oss_1b_training/results/chrome_trace_synthetic.json')); print(len(d['traceEvents']))"
    Expected Result: File exists, size > 10KB, traceEvents count > 100
    Evidence: .sisyphus/evidence/task-t7-synthetic-trace.txt
  ```

  **Commit**: YES (groups with T7-T9)
  - Message: `feat(validation): add real and simulated training runs`
  - Files: `experiments/validation/gpt_oss_1b_training/run_megatron_synthetic.sh`, `experiments/validation/gpt_oss_1b_training/results/*`

---

- [ ] **T8. Run Megatron-LM Training (Real/C4 Data) + Chrome Trace** (REQUIRES MANUAL EXECUTION — not on Snellius)

  **What to do**:
  1. Create `experiments/validation/gpt_oss_1b_training/run_megatron_real.sh` — SLURM batch script:
     - Same structure as synthetic but uses `gpt_oss_1b_real.yaml`
     - Downloads C4 dataset from HuggingFace datasets at runtime
     - Handles variable sequence length (packing or dynamic padding)
     - Save logs to `results/megatron_real.log`
  2. Submit job and wait for completion
  3. Verify outputs:
     - Chrome trace JSON exists at `results/chrome_trace_real.json`
     - Log file contains step time measurements
     - Dataset loaded successfully (check logs)

  **Must NOT do**:
  - Do NOT pre-download massive datasets to the experiment directory
  - Do NOT modify Megatron-LM source

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires running real training with dataset loading on SLURM
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T7, T9)
  - **Parallel Group**: Wave 3
  - **Blocks**: T10
  - **Blocked By**: T5

  **References**:
  - `experiments/validation/gpt_oss_1b_training/run_megatron.py` — Created in T5
  - `experiments/validation/gpt_oss_1b_training/gpt_oss_1b_real.yaml` — Created in T5
  - HuggingFace `datasets` library — C4 loading (`load_dataset("c4", "en")`)

  **Acceptance Criteria**:
  - [ ] SLURM job completes successfully
  - [ ] `results/chrome_trace_real.json` exists and size > 10KB
  - [ ] Log shows C4 dataset loaded and processed
  - [ ] Chrome trace contains traceEvents array

  **QA Scenarios**:
  ```
  Scenario: Real data training produces chrome trace
    Tool: Bash
    Preconditions: T5 complete
    Steps:
      1. sbatch experiments/validation/gpt_oss_1b_training/run_megatron_real.sh
      2. Wait for completion
      3. ls -lh experiments/validation/gpt_oss_1b_training/results/chrome_trace_real.json
      4. grep -i "c4\|dataset" experiments/validation/gpt_oss_1b_training/results/megatron_real.log | head -5
    Expected Result: Trace file exists, logs mention dataset loading
    Evidence: .sisyphus/evidence/task-t8-real-trace.txt
  ```

  **Commit**: YES (groups with T7-T9)
  - Message: `feat(validation): add real and simulated training runs`
  - Files: `experiments/validation/gpt_oss_1b_training/run_megatron_real.sh`, `experiments/validation/gpt_oss_1b_training/results/*`

---

- [x] **T9. Run Simulon Simulation + Export Chrome Trace**

  **What to do**:
  1. Ensure kernel profile YAML from T4 is available at the path referenced in `sim_training.yaml`
  2. Run `python experiments/validation/gpt_oss_1b_training/sim_training.py`
     - This can run locally (no SLURM needed) since simulon is pure Python
  3. Verify outputs:
     - `results/sim_trace.json` exists and is valid Chrome trace format
     - Script prints total step time
  4. Optionally: Compare simulon step time against real step times from T7/T8 logs (rough sanity check)

  **Must NOT do**:
  - Do NOT modify simulon source code to fix issues
  - Do NOT run on SLURM (simulon is CPU-only simulation)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Pure Python script execution, no GPU needed
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T7, T8)
  - **Parallel Group**: Wave 3
  - **Blocks**: T10
  - **Blocked By**: T4, T6

  **References**:
  - `experiments/validation/gpt_oss_1b_training/sim_training.py` — Created in T6
  - `experiments/validation/gpt_oss_1b_training/sim_training.yaml` — Created in T6
  - `src/simulon/backend/dag/chrome_trace.py` — Chrome trace export format

  **Acceptance Criteria**:
  - [ ] `sim_training.py` runs successfully and exits 0
  - [ ] `results/sim_trace.json` exists and is valid Chrome trace
  - [ ] Script prints total step time to stdout
  - [ ] traceEvents array contains ComputeNode and CommNode entries

  **QA Scenarios**:
  ```
  Scenario: Simulon produces chrome trace
    Tool: Bash
    Preconditions: T4 and T6 complete
    Steps:
      1. python experiments/validation/gpt_oss_1b_training/sim_training.py
      2. ls -lh experiments/validation/gpt_oss_1b_training/results/sim_trace.json
      3. python -c "import json; d=json.load(open('experiments/validation/gpt_oss_1b_training/results/sim_trace.json')); print(len(d['traceEvents'])); print(any('ComputeNode' in str(e) or 'CommNode' in str(e) for e in d['traceEvents'][:10]))"
    Expected Result: File exists, traceEvents > 50, contains compute/comm nodes
    Evidence: .sisyphus/evidence/task-t9-sim-trace.txt
  ```

  **Commit**: YES (groups with T7-T9)
  - Message: `feat(validation): add real and simulated training runs`
  - Files: `experiments/validation/gpt_oss_1b_training/results/sim_trace.json`

---

- [x] **T10. Collect Results and Verify All Artifacts**

  **What to do**:
  1. Create `experiments/validation/gpt_oss_1b_training/results/README.md` documenting:
     - What each artifact is
     - Real step times (synthetic vs real data)
     - Simulon predicted step time
     - File paths for all traces
     - Any notes or caveats
  2. Verify ALL expected artifacts exist and are non-empty:
     - `results/h100_profile.yaml`
     - `results/chrome_trace_synthetic.json`
     - `results/chrome_trace_real.json`
     - `results/sim_trace.json`
     - `results/megatron_synthetic.log`
     - `results/megatron_real.log`
  3. Run validation checks:
     - All JSON traces are valid Chrome trace format
     - All YAML configs are valid
     - Log files contain expected iteration counts
  4. Print summary of results for user review

  **Must NOT do**:
  - Do NOT generate comparison plots (out of scope — user wants manual comparison)
  - Do NOT delete any intermediate files

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: File verification and README creation
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (must wait for T7, T8, T9)
  - **Parallel Group**: Wave 4 (sequential)
  - **Blocks**: F1-F4
  - **Blocked By**: T7, T8, T9

  **References**:
  - `experiments/validation/simccl/results/` — Existing results structure

  **Acceptance Criteria**:
  - [ ] All 6 expected artifact files exist and are non-empty
  - [ ] `results/README.md` exists with documentation
  - [ ] All JSON traces pass validation
  - [ ] Summary printed with step times

  **QA Scenarios**:
  ```
  Scenario: All artifacts present and valid
    Tool: Bash
    Steps:
      1. for f in h100_profile.yaml chrome_trace_synthetic.json chrome_trace_real.json sim_trace.json megatron_synthetic.log megatron_real.log; do test -s experiments/validation/gpt_oss_1b_training/results/$f && echo "OK: $f" || echo "MISSING: $f"; done
      2. python -c "import json; [json.load(open(f'experiments/validation/gpt_oss_1b_training/results/chrome_trace_{s}.json')) for s in ['synthetic', 'real']]"
      3. python -c "import json; json.load(open('experiments/validation/gpt_oss_1b_training/results/sim_trace.json'))"
    Expected Result: All files OK, all JSON valid
    Evidence: .sisyphus/evidence/task-t10-artifacts-valid.txt
  ```

  **Commit**: YES
  - Message: `feat(validation): collect validation results`
  - Files: `experiments/validation/gpt_oss_1b_training/results/README.md`

---

## Final Verification Wave

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [x] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [x] F2. **Code Quality Review** — `unspecified-high`
  Check all new files for: syntax errors, missing imports, hardcoded paths, incorrect YAML formatting. Verify SLURM scripts are syntactically valid. Check Python scripts compile. Verify no source code modifications outside experiment directory.
  Output: `Files [N clean/N issues] | VERDICT`

- [x] F3. **Real Manual QA** — `unspecified-high`
  Start from clean state. Verify: Megatron-LM submodule is correctly added, install script works, training configs are valid YAML, simulon scenario is valid, all result artifacts exist and are non-empty. Check chrome traces are valid JSON with traceEvents array.
  Output: `Artifacts [N/N pass] | VERDICT`

- [x] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff/files. Verify 1:1 — everything in spec was built, nothing beyond spec was built. Check "Must NOT do" compliance. Verify no existing files were modified.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | VERDICT`

---

## Commit Strategy

- **T1-T3**: `feat(validation): add gpt-oss-1b-moe experiment foundation`
- **T4-T6**: `feat(validation): add training configs and profiling run`
- **T7-T9**: `feat(validation): add real and simulated training runs`
- **T10**: `feat(validation): collect validation results`

---

## Success Criteria

### Verification Commands
```bash
# Verify model template exists and is valid YAML
python -c "import yaml; yaml.safe_load(open('templates/model/gpt-oss-1b-moe.yaml'))"

# Verify experiment directory structure
ls experiments/validation/gpt_oss_1b_training/{run_megatron.sh,profile_h100.sh,sim_training.py,sim_training.yaml}

# Verify chrome traces are valid JSON with traceEvents
python -c "import json; d=json.load(open('experiments/validation/gpt_oss_1b_training/results/chrome_trace_synthetic.json')); assert 'traceEvents' in d"

# Verify simulon trace is valid
python -c "import json; d=json.load(open('experiments/validation/gpt_oss_1b_training/results/sim_trace.json')); assert 'traceEvents' in d"

# Verify kernel profile YAML exists
ls experiments/validation/gpt_oss_1b_training/results/h100_profile.yaml
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All deliverable files exist and are non-empty
- [ ] Chrome traces are valid and loadable
- [ ] No existing files modified (except .gitmodules for submodule)
