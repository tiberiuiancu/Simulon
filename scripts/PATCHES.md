# Simulon Development Notes

## Megatron-LM Patches for `--fake-process-group` Mode

These patches are applied to the `experiments/validate_compute/megatron-lm/` submodule to enable single-GPU profiling of multi-GPU distributed training configurations.

### Patch 1: Zero warmup microbatches (prevents OOM)

**File:** `megatron/core/pipeline_parallel/schedules.py`
**Location:** `forward_backward_pipelining_without_interleaving()` ~line 2167

**Problem:** With PP > 1, the 1F1B schedule stores activations for all warmup microbatches (PP-1 of them) before any backward pass frees memory. At GBS=15360 with DP=128, this means 15 microbatches × their full autograd graphs are alive simultaneously, causing OOM even on an H100.

**Patch:**
```python
# BEFORE:
num_warmup_microbatches = p2p_communicator.total_stages - p2p_communicator.current_stage - 1

# AFTER:
num_warmup_microbatches = 0
```

**Effect:** Only 1 microbatch's graph is alive at any time. GBS=15360 no longer OOMs.

**Proper fix:** Make this configurable via CLI flag (e.g. `--pipeline-warmup-microbatches 0`) so users can tune the memory/compute trade-off.

---

### Patch 2: All ranks build data pipeline (fixes instant exit on RANK > 0)

**File:** `megatron/training/training.py`
**Location:** `build_train_valid_test_data_loaders()` ~line 3665

**Problem:** The data loader builder has this guard:
```python
if is_distributed or mpu.get_tensor_model_parallel_rank() == 0:
    # build data, set flags = [1, 0, 0]
else:
    flags = torch.tensor([0, 0, 0], ...)

torch.distributed.broadcast(flags, 0)
```

With `--fake-process-group`, `broadcast` is a **complete no-op** in PyTorch's FakeProcessGroup (tensors are ignored). So non-TP-rank-0 ranks keep `[0, 0, 0]`, which means `do_train=False`, and they skip the training loop entirely — exiting instantly after model build.

**Patch:**
```python
# BEFORE:
if is_distributed or mpu.get_tensor_model_parallel_rank() == 0:

# AFTER:
if is_distributed or mpu.get_tensor_model_parallel_rank() == 0 or args.fake_process_group:
```

**Effect:** All ranks build their own data iterators when fake process group is active, so `do_train=True` on every rank.

**Proper fix:** Same as above — add `or args.fake_process_group` to the guard, or refactor the data pipeline initialization to not rely on broadcast when using fake backend.

---

## Scripts Added

### `scripts/train_deepseek_v3_fake_single_iter.slurm`
DeepSeek V3 (671B) fake-PG training with:
- TP=1, PP=16, EP=64, DP=2 (world_size=2048)
- 64 layers, hidden=7168, 256 experts, MLA, MTP
- GBS=15360, MBS=4, 10 iters, 2 warmup

### `scripts/train_llama_fake_single_iter.slurm`
Llama 70B dense fake-PG training with:
- TP=4, PP=4 (world_size=32, DP derived=2)
- 80 layers, hidden=8192, ffn=28672
- Same batch config as DeepSeek for fair MFU comparison

### `scripts/install_flash_attention_hopper.sh`
Standalone bash script that installs:
1. Flash Attention 2 via `pip install flash-attn`
2. Flash Attention 3 (Hopper) via `cd hopper && python setup.py install`

---

## CLI Commands Added

### `simulon install flash-attn-hopper`
Installs Flash Attention 3 (Hopper-optimized) for H100 GPUs.

**Options:**
- `--prebuilt`: Install from prebuilt wheel instead of building from source
- `--version X.Y.Z`: Specify exact flash-attn version (e.g. `2.7.4`)

**How prebuilt works:**
1. Detects Python, PyTorch, CUDA versions from current env
2. Queries GitHub API for all releases of `mjun0812/flash-attention-prebuild-wheels`
3. Searches releases for matching wheel (FA3 preferred, FA2 fallback)
4. Installs via `pip install <wheel_url>`

**Note:** Uses `sys.executable` to ensure the venv Python is used.
