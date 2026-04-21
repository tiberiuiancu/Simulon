# profiling — GPU Kernel Benchmarking + Lookup

5 files. Measures and retrieves transformer kernel timings for DAG population.

## OVERVIEW

Two sides: (1) `kernels.py` benchmarks actual GPU kernels via CUDA event timing, producing `KernelRun` records saved to GPU templates. (2) `lookup.py` retrieves timing at simulation time via nearest-neighbor matching on kernel parameters.

## STRUCTURE

```
profiling/
├── kernels.py   # benchmark_kernels() — CUDA event timing for all kernel types (564 lines)
├── lookup.py    # lookup_kernel_time() — nearest-neighbor match with extrapolation + caching
├── models.py    # Shared kernel parameter models / types
├── sweep.py     # Parameter sweep configuration for CLI profiling
└── __init__.py  # Empty
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Add new kernel benchmark | `kernels.py` | Add `_bench_{name}()` + register in `benchmark_kernels()` |
| Modify lookup/matching | `lookup.py` | Nearest-neighbor on param dict; cache via `@lru_cache` |
| Sweep parameters for CLI | `sweep.py` | Drives `simulon profile gpu` parameter combinations |

## CONVENTIONS

- **`_bench_{name}()` pattern** — each kernel has its own benchmark function: `_bench_attn_flash`, `_bench_mlp_linear1`, etc.
- **`_cuda_time()` helper** — warmup + repeated CUDA event timing, returns median ms
- **Incremental profiling** — `benchmark_kernels()` accepts existing runs and skips already-profiled configs
- **OOM tracking** — OOM'd parameter combos are recorded separately to avoid re-attempting
- **Extrapolation flag** — `lookup_kernel_time` sets `is_extrapolated=True` when no exact match found

## ANTI-PATTERNS

- **Parameter formulas MUST sync** — `kernels.py` param calculations (lines 530-542) must match `megatron_tracer.py` (lines 64-105). Drift = wrong timing
- **Grouped GEMM fallback** — `_bench_moe_expert` tries grouped_gemm first, falls back to sequential matmul. Different codepaths may give different results
- **Requires torch + CUDA** — benchmarking only works on GPU machines. Tests use `@requires_torch` / `@requires_cuda` skip markers
