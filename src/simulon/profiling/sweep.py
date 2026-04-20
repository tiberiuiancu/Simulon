"""Sweep helpers for GPU profiling CLI.

Provides:
  - parse_sweep: parse a comma-separated string of ints into a list
  - SweepResult: result container for a single (tp, ep, batch_size, seq_len) config
  - run_sweep: run benchmark_kernels over a grid of configs, catching OOM errors
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from simulon.config.common import DType
from simulon.config.dc import KernelRun


def parse_sweep(value: str) -> list[int]:
    """Parse a comma-separated string of integers into a list.

    Examples:
        "1"       -> [1]
        "1,2,4"   -> [1, 2, 4]
        "8, 16"   -> [8, 16]
    """
    return [int(v.strip()) for v in value.split(",") if v.strip()]


@dataclass
class SweepResult:
    """Result for a single profiling configuration."""

    config: dict
    runs: Optional[list[KernelRun]] = field(default=None)
    oom: bool = False
    oom_runs: list[KernelRun] = field(default_factory=list)


def _inferred_oom(
    tp: int,
    ep: int,
    batch_size: int,
    seq_len: int,
    known_ooms: list[tuple[int, int, int, int]],
) -> bool:
    """Return True if a known OOM config implies this config will also OOM.

    Dominance rule: a known-OOM (tp_oom, ep_oom, bs_oom, sl_oom) implies OOM when:
      - tp <= tp_oom  (less sharding → more memory per GPU)
      - ep <= ep_oom
      - batch_size >= bs_oom  (larger batch → more activation memory)
      - seq_len >= sl_oom
    """
    for tp_oom, ep_oom, bs_oom, sl_oom in known_ooms:
        if tp <= tp_oom and ep <= ep_oom and batch_size >= bs_oom and seq_len >= sl_oom:
            return True
    return False


def _make_oom_kernel_runs(
    kernel_params: dict,
    tp: int,
    ep: int,
    batch_size: int,
    seq_len: int,
    dtype: DType,
) -> list[KernelRun]:
    """Generate per-kernel OOM entries for a failed (tp, ep, batch_size, seq_len) config.

    For each kernel in KERNEL_MATCH_KEYS whose canonical params are fully covered by
    the current profiling config, produces a KernelRun with times_ms=[] recording that
    the kernel with these params hit OOM.
    """
    from simulon.profiling.lookup import KERNEL_MATCH_KEYS, _filter_params

    all_params = {
        "hidden_size": kernel_params["hidden_size"],
        "num_heads": kernel_params["num_heads"],
        "ffn_hidden_size": kernel_params["ffn_hidden_size"],
        "vocab_size": kernel_params["vocab_size"],
        "seq_len": seq_len,
        "batch_size": batch_size,
        "dtype": dtype.value,
        "tp": tp,
        "ep": ep,
    }
    if kernel_params.get("num_experts"):
        all_params["num_experts"] = kernel_params["num_experts"]
    if kernel_params.get("top_k"):
        all_params["top_k"] = kernel_params["top_k"]

    oom_runs: list[KernelRun] = []
    for kernel, canonical_keys in KERNEL_MATCH_KEYS.items():
        filtered = _filter_params(kernel, all_params)
        # Only emit an OOM entry when all canonical params are present (i.e. the kernel
        # would actually have been benchmarked in this profiling config).
        if frozenset(filtered.keys()) == canonical_keys:
            oom_runs.append(KernelRun(kernel=kernel, params=filtered, times_ms=[]))
    return oom_runs


def run_sweep(
    kernel_params: dict,
    tp_values: list[int],
    ep_values: list[int],
    batch_sizes: list[int],
    seq_lens: list[int],
    dtype: DType,
    epoch_num: int = 10,
    existing_runs: Optional[list[dict]] = None,
) -> list[SweepResult]:
    """Run benchmark_kernels for every combination of (tp, ep, batch_size, seq_len).

    Configs are tried in order from most memory-efficient to least, so OOM
    boundaries are discovered early.  Once a config OOMs, any config that is
    at least as memory-intensive (lower TP/EP, larger batch/seq) is inferred
    to also OOM and skipped without running.

    OOM errors are caught and recorded as SweepResult(oom=True).

    Args:
        kernel_params: Dict with keys: hidden_size, num_heads, ffn_hidden_size,
                       vocab_size, and optionally num_experts, top_k, swiglu.
        tp_values: List of tensor-parallelism degrees to sweep.
        ep_values: List of expert-parallelism degrees to sweep.
        batch_sizes: List of micro-batch sizes to sweep.
        seq_lens: List of sequence lengths to sweep.
        dtype: Compute precision.
        epoch_num: Number of timed iterations per kernel.

    Returns:
        List of SweepResult, one per (tp, ep, batch_size, seq_len) combination.
        Order matches the input product but with inferred OOMs inserted.
    """
    from itertools import product

    from simulon.profiling.kernels import benchmark_kernels

    # Sort configs from most memory-efficient (high TP/EP, small BS/SL) to least,
    # so we discover OOM boundaries as early as possible.
    sorted_configs = sorted(
        product(tp_values, ep_values, batch_sizes, seq_lens),
        key=lambda c: (-c[0], -c[1], c[2], c[3]),
    )

    # Track confirmed OOM (tp, ep, batch_size, seq_len) tuples for inference.
    known_ooms: list[tuple[int, int, int, int]] = []

    results: list[SweepResult] = []

    for tp, ep, batch_size, seq_len in sorted_configs:
        config = {"tp": tp, "ep": ep, "batch_size": batch_size, "seq_len": seq_len}

        if _inferred_oom(tp, ep, batch_size, seq_len, known_ooms):
            oom_runs = _make_oom_kernel_runs(kernel_params, tp, ep, batch_size, seq_len, dtype)
            results.append(SweepResult(config=config, runs=None, oom=True, oom_runs=oom_runs))
            continue

        try:
            runs, oom_runs = benchmark_kernels(
                hidden_size=kernel_params["hidden_size"],
                num_heads=kernel_params["num_heads"],
                ffn_hidden_size=kernel_params["ffn_hidden_size"],
                seq_len=seq_len,
                batch_size=batch_size,
                vocab_size=kernel_params["vocab_size"],
                tp=tp,
                dtype=dtype,
                epoch_num=epoch_num,
                swiglu=kernel_params.get("swiglu", False),
                num_experts=kernel_params.get("num_experts", 0),
                ep=ep,
                top_k=kernel_params.get("top_k", 1),
                num_layers=kernel_params.get("num_layers", 0),
                existing_runs=existing_runs,
            )
            # A config is OOM only if ALL kernels failed; partial OOM is normal.
            all_oom = len(runs) == 0 and len(oom_runs) > 0
            if all_oom:
                known_ooms.append((tp, ep, batch_size, seq_len))
            results.append(SweepResult(config=config, runs=runs, oom=all_oom, oom_runs=oom_runs))
        except (RuntimeError, MemoryError) as exc:
            if "out of memory" in str(exc).lower() or isinstance(exc, MemoryError):
                known_ooms.append((tp, ep, batch_size, seq_len))
                oom_runs = _make_oom_kernel_runs(kernel_params, tp, ep, batch_size, seq_len, dtype)
                results.append(SweepResult(config=config, runs=None, oom=True, oom_runs=oom_runs))
            else:
                raise

    return results
