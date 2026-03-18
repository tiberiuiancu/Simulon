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
    """
    from itertools import product

    from simulon.profiling.kernels import benchmark_kernels

    results: list[SweepResult] = []

    for tp, ep, batch_size, seq_len in product(tp_values, ep_values, batch_sizes, seq_lens):
        config = {"tp": tp, "ep": ep, "batch_size": batch_size, "seq_len": seq_len}
        try:
            runs = benchmark_kernels(
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
                existing_runs=existing_runs,
            )
            results.append(SweepResult(config=config, runs=runs, oom=False))
        except (RuntimeError, MemoryError) as exc:
            # Catch CUDA OOM and similar allocation failures
            if "out of memory" in str(exc).lower() or isinstance(exc, MemoryError):
                results.append(SweepResult(config=config, runs=None, oom=True))
            else:
                raise

    return results
