"""Unit tests for simulon.profiling.kernels.benchmark_kernels (no GPU required).

All CUDA calls are patched out so the tests run on CPU-only machines.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from simulon.config.common import DType
from simulon.config.dc import KernelRun
from simulon.profiling.kernels import benchmark_kernels

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_BASE_PARAMS = dict(
    hidden_size=4096,
    num_heads=32,
    ffn_hidden_size=11008,
    seq_len=512,
    batch_size=1,
    vocab_size=32000,
    tp=1,
    dtype=DType.bf16,
    epoch_num=5,
)

# Names of the dense (non-MoE) kernels expected in a full run.
_DENSE_KERNELS = {
    "embedding", "layernorm", "attn_qkv", "attn_flash",
    "attn_proj", "mlp_linear1", "mlp_act", "mlp_linear2", "logit",
}
_MOE_KERNELS = _DENSE_KERNELS | {"moe_route", "moe_expert"}

_FAKE_TIMES = [1.0, 2.0, 3.0, 4.0, 5.0]  # len == 5


def _mock_cuda_time(fn, warmup=3, repeats=10):
    """Replacement for _cuda_time: ignore warmup/repeats, just return fake list."""
    return list(_FAKE_TIMES)


def _patch_cuda(fn):
    """Decorator: patches CUDA availability + timing so no GPU is needed."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with patch("torch.cuda.is_available", return_value=True), \
             patch("simulon.profiling.kernels._cuda_time", side_effect=_mock_cuda_time):
            return fn(*args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Basic full-run tests
# ---------------------------------------------------------------------------


@_patch_cuda
def test_benchmark_kernels_returns_all_dense_kernels():
    runs = benchmark_kernels(**_BASE_PARAMS)
    kernel_names = {r.kernel for r in runs}
    assert kernel_names == _DENSE_KERNELS


@_patch_cuda
def test_benchmark_kernels_returns_moe_kernels_when_num_experts_set():
    runs = benchmark_kernels(**_BASE_PARAMS, num_experts=8, ep=1, top_k=2)
    kernel_names = {r.kernel for r in runs}
    assert kernel_names == _MOE_KERNELS


@_patch_cuda
def test_benchmark_kernels_no_moe_kernels_when_num_experts_zero():
    runs = benchmark_kernels(**_BASE_PARAMS, num_experts=0)
    kernel_names = {r.kernel for r in runs}
    assert "moe_route" not in kernel_names
    assert "moe_expert" not in kernel_names


@_patch_cuda
def test_benchmark_kernels_times_truncated_to_epoch_num():
    params = {**_BASE_PARAMS, "epoch_num": 3}
    runs = benchmark_kernels(**params)
    for r in runs:
        assert len(r.times_ms) == 3


@_patch_cuda
def test_benchmark_kernels_no_cuda_raises():
    # Override the outer patch to say False.
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="CUDA GPU required"):
            benchmark_kernels(**_BASE_PARAMS)


# ---------------------------------------------------------------------------
# Extend / incremental re-run logic (the key feature under test)
# ---------------------------------------------------------------------------


@_patch_cuda
def test_extend_skips_kernels_with_sufficient_timings():
    """If existing_runs already has >= epoch_num times for every kernel,
    benchmark_kernels must return an empty list (nothing to re-profile)."""

    # Build existing_runs with 5 times each (== epoch_num=5).
    existing = _build_existing_runs(epoch_num=5)
    runs = benchmark_kernels(**_BASE_PARAMS, existing_runs=existing)
    assert runs == [], "All kernels already have sufficient timings — nothing should be re-profiled"


@_patch_cuda
def test_extend_reprofiles_kernels_with_insufficient_timings():
    """If existing_runs has fewer times than epoch_num for some kernels,
    those kernels must be re-profiled."""

    # Only 2 times per kernel, but we ask for epoch_num=5.
    existing = _build_existing_runs(epoch_num=2)
    runs = benchmark_kernels(**_BASE_PARAMS, existing_runs=existing)
    kernel_names = {r.kernel for r in runs}
    # Every dense kernel should be re-profiled.
    assert kernel_names == _DENSE_KERNELS


@_patch_cuda
def test_extend_partial_existing_reruns_only_missing():
    """Only kernels absent from existing_runs should be profiled."""

    # Provide sufficient timings for all except layernorm and attn_qkv.
    skip = {"layernorm", "attn_qkv"}
    existing = _build_existing_runs(epoch_num=5, exclude=skip)
    runs = benchmark_kernels(**_BASE_PARAMS, existing_runs=existing)
    kernel_names = {r.kernel for r in runs}
    assert kernel_names == skip


@_patch_cuda
def test_extend_new_config_treated_as_missing():
    """existing_runs entries for tp=1 must not suppress profiling for tp=2
    (different params → different key)."""

    existing = _build_existing_runs(epoch_num=5, tp_override=1)
    # Ask for tp=2: params differ, so all kernels should be profiled.
    runs = benchmark_kernels(**{**_BASE_PARAMS, "tp": 2}, existing_runs=existing)
    assert len(runs) == len(_DENSE_KERNELS)


@_patch_cuda
def test_extend_empty_existing_runs_profiles_everything():
    runs = benchmark_kernels(**_BASE_PARAMS, existing_runs=[])
    assert {r.kernel for r in runs} == _DENSE_KERNELS


@_patch_cuda
def test_extend_none_existing_runs_profiles_everything():
    runs = benchmark_kernels(**_BASE_PARAMS, existing_runs=None)
    assert {r.kernel for r in runs} == _DENSE_KERNELS


# ---------------------------------------------------------------------------
# KernelRun shape / params
# ---------------------------------------------------------------------------


@_patch_cuda
def test_kernel_run_params_include_tp_seq_batch():
    runs = benchmark_kernels(**_BASE_PARAMS)
    for r in runs:
        assert r.params["tp"] == _BASE_PARAMS["tp"]
        assert r.params["seq_len"] == _BASE_PARAMS["seq_len"]
        assert r.params["batch_size"] == _BASE_PARAMS["batch_size"]


@_patch_cuda
def test_kernel_run_is_kernel_run_instance():
    runs = benchmark_kernels(**_BASE_PARAMS)
    for r in runs:
        assert isinstance(r, KernelRun)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _make_run(kernel: str, params: dict, n_times: int) -> dict:
    return {"kernel": kernel, "params": params, "times_ms": list(range(n_times))}


def _build_existing_runs(
    epoch_num: int,
    exclude: set[str] | None = None,
    tp_override: int | None = None,
) -> list[dict]:
    """Construct a fake existing_runs list for all dense kernels."""
    tp = tp_override if tp_override is not None else _BASE_PARAMS["tp"]
    base_params = {
        "hidden_size": _BASE_PARAMS["hidden_size"],
        "seq_len": _BASE_PARAMS["seq_len"],
        "batch_size": _BASE_PARAMS["batch_size"],
        "dtype": "bf16",
        "tp": tp,
    }
    kernel_extras = {
        "embedding": {},
        "layernorm": {},
        "attn_qkv": {},
        "attn_flash": {"num_heads": _BASE_PARAMS["num_heads"]},
        "attn_proj": {},
        "mlp_linear1": {"ffn_hidden_size": _BASE_PARAMS["ffn_hidden_size"]},
        "mlp_act": {"ffn_hidden_size": _BASE_PARAMS["ffn_hidden_size"], "swiglu": False},
        "mlp_linear2": {"ffn_hidden_size": _BASE_PARAMS["ffn_hidden_size"]},
        "logit": {"vocab_size": _BASE_PARAMS["vocab_size"]},
    }
    runs = []
    for kernel, extras in kernel_extras.items():
        if exclude and kernel in exclude:
            continue
        runs.append(_make_run(kernel, {**base_params, **extras}, epoch_num))
    return runs
