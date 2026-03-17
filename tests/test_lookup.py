"""Unit tests for kernel time lookup."""

import statistics

import pytest

from simulon.config.dc import GPUSpec, KernelRun
from simulon.profiling.lookup import lookup_kernel_time


def _gpu(*runs: KernelRun) -> GPUSpec:
    return GPUSpec(name="test", kernel_runs=list(runs))


def _run(kernel, params, times_ms):
    return KernelRun(kernel=kernel, params=params, times_ms=times_ms)


# ---------------------------------------------------------------------------
# Basic exact match
# ---------------------------------------------------------------------------


def test_exact_match_returns_median():
    gpu = _gpu(_run("layernorm", {"hidden_size": 4096, "seq_len": 2048, "batch_size": 1, "dtype": "bf16", "tp": 1}, [1.0, 2.0, 3.0]))
    result = lookup_kernel_time("layernorm", {"hidden_size": 4096, "seq_len": 2048, "batch_size": 1, "dtype": "bf16", "tp": 1}, gpu)
    assert result == statistics.median([1.0, 2.0, 3.0])


def test_no_match_returns_none():
    gpu = _gpu(_run("layernorm", {"hidden_size": 4096}, [1.0]))
    result = lookup_kernel_time("layernorm", {"hidden_size": 8192}, gpu)
    assert result is None


def test_wrong_kernel_returns_none():
    gpu = _gpu(_run("layernorm", {"hidden_size": 4096}, [1.0]))
    result = lookup_kernel_time("attn_qkv", {"hidden_size": 4096}, gpu)
    assert result is None


def test_empty_gpu_spec_returns_none():
    gpu = _gpu()
    result = lookup_kernel_time("layernorm", {"hidden_size": 4096}, gpu)
    assert result is None


# ---------------------------------------------------------------------------
# Partial match: query key absent from run.params
# ---------------------------------------------------------------------------


def test_partial_match_when_run_missing_tp():
    """Run has no 'tp' key — should still match on the keys that are present."""
    gpu = _gpu(_run("layernorm", {"hidden_size": 4096, "seq_len": 2048, "batch_size": 1, "dtype": "bf16"}, [2.0, 4.0]))
    result = lookup_kernel_time(
        "layernorm",
        {"hidden_size": 4096, "seq_len": 2048, "batch_size": 1, "dtype": "bf16", "tp": 2},
        gpu,
    )
    assert result == statistics.median([2.0, 4.0])


def test_partial_match_ignores_extra_run_params():
    """Run has extra params (num_heads) not in match_params — still matches."""
    gpu = _gpu(_run("attn_flash", {"hidden_size": 4096, "num_heads": 32, "seq_len": 2048, "batch_size": 1, "dtype": "bf16"}, [5.0]))
    result = lookup_kernel_time(
        "attn_flash",
        {"hidden_size": 4096, "seq_len": 2048, "batch_size": 1, "dtype": "bf16"},
        gpu,
    )
    assert result == 5.0


# ---------------------------------------------------------------------------
# Exact beats partial when both match
# ---------------------------------------------------------------------------


def test_exact_preferred_over_partial():
    """When both exact and partial runs exist, exact match is used."""
    partial_run = _run("layernorm", {"hidden_size": 4096, "seq_len": 2048, "batch_size": 1, "dtype": "bf16"}, [99.0])
    exact_run = _run("layernorm", {"hidden_size": 4096, "seq_len": 2048, "batch_size": 1, "dtype": "bf16", "tp": 2}, [1.0, 2.0, 3.0])
    gpu = _gpu(partial_run, exact_run)
    result = lookup_kernel_time(
        "layernorm",
        {"hidden_size": 4096, "seq_len": 2048, "batch_size": 1, "dtype": "bf16", "tp": 2},
        gpu,
    )
    assert result == statistics.median([1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Wrong value in overlapping key → no partial match
# ---------------------------------------------------------------------------


def test_partial_no_match_if_shared_key_differs():
    """If a key is present in both query and run but values differ, no match."""
    gpu = _gpu(_run("layernorm", {"hidden_size": 8192, "dtype": "bf16"}, [1.0]))
    result = lookup_kernel_time("layernorm", {"hidden_size": 4096, "dtype": "bf16"}, gpu)
    assert result is None


# ---------------------------------------------------------------------------
# Multiple matching runs pooled together
# ---------------------------------------------------------------------------


def test_multiple_partial_runs_pooled():
    """When multiple partial-match runs exist, their times are pooled before taking median."""
    gpu = _gpu(
        _run("layernorm", {"hidden_size": 4096, "seq_len": 2048}, [1.0, 3.0]),
        _run("layernorm", {"hidden_size": 4096, "seq_len": 2048}, [5.0, 7.0]),
    )
    result = lookup_kernel_time("layernorm", {"hidden_size": 4096, "seq_len": 2048, "tp": 1}, gpu)
    assert result == statistics.median([1.0, 3.0, 5.0, 7.0])
