"""Unit tests for kernel time lookup."""

import statistics

import pytest

from simulon.config.dc import GPUSpec, KernelRun
from simulon.profiling.lookup import lookup_kernel_time
import simulon.profiling.lookup as _lookup_module


@pytest.fixture(autouse=True)
def clear_lookup_cache():
    """Clear the in-process lookup cache before each test to avoid id() reuse false hits."""
    _lookup_module._cache.clear()
    yield
    _lookup_module._cache.clear()


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


# ---------------------------------------------------------------------------
# Proportional scaling fallback
# ---------------------------------------------------------------------------


def test_scaling_fallback_doubles_time_for_double_tokens():
    """Requesting 2× the tokens should return 2× the reference time."""
    gpu = _gpu(_run("mlp_fc1", {"hidden_size": 4096, "tp": 1, "dtype": "bf16", "batch_size": 1, "seq_len": 1024}, [10.0]))
    result = lookup_kernel_time(
        "mlp_fc1",
        {"hidden_size": 4096, "tp": 1, "dtype": "bf16", "batch_size": 1, "seq_len": 2048},
        gpu,
        warn=False,
    )
    assert result == pytest.approx(20.0)


def test_scaling_fallback_batch_and_seq():
    """Scaling applies across both batch_size and seq_len dimensions."""
    gpu = _gpu(_run("mlp_fc1", {"hidden_size": 4096, "tp": 1, "dtype": "bf16", "batch_size": 2, "seq_len": 512}, [8.0]))
    result = lookup_kernel_time(
        "mlp_fc1",
        {"hidden_size": 4096, "tp": 1, "dtype": "bf16", "batch_size": 4, "seq_len": 1024},
        gpu,
        warn=False,
    )
    # scale = (4 * 1024) / (2 * 512) = 4.0
    assert result == pytest.approx(32.0)


def test_scaling_fallback_emits_warning():
    """A UserWarning is emitted when the scaling fallback is used."""
    gpu = _gpu(_run("mlp_fc1", {"hidden_size": 4096, "tp": 1, "dtype": "bf16", "batch_size": 1, "seq_len": 1024}, [10.0]))
    with pytest.warns(UserWarning, match="scaling from"):
        lookup_kernel_time(
            "mlp_fc1",
            {"hidden_size": 4096, "tp": 1, "dtype": "bf16", "batch_size": 1, "seq_len": 2048},
            gpu,
        )


def test_scaling_fallback_not_used_when_arch_differs():
    """Scaling fallback requires matching arch params; different hidden_size → None."""
    gpu = _gpu(_run("mlp_fc1", {"hidden_size": 4096, "tp": 1, "dtype": "bf16", "batch_size": 1, "seq_len": 1024}, [10.0]))
    result = lookup_kernel_time(
        "mlp_fc1",
        {"hidden_size": 8192, "tp": 1, "dtype": "bf16", "batch_size": 1, "seq_len": 2048},
        gpu,
        warn=False,
    )
    assert result is None


def test_exact_preferred_over_scaling_fallback():
    """An exact match is used even when a scaling candidate also exists."""
    gpu = _gpu(
        _run("mlp_fc1", {"hidden_size": 4096, "tp": 1, "dtype": "bf16", "batch_size": 1, "seq_len": 1024}, [10.0]),
        _run("mlp_fc1", {"hidden_size": 4096, "tp": 1, "dtype": "bf16", "batch_size": 1, "seq_len": 2048}, [5.0]),
    )
    result = lookup_kernel_time(
        "mlp_fc1",
        {"hidden_size": 4096, "tp": 1, "dtype": "bf16", "batch_size": 1, "seq_len": 2048},
        gpu,
        warn=False,
    )
    assert result == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# AdamW lookup: num_params exact match and proportional scaling
# ---------------------------------------------------------------------------


def test_adamw_exact_match_by_num_params():
    """lookup_kernel_time returns median of run matching exact num_params and dtype."""
    gpu = _gpu(_run("adamw", {"num_params": 1_000_000, "dtype": "bf16"}, [3.0, 5.0, 7.0]))
    result = lookup_kernel_time("adamw", {"num_params": 1_000_000, "dtype": "bf16"}, gpu)
    assert result == pytest.approx(statistics.median([3.0, 5.0, 7.0]))


def test_adamw_exact_match_no_false_hit_on_dtype():
    """lookup_kernel_time does not match when dtype differs."""
    gpu = _gpu(_run("adamw", {"num_params": 1_000_000, "dtype": "fp32"}, [3.0]))
    result = lookup_kernel_time("adamw", {"num_params": 1_000_000, "dtype": "bf16"}, gpu)
    # dtype differs → no exact or partial match (both keys present, one value differs)
    assert result is None


def test_adamw_proportional_scaling_by_num_params():
    """A run at num_params=500_000 scales linearly to num_params=1_000_000 (factor 2×)."""
    gpu = _gpu(_run("adamw", {"num_params": 500_000, "dtype": "bf16"}, [4.0]))
    result = lookup_kernel_time(
        "adamw",
        {"num_params": 1_000_000, "dtype": "bf16"},
        gpu,
        warn=False,
    )
    assert result == pytest.approx(8.0)


def test_adamw_proportional_scaling_emits_warning():
    """UserWarning is emitted when num_params scaling fallback is used."""
    gpu = _gpu(_run("adamw", {"num_params": 500_000, "dtype": "bf16"}, [4.0]))
    with pytest.warns(UserWarning, match="num_params"):
        lookup_kernel_time("adamw", {"num_params": 1_000_000, "dtype": "bf16"}, gpu)
