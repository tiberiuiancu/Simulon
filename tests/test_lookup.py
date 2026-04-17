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
    gpu = _gpu(_run("layernorm", {"hidden_size": 4096, "seq_len": 2048, "batch_size": 1, "dtype": "bf16"}, [1.0, 2.0, 3.0]))
    time, extrap = lookup_kernel_time("layernorm", {"hidden_size": 4096, "seq_len": 2048, "batch_size": 1, "dtype": "bf16"}, gpu)
    assert time == statistics.median([1.0, 2.0, 3.0])
    assert extrap is False


def test_no_match_returns_none_on_dtype_mismatch():
    """dtype is never relaxed during extrapolation — mismatching dtype always returns None."""
    gpu = _gpu(_run("layernorm", {"hidden_size": 4096, "seq_len": 512, "batch_size": 1, "dtype": "fp32"}, [1.0]))
    time, extrap = lookup_kernel_time("layernorm", {"hidden_size": 4096, "seq_len": 512, "batch_size": 1, "dtype": "bf16"}, gpu)
    assert time is None
    assert extrap is False


def test_wrong_kernel_returns_none():
    gpu = _gpu(_run("layernorm", {"hidden_size": 4096}, [1.0]))
    time, extrap = lookup_kernel_time("attn_qkv", {"hidden_size": 4096}, gpu)
    assert time is None
    assert extrap is False


def test_empty_gpu_spec_returns_none():
    gpu = _gpu()
    time, extrap = lookup_kernel_time("layernorm", {"hidden_size": 4096}, gpu)
    assert time is None
    assert extrap is False


# ---------------------------------------------------------------------------
# Partial match: query key absent from run.params
# ---------------------------------------------------------------------------


def test_partial_match_when_run_missing_tp():
    """Run has no 'tp' key — should still match on the keys that are present."""
    gpu = _gpu(_run("layernorm", {"hidden_size": 4096, "seq_len": 2048, "batch_size": 1, "dtype": "bf16"}, [2.0, 4.0]))
    time, extrap = lookup_kernel_time(
        "layernorm",
        {"hidden_size": 4096, "seq_len": 2048, "batch_size": 1, "dtype": "bf16", "tp": 2},
        gpu,
    )
    assert time == statistics.median([2.0, 4.0])
    assert extrap is False


def test_partial_match_ignores_extra_run_params():
    """Run has extra params (num_heads) not in match_params — still matches."""
    gpu = _gpu(_run("attn_flash", {"hidden_size": 4096, "num_heads": 32, "seq_len": 2048, "batch_size": 1, "dtype": "bf16", "tp": 1}, [5.0]))
    time, extrap = lookup_kernel_time(
        "attn_flash",
        {"hidden_size": 4096, "seq_len": 2048, "batch_size": 1, "dtype": "bf16", "tp": 1},
        gpu,
    )
    assert time == 5.0
    assert extrap is False


# ---------------------------------------------------------------------------
# Exact beats partial when both match
# ---------------------------------------------------------------------------


def test_exact_preferred_over_partial():
    """When both exact and partial runs exist, exact match is used exclusively."""
    # partial_run is missing 'dtype' so it can only partial-match
    partial_run = _run("layernorm", {"hidden_size": 4096, "seq_len": 2048, "batch_size": 1}, [99.0])
    exact_run = _run("layernorm", {"hidden_size": 4096, "seq_len": 2048, "batch_size": 1, "dtype": "bf16"}, [1.0, 2.0, 3.0])
    gpu = _gpu(partial_run, exact_run)
    time, extrap = lookup_kernel_time(
        "layernorm",
        {"hidden_size": 4096, "seq_len": 2048, "batch_size": 1, "dtype": "bf16"},
        gpu,
    )
    assert time == statistics.median([1.0, 2.0, 3.0])
    assert extrap is False


# ---------------------------------------------------------------------------
# Wrong value in overlapping key → no partial match
# ---------------------------------------------------------------------------


def test_partial_no_match_if_dtype_differs():
    """dtype is not extrapolatable — mismatched dtype never matches."""
    gpu = _gpu(_run("layernorm", {"hidden_size": 4096, "seq_len": 512, "batch_size": 1, "dtype": "fp16"}, [1.0]))
    time, extrap = lookup_kernel_time("layernorm", {"hidden_size": 4096, "seq_len": 512, "batch_size": 1, "dtype": "bf16"}, gpu)
    assert time is None


# ---------------------------------------------------------------------------
# Multiple matching runs pooled together
# ---------------------------------------------------------------------------


def test_multiple_partial_runs_pooled():
    """When multiple partial-match runs exist, their times are pooled before taking median."""
    gpu = _gpu(
        _run("layernorm", {"hidden_size": 4096, "seq_len": 2048, "batch_size": 1, "dtype": "bf16"}, [1.0, 3.0]),
        _run("layernorm", {"hidden_size": 4096, "seq_len": 2048, "batch_size": 1, "dtype": "bf16"}, [5.0, 7.0]),
    )
    time, extrap = lookup_kernel_time("layernorm", {"hidden_size": 4096, "seq_len": 2048, "batch_size": 1, "dtype": "bf16"}, gpu)
    assert time == statistics.median([1.0, 3.0, 5.0, 7.0])
    assert extrap is False


# ---------------------------------------------------------------------------
# Structured extrapolation fallback
# ---------------------------------------------------------------------------


def test_extrapolation_doubles_time_for_double_batch():
    """Requesting 2× the batch_size should return 2× the reference time."""
    gpu = _gpu(_run("mlp_linear1", {"hidden_size": 4096, "ffn_hidden_size": 16384, "tp": 1, "dtype": "bf16", "batch_size": 1, "seq_len": 1024}, [10.0]))
    time, extrap = lookup_kernel_time(
        "mlp_linear1",
        {"hidden_size": 4096, "ffn_hidden_size": 16384, "tp": 1, "dtype": "bf16", "batch_size": 2, "seq_len": 1024},
        gpu,
        warn=False,
    )
    assert time == pytest.approx(20.0)
    assert extrap is True


def test_extrapolation_scales_seq_len():
    """Requesting 2× the seq_len should return 2× the reference time."""
    gpu = _gpu(_run("attn_qkv", {"hidden_size": 4096, "tp": 1, "dtype": "bf16", "batch_size": 1, "seq_len": 1024}, [8.0]))
    time, extrap = lookup_kernel_time(
        "attn_qkv",
        {"hidden_size": 4096, "tp": 1, "dtype": "bf16", "batch_size": 1, "seq_len": 2048},
        gpu,
        warn=False,
    )
    assert time == pytest.approx(16.0)
    assert extrap is True


def test_extrapolation_scales_tp_inversely():
    """Higher TP means less work per GPU — time should halve when TP doubles."""
    gpu = _gpu(_run("attn_qkv", {"hidden_size": 4096, "tp": 2, "dtype": "bf16", "batch_size": 1, "seq_len": 1024}, [8.0]))
    time, extrap = lookup_kernel_time(
        "attn_qkv",
        {"hidden_size": 4096, "tp": 4, "dtype": "bf16", "batch_size": 1, "seq_len": 1024},
        gpu,
        warn=False,
    )
    assert time == pytest.approx(4.0)
    assert extrap is True


def test_extrapolation_emits_warning():
    """A UserWarning is emitted when the extrapolation fallback is used."""
    gpu = _gpu(_run("mlp_linear1", {"hidden_size": 4096, "ffn_hidden_size": 16384, "tp": 1, "dtype": "bf16", "batch_size": 1, "seq_len": 1024}, [10.0]))
    with pytest.warns(UserWarning, match="scaling from"):
        lookup_kernel_time(
            "mlp_linear1",
            {"hidden_size": 4096, "ffn_hidden_size": 16384, "tp": 1, "dtype": "bf16", "batch_size": 2, "seq_len": 1024},
            gpu,
        )


def test_extrapolation_not_used_when_arch_differs():
    """Extrapolation requires matching all other params; different hidden_size → None."""
    gpu = _gpu(_run("mlp_linear1", {"hidden_size": 4096, "ffn_hidden_size": 16384, "tp": 1, "dtype": "bf16", "batch_size": 1, "seq_len": 1024}, [10.0]))
    time, _ = lookup_kernel_time(
        "mlp_linear1",
        {"hidden_size": 8192, "ffn_hidden_size": 16384, "tp": 1, "dtype": "bf16", "batch_size": 2, "seq_len": 1024},
        gpu,
        warn=False,
    )
    assert time is None


def test_exact_preferred_over_extrapolation():
    """An exact match is used even when an extrapolation candidate also exists."""
    gpu = _gpu(
        _run("mlp_linear1", {"hidden_size": 4096, "ffn_hidden_size": 16384, "tp": 1, "dtype": "bf16", "batch_size": 1, "seq_len": 1024}, [10.0]),
        _run("mlp_linear1", {"hidden_size": 4096, "ffn_hidden_size": 16384, "tp": 1, "dtype": "bf16", "batch_size": 1, "seq_len": 2048}, [5.0]),
    )
    time, extrap = lookup_kernel_time(
        "mlp_linear1",
        {"hidden_size": 4096, "ffn_hidden_size": 16384, "tp": 1, "dtype": "bf16", "batch_size": 1, "seq_len": 2048},
        gpu,
        warn=False,
    )
    assert time == pytest.approx(5.0)
    assert extrap is False


# ---------------------------------------------------------------------------
# Per-kernel param filtering
# ---------------------------------------------------------------------------


def test_kernel_params_filtered_by_canonical_keys():
    """Extra params irrelevant to a kernel are ignored during matching."""
    # layernorm does not use vocab_size or tp — those should be filtered out.
    gpu = _gpu(_run("layernorm", {"hidden_size": 4096, "seq_len": 2048, "batch_size": 1, "dtype": "bf16"}, [3.0]))
    time, extrap = lookup_kernel_time(
        "layernorm",
        # Pass extra irrelevant params; should still match exactly.
        {"hidden_size": 4096, "seq_len": 2048, "batch_size": 1, "dtype": "bf16",
         "vocab_size": 32000, "tp": 4, "num_experts": 64},
        gpu,
    )
    assert time == pytest.approx(3.0)
    assert extrap is False


def test_moe_expert_uses_ep_in_matching():
    """moe_expert includes ep in its canonical keys."""
    gpu = _gpu(
        _run("moe_expert", {"hidden_size": 4096, "ffn_hidden_size": 16384, "num_experts": 8, "ep": 4, "top_k": 2,
                             "seq_len": 2048, "batch_size": 1, "dtype": "bf16"}, [20.0]),
        _run("moe_expert", {"hidden_size": 4096, "ffn_hidden_size": 16384, "num_experts": 8, "ep": 8, "top_k": 2,
                             "seq_len": 2048, "batch_size": 1, "dtype": "bf16"}, [10.0]),
    )
    # ep=4 should match the first run, not the second.
    time, extrap = lookup_kernel_time(
        "moe_expert",
        {"hidden_size": 4096, "ffn_hidden_size": 16384, "num_experts": 8, "ep": 4, "top_k": 2,
         "seq_len": 2048, "batch_size": 1, "dtype": "bf16"},
        gpu,
    )
    assert time == pytest.approx(20.0)
    assert extrap is False


# ---------------------------------------------------------------------------
# AdamW lookup: num_params exact match and proportional scaling
# ---------------------------------------------------------------------------


def test_adamw_exact_match_by_num_params():
    """lookup_kernel_time returns median of run matching exact num_params and dtype."""
    gpu = _gpu(_run("adamw", {"num_params": 1_000_000, "dtype": "bf16"}, [3.0, 5.0, 7.0]))
    time, extrap = lookup_kernel_time("adamw", {"num_params": 1_000_000, "dtype": "bf16"}, gpu)
    assert time == pytest.approx(statistics.median([3.0, 5.0, 7.0]))
    assert extrap is False


def test_adamw_exact_match_no_false_hit_on_dtype():
    """lookup_kernel_time does not match when dtype differs."""
    gpu = _gpu(_run("adamw", {"num_params": 1_000_000, "dtype": "fp32"}, [3.0]))
    time, extrap = lookup_kernel_time("adamw", {"num_params": 1_000_000, "dtype": "bf16"}, gpu)
    assert time is None


def test_adamw_proportional_scaling_by_num_params():
    """A run at num_params=500_000 scales linearly to num_params=1_000_000 (factor 2×)."""
    gpu = _gpu(_run("adamw", {"num_params": 500_000, "dtype": "bf16"}, [4.0]))
    time, extrap = lookup_kernel_time(
        "adamw",
        {"num_params": 1_000_000, "dtype": "bf16"},
        gpu,
        warn=False,
    )
    assert time == pytest.approx(8.0)
    assert extrap is True


def test_adamw_proportional_scaling_emits_warning():
    """UserWarning is emitted when num_params scaling fallback is used."""
    gpu = _gpu(_run("adamw", {"num_params": 500_000, "dtype": "bf16"}, [4.0]))
    with pytest.warns(UserWarning, match="num_params"):
        lookup_kernel_time("adamw", {"num_params": 1_000_000, "dtype": "bf16"}, gpu)
