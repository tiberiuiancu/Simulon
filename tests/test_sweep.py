"""Unit tests for simulon.profiling.sweep (no GPU required)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from simulon.config.common import DType
from simulon.config.dc import KernelRun
from simulon.profiling.sweep import SweepResult, parse_sweep, run_sweep


# ---------------------------------------------------------------------------
# parse_sweep
# ---------------------------------------------------------------------------


def test_parse_sweep_single():
    assert parse_sweep("1") == [1]


def test_parse_sweep_multiple():
    assert parse_sweep("1,2,4") == [1, 2, 4]


def test_parse_sweep_spaces():
    assert parse_sweep("8, 16, 32") == [8, 16, 32]


def test_parse_sweep_single_large():
    assert parse_sweep("128") == [128]


# ---------------------------------------------------------------------------
# SweepResult
# ---------------------------------------------------------------------------


def test_sweep_result_defaults():
    r = SweepResult(config={"tp": 1})
    assert r.runs is None
    assert r.oom is False


def test_sweep_result_oom():
    r = SweepResult(config={"tp": 1, "ep": 1, "batch_size": 1, "seq_len": 512}, runs=None, oom=True)
    assert r.oom
    assert r.runs is None


def test_sweep_result_with_runs():
    kr = KernelRun(kernel="layernorm", params={"hidden_size": 4096}, times_ms=[1.0, 2.0])
    r = SweepResult(config={"tp": 1}, runs=[kr], oom=False)
    assert not r.oom
    assert len(r.runs) == 1


# ---------------------------------------------------------------------------
# run_sweep (mocked benchmark_kernels)
# ---------------------------------------------------------------------------

_KERNEL_PARAMS = {
    "hidden_size": 4096,
    "num_heads": 32,
    "ffn_hidden_size": 16384,
    "vocab_size": 32000,
}

_FAKE_RUNS = [
    KernelRun(kernel="layernorm", params={"hidden_size": 4096}, times_ms=[1.0, 2.0]),
    KernelRun(kernel="attn_qkv", params={"hidden_size": 4096}, times_ms=[3.0, 4.0]),
]


def _patch_benchmark(return_value=_FAKE_RUNS):
    return patch(
        "simulon.profiling.kernels.benchmark_kernels",
        return_value=return_value,
    )


def test_run_sweep_single_config():
    with _patch_benchmark() as mock_bench:
        results = run_sweep(_KERNEL_PARAMS, [1], [1], [1], [512], DType.bf16)

    assert len(results) == 1
    assert not results[0].oom
    assert results[0].runs == _FAKE_RUNS
    assert results[0].config == {"tp": 1, "ep": 1, "batch_size": 1, "seq_len": 512}
    mock_bench.assert_called_once()


def test_run_sweep_cartesian_product():
    with _patch_benchmark():
        results = run_sweep(_KERNEL_PARAMS, [1, 2], [1], [1, 2], [512], DType.bf16)

    # 2 tp * 1 ep * 2 batch * 1 seq = 4
    assert len(results) == 4
    configs = [r.config for r in results]
    assert {"tp": 1, "ep": 1, "batch_size": 1, "seq_len": 512} in configs
    assert {"tp": 2, "ep": 1, "batch_size": 2, "seq_len": 512} in configs


def test_run_sweep_oom_caught():
    def _raise_oom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")

    with patch("simulon.profiling.kernels.benchmark_kernels", side_effect=_raise_oom):
        results = run_sweep(_KERNEL_PARAMS, [1], [1], [1], [512], DType.bf16)

    assert len(results) == 1
    assert results[0].oom
    assert results[0].runs is None


def test_run_sweep_non_oom_runtime_error_propagates():
    def _raise_other(*args, **kwargs):
        raise RuntimeError("CUDA error: device-side assert triggered")

    with patch("simulon.profiling.kernels.benchmark_kernels", side_effect=_raise_other):
        with pytest.raises(RuntimeError, match="device-side assert"):
            run_sweep(_KERNEL_PARAMS, [1], [1], [1], [512], DType.bf16)


def test_run_sweep_passes_optional_params():
    moe_params = {**_KERNEL_PARAMS, "num_experts": 8, "top_k": 2, "swiglu": True}
    with _patch_benchmark() as mock_bench:
        run_sweep(moe_params, [1], [2], [1], [512], DType.bf16, epoch_num=5)

    call_kwargs = mock_bench.call_args.kwargs
    assert call_kwargs["num_experts"] == 8
    assert call_kwargs["top_k"] == 2
    assert call_kwargs["swiglu"] is True
    assert call_kwargs["ep"] == 2
    assert call_kwargs["epoch_num"] == 5


def test_run_sweep_dtype_passed_through():
    with _patch_benchmark() as mock_bench:
        run_sweep(_KERNEL_PARAMS, [1], [1], [1], [512], DType.fp16)

    assert mock_bench.call_args.kwargs["dtype"] == DType.fp16


def test_run_sweep_partial_oom_continues():
    """OOM on one config doesn't stop the sweep; remaining configs still run."""
    call_count = 0

    def _side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("out of memory")
        return _FAKE_RUNS

    with patch("simulon.profiling.kernels.benchmark_kernels", side_effect=_side_effect):
        results = run_sweep(_KERNEL_PARAMS, [1], [1], [1, 2], [512], DType.bf16)

    assert len(results) == 2
    assert results[0].oom
    assert not results[1].oom
