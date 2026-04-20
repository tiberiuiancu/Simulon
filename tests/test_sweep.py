"""Unit tests for simulon.profiling.sweep (no GPU required)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from simulon.config.common import DType
from simulon.config.dc import KernelRun
from simulon.profiling.sweep import SweepResult, _inferred_oom, _make_oom_kernel_runs, parse_sweep, run_sweep


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
    assert r.oom_runs == []


def test_sweep_result_oom():
    r = SweepResult(config={"tp": 1, "ep": 1, "batch_size": 1, "seq_len": 512}, runs=None, oom=True)
    assert r.oom
    assert r.runs is None
    assert r.oom_runs == []


def test_sweep_result_with_runs():
    kr = KernelRun(kernel="layernorm", params={"hidden_size": 4096}, times_ms=[1.0, 2.0])
    r = SweepResult(config={"tp": 1}, runs=[kr], oom=False)
    assert not r.oom
    assert len(r.runs) == 1


# ---------------------------------------------------------------------------
# _inferred_oom
# ---------------------------------------------------------------------------


def test_inferred_oom_exact_match():
    assert _inferred_oom(1, 1, 1, 512, [(1, 1, 1, 512)]) is True


def test_inferred_oom_larger_batch():
    """Larger batch with same tp/ep/seq also OOMs."""
    assert _inferred_oom(1, 1, 4, 512, [(1, 1, 1, 512)]) is True


def test_inferred_oom_lower_tp():
    """Lower TP (less sharding) with same other params also OOMs."""
    assert _inferred_oom(1, 1, 1, 512, [(2, 1, 1, 512)]) is True


def test_inferred_oom_higher_tp_is_safe():
    """Higher TP = more sharding = less memory; should NOT be inferred as OOM."""
    assert _inferred_oom(4, 1, 1, 512, [(2, 1, 1, 512)]) is False


def test_inferred_oom_smaller_batch_is_safe():
    """Smaller batch = less memory; should NOT be inferred as OOM."""
    assert _inferred_oom(1, 1, 1, 512, [(1, 1, 4, 512)]) is False


def test_inferred_oom_empty_known_ooms():
    assert _inferred_oom(1, 1, 1, 512, []) is False


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


def test_run_sweep_oom_infers_dominated_configs():
    """When bs=1 OOMs, bs=2 (same tp/ep/seq) is inferred OOM without running."""
    call_count = 0

    def _side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise RuntimeError("out of memory")

    with patch("simulon.profiling.kernels.benchmark_kernels", side_effect=_side_effect):
        results = run_sweep(_KERNEL_PARAMS, [1], [1], [1, 2], [512], DType.bf16)

    assert len(results) == 2
    assert all(r.oom for r in results)
    # benchmark_kernels should only be called once (bs=1); bs=2 is inferred
    assert call_count == 1


def test_run_sweep_oom_does_not_block_better_configs():
    """When bs=2 OOMs, bs=1 (smaller, less memory) is NOT inferred OOM and still runs."""
    call_count = 0

    def _side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        bs = kwargs.get("batch_size", 1)
        if bs == 2:
            raise RuntimeError("out of memory")
        return _FAKE_RUNS

    with patch("simulon.profiling.kernels.benchmark_kernels", side_effect=_side_effect):
        # Sorted order: bs=1 first (best), bs=2 second.
        # bs=1 succeeds; bs=2 OOMs → but bs=1 was already done, nothing to infer.
        results = run_sweep(_KERNEL_PARAMS, [1], [1], [1, 2], [512], DType.bf16)

    assert len(results) == 2
    configs_oom = {r.config["batch_size"]: r.oom for r in results}
    assert configs_oom[1] is False   # bs=1 ran and succeeded
    assert configs_oom[2] is True    # bs=2 OOMed
    assert call_count == 2


def test_run_sweep_higher_tp_runs_after_lower_tp_ooms():
    """If tp=1 OOMs, tp=2 (more sharding, less memory) still runs."""
    def _side_effect(*args, **kwargs):
        tp = kwargs.get("tp", 1)
        if tp == 1:
            raise RuntimeError("out of memory")
        return _FAKE_RUNS

    with patch("simulon.profiling.kernels.benchmark_kernels", side_effect=_side_effect):
        results = run_sweep(_KERNEL_PARAMS, [1, 2], [1], [1], [512], DType.bf16)

    configs_oom = {r.config["tp"]: r.oom for r in results}
    assert configs_oom[2] is False   # tp=2 ran fine
    assert configs_oom[1] is True    # tp=1 OOMed


# ---------------------------------------------------------------------------
# _make_oom_kernel_runs
# ---------------------------------------------------------------------------


def test_make_oom_kernel_runs_dense_model():
    """Dense model produces OOM entries for all non-MoE kernels."""
    from simulon.profiling.lookup import KERNEL_MATCH_KEYS
    runs = _make_oom_kernel_runs(_KERNEL_PARAMS, tp=1, ep=1, batch_size=1, seq_len=512, dtype=DType.bf16)
    kernels = {r.kernel for r in runs}
    # All dense kernels should be present; MoE kernels require num_experts
    assert "layernorm" in kernels
    assert "attn_qkv" in kernels
    assert "mlp_linear1" in kernels
    # MoE kernels absent when num_experts not in kernel_params
    assert "moe_expert" not in kernels
    assert "moe_route" not in kernels
    # All entries have empty times_ms
    assert all(r.times_ms == [] for r in runs)


def test_make_oom_kernel_runs_moe_model():
    """MoE model produces OOM entries including moe_expert and moe_route."""
    moe_params = {**_KERNEL_PARAMS, "num_experts": 8, "top_k": 2}
    runs = _make_oom_kernel_runs(moe_params, tp=1, ep=4, batch_size=1, seq_len=512, dtype=DType.bf16)
    kernels = {r.kernel for r in runs}
    assert "moe_expert" in kernels
    assert "moe_route" in kernels
    moe_expert_run = next(r for r in runs if r.kernel == "moe_expert")
    assert moe_expert_run.params["ep"] == 4
    assert moe_expert_run.params["num_experts"] == 8
    assert moe_expert_run.params["top_k"] == 2


def test_run_sweep_oom_populates_oom_runs():
    """OOM results must have oom_runs populated with per-kernel entries."""
    def _raise_oom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory.")

    with patch("simulon.profiling.kernels.benchmark_kernels", side_effect=_raise_oom):
        results = run_sweep(_KERNEL_PARAMS, [1], [1], [1], [512], DType.bf16)

    assert results[0].oom
    assert len(results[0].oom_runs) > 0
    kernels = {r.kernel for r in results[0].oom_runs}
    assert "layernorm" in kernels
    assert "attn_qkv" in kernels


def test_run_sweep_inferred_oom_also_has_oom_runs():
    """Inferred OOM configs must also have oom_runs (generated without running)."""
    call_count = 0

    def _side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise RuntimeError("out of memory")

    with patch("simulon.profiling.kernels.benchmark_kernels", side_effect=_side_effect):
        results = run_sweep(_KERNEL_PARAMS, [1], [1], [1, 2], [512], DType.bf16)

    inferred = next(r for r in results if r.config["batch_size"] == 2)
    assert inferred.oom
    assert len(inferred.oom_runs) > 0
    assert call_count == 1  # bs=2 was not actually run


def test_run_sweep_sorted_order_tries_best_config_first():
    """Configs are tried with highest TP/EP and smallest BS/SL first."""
    call_order = []

    def _side_effect(*args, **kwargs):
        call_order.append((kwargs["tp"], kwargs["ep"], kwargs["batch_size"], kwargs["seq_len"]))
        return _FAKE_RUNS

    with patch("simulon.profiling.kernels.benchmark_kernels", side_effect=_side_effect):
        run_sweep(_KERNEL_PARAMS, [1, 2], [1], [1, 2], [512, 1024], DType.bf16)

    # First config tried should have highest TP, smallest BS/SL
    assert call_order[0] == (2, 1, 1, 512)
    # Last config tried should have lowest TP, largest BS/SL
    assert call_order[-1] == (1, 1, 2, 1024)
