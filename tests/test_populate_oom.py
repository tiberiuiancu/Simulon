"""Unit tests for _handle_missing OOM logic in populate.py.

These tests exercise the OOM-detection path in isolation, without
building a full DAG (no GPU required, no tracing).
"""

from __future__ import annotations

import warnings

import pytest

from simulon.config.dc import GPUSpec, KernelRun
from simulon.backend.dag.populate import _handle_missing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gpu(kernel_runs=None, oom_kernel_runs=None) -> GPUSpec:
    return GPUSpec(
        name="test",
        kernel_runs=kernel_runs or [],
        oom_kernel_runs=oom_kernel_runs or [],
    )


def _oom_run(kernel: str, params: dict) -> KernelRun:
    return KernelRun(kernel=kernel, params=params, times_ms=[])


# ---------------------------------------------------------------------------
# _handle_missing: OOM path
# ---------------------------------------------------------------------------


def test_handle_missing_raises_runtime_error_on_oom():
    """When kernel+params matches a known OOM entry, RuntimeError is raised by default."""
    oom_params = {"hidden_size": 4096, "seq_len": 512, "batch_size": 1, "dtype": "bf16"}
    gpu = _gpu(oom_kernel_runs=[_oom_run("layernorm", oom_params)])
    with pytest.raises(RuntimeError, match="OOM"):
        _handle_missing("layernorm", oom_params, gpu, ignore_oom=False)


def test_handle_missing_ignores_oom_when_flag_set():
    """ignore_oom=True suppresses the RuntimeError; no exception is raised and no warning emitted."""
    oom_params = {"hidden_size": 4096, "seq_len": 512, "batch_size": 1, "dtype": "bf16"}
    gpu = _gpu(oom_kernel_runs=[_oom_run("layernorm", oom_params)])
    # Should not raise, and should not emit a warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _handle_missing("layernorm", oom_params, gpu, ignore_oom=True)


def test_handle_missing_oom_error_message_mentions_ignore_oom():
    """The RuntimeError message hints at the --ignore-oom flag."""
    oom_params = {"hidden_size": 4096, "seq_len": 512, "batch_size": 1, "dtype": "bf16"}
    gpu = _gpu(oom_kernel_runs=[_oom_run("layernorm", oom_params)])
    with pytest.raises(RuntimeError, match="--ignore-oom"):
        _handle_missing("layernorm", oom_params, gpu, ignore_oom=False)


# ---------------------------------------------------------------------------
# _handle_missing: missing data (non-OOM) path
# ---------------------------------------------------------------------------


def test_handle_missing_emits_user_warning_when_no_data():
    """When no OOM entry matches and no profiling data exists, a UserWarning is emitted."""
    gpu = _gpu()  # empty — no runs, no OOM entries
    with pytest.warns(UserWarning, match="No profiling data"):
        _handle_missing("layernorm", {"hidden_size": 4096, "seq_len": 512, "batch_size": 1, "dtype": "bf16"}, gpu, ignore_oom=False)


def test_handle_missing_no_error_when_no_data_no_oom():
    """With no OOM entry and no data, _handle_missing only warns — does not raise."""
    gpu = _gpu()
    # Should not raise; should only warn.
    with pytest.warns(UserWarning):
        _handle_missing("attn_qkv", {"hidden_size": 4096, "seq_len": 512, "batch_size": 1, "dtype": "bf16"}, gpu, ignore_oom=False)


def test_handle_missing_oom_takes_priority_over_missing_data():
    """When the OOM entry matches, RuntimeError is raised even though kernel_runs is also empty."""
    oom_params = {"hidden_size": 4096, "seq_len": 512, "batch_size": 1, "dtype": "bf16"}
    gpu = _gpu(kernel_runs=[], oom_kernel_runs=[_oom_run("layernorm", oom_params)])
    # Should raise RuntimeError, not emit a UserWarning.
    with pytest.raises(RuntimeError):
        _handle_missing("layernorm", oom_params, gpu, ignore_oom=False)


def test_handle_missing_wrong_kernel_not_oom():
    """OOM entry for a different kernel does not trigger RuntimeError for the queried kernel."""
    oom_params = {"hidden_size": 4096, "seq_len": 512, "batch_size": 1, "dtype": "bf16"}
    gpu = _gpu(oom_kernel_runs=[_oom_run("attn_qkv", oom_params)])
    # layernorm is not in OOM list → should warn, not raise.
    with pytest.warns(UserWarning):
        _handle_missing("layernorm", oom_params, gpu, ignore_oom=False)


def test_handle_missing_oom_extra_params_stripped_via_canonical_keys():
    """_handle_missing strips non-canonical params before OOM matching (same as is_kernel_oom)."""
    oom_params = {"hidden_size": 4096, "seq_len": 512, "batch_size": 1, "dtype": "bf16"}
    gpu = _gpu(oom_kernel_runs=[_oom_run("layernorm", oom_params)])
    # Query includes extra tp and vocab_size — canonical filter should strip them.
    with pytest.raises(RuntimeError, match="OOM"):
        _handle_missing(
            "layernorm",
            {"hidden_size": 4096, "seq_len": 512, "batch_size": 1, "dtype": "bf16",
             "tp": 2, "vocab_size": 32000},
            gpu,
            ignore_oom=False,
        )
