"""Unit tests for the `simulon profile gpu` CLI command (no GPU required).

benchmark_kernels / run_sweep are patched so no real hardware is touched.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from typer.testing import CliRunner

from simulon.cli import app
from simulon.config.dc import KernelRun
from simulon.profiling.sweep import SweepResult

runner = CliRunner()

# ---------------------------------------------------------------------------
# Fake data shared across tests
# ---------------------------------------------------------------------------

_FAKE_RUN = KernelRun(
    kernel="layernorm",
    params={"hidden_size": 4096, "seq_len": 512, "batch_size": 1, "dtype": "bf16", "tp": 1},
    times_ms=[1.0, 2.0, 3.0, 4.0, 5.0],
)
_FAKE_RESULT = SweepResult(config={"tp": 1, "ep": 1, "batch_size": 1, "seq_len": 512}, runs=[_FAKE_RUN], oom=False)

_ARCH_ARGS = [
    "--hidden-size", "4096",
    "--num-heads", "32",
    "--ffn-hidden-size", "11008",
    "--vocab-size", "32000",
]


def _patch_sweep(return_value=None):
    rv = return_value if return_value is not None else [_FAKE_RESULT]
    return patch("simulon.profiling.sweep.run_sweep", return_value=rv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_profile(tmp_path: Path, extra_args: list[str], sweep_rv=None) -> tuple:
    out_file = tmp_path / "gpu.yaml"
    base_args = [
        "profile", "gpu",
        "--name", "TestGPU",
        "--output", str(out_file),
        "--seq-len", "512",
    ] + _ARCH_ARGS + extra_args

    ctx = _patch_sweep(sweep_rv) if sweep_rv is not None else _patch_sweep()
    with ctx:
        result = runner.invoke(app, base_args)
    return result, out_file


# ---------------------------------------------------------------------------
# Basic invocation
# ---------------------------------------------------------------------------


def test_profile_creates_output_file(tmp_path):
    result, out_file = _run_profile(tmp_path, [])
    assert result.exit_code == 0, result.output
    assert out_file.exists()


def test_profile_output_contains_kernel_runs(tmp_path):
    _, out_file = _run_profile(tmp_path, [])
    data = yaml.safe_load(out_file.read_text())
    assert "kernel_runs" in data
    assert len(data["kernel_runs"]) >= 1


def test_profile_output_yaml_has_gpu_name(tmp_path):
    _, out_file = _run_profile(tmp_path, [])
    data = yaml.safe_load(out_file.read_text())
    assert data["name"] == "TestGPU"


# ---------------------------------------------------------------------------
# Extend: re-running appends new configs, skips existing sufficient ones
# ---------------------------------------------------------------------------


def test_extend_appends_new_runs(tmp_path):
    """Second invocation with a different config appends new kernel_runs."""
    out_file = tmp_path / "gpu.yaml"
    base = ["profile", "gpu", "--name", "TestGPU", "--output", str(out_file), "--seq-len", "512"] + _ARCH_ARGS

    run1 = KernelRun(
        kernel="layernorm",
        params={"hidden_size": 4096, "seq_len": 512, "batch_size": 1, "dtype": "bf16", "tp": 1},
        times_ms=[1.0] * 5,
    )
    run2 = KernelRun(
        kernel="layernorm",
        params={"hidden_size": 4096, "seq_len": 512, "batch_size": 2, "dtype": "bf16", "tp": 1},
        times_ms=[2.0] * 5,
    )

    with _patch_sweep([SweepResult(config={"tp": 1, "ep": 1, "batch_size": 1, "seq_len": 512}, runs=[run1], oom=False)]):
        runner.invoke(app, base + ["--batch-size", "1"])

    with _patch_sweep([SweepResult(config={"tp": 1, "ep": 1, "batch_size": 2, "seq_len": 512}, runs=[run2], oom=False)]):
        runner.invoke(app, base + ["--batch-size", "2"])

    data = yaml.safe_load(out_file.read_text())
    assert len(data["kernel_runs"]) == 2


def test_extend_does_not_duplicate_existing_run(tmp_path):
    """Re-running with the same config replaces, not duplicates, the entry."""
    out_file = tmp_path / "gpu.yaml"
    base = ["profile", "gpu", "--name", "TestGPU", "--output", str(out_file), "--seq-len", "512"] + _ARCH_ARGS

    run = KernelRun(
        kernel="layernorm",
        params={"hidden_size": 4096, "seq_len": 512, "batch_size": 1, "dtype": "bf16", "tp": 1},
        times_ms=[1.0] * 5,
    )
    result_obj = SweepResult(config={"tp": 1, "ep": 1, "batch_size": 1, "seq_len": 512}, runs=[run], oom=False)

    with _patch_sweep([result_obj]):
        runner.invoke(app, base)
    with _patch_sweep([result_obj]):
        runner.invoke(app, base)

    data = yaml.safe_load(out_file.read_text())
    layernorm_entries = [r for r in data["kernel_runs"] if r["kernel"] == "layernorm"]
    assert len(layernorm_entries) == 1, "Duplicate kernel entries should not be created"


# ---------------------------------------------------------------------------
# --purge
# ---------------------------------------------------------------------------


def test_purge_clears_existing_runs(tmp_path):
    """--purge should remove all kernel_runs present before the new profiling."""
    out_file = tmp_path / "gpu.yaml"
    base = ["profile", "gpu", "--name", "TestGPU", "--output", str(out_file), "--seq-len", "512"] + _ARCH_ARGS

    # First run: write 2 kernel entries.
    run_a = KernelRun(kernel="layernorm", params={"hidden_size": 4096, "seq_len": 512, "batch_size": 1, "dtype": "bf16", "tp": 1}, times_ms=[1.0] * 5)
    run_b = KernelRun(kernel="attn_qkv",  params={"hidden_size": 4096, "seq_len": 512, "batch_size": 1, "dtype": "bf16", "tp": 1}, times_ms=[2.0] * 5)
    with _patch_sweep([SweepResult(config={"tp": 1, "ep": 1, "batch_size": 1, "seq_len": 512}, runs=[run_a, run_b], oom=False)]):
        runner.invoke(app, base)

    data = yaml.safe_load(out_file.read_text())
    assert len(data["kernel_runs"]) == 2

    # Second run with --purge: only one new entry.
    run_c = KernelRun(kernel="layernorm", params={"hidden_size": 4096, "seq_len": 512, "batch_size": 1, "dtype": "bf16", "tp": 1}, times_ms=[9.0] * 5)
    with _patch_sweep([SweepResult(config={"tp": 1, "ep": 1, "batch_size": 1, "seq_len": 512}, runs=[run_c], oom=False)]):
        runner.invoke(app, base + ["--purge"])

    data = yaml.safe_load(out_file.read_text())
    assert len(data["kernel_runs"]) == 1
    assert data["kernel_runs"][0]["kernel"] == "layernorm"


def test_purge_on_new_file_is_harmless(tmp_path):
    """--purge on a file that doesn't exist yet should behave like a normal run."""
    result, out_file = _run_profile(tmp_path, ["--purge"])
    assert result.exit_code == 0
    assert out_file.exists()


# ---------------------------------------------------------------------------
# --overwrite
# ---------------------------------------------------------------------------


def test_overwrite_passes_empty_existing_runs_to_sweep(tmp_path):
    """--overwrite means run_sweep receives no existing_runs (forces re-profile)."""
    out_file = tmp_path / "gpu.yaml"
    base = ["profile", "gpu", "--name", "TestGPU", "--output", str(out_file), "--seq-len", "512"] + _ARCH_ARGS

    # First run writes data.
    with _patch_sweep():
        runner.invoke(app, base)

    # Second run with --overwrite: capture what run_sweep receives.
    captured_kwargs = {}

    def _fake_run_sweep(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return [_FAKE_RESULT]

    with patch("simulon.profiling.sweep.run_sweep", side_effect=_fake_run_sweep):
        runner.invoke(app, base + ["--overwrite"])

    # With --overwrite, runs_for_skip is [] so existing_runs passed to run_sweep is empty.
    assert captured_kwargs.get("existing_runs", []) == []


def test_overwrite_replaces_existing_kernel_run(tmp_path):
    """--overwrite should result in updated timings for the same kernel+params."""
    out_file = tmp_path / "gpu.yaml"
    base = ["profile", "gpu", "--name", "TestGPU", "--output", str(out_file), "--seq-len", "512"] + _ARCH_ARGS

    run_old = KernelRun(kernel="layernorm", params={"hidden_size": 4096, "seq_len": 512, "batch_size": 1, "dtype": "bf16", "tp": 1}, times_ms=[1.0] * 5)
    run_new = KernelRun(kernel="layernorm", params={"hidden_size": 4096, "seq_len": 512, "batch_size": 1, "dtype": "bf16", "tp": 1}, times_ms=[99.0] * 5)

    with _patch_sweep([SweepResult(config={"tp": 1, "ep": 1, "batch_size": 1, "seq_len": 512}, runs=[run_old], oom=False)]):
        runner.invoke(app, base)

    with _patch_sweep([SweepResult(config={"tp": 1, "ep": 1, "batch_size": 1, "seq_len": 512}, runs=[run_new], oom=False)]):
        runner.invoke(app, base + ["--overwrite"])

    data = yaml.safe_load(out_file.read_text())
    layernorm = [r for r in data["kernel_runs"] if r["kernel"] == "layernorm"]
    assert len(layernorm) == 1
    assert layernorm[0]["times_ms"] == [99.0] * 5


# ---------------------------------------------------------------------------
# --dry-run
# ---------------------------------------------------------------------------


def test_dry_run_exits_zero_without_file(tmp_path):
    out_file = tmp_path / "gpu.yaml"
    result = runner.invoke(app, [
        "profile", "gpu", "--name", "TestGPU", "--output", str(out_file),
        "--seq-len", "512", "--dry-run",
    ] + _ARCH_ARGS)
    assert result.exit_code == 0
    assert not out_file.exists(), "--dry-run must not write any file"


def test_dry_run_shows_configurations(tmp_path):
    out_file = tmp_path / "gpu.yaml"
    result = runner.invoke(app, [
        "profile", "gpu", "--name", "TestGPU", "--output", str(out_file),
        "--tp", "1,2", "--batch-size", "1,2", "--seq-len", "512", "--dry-run",
    ] + _ARCH_ARGS)
    assert "tp=1" in result.output
    assert "tp=2" in result.output
    assert "Total: 4 configurations" in result.output


def test_dry_run_does_not_call_sweep(tmp_path):
    out_file = tmp_path / "gpu.yaml"
    with patch("simulon.profiling.sweep.run_sweep") as mock_sweep:
        runner.invoke(app, [
            "profile", "gpu", "--name", "TestGPU", "--output", str(out_file),
            "--seq-len", "512", "--dry-run",
        ] + _ARCH_ARGS)
    mock_sweep.assert_not_called()


# ---------------------------------------------------------------------------
# EP filtering
# ---------------------------------------------------------------------------


def test_ep_gt1_filtered_for_dense_model(tmp_path):
    """EP > 1 configs must be silently dropped for dense (non-MoE) models."""
    out_file = tmp_path / "gpu.yaml"
    captured = {}

    def _fake_sweep(*args, **kwargs):
        captured.setdefault("calls", []).append(kwargs)
        return [_FAKE_RESULT]

    with patch("simulon.profiling.sweep.run_sweep", side_effect=_fake_sweep):
        result = runner.invoke(app, [
            "profile", "gpu", "--name", "TestGPU", "--output", str(out_file),
            "--ep", "1,2,4", "--seq-len", "512",
        ] + _ARCH_ARGS)

    assert result.exit_code == 0
    # Verify filtering via dry-run (no mock needed).
    dry = runner.invoke(app, [
        "profile", "gpu", "--name", "TestGPU", "--output", str(out_file),
        "--ep", "1,2,4", "--seq-len", "512", "--dry-run",
    ] + _ARCH_ARGS)
    assert "ep=2" not in dry.output
    assert "ep=4" not in dry.output
    assert "ep=1" in dry.output


def test_ep_gt_num_experts_filtered_for_moe_model(tmp_path):
    """EP values exceeding num_experts must be dropped for MoE models."""
    out_file = tmp_path / "gpu.yaml"
    moe_args = _ARCH_ARGS + ["--num-experts", "4", "--top-k", "2"]

    dry = runner.invoke(app, [
        "profile", "gpu", "--name", "TestGPU", "--output", str(out_file),
        "--ep", "1,2,4,8,16", "--seq-len", "512", "--dry-run",
    ] + moe_args)

    assert "ep=8" not in dry.output
    assert "ep=16" not in dry.output
    assert "ep=4" in dry.output


def test_ep_sweep_included_for_valid_moe(tmp_path):
    """All EP values that are <= num_experts should appear for MoE models."""
    out_file = tmp_path / "gpu.yaml"
    moe_args = _ARCH_ARGS + ["--num-experts", "8", "--top-k", "2"]

    dry = runner.invoke(app, [
        "profile", "gpu", "--name", "TestGPU", "--output", str(out_file),
        "--ep", "1,2,4,8", "--seq-len", "512", "--dry-run",
    ] + moe_args)

    for ep in [1, 2, 4, 8]:
        assert f"ep={ep}" in dry.output


# ---------------------------------------------------------------------------
# Missing arch fields
# ---------------------------------------------------------------------------


def test_missing_arch_fields_exits_nonzero(tmp_path):
    out_file = tmp_path / "gpu.yaml"
    result = runner.invoke(app, [
        "profile", "gpu", "--name", "TestGPU", "--output", str(out_file),
        "--hidden-size", "4096",  # missing num_heads, ffn_hidden_size, vocab_size
    ])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# OOM handling
# ---------------------------------------------------------------------------


def test_oom_config_is_skipped_gracefully(tmp_path):
    oom_result = SweepResult(config={"tp": 1, "ep": 1, "batch_size": 1, "seq_len": 512}, runs=None, oom=True)
    result, out_file = _run_profile(tmp_path, [], sweep_rv=[oom_result])
    assert result.exit_code == 0
    data = yaml.safe_load(out_file.read_text())
    assert data.get("kernel_runs", []) == []


def test_oom_config_saved_to_profile(tmp_path):
    """OOM configs must be written to oom_configs in the YAML."""
    oom_cfg = {"tp": 1, "ep": 1, "batch_size": 128, "seq_len": 8192}
    oom_result = SweepResult(config=oom_cfg, runs=None, oom=True)
    _, out_file = _run_profile(tmp_path, [], sweep_rv=[oom_result])
    data = yaml.safe_load(out_file.read_text())
    assert oom_cfg in data.get("oom_configs", [])


def test_oom_configs_deduplicated_across_runs(tmp_path):
    """Re-running with the same OOM config must not duplicate the entry."""
    oom_cfg = {"tp": 1, "ep": 1, "batch_size": 128, "seq_len": 8192}
    oom_result = SweepResult(config=oom_cfg, runs=None, oom=True)
    out_file = tmp_path / "gpu.yaml"
    base = ["profile", "gpu", "--name", "TestGPU", "--output", str(out_file), "--seq-len", "512"] + _ARCH_ARGS

    for _ in range(2):
        with _patch_sweep([oom_result]):
            runner.invoke(app, base)

    data = yaml.safe_load(out_file.read_text())
    assert data["oom_configs"].count(oom_cfg) == 1


def test_purge_clears_oom_configs(tmp_path):
    """--purge must also clear previously recorded oom_configs."""
    oom_cfg = {"tp": 1, "ep": 1, "batch_size": 128, "seq_len": 8192}
    oom_result = SweepResult(config=oom_cfg, runs=None, oom=True)
    out_file = tmp_path / "gpu.yaml"
    base = ["profile", "gpu", "--name", "TestGPU", "--output", str(out_file), "--seq-len", "512"] + _ARCH_ARGS

    # First run: record an OOM.
    with _patch_sweep([oom_result]):
        runner.invoke(app, base)
    data = yaml.safe_load(out_file.read_text())
    assert len(data.get("oom_configs", [])) == 1

    # Second run with --purge and a successful result: OOM list should be empty.
    with _patch_sweep([_FAKE_RESULT]):
        runner.invoke(app, base + ["--purge"])
    data = yaml.safe_load(out_file.read_text())
    assert data.get("oom_configs", []) == []


def test_oom_and_success_in_same_sweep(tmp_path):
    """A sweep with mixed OOM/success results saves both kernel_runs and oom_configs."""
    oom_cfg = {"tp": 1, "ep": 1, "batch_size": 128, "seq_len": 512}
    oom_result = SweepResult(config=oom_cfg, runs=None, oom=True)
    out_file = tmp_path / "gpu.yaml"
    # Two configs (batch 1 and 128): first succeeds, second OOMs.
    with patch("simulon.profiling.sweep.run_sweep", side_effect=[[_FAKE_RESULT], [oom_result]]):
        runner.invoke(app, [
            "profile", "gpu", "--name", "TestGPU", "--output", str(out_file),
            "--batch-size", "1,128", "--seq-len", "512",
        ] + _ARCH_ARGS)
    data = yaml.safe_load(out_file.read_text())
    assert len(data.get("kernel_runs", [])) >= 1
    assert oom_cfg in data.get("oom_configs", [])
