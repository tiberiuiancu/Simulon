"""CLI integration tests for --backend flag.

Uses ``typer.testing.CliRunner`` with mocked backends so no real
simulation or external binaries are required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from simulon.cli import app
from simulon.backend.atlahs_base import ATLAHSResult
from simulon.backend.dag.nodes import ExecutionDAG
from simulon.backend.dag.replayer import SimulationResult
from simulon.energy import ComponentEnergy, EnergyResult

runner = CliRunner()

_MINIMAL_SCENARIO = """
datacenter:
  datacenter:
    name: test-cluster
  cluster:
    num_nodes: 1
  node:
    gpus_per_node: 2
    gpu:
      name: H100
      memory_capacity_gb: 80.0
  network:
    scale_up:
      switch:
        port_speed: 2880Gbps
        latency: 0.000025ms

collective:
  library: nccl
  algorithm: ring

workload:
  framework: megatron
  model:
    name: test-model
    hidden_size: 256
    num_layers: 2
    num_heads: 4
    ffn_hidden_size: 1024
    vocab_size: 32000
  parallelism:
    tp: 1
    pp: 1
    num_microbatches: 2
  training:
    num_gpus: 2
    global_batch_size: 2
    micro_batch_size: 1
    sequence_length: 128
"""


@pytest.fixture
def scenario_file(tmp_path: Path) -> Path:
    path = tmp_path / "scenario.yaml"
    path.write_text(_MINIMAL_SCENARIO)
    return path


@pytest.fixture
def mock_dag() -> ExecutionDAG:
    return ExecutionDAG()


@pytest.fixture
def mock_sim_result() -> SimulationResult:
    return SimulationResult(
        total_time_ms=42.0,
        compute_ms=30.0,
        exposed_comm_ms=10.0,
        exposed_comm_by_type={"allreduce": 10.0},
        bubble_ms=2.0,
        overlapped_comm_ms=5.0,
        per_gpu_times_ms={0: 42.0, 1: 42.0},
    )


@pytest.fixture
def mock_atlahs_result() -> ATLAHSResult:
    return ATLAHSResult(
        total_time_ms=42.0,
        summary="Total time: 42.000 ms",
        per_host_times={0: 42.0},
        raw_output="Host 0: 42000000",
    )


def test_backend_analytical_default(scenario_file, mock_dag, mock_sim_result):
    with patch(
        "simulon.backend.analytical.AnalyticalBackend.simulate",
        return_value=(mock_dag, mock_sim_result),
    ):
        result = runner.invoke(app, ["simulate", str(scenario_file)])
    assert result.exit_code == 0, result.output
    assert "42.000" in result.output


def test_backend_analytical_explicit(scenario_file, mock_dag, mock_sim_result):
    with patch(
        "simulon.backend.analytical.AnalyticalBackend.simulate",
        return_value=(mock_dag, mock_sim_result),
    ):
        result = runner.invoke(app, [
            "simulate", str(scenario_file), "--backend", "analytical",
        ])
    assert result.exit_code == 0, result.output
    assert "42.000" in result.output


def test_backend_atlahs_lgs(scenario_file, mock_dag, mock_atlahs_result):
    with patch(
        "simulon.backend.atlahs_lgs.ATLAHSLGSBackend.simulate",
        return_value=(mock_dag, mock_atlahs_result),
    ):
        result = runner.invoke(app, [
            "simulate", str(scenario_file), "--backend", "atlahs-lgs",
        ])
    assert result.exit_code == 0, result.output
    assert "42.000" in result.output


def test_backend_atlahs_htsim(scenario_file, mock_dag, mock_atlahs_result):
    with patch(
        "simulon.backend.atlahs_htsim.ATLAHShtsimBackend.simulate",
        return_value=(mock_dag, mock_atlahs_result),
    ):
        result = runner.invoke(app, [
            "simulate", str(scenario_file), "--backend", "atlahs-htsim",
        ])
    assert result.exit_code == 0, result.output
    assert "42.000" in result.output


def test_backend_unknown(scenario_file):
    result = runner.invoke(app, [
        "simulate", str(scenario_file), "--backend", "nonexistent",
    ])
    assert result.exit_code != 0
    assert "Unknown backend" in result.output


def test_backend_chrome_with_atlahs(scenario_file, tmp_path):
    chrome_path = tmp_path / "trace.json"
    result = runner.invoke(app, [
        "simulate", str(scenario_file),
        "--backend", "atlahs-lgs",
        "--chrome", str(chrome_path),
    ])
    assert result.exit_code != 0
    assert "--chrome is only supported with the analytical backend" in result.output


def test_backend_atlahs_lgs_energy(scenario_file, mock_dag, mock_atlahs_result):
    mock_energy = EnergyResult(
        total_wh=0.5678,
        hardware_subtotal_wh=0.4000,
        pue_overhead_wh=0.1678,
        avg_power_kw=48.0,
        run_duration_hours=0.0001,
        breakdown=[
            ComponentEnergy(component="GPU", wh=0.3000, pct=75.0),
        ],
    )
    with patch(
        "simulon.backend.atlahs_lgs.ATLAHSLGSBackend.simulate",
        return_value=(mock_dag, mock_atlahs_result),
    ):
        with patch(
            "simulon.energy.compute_energy",
            return_value=mock_energy,
        ):
            result = runner.invoke(app, [
                "simulate", str(scenario_file),
                "--backend", "atlahs-lgs",
                "--energy",
            ])
    assert result.exit_code == 0, result.output
    assert "Energy per iteration" in result.output
    assert "0.5678" in result.output
