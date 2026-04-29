"""End-to-end CLI tests for multi-workload simulation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from typer.testing import CliRunner

from simulon.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _patch_lookup_kernel_time():
    """Provide dummy kernel timings so populate_dag() never leaves duration_ms as None."""
    with patch(
        "simulon.backend.dag.populate.lookup_kernel_time",
        return_value=(0.1, False),
    ):
        yield


def _make_multi_workload_yaml(tmp_path: Path) -> Path:
    """Write a tiny two-workload scenario YAML and return its path."""
    scenario = {
        "datacenter": {
            "datacenter": {"name": "test-cluster"},
            "cluster": {"num_nodes": 2},
            "node": {
                "gpus_per_node": 4,
                "gpu": {
                    "name": "TestGPU",
                    "memory_capacity_gb": 80.0,
                },
            },
            "network": {
                "scale_up": {
                    "switch": {
                        "port_speed": "7200Gbps",
                        "latency": "0.000025ms",
                    }
                },
                "scale_out": {
                    "nic": {
                        "speed": "400Gbps",
                        "latency": "0.005ms",
                    },
                    "topology": {
                        "type": "fat_tree",
                        "params": {"k": 2},
                    },
                },
            },
        },
        "collective": {
            "library": "nccl",
            "algorithm": "ring",
            "num_channels": 1,
        },
        "workloads": [
            {
                "name": "job-a",
                "workload": {
                    "framework": "megatron",
                    "model": {
                        "name": "job-a",
                        "hidden_size": 256,
                        "num_layers": 1,
                        "num_heads": 4,
                        "ffn_hidden_size": 512,
                        "vocab_size": 32000,
                    },
                    "parallelism": {"tp": 1, "pp": 1},
                    "training": {
                        "num_gpus": 4,
                        "global_batch_size": 4,
                        "micro_batch_size": 1,
                        "sequence_length": 128,
                    },
                },
                "start": {"offset_ms": 0.0},
            },
            {
                "name": "job-b",
                "workload": {
                    "framework": "megatron",
                    "model": {
                        "name": "job-b",
                        "hidden_size": 256,
                        "num_layers": 1,
                        "num_heads": 4,
                        "ffn_hidden_size": 512,
                        "vocab_size": 32000,
                    },
                    "parallelism": {"tp": 1, "pp": 1},
                    "training": {
                        "num_gpus": 4,
                        "global_batch_size": 4,
                        "micro_batch_size": 1,
                        "sequence_length": 128,
                    },
                },
                "start": {"offset_ms": 0.0},
            },
        ],
    }
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.dump(scenario, default_flow_style=False, sort_keys=False))
    return path


def test_simulate_multi_workload_prints_summary(tmp_path):
    """CLI simulate with multi-workload YAML prints aggregate and per-workload summaries."""
    scenario_path = _make_multi_workload_yaml(tmp_path)
    result = runner.invoke(app, [
        "simulate", str(scenario_path),
        "--ignore-missing",
    ])
    assert result.exit_code == 0, result.output
    assert "Aggregate Summary" in result.output
    assert "job-a" in result.output
    assert "job-b" in result.output


def test_simulate_multi_workload_chrome_trace(tmp_path):
    """CLI simulate --chrome produces a valid Chrome trace with traceEvents."""
    scenario_path = _make_multi_workload_yaml(tmp_path)
    chrome_path = tmp_path / "trace.json"
    result = runner.invoke(app, [
        "simulate", str(scenario_path),
        "--ignore-missing",
        "--chrome", str(chrome_path),
    ])
    assert result.exit_code == 0, result.output
    assert chrome_path.exists()
    data = json.loads(chrome_path.read_text())
    assert "traceEvents" in data
    assert len(data["traceEvents"]) > 0


def test_simulate_multi_workload_goal_trace(tmp_path):
    """CLI simulate --goal produces a GOAL trace containing calc lines."""
    scenario_path = _make_multi_workload_yaml(tmp_path)
    goal_path = tmp_path / "trace.goal"
    result = runner.invoke(app, [
        "simulate", str(scenario_path),
        "--ignore-missing",
        "--goal", str(goal_path),
    ])
    assert result.exit_code == 0, result.output
    assert goal_path.exists()
    content = goal_path.read_text()
    assert "calc" in content
