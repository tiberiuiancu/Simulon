"""Integration tests for ATLAHSLGSBackend with real LogGOPSim binary.

These tests invoke the actual ATLAHS LogGOPSim discrete-event simulator.
They are marked ``@pytest.mark.slow`` and are skipped automatically when the
binary cannot be found via :func:`find_binaries`.
"""

from __future__ import annotations

import glob as glob_mod
import subprocess
from pathlib import Path

import pytest
import yaml

from simulon.backend.atlahs_binary_finder import find_binaries
from simulon.backend.atlahs_lgs import ATLAHSLGSBackend
from simulon.config.dc import (
    ClusterSpec,
    DatacenterConfig,
    DatacenterMeta,
    GPUSpec,
    NICSpec,
    NetworkSpec,
    NodeSpec,
    ScaleOutSpec,
    ScaleUpSpec,
    SwitchSpec,
    TopologySpec,
    TopologyType,
)
from simulon.config.scenario import NcclConfig, ScenarioConfig
from simulon.config.workload import CollectiveType, CollectiveWorkload


def _lgs_binary_available() -> bool:
    try:
        _ = find_binaries()
        return True
    except RuntimeError:
        return False


LGS_AVAILABLE = _lgs_binary_available()


def make_small_collective_scenario(
    collective_type: str = "AllReduce",
    message_size_bytes: int = 1024,
    num_nodes: int = 2,
    gpus_per_node: int = 1,
) -> ScenarioConfig:
    dc = DatacenterConfig(
        datacenter=DatacenterMeta(name="test"),
        cluster=ClusterSpec(num_nodes=num_nodes),
        node=NodeSpec(
            gpus_per_node=gpus_per_node,
            gpu=GPUSpec(name="H100", memory_capacity_gb=80.0),
        ),
        network=NetworkSpec(
            scale_up=ScaleUpSpec(
                switch=SwitchSpec(port_speed="2880Gbps", latency="0.000025ms"),
            ),
            scale_out=ScaleOutSpec(
                nic=NICSpec(speed="400Gbps", latency="0.005ms"),
                topology=TopologySpec(type=TopologyType.fat_tree, params={"k": 4}),
            ),
        ),
    )
    wl = CollectiveWorkload(
        framework="collective",
        collective_type=CollectiveType(collective_type),
        message_size_bytes=message_size_bytes,
    )
    return ScenarioConfig(
        datacenter=dc,
        workload=wl,
        collective=NcclConfig(algorithm="ring", num_channels=1),
    )


@pytest.fixture
def backend():
    return ATLAHSLGSBackend()


@pytest.mark.slow
@pytest.mark.skipif(not LGS_AVAILABLE, reason="LogGOPSim binary not found")
def test_lgs_collective_total_time_positive(backend):
    """LogGOPSim produces a positive total time for a 2-rank AllReduce."""
    sc = make_small_collective_scenario(
        collective_type="AllReduce",
        message_size_bytes=1024,
        num_nodes=2,
        gpus_per_node=1,
    )
    dag, result = backend.simulate(sc)
    assert result.total_time_ms > 0


@pytest.mark.slow
@pytest.mark.skipif(not LGS_AVAILABLE, reason="LogGOPSim binary not found")
def test_lgs_collective_summary_non_empty(backend):
    """The result summary is a non-empty string."""
    sc = make_small_collective_scenario(
        collective_type="AllReduce",
        message_size_bytes=1024,
        num_nodes=2,
        gpus_per_node=1,
    )
    dag, result = backend.simulate(sc)
    assert isinstance(result.summary, str)
    assert result.summary.strip() != ""
    assert "Total time:" in result.summary


@pytest.mark.slow
@pytest.mark.skipif(not LGS_AVAILABLE, reason="LogGOPSim binary not found")
def test_lgs_collective_per_host_times_populated(backend):
    """``per_host_times`` contains an entry for every rank."""
    sc = make_small_collective_scenario(
        collective_type="AllReduce",
        message_size_bytes=1024,
        num_nodes=2,
        gpus_per_node=1,
    )
    dag, result = backend.simulate(sc)
    assert isinstance(result.per_host_times, dict)
    assert len(result.per_host_times) == 2
    for host, t in result.per_host_times.items():
        assert isinstance(host, int)
        assert t >= 0


@pytest.mark.slow
@pytest.mark.skipif(not LGS_AVAILABLE, reason="LogGOPSim binary not found")
def test_lgs_temp_files_cleaned_up(backend):
    """No ``.goal`` or ``.bin`` files are left behind in ``/tmp``."""
    tmp_prefix = "simulon_atlahs_"
    # Snapshot existing temp files before the run
    before = set(
        glob_mod.glob(f"/tmp/{tmp_prefix}*/**/*.goal", recursive=True)
        + glob_mod.glob(f"/tmp/{tmp_prefix}*/**/*.bin", recursive=True)
    )

    sc = make_small_collective_scenario(
        collective_type="AllReduce",
        message_size_bytes=1024,
        num_nodes=2,
        gpus_per_node=1,
    )
    dag, result = backend.simulate(sc)
    assert result.total_time_ms > 0

    # Snapshot temp files after the run
    after = set(
        glob_mod.glob(f"/tmp/{tmp_prefix}*/**/*.goal", recursive=True)
        + glob_mod.glob(f"/tmp/{tmp_prefix}*/**/*.bin", recursive=True)
    )

    # Any files that appeared during the run should have been cleaned up
    new_files = after - before
    assert new_files == set(), f"Leftover ATLAHS temp files: {new_files}"


@pytest.mark.slow
@pytest.mark.skipif(not LGS_AVAILABLE, reason="LogGOPSim binary not found")
def test_lgs_allgather_collective(backend):
    """LogGOPSim handles an AllGather collective without error."""
    sc = make_small_collective_scenario(
        collective_type="AllGather",
        message_size_bytes=512,
        num_nodes=2,
        gpus_per_node=1,
    )
    dag, result = backend.simulate(sc)
    assert result.total_time_ms > 0
    assert isinstance(result.summary, str)
    assert result.summary.strip() != ""


@pytest.mark.slow
@pytest.mark.skipif(not LGS_AVAILABLE, reason="LogGOPSim binary not found")
def test_lgs_cli_energy_flag(tmp_path: Path):
    """The CLI ``simulate`` command accepts ``--energy`` without crashing."""
    sc = make_small_collective_scenario(
        collective_type="AllReduce",
        message_size_bytes=1024,
        num_nodes=2,
        gpus_per_node=1,
    )
    scenario_path = tmp_path / "scenario.yaml"
    _ = scenario_path.write_text(yaml.safe_dump(sc.model_dump(mode="json")))

    cmd = [
        "uv", "run", "simulon",
        "simulate", str(scenario_path),
        "--backend", "atlahs-lgs",
        "--energy",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd="/Users/tibi/uni/t/simulon-atlahs-tree")
    assert proc.returncode == 0, f"CLI exited {proc.returncode}: {proc.stderr}"
    # For collective workloads energy is not computed, but the flag must be accepted.
    assert "Collective wall time:" in proc.stdout
