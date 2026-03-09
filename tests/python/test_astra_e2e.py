"""End-to-end test for ASTRA-Sim simulation."""

import pytest

from simulon.backend import AnalyticalBackend, AstraSimBackend
from simulon.config.common import DType
from simulon.config.dc import (
    ClusterSpec,
    DatacenterConfig,
    DatacenterMeta,
    GPUSpec,
    NICSpec,
    NodeSpec,
    ScaleOutSpec,
    ScaleOutTopologySpec,
    ScaleUpSpec,
    ScaleUpTopology,
    TopologyType,
)
from simulon.config.scenario import ScenarioConfig
from simulon.config.workload import (
    LLMSpec,
    MegatronParallelism,
    MegatronTraining,
    MegatronWorkload,
)


@pytest.fixture
def simple_scenario():
    """Create a simple test scenario."""
    datacenter = DatacenterConfig(
        datacenter=DatacenterMeta(name="test_cluster"),
        cluster=ClusterSpec(num_nodes=2),
        node=NodeSpec(
            gpus_per_node=4,
            num_switches_per_node=2,
            gpu=GPUSpec(name="H100", memory_capacity_gb=80.0),
        ),
        scale_up=ScaleUpSpec(
            topology=ScaleUpTopology.switched,
            link_bandwidth="900Gbps",
            link_latency="0.001ms",
        ),
        scale_out=ScaleOutSpec(
            nic=NICSpec(speed="400Gbps", latency="0.005ms"),
            topology=ScaleOutTopologySpec(
                type=TopologyType.fat_tree,
                params={"k": 4},
            ),
        ),
    )

    workload = MegatronWorkload(
        framework="megatron",
        model=LLMSpec(
            name="test-model",
            hidden_size=2048,
            num_layers=2,
            num_heads=16,
            swiglu=False,
            moe=False,
        ),
        parallelism=MegatronParallelism(tp=2, pp=1, ep=1, vpp=1),
        training=MegatronTraining(
            num_gpus=8,
            global_batch_size=64,
            micro_batch_size=4,
            sequence_length=1024,
            dtype=DType.bf16,
            flash_attention=False,
        ),
    )

    return ScenarioConfig(datacenter=datacenter, workload=workload)


def test_conversion_only(simple_scenario):
    """Test that conversion works without running simulation."""
    backend = AnalyticalBackend()
    results = backend.run(simple_scenario)

    assert results["status"] == "conversion_complete"
    assert results["network_backend"] == "analytical"
    assert results["topology"]["gpus_per_server"] == 4
    assert results["workload"]["all_gpus"] == 8


def test_simulation_with_flag(simple_scenario):
    """Test running actual simulation with run_simulation flag."""
    backend = AstraSimBackend(network_backend="analytical", run_simulation=True)

    try:
        results = backend.run(simple_scenario)

        # Check basic structure
        assert "status" in results
        assert "simulation" in results
        assert "network_backend" in results
        assert results["network_backend"] == "analytical"

        # Check simulation output
        sim = results["simulation"]
        assert "stdout" in sim
        assert "output_dir" in sim
        assert "workload_file" in sim

        print(f"\nSimulation output directory: {sim['output_dir']}")
        print(f"Workload file: {sim['workload_file']}")
        if sim.get("stdout"):
            print(f"Simulation stdout:\n{sim['stdout'][:500]}")

    except FileNotFoundError as e:
        pytest.skip(f"ASTRA-Sim binary not compiled: {e}")


def test_file_generation(simple_scenario, tmp_path):
    """Test that workload files are generated correctly."""
    from simulon.backend.astra_converter import TopologyConverter, WorkloadConverter
    from simulon.backend.astra_converter.file_writer import write_workload_file

    # Convert
    topo_converter = TopologyConverter()
    workload_converter = WorkloadConverter()
    workload_trace = workload_converter.convert(
        simple_scenario.workload, simple_scenario.datacenter
    )

    # Write file
    workload_file = tmp_path / "test_workload.txt"
    write_workload_file(workload_trace, workload_file)

    # Verify file exists and has content
    assert workload_file.exists()
    content = workload_file.read_text()

    # Check format
    lines = content.strip().split("\n")
    assert len(lines) >= 2  # At least header and num_layers

    # First line should have parallelism info
    assert "model_parallel_NPU_group:" in lines[0]
    assert "all_gpus:" in lines[0]

    # Second line should be number of layers
    num_layers = int(lines[1])
    assert num_layers == 4  # 2 transformer layers * 2 (attn + mlp)

    # Should have that many layer lines
    assert len(lines) >= 2 + num_layers

    print(f"\nGenerated workload file ({num_layers} layers):")
    print(content[:500])
