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


def test_basic_simulation(simple_scenario):
    """Test basic simulation runs successfully."""
    backend = AnalyticalBackend()
    results = backend.run(simple_scenario)

    # With C++ bindings, simulation actually runs
    assert results["status"] == "success"
    assert results["network_backend"] == "analytical"
    assert results["topology"]["gpus_per_server"] == 4
    assert results["workload"]["all_gpus"] == 8
    assert "simulation" in results
    assert results["simulation"]["total_time_ns"] > 0


def test_in_memory_simulation(simple_scenario):
    """Test in-memory simulation with C++ bindings."""
    backend = AstraSimBackend(network_backend="analytical")

    try:
        results = backend.run(simple_scenario)

        # Should get simulation results (not just conversion)
        assert results["status"] == "success"
        assert results["network_backend"] == "analytical"

        # Check simulation output exists
        assert "simulation" in results
        sim = results["simulation"]
        assert "total_time_ns" in sim
        assert sim["total_time_ns"] > 0  # Should have some simulation time
        assert sim["completed_layers"] == 4  # 2 transformer layers * 2 (attn + mlp)

        print(f"\nSimulation completed in {sim['total_time_ns']} ns")
        print(f"Completed {sim['completed_layers']} layers")

    except ImportError as e:
        pytest.skip(f"C++ bindings not available: {e}")
