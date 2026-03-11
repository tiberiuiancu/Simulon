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
    NetworkSpec,
    NodeSpec,
    ScaleOutSpec,
    ScaleUpSpec,
    SwitchSpec,
    TopologySpec,
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


def test_ns3_backend_unavailable_raises_import_error(simple_scenario):
    """When NS3 bindings are absent, NS3 backend raises ImportError (not NotImplementedError)."""
    try:
        from simulon._sim import run_ns3  # noqa: F401
        pytest.skip("NS3 bindings present — skipping unavailability test")
    except ImportError:
        pass

    backend = AstraSimBackend(network_backend="ns3")
    with pytest.raises(ImportError, match="NS3 backend unavailable"):
        backend.run(simple_scenario)


def test_ns3_simulation(simple_scenario):
    """Test NS3 backend end-to-end when bindings are available."""
    try:
        from simulon._sim import run_ns3  # noqa: F401
    except ImportError:
        pytest.skip("NS3 bindings not available (build with SIMULON_NS3=ON)")

    backend = AstraSimBackend(network_backend="ns3")
    results = backend.run(simple_scenario)

    assert results["network_backend"] == "ns3"
    assert results["status"] in ("success", "error")

    assert results["topology"]["gpus_per_server"] == 4
    assert results["workload"]["all_gpus"] == 8

    if results["status"] == "success":
        sim = results["simulation"]
        assert sim["total_time_ns"] > 0
        assert sim["completed_layers"] == 4
        print(f"\nNS3 simulation completed in {sim['total_time_ns']} ns")
    else:
        pytest.fail(f"NS3 simulation failed: {results['simulation']['error_message']}")
