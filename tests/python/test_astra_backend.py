"""Tests for ASTRA-Sim backend."""

import pytest

from simulon.backend.astra_sim import AstraSimBackend
from simulon.config.datacenter import (
    DatacenterConfig,
    GPUConfig,
    KernelBenchmark,
    NICConfig,
    NodeConfig,
    ScaleOutConfig,
    ScaleUpConfig,
    TopologyConfig,
)
from simulon.config.scenario import ScenarioConfig
from simulon.config.workload import (
    MegatronWorkload,
    ModelConfig,
    ParallelismConfig,
)


def test_astra_backend_basic():
    """Test basic backend run with simple scenario."""
    datacenter = DatacenterConfig(
        scale=4,
        node=NodeConfig(
            num_gpus=8,
            gpu=GPUConfig(
                name="H100",
                memory_gb=80,
                compute_tflops=1000,
                kernel_runs={
                    "matmul": KernelBenchmark(
                        mean_duration_ms=1.0, flops_multiplier=1.0
                    )
                },
            ),
            scale_up=ScaleUpConfig(
                topology="switched",
                num_switches=4,
                link_bandwidth=900,
                link_latency=1000,
            ),
            scale_out=ScaleOutConfig(
                nic=NICConfig(speed=400, latency=5000),
                topology=TopologyConfig(name="fat_tree", params={"k": 4}),
            ),
        ),
    )

    workload = MegatronWorkload(
        global_batch_size=1024,
        micro_batch_size=4,
        model=ModelConfig(
            num_layers=2,
            hidden_size=4096,
            num_attention_heads=32,
            sequence_length=2048,
            flash_attention=False,
            swiglu=False,
            moe=False,
        ),
        parallelism=ParallelismConfig(
            tensor_parallel=2,
            pipeline_parallel=2,
            expert_parallel=1,
            virtual_pipeline_parallel=1,
        ),
    )

    scenario = ScenarioConfig(
        name="test_scenario",
        datacenter=datacenter,
        workload=workload,
    )

    backend = AstraSimBackend()
    results = backend.run(scenario)

    # Verify conversion completed
    assert results["status"] == "conversion_complete"

    # Verify topology results
    topo = results["topology"]
    assert topo["gpus_per_server"] == 8
    assert topo["gpu_type"] == "H100"
    assert topo["num_nodes"] > 0
    assert topo["num_links"] > 0

    # Verify workload results
    wl = results["workload"]
    assert wl["tensor_parallel"] == 2
    assert wl["pipeline_parallel"] == 2
    assert wl["all_gpus"] == 32
    assert wl["num_layers"] == 4  # 2 transformer layers * 2 sublayers


def test_astra_backend_conversion_details():
    """Test that backend correctly converts and reports details."""
    datacenter = DatacenterConfig(
        scale=2,
        node=NodeConfig(
            num_gpus=8,
            gpu=GPUConfig(
                name="A100",
                memory_gb=80,
                compute_tflops=312,
                kernel_runs={},
            ),
            scale_up=ScaleUpConfig(
                topology="p2p",
                num_switches=0,
                link_bandwidth=600,
                link_latency=2000,
            ),
            scale_out=ScaleOutConfig(
                nic=NICConfig(speed=200, latency=10000),
                topology=TopologyConfig(name="fat_tree", params={"k": 4}),
            ),
        ),
    )

    workload = MegatronWorkload(
        global_batch_size=128,
        micro_batch_size=8,
        model=ModelConfig(
            num_layers=4,
            hidden_size=2048,
            num_attention_heads=16,
            sequence_length=1024,
            flash_attention=True,
            swiglu=True,
            moe=False,
        ),
        parallelism=ParallelismConfig(
            tensor_parallel=4,
            pipeline_parallel=2,
            expert_parallel=1,
            virtual_pipeline_parallel=2,
        ),
    )

    scenario = ScenarioConfig(
        name="test_scenario_2",
        datacenter=datacenter,
        workload=workload,
    )

    backend = AstraSimBackend()
    results = backend.run(scenario)

    # Check topology details
    topo = results["topology"]
    assert topo["gpu_type"] == "A100"
    assert topo["nv_switch_num"] == 0  # P2P topology

    # Check workload details
    wl = results["workload"]
    assert wl["parallelism_policy"] == "HYBRID_TRANSFORMER"
    assert wl["tensor_parallel"] == 4
    assert wl["pipeline_parallel"] == 2
    assert wl["virtual_pipeline_parallel"] == 2
    assert wl["all_gpus"] == 16
    assert wl["num_layers"] == 8  # 4 transformer layers * 2 sublayers
