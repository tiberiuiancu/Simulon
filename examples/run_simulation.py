#!/usr/bin/env python3
"""Example: Run ASTRA-Sim simulation through Python bindings."""

from simulon.backend import AnalyticalBackend
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


def main():
    # Define datacenter configuration
    datacenter = DatacenterConfig(
        datacenter=DatacenterMeta(name="my_cluster"),
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

    # Define workload
    workload = MegatronWorkload(
        framework="megatron",
        model=LLMSpec(
            name="llama-7b",
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            swiglu=False,
            moe=False,
        ),
        parallelism=MegatronParallelism(tp=2, pp=2, ep=1, vpp=1),
        training=MegatronTraining(
            num_gpus=8,
            global_batch_size=128,
            micro_batch_size=2,
            sequence_length=2048,
            dtype=DType.bf16,
            flash_attention=False,
        ),
    )

    scenario = ScenarioConfig(datacenter=datacenter, workload=workload)

    print("=" * 80)
    print("OPTION 1: Conversion only (no simulation)")
    print("=" * 80)

    # Option 1: Just do the conversion (fast)
    backend = AnalyticalBackend()
    results = backend.run(scenario)

    print(f"Status: {results['status']}")
    print(f"Network backend: {results['network_backend']}")
    print(f"Topology: {results['topology']['num_nodes']} nodes, "
          f"{results['topology']['num_links']} links")
    print(f"Workload: {results['workload']['num_layers']} layers, "
          f"TP={results['workload']['tensor_parallel']}, "
          f"PP={results['workload']['pipeline_parallel']}")



if __name__ == "__main__":
    main()
