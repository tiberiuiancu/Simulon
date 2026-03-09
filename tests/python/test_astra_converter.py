"""Tests for ASTRA-Sim converters."""

import pytest

from simulon.backend.astra_converter import (
    TopologyConverter,
    WorkloadConverter,
)
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
from simulon.config.workload import (
    MegatronWorkload,
    ModelConfig,
    ParallelismConfig,
)


def test_topology_converter_basic():
    """Test basic topology conversion with minimal config."""
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

    converter = TopologyConverter()
    topology = converter.convert(datacenter)

    # Verify basic counts
    total_gpus = 4 * 8  # 4 nodes * 8 GPUs
    assert len([n for n in topology.nodes if n.node_type == "gpu"]) == total_gpus
    assert topology.gpus_per_server == 8
    assert topology.gpu_type == "H100"

    # Verify NVSwitch nodes
    nvswitches_per_node = 4
    total_nvswitches = 4 * 4  # 4 nodes * 4 switches
    assert topology.nv_switch_num == total_nvswitches
    assert len([n for n in topology.nodes if n.node_type == "nvswitch"]) == total_nvswitches

    # Verify GPU IDs are in range [0, total_gpus)
    gpu_nodes = [n for n in topology.nodes if n.node_type == "gpu"]
    gpu_ids = [n.node_id for n in gpu_nodes]
    assert min(gpu_ids) == 0
    assert max(gpu_ids) == total_gpus - 1

    # Verify NVSwitch IDs start after GPUs
    nvswitch_nodes = [n for n in topology.nodes if n.node_type == "nvswitch"]
    nvswitch_ids = [n.node_id for n in nvswitch_nodes]
    assert min(nvswitch_ids) == total_gpus
    assert max(nvswitch_ids) == total_gpus + total_nvswitches - 1


def test_topology_converter_fat_tree():
    """Test fat-tree topology generation."""
    datacenter = DatacenterConfig(
        scale=8,
        node=NodeConfig(
            num_gpus=8,
            gpu=GPUConfig(
                name="H100",
                memory_gb=80,
                compute_tflops=1000,
                kernel_runs={},
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

    converter = TopologyConverter()
    topology = converter.convert(datacenter)

    # For k=4 fat-tree:
    # - 4 pods
    # - 2 leaf switches per pod (8 total)
    # - 2 aggregation switches per pod (8 total)
    # - 4 core switches
    expected_network_switches = 8 + 8 + 4  # 20 switches
    assert topology.switches_excluding_nvswitch == expected_network_switches

    # Verify network switch nodes
    network_switches = [
        n
        for n in topology.nodes
        if n.node_type in ["leaf", "aggregation", "spine"]
    ]
    assert len(network_switches) == expected_network_switches

    # Verify leaf switches exist
    leaf_switches = [n for n in topology.nodes if n.node_type == "leaf"]
    assert len(leaf_switches) == 8

    # Verify aggregation switches exist
    agg_switches = [n for n in topology.nodes if n.node_type == "aggregation"]
    assert len(agg_switches) == 8

    # Verify core (spine) switches exist
    core_switches = [n for n in topology.nodes if n.node_type == "spine"]
    assert len(core_switches) == 4


def test_topology_converter_p2p():
    """Test peer-to-peer scale-up topology."""
    datacenter = DatacenterConfig(
        scale=2,
        node=NodeConfig(
            num_gpus=8,
            gpu=GPUConfig(
                name="H100",
                memory_gb=80,
                compute_tflops=1000,
                kernel_runs={},
            ),
            scale_up=ScaleUpConfig(
                topology="p2p",
                num_switches=0,
                link_bandwidth=900,
                link_latency=1000,
            ),
            scale_out=ScaleOutConfig(
                nic=NICConfig(speed=400, latency=5000),
                topology=TopologyConfig(name="fat_tree", params={"k": 4}),
            ),
        ),
    )

    converter = TopologyConverter()
    topology = converter.convert(datacenter)

    # P2P should have no NVSwitches
    assert topology.nv_switch_num == 0
    assert len([n for n in topology.nodes if n.node_type == "nvswitch"]) == 0

    # Should have direct GPU-to-GPU links within each node
    # For 8 GPUs per node: 8 * 7 = 56 unidirectional links per node
    # 2 nodes * 56 = 112 total intra-node links
    intra_node_links = [
        link
        for link in topology.links
        if link.source < 16 and link.dest < 16  # Both endpoints are GPUs
        and link.source // 8 == link.dest // 8  # Same node
    ]
    assert len(intra_node_links) == 2 * 8 * 7


def test_workload_converter_basic():
    """Test basic workload conversion."""
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

    converter = WorkloadConverter()
    trace = converter.convert(workload, datacenter)

    # Verify parallelism
    assert trace.model_parallel_npu_group == 2  # TP
    assert trace.pipeline_model_parallelism == 2  # PP
    assert trace.expert_parallel_npu_group == 1  # EP
    assert trace.all_gpus == 32  # 4 nodes * 8 GPUs

    # Calculate expected DP and GA
    total_gpus = 32
    tp, pp, ep = 2, 2, 1
    dp = total_gpus // (tp * pp * ep)  # 32 // 4 = 8
    micro_batch = 4
    global_batch = 1024
    expected_ga = global_batch // (dp * micro_batch)  # 1024 // 32 = 32
    assert trace.ga == expected_ga

    # Verify layers: 2 transformer layers * 2 sublayers (attn + mlp) = 4 layers
    assert trace.num_layers == 4
    assert len(trace.layers) == 4

    # Check layer dependencies
    assert trace.layers[0].dependency == -1  # First layer
    assert trace.layers[1].dependency == 0  # Depends on previous
    assert trace.layers[2].dependency == 1
    assert trace.layers[3].dependency == 2

    # Verify layer IDs
    assert trace.layers[0].layer_id == "layer_0_attention"
    assert trace.layers[1].layer_id == "layer_0_mlp"
    assert trace.layers[2].layer_id == "layer_1_attention"
    assert trace.layers[3].layer_id == "layer_1_mlp"


def test_workload_converter_communication_types():
    """Test that correct communication types are generated."""
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
        global_batch_size=128,
        micro_batch_size=4,
        model=ModelConfig(
            num_layers=1,
            hidden_size=4096,
            num_attention_heads=32,
            sequence_length=2048,
            flash_attention=False,
            swiglu=False,
            moe=False,
        ),
        parallelism=ParallelismConfig(
            tensor_parallel=2,  # TP > 1
            pipeline_parallel=1,
            expert_parallel=1,
            virtual_pipeline_parallel=1,
        ),
    )

    converter = WorkloadConverter()
    trace = converter.convert(workload, datacenter)

    # With TP=2, should have TP collectives
    attn_layer = trace.layers[0]
    assert attn_layer.fwd_comm_type == "ALLGATHER_TP"
    assert attn_layer.ig_comm_type == "REDUCESCATTER_TP"

    # With DP > 1, should have DP ALLREDUCE in weight gradient
    dp = 32 // 2  # total_gpus // tp = 16
    assert dp > 1
    assert attn_layer.wg_comm_type == "ALLREDUCE_DP"


def test_workload_converter_no_tp():
    """Test workload with no tensor parallelism."""
    datacenter = DatacenterConfig(
        scale=1,
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
        global_batch_size=64,
        micro_batch_size=8,
        model=ModelConfig(
            num_layers=1,
            hidden_size=4096,
            num_attention_heads=32,
            sequence_length=2048,
            flash_attention=False,
            swiglu=False,
            moe=False,
        ),
        parallelism=ParallelismConfig(
            tensor_parallel=1,  # No TP
            pipeline_parallel=1,
            expert_parallel=1,
            virtual_pipeline_parallel=1,
        ),
    )

    converter = WorkloadConverter()
    trace = converter.convert(workload, datacenter)

    # With TP=1, should have no TP collectives
    attn_layer = trace.layers[0]
    assert attn_layer.fwd_comm_type == "NONE"
    assert attn_layer.ig_comm_type == "NONE"

    # With DP=8, should still have DP ALLREDUCE
    assert attn_layer.wg_comm_type == "ALLREDUCE_DP"


def test_workload_converter_moe():
    """Test MoE workload generates correct parallelism policy."""
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
        global_batch_size=128,
        micro_batch_size=4,
        model=ModelConfig(
            num_layers=1,
            hidden_size=4096,
            num_attention_heads=32,
            sequence_length=2048,
            flash_attention=False,
            swiglu=False,
            moe=True,
            num_experts=8,
        ),
        parallelism=ParallelismConfig(
            tensor_parallel=2,
            pipeline_parallel=2,
            expert_parallel=2,
            virtual_pipeline_parallel=1,
        ),
    )

    converter = WorkloadConverter()
    trace = converter.convert(workload, datacenter)

    # MoE should use different parallelism policy
    assert trace.parallelism_policy == "HYBRID_TRANSFORMER_FP8_MoE"
    assert trace.expert_parallel_npu_group == 2
