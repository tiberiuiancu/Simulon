"""Simple tests for ASTRA-Sim converters without C++ bindings."""

from simulon.backend.astra_converter import TopologyConverter, WorkloadConverter
from simulon.config.dc import (
    DatacenterConfig,
    DatacenterMeta,
    ClusterSpec,
    NodeSpec,
    GPUSpec,
    ScaleUpSpec,
    ScaleUpTopology,
    ScaleOutSpec,
    NICSpec,
    ScaleOutTopologySpec,
    TopologyType,
)
from simulon.config.workload import (
    MegatronWorkload,
    MegatronParallelism,
    MegatronTraining,
    LLMSpec,
)
from simulon.config.common import DType


def test_topology_converter_minimal():
    """Test topology converter with minimal configuration."""
    datacenter = DatacenterConfig(
        datacenter=DatacenterMeta(name="test"),
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

    converter = TopologyConverter()
    topology = converter.convert(datacenter)

    # Verify basic structure
    assert topology.gpus_per_server == 4
    assert topology.gpu_type == "H100"

    # Verify GPU nodes
    total_gpus = 2 * 4  # 2 nodes * 4 GPUs
    gpu_nodes = [n for n in topology.nodes if n.node_type == "gpu"]
    assert len(gpu_nodes) == total_gpus

    # Verify NVSwitch nodes (2 per node)
    nvswitch_nodes = [n for n in topology.nodes if n.node_type == "nvswitch"]
    assert len(nvswitch_nodes) == 2 * 2  # 2 nodes * 2 switches
    assert topology.nv_switch_num == 4

    # Verify network switches exist
    network_switches = [
        n for n in topology.nodes if n.node_type in ["leaf", "aggregation", "spine"]
    ]
    assert len(network_switches) > 0
    assert topology.switches_excluding_nvswitch == len(network_switches)


def test_workload_converter_minimal():
    """Test workload converter with minimal configuration."""
    datacenter = DatacenterConfig(
        datacenter=DatacenterMeta(name="test"),
        cluster=ClusterSpec(num_nodes=2),
        node=NodeSpec(
            gpus_per_node=4,
            gpu=GPUSpec(name="H100", memory_capacity_gb=80.0),
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

    converter = WorkloadConverter()
    trace = converter.convert(workload, datacenter)

    # Verify parallelism
    assert trace.model_parallel_npu_group == 2  # TP
    assert trace.pipeline_model_parallelism == 1  # PP
    assert trace.expert_parallel_npu_group == 1  # EP
    assert trace.vpp == 1
    assert trace.all_gpus == 8

    # Verify gradient accumulation
    # DP = 8 / (2 * 1 * 1) = 4
    # GA = 64 / (4 * 4) = 4
    assert trace.ga == 4

    # Verify layers (2 transformer layers * 2 sublayers)
    assert trace.num_layers == 4
    assert len(trace.layers) == 4

    # Check layer types
    assert trace.layers[0].layer_id == "layer_0_attention"
    assert trace.layers[1].layer_id == "layer_0_mlp"
    assert trace.layers[2].layer_id == "layer_1_attention"
    assert trace.layers[3].layer_id == "layer_1_mlp"

    # Check dependencies
    assert trace.layers[0].dependency == -1
    assert trace.layers[1].dependency == 0
    assert trace.layers[2].dependency == 1
    assert trace.layers[3].dependency == 2

    # Check communication types (TP=2 so should have TP collectives)
    assert trace.layers[0].fwd_comm_type == "ALLGATHER_TP"
    assert trace.layers[0].ig_comm_type == "REDUCESCATTER_TP"
    assert trace.layers[0].wg_comm_type == "ALLREDUCE_DP"


def test_bandwidth_parsing():
    """Test that bandwidth strings are parsed correctly."""
    from simulon.backend.astra_converter.topology import _parse_bandwidth

    assert _parse_bandwidth("900Gbps") == 900.0
    assert _parse_bandwidth("1Tbps") == 1000.0
    assert _parse_bandwidth("500Mbps") == 0.5
    assert _parse_bandwidth("100") == 100.0


def test_latency_parsing():
    """Test that latency strings are parsed correctly."""
    from simulon.backend.astra_converter.topology import _parse_latency

    assert _parse_latency("1ms") == 1_000_000
    assert _parse_latency("5us") == 5_000
    assert _parse_latency("100ns") == 100
    assert _parse_latency("1s") == 1_000_000_000
