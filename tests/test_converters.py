"""Tests for ASTRA-Sim topology and workload converters."""

from simulon.backend.astra_converter import TopologyConverter, WorkloadConverter
from simulon.backend.astra_converter.topology import _parse_bandwidth, _parse_latency
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
from simulon.config.workload import (
    LLMSpec,
    MegatronParallelism,
    MegatronTraining,
    MegatronWorkload,
)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def make_datacenter(
    num_nodes: int,
    gpus_per_node: int,
    fat_tree_k: int = 4,
) -> DatacenterConfig:
    return DatacenterConfig(
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
                topology=TopologySpec(type=TopologyType.fat_tree, params={"k": fat_tree_k}),
            ),
        ),
    )


def make_workload(
    num_gpus: int,
    tp: int,
    pp: int,
    ep: int = 1,
    vpp: int = 1,
    num_layers: int = 2,
    hidden_size: int = 4096,
    num_heads: int = 32,
    global_batch_size: int = 1024,
    micro_batch_size: int = 4,
    moe: bool = False,
    num_experts: int = 1,
) -> MegatronWorkload:
    return MegatronWorkload(
        framework="megatron",
        model=LLMSpec(
            name="test-model",
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            swiglu=False,
            moe=moe,
            num_experts=num_experts if moe else None,
        ),
        parallelism=MegatronParallelism(tp=tp, pp=pp, ep=ep, vpp=vpp),
        training=MegatronTraining(
            num_gpus=num_gpus,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            sequence_length=2048,
            dtype=DType.bf16,
            flash_attention=False,
        ),
    )


# ---------------------------------------------------------------------------
# Topology converter tests
# ---------------------------------------------------------------------------


def test_topology_converter_minimal():
    """Topology converter with small cluster (2 nodes, 4 GPUs)."""
    dc = make_datacenter(num_nodes=2, gpus_per_node=4)
    topology = TopologyConverter().convert(dc)

    assert topology.gpus_per_server == 4
    assert topology.gpu_type == "H100"

    total_gpus = 2 * 4
    gpu_nodes = [n for n in topology.nodes if n.node_type == "gpu"]
    assert len(gpu_nodes) == total_gpus

    nvswitch_nodes = [n for n in topology.nodes if n.node_type == "nvswitch"]
    assert len(nvswitch_nodes) == 2 * 1  # 2 nodes * 1 NVSwitch per node
    assert topology.nv_switch_num == 2

    network_switches = [
        n for n in topology.nodes if n.node_type in ["leaf", "aggregation", "spine"]
    ]
    assert len(network_switches) > 0
    assert topology.switches_excluding_nvswitch == len(network_switches)


def test_topology_converter_gpu_and_nvswitch_ids():
    """GPU IDs and NVSwitch IDs are assigned in the correct ranges."""
    dc = make_datacenter(num_nodes=4, gpus_per_node=8)
    topology = TopologyConverter().convert(dc)

    total_gpus = 4 * 8
    total_nvswitches = 4 * 1  # 1 NVSwitch per node

    gpu_nodes = [n for n in topology.nodes if n.node_type == "gpu"]
    gpu_ids = [n.node_id for n in gpu_nodes]
    assert min(gpu_ids) == 0
    assert max(gpu_ids) == total_gpus - 1

    nvswitch_nodes = [n for n in topology.nodes if n.node_type == "nvswitch"]
    assert len(nvswitch_nodes) == total_nvswitches
    assert topology.nv_switch_num == total_nvswitches
    nvswitch_ids = [n.node_id for n in nvswitch_nodes]
    assert min(nvswitch_ids) == total_gpus
    assert max(nvswitch_ids) == total_gpus + total_nvswitches - 1


def test_topology_converter_fat_tree_structure():
    """Fat-tree k=4 generates the expected switch counts."""
    # k=4 fat-tree: 4 pods, 2 leaf + 2 agg per pod, 4 core switches
    dc = make_datacenter(num_nodes=8, gpus_per_node=8, fat_tree_k=4)
    topology = TopologyConverter().convert(dc)

    leaf_switches = [n for n in topology.nodes if n.node_type == "leaf"]
    agg_switches = [n for n in topology.nodes if n.node_type == "aggregation"]
    core_switches = [n for n in topology.nodes if n.node_type == "spine"]

    assert len(leaf_switches) == 8   # 4 pods * 2
    assert len(agg_switches) == 8    # 4 pods * 2
    assert len(core_switches) == 4   # (k/2)^2

    expected_total = 8 + 8 + 4
    assert topology.switches_excluding_nvswitch == expected_total
    assert len(leaf_switches + agg_switches + core_switches) == expected_total


def test_topology_converter_no_scale_up():
    """Topology with no scale_up has no NVSwitches."""
    dc = DatacenterConfig(
        datacenter=DatacenterMeta(name="test"),
        cluster=ClusterSpec(num_nodes=2),
        node=NodeSpec(
            gpus_per_node=8,
            gpu=GPUSpec(name="H100", memory_capacity_gb=80.0),
        ),
        network=NetworkSpec(
            scale_out=ScaleOutSpec(
                nic=NICSpec(speed="400Gbps", latency="0.005ms"),
                topology=TopologySpec(type=TopologyType.fat_tree, params={"k": 4}),
            ),
        ),
    )
    topology = TopologyConverter().convert(dc)

    assert topology.nv_switch_num == 0
    assert len([n for n in topology.nodes if n.node_type == "nvswitch"]) == 0


# ---------------------------------------------------------------------------
# Workload converter tests
# ---------------------------------------------------------------------------


def test_workload_converter_minimal():
    """Workload converter with simple TP=2 config."""
    dc = make_datacenter(num_nodes=2, gpus_per_node=4)
    wl = make_workload(num_gpus=8, tp=2, pp=1, ep=1, num_layers=2)

    trace = WorkloadConverter().convert(wl, dc)

    assert trace.model_parallel_npu_group == 2
    assert trace.pipeline_model_parallelism == 1
    assert trace.expert_parallel_npu_group == 1
    assert trace.vpp == 1
    assert trace.all_gpus == 8

    # DP = 8 / (2 * 1 * 1) = 4, GA = 1024 / (4 * 4) = 64
    assert trace.ga == 64

    assert trace.num_layers == 4  # 2 transformer layers * 2 sublayers
    assert len(trace.layers) == 4
    assert trace.layers[0].layer_id == "layer_0_attention"
    assert trace.layers[1].layer_id == "layer_0_mlp"
    assert trace.layers[2].layer_id == "layer_1_attention"
    assert trace.layers[3].layer_id == "layer_1_mlp"

    assert trace.layers[0].dependency == -1
    assert trace.layers[1].dependency == 0
    assert trace.layers[2].dependency == 1
    assert trace.layers[3].dependency == 2


def test_workload_converter_gradient_accumulation():
    """GA is correctly computed from GBS, MBS, and DP."""
    dc = make_datacenter(num_nodes=4, gpus_per_node=8)
    # total_gpus=32, tp=2, pp=2, ep=1 → dp=32/(2*2*1)=8
    # GA = 1024 / (8 * 4) = 32
    wl = make_workload(
        num_gpus=32, tp=2, pp=2, num_layers=2, global_batch_size=1024, micro_batch_size=4
    )
    trace = WorkloadConverter().convert(wl, dc)

    assert trace.model_parallel_npu_group == 2
    assert trace.pipeline_model_parallelism == 2
    assert trace.all_gpus == 32
    assert trace.ga == 32
    assert trace.num_layers == 4
    assert len(trace.layers) == 4


def test_workload_converter_tp_communication_types():
    """TP > 1 generates ALLGATHER_TP forward and REDUCESCATTER_TP input-grad collectives."""
    dc = make_datacenter(num_nodes=4, gpus_per_node=8)
    wl = make_workload(num_gpus=32, tp=2, pp=1, num_layers=1)

    trace = WorkloadConverter().convert(wl, dc)
    attn_layer = trace.layers[0]

    assert attn_layer.fwd_comm_type == "ALLGATHER_TP"
    assert attn_layer.ig_comm_type == "REDUCESCATTER_TP"

    # dp = 32 // 2 = 16 > 1
    assert attn_layer.wg_comm_type == "ALLREDUCE_DP"


def test_workload_converter_no_tp():
    """TP=1 produces no TP collectives but still has DP allreduce."""
    dc = make_datacenter(num_nodes=1, gpus_per_node=8)
    wl = make_workload(num_gpus=8, tp=1, pp=1, num_layers=1)

    trace = WorkloadConverter().convert(wl, dc)
    attn_layer = trace.layers[0]

    assert attn_layer.fwd_comm_type == "NONE"
    assert attn_layer.ig_comm_type == "NONE"
    assert attn_layer.wg_comm_type == "ALLREDUCE_DP"  # dp=8 > 1


def test_workload_converter_moe():
    """MoE workload uses the HYBRID_TRANSFORMER_FP8_MoE policy."""
    dc = make_datacenter(num_nodes=4, gpus_per_node=8)
    wl = make_workload(
        num_gpus=32, tp=2, pp=2, ep=2, num_layers=1, moe=True, num_experts=8
    )

    trace = WorkloadConverter().convert(wl, dc)

    assert trace.parallelism_policy == "HYBRID_TRANSFORMER_FP8_MoE"
    assert trace.expert_parallel_npu_group == 2


# ---------------------------------------------------------------------------
# Bandwidth / latency parsing
# ---------------------------------------------------------------------------


def test_bandwidth_parsing():
    assert _parse_bandwidth("900Gbps") == 900.0
    assert _parse_bandwidth("1Tbps") == 1000.0
    assert _parse_bandwidth("500Mbps") == 0.5
    assert _parse_bandwidth("100") == 100.0


def test_latency_parsing():
    assert _parse_latency("1ms") == 1_000_000
    assert _parse_latency("5us") == 5_000
    assert _parse_latency("100ns") == 100
    assert _parse_latency("1s") == 1_000_000_000
