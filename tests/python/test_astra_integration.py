"""Integration tests for ASTRA-Sim converters with C++ bindings."""

from simulon._sim import NetworkTopology, NetworkNode, NetworkLink, WorkloadTrace, LayerTrace
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


def test_topology_python_to_cpp():
    """Test that Python NetworkTopology can be passed to C++ bindings."""
    # Create a minimal datacenter config
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

    # Convert to NetworkTopology
    converter = TopologyConverter()
    py_topology = converter.convert(datacenter)

    # Create C++ NetworkTopology and copy data
    cpp_topology = NetworkTopology()
    cpp_topology.gpus_per_server = py_topology.gpus_per_server
    cpp_topology.nv_switch_num = py_topology.nv_switch_num
    cpp_topology.switches_excluding_nvswitch = py_topology.switches_excluding_nvswitch
    cpp_topology.gpu_type = py_topology.gpu_type

    # Copy nodes
    cpp_topology.nodes = []
    for node in py_topology.nodes:
        cpp_node = NetworkNode()
        cpp_node.node_id = node.node_id
        cpp_node.node_type = node.node_type
        cpp_topology.nodes.append(cpp_node)

    # Copy links
    cpp_topology.links = []
    for link in py_topology.links:
        cpp_link = NetworkLink()
        cpp_link.source = link.source
        cpp_link.dest = link.dest
        cpp_link.bandwidth_gbps = link.bandwidth_gbps
        cpp_link.latency_ns = link.latency_ns
        cpp_link.error_rate = link.error_rate
        cpp_topology.links.append(cpp_link)

    # Verify data was copied correctly
    assert cpp_topology.gpus_per_server == 4
    assert cpp_topology.gpu_type == "H100"
    assert len(cpp_topology.nodes) == len(py_topology.nodes)
    assert len(cpp_topology.links) == len(py_topology.links)


def test_workload_python_to_cpp():
    """Test that Python WorkloadTrace can be passed to C++ bindings."""
    # Create minimal configs
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

    # Convert to WorkloadTrace
    converter = WorkloadConverter()
    py_trace = converter.convert(workload, datacenter)

    # Create C++ WorkloadTrace and copy data
    cpp_trace = WorkloadTrace()
    cpp_trace.parallelism_policy = py_trace.parallelism_policy
    cpp_trace.model_parallel_npu_group = py_trace.model_parallel_npu_group
    cpp_trace.expert_parallel_npu_group = py_trace.expert_parallel_npu_group
    cpp_trace.pipeline_model_parallelism = py_trace.pipeline_model_parallelism
    cpp_trace.ga = py_trace.ga
    cpp_trace.vpp = py_trace.vpp
    cpp_trace.all_gpus = py_trace.all_gpus
    cpp_trace.num_layers = py_trace.num_layers

    # Copy layers
    cpp_trace.layers = []
    for layer in py_trace.layers:
        cpp_layer = LayerTrace()
        cpp_layer.layer_id = layer.layer_id
        cpp_layer.dependency = layer.dependency
        cpp_layer.fwd_compute_time_ns = layer.fwd_compute_time_ns
        cpp_layer.fwd_comm_type = layer.fwd_comm_type
        cpp_layer.fwd_comm_size_bytes = layer.fwd_comm_size_bytes
        cpp_layer.ig_compute_time_ns = layer.ig_compute_time_ns
        cpp_layer.ig_comm_type = layer.ig_comm_type
        cpp_layer.ig_comm_size_bytes = layer.ig_comm_size_bytes
        cpp_layer.wg_compute_time_ns = layer.wg_compute_time_ns
        cpp_layer.wg_comm_type = layer.wg_comm_type
        cpp_layer.wg_comm_size_bytes = layer.wg_comm_size_bytes
        cpp_layer.wg_update_time_ns = layer.wg_update_time_ns
        cpp_trace.layers.append(cpp_layer)

    # Verify data was copied correctly
    assert cpp_trace.parallelism_policy == "HYBRID_TRANSFORMER"
    assert cpp_trace.model_parallel_npu_group == 2
    assert cpp_trace.num_layers == 4  # 2 transformer layers * 2 sublayers
    assert len(cpp_trace.layers) == 4
