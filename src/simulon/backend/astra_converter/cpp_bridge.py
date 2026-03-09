"""Bridge between Python dataclasses and C++ bound types."""

try:
    from simulon._sim import (
        NetworkTopology as CppNetworkTopology,
        NetworkNode as CppNetworkNode,
        NetworkLink as CppNetworkLink,
        WorkloadTrace as CppWorkloadTrace,
        LayerTrace as CppLayerTrace,
    )

    CPP_BINDINGS_AVAILABLE = True
except ImportError:
    CPP_BINDINGS_AVAILABLE = False
    CppNetworkTopology = None  # type: ignore
    CppNetworkNode = None  # type: ignore
    CppNetworkLink = None  # type: ignore
    CppWorkloadTrace = None  # type: ignore
    CppLayerTrace = None  # type: ignore

from simulon.backend.astra_converter.topology import (
    NetworkTopology,
    NetworkNode,
    NetworkLink,
)
from simulon.backend.astra_converter.workload import WorkloadTrace, LayerTrace


def to_cpp_topology(py_topo: NetworkTopology) -> "CppNetworkTopology":
    """Convert Python NetworkTopology dataclass to C++ bound type."""
    if not CPP_BINDINGS_AVAILABLE:
        raise ImportError("C++ bindings not available")

    cpp_topo = CppNetworkTopology()
    cpp_topo.gpus_per_server = py_topo.gpus_per_server
    cpp_topo.nv_switch_num = py_topo.nv_switch_num
    cpp_topo.switches_excluding_nvswitch = py_topo.switches_excluding_nvswitch
    cpp_topo.gpu_type = py_topo.gpu_type
    cpp_topo.nvlink_bandwidth_efficiency = py_topo.nvlink_bandwidth_efficiency
    cpp_topo.nic_bandwidth_efficiency = py_topo.nic_bandwidth_efficiency

    # Convert nodes
    for node in py_topo.nodes:
        cpp_node = CppNetworkNode()
        cpp_node.node_id = node.node_id
        cpp_node.node_type = node.node_type
        cpp_topo.nodes.append(cpp_node)

    # Convert links
    for link in py_topo.links:
        cpp_link = CppNetworkLink()
        cpp_link.source = link.source
        cpp_link.dest = link.dest
        cpp_link.bandwidth_gbps = link.bandwidth_gbps
        cpp_link.latency_ns = link.latency_ns
        cpp_link.error_rate = link.error_rate
        cpp_topo.links.append(cpp_link)

    return cpp_topo


def to_cpp_workload(py_workload: WorkloadTrace) -> "CppWorkloadTrace":
    """Convert Python WorkloadTrace dataclass to C++ bound type."""
    if not CPP_BINDINGS_AVAILABLE:
        raise ImportError("C++ bindings not available")

    cpp_workload = CppWorkloadTrace()
    cpp_workload.parallelism_policy = py_workload.parallelism_policy
    cpp_workload.model_parallel_npu_group = py_workload.model_parallel_npu_group
    cpp_workload.expert_parallel_npu_group = py_workload.expert_parallel_npu_group
    cpp_workload.pipeline_model_parallelism = py_workload.pipeline_model_parallelism
    cpp_workload.ga = py_workload.ga
    cpp_workload.vpp = py_workload.vpp
    cpp_workload.all_gpus = py_workload.all_gpus
    cpp_workload.num_layers = py_workload.num_layers

    # Convert layers
    for layer in py_workload.layers:
        cpp_layer = CppLayerTrace()
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
        cpp_workload.layers.append(cpp_layer)

    return cpp_workload
