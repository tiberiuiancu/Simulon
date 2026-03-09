#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "astra_bindings/topology_bridge.hh"
#include "astra_bindings/workload_bridge.hh"

namespace py = pybind11;

PYBIND11_MODULE(_sim, m) {
    m.doc() = "simulon C++ simulation backend";

    // Topology bindings
    py::class_<simulon::astra::NetworkNode>(m, "NetworkNode")
        .def(py::init<>())
        .def_readwrite("node_id", &simulon::astra::NetworkNode::node_id)
        .def_readwrite("node_type", &simulon::astra::NetworkNode::node_type);

    py::class_<simulon::astra::NetworkLink>(m, "NetworkLink")
        .def(py::init<>())
        .def_readwrite("source", &simulon::astra::NetworkLink::source)
        .def_readwrite("dest", &simulon::astra::NetworkLink::dest)
        .def_readwrite("bandwidth_gbps", &simulon::astra::NetworkLink::bandwidth_gbps)
        .def_readwrite("latency_ns", &simulon::astra::NetworkLink::latency_ns)
        .def_readwrite("error_rate", &simulon::astra::NetworkLink::error_rate);

    py::class_<simulon::astra::NetworkTopology>(m, "NetworkTopology")
        .def(py::init<>())
        .def_readwrite("nodes", &simulon::astra::NetworkTopology::nodes)
        .def_readwrite("links", &simulon::astra::NetworkTopology::links)
        .def_readwrite("gpus_per_server", &simulon::astra::NetworkTopology::gpus_per_server)
        .def_readwrite("nv_switch_num", &simulon::astra::NetworkTopology::nv_switch_num)
        .def_readwrite("switches_excluding_nvswitch", &simulon::astra::NetworkTopology::switches_excluding_nvswitch)
        .def_readwrite("gpu_type", &simulon::astra::NetworkTopology::gpu_type);

    // Workload bindings
    py::class_<simulon::astra::LayerTrace>(m, "LayerTrace")
        .def(py::init<>())
        .def_readwrite("layer_id", &simulon::astra::LayerTrace::layer_id)
        .def_readwrite("dependency", &simulon::astra::LayerTrace::dependency)
        .def_readwrite("fwd_compute_time_ns", &simulon::astra::LayerTrace::fwd_compute_time_ns)
        .def_readwrite("fwd_comm_type", &simulon::astra::LayerTrace::fwd_comm_type)
        .def_readwrite("fwd_comm_size_bytes", &simulon::astra::LayerTrace::fwd_comm_size_bytes)
        .def_readwrite("ig_compute_time_ns", &simulon::astra::LayerTrace::ig_compute_time_ns)
        .def_readwrite("ig_comm_type", &simulon::astra::LayerTrace::ig_comm_type)
        .def_readwrite("ig_comm_size_bytes", &simulon::astra::LayerTrace::ig_comm_size_bytes)
        .def_readwrite("wg_compute_time_ns", &simulon::astra::LayerTrace::wg_compute_time_ns)
        .def_readwrite("wg_comm_type", &simulon::astra::LayerTrace::wg_comm_type)
        .def_readwrite("wg_comm_size_bytes", &simulon::astra::LayerTrace::wg_comm_size_bytes)
        .def_readwrite("wg_update_time_ns", &simulon::astra::LayerTrace::wg_update_time_ns);

    py::class_<simulon::astra::WorkloadTrace>(m, "WorkloadTrace")
        .def(py::init<>())
        .def_readwrite("parallelism_policy", &simulon::astra::WorkloadTrace::parallelism_policy)
        .def_readwrite("model_parallel_npu_group", &simulon::astra::WorkloadTrace::model_parallel_npu_group)
        .def_readwrite("expert_parallel_npu_group", &simulon::astra::WorkloadTrace::expert_parallel_npu_group)
        .def_readwrite("pipeline_model_parallelism", &simulon::astra::WorkloadTrace::pipeline_model_parallelism)
        .def_readwrite("ga", &simulon::astra::WorkloadTrace::ga)
        .def_readwrite("vpp", &simulon::astra::WorkloadTrace::vpp)
        .def_readwrite("all_gpus", &simulon::astra::WorkloadTrace::all_gpus)
        .def_readwrite("num_layers", &simulon::astra::WorkloadTrace::num_layers)
        .def_readwrite("layers", &simulon::astra::WorkloadTrace::layers);
}
