/*
 * pybind11 bindings for MockNcclGroup topology functions.
 *
 * Exposes genringchannels, gettreechannels, get_nvls_tree_channels to Python
 * so that tree.py and nvls.py can query channel topology without a full
 * astra-sim runtime.
 *
 * RingChannels    = map<channel_id, map<rank, [prev, next, node_recv, node_send]>>
 * TreeChannels    = map<channel_id, map<rank, ncclTree{depth,rank,up,down[]}>>
 * NVLStreechannels = map<channel_id, map<rank, list[ncclChannelNode]>> (pointer tree flattened)
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "mocknccl/MockNcclGroup.h"
#include "mocknccl/MockNcclChannel.h"

namespace py = pybind11;
using namespace MockNccl;

// Helper: flatten NVLStreechannels (pointer-based) into a serializable form.
// Returns map<channel_id, map<rank, {depth, rank, up_rank, down_ranks[]}>>.
// up_rank = -1 if no parent (root).
using FlatNVLSChannels = std::map<int, std::map<int, std::tuple<int,int,int,std::vector<int>>>>;

static FlatNVLSChannels flatten_nvls_tree(const NVLStreechannels& nvls) {
    FlatNVLSChannels result;
    for (auto& [ch_id, rank_map] : nvls) {
        for (auto& [rank, nodes] : rank_map) {
            for (auto* node : nodes) {
                if (!node) continue;
                int up_rank = node->up ? node->up->rank : -1;
                std::vector<int> down_ranks;
                for (auto* child : node->down) {
                    if (child) down_ranks.push_back(child->rank);
                }
                result[ch_id][node->rank] = {node->depth, node->rank, up_rank, down_ranks};
            }
        }
    }
    return result;
}

PYBIND11_MODULE(_mocknccl, m) {
    m.doc() = "MockNcclGroup topology bindings for simulon";

    py::enum_<GPUType>(m, "GPUType")
        .value("H100", GPUType::H100)
        .value("H800", GPUType::H800)
        .value("H20",  GPUType::H20)
        .value("A100", GPUType::A100)
        .value("A800", GPUType::A800)
        .value("NONE", GPUType::NONE)
        .export_values();

    py::enum_<GroupType>(m, "GroupType")
        .value("TP",    GroupType::TP)
        .value("DP",    GroupType::DP)
        .value("PP",    GroupType::PP)
        .value("EP",    GroupType::EP)
        .value("DP_EP", GroupType::DP_EP)
        .value("NONE",  GroupType::NONE)
        .export_values();

    py::class_<ncclTree>(m, "NcclTree")
        .def_readonly("depth", &ncclTree::depth)
        .def_readonly("rank",  &ncclTree::rank)
        .def_readonly("up",    &ncclTree::up)
        .def_readonly("down",  &ncclTree::down);

    py::class_<MockNcclGroup>(m, "MockNcclGroup")
        .def(py::init<int,int,int,int,int,int,int,std::vector<int>,GPUType>(),
             py::arg("ngpus"),
             py::arg("gpus_per_node"),
             py::arg("tp_size"),
             py::arg("dp_size"),
             py::arg("pp_size"),
             py::arg("ep_size"),
             py::arg("dp_ep_size"),
             py::arg("nvswitches"),
             py::arg("gpu_type"))
        // RingChannels: map<ch_id, map<rank, [prev, next, node_recv, node_send]>>
        .def("genringchannels",
             &MockNcclGroup::genringchannels,
             py::arg("rank"), py::arg("type"))
        // TreeChannels: map<ch_id, map<rank, ncclTree>>
        .def("gettreechannels",
             &MockNcclGroup::gettreechannels,
             py::arg("rank"), py::arg("type"))
        // NVLStreechannels flattened: map<ch_id, map<rank, (depth,rank,up_rank,down_ranks)>>
        .def("get_nvls_tree_channels",
             [](MockNcclGroup& self, int rank, GroupType type) {
                 return flatten_nvls_tree(self.get_nvls_tree_channels(rank, type));
             },
             py::arg("rank"), py::arg("type"));
}
