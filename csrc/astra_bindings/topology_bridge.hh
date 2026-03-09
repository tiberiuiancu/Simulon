#pragma once

#include <string>
#include <vector>

namespace simulon {
namespace astra {

struct NetworkNode {
    int node_id;
    std::string node_type;
};

struct NetworkLink {
    int source;
    int dest;
    double bandwidth_gbps;
    double latency_ns;
    double error_rate;
};

struct NetworkTopology {
    std::vector<NetworkNode> nodes;
    std::vector<NetworkLink> links;
    int gpus_per_server;
    int nv_switch_num;
    int switches_excluding_nvswitch;
    std::string gpu_type;
    float nvlink_bandwidth_efficiency;  // Scale-up (intra-node) efficiency
    float nic_bandwidth_efficiency;     // Scale-out (inter-node) efficiency
};

// Forward declaration for ASTRA-Sim type
// This will be implemented once we integrate with ASTRA-Sim internals
// struct NetWorkParam;
// NetWorkParam toNetWorkParam(const NetworkTopology& topology);

}  // namespace astra
}  // namespace simulon
