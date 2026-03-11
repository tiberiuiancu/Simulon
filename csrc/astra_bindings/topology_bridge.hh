#pragma once

#include <string>
#include <vector>

#include "astra-sim/system/AstraParamParse.hh"

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

/**
 * Convert a NetworkTopology to a NetWorkParam suitable for passing to ASTRA-Sim.
 * Derives nvlink_bw, bw_per_nic, NVSwitchs, all_gpus, etc. from the topology graph.
 */
NetWorkParam toNetWorkParam(const NetworkTopology& topology);

/**
 * Serialize a NetworkTopology to the SimAI text topology file format.
 * Used by the NS3 backend which requires a file on disk.
 */
std::string toTopoFileContent(const NetworkTopology& topology);

}  // namespace astra
}  // namespace simulon
