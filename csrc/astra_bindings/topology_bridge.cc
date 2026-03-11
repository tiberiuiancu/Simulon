#include "topology_bridge.hh"

#include <map>
#include <sstream>

namespace simulon {
namespace astra {

NetWorkParam toNetWorkParam(const NetworkTopology& topology) {
    NetWorkParam param;

    // Build a node_id → node_type lookup map
    std::map<int, std::string> node_types;
    for (const auto& node : topology.nodes) {
        node_types[node.node_id] = node.node_type;
    }

    // Collect NVSwitch node IDs and count non-NVSwitch switches
    uint32_t switch_count = 0;
    for (const auto& node : topology.nodes) {
        if (node.node_type == "nvswitch") {
            param.NVswitchs.push_back(node.node_id);
        } else if (node.node_type != "gpu") {
            ++switch_count;
        }
    }

    // Average nvlink_bw from GPU↔NVSwitch links
    // Average bw_per_nic from GPU↔leaf links
    float nvlink_bw_sum = 0.0f;
    int nvlink_link_count = 0;
    float nic_bw_sum = 0.0f;
    int nic_link_count = 0;

    for (const auto& link : topology.links) {
        auto src_it = node_types.find(link.source);
        auto dst_it = node_types.find(link.dest);
        if (src_it == node_types.end() || dst_it == node_types.end()) continue;

        const std::string& src_type = src_it->second;
        const std::string& dst_type = dst_it->second;

        if ((src_type == "gpu" && dst_type == "nvswitch") ||
            (src_type == "nvswitch" && dst_type == "gpu")) {
            nvlink_bw_sum += static_cast<float>(link.bandwidth_gbps);
            ++nvlink_link_count;
        }

        if ((src_type == "gpu" && dst_type == "leaf") ||
            (src_type == "leaf" && dst_type == "gpu")) {
            nic_bw_sum += static_cast<float>(link.bandwidth_gbps);
            ++nic_link_count;
        }
    }

    float nvlink_bw = (nvlink_link_count > 0) ? nvlink_bw_sum / nvlink_link_count : -1.0f;
    float nic_bw = (nic_link_count > 0) ? nic_bw_sum / nic_link_count : -1.0f;

    // Apply efficiency factors
    if (nvlink_bw > 0.0f) nvlink_bw *= topology.nvlink_bandwidth_efficiency;
    if (nic_bw > 0.0f) nic_bw *= topology.nic_bandwidth_efficiency;

    param.nvlink_bw = nvlink_bw;
    param.bw_per_nic = nic_bw;

    // Topology counts
    param.gpus_per_server = static_cast<uint32_t>(topology.gpus_per_server);
    param.nvswitch_num = static_cast<uint32_t>(param.NVswitchs.size());
    param.switch_num = switch_count;
    param.node_num = static_cast<uint32_t>(topology.nodes.size());
    param.link_num = static_cast<uint32_t>(topology.links.size());
    param.nics_per_server = 1;  // one NIC per server by default

    // Map gpu_type string → GPUType enum
    const std::string& gt = topology.gpu_type;
    if (gt == "H100")      param.gpu_type = GPUType::H100;
    else if (gt == "H800") param.gpu_type = GPUType::H800;
    else if (gt == "H20")  param.gpu_type = GPUType::H20;
    else if (gt == "A100") param.gpu_type = GPUType::A100;
    else if (gt == "A800") param.gpu_type = GPUType::A800;
    else                   param.gpu_type = GPUType::NONE;

    // Build all_gpus: group consecutive GPU IDs by server
    int total_gpus = 0;
    for (const auto& node : topology.nodes) {
        if (node.node_type == "gpu") ++total_gpus;
    }
    int gpus = topology.gpus_per_server > 0 ? topology.gpus_per_server : 1;
    int num_servers = total_gpus / gpus;
    for (int s = 0; s < num_servers; ++s) {
        std::vector<int> server_gpus;
        for (int g = 0; g < gpus; ++g) {
            server_gpus.push_back(s * gpus + g);
        }
        param.all_gpus.push_back(server_gpus);
    }

    return param;
}

std::string toTopoFileContent(const NetworkTopology& topology) {
    // Build node type lookup
    std::map<int, std::string> node_types;
    for (const auto& node : topology.nodes) {
        node_types[node.node_id] = node.node_type;
    }

    // Collect switch IDs (non-GPU nodes, NVSwitches first then others)
    std::vector<int> all_switch_ids;
    for (const auto& node : topology.nodes) {
        if (node.node_type == "nvswitch") all_switch_ids.push_back(node.node_id);
    }
    for (const auto& node : topology.nodes) {
        if (node.node_type != "gpu" && node.node_type != "nvswitch") {
            all_switch_ids.push_back(node.node_id);
        }
    }

    int total_gpus = 0;
    for (const auto& node : topology.nodes) {
        if (node.node_type == "gpu") ++total_gpus;
    }

    int nv_switch_num = 0;
    for (const auto& node : topology.nodes) {
        if (node.node_type == "nvswitch") ++nv_switch_num;
    }

    int other_switches = static_cast<int>(all_switch_ids.size()) - nv_switch_num;
    int node_count = static_cast<int>(topology.nodes.size());
    int link_count = static_cast<int>(topology.links.size());

    std::ostringstream oss;

    // Header line: node_count gpus_per_server nv_switch_num other_switches link_count gpu_type
    oss << node_count << " "
        << topology.gpus_per_server << " "
        << nv_switch_num << " "
        << other_switches << " "
        << link_count << " "
        << topology.gpu_type << "\n";

    // Switch IDs line
    for (int i = 0; i < static_cast<int>(all_switch_ids.size()); ++i) {
        if (i > 0) oss << " ";
        oss << all_switch_ids[i];
    }
    oss << "\n";

    // Links: src dst bw_gbps latency_ns error_rate
    for (const auto& link : topology.links) {
        oss << link.source << " "
            << link.dest << " "
            << link.bandwidth_gbps << " "
            << link.latency_ns << " "
            << link.error_rate << "\n";
    }

    return oss.str();
}

}  // namespace astra
}  // namespace simulon
