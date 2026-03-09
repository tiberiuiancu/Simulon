#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace simulon {
namespace astra {

struct LayerTrace {
    std::string layer_id;
    int dependency;
    uint64_t fwd_compute_time_ns;
    std::string fwd_comm_type;
    uint64_t fwd_comm_size_bytes;
    uint64_t ig_compute_time_ns;
    std::string ig_comm_type;
    uint64_t ig_comm_size_bytes;
    uint64_t wg_compute_time_ns;
    std::string wg_comm_type;
    uint64_t wg_comm_size_bytes;
    uint64_t wg_update_time_ns;
};

struct WorkloadTrace {
    std::string parallelism_policy;
    int model_parallel_npu_group;
    int expert_parallel_npu_group;
    int pipeline_model_parallelism;
    int ga;
    int vpp;
    int all_gpus;
    int num_layers;
    std::vector<LayerTrace> layers;
};

// Forward declaration for ASTRA-Sim types
// This will be implemented once we integrate with ASTRA-Sim internals
// namespace AstraSim {
//     class Workload;
//     class Sys;
// }
// AstraSim::Workload* createWorkload(
//     const WorkloadTrace& trace,
//     AstraSim::Sys* sys,
//     const std::string& run_name
// );

}  // namespace astra
}  // namespace simulon
