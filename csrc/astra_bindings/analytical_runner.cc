/**
 * ASTRA-Sim analytical backend runner - pure in-memory implementation.
 * No file I/O, no hardcoded paths, no side effects.
 */

#include "analytical_runner.hh"

#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

// Undefine hardcoded paths before including ASTRA-Sim headers
#ifdef LOG_PATH
#undef LOG_PATH
#endif
#define LOG_PATH ""

// ASTRA-Sim includes (paths relative to csrc/)
#include "astra-sim/network_frontend/analytical/AnalyticalNetwork.h"
#include "astra-sim/network_frontend/analytical/AnaSim.h"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/workload/Workload.hh"

namespace simulon {
namespace astra {

namespace {

/**
 * Create in-memory workload from our WorkloadTrace.
 * Returns a temporary file content that Workload class can parse.
 */
std::string create_workload_content(const WorkloadTrace& workload) {
    std::ostringstream oss;

    // Line 1: Parallelism policy and parameters
    // Map our policy names to ASTRA-Sim names
    // Valid: DATA, MODEL, HYBRID_DATA_MODEL, HYBRID_MODEL_DATA, TRANSFORMER, TRANSFORMERFWDINBCKWD
    std::string policy = "TRANSFORMER";
    if (workload.parallelism_policy == "HYBRID_TRANSFORMER" ||
        workload.parallelism_policy == "hybrid") {
        policy = "HYBRID_DATA_MODEL";
    } else if (workload.parallelism_policy == "transformer") {
        policy = "TRANSFORMER";
    }

    oss << policy << " "
        << "model_parallel_NPU_group: " << workload.model_parallel_npu_group << " "
        << "pp: " << workload.pipeline_model_parallelism << " "
        << "ep: " << workload.expert_parallel_npu_group << " "
        << "vpp: " << workload.vpp << " "
        << "ga: " << workload.ga << " "
        << "all_gpus: " << workload.all_gpus << "\n";

    // Line 2: Number of layers
    oss << workload.num_layers << "\n";

    // Lines 3+: Layer specifications
    for (const auto& layer : workload.layers) {
        oss << layer.layer_id << " "
            << layer.dependency << " "
            << layer.fwd_compute_time_ns << " "
            << layer.fwd_comm_type << " "
            << layer.fwd_comm_size_bytes << " "
            << layer.ig_compute_time_ns << " "
            << layer.ig_comm_type << " "
            << layer.ig_comm_size_bytes << " "
            << layer.wg_compute_time_ns << " "
            << layer.wg_comm_type << " "
            << layer.wg_comm_size_bytes << " "
            << layer.wg_update_time_ns << "\n";
    }

    return oss.str();
}

}  // namespace

AnalyticalResults run_analytical(
    const NetworkTopology& topology,
    const WorkloadTrace& workload
) {
    AnalyticalResults results;
    results.success = false;
    results.total_time_ns = 0.0;
    results.compute_time_ns = 0.0;
    results.communication_time_ns = 0.0;
    results.completed_layers = 0;

    try {
        // Create physical dimensions for ASTRA-Sim
        std::vector<int> physical_dims;
        int total_nodes = topology.gpus_per_server > 0 ?
            (workload.all_gpus / topology.gpus_per_server) : 1;
        physical_dims.push_back(workload.all_gpus);

        std::vector<int> queues_per_dim(physical_dims.size(), 1);

        // Create analytical network backend
        AnalyticalNetWork* analytical_network = new AnalyticalNetWork(0);

        // Create workload content in memory (no file I/O)
        std::string workload_content = create_workload_content(workload);

        // Create ASTRA-Sim system (fully in-memory, no file I/O)
        AstraSim::Sys* system = nullptr;
        try {
            system = new AstraSim::Sys(
            analytical_network,  // network backend
            nullptr,             // memory API
            0,                   // id
            0,                   // npu_offset
            1,                   // num_passes
            physical_dims,       // physical dimensions
            queues_per_dim,      // queues per dimension
            "",                  // sys name
            "",                  // workload path (unused - using content instead)
            1.0,                 // comm_scale
            1.0,                 // compute_scale
            1.0,                 // injection_scale
            1,                   // total_stat_rows
            0,                   // stat_row
            "",                  // path (empty = no output)
            "",                  // run_name (empty = no output)
            false,               // separate_log (disable logging)
            false,               // rendezvous_enabled
            GPUType::H100,       // gpu_type (map from topology.gpu_type)
            physical_dims,       // all_gpus
            {},                  // NVSwitchs (empty for now)
            topology.gpus_per_server,  // ngpus_per_node
            workload_content     // workload content (in-memory)
            );
        } catch (const std::exception& e) {
            results.error_message = std::string("Failed to create Sys: ") + e.what();
            return results;
        }

        // Set additional system parameters
        system->num_gpus = workload.all_gpus;

        // Fire the workload to start simulation
        if (system->workload) {
            system->workload->fire();

            // Run the analytical simulation
            AnaSim::Run();

            // Get simulation time
            results.total_time_ns = static_cast<double>(AnaSim::Now());

            // Stop and cleanup simulation
            AnaSim::Stop();
            AnaSim::Destroy();

            results.success = true;
            results.completed_layers = workload.num_layers;
        } else {
            results.error_message = "Failed to initialize workload";
        }

        // Note: Don't delete system/network here as they may have cleanup in destructors

    } catch (const std::exception& e) {
        results.success = false;
        results.error_message = std::string("Simulation error: ") + e.what();
    } catch (...) {
        results.success = false;
        results.error_message = "Unknown simulation error";
    }

    return results;
}

}  // namespace astra
}  // namespace simulon
