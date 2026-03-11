/**
 * ASTRA-Sim analytical backend runner - pure in-memory implementation.
 * No file I/O, no hardcoded paths, no side effects.
 */

#include "analytical_runner.hh"
#include "workload_bridge.hh"
#include "topology_bridge.hh"

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
#include "astra-sim/system/AstraParamParse.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/workload/Workload.hh"

namespace simulon {
namespace astra {

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
        // Build NetWorkParam from topology graph and populate UserParam singleton
        NetWorkParam net_param = toNetWorkParam(topology);
        UserParam::getInstance()->net_work_param = net_param;

        // Create physical dimensions for ASTRA-Sim
        std::vector<int> physical_dims;
        physical_dims.push_back(workload.all_gpus);

        std::vector<int> queues_per_dim(physical_dims.size(), 1);

        // Create analytical network backend
        AnalyticalNetWork* analytical_network = new AnalyticalNetWork(0);

        // Create ASTRA-Sim system with "DIRECT_INIT" sentinel value
        // This tells Workload constructor to skip text parsing
        AstraSim::Sys* system = nullptr;
        try {
            system = new AstraSim::Sys(
            analytical_network,       // network backend
            nullptr,                  // memory API
            0,                        // id
            0,                        // npu_offset
            1,                        // num_passes
            physical_dims,            // physical dimensions
            queues_per_dim,           // queues per dimension
            "",                       // sys name
            "",                       // workload path (unused)
            1.0,                      // comm_scale
            1.0,                      // compute_scale
            1.0,                      // injection_scale
            1,                        // total_stat_rows
            0,                        // stat_row
            "",                       // path (empty = no output)
            "",                       // run_name (empty = no output)
            false,                    // separate_log (disable logging)
            false,                    // rendezvous_enabled
            net_param.gpu_type,       // gpu_type from topology
            physical_dims,            // all_gpus (flat list)
            net_param.NVswitchs,      // NVSwitch IDs from topology
            topology.gpus_per_server, // ngpus_per_node
            "DIRECT_INIT"             // sentinel: skip text parsing
            );
        } catch (const std::exception& e) {
            results.error_message = std::string("Failed to create Sys: ") + e.what();
            return results;
        }

        // Now directly initialize workload from structured data
        // No text serialization/parsing!
        initialize_workload_direct(system->workload, system, workload);

        // Set additional system parameters
        system->num_gpus = workload.all_gpus;

        // Fire the workload to start simulation
        if (system->workload && system->workload->initialized) {
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
