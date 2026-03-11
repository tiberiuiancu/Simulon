/**
 * ASTRA-Sim analytical backend runner - direct C++ integration.
 * No hardcoded paths, no subprocess calls, just direct library integration.
 */

#pragma once

#include <map>
#include <string>
#include <vector>

#include "topology_bridge.hh"
#include "workload_bridge.hh"

namespace simulon {
namespace astra {

/**
 * Results from an ASTRA-Sim analytical simulation run.
 */
struct AnalyticalResults {
    bool success;
    double total_time_ns;
    double compute_time_ns;
    double communication_time_ns;
    int completed_layers;
    std::string error_message;
    std::map<std::string, double> metrics;
};

/**
 * Run ASTRA-Sim analytical backend simulation directly.
 * No file I/O, no hardcoded paths - everything in memory.
 *
 * @param topology Network topology configuration
 * @param workload Workload trace
 * @return Simulation results
 */
AnalyticalResults run_analytical(
    const NetworkTopology& topology,
    const WorkloadTrace& workload
);

}  // namespace astra
}  // namespace simulon
