/**
 * ASTRA-Sim simulation runner wrapper for Python bindings.
 */

#pragma once

#include <map>
#include <string>
#include <vector>

namespace simulon {
namespace astra {

/**
 * Results from an ASTRA-Sim simulation run.
 */
struct SimulationResults {
    double total_time_ns;           // Total simulation time in nanoseconds
    double compute_time_ns;         // Total compute time
    double communication_time_ns;   // Total communication time
    int completed_layers;           // Number of layers completed
    std::string status;             // "success" or error message
    std::map<std::string, double> metrics;  // Additional metrics
};

/**
 * Run ASTRA-Sim analytical backend simulation.
 *
 * @param workload_path Path to workload file
 * @param result_path Path to output results directory
 * @param num_gpus Total number of GPUs
 * @param gpus_per_server GPUs per server node
 * @param gpu_type GPU type string ("A100", "H100", etc.)
 * @return Simulation results
 */
SimulationResults run_analytical_simulation(
    const std::string& workload_path,
    const std::string& result_path,
    int num_gpus,
    int gpus_per_server,
    const std::string& gpu_type
);

}  // namespace astra
}  // namespace simulon
