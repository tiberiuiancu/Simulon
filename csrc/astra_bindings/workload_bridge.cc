#include "workload_bridge.hh"

// This file will contain the implementation of createWorkload()
// once we integrate with ASTRA-Sim internals.
//
// The conversion will:
// 1. Parse parallelism_policy string to ParallelismPolicy enum
// 2. For each LayerTrace, create AstraSim::Layer object
// 3. Parse comm_type strings ("ALLGATHER_TP") to ComType + GroupType
// 4. Decode involved dimensions for each collective
// 5. Construct Layer with compute times and comm parameters

namespace simulon {
namespace astra {

// Placeholder implementation - will be filled in Phase 2
// AstraSim::Workload* createWorkload(
//     const WorkloadTrace& trace,
//     AstraSim::Sys* sys,
//     const std::string& run_name
// ) {
//     // TODO: Implement conversion
// }

}  // namespace astra
}  // namespace simulon
