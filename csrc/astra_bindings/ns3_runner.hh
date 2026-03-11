#pragma once
#include "analytical_runner.hh"  // reuse AnalyticalResults
#include "topology_bridge.hh"
#include "workload_bridge.hh"

namespace simulon {
namespace astra {

// Same return type as analytical runner
AnalyticalResults run_ns3(
    const NetworkTopology& topology,
    const WorkloadTrace& workload
);

}  // namespace astra
}  // namespace simulon
