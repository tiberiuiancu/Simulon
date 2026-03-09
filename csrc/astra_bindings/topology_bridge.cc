#include "topology_bridge.hh"

// This file will contain the implementation of toNetWorkParam()
// once we integrate with ASTRA-Sim internals.
//
// The conversion will:
// 1. Map gpu_type string to GPUType enum
// 2. Populate node_num, switch_num, link_num, etc.
// 3. Build NVswitchs vector from nodes with type "nvswitch"
// 4. Create link adjacency structures for ASTRA-Sim

namespace simulon {
namespace astra {

// Placeholder implementation - will be filled in Phase 2
// NetWorkParam toNetWorkParam(const NetworkTopology& topology) {
//     // TODO: Implement conversion
// }

}  // namespace astra
}  // namespace simulon
