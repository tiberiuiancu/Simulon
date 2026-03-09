/**
 * Direct workload construction from structured data.
 * Eliminates unnecessary text serialization/parsing.
 */

#include "workload_bridge.hh"

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

// ASTRA-Sim includes
#include "astra-sim/system/Common.hh"
#include "astra-sim/system/MockNcclGroup.h"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/workload/Layer.hh"
#include "astra-sim/workload/Workload.hh"

namespace simulon {
namespace astra {

namespace {

using namespace AstraSim;

/**
 * Convert comm type string to ASTRA-Sim ComType enum and GroupType.
 * Examples: "ALLREDUCE" -> (All_Reduce, DP)
 *           "ALLGATHER_TP" -> (All_Gather, TP)
 */
std::pair<ComType, MockNccl::GroupType> parse_comm_type(
    const std::string& comm_type_str,
    bool is_weight_grad) {

  if (comm_type_str == "NONE" || comm_type_str.empty()) {
    return {ComType::None, MockNccl::GroupType::NONE};
  }

  // Determine base communication type
  ComType comm_type = ComType::None;
  if (comm_type_str.find("ALLREDUCE") == 0) {
    comm_type = ComType::All_Reduce;
  } else if (comm_type_str.find("ALLGATHER") == 0) {
    comm_type = ComType::All_Gather;
  } else if (comm_type_str.find("REDUCESCATTER") == 0) {
    comm_type = ComType::Reduce_Scatter;
  } else if (comm_type_str.find("ALLTOALL") == 0) {
    comm_type = ComType::All_to_All;
  } else if (comm_type_str.find("ALLREDUCEALLTOALL") == 0) {
    comm_type = ComType::All_Reduce_All_to_All;
  }

  // Determine group type from suffix
  MockNccl::GroupType group_type = MockNccl::GroupType::NONE;

  if (comm_type_str.find("_DP_EP") != std::string::npos) {
    group_type = MockNccl::GroupType::DP_EP;
  } else if (comm_type_str.find("_EP") != std::string::npos) {
    group_type = MockNccl::GroupType::EP;
  } else if (comm_type_str.find("_TP") != std::string::npos) {
    group_type = MockNccl::GroupType::TP;
  } else if (comm_type_str.find("_DP") != std::string::npos) {
    group_type = MockNccl::GroupType::DP;
  } else {
    // No suffix - use defaults based on phase
    if (is_weight_grad) {
      group_type = MockNccl::GroupType::DP;  // Weight gradients are data parallel
    } else {
      group_type = MockNccl::GroupType::TP;  // Forward/input grads are tensor parallel
    }
  }

  return {comm_type, group_type};
}

/**
 * Convert parallelism policy string to enum.
 */
ParallelismPolicy parse_parallelism_policy(const std::string& policy) {
  if (policy == "DATA") {
    return ParallelismPolicy::Data;
  } else if (policy == "TRANSFORMER" || policy == "transformer" ||
             policy == "HYBRID_TRANSFORMER") {
    return ParallelismPolicy::Transformer;
  } else if (policy == "HYBRID_TRANSFORMER_FWD_IN_BCKWD") {
    return ParallelismPolicy::TransformerFwdInBckwd;
  } else if (policy == "MODEL") {
    return ParallelismPolicy::Model;
  } else if (policy == "HYBRID_DATA_MODEL" || policy == "hybrid") {
    return ParallelismPolicy::HybridDataModel;
  } else if (policy == "HYBRID_MODEL_DATA") {
    return ParallelismPolicy::HybridModelData;
  } else if (policy == "HYBRID_DLRM") {
    return ParallelismPolicy::DLRM;
  } else if (policy == "HYBRID_DLRM_ENHANCED") {
    return ParallelismPolicy::DLRMEnhanced;
  } else if (policy == "HYBRID_CUSTOMIZED") {
    return ParallelismPolicy::HybridCustomized;
  } else if (policy == "DISTRIBUTED_INFERENCE") {
    return ParallelismPolicy::DistributedInference;
  }
  return ParallelismPolicy::None;
}

/**
 * Get involved dimensions for a collective based on parallelism policy.
 * This determines which network dimensions participate in the collective.
 */
std::map<std::string, std::vector<bool>> get_involved_dimensions(
    ParallelismPolicy policy,
    int model_parallel_npu_group) {

  std::map<std::string, std::vector<bool>> result;

  // For now, simplified: all dimensions participate
  // In real ASTRA-Sim, this depends on topology and parallelism strategy
  std::vector<bool> all_dims = {true};  // Single dimension for now

  result["fwd"] = all_dims;
  result["ig"] = all_dims;
  result["wg"] = all_dims;

  return result;
}

}  // namespace

/**
 * Directly construct Workload with Layers from structured WorkloadTrace.
 * No text serialization/parsing needed!
 */
void initialize_workload_direct(
    Workload* workload,
    Sys* sys,
    const WorkloadTrace& trace) {

  // Set parallelism parameters
  workload->parallelismPolicy = parse_parallelism_policy(trace.parallelism_policy);
  workload->model_parallel_npu_group = trace.model_parallel_npu_group;
  workload->expert_parallel_npu_group = trace.expert_parallel_npu_group;
  workload->pipeline_model_parallelism = trace.pipeline_model_parallelism;
  workload->GA = trace.ga;
  workload->vpp = trace.vpp;
  workload->all_gpus = trace.all_gpus;
  workload->pp_commsize = 0;  // Will be set if PP > 1

  // Get involved dimensions for this parallelism strategy
  auto general_involved_dims = get_involved_dimensions(
      workload->parallelismPolicy,
      workload->model_parallel_npu_group);

  // Set run_type (parallelism policy as string)
  workload->run_type = trace.parallelism_policy;

  // Create layers array
  workload->SIZE = trace.num_layers;
  workload->layers = new Layer*[workload->SIZE];

  // Directly construct each layer from trace
  for (int i = 0; i < trace.num_layers; i++) {
    const auto& layer_trace = trace.layers[i];

    // Parse communication types
    auto fwd_parsed = parse_comm_type(layer_trace.fwd_comm_type, false);
    ComType fwd_comm_type = fwd_parsed.first;
    MockNccl::GroupType fwd_group_type = fwd_parsed.second;

    auto ig_parsed = parse_comm_type(layer_trace.ig_comm_type, false);
    ComType ig_comm_type = ig_parsed.first;
    MockNccl::GroupType ig_group_type = ig_parsed.second;

    auto wg_parsed = parse_comm_type(layer_trace.wg_comm_type, true);
    ComType wg_comm_type = wg_parsed.first;
    MockNccl::GroupType wg_group_type = wg_parsed.second;

    // Create layer directly - no text parsing!
    workload->layers[i] = new Layer(
        layer_trace.layer_id,
        i,
        sys,
        workload,
        // Forward pass
        layer_trace.fwd_compute_time_ns * sys->compute_scale,
        fwd_comm_type,
        fwd_group_type,
        layer_trace.fwd_comm_size_bytes * sys->comm_scale,
        general_involved_dims["fwd"],
        // Input gradient
        layer_trace.ig_compute_time_ns * sys->compute_scale,
        ig_comm_type,
        ig_group_type,
        layer_trace.ig_comm_size_bytes * sys->comm_scale,
        general_involved_dims["ig"],
        // Weight gradient
        layer_trace.wg_compute_time_ns * sys->compute_scale,
        wg_comm_type,
        wg_group_type,
        layer_trace.wg_comm_size_bytes * sys->comm_scale,
        general_involved_dims["wg"],
        layer_trace.wg_update_time_ns,
        ParallelismPolicy::None  // specific policy
    );
  }

  workload->initialized = true;
}

}  // namespace astra
}  // namespace simulon
