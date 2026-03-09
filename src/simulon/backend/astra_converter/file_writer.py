"""Write ASTRA-Sim input files from our data structures."""

import os
from pathlib import Path
from typing import Optional

try:
    from simulon._sim import NetworkTopology, WorkloadTrace
except ImportError:
    # Fallback for when C++ bindings aren't available
    NetworkTopology = None  # type: ignore
    WorkloadTrace = None  # type: ignore


def write_workload_file(
    trace: "WorkloadTrace",
    output_path: Path,
) -> None:
    """Write a workload trace to ASTRA-Sim format.

    Args:
        trace: WorkloadTrace object from converter
        output_path: Path to write the workload file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Map our parallelism policies to ASTRA-Sim names
    policy_map = {
        "transformer": "TRANSFORMER",
        "transformer_fwd_in_bckwd": "TRANSFORMERFWDINBCKWD",
        "data": "DATA",
        "model": "MODEL",
        "hybrid": "HYBRID",
    }
    policy = policy_map.get(trace.parallelism_policy.lower(), "TRANSFORMER")

    with open(output_path, "w") as f:
        # Line 1: Parallelism policy and parameters
        f.write(
            f"{policy} "
            f"model_parallel_NPU_group: {trace.model_parallel_npu_group} "
            f"pp: {trace.pipeline_model_parallelism} "
            f"ep: {trace.expert_parallel_npu_group} "
            f"vpp: {trace.vpp} "
            f"ga: {trace.ga} "
            f"all_gpus: {trace.all_gpus}\n"
        )

        # Line 2: Number of layers
        f.write(f"{trace.num_layers}\n")

        # Lines 3+: Layer specifications
        for layer in trace.layers:
            f.write(
                f"{layer.layer_id} "
                f"{layer.dependency} "
                f"{layer.fwd_compute_time_ns} "
                f"{layer.fwd_comm_type} "
                f"{layer.fwd_comm_size_bytes} "
                f"{layer.ig_compute_time_ns} "
                f"{layer.ig_comm_type} "
                f"{layer.ig_comm_size_bytes} "
                f"{layer.wg_compute_time_ns} "
                f"{layer.wg_comm_type} "
                f"{layer.wg_comm_size_bytes} "
                f"{layer.wg_update_time_ns}\n"
            )


def write_topology_file(
    topology: "NetworkTopology",
    output_path: Path,
) -> None:
    """Write network topology to ASTRA-Sim format.

    Args:
        topology: NetworkTopology object from converter
        output_path: Path to write the topology file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        # Write topology header
        f.write(f"# Topology for {topology.gpu_type}\n")
        f.write(f"# GPUs per server: {topology.gpus_per_server}\n")
        f.write(f"# NVSwitches: {topology.nv_switch_num}\n")
        f.write(f"# Network switches: {topology.switches_excluding_nvswitch}\n\n")

        # Write nodes
        f.write(f"{len(topology.nodes)}\n")  # Number of nodes
        for node in topology.nodes:
            f.write(f"{node.node_id} {node.node_type}\n")

        # Write links
        f.write(f"\n{len(topology.links)}\n")  # Number of links
        for link in topology.links:
            f.write(
                f"{link.source} {link.dest} "
                f"{link.bandwidth_gbps} {link.latency_ns} "
                f"{link.error_rate}\n"
            )


def create_astra_sim_config(
    workload_path: Path,
    topology_path: Optional[Path],
    result_dir: Path,
    num_gpus: int,
    gpus_per_server: int,
    gpu_type: str = "H100",
) -> dict:
    """Create configuration dict for ASTRA-Sim runner.

    Args:
        workload_path: Path to workload file
        topology_path: Path to topology file (None for analytical backend)
        result_dir: Directory for simulation results
        num_gpus: Total number of GPUs
        gpus_per_server: GPUs per server node
        gpu_type: GPU type (A100, H100, etc.)

    Returns:
        Configuration dictionary
    """
    result_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "workload": str(workload_path),
        "result_path": str(result_dir),
        "num_gpus": num_gpus,
        "gpus_per_server": gpus_per_server,
        "gpu_type": gpu_type,
    }

    if topology_path:
        config["topology"] = str(topology_path)

    return config
