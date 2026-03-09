"""ASTRA-Sim backend implementation for simulon.

ASTRA-Sim is the core simulation engine used by all backends. The difference between
backends is the network simulation approach:
- Analytical: Fast analytical network model (queueing theory)
- NS-3: Detailed packet-level network simulation

This module provides the base AstraSimBackend class that can use either network backend.
"""

from enum import Enum
from typing import Literal

from simulon.backend.astra_converter import TopologyConverter, WorkloadConverter
from simulon.backend.base import Backend
from simulon.config.scenario import ScenarioConfig
from simulon.config.workload import MegatronWorkload


class NetworkBackend(str, Enum):
    """Network simulation backend types."""

    ANALYTICAL = "analytical"
    NS3 = "ns3"


class AstraSimBackend(Backend):
    """Backend that uses ASTRA-Sim for distributed training simulation.

    All simulon backends are based on ASTRA-Sim, but differ in their network
    simulation approach. This class provides the unified interface.

    Args:
        network_backend: Network simulation backend to use ('analytical' or 'ns3')
    """

    def __init__(self, network_backend: Literal["analytical", "ns3"] = "analytical"):
        """Initialize ASTRA-Sim backend with specified network simulation.

        Args:
            network_backend: Network simulation backend ('analytical' or 'ns3')
        """
        self.network_backend = NetworkBackend(network_backend)

    def run(self, scenario: ScenarioConfig) -> dict:
        """Run simulation using ASTRA-Sim.

        Args:
            scenario: The scenario configuration containing datacenter and workload specs

        Returns:
            Simulation results dictionary

        Raises:
            NotImplementedError: If network backend not yet implemented
            ValueError: If workload type is not supported
        """
        # Validate workload type
        if not isinstance(scenario.workload, MegatronWorkload):
            raise ValueError(
                f"AstraSimBackend currently only supports MegatronWorkload, "
                f"got {type(scenario.workload).__name__}"
            )

        # 1. Convert topology
        topo_converter = TopologyConverter()
        network_topo = topo_converter.convert(scenario.datacenter)

        # 2. Convert workload
        workload_converter = WorkloadConverter()
        workload_trace = workload_converter.convert(
            scenario.workload, scenario.datacenter
        )

        # 3. Run simulation through C++ bindings
        if self.network_backend == NetworkBackend.ANALYTICAL:
            try:
                from simulon._sim import run_analytical
                from simulon.backend.astra_converter.cpp_bridge import (
                    to_cpp_topology,
                    to_cpp_workload,
                )

                # Convert Python dataclasses to C++ objects
                cpp_topology = to_cpp_topology(network_topo)
                cpp_workload = to_cpp_workload(workload_trace)

                sim_results = run_analytical(cpp_topology, cpp_workload)

                return {
                    "status": "success" if sim_results.success else "error",
                    "network_backend": self.network_backend.value,
                    "simulation": {
                        "total_time_ns": sim_results.total_time_ns,
                        "compute_time_ns": sim_results.compute_time_ns,
                        "communication_time_ns": sim_results.communication_time_ns,
                        "completed_layers": sim_results.completed_layers,
                        "error_message": sim_results.error_message,
                        "metrics": dict(sim_results.metrics),
                    },
                    "topology": {
                        "num_nodes": len(network_topo.nodes),
                        "num_links": len(network_topo.links),
                        "gpus_per_server": network_topo.gpus_per_server,
                        "nv_switch_num": network_topo.nv_switch_num,
                        "switches_excluding_nvswitch": network_topo.switches_excluding_nvswitch,
                        "gpu_type": network_topo.gpu_type,
                    },
                    "workload": {
                        "parallelism_policy": workload_trace.parallelism_policy,
                        "tensor_parallel": workload_trace.model_parallel_npu_group,
                        "expert_parallel": workload_trace.expert_parallel_npu_group,
                        "pipeline_parallel": workload_trace.pipeline_model_parallelism,
                        "gradient_accumulation": workload_trace.ga,
                        "virtual_pipeline_parallel": workload_trace.vpp,
                        "all_gpus": workload_trace.all_gpus,
                        "num_layers": workload_trace.num_layers,
                    },
                }
            except ImportError:
                # C++ bindings not available, return conversion results only
                return self._conversion_only_result(network_topo, workload_trace)
        elif self.network_backend == NetworkBackend.NS3:
            raise NotImplementedError("NS-3 backend not yet implemented")

        return self._conversion_only_result(network_topo, workload_trace)

    def _conversion_only_result(self, network_topo, workload_trace) -> dict:
        """Return conversion results when simulation is not available."""
        return {
            "status": "conversion_complete",
            "network_backend": self.network_backend.value,
            "topology": {
                "num_nodes": len(network_topo.nodes),
                "num_links": len(network_topo.links),
                "gpus_per_server": network_topo.gpus_per_server,
                "nv_switch_num": network_topo.nv_switch_num,
                "switches_excluding_nvswitch": network_topo.switches_excluding_nvswitch,
                "gpu_type": network_topo.gpu_type,
            },
            "workload": {
                "parallelism_policy": workload_trace.parallelism_policy,
                "tensor_parallel": workload_trace.model_parallel_npu_group,
                "expert_parallel": workload_trace.expert_parallel_npu_group,
                "pipeline_parallel": workload_trace.pipeline_model_parallelism,
                "gradient_accumulation": workload_trace.ga,
                "virtual_pipeline_parallel": workload_trace.vpp,
                "all_gpus": workload_trace.all_gpus,
                "num_layers": workload_trace.num_layers,
            },
            "note": "C++ simulation not available - conversion only",
        }
