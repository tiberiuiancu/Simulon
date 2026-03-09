"""ASTRA-Sim backend implementation for simulon."""

from simulon.backend.astra_converter import TopologyConverter, WorkloadConverter
from simulon.backend.base import Backend
from simulon.config.scenario import ScenarioSpec
from simulon.config.workload import MegatronWorkload


class AstraSimBackend(Backend):
    """Backend that uses ASTRA-Sim for distributed training simulation."""

    def run(self, scenario: ScenarioSpec) -> dict:
        """Run simulation using ASTRA-Sim.

        Args:
            scenario: The scenario configuration containing datacenter and workload specs

        Returns:
            Simulation results dictionary

        Raises:
            NotImplementedError: Full simulation not yet implemented
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

        # 3. Initialize ASTRA-Sim via C++ bindings
        # TODO: Implement AstraSimRunner C++ class
        # from simulon._sim import AstraSimRunner
        # runner = AstraSimRunner()
        # runner.initialize(network_topo, workload_trace)
        # results = runner.run_simulation()

        # For now, return conversion results for validation
        return {
            "status": "conversion_complete",
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
            "note": "Full simulation not yet implemented",
        }
