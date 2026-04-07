import logging

from simulon.backend.base import Backend
from simulon.backend.dag import DAGTracerConfig, ExecutionDAG, populate_dag, replay, SimulationResult
from simulon.backend.dag.populate import populate_network
from simulon.collective import CCLDecomposer, NCCLDecomposer, RCCLDecomposer
from simulon.config.dc import DatacenterConfig, GPUSpec
from simulon.config.resolve import load_gpu_template, resolve_gpu_spec
from simulon.config.scenario import ScenarioConfig
from simulon.config.workload import CollectiveWorkload, MegatronWorkload

logger = logging.getLogger(__name__)

_CCL_MAP: dict[str, type[CCLDecomposer]] = {
    "nccl": NCCLDecomposer,
    "rccl": RCCLDecomposer,
}


def _ccl_from_scenario(scenario: ScenarioConfig) -> CCLDecomposer:
    library = scenario.collective.library
    cls = _CCL_MAP.get(library)
    if cls is None:
        raise ValueError(f"Unknown CCL library {library!r}. Supported: {sorted(_CCL_MAP)}")
    return cls()


def _tracer_config_from_scenario(scenario: ScenarioConfig) -> DAGTracerConfig:
    c = scenario.collective
    return DAGTracerConfig(
        num_channels=c.num_channels,
        algorithm=c.algorithm,
    )


# Keep private aliases so call sites inside this module are unchanged.
_resolve_gpu_spec = resolve_gpu_spec
_load_gpu_template = load_gpu_template


class AnalyticalBackend(Backend):
    """Python analytical backend that produces a GPU-agnostic execution DAG."""

    def run(self, scenario: ScenarioConfig) -> dict:
        dag = self.run_trace(scenario)
        d = dag.to_dict()
        return {
            "status": "success",
            "compute_nodes": len(dag.compute_nodes),
            "comm_nodes": len(dag.comm_nodes),
            "edges": len(dag.edges),
            "dag": d,
        }

    def run_trace(self, scenario: ScenarioConfig, compact: bool = False) -> ExecutionDAG:
        from simulon.backend.dag.megatron_tracer import MegatronDAGTracer
        if isinstance(scenario.workload, MegatronWorkload):
            cfg = _tracer_config_from_scenario(scenario)
            cfg.compact = compact
            tracer = MegatronDAGTracer(cfg, ccl=_ccl_from_scenario(scenario))
            return tracer.trace(scenario.workload, scenario.datacenter)
        elif isinstance(scenario.workload, CollectiveWorkload):
            from simulon.backend.dag.collective_tracer import build_collective_dag
            c = scenario.collective
            return build_collective_dag(
                workload=scenario.workload,
                datacenter=scenario.datacenter,
                algorithm=c.algorithm,
                num_channels=c.num_channels,
                ccl=_ccl_from_scenario(scenario),
            )
        else:
            raise ValueError(f"AnalyticalBackend does not support {type(scenario.workload).__name__}")

    def simulate(self, scenario: ScenarioConfig, compact: bool = False) -> tuple[ExecutionDAG, SimulationResult]:
        if isinstance(scenario.workload, MegatronWorkload):
            from simulon.backend.dag.megatron_tracer import MegatronDAGTracer
            p = scenario.workload.parallelism
            t = scenario.workload.training
            num_gpus = t.num_gpus
            logger.info("Building DAG  (GPUs=%d  tp=%d  pp=%d  ep=%d  dp=%d) ...",
                        num_gpus, p.tp, p.pp, p.ep,
                        p.dp if p.dp is not None else num_gpus // (p.tp * p.pp * p.ep))
            dag = self.run_trace(scenario, compact=compact)
            logger.info("  DAG built: %d compute nodes, %d comm nodes, %d edges",
                        len(dag.compute_nodes), len(dag.comm_nodes), len(dag.edges))

            gpu_spec = _resolve_gpu_spec(scenario.datacenter)
            logger.info("Resolving compute durations (%d nodes) ...", len(dag.compute_nodes))
            populate_dag(dag, scenario.workload, gpu_spec)
            logger.info("  Compute durations resolved")

            logger.info("Populating network durations (%d comm nodes) ...", len(dag.comm_nodes))
            populate_network(dag, scenario.datacenter)
            logger.info("  Network durations resolved")

            total_nodes = len(dag.compute_nodes) + len(dag.comm_nodes)
            logger.info("Replaying DAG (%d nodes) ...", total_nodes)
            result = replay(dag)
            logger.info("  Replay done: total_time=%.3f ms", result.total_time_ms)

            return dag, result

        elif isinstance(scenario.workload, CollectiveWorkload):
            wl = scenario.workload
            dc = scenario.datacenter
            num_ranks = dc.cluster.num_nodes * dc.node.gpus_per_node
            logger.info("Building collective DAG  (type=%s  ranks=%d  size=%d bytes) ...",
                        wl.collective_type.value, num_ranks, wl.message_size_bytes)
            # compact has no meaning for collective-only DAGs (no compute nodes to fuse)
            dag = self.run_trace(scenario)
            logger.info("  DAG built: %d comm nodes", len(dag.comm_nodes))

            logger.info("Populating network durations (%d comm nodes) ...", len(dag.comm_nodes))
            populate_network(dag, scenario.datacenter)
            logger.info("  Network durations resolved")

            logger.info("Replaying DAG (%d nodes) ...", len(dag.comm_nodes))
            result = replay(dag)
            logger.info("  Replay done: total_time=%.3f ms", result.total_time_ms)

            return dag, result

        else:
            raise ValueError(f"AnalyticalBackend does not support {type(scenario.workload).__name__}")
