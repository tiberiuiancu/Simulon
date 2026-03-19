import logging

from simulon.backend.base import Backend
from simulon.backend.dag import DAGTracerConfig, ExecutionDAG, populate_dag, replay, SimulationResult
from simulon.backend.dag.populate import populate_network
from simulon.collective import CCLDecomposer, NCCLDecomposer, RCCLDecomposer
from simulon.config.dc import DatacenterConfig, GPUSpec
from simulon.config.scenario import ScenarioConfig
from simulon.config.workload import MegatronWorkload

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


def _resolve_gpu_spec(dc: DatacenterConfig) -> GPUSpec:
    gpu = dc.node.gpu
    if isinstance(gpu, str):
        return _load_gpu_template(gpu)
    if isinstance(gpu, GPUSpec) and gpu.from_:
        base = _load_gpu_template(gpu.from_)
        if gpu.name:
            base.name = gpu.name
        if gpu.flops_multiplier != 1.0:
            base.flops_multiplier = gpu.flops_multiplier
        return base
    return gpu


def _load_gpu_template(name: str) -> GPUSpec:
    import yaml
    from pathlib import Path

    template_path = Path("templates/gpu") / f"{name}.yaml"
    if not template_path.exists():
        candidates = list(Path("templates/gpu").glob("*.yaml")) if Path("templates/gpu").exists() else []
        for c in candidates:
            if c.stem.lower() == name.lower():
                template_path = c
                break
        else:
            raise FileNotFoundError(
                f"GPU template not found: {name!r}. Expected at templates/gpu/{name}.yaml"
            )
    with open(template_path) as f:
        data = yaml.safe_load(f)
    return GPUSpec.model_validate(data)


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

    def run_trace(self, scenario: ScenarioConfig) -> ExecutionDAG:
        from simulon.backend.dag.megatron_tracer import MegatronDAGTracer
        if not isinstance(scenario.workload, MegatronWorkload):
            raise ValueError(f"AnalyticalBackend only supports MegatronWorkload, got {type(scenario.workload).__name__}")
        tracer = MegatronDAGTracer(_tracer_config_from_scenario(scenario), ccl=_ccl_from_scenario(scenario))
        return tracer.trace(scenario.workload, scenario.datacenter)

    def simulate(self, scenario: ScenarioConfig) -> tuple[ExecutionDAG, SimulationResult]:
        from simulon.backend.dag.megatron_tracer import MegatronDAGTracer
        if not isinstance(scenario.workload, MegatronWorkload):
            raise ValueError(f"AnalyticalBackend only supports MegatronWorkload, got {type(scenario.workload).__name__}")

        p = scenario.workload.parallelism
        t = scenario.workload.training
        num_gpus = t.num_gpus
        logger.info("Building DAG  (GPUs=%d  tp=%d  pp=%d  ep=%d  dp=%d) ...",
                    num_gpus, p.tp, p.pp, p.ep,
                    p.dp if p.dp is not None else num_gpus // (p.tp * p.pp * p.ep))
        dag = self.run_trace(scenario)
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
