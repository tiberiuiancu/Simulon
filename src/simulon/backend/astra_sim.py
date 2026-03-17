from simulon.backend.base import Backend
from simulon.backend.dag import DAGTracer, DAGTracerConfig, ExecutionDAG, populate_dag, replay, SimulationResult
from simulon.config.dc import DatacenterConfig, GPUSpec
from simulon.config.scenario import ScenarioConfig
from simulon.config.workload import MegatronWorkload

_SUPPORTED_LIBRARIES = {"nccl"}


def _tracer_config_from_scenario(scenario: ScenarioConfig) -> DAGTracerConfig:
    c = scenario.collective
    if c.library not in _SUPPORTED_LIBRARIES:
        raise NotImplementedError(
            f"CCL library {c.library!r} is not yet implemented. "
            f"Supported: {sorted(_SUPPORTED_LIBRARIES)}"
        )
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


class AstraSimBackend(Backend):
    """Backend that produces a GPU-agnostic execution DAG."""

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
        if not isinstance(scenario.workload, MegatronWorkload):
            raise ValueError(f"AstraSimBackend only supports MegatronWorkload, got {type(scenario.workload).__name__}")
        tracer = DAGTracer(_tracer_config_from_scenario(scenario))
        return tracer.trace(scenario.workload, scenario.datacenter)

    def simulate(self, scenario: ScenarioConfig) -> tuple[ExecutionDAG, SimulationResult]:
        if not isinstance(scenario.workload, MegatronWorkload):
            raise ValueError(f"AstraSimBackend only supports MegatronWorkload, got {type(scenario.workload).__name__}")
        dag = self.run_trace(scenario)
        gpu_spec = _resolve_gpu_spec(scenario.datacenter)
        populate_dag(dag, scenario.workload, gpu_spec)
        result = replay(dag, scenario.datacenter)
        return dag, result
