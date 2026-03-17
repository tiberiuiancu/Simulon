from simulon.backend.base import Backend
from simulon.backend.dag import DAGTracer, DAGTracerConfig, ExecutionDAG, populate_dag, replay, SimulationResult
from simulon.config.dc import DatacenterConfig, GPUSpec
from simulon.config.scenario import ScenarioConfig
from simulon.config.workload import MegatronWorkload


def _resolve_gpu_spec(dc: DatacenterConfig) -> GPUSpec:
    """Return a GPUSpec, loading from template if node.gpu is a string or has from_."""
    gpu = dc.node.gpu
    if isinstance(gpu, str):
        return _load_gpu_template(gpu)
    if isinstance(gpu, GPUSpec) and gpu.from_:
        base = _load_gpu_template(gpu.from_)
        # Inline fields override template (except kernel_runs which come from template)
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

    def __init__(self, num_channels: int = 1, algorithm: str = "ring"):
        self._tracer_config = DAGTracerConfig(
            num_channels=num_channels,
            algorithm=algorithm,
        )

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
        tracer = DAGTracer(self._tracer_config)
        return tracer.trace(scenario.workload, scenario.datacenter)

    def simulate(self, scenario: ScenarioConfig) -> tuple[ExecutionDAG, SimulationResult]:
        """Run the full simulation pipeline.

        Returns (dag, result) where dag has start_ms/finish_ms/duration_ms
        populated on all nodes, ready for chrome trace export.
        """
        if not isinstance(scenario.workload, MegatronWorkload):
            raise ValueError(f"AstraSimBackend only supports MegatronWorkload, got {type(scenario.workload).__name__}")
        dag = self.run_trace(scenario)
        gpu_spec = _resolve_gpu_spec(scenario.datacenter)
        populate_dag(dag, scenario.workload, gpu_spec)
        result = replay(dag, scenario.datacenter)
        return dag, result
