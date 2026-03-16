from simulon.backend.base import Backend
from simulon.backend.dag import DAGTracer, DAGTracerConfig, ExecutionDAG
from simulon.config.scenario import ScenarioConfig
from simulon.config.workload import MegatronWorkload


class AstraSimBackend(Backend):
    """Backend that produces a GPU-agnostic execution DAG."""

    def __init__(self, num_channels: int = 1, algorithm: str = "ring"):
        self._tracer_config = DAGTracerConfig(num_channels=num_channels, algorithm=algorithm)

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
