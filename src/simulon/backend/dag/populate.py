from __future__ import annotations

from simulon.backend.dag.nodes import ExecutionDAG
from simulon.config.dc import GPUSpec
from simulon.config.workload import MegatronWorkload
from simulon.profiling.lookup import lookup_kernel_time


def populate_dag(
    dag: ExecutionDAG,
    workload: MegatronWorkload,
    gpu_spec: GPUSpec,
) -> ExecutionDAG:
    """Fill ComputeNode.duration_ms by looking up kernel times in gpu_spec.

    Mutates nodes in-place and returns the dag.
    """
    t = workload.training
    p = workload.parallelism

    match_params = {
        "hidden_size": _model_hidden_size(workload),
        "seq_len": t.sequence_length,
        "batch_size": t.micro_batch_size,
        "dtype": t.dtype.value,
        "tp": p.tp,
    }

    for node in dag.compute_nodes:
        node.duration_ms = lookup_kernel_time(node.kernel, match_params, gpu_spec)

    return dag


def _model_hidden_size(workload: MegatronWorkload) -> int | None:
    from simulon.config.workload import LLMSpec

    model = workload.model
    if isinstance(model, LLMSpec):
        return model.hidden_size
    return None
