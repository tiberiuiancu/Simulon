from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from simulon.backend.dag.nodes import ComputeNode, CommNode, DAGEdge, ExecutionDAG
from simulon.backend.dag.pipeline import PipelineScheduler
from simulon.backend.dag.layer_expander import LayerExpander
from simulon.collective import decompose_collective
from simulon.config.dc import DatacenterConfig
from simulon.config.workload import LLMSpec, MegatronWorkload


@dataclass
class DAGTracerConfig:
    num_channels: int = 1
    algorithm: str = "ring"
    dtype_bytes: int = 2  # bf16


def _resolve_model(model: str | LLMSpec) -> LLMSpec:
    """Resolve a model reference to an LLMSpec.

    If model is already an LLMSpec, return it directly.
    If it is a string (named model), attempt to load from templates/model/<name>.yaml.
    """
    if isinstance(model, LLMSpec):
        return model

    import yaml
    from pathlib import Path

    # Try to locate the template file
    template_path = Path("templates/model") / f"{model}.yaml"
    if not template_path.exists():
        # Try case-insensitive
        candidates = list(Path("templates/model").glob("*.yaml")) if Path("templates/model").exists() else []
        for c in candidates:
            if c.stem.lower() == model.lower():
                template_path = c
                break
        else:
            raise FileNotFoundError(
                f"Model template not found: {model!r}. "
                f"Expected at templates/model/{model}.yaml"
            )

    with open(template_path) as f:
        data = yaml.safe_load(f)
    return LLMSpec.model_validate(data)


class DAGTracer:
    def __init__(self, config: DAGTracerConfig | None = None):
        self.config = config or DAGTracerConfig()

    def trace(self, workload: MegatronWorkload, datacenter: DatacenterConfig) -> ExecutionDAG:
        cfg = self.config
        p = workload.parallelism
        t = workload.training

        tp = p.tp
        pp = p.pp
        dp = p.dp if p.dp is not None else t.num_gpus // (tp * pp)

        # num_microbatches
        if hasattr(p, "num_microbatches") and p.num_microbatches is not None:
            num_microbatches = p.num_microbatches
        else:
            num_microbatches = t.global_batch_size // (dp * t.micro_batch_size)

        # Resolve model spec
        model = _resolve_model(workload.model)
        num_layers = model.num_layers
        hidden_size = model.hidden_size
        seq_len = t.sequence_length
        micro_bs = t.micro_batch_size

        if num_layers is None:
            raise ValueError("Model must have num_layers defined")
        if hidden_size is None:
            raise ValueError("Model must have hidden_size defined")

        # Activation size estimate (bytes): seq_len * micro_bs * hidden_size * dtype_bytes
        activation_bytes = seq_len * micro_bs * hidden_size * cfg.dtype_bytes

        scheduler = PipelineScheduler(pp, num_microbatches)
        expander = LayerExpander()

        dag = ExecutionDAG()
        node_id_counter = 0
        flow_id_counter = 0

        def global_rank(dp_rank: int, pp_stage: int, tp_rank: int) -> int:
            return dp_rank * (tp * pp) + pp_stage * tp + tp_rank

        # Iterate over all GPUs
        for dp_rank in range(dp):
            for pp_stage in range(pp):
                for tp_rank in range(tp):
                    gpu = global_rank(dp_rank, pp_stage, tp_rank)
                    tp_group = [global_rank(dp_rank, pp_stage, r) for r in range(tp)]
                    dp_group = [global_rank(r, pp_stage, tp_rank) for r in range(dp)]

                    slots = scheduler.schedule_for_stage(pp_stage)

                    for slot in slots:
                        mb = slot.microbatch_id
                        direction = slot.direction

                        for layer_idx in range(num_layers):
                            for sublayer in ("attn", "mlp"):
                                for phase in _phases_for_direction(direction):
                                    c_nodes, comm_stubs, edges, node_id_counter = expander.expand_sublayer(
                                        sublayer_type=sublayer,
                                        phase=phase,
                                        gpu_rank=gpu,
                                        pipeline_stage=pp_stage,
                                        microbatch_id=mb,
                                        layer_idx=layer_idx,
                                        tp_group_ranks=tp_group,
                                        activation_bytes=activation_bytes,
                                        node_id_start=node_id_counter,
                                    )

                                    dag.compute_nodes.extend(c_nodes)
                                    dag.edges.extend(edges)

                                    # Fill comm stubs with flows from decompose_collective
                                    for stub in comm_stubs:
                                        if stub.collective_type in ("AllGather", "ReduceScatter", "AllReduce"):
                                            group = tp_group
                                        else:
                                            group = dp_group

                                        result, flow_id_counter, _ = decompose_collective(
                                            collective_type=stub.collective_type,
                                            group_ranks=group,
                                            data_size=stub.bytes,
                                            num_channels=cfg.num_channels,
                                            algorithm=cfg.algorithm,
                                            flow_id_start=flow_id_counter,
                                        )

                                        # Create CommNodes for each P2PFlow
                                        for flow in result.flows:
                                            comm_node = CommNode(
                                                node_id=node_id_counter,
                                                src_gpu=flow.src,
                                                dst_gpu=flow.dst,
                                                bytes=flow.flow_size,
                                                collective_type=stub.collective_type,
                                                layer_id=layer_idx,
                                                phase=phase,
                                                flow_id=flow.flow_id,
                                                parent_flow_ids=flow.parent_flow_ids,
                                            )
                                            dag.comm_nodes.append(comm_node)
                                            node_id_counter += 1

                # PP_Send at stage boundaries
                if pp > 1:
                    slots = scheduler.schedule_for_stage(pp_stage)
                    for slot in slots:
                        mb = slot.microbatch_id
                        if slot.direction == "fwd" and pp_stage < pp - 1:
                            dst_stage = pp_stage + 1
                            src_gpu = global_rank(dp_rank, pp_stage, 0)
                            dst_gpu = global_rank(dp_rank, dst_stage, 0)
                        elif slot.direction == "bwd" and pp_stage > 0:
                            dst_stage = pp_stage - 1
                            src_gpu = global_rank(dp_rank, pp_stage, 0)
                            dst_gpu = global_rank(dp_rank, dst_stage, 0)
                        else:
                            continue

                        pp_send = CommNode(
                            node_id=node_id_counter,
                            src_gpu=src_gpu,
                            dst_gpu=dst_gpu,
                            bytes=activation_bytes,
                            collective_type="PP_Send",
                            layer_id=0,
                            phase=slot.direction,
                            flow_id=flow_id_counter,
                        )
                        dag.comm_nodes.append(pp_send)
                        node_id_counter += 1
                        flow_id_counter += 1

        return dag


def _phases_for_direction(direction: str) -> list[str]:
    if direction == "fwd":
        return ["fwd"]
    else:  # bwd
        return ["bwd_ig", "bwd_wg"]
