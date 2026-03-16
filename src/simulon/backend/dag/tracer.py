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
    steady_state_only: bool = False


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


def _sublayer_entry_exit(
    c_nodes: list[ComputeNode],
    comm_stubs: list[CommNode],
    stub_to_comm_ids: dict[int, list[int]],
) -> tuple[int | None, int | None]:
    """Return (entry_node_id, exit_node_id) for a sublayer after stub patching.

    entry = first actual node (first AllGather flow if TP collective, else first compute).
    exit  = last actual node  (last ReduceScatter flow if TP collective, else last compute).
    """
    if comm_stubs:
        first_actual = stub_to_comm_ids.get(comm_stubs[0].node_id, [])
        entry_id = first_actual[0] if first_actual else (c_nodes[0].node_id if c_nodes else None)
        last_actual = stub_to_comm_ids.get(comm_stubs[-1].node_id, [])
        exit_id = last_actual[-1] if last_actual else (c_nodes[-1].node_id if c_nodes else None)
    elif c_nodes:
        entry_id = c_nodes[0].node_id
        exit_id = c_nodes[-1].node_id
    else:
        entry_id = exit_id = None
    return entry_id, exit_id


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

        # Sequential-ordering state:
        #   last_node_per_gpu: tracks the most recent exit node for each GPU across
        #     all slots/layers/sublayers, so every operation is chained to the previous.
        #   slot_last_node: records the exit node at the end of each (gpu, mb, direction)
        #     slot so PP_Send nodes can depend on the right slot's compute.
        #   slot_entry_node: records the entry (first) node of each (gpu, mb, direction)
        #     slot so incoming PP_Send flows can gate the slot's start.
        last_node_per_gpu: dict[int, int | None] = {}
        slot_last_node: dict[tuple[int, int, str], int] = {}
        slot_entry_node: dict[tuple[int, int, str], int] = {}

        # PP_Send cross-stage deps collected during the main loop and applied afterwards
        # because the destination stage may not have been visited yet.
        # Each entry: (pp_send_node_id, dst_gpu, mb, direction)
        pending_pp_deps: list[tuple[int, int, int, str]] = []

        get_slots = (
            scheduler.schedule_steady_state_for_stage
            if cfg.steady_state_only
            else scheduler.schedule_for_stage
        )

        # Iterate over all GPUs
        for dp_rank in range(dp):
            for pp_stage in range(pp):
                for tp_rank in range(tp):
                    gpu = global_rank(dp_rank, pp_stage, tp_rank)
                    tp_group = [global_rank(dp_rank, pp_stage, r) for r in range(tp)]
                    dp_group = [global_rank(r, pp_stage, tp_rank) for r in range(dp)]

                    slots = get_slots(pp_stage)

                    for slot in slots:
                        mb = slot.microbatch_id
                        direction = slot.direction
                        slot_first_entry_set = False

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

                                    # Fill comm stubs with flows from decompose_collective,
                                    # then patch edges: replace stub node_ids with actual CommNode ids.
                                    stub_to_comm_ids: dict[int, list[int]] = {}
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

                                        stub_to_comm_ids[stub.node_id] = []
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
                                            stub_to_comm_ids[stub.node_id].append(node_id_counter)
                                            node_id_counter += 1

                                    # Patch edges: stub refs → actual CommNode ids
                                    for edge in edges:
                                        srcs = stub_to_comm_ids.get(edge.src_node_id, [edge.src_node_id])
                                        dsts = stub_to_comm_ids.get(edge.dst_node_id, [edge.dst_node_id])
                                        for s in srcs:
                                            for d in dsts:
                                                dag.edges.append(DAGEdge(src_node_id=s, dst_node_id=d))

                                    # Sequential ordering: chain this sublayer after the previous
                                    # operation on this GPU (across sublayers, layers, and slots).
                                    entry_id, exit_id = _sublayer_entry_exit(
                                        c_nodes, comm_stubs, stub_to_comm_ids
                                    )

                                    # Record the first entry node of this slot (for PP_Send recv deps)
                                    if not slot_first_entry_set and entry_id is not None:
                                        slot_entry_node[(gpu, mb, direction)] = entry_id
                                        slot_first_entry_set = True

                                    prev = last_node_per_gpu.get(gpu)
                                    if prev is not None and entry_id is not None:
                                        dag.edges.append(DAGEdge(src_node_id=prev, dst_node_id=entry_id))
                                    if exit_id is not None:
                                        last_node_per_gpu[gpu] = exit_id

                        # Record slot exit so PP_Send can depend on it
                        last = last_node_per_gpu.get(gpu)
                        if last is not None:
                            slot_last_node[(gpu, mb, direction)] = last

                # PP_Send at stage boundaries
                if pp > 1:
                    slots = get_slots(pp_stage)
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

                        # PP_Send depends on the last compute of that slot on tp_rank=0
                        slot_key = (src_gpu, mb, slot.direction)
                        if slot_key in slot_last_node:
                            dag.edges.append(DAGEdge(
                                src_node_id=slot_last_node[slot_key],
                                dst_node_id=pp_send.node_id,
                            ))

                        # Defer the recv-side edges for ALL tp_ranks of the dst stage.
                        # This ensures every TP rank waits for the activations/gradients,
                        # not just tp_rank=0 (the PP_Send destination).
                        for tr in range(tp):
                            dst_gpu_tr = global_rank(dp_rank, dst_stage, tr)
                            pending_pp_deps.append((pp_send.node_id, dst_gpu_tr, mb, slot.direction))

        # Add cross-stage edges: PP_Send → first node of the receiving stage's slot.
        # This enforces that a downstream stage cannot begin a slot until it has
        # received the activations (fwd) or gradients (bwd) from its neighbour.
        for pp_send_id, dst_gpu, mb, direction in pending_pp_deps:
            key = (dst_gpu, mb, direction)
            if key in slot_entry_node:
                dag.edges.append(DAGEdge(src_node_id=pp_send_id, dst_node_id=slot_entry_node[key]))

        return dag


def _phases_for_direction(direction: str) -> list[str]:
    if direction == "fwd":
        return ["fwd"]
    else:  # bwd
        return ["bwd_ig", "bwd_wg"]
