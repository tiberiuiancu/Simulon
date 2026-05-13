from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path

from simulon.backend.dag._progress import log_progress
from simulon.backend.dag.nodes import ComputeNode, CommNode, DAGEdge, ExecutionDAG
from simulon.backend.dag.pipeline import make_scheduler
from simulon.backend.dag.layer_expander import LayerExpander
from simulon.backend.dag.tracer import DAGTracer, DAGTracerConfig
from simulon.collective import CCLDecomposer, NCCLDecomposer
from simulon.config.dc import DatacenterConfig
from simulon.config.workload import LLMSpec, MegatronDeprecatedWorkload
from simulon.profiling.models import _resolve_model

logger = logging.getLogger(__name__)


def _sublayer_entry_exit(
    c_nodes: list[ComputeNode],
    comm_stubs: list[CommNode],
    stub_to_comm_ids: dict[int, list[int]],
) -> tuple[int | None, int | None]:
    all_nodes = sorted(
        [("c", n.node_id) for n in c_nodes] + [("s", s.node_id) for s in comm_stubs],
        key=lambda x: x[1],
    )
    if not all_nodes:
        return None, None

    def _resolve(node_type: str, node_id: int, end: bool = False) -> int | None:
        if node_type == "c":
            return node_id
        actual = stub_to_comm_ids.get(node_id, [])
        if actual:
            return actual[-1] if end else actual[0]
        return None

    entry_id = None
    for node_type, node_id in all_nodes:
        entry_id = _resolve(node_type, node_id)
        if entry_id is not None:
            break

    exit_id = None
    for node_type, node_id in reversed(all_nodes):
        exit_id = _resolve(node_type, node_id, end=True)
        if exit_id is not None:
            break

    return entry_id, exit_id


def _phases_for_direction(direction: str) -> list[str]:
    if direction == "fwd":
        return ["fwd"]
    else:  # bwd
        return ["bwd_ig", "bwd_wg"]


@dataclass
class ParallelGroups:
    """Rank lists for each parallelism group on one GPU.

    Mirrors Megatron-Core's two-generator pattern (parallel_state.py):
      - ``expert_dp``:     EXPERT_DATA_PARALLEL_GROUP — fix TP, EP, PP; vary DP only.
                           Size = dp.  Used for expert-layer gradient sync.
      - ``non_expert_dp``: DATA_PARALLEL_GROUP        — fix TP, PP; vary DP × EP.
                           Size = dp × ep.  Used for non-expert gradient sync.

    When CP is added: extend ``non_expert_dp`` to also vary CP (vary DP × EP × CP),
    matching Megatron's DATA_PARALLEL_GROUP_WITH_CP.  No other field changes.
    """
    tp: list[int]
    ep: list[int]
    expert_dp: list[int]
    non_expert_dp: list[int]


def _non_expert_params_per_tp_rank(model: LLMSpec, tp: int) -> int:
    """Parameters on one TP rank that belong to non-expert sublayers (fp32 count).

    Covers: attention projections, layer-norms, embedding, logit.
    These params are replicated across the full non-expert DP group (dp × ep).
    """
    hidden = model.hidden_size or 0
    ffn = model.ffn_hidden_size or (4 * hidden)
    num_layers = model.num_layers or 0
    vocab_size = model.vocab_size or 0

    attn_per_layer = 4 * hidden * hidden // tp
    mlp_factor = 3 if model.swiglu else 2
    # Non-MoE models: the MLP is also non-expert
    non_expert_mlp = 0 if model.num_experts is not None else mlp_factor * hidden * ffn // tp
    ln_per_layer = 2 * hidden

    per_layer = attn_per_layer + non_expert_mlp + ln_per_layer
    embedding = vocab_size * hidden // tp
    logit = vocab_size * hidden // tp
    return num_layers * per_layer + embedding + logit


def _expert_params_per_tp_rank(model: LLMSpec, tp: int, ep: int) -> int:
    """Parameters on one TP rank that belong to expert sublayers (fp32 count).

    Only non-zero for MoE models.  These params are sharded across EP ranks,
    so gradient sync uses the smaller expert DP group (size = dp).
    """
    if model.num_experts is None:
        return 0
    hidden = model.hidden_size or 0
    ffn = model.ffn_hidden_size or (4 * hidden)
    num_layers = model.num_layers or 0
    num_experts = model.num_experts
    mlp_factor = 3 if model.swiglu else 2
    return num_layers * mlp_factor * hidden * ffn * (num_experts // ep) // tp


def _params_per_tp_rank(model: LLMSpec, tp: int, ep: int) -> int:
    """Total trainable parameters on one TP rank (fp32 count)."""
    return _non_expert_params_per_tp_rank(model, tp) + _expert_params_per_tp_rank(model, tp, ep)


def _emit_compute_node(
    dag: "ExecutionDAG",
    node_id_counter: int,
    gpu: int,
    kernel: str,
    microbatch_id: int,
    pipeline_stage: int,
    phase: str,
    last_node_per_gpu: dict,
    last_was_compute: dict,
    slot_entry_node: dict[tuple[int, int, str], int],
    slot_first_entry_set: bool,
    direction: str,
) -> tuple[int, bool]:
    """Emit a single compute node and wire it into the DAG chain.

    Sets last_was_compute[gpu] = False so that the subsequent sublayer's
    first kernel starts a fresh compact-fused group rather than being merged
    into this node.
    """
    n = ComputeNode(
        node_id=node_id_counter,
        gpu_rank=gpu,
        kernel=kernel,
        layer_id=-1,
        microbatch_id=microbatch_id,
        pipeline_stage=pipeline_stage,
        phase=phase,
    )
    node_id_counter += 1
    dag.compute_nodes.append(n)
    prev = last_node_per_gpu.get(gpu)
    if prev is not None:
        dag.edges.append(DAGEdge(src_node_id=prev, dst_node_id=n.node_id))
    if not slot_first_entry_set:
        slot_entry_node[(gpu, microbatch_id, direction)] = n.node_id
        slot_first_entry_set = True
    last_node_per_gpu[gpu] = n.node_id
    last_was_compute[gpu] = False
    return node_id_counter, slot_first_entry_set


def _emit_collective_nodes(
    dag: "ExecutionDAG",
    ccl: "CCLDecomposer",
    cfg: "DAGTracerConfig",
    node_id_counter: int,
    flow_id_counter: int,
    gpu: int,
    collective_type: str,
    group: list[int],
    data_bytes: int,
    phase: str,
    microbatch_id: int,
    last_node_per_gpu: dict,
    last_was_compute: dict,
    slot_entry_node: dict[tuple[int, int, str], int],
    slot_first_entry_set: bool,
    direction: str,
) -> tuple[int, int, bool]:
    """Decompose and emit collective flows, wired into the DAG chain.

    Mutates dag, last_node_per_gpu, last_was_compute, and slot_entry_node
    in-place; returns updated (node_id_counter, flow_id_counter,
    slot_first_entry_set).
    """
    result, flow_id_counter = ccl.decompose(
        collective_type=collective_type,
        group_ranks=group,
        data_size=data_bytes,
        num_channels=cfg.num_channels,
        algorithm=cfg.algorithm,
        flow_id_start=flow_id_counter,
    )
    flow_ids: list[int] = []
    for flow in result.flows:
        dag.comm_nodes.append(CommNode(
            node_id=node_id_counter,
            src_gpu=flow.src,
            dst_gpu=flow.dst,
            bytes=flow.flow_size,
            collective_type=collective_type,
            layer_id=-1,
            phase=phase,
            flow_id=flow.flow_id,
            parent_flow_ids=flow.parent_flow_ids,
        ))
        flow_ids.append(node_id_counter)
        node_id_counter += 1
    if flow_ids:
        prev = last_node_per_gpu.get(gpu)
        if prev is not None:
            for fid in flow_ids:
                dag.edges.append(DAGEdge(src_node_id=prev, dst_node_id=fid))
        if not slot_first_entry_set:
            slot_entry_node[(gpu, microbatch_id, direction)] = flow_ids[0]
            slot_first_entry_set = True
        last_node_per_gpu[gpu] = flow_ids[-1]
        last_was_compute[gpu] = False
    return node_id_counter, flow_id_counter, slot_first_entry_set


class MegatronDeprecatedDAGTracer(DAGTracer):
    def __init__(self, config: DAGTracerConfig | None = None, ccl: CCLDecomposer | None = None):
        self.config = config or DAGTracerConfig()
        self._ccl = ccl or NCCLDecomposer()

    def trace(self, workload: MegatronDeprecatedWorkload, datacenter: DatacenterConfig) -> ExecutionDAG:
        from simulon.backend.dag import cache as _cache

        cfg = self.config
        p = workload.parallelism
        t = workload.training

        tp = p.tp
        pp = p.pp
        ep = p.ep
        cp = 1  # Context Parallel — not yet in config; extend global_rank loop here when added
        dp = p.dp if p.dp is not None else t.num_gpus // (tp * pp * ep)

        if hasattr(p, "num_microbatches") and p.num_microbatches is not None:
            num_microbatches = p.num_microbatches
        else:
            num_microbatches = t.global_batch_size // (dp * t.micro_batch_size)

        model = _resolve_model(workload.model)

        _key: str | None = None
        if cfg.cache_dir is not None:
            _key = _cache._cache_key(workload, model, cfg)
            _dag = _cache.load(Path(cfg.cache_dir), _key)
            if _dag is not None:
                return _dag

        num_layers = model.num_layers
        hidden_size = model.hidden_size
        seq_len = t.sequence_length
        micro_bs = t.micro_batch_size

        if num_layers is None:
            raise ValueError("Model must have num_layers defined")
        if hidden_size is None:
            raise ValueError("Model must have hidden_size defined")

        activation_bytes = seq_len * micro_bs * hidden_size * cfg.dtype_bytes

        top_k = model.top_k or 1
        moe_data_bytes = seq_len * micro_bs * hidden_size * top_k * cfg.dtype_bytes // tp

        sublayers = ["attn", "moe" if model.num_experts is not None else "mlp"]

        scheduler = make_scheduler(p.pipeline_schedule, pp, num_microbatches)
        expander = LayerExpander()

        dag = ExecutionDAG()
        node_id_counter = 0
        flow_id_counter = 0
        pending_fused: dict[int, ComputeNode] = {}  # gpu → pending fused node (compact mode)
        last_was_compute: dict[int, bool] = {}  # gpu → was last emitted node compute?

        def global_rank(dp_rank: int, pp_stage: int, ep_rank: int, tp_rank: int, cp_rank: int = 0) -> int:
            # Rank order: tp-cp-ep-dp-pp  (tp varies fastest, pp slowest)
            # When cp=1 (current default) this reduces to the previous formula.
            # To add CP: introduce a cp loop dimension and pass cp_rank here.
            return (
                dp_rank * (pp * ep * cp * tp)
                + pp_stage * (ep * cp * tp)
                + ep_rank * (cp * tp)
                + cp_rank * tp
                + tp_rank
            )

        last_node_per_gpu: dict[int, int | None] = {}
        slot_last_node: dict[tuple[int, int, str], int] = {}
        slot_entry_node: dict[tuple[int, int, str], int] = {}
        pending_pp_deps: list[tuple[int, int, int, str]] = []

        # Step-phase invariants (loop-independent).
        # TODO: only AdamW is modeled; extend here if other optimizers are needed.
        _non_expert_params = _non_expert_params_per_tp_rank(model, tp)
        _expert_params = _expert_params_per_tp_rank(model, tp, ep)
        _dist_opt = p.distributed_optimizer
        _non_expert_dp_size = dp * ep  # DATA_PARALLEL_GROUP size (non-expert layers)
        # With distributed optimizer, each rank owns a 1/group_size shard of params.
        # Non-expert params are sharded across non_expert_dp (dp × ep ranks).
        # Expert params are sharded across expert_dp (dp ranks).
        _non_expert_opt_params = (
            _non_expert_params // pp // _non_expert_dp_size
            if (_dist_opt and _non_expert_dp_size > 1)
            else _non_expert_params // pp
        )
        _expert_opt_params = (
            _expert_params // pp // dp
            if (_dist_opt and dp > 1)
            else _expert_params // pp
        )
        _opt_num_params = (_non_expert_opt_params + _expert_opt_params)

        with log_progress("  building DAG", dp * pp * ep * tp, logger) as advance:
            for dp_rank in range(dp):
                for pp_stage in range(pp):
                    for ep_rank in range(ep):
                        for tp_rank in range(tp):
                            gpu = global_rank(dp_rank, pp_stage, ep_rank, tp_rank)
                            groups = ParallelGroups(
                                tp=[global_rank(dp_rank, pp_stage, ep_rank, r) for r in range(tp)],
                                ep=[global_rank(dp_rank, pp_stage, r, tp_rank) for r in range(ep)],
                                # EXPERT_DATA_PARALLEL_GROUP: fix TP, EP, PP; vary DP
                                expert_dp=[global_rank(r, pp_stage, ep_rank, tp_rank) for r in range(dp)],
                                # DATA_PARALLEL_GROUP: fix TP, PP; vary DP × EP
                                # When CP is added: also vary cp_rank here (vary DP × EP × CP)
                                non_expert_dp=[
                                    global_rank(r, pp_stage, ep_r, tp_rank)
                                    for r in range(dp)
                                    for ep_r in range(ep)
                                ],
                            )
                            tp_group = groups.tp
                            ep_group = groups.ep

                            slots = scheduler.schedule_for_stage(pp_stage)

                            for slot in slots:
                                mb = slot.microbatch_id
                                direction = slot.direction
                                slot_first_entry_set = False

                                # Embedding fwd (PP stage 0, forward pass only)
                                if pp_stage == 0 and direction == "fwd":
                                    node_id_counter, slot_first_entry_set = _emit_compute_node(
                                        dag, node_id_counter, gpu, "embedding", mb, pp_stage, "fwd",
                                        last_node_per_gpu, last_was_compute,
                                        slot_entry_node, slot_first_entry_set, direction,
                                    )

                                # Logit bwd (PP stage pp-1, backward pass only): gradient flows
                                # loss → logit → last transformer layer, so logit bwd precedes body bwd.
                                if pp_stage == pp - 1 and direction == "bwd":
                                    node_id_counter, slot_first_entry_set = _emit_compute_node(
                                        dag, node_id_counter, gpu, "logit", mb, pp_stage, "bwd_ig",
                                        last_node_per_gpu, last_was_compute,
                                        slot_entry_node, slot_first_entry_set, direction,
                                    )

                                for layer_idx in range(num_layers):
                                    for sublayer in sublayers:
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
                                                ep_group_ranks=ep_group,
                                                moe_data_bytes=moe_data_bytes,
                                            )

                                            if cfg.compact:
                                                # Process nodes in topological order (by node_id).
                                                # Rule: fuse a compute node into pending_fused iff
                                                # its only predecessor is the previous compute node.
                                                # Flush pending_fused on any comm node.
                                                ordered = sorted(
                                                    [("c", n) for n in c_nodes] + [("s", s) for s in comm_stubs],
                                                    key=lambda x: x[1].node_id,
                                                )
                                                for ntype, node in ordered:
                                                    if ntype == "c":
                                                        if last_was_compute.get(gpu, False):
                                                            pending_fused[gpu].fused_kernels.append(node.kernel)
                                                            pending_fused[gpu].kernel = f"{len(pending_fused[gpu].fused_kernels)} kernels"
                                                        else:
                                                            pending_fused[gpu] = ComputeNode(
                                                                node_id=node.node_id,
                                                                gpu_rank=gpu,
                                                                kernel=node.kernel,
                                                                fused_kernels=[node.kernel],
                                                                layer_id=node.layer_id,
                                                                microbatch_id=mb,
                                                                pipeline_stage=pp_stage,
                                                                phase=phase,
                                                            )
                                                            if not slot_first_entry_set:
                                                                slot_entry_node[(gpu, mb, direction)] = node.node_id
                                                                slot_first_entry_set = True
                                                        last_was_compute[gpu] = True
                                                    else:  # comm stub
                                                        if gpu in pending_fused:
                                                            pf = pending_fused.pop(gpu)
                                                            dag.compute_nodes.append(pf)
                                                            prev_id = last_node_per_gpu.get(gpu)
                                                            if prev_id is not None:
                                                                dag.edges.append(DAGEdge(src_node_id=prev_id, dst_node_id=pf.node_id))
                                                            last_node_per_gpu[gpu] = pf.node_id
                                                        if node.collective_type in ("AllGather", "ReduceScatter", "AllReduce"):
                                                            group = tp_group
                                                        elif node.collective_type == "AllToAll":
                                                            group = ep_group
                                                        else:
                                                            group = groups.expert_dp
                                                        result, flow_id_counter = self._ccl.decompose(
                                                            collective_type=node.collective_type,
                                                            group_ranks=group,
                                                            data_size=node.bytes,
                                                            num_channels=cfg.num_channels,
                                                            algorithm=cfg.algorithm,
                                                            flow_id_start=flow_id_counter,
                                                        )
                                                        flow_ids: list[int] = []
                                                        for flow in result.flows:
                                                            dag.comm_nodes.append(CommNode(
                                                                node_id=node_id_counter,
                                                                src_gpu=flow.src,
                                                                dst_gpu=flow.dst,
                                                                bytes=flow.flow_size,
                                                                collective_type=node.collective_type,
                                                                layer_id=layer_idx,
                                                                phase=phase,
                                                                flow_id=flow.flow_id,
                                                                parent_flow_ids=flow.parent_flow_ids,
                                                            ))
                                                            flow_ids.append(node_id_counter)
                                                            node_id_counter += 1
                                                        if flow_ids:
                                                            prev_id = last_node_per_gpu.get(gpu)
                                                            if prev_id is not None:
                                                                for fid in flow_ids:
                                                                    dag.edges.append(DAGEdge(src_node_id=prev_id, dst_node_id=fid))
                                                            if not slot_first_entry_set:
                                                                slot_entry_node[(gpu, mb, direction)] = flow_ids[0]
                                                                slot_first_entry_set = True
                                                            last_node_per_gpu[gpu] = flow_ids[-1]
                                                        last_was_compute[gpu] = False
                                            else:
                                                dag.compute_nodes.extend(c_nodes)

                                                stub_to_comm_ids: dict[int, list[int]] = {}
                                                for stub in comm_stubs:
                                                    if stub.collective_type in ("AllGather", "ReduceScatter", "AllReduce"):
                                                        group = tp_group
                                                    elif stub.collective_type == "AllToAll":
                                                        group = ep_group
                                                    else:
                                                        group = groups.expert_dp

                                                    result, flow_id_counter = self._ccl.decompose(
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

                                                for edge in edges:
                                                    srcs = stub_to_comm_ids.get(edge.src_node_id, [edge.src_node_id])
                                                    dsts = stub_to_comm_ids.get(edge.dst_node_id, [edge.dst_node_id])
                                                    for s in srcs:
                                                        for d in dsts:
                                                            dag.edges.append(DAGEdge(src_node_id=s, dst_node_id=d))

                                                entry_id, exit_id = _sublayer_entry_exit(
                                                    c_nodes, comm_stubs, stub_to_comm_ids
                                                )

                                                if not slot_first_entry_set and entry_id is not None:
                                                    slot_entry_node[(gpu, mb, direction)] = entry_id
                                                    slot_first_entry_set = True

                                                prev = last_node_per_gpu.get(gpu)
                                                if prev is not None:
                                                    for cn in c_nodes:
                                                        dag.edges.append(DAGEdge(src_node_id=prev, dst_node_id=cn.node_id))
                                                    for stub in comm_stubs:
                                                        for fid in stub_to_comm_ids.get(stub.node_id, []):
                                                            dag.edges.append(DAGEdge(src_node_id=prev, dst_node_id=fid))
                                                if exit_id is not None:
                                                    last_node_per_gpu[gpu] = exit_id

                                # Flush any remaining pending fused node at end of slot
                                if cfg.compact and gpu in pending_fused:
                                    pf = pending_fused.pop(gpu)
                                    dag.compute_nodes.append(pf)
                                    prev_pf = last_node_per_gpu.get(gpu)
                                    if prev_pf is not None:
                                        dag.edges.append(DAGEdge(src_node_id=prev_pf, dst_node_id=pf.node_id))
                                    last_node_per_gpu[gpu] = pf.node_id
                                    last_was_compute[gpu] = False

                                # Logit fwd + loss (PP stage pp-1, forward pass only)
                                if pp_stage == pp - 1 and direction == "fwd":
                                    node_id_counter, slot_first_entry_set = _emit_compute_node(
                                        dag, node_id_counter, gpu, "logit", mb, pp_stage, "fwd",
                                        last_node_per_gpu, last_was_compute,
                                        slot_entry_node, slot_first_entry_set, direction,
                                    )
                                    node_id_counter, slot_first_entry_set = _emit_compute_node(
                                        dag, node_id_counter, gpu, "loss_ce", mb, pp_stage, "fwd",
                                        last_node_per_gpu, last_was_compute,
                                        slot_entry_node, slot_first_entry_set, direction,
                                    )
                                    # VocabParallelCrossEntropy: 3x AllReduce over TP (bf16→fp32 cast).
                                    # See aicb/workload_generator/generate_megatron_workload.py forward().
                                    loss_tp_bytes = micro_bs * seq_len * 4  # fp32 per token
                                    for _ in range(3):
                                        node_id_counter, flow_id_counter, slot_first_entry_set = _emit_collective_nodes(
                                            dag, self._ccl, cfg, node_id_counter, flow_id_counter,
                                            gpu, "AllReduce", tp_group, loss_tp_bytes, "fwd", mb,
                                            last_node_per_gpu, last_was_compute,
                                            slot_entry_node, slot_first_entry_set, direction,
                                        )
                                    # average_losses_across_data_parallel_group: one fp32 scalar per microbatch
                                    node_id_counter, flow_id_counter, slot_first_entry_set = _emit_collective_nodes(
                                        dag, self._ccl, cfg, node_id_counter, flow_id_counter,
                                        gpu, "AllReduce", groups.non_expert_dp, 4, "fwd", mb,
                                        last_node_per_gpu, last_was_compute,
                                        slot_entry_node, slot_first_entry_set, direction,
                                    )

                                # Embedding bwd + AllReduce over TP (PP stage 0, backward pass only)
                                if pp_stage == 0 and direction == "bwd":
                                    node_id_counter, slot_first_entry_set = _emit_compute_node(
                                        dag, node_id_counter, gpu, "embedding", mb, pp_stage, "bwd_ig",
                                        last_node_per_gpu, last_was_compute,
                                        slot_entry_node, slot_first_entry_set, direction,
                                    )
                                    if tp > 1:
                                        node_id_counter, flow_id_counter, slot_first_entry_set = _emit_collective_nodes(
                                            dag, self._ccl, cfg, node_id_counter, flow_id_counter,
                                            gpu, "AllReduce", tp_group, activation_bytes, "bwd_ig", mb,
                                            last_node_per_gpu, last_was_compute,
                                            slot_entry_node, slot_first_entry_set, direction,
                                        )

                                last = last_node_per_gpu.get(gpu)
                                if last is not None:
                                    slot_last_node[(gpu, mb, direction)] = last

                            # --- Step phase ---
                            # Gradient sync then AdamW.  Two separate param groups:
                            #   non-expert: synced over non_expert_dp (DATA_PARALLEL_GROUP, size dp×ep)
                            #   expert:     synced over expert_dp (EXPERT_DATA_PARALLEL_GROUP, size dp)
                            # Pattern per group:
                            #   distributed_optimizer=True : ReduceScatter → AdamW → AllGather
                            #   distributed_optimizer=False: AllReduce → AdamW
                            # Both groups' pre-opt comms run in parallel (different ranks/data),
                            # then the single AdamW step, then post-opt comms in parallel.
                            prev = last_node_per_gpu.get(gpu)

                            # AdamW ComputeNode — emitted first so it gets a stable node_id;
                            # edges enforce ordering regardless of id assignment.
                            dag.compute_nodes.append(ComputeNode(
                                node_id=node_id_counter,
                                gpu_rank=gpu,
                                kernel="adamw",
                                layer_id=-1,
                                microbatch_id=-1,
                                pipeline_stage=pp_stage,
                                phase="step",
                                extra_params={"num_params": _opt_num_params},
                            ))
                            opt_id = node_id_counter
                            node_id_counter += 1

                            def _emit_step_flows(collective: str, group: list[int], data_size: int) -> list[int]:
                                nonlocal flow_id_counter, node_id_counter
                                result, flow_id_counter = self._ccl.decompose(
                                    collective_type=collective,
                                    group_ranks=group,
                                    data_size=data_size,
                                    num_channels=cfg.num_channels,
                                    algorithm=cfg.algorithm,
                                    flow_id_start=flow_id_counter,
                                )
                                ids = []
                                for flow in result.flows:
                                    dag.comm_nodes.append(CommNode(
                                        node_id=node_id_counter,
                                        src_gpu=flow.src, dst_gpu=flow.dst,
                                        bytes=flow.flow_size,
                                        collective_type=collective,
                                        layer_id=-1, phase="step",
                                        flow_id=flow.flow_id,
                                        parent_flow_ids=flow.parent_flow_ids,
                                    ))
                                    ids.append(node_id_counter)
                                    node_id_counter += 1
                                return ids

                            pre_opt_ids: list[int] = []
                            post_opt_ids: list[int] = []

                            # Non-expert gradient sync (DATA_PARALLEL_GROUP, size dp×ep)
                            if _non_expert_dp_size > 1 and _non_expert_params > 0:
                                ne_bytes = 4 * _non_expert_params // pp
                                if _dist_opt:
                                    pre_opt_ids += _emit_step_flows("ReduceScatter", groups.non_expert_dp, ne_bytes)
                                    post_opt_ids += _emit_step_flows(
                                        "AllGather", groups.non_expert_dp,
                                        cfg.dtype_bytes * _non_expert_params // pp,
                                    )
                                else:
                                    pre_opt_ids += _emit_step_flows("AllReduce", groups.non_expert_dp, ne_bytes)

                            # Expert gradient sync (EXPERT_DATA_PARALLEL_GROUP, size dp; EP is internal to the group)
                            if dp > 1 and _expert_params > 0:
                                exp_bytes = 4 * _expert_params // pp
                                if _dist_opt:
                                    pre_opt_ids += _emit_step_flows("ReduceScatter", groups.expert_dp, exp_bytes)
                                    post_opt_ids += _emit_step_flows(
                                        "AllGather", groups.expert_dp,
                                        cfg.dtype_bytes * _expert_params // pp,
                                    )
                                else:
                                    pre_opt_ids += _emit_step_flows("AllReduce", groups.expert_dp, exp_bytes)

                            # Wire: prev → pre_opt_ids → opt → post_opt_ids
                            # When pre_opt_ids is empty, the first block directly wires prev → opt.
                            if prev is not None:
                                for d in (pre_opt_ids if pre_opt_ids else [opt_id]):
                                    dag.edges.append(DAGEdge(src_node_id=prev, dst_node_id=d))
                            for s in pre_opt_ids:
                                dag.edges.append(DAGEdge(src_node_id=s, dst_node_id=opt_id))
                            for d in post_opt_ids:
                                dag.edges.append(DAGEdge(src_node_id=opt_id, dst_node_id=d))

                            last_node_per_gpu[gpu] = post_opt_ids[-1] if post_opt_ids else opt_id

                            advance()

                    if pp > 1:
                        slots = scheduler.schedule_for_stage(pp_stage)
                        for slot in slots:
                            mb = slot.microbatch_id
                            if slot.direction == "fwd" and pp_stage < pp - 1:
                                dst_stage = pp_stage + 1
                                src_gpu = global_rank(dp_rank, pp_stage, 0, 0)
                                dst_gpu = global_rank(dp_rank, dst_stage, 0, 0)
                            elif slot.direction == "bwd" and pp_stage > 0:
                                dst_stage = pp_stage - 1
                                src_gpu = global_rank(dp_rank, pp_stage, 0, 0)
                                dst_gpu = global_rank(dp_rank, dst_stage, 0, 0)
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

                            slot_key = (src_gpu, mb, slot.direction)
                            if slot_key in slot_last_node:
                                dag.edges.append(DAGEdge(
                                    src_node_id=slot_last_node[slot_key],
                                    dst_node_id=pp_send.node_id,
                                ))

                            for er in range(ep):
                                for tr in range(tp):
                                    dst_gpu_tr = global_rank(dp_rank, dst_stage, er, tr)
                                    pending_pp_deps.append((pp_send.node_id, dst_gpu_tr, mb, slot.direction))

        for pp_send_id, dst_gpu, mb, direction in pending_pp_deps:
            key = (dst_gpu, mb, direction)
            if key in slot_entry_node:
                dag.edges.append(DAGEdge(src_node_id=pp_send_id, dst_node_id=slot_entry_node[key]))

        if cfg.cache_dir is not None and _key is not None:
            _cache.save(Path(cfg.cache_dir), _key, dag)

        return dag
