from typing import Literal
from simulon.backend.dag.nodes import ComputeNode, CommNode, DAGEdge


# Kernel sequences per sublayer and phase
_FWD_KERNELS = {
    "attn": ["layernorm", "attn_qkv", "attn_flash", "attn_proj"],
    "mlp": ["layernorm", "mlp_linear1", "mlp_act", "mlp_linear2"],
}
_BWD_IG_KERNELS = {
    "attn": list(reversed(_FWD_KERNELS["attn"])),
    "mlp": list(reversed(_FWD_KERNELS["mlp"])),
}
_BWD_WG_KERNELS = {
    "attn": ["attn_qkv", "attn_proj"],
    "mlp": ["mlp_linear1", "mlp_linear2"],
}


class LayerExpander:
    def expand_sublayer(
        self,
        sublayer_type: Literal["attn", "mlp", "moe"],
        phase: Literal["fwd", "bwd_ig", "bwd_wg"],
        gpu_rank: int,
        pipeline_stage: int,
        microbatch_id: int,
        layer_idx: int,
        tp_group_ranks: list[int],
        activation_bytes: int,
        node_id_start: int,
        ep_group_ranks: list[int] | None = None,
        moe_data_bytes: int = 0,
    ) -> tuple[list[ComputeNode], list[CommNode], list[DAGEdge], int]:
        """Expand a sublayer into compute and comm nodes.

        For attn/mlp fwd/bwd_ig: AllGather(TP) -> kernels -> ReduceScatter(TP)
        For attn/mlp bwd_wg: kernels -> AllReduce(DP)  [DP handled at tracer level]
        For moe: TP wraps around AllToAll(EP) dispatch/combine around expert compute.

        Returns comm stubs with flow_id=-1; tracer fills these from decompose_collective.
        Returns (compute_nodes, comm_nodes, edges, next_node_id).
        """
        if sublayer_type == "moe":
            return self._expand_moe(
                phase, gpu_rank, pipeline_stage, microbatch_id, layer_idx,
                tp_group_ranks, ep_group_ranks or [], activation_bytes,
                moe_data_bytes, node_id_start,
            )

        nid = node_id_start
        compute_nodes: list[ComputeNode] = []
        comm_nodes: list[CommNode] = []
        edges: list[DAGEdge] = []

        if phase == "fwd":
            kernels = _FWD_KERNELS[sublayer_type]
        elif phase == "bwd_ig":
            kernels = _BWD_IG_KERNELS[sublayer_type]
        else:  # bwd_wg
            kernels = _BWD_WG_KERNELS[sublayer_type]

        tp = len(tp_group_ranks)
        prev_node_id: int | None = None

        if phase in ("fwd", "bwd_ig") and tp > 1:
            # AllGather stub
            ag_node = CommNode(
                node_id=nid,
                src_gpu=gpu_rank,
                dst_gpu=gpu_rank,  # placeholder
                bytes=activation_bytes,
                collective_type="AllGather",
                layer_id=layer_idx,
                phase=phase,
                flow_id=-1,
            )
            comm_nodes.append(ag_node)
            prev_node_id = nid
            nid += 1

        for kernel in kernels:
            cn = ComputeNode(
                node_id=nid,
                gpu_rank=gpu_rank,
                kernel=kernel,
                layer_id=layer_idx,
                microbatch_id=microbatch_id,
                pipeline_stage=pipeline_stage,
                phase=phase,
            )
            compute_nodes.append(cn)
            if prev_node_id is not None:
                edges.append(DAGEdge(src_node_id=prev_node_id, dst_node_id=nid))
            prev_node_id = nid
            nid += 1

        if phase in ("fwd", "bwd_ig") and tp > 1:
            # ReduceScatter stub
            rs_node = CommNode(
                node_id=nid,
                src_gpu=gpu_rank,
                dst_gpu=gpu_rank,  # placeholder
                bytes=activation_bytes,
                collective_type="ReduceScatter",
                layer_id=layer_idx,
                phase=phase,
                flow_id=-1,
            )
            comm_nodes.append(rs_node)
            if prev_node_id is not None:
                edges.append(DAGEdge(src_node_id=prev_node_id, dst_node_id=nid))
            nid += 1

        if phase == "bwd_ig" and tp > 1:
            # Second AllGather: backward of the forward ReduceScatter (RowLinear backward).
            # Matches AICB MockedMegatron: RowLinear.backward() emits AllGather(tp).
            ag2_node = CommNode(
                node_id=nid,
                src_gpu=gpu_rank,
                dst_gpu=gpu_rank,  # placeholder
                bytes=activation_bytes,
                collective_type="AllGather",
                layer_id=layer_idx,
                phase=phase,
                flow_id=-1,
            )
            comm_nodes.append(ag2_node)
            edges.append(DAGEdge(src_node_id=nid - 1, dst_node_id=nid))
            nid += 1

        return compute_nodes, comm_nodes, edges, nid

    def _expand_moe(
        self,
        phase: str,
        gpu_rank: int,
        pipeline_stage: int,
        microbatch_id: int,
        layer_idx: int,
        tp_group_ranks: list[int],
        ep_group_ranks: list[int],
        activation_bytes: int,
        moe_data_bytes: int,
        node_id_start: int,
    ) -> tuple[list[ComputeNode], list[CommNode], list[DAGEdge], int]:
        """Expand a MoE sublayer.

        fwd:    [A2A(ep)] -> [AG(tp)] -> moe_norm -> moe_route -> moe_expert -> [RS(tp)] -> [A2A(ep)]
        bwd_ig: [AG(tp)] -> moe_expert -> [A2A(ep)] -> moe_route -> moe_norm -> [RS(tp)]
        bwd_wg: moe_route -> moe_expert
        """
        nid = node_id_start
        compute_nodes: list[ComputeNode] = []
        comm_nodes: list[CommNode] = []
        edges: list[DAGEdge] = []
        tp = len(tp_group_ranks)
        ep = len(ep_group_ranks)
        prev_node_id: int | None = None

        def add_compute(kernel: str) -> None:
            nonlocal nid, prev_node_id
            cn = ComputeNode(
                node_id=nid,
                gpu_rank=gpu_rank,
                kernel=kernel,
                layer_id=layer_idx,
                microbatch_id=microbatch_id,
                pipeline_stage=pipeline_stage,
                phase=phase,
            )
            compute_nodes.append(cn)
            if prev_node_id is not None:
                edges.append(DAGEdge(src_node_id=prev_node_id, dst_node_id=nid))
            prev_node_id = nid
            nid += 1

        def add_comm(collective_type: str, data_bytes: int) -> None:
            nonlocal nid, prev_node_id
            stub = CommNode(
                node_id=nid,
                src_gpu=gpu_rank,
                dst_gpu=gpu_rank,  # placeholder
                bytes=data_bytes,
                collective_type=collective_type,
                layer_id=layer_idx,
                phase=phase,
                flow_id=-1,
            )
            comm_nodes.append(stub)
            if prev_node_id is not None:
                edges.append(DAGEdge(src_node_id=prev_node_id, dst_node_id=nid))
            prev_node_id = nid
            nid += 1

        if phase == "fwd":
            # Order matches AICB MOEMLP: permutation (A2A→AG) then unpermutation (RS→A2A).
            if ep > 1:
                add_comm("AllToAll", moe_data_bytes)   # dispatch
            if tp > 1:
                add_comm("AllGather", activation_bytes)
            add_compute("moe_norm")
            add_compute("moe_route")
            add_compute("moe_expert")
            if tp > 1:
                add_comm("ReduceScatter", activation_bytes)
            if ep > 1:
                add_comm("AllToAll", moe_data_bytes)   # combine
        elif phase == "bwd_ig":
            if tp > 1:
                add_comm("AllGather", activation_bytes)
            add_compute("moe_expert")
            if ep > 1:
                add_comm("AllToAll", moe_data_bytes)
            add_compute("moe_route")
            add_compute("moe_norm")
            if tp > 1:
                add_comm("ReduceScatter", activation_bytes)
        else:  # bwd_wg
            add_compute("moe_route")
            add_compute("moe_expert")

        return compute_nodes, comm_nodes, edges, nid
