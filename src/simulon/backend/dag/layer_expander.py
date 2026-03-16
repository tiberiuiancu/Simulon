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
        sublayer_type: Literal["attn", "mlp"],
        phase: Literal["fwd", "bwd_ig", "bwd_wg"],
        gpu_rank: int,
        pipeline_stage: int,
        microbatch_id: int,
        layer_idx: int,
        tp_group_ranks: list[int],
        activation_bytes: int,
        node_id_start: int,
    ) -> tuple[list[ComputeNode], list[CommNode], list[DAGEdge], int]:
        """Expand a sublayer into compute and comm nodes.

        For fwd/bwd_ig: AllGather(TP) -> kernels -> ReduceScatter(TP)
        For bwd_wg: kernels -> AllReduce(DP)  [DP handled at tracer level]

        Returns comm stubs with flow_id=-1; tracer fills these from decompose_collective.
        Returns (compute_nodes, comm_nodes, edges, next_node_id).
        """
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

        return compute_nodes, comm_nodes, edges, nid
