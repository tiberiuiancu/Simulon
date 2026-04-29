"""Utility for merging multiple workload DAGs into one."""

from __future__ import annotations

from dataclasses import replace

from simulon.backend.dag.nodes import CommNode, ComputeNode, DAGEdge, ExecutionDAG


def _gpu_count(dag: ExecutionDAG) -> int:
    """Return the number of distinct GPU ranks used in *dag*."""
    all_gpus: set[int] = set()
    for n in dag.compute_nodes:
        all_gpus.add(n.gpu_rank)
    for n in dag.comm_nodes:
        all_gpus.add(n.src_gpu)
        all_gpus.add(n.dst_gpu)
    return max(all_gpus, default=-1) + 1


def _max_node_id(dag: ExecutionDAG) -> int:
    """Return the maximum node_id in *dag*, or -1 if empty."""
    max_id = -1
    for n in dag.compute_nodes:
        if n.node_id > max_id:
            max_id = n.node_id
    for n in dag.comm_nodes:
        if n.node_id > max_id:
            max_id = n.node_id
    return max_id


def _max_flow_id(dag: ExecutionDAG) -> int:
    """Return the maximum flow_id in *dag*, or -1 if empty."""
    return max((n.flow_id for n in dag.comm_nodes), default=-1)


def merge_dags(dags: list[tuple[str, ExecutionDAG]]) -> tuple[ExecutionDAG, dict[int, str]]:
    """Merge multiple independent workload DAGs into a single ExecutionDAG.

    Each workload's node IDs, GPU ranks, and flow IDs are offset so that the
    merged DAG has globally unique identifiers.  No cross-workload edges are
    added.

    Args:
        dags: List of (workload_name, ExecutionDAG) pairs.

    Returns:
        A tuple of (merged ExecutionDAG, node_id_to_workload mapping).
    """
    merged = ExecutionDAG()
    node_id_to_workload: dict[int, str] = {}

    node_id_offset = 0
    gpu_rank_offset = 0
    flow_id_offset = 0

    for workload_name, dag in dags:
        for n in dag.compute_nodes:
            new_node = replace(
                n,
                node_id=n.node_id + node_id_offset,
                gpu_rank=n.gpu_rank + gpu_rank_offset,
            )
            merged.compute_nodes.append(new_node)
            node_id_to_workload[new_node.node_id] = workload_name

        for n in dag.comm_nodes:
            new_node = replace(
                n,
                node_id=n.node_id + node_id_offset,
                src_gpu=n.src_gpu + gpu_rank_offset,
                dst_gpu=n.dst_gpu + gpu_rank_offset,
                flow_id=n.flow_id + flow_id_offset,
                parent_flow_ids=[pid + flow_id_offset for pid in n.parent_flow_ids],
            )
            merged.comm_nodes.append(new_node)
            node_id_to_workload[new_node.node_id] = workload_name

        for e in dag.edges:
            new_edge = DAGEdge(
                src_node_id=e.src_node_id + node_id_offset,
                dst_node_id=e.dst_node_id + node_id_offset,
            )
            merged.edges.append(new_edge)

        node_id_offset = _max_node_id(merged) + 1
        gpu_rank_offset += _gpu_count(dag)
        flow_id_offset = _max_flow_id(merged) + 1

    return merged, node_id_to_workload
