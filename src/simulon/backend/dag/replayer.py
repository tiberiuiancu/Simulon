from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field

from simulon.backend.dag._progress import log_progress
from simulon.backend.dag.nodes import CommNode, ComputeNode, ExecutionDAG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class SimulationResult:
    total_time_ms: float
    per_gpu_times_ms: dict[int, float] = field(default_factory=dict)
    compute_time_ms: dict[int, float] = field(default_factory=dict)
    comm_time_ms: dict[int, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def replay(dag: ExecutionDAG) -> SimulationResult:
    """Critical-path walk over a fully-populated DAG.

    Assumes all node.duration_ms fields have been set before calling:
      - ComputeNode.duration_ms: filled by populate_dag()
      - CommNode.duration_ms:    filled by populate_network() (or a network simulator)

    Pure scheduler — no duration computation happens here. This means any
    network simulator (analytical, NS-3, etc.) can populate CommNode durations
    independently before replay is called.
    """
    # Build unified node map
    all_nodes: dict[int, ComputeNode | CommNode] = {}
    for n in dag.compute_nodes:
        all_nodes[n.node_id] = n
    for n in dag.comm_nodes:
        all_nodes[n.node_id] = n

    # flow_id → node_id (CommNode.parent_flow_ids uses flow_ids, not node_ids)
    flow_to_node: dict[int, int] = {n.flow_id: n.node_id for n in dag.comm_nodes}

    # Build predecessors dict and in-degrees
    predecessors: dict[int, set[int]] = {nid: set() for nid in all_nodes}
    in_degree: dict[int, int] = {nid: 0 for nid in all_nodes}

    with log_progress("  indexing edges", len(dag.edges), logger) as advance:
        for edge in dag.edges:
            predecessors[edge.dst_node_id].add(edge.src_node_id)
            in_degree[edge.dst_node_id] += 1
            advance()

    with log_progress("  indexing flow deps", len(dag.comm_nodes), logger) as advance:
        for cn in dag.comm_nodes:
            for fid in cn.parent_flow_ids:
                if fid in flow_to_node:
                    parent_nid = flow_to_node[fid]
                    if parent_nid not in predecessors[cn.node_id]:
                        predecessors[cn.node_id].add(parent_nid)
                        in_degree[cn.node_id] += 1
            advance()

    # Build successors for Kahn's algorithm
    successors: dict[int, list[int]] = defaultdict(list)
    with log_progress("  building successors", len(dag.edges) + len(dag.comm_nodes), logger) as advance:
        for edge in dag.edges:
            successors[edge.src_node_id].append(edge.dst_node_id)
            advance()
        for cn in dag.comm_nodes:
            for fid in cn.parent_flow_ids:
                if fid in flow_to_node:
                    successors[flow_to_node[fid]].append(cn.node_id)
            advance()

    # Topological sort (Kahn's algorithm)
    temp_in_degree = dict(in_degree)
    queue: deque[int] = deque(nid for nid, deg in temp_in_degree.items() if deg == 0)
    topo_order: list[int] = []
    with log_progress("  topological sort", len(all_nodes), logger) as advance:
        while queue:
            nid = queue.popleft()
            topo_order.append(nid)
            for succ in successors[nid]:
                temp_in_degree[succ] -= 1
                if temp_in_degree[succ] == 0:
                    queue.append(succ)
            advance()

    # Simulation: walk nodes in topological order
    finish_time: dict[int, float] = {}
    per_gpu_compute: dict[int, float] = defaultdict(float)
    per_gpu_comm: dict[int, float] = defaultdict(float)
    per_gpu_finish: dict[int, float] = defaultdict(float)

    with log_progress("  replaying DAG", len(topo_order), logger) as advance:
        for nid in topo_order:
            node = all_nodes[nid]
            start_time = max((finish_time[p] for p in predecessors[nid]), default=0.0)

            if isinstance(node, ComputeNode):
                duration = node.duration_ms if node.duration_ms is not None else 0.0
                finish = start_time + duration
                finish_time[nid] = finish
                node.start_ms = start_time
                node.finish_ms = finish
                per_gpu_compute[node.gpu_rank] += duration
                if finish > per_gpu_finish[node.gpu_rank]:
                    per_gpu_finish[node.gpu_rank] = finish

            else:  # CommNode
                duration = node.duration_ms if node.duration_ms is not None else 0.0
                finish = start_time + duration
                finish_time[nid] = finish
                node.start_ms = start_time
                node.finish_ms = finish
                per_gpu_comm[node.src_gpu] += duration
                per_gpu_comm[node.dst_gpu] += duration
                if finish > per_gpu_finish[node.src_gpu]:
                    per_gpu_finish[node.src_gpu] = finish
                if finish > per_gpu_finish[node.dst_gpu]:
                    per_gpu_finish[node.dst_gpu] = finish

            advance()

    total = max(per_gpu_finish.values(), default=0.0)

    return SimulationResult(
        total_time_ms=total,
        per_gpu_times_ms=dict(per_gpu_finish),
        compute_time_ms=dict(per_gpu_compute),
        comm_time_ms=dict(per_gpu_comm),
    )
