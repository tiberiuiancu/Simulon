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
    """Post-replay simulation metrics.

    Summary fields (compute_ms, exposed_comm_ms, etc.) are averaged across
    all GPU ranks.  per_gpu_times_ms contains the raw per-GPU finish times.

    Breakdown
    ---------
    The three primary components sum to total_time_ms (within floating-point
    rounding):

    * compute_ms      – GPU actively running kernels.
    * exposed_comm_ms – GPU blocked waiting for a recv to complete (dst side)
                        while no compute is running.
    * bubble_ms       – Remaining idle time: total - compute - exposed_comm.
                        In 1F1B schedules this is dominated by warm-up / drain
                        gaps.  Note: time spent only sending (src side, no
                        concurrent compute or recv) also falls here, since
                        sends are async from the GPU's perspective.

    overlapped_comm_ms is informational: comm (send + recv) that ran
    concurrently with compute and is therefore hidden from the critical path.
    It is NOT included in the three components above.
    """

    total_time_ms: float

    # --- averaged across GPUs ---
    compute_ms: float
    exposed_comm_ms: float
    exposed_comm_by_type: dict[str, float]  # collective_type -> avg ms
    bubble_ms: float
    overlapped_comm_ms: float  # informational, not in totals

    # --- raw per-GPU ---
    per_gpu_times_ms: dict[int, float] = field(default_factory=dict)
    total_flops: Optional[float] = None


# ---------------------------------------------------------------------------
# Interval helpers
# ---------------------------------------------------------------------------


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return a sorted, non-overlapping list of merged intervals."""
    if not intervals:
        return []
    merged: list[list[float]] = []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _union_duration(intervals: list[tuple[float, float]]) -> float:
    return sum(e - s for s, e in _merge_intervals(intervals))


def _intersection_duration(
    merged_a: list[tuple[float, float]],
    merged_b: list[tuple[float, float]],
) -> float:
    """Total duration of the intersection of two *already-merged* interval lists."""
    total = 0.0
    i = j = 0
    while i < len(merged_a) and j < len(merged_b):
        lo = max(merged_a[i][0], merged_b[j][0])
        hi = min(merged_a[i][1], merged_b[j][1])
        if lo < hi:
            total += hi - lo
        end_a, end_b = merged_a[i][1], merged_b[j][1]
        if end_a < end_b:
            i += 1
        elif end_b < end_a:
            j += 1
        else:
            i += 1
            j += 1
    return total


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------


def _summarize(dag: ExecutionDAG, total_time_ms: float) -> dict:
    """Derive averaged summary metrics from a fully-replayed DAG."""
    compute_by_gpu: dict[int, list[tuple[float, float]]] = defaultdict(list)
    recv_by_gpu: dict[int, list[tuple[float, float, str]]] = defaultdict(list)
    comm_by_gpu: dict[int, list[tuple[float, float]]] = defaultdict(list)

    for n in dag.compute_nodes:
        if n.start_ms is not None and n.finish_ms is not None:
            compute_by_gpu[n.gpu_rank].append((n.start_ms, n.finish_ms))

    for n in dag.comm_nodes:
        if n.start_ms is None or n.finish_ms is None:
            continue
        iv = (n.start_ms, n.finish_ms)
        recv_by_gpu[n.dst_gpu].append((n.start_ms, n.finish_ms, n.collective_type))
        comm_by_gpu[n.src_gpu].append(iv)
        comm_by_gpu[n.dst_gpu].append(iv)

    all_gpus = set(compute_by_gpu) | set(recv_by_gpu) | set(comm_by_gpu)
    if not all_gpus:
        return {
            "compute_ms": 0.0,
            "exposed_comm_ms": 0.0,
            "exposed_comm_by_type": {},
            "bubble_ms": total_time_ms,
            "overlapped_comm_ms": 0.0,
        }

    per_gpu_compute: list[float] = []
    per_gpu_exposed: list[float] = []
    per_gpu_exposed_by_type: list[dict[str, float]] = []
    per_gpu_bubble: list[float] = []
    per_gpu_overlapped: list[float] = []

    for gpu in all_gpus:
        compute_ivs = _merge_intervals(compute_by_gpu.get(gpu, []))
        recv_entries = recv_by_gpu.get(gpu, [])
        comm_ivs = _merge_intervals(comm_by_gpu.get(gpu, []))

        compute_ms = sum(e - s for s, e in compute_ivs)

        # Per-type exposed comm: group by type, compute union per type,
        # then subtract hidden portion (overlap with compute).  This avoids
        # double-counting overlapping recv nodes of the same type.
        exposed_by_type: dict[str, float] = defaultdict(float)
        by_type: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for start, finish, ctype in recv_entries:
            by_type[ctype].append((start, finish))
        for ctype, ivs in by_type.items():
            type_union = _merge_intervals(ivs)
            exp = max(
                0.0,
                _union_duration(type_union)
                - _intersection_duration(type_union, compute_ivs),
            )
            exposed_by_type[ctype] = exp

        # Compute total exposed using the union of all recv intervals to avoid
        # double-counting when two recv nodes overlap on the same GPU.
        all_recv_ivs = _merge_intervals([(s, e) for s, e, _ in recv_entries])
        exposed_total = max(
            0.0,
            _union_duration(all_recv_ivs)
            - _intersection_duration(all_recv_ivs, compute_ivs),
        )

        overlapped = _intersection_duration(comm_ivs, compute_ivs)
        bubble = max(0.0, total_time_ms - compute_ms - exposed_total)

        per_gpu_compute.append(compute_ms)
        per_gpu_exposed.append(exposed_total)
        per_gpu_exposed_by_type.append(dict(exposed_by_type))
        per_gpu_bubble.append(bubble)
        per_gpu_overlapped.append(overlapped)

    n_gpus = len(all_gpus)
    all_types = {k for d in per_gpu_exposed_by_type for k in d}
    avg_exposed_by_type = {
        t: sum(d.get(t, 0.0) for d in per_gpu_exposed_by_type) / n_gpus
        for t in sorted(all_types)
    }

    return {
        "compute_ms": sum(per_gpu_compute) / n_gpus,
        "exposed_comm_ms": sum(per_gpu_exposed) / n_gpus,
        "exposed_comm_by_type": avg_exposed_by_type,
        "bubble_ms": sum(per_gpu_bubble) / n_gpus,
        "overlapped_comm_ms": sum(per_gpu_overlapped) / n_gpus,
    }


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def replay(dag: ExecutionDAG) -> SimulationResult:
    """Critical-path walk over a fully-populated DAG.

    Assumes all node.duration_ms fields have been set before calling:
      - ComputeNode.duration_ms: already set by trace_tracer (trace-driven path)
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
    with log_progress(
        "  building successors", len(dag.edges) + len(dag.comm_nodes), logger
    ) as advance:
        for edge in dag.edges:
            successors[edge.src_node_id].append(edge.dst_node_id)
            advance()
        for cn in dag.comm_nodes:
            for fid in cn.parent_flow_ids:
                if fid in flow_to_node:
                    successors[flow_to_node[fid]].append(cn.node_id)
            advance()

    bad_edges = []
    for edge in dag.edges:
        if edge.src_node_id not in all_nodes:
            bad_edges.append(("src missing", edge.src_node_id, edge.dst_node_id))
        if edge.dst_node_id not in all_nodes:
            bad_edges.append(("dst missing", edge.src_node_id, edge.dst_node_id))
    if bad_edges:
        raise ValueError(
            f"DAG has {len(bad_edges)} edges pointing to non-existent nodes: {bad_edges[:10]}"
        )

    bad_flows = []
    for cn in dag.comm_nodes:
        for fid in cn.parent_flow_ids:
            if fid in flow_to_node:
                parent_nid = flow_to_node[fid]
                if parent_nid not in all_nodes:
                    bad_flows.append((cn.node_id, fid, parent_nid))
    if bad_flows:
        raise ValueError(
            f"DAG has {len(bad_flows)} flow deps pointing to non-existent nodes: {bad_flows[:10]}"
        )

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

    if len(topo_order) != len(all_nodes):
        missing = set(all_nodes) - set(topo_order)
        raise ValueError(
            f"Topo sort incomplete: {len(missing)} nodes not processed: {list(missing)[:10]}"
        )

    # Simulation: walk nodes in topological order
    finish_time: dict[int, float] = {}
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
                if finish > per_gpu_finish[node.gpu_rank]:
                    per_gpu_finish[node.gpu_rank] = finish

            else:  # CommNode
                duration = node.duration_ms if node.duration_ms is not None else 0.0
                finish = start_time + duration
                finish_time[nid] = finish
                node.start_ms = start_time
                node.finish_ms = finish
                if finish > per_gpu_finish[node.src_gpu]:
                    per_gpu_finish[node.src_gpu] = finish
                if finish > per_gpu_finish[node.dst_gpu]:
                    per_gpu_finish[node.dst_gpu] = finish

            advance()

    total = max(per_gpu_finish.values(), default=0.0)
    summary = _summarize(dag, total)

    return SimulationResult(
        total_time_ms=total,
        per_gpu_times_ms=dict(per_gpu_finish),
        **summary,
    )
