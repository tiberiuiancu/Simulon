from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field

from simulon.backend.dag._progress import log_progress
from simulon.backend.dag.nodes import CollectiveNode, CommNode, ComputeNode, ExecutionDAG

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
    total_flops: float | None = None

    # GPU idle caused by the host thread not having issued the work yet, averaged across
    # ranks. Reported separately because it would otherwise land inside bubble_ms and
    # silently change what every published PP-bubble number means. Zero unless host
    # modelling is enabled (NodeSpec.host_cost_us).
    host_stall_ms: float = 0.0


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
    merged_a: list[tuple[float, float]], merged_b: list[tuple[float, float]]
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


def _summarize(dag: ExecutionDAG, total_time_ms: float, network_simulation: str = "flow") -> dict:
    """Derive averaged summary metrics from a fully-replayed DAG."""
    compute_by_gpu: dict[int, list[tuple[float, float]]] = defaultdict(list)
    recv_by_gpu: dict[int, list[tuple[float, float, str]]] = defaultdict(list)
    comm_by_gpu: dict[int, list[tuple[float, float]]] = defaultdict(list)
    collective_by_gpu: dict[int, list[tuple[float, float, str]]] = defaultdict(list)

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

    if network_simulation == "collective":
        for cn in dag.collective_nodes.values():
            if cn.start_ms is None or cn.finish_ms is None or cn.duration_ms is None:
                continue
            iv = (cn.start_ms, cn.finish_ms)
            for gpu in cn.group_ranks:
                collective_by_gpu[gpu].append((cn.start_ms, cn.finish_ms, cn.collective_type))
                comm_by_gpu[gpu].append(iv)

    all_gpus = set(compute_by_gpu) | set(recv_by_gpu) | set(comm_by_gpu) | set(collective_by_gpu)
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
        collective_entries = collective_by_gpu.get(gpu, [])

        compute_ms = sum(e - s for s, e in compute_ivs)

        exposed_by_type: dict[str, float] = defaultdict(float)
        by_type: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for start, finish, ctype in recv_entries:
            by_type[ctype].append((start, finish))
        for start, finish, ctype in collective_entries:
            by_type[ctype].append((start, finish))
        for ctype, ivs in by_type.items():
            type_union = _merge_intervals(ivs)
            exp = max(
                0.0, _union_duration(type_union) - _intersection_duration(type_union, compute_ivs)
            )
            exposed_by_type[ctype] = exp

        all_recv_ivs = _merge_intervals([(s, e) for s, e, _ in recv_entries + collective_entries])
        exposed_total = max(
            0.0, _union_duration(all_recv_ivs) - _intersection_duration(all_recv_ivs, compute_ivs)
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
        t: sum(d.get(t, 0.0) for d in per_gpu_exposed_by_type) / n_gpus for t in sorted(all_types)
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


def replay(
    dag: ExecutionDAG,
    *,
    network_simulation: str = "flow",
    host_cost_us: float | None = None,
) -> SimulationResult:
    """Critical-path walk over a fully-populated DAG.

    Assumes all node.duration_ms fields have been set before calling:
      - ComputeNode.duration_ms: already set by trace_tracer (trace-driven path)
      - CommNode.duration_ms:    filled by populate_network() (flow mode)
      - CollectiveNode.duration_ms: filled by populate_collective_network() (collective mode)

    network_simulation controls whether CollectiveNodes are treated as atomic
    ("collective" mode) or whether the DAG only contains CommNodes ("flow" mode).

    HOST MODELLING (host_cost_us)
    -----------------------------
    None (default) => off, and the walk is bit-identical to the pure-GPU one.

    Set it and each rank additionally gets a serial HOST resource: the rank's host thread
    issues work in `dag.rank_program` order at `host_cost_us` per op, and a node cannot start
    on the GPU before its host thread has finished issuing everything ahead of it. GPU idle
    then emerges wherever the host cannot keep up, instead of being charged as a flat
    surcharge per collective.

    This exists because nsys profiling of Qwen3-1.7B on 4x GH200 showed 36-41% of every
    iteration is GPU idle on the pace-setting rank, and that 79-83% of that idle is host
    time in framework code with no CUDA call in flight (experiments/gap_attribution.py).
    Modelling it as a per-collective constant is what made NcclProfile.launch_latency_ms
    drift 22-26% across microbatch size: bigger microbatches make kernels longer, which
    hides more host time, and no per-collective constant can express that.

    Two properties worth preserving if this is ever refactored:
      * The host clock does NOT wait on GPU dependencies. CUDA launches are asynchronous;
        making the host block on a dependency would fully expose every issue cost and
        re-derive the flat per-collective surcharge this replaces.
      * It DOES stop at a blocking sync point (`dag.host_sync_nodes`): pipeline P2P with
        overlap-p2p-comm off. Past one of those the host genuinely cannot run ahead.
        Without this the replay reports a perfectly pipelined schedule -- PP 1->2 came out
        at +3.2% against a measured +45.9%, with the bottleneck rank's own work explaining
        the whole iteration.

        This is grounded in the source, not inferred: megatron/core/pipeline_parallel/
        p2p_communication.py:452 ends every `_communicate` with a full
        `torch.cuda.synchronize()` when `batch_p2p_comm and batch_p2p_sync`, and BOTH
        default to True (model_parallel_config.py:303,308). So each p2p exchange drains the
        whole device and blocks the host -- if anything stronger than what is modelled here,
        which only advances the host clock to that node's own finish. Corroborated by nsys:
        on the LAST pipeline stage of tp2pp2-mbs2 (dev2) there are 2051 ms of inter-kernel
        gaps per iteration, 68% of it framework time, even though that stage's host work
        (10463 ms) is well under its GPU work (12812 ms) -- i.e. idle that a freely-running-
        ahead host model predicts as zero.

    A `host_queue_kernels` bound on run-ahead (by pending CUDA launches) was tried and
    REMOVED, twice falsified. On the fake-PG grid it throttled PP=1 far harder than it
    helped deep pipelines (mean |error| 10.8% -> 12.4%); re-tested on the real-PG grid --
    where the trace fidelity that confounded the first test is fixed -- it was also found
    to be silently self-disabling: once the required budget exceeded every retired entry
    the monotone head pointer ran off the end and no constraint was applied again for that
    rank, so a *tighter* queue throttled *less* (q=1024 gave tp1pp1-mbs1 +19.5%, q=256 gave
    +0.3%, identical to no bound at all). Beyond the bug, the DAG's granularity is one node
    per slot -- ~760 kernels on these cells -- which is coarser than any plausible queue
    depth, so the bound cannot be expressed here without inventing a fitted parameter.
    The physics it aimed at (host cannot run ahead forever) is instead carried by
    `dag.host_sync_nodes`, which is grounded: see below.
      * It is a pre-pass, not a clock updated inside the walk, so the result cannot depend
        on how Kahn's algorithm breaks ties.
    """
    # Build unified node map
    all_nodes: dict[int, ComputeNode | CommNode | CollectiveNode] = {}
    for n in dag.compute_nodes:
        all_nodes[n.node_id] = n
    for n in dag.comm_nodes:
        all_nodes[n.node_id] = n
    if network_simulation == "collective":
        for n in dag.collective_nodes.values():
            all_nodes[n.node_id] = n

    # flow_id → node_id (CommNode.parent_flow_ids uses flow_ids, not node_ids)
    flow_to_node: dict[int, int] = {n.flow_id: n.node_id for n in dag.comm_nodes}

    # Build predecessors dict and in-degrees
    predecessors: dict[int, set[int]] = {nid: set() for nid in all_nodes}
    in_degree: dict[int, int] = dict.fromkeys(all_nodes, 0)

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

    # Host issue times. Two rules, and the distinction between them is the whole point:
    #   * the host does NOT wait on GPU dependencies -- CUDA launches are asynchronous, so
    #     making it block everywhere would fully expose every issue cost and re-derive the
    #     flat per-collective surcharge this replaces;
    #   * but it DOES stop at a blocking sync point (dag.host_sync_nodes: pipeline P2P with
    #     overlap-p2p-comm off, where Megatron issues a blocking recv). Past one of those
    #     the host genuinely cannot run ahead.
    # Without the second rule the host runs ahead across the entire iteration and the replay
    # reports a perfectly-pipelined schedule: PP 1->2 came out at +3.2% against a measured
    # +45.9%, with the bottleneck rank's own work (10796 ms) explaining the whole 10928 ms
    # iteration while hardware needed 14479 ms.
    # A collective appears in every participant's program, so take the latest of its issue
    # times -- it is not on the wire until the last rank has issued it.
    host_ready: dict[int, float] = {}
    rank_of_node: dict[int, list[int]] = {}
    if host_cost_us is not None:
        if not dag.rank_program:
            raise ValueError(
                "host modelling requested but the DAG carries no rank_program. Host time "
                "comes from kernel-timing traces (SIMULON_TRACE_KERNEL_TIME=1); a span-only "
                "trace already has the host gaps baked into its wall-clock spans, so "
                "modelling host time on top of it would double-count."
            )
        for rank, program in dag.rank_program.items():
            for nid in program:
                rank_of_node.setdefault(nid, []).append(rank)

    # Simulation: walk nodes in topological order
    finish_time: dict[int, float] = {}
    per_gpu_finish: dict[int, float] = defaultdict(float)
    host_stall_by_gpu: dict[int, float] = defaultdict(float)

    host_clock: dict[int, float] = defaultdict(float)
    with log_progress("  replaying DAG", len(topo_order), logger) as advance:
        for nid in topo_order:
            node = all_nodes[nid]
            start_time = max((finish_time[p] for p in predecessors[nid]), default=0.0)
            ranks = rank_of_node.get(nid)
            if ranks is not None:
                cost = getattr(node, "host_ops", 0.0) * host_cost_us / 1000.0
                issued = max(host_clock[r] for r in ranks) + cost
                for r in ranks:
                    host_clock[r] = issued
                if issued > start_time:
                    if isinstance(node, ComputeNode):
                        host_stall_by_gpu[node.gpu_rank] += issued - start_time
                    start_time = issued

            if isinstance(node, ComputeNode):
                duration = node.duration_ms if node.duration_ms is not None else 0.0
                finish = start_time + duration
                finish_time[nid] = finish
                node.start_ms = start_time
                node.finish_ms = finish
                if finish > per_gpu_finish[node.gpu_rank]:
                    per_gpu_finish[node.gpu_rank] = finish

            else:
                duration = node.duration_ms if node.duration_ms is not None else 0.0
                finish = start_time + duration
                finish_time[nid] = finish
                node.start_ms = start_time
                node.finish_ms = finish
                if nid in dag.host_sync_nodes and ranks is not None:
                    # Blocking recv: the host thread is parked until it completes, so it
                    # cannot have been issuing work behind this point.
                    for r in ranks:
                        if finish > host_clock[r]:
                            host_clock[r] = finish
                if isinstance(node, CommNode):
                    if finish > per_gpu_finish[node.src_gpu]:
                        per_gpu_finish[node.src_gpu] = finish
                    if finish > per_gpu_finish[node.dst_gpu]:
                        per_gpu_finish[node.dst_gpu] = finish
                elif isinstance(node, CollectiveNode):
                    for gpu in node.group_ranks:
                        if finish > per_gpu_finish[gpu]:
                            per_gpu_finish[gpu] = finish

            advance()

    total = max(per_gpu_finish.values(), default=0.0)
    summary = _summarize(dag, total, network_simulation)
    n_gpu = len(per_gpu_finish) or 1

    return SimulationResult(
        total_time_ms=total,
        per_gpu_times_ms=dict(per_gpu_finish),
        host_stall_ms=sum(host_stall_by_gpu.values()) / n_gpu,
        **summary,
    )
