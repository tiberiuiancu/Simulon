"""GOAL (Group Operation Assembly Language) trace export for an ExecutionDAG.

GOAL is the workload format consumed by ATLAHS (and LogGOPSim). It describes
computation and communication patterns as a DAG of tasks per rank, and is used
as input to a network simulator that models transfer timing independently.

Format summary (text):
    num_ranks <N>

    rank <r> {
        <label>: calc <size_ns>
        <label>: send <bytes>b to <dst_rank> tag <tag>
        <label>: recv <bytes>b from <src_rank> tag <tag>
        <label2> requires <label1>   # label2 cannot start before label1 finishes
    }

Mapping from simulon DAG:
    ComputeNode  → calc <duration_ns>     label c{node_id} in gpu_rank's block
    CommNode     → send <bytes>b ...      label s{node_id} in src_gpu's block
                   recv <bytes>b ...      label r{node_id} in dst_gpu's block
    DAGEdge      → requires, inferred per the table below

CommNode.duration_ms is intentionally ignored: ATLAHS models network timing
from its own network model; only the byte count matters here.

Dependency inference for DAGEdge(X → Y):
    X = ComputeNode  →  src_label = c{X.node_id},  src_rank = X.gpu_rank
    X = CommNode     →  src_label = r{X.node_id},  src_rank = X.dst_gpu   (recv side produces)
    Y = ComputeNode  →  dst_label = c{Y.node_id},  dst_rank = Y.gpu_rank
    Y = CommNode     →  dst_label = s{Y.node_id},  dst_rank = Y.src_gpu   (send side consumes)

Most DAG edges are intra-rank under this mapping. The one exception is the
PP_Send fan-out: one CommNode (src=stage_N_rank0, dst=stage_{N+1}_rank0) is
wired by the tracer to the entry nodes of *all* TP/EP ranks on the destination
stage.  For the non-primary ranks (tr>0 or er>0) this yields src_rank ≠
dst_rank in GOAL terms, so those edges are skipped. Their sequencing is
preserved implicitly: each sublayer starts with an AllGather that pulls data
from rank0, so the non-primary ranks already depend on rank0's recv chain.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from simulon.backend.dag.nodes import CommNode, ComputeNode, ExecutionDAG


def dag_to_goal(dag: ExecutionDAG) -> str:
    """Convert a timing-populated ExecutionDAG to GOAL text format.

    Args:
        dag: ExecutionDAG after trace-driven build and replay() have been called
             so that ComputeNode.duration_ms is set on every compute node.

    Returns:
        GOAL schedule as a string, ready to write to a .goal file.

    Raises:
        ValueError: If any ComputeNode.duration_ms is None (unpopulated DAG),
                    or if an edge references a node_id not found in the DAG.
    """
    if not dag.compute_nodes and not dag.comm_nodes:
        raise ValueError("ExecutionDAG is empty; nothing to export to GOAL.")

    # Validate compute timing is populated on every compute node.
    for n in dag.compute_nodes:
        if n.duration_ms is None:
            raise ValueError(
                f"ComputeNode {n.node_id} (kernel={n.kernel!r}, gpu={n.gpu_rank}) "
                "has no duration_ms. Run trace generation first, then replay()."
            )

    # Collect all GPU ranks and derive num_ranks.
    all_ranks: set[int] = set()
    for n in dag.compute_nodes:
        all_ranks.add(n.gpu_rank)
    for n in dag.comm_nodes:
        all_ranks.add(n.src_gpu)
        all_ranks.add(n.dst_gpu)

    num_ranks = max(all_ranks) + 1
    expected_ranks = set(range(num_ranks))
    if all_ranks != expected_ranks:
        raise ValueError(
            f"GPU ranks are not contiguous 0..{num_ranks - 1}: found {sorted(all_ranks)}"
        )

    # Build node lookup maps: node_id → node
    compute_by_id: dict[int, ComputeNode] = {n.node_id: n for n in dag.compute_nodes}
    comm_by_id: dict[int, CommNode] = {n.node_id: n for n in dag.comm_nodes}

    # Group nodes by rank, sorted by node_id for deterministic output.
    compute_by_rank: dict[int, list[ComputeNode]] = defaultdict(list)
    sends_by_rank: dict[int, list[CommNode]] = defaultdict(list)
    recvs_by_rank: dict[int, list[CommNode]] = defaultdict(list)

    for n in sorted(dag.compute_nodes, key=lambda x: x.node_id):
        compute_by_rank[n.gpu_rank].append(n)
    for n in sorted(dag.comm_nodes, key=lambda x: x.node_id):
        sends_by_rank[n.src_gpu].append(n)
        recvs_by_rank[n.dst_gpu].append(n)

    # Build dependency list per rank from DAGEdges.
    # Each entry is (dst_label, src_label), written as "{dst_label} requires {src_label}".
    deps_by_rank: dict[int, list[tuple[str, str]]] = defaultdict(list)

    for edge in dag.edges:
        src_node = compute_by_id.get(edge.src_node_id) or comm_by_id.get(edge.src_node_id)
        dst_node = compute_by_id.get(edge.dst_node_id) or comm_by_id.get(edge.dst_node_id)

        if src_node is None:
            raise ValueError(f"DAGEdge references unknown src_node_id {edge.src_node_id}.")
        if dst_node is None:
            raise ValueError(f"DAGEdge references unknown dst_node_id {edge.dst_node_id}.")

        # Producer: ComputeNode produces at gpu_rank; CommNode produces at dst_gpu
        # (the recv completes there and data becomes available).
        if isinstance(src_node, ComputeNode):
            src_label = f"c{edge.src_node_id}"
            src_rank = src_node.gpu_rank
        else:
            src_label = f"r{edge.src_node_id}"
            src_rank = src_node.dst_gpu

        # Consumer: ComputeNode consumes at gpu_rank; CommNode initiates a send
        # from src_gpu.
        if isinstance(dst_node, ComputeNode):
            dst_label = f"c{edge.dst_node_id}"
            dst_rank = dst_node.gpu_rank
        else:
            dst_label = f"s{edge.dst_node_id}"
            dst_rank = dst_node.src_gpu

        if src_rank != dst_rank:
            # PP_Send fan-out: the tracer wires one PP_Send CommNode (dst_gpu =
            # stage_{N+1} rank0) to *all* TP/EP entry nodes on the destination
            # stage. For non-primary ranks this is a cross-rank edge in GOAL
            # terms. Skip it — those ranks' ordering is preserved by the
            # AllGather that starts every sublayer on rank0's recv chain.
            continue

        deps_by_rank[src_rank].append((dst_label, src_label))

    # Emit GOAL text
    lines: list[str] = [f"num_ranks {num_ranks}"]

    for rank in range(num_ranks):
        lines.append(f"\nrank {rank} {{")

        for n in compute_by_rank.get(rank, []):
            dur_ns = int(n.duration_ms * 1_000_000)  # ms → ns  (1 ms = 1 000 000 ns)
            lines.append(f"c{n.node_id}: calc {dur_ns}")

        for n in sends_by_rank.get(rank, []):
            lines.append(f"s{n.node_id}: send {n.bytes}b to {n.dst_gpu} tag {n.node_id}")

        for n in recvs_by_rank.get(rank, []):
            lines.append(f"r{n.node_id}: recv {n.bytes}b from {n.src_gpu} tag {n.node_id}")

        for dst_label, src_label in sorted(deps_by_rank.get(rank, [])):
            lines.append(f"{dst_label} requires {src_label}")

        lines.append("}")

    return "\n".join(lines) + "\n"


def write_goal_trace(dag: ExecutionDAG, path: str | Path) -> None:
    """Write a GOAL trace file from a timing-populated ExecutionDAG.

    Args:
        dag:  ExecutionDAG after trace-driven build and replay() have been called.
        path: Output path for the .goal file.

    Raises:
        ValueError: If any ComputeNode.duration_ms is None.
    """
    goal = dag_to_goal(dag)
    with open(path, "w", encoding="utf-8") as f:
        f.write(goal)
