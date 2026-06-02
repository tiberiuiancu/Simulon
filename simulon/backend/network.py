from __future__ import annotations

import logging

from simulon.backend.dag._progress import log_progress
from simulon.backend.dag.nodes import CommNode, DAGEdge, ExecutionDAG
from simulon.collective.decompose import decompose_collective

logger = logging.getLogger(__name__)


def decompose_collectives_in_dag(dag: ExecutionDAG) -> ExecutionDAG:
    """Replace CollectiveNodes with decomposed P2P CommNodes.

    Mutates the DAG in-place.  Uses dag.edges (real edges) instead of the
    old pending_edges mechanism.
    """
    if not dag.collective_nodes:
        return dag

    node_id_to_rank: dict[int, int] = {}
    for n in dag.compute_nodes:
        node_id_to_rank[n.node_id] = n.gpu_rank
    for n in dag.comm_nodes:
        node_id_to_rank[n.node_id] = n.src_gpu

    max_node_id = max(
        [n.node_id for n in dag.compute_nodes]
        + [n.node_id for n in dag.comm_nodes]
        + [n.node_id for n in dag.collective_nodes.values()],
        default=-1,
    )
    node_id = max_node_id + 1

    max_flow_id = max([n.flow_id for n in dag.comm_nodes], default=-1)
    flow_id = max_flow_id + 1

    collective_ids = set(dag.collective_nodes.keys())
    normal_edges: list[DAGEdge] = []
    incoming: dict[int, list[DAGEdge]] = {cid: [] for cid in collective_ids}
    outgoing: dict[int, list[DAGEdge]] = {cid: [] for cid in collective_ids}

    for edge in dag.edges:
        if edge.dst_node_id in collective_ids:
            incoming[edge.dst_node_id].append(edge)
        elif edge.src_node_id in collective_ids:
            outgoing[edge.src_node_id].append(edge)
        else:
            normal_edges.append(edge)

    dag.edges = normal_edges

    collective_nodes = sorted(dag.collective_nodes.values(), key=lambda n: n.node_id)

    with log_progress("  decomposing collectives", len(collective_nodes), logger) as advance:
        for C in collective_nodes:
            result, next_flow_id = decompose_collective(
                collective_type=C.collective_type,
                group_ranks=C.group_ranks,
                data_size=C.data_size,
                num_channels=C.num_channels,
                algorithm=C.algorithm,
                flow_id_start=flow_id,
            )

            first_p2p_id = node_id
            if not result.flows:
                advance()
                continue

            for flow in result.flows:
                dag.add_comm_node(
                    CommNode(
                        node_id=node_id,
                        src_gpu=flow.src,
                        dst_gpu=flow.dst,
                        bytes=flow.flow_size,
                        collective_type=C.collective_type,
                        layer_id=-1,
                        phase=C.phase,
                        flow_id=flow.flow_id,
                        parent_flow_ids=flow.parent_flow_ids,
                    )
                )
                node_id_to_rank[node_id] = flow.src
                node_id += 1

            entry_flows: dict[int, list[int]] = {}
            exit_flows: dict[int, list[int]] = {}

            for idx, flow in enumerate(result.flows):
                nid = first_p2p_id + idx
                if not flow.parent_flow_ids:
                    entry_flows.setdefault(flow.src, []).append(nid)
                    entry_flows.setdefault(flow.dst, []).append(nid)
                if not flow.child_flow_ids:
                    exit_flows.setdefault(flow.src, []).append(nid)
                    exit_flows.setdefault(flow.dst, []).append(nid)

            for edge in incoming.get(C.node_id, []):
                src_rank = node_id_to_rank.get(edge.src_node_id)
                if src_rank is not None and src_rank in entry_flows:
                    for entry_nid in entry_flows[src_rank]:
                        dag.add_edge(DAGEdge(src_node_id=edge.src_node_id, dst_node_id=entry_nid))
                else:
                    for nids in entry_flows.values():
                        for entry_nid in nids:
                            dag.add_edge(
                                DAGEdge(src_node_id=edge.src_node_id, dst_node_id=entry_nid)
                            )

            for edge in outgoing.get(C.node_id, []):
                dst_rank = node_id_to_rank.get(edge.dst_node_id)
                if dst_rank is not None and dst_rank in exit_flows:
                    for exit_nid in exit_flows[dst_rank]:
                        dag.add_edge(DAGEdge(src_node_id=exit_nid, dst_node_id=edge.dst_node_id))
                else:
                    for nids in exit_flows.values():
                        for exit_nid in nids:
                            dag.add_edge(
                                DAGEdge(src_node_id=exit_nid, dst_node_id=edge.dst_node_id)
                            )

            flow_id = next_flow_id
            advance()

    dag.collective_nodes.clear()
    return dag
