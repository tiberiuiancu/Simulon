from simulon.backend.dag.nodes import CommNode, ExecutionDAG
from simulon.collective import CCLDecomposer
from simulon.collective.decompose import decompose_collective
from simulon.config.dc import DatacenterConfig
from simulon.config.workload import CollectiveWorkload


def build_collective_dag(
    workload: CollectiveWorkload,
    datacenter: DatacenterConfig,
    algorithm: str,
    num_channels: int,
    ccl: CCLDecomposer,
) -> ExecutionDAG:
    """Build a CommNode-only ExecutionDAG for a single collective operation."""
    num_ranks = datacenter.cluster.num_nodes * datacenter.node.gpus_per_node
    group_ranks = list(range(num_ranks))

    result, _ = decompose_collective(
        collective_type=workload.collective_type.value,
        group_ranks=group_ranks,
        data_size=workload.message_size_bytes,
        num_channels=num_channels,
        algorithm=algorithm,
        flow_id_start=0,
    )

    dag = ExecutionDAG()
    for node_id, flow in enumerate(result.flows):
        dag.comm_nodes.append(CommNode(
            node_id=node_id,
            src_gpu=flow.src,
            dst_gpu=flow.dst,
            bytes=flow.flow_size,
            collective_type=workload.collective_type.value,
            layer_id=0,
            phase="collective",
            flow_id=flow.flow_id,
            parent_flow_ids=list(flow.parent_flow_ids),
        ))

    return dag
