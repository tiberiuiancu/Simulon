from simulon.backend.dag.nodes import CollectiveNode, ExecutionDAG
from simulon.collective import CCLDecomposer
from simulon.config.dc import DatacenterConfig
from simulon.config.resolve import resolve_node_spec
from simulon.config.workload import CollectiveWorkload


def build_collective_dag(
    workload: CollectiveWorkload,
    datacenter: DatacenterConfig,
    algorithm: str,
    num_channels: int,
    ccl: CCLDecomposer,
) -> ExecutionDAG:
    node = resolve_node_spec(datacenter)
    gpus_per_node = node.gpus_per_node
    if gpus_per_node is None:
        raise ValueError("node.gpus_per_node must be set after resolution")
    num_ranks = datacenter.cluster.num_nodes * gpus_per_node
    group_ranks = list(range(num_ranks))

    dag = ExecutionDAG()
    dag.add_collective_node(
        CollectiveNode(
            node_id=0,
            collective_type=workload.collective_type.value,
            group_ranks=group_ranks,
            data_size=workload.message_size_bytes,
            name="collective",
            timestamp_ms=0.0,
            layer_id=0,
            phase="collective",
            algorithm=algorithm,
            num_channels=num_channels,
        )
    )
    return dag
