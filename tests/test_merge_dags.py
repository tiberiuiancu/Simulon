"""Tests for DAG merge utility."""

import pytest

from simulon.backend.dag.merge import merge_dags
from simulon.backend.dag.nodes import CommNode, ComputeNode, DAGEdge, ExecutionDAG


def _compute(node_id: int, gpu_rank: int = 0) -> ComputeNode:
    return ComputeNode(
        node_id=node_id,
        gpu_rank=gpu_rank,
        kernel="layernorm",
        layer_id=0,
        microbatch_id=0,
        pipeline_stage=0,
        phase="fwd",
    )


def _comm(
    node_id: int,
    flow_id: int,
    src_gpu: int = 0,
    dst_gpu: int = 1,
    parent_flow_ids: list[int] | None = None,
) -> CommNode:
    return CommNode(
        node_id=node_id,
        src_gpu=src_gpu,
        dst_gpu=dst_gpu,
        bytes=1024,
        collective_type="AllReduce",
        layer_id=0,
        phase="fwd",
        flow_id=flow_id,
        parent_flow_ids=parent_flow_ids or [],
    )


def _edge(src: int, dst: int) -> DAGEdge:
    return DAGEdge(src_node_id=src, dst_node_id=dst)


class TestMergeDags:
    def test_empty_list_returns_empty_dag(self):
        merged, mapping = merge_dags([])
        assert merged.compute_nodes == []
        assert merged.comm_nodes == []
        assert merged.edges == []
        assert mapping == {}

    def test_single_dag_identity(self):
        dag = ExecutionDAG(
            compute_nodes=[_compute(0, 0), _compute(1, 1)],
            comm_nodes=[_comm(2, 0, 0, 1, [])],
            edges=[_edge(0, 1), _edge(1, 2)],
        )
        merged, mapping = merge_dags([("a", dag)])

        assert len(merged.compute_nodes) == 2
        assert len(merged.comm_nodes) == 1
        assert len(merged.edges) == 2
        assert merged.compute_nodes[0].node_id == 0
        assert merged.compute_nodes[1].node_id == 1
        assert merged.comm_nodes[0].node_id == 2
        assert mapping == {0: "a", 1: "a", 2: "a"}

    def test_single_dag_creates_new_objects(self):
        original = ExecutionDAG(
            compute_nodes=[_compute(0, 0)],
            comm_nodes=[_comm(1, 0, 0, 1)],
            edges=[_edge(0, 1)],
        )
        merged, _ = merge_dags([("a", original)])

        assert merged.compute_nodes[0] is not original.compute_nodes[0]
        assert merged.comm_nodes[0] is not original.comm_nodes[0]
        assert merged.edges[0] is not original.edges[0]

    def test_two_dags_node_ids_unique(self):
        dag_a = ExecutionDAG(
            compute_nodes=[_compute(0, 0), _compute(1, 0)],
        )
        dag_b = ExecutionDAG(
            compute_nodes=[_compute(0, 0), _compute(1, 0)],
        )
        merged, _ = merge_dags([("a", dag_a), ("b", dag_b)])

        node_ids = [n.node_id for n in merged.compute_nodes]
        assert node_ids == [0, 1, 2, 3]

    def test_two_dags_gpu_ranks_offset(self):
        dag_a = ExecutionDAG(
            compute_nodes=[_compute(0, 0), _compute(1, 1)],
            comm_nodes=[_comm(2, 0, 0, 1)],
        )
        dag_b = ExecutionDAG(
            compute_nodes=[_compute(0, 0), _compute(1, 1)],
            comm_nodes=[_comm(2, 0, 0, 1)],
        )
        merged, _ = merge_dags([("a", dag_a), ("b", dag_b)])

        a_compute = merged.compute_nodes[:2]
        b_compute = merged.compute_nodes[2:]
        a_comm = merged.comm_nodes[0]
        b_comm = merged.comm_nodes[1]

        assert a_compute[0].gpu_rank == 0
        assert a_compute[1].gpu_rank == 1
        assert b_compute[0].gpu_rank == 2
        assert b_compute[1].gpu_rank == 3

        assert a_comm.src_gpu == 0 and a_comm.dst_gpu == 1
        assert b_comm.src_gpu == 2 and b_comm.dst_gpu == 3

    def test_two_dags_flow_ids_unique(self):
        dag_a = ExecutionDAG(
            comm_nodes=[_comm(0, 0), _comm(1, 1)],
        )
        dag_b = ExecutionDAG(
            comm_nodes=[_comm(0, 0), _comm(1, 1)],
        )
        merged, _ = merge_dags([("a", dag_a), ("b", dag_b)])

        flow_ids = [n.flow_id for n in merged.comm_nodes]
        assert flow_ids == [0, 1, 2, 3]

    def test_two_dags_parent_flow_ids_offset(self):
        dag_a = ExecutionDAG(
            comm_nodes=[
                _comm(0, 0, parent_flow_ids=[]),
                _comm(1, 1, parent_flow_ids=[0]),
            ],
        )
        dag_b = ExecutionDAG(
            comm_nodes=[
                _comm(0, 0, parent_flow_ids=[]),
                _comm(1, 1, parent_flow_ids=[0]),
            ],
        )
        merged, _ = merge_dags([("a", dag_a), ("b", dag_b)])

        a_comm = merged.comm_nodes[:2]
        b_comm = merged.comm_nodes[2:]

        assert a_comm[0].parent_flow_ids == []
        assert a_comm[1].parent_flow_ids == [0]
        assert b_comm[0].parent_flow_ids == []
        assert b_comm[1].parent_flow_ids == [2]

    def test_two_dags_edges_offset(self):
        dag_a = ExecutionDAG(
            compute_nodes=[_compute(0, 0)],
            comm_nodes=[_comm(1, 0)],
            edges=[_edge(0, 1)],
        )
        dag_b = ExecutionDAG(
            compute_nodes=[_compute(0, 0)],
            comm_nodes=[_comm(1, 0)],
            edges=[_edge(0, 1)],
        )
        merged, _ = merge_dags([("a", dag_a), ("b", dag_b)])

        assert merged.edges[0].src_node_id == 0 and merged.edges[0].dst_node_id == 1
        assert merged.edges[1].src_node_id == 2 and merged.edges[1].dst_node_id == 3

    def test_input_dags_not_modified(self):
        orig_compute = _compute(0, 0)
        orig_comm = _comm(1, 0, parent_flow_ids=[])
        orig_edge = _edge(0, 1)
        dag = ExecutionDAG(
            compute_nodes=[orig_compute],
            comm_nodes=[orig_comm],
            edges=[orig_edge],
        )
        merge_dags([("a", dag)])

        assert orig_compute.node_id == 0
        assert orig_compute.gpu_rank == 0
        assert orig_comm.node_id == 1
        assert orig_comm.src_gpu == 0
        assert orig_comm.dst_gpu == 1
        assert orig_comm.flow_id == 0
        assert orig_comm.parent_flow_ids == []
        assert orig_edge.src_node_id == 0
        assert orig_edge.dst_node_id == 1

    def test_node_id_to_workload_mapping(self):
        dag_a = ExecutionDAG(
            compute_nodes=[_compute(0, 0)],
            comm_nodes=[_comm(1, 0)],
        )
        dag_b = ExecutionDAG(
            compute_nodes=[_compute(0, 0), _compute(1, 0)],
            comm_nodes=[_comm(2, 0), _comm(3, 1)],
        )
        _, mapping = merge_dags([("foo", dag_a), ("bar", dag_b)])

        assert mapping == {
            0: "foo",
            1: "foo",
            2: "bar",
            3: "bar",
            4: "bar",
            5: "bar",
        }

    def test_dag_with_only_comm_nodes(self):
        dag = ExecutionDAG(
            comm_nodes=[_comm(0, 0), _comm(1, 1)],
        )
        merged, _ = merge_dags([("a", dag)])
        assert len(merged.comm_nodes) == 2
        assert len(merged.compute_nodes) == 0

    def test_three_dags_cumulative_offsets(self):
        dag_a = ExecutionDAG(
            compute_nodes=[_compute(0, 0)],
            comm_nodes=[_comm(1, 0, 0, 0)],
        )
        dag_b = ExecutionDAG(
            compute_nodes=[_compute(0, 0), _compute(1, 1)],
            comm_nodes=[_comm(2, 0, 0, 1), _comm(3, 1, 1, 0)],
        )
        dag_c = ExecutionDAG(
            compute_nodes=[_compute(0, 0)],
            comm_nodes=[_comm(1, 0, 0, 0)],
        )
        merged, mapping = merge_dags([("a", dag_a), ("b", dag_b), ("c", dag_c)])

        assert len(merged.compute_nodes) == 4
        assert len(merged.comm_nodes) == 4

        assert merged.compute_nodes[0].node_id == 0 and merged.compute_nodes[0].gpu_rank == 0
        assert merged.compute_nodes[1].node_id == 2 and merged.compute_nodes[1].gpu_rank == 1
        assert merged.compute_nodes[2].node_id == 3 and merged.compute_nodes[2].gpu_rank == 2
        assert merged.compute_nodes[3].node_id == 6 and merged.compute_nodes[3].gpu_rank == 3

        assert merged.comm_nodes[0].node_id == 1 and merged.comm_nodes[0].flow_id == 0
        assert merged.comm_nodes[1].node_id == 4 and merged.comm_nodes[1].flow_id == 1
        assert merged.comm_nodes[2].node_id == 5 and merged.comm_nodes[2].flow_id == 2
        assert merged.comm_nodes[3].node_id == 7 and merged.comm_nodes[3].flow_id == 3

        assert mapping[0] == "a"
        assert mapping[1] == "a"
        assert mapping[2] == "b"
        assert mapping[6] == "c"

    def test_replayer_runs_on_merged_dag(self):
        from simulon.backend.dag.replayer import replay

        dag_a = ExecutionDAG(
            compute_nodes=[
                ComputeNode(
                    node_id=0,
                    gpu_rank=0,
                    kernel="layernorm",
                    layer_id=0,
                    microbatch_id=0,
                    pipeline_stage=0,
                    phase="fwd",
                    duration_ms=1.0,
                ),
            ],
        )
        dag_b = ExecutionDAG(
            compute_nodes=[
                ComputeNode(
                    node_id=0,
                    gpu_rank=0,
                    kernel="layernorm",
                    layer_id=0,
                    microbatch_id=0,
                    pipeline_stage=0,
                    phase="fwd",
                    duration_ms=2.0,
                ),
            ],
        )
        merged, _ = merge_dags([("a", dag_a), ("b", dag_b)])
        result = replay(merged)

        assert result.total_time_ms == 2.0
        assert result.per_gpu_times_ms[0] == 1.0
        assert result.per_gpu_times_ms[1] == 2.0

    def test_comm_node_with_no_parent_flow_ids(self):
        dag = ExecutionDAG(
            comm_nodes=[_comm(0, 0, parent_flow_ids=[])],
        )
        merged, _ = merge_dags([("a", dag)])
        assert merged.comm_nodes[0].parent_flow_ids == []
