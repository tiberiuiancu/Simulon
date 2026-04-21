"""Tests for goal_trace.dag_to_goal() and write_goal_trace()."""

from __future__ import annotations

import pytest

from simulon.backend.dag.goal_trace import dag_to_goal, write_goal_trace
from simulon.backend.dag.nodes import CommNode, ComputeNode, DAGEdge, ExecutionDAG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute(node_id: int, gpu_rank: int, duration_ms: float | None = 1.0) -> ComputeNode:
    """Minimal populated ComputeNode."""
    return ComputeNode(
        node_id=node_id,
        gpu_rank=gpu_rank,
        kernel="mlp_linear1",
        layer_id=0,
        microbatch_id=0,
        pipeline_stage=0,
        phase="fwd",
        duration_ms=duration_ms,
    )


def _comm(
    node_id: int,
    src_gpu: int,
    dst_gpu: int,
    bytes_: int = 1024,
    duration_ms: float | None = 0.5,
) -> CommNode:
    """Minimal CommNode; duration_ms set to verify it does NOT appear in output."""
    return CommNode(
        node_id=node_id,
        src_gpu=src_gpu,
        dst_gpu=dst_gpu,
        bytes=bytes_,
        collective_type="AllGather",
        layer_id=0,
        phase="fwd",
        flow_id=node_id,
        duration_ms=duration_ms,
    )


# ---------------------------------------------------------------------------
# Scenario 1: single-rank, single compute node
# ---------------------------------------------------------------------------

class TestSingleRankSingleCompute:
    def test_num_ranks_is_one(self):
        """num_ranks header must reflect a single GPU rank."""
        dag = ExecutionDAG(compute_nodes=[_compute(0, gpu_rank=0, duration_ms=2.0)])
        out = dag_to_goal(dag)
        assert "num_ranks 1" in out

    def test_calc_line_present(self):
        """calc line for duration_ms=2.0 must emit 2000000 ns."""
        dag = ExecutionDAG(compute_nodes=[_compute(0, gpu_rank=0, duration_ms=2.0)])
        out = dag_to_goal(dag)
        assert "c0: calc 2000000" in out

    def test_rank_block_structure(self):
        """rank 0 block must open and close."""
        dag = ExecutionDAG(compute_nodes=[_compute(0, gpu_rank=0, duration_ms=2.0)])
        out = dag_to_goal(dag)
        assert "rank 0 {" in out
        assert "}" in out


# ---------------------------------------------------------------------------
# Scenario 2: same-rank compute → compute dependency
# ---------------------------------------------------------------------------

class TestComputeToComputeDependency:
    def test_requires_line_present(self):
        """c1 requires c0 must appear when ComputeNode 0 → ComputeNode 1 on same rank."""
        n0 = _compute(0, gpu_rank=0, duration_ms=1.0)
        n1 = _compute(1, gpu_rank=0, duration_ms=1.0)
        dag = ExecutionDAG(
            compute_nodes=[n0, n1],
            edges=[DAGEdge(src_node_id=0, dst_node_id=1)],
        )
        out = dag_to_goal(dag)
        assert "c1 requires c0" in out

    def test_no_spurious_requires(self):
        """Only the declared dependency should appear — no extra requires lines."""
        n0 = _compute(0, gpu_rank=0, duration_ms=1.0)
        n1 = _compute(1, gpu_rank=0, duration_ms=1.0)
        dag = ExecutionDAG(
            compute_nodes=[n0, n1],
            edges=[DAGEdge(src_node_id=0, dst_node_id=1)],
        )
        out = dag_to_goal(dag)
        assert out.count("requires") == 1


# ---------------------------------------------------------------------------
# Scenario 3: two-rank P2P (send/recv pair)
# ---------------------------------------------------------------------------

class TestTwoRankP2P:
    def setup_method(self):
        comm = _comm(node_id=5, src_gpu=0, dst_gpu=1, bytes_=4096, duration_ms=0.25)
        self.dag = ExecutionDAG(comm_nodes=[comm])
        self.out = dag_to_goal(self.dag)

    def test_num_ranks_two(self):
        """num_ranks must be 2 when comm spans GPUs 0 and 1."""
        assert "num_ranks 2" in self.out

    def test_send_in_rank0_block(self):
        """send line with correct bytes and destination must appear in rank 0's block."""
        assert "s5: send 4096b to 1 tag 5" in self.out

    def test_recv_in_rank1_block(self):
        """recv line with correct bytes and source must appear in rank 1's block."""
        assert "r5: recv 4096b from 0 tag 5" in self.out

    def test_tags_match(self):
        """send and recv must share the same tag (= node_id)."""
        assert "tag 5" in self.out
        # Both send and recv carry tag 5
        assert self.out.count("tag 5") == 2

    def test_duration_ms_not_in_output(self):
        """CommNode.duration_ms must NOT appear anywhere in GOAL output."""
        assert "0.25" not in self.out
        assert "duration" not in self.out


# ---------------------------------------------------------------------------
# Scenario 4: AllGather pattern — comm → compute dependency (recv side)
# ---------------------------------------------------------------------------

class TestCommToComputeDependency:
    def test_compute_requires_recv(self):
        """c{compute_id} requires r{comm_id} must appear in dst_gpu's rank block."""
        comm = _comm(node_id=10, src_gpu=0, dst_gpu=1, bytes_=512)
        calc = _compute(node_id=11, gpu_rank=1, duration_ms=1.0)
        dag = ExecutionDAG(
            compute_nodes=[calc],
            comm_nodes=[comm],
            edges=[DAGEdge(src_node_id=10, dst_node_id=11)],
        )
        out = dag_to_goal(dag)
        # comm recv completes at dst_gpu=1; calc is also on gpu_rank=1 → intra-rank dep
        assert "c11 requires r10" in out


# ---------------------------------------------------------------------------
# Scenario 5: ReduceScatter pattern — compute → comm dependency (send side)
# ---------------------------------------------------------------------------

class TestComputeToCommDependency:
    def test_send_requires_compute(self):
        """s{comm_id} requires c{compute_id} must appear in src_gpu's rank block."""
        calc = _compute(node_id=20, gpu_rank=0, duration_ms=1.0)
        comm = _comm(node_id=21, src_gpu=0, dst_gpu=1, bytes_=512)
        dag = ExecutionDAG(
            compute_nodes=[calc],
            comm_nodes=[comm],
            edges=[DAGEdge(src_node_id=20, dst_node_id=21)],
        )
        out = dag_to_goal(dag)
        # calc on gpu_rank=0; comm send side at src_gpu=0 → intra-rank dep
        assert "s21 requires c20" in out


# ---------------------------------------------------------------------------
# Scenario 6: ring step — comm → comm dependency
# ---------------------------------------------------------------------------

class TestCommToCommDependency:
    def test_send_requires_recv_in_shared_rank(self):
        """s{Y.node_id} requires r{X.node_id} in rank 1 when X.dst_gpu == Y.src_gpu == 1."""
        comm_x = _comm(node_id=30, src_gpu=0, dst_gpu=1, bytes_=256)
        comm_y = _comm(node_id=31, src_gpu=1, dst_gpu=2, bytes_=256)
        dag = ExecutionDAG(
            comm_nodes=[comm_x, comm_y],
            edges=[DAGEdge(src_node_id=30, dst_node_id=31)],
        )
        out = dag_to_goal(dag)
        # src (X) → r30 at dst_gpu=1; dst (Y) → s31 at src_gpu=1 → intra-rank dep
        assert "s31 requires r30" in out

    def test_num_ranks_covers_all_gpus(self):
        """num_ranks must be max_rank+1 = 3 for ranks {0,1,2}."""
        comm_x = _comm(node_id=30, src_gpu=0, dst_gpu=1, bytes_=256)
        comm_y = _comm(node_id=31, src_gpu=1, dst_gpu=2, bytes_=256)
        dag = ExecutionDAG(comm_nodes=[comm_x, comm_y])
        out = dag_to_goal(dag)
        assert "num_ranks 3" in out


# ---------------------------------------------------------------------------
# Scenario 7: cross-rank edge is silently skipped (PP_Send fan-out)
# ---------------------------------------------------------------------------

class TestCrossRankEdgeSkipped:
    def test_no_requires_for_cross_rank_edge(self):
        """An edge where src_rank != dst_rank in GOAL terms must produce no requires line."""
        # PP_Send: src=0, dst=2; ComputeNode on gpu_rank=1 (not dst=2)
        pp_send = _comm(node_id=40, src_gpu=0, dst_gpu=2, bytes_=128)
        calc = _compute(node_id=41, gpu_rank=1, duration_ms=1.0)
        dag = ExecutionDAG(
            compute_nodes=[calc],
            comm_nodes=[pp_send],
            edges=[DAGEdge(src_node_id=40, dst_node_id=41)],
        )
        out = dag_to_goal(dag)
        # comm recv at dst_gpu=2; calc at gpu_rank=1 → cross-rank → skipped
        assert "requires" not in out

    def test_no_error_on_cross_rank_edge(self):
        """Cross-rank edges must not raise any exception."""
        pp_send = _comm(node_id=40, src_gpu=0, dst_gpu=2, bytes_=128)
        calc = _compute(node_id=41, gpu_rank=1, duration_ms=1.0)
        dag = ExecutionDAG(
            compute_nodes=[calc],
            comm_nodes=[pp_send],
            edges=[DAGEdge(src_node_id=40, dst_node_id=41)],
        )
        dag_to_goal(dag)  # must not raise


# ---------------------------------------------------------------------------
# Error conditions
# ---------------------------------------------------------------------------

class TestErrorConditions:
    def test_unpopulated_compute_raises(self):
        """ValueError must be raised if any ComputeNode.duration_ms is None."""
        dag = ExecutionDAG(compute_nodes=[_compute(0, gpu_rank=0, duration_ms=None)])
        with pytest.raises(ValueError, match="duration_ms"):
            dag_to_goal(dag)

    def test_unknown_src_node_id_raises(self):
        """ValueError must be raised when a DAGEdge references a non-existent src_node_id."""
        calc = _compute(0, gpu_rank=0, duration_ms=1.0)
        dag = ExecutionDAG(
            compute_nodes=[calc],
            edges=[DAGEdge(src_node_id=999, dst_node_id=0)],
        )
        with pytest.raises(ValueError, match="src_node_id"):
            dag_to_goal(dag)

    def test_unknown_dst_node_id_raises(self):
        """ValueError must be raised when a DAGEdge references a non-existent dst_node_id."""
        calc = _compute(0, gpu_rank=0, duration_ms=1.0)
        dag = ExecutionDAG(
            compute_nodes=[calc],
            edges=[DAGEdge(src_node_id=0, dst_node_id=999)],
        )
        with pytest.raises(ValueError, match="dst_node_id"):
            dag_to_goal(dag)

    def test_non_contiguous_ranks_raises(self):
        """ValueError must be raised when GPU ranks have a gap (e.g. {0, 2} without 1)."""
        dag = ExecutionDAG(
            compute_nodes=[
                _compute(0, gpu_rank=0, duration_ms=1.0),
                _compute(1, gpu_rank=2, duration_ms=1.0),
            ]
        )
        with pytest.raises(ValueError, match="contiguous"):
            dag_to_goal(dag)

    def test_empty_dag_raises(self):
        """ValueError must be raised for a completely empty ExecutionDAG."""
        with pytest.raises(ValueError, match="empty"):
            dag_to_goal(ExecutionDAG())


# ---------------------------------------------------------------------------
# Scenario 12: ms → ns conversion precision
# ---------------------------------------------------------------------------

class TestMsToNsConversion:
    def test_half_ms_produces_500000_ns(self):
        """duration_ms=0.5 must produce calc 500000 (0.5 ms = 500 000 ns)."""
        dag = ExecutionDAG(compute_nodes=[_compute(0, gpu_rank=0, duration_ms=0.5)])
        out = dag_to_goal(dag)
        assert "c0: calc 500000" in out

    def test_fractional_ms_truncates(self):
        """int() truncation: 1.9999 ms → 1999900 ns."""
        dag = ExecutionDAG(compute_nodes=[_compute(0, gpu_rank=0, duration_ms=1.9999)])
        out = dag_to_goal(dag)
        assert "c0: calc 1999900" in out


# ---------------------------------------------------------------------------
# write_goal_trace integration
# ---------------------------------------------------------------------------

class TestWriteGoalTrace:
    def test_writes_file_with_correct_content(self, tmp_path):
        """write_goal_trace must produce a file whose content matches dag_to_goal."""
        dag = ExecutionDAG(compute_nodes=[_compute(0, gpu_rank=0, duration_ms=1.0)])
        out_path = tmp_path / "test.goal"
        write_goal_trace(dag, out_path)
        assert out_path.exists()
        assert out_path.read_text() == dag_to_goal(dag)

    def test_write_propagates_value_error(self, tmp_path):
        """write_goal_trace must propagate ValueError from dag_to_goal."""
        dag = ExecutionDAG(compute_nodes=[_compute(0, gpu_rank=0, duration_ms=None)])
        with pytest.raises(ValueError):
            write_goal_trace(dag, tmp_path / "bad.goal")
