"""Unit tests for the DAG replayer and supporting utilities."""

import pytest

from simulon.backend.dag.nodes import CommNode, ComputeNode, DAGEdge, ExecutionDAG
from simulon.backend.dag.populate import (
    _get_link_params,
    _parse_latency,
    _parse_speed,
    populate_network,
)
from simulon.backend.dag.replayer import (
    SimulationResult,
    _intersection_duration,
    _merge_intervals,
    replay,
)
from simulon.config.dc import (
    DatacenterConfig,
    DatacenterMeta,
    ClusterSpec,
    NetworkSpec,
    NICSpec,
    NodeSpec,
    ScaleOutSpec,
    ScaleUpSpec,
    SwitchSpec,
    GPUSpec,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dc(
    gpus_per_node: int = 4,
    nvswitch_speed: str = "100Gbps",
    nvswitch_latency: str = "0.001ms",
    nic_speed: str = "100Gbps",
    nic_latency: str = "0.01ms",
    nic_efficiency: float = 1.0,
) -> DatacenterConfig:
    return DatacenterConfig(
        datacenter=DatacenterMeta(name="test"),
        cluster=ClusterSpec(num_nodes=4),
        node=NodeSpec(
            gpus_per_node=gpus_per_node,
            gpu=GPUSpec(name="test-gpu"),
        ),
        network=NetworkSpec(
            scale_up=ScaleUpSpec(
                switch=SwitchSpec(port_speed=nvswitch_speed, latency=nvswitch_latency)
            ),
            scale_out=ScaleOutSpec(
                nic=NICSpec(speed=nic_speed, latency=nic_latency, bandwidth_efficiency=nic_efficiency)
            ),
        ),
    )


def _dag(*nodes, edges=(), comm_nodes=()):
    dag = ExecutionDAG()
    for n in nodes:
        if isinstance(n, ComputeNode):
            dag.compute_nodes.append(n)
        else:
            dag.comm_nodes.append(n)
    dag.comm_nodes.extend(comm_nodes)
    dag.edges.extend(edges)
    return dag


def _compute(node_id, gpu_rank=0, duration_ms=1.0):
    return ComputeNode(
        node_id=node_id,
        gpu_rank=gpu_rank,
        kernel="layernorm",
        layer_id=0,
        microbatch_id=0,
        pipeline_stage=0,
        phase="fwd",
        duration_ms=duration_ms,
    )


def _comm(node_id, src_gpu, dst_gpu, bytes=1000, flow_id=None, parent_flow_ids=None):
    return CommNode(
        node_id=node_id,
        src_gpu=src_gpu,
        dst_gpu=dst_gpu,
        bytes=bytes,
        collective_type="PP_Send",
        layer_id=0,
        phase="fwd",
        flow_id=node_id if flow_id is None else flow_id,
        parent_flow_ids=parent_flow_ids or [],
    )


# ---------------------------------------------------------------------------
# _parse_speed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s, expected_bytes_per_ms", [
    ("1Gbps",    1e9 / 8 / 1000),
    ("400Gbps",  400e9 / 8 / 1000),
    ("1GBps",    1e9 / 1000),
    ("100Mbps",  100e6 / 8 / 1000),
    ("100MBps",  100e6 / 1000),
    ("2880Gbps", 2880e9 / 8 / 1000),
])
def test_parse_speed(s, expected_bytes_per_ms):
    assert abs(_parse_speed(s) - expected_bytes_per_ms) < 1e-3


def test_parse_speed_invalid():
    with pytest.raises(ValueError):
        _parse_speed("fast")


# ---------------------------------------------------------------------------
# _parse_latency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s, expected_ms", [
    ("1ms",    1.0),
    ("0.5ms",  0.5),
    ("1us",    0.001),
    ("500us",  0.5),
    ("1ns",    1e-6),
    ("100ns",  1e-4),
    ("2.5e-5ms", 2.5e-5),
])
def test_parse_latency(s, expected_ms):
    assert abs(_parse_latency(s) - expected_ms) < 1e-12


def test_parse_latency_invalid():
    with pytest.raises(ValueError):
        _parse_latency("fast")


# ---------------------------------------------------------------------------
# _get_link_params: intra vs inter node
# ---------------------------------------------------------------------------


def test_get_link_params_intra_node():
    dc = _dc(gpus_per_node=4, nvswitch_speed="100Gbps", nvswitch_latency="0.001ms")
    bw, lat = _get_link_params(0, 3, dc)
    assert abs(bw - _parse_speed("100Gbps")) < 1
    assert abs(lat - 0.001) < 1e-9


def test_get_link_params_inter_node():
    dc = _dc(gpus_per_node=4, nic_speed="100Gbps", nic_latency="0.01ms", nic_efficiency=1.0)
    bw, lat = _get_link_params(0, 4, dc)  # GPU 0 (node 0) → GPU 4 (node 1)
    assert abs(bw - _parse_speed("100Gbps")) < 1
    assert abs(lat - 0.01) < 1e-9


def test_get_link_params_nic_efficiency():
    dc = _dc(gpus_per_node=4, nic_speed="100Gbps", nic_efficiency=0.5)
    bw, _ = _get_link_params(0, 4, dc)
    assert abs(bw - _parse_speed("100Gbps") * 0.5) < 1


def test_get_link_params_boundary_gpus():
    """GPU 3 and GPU 4 are on different nodes when gpus_per_node=4."""
    dc = _dc(gpus_per_node=4, nvswitch_speed="400Gbps", nic_speed="100Gbps", nic_efficiency=1.0)
    bw_intra, _ = _get_link_params(0, 3, dc)   # same node
    bw_inter, _ = _get_link_params(3, 4, dc)   # different nodes
    assert abs(bw_intra - _parse_speed("400Gbps")) < 1
    assert abs(bw_inter - _parse_speed("100Gbps")) < 1


# ---------------------------------------------------------------------------
# Single CommNode
# ---------------------------------------------------------------------------


def test_single_comm_node_duration():
    """Duration = latency + bytes / bandwidth; both src and dst get that finish time."""
    dc = _dc(gpus_per_node=4, nvswitch_speed="1GBps", nvswitch_latency="0ms")
    bytes_ = 1_000_000  # 1 MB
    bw = _parse_speed("1GBps")   # 1e6 bytes/ms
    expected_duration = bytes_ / bw  # 1 ms

    dag = _dag(_comm(0, src_gpu=0, dst_gpu=1, bytes=bytes_))
    populate_network(dag, dc)
    result = replay(dag)

    assert abs(result.total_time_ms - expected_duration) < 1e-9
    assert abs(result.per_gpu_times_ms[0] - expected_duration) < 1e-9  # src
    assert abs(result.per_gpu_times_ms[1] - expected_duration) < 1e-9  # dst


def test_single_comm_node_includes_latency():
    """Latency is added on top of the transfer time."""
    dc = _dc(gpus_per_node=4, nvswitch_speed="1GBps", nvswitch_latency="0.5ms")
    bw = _parse_speed("1GBps")
    bytes_ = 1_000_000
    expected = 0.5 + bytes_ / bw

    dag = _dag(_comm(0, src_gpu=0, dst_gpu=1, bytes=bytes_))
    populate_network(dag, dc)
    result = replay(dag)

    assert abs(result.total_time_ms - expected) < 1e-9


# ---------------------------------------------------------------------------
# Independent flows run in parallel (no port contention modeled)
# ---------------------------------------------------------------------------


def test_independent_flows_run_in_parallel():
    """Flows with different src and dst GPUs run in parallel."""
    dc = _dc(gpus_per_node=4, nvswitch_speed="1GBps", nvswitch_latency="0ms")
    bytes_ = 1_000_000  # 1 ms each

    dag = _dag(
        _comm(0, src_gpu=0, dst_gpu=1, bytes=bytes_),
        _comm(1, src_gpu=2, dst_gpu=3, bytes=bytes_),
    )
    populate_network(dag, dc)
    result = replay(dag)

    # Parallel: 1 ms total
    assert abs(result.total_time_ms - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Compute node
# ---------------------------------------------------------------------------


def test_single_compute_node():
    dc = _dc()
    dag = _dag(_compute(0, gpu_rank=0, duration_ms=5.0))
    populate_network(dag, dc)
    result = replay(dag)

    assert abs(result.total_time_ms - 5.0) < 1e-9
    assert abs(result.per_gpu_times_ms[0] - 5.0) < 1e-9
    assert abs(result.compute_ms - 5.0) < 1e-9


def test_compute_node_none_duration():
    """ComputeNode with duration_ms=None contributes 0 ms."""
    dc = _dc()
    n = _compute(0, gpu_rank=0, duration_ms=None)
    dag = _dag(n)
    populate_network(dag, dc)
    result = replay(dag)
    assert result.total_time_ms == 0.0


# ---------------------------------------------------------------------------
# Dependency chain: compute → comm
# ---------------------------------------------------------------------------


def test_compute_before_comm():
    """CommNode must wait for its ComputeNode predecessor to finish."""
    dc = _dc(gpus_per_node=4, nvswitch_speed="1GBps", nvswitch_latency="0ms")
    bytes_ = 1_000_000  # 1 ms comm

    c = _compute(0, gpu_rank=0, duration_ms=3.0)
    m = _comm(1, src_gpu=0, dst_gpu=1, bytes=bytes_)
    dag = _dag(c, m, edges=[DAGEdge(src_node_id=0, dst_node_id=1)])
    populate_network(dag, dc)
    result = replay(dag)

    # Comm starts at t=3, finishes at t=4
    assert abs(result.total_time_ms - 4.0) < 1e-9
    assert abs(result.per_gpu_times_ms[0] - 4.0) < 1e-9  # src of comm
    assert abs(result.per_gpu_times_ms[1] - 4.0) < 1e-9  # dst of comm


# ---------------------------------------------------------------------------
# parent_flow_ids ordering
# ---------------------------------------------------------------------------


def test_parent_flow_ids_ordering():
    """A CommNode with parent_flow_ids must wait for the parent flow to finish."""
    dc = _dc(gpus_per_node=4, nvswitch_speed="1GBps", nvswitch_latency="0ms")
    bytes_ = 1_000_000  # 1 ms each

    parent = CommNode(
        node_id=0, src_gpu=0, dst_gpu=1, bytes=bytes_,
        collective_type="AllGather", layer_id=0, phase="fwd",
        flow_id=10, parent_flow_ids=[],
    )
    child = CommNode(
        node_id=1, src_gpu=1, dst_gpu=2, bytes=bytes_,
        collective_type="AllGather", layer_id=0, phase="fwd",
        flow_id=11, parent_flow_ids=[10],  # depends on flow_id=10
    )
    dag = ExecutionDAG(comm_nodes=[parent, child])
    populate_network(dag, dc)
    result = replay(dag)

    # Parent finishes at 1ms, child starts at 1ms, finishes at 2ms
    assert abs(result.total_time_ms - 2.0) < 1e-9


# ---------------------------------------------------------------------------
# comm_time_ms accounts for both src and dst
# ---------------------------------------------------------------------------


def test_comm_time_counted_for_both_endpoints():
    """Both src and dst GPU are accounted for in the summary.

    The dst GPU (receiver) waits for the recv → exposed_comm.
    The src GPU (sender) has no compute or recv → bubble.
    Averaged across 2 GPUs: exposed_comm = 0.5 ms, bubble = 0.5 ms.
    """
    dc = _dc(gpus_per_node=4, nvswitch_speed="1GBps", nvswitch_latency="0ms")
    bytes_ = 1_000_000  # 1 ms

    dag = _dag(_comm(0, src_gpu=0, dst_gpu=1, bytes=bytes_))
    populate_network(dag, dc)
    result = replay(dag)

    assert abs(result.exposed_comm_ms - 0.5) < 1e-9   # avg: 0 (src) + 1.0 (dst) / 2
    assert abs(result.bubble_ms - 0.5) < 1e-9          # avg: 1.0 (src) + 0 (dst) / 2


# ---------------------------------------------------------------------------
# per_gpu_times_ms covers all participating GPUs
# ---------------------------------------------------------------------------


def test_per_gpu_finish_covers_src_gpu():
    """A GPU that only sends (never receives) must appear in per_gpu_times_ms."""
    dc = _dc(gpus_per_node=4, nvswitch_speed="1GBps", nvswitch_latency="0ms")
    bytes_ = 1_000_000

    dag = _dag(_comm(0, src_gpu=0, dst_gpu=1, bytes=bytes_))
    populate_network(dag, dc)
    result = replay(dag)

    assert 0 in result.per_gpu_times_ms
    assert 1 in result.per_gpu_times_ms


# ---------------------------------------------------------------------------
# Inter-node classification
# ---------------------------------------------------------------------------


def test_inter_node_uses_nic_bandwidth():
    """Flows between GPUs on different nodes use NIC bandwidth, not NVSwitch."""
    dc = _dc(
        gpus_per_node=2,
        nvswitch_speed="100GBps",  # very fast intra-node
        nic_speed="1GBps",         # slow inter-node
        nic_latency="0ms",
        nic_efficiency=1.0,
    )
    bytes_ = 1_000_000  # 1 ms at 1 GBps

    # GPU 0 (node 0) → GPU 2 (node 1)
    dag = _dag(_comm(0, src_gpu=0, dst_gpu=2, bytes=bytes_))
    populate_network(dag, dc)
    result = replay(dag)

    assert abs(result.total_time_ms - 1.0) < 1e-9


def test_intra_node_uses_nvswitch_bandwidth():
    """Flows between GPUs on the same node use NVSwitch bandwidth."""
    dc = _dc(
        gpus_per_node=4,
        nvswitch_speed="1GBps",
        nvswitch_latency="0ms",
        nic_speed="100MBps",  # slow inter-node (irrelevant here)
        nic_efficiency=1.0,
    )
    bytes_ = 1_000_000  # 1 ms at 1 GBps

    # GPU 0 → GPU 3, same node
    dag = _dag(_comm(0, src_gpu=0, dst_gpu=3, bytes=bytes_))
    populate_network(dag, dc)
    result = replay(dag)

    assert abs(result.total_time_ms - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# _merge_intervals
# ---------------------------------------------------------------------------


class TestMergeIntervals:
    def test_empty(self):
        """Empty input returns empty list."""
        assert _merge_intervals([]) == []

    def test_single(self):
        """A single interval is returned unchanged."""
        assert _merge_intervals([(1.0, 3.0)]) == [(1.0, 3.0)]

    def test_non_overlapping(self):
        """Non-overlapping intervals are returned in sorted order."""
        assert _merge_intervals([(0.0, 1.0), (2.0, 3.0)]) == [(0.0, 1.0), (2.0, 3.0)]

    def test_non_overlapping_unsorted(self):
        """Out-of-order non-overlapping intervals are sorted first."""
        assert _merge_intervals([(2.0, 3.0), (0.0, 1.0)]) == [(0.0, 1.0), (2.0, 3.0)]

    def test_overlapping(self):
        """Overlapping intervals are merged into one span."""
        assert _merge_intervals([(0.0, 2.0), (1.0, 3.0)]) == [(0.0, 3.0)]

    def test_adjacent(self):
        """Adjacent intervals (end == start) are merged because s <= end."""
        assert _merge_intervals([(0.0, 1.0), (1.0, 2.0)]) == [(0.0, 2.0)]

    def test_contained(self):
        """An interval fully inside another is absorbed into the outer span."""
        assert _merge_intervals([(0.0, 5.0), (1.0, 3.0)]) == [(0.0, 5.0)]

    def test_multiple_overlapping_collapse_to_one(self):
        """Three mutually-overlapping intervals collapse to a single interval."""
        result = _merge_intervals([(0.0, 2.0), (1.0, 3.0), (2.5, 4.0)])
        assert result == [(0.0, 4.0)]

    def test_two_disjoint_groups(self):
        """Two disjoint clusters each merge independently."""
        result = _merge_intervals([(0.0, 2.0), (1.0, 3.0), (5.0, 7.0), (6.0, 8.0)])
        assert result == [(0.0, 3.0), (5.0, 8.0)]


# ---------------------------------------------------------------------------
# _intersection_duration
# ---------------------------------------------------------------------------


class TestIntersectionDuration:
    def test_both_empty(self):
        """No intervals on either side → zero intersection."""
        assert _intersection_duration([], []) == pytest.approx(0.0)

    def test_first_empty(self):
        """First list empty → zero intersection."""
        assert _intersection_duration([], [(0.0, 1.0)]) == pytest.approx(0.0)

    def test_second_empty(self):
        """Second list empty → zero intersection."""
        assert _intersection_duration([(0.0, 1.0)], []) == pytest.approx(0.0)

    def test_no_overlap(self):
        """Disjoint intervals produce zero intersection."""
        assert _intersection_duration([(0.0, 1.0)], [(2.0, 3.0)]) == pytest.approx(0.0)

    def test_partial_overlap(self):
        """Partially overlapping intervals: intersection equals the overlap length."""
        assert _intersection_duration([(0.0, 2.0)], [(1.0, 3.0)]) == pytest.approx(1.0)

    def test_full_containment(self):
        """Smaller interval fully inside larger: intersection equals the smaller length."""
        assert _intersection_duration([(0.0, 4.0)], [(1.0, 3.0)]) == pytest.approx(2.0)

    def test_identical_intervals(self):
        """Identical intervals: intersection equals the full interval length."""
        assert _intersection_duration([(0.0, 5.0)], [(0.0, 5.0)]) == pytest.approx(5.0)

    def test_multiple_intervals_each_side(self):
        """Multiple intervals per side: sum of all pairwise overlaps is correct.

        a=[(0,2),(4,6)], b=[(1,5)] → overlaps at (1,2)=1 and (4,5)=1 → total=2.
        """
        a = [(0.0, 2.0), (4.0, 6.0)]
        b = [(1.0, 5.0)]
        assert _intersection_duration(a, b) == pytest.approx(2.0)

    def test_touching_boundary_zero(self):
        """Adjacent intervals that touch but do not overlap contribute nothing (hi == lo)."""
        assert _intersection_duration([(0.0, 1.0)], [(1.0, 2.0)]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Helper: CommNode with pre-set duration_ms (bypasses populate_network)
# ---------------------------------------------------------------------------


def _comm_preset(node_id, src_gpu, dst_gpu, duration_ms, collective_type="AllReduce"):
    """Build a CommNode with duration already set, skipping populate_network."""
    n = CommNode(
        node_id=node_id,
        src_gpu=src_gpu,
        dst_gpu=dst_gpu,
        bytes=0,
        collective_type=collective_type,
        layer_id=0,
        phase="fwd",
        flow_id=node_id,
        parent_flow_ids=[],
    )
    n.duration_ms = duration_ms
    return n


# ---------------------------------------------------------------------------
# _summarize / SimulationResult metrics via replay()
# ---------------------------------------------------------------------------


class TestSummarize:

    # ------------------------------------------------------------------
    # Pure compute
    # ------------------------------------------------------------------

    def test_pure_compute_no_comm_no_bubble(self):
        """Pure compute on one GPU: compute_ms correct, exposed_comm and bubble both 0."""
        dag = _dag(_compute(0, gpu_rank=0, duration_ms=5.0))
        result = replay(dag)

        assert result.compute_ms == pytest.approx(5.0)
        assert result.exposed_comm_ms == pytest.approx(0.0)
        assert result.bubble_ms == pytest.approx(0.0)
        assert result.overlapped_comm_ms == pytest.approx(0.0)
        assert result.exposed_comm_by_type == {}

    def test_two_gpus_different_compute_durations_avg(self):
        """Two GPUs with different compute durations: averages are correct.

        GPU0: 4ms, GPU1: 2ms.  total=4.  avg_compute=3, avg_bubble=1.
        """
        c0 = _compute(0, gpu_rank=0, duration_ms=4.0)
        c1 = _compute(1, gpu_rank=1, duration_ms=2.0)
        dag = _dag(c0, c1)
        result = replay(dag)

        assert result.total_time_ms == pytest.approx(4.0)
        assert result.compute_ms == pytest.approx(3.0)   # avg(4, 2)
        assert result.bubble_ms == pytest.approx(1.0)    # avg(0, 2)
        assert result.exposed_comm_ms == pytest.approx(0.0)

    def test_three_gpus_compute_avg(self):
        """Three GPUs compute independently; averages span all three."""
        c0 = _compute(0, gpu_rank=0, duration_ms=4.0)
        c1 = _compute(1, gpu_rank=1, duration_ms=2.0)
        c2 = _compute(2, gpu_rank=2, duration_ms=6.0)
        dag = _dag(c0, c1, c2)
        result = replay(dag)

        assert result.total_time_ms == pytest.approx(6.0)
        assert result.compute_ms == pytest.approx(4.0)   # avg(4, 2, 6)
        assert result.bubble_ms == pytest.approx(2.0)    # avg(2, 4, 0)

    # ------------------------------------------------------------------
    # Pure recv (no compute on dst GPU)
    # ------------------------------------------------------------------

    def test_pure_recv_fully_exposed(self):
        """Recv with no overlapping compute is fully exposed on the dst GPU.

        Src GPU (sender) contributes no exposed_comm.
        Averaged over 2 GPUs: exposed_comm = recv_duration / 2.
        """
        dc = _dc(gpus_per_node=4, nvswitch_speed="1GBps", nvswitch_latency="0ms")
        bytes_ = 2_000_000  # 2 ms at 1 GBps

        dag = _dag(_comm(0, src_gpu=0, dst_gpu=1, bytes=bytes_))
        populate_network(dag, dc)
        result = replay(dag)

        # dst GPU: exposed=2ms.  src GPU: exposed=0.  avg = 1ms.
        assert result.exposed_comm_ms == pytest.approx(1.0)
        assert result.compute_ms == pytest.approx(0.0)
        # src GPU has 2ms where it is neither computing nor receiving → bubble
        assert result.bubble_ms == pytest.approx(1.0)   # avg(2, 0) / 2

    # ------------------------------------------------------------------
    # Sequential compute then recv (no overlap)
    # ------------------------------------------------------------------

    def test_sequential_compute_then_recv_on_dst(self):
        """Compute (3ms) then recv (1ms) on dst GPU: both fully accounted.

        GPU1 (dst): compute=3, exposed_comm=1, bubble=0.
        GPU0 (src): send starts at t=3, bubble=3+0=3 (no recv, compute=0).
        Actually GPU0 finishes at t=4 (blocking during send);
        bubble for GPU0 = total(4) - 0 - 0 = 4.
        Averages: compute=1.5, exposed_comm=0.5, bubble=2.
        """
        dc = _dc(gpus_per_node=4, nvswitch_speed="1GBps", nvswitch_latency="0ms")
        bytes_ = 1_000_000  # 1 ms

        c = _compute(0, gpu_rank=1, duration_ms=3.0)
        m = _comm(1, src_gpu=0, dst_gpu=1, bytes=bytes_)
        dag = _dag(c, m, edges=[DAGEdge(src_node_id=0, dst_node_id=1)])
        populate_network(dag, dc)
        result = replay(dag)

        assert result.total_time_ms == pytest.approx(4.0)
        assert result.compute_ms == pytest.approx(1.5)        # avg(3, 0)
        assert result.exposed_comm_ms == pytest.approx(0.5)   # avg(1, 0)
        assert result.bubble_ms == pytest.approx(2.0)         # avg(0, 4)

    # ------------------------------------------------------------------
    # Compute overlapping recv (comm hidden)
    # ------------------------------------------------------------------

    def test_recv_fully_inside_compute_window_zero_exposed(self):
        """Recv fully inside compute window: exposed_comm == 0, overlapped_comm > 0.

        Setup: GPU0 runs compute 0-5ms.  GPU1 sends, GPU0 receives 1-3ms.
        Overlap = 2ms.  Exposed = 0.
        """
        c = _compute(0, gpu_rank=0, duration_ms=5.0)
        m = _comm_preset(1, src_gpu=1, dst_gpu=0, duration_ms=2.0)
        # delay comm start to t=1 via a 1ms pre-compute on GPU0
        c_pre = _compute(2, gpu_rank=0, duration_ms=1.0)
        dag = _dag(c_pre, c, m, edges=[
            DAGEdge(src_node_id=2, dst_node_id=0),  # c starts after c_pre
            DAGEdge(src_node_id=2, dst_node_id=1),  # comm starts after c_pre
        ])
        result = replay(dag)

        # c_pre [0,1], c [1,6], comm [1,3]
        assert result.total_time_ms == pytest.approx(6.0)
        assert result.exposed_comm_ms == pytest.approx(0.0)
        # GPU0: compute=[0,1]+[1,6]=[0,6]=6ms, comm=[1,3], overlap=2ms
        # GPU1: no compute, no recv, bubble=6ms
        assert result.overlapped_comm_ms == pytest.approx(2.0 / 2)  # avg over 2 GPUs

    def test_recv_partially_overlapping_compute(self):
        """Recv partially overlapping compute: exposed = recv duration - overlap.

        GPU0: compute 0-2ms, recv 1-4ms.  Overlap with [0,2] = 1ms.  Exposed = 2ms.
        But GPU0 has c_pre [0,1] + c [1,3] = merged [0,3].
        Recv [1,4] ∩ [0,3] = [1,3] = 2ms hidden.  Exposed = 3 - 2 = 1ms.
        GPU1: sender, no recv, bubble = 4ms.
        Averages: compute=1.5, exposed=0.5, bubble=2.
        """
        c_pre = _compute(2, gpu_rank=0, duration_ms=1.0)
        c = _compute(0, gpu_rank=0, duration_ms=2.0)
        m = _comm_preset(1, src_gpu=1, dst_gpu=0, duration_ms=3.0)
        dag = _dag(c_pre, c, m, edges=[
            DAGEdge(src_node_id=2, dst_node_id=0),
            DAGEdge(src_node_id=2, dst_node_id=1),
        ])
        result = replay(dag)

        # c_pre [0,1], c [1,3], comm [1,4]
        assert result.total_time_ms == pytest.approx(4.0)
        assert result.compute_ms == pytest.approx(1.5)        # avg(3, 0)
        assert result.exposed_comm_ms == pytest.approx(0.5)   # avg(1, 0)
        assert result.bubble_ms == pytest.approx(2.0)         # avg(0, 4)

    # ------------------------------------------------------------------
    # exposed_comm_by_type bucketing
    # ------------------------------------------------------------------

    def test_exposed_comm_by_type_two_distinct_types(self):
        """Two recv nodes on same GPU with different types bucket to separate keys.

        GPU0 receives AllReduce (2ms at t=0-2) then ReduceScatter (1ms at t=2-3).
        GPU1 and GPU2 are pure senders (no recv).
        Avg over 3 GPUs: AllReduce = 2/3 ms, ReduceScatter = 1/3 ms.
        """
        ar = _comm_preset(0, src_gpu=1, dst_gpu=0, duration_ms=2.0, collective_type="AllReduce")
        rs = _comm_preset(1, src_gpu=2, dst_gpu=0, duration_ms=1.0, collective_type="ReduceScatter")
        dag = _dag(ar, rs, edges=[DAGEdge(src_node_id=0, dst_node_id=1)])
        result = replay(dag)

        assert result.total_time_ms == pytest.approx(3.0)
        assert "AllReduce" in result.exposed_comm_by_type
        assert "ReduceScatter" in result.exposed_comm_by_type
        assert result.exposed_comm_by_type["AllReduce"] == pytest.approx(2.0 / 3)
        assert result.exposed_comm_by_type["ReduceScatter"] == pytest.approx(1.0 / 3)

    def test_exposed_comm_by_type_hidden_recv_zero(self):
        """Recv fully hidden by compute records 0 exposed for its collective_type."""
        c = _compute(0, gpu_rank=0, duration_ms=5.0)
        m = _comm_preset(1, src_gpu=1, dst_gpu=0, duration_ms=2.0, collective_type="AllGather")
        c_pre = _compute(2, gpu_rank=0, duration_ms=1.0)
        dag = _dag(c_pre, c, m, edges=[
            DAGEdge(src_node_id=2, dst_node_id=0),
            DAGEdge(src_node_id=2, dst_node_id=1),
        ])
        result = replay(dag)

        assert result.exposed_comm_by_type.get("AllGather", 0.0) == pytest.approx(0.0)

    # ------------------------------------------------------------------
    # bubble_ms: idle time invariants
    # ------------------------------------------------------------------

    def test_bubble_ms_equals_idle_time(self):
        """bubble_ms accounts for GPU time that is neither compute nor exposed_comm.

        GPU1 (dst): compute=3ms (0-3), then recv=2ms (3-5).  bubble=0.
        GPU0 (src): no compute, send 3-5ms.  bubble = total(5) - 0 - 0 = 5.
        Avg bubble = (0 + 5) / 2 = 2.5.
        """
        dc = _dc(gpus_per_node=4, nvswitch_speed="1GBps", nvswitch_latency="0ms")
        bytes_ = 2_000_000  # 2 ms

        c = _compute(0, gpu_rank=1, duration_ms=3.0)
        m = _comm(1, src_gpu=0, dst_gpu=1, bytes=bytes_)
        dag = _dag(c, m, edges=[DAGEdge(src_node_id=0, dst_node_id=1)])
        populate_network(dag, dc)
        result = replay(dag)

        assert result.total_time_ms == pytest.approx(5.0)
        assert result.bubble_ms == pytest.approx(2.5)
        assert result.compute_ms == pytest.approx(1.5)
        assert result.exposed_comm_ms == pytest.approx(1.0)

    def test_components_sum_to_total_time(self):
        """compute_ms + exposed_comm_ms + bubble_ms == total_time_ms for any DAG."""
        dc = _dc(gpus_per_node=4, nvswitch_speed="1GBps", nvswitch_latency="0ms")
        bytes_ = 1_000_000

        c = _compute(0, gpu_rank=0, duration_ms=2.0)
        m = _comm(1, src_gpu=0, dst_gpu=1, bytes=bytes_)
        dag = _dag(c, m, edges=[DAGEdge(src_node_id=0, dst_node_id=1)])
        populate_network(dag, dc)
        result = replay(dag)

        total_components = result.compute_ms + result.exposed_comm_ms + result.bubble_ms
        assert total_components == pytest.approx(result.total_time_ms)

    def test_components_sum_pure_compute(self):
        """Summation invariant holds for a pure-compute single-GPU DAG."""
        dag = _dag(_compute(0, gpu_rank=0, duration_ms=7.0))
        result = replay(dag)

        total_components = result.compute_ms + result.exposed_comm_ms + result.bubble_ms
        assert total_components == pytest.approx(result.total_time_ms)

    def test_components_sum_pure_recv(self):
        """Summation invariant holds for a pure-recv two-GPU DAG."""
        dc = _dc(gpus_per_node=4, nvswitch_speed="1GBps", nvswitch_latency="0ms")
        dag = _dag(_comm(0, src_gpu=0, dst_gpu=1, bytes=3_000_000))
        populate_network(dag, dc)
        result = replay(dag)

        total_components = result.compute_ms + result.exposed_comm_ms + result.bubble_ms
        assert total_components == pytest.approx(result.total_time_ms)

    # ------------------------------------------------------------------
    # TIB-113: two overlapping recvs on the same GPU (no compute)
    # ------------------------------------------------------------------

    def test_overlapping_recvs_no_double_count(self):
        """Two overlapping recv nodes on one GPU must not double-count exposed time.

        Regression for TIB-113: before the fix, exposed_total was computed by
        summing per-recv durations, so overlapping recvs inflated it beyond the
        actual wall-clock wait time, clamping bubble_ms to zero and causing
        component percentages to exceed 100%.

        Setup (no compute anywhere):
          GPU 0: dst of both recvs, both start at t=0 in parallel.
            - AllReduce (src=1 → dst=0): duration 4 ms  [0, 4]
            - AllGather  (src=2 → dst=0): duration 3 ms  [0, 3]
          GPU 1: pure sender, no recv. Finishes at t=4.
          GPU 2: pure sender, no recv. Finishes at t=4.

        total_time_ms = 4 ms.

        Per GPU 0: union of recv intervals = [0,4] → exposed_total = 4 ms.
          Per-type sum = AllReduce(4) + AllGather(3) = 7 ms > exposed_total.
        Per GPU 1: bubble = 4, exposed = 0.
        Per GPU 2: bubble = 4, exposed = 0.

        Averages (3 GPUs):
          compute_ms      = 0
          exposed_comm_ms = 4/3
          bubble_ms       = (0 + 4 + 4) / 3 = 8/3
          sum             = 4/3 + 8/3 = 12/3 = 4  ✓
        """
        ar = _comm_preset(0, src_gpu=1, dst_gpu=0, duration_ms=4.0, collective_type="AllReduce")
        ag = _comm_preset(1, src_gpu=2, dst_gpu=0, duration_ms=3.0, collective_type="AllGather")
        dag = _dag(ar, ag)  # no edges: both start at t=0 in parallel
        result = replay(dag)

        assert result.total_time_ms == pytest.approx(4.0)

        # Invariant 1: components sum to total_time_ms
        total_components = result.compute_ms + result.exposed_comm_ms + result.bubble_ms
        assert total_components == pytest.approx(result.total_time_ms)

        # Invariant 2: bubble must not be clamped to zero (GPU 0 idle after
        # the shorter recv finishes; GPUs 1 and 2 are idle the entire time)
        assert result.bubble_ms > 0.0

        # Invariant 3: per-type sum exceeds the de-duplicated total,
        # confirming the union-based computation removed the overlap
        per_type_sum = (
            result.exposed_comm_by_type.get("AllReduce", 0.0)
            + result.exposed_comm_by_type.get("AllGather", 0.0)
        )
        assert result.exposed_comm_ms < per_type_sum
