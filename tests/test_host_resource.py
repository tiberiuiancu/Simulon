"""The two-resource (host + GPU) replay.

Background: nsys profiling of Qwen3-1.7B on 4x GH200 showed 36-41% of every iteration is
GPU idle on the pace-setting rank, and that 79-83% of that idle is host time in framework
code with no CUDA call in flight (experiments/gap_attribution.py). The replay models the
host as a serial per-rank resource so that idle emerges from the schedule instead of being
charged as a flat surcharge per collective.
"""

from __future__ import annotations

import pytest

from simulon.backend.dag.nodes import CollectiveNode, ComputeNode, DAGEdge, ExecutionDAG
from simulon.backend.dag.replayer import replay


def _chain(dag: ExecutionDAG, rank: int, n: int, gpu_ms: float, host_ops: float, start_id: int):
    """n serial compute nodes on one rank, chained head-to-tail like the tracer builds them."""
    prev = None
    for i in range(n):
        nid = start_id + i
        node = ComputeNode(
            node_id=nid, gpu_rank=rank, kernel="compute", layer_id=-1, microbatch_id=0,
            pipeline_stage=0, phase="fwd", duration_ms=gpu_ms, host_ops=host_ops,
        )
        dag.add_compute_node(node)
        dag.rank_program.setdefault(rank, []).append(nid)
        if prev is not None:
            dag.add_edge(DAGEdge(src_node_id=prev, dst_node_id=nid))
        prev = nid
    return prev


def test_host_modelling_is_off_by_default():
    """Inertness gate: without host_cost_us the walk must be the pure-GPU one.

    This is what makes the change safe to land ahead of any template opting in.
    """
    dag = ExecutionDAG()
    _chain(dag, 0, 10, gpu_ms=1.0, host_ops=99.0, start_id=0)
    result = replay(dag, network_simulation="collective")
    assert result.total_time_ms == pytest.approx(10.0)
    assert result.host_stall_ms == 0.0


def test_gpu_bound_rank_ignores_host():
    """Host cost below GPU duration is hidden: only the first issue is ever exposed.

    The GPU cannot run node 0 until the host has issued it, so the timeline is one issue of
    fill (1 ms) plus the GPU chain. Every later issue overlaps the kernel before it. Across
    a real iteration's ~187k nodes that fill is negligible; what matters is that it does
    not accumulate.
    """
    dag = ExecutionDAG()
    _chain(dag, 0, 10, gpu_ms=5.0, host_ops=1.0, start_id=0)
    result = replay(dag, network_simulation="collective", host_cost_us=1000.0)
    assert result.total_time_ms == pytest.approx(51.0)  # 1 ms fill + 50 ms of kernels
    assert result.host_stall_ms == pytest.approx(1.0)  # the fill, and nothing after it


def test_host_bound_rank_is_paced_by_the_host():
    """Host cost above GPU duration sets the pace, and the excess is reported as stall.

    10 nodes x 2 ms host vs 1 ms GPU: the host finishes issuing at 20 ms and the last
    kernel then runs 1 ms. The 10 ms difference is idle the old model had to book as a
    per-collective surcharge.
    """
    dag = ExecutionDAG()
    _chain(dag, 0, 10, gpu_ms=1.0, host_ops=2.0, start_id=0)
    result = replay(dag, network_simulation="collective", host_cost_us=1000.0)
    assert result.total_time_ms == pytest.approx(21.0)
    assert result.host_stall_ms == pytest.approx(11.0)


def test_larger_kernels_hide_more_host_time():
    """The mbs mechanism, in miniature.

    Same program (same op count, same host cost), bigger kernels: the iteration grows by
    much less than the GPU work does, because the host time stops being exposed. A flat
    per-collective constant cannot express this, which is why launch_latency_ms had to be
    re-fitted at every microbatch size.
    """
    small = ExecutionDAG()
    _chain(small, 0, 10, gpu_ms=1.0, host_ops=3.0, start_id=0)
    big = ExecutionDAG()
    _chain(big, 0, 10, gpu_ms=3.0, host_ops=3.0, start_id=0)

    r_small = replay(small, network_simulation="collective", host_cost_us=1000.0)
    r_big = replay(big, network_simulation="collective", host_cost_us=1000.0)

    # The property that matters is how much host time is EXPOSED, i.e. how far the
    # iteration sits above the pure-GPU critical path.
    exposed_small = r_small.total_time_ms - 10 * 1.0
    exposed_big = r_big.total_time_ms - 10 * 3.0

    assert exposed_small == pytest.approx(21.0)  # host-bound: nearly all of it is exposed
    assert exposed_big == pytest.approx(3.0)  # balanced: only the one-op fill remains
    assert exposed_big < exposed_small / 5


def test_collective_waits_for_the_latest_issuing_rank():
    """A shared collective cannot start until every participant's host has issued it.

    Rank 0 is host-bound (issues at 20 ms), rank 1 is fast (issues at 2 ms). The collective
    is one node shared by both ranks, so skew is emergent -- no rank is designated the
    straggler and no per-collective constant is involved.
    """
    dag = ExecutionDAG()
    last0 = _chain(dag, 0, 10, gpu_ms=0.1, host_ops=2.0, start_id=0)
    last1 = _chain(dag, 1, 10, gpu_ms=0.1, host_ops=0.2, start_id=100)

    coll = CollectiveNode(
        node_id=200, collective_type="AllReduce", group_ranks=[0, 1], data_size=1024,
        name="ar", timestamp_ms=0.0, layer_id=-1, phase="fwd", algorithm="ring",
        num_channels=1, duration_ms=1.0, host_ops=0.0,
    )
    dag.add_collective_node(coll)
    for r in (0, 1):
        dag.rank_program.setdefault(r, []).append(200)
    dag.add_edge(DAGEdge(src_node_id=last0, dst_node_id=200))
    dag.add_edge(DAGEdge(src_node_id=last1, dst_node_id=200))

    replay(dag, network_simulation="collective", host_cost_us=1000.0)
    # Rank 0's host issues its 10 nodes over 20 ms and the last kernel takes 0.1 ms more;
    # rank 1 was ready at ~2.1 ms and waits. The wait is emergent from the two programs.
    assert coll.start_ms == pytest.approx(20.1)
    assert coll.finish_ms == pytest.approx(21.1)


def test_refuses_when_the_trace_carries_no_host_data():
    """Span traces already contain host stalls inside their wall-clock spans.

    Modelling host time on top of one would double-count, so this must fail loudly rather
    than silently produce a plausible number.
    """
    dag = ExecutionDAG()
    node = ComputeNode(
        node_id=0, gpu_rank=0, kernel="compute", layer_id=-1, microbatch_id=0,
        pipeline_stage=0, phase="fwd", duration_ms=1.0,
    )
    dag.add_compute_node(node)  # note: no rank_program
    with pytest.raises(ValueError, match="rank_program"):
        replay(dag, network_simulation="collective", host_cost_us=1000.0)
