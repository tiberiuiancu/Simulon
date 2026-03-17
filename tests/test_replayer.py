"""Unit tests for the DAG replayer and supporting utilities."""

import pytest

from simulon.backend.dag.nodes import CommNode, ComputeNode, DAGEdge, ExecutionDAG
from simulon.backend.dag.replayer import (
    SimulationResult,
    _get_link_params,
    _parse_latency,
    _parse_speed,
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
    result = replay(dag, dc)

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
    result = replay(dag, dc)

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
    result = replay(dag, dc)

    # Parallel: 1 ms total
    assert abs(result.total_time_ms - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Compute node
# ---------------------------------------------------------------------------


def test_single_compute_node():
    dc = _dc()
    dag = _dag(_compute(0, gpu_rank=0, duration_ms=5.0))
    result = replay(dag, dc)

    assert abs(result.total_time_ms - 5.0) < 1e-9
    assert abs(result.per_gpu_times_ms[0] - 5.0) < 1e-9
    assert abs(result.compute_time_ms[0] - 5.0) < 1e-9


def test_compute_node_none_duration():
    """ComputeNode with duration_ms=None contributes 0 ms."""
    dc = _dc()
    n = _compute(0, gpu_rank=0, duration_ms=None)
    dag = _dag(n)
    result = replay(dag, dc)
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
    result = replay(dag, dc)

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
    result = replay(dag, dc)

    # Parent finishes at 1ms, child starts at 1ms, finishes at 2ms
    assert abs(result.total_time_ms - 2.0) < 1e-9


# ---------------------------------------------------------------------------
# comm_time_ms accounts for both src and dst
# ---------------------------------------------------------------------------


def test_comm_time_counted_for_both_endpoints():
    """comm_time_ms[gpu] is incremented for both src_gpu and dst_gpu."""
    dc = _dc(gpus_per_node=4, nvswitch_speed="1GBps", nvswitch_latency="0ms")
    bytes_ = 1_000_000  # 1 ms

    dag = _dag(_comm(0, src_gpu=0, dst_gpu=1, bytes=bytes_))
    result = replay(dag, dc)

    assert abs(result.comm_time_ms[0] - 1.0) < 1e-9  # src counted
    assert abs(result.comm_time_ms[1] - 1.0) < 1e-9  # dst counted


# ---------------------------------------------------------------------------
# per_gpu_times_ms covers all participating GPUs
# ---------------------------------------------------------------------------


def test_per_gpu_finish_covers_src_gpu():
    """A GPU that only sends (never receives) must appear in per_gpu_times_ms."""
    dc = _dc(gpus_per_node=4, nvswitch_speed="1GBps", nvswitch_latency="0ms")
    bytes_ = 1_000_000

    dag = _dag(_comm(0, src_gpu=0, dst_gpu=1, bytes=bytes_))
    result = replay(dag, dc)

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
    result = replay(dag, dc)

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
    result = replay(dag, dc)

    assert abs(result.total_time_ms - 1.0) < 1e-9
