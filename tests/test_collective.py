"""Tests for collective decomposition and ring algorithms."""

import pytest

from simulon.collective.common import P2PFlow
from simulon.collective.ring import (
    ring_all_gather,
    ring_all_reduce,
    ring_all_to_all,
    ring_reduce_scatter,
)
from simulon.collective.nvls import nvls_all_reduce
from simulon.collective.decompose import decompose_collective, CollectiveResult


# ---------------------------------------------------------------------------
# Ring ReduceScatter
# ---------------------------------------------------------------------------


def test_ring_reduce_scatter_flow_count():
    """N=4, C=1: nsteps=3, 4 ranks * 1 channel * 3 steps = 12 flows."""
    flows, nfid = ring_reduce_scatter([0, 1, 2, 3], data_size=1024, num_channels=1)
    assert len(flows) == 12
    assert nfid == 12


def test_ring_reduce_scatter_flow_count_multichannel():
    """N=4, C=2: 4 * 2 * 3 = 24 flows."""
    flows, nfid = ring_reduce_scatter([0, 1, 2, 3], data_size=1024, num_channels=2)
    assert len(flows) == 24
    assert nfid == 24


def test_ring_reduce_scatter_step0_has_no_parents():
    """Step 0 flows have empty parent_flow_ids."""
    flows, _ = ring_reduce_scatter([0, 1, 2, 3], data_size=1024)
    step0 = flows[:4]  # first 4 flows (N=4, C=1)
    for f in step0:
        assert f.parent_flow_ids == []


def test_ring_reduce_scatter_step1_has_parents():
    """Step 1 flows reference step 0 flows as parents."""
    flows, _ = ring_reduce_scatter([0, 1, 2, 3], data_size=1024)
    step1 = flows[4:8]
    for f in step1:
        assert len(f.parent_flow_ids) == 1
        parent_fid = f.parent_flow_ids[0]
        assert 0 <= parent_fid < 4  # step0 flow ids are 0-3


def test_ring_reduce_scatter_child_backfill():
    """Each flow's child_flow_ids contains flows that list it as parent."""
    flows, _ = ring_reduce_scatter([0, 1, 2, 3], data_size=1024)
    fid_to_flow = {f.flow_id: f for f in flows}
    for f in flows:
        for child_fid in f.child_flow_ids:
            child = fid_to_flow[child_fid]
            assert f.flow_id in child.parent_flow_ids


def test_ring_reduce_scatter_conn_type():
    flows, _ = ring_reduce_scatter([0, 1, 2, 3], data_size=1024)
    for f in flows:
        assert f.conn_type == "RING"


def test_ring_reduce_scatter_single_rank():
    """N=1 produces no flows."""
    flows, nfid = ring_reduce_scatter([0], data_size=1024)
    assert flows == []
    assert nfid == 0


def test_ring_reduce_scatter_flow_id_start():
    """flow_id_start is respected."""
    flows, nfid = ring_reduce_scatter([0, 1, 2], data_size=1024, flow_id_start=100)
    assert flows[0].flow_id == 100
    assert nfid == 100 + len(flows)


# ---------------------------------------------------------------------------
# Ring AllGather
# ---------------------------------------------------------------------------


def test_ring_all_gather_flow_count():
    """N=4, C=1: 4 * 1 * 3 = 12 flows."""
    flows, nfid = ring_all_gather([0, 1, 2, 3], data_size=1024)
    assert len(flows) == 12
    assert nfid == 12


def test_ring_all_gather_step0_has_no_parents():
    flows, _ = ring_all_gather([0, 1, 2, 3], data_size=1024)
    step0 = flows[:4]
    for f in step0:
        assert f.parent_flow_ids == []


def test_ring_all_gather_step1_parent_from_sender():
    """Step 1, rank i's parent is step 0 flow from rank (i-1)%N."""
    flows, _ = ring_all_gather([0, 1, 2, 3], data_size=1024)
    # step 1 flows: indices 4..7
    step1 = flows[4:8]
    for f in step1:
        assert len(f.parent_flow_ids) == 1


# ---------------------------------------------------------------------------
# Ring AllReduce
# ---------------------------------------------------------------------------


def test_ring_all_reduce_flow_count():
    """AllReduce = RS + AG: 12 + 12 = 24 for N=4."""
    flows, nfid = ring_all_reduce([0, 1, 2, 3], data_size=1024)
    assert len(flows) == 24
    assert nfid == 24


def test_ring_all_reduce_ag_step0_has_rs_parents():
    """AG step-0 flows should have RS final-step flows as parents."""
    N = 4
    flows, _ = ring_all_reduce([0, 1, 2, 3], data_size=1024)
    ag_flows = flows[12:]  # second half
    ag_step0 = ag_flows[:N]
    for f in ag_step0:
        assert len(f.parent_flow_ids) == 1
        parent_fid = f.parent_flow_ids[0]
        # RS final step (step N-2=2): flow ids 8..11
        assert 8 <= parent_fid <= 11


def test_ring_all_reduce_single_rank():
    flows, nfid = ring_all_reduce([5], data_size=1024)
    assert flows == []
    assert nfid == 0


# ---------------------------------------------------------------------------
# Ring AllToAll
# ---------------------------------------------------------------------------


def test_ring_all_to_all_flow_count():
    """N=4: 4*3 = 12 flows."""
    flows, nfid = ring_all_to_all([0, 1, 2, 3], data_size=1024)
    assert len(flows) == 12
    assert nfid == 12


def test_ring_all_to_all_no_self_flows():
    flows, _ = ring_all_to_all([0, 1, 2, 3], data_size=1024)
    for f in flows:
        assert f.src != f.dst


def test_ring_all_to_all_no_parents():
    flows, _ = ring_all_to_all([0, 1, 2, 3], data_size=1024)
    for f in flows:
        assert f.parent_flow_ids == []


def test_ring_all_to_all_chunk_size():
    """Each flow carries data_size // N bytes."""
    N = 4
    data_size = 4096
    flows, _ = ring_all_to_all(list(range(N)), data_size=data_size)
    for f in flows:
        assert f.flow_size == data_size // N


# ---------------------------------------------------------------------------
# NVLS AllReduce (stub — not yet implemented)
# ---------------------------------------------------------------------------


def test_nvls_all_reduce_not_implemented():
    with pytest.raises(NotImplementedError):
        nvls_all_reduce([0, 1, 2, 3], data_size=1024)


# ---------------------------------------------------------------------------
# decompose_collective
# ---------------------------------------------------------------------------


def test_decompose_ring_all_reduce():
    result, nfid = decompose_collective(
        "AllReduce", [0, 1, 2, 3], data_size=1024, algorithm="ring"
    )
    assert isinstance(result, CollectiveResult)
    assert len(result.flows) == 24


def test_decompose_ring_reduce_scatter():
    result, nfid = decompose_collective(
        "ReduceScatter", [0, 1, 2, 3], data_size=1024, algorithm="ring"
    )
    assert len(result.flows) == 12


def test_decompose_ring_all_gather():
    result, nfid = decompose_collective(
        "AllGather", [0, 1, 2, 3], data_size=1024, algorithm="ring"
    )
    assert len(result.flows) == 12


def test_decompose_ring_all_to_all():
    result, nfid = decompose_collective(
        "AllToAll", [0, 1, 2, 3], data_size=1024, algorithm="ring"
    )
    assert len(result.flows) == 12


def test_decompose_nvls_not_implemented():
    with pytest.raises(NotImplementedError):
        decompose_collective("AllReduce", list(range(8)), data_size=4096, algorithm="nvls")


def test_decompose_nvls_non_allreduce_unsupported():
    """Non-AllReduce collectives with nvls algorithm raise ValueError (no silent fallback)."""
    with pytest.raises(ValueError, match="Unsupported combination"):
        decompose_collective("AllGather", [0, 1, 2, 3], data_size=1024, algorithm="nvls")


def test_decompose_unknown_algorithm():
    with pytest.raises(ValueError, match="Unsupported combination"):
        decompose_collective("AllReduce", [0, 1, 2, 3], data_size=1024, algorithm="unknown")


def test_decompose_unknown_collective():
    with pytest.raises(ValueError, match="Unsupported combination"):
        decompose_collective("Broadcast", [0, 1, 2, 3], data_size=1024, algorithm="ring")


def test_decompose_flow_id_continuity():
    """next_flow_id from one call can be used as flow_id_start for next."""
    result1, nfid1 = decompose_collective("AllGather", [0, 1, 2], data_size=768)
    result2, nfid2 = decompose_collective(
        "ReduceScatter", [0, 1, 2], data_size=768, flow_id_start=nfid1
    )
    all_fids = [f.flow_id for f in result1.flows] + [f.flow_id for f in result2.flows]
    assert len(all_fids) == len(set(all_fids)), "Duplicate flow IDs across consecutive calls"
