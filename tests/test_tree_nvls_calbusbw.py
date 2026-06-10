"""Tests for tree AllReduce, NVLS Tree AllReduce, calbusbw, and nccl_profile loading.

Covers:
- tree_all_reduce: structural invariants (conn_type, chunk_size, single-rank no-op,
  reduce-up dependency chain, broadcast-down dependency chain, child_flow_ids backfill,
  flow_id_start, multichannel)
- nvls_tree_all_reduce: single-node fallback, multi-node phase counts and conn_types,
  child_flow_ids backfill
- cal_busbw: ValueError with no profile, intra_bw from profile, inter_bw=None for
  single-node, inter_bw set for multi-node, algorithm auto-selection
- load_nccl_profile: known GPU returns NcclProfile, unknown returns None
"""
from __future__ import annotations

import pytest

from simulon.collective.tree import tree_all_reduce
from simulon.collective.nvls import nvls_all_reduce, nvls_tree_all_reduce
from simulon.config.nccl_profile import NcclDataPoint, NcclAlgoMeasurements, NcclProfile
from simulon.collective.calbusbw import cal_busbw


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profile(
    allreduce_ring_bw: float = 200.0,
    allgather_ring_bw: float = 180.0,
    reducescatter_ring_bw: float = 180.0,
    alltoall_ring_bw: float = 160.0,
    allreduce_tree_bw: float = 0.0,
    allreduce_nvls_bw: float = 0.0,
    allreduce_nvls_tree_bw: float = 0.0,
) -> NcclProfile:
    """Build a minimal NcclProfile with a single data point at 1 GB."""
    size = 1 << 30  # 1 GB

    def _pts(bw: float) -> list[NcclDataPoint]:
        return [NcclDataPoint(size_bytes=size, bus_bw_GBps=bw)] if bw > 0 else []

    return NcclProfile(
        gpus_per_node=8,
        AllReduce=NcclAlgoMeasurements(
            ring=_pts(allreduce_ring_bw),
            tree=_pts(allreduce_tree_bw),
            nvls=_pts(allreduce_nvls_bw),
            nvls_tree=_pts(allreduce_nvls_tree_bw),
        ),
        AllGather=NcclAlgoMeasurements(ring=_pts(allgather_ring_bw)),
        ReduceScatter=NcclAlgoMeasurements(ring=_pts(reducescatter_ring_bw)),
        AllToAll=NcclAlgoMeasurements(ring=_pts(alltoall_ring_bw)),
    )


# ---------------------------------------------------------------------------
# tree_all_reduce — structural invariants
# ---------------------------------------------------------------------------


class TestTreeAllReduce:
    def test_single_rank_returns_no_flows(self):
        """N=1 is a no-op: returns empty list and unchanged flow_id_start."""
        flows, nfid = tree_all_reduce([7], data_size=1024, num_channels=1, flow_id_start=5)
        assert flows == []
        assert nfid == 5

    def test_conn_type_is_tree(self):
        """All flows produced by tree_all_reduce carry conn_type='TREE'."""
        flows, _ = tree_all_reduce([0, 1, 2, 3], data_size=4096, num_channels=1)
        assert len(flows) > 0
        for f in flows:
            assert f.conn_type == "TREE"

    def test_chunk_size_equals_data_divided_by_channels(self):
        """Each flow carries data_size // num_channels bytes."""
        data_size = 4096
        num_channels = 2
        flows, _ = tree_all_reduce([0, 1, 2, 3], data_size=data_size, num_channels=num_channels)
        expected_chunk = data_size // num_channels
        for f in flows:
            assert f.flow_size == expected_chunk

    def test_flow_id_start_is_respected(self):
        """flow_id_start offsets all generated flow IDs."""
        start = 42
        flows, nfid = tree_all_reduce([0, 1, 2, 3], data_size=4096, num_channels=1,
                                       flow_id_start=start)
        assert flows[0].flow_id == start
        assert nfid == start + len(flows)

    def test_flow_ids_are_contiguous_and_unique(self):
        """Flow IDs form a contiguous sequence with no duplicates."""
        flows, nfid = tree_all_reduce([0, 1, 2, 3], data_size=4096, num_channels=1,
                                       flow_id_start=0)
        fids = [f.flow_id for f in flows]
        assert len(fids) == len(set(fids))
        assert min(fids) == 0
        assert max(fids) == len(flows) - 1
        assert nfid == len(flows)

    def test_reduce_up_leaves_have_no_parents(self):
        """Leaf nodes in the reduce-up phase have no parent dependencies."""
        flows, _ = tree_all_reduce([0, 1, 2, 3], data_size=4096, num_channels=1)
        # Reduce-up flows form the first phase; leaf flows have empty parent_flow_ids.
        reduce_flows = [f for f in flows if f.dst in [0, 1, 2, 3] and f.src in [0, 1, 2, 3]]
        no_parent = [f for f in reduce_flows if f.parent_flow_ids == []]
        assert len(no_parent) > 0, "At least one leaf reduce-up flow must have no parents"

    def test_broadcast_flows_depend_on_reduce_phase(self):
        """Broadcast flows have at least one parent (from the reduce-up phase)."""
        flows, _ = tree_all_reduce([0, 1, 2, 3], data_size=4096, num_channels=1)
        # Broadcast flows (root → children) always have >= 1 dependency.
        # Identify them: they are not the root's reduce-up send (root has no reduce-up send).
        # The root's broadcast sends to children must depend on reduce-up flows.
        fids_with_parents = [f for f in flows if len(f.parent_flow_ids) > 0]
        assert len(fids_with_parents) > 0

    def test_child_flow_ids_backfill_consistent(self):
        """For every flow F, F's child_flow_ids lists each flow that names F as parent."""
        flows, _ = tree_all_reduce([0, 1, 2, 3], data_size=4096, num_channels=1)
        fid_to_flow = {f.flow_id: f for f in flows}
        for f in flows:
            for child_fid in f.child_flow_ids:
                child = fid_to_flow[child_fid]
                assert f.flow_id in child.parent_flow_ids, (
                    f"Flow {f.flow_id} lists {child_fid} as child, "
                    f"but {child_fid} does not list {f.flow_id} as parent"
                )

    def test_parent_flow_ids_reference_valid_flows(self):
        """All parent_flow_ids reference flow IDs that exist in the result."""
        flows, _ = tree_all_reduce([0, 1, 2, 3], data_size=4096, num_channels=1)
        valid_fids = {f.flow_id for f in flows}
        for f in flows:
            for pid in f.parent_flow_ids:
                assert pid in valid_fids, f"Flow {f.flow_id} references missing parent {pid}"

    def test_no_self_flows(self):
        """No flow sends to itself (src != dst)."""
        flows, _ = tree_all_reduce([0, 1, 2, 3], data_size=4096, num_channels=1)
        for f in flows:
            assert f.src != f.dst

    def test_src_dst_within_group(self):
        """All src and dst ranks belong to group_ranks."""
        group = [0, 1, 2, 3]
        flows, _ = tree_all_reduce(group, data_size=4096, num_channels=1)
        group_set = set(group)
        for f in flows:
            assert f.src in group_set
            assert f.dst in group_set

    def test_multichannel_doubles_flow_count(self):
        """Doubling num_channels doubles the number of flows."""
        flows1, _ = tree_all_reduce([0, 1, 2, 3], data_size=4096, num_channels=1)
        flows2, _ = tree_all_reduce([0, 1, 2, 3], data_size=4096, num_channels=2)
        assert len(flows2) == 2 * len(flows1)

    def test_two_ranks_produces_exactly_two_flows(self):
        """N=2: one reduce-up flow and one broadcast-down flow — exactly 2 flows total."""
        flows, nfid = tree_all_reduce([0, 1], data_size=1024, num_channels=1)
        assert len(flows) == 2
        assert nfid == 2

    def test_four_ranks_tree_reduce_has_n_minus_1_reduce_flows(self):
        """For N ranks, the reduce-up phase has exactly N-1 flows (one per non-root)."""
        N = 4
        flows, _ = tree_all_reduce(list(range(N)), data_size=4096, num_channels=1)
        # Total flows = (N-1) reduce-up + (N-1) broadcast-down = 2*(N-1) for a binary tree
        # Verify total equals 2*(N-1)
        assert len(flows) == 2 * (N - 1)

    def test_decompose_collective_routes_to_tree(self):
        """decompose_collective with algorithm='tree' and AllReduce calls tree_all_reduce."""
        from simulon.collective.decompose import decompose_collective, CollectiveResult
        result, nfid = decompose_collective(
            "AllReduce", [0, 1, 2, 3], data_size=4096, algorithm="tree", num_channels=1
        )
        assert isinstance(result, CollectiveResult)
        assert len(result.flows) > 0
        for f in result.flows:
            assert f.conn_type == "TREE"


# ---------------------------------------------------------------------------
# nvls_tree_all_reduce
# ---------------------------------------------------------------------------


class TestNvlsTreeAllReduce:
    def test_single_node_falls_back_to_nvls_all_reduce(self):
        """Single-node nvls_tree_all_reduce produces the same flows as nvls_all_reduce."""
        group = list(range(4))  # 4 consecutive ranks → 1 node (gpus_per_node=4)
        data_size = 4096
        flows_tree, _ = nvls_tree_all_reduce(group, data_size=data_size, num_channels=1)
        flows_nvls, _ = nvls_all_reduce(group, data_size=data_size, num_channels=1)
        # Same count and same structure
        assert len(flows_tree) == len(flows_nvls)

    def test_multi_node_produces_three_phases(self):
        """Multi-node NVLS tree has intra-reduce (NVLS), inter-node (NET), and scatter (NVLS)."""
        # 16 ranks, stride=8 → inferred gpus_per_node=8, 2 nodes
        group = list(range(16))
        flows, _ = nvls_tree_all_reduce(group, data_size=16 * 1024, num_channels=1)

        nvls_flows = [f for f in flows if f.conn_type == "NVLS"]
        net_flows = [f for f in flows if f.conn_type == "NET"]

        # Must have both NVLS and NET flows
        assert len(nvls_flows) > 0
        assert len(net_flows) > 0

    def test_multi_node_intra_reduce_no_parents(self):
        """Intra-node reduce flows (GPU→switch) in phase 1 have no parent dependencies."""
        group = list(range(16))
        flows, _ = nvls_tree_all_reduce(group, data_size=16 * 1024, num_channels=1)
        gpu_set = set(group)
        switch_ids = {f.dst for f in flows if f.src in gpu_set} - gpu_set

        reduce_flows = [f for f in flows if f.dst in switch_ids]
        # The very first batch (intra-node reduce) has no parents
        no_parent_reduce = [f for f in reduce_flows if f.parent_flow_ids == []]
        assert len(no_parent_reduce) > 0

    def test_multi_node_scatter_has_parents(self):
        """Intra-node scatter flows (switch→GPU) in phase 3 always have dependencies."""
        group = list(range(16))
        flows, _ = nvls_tree_all_reduce(group, data_size=16 * 1024, num_channels=1)
        gpu_set = set(group)
        switch_ids = {f.dst for f in flows if f.src in gpu_set} - gpu_set

        scatter_flows = [f for f in flows if f.src in switch_ids]
        for f in scatter_flows:
            assert len(f.parent_flow_ids) > 0, "Scatter flows must wait for inter-node phase"

    def test_child_flow_ids_backfill_consistent(self):
        """child_flow_ids are consistent with parent_flow_ids for multi-node case."""
        group = list(range(16))
        flows, _ = nvls_tree_all_reduce(group, data_size=16 * 1024, num_channels=1)
        fid_to_flow = {f.flow_id: f for f in flows}
        for f in flows:
            for child_fid in f.child_flow_ids:
                child = fid_to_flow[child_fid]
                assert f.flow_id in child.parent_flow_ids

    def test_single_rank_returns_empty(self):
        """N=1 is a no-op."""
        flows, nfid = nvls_tree_all_reduce([3], data_size=1024)
        assert flows == []
        assert nfid == 0

    def test_flow_ids_unique(self):
        """All flow IDs are unique in multi-node case."""
        group = list(range(16))
        flows, _ = nvls_tree_all_reduce(group, data_size=16 * 1024, num_channels=1)
        fids = [f.flow_id for f in flows]
        assert len(fids) == len(set(fids))

    def test_decompose_collective_routes_to_nvls_tree(self):
        """decompose_collective with algorithm='nvls_tree' and AllReduce works."""
        from simulon.collective.decompose import decompose_collective, CollectiveResult
        group = list(range(16))
        result, nfid = decompose_collective(
            "AllReduce", group, data_size=16 * 1024, algorithm="nvls_tree", num_channels=1
        )
        assert isinstance(result, CollectiveResult)
        assert len(result.flows) > 0


# ---------------------------------------------------------------------------
# NVLS AllReduce — additional multichannel invariants not in test_collective.py
# ---------------------------------------------------------------------------


class TestNvlsAllReduceMultichannel:
    def test_multichannel_doubles_flow_count(self):
        """2 channels produces 2x the flows of 1 channel."""
        N = 4
        flows1, _ = nvls_all_reduce(list(range(N)), data_size=4096, num_channels=1)
        flows2, _ = nvls_all_reduce(list(range(N)), data_size=4096, num_channels=2)
        assert len(flows2) == 2 * len(flows1)

    def test_chunk_size_divided_by_channels(self):
        """Each flow carries data_size // num_channels bytes."""
        data_size = 8192
        num_channels = 4
        flows, _ = nvls_all_reduce([0, 1, 2, 3], data_size=data_size, num_channels=num_channels)
        expected = data_size // num_channels
        for f in flows:
            assert f.flow_size == expected

    def test_flow_id_start_respected(self):
        """flow_id_start offsets all generated flow IDs."""
        start = 100
        flows, nfid = nvls_all_reduce([0, 1, 2, 3], data_size=4096, flow_id_start=start)
        assert flows[0].flow_id == start
        assert nfid == start + len(flows)

    def test_conn_type_is_nvls(self):
        """All nvls_all_reduce flows carry conn_type='NVLS'."""
        flows, _ = nvls_all_reduce([0, 1, 2, 3], data_size=4096)
        for f in flows:
            assert f.conn_type == "NVLS"

    def test_single_rank_is_noop(self):
        """N=1 produces no flows."""
        flows, nfid = nvls_all_reduce([5], data_size=1024)
        assert flows == []
        assert nfid == 0


# ---------------------------------------------------------------------------
# cal_busbw
# ---------------------------------------------------------------------------


class TestCalBusbw:
    def test_raises_value_error_with_no_profile(self):
        """cal_busbw raises ValueError when nccl_profile is None."""
        with pytest.raises(ValueError, match="No NCCL profile"):
            cal_busbw(
                collective_type="AllReduce",
                message_size_bytes=1 << 20,
                num_nodes=1,
                gpus_per_node=8,
                nics_per_node=1.0,
                nic_bw_GBps=50.0,
                nccl_profile=None,
                algorithm="ring",
            )

    def test_raises_value_error_for_unknown_collective(self):
        """cal_busbw raises ValueError for a collective_type not in the profile."""
        profile = _make_profile()
        with pytest.raises(ValueError):
            cal_busbw(
                collective_type="Broadcast",
                message_size_bytes=1 << 20,
                num_nodes=1,
                gpus_per_node=8,
                nics_per_node=1.0,
                nic_bw_GBps=50.0,
                nccl_profile=profile,
                algorithm="ring",
            )

    def test_single_node_inter_bw_is_none(self):
        """Single-node call returns inter_bw_GBps=None."""
        profile = _make_profile(allreduce_ring_bw=200.0)
        _, intra_bw, inter_bw = cal_busbw(
            collective_type="AllReduce",
            message_size_bytes=1 << 30,
            num_nodes=1,
            gpus_per_node=8,
            nics_per_node=1.0,
            nic_bw_GBps=50.0,
            nccl_profile=profile,
            algorithm="ring",
        )
        assert inter_bw is None
        assert intra_bw == pytest.approx(200.0)

    def test_multi_node_inter_bw_is_set(self):
        """Multi-node call returns a positive inter_bw_GBps."""
        profile = _make_profile(allreduce_ring_bw=200.0)
        _, intra_bw, inter_bw = cal_busbw(
            collective_type="AllReduce",
            message_size_bytes=1 << 30,
            num_nodes=2,
            gpus_per_node=8,
            nics_per_node=1.0,
            nic_bw_GBps=50.0,
            nccl_profile=profile,
            algorithm="ring",
        )
        assert inter_bw is not None
        assert inter_bw > 0.0
        assert intra_bw == pytest.approx(200.0)

    def test_intra_bw_matches_profile_at_exact_size(self):
        """When message_size == a profile data point, intra_bw equals that data point's bw."""
        size = 1 << 30  # 1 GB — matches our _make_profile's single point
        profile = _make_profile(allreduce_ring_bw=250.0)
        _, intra_bw, _ = cal_busbw(
            collective_type="AllReduce",
            message_size_bytes=size,
            num_nodes=1,
            gpus_per_node=8,
            nics_per_node=1.0,
            nic_bw_GBps=50.0,
            nccl_profile=profile,
            algorithm="ring",
        )
        assert intra_bw == pytest.approx(250.0)

    def test_auto_selects_ring_when_only_ring_available(self):
        """auto algorithm selects 'ring' when it is the only candidate with bw > 0."""
        profile = _make_profile(allreduce_ring_bw=200.0, allreduce_tree_bw=0.0)
        selected_algo, _, _ = cal_busbw(
            collective_type="AllReduce",
            message_size_bytes=1 << 30,
            num_nodes=1,
            gpus_per_node=8,
            nics_per_node=1.0,
            nic_bw_GBps=50.0,
            nccl_profile=profile,
            algorithm="auto",
        )
        assert selected_algo == "ring"

    def test_auto_selects_higher_bw_algorithm(self):
        """auto algorithm selects the algorithm with the highest bus BW."""
        # tree_bw > ring_bw → auto should pick tree
        profile = _make_profile(allreduce_ring_bw=200.0, allreduce_tree_bw=300.0)
        selected_algo, intra_bw, _ = cal_busbw(
            collective_type="AllReduce",
            message_size_bytes=1 << 30,
            num_nodes=1,
            gpus_per_node=8,
            nics_per_node=1.0,
            nic_bw_GBps=50.0,
            nccl_profile=profile,
            algorithm="auto",
        )
        assert selected_algo == "tree"
        assert intra_bw == pytest.approx(300.0)

    def test_auto_does_not_select_nvls_for_multi_node(self):
        """auto does not select nvls (single-node algorithm) when num_nodes > 1."""
        profile = _make_profile(
            allreduce_ring_bw=200.0,
            allreduce_nvls_bw=500.0,  # very high, but single-node only
        )
        selected_algo, _, _ = cal_busbw(
            collective_type="AllReduce",
            message_size_bytes=1 << 30,
            num_nodes=2,
            gpus_per_node=8,
            nics_per_node=1.0,
            nic_bw_GBps=50.0,
            nccl_profile=profile,
            algorithm="auto",
        )
        assert selected_algo != "nvls"

    def test_auto_non_allreduce_always_ring(self):
        """auto always selects 'ring' for non-AllReduce collectives."""
        profile = _make_profile(allgather_ring_bw=180.0)
        selected_algo, intra_bw, _ = cal_busbw(
            collective_type="AllGather",
            message_size_bytes=1 << 30,
            num_nodes=1,
            gpus_per_node=8,
            nics_per_node=1.0,
            nic_bw_GBps=50.0,
            nccl_profile=profile,
            algorithm="auto",
        )
        assert selected_algo == "ring"
        assert intra_bw == pytest.approx(180.0)

    def test_explicit_ring_algorithm_returns_ring_bw(self):
        """Explicit algorithm='ring' reads from ring measurements."""
        profile = _make_profile(allreduce_ring_bw=175.0)
        selected_algo, intra_bw, _ = cal_busbw(
            collective_type="AllReduce",
            message_size_bytes=1 << 30,
            num_nodes=1,
            gpus_per_node=8,
            nics_per_node=1.0,
            nic_bw_GBps=50.0,
            nccl_profile=profile,
            algorithm="ring",
        )
        assert selected_algo == "ring"
        assert intra_bw == pytest.approx(175.0)

    def test_explicit_tree_algorithm_returns_tree_bw(self):
        """Explicit algorithm='tree' reads from tree measurements."""
        profile = _make_profile(allreduce_ring_bw=175.0, allreduce_tree_bw=220.0)
        selected_algo, intra_bw, _ = cal_busbw(
            collective_type="AllReduce",
            message_size_bytes=1 << 30,
            num_nodes=1,
            gpus_per_node=8,
            nics_per_node=1.0,
            nic_bw_GBps=50.0,
            nccl_profile=profile,
            algorithm="tree",
        )
        assert selected_algo == "tree"
        assert intra_bw == pytest.approx(220.0)

    def test_inter_bw_scales_with_nic_bw(self):
        """inter_bw_GBps scales proportionally with nic_bw_GBps for fixed topology."""
        profile = _make_profile(allreduce_ring_bw=200.0)
        size = 1 << 30
        _, _, inter_bw_1 = cal_busbw("AllReduce", size, 2, 8, 1.0, 50.0, profile, "ring")
        _, _, inter_bw_2 = cal_busbw("AllReduce", size, 2, 8, 1.0, 100.0, profile, "ring")
        assert inter_bw_1 is not None and inter_bw_2 is not None
        assert inter_bw_2 == pytest.approx(inter_bw_1 * 2, rel=1e-6)

    def test_larger_message_clamps_to_max_profile_bw(self):
        """Message size beyond the last profile point clamps to the last measured bw."""
        size = 1 << 30  # our single profile point
        large_size = size * 1000  # far beyond the table
        profile = _make_profile(allreduce_ring_bw=200.0)
        _, intra_bw_large, _ = cal_busbw(
            "AllReduce", large_size, 1, 8, 1.0, 50.0, profile, "ring"
        )
        assert intra_bw_large == pytest.approx(200.0)

    def test_smaller_message_clamps_to_min_profile_bw(self):
        """Message size below the first profile point clamps to the first measured bw."""
        small_size = 1  # far below
        profile = _make_profile(allreduce_ring_bw=166.0)
        _, intra_bw_small, _ = cal_busbw(
            "AllReduce", small_size, 1, 8, 1.0, 50.0, profile, "ring"
        )
        assert intra_bw_small == pytest.approx(166.0)


# ---------------------------------------------------------------------------
# load_nccl_profile
# ---------------------------------------------------------------------------


class TestLoadNcclProfile:
    """Tests run from repo root (where templates/gpu/ lives)."""

    def test_h100_returns_nccl_profile(self):
        """load_nccl_profile('h100') returns an NcclProfile when h100.nccl.yaml exists."""
        from simulon.config.resolve import load_nccl_profile
        profile = load_nccl_profile("h100")
        assert profile is not None
        assert isinstance(profile, NcclProfile)

    def test_h100_profile_has_allreduce_ring_points(self):
        """h100.nccl.yaml has ring AllReduce measurements."""
        from simulon.config.resolve import load_nccl_profile
        profile = load_nccl_profile("h100")
        assert profile is not None
        assert len(profile.AllReduce.ring) > 0

    def test_h100_profile_gpus_per_node(self):
        """h100.nccl.yaml specifies gpus_per_node=8."""
        from simulon.config.resolve import load_nccl_profile
        profile = load_nccl_profile("h100")
        assert profile is not None
        assert profile.gpus_per_node == 8

    def test_nonexistent_gpu_returns_none(self):
        """load_nccl_profile returns None for an unknown GPU name."""
        from simulon.config.resolve import load_nccl_profile
        profile = load_nccl_profile("nonexistent_gpu_xyz")
        assert profile is None

    def test_case_insensitive_lookup(self):
        """load_nccl_profile('H100') (uppercase) resolves the same as 'h100'."""
        from simulon.config.resolve import load_nccl_profile
        profile_lower = load_nccl_profile("h100")
        profile_upper = load_nccl_profile("H100")
        # Both should either return a profile or None; they must agree
        if profile_lower is None:
            assert profile_upper is None
        else:
            assert profile_upper is not None
            assert profile_upper.gpus_per_node == profile_lower.gpus_per_node
