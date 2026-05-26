"""NVLS (NVLink SHARP) collective algorithms.

NVLS offloads reduction to the NVSwitch fabric, enabling in-network AllReduce
within a single NVLink domain (one node). Two variants exist in NCCL:

  NVLS       — single-node: each GPU gathers to NVSwitch, which reduces and
                scatters back. Modeled as a star topology through a virtual
                switch node (ID in the _SWITCH_BASE namespace).
  NVLS_TREE  — multi-node: intra-node NVLS combined with an inter-node tree
                for scale-out.

Key design constraint:
  NVLS must only be selected for INTRA-NODE groups. The algorithm must be
  chosen per-collective based on whether all ranks share the same NVSwitch
  domain — NOT as a global flag applied to all collectives. Applying it to
  inter-node DP AllReduce or to AllGather/ReduceScatter is wrong.

Virtual switch node IDs:
  Switch nodes use IDs starting at _SWITCH_BASE (1_000_000), well beyond any
  realistic GPU rank. Using max(group_ranks)+1 would collide with real GPU
  ranks when nvls_all_reduce is called on a non-highest-ranked node group
  (e.g., ranks [0..7] in a 2-node system would give switch_id=8 = GPU 8).

References:
  - NCCL MockNccl.h: NCCL_ALGO_NVLS = 4, NCCL_ALGO_NVLS_TREE = 5
  - SimAI calbusbw.cc: calcNVLSBusBw — requires gpus_per_node == 8
  - astra-sim MockNcclGroup.cc: genAllreduceNVLSFlowModels (single-node),
    generate_flow_model_nvls_tree_allreduce_up/down (multi-node tree)
"""

from __future__ import annotations

from simulon.collective.common import P2PFlow

# Fallback virtual switch node IDs used when the C++ binding is unavailable.
# Starting at 1_000_000 avoids colliding with any realistic GPU rank.
_SWITCH_BASE = 1_000_000


def _get_nvls_switch_ids(N: int, num_nodes: int, gpus_per_node: int) -> list[int]:
    """Return the virtual NVSwitch node ID for each node using MockNcclGroup.

    For single-node groups, uses get_nvls_channels (returns TreeChannels-style dict).
    For multi-node groups, uses get_nvls_tree_channels (returns flattened tuple with
    switch_id per entry). Falls back to _SWITCH_BASE + n if the C++ binding is
    unavailable or the query fails.
    """
    try:
        import simulon._mocknccl as _m

        nvswitches = [N + i for i in range(num_nodes)]
        g = _m.MockNcclGroup(N, gpus_per_node, N, 1, 1, 1, 1, nvswitches, _m.GPUType.H100)
        if num_nodes == 1:
            # Single-node: get_nvls_channels returns TreeChannels-style dict.
            # The last entry (key = gpus_per_node) is the virtual switch.
            raw = g.get_nvls_channels(0, _m.GroupType.TP)
            ch0 = raw[min(raw.keys())]
            switch_id = next((rank for rank in ch0 if rank >= gpus_per_node), _SWITCH_BASE)
            return [switch_id]
        else:
            # Multi-node: get_nvls_tree_channels returns (depth, rank, switch_id, children).
            raw = g.get_nvls_tree_channels(0, _m.GroupType.TP)
            ch0 = raw[min(raw.keys())]
            switch_ids: list[int] = []
            for node_idx in range(num_nodes):
                rep_rank = node_idx * gpus_per_node
                entry = ch0.get(rep_rank)
                if entry is not None:
                    switch_ids.append(entry[2])  # switch_id field
                else:
                    switch_ids.append(_SWITCH_BASE + node_idx)
            return switch_ids
    except Exception:
        return [_SWITCH_BASE + n for n in range(num_nodes)]


def nvls_all_reduce(
    group_ranks: list[int], data_size: int, num_channels: int = 1, flow_id_start: int = 0
) -> tuple[list[P2PFlow], int]:
    """NVLS AllReduce — single-node star topology through virtual NVSwitch.

    Models the NVSwitch-accelerated AllReduce as two phases:

    Phase 1 — Reduce (gather to switch):
      All N GPUs send chunk_size bytes to the virtual switch simultaneously.
      No dependencies between reduce flows (they all start at once).

    Phase 2 — Broadcast (scatter from switch):
      The virtual switch sends chunk_size bytes back to each GPU.
      Each broadcast flow depends on ALL reduce flows completing, because
      the switch must have the fully-reduced value before scattering.

    The virtual switch node ID is max(group_ranks) + 1, outside the real
    GPU rank space. The replayer assigns this node its own timeline.

    num_channels: each channel carries data_size // num_channels bytes and
    runs independently in parallel.
    """
    N = len(group_ranks)
    if N == 1:
        return [], flow_id_start

    # Single-node: one switch for all GPUs. Query switch ID from MockNcclGroup.
    switch_id = _get_nvls_switch_ids(N, 1, N)[0]
    chunk_size = data_size // num_channels
    chunk_count = num_channels

    fid = flow_id_start
    all_flows: list[P2PFlow] = []

    for ch in range(num_channels):
        # --- Phase 1: Reduce (GPU → switch) ---
        reduce_flow_ids: list[int] = []
        for gpu_rank in group_ranks:
            flow = P2PFlow(
                flow_id=fid,
                src=gpu_rank,
                dst=switch_id,
                flow_size=chunk_size,
                parent_flow_ids=[],  # all start in parallel
                child_flow_ids=[],
                channel_id=ch,
                chunk_id=ch,
                chunk_count=chunk_count,
                conn_type="NVLS",
            )
            reduce_flow_ids.append(fid)
            all_flows.append(flow)
            fid += 1

        # --- Phase 2: Broadcast (switch → GPU) ---
        # Each broadcast flow depends on ALL reduce flows completing.
        for gpu_rank in group_ranks:
            flow = P2PFlow(
                flow_id=fid,
                src=switch_id,
                dst=gpu_rank,
                flow_size=chunk_size,
                parent_flow_ids=list(reduce_flow_ids),
                child_flow_ids=[],
                channel_id=ch,
                chunk_id=ch,
                chunk_count=chunk_count,
                conn_type="NVLS",
            )
            all_flows.append(flow)
            fid += 1

    # Back-fill child_flow_ids from parent_flow_ids
    fid_to_flow = {f.flow_id: f for f in all_flows}
    for flow in all_flows:
        for pid in flow.parent_flow_ids:
            parent_flow = fid_to_flow.get(pid)
            if parent_flow and flow.flow_id not in parent_flow.child_flow_ids:
                parent_flow.child_flow_ids.append(flow.flow_id)

    return all_flows, fid


def nvls_tree_all_reduce(
    group_ranks: list[int], data_size: int, num_channels: int = 1, flow_id_start: int = 0
) -> tuple[list[P2PFlow], int]:
    """NVLS Tree AllReduce — intra-node NVLS + inter-node tree (multi-node).

    Phase 1: ReduceScatter via intra-node NVLS (reduce within each node).
    Phase 2: Inter-node AllReduce via tree topology.
    Phase 3: AllGather via intra-node NVLS (broadcast back within each node).

    The inter-node phase uses the tree topology from MockNcclGroup, modeling
    node-level representatives communicating via NIC.

    This algorithm requires gpus_per_node == 8 (NVSwitch constraint).
    """
    from simulon.collective.tree import _get_tree_topology, _infer_gpus_per_node

    N = len(group_ranks)
    if N == 1:
        return [], flow_id_start

    gpus_per_node = _infer_gpus_per_node(group_ranks)
    num_nodes = max(1, N // gpus_per_node)

    if num_nodes == 1:
        # Single-node: degenerate to plain NVLS
        return nvls_all_reduce(group_ranks, data_size, num_channels, flow_id_start)

    chunk_size = data_size // num_channels
    chunk_count = num_channels

    fid = flow_id_start
    all_flows: list[P2PFlow] = []

    # Node groups: ranks grouped by node
    node_groups: list[list[int]] = []
    for n in range(num_nodes):
        node_groups.append(group_ranks[n * gpus_per_node : (n + 1) * gpus_per_node])

    # One virtual switch per node; IDs from MockNcclGroup (N + node_index).
    switch_ids = _get_nvls_switch_ids(N, num_nodes, gpus_per_node)

    # Inter-node tree: use the first GPU of each node as the node representative
    node_representatives = [node_groups[n][0] for n in range(num_nodes)]
    inter_topology = _get_tree_topology(num_nodes, num_nodes, 1, num_channels)

    for ch in range(num_channels):
        ch_topo = inter_topology.get(ch, inter_topology[min(inter_topology)])

        # ---------------------------------------------------------------
        # Phase 1: Intra-node ReduceScatter (NVLS gather within each node)
        # Each GPU sends chunk to its node's virtual switch.
        # ---------------------------------------------------------------
        intra_reduce_fids: dict[int, list[int]] = {n: [] for n in range(num_nodes)}
        for n, (node_ranks, switch_id) in enumerate(zip(node_groups, switch_ids, strict=False)):
            for gpu_rank in node_ranks:
                flow = P2PFlow(
                    flow_id=fid,
                    src=gpu_rank,
                    dst=switch_id,
                    flow_size=chunk_size,
                    parent_flow_ids=[],
                    child_flow_ids=[],
                    channel_id=ch,
                    chunk_id=ch,
                    chunk_count=chunk_count,
                    conn_type="NVLS",
                )
                intra_reduce_fids[n].append(fid)
                all_flows.append(flow)
                fid += 1

        # ---------------------------------------------------------------
        # Phase 2: Inter-node tree AllReduce (between node representatives)
        # Uses tree topology; node representative = group_ranks[node * gpus_per_node]
        # Flows depend on intra-node reduce completing at each node.
        # ---------------------------------------------------------------
        inter_reduce_flow_ids: dict[int, int] = {}  # node_idx → flow_id of reduce-up

        def _reduce_order_inter(topo):
            root = next(r for r, (p, _) in topo.items() if p == -1)
            visited = []
            queue = [root]
            while queue:
                cur = queue.pop(0)
                visited.append(cur)
                _, children = topo[cur]
                queue.extend(children)
            return list(reversed(visited))

        order = _reduce_order_inter(ch_topo)
        for node_idx in order:
            parent_node, children_nodes = ch_topo[node_idx]
            if parent_node == -1:
                continue
            # Dependency: intra-node reduce at this node + children's inter-node reduce flows
            parent_fids = list(intra_reduce_fids[node_idx])
            parent_fids += [
                inter_reduce_flow_ids[c] for c in children_nodes if c in inter_reduce_flow_ids
            ]
            src = node_representatives[node_idx]
            dst = node_representatives[parent_node]
            flow = P2PFlow(
                flow_id=fid,
                src=src,
                dst=dst,
                flow_size=chunk_size,
                parent_flow_ids=parent_fids,
                child_flow_ids=[],
                channel_id=ch,
                chunk_id=ch,
                chunk_count=chunk_count,
                conn_type="NET",
            )
            inter_reduce_flow_ids[node_idx] = fid
            all_flows.append(flow)
            fid += 1

        # Inter-node broadcast (from root downward)
        inter_bcast_flow_ids: dict[int, int] = {}
        root_node = next(r for r, (p, _) in ch_topo.items() if p == -1)
        _, root_children_nodes = ch_topo[root_node]
        root_deps = list(intra_reduce_fids[root_node]) + [
            inter_reduce_flow_ids[c] for c in root_children_nodes if c in inter_reduce_flow_ids
        ]

        queue_bfs = [root_node]
        while queue_bfs:
            cur_node = queue_bfs.pop(0)
            _, child_nodes = ch_topo[cur_node]
            for child_node in child_nodes:
                parent_bcast = inter_bcast_flow_ids.get(cur_node)
                if cur_node == root_node:
                    deps = root_deps
                else:
                    deps = [parent_bcast] if parent_bcast is not None else []
                src = node_representatives[cur_node]
                dst = node_representatives[child_node]
                flow = P2PFlow(
                    flow_id=fid,
                    src=src,
                    dst=dst,
                    flow_size=chunk_size,
                    parent_flow_ids=deps,
                    child_flow_ids=[],
                    channel_id=ch,
                    chunk_id=ch,
                    chunk_count=chunk_count,
                    conn_type="NET",
                )
                inter_bcast_flow_ids[child_node] = fid
                all_flows.append(flow)
                fid += 1
            queue_bfs.extend(child_nodes)

        # ---------------------------------------------------------------
        # Phase 3: Intra-node AllGather (NVLS scatter back within each node)
        # Each node's switch sends back to all GPUs in that node.
        # Depends on: inter-node broadcast reaching this node's representative.
        # ---------------------------------------------------------------
        for n, (node_ranks, switch_id) in enumerate(zip(node_groups, switch_ids, strict=False)):
            # The node's switch can scatter once the inter-node bcast arrives
            # (or for root node, once the inter-node reduce is done).
            if n == root_node:
                scatter_deps = list(intra_reduce_fids[n]) + [
                    inter_reduce_flow_ids[c]
                    for c in root_children_nodes
                    if c in inter_reduce_flow_ids
                ]
            else:
                inter_bcast = inter_bcast_flow_ids.get(n)
                scatter_deps = [inter_bcast] if inter_bcast is not None else []

            for gpu_rank in node_ranks:
                flow = P2PFlow(
                    flow_id=fid,
                    src=switch_id,
                    dst=gpu_rank,
                    flow_size=chunk_size,
                    parent_flow_ids=list(scatter_deps),
                    child_flow_ids=[],
                    channel_id=ch,
                    chunk_id=ch,
                    chunk_count=chunk_count,
                    conn_type="NVLS",
                )
                all_flows.append(flow)
                fid += 1

    # Back-fill child_flow_ids
    fid_to_flow = {f.flow_id: f for f in all_flows}
    for flow in all_flows:
        for pid in flow.parent_flow_ids:
            parent_flow = fid_to_flow.get(pid)
            if parent_flow and flow.flow_id not in parent_flow.child_flow_ids:
                parent_flow.child_flow_ids.append(flow.flow_id)

    return all_flows, fid
