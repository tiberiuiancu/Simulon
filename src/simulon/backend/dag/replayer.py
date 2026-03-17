from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass, field

from simulon.backend.dag.nodes import CommNode, ComputeNode, ExecutionDAG, NVSwitchNode
from simulon.config.dc import DatacenterConfig, NICSpec, SwitchSpec


# ---------------------------------------------------------------------------
# String-parsing helpers
# ---------------------------------------------------------------------------


def _parse_speed(s: str) -> float:
    """Parse a bandwidth string to bytes per millisecond.

    Handles: Gbps, Mbps, GBps, MBps
    """
    m = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*(G|M)(b|B)ps", s.strip())
    if not m:
        raise ValueError(f"Cannot parse bandwidth: {s!r}")
    value = float(m.group(1))
    magnitude = m.group(2)
    unit = m.group(3)
    if unit == "b":
        bits_per_sec = value * (1e9 if magnitude == "G" else 1e6)
    else:  # "B" → bytes → bits
        bits_per_sec = value * 8 * (1e9 if magnitude == "G" else 1e6)
    bytes_per_ms = bits_per_sec / 8 / 1000
    return bytes_per_ms


def _parse_latency(s: str) -> float:
    """Parse a latency string to milliseconds.

    Handles: ms, us, ns
    """
    m = re.fullmatch(r"([0-9]*\.?[0-9]+(?:e[+-]?\d+)?)\s*(ms|us|ns)", s.strip())
    if not m:
        raise ValueError(f"Cannot parse latency: {s!r}")
    value = float(m.group(1))
    unit = m.group(2)
    if unit == "ms":
        return value
    elif unit == "us":
        return value / 1000
    else:  # ns
        return value / 1_000_000


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class SimulationResult:
    total_time_ms: float
    per_gpu_times_ms: dict[int, float] = field(default_factory=dict)
    compute_time_ms: dict[int, float] = field(default_factory=dict)
    comm_time_ms: dict[int, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Link helpers
# ---------------------------------------------------------------------------


def _get_link_params(
    src_gpu: int,
    dst_gpu: int,
    datacenter: DatacenterConfig,
) -> tuple[float, float]:
    """Return (bandwidth_bytes_per_ms, latency_ms) for the logical link src_gpu→dst_gpu.

    Bandwidth model (single-hop, no multi-switch routing):
      Intra-node: NVSwitch port speed — bottleneck is the GPU's NVSwitch port.
        Full bisection assumed; no switch fabric congestion modeled.
      Inter-node: NIC speed × bandwidth_efficiency — bottleneck is the NIC.
        Leaf/spine switch uplink contention is NOT modeled (non-blocking fabric assumed).

    Serialization is on per-GPU output ports and input ports independently,
    so concurrent sends from different sources to the same GPU contend on the
    destination's input port, and concurrent sends from one GPU to different
    destinations contend on the source's output port.
    """
    gpus_per_node = datacenter.node.gpus_per_node
    is_intra = (src_gpu // gpus_per_node) == (dst_gpu // gpus_per_node)

    network = datacenter.network

    if is_intra:
        switch_spec: SwitchSpec | None = None
        if network and network.scale_up and network.scale_up.switch:
            sw = network.scale_up.switch
            if isinstance(sw, SwitchSpec):
                switch_spec = sw
        bw = _parse_speed(switch_spec.port_speed) if (switch_spec and switch_spec.port_speed) else _parse_speed("2880Gbps")
        latency_ms = _parse_latency(switch_spec.latency) if (switch_spec and switch_spec.latency) else 0.0
    else:
        nic_spec: NICSpec | None = None
        if network and network.scale_out and network.scale_out.nic:
            nic = network.scale_out.nic
            if isinstance(nic, NICSpec):
                nic_spec = nic
        if nic_spec and nic_spec.speed:
            bw = _parse_speed(nic_spec.speed) * nic_spec.bandwidth_efficiency
        else:
            bw = _parse_speed("400Gbps") * 0.85
        latency_ms = _parse_latency(nic_spec.latency) if (nic_spec and nic_spec.latency) else 0.0

    return bw, latency_ms


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def replay(dag: ExecutionDAG, datacenter: DatacenterConfig) -> SimulationResult:
    """Simulate the DAG and return timing estimates.

    Matches ASTRA-Sim's analytical backend: pure critical-path scheduling.
      - ComputeNode: start after all predecessors finish; duration = node.duration_ms.
      - CommNode: start after all predecessors finish; duration = latency + bytes/bandwidth.
      - No port/link contention modeled. Sequential ordering within a collective
        comes from CommNode.parent_flow_ids, not resource serialization.

    Routing assumption: single logical hop per flow; no intermediate switch
    congestion modeled. See _get_link_params for details.
    """
    # NVSwitch virtual ranks — excluded from per-GPU timing stats
    nvswitch_ranks: set[int] = {n.nvswitch_rank for n in dag.nvswitch_nodes}

    # Build unified node map
    all_nodes: dict[int, ComputeNode | CommNode | NVSwitchNode] = {}
    for n in dag.compute_nodes:
        all_nodes[n.node_id] = n
    for n in dag.comm_nodes:
        all_nodes[n.node_id] = n
    for n in dag.nvswitch_nodes:
        all_nodes[n.node_id] = n

    # flow_id → node_id: CommNode and NVSwitchNode both expose flow_id so that
    # scatter flows can list the NVSwitchNode's flow_id in parent_flow_ids.
    flow_to_node: dict[int, int] = {n.flow_id: n.node_id for n in dag.comm_nodes}
    for n in dag.nvswitch_nodes:
        flow_to_node[n.flow_id] = n.node_id

    # Build predecessors dict and in-degrees
    predecessors: dict[int, set[int]] = {nid: set() for nid in all_nodes}
    in_degree: dict[int, int] = {nid: 0 for nid in all_nodes}

    for edge in dag.edges:
        predecessors[edge.dst_node_id].add(edge.src_node_id)
        in_degree[edge.dst_node_id] += 1

    # Resolve parent_flow_ids for both CommNodes and NVSwitchNodes
    nodes_with_flow_parents: list[CommNode | NVSwitchNode] = list(dag.comm_nodes) + list(dag.nvswitch_nodes)
    for cn in nodes_with_flow_parents:
        for fid in cn.parent_flow_ids:
            if fid in flow_to_node:
                parent_nid = flow_to_node[fid]
                if parent_nid not in predecessors[cn.node_id]:
                    predecessors[cn.node_id].add(parent_nid)
                    in_degree[cn.node_id] += 1

    # Build successors for Kahn's algorithm
    successors: dict[int, list[int]] = defaultdict(list)
    for edge in dag.edges:
        successors[edge.src_node_id].append(edge.dst_node_id)
    for cn in nodes_with_flow_parents:
        for fid in cn.parent_flow_ids:
            if fid in flow_to_node:
                successors[flow_to_node[fid]].append(cn.node_id)

    # Topological sort (Kahn's algorithm)
    temp_in_degree = dict(in_degree)
    queue: deque[int] = deque(nid for nid, deg in temp_in_degree.items() if deg == 0)
    topo_order: list[int] = []
    while queue:
        nid = queue.popleft()
        topo_order.append(nid)
        for succ in successors[nid]:
            temp_in_degree[succ] -= 1
            if temp_in_degree[succ] == 0:
                queue.append(succ)

    # Simulation: walk nodes in topological order
    finish_time: dict[int, float] = {}
    per_gpu_compute: dict[int, float] = defaultdict(float)
    per_gpu_comm: dict[int, float] = defaultdict(float)
    per_gpu_finish: dict[int, float] = defaultdict(float)

    for nid in topo_order:
        node = all_nodes[nid]
        start_time = max((finish_time[p] for p in predecessors[nid]), default=0.0)

        if isinstance(node, ComputeNode):
            duration = node.duration_ms if node.duration_ms is not None else 0.0
            finish = start_time + duration
            finish_time[nid] = finish
            node.start_ms = start_time
            node.finish_ms = finish
            per_gpu_compute[node.gpu_rank] += duration
            if finish > per_gpu_finish[node.gpu_rank]:
                per_gpu_finish[node.gpu_rank] = finish

        elif isinstance(node, NVSwitchNode):
            # NVSwitch reduces at wire speed; its latency is captured by the
            # gather/scatter P2PFlow transfer times. Duration = 0.
            duration = 0.0
            finish = start_time + duration
            finish_time[nid] = finish
            node.duration_ms = duration
            node.start_ms = start_time
            node.finish_ms = finish
            # NVSwitch rank intentionally excluded from per_gpu_finish so it
            # doesn't inflate the GPU count in SimulationResult.

        else:  # CommNode
            bw, latency_ms = _get_link_params(node.src_gpu, node.dst_gpu, datacenter)
            duration = latency_ms + (node.bytes / bw if bw > 0 else 0.0)
            finish = start_time + duration
            finish_time[nid] = finish
            node.duration_ms = duration
            node.start_ms = start_time
            node.finish_ms = finish
            if node.src_gpu not in nvswitch_ranks:
                per_gpu_comm[node.src_gpu] += duration
                if finish > per_gpu_finish[node.src_gpu]:
                    per_gpu_finish[node.src_gpu] = finish
            if node.dst_gpu not in nvswitch_ranks:
                per_gpu_comm[node.dst_gpu] += duration
                if finish > per_gpu_finish[node.dst_gpu]:
                    per_gpu_finish[node.dst_gpu] = finish

    total = max(per_gpu_finish.values(), default=0.0)

    return SimulationResult(
        total_time_ms=total,
        per_gpu_times_ms=dict(per_gpu_finish),
        compute_time_ms=dict(per_gpu_compute),
        comm_time_ms=dict(per_gpu_comm),
    )
