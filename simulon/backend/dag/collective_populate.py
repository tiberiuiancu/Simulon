"""Populate CollectiveNode.duration_ms using the analytical collective-level model.

This is the "collective" network-simulation backend: instead of decomposing
collectives into P2P flows (CommNode) and resolving per-flow bandwidth, we
compute a single scalar duration for the whole collective using the same
formula as SimAI's analytical mode.

Inputs
------
- NcclProfile (single-node: direct lookup)
- cal_busbw result (multi-node: theoretical bus bandwidth)
- NIC efficiency table (multi-node: actual / theoretical ratio)

Formula (SimAI analytical)
--------------------------
For ring/tree collectives:
  AllReduce:   duration_ms = size_bytes * 1e-6 / (ratio * busbw) * 2 * (nranks-1)/nranks
  Others:      duration_ms = size_bytes * 1e-6 / (ratio * busbw) * (nranks-1)/nranks

Single-node: use the measured bus_bw_GBps from nccl-tests directly.
"""

from __future__ import annotations

import logging

from simulon.backend.dag._progress import log_progress
from simulon.backend.dag.network_populate import _get_link_params
from simulon.backend.dag.nodes import CollectiveNode, ExecutionDAG
from simulon.collective.calbusbw import _interp_profile, cal_busbw
from simulon.config.dc import DatacenterConfig
from simulon.config.nccl_profile import NcclProfile
from simulon.config.resolve import resolve_nccl_profile, resolve_node_spec

logger = logging.getLogger(__name__)


def _single_node_duration_ms(collective_node: CollectiveNode, nccl_profile: NcclProfile) -> float:
    """Look up the measured duration from nccl-tests profile.

    Uses bus_bw_GBps from the profile and the SimAI formula:
      duration = size_bytes / (busbw * 1e9) * factor
    where factor is 2*(nranks-1)/nranks for AllReduce, (nranks-1)/nranks otherwise.

    The nccl-tests bus_bw already includes algorithmic overhead, so this gives
    the actual measured wall time.
    """
    collective_type = collective_node.collective_type
    nranks = len(collective_node.group_ranks)
    size_bytes = collective_node.data_size
    algorithm = collective_node.algorithm

    # NVLink busbw depends on how many GPUs are in the communicator (a TP=2 group
    # over 2 GPUs reaches only ~1/3 of the 4-GPU busbw). Select the rank-matched
    # sub-profile if one was measured; otherwise fall back to the full-node curve.
    nccl_profile = nccl_profile.for_nranks(nranks)

    algo_measurements = getattr(nccl_profile, collective_type, None)
    if algo_measurements is None:
        raise ValueError(
            f"NcclProfile has no measurements for {collective_type!r}. "
            f"Cannot compute single-node collective duration."
        )

    if algorithm == "ring":
        bus_bw = _interp_profile(size_bytes, algo_measurements.ring)
    elif algorithm == "tree":
        bus_bw = _interp_profile(size_bytes, algo_measurements.tree)
    elif algorithm == "nvls":
        bus_bw = _interp_profile(size_bytes, algo_measurements.nvls)
    elif algorithm == "nvls_tree":
        bus_bw = _interp_profile(size_bytes, algo_measurements.nvls_tree)
    else:
        bus_bw = _interp_profile(size_bytes, algo_measurements.ring)

    if bus_bw is None or bus_bw <= 0:
        raise ValueError(
            f"No valid profile measurement for {collective_type!r} "
            f"algorithm={algorithm!r} size={size_bytes}. "
            f"Cannot compute single-node collective duration."
        )

    # SimAI formula: duration = size / busbw * factor
    # bus_bw is in GB/s
    # We want ms: size / (bus_bw * 1e9) * 1000 = size * 1e-6 / bus_bw
    base = size_bytes * 1e-6 / bus_bw

    if collective_type == "AllReduce":
        return base * 2 * (nranks - 1) / nranks
    else:
        return base * (nranks - 1) / nranks


def _multi_node_duration_ms(
    collective_node: CollectiveNode,
    gpus_per_node: int,
    nic_bw_GBps: float,
    nics_per_node: int,
    nccl_profile: NcclProfile,
) -> float:
    """Compute multi-node collective duration using cal_busbw + NIC efficiency.

    For single-node portion, use the same nccl profile lookup.
    For multi-node, use cal_busbw to get theoretical bus bandwidth, then
    apply NIC efficiency ratio from nic_efficiency_defaults.yaml.
    """
    collective_type = collective_node.collective_type
    nranks = len(collective_node.group_ranks)
    size_bytes = collective_node.data_size
    algorithm = collective_node.algorithm

    # Count unique nodes from actual rank-to-node mapping rather than assuming
    # all gpus_per_node GPUs on each node participate. This matters for DP
    # groups where only 1 GPU per node is in the collective (e.g. TP=4, DP=16).
    node_count = len({r // gpus_per_node for r in collective_node.group_ranks})
    effective_gpus_per_node = nranks // node_count
    # Scale nics proportionally: on quad-rail nodes each GPU has its own NIC,
    # so a collective using k GPUs per node uses k NICs per node.
    effective_nics_per_node = nics_per_node * effective_gpus_per_node / gpus_per_node

    # Prefer a DIRECT measurement when this exact topology was profiled with
    # nccl-tests (e.g. "2n4g"). The measured busbw bakes in NIC bandwidth, rail
    # count and the inter-node fabric, so it reproduces the real collective time
    # via the same SimAI formula — no NIC-efficiency model needed. Fall back to
    # cal_busbw for unmeasured topologies.
    topo_profile = nccl_profile.for_topology(node_count, effective_gpus_per_node)
    if topo_profile is not None:
        algo = getattr(topo_profile, collective_type, None)
        meas_busbw = _interp_profile(size_bytes, algo.ring) if algo is not None else None
        if meas_busbw is not None and meas_busbw > 0:
            base = size_bytes * 1e-6 / meas_busbw
            if collective_type == "AllReduce":
                return base * 2 * (nranks - 1) / nranks
            return base * (nranks - 1) / nranks

    selected_algorithm, intra_bw_GBps, inter_bw_GBps = cal_busbw(
        collective_type=collective_type,
        message_size_bytes=size_bytes,
        num_nodes=node_count,
        gpus_per_node=effective_gpus_per_node,
        nics_per_node=effective_nics_per_node,
        nic_bw_GBps=nic_bw_GBps,
        nccl_profile=nccl_profile,
        algorithm=algorithm,
    )

    # For single-node (node_count == 1), use the intra-node BW directly
    if node_count == 1:
        busbw = intra_bw_GBps
        ratio = 1.0  # No NIC efficiency adjustment for single-node
    else:
        # For multi-node, cal_busbw already returned the effective inter-node BW
        # (nic_bw * nics_per_node * NIC_efficiency).  Do NOT apply the efficiency
        # ratio again — that would double-count it and inflate duration by ~2×.
        busbw = inter_bw_GBps if inter_bw_GBps is not None else intra_bw_GBps
        if busbw is None or busbw <= 0:
            raise ValueError(
                f"cal_busbw returned invalid busbw for {collective_type!r} "
                f"nodes={node_count} gpus_per_node={gpus_per_node}"
            )
        ratio = 1.0

    # SimAI formula: duration = size / busbw * factor
    # busbw is in GB/s
    # We want ms: size / (busbw * 1e9) * 1000 = size * 1e-6 / busbw
    base = size_bytes * 1e-6 / (ratio * busbw)

    if collective_type == "AllReduce":
        return base * 2 * (nranks - 1) / nranks
    else:
        return base * (nranks - 1) / nranks


def populate_collective_network(dag: ExecutionDAG, datacenter: DatacenterConfig) -> ExecutionDAG:
    """Compute duration_ms for every CollectiveNode in the DAG.

    Single-node collectives: use nccl-tests profile directly.
    Multi-node collectives: use cal_busbw + NIC efficiency ratio.

    Mutates nodes in-place and returns the dag.
    """
    nccl_profile = resolve_nccl_profile(datacenter)
    if nccl_profile is None:
        raise ValueError(
            "No NCCL profile available. Cannot populate collective network durations. "
            "Provide a node template with an nccl profile."
        )

    resolved_node = resolve_node_spec(datacenter)
    gpus_per_node = resolved_node.gpus_per_node
    if gpus_per_node is None:
        raise ValueError("node.gpus_per_node must be set after resolution")

    from simulon.backend.dag.network_populate import _ensure_defaults, _parse_latency, _parse_speed

    _ensure_defaults()
    if resolved_node.scale_up and resolved_node.scale_up.switch:
        sw = resolved_node.scale_up.switch
        if isinstance(sw, str):
            raise ValueError(
                f"node.scale_up.switch is a string reference {sw!r} -- string switch templates "
                "not yet supported. Specify inline or use a node template with inline switch spec."
            )
        intra_bw = _parse_speed(sw.port_speed) if sw.port_speed else 0.0
        intra_latency = _parse_latency(sw.latency) if sw.latency else 0.0
    else:
        intra_bw, intra_latency = 0.0, 0.0

    if resolved_node.scale_out and resolved_node.scale_out.nic:
        nic = resolved_node.scale_out.nic
        if isinstance(nic, str):
            raise ValueError(
                f"node.scale_out.nic is a string reference {nic!r} -- string NIC templates "
                "not yet supported. Specify inline or use a node template with inline NIC spec."
            )
        inter_bw = _parse_speed(nic.speed) * nic.bandwidth_efficiency if nic.speed else 0.0
        inter_latency = _parse_latency(nic.latency) if nic.latency else 0.0
    else:
        inter_bw, inter_latency = 0.0, 0.0

    from simulon.backend.analytical import _nic_bw_GBps

    nic_bw_GBps, nics_per_node = _nic_bw_GBps(datacenter)

    launch_latency_ms = nccl_profile.launch_latency_ms

    with log_progress(
        "  populating collective durations", len(dag.collective_nodes), logger
    ) as advance:
        for node in dag.collective_nodes.values():
            node_count = len({r // gpus_per_node for r in node.group_ranks})

            if node_count == 1:
                node.duration_ms = _single_node_duration_ms(node, nccl_profile)
            else:
                node.duration_ms = _multi_node_duration_ms(
                    node, gpus_per_node, nic_bw_GBps, nics_per_node, nccl_profile
                )
            node.duration_ms += launch_latency_ms
            advance()

    pp_sends = [n for n in dag.comm_nodes if n.collective_type == "PP_Send"]
    with log_progress("  populating PP_Send durations", len(pp_sends), logger) as advance:
        for comm_node in pp_sends:
            bw, latency_ms = _get_link_params(
                comm_node.src_gpu,
                comm_node.dst_gpu,
                gpus_per_node,
                intra_bw,
                intra_latency,
                inter_bw,
                inter_latency,
            )
            comm_node.duration_ms = latency_ms + (comm_node.bytes / bw if bw > 0 else 0.0)
            advance()

    return dag
