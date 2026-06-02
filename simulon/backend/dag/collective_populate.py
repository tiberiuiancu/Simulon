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
    # bus_bw is in GB/s = 1e9 bytes/s
    # We want ms: size / (bus_bw * 1e9) * 1000 = size * 1e-6 / bus_bw
    base = size_bytes * 1e-6 / bus_bw

    if collective_type == "AllReduce":
        return base * 2 * (nranks - 1) / nranks
    else:
        return base * (nranks - 1) / nranks


def _multi_node_duration_ms(
    collective_node: CollectiveNode, dc: DatacenterConfig, nccl_profile: NcclProfile
) -> float:
    """Compute multi-node collective duration using cal_busbw + NIC efficiency.

    For single-node portion, use the same nccl profile lookup.
    For multi-node, use cal_busbw to get theoretical bus bandwidth, then
    apply NIC efficiency ratio from nic_efficiency_defaults.yaml.
    """
    from simulon.backend.analytical import _nic_bw_GBps
    from simulon.collective.calbusbw import _nic_efficiency

    collective_type = collective_node.collective_type
    nranks = len(collective_node.group_ranks)
    size_bytes = collective_node.data_size
    algorithm = collective_node.algorithm

    resolved_node = resolve_node_spec(dc)
    gpus_per_node = resolved_node.gpus_per_node
    if gpus_per_node is None:
        raise ValueError("node.gpus_per_node must be set after resolution")

    node_count = nranks // gpus_per_node

    # Get NIC bandwidth
    nic_bw, nics_per_node = _nic_bw_GBps(dc)

    # Call cal_busbw to get the effective bus bandwidth
    selected_algorithm, intra_bw_GBps, inter_bw_GBps = cal_busbw(
        collective_type=collective_type,
        message_size_bytes=size_bytes,
        num_nodes=node_count,
        gpus_per_node=gpus_per_node,
        nics_per_node=nics_per_node,
        nic_bw_GBps=nic_bw,
        nccl_profile=nccl_profile,
        algorithm=algorithm,
    )

    # For single-node (node_count == 1), use the intra-node BW directly
    if node_count == 1:
        busbw = intra_bw_GBps
        ratio = 1.0  # No NIC efficiency adjustment for single-node
    else:
        # For multi-node, the bottleneck is the inter-node BW
        # We apply NIC efficiency ratio
        busbw = inter_bw_GBps if inter_bw_GBps is not None else intra_bw_GBps
        if busbw is None or busbw <= 0:
            raise ValueError(
                f"cal_busbw returned invalid busbw for {collective_type!r} "
                f"nodes={node_count} gpus_per_node={gpus_per_node}"
            )
        ratio = _nic_efficiency(size_bytes, node_count)

    # SimAI formula
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

    for node in dag.collective_nodes.values():
        nranks = len(node.group_ranks)
        resolved_node = resolve_node_spec(datacenter)
        gpus_per_node = resolved_node.gpus_per_node
        if gpus_per_node is None:
            raise ValueError("node.gpus_per_node must be set after resolution")

        node_count = nranks // gpus_per_node

        if node_count == 1:
            node.duration_ms = _single_node_duration_ms(node, nccl_profile)
        else:
            node.duration_ms = _multi_node_duration_ms(node, datacenter, nccl_profile)

    return dag
