"""Python port of SimAI's calbusbw.cc bandwidth calculation logic.

Derives effective bus bandwidth (and selects the best algorithm) for a
collective operation given hardware topology and optional nccl-tests measurements.

Key differences from SimAI:
- Instead of hardcoded GPU-type constants, uses NcclProfile for intra-node BW.
- For inter-node, uses nic_bw_GBps from datacenter spec scaled by NIC efficiency
  from the bundled nic_efficiency_defaults.yaml table.
- Returns separate intra and inter-node BWs for populate_network.
- Fails loudly if no profile is available.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import yaml

from simulon.config.nccl_profile import NcclAlgoMeasurements, NcclProfile

# ---------------------------------------------------------------------------
# NIC efficiency table (lazy-loaded)
# ---------------------------------------------------------------------------

_NIC_EFF_CACHE: dict[int, list[tuple[int, float]]] | None = None


def _load_nic_efficiency() -> dict[int, list[tuple[int, float]]]:
    """Load the NIC efficiency table from the bundled YAML file.

    Returns a dict mapping node_count → sorted list of (size_bytes, efficiency).
    """
    global _NIC_EFF_CACHE
    if _NIC_EFF_CACHE is not None:
        return _NIC_EFF_CACHE

    data_path = Path(__file__).parent.parent / "data" / "nic_efficiency_defaults.yaml"
    with open(data_path) as f:
        raw = yaml.safe_load(f)

    table: dict[int, list[tuple[int, float]]] = {}
    for key, rows in raw.items():
        if not isinstance(rows, list):
            continue
        n_nodes = int(key.split("_")[0])
        table[n_nodes] = sorted((r["size_bytes"], r["efficiency"]) for r in rows)

    _NIC_EFF_CACHE = table
    return table


# ---------------------------------------------------------------------------
# AllToAll ratio table (lazy-loaded)
# ---------------------------------------------------------------------------

_ALLTOALL_RATIO_CACHE: dict[int, list[tuple[int, float]]] | None = None


def _load_alltoall_ratio() -> dict[int, list[tuple[int, float]]]:
    """Load the AllToAll ratio table from the bundled CSV file.

    The CSV maps node_count → sorted list of (size_bytes, ratio).
    """
    global _ALLTOALL_RATIO_CACHE
    if _ALLTOALL_RATIO_CACHE is not None:
        return _ALLTOALL_RATIO_CACHE

    data_path = Path(__file__).parent.parent / "data" / "alltoall_ratio_defaults.csv"
    with open(data_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        col_to_nodes: dict[int, int] = {}
        for i, h in enumerate(header):
            if "Node" in h:
                nodes_str = h.split("Node")[0]
                if nodes_str.isdigit():
                    col_to_nodes[i] = int(nodes_str)

        table: dict[int, list[tuple[int, float]]] = {n: [] for n in col_to_nodes.values()}
        for row in reader:
            size = int(row[0])
            for col, nodes in col_to_nodes.items():
                val = row[col].strip()
                if val:
                    table[nodes].append((size, float(val)))

    table = {n: sorted(v) for n, v in table.items() if v}
    _ALLTOALL_RATIO_CACHE = table
    return table


def _nic_efficiency(message_size_bytes: int, num_nodes: int) -> float:
    """Interpolate NIC efficiency for the given message size and node count.

    Uses log-linear interpolation on message size between the two nearest
    table entries. Interpolates linearly between adjacent node-count curves.
    Node counts > 128 clamp to the 128-node curve.
    """
    table = _load_nic_efficiency()
    available_nodes = sorted(table.keys())

    if num_nodes <= available_nodes[0]:
        return _interp_size(message_size_bytes, table[available_nodes[0]])
    if num_nodes >= available_nodes[-1]:
        return _interp_size(message_size_bytes, table[available_nodes[-1]])

    # Linear interpolation between the two bracketing node counts
    lo = max(n for n in available_nodes if n <= num_nodes)
    hi = min(n for n in available_nodes if n >= num_nodes)
    if lo == hi:
        return _interp_size(message_size_bytes, table[lo])

    eff_lo = _interp_size(message_size_bytes, table[lo])
    eff_hi = _interp_size(message_size_bytes, table[hi])
    t = (num_nodes - lo) / (hi - lo)
    return eff_lo + t * (eff_hi - eff_lo)


def _interp_size(size: int, curve: list[tuple[int, float]]) -> float:
    """Log-linear interpolation of efficiency by message size."""
    if not curve:
        return 1.0
    if size <= curve[0][0]:
        return curve[0][1]
    if size >= curve[-1][0]:
        return curve[-1][1]

    for i in range(len(curve) - 1):
        s0, e0 = curve[i]
        s1, e1 = curve[i + 1]
        if s0 <= size <= s1:
            # Log-linear interpolation
            log_t = math.log(size / s0) / math.log(s1 / s0)
            return e0 + log_t * (e1 - e0)
    return curve[-1][1]


def _alltoall_ratio(message_size_bytes: int, num_nodes: int) -> float:
    """AllToAll ratio with SimAI-style normalization (divide by max)."""
    table = _load_alltoall_ratio()
    available_nodes = sorted(table.keys())

    if not available_nodes:
        return 1.0
    if num_nodes <= available_nodes[0]:
        lo_node = available_nodes[0]
        lo_val = _interp_size(message_size_bytes, table[lo_node])
        mx_val = max(r for _, r in table[lo_node])
        return lo_val / mx_val
    if num_nodes >= available_nodes[-1]:
        hi_node = available_nodes[-1]
        hi_val = _interp_size(message_size_bytes, table[hi_node])
        mx_val = max(r for _, r in table[hi_node])
        return hi_val / mx_val

    lo = max(n for n in available_nodes if n <= num_nodes)
    hi = min(n for n in available_nodes if n >= num_nodes)
    if lo == hi:
        return _interp_size(message_size_bytes, table[lo]) / max(r for _, r in table[lo])

    eff_lo = _interp_size(message_size_bytes, table[lo]) / max(r for _, r in table[lo])
    eff_hi = _interp_size(message_size_bytes, table[hi]) / max(r for _, r in table[hi])
    t = (num_nodes - lo) / (hi - lo)
    return eff_lo + t * (eff_hi - eff_lo)


# ---------------------------------------------------------------------------
# Profile interpolation
# ---------------------------------------------------------------------------


def _interp_profile(size: int, points: list) -> float | None:
    """Interpolate bus_bw_GBps from a list of NcclDataPoint for the given size.

    Uses log-linear interpolation. Returns None if the list is empty.
    """
    if not points:
        return None

    sorted_pts = sorted(points, key=lambda p: p.size_bytes)
    if size <= sorted_pts[0].size_bytes:
        return sorted_pts[0].bus_bw_GBps
    if size >= sorted_pts[-1].size_bytes:
        return sorted_pts[-1].bus_bw_GBps

    for i in range(len(sorted_pts) - 1):
        p0 = sorted_pts[i]
        p1 = sorted_pts[i + 1]
        if p0.size_bytes <= size <= p1.size_bytes:
            log_t = math.log(size / p0.size_bytes) / math.log(p1.size_bytes / p0.size_bytes)
            return p0.bus_bw_GBps + log_t * (p1.bus_bw_GBps - p0.bus_bw_GBps)
    return sorted_pts[-1].bus_bw_GBps


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def cal_busbw(
    collective_type: str,
    message_size_bytes: int,
    num_nodes: int,
    gpus_per_node: int,
    nics_per_node: float,
    nic_bw_GBps: float,
    nccl_profile: NcclProfile | None,
    algorithm: str = "auto",
) -> tuple[str, float, float | None]:
    """Derive effective bus bandwidth for a collective operation.

    Parameters
    ----------
    collective_type:
        One of "AllReduce", "AllGather", "ReduceScatter", "AllToAll".
    message_size_bytes:
        Total message size in bytes (pre-collective).
    num_nodes:
        Number of nodes in the collective group.
    gpus_per_node:
        GPUs per node.
    nics_per_node:
        Number of NICs per node (typically 1.0 for single-port setups).
    nic_bw_GBps:
        Raw NIC bandwidth per port in GB/s (from datacenter spec).
    nccl_profile:
        Measured nccl-tests profile. Required.
    algorithm:
        "auto" | "ring" | "tree" | "nvls" | "nvls_tree"

    Returns
    -------
    (selected_algorithm, intra_bw_GBps, inter_bw_GBps)
        intra_bw_GBps: effective BW for intra-node P2P flows (NVLink).
        inter_bw_GBps: effective BW for inter-node P2P flows (NIC), or None
                       for single-node collectives.

    Raises
    ------
    ValueError
        If no profile is available and the BW cannot be derived.
    """
    if nccl_profile is None:
        raise ValueError(
            f"No NCCL profile loaded for this GPU. Cannot derive effective bandwidth "
            f"for {collective_type}. Either provide a <gpu>.nccl.yaml profile or "
            f"attach an nccl profile to your node template."
        )

    algo_measurements: NcclAlgoMeasurements = getattr(nccl_profile, collective_type, None)
    if algo_measurements is None:
        raise ValueError(
            f"NcclProfile has no measurements for collective type {collective_type!r}. "
            f"Supported: AllReduce, AllGather, ReduceScatter, AllToAll."
        )

    # -----------------------------------------------------------------------
    # Intra-node BW: from profile (1-node measurement, NVLink)
    # -----------------------------------------------------------------------
    intra_bw_GBps: float | None = None

    if algorithm in ("auto", "ring"):
        intra_bw_GBps = _interp_profile(message_size_bytes, algo_measurements.ring)
    elif algorithm == "tree":
        intra_bw_GBps = _interp_profile(message_size_bytes, algo_measurements.tree)
    elif algorithm == "nvls":
        intra_bw_GBps = _interp_profile(message_size_bytes, algo_measurements.nvls)
    elif algorithm == "nvls_tree":
        # nvls_tree falls back to plain nvls topology on single-node; use the nvls
        # BW curve in that case so the bandwidth matches the actual flow model used.
        if num_nodes == 1:
            intra_bw_GBps = _interp_profile(message_size_bytes, algo_measurements.nvls)
        else:
            intra_bw_GBps = _interp_profile(message_size_bytes, algo_measurements.nvls_tree)

    # -----------------------------------------------------------------------
    # Algorithm auto-selection (AllReduce only; mirrors SimAI)
    # -----------------------------------------------------------------------
    selected_algorithm = algorithm

    if algorithm == "auto":
        if collective_type == "AllReduce":
            ring_bw = _interp_profile(message_size_bytes, algo_measurements.ring) or 0.0
            tree_bw = _interp_profile(message_size_bytes, algo_measurements.tree) or 0.0
            nvls_bw = _interp_profile(message_size_bytes, algo_measurements.nvls) or 0.0
            nvls_tree_bw = _interp_profile(message_size_bytes, algo_measurements.nvls_tree) or 0.0

            candidates: dict[str, float] = {"ring": ring_bw}
            if tree_bw > 0:
                candidates["tree"] = tree_bw
            # Only consider NVLS if the profile has measurements for it
            if nvls_bw > 0 and num_nodes == 1:
                candidates["nvls"] = nvls_bw
            if nvls_tree_bw > 0 and num_nodes > 1:
                candidates["nvls_tree"] = nvls_tree_bw

            selected_algorithm = max(candidates, key=lambda k: candidates[k])
            intra_bw_GBps = candidates[selected_algorithm]
        else:
            # Non-AllReduce: always ring
            selected_algorithm = "ring"
            intra_bw_GBps = _interp_profile(message_size_bytes, algo_measurements.ring)

    if intra_bw_GBps is None:
        raise ValueError(
            f"No profile measurements found for algorithm={algorithm!r} and "
            f"collective={collective_type!r}. Add measurements to your .nccl.yaml file."
        )

    # -----------------------------------------------------------------------
    # Inter-node BW: collective-specific formulas
    # -----------------------------------------------------------------------
    inter_bw_GBps: float | None = None
    if num_nodes > 1:
        if collective_type == "AllToAll":
            nranks = num_nodes * gpus_per_node
            base_bw = (
                nic_bw_GBps
                * nics_per_node
                / gpus_per_node
                * (nranks - 1)
                / ((num_nodes - 1) * gpus_per_node)
            )
            ratio = _alltoall_ratio(message_size_bytes, num_nodes)
            inter_bw_GBps = base_bw * ratio
        else:
            eff = _nic_efficiency(message_size_bytes, num_nodes)
            inter_bw_GBps = nic_bw_GBps * nics_per_node * eff

    return selected_algorithm, intra_bw_GBps, inter_bw_GBps
