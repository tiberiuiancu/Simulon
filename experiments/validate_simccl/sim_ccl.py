"""Simulate CCL collectives with simulon and write nccl-tests-compatible JSON.

Sweeps AllReduce, AllGather, ReduceScatter over message sizes 8 MB – 8192 MB
for three cluster configs: 1×4 GPUs, 2×4 GPUs, 4×4 GPUs.

Usage (from repo root):
    uv run python experiments/validate_simccl/sim_ccl.py
    uv run python experiments/validate_simccl/sim_ccl.py --cluster jupiter
    uv run python experiments/validate_simccl/sim_ccl.py --output-dir /path/to/results
"""

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from simulon.backend.analytical import simulate as run_simulation
from simulon.config.dc import (
    DatacenterConfig,
    DatacenterMeta,
    NICSpec,
    NodeSpec,
    ScaleOutSpec,
    TopologySpec,
    TopologyType,
)
from simulon.config.scenario import NcclConfig, ScenarioConfig
from simulon.config.workload import CollectiveType, CollectiveWorkload

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------

COLLECTIVES = ["AllReduce", "AllGather", "ReduceScatter", "AllToAll"]

CONFIGS = [
    {"label": "1n4g", "num_nodes": 1, "gpus_per_node": 4},
    {"label": "2n4g", "num_nodes": 2, "gpus_per_node": 4},
    {"label": "4n4g", "num_nodes": 4, "gpus_per_node": 4},
]

# 8 MB → 8192 MB, doubling each step (11 points)
MESSAGE_SIZES_BYTES = [8 * 1024 * 1024 * (2**i) for i in range(11)]

# ---------------------------------------------------------------------------
# Hardware config
# ---------------------------------------------------------------------------

_CLUSTERS = {
    "snellius": {
        "node_template": "snellius-h100-4g",
        # Quad-rail NDR200: 4 × 200 Gbps = 800 Gbps per node
        "nic_speed": "200Gbps",
        "nic_latency": "0.005ms",
        "nics_per_node": 4,
    },
    "jupiter": {
        "node_template": "jupiter-gh200-4g",
        # Quad-rail HDR: 4 × 200 Gbps = 800 Gbps per node
        "nic_speed": "200Gbps",
        "nic_latency": "0.005ms",
        "nics_per_node": 4,
    },
}


def _make_datacenter(num_nodes: int, gpus_per_node: int, cluster: str) -> DatacenterConfig:
    cfg = _CLUSTERS[cluster]
    return DatacenterConfig(
        datacenter=DatacenterMeta(name=f"{num_nodes}n{gpus_per_node}g"),
        num_nodes=num_nodes,
        node=NodeSpec(
            from_=cfg["node_template"],
            nics_per_node=cfg["nics_per_node"],
            scale_out=ScaleOutSpec(
                nic=NICSpec(speed=cfg["nic_speed"], latency=cfg["nic_latency"]),
                topology=TopologySpec(type=TopologyType.fat_tree, params={"k": 4}),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Bus bandwidth
# ---------------------------------------------------------------------------


# Correction factors from nccl-tests PERFORMANCE.md
def _bus_bw(collective: str, alg_bw_GBps: float, n: int) -> float:
    factors = {
        "AllReduce": 2 * (n - 1) / n,
        "AllGather": (n - 1) / n,
        "ReduceScatter": (n - 1) / n,
        "AllToAll": (n - 1) / n,
    }
    if collective not in factors:
        raise ValueError(f"No bus-bw correction factor defined for collective {collective!r}")
    return alg_bw_GBps * factors[collective]


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulate_config(collective: str, num_nodes: int, gpus_per_node: int, cluster: str) -> dict:
    """Run all message-size points for one (collective, cluster) combo."""
    dc = _make_datacenter(num_nodes, gpus_per_node, cluster)
    num_ranks = num_nodes * gpus_per_node
    results = []

    for size in MESSAGE_SIZES_BYTES:
        scenario = ScenarioConfig(
            datacenter=dc,
            workload=CollectiveWorkload(
                framework="collective",
                collective_type=CollectiveType(collective),
                message_size_bytes=size,
            ),
            collective=NcclConfig(
                algorithm="auto",  # calbusbw selects ring and interpolates BW from nccl profile
                num_channels=1,
            ),
        )
        _, result = run_simulation(scenario)

        time_us = result.total_time_ms * 1000
        alg_bw = (size / 1e9) / (result.total_time_ms / 1000)  # GB/s
        bus_bw = _bus_bw(collective, alg_bw, num_ranks)

        results.append(
            {"size": size, "out_of_place": {"time": time_us, "alg_bw": alg_bw, "bus_bw": bus_bw}}
        )

    return {
        "version": 1,
        "config": {
            "collective": collective,
            "num_nodes": num_nodes,
            "gpus_per_node": gpus_per_node,
            "ngpus": num_ranks,
        },
        "results": results,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cluster",
        choices=list(_CLUSTERS),
        default="snellius",
        help="Cluster to simulate (default: snellius)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/validate_simccl/results"),
        help="Directory to write JSON results (default: experiments/validate_simccl/results)",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for cfg in CONFIGS:
        for collective in COLLECTIVES:
            label = cfg["label"]
            print(f"[sim] {collective:15s}  {label} ...", flush=True)  # noqa: T201
            data = simulate_config(collective, cfg["num_nodes"], cfg["gpus_per_node"], args.cluster)
            out = args.output_dir / f"sim_{collective.lower()}_{label}_{args.cluster}.json"
            out.write_text(json.dumps(data, indent=2))
            print(f"      -> {out}")  # noqa: T201


if __name__ == "__main__":
    main()
