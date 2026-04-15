#!/usr/bin/env python3
"""Simulate CCL collectives with simulon and write nccl-tests-compatible JSON.

Sweeps AllReduce, AllGather, ReduceScatter over message sizes 8 MB – 8192 MB
for three cluster configs: 1×4 GPUs, 2×4 GPUs, 4×4 GPUs (H100 + NVSwitch 4 +
InfiniBand HDR100).

Usage (from repo root):
    uv run python experiments/validation/simccl/sim_ccl.py
    uv run python experiments/validation/simccl/sim_ccl.py --output-dir /path/to/results
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from simulon.backend.analytical import AnalyticalBackend
from simulon.config.dc import (
    ClusterSpec,
    DatacenterConfig,
    DatacenterMeta,
    GPUSpec,
    NICSpec,
    NetworkSpec,
    NodeSpec,
    ScaleOutSpec,
    ScaleUpSpec,
    SwitchSpec,
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

# H100 NVLink 4: theoretical 450 GB/s per direction, but ring AllReduce on Snellius 1n4g
# achieves ~319 GB/s effective bandwidth at saturation.  NVSwitch wire latency is ~25 ns;
# the dominant per-step overhead (~5.95 µs) comes from NCCL kernel launch + sync and is
# captured in per_step_latency_us below (calibrated via linear fit: time = nsteps*lat + nsteps*S/(N*bw)).
_NVSWITCH4_PORT_SPEED = "2554Gbps"  # 319.2 GB/s effective (vs 450 GB/s theoretical)
_NVSWITCH4_LATENCY = (
    "0.000025ms"  # 25 ns wire latency (NCCL overhead goes in per_step_latency_us)
)

# InfiniBand HDR100: 100 Gbps per direction per port (× 0.85 efficiency applied by simulon).
# Snellius nodes have 1 IB port per node (not per GPU).  For ring collectives this is
# correct: only one GPU per node uses the inter-node link at a time, so each inter-node
# P2P flow gets the full 100 Gbps link.  Would need adjustment for algorithms where
# multiple GPUs communicate inter-node simultaneously (e.g. AllToAll).
_IB_HDR100_SPEED = "100Gbps"
_IB_HDR100_LATENCY = "0.005ms"  # 5 µs


def _make_datacenter(num_nodes: int, gpus_per_node: int) -> DatacenterConfig:
    return DatacenterConfig(
        datacenter=DatacenterMeta(name=f"{num_nodes}n{gpus_per_node}g"),
        cluster=ClusterSpec(num_nodes=num_nodes),
        node=NodeSpec(
            gpus_per_node=gpus_per_node,
            gpu=GPUSpec(name="H100", memory_capacity_gb=80.0),
        ),
        network=NetworkSpec(
            scale_up=ScaleUpSpec(
                switch=SwitchSpec(
                    port_speed=_NVSWITCH4_PORT_SPEED,
                    latency=_NVSWITCH4_LATENCY,
                ),
            ),
            scale_out=ScaleOutSpec(
                nic=NICSpec(speed=_IB_HDR100_SPEED, latency=_IB_HDR100_LATENCY),
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
        raise ValueError(
            f"No bus-bw correction factor defined for collective {collective!r}"
        )
    return alg_bw_GBps * factors[collective]


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulate_config(collective: str, num_nodes: int, gpus_per_node: int) -> dict:
    """Run all message-size points for one (collective, cluster) combo."""
    dc = _make_datacenter(num_nodes, gpus_per_node)
    num_ranks = num_nodes * gpus_per_node
    backend = AnalyticalBackend()
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
                algorithm="ring",
                num_channels=1,
                per_step_latency_us={
                    "AllReduce":      4.07,
                    "AllGather":      9.09,
                    "ReduceScatter":  8.24,
                    "AllToAll":       8.37,
                },
                per_collective_bw_GBps={
                    "AllReduce":    316.9,
                    "AllGather":    305.3,
                    "ReduceScatter": 297.5,
                    "AllToAll":     271.3,
                },
            ),
        )
        _, result = backend.simulate(scenario)

        time_us = result.total_time_ms * 1000
        alg_bw = (size / 1e9) / (result.total_time_ms / 1000)  # GB/s
        bus_bw = _bus_bw(collective, alg_bw, num_ranks)

        results.append(
            {
                "size": size,
                "out_of_place": {
                    "time": time_us,
                    "alg_bw": alg_bw,
                    "bus_bw": bus_bw,
                },
            }
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
        "--output-dir",
        type=Path,
        default=Path("experiments/validation/simccl/results"),
        help="Directory to write JSON results (default: experiments/validation/simccl/results)",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for cfg in CONFIGS:
        for collective in COLLECTIVES:
            label = cfg["label"]
            print(f"[sim] {collective:15s}  {label} ...", flush=True)
            data = simulate_config(collective, cfg["num_nodes"], cfg["gpus_per_node"])
            out = args.output_dir / f"sim_{collective.lower()}_{label}.json"
            out.write_text(json.dumps(data, indent=2))
            print(f"      -> {out}")


if __name__ == "__main__":
    main()
