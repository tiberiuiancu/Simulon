#!/usr/bin/env python3
"""Convert nccl-tests JSON results into a Simulon node-template ``nccl:`` block.

When moving Simulon to a new cluster, the network model is calibrated from
nccl-tests bus-bandwidth curves. nccl-tests writes one JSON file per collective
(``-J`` flag); this script reads them and emits the ``nccl:`` YAML block that goes
into ``templates/node/<cluster>.yaml`` — replacing what was previously a manual,
error-prone transcription of bandwidth tables.

Usage (from repo root, after running experiments/validate_simccl/run_nccl_*.sh):

    python scripts/nccl_to_node_template.py \
        --results-dir experiments/validate_simccl/results \
        --config 1n4g --cluster mycluster --gpus-per-node 4

Paste the printed block under your node template. ``launch_latency_ms`` is left as
a placeholder — see the note printed at the end for how to derive it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# nccl-tests binary names map to Simulon collective keys.
_COLLECTIVES = {
    "allreduce": "AllReduce",
    "allgather": "AllGather",
    "reducescatter": "ReduceScatter",
    "alltoall": "AllToAll",
}


def _load_curve(path: Path) -> list[tuple[int, float]]:
    """Return [(size_bytes, bus_bw_GBps), ...] from one nccl-tests JSON file."""
    data = json.loads(path.read_text())
    out = []
    for r in data.get("results", []):
        size = int(r["size"])
        # out_of_place matches how Megatron issues collectives (fresh buffers).
        bw = float(r["out_of_place"]["bus_bw"])
        out.append((size, bw))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, required=True,
                    help="Directory with nccl_<collective>_<config>_<cluster>.json files")
    ap.add_argument("--config", required=True,
                    help="Config label in the filenames, e.g. 1n4g")
    ap.add_argument("--cluster", required=True,
                    help="Cluster suffix in the filenames, e.g. jupiter (or '' if none)")
    ap.add_argument("--gpus-per-node", type=int, default=4)
    ap.add_argument("--algorithm", default="ring",
                    help="NCCL algorithm label for the curve (default: ring)")
    args = ap.parse_args()

    suffix = f"_{args.cluster}" if args.cluster else ""
    lines = [
        "nccl:",
        f"  name: {args.cluster or 'cluster'}-{args.gpus_per_node}g",
        f"  gpus_per_node: {args.gpus_per_node}",
        "  # launch_latency_ms: <DERIVE — see note below>",
    ]

    found = 0
    for fname, key in _COLLECTIVES.items():
        path = args.results_dir / f"nccl_{fname}_{args.config}{suffix}.json"
        if not path.is_file():
            print(f"# WARNING: missing {path} — skipping {key}")
            continue
        curve = _load_curve(path)
        if not curve:
            print(f"# WARNING: no results in {path}")
            continue
        found += 1
        lines.append(f"  {key}:")
        lines.append(f"    {args.algorithm}:")
        for size, bw in curve:
            lines.append(f"      - {{size_bytes: {size:>13}, bus_bw_GBps: {bw:9.3f}}}")

    print("\n".join(lines))
    print()
    print(f"# Parsed {found}/{len(_COLLECTIVES)} collectives from {args.results_dir}")
    print("# NOTE on launch_latency_ms: per-call overhead beyond tight-loop bus bw.")
    print("#   Derive from a real run as: (wall gap with N collectives − ideal-bw time) / N,")
    print("#   or measure (SP=true gap − SP=false gap)/(extra collective count). See the")
    print("#   jupiter-gh200-4g.yaml comment for a worked example (~0.111 ms there).")


if __name__ == "__main__":
    main()
