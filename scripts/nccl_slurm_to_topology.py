#!/usr/bin/env python3
"""Turn large-scale nccl-tests Slurm outputs into `nccl.by_topology` entries for a node template.

The 64-1024-node AllReduce sweeps in ~/nccl-tests (jobs 358773-358779, 376608, 409071; April
2026, NCCL 2.29.7, 4 GPUs/node, 4 B .. 8 GB) are the only measurements Simulon has of the
inter-node fabric at the scale the 256-1024-node search runs at. This script reads the plain
`*_perf` stdout tables (no `-J` JSON was written) and emits YAML for `templates/node/*.yaml`.

Two derived families are emitted alongside the measured AllReduce curves, each marked in
the YAML with its provenance so nothing derived can pass for measured:

  * ReduceScatter / AllGather at Nn4g = AllReduce(Nn4g) x a per-collective ratio measured on
    the closest profiled topology (default: the 16n1g / 4n4g ring curves in the template,
    where RS/AG busbw is 0.5-0.6x AllReduce at >= 1 GB). Pass --rs-ratio/--ag-ratio to
    override with a directly measured value.
  * Nn1g (one GPU per node -- the DP-group shape of every TP=4 run) = AllReduce(Nn4g)
    scaled by the ONE measured 1g/4g pair, job 409071 at 64 nodes / 32 GB: 24.55 vs 80.13
    GB/s (0.306). A single GPU cannot drive its node's four NICs, so the 1g communicator is
    NIC-bound at ~25 GB/s; the 4g curve carries the fabric's scale dependence.

Usage:
    python scripts/nccl_slurm_to_topology.py ~/nccl-tests/slurm-358778.out:64 \
        ~/nccl-tests/slurm-358777.out:128 ... --min-bytes 8388608 >> templates/node/jupiter-gh200-4g.yaml
(append under `nccl.by_topology:`; the output is indented for that position)
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

_ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+(float|half|bfloat16|int8|double)\s+\S+\s+\S+\s+"
                  r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\S+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")


def read_curve(path: Path, min_bytes: int) -> list[tuple[int, float]]:
    """[(size_bytes, out-of-place busbw GB/s)] from a *_perf stdout table."""
    out = []
    for line in path.read_text(errors="replace").splitlines():
        m = _ROW.match(line)
        if m and int(m.group(1)) >= min_bytes:
            out.append((int(m.group(1)), float(m.group(6))))
    return out


def emit(nodes: int, gpn: int, curves: dict[str, list[tuple[int, float]]], note: str, indent: str = "    ") -> str:
    lines = [f"{indent}{nodes}n{gpn}g:", f"{indent}  gpus_per_node: {gpn}"]
    for note_line in note.splitlines():
        lines.append(f"{indent}  # {note_line}")
    for coll, curve in curves.items():
        lines.append(f"{indent}  {coll}:")
        lines.append(f"{indent}    ring:")
        for size, bw in curve:
            lines.append(f"{indent}      - {{size_bytes: {size:>13d}, bus_bw_GBps: {bw:8.3f}}}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="<slurm-out>:<nodes> pairs (AllReduce sweeps, 4 GPUs/node)")
    ap.add_argument("--gpus-per-node", type=int, default=4)
    ap.add_argument("--min-bytes", type=int, default=8388608)
    ap.add_argument("--rs-ratio", type=float, default=0.55, help="ReduceScatter/AllReduce busbw ratio")
    ap.add_argument("--ag-ratio", type=float, default=0.55, help="AllGather/AllReduce busbw ratio")
    ap.add_argument("--one-gpu-ratio", type=float, default=24.55 / 80.13,
                    help="Nn1g/Nn4g AllReduce busbw ratio (job 409071, 64 nodes, 32 GB)")
    ap.add_argument("--no-1g", action="store_true", help="do not emit the derived Nn1g entries")
    ap.add_argument("--source-note", default="NCCL 2.29.7+cuda13.0, Stages/2026, NCCL_IB_HCA=mlx5, April 2026")
    a = ap.parse_args()

    blocks = []
    for spec in a.runs:
        path, nodes = spec.rsplit(":", 1)
        nodes = int(nodes)
        ar = read_curve(Path(path), a.min_bytes)
        if not ar:
            raise SystemExit(f"no table rows in {path}")
        job = re.search(r"(\d{5,})", Path(path).name)
        job = job.group(1) if job else Path(path).name
        rs = [(s, round(b * a.rs_ratio, 3)) for s, b in ar]
        ag = [(s, round(b * a.ag_ratio, 3)) for s, b in ar]
        note4 = (f"MEASURED AllReduce: nccl-tests job {job}, {nodes} nodes x {a.gpus_per_node} GPUs ({a.source_note}).\n"
                 f"DERIVED ReduceScatter/AllGather = AllReduce x {a.rs_ratio:.2f}/{a.ag_ratio:.2f} "
                 f"(ratio measured on the profiled 16n1g/4n4g curves at >= 1 GB).")
        blocks.append(emit(nodes, a.gpus_per_node, {"AllReduce": ar, "ReduceScatter": rs, "AllGather": ag}, note4))
        if not a.no_1g:
            r = a.one_gpu_ratio
            ar1 = [(s, round(b * r, 3)) for s, b in ar]
            note1 = (f"DERIVED, not measured: AllReduce({nodes}n4g, job {job}) x {r:.3f}, the single measured 1g/4g pair\n"
                     f"(job 409071: 64 nodes, 32 GB, 24.55 vs 80.13 GB/s). One GPU per node is NIC-bound (~25 GB/s);\n"
                     f"the 4g curve supplies the fabric's scale dependence. RS/AG further x {a.rs_ratio:.2f}/{a.ag_ratio:.2f}.\n"
                     f"Replace with a measured {nodes}n1g sweep when one exists.")
            blocks.append(emit(nodes, 1, {"AllReduce": ar1,
                                          "ReduceScatter": [(s, round(b * a.rs_ratio, 3)) for s, b in ar1],
                                          "AllGather": [(s, round(b * a.ag_ratio, 3)) for s, b in ar1]}, note1))
    print("\n".join(blocks))


if __name__ == "__main__":
    main()
