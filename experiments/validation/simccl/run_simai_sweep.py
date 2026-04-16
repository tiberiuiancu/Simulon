#!/usr/bin/env python3
"""Run SimAI collective sweep in analytical and/or NS3 mode.

Sweeps AllReduce, AllGather, ReduceScatter, AllToAll over message sizes
8 MB – 8192 MB for cluster configs: 1×4, 2×4, 4×4 GPUs
(H100 NVSwitch4 + IB HDR100, matching sim_ccl.py).

Outputs one JSON file per (mode, collective, config) in nccl-tests-compatible
format alongside the existing sim_ccl.py / nccl-tests results, so they can
all be fed into the same plot.py.

Usage (from repo root):
    uv run python experiments/validation/simccl/run_simai_sweep.py
    uv run python experiments/validation/simccl/run_simai_sweep.py --mode ns3
    uv run python experiments/validation/simccl/run_simai_sweep.py --mode both

Requires SimAI binaries (build from ~/uni/t/simai-original):
    Analytical:  cd ~/uni/t/simai-original && ./scripts/build.sh -c analytical
    NS3:         cd ~/uni/t/simai-original && ./scripts/build.sh -c ns3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sweep parameters (matching sim_ccl.py)
# ---------------------------------------------------------------------------

COLLECTIVES = ["AllReduce", "AllGather", "ReduceScatter", "AllToAll"]

# SimAI workload token for each collective
_SIMAI_COLL = {
    "AllReduce":     "ALLREDUCE",
    "AllGather":     "ALLGATHER",
    "ReduceScatter": "REDUCESCATTER",
    "AllToAll":      "ALLTOALL",
}

CONFIGS = [
    {"label": "1n4g", "num_nodes": 1, "gpus_per_node": 4},
    {"label": "2n4g", "num_nodes": 2, "gpus_per_node": 4},
    {"label": "4n4g", "num_nodes": 4, "gpus_per_node": 4},
]

# 8 MB → 8192 MB, doubling each step (11 points) — same as sim_ccl.py
MESSAGE_SIZES_BYTES = [8 * 1024 * 1024 * (2**i) for i in range(11)]


# ---------------------------------------------------------------------------
# Hardware parameters (matching Snellius H100 NVSwitch4 + IB HDR100 setup)
# ---------------------------------------------------------------------------

# H100 SXM5 theoretical NVLink 4 peak: 20.6 GB/s × 18 links = 370.8 GB/s
# calbusbw then multiplies by nvlink_ratio.csv to get effective BW.
# Passing the theoretical value is correct; the ratio CSV corrects it.
_NV_BW_GBps = 370.8

# IB HDR100: 100 Gbps = 12.5 GB/s per port
_NIC_BW_GBps = 12.5
_NICS_PER_NODE = 1
_GPU_TYPE = "H100"

# NS3 topology link speeds (what gets written into the topo file)
_NS3_NVLINK_BW = "2554Gbps"   # 319.25 GB/s × 8 = 2554 Gbps, effective rate
_NS3_NIC_BW    = "100Gbps"    # IB HDR100
_NS3_NIC_LAT   = "0.005ms"    # 5 µs


# ---------------------------------------------------------------------------
# Workload file generation
# ---------------------------------------------------------------------------

def _make_workload(collective: str, num_gpus: int) -> str:
    """Return a HYBRID_TRANSFORMER workload with one layer per message size.

    All compute is 0; only the weight-grad collective varies.
    The CSV output column ``wg total comm`` gives the per-layer collective
    time in µs, with algbw and busbw in the adjacent columns.

    Column order (space-separated):
        name  dep  fwd_comp  fwd_comm  fwd_size
              ig_comp  ig_comm  ig_size
              wg_comp  wg_comm  wg_size  repeat

    The collective goes in the FWD column (TP group = model_parallel_NPU_group = all GPUs).
    Putting it in the WG column would target the DP group, which is size 1 when TP = all_gpus,
    so zero streams would be injected.  Compute is set to 1 tick (negligible ~1 ns).
    """
    coll = _SIMAI_COLL[collective]
    header = (
        f"HYBRID_TRANSFORMER_FWD_IN_BCKWD "
        f"model_parallel_NPU_group: {num_gpus} "
        f"ep: 1 pp: 1 vpp: 1 ga: 1 all_gpus: {num_gpus} "
        f"checkpoints: 0 checkpoint_initiates: 0"
    )
    lines = [header, str(len(MESSAGE_SIZES_BYTES))]
    for i, size in enumerate(MESSAGE_SIZES_BYTES):
        # fwd_comp=1 fwd_comm=COLL fwd_size=SIZE  ig_comp=1 ig_comm=NONE ig_size=0
        # wg_comp=1 wg_comm=NONE wg_size=0  repeat=1
        lines.append(
            f"layer_{i:03d} -1 1 {coll} {size} 1 NONE 0 1 NONE 0 1"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------
# EndToEnd.csv columns (0-indexed after splitting on ','):
#   0  layer_name
#   1  run_name
#   2  fwd compute (µs)
#   3  wg compute  (µs)
#   4  ig compute  (µs)
#   5  fwd exposed comm (µs)
#   6  wg exposed comm  (µs)
#   7  ig exposed comm  (µs)
#   8  fwd total comm   (µs)   ← our collective time (collective is in fwd column)
#   9  fwd algbw  (GB/s)       ← algbw
#  10  fwd busbw  (GB/s)       ← busbw
#  11  wg total comm    (µs)
#  12  wg algbw   (GB/s)
#  13  wg busbw   (GB/s)
#  14  ig total comm    (µs)
#  15  ig algbw   (GB/s)
#  16  ig busbw   (GB/s)
#  17  workload finished at (µs)
#
# Summary rows start with "SUM", "total exposed comm", or empty first field.
# The first row of the file is a different summary line (File name, Expose DP comm, ...);
# it is skipped because its first field is not a layer name.

_SKIP_PREFIXES = {"layer_name", "SUM", "total exposed comm", ""}


def _parse_endtoend_csv(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(csv_path) as fh:
        for raw in fh:
            parts = [p.strip() for p in raw.strip().split(",")]
            if not parts or parts[0] in _SKIP_PREFIXES:
                continue
            if len(parts) < 14:
                continue
            try:
                rows.append(
                    {
                        "name":          parts[0],
                        "comm_us":       float(parts[8]),
                        "algbw_GBps":    float(parts[9]),
                        "busbw_GBps":    float(parts[10]),
                    }
                )
            except (ValueError, IndexError):
                continue
    return rows


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def _find_binary(simai_root: Path, name: str) -> Path | None:
    candidates = [
        simai_root / "bin" / name,
        simai_root / "astra-sim-alibabacloud/build/simai_analytical/build/simai_analytical/SimAI_analytical",
        simai_root / "astra-sim-alibabacloud/extern/network_backend/ns3-interface/simulation/build/scratch/ns3.36.1-AstraSimNetwork-debug",
    ]
    # Pick the right candidates per binary
    if name == "SimAI_analytical":
        check = [
            simai_root / "bin/SimAI_analytical",
            simai_root / "astra-sim-alibabacloud/build/simai_analytical/build/simai_analytical/SimAI_analytical",
        ]
    else:
        check = [
            simai_root / "bin/SimAI_simulator",
            simai_root / "astra-sim-alibabacloud/extern/network_backend/ns3-interface/simulation/build/scratch/ns3.36.1-AstraSimNetwork-debug",
        ]
    for p in check:
        resolved = p.resolve() if p.is_symlink() else p
        if resolved.exists():
            return resolved
    return None


# ---------------------------------------------------------------------------
# Calibration from nccl-tests measurements
# ---------------------------------------------------------------------------

# SimAI analytical uses GBps=1/2^30 (GiB basis) internally but hardware specs
# are in SI GB/s, introducing a 2^30/1e9 = 1.0737x inflation factor.
# To cancel it: nv_calibrated = desired_busbw / 1.0737
_SIMAI_INFLATION = 2**30 / 1e9  # ≈ 1.0737


def _nccl_plateau_busbw(nccl_dir: Path, collective: str, label: str) -> float | None:
    """Return the large-message plateau bus_bw (GB/s) from a nccl-tests JSON.

    Uses the average of the top-3 largest message sizes to smooth measurement
    noise. Returns None if the file is not found.
    """
    path = nccl_dir / f"nccl_{collective.lower()}_{label}.json"
    if not path.exists():
        return None
    import json
    with open(path) as f:
        d = json.load(f)
    results = d.get("results", [])
    if not results:
        return None
    bws = [r["out_of_place"]["bus_bw"] for r in results if r["out_of_place"]["bus_bw"] > 0]
    if not bws:
        return None
    return sum(bws[-3:]) / len(bws[-3:])


def _calibrated_nv(plateau_busbw: float) -> float:
    """Back-solve the -nv value that makes SimAI output plateau_busbw.

    For single-node SimAI analytical: bus_bw = nv × _SIMAI_INFLATION
    (ratio=1 always because cal_ratio returns 1 for all collectives at
    large message sizes due to getValue() normalising by the last CSV row).
    """
    return plateau_busbw / _SIMAI_INFLATION


# ---------------------------------------------------------------------------
# Analytical mode
# ---------------------------------------------------------------------------

def run_analytical(
    simai_root: Path,
    collective: str,
    num_nodes: int,
    gpus_per_node: int,
    nccl_dir: Path | None = None,
) -> list[dict] | None:
    """Run SimAI_analytical; return per-size rows or None on failure.

    When nccl_dir is provided the -nv (single-node) or -nic (multi-node)
    parameter is calibrated from nccl-tests measurements so that SimAI's
    large-message plateau matches real hardware.  Without calibration SimAI
    inflates results by ~7% (GiB/GB unit mismatch) and ignores the nvlink
    efficiency ratio for AllReduce entirely.
    """
    import subprocess

    binary = _find_binary(simai_root, "SimAI_analytical")
    if binary is None:
        logger.error(
            "SimAI_analytical not found. Build with:\n"
            "  cd %s && ./scripts/build.sh -c analytical",
            simai_root,
        )
        return None

    label = f"{num_nodes}n{gpus_per_node}g"
    nv_bw  = _NV_BW_GBps
    nic_bw = _NIC_BW_GBps

    if nccl_dir is not None:
        plateau = _nccl_plateau_busbw(nccl_dir, collective, label)
        if plateau is not None:
            if num_nodes == 1:
                # Single-node: NVLink is the bottleneck; calibrate -nv
                nv_bw = _calibrated_nv(plateau)
                logger.debug(
                    "%s %s: nccl plateau=%.1f GB/s → -nv=%.1f",
                    collective, label, plateau, nv_bw,
                )
            else:
                # Multi-node: NIC is the bottleneck; calibrate -nic
                nic_bw = plateau / _SIMAI_INFLATION
                logger.debug(
                    "%s %s: nccl plateau=%.1f GB/s → -nic=%.1f",
                    collective, label, plateau, nic_bw,
                )
        else:
            logger.debug(
                "%s %s: no nccl data in %s, using hardware defaults",
                collective, label, nccl_dir,
            )

    num_gpus = num_nodes * gpus_per_node
    workload_text = _make_workload(collective, num_gpus)
    result_prefix = f"sweep_{collective}_{num_gpus}g_{uuid.uuid4().hex[:8]}_"
    results_dir = simai_root / "results"
    results_dir.mkdir(exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", dir=simai_root, delete=False
    ) as wf:
        wf.write(workload_text)
        workload_path = Path(wf.name)

    try:
        cmd = [
            str(binary),
            "-w", str(workload_path),
            "-g",     str(num_gpus),
            "-g_p_s", str(gpus_per_node),
            "-r",     result_prefix,
            "-g_type", _GPU_TYPE,
            "-nv",    str(nv_bw),
            "-nic",   str(nic_bw),
            "-n_p_s", str(_NICS_PER_NODE),
        ]
        logger.debug("cmd: %s", " ".join(cmd))
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            # CWD must be simai_root: ratio CSV paths are relative to it
            cwd=str(simai_root),
        )
        if proc.returncode != 0:
            logger.error(
                "SimAI_analytical failed (rc=%d)\nstderr:\n%s",
                proc.returncode, proc.stderr[-2000:],
            )
            return None
    except subprocess.TimeoutExpired:
        logger.error("SimAI_analytical timed out")
        return None
    finally:
        workload_path.unlink(missing_ok=True)

    # Result file: <simai_root>/results/<prefix>EndToEnd.csv
    csv_path = results_dir / f"{result_prefix}EndToEnd.csv"
    if not csv_path.exists():
        # Fallback: search for any newly created EndToEnd.csv
        csvs = sorted(
            results_dir.glob(f"{result_prefix}*EndToEnd.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not csvs:
            logger.error("EndToEnd.csv not found (prefix=%s)", result_prefix)
            logger.debug("stdout:\n%s", proc.stdout[-2000:])
            return None
        csv_path = csvs[0]

    try:
        rows = _parse_endtoend_csv(csv_path)
    finally:
        # Clean up result files written by SimAI
        for f in results_dir.glob(f"{result_prefix}*"):
            f.unlink(missing_ok=True)

    return rows


# ---------------------------------------------------------------------------
# NS3 mode
# ---------------------------------------------------------------------------

def _gen_topology(
    simai_root: Path, num_nodes: int, gpus_per_node: int, work_dir: Path
) -> Path | None:
    """Generate a Spectrum-X NS3 topology file; return path (without extension)."""
    import subprocess

    gen = (
        simai_root
        / "astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py"
    )
    if not gen.exists():
        logger.error("gen_Topo_Template.py not found at %s", gen)
        return None

    num_gpus = num_nodes * gpus_per_node
    cmd = [
        sys.executable, str(gen),
        "-topo", "Spectrum-X",
        "-g",    str(num_gpus),
        "-gps",  str(gpus_per_node),
        "-gt",   _GPU_TYPE,
        "-bw",   _NS3_NIC_BW,
        "-nvbw", _NS3_NVLINK_BW,
        "-l",    _NS3_NIC_LAT,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(work_dir))
    if proc.returncode != 0:
        logger.error("Topology generation failed:\n%s", proc.stderr)
        return None

    # Find the generated file (no extension, name starts with Spectrum-X)
    candidates = [
        p for p in work_dir.iterdir()
        if p.name.startswith("Spectrum-X") and not p.suffix
    ]
    if not candidates:
        logger.error("No topology file found in %s after generation", work_dir)
        return None
    return candidates[0]


def run_ns3(
    simai_root: Path,
    collective: str,
    num_nodes: int,
    gpus_per_node: int,
    as_send_lat: int = 3,
) -> list[dict] | None:
    """Run SimAI_simulator (NS3); return per-size rows or None on failure."""
    import subprocess

    binary = _find_binary(simai_root, "SimAI_simulator")
    if binary is None:
        logger.error(
            "SimAI_simulator not found. Build with:\n"
            "  cd %s && ./scripts/build.sh -c ns3",
            simai_root,
        )
        return None

    conf = simai_root / "astra-sim-alibabacloud/inputs/config/SimAI.conf"
    if not conf.exists():
        logger.error("SimAI.conf not found at %s", conf)
        return None

    num_gpus = num_nodes * gpus_per_node
    workload_text = _make_workload(collective, num_gpus)

    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir)
        workload_file = tmpdir / "workload.txt"
        workload_file.write_text(workload_text)

        topo = _gen_topology(simai_root, num_nodes, gpus_per_node, tmpdir)
        if topo is None:
            return None

        import os
        num_threads = os.cpu_count() or 1
        cmd = [
            str(binary),
            "-t", str(num_threads),
            "-w", str(workload_file),
            "-n", str(topo),       # gen_Topo_Template writes no extension
            "-c", str(conf),
        ]
        env = os.environ.copy()
        env["AS_SEND_LAT"] = str(as_send_lat)
        logger.debug("NS3 cmd: %s", " ".join(cmd))
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,
                cwd=str(tmpdir),
                env=env,
            )
            if proc.returncode != 0:
                logger.error(
                    "SimAI NS3 failed (rc=%d)\nstderr:\n%s",
                    proc.returncode, proc.stderr[-2000:],
                )
                return None
        except subprocess.TimeoutExpired:
            logger.error("SimAI NS3 timed out (2 h)")
            return None

        # NS3 writes to ./ncclFlowModel_EndToEnd.csv (RESULT_PATH = "./ncclFlowModel_")
        csvs = sorted(tmpdir.rglob("*EndToEnd.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not csvs:
            logger.error("EndToEnd.csv not found after NS3 run")
            logger.debug("stdout:\n%s", proc.stdout[-2000:])
            return None

        return _parse_endtoend_csv(csvs[0])


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

_BUS_BW_FACTOR = {
    "AllReduce":     lambda n: 2 * (n - 1) / n,
    "AllGather":     lambda n: (n - 1) / n,
    "ReduceScatter": lambda n: (n - 1) / n,
    "AllToAll":      lambda n: (n - 1) / n,
}


def _to_json(
    mode: str,
    collective: str,
    num_nodes: int,
    gpus_per_node: int,
    rows: list[dict],
) -> dict:
    num_gpus = num_nodes * gpus_per_node
    factor = _BUS_BW_FACTOR[collective](num_gpus)
    results = []
    for i, row in enumerate(rows):
        if i >= len(MESSAGE_SIZES_BYTES):
            break
        size = MESSAGE_SIZES_BYTES[i]
        time_us = row["comm_us"]
        # Compute alg_bw and bus_bw from actual simulation time, same as nccl-tests.
        # SimAI's CSV columns 9/10 are calbusbw-theory values, not size/time — using
        # the actual time here gives a consistent basis for cross-tool comparison.
        alg_bw = (size / 1e9) / (time_us / 1e6) if time_us > 0 else float("inf")
        bus_bw = alg_bw * factor
        results.append(
            {
                "size": size,
                "out_of_place": {
                    "time":    time_us,
                    "alg_bw": alg_bw,
                    "bus_bw": bus_bw,
                },
            }
        )
    return {
        "version": 1,
        "mode": mode,
        "config": {
            "collective":   collective,
            "num_nodes":    num_nodes,
            "gpus_per_node": gpus_per_node,
            "ngpus":        num_gpus,
        },
        "results": results,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mode",
        choices=["analytical", "ns3", "both"],
        default="analytical",
        help="Simulation mode to run (default: analytical)",
    )
    parser.add_argument(
        "--simai-root",
        type=Path,
        default=Path("~/uni/t/simai-original").expanduser(),
        help="Path to SimAI repo root (default: ~/uni/t/simai-original)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/validation/simccl/results"),
        help="Directory to write JSON results (default: experiments/validation/simccl/results)",
    )
    parser.add_argument(
        "--collective",
        choices=["all"] + [c.lower() for c in COLLECTIVES],
        default="all",
        help="Collective to run (default: all)",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        choices=[1, 2, 4],
        default=None,
        help="Number of nodes to run (1, 2, or 4). Default: all configs.",
    )
    parser.add_argument(
        "--nccl-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing nccl_<collective>_<config>.json files. "
            "When provided, -nv/-nic are calibrated from the measured plateau "
            "bus_bw so that SimAI analytical matches real hardware at large "
            "message sizes (default: same as --output-dir)"
        ),
    )
    parser.add_argument(
        "--lat",
        type=int,
        default=3,
        help="AS_SEND_LAT value for NS3 mode (default: 3, per SimAI README)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    nccl_dir = args.nccl_dir if args.nccl_dir is not None else args.output_dir
    modes = ["analytical", "ns3"] if args.mode == "both" else [args.mode]
    collectives = (
        COLLECTIVES if args.collective == "all"
        else [c for c in COLLECTIVES if c.lower() == args.collective]
    )
    configs = (
        CONFIGS if args.nodes is None
        else [cfg for cfg in CONFIGS if cfg["num_nodes"] == args.nodes]
    )

    for cfg in configs:
        for collective in collectives:
            for mode in modes:
                label = cfg["label"]
                print(f"[simai-{mode}] {collective:15s}  {label} ...", flush=True)

                if mode == "analytical":
                    rows = run_analytical(
                        args.simai_root, collective,
                        cfg["num_nodes"], cfg["gpus_per_node"],
                        nccl_dir=nccl_dir,
                    )
                else:
                    rows = run_ns3(
                        args.simai_root, collective,
                        cfg["num_nodes"], cfg["gpus_per_node"],
                        as_send_lat=args.lat,
                    )

                if rows is None:
                    print("      -> FAILED")
                    continue

                n_expected = len(MESSAGE_SIZES_BYTES)
                if len(rows) != n_expected:
                    logger.warning(
                        "Got %d rows, expected %d — some sizes may be missing",
                        len(rows), n_expected,
                    )

                data = _to_json(mode, collective, cfg["num_nodes"], cfg["gpus_per_node"], rows)
                out = (
                    args.output_dir
                    / f"simai_{mode}_{collective.lower()}_{label}.json"
                )
                out.write_text(json.dumps(data, indent=2))
                print(f"      -> {out}")


if __name__ == "__main__":
    main()
