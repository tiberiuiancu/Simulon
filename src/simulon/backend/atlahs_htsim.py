"""ATLAHS htsim backend.

Runs the GOAL schedule through the ATLAHS htsim packet-level simulator
and returns the predicted end-to-end runtime.
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import cast

from simulon.backend.atlahs_base import ATLAHSBackendBase, ATLAHSResult
from simulon.backend.atlahs_binary_finder import find_binaries
from simulon.backend.dag import DAGTracerConfig, ExecutionDAG
from simulon.backend.htsim_topology import generate_topology
from simulon.config.dc import DatacenterConfig, NICSpec
from simulon.config.resolve import resolve_node_spec, resolve_scale_out
from simulon.config.scenario import ScenarioConfig
from simulon.config.workload import CollectiveWorkload, MegatronWorkload

logger = logging.getLogger(__name__)

_DEFAULT_MTU = 4096
_DEFAULT_PATHS = 128
_DEFAULT_QUEUE = 1_000_000

_SPEED_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*(Gbps|Mbps|Kbps|bps)$", re.IGNORECASE)


def _parse_speed_mbps(speed: str | None) -> int:
    """Parse a speed string like '400Gbps' into Mbps."""
    if speed is None:
        return 200_000  # 200 Gbps default
    m = _SPEED_RE.match(speed.strip())
    if not m:
        raise ValueError(f"Cannot parse NIC speed: {speed!r}")
    val = float(m.group(1))
    unit = m.group(2).lower()
    if unit == "gbps":
        return int(val * 1_000)
    if unit == "mbps":
        return int(val)
    if unit == "kbps":
        return int(val / 1_000)
    return int(val / 1_000_000)


def _tracer_config_from_scenario(scenario: ScenarioConfig) -> DAGTracerConfig:
    c = scenario.collective
    algorithm = c.algorithm if c.algorithm != "auto" else "ring"
    return DAGTracerConfig(
        num_channels=c.num_channels,
        algorithm=algorithm,
    )


class ATLAHShtsimBackend(ATLAHSBackendBase):
    """Backend that runs ATLAHS htsim on a GOAL schedule."""

    def __init__(self) -> None:
        self.scenario: ScenarioConfig | None = None

    def run_trace(self, scenario: ScenarioConfig, compact: bool = False) -> ExecutionDAG:
        """Build an ExecutionDAG for *scenario*."""
        self.scenario = scenario
        cfg = _tracer_config_from_scenario(scenario)
        cfg.compact = compact

        datacenter = cast(DatacenterConfig, scenario.datacenter)

        if isinstance(scenario.workload, MegatronWorkload):
            from simulon.backend.dag.trace_tracer import MegatronDagTracer
            from simulon.collective import NCCLDecomposer

            tracer = MegatronDagTracer(cfg, ccl=NCCLDecomposer())
            return tracer.trace(scenario.workload, datacenter)

        if isinstance(scenario.workload, CollectiveWorkload):
            from simulon.backend.dag.collective_tracer import build_collective_dag
            from simulon.collective import NCCLDecomposer

            return build_collective_dag(
                workload=scenario.workload,
                datacenter=datacenter,
                algorithm=cfg.algorithm,
                num_channels=cfg.num_channels,
                ccl=NCCLDecomposer(),
            )

        raise ValueError(
            f"ATLAHShtsimBackend does not support {type(scenario.workload).__name__}"
        )

    def _run_simulator(self, bin_path: Path) -> ATLAHSResult:
        """Invoke htsim_uec on the compiled GOAL binary and parse the result."""
        if self.scenario is None:
            raise RuntimeError("run_trace must be called before _run_simulator")

        binaries = find_binaries()
        htsim_path = Path(binaries["htsim_uec"])

        dc = cast(DatacenterConfig, self.scenario.datacenter)
        scale_out = resolve_scale_out(dc)
        nic = scale_out.nic if scale_out is not None else None
        if isinstance(nic, str):
            linkspeed_mbps = 200_000
        elif isinstance(nic, NICSpec):
            linkspeed_mbps = _parse_speed_mbps(nic.speed)
        else:
            linkspeed_mbps = 200_000

        node = resolve_node_spec(dc)
        gpus_per_node = node.gpus_per_node
        if gpus_per_node is None:
            raise ValueError("datacenter.node.gpus_per_node is required for htsim simulation")
        total_nodes = dc.cluster.num_nodes * gpus_per_node

        topo_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".topo", delete=False) as f:
                topo_path = Path(f.name)
                f.write(generate_topology(dc))

            cmd = [
                str(htsim_path),
                "-topo", str(topo_path),
                "-goal", str(bin_path),
                "-linkspeed", str(linkspeed_mbps),
                "-nodes", str(total_nodes),
                "-strat", "ecmp_host",
                "-mtu", str(_DEFAULT_MTU),
                "-paths", str(_DEFAULT_PATHS),
                "-q", str(_DEFAULT_QUEUE),
            ]

            logger.info("Running htsim: %s", " ".join(cmd))

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("htsim timed out after 300s") from exc

            if proc.returncode != 0:
                stderr = proc.stderr.strip() if proc.stderr else "(no stderr)"
                raise RuntimeError(f"htsim failed (exit {proc.returncode}): {stderr}")

            output = proc.stdout
            per_host_times: dict[int, float] = {}

            for line in output.splitlines():
                m = re.search(r"^Host\s+(\d+):\s+(\d+)\s*$", line)
                if m:
                    per_host_times[int(m.group(1))] = int(m.group(2)) / 1e6

            max_host_match = re.search(r"Maximum finishing time at host\s+(\d+):\s+(\d+)", output)
            htsim_time_match = re.search(r"It terminates!\s+Htsim time\s+(\d+)", output)

            if max_host_match:
                total_time_ns = int(max_host_match.group(2))
            elif htsim_time_match:
                total_time_ns = int(htsim_time_match.group(1))
            elif per_host_times:
                total_time_ns = max(int(t * 1e6) for t in per_host_times.values())
            else:
                raise RuntimeError(f"Could not parse htsim output: {output}")

            total_time_ms = total_time_ns / 1e6

            summary_lines = [f"Total time: {total_time_ms:.3f} ms"]
            for host, t in sorted(per_host_times.items()):
                summary_lines.append(f"  Host {host}: {t:.3f} ms")
            summary = "\n".join(summary_lines)

            return ATLAHSResult(
                total_time_ms=total_time_ms,
                summary=summary,
                per_host_times=per_host_times,
                raw_output=output,
            )

        finally:
            if topo_path is not None:
                try:
                    topo_path.unlink(missing_ok=True)
                except OSError:
                    pass
