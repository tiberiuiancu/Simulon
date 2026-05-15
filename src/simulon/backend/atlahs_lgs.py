"""ATLAHS LogGOPSim backend.

Runs the GOAL schedule through the ATLAHS LogGOPSim discrete-event simulator
and returns the predicted end-to-end runtime.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import cast

from simulon.backend.atlahs_base import ATLAHSBackendBase, ATLAHSResult
from simulon.backend.atlahs_binary_finder import find_binaries
from simulon.backend.dag import DAGTracerConfig, ExecutionDAG
from simulon.config.dc import DatacenterConfig
from simulon.config.scenario import ScenarioConfig
from simulon.config.workload import CollectiveWorkload, MegatronWorkload

logger = logging.getLogger(__name__)

_DEFAULT_LOGGOPS: dict[str, float | int] = {
    "L": 5000,
    "o": 250,
    "g": 5,
    "G": 0.04,
    "O": 0,
    "S": 0,
}


def _tracer_config_from_scenario(scenario: ScenarioConfig) -> DAGTracerConfig:
    c = scenario.collective
    algorithm = c.algorithm if c.algorithm != "auto" else "ring"
    return DAGTracerConfig(
        num_channels=c.num_channels,
        algorithm=algorithm,
    )


class ATLAHSLGSBackend(ATLAHSBackendBase):
    """Backend that runs ATLAHS LogGOPSim on a GOAL schedule."""

    def __init__(self) -> None:
        self._loggops_params: dict[str, float | int] | None = None

    def run_trace(self, scenario: ScenarioConfig, compact: bool = False) -> ExecutionDAG:
        """Build an ExecutionDAG for *scenario*."""
        try:
            import importlib

            mod = importlib.import_module("simulon.backend.atlahs_lgs_params")
            mapper = getattr(mod, "map_datacenter_to_loggops")
            self._loggops_params = mapper(scenario.datacenter)
        except (ImportError, AttributeError):
            self._loggops_params = None

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
            f"ATLAHSLGSBackend does not support {type(scenario.workload).__name__}"
        )

    def _run_simulator(self, bin_path: Path) -> ATLAHSResult:
        """Invoke LogGOPSim on the compiled GOAL binary and parse the result."""
        binaries = find_binaries()
        loggopsim_path = Path(binaries["LogGOPSim"])

        params = self._loggops_params if self._loggops_params is not None else _DEFAULT_LOGGOPS

        cmd = [str(loggopsim_path), "-f", str(bin_path)]
        for key in ("L", "o", "g", "G", "O", "S"):
            val = params.get(key)
            if val is not None:
                cmd.extend([f"-{key}", str(val)])

        logger.info("Running LogGOPSim: %s", " ".join(cmd))

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("LogGOPSim timed out after 300s") from exc

        if proc.returncode != 0:
            stderr = proc.stderr.strip() if proc.stderr else "(no stderr)"
            raise RuntimeError(f"LogGOPSim failed (exit {proc.returncode}): {stderr}")

        output = proc.stdout
        matches = re.findall(r"[hH]ost \d+: (\d+)", output)
        if not matches:
            raise RuntimeError(f"Could not parse LogGOPSim output: {output}")

        per_host_times: dict[int, float] = {}
        for i, m in enumerate(matches):
            per_host_times[i] = int(m) / 1e6

        total_time_ms = max(per_host_times.values())

        summary_lines = [f"Total time: {total_time_ms:.3f} ms"]
        for host, t in per_host_times.items():
            summary_lines.append(f"  Host {host}: {t:.3f} ms")
        summary = "\n".join(summary_lines)

        return ATLAHSResult(
            total_time_ms=total_time_ms,
            summary=summary,
            per_host_times=per_host_times,
            raw_output=output,
        )
