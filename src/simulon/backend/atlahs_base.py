"""ATLAHS backend base class.

Provides a shared simulation pipeline for all ATLAHS-based backends:
build DAG → populate compute timing → export to GOAL → txt2bin → run simulator.

Network population and DAG replay are skipped because ATLAHS simulators
model network timing independently from the GOAL schedule.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

from simulon.backend.atlahs_binary_finder import find_binaries
from simulon.backend.base import Backend, BackendResult
from simulon.backend.dag import ExecutionDAG
from simulon.backend.dag.goal_trace import write_goal_trace
from simulon.config.scenario import ScenarioConfig

logger = logging.getLogger(__name__)


@dataclass
class ATLAHSResult:
    """Result returned by an ATLAHS simulator backend."""

    total_time_ms: float
    summary: str
    per_host_times: dict[int, float]
    raw_output: str


class ATLAHSBackendBase(Backend):
    """Abstract base for ATLAHS simulator backends.

    Subclasses must implement :meth:`run_trace` and :meth:`_run_simulator`.
    The :meth:`simulate` orchestration follows this flow:

        1. ``run_trace()`` → build the ExecutionDAG
        2. ``write_goal_trace()`` → export DAG to GOAL text
        3. ``txt2bin`` → convert GOAL text to binary
        4. ``_run_simulator()`` → subclass-specific simulator invocation
    """

    @abstractmethod
    def run_trace(self, scenario: ScenarioConfig, compact: bool = False) -> ExecutionDAG:
        """Build an ExecutionDAG for *scenario*.

        Subclasses typically delegate to a DAG tracer (e.g.
        :class:`MegatronDAGTracer`) or collective tracer.
        """
        ...

    @abstractmethod
    def _run_simulator(self, bin_path: Path) -> ATLAHSResult:
        """Invoke the ATLAHS simulator on the compiled GOAL binary.

        Args:
            bin_path: Absolute path to the ``.bin`` file produced by txt2bin.

        Returns:
            Parsed simulation result.
        """
        ...

    def _find_txt2bin(self) -> Path:
        """Locate the txt2bin executable."""
        binaries = find_binaries()
        return Path(binaries["txt2bin"])

    def run(self, scenario: ScenarioConfig) -> dict[str, object]:
        """Run the simulation and return a plain dict."""
        dag, result = self.simulate(scenario)
        summary = getattr(result, "summary", None)
        return {
            "status": "success",
            "compute_nodes": len(dag.compute_nodes),
            "comm_nodes": len(dag.comm_nodes),
            "edges": len(dag.edges),
            "dag": dag.to_dict(),
            "result": {
                "total_time_ms": result.total_time_ms,
                "summary": summary,
            },
        }

    def simulate(
        self,
        scenario: ScenarioConfig,
        compact: bool = False,
        ignore_oom: bool = False,
        ignore_missing: bool = False,
    ) -> tuple[ExecutionDAG, BackendResult]:
        """Build DAG, export to GOAL, convert via txt2bin, and run the simulator.

        Temp files are created inside a :class:`tempfile.TemporaryDirectory` and
        are cleaned up in a ``finally`` block even if the simulator crashes.
        """
        logger.info("ATLAHS: building DAG ...")
        dag = self.run_trace(scenario, compact=compact)
        logger.info(
            "  DAG built: %d compute nodes, %d comm nodes, %d edges",
            len(dag.compute_nodes),
            len(dag.comm_nodes),
            len(dag.edges),
        )

        tmp_dir: tempfile.TemporaryDirectory[str] | None = None
        try:
            tmp_dir = tempfile.TemporaryDirectory(prefix="simulon_atlahs_")
            tmp_path = Path(tmp_dir.name)
            goal_path = tmp_path / "schedule.goal"
            bin_path = tmp_path / "schedule.bin"

            logger.info("ATLAHS: writing GOAL trace ...")
            write_goal_trace(dag, goal_path)
            logger.info("  GOAL trace written to %s", goal_path)

            txt2bin = self._find_txt2bin()
            logger.info("ATLAHS: running txt2bin (%s) ...", txt2bin)
            _ = subprocess.run(
                [str(txt2bin), "-i", str(goal_path), "-o", str(bin_path)],
                check=True,
                capture_output=True,
            )
            logger.info("  txt2bin done -> %s", bin_path)

            logger.info("ATLAHS: running simulator ...")
            result = self._run_simulator(bin_path)
            logger.info("  Simulator done: total_time=%.3f ms", result.total_time_ms)

            return dag, result

        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
            logger.error("txt2bin failed: %s", stderr)
            raise RuntimeError(f"txt2bin conversion failed: {stderr}") from exc

        finally:
            if tmp_dir is not None:
                tmp_dir.cleanup()
