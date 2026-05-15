"""Tests for ATLAHS backends with shared mock binary setup.

Tests the full simulate() pipeline — DAG → GOAL → txt2bin → simulator —
using mock shell scripts discovered via the ``SIMULON_ATLAHS_BIN_DIR`` env var
rather than patching ``find_binaries`` (which the per-backend unit tests do).
"""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from simulon.backend.atlahs_base import ATLAHSResult
from simulon.backend.dag.goal_trace import dag_to_goal, write_goal_trace
from simulon.backend.dag.nodes import ComputeNode, CommNode, DAGEdge, ExecutionDAG
from simulon.backend.atlahs_htsim import ATLAHShtsimBackend
from simulon.backend.atlahs_lgs import ATLAHSLGSBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_script(path: Path, content: str) -> None:
    """Write an executable shell script at *path*."""
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IRUSR | stat.S_IXUSR)


def _make_minimal_dag() -> ExecutionDAG:
    """Single compute node (rank 0, 1.0 ms) — sufficient for GOAL export."""
    dag = ExecutionDAG()
    dag.compute_nodes.append(
        ComputeNode(
            node_id=0,
            gpu_rank=0,
            kernel="layernorm",
            layer_id=0,
            microbatch_id=0,
            pipeline_stage=0,
            phase="fwd",
            duration_ms=1.0,
        )
    )
    return dag


def _make_multi_rank_dag() -> ExecutionDAG:
    """Two ranks with one comm edge — richer GOAL content."""
    dag = ExecutionDAG()

    dag.compute_nodes.append(
        ComputeNode(
            node_id=0,
            gpu_rank=0,
            kernel="layernorm",
            layer_id=0,
            microbatch_id=0,
            pipeline_stage=0,
            phase="fwd",
            duration_ms=1.0,
        )
    )
    dag.compute_nodes.append(
        ComputeNode(
            node_id=1,
            gpu_rank=1,
            kernel="attn_qkv",
            layer_id=0,
            microbatch_id=0,
            pipeline_stage=0,
            phase="fwd",
            duration_ms=2.0,
        )
    )
    dag.comm_nodes.append(
        CommNode(
            node_id=2,
            src_gpu=0,
            dst_gpu=1,
            bytes=4096,
            collective_type="AllReduce",
            layer_id=0,
            phase="fwd",
            flow_id=0,
            parent_flow_ids=[],
        )
    )
    dag.edges.append(DAGEdge(src_node_id=0, dst_node_id=2))
    dag.edges.append(DAGEdge(src_node_id=2, dst_node_id=1))
    return dag


# ---------------------------------------------------------------------------
# Fixtures — mock binary directories
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_bin_dir(tmp_path: Path) -> Path:
    """Create mock ATLAHS binaries and point ``SIMULON_ATLAHS_BIN_DIR`` at them.

    Provides executable scripts for all four required binaries:
    ``txt2bin``, ``LogGOPSim``, ``htsim_uec`` and ``schedgen``.
    """
    bin_dir = tmp_path / "mock_bin"
    bin_dir.mkdir()

    # txt2bin: copies input file to output file
    _make_mock_script(
        bin_dir / "txt2bin",
        """#!/bin/sh
while getopts i:o: opt; do case $opt in i) input="$OPTARG" ;; o) output="$OPTARG" ;; esac; done
cp "$input" "$output"
""",
    )
    # LogGOPSim: single host timing line
    _make_mock_script(bin_dir / "LogGOPSim", '#!/bin/sh\necho "Host 0: 1000000"\n')
    # htsim_uec: host timing + Htsim time line
    _make_mock_script(
        bin_dir / "htsim_uec",
        '#!/bin/sh\necho "Host 0: 1000000"\necho "It terminates! Htsim time 1000000"\n',
    )
    # schedgen: stub (required by find_binaries)
    _make_mock_script(bin_dir / "schedgen", "#!/bin/sh\necho stub\n")

    old = os.environ.get("SIMULON_ATLAHS_BIN_DIR")
    os.environ["SIMULON_ATLAHS_BIN_DIR"] = str(bin_dir)
    yield bin_dir
    if old is None:
        os.environ.pop("SIMULON_ATLAHS_BIN_DIR", None)
    else:
        os.environ["SIMULON_ATLAHS_BIN_DIR"] = old


@pytest.fixture
def empty_bin_dir(tmp_path: Path) -> Path:
    """Empty directory — used to test missing-binary errors."""
    d = tmp_path / "empty_bin"
    d.mkdir()
    old = os.environ.get("SIMULON_ATLAHS_BIN_DIR")
    os.environ["SIMULON_ATLAHS_BIN_DIR"] = str(d)
    yield d
    if old is None:
        os.environ.pop("SIMULON_ATLAHS_BIN_DIR", None)
    else:
        os.environ["SIMULON_ATLAHS_BIN_DIR"] = old


# ---------------------------------------------------------------------------
# Fixtures — backends and DAGs
# ---------------------------------------------------------------------------


@pytest.fixture
def lgs_backend() -> ATLAHSLGSBackend:
    return ATLAHSLGSBackend()


@pytest.fixture
def htsim_backend() -> ATLAHShtsimBackend:
    return ATLAHShtsimBackend()


@pytest.fixture
def htsim_setup(htsim_backend: ATLAHShtsimBackend) -> ATLAHShtsimBackend:
    """Configure htsim backend with a mock scenario so ``_run_simulator`` works."""
    scenario = MagicMock()
    # Datacenter fields accessed by _run_simulator
    scenario.datacenter.cluster.num_nodes = 2
    scenario.datacenter.node.gpus_per_node = 4
    scenario.datacenter.node.from_ = ""  # prevent template loading
    # These become MagicMock implicitly — fine since generate_topology is mocked
    htsim_backend.scenario = scenario
    return htsim_backend


@pytest.fixture
def minimal_dag() -> ExecutionDAG:
    return _make_minimal_dag()


@pytest.fixture
def multi_rank_dag() -> ExecutionDAG:
    return _make_multi_rank_dag()


# ===================================================================
# GOAL generation validity (direct unit tests, no binaries needed)
# ===================================================================


class TestGoalGeneration:
    """Direct ``dag_to_goal()`` format tests — no binaries required."""

    def test_single_rank_goal(self, minimal_dag: ExecutionDAG) -> None:
        goal = dag_to_goal(minimal_dag)
        assert "num_ranks 1" in goal
        assert "rank 0 {" in goal
        assert "c0: calc 1000000" in goal  # 1.0 ms → 1 000 000 ns
        assert goal.strip().endswith("}")

    def test_multi_rank_goal(self, multi_rank_dag: ExecutionDAG) -> None:
        goal = dag_to_goal(multi_rank_dag)
        assert "num_ranks 2" in goal
        assert "c0: calc 1000000" in goal
        assert "c1: calc 2000000" in goal  # 2.0 ms → 2 000 000 ns
        assert "s2: send 4096b to 1 tag 2" in goal
        assert "r2: recv 4096b from 0 tag 2" in goal

    def test_empty_dag_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            dag_to_goal(ExecutionDAG())

    def test_unpopulated_dag_raises(self) -> None:
        dag = ExecutionDAG()
        dag.compute_nodes.append(
            ComputeNode(
                node_id=0,
                gpu_rank=0,
                kernel="layernorm",
                layer_id=0,
                microbatch_id=0,
                pipeline_stage=0,
                phase="fwd",
                duration_ms=None,
            )
        )
        with pytest.raises(ValueError, match="duration_ms"):
            dag_to_goal(dag)


# ===================================================================
# LGS Backend: full simulate() end-to-end
# ===================================================================


class TestLGSBackendSimulate:
    """``ATLAHSLGSBackend.simulate()`` exercised through the whole pipeline."""

    # ------------------------------------------------------------------
    # Happy path
    # ------------------------------------------------------------------

    def test_simulate_returns_dag_and_result(
        self,
        lgs_backend: ATLAHSLGSBackend,
        mock_bin_dir: Path,
        minimal_dag: ExecutionDAG,
    ) -> None:
        """``simulate()`` returns ``(dag, result)`` with ``total_time_ms == 1.0``."""
        with patch.object(lgs_backend, "run_trace", return_value=minimal_dag):
            dag, result = lgs_backend.simulate(MagicMock())

        assert dag is minimal_dag
        assert result.total_time_ms == pytest.approx(1.0)
        assert isinstance(result, ATLAHSResult)

    def test_simulate_per_host_times(
        self,
        lgs_backend: ATLAHSLGSBackend,
        mock_bin_dir: Path,
        minimal_dag: ExecutionDAG,
    ) -> None:
        """``per_host_times`` dict and summary string are populated correctly."""
        with patch.object(lgs_backend, "run_trace", return_value=minimal_dag):
            _, result = lgs_backend.simulate(MagicMock())

        assert result.per_host_times == {0: 1.0}
        assert "Host 0: 1.000 ms" in result.summary

    # ------------------------------------------------------------------
    # GOAL file content (spy on write_goal_trace to capture before cleanup)
    # ------------------------------------------------------------------

    def test_goal_file_content(
        self,
        lgs_backend: ATLAHSLGSBackend,
        mock_bin_dir: Path,
        minimal_dag: ExecutionDAG,
        tmp_path: Path,
    ) -> None:
        """GOAL file written during ``simulate()`` has valid content."""
        import simulon.backend.atlahs_base as atlahs_base_mod

        goal_copy = tmp_path / "lgs_goal.goal"
        _original_write = atlahs_base_mod.write_goal_trace

        def _saving_write(dag: ExecutionDAG, path: str | Path) -> None:
            _original_write(dag, path)
            import shutil

            shutil.copy2(path, goal_copy)

        with patch.object(lgs_backend, "run_trace", return_value=minimal_dag), \
             patch.object(atlahs_base_mod, "write_goal_trace", _saving_write):
            lgs_backend.simulate(MagicMock())

        content = goal_copy.read_text()
        assert "num_ranks 1" in content
        assert "c0: calc 1000000" in content

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_error_missing_binary(
        self,
        lgs_backend: ATLAHSLGSBackend,
        empty_bin_dir: Path,
        minimal_dag: ExecutionDAG,
    ) -> None:
        """``RuntimeError`` when ATLAHS binaries cannot be found."""
        with patch.object(lgs_backend, "run_trace", return_value=minimal_dag), \
             patch(
                 "simulon.backend.atlahs_binary_finder._get_platform_tag",
                 return_value="_nonexistent_",
             ), \
             patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="Missing ATLAHS binaries"):
                lgs_backend.simulate(MagicMock())

    def test_error_nonzero_exit(
        self,
        lgs_backend: ATLAHSLGSBackend,
        mock_bin_dir: Path,
        minimal_dag: ExecutionDAG,
    ) -> None:
        """``RuntimeError`` when LogGOPSim exits with a non-zero code."""
        failing = mock_bin_dir / "LogGOPSim"
        failing.write_text('#!/bin/sh\necho "error" >&2\nexit 1\n')
        failing.chmod(failing.stat().st_mode | stat.S_IEXEC)

        with patch.object(lgs_backend, "run_trace", return_value=minimal_dag):
            with pytest.raises(RuntimeError, match="LogGOPSim failed"):
                lgs_backend.simulate(MagicMock())

    def test_error_timeout(
        self,
        lgs_backend: ATLAHSLGSBackend,
        mock_bin_dir: Path,
        minimal_dag: ExecutionDAG,
    ) -> None:
        """``RuntimeError`` when LogGOPSim times out."""
        with patch.object(lgs_backend, "run_trace", return_value=minimal_dag), \
             patch.object(
                 lgs_backend,
                 "_run_simulator",
                 side_effect=RuntimeError("LogGOPSim timed out after 300s"),
             ):
            with pytest.raises(RuntimeError, match="timed out"):
                lgs_backend.simulate(MagicMock())

    # ------------------------------------------------------------------
    # Temp file cleanup
    # ------------------------------------------------------------------

    def test_temp_file_cleanup(
        self,
        lgs_backend: ATLAHSLGSBackend,
        mock_bin_dir: Path,
        minimal_dag: ExecutionDAG,
    ) -> None:
        """Temp directory is cleaned up after a successful ``simulate()``."""
        real_td = tempfile.TemporaryDirectory(prefix="test_lgs_ok_")
        tracked_name = real_td.name

        with patch.object(lgs_backend, "run_trace", return_value=minimal_dag), \
             patch(
                 "simulon.backend.atlahs_base.tempfile.TemporaryDirectory",
                 return_value=real_td,
             ):
            lgs_backend.simulate(MagicMock())

        assert not Path(tracked_name).exists(), (
            f"Temp dir {tracked_name} was not cleaned up"
        )

    def test_temp_file_cleanup_on_error(
        self,
        lgs_backend: ATLAHSLGSBackend,
        mock_bin_dir: Path,
        minimal_dag: ExecutionDAG,
    ) -> None:
        """Temp directory is cleaned up even when simulation raises."""
        real_td = tempfile.TemporaryDirectory(prefix="test_lgs_err_")
        tracked_name = real_td.name

        with patch.object(lgs_backend, "run_trace", return_value=minimal_dag), \
             patch(
                 "simulon.backend.atlahs_base.tempfile.TemporaryDirectory",
                 return_value=real_td,
             ), \
             patch.object(
                 lgs_backend, "_run_simulator", side_effect=RuntimeError("boom")
             ):
            with pytest.raises(RuntimeError):
                lgs_backend.simulate(MagicMock())

        assert not Path(tracked_name).exists(), (
            f"Temp dir {tracked_name} was not cleaned up after error"
        )


# ===================================================================
# htsim Backend: full simulate() end-to-end
# ===================================================================


class TestHtsimBackendSimulate:
    """``ATLAHShtsimBackend.simulate()`` exercised through the whole pipeline."""

    # ------------------------------------------------------------------
    # Happy path
    # ------------------------------------------------------------------

    def test_simulate_returns_dag_and_result(
        self,
        htsim_setup: ATLAHShtsimBackend,
        mock_bin_dir: Path,
        minimal_dag: ExecutionDAG,
    ) -> None:
        """``simulate()`` returns ``(dag, result)`` with ``total_time_ms == 1.0``."""
        with patch.object(htsim_setup, "run_trace", return_value=minimal_dag), \
             patch("simulon.backend.atlahs_htsim.generate_topology", return_value="mock topo"):
            dag, result = htsim_setup.simulate(MagicMock())

        assert dag is minimal_dag
        assert result.total_time_ms == pytest.approx(1.0)
        assert isinstance(result, ATLAHSResult)

    def test_simulate_per_host_times(
        self,
        htsim_setup: ATLAHShtsimBackend,
        mock_bin_dir: Path,
        minimal_dag: ExecutionDAG,
    ) -> None:
        """``per_host_times`` dict and summary string are populated correctly."""
        with patch.object(htsim_setup, "run_trace", return_value=minimal_dag), \
             patch("simulon.backend.atlahs_htsim.generate_topology", return_value="mock topo"):
            _, result = htsim_setup.simulate(MagicMock())

        assert result.per_host_times == {0: 1.0}
        assert "Host 0: 1.000 ms" in result.summary

    # ------------------------------------------------------------------
    # GOAL file content
    # ------------------------------------------------------------------

    def test_goal_file_content(
        self,
        htsim_setup: ATLAHShtsimBackend,
        mock_bin_dir: Path,
        minimal_dag: ExecutionDAG,
        tmp_path: Path,
    ) -> None:
        """GOAL file written during ``simulate()`` has valid content."""
        import simulon.backend.atlahs_base as atlahs_base_mod

        goal_copy = tmp_path / "htsim_goal.goal"
        _original_write = atlahs_base_mod.write_goal_trace

        def _saving_write(dag: ExecutionDAG, path: str | Path) -> None:
            _original_write(dag, path)
            import shutil

            shutil.copy2(path, goal_copy)

        with patch.object(htsim_setup, "run_trace", return_value=minimal_dag), \
             patch.object(atlahs_base_mod, "write_goal_trace", _saving_write), \
             patch("simulon.backend.atlahs_htsim.generate_topology", return_value="mock topo"):
            htsim_setup.simulate(MagicMock())

        content = goal_copy.read_text()
        assert "num_ranks 1" in content
        assert "c0: calc 1000000" in content

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_error_missing_binary(
        self,
        htsim_backend: ATLAHShtsimBackend,
        empty_bin_dir: Path,
        minimal_dag: ExecutionDAG,
    ) -> None:
        """``RuntimeError`` when ATLAHS binaries cannot be found."""
        with patch.object(htsim_backend, "run_trace", return_value=minimal_dag), \
             patch(
                 "simulon.backend.atlahs_binary_finder._get_platform_tag",
                 return_value="_nonexistent_",
             ), \
             patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="Missing ATLAHS binaries"):
                htsim_backend.simulate(MagicMock())

    def test_error_nonzero_exit(
        self,
        htsim_setup: ATLAHShtsimBackend,
        mock_bin_dir: Path,
        minimal_dag: ExecutionDAG,
    ) -> None:
        """``RuntimeError`` when htsim_uec exits with a non-zero code."""
        failing = mock_bin_dir / "htsim_uec"
        failing.write_text('#!/bin/sh\necho "error" >&2\nexit 1\n')
        failing.chmod(failing.stat().st_mode | stat.S_IEXEC)

        with patch.object(htsim_setup, "run_trace", return_value=minimal_dag), \
             patch("simulon.backend.atlahs_htsim.generate_topology", return_value="mock topo"):
            with pytest.raises(RuntimeError, match="htsim failed"):
                htsim_setup.simulate(MagicMock())

    def test_error_timeout(
        self,
        htsim_setup: ATLAHShtsimBackend,
        mock_bin_dir: Path,
        minimal_dag: ExecutionDAG,
    ) -> None:
        """``RuntimeError`` when htsim_uec times out."""
        with patch.object(htsim_setup, "run_trace", return_value=minimal_dag), \
             patch.object(
                 htsim_setup,
                 "_run_simulator",
                 side_effect=RuntimeError("htsim timed out after 300s"),
             ), \
             patch("simulon.backend.atlahs_htsim.generate_topology", return_value="mock topo"):
            with pytest.raises(RuntimeError, match="timed out"):
                htsim_setup.simulate(MagicMock())

    # ------------------------------------------------------------------
    # Temp file cleanup
    # ------------------------------------------------------------------

    def test_temp_file_cleanup(
        self,
        htsim_setup: ATLAHShtsimBackend,
        mock_bin_dir: Path,
        minimal_dag: ExecutionDAG,
    ) -> None:
        """Temp directory is cleaned up after a successful ``simulate()``."""
        real_td = tempfile.TemporaryDirectory(prefix="test_htsim_ok_")
        tracked_name = real_td.name

        with patch.object(htsim_setup, "run_trace", return_value=minimal_dag), \
             patch(
                 "simulon.backend.atlahs_base.tempfile.TemporaryDirectory",
                 return_value=real_td,
             ), \
             patch("simulon.backend.atlahs_htsim.generate_topology", return_value="mock topo"):
            htsim_setup.simulate(MagicMock())

        assert not Path(tracked_name).exists(), (
            f"Temp dir {tracked_name} was not cleaned up"
        )

    def test_temp_file_cleanup_on_error(
        self,
        htsim_setup: ATLAHShtsimBackend,
        mock_bin_dir: Path,
        minimal_dag: ExecutionDAG,
    ) -> None:
        """Temp directory is cleaned up even when simulation raises."""
        real_td = tempfile.TemporaryDirectory(prefix="test_htsim_err_")
        tracked_name = real_td.name

        with patch.object(htsim_setup, "run_trace", return_value=minimal_dag), \
             patch(
                 "simulon.backend.atlahs_base.tempfile.TemporaryDirectory",
                 return_value=real_td,
             ), \
             patch("simulon.backend.atlahs_htsim.generate_topology", return_value="mock topo"), \
             patch.object(
                 htsim_setup, "_run_simulator", side_effect=RuntimeError("boom")
             ):
            with pytest.raises(RuntimeError):
                htsim_setup.simulate(MagicMock())

        assert not Path(tracked_name).exists(), (
            f"Temp dir {tracked_name} was not cleaned up after error"
        )
