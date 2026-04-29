"""Tests for ATLAHShtsimBackend."""

from __future__ import annotations

import stat
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from simulon.backend.atlahs_htsim import ATLAHShtsimBackend


@pytest.fixture
def backend():
    return ATLAHShtsimBackend()


def _make_mock_binary(tmp_path: Path, script: str) -> Path:
    """Write a shell script that acts as a mock htsim_uec executable."""
    mock_bin = tmp_path / "htsim_uec"
    mock_bin.write_text(script)
    mock_bin.chmod(mock_bin.stat().st_mode | stat.S_IEXEC)
    return mock_bin


def _make_mock_scenario():
    """Create a minimal mock scenario for htsim backend tests."""
    scenario = MagicMock()
    scenario.datacenter = MagicMock()
    scenario.datacenter.cluster.num_nodes = 2
    scenario.datacenter.node = MagicMock()
    scenario.datacenter.node.gpus_per_node = 4
    scenario.datacenter.node.from_ = None
    scenario.datacenter.scale_out = MagicMock()
    scenario.datacenter.scale_out.nic = MagicMock()
    scenario.datacenter.scale_out.nic.speed = "400Gbps"
    return scenario


def test_run_simulator_parses_host_time(backend, tmp_path):
    """Mock binary output 'Host 0: 1000000' yields total_time_ms=1.0."""
    mock_bin = _make_mock_binary(
        tmp_path,
        script='#!/bin/sh\necho "Host 0: 1000000"\necho "Maximum finishing time at host 0: 1000000 (0.001 s)"\n',
    )
    backend.scenario = _make_mock_scenario()
    with patch(
        "simulon.backend.atlahs_htsim.find_binaries",
        return_value={"htsim_uec": str(mock_bin)},
    ):
        with patch("simulon.backend.atlahs_htsim.generate_topology", return_value="mock topo"):
            result = backend._run_simulator(tmp_path / "dummy.bin")

    assert result.total_time_ms == pytest.approx(1.0)
    assert result.per_host_times == {0: 1.0}
    assert "Host 0: 1.000 ms" in result.summary


def test_run_simulator_multiple_hosts(backend, tmp_path):
    """Maximum host time is chosen when multiple hosts are reported."""
    mock_bin = _make_mock_binary(
        tmp_path,
        script='#!/bin/sh\necho "Host 0: 500000"\necho "Host 1: 2000000"\necho "Maximum finishing time at host 1: 2000000 (0.002 s)"\n',
    )
    backend.scenario = _make_mock_scenario()
    with patch(
        "simulon.backend.atlahs_htsim.find_binaries",
        return_value={"htsim_uec": str(mock_bin)},
    ):
        with patch("simulon.backend.atlahs_htsim.generate_topology", return_value="mock topo"):
            result = backend._run_simulator(tmp_path / "dummy.bin")

    assert result.total_time_ms == pytest.approx(2.0)
    assert result.per_host_times == {0: 0.5, 1: 2.0}


def test_run_simulator_nonzero_exit_raises(backend, tmp_path):
    """Non-zero exit code from htsim raises RuntimeError with stderr."""
    mock_bin = _make_mock_binary(
        tmp_path,
        script='#!/bin/sh\necho "something broke" >&2\nexit 1\n',
    )
    backend.scenario = _make_mock_scenario()
    with patch(
        "simulon.backend.atlahs_htsim.find_binaries",
        return_value={"htsim_uec": str(mock_bin)},
    ):
        with patch("simulon.backend.atlahs_htsim.generate_topology", return_value="mock topo"):
            with pytest.raises(RuntimeError, match="htsim failed"):
                backend._run_simulator(tmp_path / "dummy.bin")


def test_run_simulator_no_matches_raises(backend, tmp_path):
    """Stdout without host times raises RuntimeError."""
    mock_bin = _make_mock_binary(
        tmp_path,
        script='#!/bin/sh\necho "no timing here"\n',
    )
    backend.scenario = _make_mock_scenario()
    with patch(
        "simulon.backend.atlahs_htsim.find_binaries",
        return_value={"htsim_uec": str(mock_bin)},
    ):
        with patch("simulon.backend.atlahs_htsim.generate_topology", return_value="mock topo"):
            with pytest.raises(RuntimeError, match="Could not parse htsim output"):
                backend._run_simulator(tmp_path / "dummy.bin")


def test_run_simulator_uses_htsim_time_fallback(backend, tmp_path):
    """Uses 'Htsim time' fallback when no Maximum finishing time line is present."""
    mock_bin = _make_mock_binary(
        tmp_path,
        script='#!/bin/sh\necho "Host 0: 1500000"\necho "It terminates! Htsim time 1500000"\n',
    )
    backend.scenario = _make_mock_scenario()
    with patch(
        "simulon.backend.atlahs_htsim.find_binaries",
        return_value={"htsim_uec": str(mock_bin)},
    ):
        with patch("simulon.backend.atlahs_htsim.generate_topology", return_value="mock topo"):
            result = backend._run_simulator(tmp_path / "dummy.bin")

    assert result.total_time_ms == pytest.approx(1.5)
