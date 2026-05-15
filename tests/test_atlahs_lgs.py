"""Tests for ATLAHSLGSBackend."""

from __future__ import annotations

import stat
from pathlib import Path
from unittest.mock import patch

import pytest

from simulon.backend.atlahs_lgs import ATLAHSLGSBackend


@pytest.fixture
def backend():
    return ATLAHSLGSBackend()


def _make_mock_binary(tmp_path: Path, script: str) -> Path:
    """Write a shell script that acts as a mock LogGOPSim executable."""
    mock_bin = tmp_path / "LogGOPSim"
    mock_bin.write_text(script)
    mock_bin.chmod(mock_bin.stat().st_mode | stat.S_IEXEC)
    return mock_bin


def test_run_simulator_parses_host_time(backend, tmp_path):
    """Mock binary output 'Host 0: 1000000' yields total_time_ms=1.0."""
    mock_bin = _make_mock_binary(
        tmp_path,
        script='#!/bin/sh\necho "Host 0: 1000000"\n',
    )
    with patch(
        "simulon.backend.atlahs_lgs.find_binaries",
        return_value={"LogGOPSim": str(mock_bin)},
    ):
        result = backend._run_simulator(tmp_path / "dummy.bin")

    assert result.total_time_ms == pytest.approx(1.0)
    assert result.per_host_times == {0: 1.0}
    assert "Host 0: 1.000 ms" in result.summary


def test_run_simulator_multiple_hosts(backend, tmp_path):
    """Maximum host time is chosen when multiple hosts are reported."""
    mock_bin = _make_mock_binary(
        tmp_path,
        script='#!/bin/sh\necho "Host 0: 500000"\necho "Host 1: 2000000"\n',
    )
    with patch(
        "simulon.backend.atlahs_lgs.find_binaries",
        return_value={"LogGOPSim": str(mock_bin)},
    ):
        result = backend._run_simulator(tmp_path / "dummy.bin")

    assert result.total_time_ms == pytest.approx(2.0)
    assert result.per_host_times == {0: 0.5, 1: 2.0}


def test_run_simulator_nonzero_exit_raises(backend, tmp_path):
    """Non-zero exit code from LogGOPSim raises RuntimeError with stderr."""
    mock_bin = _make_mock_binary(
        tmp_path,
        script='#!/bin/sh\necho "something broke" >&2\nexit 1\n',
    )
    with patch(
        "simulon.backend.atlahs_lgs.find_binaries",
        return_value={"LogGOPSim": str(mock_bin)},
    ):
        with pytest.raises(RuntimeError, match="LogGOPSim failed"):
            backend._run_simulator(tmp_path / "dummy.bin")


def test_run_simulator_no_matches_raises(backend, tmp_path):
    """Stdout without host times raises RuntimeError."""
    mock_bin = _make_mock_binary(
        tmp_path,
        script='#!/bin/sh\necho "no timing here"\n',
    )
    with patch(
        "simulon.backend.atlahs_lgs.find_binaries",
        return_value={"LogGOPSim": str(mock_bin)},
    ):
        with pytest.raises(RuntimeError, match="Could not parse LogGOPSim output"):
            backend._run_simulator(tmp_path / "dummy.bin")


def test_run_simulator_uses_loggops_params(backend, tmp_path):
    """Custom LogGOPS parameters are forwarded as CLI flags."""
    mock_bin = _make_mock_binary(
        tmp_path,
        script='#!/bin/sh\nfor arg in "$@"; do echo "$arg"; done\necho "Host 0: 1000"\n',
    )
    backend._loggops_params = {"L": 1234, "o": 99, "g": 1, "G": 0.5, "O": 7, "S": 42}
    with patch(
        "simulon.backend.atlahs_lgs.find_binaries",
        return_value={"LogGOPSim": str(mock_bin)},
    ):
        result = backend._run_simulator(tmp_path / "dummy.bin")

    raw = result.raw_output
    assert "-L" in raw
    assert "1234" in raw
    assert "-o" in raw
    assert "99" in raw
    assert "-g" in raw
    assert "1" in raw
    assert "-G" in raw
    assert "0.5" in raw
    assert "-O" in raw
    assert "7" in raw
    assert "-S" in raw
    assert "42" in raw
