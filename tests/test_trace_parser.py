import json
import tempfile
from pathlib import Path
from typing import cast

import pytest

from simulon.backend.dag.trace_parser import (
    TraceFileParser,  # pyright: ignore[reportMissingTypeStubs]
)


def _write_trace(data: dict[str, object]) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as fp:
        json.dump(data, fp)
        return Path(fp.name)


def _base_trace() -> dict[str, object]:
    return {
        "trace_format_version": "1.0",
        "rank": 0,
        "world_size": 2,
        "pipeline_stage": 0,
        "events": [
            {
                "type": "collective",
                "timestamp_ms": 1.5,
                "metadata": {"collective_type": "AllReduce", "bytes": 1024, "group_ranks": [0, 1]},
            },
            {"type": "slot_begin", "timestamp_ms": 2, "metadata": {"slot": "fwd"}},
            {"type": "slot_end", "timestamp_ms": 3, "metadata": {"slot": "fwd"}},
        ],
    }


def test_parse_valid_trace():
    path = _write_trace(_base_trace())
    try:
        parsed = TraceFileParser.parse(path)
    finally:
        path.unlink(missing_ok=True)

    assert parsed.trace_format_version == "1.0"
    assert parsed.rank == 0
    assert parsed.world_size == 2
    assert parsed.pipeline_stage == 0
    assert [event.type for event in parsed.events] == ["collective", "slot_begin", "slot_end"]


def test_parse_rejects_invalid_version():
    data = _base_trace()
    data["trace_format_version"] = "2.0"
    path = _write_trace(data)
    try:
        with pytest.raises(ValueError, match="Unsupported trace format version"):
            _ = TraceFileParser.parse(path)
    finally:
        path.unlink(missing_ok=True)


def test_parse_empty_events():
    data = _base_trace()
    data["events"] = []
    path = _write_trace(data)
    try:
        parsed = TraceFileParser.parse(path)
    finally:
        path.unlink(missing_ok=True)

    assert parsed.events == []
    assert parsed.rank == 0


def test_parse_collective_metadata_preserved():
    path = _write_trace(_base_trace())
    try:
        parsed = TraceFileParser.parse(path)
    finally:
        path.unlink(missing_ok=True)

    collective = parsed.events[0]
    assert collective.type == "collective"
    assert collective.metadata["collective_type"] == "AllReduce"
    assert collective.metadata["bytes"] == 1024


def test_parse_with_energy_fields():
    data = _base_trace()
    data["total_flops"] = 1234567890
    data["energy_kwh"] = 0.42
    data["co2eq_kg"] = 0.05
    path = _write_trace(data)
    try:
        parsed = TraceFileParser.parse(path)
    finally:
        path.unlink(missing_ok=True)

    assert parsed.total_flops == 1234567890.0
    assert parsed.energy_kwh == 0.42
    assert parsed.co2eq_kg == 0.05


def test_parse_legacy_without_energy_fields():
    data = _base_trace()
    path = _write_trace(data)
    try:
        parsed = TraceFileParser.parse(path)
    finally:
        path.unlink(missing_ok=True)

    assert parsed.energy_kwh is None
    assert parsed.co2eq_kg is None


def test_parse_rejects_unsupported_event_type():
    data = _base_trace()
    events = cast(list[dict[str, object]], data["events"])
    events.append({"type": "bogus", "timestamp_ms": 4, "metadata": {}})
    path = _write_trace(data)
    try:
        with pytest.raises(ValueError, match="Unsupported event type"):
            _ = TraceFileParser.parse(path)
    finally:
        path.unlink(missing_ok=True)
