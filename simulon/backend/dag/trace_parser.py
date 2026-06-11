from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict, cast


@dataclass
class TraceEvent:
    type: str  # collective | slot_begin | slot_end
    timestamp_ms: float
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class TraceFile:
    trace_format_version: str
    rank: int
    world_size: int
    pipeline_stage: int
    events: list[TraceEvent]
    total_flops: float | None = None
    energy_kwh: float | None = None
    co2eq_kg: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "trace_format_version": self.trace_format_version,
            "rank": self.rank,
            "world_size": self.world_size,
            "pipeline_stage": self.pipeline_stage,
            "events": [
                {"type": e.type, "timestamp_ms": e.timestamp_ms, "metadata": e.metadata}
                for e in self.events
            ],
            "total_flops": self.total_flops,
            "energy_kwh": self.energy_kwh,
            "co2eq_kg": self.co2eq_kg,
        }

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class TraceEventInput(TypedDict):
    type: str
    timestamp_ms: int | float
    metadata: dict[str, object]


class TraceFileParser:
    @staticmethod
    def parse(path: str | Path) -> TraceFile:
        trace_path = Path(path)
        data_obj = json.loads(trace_path.read_text())  # pyright: ignore[reportAny]
        if not isinstance(data_obj, dict):
            raise ValueError("Trace file must contain a JSON object")
        data = cast(dict[str, object], data_obj)

        required_fields = ("trace_format_version", "rank", "world_size", "pipeline_stage", "events")
        missing = [field_name for field_name in required_fields if field_name not in data]
        if missing:
            raise ValueError(f"Missing required top-level field(s): {', '.join(missing)}")

        trace_format_version = data["trace_format_version"]
        if trace_format_version != "1.0":
            raise ValueError(f"Unsupported trace format version: {trace_format_version!r}")

        rank_obj = data["rank"]
        world_size_obj = data["world_size"]
        pipeline_stage_obj = data["pipeline_stage"]
        if not isinstance(rank_obj, int):
            raise ValueError("rank must be an integer")
        if not isinstance(world_size_obj, int):
            raise ValueError("world_size must be an integer")
        if not isinstance(pipeline_stage_obj, int):
            raise ValueError("pipeline_stage must be an integer")

        events_obj = data["events"]
        if not isinstance(events_obj, list):
            raise ValueError("events must be a list")
        events_data = cast(list[TraceEventInput], events_obj)

        total_flops = None
        if "total_flops" in data:
            tf = data["total_flops"]
            if isinstance(tf, int | float):
                total_flops = float(tf)

        energy_kwh = None
        if "energy_kwh" in data:
            ek = data["energy_kwh"]
            if isinstance(ek, int | float):
                energy_kwh = float(ek)

        co2eq_kg = None
        if "co2eq_kg" in data:
            ck = data["co2eq_kg"]
            if isinstance(ck, int | float):
                co2eq_kg = float(ck)

        events: list[TraceEvent] = []
        for event_data in events_data:
            event_type = event_data["type"]
            metadata = event_data["metadata"]

            bytes_value = metadata.get("bytes", 0)
            if event_type == "collective":
                if not isinstance(bytes_value, int | float):
                    raise ValueError("collective events must have metadata.bytes as int or float")
                if bytes_value < 0:
                    raise ValueError("collective events must have metadata.bytes >= 0")

            if event_type not in {"collective", "slot_begin", "slot_end"}:
                raise ValueError(f"Unsupported event type: {event_type!r}")

            timestamp_ms = float(event_data["timestamp_ms"])

            events.append(TraceEvent(type=event_type, timestamp_ms=timestamp_ms, metadata=metadata))

        return TraceFile(
            trace_format_version=str(trace_format_version),
            rank=rank_obj,
            world_size=world_size_obj,
            pipeline_stage=pipeline_stage_obj,
            events=events,
            total_flops=total_flops,
            energy_kwh=energy_kwh,
            co2eq_kg=co2eq_kg,
        )
