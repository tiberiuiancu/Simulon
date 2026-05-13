# Trace Format v1.0

This file defines the JSON trace schema produced by `CudaEventTracer.finish_iteration()`.

## Top-level schema

```json
{
  "trace_format_version": "1.0",
  "rank": 0,
  "world_size": 8,
  "pipeline_stage": 0,
  "events": []
}
```

### Fields

- `trace_format_version` — string, must be exactly `"1.0"`.
- `rank` — integer global rank of the traced worker.
- `world_size` — integer total number of ranks in the run.
- `pipeline_stage` — integer pipeline stage index for the worker.
- `events` — array of trace events in chronological order.

## Event schema

Each event object has:

- `type` — one of:
  - `collective`
  - `slot_begin`
  - `slot_end`
- `timestamp_ms` — float timestamp in milliseconds.
- `metadata` — free-form object with event-specific payload.

### Collective event metadata

Collective events must include:

- `bytes` — integer number of bytes transferred; must be `> 0`.

Other metadata keys are allowed and may include collective-specific details.

### Slot events

- `slot_begin` marks the start of a traced pipeline slot.
- `slot_end` marks the end of a traced pipeline slot.

Slot events may carry additional metadata, but no required keys beyond `timestamp_ms` and `metadata`.

## Example

```json
{
  "trace_format_version": "1.0",
  "rank": 0,
  "world_size": 8,
  "pipeline_stage": 0,
  "events": [
    {
      "type": "slot_begin",
      "timestamp_ms": 12.5,
      "metadata": {"microbatch_id": 0}
    },
    {
      "type": "collective",
      "timestamp_ms": 13.0,
      "metadata": {"collective_type": "AllReduce", "bytes": 4096}
    },
    {
      "type": "slot_end",
      "timestamp_ms": 18.25,
      "metadata": {"microbatch_id": 0}
    }
  ]
}
```
