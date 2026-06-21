"""Schedule synthesis from per-layer primitives.

A profiled trace measures, per pipeline stage, the wall-clock cost of one
forward and one backward microbatch as an ordered stream of *items*: compute
gaps (durations between recorded events) interleaved with TP collectives
(AllGather / ReduceScatter, with their byte volumes).

Crucially, that stream is *periodic* over transformer layers — a slot is
``[pre-amble] + num_layers x [per-layer items]``.  Because the period is fixed
and ``num_layers / pp`` is known from config, the stream can be segmented into
per-layer building blocks.

Once we have per-layer fwd/bwd blocks for each stage, the whole sweep becomes
*schedule arithmetic*: we compose those blocks under any Megatron schedule
(plain 1F1B, interleaved-1F1B / VPP, recompute) without new hardware traces.
The only genuinely irreducible axis is the tensor-parallel degree, which changes
the per-layer GEMM shapes and therefore must be measured.

This module is built so that the *identity* recomposition — same pp, same mbs,
no VPP, no recompute — reproduces the original event stream exactly.  That
round-trip is the correctness gate the higher-level synthesis builds on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from simulon.backend.dag.trace_parser import TraceEvent, TraceFile

# Collectives recorded inside a slot follow a fixed per-layer period.  With
# sequence parallelism each transformer layer contributes one AllGather +
# ReduceScatter for attention and one AllGather + ReduceScatter for the MLP.
COLLECTIVES_PER_LAYER = 4


@dataclass(frozen=True)
class ComputeItem:
    """A compute gap of ``duration_ms`` between two recorded events."""

    duration_ms: float


@dataclass(frozen=True)
class CollectiveItem:
    """A TP collective recorded inside a slot, with its measured byte volume."""

    name: str
    collective_type: str
    bytes: int
    metadata: dict = field(default_factory=dict)


# An item stream is the flat, ordered representation of one slot's interior.
Item = ComputeItem | CollectiveItem


@dataclass
class LayerBlock:
    """The ordered item stream for a single transformer layer in one direction."""

    items: list[Item]

    @property
    def compute_ms(self) -> float:
        return sum(i.duration_ms for i in self.items if isinstance(i, ComputeItem))


@dataclass
class StagePrimitive:
    """Per-layer building blocks for one pipeline stage, both directions.

    ``preamble[direction]`` holds the items that precede the first layer (e.g.
    the leading reduce-scatter and embedding compute) and ``postamble`` the
    items after the last layer.  ``layers[direction]`` is the list of per-layer
    blocks.  Reassembling preamble + layers + postamble reproduces the original
    slot interior exactly.
    """

    num_layers: int
    preamble: dict[str, list[Item]] = field(default_factory=dict)
    layers: dict[str, list[LayerBlock]] = field(default_factory=dict)
    postamble: dict[str, list[Item]] = field(default_factory=dict)


def _slot_streams(trace: TraceFile) -> dict[tuple[int, str], list]:
    """Group a trace's events into per-(microbatch, direction) item streams.

    Returns a mapping ``(microbatch_id, direction) -> list of (kind, payload)``
    where the stream alternates compute gaps and collective events, mirroring
    the timestamp deltas between consecutive in-slot events.
    """
    events = sorted(trace.events, key=lambda e: e.timestamp_ms)
    streams: dict[tuple[int, str], list[Item]] = {}
    in_slot = False
    key: tuple[int, str] | None = None
    prev_ts: float | None = None
    for i, ev in enumerate(events):
        if ev.type == "slot_begin":
            mb = ev.metadata.get("microbatch_id", -1)
            direction = _normalize_direction(ev.metadata)
            key = (int(mb), direction)
            streams.setdefault(key, [])
            in_slot = True
            prev_ts = ev.timestamp_ms
        elif ev.type == "slot_end":
            if in_slot and key is not None and prev_ts is not None:
                gap = ev.timestamp_ms - prev_ts
                if gap > 0:
                    streams[key].append(ComputeItem(duration_ms=gap))
            in_slot = False
            key = None
            prev_ts = None
        elif in_slot and key is not None:
            # The gap *before* this event is compute attributable to the slot.
            if prev_ts is not None:
                gap = ev.timestamp_ms - prev_ts
                if gap > 0:
                    streams[key].append(ComputeItem(duration_ms=gap))
            if ev.type == "collective":
                ct = str(ev.metadata.get("collective_type", ""))
                # PP transfers are inter-slot in 1F1B; intra-slot collectives are TP.
                if ct not in ("PP_Send", "PP_Recv"):
                    streams[key].append(
                        CollectiveItem(
                            name=str(ev.metadata.get("name", "")),
                            collective_type=ct,
                            bytes=int(ev.metadata.get("bytes", 0) or 0),
                            metadata=dict(ev.metadata),
                        )
                    )
            prev_ts = ev.timestamp_ms
    return streams


def _normalize_direction(metadata: dict) -> str:
    phase = str(metadata.get("phase", "") or "")
    if phase == "fwd":
        return "fwd"
    if phase in ("bwd", "bwd_ig", "bwd_wg"):
        return "bwd"
    return str(metadata.get("direction") or metadata.get("slot") or phase or "")


def _segment_into_layers(stream: list[Item], num_layers: int) -> tuple[list, list, list]:
    """Split a slot's item stream into (preamble, [layer blocks], postamble).

    Layers are delimited by their collective period: each layer owns
    ``COLLECTIVES_PER_LAYER`` collectives plus the compute items interleaved
    with and trailing them.  Anything before the first layer's collectives is
    the preamble; anything after the last layer's final collective is the
    postamble.
    """
    # Index collectives.
    coll_positions = [i for i, it in enumerate(stream) if isinstance(it, CollectiveItem)]
    if num_layers <= 0 or len(coll_positions) < num_layers * COLLECTIVES_PER_LAYER:
        # Not enough structure to segment — treat the whole slot as preamble.
        return list(stream), [], []

    # The trailing collectives belong to the layers; any leading collectives
    # beyond the layer budget form the preamble (e.g. a leading reduce-scatter).
    layer_coll_count = num_layers * COLLECTIVES_PER_LAYER
    first_layer_coll = coll_positions[len(coll_positions) - layer_coll_count]

    preamble = stream[:first_layer_coll]

    # Walk the remaining stream, emitting a new layer every COLLECTIVES_PER_LAYER
    # collectives.  Compute items attach to the current (open) layer.
    layers: list[LayerBlock] = []
    current: list[Item] = []
    coll_in_layer = 0
    rest = stream[first_layer_coll:]
    consumed_upto = len(stream)
    for idx, it in enumerate(rest):
        current.append(it)
        if isinstance(it, CollectiveItem):
            coll_in_layer += 1
            if coll_in_layer == COLLECTIVES_PER_LAYER:
                layers.append(LayerBlock(items=current))
                current = []
                coll_in_layer = 0
                if len(layers) == num_layers:
                    consumed_upto = first_layer_coll + idx + 1
                    break

    postamble = stream[consumed_upto:]
    return preamble, layers, postamble


def extract_stage_primitive(trace: TraceFile, num_layers: int) -> StagePrimitive:
    """Decompose one stage's trace into per-layer fwd/bwd building blocks."""
    streams = _slot_streams(trace)
    prim = StagePrimitive(num_layers=num_layers)
    for (_mb, direction), stream in streams.items():
        if direction not in ("fwd", "bwd"):
            continue
        if direction in prim.layers:
            continue  # one representative microbatch per direction is enough
        pre, layers, post = _segment_into_layers(stream, num_layers)
        prim.preamble[direction] = pre
        prim.layers[direction] = layers
        prim.postamble[direction] = post
    return prim


def assemble_slot_items(prim: StagePrimitive, direction: str) -> list[Item]:
    """Reassemble a full slot's item stream from per-layer blocks (identity)."""
    items: list[Item] = list(prim.preamble.get(direction, []))
    for block in prim.layers.get(direction, []):
        items.extend(block.items)
    items.extend(prim.postamble.get(direction, []))
    return items
