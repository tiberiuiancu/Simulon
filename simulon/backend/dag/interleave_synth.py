"""Synthesize an interleaved-1F1B (VPP / explicit-layout) trace from a plain-PP trace.

WHY THIS EXISTS. Of the 141 measured Qwen3-32B configurations on this cluster, 98 -- every
production run and everything above 16 nodes -- use an interleaved pipeline schedule with an
explicit ``--pipeline-model-parallel-layout``. Simulon could not replay any of them, so the
regime the simulator exists to choose within was exactly the regime it could not speak about.

WHY IT CAN BE SYNTHESIZED RATHER THAN CAPTURED. Two facts, both measured (job 1854038):

  * Per-chunk compute is LINEAR IN THE CHUNK'S LAYER COUNT. Fitting one ``t_layer`` across 34
    slot classes spanning PP2/PP4/PP8, DP1/DP4, plain and the 16-chunk production layout gives
    a mean residual of 3.6% and a maximum of 5.8%; equal-layer chunks within a stage agree to
    ~1.3%. A 5-layer chunk costs 1.23x a 4-layer chunk (5/4 = 1.25). There is no short-chunk
    kernel penalty to capture, so the `PP*VPP <= 16` rule in the scaling guide is a property
    of the SCHEDULE, which this module reproduces, not of the kernels.
  * The embedding costs ~0 layer-equivalents and the output head ~1.9-2.4, measured directly
    by comparing the chunks that own them against those that do not.

So a chunk's work is `n_layers` per-layer blocks plus, where the layout says so, the
embedding preamble or the output-head postamble -- all taken from a REAL trace of the same
(model, TP, mbs, seq) via ``schedule_synth.extract_stage_primitive``. Nothing here is a fitted
constant: ``t_layer`` is derived per-direction from the source trace being replayed, never
baked in, so it carries that trace's own hardware and kernel mix.

WHAT IS STILL ASSUMED. The schedule itself. The chunk-id rule and warmup depth below are
Megatron's, and were CHECKED against the captured layout trace: the chunk-id formula
reproduces all 512 recorded slots on stages 0 and 3 exactly, and the measured count of
forwards before the first backward is the warmup formula + 1 (the steady loop opens with a
forward). What is not verified here is the P2P wiring at chunk boundaries under failure of
those assumptions -- see ``tests/test_interleave_synth.py`` for what is gated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from simulon.backend.dag.schedule_synth import (
    CollectiveItem,
    ComputeItem,
    Item,
    StagePrimitive,
    extract_stage_primitive,
)
from simulon.backend.dag.trace_parser import TraceEvent, TraceFile, TraceFileParser

_SLOT_METRICS = ("kernel_device_ms", "host_ms", "host_ops", "launch_count", "kernel_count")


# ---------------------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LayoutGroup:
    """One "|"-separated pipeline group of a --pipeline-model-parallel-layout."""

    n_layers: int
    has_embedding: bool
    has_head: bool


def parse_layout(layout: str) -> list[LayoutGroup]:
    """Parse Megatron's layout syntax: E (input embedding), t / t*N (layers), L (loss head).

    Configs often carry it backslash-escaped ("Et\\*5\\|t\\*4"), so strip those first.
    """
    out: list[LayoutGroup] = []
    for grp in layout.replace("\\", "").split("|"):
        n, i = 0, 0
        while i < len(grp):
            if grp[i] == "t":
                if i + 1 < len(grp) and grp[i + 1] == "*":
                    j = i + 2
                    while j < len(grp) and grp[j].isdigit():
                        j += 1
                    n += int(grp[i + 2:j] or 1)
                    i = j
                    continue
                n += 1
            i += 1
        out.append(LayoutGroup(n, "E" in grp, "L" in grp))
    return out


def uniform_layout(num_layers: int, pp: int, chunks: int) -> list[LayoutGroup]:
    """The layout implied by --num-layers-per-virtual-pipeline-stage (equal chunks)."""
    n_groups = pp * chunks
    per = num_layers // n_groups
    return [LayoutGroup(per, g == 0, g == n_groups - 1) for g in range(n_groups)]


def group_of(stage: int, chunk: int, pp: int) -> int:
    """Megatron assigns layout group ``stage + chunk*pp`` to (stage, chunk)."""
    return stage + chunk * pp


def layer_span(groups: list[LayoutGroup], g: int) -> range:
    """Global layer indices owned by layout group *g* (groups are in pipeline order)."""
    start = sum(x.n_layers for x in groups[:g])
    return range(start, start + groups[g].n_layers)


# ---------------------------------------------------------------------------------------
# schedule
# ---------------------------------------------------------------------------------------

def chunk_id(virtual_mb: int, forward: bool, pp: int, chunks: int) -> int:
    """Megatron's model-chunk rule. Verified against every recorded slot of job 1854038."""
    c = (virtual_mb % (pp * chunks)) // pp
    return c if forward else chunks - c - 1


def num_warmup(stage: int, pp: int, chunks: int, n_mb: int) -> int:
    """Warmup forward slots before the steady 1F1B loop, from Megatron's scheduler.

    The `(pp - stage - 1) * 2 + (chunks - 1) * pp` form is the INTERLEAVED scheduler's, and
    it must not be applied when there is only one chunk: Megatron then runs the separate
    non-interleaved scheduler, whose warmup is `pp - stage - 1` (so `pp - stage` forwards
    before the first backward, which is what pp_synth measured against real PP2/PP4 traces).
    Applying the interleaved form at chunks=1 fills the pipeline twice as deep and produced a
    31 s bubble against a 5.7 s measurement on 8n/PP8 -- a layout with exactly `pp` groups is
    an uneven plain split, not an interleaved schedule.
    """
    total = n_mb * chunks
    if chunks <= 1:
        return max(0, min(pp - stage - 1, total))
    return min((pp - stage - 1) * 2 + (chunks - 1) * pp, total)


def interleaved_actions(stage: int, pp: int, chunks: int, n_mb: int) -> list[tuple[str, int, int]]:
    """Ordered [(direction, virtual_microbatch_id, model_chunk_id)] for one stage.

    Warmup forwards, then the steady loop (one forward then one backward per step -- which is
    why the measured count of forwards before the first backward is num_warmup + 1), then the
    cooldown backwards.
    """
    total = n_mb * chunks
    nw = num_warmup(stage, pp, chunks, n_mb)
    out: list[tuple[str, int, int]] = []
    for v in range(nw):
        out.append(("fwd", v, chunk_id(v, True, pp, chunks)))
    n_steady = total - nw
    for i in range(n_steady):
        f = nw + i
        out.append(("fwd", f, chunk_id(f, True, pp, chunks)))
        out.append(("bwd", i, chunk_id(i, False, pp, chunks)))
    for i in range(n_steady, total):
        out.append(("bwd", i, chunk_id(i, False, pp, chunks)))
    return out


# ---------------------------------------------------------------------------------------
# per-layer primitives from the source trace
# ---------------------------------------------------------------------------------------

@dataclass
class LayerLibrary:
    """Per-layer building blocks plus the endpoint terms, from a plain-PP source trace.

    ``blocks[direction][i]`` is the item stream of global layer *i*. ``metric[direction][key]``
    is the per-LAYER share of a per-slot scalar (kernel_device_ms, host_ms, ...), and
    ``embedding``/``head`` are the extra shares owned by the first/last group. All are derived
    from the source trace, so replacing the source replaces every number here.
    """

    blocks: dict[str, list[list[Item]]]
    metric: dict[str, dict[str, float]]
    embedding: dict[str, dict[str, float]]
    head: dict[str, dict[str, float]]


def _slot_metrics(trace: TraceFile, direction: str) -> dict[str, float]:
    """The representative slot's per-slot scalars for one direction."""
    best: dict[str, float] = {}
    for ev in trace.events:
        if ev.type != "slot_begin" or ev.metadata.get("direction") != direction:
            continue
        cur = {k: float(v) for k, v in ev.metadata.items() if k in _SLOT_METRICS and v is not None}
        if cur and (not best or cur.get("kernel_device_ms", 0) > best.get("kernel_device_ms", 0)):
            best = cur
    return best


def build_layer_library(stage_traces: list[TraceFile], num_layers: int) -> LayerLibrary:
    """Decompose a plain-PP trace into per-layer blocks and endpoint terms.

    The interior stages (those owning neither the embedding nor the head) define the per-layer
    metric share; the first and last stages' excess over that defines the embedding and head
    terms. With pp < 3 there is no interior stage, so the two endpoints are solved jointly
    from the assumption the measurements support -- embedding ~ 0 -- and the head takes the
    remainder.
    """
    pp = len(stage_traces)
    per_stage = num_layers // pp
    prims = [extract_stage_primitive(t, per_stage) for t in stage_traces]

    blocks: dict[str, list[list[Item]]] = {}
    metric: dict[str, dict[str, float]] = {}
    embedding: dict[str, dict[str, float]] = {}
    head: dict[str, dict[str, float]] = {}

    for direction in ("fwd", "bwd"):
        # forward: stage s owns layers [s*per_stage, (s+1)*per_stage). backward blocks are
        # recorded last-layer-first, so reverse each stage's list to index by global layer.
        per_layer_blocks: list[list[Item]] = []
        for s, prim in enumerate(prims):
            lb = prim.layers.get(direction, [])
            if direction == "bwd":
                lb = list(reversed(lb))
            per_layer_blocks.extend(b.items for b in lb)
        if len(per_layer_blocks) < num_layers:      # segmentation failed -> pad by repetition
            if not per_layer_blocks:
                per_layer_blocks = [[] for _ in range(num_layers)]
            while len(per_layer_blocks) < num_layers:
                per_layer_blocks.append(list(per_layer_blocks[-1]))
        blocks[direction] = per_layer_blocks[:num_layers]

        slot = [_slot_metrics(t, direction) for t in stage_traces]
        keys = sorted({k for m in slot for k in m})
        interior = [s for s in range(pp) if 0 < s < pp - 1]
        per_layer: dict[str, float] = {}
        for k in keys:
            if interior:
                per_layer[k] = sum(slot[s].get(k, 0.0) for s in interior) / (len(interior) * per_stage)
            else:
                # no interior stage: embedding ~ 0 (measured), so stage 0 sets the per-layer rate
                per_layer[k] = slot[0].get(k, 0.0) / per_stage
        metric[direction] = per_layer
        embedding[direction] = {k: max(0.0, slot[0].get(k, 0.0) - per_layer[k] * per_stage) for k in keys}
        head[direction] = {k: max(0.0, slot[pp - 1].get(k, 0.0) - per_layer[k] * per_stage) for k in keys}
    return LayerLibrary(blocks, metric, embedding, head)


# ---------------------------------------------------------------------------------------
# synthesis
# ---------------------------------------------------------------------------------------

def _chunk_items(lib: LayerLibrary, groups: list[LayoutGroup], g: int, direction: str,
                 prims: list[StagePrimitive]) -> tuple[list[Item], dict[str, float]]:
    """The item stream and per-slot scalars for layout group *g* in one direction."""
    grp = groups[g]
    span = layer_span(groups, g)
    items: list[Item] = []
    if grp.has_embedding and direction == "fwd":
        items.extend(prims[0].preamble.get(direction, []))
    order = span if direction == "fwd" else reversed(span)
    for i in order:
        items.extend(lib.blocks[direction][i] if i < len(lib.blocks[direction]) else [])
    if grp.has_head:
        items.extend(prims[-1].postamble.get(direction, []))

    metrics: dict[str, float] = {}
    for k, per in lib.metric[direction].items():
        v = per * grp.n_layers
        if grp.has_embedding:
            v += lib.embedding[direction].get(k, 0.0)
        if grp.has_head:
            v += lib.head[direction].get(k, 0.0)
        metrics[k] = v
    return items, metrics


def _pp_bytes(stage_traces: list[TraceFile]) -> int:
    for tr in stage_traces:
        for ev in tr.events:
            if ev.type == "collective" and ev.metadata.get("collective_type") in ("PP_Send", "PP_Recv"):
                b = int(ev.metadata.get("bytes", 0) or 0)
                if b > 0:
                    return b
    return 0


def synthesize_interleaved(stage_traces: list[TraceFile], target_pp: int, chunks: int,
                           n_mb: int, groups: list[LayoutGroup],
                           num_layers: int) -> dict[int, TraceFile]:
    """Build one interleaved trace per target stage. Returns {representative rank: TraceFile}."""
    if len(groups) != target_pp * chunks:
        raise ValueError(f"layout has {len(groups)} groups, expected pp*chunks="
                         f"{target_pp * chunks}")
    src_pp = len(stage_traces)
    world = stage_traces[0].world_size * target_pp // src_pp
    rps = world // target_pp
    lib = build_layer_library(stage_traces, num_layers)
    prims = [extract_stage_primitive(t, num_layers // src_pp) for t in stage_traces]
    pp_bytes = _pp_bytes(stage_traces)

    out: dict[int, TraceFile] = {}
    for stage in range(target_pp):
        rank = stage * rps
        prev_rank, next_rank = rank - rps, rank + rps
        events: list[TraceEvent] = []
        t = 0.0
        spacer = 0.001
        for direction, vmb, chunk in interleaved_actions(stage, target_pp, chunks, n_mb):
            g = group_of(stage, chunk, target_pp)
            items, metrics = _chunk_items(lib, groups, g, direction, prims)
            # inbound P2P: within a chunk from the previous stage; at stage 0 the input comes
            # from the last stage's previous chunk (the virtual pipeline wraps round)
            if direction == "fwd" and not (stage == 0 and chunk == 0):
                src = prev_rank if stage > 0 else (target_pp - 1) * rps
                events.append(TraceEvent("collective", t, {
                    "name": "recv_forward", "collective_type": "PP_Recv", "bytes": pp_bytes,
                    "group_ranks": [src, rank], "microbatch_id": vmb, "direction": "fwd"}))
            elif direction == "bwd" and not (stage == target_pp - 1 and chunk == chunks - 1):
                src = next_rank if stage < target_pp - 1 else 0
                events.append(TraceEvent("collective", t, {
                    "name": "recv_backward", "collective_type": "PP_Recv", "bytes": pp_bytes,
                    "group_ranks": [rank, src], "microbatch_id": vmb, "direction": "bwd"}))
            t += spacer

            events.append(TraceEvent("slot_begin", t, {
                "microbatch_id": vmb, "direction": direction, "pipeline_stage": stage,
                "model_chunk_id": chunk, **metrics}))
            for it in items:
                if isinstance(it, ComputeItem):
                    t += it.duration_ms
                elif isinstance(it, CollectiveItem):
                    md = dict(it.metadata)
                    md["microbatch_id"] = None
                    md["direction"] = None
                    events.append(TraceEvent("collective", t, md))
            events.append(TraceEvent("slot_end", t, {
                "microbatch_id": vmb, "direction": direction, "pipeline_stage": stage}))
            t += spacer

            if direction == "fwd" and not (stage == target_pp - 1 and chunk == chunks - 1):
                dst = next_rank if stage < target_pp - 1 else 0
                events.append(TraceEvent("collective", t, {
                    "name": "send_forward", "collective_type": "PP_Send", "bytes": pp_bytes,
                    "group_ranks": [rank, dst], "microbatch_id": vmb, "direction": "fwd"}))
            elif direction == "bwd" and not (stage == 0 and chunk == 0):
                dst = prev_rank if stage > 0 else (target_pp - 1) * rps
                events.append(TraceEvent("collective", t, {
                    "name": "send_backward", "collective_type": "PP_Send", "bytes": pp_bytes,
                    "group_ranks": [dst, rank], "microbatch_id": vmb, "direction": "bwd"}))
            t += spacer

        # one optimizer step per iteration, summed over the chunks this stage owns
        step_metrics: dict[str, float] = {}
        for tr in stage_traces:
            for ev in tr.events:
                if ev.type == "slot_begin" and ev.metadata.get("direction") == "step":
                    for k, v in ev.metadata.items():
                        if k in _SLOT_METRICS and v is not None:
                            step_metrics[k] = step_metrics.get(k, 0.0) + float(v)
                    break
        if step_metrics:
            scale = 1.0 / max(1, src_pp)          # per stage, not summed over the source
            events.append(TraceEvent("slot_begin", t, {
                "microbatch_id": 0, "direction": "step", "pipeline_stage": stage,
                **{k: v * scale for k, v in step_metrics.items()}}))
            t += spacer
            events.append(TraceEvent("slot_end", t, {
                "microbatch_id": 0, "direction": "step", "pipeline_stage": stage}))

        src = stage_traces[min(stage * max(1, src_pp // target_pp), src_pp - 1)]
        out[rank] = TraceFile(
            trace_format_version=src.trace_format_version, rank=rank, world_size=world,
            pipeline_stage=stage, events=events, total_flops=src.total_flops,
            energy_kwh=src.energy_kwh, co2eq_kg=src.co2eq_kg)
    return out


def derive_interleaved_from_dir(source_dir: Path, dest_dir: Path, src_pp: int, target_pp: int,
                                chunks: int, n_mb: int, num_layers: int,
                                layout: str | None = None) -> Path:
    """Read a plain-PP trace dir, write an interleaved trace dir, return dest_dir."""
    import yaml

    rank0 = TraceFileParser.parse(str(source_dir / "trace_rank_0.json"))
    rps_src = rank0.world_size // src_pp
    stage_traces = [TraceFileParser.parse(str(source_dir / f"trace_rank_{s * rps_src}.json"))
                    for s in range(src_pp)]
    groups = parse_layout(layout) if layout else uniform_layout(num_layers, target_pp, chunks)
    derived = synthesize_interleaved(stage_traces, target_pp, chunks, n_mb, groups, num_layers)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for rank, tf in derived.items():
        (dest_dir / f"trace_rank_{rank}.json").write_text(tf.to_json(indent=None))
    src_wl = source_dir / "workload.yaml"
    if src_wl.exists():
        data = yaml.safe_load(src_wl.read_text())
        if isinstance(data, dict) and isinstance(data.get("config"), dict):
            cfg = data["config"]
            cfg["pipeline-model-parallel-size"] = target_pp
            for k in ("num-gpus", "num_gpus"):
                if k in cfg and cfg[k]:
                    cfg[k] = int(cfg[k]) * target_pp // src_pp
            if layout:
                cfg["pipeline-model-parallel-layout"] = layout
            else:
                cfg["num-layers-per-virtual-pipeline-stage"] = num_layers // (target_pp * chunks)
            # interleaved schedules keep overlap_p2p_comm; Megatron only forces it off for
            # the non-interleaved path
            cfg["overlap-p2p-comm"] = True
            (dest_dir / "workload.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    return dest_dir
