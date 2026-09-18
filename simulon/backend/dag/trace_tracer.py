from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from pydantic import ConfigDict

from simulon.backend.dag._progress import log_progress
from simulon.backend.dag.nodes import CollectiveNode, CommNode, ComputeNode, DAGEdge, ExecutionDAG
from simulon.backend.dag.trace_parser import TraceFileParser
from simulon.backend.dag.tracer import DAGTracer, DAGTracerConfig
from simulon.collective import CCLDecomposer
from simulon.config.dc import DatacenterConfig
from simulon.config.resolve import resolve_gpu_spec
from simulon.config.workload import MegatronWorkload

logger = logging.getLogger(__name__)

"""Megatron-LM rank formula and parallelism helpers."""

_COLLECTIVE_GROUPS = None


class RankCoords(NamedTuple):
    """
    See https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe#moe-parallel-folding
    """

    # attention ranks
    tp: int
    cp: int
    dp: int

    # expert ranks
    etp: int
    ep: int
    edp: int

    # shared pp
    pp: int


@dataclass(frozen=True)
class ParallelConfig:
    tp: int
    cp: int
    ep: int
    dp: int
    pp: int
    etp: int
    edp: int
    num_gpus: int
    overlap_p2p_comm: bool = True

    @classmethod
    def from_workload(cls, workload: MegatronWorkload) -> ParallelConfig:
        def get_cfg_int(cfg: ConfigDict, key: str, default: int = None):
            val = cfg.get(key, default)
            return None if val is None else int(val)

        cfg = workload.config
        tp = get_cfg_int(cfg, "tensor-model-parallel-size", 1)
        pp = get_cfg_int(cfg, "pipeline-model-parallel-size", 1)
        ep = get_cfg_int(cfg, "expert-model-parallel-size", 1)
        cp = get_cfg_int(cfg, "context-model-parallel-size", 1)
        etp = get_cfg_int(cfg, "expert-tensor-parallel-size", tp)
        num_gpus = get_cfg_int(cfg, "num-gpus")

        if num_gpus is None:
            raise ValueError("num-gpus must be specified in the workload config")

        non_expert_model_size = tp * cp * pp
        if num_gpus % non_expert_model_size != 0:
            raise ValueError(
                f"num-gpus ({num_gpus}) not divisible by tp*cp*pp ({non_expert_model_size}). "
                f"Config: tp={tp}, cp={cp}, pp={pp}"
            )
        dp = num_gpus // non_expert_model_size

        expert_model_size = etp * ep * pp
        if num_gpus % expert_model_size != 0:
            raise ValueError(
                f"num-gpus ({num_gpus}) not divisible by etp*ep*pp ({expert_model_size}). "
                f"Config: etp={etp}, ep={ep}, pp={pp}"
            )
        edp = num_gpus // expert_model_size

        overlap_p2p_comm = cfg.get("overlap-p2p-comm", True)
        if isinstance(overlap_p2p_comm, str):
            overlap_p2p_comm = overlap_p2p_comm.lower() in ("true", "1", "yes")

        return cls(
            tp=tp,
            cp=cp,
            ep=ep,
            dp=dp,
            pp=pp,
            num_gpus=num_gpus,
            etp=etp,
            edp=edp,
            overlap_p2p_comm=overlap_p2p_comm,
        )

    @property
    def world_size(self) -> int:
        return self.num_gpus

    @property
    def ranks_per_stage(self) -> int:
        return self.num_gpus // self.pp


# Rank conversion helpers


def _global_rank_attention(
    tp_rank: int, cp_rank: int, dp_rank: int, pp_stage: int, config: ParallelConfig
) -> int:
    for name, value, size in (
        ("tp_rank", tp_rank, config.tp),
        ("cp_rank", cp_rank, config.cp),
        ("dp_rank", dp_rank, config.dp),
        ("pp_stage", pp_stage, config.pp),
    ):
        if not (0 <= value < size):
            raise ValueError(f"{name}={value} out of range [0, {size}) for config {config}")

    return (
        tp_rank
        + cp_rank * config.tp
        + dp_rank * config.cp * config.tp
        + pp_stage * config.tp * config.cp * config.dp
    )


def _global_rank_expert(
    etp_rank: int, ep_rank: int, edp_rank: int, pp_stage: int, config: ParallelConfig
) -> int:
    for name, value, size in (
        ("tp_rank", etp_rank, config.etp),
        ("cp_rank", ep_rank, config.ep),
        ("dp_rank", edp_rank, config.edp),
        ("pp_stage", pp_stage, config.pp),
    ):
        if not (0 <= value < size):
            raise ValueError(f"{name}={value} out of range [0, {size}) for config {config}")

    return (
        etp_rank
        + ep_rank * config.etp
        + edp_rank * config.ep * config.etp
        + pp_stage * config.etp * config.ep * config.edp
    )


_global_rank = _global_rank_attention  # backward-compat alias used by tests


def _decompose_rank(rank: int, config: ParallelConfig) -> RankCoords:
    """Convert a global rank back to decomposed coordinates."""
    if not (0 <= rank < config.world_size):
        raise ValueError(f"rank={rank} out of range [0, {config.world_size}) for config {config}")
    tp = rank % config.tp
    cp = (rank // config.tp) % config.cp
    dp = (rank // (config.tp * config.cp)) % config.dp

    etp = rank % config.etp
    ep = (rank // config.etp) % config.ep
    edp = (rank // (config.etp * config.ep)) % config.edp

    pp = rank // config.ranks_per_stage
    return RankCoords(tp=tp, cp=cp, dp=dp, etp=etp, ep=ep, edp=edp, pp=pp)


def _stage_of(rank: int, config: ParallelConfig) -> int:
    """Return the pipeline-parallel stage index for a global rank."""
    return rank // config.ranks_per_stage


def _ranks_for_stage(pp_stage: int, config: ParallelConfig) -> list[int]:
    """Return every global rank that belongs to a given PP stage."""
    start = pp_stage * config.ranks_per_stage
    end = (pp_stage + 1) * config.ranks_per_stage
    return list(range(start, end))


def _get_tp_group_ranks(rank: int, config: ParallelConfig) -> list[int]:
    """Return every global rank in the TP group containing *rank*."""
    coords = _decompose_rank(rank, config)
    return [
        _global_rank_attention(tp, coords.cp, coords.dp, coords.pp, config)
        for tp in range(config.tp)
    ]


def _get_cp_group_ranks(rank: int, config: ParallelConfig) -> list[int]:
    """Return every global rank in the CP group containing *rank*."""
    coords = _decompose_rank(rank, config)
    return [
        _global_rank_attention(coords.tp, cp, coords.dp, coords.pp, config)
        for cp in range(config.cp)
    ]


def _get_dp_group_ranks(rank: int, config: ParallelConfig) -> list[int]:
    """Return every global rank in the DP group containing *rank*."""
    coords = _decompose_rank(rank, config)
    return [
        _global_rank_attention(coords.tp, coords.cp, dp, coords.pp, config)
        for dp in range(config.dp)
    ]


def _get_etp_group_ranks(rank: int, config: ParallelConfig) -> list[int]:
    """Return every global rank in the EP group containing *rank*."""
    coords = _decompose_rank(rank, config)
    return [
        _global_rank_expert(etp, coords.ep, coords.edp, coords.pp, config)
        for etp in range(config.etp)
    ]


def _get_ep_group_ranks(rank: int, config: ParallelConfig) -> list[int]:
    """Return every global rank in the EP group containing *rank*."""
    coords = _decompose_rank(rank, config)
    return [
        _global_rank_expert(coords.etp, ep, coords.edp, coords.pp, config)
        for ep in range(config.ep)
    ]


def _get_edp_group_ranks(rank: int, config: ParallelConfig) -> list[int]:
    """Return every global rank in the EP group containing *rank*."""
    coords = _decompose_rank(rank, config)
    return [
        _global_rank_expert(coords.etp, coords.ep, edp, coords.pp, config)
        for edp in range(config.edp)
    ]


def _make_collective_groups(config: ParallelConfig):
    funcs = [
        _get_tp_group_ranks,
        _get_cp_group_ranks,
        _get_dp_group_ranks,
        _get_etp_group_ranks,
        _get_ep_group_ranks,
        _get_edp_group_ranks,
    ]

    global _COLLECTIVE_GROUPS
    groups: dict[int, list[list[int]]] = {}
    for rank in range(config.world_size):
        groups[rank] = [func(rank, config) for func in funcs]

    _COLLECTIVE_GROUPS = groups


@dataclass(frozen=True)
class _PendingPPTransfer:
    remapped_src: int
    remapped_dst: int
    bytes: int
    microbatch_id: int
    direction: str


def _resolve_traces_dir(datacenter: DatacenterConfig, workload: MegatronWorkload) -> Path:
    """Resolve the directory that contains per-rank trace files."""
    import os

    env_traces_dir = os.environ.get("SIMULON_TRACES_DIR")
    if env_traces_dir:
        return Path(env_traces_dir)
    if workload.traces_dir is not None:
        return Path(workload.traces_dir)
    from simulon.config.resolve import resolve_gpu_spec, workload_hash

    gpu_spec = resolve_gpu_spec(datacenter)
    gpu_name = (gpu_spec.name or "default").lower().replace(" ", "-")
    h = workload_hash(workload)
    p = Path("templates/gpu") / gpu_name / "traces" / h
    if not p.exists():
        raise ValueError(
            f"Traces not found at {p}. "
            "Either set traces_dir in workload or ensure traces exist "
            "in the GPU-specific hashed path."
        )
    return p


def _load_trace_mbs(traces_dir: Path) -> int | None:
    """Read micro-batch-size from the workload.yaml stored in the trace directory."""
    wl_path = traces_dir / "workload.yaml"
    if not wl_path.exists():
        return None
    import yaml
    data = yaml.safe_load(wl_path.read_text())
    cfg = data.get("config", {}) if isinstance(data, dict) else {}
    v = cfg.get("micro-batch-size") or cfg.get("micro_batch_size")
    return int(v) if v is not None else None


def _trace_microbatch_count(traces_dir: Path) -> int | None:
    """Number of microbatches the trace was captured with: ``gbs / (mbs * dp)``.

    Read from the trace directory's own workload.yaml, so it reflects what was
    traced rather than what is being simulated.
    """
    wl_path = traces_dir / "workload.yaml"
    if not wl_path.exists():
        return None
    import yaml
    data = yaml.safe_load(wl_path.read_text())
    cfg = data.get("config", {}) if isinstance(data, dict) else {}

    def _get(*names):
        for n in names:
            v = cfg.get(n)
            if v is not None:
                return int(v)
        return None

    gbs = _get("global-batch-size", "global_batch_size")
    mbs = _get("micro-batch-size", "micro_batch_size")
    num_gpus = _get("num-gpus", "num_gpus")
    tp = _get("tensor-model-parallel-size", "tensor_model_parallel_size") or 1
    pp = _get("pipeline-model-parallel-size", "pipeline_model_parallel_size") or 1
    cp = _get("context-parallel-size", "context_parallel_size") or 1
    if not gbs or not mbs or not num_gpus:
        return None
    dp = num_gpus // (tp * pp * cp)
    if dp <= 0:
        return None
    count = gbs // (mbs * dp)
    return count or None


def _extrapolate_trace_for_mbs(
    trace,
    trace_mbs: int,
    new_mbs: int,
    new_num_microbatches: int,
    keep_step: bool = False,
):
    """Return a scaled copy of *trace* simulating a different micro-batch size.

    Two complementary transforms are applied:
    - All timestamps are scaled by ``new_mbs / trace_mbs`` so that per-microbatch
      compute and TP collective overlap stretch proportionally.
    - Events belonging to microbatch_ids >= *new_num_microbatches* (and the
      optimizer-step slot) are dropped, reducing the pipeline fill to match the
      requested global-batch-size / (new_mbs * dp).

    PP activation-transfer sizes are NOT scaled here; they are computed from the
    workload config by ``_compute_activation_bytes``.  TP collective bytes ARE
    scaled because their size is proportional to mbs.

    The gaps left by filtered-out microbatches in the middle of 1F1B schedules
    naturally model the exposed pipeline-drain bubble — no special treatment needed.
    """
    from simulon.backend.dag.trace_parser import TraceEvent, TraceFile

    scale = new_mbs / trace_mbs

    events_sorted = sorted(trace.events, key=lambda e: e.timestamp_ms)

    in_slot = False
    slot_kept = False
    new_events = []

    for ev in events_sorted:
        if ev.type == "slot_begin":
            mb_id = ev.metadata.get("microbatch_id", -1)
            direction = ev.metadata.get("direction") or ev.metadata.get("phase", "")
            in_slot = True
            # The optimizer step is per ITERATION, not per microbatch: dropping it was
            # harmless for the original mbs-scaling use (its kernels would be mis-scaled)
            # but wrong for a trim-only replay, where it silently removed a fixed cost that
            # matters most exactly where M is small. keep_step=True keeps it unscaled.
            is_step = direction == "step"
            slot_kept = (is_step and keep_step) or (
                not is_step and isinstance(mb_id, int) and 0 <= mb_id < new_num_microbatches)
            if not slot_kept:
                continue

        elif ev.type == "slot_end":
            was_kept = slot_kept
            in_slot = False
            slot_kept = False
            if not was_kept:
                continue

        elif not in_slot:
            # Inter-slot event: PP_Send / PP_Recv tagged with microbatch_id
            mb_id = ev.metadata.get("microbatch_id", None)
            if mb_id is not None:
                try:
                    if not (0 <= int(mb_id) < new_num_microbatches):
                        continue
                except (TypeError, ValueError):
                    pass

        elif not slot_kept:
            continue  # inside a filtered-out slot

        new_meta = dict(ev.metadata)
        if ev.type == "slot_begin":
            # kernel_device_ms is a DURATION, so it has to be scaled with the timestamps.
            # Without this it stayed at its traced value while the slot span was multiplied
            # by `scale`, so _slot_compute_scales returned kd/(span*scale) and the
            # extrapolated compute came out equal to the *traced* mbs's kernel time
            # regardless of the target mbs -- silently, and on every EXTRAP leg scored
            # against a kernel-timing registry.
            #
            # Scaling it per slot while the microbatch COUNT drops by the same factor leaves
            # the iteration's total kernel time roughly invariant in mbs, which is what the
            # hardware shows (10596 ms at mbs1 vs 10802 ms at mbs2, +1.9%, measured by nsys).
            kd = ev.metadata.get("kernel_device_ms")
            if isinstance(kd, (int, float)):
                new_meta["kernel_device_ms"] = kd * scale
            # host_ops, host_ms, launch_count and kernel_count are deliberately NOT scaled.
            # A larger microbatch makes each tensor bigger, not the program longer: the slot
            # runs the same aten ops, so its host-side framework cost is mbs-invariant while
            # its GPU time grows. That asymmetry IS the mechanism -- with the microbatch
            # count halving, total GPU work stays flat while total host work halves, which
            # is what makes larger mbs faster and what the old per-collective constant had
            # to keep re-fitting. See experiments/gap_attribution.py.
        if ev.type == "collective":
            ct = str(ev.metadata.get("collective_type", ""))
            if ct not in ("PP_Send", "PP_Recv"):
                raw_bytes = ev.metadata.get("bytes")
                if raw_bytes is not None:
                    new_meta["bytes"] = int(raw_bytes * scale)

        new_events.append(TraceEvent(
            type=ev.type,
            timestamp_ms=ev.timestamp_ms * scale,
            metadata=new_meta,
        ))

    return TraceFile(
        trace_format_version=trace.trace_format_version,
        rank=trace.rank,
        world_size=trace.world_size,
        pipeline_stage=trace.pipeline_stage,
        events=new_events,
        total_flops=trace.total_flops,  # unchanged: same gbs → same total FLOPs
        energy_kwh=trace.energy_kwh,
        co2eq_kg=trace.co2eq_kg,
    )


def _slot_direction(ev) -> str:
    return str(ev.metadata.get("direction") or ev.metadata.get("phase", ""))


def _split_trace_items(events_sorted: list) -> list:
    """Group a timestamp-ordered event list into ``("slot", [events])`` /
    ``("event", ev)`` items, preserving execution order.

    A slot runs from ``slot_begin`` to its matching ``slot_end``; anything outside
    is a standalone inter-slot event (PP_Send / PP_Recv).
    """
    items: list = []
    cur: list | None = None
    for ev in events_sorted:
        if ev.type == "slot_begin":
            cur = [ev]
        elif ev.type == "slot_end":
            if cur is None:
                items.append(("event", ev))
            else:
                cur.append(ev)
                items.append(("slot", cur))
                cur = None
        elif cur is not None:
            cur.append(ev)
        else:
            items.append(("event", ev))
    if cur is not None:  # unterminated slot — keep rather than drop
        items.append(("slot", cur))
    return items


def _replicate_trace_microbatches(trace, target_num_mb: int):
    """Return a copy of *trace* whose steady state runs *target_num_mb* microbatches.

    The inverse of the trimming done by :func:`_extrapolate_trace_for_mbs`, which can
    only *remove* microbatches. Raising the global batch size raises the microbatch
    count (``gbs / (mbs * dp)``), so simulating a larger GBS than was traced requires
    *adding* steady-state microbatches.

    1F1B is periodic: after the warmup fill (``pp - 1 - stage`` forward passes) the
    schedule is a repeating forward/backward pair, and every microbatch costs the
    same. So the extra microbatches are produced by splicing copies of the **last
    steady-state (forward, backward) pair** back in at the end of the steady region,
    which preserves the warmup depth and the cooldown drain that set the bubble.

    Timestamps of each copy are shifted by the unit's span, and everything after the
    splice point is shifted by ``k * span``, so per-event durations and inter-event
    gaps are reproduced exactly. Microbatch ids are then renumbered per direction in
    execution order, which is what 1F1B replay expects.
    """
    from simulon.backend.dag.trace_parser import TraceEvent, TraceFile

    # Work on copies throughout: the caller's trace objects may be shared between
    # ranks (see _load_or_derive_trace), so renumbering in place would corrupt every
    # rank replayed after this one.
    events_sorted = [
        TraceEvent(type=e.type, timestamp_ms=e.timestamp_ms, metadata=dict(e.metadata))
        for e in sorted(trace.events, key=lambda e: e.timestamp_ms)
    ]
    items = _split_trace_items(events_sorted)

    def _is_mb_slot(item) -> bool:
        return item[0] == "slot" and _slot_direction(item[1][0]) not in ("step", "")

    mb_ids = {
        item[1][0].metadata.get("microbatch_id")
        for item in items
        if _is_mb_slot(item)
    }
    mb_ids.discard(None)
    traced_num_mb = len(mb_ids)
    if traced_num_mb == 0 or target_num_mb <= traced_num_mb:
        return trace

    # Locate the last forward slot and the first backward slot after it: together
    # they form one steady-state period.
    last_fwd = None
    for i, item in enumerate(items):
        if _is_mb_slot(item) and _slot_direction(item[1][0]).startswith("f"):
            last_fwd = i
    if last_fwd is None:
        return trace
    unit_end = None
    for j in range(last_fwd + 1, len(items)):
        if _is_mb_slot(items[j]) and _slot_direction(items[j][1][0]).startswith("b"):
            unit_end = j
            break
    if unit_end is None:
        return trace

    def _dir_key(direction: str) -> str | None:
        if direction.startswith("f"):
            return "f"
        if direction.startswith("b"):
            return "b"
        return None

    # Record each inter-slot PP event's offset from the preceding same-direction
    # slot *in the original ordering*, before any splicing. It has to be measured
    # here rather than recomputed during the renumber walk: after splicing, a copy's
    # leading PP_Recv is preceded by the previous copy's trailing backward slot, so a
    # recomputed delta collapses to 0 and two backward slots end up waiting on the
    # same transfer.
    _anchor: dict[str, int] = {}
    for kind, payload in items:
        if kind == "slot":
            direction = _slot_direction(payload[0])
            d = _dir_key(direction) if direction not in ("step", "") else None
            mb = payload[0].metadata.get("microbatch_id")
            if d is not None and mb is not None:
                _anchor[d] = int(mb)
        else:
            mb = payload.metadata.get("microbatch_id")
            d = _dir_key(_slot_direction(payload))
            if mb is not None and d is not None:
                payload.metadata["_mb_delta"] = int(mb) - _anchor.get(d, -1)

    unit = items[last_fwd : unit_end + 1]

    def _item_events(item):
        return item[1] if item[0] == "slot" else [item[1]]

    # The steady-state period is the stride between consecutive forward slots, NOT
    # the unit's own max-min span: the latter omits the trailing gap between the
    # unit's last event and where the next period begins, which would compress every
    # replicated copy.
    fwd_starts = [
        item[1][0].timestamp_ms
        for item in items
        if _is_mb_slot(item) and _slot_direction(item[1][0]).startswith("f")
    ]
    span = fwd_starts[-1] - fwd_starts[-2] if len(fwd_starts) >= 2 else 0.0
    if span <= 0:
        # Degenerate trace (single forward slot): fall back to the mean period.
        total = max(e.timestamp_ms for e in events_sorted) - min(
            e.timestamp_ms for e in events_sorted
        )
        span = total / max(traced_num_mb, 1)

    k = target_num_mb - traced_num_mb

    def _shift(item, dt):
        if item[0] == "slot":
            return ("slot", [
                TraceEvent(type=e.type, timestamp_ms=e.timestamp_ms + dt, metadata=dict(e.metadata))
                for e in item[1]
            ])
        e = item[1]
        return ("event", TraceEvent(
            type=e.type, timestamp_ms=e.timestamp_ms + dt, metadata=dict(e.metadata)
        ))

    new_items = list(items[: unit_end + 1])
    for c in range(k):
        dt = (c + 1) * span
        new_items.extend(_shift(it, dt) for it in unit)
    new_items.extend(_shift(it, k * span) for it in items[unit_end + 1 :])

    # Renumber microbatch ids per direction in execution order.
    #
    # Standalone PP events sit *around* their slot, not inside it, and which side
    # depends on the event: a PP_Send trails the forward slot whose activations it
    # ships, while a PP_Recv leads the backward slot whose gradients it awaits.
    # Binding both to the last-seen slot mis-assigns every PP_Recv by one microbatch
    # and injects false cross-microbatch dependencies.
    #
    # So instead of hardcoding that asymmetry, preserve each event's own offset:
    # measure its original id relative to the preceding same-direction slot, and
    # reapply that delta after renumbering. Events with no preceding slot of their
    # direction anchor at -1, which makes a leading PP_Recv(mb=0) resolve to 0.
    counters: dict[str, int] = {}
    last_new_by_dir: dict[str, int] = {}

    out: list = []
    for kind, payload in new_items:
        if kind == "slot":
            direction = _slot_direction(payload[0])
            d = _dir_key(direction) if direction not in ("step", "") else None
            if d is not None:
                new_id = counters.get(d, 0)
                counters[d] = new_id + 1
                last_new_by_dir[d] = new_id
                for e in payload:
                    if e.metadata.get("microbatch_id") is not None:
                        e.metadata["microbatch_id"] = new_id
            out.extend(payload)
        else:
            e = payload
            delta = e.metadata.pop("_mb_delta", None)
            if delta is not None:
                d = _dir_key(_slot_direction(e))
                if d is not None:
                    e.metadata["microbatch_id"] = last_new_by_dir.get(d, -1) + delta
            out.append(e)

    out.sort(key=lambda e: e.timestamp_ms)
    return TraceFile(
        trace_format_version=trace.trace_format_version,
        rank=trace.rank,
        world_size=trace.world_size,
        pipeline_stage=trace.pipeline_stage,
        events=out,
        # More microbatches at the same mbs ⇒ proportionally more FLOPs.
        total_flops=(
            None if trace.total_flops is None
            else int(trace.total_flops * target_num_mb / traced_num_mb)
        ),
        energy_kwh=trace.energy_kwh,
        co2eq_kg=trace.co2eq_kg,
    )


def _compute_activation_bytes(workload: MegatronWorkload) -> int:
    """Fallback activation bytes for PP transfers without explicit 'bytes'."""
    cfg = workload.config
    seq_len = int(cfg.get("seq-length", 2048))
    micro_bs = int(cfg.get("micro-batch-size", 1))
    hidden_size = int(cfg.get("hidden-size", 0))
    dtype_str = str(cfg.get("dtype", "bf16")).lower()
    dtype_bytes = 4 if dtype_str == "fp32" else 1 if dtype_str == "fp8" else 2
    return seq_len * micro_bs * hidden_size * dtype_bytes


def _traced_topology(traces_dir: Path) -> tuple[int | None, int | None]:
    """(world_size, pp) the trace directory was captured at, from its workload.yaml."""
    wl_path = traces_dir / "workload.yaml"
    if not wl_path.exists():
        return (None, None)
    import yaml
    data = yaml.safe_load(wl_path.read_text())
    cfg = data.get("config", {}) if isinstance(data, dict) else {}

    def _get(*names):
        for n in names:
            v = cfg.get(n)
            if v is not None:
                return int(v)
        return None

    return (
        _get("num-gpus", "num_gpus"),
        _get("pipeline-model-parallel-size", "pipeline_model_parallel_size") or 1,
    )


def _traces_by_stage(traces_dir: Path, config: ParallelConfig) -> dict[int, Path]:
    """Map PP stage -> an available trace file, independent of the simulated scale.

    Trace files are named by ABSOLUTE rank in the world the trace was generated at
    (e.g. trace_rank_32.json is stage 1 of PP=2 on 64 GPUs). Deriving the stage from
    the *simulated* world size therefore breaks as soon as the node count differs
    from the traced one: at 256 GPUs the same stage starts at rank 128, and the
    lookup fails with "Last PP stage trace missing" even though a perfectly good
    trace for that stage is present.

    Per-GPU work is scale-invariant for fixed TP/PP/mbs, so a stage's trace is
    reusable at any node count — we just have to resolve it by STAGE, using the
    traced topology to interpret the filenames.
    """
    traced_world, traced_pp = _traced_topology(traces_dir)
    if not traced_world or not traced_pp:
        traced_world, traced_pp = config.world_size, config.pp
    ranks_per_stage = max(1, traced_world // max(1, traced_pp))

    by_stage: dict[int, Path] = {}
    for path in sorted(traces_dir.glob("trace_rank_*.json")):
        try:
            r = int(path.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        stage = min(r // ranks_per_stage, traced_pp - 1)
        by_stage.setdefault(stage, path)
    return by_stage


def _stage_has_exact_trace(pp_stage: int, traces_dir: Path, config: ParallelConfig) -> bool:
    """Return True if a trace exists for this PP stage (at any traced scale)."""
    if any(
        (traces_dir / f"trace_rank_{r}.json").exists() for r in _ranks_for_stage(pp_stage, config)
    ):
        return True
    return pp_stage in _traces_by_stage(traces_dir, config)


def _load_first_traced_rank_in_stage(pp_stage: int, traces_dir: Path, config: ParallelConfig):
    """Return (trace, rank) of a trace for this stage, resolved scale-independently.

    The returned rank is remapped into the SIMULATED world (first rank of the stage)
    so that downstream collective remapping works against the current topology.
    """
    for r in _ranks_for_stage(pp_stage, config):
        path = traces_dir / f"trace_rank_{r}.json"
        if path.exists():
            return (TraceFileParser.parse(str(path)), r)

    by_stage = _traces_by_stage(traces_dir, config)
    path = by_stage.get(pp_stage)
    if path is not None:
        # Present it as the stage's first rank in the CURRENT world so that
        # _remap_collectives rebuilds the groups against this topology.
        local_rank = _ranks_for_stage(pp_stage, config)[0]
        return (TraceFileParser.parse(str(path)), local_rank)

    if by_stage:  # any stage at all (matches the old middle-stage fallback)
        stage, path = sorted(by_stage.items())[0]
        return (TraceFileParser.parse(str(path)), _ranks_for_stage(pp_stage, config)[0])

    raise ValueError(f"No trace files found in {traces_dir}")


def _traced_parallel_config(traces_dir: Path, fallback: ParallelConfig) -> ParallelConfig:
    """Reconstruct the ParallelConfig the trace was captured under."""
    wl_path = traces_dir / "workload.yaml"
    if not wl_path.exists():
        return fallback
    import yaml
    data = yaml.safe_load(wl_path.read_text())
    cfg = data.get("config", {}) if isinstance(data, dict) else {}

    def _g(*names, default=None):
        for n in names:
            v = cfg.get(n)
            if v is not None:
                return int(v)
        return default

    num_gpus = _g("num-gpus", "num_gpus")
    if not num_gpus:
        return fallback
    tp = _g("tensor-model-parallel-size", "tensor_model_parallel_size", default=1) or 1
    pp = _g("pipeline-model-parallel-size", "pipeline_model_parallel_size", default=1) or 1
    cp = _g("context-parallel-size", "context_parallel_size", default=1) or 1
    dp = max(1, num_gpus // (tp * pp * cp))
    return ParallelConfig(tp=tp, cp=cp, ep=fallback.ep, dp=dp, pp=pp,
                          etp=fallback.etp, edp=fallback.edp, num_gpus=num_gpus)


def _build_groups_for(config: ParallelConfig) -> dict[int, list[list[int]]]:
    funcs = [
        _get_tp_group_ranks, _get_cp_group_ranks, _get_dp_group_ranks,
        _get_etp_group_ranks, _get_ep_group_ranks, _get_edp_group_ranks,
    ]
    return {r: [f(r, config) for f in funcs] for r in range(config.world_size)}


# Groups of the world the TRACE was captured in. Needed whenever the simulated scale
# differs from the traced scale: a traced collective must be identified by its GROUP
# TYPE (tp / cp / dp / ...) in the traced world, then re-formed as the corresponding
# group in the simulated world. Matching by rank membership alone silently fails at a
# different node count — nothing matches, the stale traced rank list survives, and DP
# collectives are never rebuilt for the new dp.
_TRACED_COLLECTIVE_GROUPS = None
_TRACED_CONFIG = None


def _remap_collective(group: list[int], from_rank: int, to_rank: int):
    group = set(group)
    for from_group, to_group in zip(
        _COLLECTIVE_GROUPS[from_rank], _COLLECTIVE_GROUPS[to_rank], strict=False
    ):
        if set(from_group) == group:
            return to_group
    # Different simulated scale: identify the group TYPE in the traced world, then
    # return the same type's group for to_rank in the simulated world.
    if _TRACED_COLLECTIVE_GROUPS is not None:
        for src_rank, traced_groups in _TRACED_COLLECTIVE_GROUPS.items():
            for idx, traced_group in enumerate(traced_groups):
                if set(traced_group) == group:
                    cur = _COLLECTIVE_GROUPS.get(to_rank, [])
                    if idx < len(cur) and cur[idx]:
                        return cur[idx]
            del src_rank
    return None


def _remap_collectives(source_trace, from_rank: int, to_rank: int, config: ParallelConfig):
    """Return a new TraceFile with collectives remapped for *to_rank*."""
    new_events = []
    for ev in source_trace.events:
        if ev.type != "collective":
            new_events.append(ev)
            continue
        ct = str(ev.metadata.get("collective_type", ""))
        group_ranks_raw = ev.metadata.get("group_ranks", [])
        group_ranks = list(group_ranks_raw) if isinstance(group_ranks_raw, list | tuple) else []

        if ct in ("PP_Send", "PP_Recv"):
            if _TRACED_COLLECTIVE_GROUPS is not None and len(group_ranks) >= 2:
                # Scale-out: the traced group_ranks are ranks in the TRACED world, so
                # shifting them by a constant delta lands on the wrong peers (stage
                # stride differs between worlds). Rebuild the pair from the SIMULATED
                # topology: same TP/CP/DP coordinates, adjacent PP stage.
                coords = _decompose_rank(to_rank, config)
                # Direction matters: in the FORWARD pass activations flow up the
                # pipeline (Send -> stage+1, Recv <- stage-1); in the BACKWARD pass
                # gradients flow back down (Send -> stage-1, Recv <- stage+1).
                # Assuming forward-only happens to work at PP=2 (a single neighbour)
                # but wires cycles into the DAG at PP>=4 -> "Topo sort incomplete".
                _d = str(ev.metadata.get("direction") or ev.metadata.get("phase", ""))
                backward = _d.startswith("b")
                step = -1 if backward else 1
                nbr = coords.pp + step if ct == "PP_Send" else coords.pp - step
                if 0 <= nbr < config.pp:
                    peer = _global_rank_attention(coords.tp, coords.cp, coords.dp, nbr, config)
                    new_group = [to_rank, peer] if ct == "PP_Send" else [peer, to_rank]
                    new_events.append(type(ev)(
                        type=ev.type,
                        timestamp_ms=ev.timestamp_ms,
                        metadata={**ev.metadata, "group_ranks": new_group},
                    ))
                    continue
                new_events.append(ev)
                continue
            if len(group_ranks) >= 2:
                delta = to_rank - from_rank
                new_group = [group_ranks[0] + delta, group_ranks[1] + delta]
                if all(0 <= g < config.world_size for g in new_group):
                    new_ev = type(ev)(
                        type=ev.type,
                        timestamp_ms=ev.timestamp_ms,
                        metadata={**ev.metadata, "group_ranks": new_group},
                    )
                    new_events.append(new_ev)
                    continue
            new_events.append(ev)
            continue

        new_group = _remap_collective(group_ranks, from_rank, to_rank)
        if new_group is not None:
            new_ev = type(ev)(
                type=ev.type,
                timestamp_ms=ev.timestamp_ms,
                metadata={**ev.metadata, "group_ranks": new_group},
            )
            new_events.append(new_ev)
        else:
            new_events.append(ev)
    from simulon.backend.dag.trace_parser import TraceFile

    return TraceFile(
        trace_format_version=source_trace.trace_format_version,
        rank=to_rank,
        world_size=source_trace.world_size,
        pipeline_stage=_stage_of(to_rank, config),
        events=new_events,
        total_flops=source_trace.total_flops,
        energy_kwh=source_trace.energy_kwh,
        co2eq_kg=source_trace.co2eq_kg,
    )


def _virtual_chunks(cfg: dict, pp: int) -> int:
    """Virtual pipeline chunks per stage: from an explicit layout, or from VPP, else 1."""
    layout = cfg.get("pipeline-model-parallel-layout", cfg.get("pipeline_model_parallel_layout"))
    if layout:
        return max(1, len(str(layout).replace("\\", "").split("|")) // max(1, pp))
    nlvps = cfg.get("num-layers-per-virtual-pipeline-stage",
                    cfg.get("num_layers_per_virtual_pipeline_stage"))
    n_layers = int(cfg.get("num-layers", cfg.get("num_layers", 0)) or 0)
    if nlvps and n_layers:
        return max(1, n_layers // (max(1, pp) * int(nlvps)))
    return 1


def _pipeline_layout_groups(layout: str) -> list[tuple[int, bool, bool]]:
    """Parse --pipeline-model-parallel-layout into [(n_layers, has_embedding, has_loss)].

    Megatron's syntax is a "|"-separated list of pipeline groups, each a concatenation of
    stage items: E (input embedding), t / t*N (N transformer layers), L (loss + output head).
    Configs often carry it backslash-escaped ("Et\\*5\\|t\\*4"), so strip those first.
    """
    out: list[tuple[int, bool, bool]] = []
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
        out.append((n, "E" in grp, "L" in grp))
    return out


def _stage_param_count(workload: MegatronWorkload, config: ParallelConfig, stage: int) -> int:
    """Parameters physically held by one pipeline stage (before the TP split).

    MEASURED, not assumed (job 1854038, the first capture with the DP collectives
    instrumented): the per-stage parameter count derived here reproduces the recorded
    all-gather bytes to +0.01% on every stage of every cell, while the previous uniform
    N/(TP*PP) was off by -7.9% to +13.7% on the production 16-chunk layout -- stage 0 owns
    the input embedding plus 17 layers there, stage 3 owns 15 layers plus the output head,
    and the stages in between own 16. With PP=2 and untied embeddings the two stages happen
    to tie, which is why the uniform form looked right until a layout was tested.

    Interleaved chunk assignment follows Megatron's rule: chunk c of stage s is layout group
    s + c*pp (confirmed against the measured per-stage bytes above).
    """
    from simulon.backend.memory import ModelDims, embedding_param_count, layer_param_count
    cfg = workload.config
    d = ModelDims.from_config(cfg)
    layer_p, emb_p = layer_param_count(d), embedding_param_count(d)
    pp = max(1, config.pp)
    layout = cfg.get("pipeline-model-parallel-layout", cfg.get("pipeline_model_parallel_layout"))
    if layout:
        groups = _pipeline_layout_groups(str(layout))
        mine = [groups[i] for i in range(stage, len(groups), pp)]
    else:
        # uniform split; input embedding on the first stage, output head on the last
        mine = [(d.num_layers // pp, stage == 0, stage == pp - 1)]
    layers = sum(m[0] for m in mine)
    embeds = sum(int(m[1]) + (0 if d.tie_embeddings else int(m[2])) for m in mine)
    return layers * layer_p + embeds * emb_p


def _dp_grad_sync_bytes(workload: MegatronWorkload, config: ParallelConfig,
                        stage: int) -> tuple[int, int]:
    """(grad reduce-scatter bytes, param all-gather bytes) for one rank of *stage*.

    Distributed optimizer: gradients are reduce-scattered over the DP group in fp32 unless
    --grad-reduce-in-bf16; the updated bf16 params (fp8 with --fp8-param-gather) are
    all-gathered back. Full-buffer sizes, which is what nccl-tests calls "size" and what the
    tracer records (param_and_grad_buffer._record_dp_collective).
    """
    cfg = workload.config
    per_rank = _stage_param_count(workload, config, stage) // config.tp
    grad_bytes = 2 if cfg.get("grad-reduce-in-bf16", cfg.get("grad_reduce_in_bf16", False)) else 4
    param_bytes = 1 if cfg.get("fp8-param-gather", cfg.get("fp8_param_gather", False)) else 2
    return per_rank * grad_bytes, per_rank * param_bytes


def _inject_dp_grad_sync(trace, rank: int, config: ParallelConfig, workload: MegatronWorkload):
    """Add the distributed-optimizer DP collectives a DP=1 capture cannot contain.

    A real-process-group trace is captured at DP=1 (the full TP x PP group and nothing else),
    so its optimizer-step slot holds no collectives: the grad reduce-scatter and param
    all-gather are no-ops at DP=1. Replaying such a trace at DP>1 without this step models
    ZERO data-parallel communication -- the dominant scale-out cost. Caught by
    experiments/ground.py: at 512 nodes / PP1 the sim was -61% with 241 ms of exposed comm.

    Placement follows Megatron's schedule, and is now MEASURED (job 1854038, TP4/PP2/DP4):
      * overlap ON  -- the buffers are BUCKETED and the buckets are spread across the phase:
                       64 ReduceScatter events landing inside backward slots and 64
                       AllGather events inside forward slots, one iteration, ~265 MB and
                       ~132 MB each. So the whole backward phase hides the grad reduce, not
                       just its last slot; this synthesizes one async collective per slot
                       carrying an equal share of the total, which is that structure at the
                       granularity the DAG has. A trailing zero-byte synchronous marker at
                       the step makes the optimizer wait for the last one.
      * overlap OFF -- one single coalesced collective each, at the end: the measured trace
                       has exactly 1 ReduceScatter between the last backward and the step,
                       and 1 AllGather inside the step slot.
    Explicit async_op on these events bypasses the SP-dependent inference.
    """
    from simulon.backend.dag.trace_parser import TraceEvent
    if config.dp <= 1:
        return trace
    coords = _decompose_rank(rank, config)
    group = [_global_rank_attention(coords.tp, coords.cp, d, coords.pp, config) for d in range(config.dp)]
    rs_bytes, ag_bytes = _dp_grad_sync_bytes(workload, config, coords.pp)
    cfg = workload.config
    ogr = bool(cfg.get("overlap-grad-reduce", cfg.get("overlap_grad_reduce", False)))
    opg = bool(cfg.get("overlap-param-gather", cfg.get("overlap_param_gather", False)))

    ev = sorted(trace.events, key=lambda e: e.timestamp_ms)
    slot_begins = [e for e in ev if e.type == "slot_begin"]
    steps = [e for e in slot_begins if e.metadata.get("direction") == "step"]
    bwds = [e for e in slot_begins if e.metadata.get("direction") == "bwd"]
    fwds = [e for e in slot_begins if e.metadata.get("direction") == "fwd"]
    t_end = ev[-1].timestamp_ms if ev else 0.0
    eps = 0.001
    if steps:
        t_step = steps[0].timestamp_ms
        step_end = next((e.timestamp_ms for e in ev if e.type == "slot_end" and e.timestamp_ms > t_step), t_end)
    else:
        t_step = step_end = t_end + eps

    def mk(ct, name, nbytes, t, async_op, direction):
        return TraceEvent("collective", t, {
            "name": name, "collective_type": ct, "bytes": int(nbytes), "group_ranks": list(group),
            "microbatch_id": None, "direction": direction, "async_op": async_op, "synthesized": "dp_grad_sync"})

    def spread(ct, name, total, slots, direction):
        """One async collective per slot, each carrying an equal share of the buffer."""
        n = len(slots)
        share, rem = divmod(int(total), n)
        return [mk(ct, name, share + (1 if i < rem else 0), s.timestamp_ms + eps, True, direction)
                for i, s in enumerate(slots)]

    new = []
    if ogr and bwds:
        new += spread("ReduceScatter", "dist_opt_grad_reduce_scatter", rs_bytes, bwds, "bwd")
        # the optimizer cannot start until the last bucket has landed
        new.append(mk("ReduceScatter", "dist_opt_grad_reduce_wait", 0, t_step + eps, False, "step"))
    else:
        new.append(mk("ReduceScatter", "dist_opt_grad_reduce_scatter", rs_bytes, t_step + eps, False, "step"))
    if opg and fwds:
        new += spread("AllGather", "dist_opt_param_all_gather", ag_bytes, fwds, "fwd")
    else:
        new.append(mk("AllGather", "dist_opt_param_all_gather", ag_bytes, step_end + eps, False, "step"))
    trace.events = sorted(ev + new, key=lambda e: e.timestamp_ms)
    return trace


def _load_or_derive_trace(rank: int, traces_dir: Path, config: ParallelConfig, stage_traces: dict):
    """Load exact trace, or derive from a sibling rank in the same PP stage."""
    exact_path = traces_dir / f"trace_rank_{rank}.json"
    if exact_path.exists():
        trace = TraceFileParser.parse(str(exact_path))
        stage = _stage_of(rank, config)
        if stage not in stage_traces:
            stage_traces[stage] = trace
        return trace
    stage = _stage_of(rank, config)
    if stage in stage_traces:
        src_trace = stage_traces[stage]
        src_rank = src_trace.rank
        return _remap_collectives(src_trace, src_rank, rank, config)
    src_trace, src_rank = _load_first_traced_rank_in_stage(stage, traces_dir, config)
    stage_traces[stage] = src_trace
    return _remap_collectives(src_trace, src_rank, rank, config)


def _process_slot_begin(
    event,
    rank: int,
    active_microbatch_id: list,
    active_direction: list,
    slot_node_ids: list,
    slot_first_timestamp: dict,
) -> None:
    active_microbatch_id[0] = event.metadata.get("microbatch_id", -1)
    raw_phase = event.metadata.get("phase", "")
    active_phase = str(raw_phase)
    if active_phase == "fwd":
        active_direction[0] = "fwd"
    elif active_phase in ("bwd", "bwd_ig", "bwd_wg"):
        active_direction[0] = "bwd"
    else:
        active_direction[0] = str(
            event.metadata.get("direction") or event.metadata.get("slot") or active_phase
        )
    slot_node_ids.clear()
    key = (rank, active_microbatch_id[0], active_direction[0])
    if key not in slot_first_timestamp:
        slot_first_timestamp[key] = event.timestamp_ms


def _process_slot_end(
    event,
    rank: int,
    active_microbatch_id: list,
    active_direction: list,
    slot_node_ids: list,
    slot_nodes: dict,
    slot_entry_node: dict,
    slot_last_node: dict,
    slot_last_timestamp: dict,
) -> None:
    if slot_node_ids:
        key = (rank, active_microbatch_id[0], active_direction[0])
        slot_nodes.setdefault(key, []).extend(slot_node_ids)
        slot_entry_node[key] = slot_node_ids[0]
        slot_last_node[key] = slot_node_ids[-1]
        slot_last_timestamp[key] = event.timestamp_ms
    slot_node_ids.clear()


def _add_compute_node(
    dag: ExecutionDAG,
    rank: int,
    config: ParallelConfig,
    duration_ms: float,
    microbatch_id: int,
    direction: str,
    node_id: list,
    last_node_by_rank: dict[int, CollectiveNode | ComputeNode],
    slot_node_ids: list,
    host_ops: float = 0.0,
    kernel_ct: float = 0.0,
) -> None:
    cn = ComputeNode(
        node_id=node_id[0],
        gpu_rank=rank,
        kernel="compute",
        layer_id=-1,
        microbatch_id=microbatch_id,
        pipeline_stage=_stage_of(rank, config),
        phase=direction,
        duration_ms=duration_ms,
        host_ops=host_ops,
        kernel_ct=kernel_ct,
    )
    dag.add_compute_node(cn)
    dag.rank_program.setdefault(rank, []).append(node_id[0])
    slot_node_ids.append(node_id[0])
    if rank in last_node_by_rank:
        dag.add_edge(DAGEdge(src_node_id=last_node_by_rank[rank].node_id, dst_node_id=node_id[0]))
    last_node_by_rank[rank] = cn
    node_id[0] += 1


def _localize_to_global_ranks(local_group: list[int], rank: int) -> list[int] | None:
    """Translate 0-indexed local communicator ranks to global ranks for *rank*.

    Traces sometimes record intra-communicator local ranks (0, 1, 2, 3) rather
    than global GPU ranks.  When *rank* is not in *local_group* and the group
    looks like {0..n-1}, find the matching process-group for *rank* by size and
    return its global ranks in the same relative order.
    """
    n = len(local_group)
    if rank in local_group or set(local_group) != set(range(n)):
        return None
    for global_group in _COLLECTIVE_GROUPS.get(rank, []):
        if len(global_group) == n:
            sorted_global = sorted(global_group)
            return [sorted_global[i] for i in local_group]
    return None


def _add_non_pp_collective(
    dag: ExecutionDAG,
    event,
    rank: int,
    active_microbatch_id: int,
    direction: str,
    node_id: list,
    last_node_by_rank: dict[int, CollectiveNode | ComputeNode],
    slot_node_ids: list,
    tracer_cfg: DAGTracerConfig,
    _collective_registry: dict,
    last_async_collective_by_rank: dict[int, CollectiveNode | ComputeNode] | None = None,
) -> None:
    collective_type = str(event.metadata.get("collective_type", ""))
    group_ranks_raw = event.metadata.get("group_ranks", [])
    group_ranks = list(group_ranks_raw) if isinstance(group_ranks_raw, list | tuple) else []
    data_size = int(event.metadata.get("bytes", 0))
    name = str(event.metadata.get("name", ""))
    timestamp_ms = float(event.timestamp_ms)

    if len(group_ranks) < 2:
        return

    # Traces may record local (0-indexed) communicator ranks instead of global
    # ranks.  Translate to global ranks so collectives from different PP stages
    # that happen to share the same local ranks don't collapse into one node.
    global_ranks = _localize_to_global_ranks(group_ranks, rank)
    if global_ranks is not None:
        group_ranks = global_ranks

    match_key = (collective_type, frozenset(group_ranks), name, round(timestamp_ms, 3), data_size)

    collective = _collective_registry.get(match_key)
    if collective is not None:
        collective_id = collective.node_id
    else:
        collective = CollectiveNode(
            node_id=node_id[0],
            collective_type=collective_type,
            group_ranks=group_ranks,
            data_size=data_size,
            name=name,
            timestamp_ms=timestamp_ms,
            layer_id=-1,
            phase=direction,
            algorithm=tracer_cfg.algorithm,
            num_channels=tracer_cfg.num_channels,
        )
        dag.add_collective_node(collective)
        _collective_registry[match_key] = collective
        collective_id = node_id[0]
        node_id[0] += 1

    # Every participant issues the collective from its own host thread, so it belongs in
    # each rank's program -- including when it is async below, because an async collective
    # still costs the host thread the time to issue it even though the GPU does not block.
    dag.rank_program.setdefault(rank, []).append(collective_id)

    if rank in last_node_by_rank:
        dag.add_edge(
            DAGEdge(src_node_id=last_node_by_rank[rank].node_id, dst_node_id=collective_id)
        )

    if rank in (last_async_collective_by_rank or {}):
        dag.add_edge(
            DAGEdge(
                src_node_id=last_async_collective_by_rank[rank].node_id, dst_node_id=collective_id
            )
        )

    # The tracer never recorded async_op (always None in existing traces), so infer
    # it structurally: with sequence_parallel=false the TP collective is AllReduce, so
    # ReduceScatter / AllGather come ONLY from the distributed optimizer (grad
    # reduce-scatter + param all-gather). Those run on a NCCL side stream and overlap
    # compute on the real hardware even with overlap_grad_reduce/param_gather off (the
    # val40 runs measured PP=1 hiding ~90% of this comm). Only honored when the caller
    # opts in via overlap_async_collectives; default False leaves every other sim
    # untouched. NOTE: assumes SP=false (the only regime we validate) — under SP=true
    # RS/AG would be TP collectives and must NOT be treated as async.
    async_op = event.metadata.get("async_op")
    if async_op is None:
        if name.startswith(("dist_opt_", "ddp_grad_")):
            # A distributed-optimizer collective, recorded by
            # param_and_grad_buffer._record_dp_collective. Whether it is asynchronous is
            # decided by the run's overlap flags, NOT by the SP rule below: under SP=true
            # the TP collectives are ReduceScatter/AllGather too and are synchronous, so
            # applying that rule here left the DP traffic fully exposed. Measured on the
            # DP=4 capture (job 1854038): the sim exposed 2074 ms with the overlap flags ON
            # against 2443 ms with them OFF, while the two measured runs differ by 1.5%.
            # Captures from 2026-09-17 on record async_op directly and skip this branch.
            async_op = (tracer_cfg.overlap_grad_reduce if collective_type == "ReduceScatter"
                        else tracer_cfg.overlap_param_gather)
        else:
            # Traced RS/AG are the dist-optimizer's only when SP is off; under SP=true they
            # are TP collectives and must stay synchronous.
            async_op = (collective_type in ("ReduceScatter", "AllGather")
                        and not tracer_cfg.sequence_parallel)
    else:
        async_op = bool(async_op)
    if async_op and tracer_cfg.overlap_async_collectives:
        # Async collectives launch on a separate CUDA stream and don't block
        # subsequent compute on this rank. Keep the predecessor edge (collective
        # waits for prior work) but don't make this collective the new "last
        # node" — the next compute node will chain from the previous one instead.
        # Track it separately so the next collective still waits for this one.
        if last_async_collective_by_rank is not None:
            last_async_collective_by_rank[rank] = collective
        return

    if last_async_collective_by_rank is not None:
        last_async_collective_by_rank.pop(rank, None)
    last_node_by_rank[rank] = collective
    slot_node_ids.append(collective_id)


def _add_pp_transfer(
    event,
    rank: int,
    active_microbatch_id: int,
    active_direction: str,
    activation_bytes: int,
    pending: list,
) -> None:
    group_ranks_raw = event.metadata.get("group_ranks", [])
    group_ranks = list(group_ranks_raw) if isinstance(group_ranks_raw, list | tuple) else []
    direction = str(event.metadata.get("direction", active_direction))
    if direction == "fwd":
        traced_src = group_ranks[0] if len(group_ranks) > 0 else -1
        traced_dst = group_ranks[1] if len(group_ranks) > 1 else -1
    elif direction == "bwd":
        traced_src = group_ranks[1] if len(group_ranks) > 1 else -1
        traced_dst = group_ranks[0] if len(group_ranks) > 0 else -1
    else:
        traced_src = group_ranks[0] if len(group_ranks) > 0 else -1
        traced_dst = group_ranks[1] if len(group_ranks) > 1 else -1
    data_size = int(
        event.metadata.get("bytes") if event.metadata.get("bytes") is not None else activation_bytes
    )
    _mb = event.metadata.get("microbatch_id")
    pp_mb = int(_mb) if _mb is not None else active_microbatch_id
    pending.append(
        _PendingPPTransfer(
            remapped_src=traced_src,
            remapped_dst=traced_dst,
            bytes=data_size,
            microbatch_id=pp_mb,
            direction=direction,
        )
    )


def _handle_event_gap(
    dag: ExecutionDAG,
    event,
    next_event,
    rank: int,
    config: ParallelConfig,
    active_microbatch_id: list,
    active_direction: list,
    node_id: list,
    flow_id: list,
    activation_bytes: int,
    tracer_cfg: DAGTracerConfig,
    slot_node_ids: list,
    pending_pp_transfers: list,
    last_node_by_rank: dict[int, CollectiveNode | ComputeNode],
    _collective_registry: dict,
    last_async_collective_by_rank: dict[int, CollectiveNode | ComputeNode] | None = None,
    flops_multiplier: float = 1.0,
    compute_scale: float = 1.0,
    host_ops_rate: float = 0.0,
    kernel_rate: float = 0.0,
) -> None:
    # compute_scale rescales the wall-clock gap to measured per-kernel DEVICE time when
    # the trace carries kernel_device_ms (see _slot_compute_scales). It is 1.0 for legacy
    # traces (no kernel timing) and for collective-transfer gaps, so behavior is
    # unchanged unless the kernel-timed field is present. It multiplies COMPUTE gaps only;
    # collective durations come from the comm model, never the trace wall-clock.
    raw_gap_ms = next_event.timestamp_ms - event.timestamp_ms
    duration_ms = raw_gap_ms / flops_multiplier * compute_scale
    # Ops are apportioned from the RAW gap, deliberately: op count is a property of the
    # traced program, not of the rescaled device time. Rescaling it too would make the host
    # term shrink whenever the GPU term does, which is exactly the coupling that has to be
    # broken for the mbs and TP axes to come out.
    host_ops = raw_gap_ms * host_ops_rate if host_ops_rate else 0.0
    kernel_ct = raw_gap_ms * kernel_rate if kernel_rate else 0.0
    # A negative/zero raw gap is a malformed trace; skip it. But do NOT use duration_ms for
    # this test: with kernel timing, gaps OUTSIDE any slot are deliberately given
    # compute_scale = 0 (see below), and returning here would silently drop the collective
    # that the gap leads into.
    if raw_gap_ms <= 0:
        return
    if duration_ms < 0:
        duration_ms = 0.0
    if event.type == "collective":
        ct = str(event.metadata.get("collective_type", ""))
    if event.type == "slot_begin":
        _add_compute_node(
            dag,
            rank,
            config,
            duration_ms,
            active_microbatch_id[0],
            active_direction[0],
            node_id,
            last_node_by_rank,
            slot_node_ids,
            host_ops,
            kernel_ct,
        )
    elif event.type == "collective":
        ct = str(event.metadata.get("collective_type", ""))
        if ct in ("PP_Send", "PP_Recv"):
            _add_pp_transfer(
                event,
                rank,
                active_microbatch_id[0],
                active_direction[0],
                activation_bytes,
                pending_pp_transfers,
            )
        elif len(event.metadata.get("group_ranks", [])) > 1:
            _add_non_pp_collective(
                dag,
                event,
                rank,
                active_microbatch_id[0],
                active_direction[0],
                node_id,
                last_node_by_rank,
                slot_node_ids,
                tracer_cfg,
                _collective_registry,
                last_async_collective_by_rank,
            )
            _add_compute_node(
                dag,
                rank,
                config,
                duration_ms,
                active_microbatch_id[0],
                active_direction[0],
                node_id,
                last_node_by_rank,
                slot_node_ids,
                host_ops,
                kernel_ct,
            )
    elif event.type != "slot_end":
        _add_compute_node(
            dag,
            rank,
            config,
            duration_ms,
            active_microbatch_id[0],
            active_direction[0],
            node_id,
            last_node_by_rank,
            slot_node_ids,
            host_ops,
            kernel_ct,
        )


def _slot_compute_scales(events) -> dict[int, float]:
    """Per-slot factor that turns the trace's wall-clock compute into measured per-kernel
    DEVICE time. For each slot carrying metadata.kernel_device_ms (from a trace generated
    with SIMULON_TRACE_KERNEL_TIME=1), scale = kernel_device_ms / (slot_end - slot_begin).
    Applied to every compute gap inside the slot, it makes that slot's total compute equal
    the pure kernel time while preserving the intra-slot shape. Returns {} for legacy
    traces (no kernel timing) => callers fall back to the untouched wall-clock spans.

    WHY: the fake-PG wall-clock span pads each kernel with a CPU-launch idle gap that does
    not shrink as larger mbs/lower TP make the GEMMs more tensor-core efficient; the pure
    kernel device time does. This wires the measured efficiency in with no fitted constant.
    """
    scales: dict[int, float] = {}
    open_idx: int | None = None
    open_ts = 0.0
    kd: float | None = None
    for i, e in enumerate(events):
        if e.type == "slot_begin":
            open_idx, open_ts = i, e.timestamp_ms
            v = e.metadata.get("kernel_device_ms")
            kd = float(v) if isinstance(v, (int, float)) else None
        elif e.type == "slot_end" and open_idx is not None:
            span = e.timestamp_ms - open_ts
            if kd is not None and span > 0:
                scales[open_idx] = kd / span
            open_idx, kd = None, None
    return scales


def _slot_kernel_rates(events) -> dict[int, float]:
    """Per-slot KERNELS-per-wall-ms, for slots carrying metadata.kernel_count.

    Feeds the bounded launch queue in replay(): the host can only run ahead of the GPU by a
    fixed number of pending kernel launches, so the replay needs to know how many launches
    each compute node represents. Same apportioning as _slot_host_rates.
    """
    rates: dict[int, float] = {}
    open_idx: int | None = None
    open_ts = 0.0
    kc: float | None = None
    for i, e in enumerate(events):
        if e.type == "slot_begin":
            open_idx, open_ts = i, e.timestamp_ms
            v = e.metadata.get("kernel_count")
            kc = float(v) if isinstance(v, (int, float)) else None
        elif e.type == "slot_end" and open_idx is not None:
            span = e.timestamp_ms - open_ts
            if kc is not None and span > 0:
                rates[open_idx] = kc / span
            open_idx, kc = None, None
    return rates


def _slot_host_rates(events) -> dict[int, float]:
    """Per-slot framework-OPS-per-wall-ms, for slots carrying metadata.host_ops.

    Mirrors _slot_compute_scales but for the OTHER resource. Expressed as a rate so the ops
    can be spread over the slot's compute gaps the same way compute_scale is, which keeps
    intra-slot collective issue times in the right place rather than charging the whole
    slot's host cost at its first node.

    OPS, NOT THE MEASURED host_ms. The tracer records both, and host_ms (the union of the
    slot's cpu_op intervals) looks like the more direct measurement -- but those records
    include time spent BLOCKED on the GPU, so wherever the GPU binds the union saturates
    toward the wall-clock span. Measured on job 1224555: 19.9 us/op at mbs1 rising to 53.1
    at mbs4, with host_ms reaching 88% of span. Using it would make host time grow with GPU
    time, destroying the very decoupling the two-resource replay exists to model. The op
    count has no such contamination -- it came back invariant across TP (8995/9146/9141 at
    TP=1/2/4) and across mbs (8995/9033/9069 at mbs=1/2/4) and scaling as 1/PP, exactly as
    the program structure demands. Cost per op comes from NodeSpec.host_cost_us.

    Returns {} for traces without host counts => host modelling is simply unavailable, and
    replay() refuses rather than silently modelling zero (see its host_cost_us docs).
    """
    rates: dict[int, float] = {}
    open_idx: int | None = None
    open_ts = 0.0
    hm: float | None = None
    for i, e in enumerate(events):
        if e.type == "slot_begin":
            open_idx, open_ts = i, e.timestamp_ms
            v = e.metadata.get("host_ops")
            hm = float(v) if isinstance(v, (int, float)) else None
        elif e.type == "slot_end" and open_idx is not None:
            span = e.timestamp_ms - open_ts
            if hm is not None and span > 0:
                rates[open_idx] = hm / span
            open_idx, hm = None, None
    return rates


def _add_trace_to_dag(
    dag: ExecutionDAG,
    trace,
    rank: int,
    config: ParallelConfig,
    node_id: list,
    flow_id: list,
    activation_bytes: int,
    tracer_cfg: DAGTracerConfig,
    slot_nodes: dict,
    slot_entry_node: dict,
    slot_last_node: dict,
    slot_first_timestamp: dict,
    slot_last_timestamp: dict,
    pending_pp_transfers: list,
    last_node_by_rank: dict[int, CollectiveNode | ComputeNode],
    _collective_registry: dict,
    last_async_collective_by_rank: dict[int, CollectiveNode | ComputeNode],
    flops_multiplier: float = 1.0,
) -> None:
    events = sorted(trace.events, key=lambda e: e.timestamp_ms)
    active_microbatch_id: list = [-1]
    active_direction: list = [""]
    slot_node_ids: list[int] = []
    slot_scales = _slot_compute_scales(events)  # {} unless trace has kernel_device_ms
    slot_host_rates = _slot_host_rates(events)  # {} unless trace has host_ops
    slot_kernel_rates = _slot_kernel_rates(events)  # {} unless trace has kernel_count
    compute_scale = 1.0
    host_ops_rate = 0.0
    kernel_rate = 0.0
    for i in range(len(events)):
        event = events[i]
        if event.type == "slot_begin":
            compute_scale = slot_scales.get(i, 1.0)
            host_ops_rate = slot_host_rates.get(i, 0.0)
            kernel_rate = slot_kernel_rates.get(i, 0.0)
            _process_slot_begin(
                event,
                rank,
                active_microbatch_id,
                active_direction,
                slot_node_ids,
                slot_first_timestamp,
            )
        elif event.type == "slot_end":
            # Gaps OUTSIDE a slot contribute no compute when the trace carries kernel
            # timing. Under a fake process group that time is negligible, but under a REAL
            # one (--trace-real-process-group) the space between slots is where the rank
            # sits waiting on collectives and P2P -- booking it as compute at wall-clock
            # rate inflated tp1pp1-mbs1 from 8986 ms of measured kernel time to 10217 ms,
            # i.e. +6.7% -> +18.2% against hardware. Kernel time that genuinely falls
            # outside every slot is reported separately as unattributed_kernel_device_ms,
            # which check_trace_health.py already gates at 5%.
            compute_scale = 0.0 if slot_scales else 1.0
            host_ops_rate = 0.0
            kernel_rate = 0.0
            _process_slot_end(
                event,
                rank,
                active_microbatch_id,
                active_direction,
                slot_node_ids,
                slot_nodes,
                slot_entry_node,
                slot_last_node,
                slot_last_timestamp,
            )
        if i + 1 < len(events):
            _handle_event_gap(
                dag,
                event,
                events[i + 1],
                rank,
                config,
                active_microbatch_id,
                active_direction,
                node_id,
                flow_id,
                activation_bytes,
                tracer_cfg,
                slot_node_ids,
                pending_pp_transfers,
                last_node_by_rank,
                _collective_registry,
                last_async_collective_by_rank,
                flops_multiplier,
                compute_scale,
                host_ops_rate,
                kernel_rate,
            )


def _wire_slot_edges(dag: ExecutionDAG, slot_nodes: dict) -> None:
    with log_progress("  wiring slot edges", len(slot_nodes), logger) as advance:
        for node_ids in slot_nodes.values():
            for i in range(len(node_ids) - 1):
                dag.add_edge(DAGEdge(src_node_id=node_ids[i], dst_node_id=node_ids[i + 1]))
            advance()


def _should_skip_pp_pair(src_stage: int, dst_stage: int, direction: str) -> bool:
    return (direction == "fwd" and src_stage > dst_stage) or (
        direction == "bwd" and src_stage < dst_stage
    )


def _compute_next_slot_by_key(slot_first_timestamp: dict) -> dict:
    keys_by_rank: dict[int, list] = {}
    for key in slot_first_timestamp:
        keys_by_rank.setdefault(key[0], []).append(key)
    next_slot: dict = {}
    for rank_keys in keys_by_rank.values():
        rank_keys.sort(key=lambda k: slot_first_timestamp[k])
        for i in range(len(rank_keys) - 1):
            next_slot[rank_keys[i]] = rank_keys[i + 1]
    return next_slot


def _create_pp_send(
    src: int,
    dst: int,
    record,
    src_node: int,
    dst_node: int,
    node_id: list,
    flow_id: list,
    dag: ExecutionDAG,
    slot_entry_node: dict,
    next_slot_by_key: dict | None,
    sync_send: bool,
    dst_prev_node: int | None = None,
) -> None:
    pp_send = CommNode(
        node_id=node_id[0],
        src_gpu=src,
        dst_gpu=dst,
        bytes=record.bytes,
        collective_type="PP_Send",
        layer_id=-1,
        phase=record.direction,
        flow_id=flow_id[0],
    )
    dag.add_comm_node(pp_send)
    # Both peers issue this transfer from their own host thread, so it belongs in both
    # programs -- and with sync_send it is where the host stops running ahead (below).
    for _r in (src, dst):
        dag.rank_program.setdefault(_r, []).append(pp_send.node_id)
    dag.add_edge(DAGEdge(src_node_id=src_node, dst_node_id=pp_send.node_id))
    dag.add_edge(DAGEdge(src_node_id=pp_send.node_id, dst_node_id=dst_node))
    if sync_send and next_slot_by_key:
        src_key = (src, record.microbatch_id, record.direction)
        next_key = next_slot_by_key.get(src_key)
        if next_key:
            next_node = slot_entry_node.get(next_key)
            if next_node is not None:
                dag.add_edge(DAGEdge(src_node_id=pp_send.node_id, dst_node_id=next_node))
    # In synchronous P2P mode, the destination rank must finish its current
    # compute before it can post the recv. Without this edge the PP_Send
    # (which represents the combined send+recv) can be scheduled concurrently
    # with compute on the destination rank, which is physically impossible
    # when batch_p2p_comm=True (the GPU is blocked on the recv call).
    if sync_send and dst_prev_node is not None:
        dag.add_edge(DAGEdge(src_node_id=dst_prev_node, dst_node_id=pp_send.node_id))
    if sync_send:
        # The recv is BLOCKING, so the host thread on both peers stops here and cannot keep
        # issuing work behind it. Recording that is what stops the two-resource replay from
        # letting the host run ahead across the whole iteration -- which made it report a
        # perfectly-pipelined schedule (PP 1->2 sim +3.2% vs hardware +45.9%).
        dag.host_sync_nodes.add(pp_send.node_id)
    node_id[0] += 1
    flow_id[0] += 1


def _resolve_pp_nodes(
    src: int, dst: int, record, slot_entry_node: dict, slot_last_node: dict
) -> tuple[int | None, int | None]:
    src_key = (src, record.microbatch_id, record.direction)
    dst_key = (dst, record.microbatch_id, record.direction)
    src_node = slot_last_node.get(src_key)
    dst_node = slot_entry_node.get(dst_key)
    if dst_node is None and record.direction == "bwd":
        for bwd_phase in ("bwd_ig", "bwd_wg"):
            alt_key = (dst, record.microbatch_id, bwd_phase)
            if alt_key in slot_entry_node:
                dst_node = slot_entry_node[alt_key]
                break
    return src_node, dst_node


def _wire_pp_transfers(
    dag: ExecutionDAG,
    pending: list,
    config: ParallelConfig,
    slot_entry_node: dict,
    slot_last_node: dict,
    node_id: list,
    flow_id: list,
    next_slot_by_key: dict | None,
    sync_send: bool,
) -> tuple[int, int]:
    prev_slot_by_key: dict = {}
    if sync_send and next_slot_by_key:
        prev_slot_by_key = {v: k for k, v in next_slot_by_key.items()}
    seen: set[tuple[int, int, int, str]] = set()
    with log_progress("  wiring PP transfers", len(pending), logger) as advance:
        for record in pending:
            pp_stride = config.ranks_per_stage
            src_stage = record.remapped_src // pp_stride
            dst_stage = record.remapped_dst // pp_stride
            if src_stage == dst_stage:
                advance()
                continue
            if not (
                0 <= record.remapped_src < config.world_size
                and 0 <= record.remapped_dst < config.world_size
            ):
                advance()
                continue
            src_ranks = _ranks_for_stage(src_stage, config)
            dst_ranks = _ranks_for_stage(dst_stage, config)
            for src, dst in zip(src_ranks, dst_ranks, strict=False):
                dedup = (src, dst, record.microbatch_id, record.direction)
                if dedup in seen:
                    continue
                seen.add(dedup)
                src_node, dst_node = _resolve_pp_nodes(
                    src, dst, record, slot_entry_node, slot_last_node
                )
                if src_node is None or dst_node is None:
                    continue
                if _should_skip_pp_pair(src_stage, dst_stage, record.direction):
                    continue
                dst_prev_node: int | None = None
                if sync_send and prev_slot_by_key:
                    dst_key = (dst, record.microbatch_id, record.direction)
                    if dst_node is not None and dst_key in slot_entry_node:
                        prev_key = prev_slot_by_key.get(dst_key)
                        if prev_key is not None:
                            dst_prev_node = slot_last_node.get(prev_key)
                _create_pp_send(
                    src,
                    dst,
                    record,
                    src_node,
                    dst_node,
                    node_id,
                    flow_id,
                    dag,
                    slot_entry_node,
                    next_slot_by_key,
                    sync_send,
                    dst_prev_node,
                )
            advance()
    return node_id[0], flow_id[0]


def _wire_cross_slot_edges(
    dag: ExecutionDAG, slot_first_timestamp: dict, slot_last_node: dict, slot_entry_node: dict
) -> None:
    keys_by_rank: dict[int, list] = {}
    for key in slot_first_timestamp:
        keys_by_rank.setdefault(key[0], []).append(key)
    with log_progress("  wiring cross-slot edges", len(keys_by_rank), logger) as advance:
        for _, keys in keys_by_rank.items():
            keys.sort(key=lambda k: slot_first_timestamp[k])
            for i in range(len(keys) - 1):
                prev_last = slot_last_node.get(keys[i])
                next_first = slot_entry_node.get(keys[i + 1])
                if prev_last is not None and next_first is not None:
                    dag.add_edge(DAGEdge(src_node_id=prev_last, dst_node_id=next_first))
            advance()


def _wire_bwd_to_step(
    dag: ExecutionDAG, slot_last_node: dict, slot_entry_node: dict, config: ParallelConfig
) -> None:
    with log_progress("  wiring bwd-to-step", config.world_size, logger) as advance:
        for rank in range(config.world_size):
            bwd_keys = [k for k in slot_last_node if k[0] == rank and k[2] == "bwd"]
            if bwd_keys:
                last_bwd_key = max(bwd_keys, key=lambda k: k[1])
                step_key = (rank, 0, "step")
                if step_key in slot_entry_node:
                    dag.add_edge(
                        DAGEdge(
                            src_node_id=slot_last_node[last_bwd_key],
                            dst_node_id=slot_entry_node[step_key],
                        )
                    )
            advance()


class MegatronDagTracer(DAGTracer):
    cfg: DAGTracerConfig
    ccl: CCLDecomposer

    def __init__(self, cfg: DAGTracerConfig, ccl: CCLDecomposer):
        self.cfg = cfg
        self.ccl = ccl

    def trace(self, workload: MegatronWorkload, datacenter: DatacenterConfig) -> ExecutionDAG:
        dag = ExecutionDAG()
        config = ParallelConfig.from_workload(workload)
        _make_collective_groups(config)
        traces_dir = _resolve_traces_dir(datacenter, workload)

        # Scale-out support: when the simulated world differs from the traced one,
        # build the traced world's collective groups too, so a traced collective can
        # be identified by group TYPE and re-formed for the simulated topology.
        global _TRACED_COLLECTIVE_GROUPS, _TRACED_CONFIG
        _TRACED_CONFIG = _traced_parallel_config(traces_dir, config)
        if _TRACED_CONFIG.world_size != config.world_size:
            logger.info(
                "  Scale-out replay: traced world=%d GPUs (dp=%d) -> simulated world=%d GPUs (dp=%d)",
                _TRACED_CONFIG.world_size, _TRACED_CONFIG.dp, config.world_size, config.dp,
            )
            _TRACED_COLLECTIVE_GROUPS = _build_groups_for(_TRACED_CONFIG)
        else:
            _TRACED_COLLECTIVE_GROUPS = None
        activation_bytes = _compute_activation_bytes(workload)
        flops_multiplier = resolve_gpu_spec(datacenter).flops_multiplier

        # MBS extrapolation: if the requested mbs differs from the traced mbs,
        # scale compute timings and TP collective sizes, and trim to the correct
        # number of microbatches.  PP transfer sizes are always recomputed from
        # the workload config (via activation_bytes), so no adjustment is needed.
        new_mbs = int(workload.config.get("micro-batch-size", 1))
        gbs = int(workload.config.get("global-batch-size", 0))
        new_num_microbatches = (gbs // (new_mbs * config.dp)) if gbs and new_mbs and config.dp else None
        # An INTERLEAVED schedule records one slot per (microbatch, model chunk) and keys it
        # by the VIRTUAL microbatch id, which runs 0..n_mb*chunks-1. Trimming and replication
        # below filter on that id, so they must be told the virtual count -- otherwise a
        # 4-chunk trace looks like it has 4x too many microbatches and three quarters of its
        # slots are silently dropped.
        _chunks = _virtual_chunks(workload.config, config.pp)
        if new_num_microbatches is not None and _chunks > 1:
            new_num_microbatches *= _chunks
        trace_mbs = _load_trace_mbs(traces_dir)
        if trace_mbs is None and new_mbs != 1:
            # Without the traced mbs there is no way to know whether extrapolation is
            # needed, and the old behaviour was to skip it silently -- returning the
            # TRACED mbs's iteration time for whatever mbs was asked for. That is
            # invisible in the result and fatal to an mbs sweep: every point comes back
            # equal to the anchor. Real-PG captures hit this because workload.yaml is
            # written by `simulon trace` (cli/trace.py), which a direct pretrain_gpt.py
            # run bypasses.
            raise ValueError(
                f"{traces_dir} has no workload.yaml, so its micro-batch-size is unknown, "
                f"but micro-batch-size={new_mbs} was requested. Refusing to guess: "
                f"without it the traced mbs is silently returned instead. Write a "
                f"workload.yaml into the trace directory (see simulon/cli/trace.py)."
            )
        _mbs_scale_needed = (
            trace_mbs is not None
            and trace_mbs != new_mbs
            and new_num_microbatches is not None
            and new_num_microbatches > 0
        )
        if _mbs_scale_needed:
            logger.info(
                "MBS extrapolation: trace_mbs=%d → new_mbs=%d "
                "(scale=%.2f, microbatches %d→%d)",
                trace_mbs, new_mbs, new_mbs / trace_mbs,
                gbs // (trace_mbs * config.dp), new_num_microbatches,
            )

        # Microbatch replication: the mbs extrapolation above can only *trim*
        # microbatches, so a target GBS larger than the traced one needs the steady
        # state extended instead. Decoupling trace GBS from simulated GBS makes GBS
        # a free axis (one GBS=256 trace serves GBS=4096).
        traced_num_microbatches = _trace_microbatch_count(traces_dir)
        _replicate_needed = (
            new_num_microbatches is not None
            and traced_num_microbatches is not None
            and new_num_microbatches > traced_num_microbatches
        )
        _allow_pp_replication = os.environ.get("SIMULON_UNSAFE_PP_REPLICATION") == "1"
        if _replicate_needed and config.pp > 1 and not _allow_pp_replication:
            # KNOWN DEFECT — do not silently return wrong numbers.
            # PP=1 replication is validated (cost is exactly linear in microbatch
            # count: predicted 128,698 ms vs actual 128,707 at GBS=4096). For PP>1
            # the replicated pipeline leaks ~7.5 ms of idle per added microbatch, so
            # the bubble GROWS with microbatch count instead of staying at its
            # fill+drain value: pp=2 bubble 642 ms @32mb → 1,360 ms @128mb, where it
            # should stay ~constant. Suspected cause is microbatch-id remapping of
            # the standalone PP_Send/PP_Recv events (a PP_Recv precedes the forward
            # slot it feeds, so last-seen-id attribution is off by one), which
            # creates cross-microbatch dependencies that serialise the pipeline.
            raise ValueError(
                f"Microbatch replication is not yet correct for pp>1 (pp={config.pp}): "
                f"the pipeline bubble grows with microbatch count. Trace has "
                f"{traced_num_microbatches} microbatches, simulation needs "
                f"{new_num_microbatches}. Either regenerate the trace at the target "
                f"global-batch-size, or simulate at a gbs whose microbatch count is "
                f"<= the traced one."
            )
        if _replicate_needed:
            logger.info(
                "Microbatch replication: %d → %d microbatches (steady-state splice)",
                traced_num_microbatches, new_num_microbatches,
            )

        # Trim-only: same mbs as the trace but FEWER microbatches per DP rank (a bigger DP
        # group or a smaller GBS than the capture). Until 2026-09-16 trimming only ran inside
        # the mbs-extrapolation branch, so a trace captured at M=64 replayed all 64 microbatches
        # for a run that needed 8 -- gbs=64 and gbs=256 returned the identical iteration time
        # (experiments/ground.py caught it: 16n TP4/PP2 at gbs64 came out +389%).
        _trim_only = (
            not _mbs_scale_needed
            and new_num_microbatches is not None
            and new_num_microbatches > 0
            and traced_num_microbatches is not None
            and new_num_microbatches < traced_num_microbatches
        )
        if _trim_only:
            logger.info(
                "Microbatch trim: %d → %d microbatches (same mbs, smaller GBS/DP share)",
                traced_num_microbatches, new_num_microbatches,
            )

        # Validate first and last stage traces exist
        if not _stage_has_exact_trace(0, traces_dir, config):
            raise ValueError(
                f"First PP stage (0) trace missing. "
                f"Expected one of: {[traces_dir / f'trace_rank_{r}.json' for r in _ranks_for_stage(0, config)]}"
            )
        if not _stage_has_exact_trace(config.pp - 1, traces_dir, config):
            raise ValueError(
                f"Last PP stage ({config.pp - 1}) trace missing. "
                f"Expected one of: {[traces_dir / f'trace_rank_{r}.json' for r in _ranks_for_stage(config.pp - 1, config)]}"
            )

        stage_traces: dict = {}
        slot_nodes: dict = {}
        slot_entry_node: dict = {}
        slot_last_node: dict = {}
        slot_first_timestamp: dict = {}
        slot_last_timestamp: dict = {}
        pending_pp_transfers: list = []
        last_node_by_rank: dict[int, CollectiveNode | ComputeNode] = {}
        last_async_collective_by_rank: dict[int, CollectiveNode | ComputeNode] = {}

        node_id = [0]
        flow_id = [0]

        total_energy_kwh = 0.0
        total_co2eq_kg = 0.0

        with log_progress("  building DAG", config.world_size, logger) as advance:
            _collective_registry: dict = {}
            for rank in range(config.world_size):
                trace = _load_or_derive_trace(rank, traces_dir, config, stage_traces)
                if _replicate_needed:
                    trace = _replicate_trace_microbatches(trace, new_num_microbatches)
                if _mbs_scale_needed:
                    trace = _extrapolate_trace_for_mbs(
                        trace, trace_mbs, new_mbs, new_num_microbatches
                    )
                elif _trim_only:
                    _m = trace_mbs or new_mbs   # scale 1.0: drop microbatches only
                    trace = _extrapolate_trace_for_mbs(trace, _m, _m, new_num_microbatches,
                                                       keep_step=True)
                if (
                    config.dp > 1
                    and _TRACED_CONFIG is not None
                    and _TRACED_CONFIG.dp == 1
                ):
                    trace = _inject_dp_grad_sync(trace, rank, config, workload)
                exact_path = traces_dir / f"trace_rank_{rank}.json"
                if exact_path.exists():
                    dag.profiled_ranks.add(rank)
                # total_flops in the trace is a WHOLE-WORLD quantity for the traced
                # topology. Simulating a different node count must rescale it, or MFU
                # is wrong by the world-size ratio (observed: 33.7% -> 7.7% going 16n
                # -> 64n purely from this). Throughput (tok/s/GPU) is unaffected.
                if (
                    dag.total_flops is None
                    and trace.total_flops is not None
                    and _TRACED_CONFIG is not None
                    and _TRACED_CONFIG.world_size
                    and _TRACED_CONFIG.world_size != config.world_size
                ):
                    dag.total_flops = trace.total_flops * (
                        config.world_size / _TRACED_CONFIG.world_size
                    )
                elif dag.total_flops is None and trace.total_flops is not None:
                    dag.total_flops = trace.total_flops
                if trace.energy_kwh is not None:
                    total_energy_kwh += trace.energy_kwh
                if trace.co2eq_kg is not None:
                    total_co2eq_kg += trace.co2eq_kg

                _add_trace_to_dag(
                    dag,
                    trace,
                    rank,
                    config,
                    node_id,
                    flow_id,
                    activation_bytes,
                    self.cfg,
                    slot_nodes,
                    slot_entry_node,
                    slot_last_node,
                    slot_first_timestamp,
                    slot_last_timestamp,
                    pending_pp_transfers,
                    last_node_by_rank,
                    _collective_registry,
                    last_async_collective_by_rank,
                    flops_multiplier,
                )
                advance()

        if total_energy_kwh > 0:
            dag.energy_kwh = total_energy_kwh
        if total_co2eq_kg > 0:
            dag.co2eq_kg = total_co2eq_kg

        _wire_slot_edges(dag, slot_nodes)
        next_slot_by_key = _compute_next_slot_by_key(slot_first_timestamp)
        node_id[0], flow_id[0] = _wire_pp_transfers(
            dag,
            pending_pp_transfers,
            config,
            slot_entry_node,
            slot_last_node,
            node_id,
            flow_id,
            next_slot_by_key=next_slot_by_key,
            sync_send=not config.overlap_p2p_comm,
        )
        _wire_cross_slot_edges(dag, slot_first_timestamp, slot_last_node, slot_entry_node)
        _wire_bwd_to_step(dag, slot_last_node, slot_entry_node, config)
        return dag
