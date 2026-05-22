from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, cast

from simulon.backend.dag.nodes import (
    CollectiveNode,
    ComputeNode,
    CommNode,
    DAGEdge,
    ExecutionDAG,
)
from simulon.backend.dag._progress import log_progress
from simulon.backend.dag.trace_parser import TraceFileParser
from simulon.backend.dag.tracer import DAGTracer, DAGTracerConfig
from simulon.collective import CCLDecomposer
from simulon.collective.decompose import decompose_collective
from simulon.config.dc import DatacenterConfig
from simulon.config.workload import MegatronWorkload

logger = logging.getLogger(__name__)

"""Megatron-LM rank formula and parallelism helpers."""


class RankCoords(NamedTuple):
    """Decomposed parallelism coordinates (tp, cp, ep, dp, pp)."""

    tp: int
    cp: int
    ep: int
    dp: int
    pp: int


@dataclass(frozen=True)
class ParallelConfig:
    """Immutable parallelism dimensions."""

    tp: int
    cp: int
    ep: int
    dp: int
    pp: int
    num_gpus: int

    def __post_init__(self) -> None:
        if self.tp < 1 or self.cp < 1 or self.ep < 1 or self.dp < 1 or self.pp < 1:
            raise ValueError(
                "All parallelism dimensions must be >= 1, got "
                f"tp={self.tp}, cp={self.cp}, ep={self.ep}, dp={self.dp}, pp={self.pp}"
            )
        expected = self.tp * self.cp * self.ep * self.dp * self.pp
        if expected != self.num_gpus:
            raise ValueError(
                f"Parallelism product ({expected}) != num_gpus ({self.num_gpus}). "
                f"Config: tp={self.tp}, cp={self.cp}, ep={self.ep}, dp={self.dp}, pp={self.pp}"
            )

    @classmethod
    def from_workload(cls, workload: MegatronWorkload) -> "ParallelConfig":
        cfg = workload.config
        tp = int(cfg.get("tensor-model-parallel-size", 1))
        pp = int(cfg.get("pipeline-model-parallel-size", 1))
        ep = int(cfg.get("expert-model-parallel-size", 1))
        cp = 1  # hardcoded — see note above
        num_gpus = int(cfg.get("num_gpus", cfg.get("num-gpus", tp * pp * ep)))
        dp = max(1, num_gpus // (tp * pp * ep))
        return cls(tp=tp, cp=cp, ep=ep, dp=dp, pp=pp, num_gpus=num_gpus)

    @property
    def world_size(self) -> int:
        return self.num_gpus

    @property
    def ranks_per_stage(self) -> int:
        return self.tp * self.cp * self.ep * self.dp


# Rank conversion helpers


def _global_rank(
    tp_rank: int,
    cp_rank: int,
    ep_rank: int,
    dp_rank: int,
    pp_stage: int,
    config: ParallelConfig,
) -> int:
    """Compute the global rank from decomposed parallelism coordinates."""
    for name, value, size in (
        ("tp_rank", tp_rank, config.tp),
        ("cp_rank", cp_rank, config.cp),
        ("ep_rank", ep_rank, config.ep),
        ("dp_rank", dp_rank, config.dp),
        ("pp_stage", pp_stage, config.pp),
    ):
        if not (0 <= value < size):
            raise ValueError(
                f"{name}={value} out of range [0, {size}) for config {config}"
            )
    return (
        tp_rank
        + cp_rank * config.tp
        + ep_rank * config.tp * config.cp
        + dp_rank * config.tp * config.cp * config.ep
        + pp_stage * config.tp * config.cp * config.ep * config.dp
    )


def _decompose_rank(rank: int, config: ParallelConfig) -> RankCoords:
    """Convert a global rank back to decomposed ``(tp, cp, ep, dp, pp)`` coordinates."""
    if not (0 <= rank < config.world_size):
        raise ValueError(
            f"rank={rank} out of range [0, {config.world_size}) for config {config}"
        )
    tp = rank % config.tp
    cp = (rank // config.tp) % config.cp
    ep = (rank // (config.tp * config.cp)) % config.ep
    dp = (rank // (config.tp * config.cp * config.ep)) % config.dp
    pp = rank // config.ranks_per_stage
    return RankCoords(tp=tp, cp=cp, ep=ep, dp=dp, pp=pp)


def _stage_of(rank: int, config: ParallelConfig) -> int:
    """Return the pipeline-parallel stage index for a global rank."""
    return rank // config.ranks_per_stage


def _ranks_for_stage(pp_stage: int, config: ParallelConfig) -> list[int]:
    """Return every global rank that belongs to a given PP stage."""
    ranks: list[int] = []
    for dp_rank in range(config.dp):
        for ep_rank in range(config.ep):
            for cp_rank in range(config.cp):
                for tp_rank in range(config.tp):
                    ranks.append(
                        _global_rank(
                            tp_rank, cp_rank, ep_rank, dp_rank, pp_stage, config
                        )
                    )
    return ranks


# Process-group membership helpers


def _ranks_in_same_dp_group(
    group_ranks: Iterable[int], reference_rank: int, config: ParallelConfig
) -> bool:
    """Return ``True`` iff *group_ranks* is exactly the DP group of *reference_rank*."""
    return set(group_ranks) == set(_get_dp_group_ranks(reference_rank, config))


def _ranks_in_same_ep_group(
    group_ranks: Iterable[int], reference_rank: int, config: ParallelConfig
) -> bool:
    """Return ``True`` iff *group_ranks* is exactly the EP group of *reference_rank*."""
    return set(group_ranks) == set(_get_ep_group_ranks(reference_rank, config))


def _ranks_in_same_tp_group(
    group_ranks: Iterable[int], reference_rank: int, config: ParallelConfig
) -> bool:
    """Return ``True`` iff *group_ranks* is exactly the TP group of *reference_rank*."""
    return set(group_ranks) == set(_get_tp_group_ranks(reference_rank, config))


def _ranks_in_same_cp_group(
    group_ranks: Iterable[int], reference_rank: int, config: ParallelConfig
) -> bool:
    """Return ``True`` iff *group_ranks* is exactly the CP group of *reference_rank*."""
    return set(group_ranks) == set(_get_cp_group_ranks(reference_rank, config))


# Process-group builders


def _get_dp_group_ranks(rank: int, config: ParallelConfig) -> list[int]:
    """Return every global rank in the DP group containing *rank*."""
    coords = _decompose_rank(rank, config)
    return [
        _global_rank(coords.tp, coords.cp, coords.ep, dp, coords.pp, config)
        for dp in range(config.dp)
    ]


def _get_ep_group_ranks(rank: int, config: ParallelConfig) -> list[int]:
    """Return every global rank in the EP group containing *rank*."""
    coords = _decompose_rank(rank, config)
    return [
        _global_rank(coords.tp, coords.cp, ep, coords.dp, coords.pp, config)
        for ep in range(config.ep)
    ]


def _get_tp_group_ranks(rank: int, config: ParallelConfig) -> list[int]:
    """Return every global rank in the TP group containing *rank*."""
    coords = _decompose_rank(rank, config)
    return [
        _global_rank(tp, coords.cp, coords.ep, coords.dp, coords.pp, config)
        for tp in range(config.tp)
    ]


def _get_cp_group_ranks(rank: int, config: ParallelConfig) -> list[int]:
    """Return every global rank in the CP group containing *rank*."""
    coords = _decompose_rank(rank, config)
    return [
        _global_rank(coords.tp, cp, coords.ep, coords.dp, coords.pp, config)
        for cp in range(config.cp)
    ]


@dataclass(frozen=True)
class _PendingPPTransfer:
    remapped_src: int
    remapped_dst: int
    bytes: int
    microbatch_id: int
    direction: str


def _resolve_traces_dir(
    datacenter: DatacenterConfig, workload: MegatronWorkload
) -> Path:
    """Resolve the directory that contains per-rank trace files."""
    traces_dir = (
        datacenter.datacenter.traces_dir
        if datacenter and datacenter.datacenter
        else None
    )
    if traces_dir is not None:
        return Path(traces_dir)
    from simulon.config.resolve import workload_hash, resolve_gpu_spec

    try:
        gpu_spec = resolve_gpu_spec(datacenter, include_profile=False)
        gpu_name = (gpu_spec.name or "default").lower().replace(" ", "-")
    except Exception:
        gpu_name = "default"
    h = workload_hash(workload)
    p = Path("templates/gpu") / gpu_name / "traces" / h
    if not p.exists():
        raise ValueError(
            f"Traces not found at {p}. "
            "Either set traces_dir in datacenter.datacenter or ensure traces exist "
            "in the GPU-specific hashed path."
        )
    return p


def _compute_activation_bytes(workload: MegatronWorkload) -> int:
    """Fallback activation bytes for PP transfers without explicit 'bytes'."""
    cfg = workload.config
    seq_len = int(cfg.get("seq-length", 2048))
    micro_bs = int(cfg.get("micro-batch-size", 1))
    hidden_size = int(cfg.get("hidden-size", 0))
    dtype_str = str(cfg.get("dtype", "bf16")).lower()
    dtype_bytes = 4 if dtype_str == "fp32" else 1 if dtype_str == "fp8" else 2
    return seq_len * micro_bs * hidden_size * dtype_bytes


def _stage_has_exact_trace(
    pp_stage: int, traces_dir: Path, config: ParallelConfig
) -> bool:
    """Return True if any rank in the stage has a trace_rank_{rank}.json file."""
    return any(
        (traces_dir / f"trace_rank_{r}.json").exists()
        for r in _ranks_for_stage(pp_stage, config)
    )


def _load_exact_trace(rank: int, traces_dir: Path):
    """Load trace_rank_{rank}.json from traces_dir."""
    path = traces_dir / f"trace_rank_{rank}.json"
    if not path.exists():
        raise ValueError(f"Trace file not found: {path}")
    return TraceFileParser.parse(str(path))


def _load_first_traced_rank_in_stage(
    pp_stage: int, traces_dir: Path, config: ParallelConfig
):
    """Return (trace, rank) of the first exact trace found in the stage."""
    for r in _ranks_for_stage(pp_stage, config):
        path = traces_dir / f"trace_rank_{r}.json"
        if path.exists():
            return (TraceFileParser.parse(str(path)), r)
    # Fallback: any exact trace anywhere (matches old middle-stage fallback).
    for rank in range(config.world_size):
        path = traces_dir / f"trace_rank_{rank}.json"
        if path.exists():
            return (TraceFileParser.parse(str(path)), rank)
    raise ValueError(f"No trace files found in {traces_dir}")


def _remap_collectives(
    source_trace, from_rank: int, to_rank: int, config: ParallelConfig
):
    """Return a new TraceFile with collectives remapped for *to_rank*."""
    new_events = []
    for ev in source_trace.events:
        if ev.type != "collective":
            new_events.append(ev)
            continue
        ct = str(ev.metadata.get("collective_type", ""))
        group_ranks_raw = ev.metadata.get("group_ranks", [])
        group_ranks = (
            list(group_ranks_raw) if isinstance(group_ranks_raw, (list, tuple)) else []
        )
        ev_name = str(ev.metadata.get("name", ""))

        if ct in ("PP_Send", "PP_Recv"):
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

        new_group = None
        if _ranks_in_same_dp_group(group_ranks, from_rank, config):
            new_group = _get_dp_group_ranks(to_rank, config)
        elif _ranks_in_same_ep_group(group_ranks, from_rank, config):
            new_group = _get_ep_group_ranks(to_rank, config)
        elif _ranks_in_same_tp_group(group_ranks, from_rank, config):
            new_group = _get_tp_group_ranks(to_rank, config)
        elif _ranks_in_same_cp_group(group_ranks, from_rank, config):
            new_group = _get_cp_group_ranks(to_rank, config)
        elif (
            "DistributedDataParallel" in ev_name
            or "Distributed_DataParallel" in ev_name
        ):
            new_group = _get_dp_group_ranks(to_rank, config)

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
    )


def _load_or_derive_trace(
    rank: int,
    traces_dir: Path,
    config: ParallelConfig,
    stage_traces: dict,
):
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
            event.metadata.get("direction")
            or event.metadata.get("slot")
            or active_phase
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
    )
    dag.add_compute_node(cn)
    slot_node_ids.append(node_id[0])
    if rank in last_node_by_rank:
        dag.add_edge(
            DAGEdge(src_node_id=last_node_by_rank[rank].node_id, dst_node_id=node_id[0])
        )
    last_node_by_rank[rank] = cn
    node_id[0] += 1


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
) -> None:
    collective_type = str(event.metadata.get("collective_type", ""))
    group_ranks_raw = event.metadata.get("group_ranks", [])
    group_ranks = (
        list(group_ranks_raw) if isinstance(group_ranks_raw, (list, tuple)) else []
    )
    data_size = int(event.metadata.get("bytes", 0))
    name = str(event.metadata.get("name", ""))
    timestamp_ms = float(event.timestamp_ms)

    match_key = (
        collective_type,
        frozenset(group_ranks),
        name,
        round(timestamp_ms, 3),
        data_size,
    )

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

    if rank in last_node_by_rank:
        dag.add_edge(
            DAGEdge(
                src_node_id=last_node_by_rank[rank].node_id,
                dst_node_id=collective_id,
            )
        )
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
    group_ranks = (
        list(group_ranks_raw) if isinstance(group_ranks_raw, (list, tuple)) else []
    )
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
        event.metadata.get("bytes")
        if event.metadata.get("bytes") is not None
        else activation_bytes
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
) -> None:
    duration_ms = next_event.timestamp_ms - event.timestamp_ms
    if duration_ms <= 0:
        return
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
        else:
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
        )


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
) -> None:
    events = sorted(trace.events, key=lambda e: e.timestamp_ms)
    active_microbatch_id: list = [-1]
    active_direction: list = [""]
    slot_node_ids: list[int] = []
    for i in range(len(events)):
        event = events[i]
        if event.type == "slot_begin":
            _process_slot_begin(
                event,
                rank,
                active_microbatch_id,
                active_direction,
                slot_node_ids,
                slot_first_timestamp,
            )
        elif event.type == "slot_end":
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
            )


def _wire_slot_edges(dag: ExecutionDAG, slot_nodes: dict) -> None:
    for node_ids in slot_nodes.values():
        for i in range(len(node_ids) - 1):
            dag.add_edge(DAGEdge(src_node_id=node_ids[i], dst_node_id=node_ids[i + 1]))


def _should_skip_pp_pair(src_stage: int, dst_stage: int, direction: str) -> bool:
    if direction == "fwd" and src_stage > dst_stage:
        return True
    if direction == "bwd" and src_stage < dst_stage:
        return True
    return False


def _create_pp_send(
    src: int,
    dst: int,
    record,
    src_node: int,
    dst_node: int,
    node_id: list,
    flow_id: list,
    dag: ExecutionDAG,
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
    dag.add_edge(DAGEdge(src_node_id=src_node, dst_node_id=pp_send.node_id))
    dag.add_edge(DAGEdge(src_node_id=pp_send.node_id, dst_node_id=dst_node))
    node_id[0] += 1
    flow_id[0] += 1


def _resolve_pp_nodes(
    src: int,
    dst: int,
    record,
    slot_entry_node: dict,
    slot_last_node: dict,
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
) -> tuple[int, int]:
    seen: set[tuple[int, int, int, str]] = set()
    for record in pending:
        pp_stride = config.ranks_per_stage
        src_stage = record.remapped_src // pp_stride
        dst_stage = record.remapped_dst // pp_stride
        if src_stage == dst_stage:
            continue
        if not (
            0 <= record.remapped_src < config.world_size
            and 0 <= record.remapped_dst < config.world_size
        ):
            continue
        src_ranks = _ranks_for_stage(src_stage, config)
        dst_ranks = _ranks_for_stage(dst_stage, config)
        for src, dst in zip(src_ranks, dst_ranks):
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
            _create_pp_send(src, dst, record, src_node, dst_node, node_id, flow_id, dag)
    return node_id[0], flow_id[0]


def _wire_cross_slot_edges(
    dag: ExecutionDAG,
    slot_first_timestamp: dict,
    slot_last_node: dict,
    slot_entry_node: dict,
) -> None:
    keys_by_rank: dict[int, list] = {}
    for key in slot_first_timestamp:
        keys_by_rank.setdefault(key[0], []).append(key)
    for _, keys in keys_by_rank.items():
        keys.sort(key=lambda k: slot_first_timestamp[k])
        for i in range(len(keys) - 1):
            prev_last = slot_last_node.get(keys[i])
            next_first = slot_entry_node.get(keys[i + 1])
            if prev_last is not None and next_first is not None:
                dag.add_edge(DAGEdge(src_node_id=prev_last, dst_node_id=next_first))


def _wire_bwd_to_step(
    dag: ExecutionDAG,
    slot_last_node: dict,
    slot_entry_node: dict,
    config: ParallelConfig,
) -> None:
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


def _decompose_collectives_in_dag(
    dag: ExecutionDAG,
    tracer_cfg: DAGTracerConfig,
    node_id: list,
    flow_id: list,
) -> None:
    if not dag.collective_nodes:
        return

    for C in sorted(dag.collective_nodes.values(), key=lambda n: n.node_id):
        result, next_flow_id = decompose_collective(
            collective_type=C.collective_type,
            group_ranks=C.group_ranks,
            data_size=C.data_size,
            num_channels=tracer_cfg.num_channels,
            algorithm=tracer_cfg.algorithm,
            flow_id_start=flow_id[0],
        )
        first_p2p_id = node_id[0]
        if len(result.flows) == 0:
            continue
        for flow in result.flows:
            dag.add_comm_node(
                CommNode(
                    node_id=node_id[0],
                    src_gpu=flow.src,
                    dst_gpu=flow.dst,
                    bytes=flow.flow_size,
                    collective_type=C.collective_type,
                    layer_id=-1,
                    phase=C.phase,
                    flow_id=flow.flow_id,
                    parent_flow_ids=list(flow.parent_flow_ids),
                )
            )
            node_id[0] += 1
        last_p2p_id = node_id[0] - 1

        for i in range(len(result.flows) - 1):
            dag.add_edge(
                DAGEdge(
                    src_node_id=first_p2p_id + i,
                    dst_node_id=first_p2p_id + i + 1,
                )
            )

        for edge in C.pending_edges or []:
            if edge.dst_node_id == C.node_id:
                edge.dst_node_id = first_p2p_id
            if edge.src_node_id == C.node_id:
                edge.src_node_id = last_p2p_id
            dag.add_edge(edge)

        flow_id[0] = next_flow_id

    dag.collective_nodes.clear()


class MegatronDagTracer(DAGTracer):
    cfg: DAGTracerConfig
    ccl: CCLDecomposer

    def __init__(self, cfg: DAGTracerConfig, ccl: CCLDecomposer):
        self.cfg = cfg
        self.ccl = ccl

    def trace(
        self, workload: MegatronWorkload, datacenter: DatacenterConfig
    ) -> ExecutionDAG:
        dag = ExecutionDAG()
        config = ParallelConfig.from_workload(workload)
        traces_dir = _resolve_traces_dir(datacenter, workload)
        activation_bytes = _compute_activation_bytes(workload)

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

        node_id = [0]
        flow_id = [0]

        with log_progress("  building DAG", config.world_size, logger) as advance:
            _collective_registry: dict = {}
            for rank in range(config.world_size):
                trace = _load_or_derive_trace(rank, traces_dir, config, stage_traces)
                exact_path = traces_dir / f"trace_rank_{rank}.json"
                if exact_path.exists():
                    dag.profiled_ranks.add(rank)
                if dag.total_flops is None and trace.total_flops is not None:
                    dag.total_flops = trace.total_flops
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
                )
                advance()

        _wire_slot_edges(dag, slot_nodes)
        node_id[0], flow_id[0] = _wire_pp_transfers(
            dag,
            pending_pp_transfers,
            config,
            slot_entry_node,
            slot_last_node,
            node_id,
            flow_id,
        )
        _wire_cross_slot_edges(
            dag, slot_first_timestamp, slot_last_node, slot_entry_node
        )
        _wire_bwd_to_step(dag, slot_last_node, slot_entry_node, config)
        _decompose_collectives_in_dag(dag, self.cfg, node_id, flow_id)
        return dag
