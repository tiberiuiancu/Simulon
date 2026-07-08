from __future__ import annotations

import logging
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
    traces_dir = datacenter.datacenter.traces_dir if datacenter and datacenter.datacenter else None
    if traces_dir is not None:
        return Path(traces_dir)
    from simulon.config.resolve import resolve_gpu_spec, workload_hash

    gpu_spec = resolve_gpu_spec(datacenter)
    gpu_name = (gpu_spec.name or "default").lower().replace(" ", "-")
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


def _stage_has_exact_trace(pp_stage: int, traces_dir: Path, config: ParallelConfig) -> bool:
    """Return True if any rank in the stage has a trace_rank_{rank}.json file."""
    return any(
        (traces_dir / f"trace_rank_{r}.json").exists() for r in _ranks_for_stage(pp_stage, config)
    )


def _load_first_traced_rank_in_stage(pp_stage: int, traces_dir: Path, config: ParallelConfig):
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


def _remap_collective(group: list[int], from_rank: int, to_rank: int):
    group = set(group)
    for from_group, to_group in zip(
        _COLLECTIVE_GROUPS[from_rank], _COLLECTIVE_GROUPS[to_rank], strict=False
    ):
        if set(from_group) == group:
            return to_group


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

    if rank in last_node_by_rank:
        dag.add_edge(
            DAGEdge(src_node_id=last_node_by_rank[rank].node_id, dst_node_id=collective_id)
        )

    async_op = bool(event.metadata.get("async_op", False))
    if async_op and tracer_cfg.overlap_async_collectives:
        # Async collectives launch on a separate CUDA stream and don't block
        # subsequent compute on this rank. Keep the predecessor edge (collective
        # waits for prior work) but don't make this collective the new "last
        # node" — the next compute node will chain from the previous one instead.
        return

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
    flops_multiplier: float = 1.0,
) -> None:
    duration_ms = (next_event.timestamp_ms - event.timestamp_ms) / flops_multiplier
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
    flops_multiplier: float = 1.0,
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
                flops_multiplier,
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
        activation_bytes = _compute_activation_bytes(workload)
        flops_multiplier = resolve_gpu_spec(datacenter).flops_multiplier

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

        total_energy_kwh = 0.0
        total_co2eq_kg = 0.0

        with log_progress("  building DAG", config.world_size, logger) as advance:
            _collective_registry: dict = {}
            for rank in range(config.world_size):
                trace = _load_or_derive_trace(rank, traces_dir, config, stage_traces)
                exact_path = traces_dir / f"trace_rank_{rank}.json"
                if exact_path.exists():
                    dag.profiled_ranks.add(rank)
                if dag.total_flops is None and trace.total_flops is not None:
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
