from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, cast

from simulon.backend.dag.nodes import ComputeNode, CommNode, DAGEdge, ExecutionDAG
from simulon.backend.dag.trace_parser import TraceFileParser
from simulon.backend.dag.tracer import DAGTracer, DAGTracerConfig
from simulon.collective import CCLDecomposer
from simulon.collective.decompose import decompose_collective
from simulon.config.dc import DatacenterConfig
from simulon.config.workload import MegatronWorkload


# =============================================================================
# Rank formula and parallelism helpers
# =============================================================================
#
# Megatron-LM global rank ordering (innermost -> outermost):
#     TP -> CP -> EP -> DP -> PP
#
# A global rank is computed as:
#     rank = tp
#          + cp * TP
#          + ep * TP*CP
#          + dp * TP*CP*EP
#          + pp * TP*CP*EP*DP
#
# Dimensions that vary faster are to the RIGHT in the linearized address.
# Example: TP=2, CP=1, EP=1, DP=2, PP=4
#   - Ranks 0..1  : PP stage 0, DP group 0 (TP=0,1)
#   - Ranks 2..3  : PP stage 0, DP group 1 (TP=0,1)
#   - Ranks 4..7  : PP stage 1, DP groups 0,1
#   - ... and so on.
#
# We keep CP hardcoded to 1 for now; the formula still accepts a cp_rank so
# that the code is forward-compatible.


class RankCoords(NamedTuple):
    """Decomposed parallelism coordinates for a single global rank.

    The ordering mirrors Megatron-LM's ``parallel_state.py``:
    ``TP -> CP -> EP -> DP -> PP`` (innermost to outermost).
    """

    tp: int
    cp: int
    ep: int
    dp: int
    pp: int


@dataclass(frozen=True)
class ParallelConfig:
    """Immutable parallelism dimensions extracted from a ``MegatronWorkload``.

    Attributes
    ----------
    tp : int
        Tensor-model-parallel size.
    cp : int
        Context-parallel size (currently hardcoded to 1).
    ep : int
        Expert-model-parallel size.
    dp : int
        Data-parallel size.
    pp : int
        Pipeline-model-parallel size.
    num_gpus : int
        Total number of GPUs (``world_size``).

    Notes
    -----
    The product ``tp * cp * ep * dp * pp`` must equal ``num_gpus``.
    If it does not, the config is considered invalid and ``validate()``
    (invoked by ``from_workload``) raises ``ValueError``.
    """

    tp: int
    cp: int
    ep: int
    dp: int
    pp: int
    num_gpus: int

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_workload(cls, workload: MegatronWorkload) -> "ParallelConfig":
        """Build a ``ParallelConfig`` from a ``MegatronWorkload`` config dict.

        The workload's flat ``config`` dictionary is queried for keys like
        ``tensor-model-parallel-size``, ``pipeline-model-parallel-size``,
        ``expert-model-parallel-size``, and ``num_gpus``.

        CP (context parallel) is hardcoded to 1 because simulon does not
        support CP > 1 yet.
        """
        cfg = workload.config
        tp = int(cfg.get("tensor-model-parallel-size", 1))
        pp = int(cfg.get("pipeline-model-parallel-size", 1))
        ep = int(cfg.get("expert-model-parallel-size", 1))
        cp = 1  # hardcoded — see note above
        num_gpus = int(cfg.get("num_gpus", cfg.get("num-gpus", tp * pp * ep)))
        dp = max(1, num_gpus // (tp * pp * ep))
        return cls(tp=tp, cp=cp, ep=ep, dp=dp, pp=pp, num_gpus=num_gpus)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def world_size(self) -> int:
        """Alias for ``num_gpus``."""
        return self.num_gpus

    @property
    def ranks_per_stage(self) -> int:
        """Number of global ranks that belong to a single PP stage.

        This is ``TP * CP * EP * DP`` — every dimension *except* PP.
        """
        return self.tp * self.cp * self.ep * self.dp


# ------------------------------------------------------------------------------
# Rank conversion helpers
# ------------------------------------------------------------------------------

def _global_rank(
    tp_rank: int,
    cp_rank: int,
    ep_rank: int,
    dp_rank: int,
    pp_stage: int,
    config: ParallelConfig,
) -> int:
    """Compute the global rank from decomposed parallelism coordinates.

    The formula follows Megatron-LM ``parallel_state.py`` ordering:
    ``TP -> CP -> EP -> DP -> PP`` (innermost to outermost).

    .. math::

        rank = tp
             + cp \\times TP
             + ep \\times TP \\times CP
             + dp \\times TP \\times CP \\times EP
             + pp \\times TP \\times CP \\times EP \\times DP

    Parameters
    ----------
    tp_rank, cp_rank, ep_rank, dp_rank, pp_stage : int
        Coordinates inside each parallelism dimension.
    config : ParallelConfig
        The parallelism configuration defining the sizes of each dimension.

    Returns
    -------
    int
        The global linearized rank (0 .. ``config.world_size - 1``).

    Raises
    ------
    ValueError
        If any coordinate is out of range for the given ``config``.

    Examples
    --------
    >>> pc = ParallelConfig(tp=2, cp=1, ep=1, dp=2, pp=4, num_gpus=16)
    >>> _global_rank(0, 0, 0, 0, 0, pc)
    0
    >>> _global_rank(1, 0, 0, 0, 0, pc)
    1
    >>> _global_rank(0, 0, 0, 1, 0, pc)   # DP group 1, same PP stage 0
    2
    >>> _global_rank(0, 0, 0, 0, 1, pc)   # PP stage 1
    4
    >>> _global_rank(0, 0, 0, 0, 3, pc)   # PP stage 3
    12
    """
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
    """Convert a global rank back to decomposed ``(tp, cp, ep, dp, pp)`` coordinates.

    This is the exact inverse of :func:`_global_rank`:

    .. code-block:: python

        assert _global_rank(*_decompose_rank(r, config), config) == r
        for all valid r in 0 .. config.world_size - 1

    The decomposition uses successive modulo and integer-division by the stride
    of each dimension, starting from the innermost (TP) and moving outward.

    Parameters
    ----------
    rank : int
        Global linearized rank.
    config : ParallelConfig
        The parallelism configuration.

    Returns
    -------
    RankCoords
        Named tuple with fields ``tp, cp, ep, dp, pp``.

    Raises
    ------
    ValueError
        If ``rank`` is outside ``[0, config.world_size)``.

    Examples
    --------
    >>> pc = ParallelConfig(tp=2, cp=1, ep=1, dp=2, pp=4, num_gpus=16)
    >>> _decompose_rank(0, pc)
    RankCoords(tp=0, cp=0, ep=0, dp=0, pp=0)
    >>> _decompose_rank(3, pc)
    RankCoords(tp=1, cp=0, ep=0, dp=1, pp=0)
    >>> _decompose_rank(7, pc)
    RankCoords(tp=1, cp=0, ep=0, dp=1, pp=1)
    >>> _decompose_rank(15, pc)
    RankCoords(tp=1, cp=0, ep=0, dp=1, pp=3)
    """
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
    """Return the pipeline-parallel stage index for a global rank.

    This is simply ``rank // ranks_per_stage``.

    Examples
    --------
    >>> pc = ParallelConfig(tp=2, cp=1, ep=1, dp=2, pp=4, num_gpus=16)
    >>> _stage_of(0, pc)
    0
    >>> _stage_of(3, pc)
    0
    >>> _stage_of(4, pc)
    1
    >>> _stage_of(15, pc)
    3
    """
    return rank // config.ranks_per_stage


def _ranks_for_stage(pp_stage: int, config: ParallelConfig) -> list[int]:
    """Return every global rank that belongs to a given PP stage.

    The ranks are generated by iterating over every combination of
    ``dp``, ``ep``, ``cp``, and ``tp`` (outermost to innermost) and
    computing the global rank with :func:`_global_rank`.

    Parameters
    ----------
    pp_stage : int
        The pipeline stage index (0 .. ``config.pp - 1``).
    config : ParallelConfig
        Parallelism configuration.

    Returns
    -------
    list[int]
        Ordered list of global ranks belonging to ``pp_stage``.

    Examples
    --------
    >>> pc = ParallelConfig(tp=2, cp=1, ep=1, dp=2, pp=4, num_gpus=16)
    >>> _ranks_for_stage(0, pc)
    [0, 1, 2, 3]
    >>> _ranks_for_stage(1, pc)
    [4, 5, 6, 7]
    >>> _ranks_for_stage(3, pc)
    [12, 13, 14, 15]
    """
    ranks: list[int] = []
    for dp_rank in range(config.dp):
        for ep_rank in range(config.ep):
            for cp_rank in range(config.cp):
                for tp_rank in range(config.tp):
                    ranks.append(
                        _global_rank(tp_rank, cp_rank, ep_rank, dp_rank, pp_stage, config)
                    )
    return ranks


# ------------------------------------------------------------------------------
# Process-group membership helpers
# ------------------------------------------------------------------------------

def _ranks_in_same_dp_group(
    group_ranks: Iterable[int], reference_rank: int, config: ParallelConfig
) -> bool:
    """Return ``True`` iff every rank in *group_ranks* shares the same DP group as *reference_rank*.

    Two ranks are in the same DP group when **all other dimensions are identical**:
    ``tp``, ``cp``, ``ep``, and ``pp`` must match; only ``dp`` may differ.

    Parameters
    ----------
    group_ranks : Iterable[int]
        Ranks to validate (e.g. from a trace event's ``group_ranks``).
    reference_rank : int
        The rank whose DP group is the reference.
    config : ParallelConfig
        Parallelism configuration.

    Returns
    -------
    bool

    Examples
    --------
    >>> pc = ParallelConfig(tp=2, cp=1, ep=1, dp=2, pp=4, num_gpus=16)
    >>> _ranks_in_same_dp_group([0, 2], 0, pc)   # same PP stage, same TP, different DP
    True
    >>> _ranks_in_same_dp_group([0, 1], 0, pc)   # different TP => different DP group
    False
    """
    ref = _decompose_rank(reference_rank, config)
    for r in group_ranks:
        c = _decompose_rank(r, config)
        if (
            c.tp != ref.tp
            or c.cp != ref.cp
            or c.ep != ref.ep
            or c.pp != ref.pp
        ):
            return False
    return True


def _ranks_in_same_ep_group(
    group_ranks: Iterable[int], reference_rank: int, config: ParallelConfig
) -> bool:
    """Return ``True`` iff every rank in *group_ranks* shares the same EP group as *reference_rank*.

    Same-EP-group condition: ``tp``, ``cp``, ``dp``, and ``pp`` match; only ``ep`` may differ.
    """
    ref = _decompose_rank(reference_rank, config)
    for r in group_ranks:
        c = _decompose_rank(r, config)
        if (
            c.tp != ref.tp
            or c.cp != ref.cp
            or c.dp != ref.dp
            or c.pp != ref.pp
        ):
            return False
    return True


def _ranks_in_same_tp_group(
    group_ranks: Iterable[int], reference_rank: int, config: ParallelConfig
) -> bool:
    """Return ``True`` iff every rank in *group_ranks* shares the same TP group as *reference_rank*.

    Same-TP-group condition: ``cp``, ``ep``, ``dp``, and ``pp`` match; only ``tp`` may differ.
    """
    ref = _decompose_rank(reference_rank, config)
    for r in group_ranks:
        c = _decompose_rank(r, config)
        if (
            c.cp != ref.cp
            or c.ep != ref.ep
            or c.dp != ref.dp
            or c.pp != ref.pp
        ):
            return False
    return True


def _ranks_in_same_cp_group(
    group_ranks: Iterable[int], reference_rank: int, config: ParallelConfig
) -> bool:
    """Return ``True`` iff every rank in *group_ranks* shares the same CP group as *reference_rank*.

    Same-CP-group condition: ``tp``, ``ep``, ``dp``, and ``pp`` match; only ``cp`` may differ.
    """
    ref = _decompose_rank(reference_rank, config)
    for r in group_ranks:
        c = _decompose_rank(r, config)
        if (
            c.tp != ref.tp
            or c.ep != ref.ep
            or c.dp != ref.dp
            or c.pp != ref.pp
        ):
            return False
    return True


# ------------------------------------------------------------------------------
# Process-group builders
# ------------------------------------------------------------------------------

def _get_dp_group_ranks(rank: int, config: ParallelConfig) -> list[int]:
    """Return every global rank in the DP group containing *rank*.

    A DP group contains all ranks where ``tp``, ``cp``, ``ep``, and ``pp`` are
    fixed to the coordinates of *rank*, while ``dp`` varies across ``0 .. DP-1``.

    Parameters
    ----------
    rank : int
        Any member of the desired DP group.
    config : ParallelConfig
        Parallelism configuration.

    Returns
    -------
    list[int]
        Ordered list ``[rank(dp=0), rank(dp=1), ..., rank(dp=DP-1)]``.

    Examples
    --------
    >>> pc = ParallelConfig(tp=2, cp=1, ep=1, dp=2, pp=4, num_gpus=16)
    >>> _get_dp_group_ranks(0, pc)   # TP=0, DP varies
    [0, 2]
    >>> _get_dp_group_ranks(1, pc)   # TP=1, DP varies
    [1, 3]
    """
    coords = _decompose_rank(rank, config)
    return [
        _global_rank(coords.tp, coords.cp, coords.ep, dp, coords.pp, config)
        for dp in range(config.dp)
    ]


def _get_ep_group_ranks(rank: int, config: ParallelConfig) -> list[int]:
    """Return every global rank in the EP group containing *rank*.

    An EP group contains all ranks where ``tp``, ``cp``, ``dp``, and ``pp`` are
    fixed to the coordinates of *rank*, while ``ep`` varies across ``0 .. EP-1``.
    """
    coords = _decompose_rank(rank, config)
    return [
        _global_rank(coords.tp, coords.cp, ep, coords.dp, coords.pp, config)
        for ep in range(config.ep)
    ]


def _get_tp_group_ranks(rank: int, config: ParallelConfig) -> list[int]:
    """Return every global rank in the TP group containing *rank*.

    A TP group contains all ranks where ``cp``, ``ep``, ``dp``, and ``pp`` are
    fixed to the coordinates of *rank*, while ``tp`` varies across ``0 .. TP-1``.
    """
    coords = _decompose_rank(rank, config)
    return [
        _global_rank(tp, coords.cp, coords.ep, coords.dp, coords.pp, config)
        for tp in range(config.tp)
    ]


def _get_cp_group_ranks(rank: int, config: ParallelConfig) -> list[int]:
    """Return every global rank in the CP group containing *rank*.

    A CP group contains all ranks where ``tp``, ``ep``, ``dp``, and ``pp`` are
    fixed to the coordinates of *rank*, while ``cp`` varies across ``0 .. CP-1``.
    """
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




def _resolve_traces_dir(datacenter: DatacenterConfig, workload: MegatronWorkload) -> Path:
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


def _stage_has_exact_trace(pp_stage: int, traces_dir: Path, config: ParallelConfig) -> bool:
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
        if ct in ("PP_Send", "PP_Recv"):
            new_events.append(ev)
            continue
        group_ranks_raw = ev.metadata.get("group_ranks", [])
        group_ranks = list(group_ranks_raw) if isinstance(group_ranks_raw, (list, tuple)) else []
        ev_name = str(ev.metadata.get("name", ""))
        new_group = None
        if "DistributedDataParallel" in ev_name or "Distributed_DataParallel" in ev_name:
            new_group = _get_dp_group_ranks(to_rank, config)
        elif _ranks_in_same_dp_group(group_ranks, from_rank, config):
            new_group = _get_dp_group_ranks(to_rank, config)
        elif _ranks_in_same_ep_group(group_ranks, from_rank, config):
            new_group = _get_ep_group_ranks(to_rank, config)
        elif _ranks_in_same_tp_group(group_ranks, from_rank, config):
            new_group = _get_tp_group_ranks(to_rank, config)
        elif _ranks_in_same_cp_group(group_ranks, from_rank, config):
            new_group = _get_cp_group_ranks(to_rank, config)
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
    active_microbatch_id[0] = -1
    active_direction[0] = ""


def _add_compute_node(
    dag: ExecutionDAG,
    rank: int,
    config: ParallelConfig,
    duration_ms: float,
    microbatch_id: int,
    direction: str,
    node_id: list,
    last_node_by_rank: dict[int, int],
    slot_node_ids: list,
) -> None:
    cn = ComputeNode(
        node_id=node_id[0],
        gpu_rank=rank,
        kernel="trace_compute",
        layer_id=-1,
        microbatch_id=microbatch_id,
        pipeline_stage=_stage_of(rank, config),
        phase=direction,
        duration_ms=duration_ms,
    )
    dag.compute_nodes.append(cn)
    slot_node_ids.append(node_id[0])
    if rank in last_node_by_rank:
        dag.edges.append(DAGEdge(src_node_id=last_node_by_rank[rank], dst_node_id=node_id[0]))
    last_node_by_rank[rank] = node_id[0]
    node_id[0] += 1


def _add_non_pp_collective(
    dag: ExecutionDAG,
    event,
    rank: int,
    active_microbatch_id: int,
    direction: str,
    node_id: list,
    flow_id: list,
    last_node_by_rank: dict[int, int],
    slot_node_ids: list,
    tracer_cfg: DAGTracerConfig,
) -> None:
    collective_type = str(event.metadata.get("collective_type", ""))
    group_ranks_raw = event.metadata.get("group_ranks", [])
    group_ranks = list(group_ranks_raw) if isinstance(group_ranks_raw, (list, tuple)) else []
    data_size = int(event.metadata.get("bytes", 0))
    if not group_ranks:
        return
    result, next_flow_id = decompose_collective(
        collective_type=collective_type,
        group_ranks=group_ranks,
        data_size=data_size,
        num_channels=tracer_cfg.num_channels,
        algorithm=tracer_cfg.algorithm,
        flow_id_start=flow_id[0],
    )
    flow_id[0] = next_flow_id
    for flow in result.flows:
        comm_node = CommNode(
            node_id=node_id[0],
            src_gpu=flow.src,
            dst_gpu=flow.dst,
            bytes=flow.flow_size,
            collective_type=collective_type,
            layer_id=-1,
            phase=direction,
            flow_id=flow.flow_id,
            parent_flow_ids=flow.parent_flow_ids,
        )
        dag.comm_nodes.append(comm_node)
        if flow.src == rank:
            slot_node_ids.append(node_id[0])
            if rank in last_node_by_rank:
                dag.edges.append(DAGEdge(src_node_id=last_node_by_rank[rank], dst_node_id=node_id[0]))
            last_node_by_rank[rank] = node_id[0]
        if flow.dst == rank and flow.dst != flow.src:
            slot_node_ids.append(node_id[0])
            if rank in last_node_by_rank:
                dag.edges.append(DAGEdge(src_node_id=last_node_by_rank[rank], dst_node_id=node_id[0]))
            last_node_by_rank[rank] = node_id[0]
        node_id[0] += 1


def _add_pp_transfer(
    event,
    rank: int,
    active_microbatch_id: int,
    active_direction: str,
    activation_bytes: int,
    pending: list,
) -> None:
    group_ranks_raw = event.metadata.get("group_ranks", [])
    group_ranks = list(group_ranks_raw) if isinstance(group_ranks_raw, (list, tuple)) else []
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
    data_size = int(event.metadata.get("bytes", activation_bytes))
    pp_mb = int(event.metadata.get("microbatch_id", active_microbatch_id))
    pending.append(_PendingPPTransfer(
        remapped_src=traced_src,
        remapped_dst=traced_dst,
        bytes=data_size,
        microbatch_id=pp_mb,
        direction=direction,
    ))


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
    last_node_by_rank: dict[int, int],
) -> None:
    events = sorted(trace.events, key=lambda e: e.timestamp_ms)
    active_microbatch_id: list = [-1]
    active_direction: list = [""]
    slot_node_ids: list[int] = []
    for i in range(len(events)):
        event = events[i]
        if event.type == "slot_begin":
            _process_slot_begin(
                event, rank, active_microbatch_id, active_direction,
                slot_node_ids, slot_first_timestamp
            )
        elif event.type == "slot_end":
            _process_slot_end(
                event, rank, active_microbatch_id, active_direction,
                slot_node_ids, slot_nodes, slot_entry_node,
                slot_last_node, slot_last_timestamp
            )
        if i + 1 < len(events):
            next_event = events[i + 1]
            duration_ms = next_event.timestamp_ms - event.timestamp_ms
            if duration_ms <= 0:
                continue
            is_pp = False
            if event.type == "collective":
                ct = str(event.metadata.get("collective_type", ""))
                is_pp = ct in ("PP_Send", "PP_Recv")
            if event.type == "slot_begin" or (event.type == "collective" and not is_pp):
                _add_compute_node(
                    dag, rank, config, duration_ms,
                    active_microbatch_id[0], active_direction[0],
                    node_id, last_node_by_rank, slot_node_ids
                )
            if event.type == "collective":
                collective_type = str(event.metadata.get("collective_type", ""))
                if collective_type in ("PP_Send", "PP_Recv"):
                    _add_pp_transfer(
                        event, rank, active_microbatch_id[0],
                        active_direction[0], activation_bytes, pending_pp_transfers
                    )
                else:
                    _add_non_pp_collective(
                        dag, event, rank, active_microbatch_id[0],
                        active_direction[0], node_id, flow_id,
                        last_node_by_rank, slot_node_ids, tracer_cfg
                    )


def _wire_slot_edges(dag: ExecutionDAG, slot_nodes: dict) -> None:
    for node_ids in slot_nodes.values():
        for i in range(len(node_ids) - 1):
            dag.edges.append(DAGEdge(src_node_id=node_ids[i], dst_node_id=node_ids[i + 1]))


def _wire_pp_transfers(
    dag: ExecutionDAG,
    pending: list,
    config: ParallelConfig,
    slot_entry_node: dict,
    slot_last_node: dict,
    node_id: list,
    flow_id: list,
) -> None:
    seen: set[tuple[int, int, int, str]] = set()
    for record in pending:
        pp_stride = config.ranks_per_stage
        src_stage = record.remapped_src // pp_stride
        dst_stage = record.remapped_dst // pp_stride
        if src_stage == dst_stage:
            continue
        if not (0 <= record.remapped_src < config.world_size
                and 0 <= record.remapped_dst < config.world_size):
            continue
        src_ranks = _ranks_for_stage(src_stage, config)
        dst_ranks = _ranks_for_stage(dst_stage, config)
        for src, dst in zip(src_ranks, dst_ranks):
            dedup = (src, dst, record.microbatch_id, record.direction)
            if dedup in seen:
                continue
            seen.add(dedup)
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
            if src_node is not None and dst_node is not None:
                if record.direction == "fwd" and src_stage > dst_stage:
                    continue
                if record.direction == "bwd" and src_stage < dst_stage:
                    continue
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
                dag.comm_nodes.append(pp_send)
                dag.edges.append(DAGEdge(src_node_id=src_node, dst_node_id=pp_send.node_id))
                dag.edges.append(DAGEdge(src_node_id=pp_send.node_id, dst_node_id=dst_node))
                node_id[0] += 1
                flow_id[0] += 1
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
    for rank, keys in keys_by_rank.items():
        keys.sort(key=lambda k: slot_first_timestamp[k])
        for i in range(len(keys) - 1):
            prev_last = slot_last_node.get(keys[i])
            next_first = slot_entry_node.get(keys[i + 1])
            if prev_last is not None and next_first is not None:
                dag.edges.append(DAGEdge(src_node_id=prev_last, dst_node_id=next_first))


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
                dag.edges.append(DAGEdge(
                    src_node_id=slot_last_node[last_bwd_key],
                    dst_node_id=slot_entry_node[step_key],
                ))


class MegatronDagTracer(DAGTracer):
    cfg: DAGTracerConfig
    ccl: CCLDecomposer

    def __init__(self, cfg: DAGTracerConfig, ccl: CCLDecomposer):
        self.cfg = cfg
        self.ccl = ccl

    def trace(self, workload: MegatronWorkload, datacenter: DatacenterConfig) -> ExecutionDAG:
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
        last_node_by_rank: dict[int, int] = {}

        node_id = [0]
        flow_id = [0]

        for rank in range(config.world_size):
            trace = _load_or_derive_trace(rank, traces_dir, config, stage_traces)
            if dag.total_flops is None and trace.total_flops is not None:
                dag.total_flops = trace.total_flops
            _add_trace_to_dag(
                dag, trace, rank, config, node_id, flow_id,
                activation_bytes, self.cfg,
                slot_nodes, slot_entry_node, slot_last_node,
                slot_first_timestamp, slot_last_timestamp,
                pending_pp_transfers, last_node_by_rank,
            )

        _wire_slot_edges(dag, slot_nodes)
        node_id[0], flow_id[0] = _wire_pp_transfers(
            dag, pending_pp_transfers, config,
            slot_entry_node, slot_last_node, node_id, flow_id,
        )
        _wire_cross_slot_edges(dag, slot_first_timestamp, slot_last_node, slot_entry_node)
        _wire_bwd_to_step(dag, slot_last_node, slot_entry_node, config)
        return dag

