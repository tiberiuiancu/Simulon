from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from simulon.backend.dag.nodes import ComputeNode, CommNode, DAGEdge, ExecutionDAG
from simulon.backend.dag.trace_parser import TraceFileParser
from simulon.backend.dag.tracer import DAGTracer, DAGTracerConfig
from simulon.collective import CCLDecomposer
from simulon.collective.decompose import decompose_collective
from simulon.config.dc import DatacenterConfig
from simulon.config.workload import MegatronWorkload


@dataclass(frozen=True)
class _PendingPPTransfer:
    remapped_src: int
    remapped_dst: int
    bytes: int
    microbatch_id: int
    direction: str


class MegatronDagTracer(DAGTracer):
    cfg: DAGTracerConfig
    ccl: CCLDecomposer

    def __init__(self, cfg: DAGTracerConfig, ccl: CCLDecomposer):
        self.cfg = cfg
        self.ccl = ccl

    def trace(self, workload: MegatronWorkload, datacenter: DatacenterConfig) -> ExecutionDAG:
        dag = ExecutionDAG()
        node_id_counter = 0
        flow_id_counter = 0

        cfg = workload.config
        tp = int(cfg.get("tensor-model-parallel-size", 1))
        pp = int(cfg.get("pipeline-model-parallel-size", 1))
        ep = int(cfg.get("expert-model-parallel-size", 1))
        cp = 1
        num_gpus = int(cfg.get("num_gpus", tp * pp * ep))
        dp_pipeline = max(1, num_gpus // (tp * pp * cp))
        dp = max(1, num_gpus // (tp * pp * ep))
        num_microbatches = int(
            cfg.get("num_microbatches")
            or (
                int(cfg.get("global-batch-size", 0))
                // max(1, dp_pipeline * int(cfg.get("micro-batch-size", 1)))
            )
        )

        # Activation bytes fallback (used only when trace event has no bytes field)
        seq_len = int(cfg.get("seq-length", 2048))
        micro_bs = int(cfg.get("micro-batch-size", 1))
        hidden_size = int(cfg.get("hidden-size", 0))
        dtype_str = str(cfg.get("dtype", "bf16")).lower()
        dtype_bytes = 4 if dtype_str == "fp32" else 1 if dtype_str == "fp8" else 2
        activation_bytes = seq_len * micro_bs * hidden_size * dtype_bytes

        def global_rank(dp_rank: int, pp_stage: int, ep_rank: int, tp_rank: int, cp_rank: int = 0) -> int:
            return (
                dp_rank * (pp * ep * cp * tp)
                + pp_stage * (ep * cp * tp)
                + ep_rank * (cp * tp)
                + cp_rank * tp
                + tp_rank
            )

        def ranks_for_stage(pp_stage: int) -> list[int]:
            ranks: list[int] = []
            for dp_rank in range(dp):
                for ep_rank in range(ep):
                    for tp_rank in range(tp):
                        for cp_rank in range(cp):
                            ranks.append(global_rank(dp_rank, pp_stage, ep_rank, tp_rank, cp_rank))
            return ranks

        def _remap_rank(traced_rank: int, traced_stage: int, target_stage: int) -> int:
            offset = (target_stage - traced_stage) * (ep * cp * tp)
            return traced_rank + offset

        traces_dir = (
            datacenter.datacenter.traces_dir
            if datacenter and datacenter.datacenter
            else None
        )
        if traces_dir is None:
            raise ValueError(
                "traces_dir must be set in datacenter.datacenter for trace-driven workloads"
            )
        traces_dir = Path(traces_dir)

        trace_paths: dict[int, str] = {}
        for pp_stage in range(pp):
            path = traces_dir / f"trace_pp_stage_{pp_stage}.json"
            if path.exists():
                trace_paths[pp_stage] = str(path)

        if 0 not in trace_paths:
            raise ValueError(
                f"First PP stage (0) trace missing: {traces_dir / 'trace_pp_stage_0.json'}. "
                "Run trace generation for the first stage."
            )
        if (pp - 1) not in trace_paths:
            raise ValueError(
                f"Last PP stage ({pp - 1}) trace missing: {traces_dir / f'trace_pp_stage_{pp - 1}.json'}. "
                "Run trace generation for the last stage."
            )

        if pp > 2:
            fallback_middle: str | None = None
            for candidate in range(1, pp - 1):
                if candidate in trace_paths:
                    fallback_middle = trace_paths[candidate]
                    break
            for pp_stage in range(1, pp - 1):
                if pp_stage not in trace_paths:
                    if fallback_middle is None:
                        raise ValueError(
                            f"Middle PP stage ({pp_stage}) trace missing and no fallback available. "
                            f"Expected one of: {[traces_dir / f'trace_pp_stage_{s}.json' for s in range(1, pp - 1)]}"
                        )
                    trace_paths[pp_stage] = fallback_middle

        slot_nodes: dict[tuple[int, int, str], list[int]] = {}
        slot_entry_node: dict[tuple[int, int, str], int] = {}
        slot_last_node: dict[tuple[int, int, str], int] = {}
        pending_pp_transfers: list[_PendingPPTransfer] = []

        # Track first slot_begin timestamp for each key (cross-slot ordering)
        slot_first_timestamp: dict[tuple[int, int, str], float] = {}

        for target_stage, path in trace_paths.items():
            trace_file = TraceFileParser.parse(path)
            events = sorted(trace_file.events, key=lambda e: e.timestamp_ms)
            # The file header pipeline_stage is often 0 in fake-process-group runs.
            # Use the first slot_begin event metadata for the actual traced stage.
            traced_stage = next(
                (
                    cast(int, e.metadata.get("pipeline_stage", trace_file.pipeline_stage))
                    for e in events
                    if e.type == "slot_begin"
                ),
                trace_file.pipeline_stage,
            )
            replica_ranks = ranks_for_stage(target_stage)

            active_microbatch_id = -1
            active_direction = ""
            slot_node_ids_by_replica: dict[int, list[int]] = {r: [] for r in replica_ranks}
            last_node_by_replica: dict[int, int] = {}

            for i in range(len(events)):
                event = events[i]

                if event.type == "slot_begin":
                    active_microbatch_id = cast(int, event.metadata.get("microbatch_id", -1))
                    raw_phase = event.metadata.get("phase", "")
                    active_phase = str(raw_phase)
                    if active_phase == "fwd":
                        active_direction = "fwd"
                    elif active_phase in ("bwd", "bwd_ig", "bwd_wg"):
                        active_direction = "bwd"
                    else:
                        active_direction = str(
                            event.metadata.get("direction")
                            or event.metadata.get("slot")
                            or active_phase
                        )
                    for r in replica_ranks:
                        slot_node_ids_by_replica[r] = []

                    # Record first timestamp for cross-slot ordering
                    for r in replica_ranks:
                        key = (r, active_microbatch_id, active_direction)
                        if key not in slot_first_timestamp:
                            slot_first_timestamp[key] = event.timestamp_ms

                elif event.type == "slot_end":
                    for r in replica_ranks:
                        if slot_node_ids_by_replica[r]:
                            key = (r, active_microbatch_id, active_direction)
                            slot_nodes.setdefault(key, []).extend(slot_node_ids_by_replica[r])
                            slot_entry_node[key] = slot_node_ids_by_replica[r][0]
                            slot_last_node[key] = slot_node_ids_by_replica[r][-1]
                        slot_node_ids_by_replica[r] = []
                    active_microbatch_id = -1
                    active_direction = ""

                if i + 1 < len(events):
                    next_event = events[i + 1]
                    duration_ms = next_event.timestamp_ms - event.timestamp_ms
                    if duration_ms > 0:
                        is_pp = False
                        if event.type == "collective":
                            ct = str(event.metadata.get("collective_type", ""))
                            is_pp = ct in ("PP_Send", "PP_Recv")
                        if event.type == "slot_begin" or (event.type == "collective" and not is_pp):
                            for r in replica_ranks:
                                cn = ComputeNode(
                                    node_id=node_id_counter,
                                    gpu_rank=r,
                                    kernel="trace_compute",
                                    layer_id=-1,
                                    microbatch_id=active_microbatch_id,
                                    pipeline_stage=target_stage,
                                    phase=active_direction,
                                    duration_ms=duration_ms,
                                )
                                dag.compute_nodes.append(cn)
                                slot_node_ids_by_replica[r].append(node_id_counter)
                                if r in last_node_by_replica:
                                    dag.edges.append(DAGEdge(src_node_id=last_node_by_replica[r], dst_node_id=node_id_counter))
                                last_node_by_replica[r] = node_id_counter
                                node_id_counter += 1
                        if event.type == "collective":
                            collective_type = str(event.metadata.get("collective_type", ""))
                            if collective_type in ("PP_Send", "PP_Recv"):
                                # Collect both PP_Send and PP_Recv for Pass 2 wiring.
                                # They represent the same physical transfer; deduplication
                                # happens later by (src, dst, microbatch, direction).
                                group_ranks_raw = event.metadata.get("group_ranks", [])
                                group_ranks = cast(
                                    list[int],
                                    list(group_ranks_raw) if isinstance(group_ranks_raw, (list, tuple)) else [],
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

                                remapped_src = _remap_rank(traced_src, traced_stage, target_stage)
                                remapped_dst = _remap_rank(traced_dst, traced_stage, target_stage)

                                data_size = cast(int, event.metadata.get("bytes", activation_bytes))

                                # PP events may occur after slot_end, so active_microbatch_id is -1.
                                # Use the trace event metadata which carries the correct microbatch_id.
                                pp_mb = cast(int, event.metadata.get("microbatch_id", active_microbatch_id))
                                pending_pp_transfers.append(_PendingPPTransfer(
                                    remapped_src=remapped_src,
                                    remapped_dst=remapped_dst,
                                    bytes=data_size,
                                    microbatch_id=pp_mb,
                                    direction=direction,
                                ))
                                continue
                            else:
                                group_ranks_raw = event.metadata.get("group_ranks", [])
                                group_ranks = cast(
                                    list[int],
                                    list(group_ranks_raw) if isinstance(group_ranks_raw, (list, tuple)) else [],
                                )
                                remapped_group_ranks = [_remap_rank(r, traced_stage, target_stage) for r in group_ranks]
                                data_size = cast(int, event.metadata.get("bytes", 0))

                                result, flow_id_counter = decompose_collective(
                                    collective_type=collective_type,
                                    group_ranks=remapped_group_ranks,
                                    data_size=data_size,
                                    num_channels=self.cfg.num_channels,
                                    algorithm=self.cfg.algorithm,
                                    flow_id_start=flow_id_counter,
                                )

                                for flow in result.flows:
                                    comm_node = CommNode(
                                        node_id=node_id_counter,
                                        src_gpu=flow.src,
                                        dst_gpu=flow.dst,
                                        bytes=flow.flow_size,
                                        collective_type=collective_type,
                                        layer_id=-1,
                                        phase=active_direction,
                                        flow_id=flow.flow_id,
                                        parent_flow_ids=flow.parent_flow_ids,
                                    )
                                    dag.comm_nodes.append(comm_node)
                                    if flow.src in slot_node_ids_by_replica:
                                        slot_node_ids_by_replica[flow.src].append(node_id_counter)
                                        if flow.src in last_node_by_replica:
                                            dag.edges.append(DAGEdge(src_node_id=last_node_by_replica[flow.src], dst_node_id=node_id_counter))
                                        last_node_by_replica[flow.src] = node_id_counter
                                    if flow.dst in slot_node_ids_by_replica and flow.dst != flow.src:
                                        slot_node_ids_by_replica[flow.dst].append(node_id_counter)
                                        if flow.dst in last_node_by_replica:
                                            dag.edges.append(DAGEdge(src_node_id=last_node_by_replica[flow.dst], dst_node_id=node_id_counter))
                                        last_node_by_replica[flow.dst] = node_id_counter
                                    node_id_counter += 1

        for node_ids in slot_nodes.values():
            for i in range(len(node_ids) - 1):
                dag.edges.append(DAGEdge(src_node_id=node_ids[i], dst_node_id=node_ids[i + 1]))

        seen_transfers: set[tuple[int, int, int, str]] = set()
        for record in pending_pp_transfers:
            # The trace records one representative PP transfer per stage pair.
            # Replicate it across all corresponding ranks (TP/EP/CP/DP replicas).
            ranks_per_stage = ep * cp * tp
            dp_stride = pp * ranks_per_stage
            src_stage = (record.remapped_src % dp_stride) // ranks_per_stage
            dst_stage = (record.remapped_dst % dp_stride) // ranks_per_stage
            src_ranks = ranks_for_stage(src_stage)
            dst_ranks = ranks_for_stage(dst_stage)

            for src, dst in zip(src_ranks, dst_ranks):
                dedup_key = (src, dst, record.microbatch_id, record.direction)
                if dedup_key in seen_transfers:
                    continue
                seen_transfers.add(dedup_key)

                src_key = (src, record.microbatch_id, record.direction)
                dst_key = (dst, record.microbatch_id, record.direction)

                src_node_id = slot_last_node.get(src_key)
                dst_node_id = slot_entry_node.get(dst_key)

                if dst_node_id is None and record.direction == "bwd":
                    for bwd_phase in ("bwd_ig", "bwd_wg"):
                        alt_key = (dst, record.microbatch_id, bwd_phase)
                        if alt_key in slot_entry_node:
                            dst_node_id = slot_entry_node[alt_key]
                            break

                if src_node_id is not None and dst_node_id is not None:
                    pp_send = CommNode(
                        node_id=node_id_counter,
                        src_gpu=src,
                        dst_gpu=dst,
                        bytes=record.bytes,
                        collective_type="PP_Send",
                        layer_id=-1,
                        phase=record.direction,
                        flow_id=flow_id_counter,
                    )
                    dag.comm_nodes.append(pp_send)
                    dag.edges.append(DAGEdge(src_node_id=src_node_id, dst_node_id=pp_send.node_id))
                    dag.edges.append(DAGEdge(src_node_id=pp_send.node_id, dst_node_id=dst_node_id))
                    node_id_counter += 1
                    flow_id_counter += 1

        keys_by_rank: dict[int, list[tuple[int, int, str]]] = {}
        for key in slot_first_timestamp:
            rank = key[0]
            keys_by_rank.setdefault(rank, []).append(key)

        for rank, keys in keys_by_rank.items():
            keys.sort(key=lambda k: slot_first_timestamp[k])
            for i in range(len(keys) - 1):
                prev_key = keys[i]
                next_key = keys[i + 1]
                prev_last = slot_last_node.get(prev_key)
                next_first = slot_entry_node.get(next_key)
                if prev_last is not None and next_first is not None:
                    dag.edges.append(DAGEdge(src_node_id=prev_last, dst_node_id=next_first))

        for target_stage in trace_paths.keys():
            replica_ranks = ranks_for_stage(target_stage)
            for r in replica_ranks:
                bwd_keys = [k for k in slot_last_node if k[0] == r and k[2] == "bwd"]
                if bwd_keys:
                    last_bwd_key = max(bwd_keys, key=lambda k: k[1])
                    step_key = (r, 0, "step")
                    if step_key in slot_entry_node:
                        dag.edges.append(DAGEdge(
                            src_node_id=slot_last_node[last_bwd_key],
                            dst_node_id=slot_entry_node[step_key],
                        ))

        return dag
