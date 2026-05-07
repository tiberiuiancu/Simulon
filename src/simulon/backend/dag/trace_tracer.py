from __future__ import annotations

from pathlib import Path
from typing import cast

from simulon.backend.dag.nodes import ComputeNode, CommNode, DAGEdge, ExecutionDAG
from simulon.backend.dag.trace_parser import TraceFileParser
from simulon.backend.dag.tracer import DAGTracer, DAGTracerConfig
from simulon.backend.dag.pipeline import OneFOneBScheduler
from simulon.collective import CCLDecomposer
from simulon.collective.decompose import decompose_collective
from simulon.config.dc import DatacenterConfig
from simulon.config.workload import MegatronWorkload


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

        scheduler = OneFOneBScheduler(pp, num_microbatches)

        # Activation bytes for PP_Send (seq_len * micro_bs * hidden_size * dtype_bytes)
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
        pp_events: dict[int, list[tuple[int, str, int, int, int, str]]] = {}
        slot_entry_node: dict[tuple[int, int, str], int] = {}
        slot_last_node: dict[tuple[int, int, str], int] = {}

        for pp_stage, path in trace_paths.items():
            trace_file = TraceFileParser.parse(path)
            events = sorted(trace_file.events, key=lambda e: e.timestamp_ms)
            replica_ranks = ranks_for_stage(pp_stage)

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
                                    pipeline_stage=pp_stage,
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
                                if pp <= 1:
                                    continue
                                group_ranks_raw = event.metadata.get("group_ranks", [])
                                group_ranks = cast(
                                    list[int],
                                    list(group_ranks_raw) if isinstance(group_ranks_raw, (list, tuple)) else [],
                                )
                                data_size = cast(int, event.metadata.get("bytes", 0))
                                mb = cast(int, event.metadata.get("microbatch_id", active_microbatch_id))
                                direction = str(event.metadata.get("direction", active_direction))

                                if len(group_ranks) == 2:
                                    src_gpu = group_ranks[0]
                                    dst_gpu = group_ranks[1]
                                else:
                                    src_gpu = replica_ranks[0]
                                    dst_gpu = replica_ranks[0]

                                cn = CommNode(
                                    node_id=node_id_counter,
                                    src_gpu=src_gpu,
                                    dst_gpu=dst_gpu,
                                    bytes=data_size,
                                    collective_type=collective_type,
                                    layer_id=-1,
                                    phase=direction,
                                    flow_id=flow_id_counter,
                                )
                                dag.comm_nodes.append(cn)
                                if src_gpu in slot_node_ids_by_replica:
                                    slot_node_ids_by_replica[src_gpu].append(node_id_counter)
                                    if src_gpu in last_node_by_replica:
                                        dag.edges.append(DAGEdge(src_node_id=last_node_by_replica[src_gpu], dst_node_id=node_id_counter))
                                    last_node_by_replica[src_gpu] = node_id_counter
                                if dst_gpu in slot_node_ids_by_replica and dst_gpu != src_gpu:
                                    slot_node_ids_by_replica[dst_gpu].append(node_id_counter)
                                    if dst_gpu in last_node_by_replica:
                                        dag.edges.append(DAGEdge(src_node_id=last_node_by_replica[dst_gpu], dst_node_id=node_id_counter))
                                    last_node_by_replica[dst_gpu] = node_id_counter
                                pp_events.setdefault(pp_stage, []).append(
                                    (node_id_counter, collective_type, src_gpu, dst_gpu, mb, direction)
                                )
                                node_id_counter += 1
                                flow_id_counter += 1
                            else:
                                group_ranks_raw = event.metadata.get("group_ranks", [])
                                group_ranks = cast(
                                    list[int],
                                    list(group_ranks_raw) if isinstance(group_ranks_raw, (list, tuple)) else [],
                                )
                                data_size = cast(int, event.metadata.get("bytes", 0))

                                result, flow_id_counter = decompose_collective(
                                    collective_type=collective_type,
                                    group_ranks=group_ranks,
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

        if pp > 1:
            for pp_stage, events_list in pp_events.items():
                for (node_id, collective_type, src_gpu, dst_gpu, mb, direction) in events_list:
                    if collective_type == "PP_Send":
                        if direction == "fwd" and pp_stage < pp - 1:
                            dst_stage = pp_stage + 1
                        elif direction == "bwd" and pp_stage > 0:
                            dst_stage = pp_stage - 1
                        else:
                            continue

                        found_recv = False
                        for dst_events in pp_events.get(dst_stage, []):
                            (dst_node_id, dst_type, dst_src, dst_dst, dst_mb, dst_dir) = dst_events
                            if dst_type == "PP_Recv" and dst_mb == mb and dst_dir == direction:
                                dag.edges.append(DAGEdge(src_node_id=node_id, dst_node_id=dst_node_id))
                                found_recv = True
                                break

                        if not found_recv:
                            dst_key = (dst_gpu, mb, direction)
                            if dst_key in slot_entry_node:
                                dag.edges.append(DAGEdge(src_node_id=node_id, dst_node_id=slot_entry_node[dst_key]))

        if pp > 1 and not pp_events:
            for dp_rank in range(dp):
                for pp_stage in range(pp):
                    if pp_stage not in trace_paths:
                        continue
                    slots = scheduler.schedule_for_stage(pp_stage)
                    for slot in slots:
                        mb = slot.microbatch_id
                        direction = slot.direction
                        if direction == "fwd" and pp_stage < pp - 1:
                            dst_stage = pp_stage + 1
                        elif direction == "bwd" and pp_stage > 0:
                            dst_stage = pp_stage - 1
                        else:
                            continue

                        for ep_rank in range(ep):
                            for tp_rank in range(tp):
                                for cp_rank in range(cp):
                                    src_gpu = global_rank(dp_rank, pp_stage, ep_rank, tp_rank, cp_rank)
                                    dst_gpu = global_rank(dp_rank, dst_stage, ep_rank, tp_rank, cp_rank)

                                    pp_send = CommNode(
                                        node_id=node_id_counter,
                                        src_gpu=src_gpu,
                                        dst_gpu=dst_gpu,
                                        bytes=activation_bytes,
                                        collective_type="PP_Send",
                                        layer_id=0,
                                        phase=direction,
                                        flow_id=flow_id_counter,
                                    )
                                    dag.comm_nodes.append(pp_send)
                                    node_id_counter += 1
                                    flow_id_counter += 1

                                    src_key = (src_gpu, mb, direction)
                                    if src_key in slot_last_node:
                                        dag.edges.append(
                                            DAGEdge(src_node_id=slot_last_node[src_key], dst_node_id=pp_send.node_id)
                                        )

                                    dst_key = (dst_gpu, mb, direction)
                                    if dst_key in slot_entry_node:
                                        dag.edges.append(
                                            DAGEdge(src_node_id=pp_send.node_id, dst_node_id=slot_entry_node[dst_key])
                                        )
                                    else:
                                        if direction == "bwd":
                                            for bwd_phase in ("bwd_ig", "bwd_wg"):
                                                dst_key2 = (dst_gpu, mb, bwd_phase)
                                                if dst_key2 in slot_entry_node:
                                                    dag.edges.append(
                                                        DAGEdge(
                                                            src_node_id=pp_send.node_id,
                                                            dst_node_id=slot_entry_node[dst_key2],
                                                        )
                                                    )
                                                    break

        for dp_rank in range(dp):
            for pp_stage in range(pp):
                for ep_rank in range(ep):
                    for tp_rank in range(tp):
                        for cp_rank in range(cp):
                            replica_rank = global_rank(dp_rank, pp_stage, ep_rank, tp_rank, cp_rank)
                            slots = scheduler.schedule_for_stage(pp_stage)
                            prev_node_id: int | None = None
                            for slot in slots:
                                mb = slot.microbatch_id
                                direction = slot.direction
                                key = (replica_rank, mb, direction)
                                if key in slot_last_node:
                                    if prev_node_id is not None:
                                        prev_slot_idx = slots.index(slot) - 1
                                        if prev_slot_idx >= 0:
                                            prev_slot = slots[prev_slot_idx]
                                            prev_key = (replica_rank, prev_slot.microbatch_id, prev_slot.direction)
                                            if prev_key in slot_entry_node:
                                                prev_last = slot_last_node.get(prev_key)
                                                this_first = slot_entry_node.get(key)
                                                if prev_last is not None and this_first is not None:
                                                    dag.edges.append(DAGEdge(src_node_id=prev_last, dst_node_id=this_first))
                                    prev_node_id = slot_last_node[key]

        return dag
