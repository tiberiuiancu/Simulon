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
        dp = max(1, num_gpus // (tp * pp * ep))
        num_microbatches = int(
            cfg.get("num_microbatches")
            or (
                int(cfg.get("global-batch-size", 0))
                // max(1, dp * int(cfg.get("micro-batch-size", 1)))
            )
        )

        scheduler = OneFOneBScheduler(pp, num_microbatches)

        slot_entry_node: dict[tuple[int, int, str], int] = {}
        slot_last_node: dict[tuple[int, int, str], int] = {}

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

        for pp_stage, path in trace_paths.items():
            trace_file = TraceFileParser.parse(path)
            events = sorted(trace_file.events, key=lambda e: e.timestamp_ms)
            replica_ranks = ranks_for_stage(pp_stage)

            for replica_rank in replica_ranks:
                last_node_id: int | None = None
                active_microbatch_id = -1
                active_phase = ""
                active_direction = ""
                pending_entry = False

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
                            active_direction = str(event.metadata.get("direction", active_phase))
                        pending_entry = True
                    elif event.type == "slot_end":
                        active_microbatch_id = -1
                        active_phase = ""
                        active_direction = ""
                        pending_entry = False

                    if i + 1 < len(events):
                        next_event = events[i + 1]
                        duration_ms = next_event.timestamp_ms - event.timestamp_ms
                        if duration_ms > 0:
                            cn = ComputeNode(
                                node_id=node_id_counter,
                                gpu_rank=replica_rank,
                                kernel="trace_compute",
                                layer_id=-1,
                                microbatch_id=active_microbatch_id,
                                pipeline_stage=pp_stage,
                                phase=active_phase,
                                duration_ms=duration_ms,
                            )
                            dag.compute_nodes.append(cn)
                            if last_node_id is not None:
                                dag.edges.append(DAGEdge(src_node_id=last_node_id, dst_node_id=node_id_counter))
                            last_node_id = node_id_counter
                            node_id_counter += 1

                            if pending_entry:
                                slot_key = (replica_rank, active_microbatch_id, active_direction)
                                if slot_key not in slot_entry_node:
                                    slot_entry_node[slot_key] = cn.node_id
                                pending_entry = False
                            if active_direction:
                                slot_key = (replica_rank, active_microbatch_id, active_direction)
                                slot_last_node[slot_key] = cn.node_id

                    if event.type == "collective":
                        collective_type = str(event.metadata.get("collective_type", ""))
                        if collective_type in ("PP_Send", "PP_Recv"):
                            continue
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

                        flow_node_ids: list[int] = []
                        for flow in result.flows:
                            comm_node = CommNode(
                                node_id=node_id_counter,
                                src_gpu=flow.src,
                                dst_gpu=flow.dst,
                                bytes=flow.flow_size,
                                collective_type=collective_type,
                                layer_id=-1,
                                phase=active_phase,
                                flow_id=flow.flow_id,
                                parent_flow_ids=flow.parent_flow_ids,
                            )
                            dag.comm_nodes.append(comm_node)
                            flow_node_ids.append(node_id_counter)
                            node_id_counter += 1

                        if flow_node_ids:
                            if last_node_id is not None:
                                for fid in flow_node_ids:
                                    dag.edges.append(DAGEdge(src_node_id=last_node_id, dst_node_id=fid))
                            last_node_id = flow_node_ids[-1]

                            if pending_entry:
                                slot_key = (replica_rank, active_microbatch_id, active_direction)
                                if slot_key not in slot_entry_node:
                                    slot_entry_node[slot_key] = flow_node_ids[0]
                                pending_entry = False
                            if active_direction:
                                slot_key = (replica_rank, active_microbatch_id, active_direction)
                                slot_last_node[slot_key] = flow_node_ids[-1]

        if pp > 1:
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

                        src_gpu = global_rank(dp_rank, pp_stage, 0, 0)
                        dst_gpu = global_rank(dp_rank, dst_stage, 0, 0)

                        pp_send = CommNode(
                            node_id=node_id_counter,
                            src_gpu=src_gpu,
                            dst_gpu=dst_gpu,
                            bytes=0,
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

                        for ep_rank in range(ep):
                            for tp_rank in range(tp):
                                for cp_rank in range(cp):
                                    dst_gpu_tr = global_rank(dp_rank, dst_stage, ep_rank, tp_rank, cp_rank)
                                    dst_key = (dst_gpu_tr, mb, direction)
                                    if dst_key in slot_entry_node:
                                        dag.edges.append(
                                            DAGEdge(src_node_id=pp_send.node_id, dst_node_id=slot_entry_node[dst_key])
                                        )
                                    else:
                                        if direction == "bwd":
                                            for bwd_phase in ("bwd_ig", "bwd_wg"):
                                                dst_key2 = (dst_gpu_tr, mb, bwd_phase)
                                                if dst_key2 in slot_entry_node:
                                                    dag.edges.append(
                                                        DAGEdge(
                                                            src_node_id=pp_send.node_id,
                                                            dst_node_id=slot_entry_node[dst_key2],
                                                        )
                                                    )
                                                    break

        return dag
