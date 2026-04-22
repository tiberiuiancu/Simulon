import logging
from pathlib import Path
from dataclasses import dataclass, field

import yaml
from pydantic import TypeAdapter

from simulon.backend.base import Backend
from simulon.backend.dag import DAGTracerConfig, ExecutionDAG, populate_dag, replay, SimulationResult
from simulon.backend.dag.merge import merge_dags
from simulon.backend.dag.populate import populate_network
from simulon.backend.dag.replayer import summarize_subset
from simulon.collective import CCLDecomposer, NCCLDecomposer, RCCLDecomposer
from simulon.collective.calbusbw import cal_busbw
from simulon.config.dc import DatacenterConfig, GPUSpec, NICSpec
from simulon.config.placement import NodeSlice, place_workloads
from simulon.config.resolve import (
    resolve_gpu_spec,
    resolve_nccl_profile,
    resolve_node_spec,
    resolve_scale_out,
)
from simulon.config.scenario import ScenarioConfig
from simulon.config.workload import CollectiveWorkload, InferenceWorkload, MegatronWorkload, WorkloadConfig

logger = logging.getLogger(__name__)


@dataclass
class SimulationOutput:
    dag: ExecutionDAG
    result: SimulationResult
    by_workload: dict[str, SimulationResult] = field(default_factory=dict)
    start_offsets: dict[int, float] = field(default_factory=dict)
    node_id_to_workload: dict[int, str] = field(default_factory=dict)

    def __iter__(self):
        return iter((self.dag, self.result))

    def __getitem__(self, idx: int):
        return (self.dag, self.result)[idx]

_CCL_MAP: dict[str, type[CCLDecomposer]] = {
    "nccl": NCCLDecomposer,
    "rccl": RCCLDecomposer,
}


def _ccl_from_scenario(scenario: ScenarioConfig) -> CCLDecomposer:
    library = scenario.collective.library
    cls = _CCL_MAP.get(library)
    if cls is None:
        raise ValueError(f"Unknown CCL library {library!r}. Supported: {sorted(_CCL_MAP)}")
    return cls()


def _tracer_config_from_scenario(scenario: ScenarioConfig) -> DAGTracerConfig:
    c = scenario.collective
    # "auto" is resolved per-collective in simulate(); for MegatronWorkload tracing,
    # fall back to "ring" as the default algorithm.
    algorithm = c.algorithm if c.algorithm != "auto" else "ring"
    return DAGTracerConfig(
        num_channels=c.num_channels,
        algorithm=algorithm,
    )


def _nic_bw_GBps(dc: DatacenterConfig) -> tuple[float, int]:
    """Return (nic_bw_GBps, nics_per_node) from datacenter spec, or defaults."""
    scale_out = resolve_scale_out(dc)
    if scale_out and scale_out.nic:
        nic = scale_out.nic
        if isinstance(nic, NICSpec) and nic.speed:
            from simulon.backend.dag.populate import _parse_speed
            bw = _parse_speed(nic.speed) / 1e6  # bytes/ms → GB/s
            return bw, nic.nics_per_node
    return 400e9 / 8 / 1e9, 1  # 400 Gbps → 50 GB/s, 1 NIC


def _dtype_bytes(dtype) -> int:
    from simulon.config.common import DType
    return {DType.fp32: 4, DType.fp16: 2, DType.bf16: 2, DType.fp8: 1}.get(dtype, 2)


def _tp_message_size(workload: MegatronWorkload) -> int:
    """Representative TP AllReduce message size in bytes.

    Uses hidden_size × seq_len × micro_batch_size × dtype_bytes — the typical
    size of the activation tensor passed through a TP AllReduce/ReduceScatter.
    Falls back to 256 MiB if hidden_size is not resolvable.
    """
    from simulon.backend.dag.populate import _model_hidden_size
    hidden = _model_hidden_size(workload)
    if hidden is None:
        return 256 * 1024 * 1024
    t = workload.training
    return hidden * t.sequence_length * t.micro_batch_size * _dtype_bytes(t.dtype)


# Keep private aliases so call sites inside this module are unchanged.
_resolve_gpu_spec = resolve_gpu_spec
_workload_adapter = TypeAdapter(WorkloadConfig)


def _load_datacenter(dc: DatacenterConfig | Path) -> DatacenterConfig:
    if isinstance(dc, DatacenterConfig):
        return dc
    with open(dc) as f:
        return DatacenterConfig.model_validate(yaml.safe_load(f))


def _load_workload(workload: WorkloadConfig | Path) -> WorkloadConfig:
    if not isinstance(workload, Path):
        return workload
    with open(workload) as f:
        return _workload_adapter.validate_python(yaml.safe_load(f))


class AnalyticalBackend(Backend):
    """Python analytical backend that produces a GPU-agnostic execution DAG."""

    def _resolve_workloads(self, scenario: ScenarioConfig) -> list[tuple[str, WorkloadConfig, NodeSlice]]:
        datacenter = _load_datacenter(scenario.datacenter)
        placements = place_workloads(scenario.workloads, datacenter)
        resolved: list[tuple[str, WorkloadConfig, NodeSlice]] = []
        for instance in scenario.workloads:
            resolved.append((instance.name, _load_workload(instance.workload), placements[instance.name]))
        return resolved

    def _slice_datacenter(self, datacenter: DatacenterConfig, node_slice: NodeSlice) -> DatacenterConfig:
        gpus_per_node = datacenter.node.gpus_per_node
        if gpus_per_node is None:
            raise ValueError("datacenter.node.gpus_per_node is required for slicing")
        data = datacenter.model_dump()
        data["cluster"]["num_nodes"] = node_slice.num_gpus // gpus_per_node
        return DatacenterConfig.model_validate(data)

    def _trace_all_workloads(
        self,
        scenario: ScenarioConfig,
        compact: bool = False,
        _resolved_algorithm: str | None = None,
    ) -> list[tuple[str, ExecutionDAG]]:
        datacenter = _load_datacenter(scenario.datacenter)
        resolved = self._resolve_workloads(scenario)
        results: list[tuple[str, ExecutionDAG]] = []

        for name, workload, node_slice in resolved:
            sliced_dc = self._slice_datacenter(datacenter, node_slice)

            if isinstance(workload, MegatronWorkload):
                from simulon.backend.dag.megatron_tracer import MegatronDAGTracer
                cfg = _tracer_config_from_scenario(scenario)
                cfg.compact = compact
                if _resolved_algorithm is not None:
                    cfg.algorithm = _resolved_algorithm
                tracer = MegatronDAGTracer(cfg, ccl=_ccl_from_scenario(scenario))
                dag = tracer.trace(workload, sliced_dc)
            elif isinstance(workload, CollectiveWorkload):
                from simulon.backend.dag.collective_tracer import build_collective_dag
                c = scenario.collective
                algorithm = _resolved_algorithm or (c.algorithm if c.algorithm != "auto" else "ring")
                dag = build_collective_dag(
                    workload=workload,
                    datacenter=sliced_dc,
                    algorithm=algorithm,
                    num_channels=c.num_channels,
                    ccl=_ccl_from_scenario(scenario),
                )
            elif isinstance(workload, InferenceWorkload):
                raise NotImplementedError(
                    "AnalyticalBackend does not yet support InferenceWorkload"
                )
            else:
                raise ValueError(f"AnalyticalBackend does not support {type(workload).__name__}")

            results.append((name, dag))

        return results

    def run_trace(self, scenario: ScenarioConfig, compact: bool = False, _resolved_algorithm: str | None = None) -> ExecutionDAG:
        dags = self._trace_all_workloads(scenario, compact=compact, _resolved_algorithm=_resolved_algorithm)
        if len(dags) != 1:
            raise ValueError(
                f"run_trace() requires exactly one workload; found {len(dags)}. "
                f"Use _trace_all_workloads() for multi-workload scenarios."
            )
        return dags[0][1]

    def run(self, scenario: ScenarioConfig) -> dict:
        dags = self._trace_all_workloads(scenario)
        if len(dags) == 1:
            dag = dags[0][1]
            d = dag.to_dict()
            return {
                "status": "success",
                "compute_nodes": len(dag.compute_nodes),
                "comm_nodes": len(dag.comm_nodes),
                "edges": len(dag.edges),
                "dag": d,
            }
        return {
            "status": "success",
            "workloads": {
                name: {
                    "compute_nodes": len(dag.compute_nodes),
                    "comm_nodes": len(dag.comm_nodes),
                    "edges": len(dag.edges),
                    "dag": dag.to_dict(),
                }
                for name, dag in dags
            },
        }

    def _simulate_single_workload(
        self,
        scenario: ScenarioConfig,
        compact: bool = False,
        ignore_oom: bool = False,
        ignore_missing: bool = False,
    ) -> SimulationOutput:
        if isinstance(scenario.workload, MegatronWorkload):
            from simulon.backend.dag.megatron_tracer import MegatronDAGTracer
            p = scenario.workload.parallelism
            t = scenario.workload.training
            num_gpus = t.num_gpus
            logger.info("Building DAG  (GPUs=%d  tp=%d  pp=%d  ep=%d  dp=%d) ...",
                        num_gpus, p.tp, p.pp, p.ep,
                        p.dp if p.dp is not None else num_gpus // (p.tp * p.pp * p.ep))

            # Resolve algorithm and BW overrides from nccl profile (if available).
            dc = scenario.datacenter
            resolved_node = resolve_node_spec(dc)
            gpus_per_node = resolved_node.gpus_per_node or num_gpus
            nccl_profile = resolve_nccl_profile(dc)
            intra_override: float | None = None
            inter_override: float | None = None
            resolved_algo: str | None = None

            if nccl_profile is not None:
                tp_msg_size = _tp_message_size(scenario.workload)
                nic_bw, nics_per_node = _nic_bw_GBps(dc)
                try:
                    selected_algo, intra_bw_GBps, inter_bw_GBps = cal_busbw(
                        collective_type="AllReduce",
                        message_size_bytes=tp_msg_size,
                        num_nodes=dc.cluster.num_nodes,
                        gpus_per_node=gpus_per_node,
                        nics_per_node=nics_per_node,
                        nic_bw_GBps=nic_bw,
                        nccl_profile=nccl_profile,
                        algorithm=scenario.collective.algorithm,
                    )
                    resolved_algo = selected_algo
                    intra_override = intra_bw_GBps * 1e6  # GB/s → bytes/ms
                    inter_override = inter_bw_GBps * 1e6 if inter_bw_GBps is not None else None
                    logger.info(
                        "Network calibration from nccl profile (algo=%s, intra=%.1f GB/s, inter=%s) ...",
                        selected_algo, intra_bw_GBps,
                        f"{inter_bw_GBps:.1f} GB/s" if inter_bw_GBps is not None else "N/A",
                    )
                except ValueError as e:
                    logger.warning("cal_busbw failed for Megatron workload: %s — using raw link BW.", e)

            dag = self.run_trace(scenario, compact=compact, _resolved_algorithm=resolved_algo)
            logger.info("  DAG built: %d compute nodes, %d comm nodes, %d edges",
                        len(dag.compute_nodes), len(dag.comm_nodes), len(dag.edges))

            gpu_spec = _resolve_gpu_spec(scenario.datacenter)
            logger.info("Resolving compute durations (%d nodes) ...", len(dag.compute_nodes))
            populate_dag(dag, scenario.workload, gpu_spec, ignore_oom=ignore_oom, ignore_missing=ignore_missing)
            logger.info("  Compute durations resolved")

            logger.info("Populating network durations (%d comm nodes) ...", len(dag.comm_nodes))
            populate_network(
                dag, dc,
                bw_override_bytes_per_ms=intra_override,
                inter_bw_override_bytes_per_ms=inter_override,
            )
            logger.info("  Network durations resolved")

            total_nodes = len(dag.compute_nodes) + len(dag.comm_nodes)
            logger.info("Replaying DAG (%d nodes) ...", total_nodes)
            result = replay(dag)
            logger.info("  Replay done: total_time=%.3f ms", result.total_time_ms)

            return SimulationOutput(dag=dag, result=result)

        elif isinstance(scenario.workload, CollectiveWorkload):
            wl = scenario.workload
            dc = scenario.datacenter
            resolved_node = resolve_node_spec(dc)
            gpus_per_node = resolved_node.gpus_per_node
            if gpus_per_node is None:
                raise ValueError("node.gpus_per_node must be set after resolution")
            num_ranks = dc.cluster.num_nodes * gpus_per_node
            collective_type = wl.collective_type.value

            # Derive BW from nccl profile + NIC efficiency table.
            nccl_profile = resolve_nccl_profile(dc)

            nic_bw, nics_per_node = _nic_bw_GBps(dc)
            selected_algo, intra_bw_GBps, inter_bw_GBps = cal_busbw(
                collective_type=collective_type,
                message_size_bytes=wl.message_size_bytes,
                num_nodes=dc.cluster.num_nodes,
                gpus_per_node=gpus_per_node,
                nics_per_node=nics_per_node,
                nic_bw_GBps=nic_bw,
                nccl_profile=nccl_profile,
                algorithm=scenario.collective.algorithm,
            )
            intra_override = intra_bw_GBps * 1e6  # GB/s → bytes/ms
            inter_override = inter_bw_GBps * 1e6 if inter_bw_GBps is not None else None
            logger.info(
                "Populating network (algo=%s, intra_bw=%.1f GB/s, inter_bw=%s) ...",
                selected_algo, intra_bw_GBps,
                f"{inter_bw_GBps:.1f} GB/s" if inter_bw_GBps is not None else "N/A",
            )

            logger.info("Building collective DAG  (type=%s  ranks=%d  size=%d bytes  algo=%s) ...",
                        collective_type, num_ranks, wl.message_size_bytes, selected_algo)
            dag = self.run_trace(scenario, _resolved_algorithm=selected_algo)
            logger.info("  DAG built: %d comm nodes", len(dag.comm_nodes))

            populate_network(
                dag, dc,
                bw_override_bytes_per_ms=intra_override,
                inter_bw_override_bytes_per_ms=inter_override,
            )
            logger.info("  Network durations resolved")

            logger.info("Replaying DAG (%d nodes) ...", len(dag.comm_nodes))
            result = replay(dag)
            logger.info("  Replay done: total_time=%.3f ms", result.total_time_ms)

            return SimulationOutput(dag=dag, result=result)

        else:
            raise ValueError(f"AnalyticalBackend does not support {type(scenario.workload).__name__}")

    def _simulate_multi_workload(
        self,
        scenario: ScenarioConfig,
        compact: bool = False,
        ignore_oom: bool = False,
        ignore_missing: bool = False,
    ) -> SimulationOutput:
        datacenter = _load_datacenter(scenario.datacenter)
        resolved = self._resolve_workloads(scenario)
        gpu_spec = _resolve_gpu_spec(datacenter)

        entries: list[tuple[str, ExecutionDAG, SimulationResult, NodeSlice, WorkloadConfig]] = []
        for name, workload, node_slice in resolved:
            sliced_dc = self._slice_datacenter(datacenter, node_slice)

            if isinstance(workload, MegatronWorkload):
                from simulon.backend.dag.megatron_tracer import MegatronDAGTracer
                cfg = _tracer_config_from_scenario(scenario)
                cfg.compact = compact
                cfg.cache_dir = None
                tracer = MegatronDAGTracer(cfg, ccl=_ccl_from_scenario(scenario))
                dag = tracer.trace(workload, sliced_dc)
            elif isinstance(workload, CollectiveWorkload):
                from simulon.backend.dag.collective_tracer import build_collective_dag
                c = scenario.collective
                algorithm = c.algorithm if c.algorithm != "auto" else "ring"
                dag = build_collective_dag(
                    workload=workload,
                    datacenter=sliced_dc,
                    algorithm=algorithm,
                    num_channels=c.num_channels,
                    ccl=_ccl_from_scenario(scenario),
                )
            elif isinstance(workload, InferenceWorkload):
                raise NotImplementedError(
                    "AnalyticalBackend does not yet support InferenceWorkload"
                )
            else:
                raise ValueError(f"AnalyticalBackend does not support {type(workload).__name__}")

            if isinstance(workload, MegatronWorkload):
                populate_dag(dag, workload, gpu_spec, ignore_oom=ignore_oom, ignore_missing=ignore_missing)

            populate_network(dag, sliced_dc)
            result = replay(dag)
            entries.append((name, dag, result, node_slice, workload))

        result_by_name = {name: result for name, _, result, _, _ in entries}
        instance_by_name = {instance.name: instance for instance in scenario.workloads}

        effective_start: dict[str, float] = {}

        def _compute_effective_start(name: str) -> float:
            if name in effective_start:
                return effective_start[name]
            instance = instance_by_name[name]
            dep_finish = 0.0
            for dep in instance.start.after_finish:
                dep_finish = max(dep_finish, result_by_name[dep].total_time_ms)
            start = max(instance.start.offset_ms, dep_finish)
            effective_start[name] = start
            return start

        for name, _, _, _, _ in entries:
            _compute_effective_start(name)

        start_offsets: dict[int, float] = {}
        for name, _, _, node_slice, _ in entries:
            offset = effective_start[name]
            for gpu in range(node_slice.start_gpu_rank, node_slice.start_gpu_rank + node_slice.num_gpus):
                start_offsets[gpu] = offset

        dags = [(name, dag) for name, dag, _, _, _ in entries]
        merged_dag, node_id_to_workload = merge_dags(dags)

        populate_network(merged_dag, datacenter)
        aggregate_result = replay(merged_dag, start_offsets=start_offsets)

        per_workload: dict[str, SimulationResult] = {}
        for name, _, _, _, _ in entries:
            workload_node_ids = {nid for nid, wl_name in node_id_to_workload.items() if wl_name == name}
            per_workload[name] = summarize_subset(merged_dag, workload_node_ids)

        workload_labels: dict[int, str] = {}
        for name, _, _, node_slice, _ in entries:
            for gpu in range(node_slice.start_gpu_rank, node_slice.start_gpu_rank + node_slice.num_gpus):
                workload_labels[gpu] = name

        return SimulationOutput(
            dag=merged_dag,
            result=aggregate_result,
            by_workload=per_workload,
            start_offsets=start_offsets,
            node_id_to_workload=node_id_to_workload,
        )

    def simulate(self, scenario: ScenarioConfig, compact: bool = False, ignore_oom: bool = False, ignore_missing: bool = False) -> SimulationOutput:
        if len(scenario.workloads) == 1:
            return self._simulate_single_workload(scenario, compact=compact, ignore_oom=ignore_oom, ignore_missing=ignore_missing)
        else:
            return self._simulate_multi_workload(scenario, compact=compact, ignore_oom=ignore_oom, ignore_missing=ignore_missing)
