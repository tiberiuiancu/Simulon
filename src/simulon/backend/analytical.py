import logging

from simulon.backend.base import Backend
from simulon.backend.dag import DAGTracerConfig, ExecutionDAG, populate_dag, replay, SimulationResult
from simulon.backend.dag.populate import populate_network
from simulon.collective import CCLDecomposer, NCCLDecomposer, RCCLDecomposer
from simulon.collective.calbusbw import cal_busbw
from simulon.config.dc import DatacenterConfig, GPUSpec, NICSpec
from simulon.config.resolve import (
    resolve_gpu_spec,
    resolve_nccl_profile,
    resolve_node_spec,
    resolve_scale_out,
)
from simulon.config.scenario import ScenarioConfig
from simulon.config.workload import CollectiveWorkload, MegatronWorkload

logger = logging.getLogger(__name__)

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


def _nic_bw_GBps(dc: DatacenterConfig) -> float:
    """Return NIC bandwidth in GB/s from datacenter spec, or default 400 Gbps × 0.85."""
    scale_out = resolve_scale_out(dc)
    if scale_out and scale_out.nic:
        nic = scale_out.nic
        if isinstance(nic, NICSpec) and nic.speed:
            from simulon.backend.dag.populate import _parse_speed
            # _parse_speed returns bytes/ms; convert to GB/s
            return _parse_speed(nic.speed) / 1e6
    return 400e9 / 8 / 1e9  # 400 Gbps → 50 GB/s


# Keep private aliases so call sites inside this module are unchanged.
_resolve_gpu_spec = resolve_gpu_spec


class AnalyticalBackend(Backend):
    """Python analytical backend that produces a GPU-agnostic execution DAG."""

    def run(self, scenario: ScenarioConfig) -> dict:
        dag = self.run_trace(scenario)
        d = dag.to_dict()
        return {
            "status": "success",
            "compute_nodes": len(dag.compute_nodes),
            "comm_nodes": len(dag.comm_nodes),
            "edges": len(dag.edges),
            "dag": d,
        }

    def run_trace(self, scenario: ScenarioConfig, compact: bool = False, _resolved_algorithm: str | None = None) -> ExecutionDAG:
        from simulon.backend.dag.megatron_tracer import MegatronDAGTracer
        if isinstance(scenario.workload, MegatronWorkload):
            cfg = _tracer_config_from_scenario(scenario)
            cfg.compact = compact
            tracer = MegatronDAGTracer(cfg, ccl=_ccl_from_scenario(scenario))
            return tracer.trace(scenario.workload, scenario.datacenter)
        elif isinstance(scenario.workload, CollectiveWorkload):
            from simulon.backend.dag.collective_tracer import build_collective_dag
            c = scenario.collective
            # "auto" must be resolved before decompose_collective is called;
            # if not resolved externally, fall back to "ring".
            algorithm = _resolved_algorithm or (c.algorithm if c.algorithm != "auto" else "ring")
            return build_collective_dag(
                workload=scenario.workload,
                datacenter=scenario.datacenter,
                algorithm=algorithm,
                num_channels=c.num_channels,
                ccl=_ccl_from_scenario(scenario),
            )
        else:
            raise ValueError(f"AnalyticalBackend does not support {type(scenario.workload).__name__}")

    def simulate(self, scenario: ScenarioConfig, compact: bool = False) -> tuple[ExecutionDAG, SimulationResult]:
        if isinstance(scenario.workload, MegatronWorkload):
            from simulon.backend.dag.megatron_tracer import MegatronDAGTracer
            p = scenario.workload.parallelism
            t = scenario.workload.training
            num_gpus = t.num_gpus
            logger.info("Building DAG  (GPUs=%d  tp=%d  pp=%d  ep=%d  dp=%d) ...",
                        num_gpus, p.tp, p.pp, p.ep,
                        p.dp if p.dp is not None else num_gpus // (p.tp * p.pp * p.ep))
            dag = self.run_trace(scenario, compact=compact)
            logger.info("  DAG built: %d compute nodes, %d comm nodes, %d edges",
                        len(dag.compute_nodes), len(dag.comm_nodes), len(dag.edges))

            gpu_spec = _resolve_gpu_spec(scenario.datacenter)
            logger.info("Resolving compute durations (%d nodes) ...", len(dag.compute_nodes))
            populate_dag(dag, scenario.workload, gpu_spec)
            logger.info("  Compute durations resolved")

            logger.info("Populating network durations (%d comm nodes) ...", len(dag.comm_nodes))
            populate_network(dag, scenario.datacenter)
            logger.info("  Network durations resolved")

            total_nodes = len(dag.compute_nodes) + len(dag.comm_nodes)
            logger.info("Replaying DAG (%d nodes) ...", total_nodes)
            result = replay(dag)
            logger.info("  Replay done: total_time=%.3f ms", result.total_time_ms)

            return dag, result

        elif isinstance(scenario.workload, CollectiveWorkload):
            wl = scenario.workload
            dc = scenario.datacenter
            resolved_node = resolve_node_spec(dc)
            gpus_per_node = resolved_node.gpus_per_node
            if gpus_per_node is None:
                raise ValueError("node.gpus_per_node must be set after resolution")
            num_ranks = dc.cluster.num_nodes * gpus_per_node
            collective_type = wl.collective_type.value

            per_step_latency_ms = (
                scenario.collective.per_step_latency_us.get(collective_type, 0.0) / 1000.0
            )

            # Manual override takes full precedence (backward compat).
            manual_bw_GBps = scenario.collective.per_collective_bw_GBps.get(collective_type)
            if manual_bw_GBps is not None:
                intra_override = manual_bw_GBps * 1e6  # GB/s → bytes/ms
                inter_override = None
                c_algo = scenario.collective.algorithm
                selected_algo = c_algo if c_algo != "auto" else "ring"
                logger.info(
                    "Populating network (manual bw override=%.1f GB/s, algo=%s) ...",
                    manual_bw_GBps, selected_algo,
                )
            else:
                # Derive BW from nccl profile + NIC efficiency table.
                nccl_profile = resolve_nccl_profile(dc)

                nic_bw = _nic_bw_GBps(dc)
                # TODO: NICSpec has no nics_per_node field. Hardcoded to 1.0.
                # Add nics_per_node to NICSpec / ScaleOutSpec and read it here.
                selected_algo, intra_bw_GBps, inter_bw_GBps = cal_busbw(
                    collective_type=collective_type,
                    message_size_bytes=wl.message_size_bytes,
                    num_nodes=dc.cluster.num_nodes,
                    gpus_per_node=gpus_per_node,
                    nics_per_node=1.0,
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
                per_step_latency_ms=per_step_latency_ms,
                bw_override_bytes_per_ms=intra_override,
                inter_bw_override_bytes_per_ms=inter_override,
            )
            logger.info("  Network durations resolved")

            logger.info("Replaying DAG (%d nodes) ...", len(dag.comm_nodes))
            result = replay(dag)
            logger.info("  Replay done: total_time=%.3f ms", result.total_time_ms)

            return dag, result

        else:
            raise ValueError(f"AnalyticalBackend does not support {type(scenario.workload).__name__}")
