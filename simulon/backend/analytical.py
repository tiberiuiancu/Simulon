"""Standalone simulation functions for the analytical backend.

Supports two network-simulation modes:
  - "flow"    : decompose collectives into P2P flows, populate per-flow BW,
                replay the flow-level DAG (existing behaviour).
  - "collective": keep CollectiveNode intact, compute a single scalar duration
                  per collective using the SimAI analytical formula,
                  replay the DAG with CollectiveNodes as atomic nodes.
"""

from __future__ import annotations

import logging

from simulon.backend.dag import DAGTracerConfig, ExecutionDAG, SimulationResult, replay
from simulon.backend.dag.collective_populate import populate_collective_network
from simulon.backend.dag.network_populate import populate_network
from simulon.backend.dag.trace_tracer import MegatronDagTracer
from simulon.backend.network import decompose_collectives_in_dag
from simulon.collective import CCLDecomposer, NCCLDecomposer, RCCLDecomposer
from simulon.config.dc import DatacenterConfig, NICSpec
from simulon.config.resolve import resolve_nccl_profile, resolve_node_spec, resolve_scale_out
from simulon.config.scenario import ScenarioConfig
from simulon.config.workload import CollectiveWorkload, MegatronWorkload

logger = logging.getLogger(__name__)

_CCL_MAP: dict[str, type[CCLDecomposer]] = {"nccl": NCCLDecomposer, "rccl": RCCLDecomposer}


def _ccl_from_scenario(scenario: ScenarioConfig) -> CCLDecomposer:
    library = scenario.collective.library
    cls = _CCL_MAP.get(library)
    if cls is None:
        raise ValueError(f"Unknown CCL library {library!r}. Supported: {sorted(_CCL_MAP)}")
    return cls()


def _tracer_config_from_scenario(scenario: ScenarioConfig) -> DAGTracerConfig:
    c = scenario.collective
    algorithm = c.algorithm if c.algorithm != "auto" else "ring"
    return DAGTracerConfig(num_channels=c.num_channels, algorithm=algorithm)


def _nic_bw_GBps(dc: DatacenterConfig) -> tuple[float, int]:
    node = resolve_node_spec(dc)
    nics_per_node = node.nics_per_node

    scale_out = resolve_scale_out(dc)
    if scale_out and scale_out.nic:
        nic = scale_out.nic
        if isinstance(nic, NICSpec) and nic.speed:
            from simulon.backend.dag.network_populate import _parse_speed

            bw = _parse_speed(nic.speed) / 1e6
            return bw, nics_per_node
    return 400e9 / 8 / 1e9, nics_per_node


def run_trace(scenario: ScenarioConfig, _resolved_algorithm: str | None = None) -> ExecutionDAG:
    """Build an ExecutionDAG from a scenario without populating durations."""
    if isinstance(scenario.workload, MegatronWorkload):
        cfg = _tracer_config_from_scenario(scenario)
        if _resolved_algorithm is not None:
            cfg.algorithm = _resolved_algorithm
        tracer = MegatronDagTracer(cfg, ccl=_ccl_from_scenario(scenario))
        return tracer.trace(scenario.workload, scenario.datacenter)

    elif isinstance(scenario.workload, CollectiveWorkload):
        from simulon.backend.dag.collective_tracer import build_collective_dag

        c = scenario.collective
        algorithm = _resolved_algorithm or (c.algorithm if c.algorithm != "auto" else "ring")
        return build_collective_dag(
            workload=scenario.workload,
            datacenter=scenario.datacenter,
            algorithm=algorithm,
            num_channels=c.num_channels,
            ccl=_ccl_from_scenario(scenario),
        )
    else:
        raise ValueError(f"Unsupported workload type {type(scenario.workload).__name__}")


def simulate(
    scenario: ScenarioConfig, *, network_simulation: str = "collective"
) -> tuple[ExecutionDAG, SimulationResult]:
    """Run the full simulation pipeline.

    Parameters
    ----------
    scenario : ScenarioConfig
        The scenario to simulate.
    network_simulation : {"flow", "collective"}
        "flow"     — decompose collectives into P2P flows (existing behaviour).
        "collective" — keep collectives intact, use SimAI analytical formula.
    compact, ignore_oom, ignore_missing :
        Kept for backward compatibility (currently no-ops at this layer).
    """
    if network_simulation not in ("flow", "collective"):
        raise ValueError(
            f"network_simulation must be 'flow' or 'collective', got {network_simulation!r}"
        )

    intra_override: float | None = None
    inter_override: float | None = None
    resolved_algorithm: str | None = None

    if isinstance(scenario.workload, MegatronWorkload):
        cfg = scenario.workload.config
        tp = int(cfg.get("tensor-model-parallel-size", 1))
        pp = int(cfg.get("pipeline-model-parallel-size", 1))
        ep = int(cfg.get("expert-model-parallel-size", 1))
        num_gpus = int(cfg.get("num_gpus", cfg.get("num-gpus", tp * pp * ep)))
        dp = max(1, num_gpus // (tp * pp * ep))
        logger.info(
            "Building DAG  (GPUs=%d  tp=%d  pp=%d  ep=%d  dp=%d) ...", num_gpus, tp, pp, ep, dp
        )

    elif isinstance(scenario.workload, CollectiveWorkload):
        wl = scenario.workload
        dc = scenario.datacenter
        resolved_node = resolve_node_spec(dc)
        gpus_per_node = resolved_node.gpus_per_node
        if gpus_per_node is None:
            raise ValueError("node.gpus_per_node must be set after resolution")
        num_ranks = dc.num_nodes * gpus_per_node
        collective_type = wl.collective_type.value

        nccl_profile = resolve_nccl_profile(dc)
        nic_bw, nics_per_node = _nic_bw_GBps(dc)
        from simulon.collective.calbusbw import cal_busbw as _cal_busbw

        resolved_algorithm, intra_bw_GBps, inter_bw_GBps = _cal_busbw(
            collective_type=collective_type,
            message_size_bytes=wl.message_size_bytes,
            num_nodes=dc.num_nodes,
            gpus_per_node=gpus_per_node,
            nics_per_node=nics_per_node,
            nic_bw_GBps=nic_bw,
            nccl_profile=nccl_profile,
            algorithm=scenario.collective.algorithm,
        )
        intra_override = intra_bw_GBps * 1e6
        inter_override = inter_bw_GBps * 1e6 if inter_bw_GBps is not None else None
        logger.info(
            "Populating network (algo=%s, intra_bw=%.1f GB/s, inter_bw=%s) ...",
            resolved_algorithm,
            intra_bw_GBps,
            f"{inter_bw_GBps:.1f} GB/s" if inter_bw_GBps is not None else "N/A",
        )
        logger.info(
            "Building collective DAG  (type=%s  ranks=%d  size=%d bytes  algo=%s) ...",
            collective_type,
            num_ranks,
            wl.message_size_bytes,
            resolved_algorithm,
        )

    dag = run_trace(scenario, _resolved_algorithm=resolved_algorithm)
    logger.info(
        "  DAG built: %d compute nodes, %d collective nodes, %d edges",
        len(dag.compute_nodes),
        len(dag.collective_nodes),
        len(dag.edges),
    )

    if network_simulation == "flow":
        decompose_collectives_in_dag(dag)
        logger.info("  After decomposition: %d comm nodes", len(dag.comm_nodes))

        dc = scenario.datacenter
        logger.info("Populating network durations (%d comm nodes) ...", len(dag.comm_nodes))
        populate_network(
            dag,
            dc,
            bw_override_bytes_per_ms=intra_override,
            inter_bw_override_bytes_per_ms=inter_override,
        )
        logger.info("  Network durations resolved")
    else:  # network_simulation == "collective"
        dc = scenario.datacenter
        logger.info(
            "Populating collective network durations (%d collective nodes) ...",
            len(dag.collective_nodes),
        )
        populate_collective_network(dag, dc)
        logger.info("  Collective durations resolved")

    total_nodes = len(dag.compute_nodes) + len(dag.comm_nodes) + len(dag.collective_nodes)
    logger.info("Replaying DAG (%d nodes) ...", total_nodes)
    result = replay(dag, network_simulation=network_simulation)
    if dag.total_flops is not None:
        result.total_flops = dag.total_flops

    logger.info("  Replay done: total_time=%.3f ms", result.total_time_ms)

    return dag, result
