from __future__ import annotations

from typing import TYPE_CHECKING, Union

from simulon.config.dc import DatacenterConfig, GPUSpec
from simulon.config.resolve import resolve_node_spec
from simulon.config.scenario import ScenarioConfig
from simulon.config.workload import CollectiveWorkload, MegatronWorkload

if TYPE_CHECKING:
    from simulon.backend.dag.replayer import SimulationResult


def extract_params(scenario: ScenarioConfig) -> dict[str, Union[str, int, float, bool]]:
    params: dict[str, Union[str, int, float, bool]] = {}

    c = scenario.collective
    params["collective.library"] = c.library
    params["collective.algorithm"] = c.algorithm
    params["collective.num_channels"] = c.num_channels

    wl = scenario.workload
    if isinstance(wl, MegatronWorkload):
        cfg = wl.config
        params["workload.framework"] = wl.framework
        for key in (
            "tensor-model-parallel-size",
            "pipeline-model-parallel-size",
            "expert-model-parallel-size",
            "micro-batch-size",
            "global-batch-size",
            "seq-length",
            "num-layers",
            "hidden-size",
            "num-attention-heads",
            "ffn-hidden-size",
            "vocab-size",
        ):
            if key in cfg:
                params[f"workload.{key.replace('-', '_')}"] = cfg[key]
        if "num_gpus" in cfg:
            params["workload.num_gpus"] = cfg["num_gpus"]
    elif isinstance(wl, CollectiveWorkload):
        params["workload.framework"] = wl.framework
        params["workload.collective_type"] = wl.collective_type.value
        params["workload.message_size_bytes"] = wl.message_size_bytes

    from pathlib import Path
    if isinstance(scenario.datacenter, Path):
        params["datacenter.config_path"] = str(scenario.datacenter)
    elif isinstance(scenario.datacenter, DatacenterConfig):
        node = resolve_node_spec(scenario.datacenter)
        gpu = node.gpu
        if isinstance(gpu, str):
            params["datacenter.gpu"] = gpu
        elif isinstance(gpu, GPUSpec):
            gpu_name = gpu.name or gpu.from_
            if gpu_name is not None:
                params["datacenter.gpu"] = gpu_name

    return params


def extract_metrics(result: SimulationResult) -> dict[str, float]:
    metrics: dict[str, float] = {
        "total_time_ms": result.total_time_ms,
        "compute_ms": result.compute_ms,
        "exposed_comm_ms": result.exposed_comm_ms,
        "bubble_ms": result.bubble_ms,
        "overlapped_comm_ms": result.overlapped_comm_ms,
    }
    for ctype, ms in result.exposed_comm_by_type.items():
        metrics[f"exposed_comm.{ctype}_ms"] = ms
    if result.total_flops is not None:
        metrics["total_flops"] = result.total_flops
    return metrics
