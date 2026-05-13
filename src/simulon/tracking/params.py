from __future__ import annotations

from typing import TYPE_CHECKING, Union

from simulon.config.dc import DatacenterConfig, GPUSpec
from simulon.config.resolve import resolve_node_spec
from simulon.config.scenario import ScenarioConfig
from simulon.config.workload import CollectiveWorkload, LLMSpec, MegatronDeprecatedWorkload

if TYPE_CHECKING:
    from simulon.backend.dag.replayer import SimulationResult


def extract_params(scenario: ScenarioConfig) -> dict[str, Union[str, int, float, bool]]:
    """Return a flat dict of dotted-key params covering everything needed to reproduce a run."""
    params: dict[str, Union[str, int, float, bool]] = {}

    # --- collective ---
    c = scenario.collective
    params["collective.library"] = c.library
    params["collective.algorithm"] = c.algorithm
    params["collective.num_channels"] = c.num_channels

    # --- workload ---
    wl = scenario.workload
    if isinstance(wl, MegatronDeprecatedWorkload):
        p = wl.parallelism
        t = wl.training

        params["workload.framework"] = wl.framework
        params["workload.tp"] = p.tp
        params["workload.pp"] = p.pp
        params["workload.ep"] = p.ep
        params["workload.sp"] = p.sp
        params["workload.vpp"] = p.vpp
        params["workload.distributed_optimizer"] = p.distributed_optimizer
        params["workload.pipeline_schedule"] = p.pipeline_schedule
        if p.dp is not None:
            params["workload.dp"] = p.dp
        if p.num_microbatches is not None:
            params["workload.num_microbatches"] = p.num_microbatches

        params["training.num_gpus"] = t.num_gpus
        params["training.global_batch_size"] = t.global_batch_size
        params["training.micro_batch_size"] = t.micro_batch_size
        params["training.sequence_length"] = t.sequence_length
        params["training.dtype"] = t.dtype.value
        params["training.flash_attention"] = t.flash_attention
        params["training.iterations"] = t.iterations

        if isinstance(wl.model, str):
            params["model.name"] = wl.model
        elif isinstance(wl.model, LLMSpec):
            if wl.model.name is not None:
                params["model.name"] = wl.model.name
            if wl.model.from_ is not None:
                params["model.from"] = wl.model.from_
            if wl.model.hidden_size is not None:
                params["model.hidden_size"] = wl.model.hidden_size
            if wl.model.num_layers is not None:
                params["model.num_layers"] = wl.model.num_layers
            if wl.model.num_heads is not None:
                params["model.num_heads"] = wl.model.num_heads
            if wl.model.ffn_hidden_size is not None:
                params["model.ffn_hidden_size"] = wl.model.ffn_hidden_size
            if wl.model.vocab_size is not None:
                params["model.vocab_size"] = wl.model.vocab_size
            if wl.model.num_experts is not None:
                params["model.num_experts"] = wl.model.num_experts
            if wl.model.top_k is not None:
                params["model.top_k"] = wl.model.top_k
            params["model.swiglu"] = wl.model.swiglu

    elif isinstance(wl, CollectiveWorkload):
        params["workload.framework"] = wl.framework
        params["workload.collective_type"] = wl.collective_type.value
        params["workload.message_size_bytes"] = wl.message_size_bytes

    # --- datacenter: GPU name ---
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
    """Return a flat dict of metrics from a SimulationResult."""
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
