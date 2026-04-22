"""CLI output formatting helpers for simulon simulate command."""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer

from simulon.config.resolve import resolve_node_spec

if TYPE_CHECKING:
    from simulon.backend.dag import SimulationResult
    from simulon.config.dc import DatacenterConfig
    from simulon.config.workload import MegatronWorkload


def _print_summary(result: SimulationResult, workload: MegatronWorkload | None = None) -> None:
    """Print a human-readable simulation summary to stdout.

    Args:
        result: SimulationResult from AnalyticalBackend.simulate().
        workload: Optional MegatronWorkload for throughput metrics.
    """
    total = result.total_time_ms
    n_gpus = len(result.per_gpu_times_ms)

    def _pct(ms: float) -> str:
        return f"{ms / total * 100:5.1f}%" if total > 0 else "  n/a"

    typer.echo(f"\nIteration wall time:  {total:.3f} ms")
    typer.echo(f"  (metrics below averaged across {n_gpus} GPUs)")
    typer.echo("")
    typer.echo(f"  Compute:        {result.compute_ms:9.3f} ms  {_pct(result.compute_ms)}")
    typer.echo(f"  Exposed comm:   {result.exposed_comm_ms:9.3f} ms  {_pct(result.exposed_comm_ms)}")
    for ctype, ms in sorted(result.exposed_comm_by_type.items(), key=lambda x: -x[1]):
        typer.echo(f"    {ctype + ':':22s} {ms:9.3f} ms")
    typer.echo(f"  Bubble:         {result.bubble_ms:9.3f} ms  {_pct(result.bubble_ms)}")
    typer.echo("")
    typer.echo(f"  Overlapped comm:{result.overlapped_comm_ms:9.3f} ms        "
               "  (hidden by compute, not in totals above)")
    typer.echo("")

    if workload is not None and total > 0:
        from simulon.profiling.models import _resolve_model
        t = workload.training
        tokens_per_iter = t.global_batch_size * t.sequence_length
        iter_time_s = total / 1000.0
        throughput_tps = tokens_per_iter / iter_time_s
        per_gpu_tps = throughput_tps / n_gpus
        typer.echo(f"Throughput:           {per_gpu_tps:,.1f} tokens/s ({throughput_tps:,.1f} tokens/s)")
        resolved = _resolve_model(workload.model)
        if resolved.gflops_per_train_token is not None:
            tflops = throughput_tps * resolved.gflops_per_train_token / 1e3
            per_gpu_tflops = tflops / n_gpus
            typer.echo(f"                      {per_gpu_tflops:.2f} TFLOPs/s ({tflops:.2f} TFLOPs/s)")
        typer.echo("")


def _print_collective_summary(workload, result: SimulationResult, datacenter: DatacenterConfig) -> None:
    node = resolve_node_spec(datacenter)
    gpus_per_node = node.gpus_per_node
    if gpus_per_node is None:
        gpus_per_node = 0
    num_ranks = datacenter.cluster.num_nodes * gpus_per_node
    typer.echo(f"\nCollective wall time:  {result.total_time_ms:.3f} ms")
    typer.echo(f"  Type:          {workload.collective_type.value}")
    typer.echo(f"  Message size:  {workload.message_size_bytes:,} bytes")
    typer.echo(f"  Ranks:         {num_ranks}")
    typer.echo("")


def _print_energy_summary(result) -> None:
    """Print a human-readable energy summary to stdout."""
    typer.echo(f"Energy per iteration:  {result.total_wh:.4f} Wh"
               f"   (avg cluster power: {result.avg_power_kw:.2f} kW)")
    typer.echo(f"  Hardware subtotal:   {result.hardware_subtotal_wh:.4f} Wh")
    for comp in result.breakdown:
        label = comp.component + ":"
        typer.echo(f"    {label:26s} {comp.wh:10.4f} Wh  ({comp.pct:5.1f}%)")
    typer.echo(f"  PUE overhead:        {result.pue_overhead_wh:.4f} Wh")
    typer.echo("")


def _print_cost_summary(result) -> None:
    """Print a human-readable cost summary to stdout."""

    def _fmt(v: float) -> str:
        return f"${v:,.0f}"

    typer.echo("Cost model")
    capex = result.capex
    range_str = ""
    if capex.min is not None and capex.max is not None:
        range_str = f"  [{_fmt(capex.min)} \u2013 {_fmt(capex.max)}]"
    typer.echo(f"  CAPEX total:    {_fmt(capex.total)}{range_str}")
    for comp in capex.breakdown:
        label = comp.component + ":"
        range_comp = ""
        if comp.min is not None and comp.max is not None:
            range_comp = f"  [{_fmt(comp.min)} \u2013 {_fmt(comp.max)}]"
        typer.echo(f"    {label:26s} {_fmt(comp.total):>14s}{range_comp}  ({comp.pct:5.1f}%)")
    typer.echo(f"  OPEX per run:   {_fmt(result.opex_per_run)}")
    if result.cost_per_run is not None:
        cpr = result.cost_per_run
        typer.echo(
            f"  Cost per run:   {_fmt(cpr.total)}"
            f"  (capex {_fmt(cpr.capex_component)} + opex {_fmt(cpr.opex_component)})"
        )
    typer.echo("")
