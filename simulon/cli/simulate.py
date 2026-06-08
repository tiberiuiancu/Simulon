import json
import os
from pathlib import Path

import typer
import yaml

from simulon.backend.analytical import simulate as run_simulation
from simulon.backend.dag.chrome_trace import to_chrome_trace
from simulon.config.resolve import resolve_datacenter, resolve_node_spec, resolve_workload, workload_hash
from simulon.config.scenario import ScenarioConfig
from simulon.config.workload import MegatronWorkload
from simulon.tracking import get_trackers
from simulon.tracking.params import extract_metrics, extract_params


def simulate(
    scenario: str = typer.Argument(..., help="Path to scenario.yaml"),
    summary: bool = typer.Option(
        True, "--summary/--no-summary", help="Print iteration summary to stdout"
    ),
    chrome: Path | None = typer.Option(
        None, "--chrome", help="Write Chrome/Perfetto trace to this path"
    ),
    chrome_compact: bool = typer.Option(
        False, "--chrome-compact", help="Only include profiled ranks (excludes extrapolated traces)"
    ),
    dag_out: Path | None = typer.Option(
        None, "--dag", help="Write timing-populated DAG JSON to this path"
    ),
    goal: Path | None = typer.Option(
        None, "--goal", help="Write GOAL trace to this path for use with ATLAHS/LogGOPSim"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable backend progress logging"),
    energy: bool = typer.Option(
        False, "--energy", help="Compute and print per-iteration energy breakdown"
    ),
    cost: bool = typer.Option(
        False, "--cost", help="Compute and print cost breakdown (implies --energy)"
    ),
    trace: bool = typer.Option(
        False,
        "--trace",
        help="Auto-generate execution traces if they are missing before simulating",
    ),
    network_simulation: str = typer.Option(
        "collective",
        "--network-simulation",
        help="Network simulation backend: 'flow' (per-flow BW) or 'collective' (analytical)",
    ),
):
    """Run simulation and print an iteration summary.

    Optionally export a Chrome/Perfetto trace (--chrome) and/or a
    timing-populated DAG JSON (--dag) for offline analysis.
    """
    import tempfile

    trackers = get_trackers(scenario)

    with open(scenario) as f:
        raw = yaml.safe_load(f)
    sc = ScenarioConfig.model_validate(raw)

    if isinstance(sc.datacenter, Path):
        sc.datacenter = resolve_datacenter(sc.datacenter)

    if isinstance(sc.workload, Path):
        sc.workload = resolve_workload(sc.workload)

    if trace and isinstance(sc.workload, MegatronWorkload):
        from simulon.cli.trace import generate_trace

        typer.echo("Ensuring traces exist ...")
        generate_trace(scenario=scenario)

    if verbose:
        import logging

        logging.basicConfig(format="%(message)s", level=logging.INFO)

    trackers = get_trackers()

    try:
        for tracker in trackers:
            tracker.start_run()

        from simulon.config.resolve import resolve_gpu_spec

        try:
            gpu_spec = resolve_gpu_spec(sc.datacenter)
        except Exception:
            gpu_spec = None

        dag, result = run_simulation(sc, network_simulation=network_simulation)

        if trackers:
            params = extract_params(sc)

            if isinstance(sc.workload, MegatronWorkload):
                params["workload_hash"] = workload_hash(sc.workload)

            metrics = extract_metrics(result)

            if isinstance(sc.workload, MegatronWorkload):
                derived = _compute_training_metrics(result, sc.workload, gpu_spec)
                metrics.update({k: v for k, v in derived.items() if v is not None})

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", prefix="scenario_", delete=False
            ) as tmp:
                yaml.dump(raw, tmp)
                scenario_artifact_path = Path(tmp.name)
            try:
                for tracker in trackers:
                    tracker.log_params(params)
                    tracker.log_metrics(metrics)
                    tracker.log_artifact(scenario_artifact_path)
            finally:
                scenario_artifact_path.unlink(missing_ok=True)

        if summary:
            from simulon.config.workload import CollectiveWorkload

            if isinstance(sc.workload, CollectiveWorkload):
                _print_collective_summary(sc.workload, result, sc.datacenter)
            else:
                _print_summary(result, sc.workload, gpu_spec)

        if isinstance(sc.workload, MegatronWorkload):
            energy_result = None
            if energy or cost:
                from simulon.energy import compute_energy

                energy_result = compute_energy(dag, sc)
                if energy_result is not None:
                    _print_energy_summary(energy_result)

            if cost and energy_result is not None:
                from simulon.cost import compute_cost

                cost_result = compute_cost(sc, energy_result)
                _print_cost_summary(cost_result)

        if chrome is not None:
            if isinstance(sc.workload, MegatronWorkload):
                from simulon.backend.dag.trace_tracer import ParallelConfig

                config = ParallelConfig.from_workload(sc.workload)
            else:
                from simulon.backend.dag.trace_tracer import ParallelConfig

                config = ParallelConfig(tp=1, cp=1, ep=1, dp=1, pp=1, etp=1, edp=1, num_gpus=1)
            trace_dict = to_chrome_trace(dag, config=config, only_profiled=chrome_compact)
            with open(chrome, "w") as f:
                json.dump(trace_dict, f)
            typer.echo(f"Chrome trace written to {chrome}  (open in https://ui.perfetto.dev)")
            for tracker in trackers:
                tracker.log_artifact(chrome)

        if dag_out is not None:
            with open(dag_out, "w") as f:
                json.dump(dag.to_dict(), f)
            typer.echo(f"DAG written to {dag_out}")
            for tracker in trackers:
                tracker.log_artifact(dag_out)

        if goal is not None:
            from simulon.backend.dag.goal_trace import write_goal_trace

            write_goal_trace(dag, goal)
            typer.echo(f"GOAL trace written to {goal}  (feed to ATLAHS/LogGOPSim via txt2bin)")
            for tracker in trackers:
                tracker.log_artifact(goal)

    finally:
        for tracker in trackers:
            tracker.end_run()


def _print_summary(result, workload=None, gpu_spec=None) -> None:
    """Print a human-readable simulation summary to stdout.

    Args:
        result: SimulationResult from AnalyticalBackend.simulate().
        workload: Optional workload for throughput metrics.
        gpu_spec: Optional resolved GPUSpec for GPU peak TFLOPs (needed for MFU).
    """

    total = result.total_time_ms
    n_gpus = len(result.per_gpu_times_ms)

    def _pct(ms: float) -> str:
        return f"{ms / total * 100:5.1f}%" if total > 0 else "  n/a"

    typer.echo(f"\nIteration wall time:  {total:.3f} ms")
    typer.echo(f"  (metrics below averaged across {n_gpus} GPUs)")
    typer.echo("")
    typer.echo(f"  Compute:        {result.compute_ms:9.3f} ms  {_pct(result.compute_ms)}")
    typer.echo(
        f"  Exposed comm:   {result.exposed_comm_ms:9.3f} ms  {_pct(result.exposed_comm_ms)}"
    )
    for ctype, ms in sorted(result.exposed_comm_by_type.items(), key=lambda x: -x[1]):
        typer.echo(f"    {ctype + ':':22s} {ms:9.3f} ms")
    typer.echo(f"  Bubble:         {result.bubble_ms:9.3f} ms  {_pct(result.bubble_ms)}")
    typer.echo("")
    typer.echo(
        f"  Overlapped comm:{result.overlapped_comm_ms:9.3f} ms        "
        "  (hidden by compute, not in totals above)"
    )
    typer.echo("")

    if workload is not None and total > 0:
        metrics = _compute_training_metrics(result, workload, gpu_spec)
        if metrics:
            typer.echo(
                f"Throughput:           {metrics['per_gpu_tps']:,.1f} tokens/s "
                f"({metrics['throughput_tps']:,.1f} tokens/s)"
            )
            if metrics.get("per_gpu_tflops") is not None:
                typer.echo(
                    f"                      {metrics['per_gpu_tflops']:.2f} TFLOPs/s "
                    f"({metrics['tflops']:.2f} TFLOPs/s)"
                )
            if metrics.get("mfu_pct") is not None:
                typer.echo(f"  MFU:                {metrics['mfu_pct']:.1f}%")
            typer.echo("")


def _compute_training_metrics(result, workload, gpu_spec=None) -> dict[str, float]:
    """Compute throughput, TFLOPs, and MFU from a simulation result."""
    try:
        total = result.total_time_ms
        n_gpus = len(result.per_gpu_times_ms)
        if total <= 0 or n_gpus == 0:
            return {}

        from simulon.config.workload import MegatronWorkload

        if not isinstance(workload, MegatronWorkload):
            return {}

        cfg = workload.config
        tokens_per_iter = cfg.get("global-batch-size", 0) * cfg.get("seq-length", 0)

        iter_time_s = total / 1000.0
        if iter_time_s <= 0:
            return {}
        throughput_tps = tokens_per_iter / iter_time_s
        per_gpu_tps = throughput_tps / n_gpus

        gflops_per_token = None
        if result.total_flops is not None and tokens_per_iter > 0:
            gflops_per_token = result.total_flops / tokens_per_iter / 1e9

        metrics: dict[str, float] = {"throughput_tps": throughput_tps, "per_gpu_tps": per_gpu_tps}
        if gflops_per_token is not None:
            tflops = throughput_tps * gflops_per_token / 1e3
            per_gpu_tflops = tflops / n_gpus
            metrics["gflops_per_token"] = gflops_per_token
            metrics["tflops"] = tflops
            metrics["per_gpu_tflops"] = per_gpu_tflops

            if gpu_spec is not None and gpu_spec.peak_tflops_bf16 is not None:
                metrics["mfu_pct"] = (per_gpu_tflops / gpu_spec.peak_tflops_bf16) * 100
        return metrics
    except Exception:
        return {}


def _print_collective_summary(workload, result, datacenter) -> None:
    node = resolve_node_spec(datacenter)
    gpus_per_node = node.gpus_per_node
    if gpus_per_node is None:
        gpus_per_node = 0
    num_ranks = datacenter.num_nodes * gpus_per_node
    typer.echo(f"\nCollective wall time:  {result.total_time_ms:.3f} ms")
    typer.echo(f"  Type:          {workload.collective_type.value}")
    typer.echo(f"  Message size:  {workload.message_size_bytes:,} bytes")
    typer.echo(f"  Ranks:         {num_ranks}")
    typer.echo("")


def _print_energy_summary(result) -> None:
    """Print a human-readable energy summary to stdout."""
    typer.echo(
        f"Energy per iteration:  {result.total_wh:.4f} Wh"
        f"   (avg cluster power: {result.avg_power_kw:.2f} kW)"
    )
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
        range_str = f"  [ {_fmt(capex.min)} \u2013 {_fmt(capex.max)}]"
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
