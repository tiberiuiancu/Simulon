"""simulon compare — run one or more scenarios and compare against reference measurements."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import yaml

from simulon.backend.analytical import simulate as run_simulation
from simulon.config.resolve import resolve_datacenter, resolve_workload
from simulon.config.scenario import ScenarioConfig
from simulon.config.workload import MegatronWorkload


def _load_scenario(scenario_path: str) -> ScenarioConfig:
    with open(scenario_path) as f:
        raw = yaml.safe_load(f)
    sc = ScenarioConfig.model_validate(raw)
    if isinstance(sc.datacenter, Path):
        sc.datacenter = resolve_datacenter(sc.datacenter)
    if isinstance(sc.workload, Path):
        sc.workload = resolve_workload(sc.workload)
    return sc


def _load_reference(ref_path: Path) -> dict:
    with open(ref_path) as f:
        return yaml.safe_load(f) or {}


def _find_default_ref(scenario_path: str) -> Path | None:
    ref = Path(scenario_path).parent / "reference.yaml"
    return ref if ref.is_file() else None


def _compute_metrics(result, workload, gpu_spec=None) -> dict[str, float]:
    metrics: dict[str, float] = {"iter_time_ms": result.total_time_ms}
    if workload is not None and isinstance(workload, MegatronWorkload):
        cfg = workload.config
        tokens_per_iter = cfg.get("global-batch-size", 0) * cfg.get("seq-length", 0)
        if tokens_per_iter > 0 and result.total_time_ms > 0:
            tps = tokens_per_iter / (result.total_time_ms / 1000.0)
            metrics["throughput_tps"] = tps
            metrics["per_gpu_tps"] = tps / max(1, len(result.per_gpu_times_ms))

        if result.total_flops is not None and tokens_per_iter > 0:
            gflops_per_token = result.total_flops / tokens_per_iter / 1e9
            tflops = metrics.get("throughput_tps", 0) * gflops_per_token / 1e3
            n = max(1, len(result.per_gpu_times_ms))
            per_gpu_tflops = tflops / n
            metrics["tflops"] = tflops
            metrics["per_gpu_tflops"] = per_gpu_tflops

            if gpu_spec is not None and gpu_spec.peak_tflops_bf16 is not None:
                metrics["mfu_pct"] = (per_gpu_tflops / gpu_spec.peak_tflops_bf16) * 100

    return metrics


_METRIC_LABELS: dict[str, tuple[str, str]] = {
    "iter_time_ms": ("Iteration time", "ms"),
    "throughput_tps": ("Throughput", "t/s"),
    "per_gpu_tps": ("Per-GPU throughput", "t/s"),
    "tflops": ("TFLOPs/s (total)", "TF/s"),
    "per_gpu_tflops": ("Per-GPU TFLOPs/s", "TF/s"),
    "mfu_pct": ("MFU", "%"),
}


def _fmt_val(key: str, val: float) -> str:
    unit = _METRIC_LABELS.get(key, ("", ""))[1]
    if unit == "%":
        return f"{val:.2f}%"
    elif unit in ("ms",):
        return f"{val:,.1f} ms"
    elif unit in ("t/s",):
        return f"{val:,.0f} t/s"
    elif unit in ("TF/s",):
        return f"{val:.2f} TF/s"
    return f"{val:.3f}"


def _fmt_err(err_pct: float) -> str:
    sign = "+" if err_pct >= 0 else ""
    return f"{sign}{err_pct:.2f}%"


def _print_comparison(
    label: str,
    simulated: dict[str, float],
    reference: dict[str, float] | None,
    breakdown: dict | None = None,
) -> None:
    sep = "─" * 72
    typer.echo(f"\n{label}")
    typer.echo(sep)

    keys = list(_METRIC_LABELS.keys())
    if reference:
        typer.echo(f"  {'Metric':<28s}  {'Simulated':>14s}  {'Reference':>14s}  {'Error':>8s}")
        typer.echo(f"  {'':─<28s}  {'':─>14s}  {'':─>14s}  {'':─>8s}")
        for key in keys:
            if key not in simulated:
                continue
            sim_val = simulated[key]
            sim_str = _fmt_val(key, sim_val)
            ref_val = reference.get(key)
            if ref_val is not None:
                err = (sim_val - ref_val) / ref_val * 100
                ref_str = _fmt_val(key, ref_val)
                err_str = _fmt_err(err)
            else:
                ref_str = "—"
                err_str = "—"
            label_str = _METRIC_LABELS[key][0]
            typer.echo(f"  {label_str:<28s}  {sim_str:>14s}  {ref_str:>14s}  {err_str:>8s}")
    else:
        typer.echo(f"  {'Metric':<28s}  {'Simulated':>14s}")
        typer.echo(f"  {'':─<28s}  {'':─>14s}")
        for key in keys:
            if key not in simulated:
                continue
            label_str = _METRIC_LABELS[key][0]
            typer.echo(f"  {label_str:<28s}  {_fmt_val(key, simulated[key]):>14s}")

    if breakdown:
        typer.echo("")
        typer.echo(f"  {'Breakdown':<28s}  {'ms':>14s}  {'%':>8s}")
        typer.echo(f"  {'':─<28s}  {'':─>14s}  {'':─>8s}")
        total = simulated["iter_time_ms"]
        items = [
            ("Compute", breakdown.get("compute_ms", 0)),
            ("Exposed comm", breakdown.get("exposed_comm_ms", 0)),
            ("Bubble", breakdown.get("bubble_ms", 0)),
        ]
        for name, val in items:
            if val > 0:
                pct = val / total * 100 if total > 0 else 0
                typer.echo(f"  {name:<28s}  {val:>11,.1f} ms  {pct:>6.1f}%")

    typer.echo("")


def compare(
    scenarios: list[str] = typer.Argument(..., help="One or more scenario.yaml paths"),
    ref: Optional[list[Path]] = typer.Option(
        None,
        "--ref",
        help=(
            "Reference measurements YAML (one per scenario, or a single file applied to all). "
            "Defaults to reference.yaml next to each scenario if it exists."
        ),
    ),
    network_simulation: str = typer.Option(
        "collective",
        "--network-simulation",
        help="Network simulation backend: 'flow' or 'collective'",
    ),
    breakdown: bool = typer.Option(
        True,
        "--breakdown/--no-breakdown",
        help="Show compute/comm/bubble breakdown",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Write results to a YAML file (simulated metrics + errors vs reference).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable backend progress logging"),
):
    """Run one or more scenarios and compare results against reference measurements.

    Reference files use a simple YAML format::

        iter_time_ms: 11654
        throughput_tps: 89314   # optional
        mfu_pct: 35.2           # optional

    If --ref is omitted, simulon looks for reference.yaml next to each scenario.
    """
    if verbose:
        import logging

        logging.basicConfig(format="%(message)s", level=logging.INFO)

    # Resolve reference files: one per scenario, or broadcast a single ref to all
    refs: list[Path | None]
    if ref:
        if len(ref) == 1:
            refs = [ref[0]] * len(scenarios)
        elif len(ref) == len(scenarios):
            refs = list(ref)
        else:
            typer.echo(
                f"Error: --ref count ({len(ref)}) must be 1 or match scenario count ({len(scenarios)})",
                err=True,
            )
            raise typer.Exit(1)
    else:
        refs = [_find_default_ref(s) for s in scenarios]

    for scenario_path, ref_path in zip(scenarios, refs):
        sc = _load_scenario(scenario_path)

        try:
            from simulon.config.resolve import resolve_gpu_spec

            gpu_spec = resolve_gpu_spec(sc.datacenter)
        except Exception:
            gpu_spec = None

        dag, result = run_simulation(sc, network_simulation=network_simulation)

        simulated = _compute_metrics(result, sc.workload, gpu_spec)
        reference = _load_reference(ref_path) if ref_path else None

        bd = None
        if breakdown:
            bd = {
                "compute_ms": result.compute_ms,
                "exposed_comm_ms": result.exposed_comm_ms,
                "bubble_ms": result.bubble_ms,
            }

        label = f"Scenario: {scenario_path}"
        if ref_path:
            label += f"  (ref: {ref_path})"

        _print_comparison(label, simulated, reference, bd)

        if output is not None:
            out_records: dict = {"scenario": scenario_path, "simulated": simulated}
            if reference:
                errors = {}
                for key, sim_val in simulated.items():
                    ref_val = reference.get(key)
                    if ref_val is not None:
                        errors[key] = round((sim_val - ref_val) / ref_val * 100, 4)
                out_records["reference"] = reference
                out_records["error_pct"] = errors
            if bd:
                out_records["breakdown"] = bd
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w") as f:
                yaml.dump(out_records, f, default_flow_style=False, sort_keys=False)
            typer.echo(f"Results saved to {output}")
