from itertools import product
from pathlib import Path
from typing import Optional

import typer
import yaml

def _dump_profile(data: dict, f) -> None:
    """Write a GPU profile YAML with compact one-line-per-kernel_run formatting."""
    top = {k: v for k, v in data.items() if k != "kernel_runs"}
    f.write(yaml.dump(top, default_flow_style=False, sort_keys=False))
    runs = data.get("kernel_runs", [])
    if runs:
        f.write("kernel_runs:\n")
        for run in runs:
            f.write(f"  - {yaml.dump(run, default_flow_style=True, sort_keys=False).strip()}\n")


app = typer.Typer(name="simulon", help="AI cluster simulator")
profile_app = typer.Typer(help="Profile local hardware and save templates.")
app.add_typer(profile_app, name="profile")

from simulon.cli.install import app as install_app  # noqa: E402
app.add_typer(install_app, name="install", help="Install third-party components (apex, deepgemm, m4).")


@app.command()
def simulate(
    scenario: str = typer.Argument(..., help="Path to scenario.yaml"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path for trace.json (default: trace.json)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Print per-GPU timing summary"),
):
    """Run simulation and write a Chrome/Perfetto trace."""
    import json
    from simulon.backend.astra_sim import AstraSimBackend
    from simulon.backend.dag.chrome_trace import to_chrome_trace
    from simulon.config.scenario import ScenarioConfig
    from simulon.config.workload import MegatronWorkload

    with open(scenario) as f:
        raw = yaml.safe_load(f)
    sc = ScenarioConfig.model_validate(raw)

    if not isinstance(sc.workload, MegatronWorkload):
        typer.echo("Error: simulate only supports MegatronWorkload scenarios.", err=True)
        raise typer.Exit(1)

    backend = AstraSimBackend()
    dag, result = backend.simulate(sc)

    p = sc.workload.parallelism
    t = sc.workload.training
    tp = p.tp
    pp_val = p.pp
    ep = p.ep
    dp = p.dp if p.dp is not None else t.num_gpus // (tp * pp_val * ep)

    trace_dict = to_chrome_trace(dag, tp=tp, pp=pp_val, dp=dp, ep=ep)

    if output is None:
        output = Path("trace.json")
    with open(output, "w") as f:
        json.dump(trace_dict, f)

    typer.echo(f"Trace written to {output}")
    typer.echo(f"  GPUs: {len(result.per_gpu_times_ms)}  |  Total: {result.total_time_ms:.3f} ms")
    typer.echo(f"  Load in https://ui.perfetto.dev or chrome://tracing")

    if verbose:
        typer.echo("")
        typer.echo("Per-GPU finish times (ms):")
        for gpu_rank in sorted(result.per_gpu_times_ms):
            compute = result.compute_time_ms.get(gpu_rank, 0.0)
            comm = result.comm_time_ms.get(gpu_rank, 0.0)
            finish = result.per_gpu_times_ms[gpu_rank]
            typer.echo(f"  GPU {gpu_rank:3d}: finish={finish:.3f}  compute={compute:.3f}  comm={comm:.3f}")


@profile_app.command("gpu")
def profile_gpu(
    name: str = typer.Option(..., "--name", "-n", help="GPU model name (e.g. H100-SXM5-80GB)"),
    vendor: Optional[str] = typer.Option(None, help="GPU vendor: nvidia | amd"),
    memory_capacity_gb: Optional[float] = typer.Option(None, help="HBM capacity in GB"),
    tdp_w: Optional[float] = typer.Option(None, help="TDP in watts"),
    flops_multiplier: float = typer.Option(1.0, help="Scalar multiplier applied to all profiled FLOP rates"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Load arch from templates/model/<name>.yaml"),
    hidden_size: Optional[int] = typer.Option(None, help="Transformer hidden dimension"),
    num_heads: Optional[int] = typer.Option(None, help="Number of attention heads"),
    ffn_hidden_size: Optional[int] = typer.Option(None, help="FFN intermediate dimension"),
    vocab_size: Optional[int] = typer.Option(None, help="Vocabulary size"),
    num_experts: Optional[int] = typer.Option(None, help="Number of MoE experts (0 = dense)"),
    top_k: Optional[int] = typer.Option(None, help="Top-k routing for MoE"),
    dtype: str = typer.Option("bf16", help="Compute dtype: fp32 | fp16 | bf16 | fp8"),
    tp: str = typer.Option("1", help="TP degree(s), comma-separated (e.g. 1,2,4,8)"),
    ep: str = typer.Option("1", help="EP degree(s), comma-separated"),
    batch_size: str = typer.Option("1", help="Micro-batch size(s), comma-separated"),
    seq_len: str = typer.Option("2048", help="Sequence length(s), comma-separated"),
    swiglu: bool = typer.Option(False, help="Use SwiGLU activation shape for mlp_act"),
    epoch_num: int = typer.Option(10, help="Number of timed iterations per kernel"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Re-profile and replace all matching existing kernel entries"),
    purge: bool = typer.Option(False, "--purge", help="Clear all existing kernel_runs before profiling"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print sweep configurations without running"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output path for the template YAML. Defaults to templates/gpu/<name>.yaml",
    ),
):
    """Profile the local GPU and save a hardware template.

    Appends new kernel_runs to an existing YAML file (or creates it).
    Use --model to load arch from a template, and --tp/--ep/--batch-size/--seq-len
    to sweep over multiple configurations in one invocation.
    """
    from simulon.config.common import DType
    from simulon.config.dc import GPUSpec
    from simulon.profiling.sweep import SweepResult, parse_sweep, run_sweep

    dtype_enum = DType(dtype)
    tp_values = parse_sweep(tp)
    ep_values = parse_sweep(ep)
    batch_sizes = parse_sweep(batch_size)
    seq_lens = parse_sweep(seq_len)

    # Build kernel_params: start from model template, then apply manual overrides.
    kernel_params: dict = {}

    if model is not None:
        from simulon.profiling.models import load_model_template, model_to_kernel_params
        try:
            tmpl = load_model_template(model)
        except FileNotFoundError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1)
        kernel_params = model_to_kernel_params(tmpl)

    if hidden_size is not None:
        kernel_params["hidden_size"] = hidden_size
    if num_heads is not None:
        kernel_params["num_heads"] = num_heads
    if ffn_hidden_size is not None:
        kernel_params["ffn_hidden_size"] = ffn_hidden_size
    if vocab_size is not None:
        kernel_params["vocab_size"] = vocab_size
    if num_experts is not None:
        kernel_params["num_experts"] = num_experts
    if top_k is not None:
        kernel_params["top_k"] = top_k
    if swiglu:
        kernel_params["swiglu"] = True

    required = ["hidden_size", "num_heads", "ffn_hidden_size", "vocab_size"]
    missing = [k for k in required if k not in kernel_params]
    if missing:
        typer.echo(
            f"Error: missing required arch fields: {missing}. "
            "Use --model or pass them directly.",
            err=True,
        )
        raise typer.Exit(1)

    is_moe = kernel_params.get("num_experts", 0) > 0
    num_experts = kernel_params.get("num_experts", 0)
    configs = [
        (t, e, b, s)
        for t, e, b, s in product(tp_values, ep_values, batch_sizes, seq_lens)
        if not (e > 1 and not is_moe)
        and (not is_moe or e <= num_experts)
    ]

    if dry_run:
        label = model or name
        typer.echo(f"Sweep configurations for GPU '{name}' (arch: {label}):")
        for t, e, b, s in configs:
            typer.echo(f"  tp={t} ep={e} bs={b} seq={s}")
        typer.echo(f"Total: {len(configs)} configurations")
        raise typer.Exit(0)

    # Load or initialise the existing profile.
    if output is None:
        safe_name = name.lower().replace(" ", "-")
        output = Path("templates/gpu") / f"{safe_name}.yaml"

    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        with open(output) as f:
            existing = yaml.safe_load(f) or {}
        existing_runs: list[dict] = existing.get("kernel_runs", [])
    else:
        existing = {
            "name": name,
            "vendor": vendor,
            "memory_capacity_gb": memory_capacity_gb,
            "tdp_w": tdp_w,
            "flops_multiplier": flops_multiplier,
        }
        existing = {k: v for k, v in existing.items() if v is not None}
        existing_runs = []

    if purge:
        existing_runs = []

    # For skip logic: pass existing_runs unless --overwrite (forces re-profiling).
    runs_for_skip = [] if overwrite else existing_runs

    # Run sweep with progress display.
    label = model or name
    results: list[SweepResult] = []

    try:
        from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
        ) as progress:
            task_id = progress.add_task(f"Profiling {label}", total=len(configs))
            for t, e, b, s in configs:
                progress.update(task_id, description=f"Profiling {label}  tp={t} ep={e} bs={b} seq={s}")
                single = run_sweep(kernel_params, [t], [e], [b], [s], dtype_enum, epoch_num, existing_runs=runs_for_skip)
                r = single[0] if single else SweepResult(
                    config={"tp": t, "ep": e, "batch_size": b, "seq_len": s}, runs=None, oom=True
                )
                results.append(r)
                if r.oom:
                    progress.update(task_id, advance=1, description=f"[red]✗ OOM  tp={t} ep={e} bs={b} seq={s}")
                else:
                    progress.update(task_id, advance=1)

    except ImportError:
        for t, e, b, s in configs:
            typer.echo(f"  Running tp={t} ep={e} bs={b} seq={s} ...")
            single = run_sweep(kernel_params, [t], [e], [b], [s], dtype_enum, epoch_num, existing_runs=runs_for_skip)
            r = single[0] if single else SweepResult(
                config={"tp": t, "ep": e, "batch_size": b, "seq_len": s}, runs=None, oom=True
            )
            results.append(r)
            typer.echo("    ✗ OOM" if r.oom else "    ✓ Done")

    # Merge new runs into existing_runs, replacing any entries with the same
    # (kernel, params) key (handles re-runs of entries with insufficient timings
    # and --overwrite).
    completed = [r for r in results if not r.oom]
    oom_count = sum(1 for r in results if r.oom)
    all_new_runs = [kr for r in completed if r.runs for kr in r.runs]

    if all_new_runs:
        new_keys = {(kr.kernel, frozenset(kr.params.items())) for kr in all_new_runs}
        existing_runs = [
            r for r in existing_runs
            if (r["kernel"], frozenset(r["params"].items())) not in new_keys
        ]
        existing_runs.extend(kr.model_dump() for kr in all_new_runs)

    existing["kernel_runs"] = existing_runs

    with open(output, "w") as f:
        _dump_profile(existing, f)

    typer.echo(f"Saved {len(all_new_runs)} kernel runs to {output}")
    typer.echo(f"Completed {len(completed)}/{len(results)} configs, {oom_count} skipped (OOM)")

    GPUSpec.model_validate(existing)
    typer.echo("Profile validated successfully.")
