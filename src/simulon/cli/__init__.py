import json
import os
from itertools import product
from pathlib import Path
from typing import Optional

import subprocess
import sys

import typer
import yaml

def _dump_profile(data: dict, f) -> None:
    """Write a GPU profile YAML with compact one-line-per-entry formatting."""
    top = {k: v for k, v in data.items() if k not in ("kernel_runs", "oom_kernel_runs")}
    if top:
        f.write(yaml.dump(top, default_flow_style=False, sort_keys=False))
    runs = data.get("kernel_runs", [])
    if runs:
        f.write("kernel_runs:\n")
        for run in runs:
            f.write(f"  - {yaml.dump(run, default_flow_style=True, sort_keys=False).strip()}\n")
    oom_kr = data.get("oom_kernel_runs", [])
    if oom_kr:
        f.write("oom_kernel_runs:\n")
        for run in oom_kr:
            f.write(f"  - {yaml.dump(run, default_flow_style=True, sort_keys=False).strip()}\n")


app = typer.Typer(name="simulon", help="AI cluster simulator")
profile_app = typer.Typer(help="Profile local hardware and save templates.")
app.add_typer(profile_app, name="profile")

trace_app = typer.Typer(help="Trace generation commands.")
app.add_typer(trace_app, name="trace")

from simulon.cli.install import app as install_app  # noqa: E402
app.add_typer(install_app, name="install", help="Install third-party components (apex, deepgemm, m4).")

from simulon.config.resolve import resolve_node_spec


@app.command()
def simulate(
    scenario: str = typer.Argument(..., help="Path to scenario.yaml"),
    summary: bool = typer.Option(True, "--summary/--no-summary", help="Print iteration summary to stdout"),
    chrome: Optional[Path] = typer.Option(None, "--chrome", help="Write Chrome/Perfetto trace to this path"),
    dag_out: Optional[Path] = typer.Option(None, "--dag", help="Write timing-populated DAG JSON to this path"),
    goal: Optional[Path] = typer.Option(None, "--goal", help="Write GOAL trace to this path for use with ATLAHS/LogGOPSim"),
    compact: bool = typer.Option(False, "--compact", help="Fuse consecutive compute-only sublayers into single DAG nodes"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable backend progress logging"),
    energy: bool = typer.Option(False, "--energy", help="Compute and print per-iteration energy breakdown"),
    cost: bool = typer.Option(False, "--cost", help="Compute and print cost breakdown (implies --energy)"),
    ignore_oom: bool = typer.Option(False, "--ignore-oom", help="Suppress errors for configs matching OOM profile entries"),
    ignore_missing: bool = typer.Option(False, "--ignore-missing", help="Suppress errors for kernels with no profiling data (treat as 0 duration)"),
):
    """Run simulation and print an iteration summary.

    Optionally export a Chrome/Perfetto trace (--chrome) and/or a
    timing-populated DAG JSON (--dag) for offline analysis.
    """
    import json
    import tempfile
    from simulon.backend.analytical import AnalyticalBackend
    from simulon.backend.dag.chrome_trace import to_chrome_trace
    from simulon.config.scenario import ScenarioConfig
    from simulon.config.workload import MegatronDeprecatedWorkload, MegatronWorkload
    from simulon.tracking import get_trackers
    from simulon.tracking.params import extract_metrics, extract_params

    with open(scenario) as f:
        raw = yaml.safe_load(f)
    sc = ScenarioConfig.model_validate(raw)

    if verbose:
        import logging
        logging.basicConfig(format="%(message)s", level=logging.INFO)

    trackers = get_trackers()

    try:
        for tracker in trackers:
            tracker.start_run()

        backend = AnalyticalBackend()
        dag, result = backend.simulate(sc, compact=compact, ignore_oom=ignore_oom, ignore_missing=ignore_missing)

        if trackers:
            params = extract_params(sc)
            metrics = extract_metrics(result)
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
            from simulon.config.workload import CollectiveWorkload, MegatronDeprecatedWorkload as _MW
            if isinstance(sc.workload, CollectiveWorkload):
                _print_collective_summary(sc.workload, result, sc.datacenter)
            elif isinstance(sc.workload, _MW):
                _print_summary(result, sc.workload)
            else:
                _print_summary(result)

        if isinstance(sc.workload, MegatronDeprecatedWorkload):
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
                p = sc.workload.parallelism
                t = sc.workload.training
                tp = p.tp
                pp_val = p.pp
                ep = p.ep
                dp = p.dp if p.dp is not None else t.num_gpus // (tp * pp_val * ep)
                trace_dict = to_chrome_trace(dag, tp=tp, pp=pp_val, dp=dp, ep=ep)
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


def _print_summary(result, workload=None) -> None:
    """Print a human-readable simulation summary to stdout.

    Args:
        result: SimulationResult from AnalyticalBackend.simulate().
        workload: Optional MegatronDeprecatedWorkload for throughput metrics.
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


def _print_collective_summary(workload, result, datacenter) -> None:
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

    # Determine spec and profile paths.
    if output is None:
        safe_name = name.lower().replace(" ", "-")
        spec_path = Path("templates/gpu") / f"{safe_name}.yaml"
    else:
        spec_path = output

    profile_path = spec_path.with_suffix('').with_suffix('.profile.yaml')
    spec_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing profile data (kernel_runs / oom_kernel_runs).
    if profile_path.exists():
        with open(profile_path) as f:
            profile_data: dict = yaml.safe_load(f) or {}
        existing_runs: list[dict] = profile_data.get("kernel_runs", [])
        existing_oom_kr: list[dict] = profile_data.get("oom_kernel_runs", [])
    else:
        profile_data = {}
        existing_runs = []
        existing_oom_kr = []

    # Create hardware spec file if it doesn't exist yet (skip for --dry-run).
    _hw_args_provided = any(v is not None for v in [vendor, memory_capacity_gb, tdp_w]) or flops_multiplier != 1.0
    if not dry_run and not spec_path.exists():
        spec_dict: dict = {
            "name": name,
            "vendor": vendor,
            "memory_capacity_gb": memory_capacity_gb,
            "flops_multiplier": flops_multiplier,
        }
        spec_dict = {k: v for k, v in spec_dict.items() if v is not None}
        if tdp_w is not None:
            spec_dict["power_model"] = {"type": "constant", "tdp_w": tdp_w}
        with open(spec_path, "w") as f:
            yaml.dump(spec_dict, f, default_flow_style=False, sort_keys=False)
    elif spec_path.exists() and _hw_args_provided:
        typer.echo(
            f"Warning: {spec_path} already exists — hardware fields (--vendor, --memory-capacity-gb, "
            "--tdp-w, --flops-multiplier) were ignored. Edit the spec file directly to update them.",
            err=True,
        )

    if purge:
        existing_runs = []
        existing_oom_kr = []

    # For skip logic: pass existing_runs unless --overwrite (forces re-profiling).
    runs_for_skip = [] if overwrite else existing_runs

    label = model or name

    # Build skip-filter structures (used for both dry-run output and actual sweep).
    # Use canonical filtered params for both sets so that non-canonical fields stored
    # in profiling runs (e.g. swiglu in mlp_act) don't break the matching.
    from simulon.profiling.lookup import _filter_params as _fp
    _sufficient: set[tuple] = set()
    for run in runs_for_skip:
        key = (run["kernel"], frozenset(_fp(run["kernel"], run["params"]).items()))
        if len(run["times_ms"]) >= epoch_num:
            _sufficient.add(key)
    _oom_kr_set: set[tuple] = {
        (r["kernel"], frozenset(r["params"].items())) for r in existing_oom_kr
    }

    dtype_str = dtype_enum.value
    ffn_hidden_size_val = kernel_params["ffn_hidden_size"]
    hidden_size_val = kernel_params["hidden_size"]
    num_heads_val = kernel_params["num_heads"]
    vocab_size_val = kernel_params["vocab_size"]
    num_experts_val = kernel_params.get("num_experts", 0)
    top_k_val = kernel_params.get("top_k", 1)
    swiglu_val = kernel_params.get("swiglu", False)
    num_layers_val = kernel_params.get("num_layers", 0)

    def _adamw_num_params(tp: int, ep: int) -> int:
        """Compute params per TP rank — mirrors kernels.py benchmark_kernels formula."""
        mlp_factor = 3 if swiglu_val else 2
        if num_experts_val > 0:
            mlp_per_layer = mlp_factor * hidden_size_val * ffn_hidden_size_val * (num_experts_val // ep) // tp
        else:
            mlp_per_layer = mlp_factor * hidden_size_val * ffn_hidden_size_val // tp
        attn_per_layer = 4 * hidden_size_val * hidden_size_val // tp
        ln_per_layer = 2 * hidden_size_val
        per_layer = attn_per_layer + mlp_per_layer + ln_per_layer
        embedding = vocab_size_val * hidden_size_val // tp
        logit = vocab_size_val * hidden_size_val // tp
        return num_layers_val * per_layer + embedding + logit

    def _config_done(t: int, e: int, b: int, s: int) -> bool:
        # Note: _sufficient and _oom_kr_set are frozen at sweep start and not updated
        # mid-sweep. Within a single sweep invocation, inferred-OOM skipping is handled
        # by run_sweep's internal _inferred_oom logic, not by _config_done.
        base = {"hidden_size": hidden_size_val, "seq_len": s, "batch_size": b, "dtype": dtype_str, "tp": t}
        # Complete params used for OOM key lookup — must match what _make_oom_kernel_runs uses.
        all_params_for_oom: dict = {
            "hidden_size": hidden_size_val, "num_heads": num_heads_val,
            "ffn_hidden_size": ffn_hidden_size_val, "vocab_size": vocab_size_val,
            "seq_len": s, "batch_size": b, "dtype": dtype_str, "tp": t, "ep": e,
        }
        if num_experts_val > 0:
            all_params_for_oom["num_experts"] = num_experts_val
        if kernel_params.get("top_k"):
            all_params_for_oom["top_k"] = top_k_val
        expected = [
            ("embedding", {}),
            ("layernorm", {}),
            ("attn_qkv", {}),
            ("attn_flash", {"num_heads": num_heads_val}),
            ("attn_proj", {}),
            ("mlp_linear1", {"ffn_hidden_size": ffn_hidden_size_val}),
            ("mlp_act", {"ffn_hidden_size": ffn_hidden_size_val, "swiglu": swiglu_val}),
            ("mlp_linear2", {"ffn_hidden_size": ffn_hidden_size_val}),
            ("logit", {"vocab_size": vocab_size_val}),
        ]
        if num_experts_val > 0:
            expected += [
                ("moe_norm", {}),
                ("moe_route", {"num_experts": num_experts_val}),
                ("moe_expert", {"num_experts": num_experts_val, "ep": e, "top_k": top_k_val, "ffn_hidden_size": ffn_hidden_size_val}),
            ]
        # A config is done when every expected kernel is either sufficiently profiled or known-OOM.
        for kernel, extra in expected:
            sufficient_key = (kernel, frozenset(_fp(kernel, {**base, **extra, "ep": e}).items()))
            oom_key = (kernel, frozenset(_fp(kernel, all_params_for_oom).items()))
            if sufficient_key not in _sufficient and oom_key not in _oom_kr_set:
                return False
        if num_layers_val > 0:
            adamw_key = ("adamw", frozenset({"num_params": _adamw_num_params(t, e), "dtype": dtype_str}.items()))
            if adamw_key not in _sufficient and adamw_key not in _oom_kr_set:
                return False
        return True

    pending = [(t, e, b, s) for t, e, b, s in configs if not _config_done(t, e, b, s)]
    skipped = len(configs) - len(pending)

    if dry_run:
        typer.echo(f"Sweep configurations for GPU '{name}' (arch: {label}):")
        for t, e, b, s in pending:
            typer.echo(f"  tp={t} ep={e} bs={b} seq={s}")
        typer.echo(f"Total: {len(pending)} configurations to run, {skipped} already done (skipped)")
        raise typer.Exit(0)

    # Run sweep with progress display.
    results: list[SweepResult] = []

    try:
        from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
        ) as progress:
            task_id = progress.add_task(f"Profiling {label}", total=len(pending))
            for t, e, b, s in pending:
                progress.update(task_id, description=f"Profiling {label}  tp={t} ep={e} bs={b} seq={s}")
                single = run_sweep(kernel_params, [t], [e], [b], [s], dtype_enum, epoch_num, existing_runs=runs_for_skip)
                # run_sweep always returns exactly one result for a single-config call.
                r = single[0] if single else SweepResult(
                    config={"tp": t, "ep": e, "batch_size": b, "seq_len": s}, runs=None, oom=True
                )
                results.append(r)
                if r.oom:
                    progress.update(task_id, advance=1, description=f"[red]✗ OOM  tp={t} ep={e} bs={b} seq={s}")
                else:
                    progress.update(task_id, advance=1)

    except ImportError:
        for t, e, b, s in pending:
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

    profile_data["kernel_runs"] = existing_runs

    # Merge new per-kernel OOM runs (for simulation-time checking and dry-run skip logic).
    new_oom_kr = [kr for r in results if r.oom for kr in r.oom_runs]
    if new_oom_kr:
        existing_oom_kr_set = {
            (r["kernel"], frozenset(r["params"].items())) for r in existing_oom_kr
        }
        for kr in new_oom_kr:
            key = (kr.kernel, frozenset(kr.params.items()))
            if key not in existing_oom_kr_set:
                existing_oom_kr.append(kr.model_dump())
                existing_oom_kr_set.add(key)
    profile_data["oom_kernel_runs"] = existing_oom_kr

    with open(profile_path, "w") as f:
        _dump_profile(profile_data, f)

    typer.echo(f"Saved {len(all_new_runs)} kernel runs to {profile_path}")
    typer.echo(f"Completed {len(completed)}/{len(results)} configs, {oom_count} skipped (OOM)")

    with open(spec_path) as f:
        merged = yaml.safe_load(f) or {}
    merged.update(profile_data)
    GPUSpec.model_validate(merged)
    typer.echo("Profile validated successfully.")


# ---------------------------------------------------------------------------
# simulon profile node
# ---------------------------------------------------------------------------

_COLLECTIVE_TYPES = ["allreduce", "allgather", "reducescatter", "alltoall"]


def _parse_nccl_json(path: Path) -> list[dict]:
    """Parse a nccl-tests JSON output file into a list of {size_bytes, bus_bw_GBps}."""
    data = json.loads(path.read_text())
    return [
        {"size_bytes": r["size"], "bus_bw_GBps": round(r["out_of_place"]["bus_bw"], 3)}
        for r in data.get("results", [])
        if r.get("out_of_place", {}).get("bus_bw") is not None
    ]


def _gpu_count_from_json(data: dict) -> int | None:
    """Extract GPU count from nccl-tests JSON config block."""
    config = data.get("config", {})
    devices = config.get("devices", [])
    if devices:
        return len(devices)
    ngpus = config.get("ngpus")
    if ngpus:
        return int(ngpus)
    return None


@profile_app.command("node")
def profile_node(
    gpu: str = typer.Option(..., "--gpu", "-g", help="GPU template name (e.g. h100)"),
    input_json: Optional[Path] = typer.Option(
        None,
        "--input-json",
        help="Directory containing pre-run nccl-tests JSON files "
        "(e.g. *allreduce*.json from SLURM runs). Alternative to --nccl-tests-dir.",
    ),
    nccl_tests_dir: Optional[Path] = typer.Option(
        None,
        "--nccl-tests-dir",
        help="Path to nccl-tests build directory. Runs tests live (requires MPI environment).",
    ),
    gpus_per_node: Optional[int] = typer.Option(
        None,
        "--gpus-per-node",
        "-n",
        help="GPU count. Detected from JSON config if not provided.",
    ),
    port_speed: Optional[str] = typer.Option(
        None,
        "--port-speed",
        help="NVSwitch port speed for scale_up (e.g. 2554Gbps). "
        "Detected from input JSON if not provided.",
    ),
    latency: str = typer.Option(
        "0.000025ms",
        "--latency",
        help="NVSwitch wire latency (e.g. 0.000025ms = 25 ns).",
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        help="Node template name. Defaults to <gpu>-<gpus_per_node>g.",
    ),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        "-o",
        help="Output path. Defaults to templates/node/<name>.yaml.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print YAML without writing.",
    ),
):
    """Generate a node template from nccl-tests measurements.

    Two modes:

    \b
    1. --input-json DIR   Parse existing JSON files from SLURM runs.
       Looks for files matching *allreduce*.json, *allgather*.json,
       *reduce_scatter*.json, *alltoall*.json in the given directory.

    \b
    2. --nccl-tests-dir DIR  Run nccl-tests live (requires MPI).
       Runs allreduce_perf, allgather_perf, reduce_scatter_perf, alltoall_perf.

    The output is a templates/node/<name>.yaml carrying GPU reference, scale_up,
    and embedded NCCL measurements. Use this template in datacenter.node to
    skip the network.scale_up + hard-coded per-collective_bw_GBps pattern.
    """
    import json
    from simulon.config.dc import NodeSpec
    from simulon.config.nccl_profile import NcclAlgoMeasurements, NcclDataPoint, NcclProfile

    if input_json is None and nccl_tests_dir is None:
        typer.echo("Error: pass --input-json DIR or --nccl-tests-dir DIR", err=True)
        raise typer.Exit(1)
    if input_json is not None and nccl_tests_dir is not None:
        typer.echo("Error: pass only one of --input-json or --nccl-tests-dir", err=True)
        raise typer.Exit(1)

    measurements: dict[str, list[dict]] = {}

    if input_json is not None:
        if not input_json.is_dir():
            typer.echo(f"Error: {input_json} is not a directory", err=True)
            raise typer.Exit(1)
        found_any = False
        for coll in _COLLECTIVE_TYPES:
            matches = list(input_json.glob(f"*{coll}*.json"))
            if not matches:
                matches = list(input_json.glob(f"*{coll.replace('_', '')}*.json"))
            if len(matches) > 1:
                typer.echo(
                    f"  Warning: multiple JSON files match '{coll}': "
                    f"{[p.name for p in matches]}. Using {matches[0].name}.",
                    err=True,
                )
                matches = matches[:1]
            for path in matches:
                try:
                    data = json.loads(path.read_text())
                    if gpus_per_node is None:
                        gpus_per_node = _gpu_count_from_json(data)
                    key = coll
                    measurements[key] = _parse_nccl_json(path)
                    found_any = True
                    typer.echo(f"  {path.name}: {len(measurements[key])} points")
                except Exception as exc:
                    typer.echo(f"  Warning: {path.name}: {exc}", err=True)
        if not found_any:
            typer.echo(
                f"Error: no nccl-tests JSON files found in {input_json}. "
                "Expected files matching *allreduce*.json, *allgather*.json, etc.",
                err=True,
            )
            raise typer.Exit(1)

    elif nccl_tests_dir is not None:
        typer.echo("Live nccl-tests mode not yet implemented. Use --input-json.", err=True)
        raise typer.Exit(1)

    if gpus_per_node is None:
        typer.echo("Error: could not detect gpus_per_node from JSON. Pass --gpus-per-node.", err=True)
        raise typer.Exit(1)

    detected_name = name or f"{gpu}-{gpus_per_node}g"
    out_path = out or Path("templates/node") / f"{detected_name}.yaml"

    # Build NcclProfile
    np_data: dict = {
        "gpus_per_node": gpus_per_node,
        "name": detected_name,
    }
    _coll_key_map = {
        "allreduce": "AllReduce",
        "allgather": "AllGather",
        "reducescatter": "ReduceScatter",
        "alltoall": "AllToAll",
    }
    for coll, points in measurements.items():
        key = _coll_key_map.get(coll, coll.capitalize())
        np_data[key] = {"ring": [{"size_bytes": p["size_bytes"], "bus_bw_GBps": p["bus_bw_GBps"]} for p in points]}

    nccl_profile = NcclProfile.model_validate(np_data)

    # Build NodeSpec — only include switch fields that were explicitly provided.
    switch_data: dict = {}
    if port_speed is not None:
        switch_data["port_speed"] = port_speed
    switch_data["latency"] = latency  # always include; user controls the value

    node_data: dict = {
        "name": detected_name,
        "from": gpu,
        "gpus_per_node": gpus_per_node,
        "scale_up": {"switch": switch_data},
        "nccl": nccl_profile.model_dump(),
    }

    node_spec = NodeSpec.model_validate(node_data)
    output_data = node_spec.model_dump(by_alias=True, exclude_unset=True)

    yaml_out = yaml.dump(output_data, default_flow_style=False, sort_keys=False)
    if dry_run:
        typer.echo(yaml_out)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(yaml_out)
        typer.echo(f"Written to {out_path}")
        NodeSpec.model_validate(yaml.safe_load(out_path.read_text()))
        typer.echo("Validated successfully.")


# ---------------------------------------------------------------------------
# simulon trace generate
# ---------------------------------------------------------------------------

_MEGATRON_ENTRYPOINT = Path(__file__).parents[3] / "vendor" / "Megatron-LM-traced" / "pretrain_gpt.py"


@trace_app.command("generate")
def generate_trace(
    scenario: str = typer.Argument(..., help="Path to scenario.yaml"),
    output_dir: Path = typer.Option(Path("./traces"), "--output-dir", "-o", help="Directory to write trace files"),
    stages: Optional[list[int]] = typer.Option(None, "--stage", help="Specific PP stages to trace (default: first, middle, last)"),
):
    """Generate per-PP-stage execution traces by running Megatron-LM with fake process groups."""
    from simulon.config.scenario import ScenarioConfig
    from simulon.config.workload import LLMSpec, MegatronDeprecatedWorkload, MegatronWorkload

    with open(scenario) as f:
        raw = yaml.safe_load(f)
    sc = ScenarioConfig.model_validate(raw)

    if not isinstance(sc.workload, (MegatronWorkload, MegatronDeprecatedWorkload)):
        raise typer.BadParameter("Scenario workload must be a Megatron workload")

    derived_args: dict[str, str | int | bool]

    if isinstance(sc.workload, MegatronDeprecatedWorkload):
        workload = sc.workload
        p = workload.parallelism
        t = workload.training

        derived_args = {
            "--tensor-model-parallel-size": p.tp,
            "--pipeline-model-parallel-size": p.pp,
            "--micro-batch-size": t.micro_batch_size,
            "--global-batch-size": t.global_batch_size,
            "--seq-length": t.sequence_length,
        }
        if p.ep > 1:
            derived_args["--expert-model-parallel-size"] = p.ep

        model = workload.model
        if isinstance(model, LLMSpec):
            if model.num_layers is not None:
                derived_args["--num-layers"] = model.num_layers
            if model.hidden_size is not None:
                derived_args["--hidden-size"] = model.hidden_size
            if model.num_heads is not None:
                derived_args["--num-attention-heads"] = model.num_heads
            if model.ffn_hidden_size is not None:
                derived_args["--ffn-hidden-size"] = model.ffn_hidden_size

        if workload.megatron_args:
            for key, value in workload.megatron_args.items():
                flag = f"--{key}"
                if isinstance(value, bool):
                    if value:
                        derived_args[flag] = True
                    elif flag in derived_args:
                        del derived_args[flag]
                else:
                    derived_args[flag] = value

        pp = p.pp
        tp = p.tp
        world_size = t.num_gpus
    else:
        workload = sc.workload
        cfg = workload.config
        derived_args = {}
        for key in ("tensor-model-parallel-size", "pipeline-model-parallel-size",
                    "micro-batch-size", "global-batch-size", "seq-length",
                    "expert-model-parallel-size", "num-layers", "hidden-size",
                    "num-attention-heads", "ffn-hidden-size"):
            if key in cfg:
                derived_args[f"--{key}"] = cfg[key]
        skip = {"num_gpus", "num-gpus", "num_microbatches", "num-microbatches"}
        for key, value in cfg.items():
            if key == "distributed-optimizer":
                flag = "--use-distributed-optimizer"
            else:
                flag = f"--{key}"
            if flag not in derived_args and key not in skip:
                if isinstance(value, bool):
                    if value:
                        derived_args[flag] = True
                else:
                    derived_args[flag] = value

        pp = int(cfg.get("pipeline-model-parallel-size", 1))
        tp = int(cfg.get("tensor-model-parallel-size", 1))
        world_size = cfg.get("num_gpus", cfg.get("num-gpus"))
    if stages is not None:
        stages_to_trace = stages
    else:
        if pp == 1:
            stages_to_trace = [0]
        elif pp == 2:
            stages_to_trace = [0, pp - 1]
        else:
            stages_to_trace = [0, pp // 2, pp - 1]

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for stage in stages_to_trace:
        rank = stage * tp
        cmd: list[str] = [sys.executable, str(_MEGATRON_ENTRYPOINT)]
        for flag, value in derived_args.items():
            cmd.append(flag)
            if value is not True:
                cmd.append(str(value))
        cmd.extend([
            "--fake-process-group",
            "--rank", str(rank),
            "--trace-dir", str(output_dir),
        ])

        typer.echo(f"Tracing PP stage {stage} (rank {rank}) ...")
        env = os.environ.copy()
        if world_size is not None:
            env["WORLD_SIZE"] = str(world_size)
        try:
            subprocess.run(cmd, check=True, env=env)
        except FileNotFoundError as exc:
            typer.echo(f"Error: could not run Megatron entry point: {exc}", err=True)
            raise typer.Exit(1)
        except subprocess.CalledProcessError as exc:
            typer.echo(f"Error: Megatron exited with code {exc.returncode} for PP stage {stage}", err=True)
            raise typer.Exit(1)

    trace_files = sorted(output_dir.glob("trace_pp_stage_*.json"))
    typer.echo(f"\nTrace generation complete. {len(trace_files)} file(s) in {output_dir}:")
    for tf in trace_files:
        typer.echo(f"  {tf.name}")
