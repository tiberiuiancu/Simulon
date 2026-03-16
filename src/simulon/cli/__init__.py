from pathlib import Path
from typing import Optional

import typer
import yaml

app = typer.Typer(name="simulon", help="AI cluster simulator")
profile_app = typer.Typer(help="Profile local hardware and save templates.")
app.add_typer(profile_app, name="profile")


@app.command()
def simulate(
    backend: str = typer.Argument(..., help="Simulation backend: analytical | ns3"),
    scenario: str = typer.Argument(..., help="Path to scenario.yaml"),
):
    """Run a simulation scenario."""
    raise NotImplementedError


@app.command()
def trace(
    scenario: str = typer.Argument(..., help="Path to scenario.yaml"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path for dag.json"),
    num_channels: int = typer.Option(1, "--num-channels", help="Number of ring channels"),
    algorithm: str = typer.Option("ring", "--algorithm", help="Collective algorithm: ring | nvls"),
):
    """Extract a GPU-agnostic execution DAG from a scenario."""
    import json
    from simulon.backend.astra_sim import AstraSimBackend
    from simulon.config.scenario import ScenarioConfig

    with open(scenario) as f:
        raw = yaml.safe_load(f)
    sc = ScenarioConfig.model_validate(raw)

    backend = AstraSimBackend(num_channels=num_channels, algorithm=algorithm)
    dag = backend.run_trace(sc)

    if output is None:
        output = Path("dag.json")

    with open(output, "w") as f:
        f.write(dag.to_json())

    typer.echo(f"DAG written to {output}")
    typer.echo(f"  compute_nodes: {len(dag.compute_nodes)}")
    typer.echo(f"  comm_nodes:    {len(dag.comm_nodes)}")
    typer.echo(f"  edges:         {len(dag.edges)}")


@profile_app.command("gpu")
def profile_gpu(
    name: str = typer.Option(..., "--name", "-n", help="GPU model name (e.g. H100-SXM5-80GB)"),
    vendor: Optional[str] = typer.Option(None, help="GPU vendor: nvidia | amd"),
    memory_capacity_gb: Optional[float] = typer.Option(None, help="HBM capacity in GB"),
    tdp_w: Optional[float] = typer.Option(None, help="TDP in watts"),
    flops_multiplier: float = typer.Option(1.0, help="Scalar multiplier applied to all profiled FLOP rates"),
    hidden_size: int = typer.Option(..., help="Transformer hidden dimension"),
    num_heads: int = typer.Option(..., help="Number of attention heads"),
    ffn_hidden_size: int = typer.Option(..., help="FFN intermediate dimension"),
    seq_len: int = typer.Option(..., help="Sequence length"),
    batch_size: int = typer.Option(1, help="Micro-batch size"),
    vocab_size: int = typer.Option(..., help="Vocabulary size"),
    dtype: str = typer.Option("bf16", help="Compute dtype: fp32 | fp16 | bf16 | fp8"),
    tp: int = typer.Option(1, help="Tensor Parallelism degree"),
    epoch_num: int = typer.Option(10, help="Number of timed iterations per kernel"),
    swiglu: bool = typer.Option(False, help="Use SwiGLU activation shape for mlp_act"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output path for the template YAML. Defaults to templates/gpu/<name>.yaml",
    ),
):
    """Profile the local GPU and save a hardware template.

    Appends new kernel_runs to an existing YAML file (or creates it).
    Run multiple times with different configs to build a rich profile.
    """
    from simulon.config.common import DType
    from simulon.config.dc import GPUSpec, KernelRun
    from simulon.profiling.kernels import benchmark_kernels

    dtype_enum = DType(dtype)

    typer.echo(f"Benchmarking kernels on GPU: {name}")
    typer.echo(
        f"  hidden={hidden_size}, heads={num_heads}, ffn={ffn_hidden_size}, "
        f"seq={seq_len}, batch={batch_size}, vocab={vocab_size}, "
        f"dtype={dtype}, tp={tp}, epochs={epoch_num}"
    )

    kernel_runs: list[KernelRun] = benchmark_kernels(
        hidden_size=hidden_size,
        num_heads=num_heads,
        ffn_hidden_size=ffn_hidden_size,
        seq_len=seq_len,
        batch_size=batch_size,
        vocab_size=vocab_size,
        tp=tp,
        dtype=dtype_enum,
        epoch_num=epoch_num,
        swiglu=swiglu,
    )

    # Resolve output path
    if output is None:
        safe_name = name.lower().replace(" ", "-")
        output = Path("templates/gpu") / f"{safe_name}.yaml"

    output.parent.mkdir(parents=True, exist_ok=True)

    # Load existing YAML or start fresh
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
        # Remove None values for cleaner YAML
        existing = {k: v for k, v in existing.items() if v is not None}
        existing_runs = []

    # Append new runs
    new_runs = [kr.model_dump() for kr in kernel_runs]
    existing_runs.extend(new_runs)
    existing["kernel_runs"] = existing_runs

    with open(output, "w") as f:
        yaml.dump(existing, f, default_flow_style=False, sort_keys=False)

    typer.echo(f"Saved {len(kernel_runs)} kernel runs to {output}")

    # Validate by round-tripping through GPUSpec
    GPUSpec.model_validate(existing)
    typer.echo("Profile validated successfully.")
