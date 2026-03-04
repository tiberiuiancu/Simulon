from pathlib import Path
from typing import Optional

import typer

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


@profile_app.command("gpu")
def profile_gpu(
    name: str = typer.Option(..., "--name", "-n", help="GPU model name (e.g. H100-SXM5-80GB)"),
    vendor: Optional[str] = typer.Option(None, help="GPU vendor: nvidia | amd"),
    memory_capacity_gb: Optional[float] = typer.Option(None, help="HBM capacity in GB"),
    tdp_w: Optional[float] = typer.Option(None, help="TDP in watts"),
    flops_multiplier: float = typer.Option(1.0, help="Scalar multiplier applied to all profiled FLOP rates"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output path for the template YAML. Defaults to templates/gpu/<name>.yaml",
    ),
):
    """Profile the local GPU and save a hardware template."""
    raise NotImplementedError
