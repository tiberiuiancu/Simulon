from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Annotated

import typer


app = typer.Typer(no_args_is_help=True)

# Cache locations for third-party components
_APEX_GIT_URL = "https://github.com/NVIDIA/apex.git"
_APEX_CACHE_DIR = Path.home() / ".cache" / "simulon" / "apex"

_DEEPGEMM_GIT_URL = "https://github.com/deepseek-ai/DeepGEMM.git"
_DEEPGEMM_CACHE_DIR = Path.home() / ".cache" / "simulon" / "DeepGEMM"


def _clone_repo(git_url: str, dest: Path) -> Path:
    """Clone a git repo to the given destination, removing it first if it exists."""
    typer.echo(f"Cloning {git_url} into {dest} ...")
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "git", "clone", "--recurse-submodules", git_url, str(dest)
    ], check=True)
    return dest


@app.command()
def apex(
    force: Annotated[
        bool,
        typer.Option("--force", help="Reinstall even if already installed."),
    ] = False,
    git_url: Annotated[
        str,
        typer.Option("--git-url", help="Git URL to clone apex from if not found."),
    ] = _APEX_GIT_URL,
    src: Annotated[
        Path | None,
        typer.Option("--src", help="Path to apex source directory.", exists=True, file_okay=False),
    ] = None,
    skip_cuda_version_check: Annotated[
        bool,
        typer.Option(
            "--skip-cuda-version-check",
            help="Patch apex setup.py to skip CUDA version check (use if you get a CUDA version mismatch error)",
        ),
    ] = False,
) -> None:
    """Install NVIDIA Apex (CUDA extensions for PyTorch).

    If you see a RuntimeError about CUDA version mismatch, you can use --skip-cuda-version-check to patch setup.py and skip the check (at your own risk).
    """
    apex_src = src or (_APEX_CACHE_DIR if _APEX_CACHE_DIR.is_dir() else None)
    if not apex_src or force:
        apex_src = _clone_repo(git_url, _APEX_CACHE_DIR)

    setup_py = apex_src / "setup.py"
    if skip_cuda_version_check and setup_py.is_file():
        text = setup_py.read_text()
        replaced = text.replace(
            "check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)",
            "#check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)"
        )
        if text == replaced:
            typer.echo("Warning: Could not find CUDA version check call in setup.py to patch.")
        else:
            setup_py.write_text(replaced)
            typer.echo("Patched apex setup.py to comment out CUDA version check call.")
            typer.echo("  (You are bypassing a safety check. See https://github.com/NVIDIA/apex/pull/323#discussion_r287021798)")

    install_cmd = [
        "bash", "-c",
        'NVCC_APPEND_FLAGS="--threads $(nproc)" '
        'APEX_PARALLEL_BUILD=$(nproc) '
        'APEX_CPP_EXT=1 APEX_CUDA_EXT=1 APEX_FAST_LAYER_NORM=1 '
        'pip install -v --no-build-isolation .'
    ]
    typer.echo(f"Installing apex from {apex_src} ...")
    subprocess.run(install_cmd, cwd=str(apex_src), check=True)
    typer.echo("Apex installed successfully.")


@app.command()
def deepgemm(
    force: Annotated[
        bool,
        typer.Option("--force", help="Reinstall even if already installed."),
    ] = False,
    git_url: Annotated[
        str,
        typer.Option("--git-url", help="Git URL to clone DeepGEMM from if not found."),
    ] = _DEEPGEMM_GIT_URL,
    src: Annotated[
        Path | None,
        typer.Option("--src", help="Path to DeepGEMM source directory.", exists=True, file_okay=False),
    ] = None,
) -> None:
    """Install DeepGEMM (CUDA kernel for DeepSeek)."""
    deepgemm_src = src or (_DEEPGEMM_CACHE_DIR if _DEEPGEMM_CACHE_DIR.is_dir() else None)
    if not deepgemm_src or force:
        deepgemm_src = _clone_repo(git_url, _DEEPGEMM_CACHE_DIR)
    install_sh = deepgemm_src / "install.sh"
    if not install_sh.is_file():
        typer.echo(f"Error: install.sh not found in {deepgemm_src}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Installing DeepGEMM from {deepgemm_src} ...")
    subprocess.run(["bash", str(install_sh)], cwd=str(deepgemm_src), check=True)
    typer.echo("DeepGEMM installed successfully.")
