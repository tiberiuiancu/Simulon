from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True)

# Cache locations for third-party components
_APEX_GIT_URL = "https://github.com/NVIDIA/apex.git"
_APEX_CACHE_DIR = Path.home() / ".cache" / "simulon" / "apex"

_DEEPGEMM_GIT_URL = "https://github.com/deepseek-ai/DeepGEMM.git"
_DEEPGEMM_CACHE_DIR = Path.home() / ".cache" / "simulon" / "DeepGEMM"

_FLASH_ATTN_GIT_URL = "git@github.com:Dao-AILab/flash-attention.git"
_FLASH_ATTN_CACHE_DIR = Path.home() / ".cache" / "simulon" / "flash-attention"


def _clone_repo(git_url: str, dest: Path) -> Path:
    """Clone a git repo to the given destination, removing it first if it exists."""
    typer.echo(f"Cloning {git_url} into {dest} ...")
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--recurse-submodules", git_url, str(dest)], check=True)
    return dest


@app.command()
def apex(
    force: Annotated[
        bool, typer.Option("--force", help="Reinstall even if already installed.")
    ] = False,
    git_url: Annotated[
        str, typer.Option("--git-url", help="Git URL to clone apex from if not found.")
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
            "#check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)",
        )
        if text == replaced:
            typer.echo("Warning: Could not find CUDA version check call in setup.py to patch.")
        else:
            setup_py.write_text(replaced)
            typer.echo("Patched apex setup.py to comment out CUDA version check call.")
            typer.echo(
                "  (You are bypassing a safety check. See https://github.com/NVIDIA/apex/pull/323#discussion_r287021798)"
            )

    install_cmd = [
        "bash",
        "-c",
        'NVCC_APPEND_FLAGS="--threads $(nproc)" '
        "APEX_PARALLEL_BUILD=$(nproc) "
        "APEX_CPP_EXT=1 APEX_CUDA_EXT=1 APEX_FAST_LAYER_NORM=1 "
        f'"{sys.executable}" setup.py install --cpp_ext --cuda_ext',
    ]
    typer.echo(f"Installing apex from {apex_src} ...")
    subprocess.run(install_cmd, cwd=str(apex_src), check=True)
    typer.echo("Apex installed successfully.")


@app.command()
def deepgemm(
    force: Annotated[
        bool, typer.Option("--force", help="Reinstall even if already installed.")
    ] = False,
    git_url: Annotated[
        str, typer.Option("--git-url", help="Git URL to clone DeepGEMM from if not found.")
    ] = _DEEPGEMM_GIT_URL,
    src: Annotated[
        Path | None,
        typer.Option(
            "--src", help="Path to DeepGEMM source directory.", exists=True, file_okay=False
        ),
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


@app.command()
def flash_attn_hopper(
    force: Annotated[
        bool, typer.Option("--force", help="Reinstall even if already installed.")
    ] = False,
    prebuilt: Annotated[
        bool,
        typer.Option(
            "--prebuilt", help="Install from prebuilt wheel instead of building from source."
        ),
    ] = False,
    version: Annotated[
        str | None,
        typer.Option("--version", help="Exact flash-attn version to install (e.g. 2.7.3)."),
    ] = None,
    git_url: Annotated[
        str, typer.Option("--git-url", help="Git URL to clone flash-attention from.")
    ] = _FLASH_ATTN_GIT_URL,
    src: Annotated[
        Path | None,
        typer.Option(
            "--src", help="Path to flash-attention source directory.", exists=True, file_okay=False
        ),
    ] = None,
) -> None:
    """Install Flash Attention 3 (Hopper-optimized) for H100 GPUs.

    By default clones Dao-AILab/flash-attention and builds the hopper/
    subdirectory. Use --prebuilt to install from a prebuilt wheel instead.
    """
    if prebuilt:
        _install_flash_attn_prebuilt(version=version)
        return

    flash_src = src or (_FLASH_ATTN_CACHE_DIR if _FLASH_ATTN_CACHE_DIR.is_dir() else None)
    if not flash_src or force:
        flash_src = _clone_repo(git_url, _FLASH_ATTN_CACHE_DIR)

    if version:
        typer.echo(f"Checking out flash-attn version {version} ...")
        subprocess.run(["git", "-C", str(flash_src), "checkout", f"v{version}"], check=True)

    hopper_dir = flash_src / "hopper"
    if not hopper_dir.is_dir():
        typer.echo(f"Error: {hopper_dir} not found. Is this the flash-attention repo?", err=True)
        raise typer.Exit(1)

    setup_py = hopper_dir / "setup.py"
    if not setup_py.is_file():
        typer.echo(f"Error: {setup_py} not found.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Installing Flash Attention 3 (Hopper) from {hopper_dir} ...")
    typer.echo(f"Using Python interpreter: {sys.executable}")
    subprocess.run([sys.executable, str(setup_py), "install"], cwd=str(hopper_dir), check=True)
    typer.echo("Flash Attention 3 (Hopper) installed successfully.")


def _install_flash_attn_prebuilt(version: str | None = None) -> None:
    """Install Flash Attention from prebuilt wheels (mjun0812/flash-attention-prebuild-wheels)."""
    import json
    import urllib.request

    import torch

    py_major, py_minor = sys.version_info[:2]
    py_tag = f"cp{py_major}{py_minor}"

    torch_version = torch.__version__.split("+")[0]
    torch_major_minor = ".".join(torch_version.split(".")[:2])

    cuda_version = torch.version.cuda
    if cuda_version is None:
        typer.echo("Error: PyTorch is not built with CUDA support.", err=True)
        raise typer.Exit(1)
    cuda_major_minor = cuda_version.replace(".", "")[:3]

    if version:
        fa3_wheel_prefix = f"flash_attn-{version}+cu{cuda_major_minor}torch{torch_major_minor}-{py_tag}-{py_tag}-linux_x86_64.whl"
    else:
        fa3_wheel_prefix = (
            f"flash_attn-3."
            f"+cu{cuda_major_minor}torch{torch_major_minor}-"
            f"{py_tag}-{py_tag}-linux_x86_64.whl"
        )

    typer.echo("Searching for prebuilt wheel across all releases ...")
    typer.echo(
        f"  Detected: Python {py_major}.{py_minor}, PyTorch {torch_version}, CUDA {cuda_version}"
    )

    page = 1
    per_page = 30
    fa3_match = None
    fa2_match = None
    while True:
        api_url = (
            f"https://api.github.com/repos/mjun0812/flash-attention-prebuild-wheels/"
            f"releases?per_page={per_page}&page={page}"
        )
        try:
            req = urllib.request.Request(api_url, headers={"Accept": "application/vnd.github+json"})
            with urllib.request.urlopen(req, timeout=15) as response:
                releases = json.loads(response.read().decode())
        except Exception as exc:
            typer.echo(f"Error: Failed to fetch releases from GitHub: {exc}", err=True)
            raise typer.Exit(1) from exc

        if not releases:
            break

        for release in releases:
            tag = release.get("tag_name", "")
            assets = release.get("assets", [])
            asset_names = [a.get("name", "") for a in assets]
            for name in asset_names:
                if version:
                    if name == fa3_wheel_prefix and fa3_match is None:
                        fa3_match = (tag, name)
                else:
                    if name.startswith("flash_attn-3.") and name.endswith(
                        f"+cu{cuda_major_minor}torch{torch_major_minor}-{py_tag}-{py_tag}-linux_x86_64.whl"
                    ):
                        if fa3_match is None:
                            fa3_match = (tag, name)
                    elif (
                        name.startswith("flash_attn-2.")
                        and name.endswith(
                            f"+cu{cuda_major_minor}torch{torch_major_minor}-{py_tag}-{py_tag}-linux_x86_64.whl"
                        )
                        and fa2_match is None
                    ):
                        fa2_match = (tag, name)

            if fa3_match and fa2_match:
                break
        if fa3_match and fa2_match:
            break
        if len(releases) < per_page:
            break
        page += 1

    tag, wheel_name = None, None
    if fa3_match:
        tag, wheel_name = fa3_match
        typer.echo(f"  Found Flash Attention 3 wheel: {wheel_name} in {tag}")
    elif fa2_match:
        tag, wheel_name = fa2_match
        typer.echo(f"  Found Flash Attention 2 wheel: {wheel_name} in {tag}")

    if tag and wheel_name:
        wheel_url = (
            f"https://github.com/mjun0812/flash-attention-prebuild-wheels/"
            f"releases/download/{tag}/{wheel_name}"
        )
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", wheel_url], check=True)
            typer.echo("Flash Attention installed successfully from prebuilt wheel.")
            return
        except subprocess.CalledProcessError as exc:
            typer.echo(f"Error: pip install failed for {wheel_url}: {exc}", err=True)
            raise typer.Exit(1) from exc

    typer.echo("Error: Could not find a prebuilt wheel for your environment.", err=True)
    typer.echo(
        f"  Searched across all releases for Python {py_major}.{py_minor}, PyTorch {torch_major_minor}, CUDA {cuda_major_minor}.",
        err=True,
    )
    typer.echo("Try installing from source instead (omit --prebuilt).", err=True)
    raise typer.Exit(1)
