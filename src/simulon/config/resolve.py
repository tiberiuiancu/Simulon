"""Utilities for resolving hardware specs that may reference named templates."""
from __future__ import annotations

import warnings
from pathlib import Path

import yaml

from simulon.config.dc import (
    DatacenterConfig,
    GPUSpec,
    NodeSpec,
    ScaleOutSpec,
)
from simulon.config.nccl_profile import NcclProfile


def _load_profile_data(template_path: Path) -> dict:
    profile_path = template_path.with_suffix("").with_suffix(".profile.yaml")
    if profile_path.exists():
        with open(profile_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def load_gpu_template(name: str) -> GPUSpec:
    """Load a GPU spec from a named YAML template file.

    Searches templates/gpu/<name>.yaml (case-insensitive fallback).
    Merges kernel_runs / oom_configs from the companion .profile.yaml if present.
    """
    template_path = Path("templates/gpu") / f"{name}.yaml"
    if not template_path.exists():
        candidates = (
            list(Path("templates/gpu").glob("*.yaml"))
            if Path("templates/gpu").exists()
            else []
        )
        for c in candidates:
            if c.stem.lower() == name.lower():
                template_path = c
                break
        else:
            raise FileNotFoundError(
                f"GPU template not found: {name!r}. Expected at templates/gpu/{name}.yaml"
            )
    with open(template_path) as f:
        data = yaml.safe_load(f)
    data.update(_load_profile_data(template_path))
    return GPUSpec.model_validate(data)


def load_nccl_profile(name: str) -> NcclProfile | None:
    """Load an NCCL measurement profile from <name>.nccl.yaml alongside the GPU template.

    Returns None if no companion .nccl.yaml file exists.
    """
    template_path = Path("templates/gpu") / f"{name}.yaml"
    # Case-insensitive fallback (same as load_gpu_template)
    if not template_path.exists() and Path("templates/gpu").exists():
        for c in Path("templates/gpu").glob("*.yaml"):
            if c.stem.lower() == name.lower():
                template_path = c
                break
    nccl_path = template_path.with_suffix("").with_suffix(".nccl.yaml")
    if not nccl_path.exists():
        return None
    with open(nccl_path) as f:
        return NcclProfile.model_validate(yaml.safe_load(f))


def load_node_template(name: str) -> NodeSpec:
    """Load a node spec from a named YAML template file.

    Searches templates/node/<name>.yaml (case-insensitive fallback).
    """
    template_path = Path("templates/node") / f"{name}.yaml"
    if not template_path.exists():
        candidates = (
            list(Path("templates/node").glob("*.yaml"))
            if Path("templates/node").exists()
            else []
        )
        for c in candidates:
            if c.stem.lower() == name.lower():
                template_path = c
                break
        else:
            raise FileNotFoundError(
                f"Node template not found: {name!r}. Expected at templates/node/{name}.yaml"
            )
    with open(template_path) as f:
        data = yaml.safe_load(f)
    return NodeSpec.model_validate(data)


def resolve_node_spec(dc: DatacenterConfig) -> NodeSpec:
    """Return the effective NodeSpec for a datacenter config.

    Handles three forms:
    - Bare string (``node: "dgx-h100"``) — coerced to NodeSpec(from_=name) by Pydantic.
    - ``from:`` with field overrides — loads the named template, then applies overrides.
    - Fully inline spec — returned as-is.
    """
    node = dc.node
    if node.from_:
        base = load_node_template(node.from_)
        base_dict = base.model_dump(by_alias=False)
        overrides = node.model_dump(exclude_unset=True, by_alias=False)
        overrides.pop("from_", None)
        base_dict.update(overrides)
        return NodeSpec.model_validate(base_dict)
    return node


def resolve_gpu_spec(dc: DatacenterConfig) -> GPUSpec:
    """Return the effective GPUSpec for a datacenter config.

    Handles three forms:
    - Short string reference (``gpu: H100``) — loads the named template.
    - ``from:`` with field overrides — loads the named template, then applies all
      explicitly-set override fields (power_model, cost, name, etc.).
    - Fully inline spec — returned as-is.
    """
    node = resolve_node_spec(dc)
    gpu = node.gpu
    if isinstance(gpu, str):
        return load_gpu_template(gpu)
    if isinstance(gpu, GPUSpec) and gpu.from_:
        base = load_gpu_template(gpu.from_)
        # Merge override fields into the base dict and re-validate so that nested
        # objects (cost, power_model, …) are properly coerced by Pydantic, not stored
        # as raw dicts from model_dump().
        base_dict = base.model_dump(by_alias=False)
        overrides = gpu.model_dump(exclude_unset=True, by_alias=False)
        overrides.pop("from_", None)
        base_dict.update(overrides)
        return GPUSpec.model_validate(base_dict)
    return gpu


def resolve_nccl_profile(dc: DatacenterConfig) -> NcclProfile | None:
    """Return the NCCL profile for a datacenter config.

    Priority:
    1. Embedded node.nccl (node template carries its own profile)
    2. load_nccl_profile(gpu_name) fallback (companion .nccl.yaml next to GPU template)
    """
    node = resolve_node_spec(dc)
    if node.nccl is not None:
        return node.nccl
    gpu_spec = resolve_gpu_spec(dc)
    gpu_name = gpu_spec.name if gpu_spec.name else None
    return load_nccl_profile(gpu_name) if gpu_name else None


def resolve_scale_out(dc: DatacenterConfig) -> ScaleOutSpec | None:
    """Return the effective scale-out spec for a datacenter config.

    Returns dc.scale_out if set, otherwise falls back to dc.network.scale_out
    with a DeprecationWarning.
    """
    if dc.scale_out is not None:
        return dc.scale_out
    if dc.network is not None and dc.network.scale_out is not None:
        warnings.warn(
            "datacenter.network.scale_out is deprecated. "
            "Move scale_out to the top-level datacenter.scale_out field.",
            DeprecationWarning,
            stacklevel=2,
        )
        return dc.network.scale_out
    return None
