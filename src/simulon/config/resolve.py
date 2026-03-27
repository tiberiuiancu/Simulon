"""Utilities for resolving hardware specs that may reference named templates."""
from __future__ import annotations

from pathlib import Path

import yaml

from simulon.config.dc import DatacenterConfig, GPUSpec


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


def resolve_gpu_spec(dc: DatacenterConfig) -> GPUSpec:
    """Return the effective GPUSpec for a datacenter config.

    Handles three forms:
    - Short string reference (``gpu: H100``) — loads the named template.
    - ``from:`` with field overrides — loads the named template, then applies all
      explicitly-set override fields (power_model, cost, name, etc.).
    - Fully inline spec — returned as-is.
    """
    gpu = dc.node.gpu
    if isinstance(gpu, str):
        return load_gpu_template(gpu)
    if isinstance(gpu, GPUSpec) and gpu.from_:
        base = load_gpu_template(gpu.from_)
        # Apply every field that was explicitly provided in the override
        # (exclude_unset=True skips fields that were left at their defaults).
        overrides = gpu.model_dump(exclude_unset=True, by_alias=False)
        overrides.pop("from_", None)  # don't carry the template reference onto base
        for key, val in overrides.items():
            setattr(base, key, val)
        return base
    return gpu
