"""Utilities for resolving hardware specs that may reference named templates."""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import yaml


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* into *base*, returning a new dict.

    Nested dicts are merged rather than replaced, so a partial sub-object
    override (e.g. only changing one field inside scale_up.switch) does not
    wipe sibling fields that were not explicitly overridden.
    """
    result = dict(base)
    for key, val in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result

from simulon.config.dc import (
    DatacenterConfig,
    GPUSpec,
    NodeSpec,
    ScaleOutSpec,
)
from simulon.config.nccl_profile import NcclProfile
from simulon.config.workload import MegatronWorkload


def _load_profile_data(template_path: Path) -> dict:
    profile_path = template_path.with_suffix("").with_suffix(".profile.yaml")
    if profile_path.exists():
        with open(profile_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def load_gpu_template(name: str, include_profile: bool = True) -> GPUSpec:
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
    if include_profile:
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
        return NodeSpec.model_validate(_deep_merge(base_dict, overrides))
    return node


def resolve_gpu_spec(dc: DatacenterConfig, include_profile: bool = True) -> GPUSpec:
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
        return load_gpu_template(gpu, include_profile=include_profile)
    if isinstance(gpu, GPUSpec) and gpu.from_:
        base = load_gpu_template(gpu.from_, include_profile=include_profile)
        # Merge override fields into the base dict and re-validate so that nested
        # objects (cost, power_model, …) are properly coerced by Pydantic, not stored
        # as raw dicts from model_dump().
        base_dict = base.model_dump(by_alias=False)
        overrides = gpu.model_dump(exclude_unset=True, by_alias=False)
        overrides.pop("from_", None)
        return GPUSpec.model_validate(_deep_merge(base_dict, overrides))
    return gpu


def resolve_nccl_profile(dc: DatacenterConfig) -> NcclProfile | None:
    """Return the NCCL profile for a datacenter config.

    Priority:
    1. Embedded node.nccl (node template carries its own profile)
    2. load_nccl_profile(gpu_name) fallback (companion .nccl.yaml next to GPU template)

    Uses the already-resolved node to avoid loading the node template twice.
    """
    node = resolve_node_spec(dc)
    if node.nccl is not None:
        return node.nccl
    # Resolve GPU from the already-resolved node to avoid a second template load.
    gpu = node.gpu
    if isinstance(gpu, str):
        gpu_spec = load_gpu_template(gpu)
    elif isinstance(gpu, GPUSpec) and gpu.from_:
        gpu_spec = load_gpu_template(gpu.from_)
    else:
        gpu_spec = gpu
    if gpu_spec is None:
        return None
    # Use the template file stem (the 'from_' name or the gpu string) to find the
    # companion .nccl.yaml, not gpu_spec.name (which may have spaces or differ in case).
    gpu_template_name: str | None = None
    if isinstance(node.gpu, str):
        gpu_template_name = node.gpu
    elif isinstance(node.gpu, GPUSpec) and node.gpu.from_:
        gpu_template_name = node.gpu.from_
    return load_nccl_profile(gpu_template_name) if gpu_template_name else None


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


def _load_workload_yaml(path_or_name: str, source_file: Path | None) -> dict:
    """Load workload YAML, checking templates/workload/ first, then relative path."""
    template_path = Path("templates/workload") / f"{path_or_name}.yaml"
    if template_path.exists():
        with open(template_path) as f:
            return yaml.safe_load(f)

    if source_file is not None:
        rel_path = source_file.parent / path_or_name
    else:
        rel_path = Path(path_or_name)
    return yaml.safe_load(rel_path.read_text())


def resolve_workload(
    path_or_dict: Path | str | dict,
    _visited: set[str] | None = None,
    _source_file: Path | None = None,
) -> MegatronWorkload:
    """Resolve a MegatronWorkload from a path, YAML string, or inline dict.

    Supports ``from:`` inheritance: loads the base workload YAML
    (from ``templates/workload/<name>.yaml`` or a relative path),
    deep-merges overrides, and validates the result.

    Raises ``ValueError`` on circular ``from:`` chains.
    """
    if _visited is None:
        _visited = set()

    if isinstance(path_or_dict, (Path, str)):
        source = Path(path_or_dict) if isinstance(path_or_dict, str) else path_or_dict
        with open(source) as f:
            data = yaml.safe_load(f)
        source_file = source
    else:
        data = path_or_dict
        source_file = _source_file

    from_name = data.get("from")
    if from_name is not None:
        if from_name in _visited:
            raise ValueError("Circular workload inheritance detected")
        _visited.add(from_name)
        base_data = _load_workload_yaml(from_name, source_file)
        base_wl = resolve_workload(base_data, _visited=_visited, _source_file=source_file)
        base_dict = base_wl.model_dump(by_alias=False)
        overrides = {k: v for k, v in data.items() if k != "from"}
        merged = _deep_merge(base_dict, overrides)
        return MegatronWorkload.model_validate(merged)

    return MegatronWorkload.model_validate(data)


# Compute-relevant config keys for workload hashing.
# Data-path / runtime keys are intentionally excluded.
_COMPUTE_KEYS = frozenset({
    "tensor-model-parallel-size",
    "tp",
    "pipeline-model-parallel-size",
    "pp",
    "expert-model-parallel-size",
    "ep",
    "num_microbatches",
    "num-microbatches",
    "pipeline_schedule",
    "num_layers",
    "num-layers",
    "hidden_size",
    "hidden-size",
    "num_attention_heads",
    "num-attention-heads",
    "ffn_hidden_size",
    "ffn-hidden-size",
    "vocab_size",
    "vocab-size",
    "seq_length",
    "seq-length",
    "sequence_length",
    "global_batch_size",
    "global-batch-size",
    "micro_batch_size",
    "micro-batch-size",
    "dtype",
    "num_gpus",
    "num-gpus",
    "flash_attention",
    "flash-attention",
    "swiglu",
    "num_experts",
    "num-experts",
    "top_k",
    "top-k",
})


def workload_hash(workload: MegatronWorkload) -> str:
    """Return a 16-character hex hash of the workload's compute-relevant config.

    Only fields in ``_COMPUTE_KEYS`` are included, so changing dataset paths,
    tokenizer choices, or memory snapshots does not change the hash.
    """
    filtered = {k: v for k, v in workload.config.items() if k in _COMPUTE_KEYS}
    return hashlib.sha256(
        json.dumps(filtered, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
