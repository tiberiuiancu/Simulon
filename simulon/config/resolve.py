"""Utilities for resolving hardware specs that may reference named templates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* into *base*, returning a new dict.

    Nested dicts are merged rather than replaced, so a partial sub-object
    override (e.g. only changing one field inside scale_up.switch) does not
    wipe sibling fields that were not explicitly overridden.

    ``None`` override values are ignored so that unset fields in a partial
    override do not clobber values inherited from the base template.
    """
    result = dict(base)
    for key, val in overrides.items():
        if val is None:
            continue
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


from simulon.config.dc import DatacenterConfig, GPUSpec, NodeSpec, ScaleOutSpec  # noqa: E402
from simulon.config.nccl_profile import NcclProfile  # noqa: E402
from simulon.config.workload import MegatronWorkload  # noqa: E402


def load_gpu_template(name: str) -> GPUSpec:
    """Load a GPU spec from a named YAML template file or a direct path.

    If *name* ends with ``.yaml`` or contains path separators, it is treated
    as a filesystem path. Otherwise searches ``templates/gpu/<name>.yaml``.
    """
    if name.endswith(".yaml") or "/" in name or "\\" in name:
        template_path = Path(name)
        if not template_path.exists():
            raise FileNotFoundError(f"GPU template not found: {name!r}")
    else:
        template_path = Path("templates/gpu") / f"{name}.yaml"
        if not template_path.exists():
            candidates = (
                list(Path("templates/gpu").glob("*.yaml")) if Path("templates/gpu").exists() else []
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
    """Load a node spec from a named YAML template file or a direct path.

    If *name* ends with ``.yaml`` or contains path separators, it is treated
    as a filesystem path. Otherwise searches ``templates/node/<name>.yaml``.
    """
    if name.endswith(".yaml") or "/" in name or "\\" in name:
        template_path = Path(name)
        if not template_path.exists():
            raise FileNotFoundError(f"Node template not found: {name!r}")
    else:
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

    Reads from the resolved node's scale_out field.
    """
    node = resolve_node_spec(dc)
    return node.scale_out


def _load_workload_yaml(path_or_name: str, source_file: Path | None) -> dict:
    """Load workload YAML, checking templates/workload/ first, then relative path."""
    template_path = Path("templates/workload") / f"{path_or_name}.yaml"
    if template_path.exists():
        with open(template_path) as f:
            return yaml.safe_load(f)

    rel_path = source_file.parent / path_or_name if source_file is not None else Path(path_or_name)
    return yaml.safe_load(rel_path.read_text())


def resolve_datacenter(path: Path | str) -> DatacenterConfig:
    """Load a DatacenterConfig from a YAML file path."""
    source = Path(path) if isinstance(path, str) else path
    with open(source) as f:
        data = yaml.safe_load(f)
    return DatacenterConfig.model_validate(data)


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

    if isinstance(path_or_dict, Path | str):
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


# Keys intentionally excluded from workload hashing.
# Everything else in workload.config is automatically included.
# Excluded categories: data/runtime/infra flags, tokenizer choices,
# memory/GC tuning, and architecture knobs that don't change compute graph.
_EXCLUDED_HASH_KEYS = frozenset(
    {
        # data / runtime / infra
        "framework",
        "seed",
        "split",
        "reset-position-ids",
        "reset-attention-mask",
        "eod-mask-loss",
        "tokenizer-type",
        "tokenizer-model",
        "mock-data",
        "mmap-bin-files",
        "manual-gc",
        "manual-gc-interval",
        "num-workers",
        "override-opt_param_scheduler",
        "override-opt-param-scheduler",
        # architecture knobs that don't change the compute graph
        "q-lora-rank",
        "init-method-std",
        "make-vocab-size-divisible-by",
    }
)


def workload_hash(workload: MegatronWorkload) -> str:
    """Return a 16-character hex hash of the workload's compute-relevant config.

    Every key in ``workload.config`` is included **except** those listed in
    ``_EXCLUDED_HASH_KEYS``.  Adding a new config flag therefore automatically
    changes the hash; data-path / runtime flags must be explicitly added to the
    exclusion list to keep the hash stable.
    """
    filtered = {k: v for k, v in workload.config.items() if k not in _EXCLUDED_HASH_KEYS}
    return hashlib.sha256(json.dumps(filtered, sort_keys=True, default=str).encode()).hexdigest()[
        :16
    ]
