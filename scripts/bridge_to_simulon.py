#!/usr/bin/env python3
"""Convert a Megatron-Bridge recipe ConfigContainer into a simulon workload YAML.

Usage:
    python scripts/bridge_to_simulon.py \
        --recipe megatron.bridge.recipes.deepseek.deepseek_v3 \
        --function deepseek_v3_pretrain_config \
        --output deepseek_v3.yaml

    python scripts/bridge_to_simulon.py \
        --recipe experiments/usecase_deepseek/deepseek_config.py \
        --function deepseek_v3_pretrain_config \
        --output deepseek_v3.yaml

The script imports the Bridge recipe, runs the config function, and flattens the
nested ConfigContainer into a flat dict of Megatron-LM CLI flags that simulon
expects in workload.config.  CLI flag names are derived directly from dataclass
field metadata (argparse_meta) so any new field added to TransformerConfig is
automatically picked up without whitelist maintenance.
"""

import argparse
import dataclasses
import importlib
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------


_KNOWN_REVERSE_OVERRIDES: dict[str, str] = {
    "num_moe_experts": "num-experts",
    "fp8_param": "fp8-param-gather",
    "fp4_param": "fp4-param-gather",
    "grad_reduce_in_fp32": "grad-reduce-in-bf16",
    "fp8": "fp8-format",
    "fp4": "fp4-format",
    "use_precision_aware_optimizer": "use-precision-aware-optimizer",
    "layernorm_zero_centered_gamma": "layernorm-zero-centered-gamma",
    "random_seed": "seed",
    "sequence_length": "seq-length",
    "virtual_pipeline_model_parallel_size": "num-virtual-stages-per-pipeline-rank",
    "apply_rope_fusion": "no-|rope-fusion",
    # Inverted booleans (store_false in argparse — default True, emit --no-* when False)
    "masked_softmax_fusion": "no-|masked-softmax-fusion",
    "gradient_accumulation_fusion": "no-|gradient-accumulation-fusion",
    "bias_dropout_fusion": "no-|bias-dropout-fusion",
}


_SKIP_INTERNAL: frozenset[str] = {
    "params_dtype",
    "pipeline_dtype",
    "data_parallel_size",
    "world_size",
    "rank",
    "local_rank",
    "device_count",
    "deallocate_pipeline_outputs",
    "persist_layer_norm",
    "activation_func",
    "gated_linear_unit",
    "bias_activation_fusion",
    "config_logger_dir",
    "rotary_interleaved",
    "batch_p2p_comm",
    "input_data",
    "lazy_init",
    "use_megatron_fsdp",
    "average_in_collective",
    "transformer_layer_spec",
    "hf_model_id",
    "check_for_nan_in_grad",
}

# Flags related to training / logging / checkpointing / profiling we strip
_TRAINING_FLAG_PATTERNS: set[str] = {
    "tensorboard",
    "log-interval",
    "save",
    "load",
    "ckpt",
    "eval",
    "lr-",
    "warmup",
    "weight-decay",
    "profile",
    "train-iters",
    "exit-signal",
    "check-optimizer",
    "decrease-batch",
    "empty-unused",
    "skip-train",
    "timing-log",
    "use-nsys",
    "use-pytorch-profiler",
    "flight-recorder",
    "distributed-timeout",
    "distributed-backend",
}

# ---------------------------------------------------------------------------


def _field_cli_name(field: dataclasses.Field) -> str:
    """Derive CLI flag key from a dataclass field using argparse_meta or kebab-case default."""
    meta = field.metadata.get("argparse_meta", {})
    arg_names = meta.get("arg_names", [])
    if arg_names:
        return arg_names[0].lstrip("-")
    return field.name.replace("_", "-")


def _is_torch_dtype(value: Any) -> bool:
    try:
        import torch

        return isinstance(value, torch.dtype)
    except Exception:
        return False


def _torch_dtype_to_str(value: Any) -> str | None:
    import torch

    mapping = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        torch.float8_e4m3fn: "fp8",
        torch.float8_e5m2: "fp8",
    }
    return mapping.get(value, str(value).split(".")[-1])


def _convert_value(value: Any) -> Any:
    if _is_torch_dtype(value):
        return _torch_dtype_to_str(value)
    if isinstance(value, list | tuple):
        return [_convert_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _convert_value(v) for k, v in value.items()}
    return value


def _list_to_cli_string(value: list | tuple) -> str:
    if not value:
        return ""
    if isinstance(value[0], list | tuple):
        layer_chars = {"embedding": "E", "decoder": "t", "mtp": "m", "loss": "L"}
        stages = []
        for stage in value:
            stages.append("".join(layer_chars.get(item, str(item)[0]) for item in stage))
        return "|".join(stages)
    if all(isinstance(x, int) for x in value):
        return str(value).replace(" ", "")
    return str(value).replace(" ", "")


def _should_skip_field(field: dataclasses.Field, value: Any, defaults: dict[str, Any]) -> bool:
    default_val = defaults.get(field.name)
    return (
        not field.init
        or field.name in _SKIP_INTERNAL
        or field.name.startswith("_")
        or value == default_val
        or (
            value is None
            and (
                default_val is None
                or (isinstance(default_val, dataclasses.Field) and default_val.default is None)
            )
        )
    )


def _subconfig_to_flags(sub_instance: Any) -> dict[str, Any]:
    """Walk a single sub-config dataclass instance and return {cli_key: value}."""
    flags = {}
    if sub_instance is None or not dataclasses.is_dataclass(sub_instance):
        return flags

    # Collect default values from the class so we can skip unchanged fields
    defaults_map: dict[str, Any] = {}
    for f in dataclasses.fields(sub_instance):
        if not f.init:
            continue
        default_val = f.default_factory() if callable(f.default_factory) else f.default
        defaults_map[f.name] = default_val

    for field in dataclasses.fields(sub_instance):
        if not field.init:
            continue
        val = getattr(sub_instance, field.name, None)
        if _should_skip_field(field, val, defaults_map):
            continue

        if val is None:
            continue

        val = _convert_value(val)
        if isinstance(val, list | tuple):
            val = _list_to_cli_string(val)

        override = _KNOWN_REVERSE_OVERRIDES.get(field.name)
        if override:
            if override.startswith("no-|"):
                body = override.replace("no-|", "")
                if not val:
                    flags[f"no-{body}"] = True
            else:
                flags[override] = val
            continue

        cli_key = _field_cli_name(field)

        # Skip lists converted to empty strings
        if val == "":
            continue

        flags[cli_key] = val

    return flags


def _build_simulon_config(cfg: Any) -> dict[str, Any]:
    """Flatten a ConfigContainer into a simulon workload.config dict."""
    # Sub-configs to walk, ordered by priority.  Only fields with explicit overrides are
    # needed from optimizer/ddp; model carries the bulk of config.
    result: dict[str, Any] = {}
    for attr_name in (
        "model",
        "mixed_precision",
        "optimizer",
        "ddp",
        "train",
        "scheduler",
        "dataset",
        "tokenizer",
    ):
        sub = getattr(cfg, attr_name, None)
        if sub is not None:
            result.update(_subconfig_to_flags(sub))

    # Handle special fields that have explicit argparse_meta in TransformerConfig but need
    # their value directly from the original field name.  These are already picked up by the
    # loop above because they live on cfg.model, but their CLI names were set by argparse_meta.
    # The override map already covers the renaming.

    # Strip training-only flags by name patterns
    for key in list(result):
        for pat in _TRAINING_FLAG_PATTERNS:
            if key.startswith(pat) or pat in key:
                del result[key]
                break

    mp = getattr(cfg, "mixed_precision", None)
    if mp is not None:
        for attr_name, flag_name in (("fp8", "fp8-format"), ("fp8_recipe", "fp8-recipe")):
            raw = getattr(mp, attr_name, None)
            if raw is not None:
                override_name = _KNOWN_REVERSE_OVERRIDES.get(attr_name, flag_name)
                if override_name not in result:
                    result[override_name] = raw

    # Force overrides
    result["mock-data"] = True
    result["split"] = "1000,0,0"
    result["moe-token-dispatcher-type"] = "alltoall"

    return result


def main() -> None:
    arg_parser = argparse.ArgumentParser(
        description="Convert Megatron-Bridge recipe to simulon workload YAML"
    )
    arg_parser.add_argument(
        "--recipe",
        required=True,
        help=(
            "Dotted module path or file path to the Bridge recipe "
            "(e.g., megatron.bridge.recipes.deepseek.deepseek_v3 or "
            "experiments/usecase_deepseek/deepseek_config.py)"
        ),
    )
    arg_parser.add_argument(
        "--function",
        required=True,
        help="Name of the config function inside the recipe module (e.g., deepseek_v3_pretrain_config)",
    )
    arg_parser.add_argument(
        "--output", required=True, help="Path to write the generated workload YAML"
    )
    arg_parser.add_argument(
        "--megatron-bridge-path",
        default=None,
        help="Path to Megatron-Bridge source (added to PYTHONPATH). If not set, assumes it's installed.",
    )
    args = arg_parser.parse_args()

    perf_utils_dir = (
        Path(__file__).resolve().parents[1]
        / "vendor"
        / "Megatron-Bridge"
        / "scripts"
        / "performance"
    )
    if perf_utils_dir.exists() and str(perf_utils_dir) not in sys.path:
        sys.path.insert(0, str(perf_utils_dir))

    # Import recipe module
    recipe_arg = args.recipe
    recipe_path = Path(recipe_arg)
    is_file_path = "/" in recipe_arg or "\\" in recipe_arg or recipe_path.suffix == ".py"

    if is_file_path:
        if recipe_path.suffix != ".py":
            recipe_path = recipe_path.with_suffix(".py")
        if not recipe_path.exists():
            logger.error("Recipe file not found: %s", recipe_path)
            sys.exit(1)
        abs_path = recipe_path.resolve()
        module_name = abs_path.stem
        sys.path.insert(0, str(abs_path.parent))
        try:
            recipe_mod = importlib.import_module(module_name)
        except ImportError as e:
            logger.error("Cannot import recipe module '%s': %s", module_name, e)
            sys.exit(1)
    else:
        try:
            recipe_mod = importlib.import_module(recipe_arg)
        except ImportError as e:
            logger.error("Cannot import recipe module '%s': %s", recipe_arg, e)
            sys.exit(1)

    # Get config function
    if not hasattr(recipe_mod, args.function):
        available = [n for n in dir(recipe_mod) if not n.startswith("_")]
        logger.error(
            "Function '%s' not found in %s. Available: %s", args.function, args.recipe, available
        )
        sys.exit(1)

    config_fn = getattr(recipe_mod, args.function)
    if not callable(config_fn):
        logger.error("'%s' is not callable", args.function)
        sys.exit(1)

    # Build the Bridge ConfigContainer
    logger.info("Running %s.%s() ...", args.recipe, args.function)
    cfg = config_fn()

    # Flatten to simulon flags
    simulon_config = _build_simulon_config(cfg)

    workload = {"framework": "megatron", "config": simulon_config}

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(workload, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logger.info("Wrote workload YAML to %s", output_path)
    logger.info("  Total flags: %d", len(simulon_config))


if __name__ == "__main__":
    main()
