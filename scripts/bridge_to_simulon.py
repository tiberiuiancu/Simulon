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
expects in workload.config.
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

# Fields that are computed/derived and should NOT be emitted as CLI flags.
_DERIVED_FIELDS = {
    # Computed internally by Megatron-LM from other flags
    "params_dtype",
    "pipeline_dtype",
    "data_parallel_size",
    "world_size",
    "rank",
    "local_rank",
    "device_count",
    "deallocate_pipeline_outputs",
    "persist_layer_norm",
    "layernorm_zero_centered_gamma",
    "activation_func",
    "gated_linear_unit",
    "bias_activation_fusion",
    "config_logger_dir",
    "num_moe_experts",
    "rotary_interleaved",
    "fp8_param",
    "batch_p2p_comm",
    # Internal / init=False fields
    "input_data",  # PipelineParallelLayerLayout
    "lazy_init",
    "use_megatron_fsdp",  # Often overridden by Bridge logic
    "average_in_collective",  # Internal DDP field
    "transformer_layer_spec",  # Python callable (functools.partial), not a CLI flag
}

# Fields that exist on multiple sub-configs but should only take the value
# from a specific source when conflicts occur.
_SOURCE_PRIORITY = {
    "use_distributed_optimizer": "ddp",
    "overlap_grad_reduce": "ddp",
    "overlap_param_gather": "ddp",
    "check_for_nan_in_grad": "ddp",
    "grad_reduce_in_fp32": "ddp",
    "data_parallel_sharding_strategy": "ddp",
}

# Sub-config paths to walk (dotted names relative to ConfigContainer)
_SUBCONFIG_PATHS = [
    "model",
    "optimizer",
    "ddp",
    "train",
    "scheduler",
    "dataset",
    "tokenizer",
    "checkpoint",
    "dist",
    "comm_overlap",
    "mixed_precision",
    "validation",
    "logger",
    "rng",
    "straggler",
    "profiling",
    "peft",
    "inprocess_restart",
]

_TRAINING_ONLY_FIELDS = {
    "tensorboard-dir",
    "tensorboard-log-interval",
    "tensorboard-queue-size",
    "log-timers-to-tensorboard",
    "log-loss-scale-to-tensorboard",
    "log-validation-ppl-to-tensorboard",
    "log-memory-to-tensorboard",
    "log-device-memory-used",
    "log-l2-norm-grad-to-tensorboard",
    "log-runtime-to-tensorboard",
    "log-world-size-to-tensorboard",
    "log-energy",
    "log-progress",
    "log-throughput",
    "log-throughput-to-tensorboard",
    "log-params-norm",
    "log-num-zeros-in-grad",
    "log-interval",
    "logging-level",
    "filter-warnings",
    "set-level-for-all-loggers",
    "skip-train-metrics-log",
    "eval-iters",
    "eval-interval",
    "full-validation",
    "multiple-validation-sets",
    "drop-last-partial-validation-sequence",
    "save",
    "save-interval",
    "save-optim",
    "save-rng",
    "load",
    "load-optim",
    "load-rng",
    "load-main-params-from-ckpt",
    "ckpt-format",
    "auto-detect-ckpt-format",
    "fully-parallel-save",
    "async-save",
    "async-strategy",
    "use-persistent-ckpt-worker",
    "fully-parallel-load",
    "finetune",
    "use-checkpoint-args",
    "use-mp-args-from-checkpoint-args",
    "use-tokenizer-model-from-checkpoint-args",
    "exit-on-missing-checkpoint",
    "replication",
    "replication-factor",
    "storage-writers-per-rank",
    "dist-ckpt-strictness",
    "save-tokenizer-assets",
    "ckpt-convert-update-legacy-dist-opt-format",
    "strict-fsdp-dtensor-load",
    "dist-ckpt-save-pre-mcore-014",
    "dist-ckpt-optim-fully-reshardable",
    "distrib-optim-fully-reshardable-mem-efficient",
    "train-iters",
    "exit-signal-handler",
    "exit-signal",
    "exit-signal-handler-for-dataloader",
    "exit-signal-handler-for-training",
    "check-optimizer-step-success",
    "decrease-batch-size-if-needed",
    "empty-unused-memory-level",
    "skip-train",
    "skip-sync-grad-norm-across-mp",
    "lr",
    "min-lr",
    "lr-decay-style",
    "lr-wsd-decay-style",
    "lr-warmup-iters",
    "lr-warmup-samples",
    "lr-warmup-init",
    "override-opt-param-scheduler",
    "use-checkpoint-opt-param-scheduler",
    "start-weight-decay",
    "end-weight-decay",
    "weight-decay-incr-style",
    "timing-log-level",
    "timing-log-option",
    "use-nsys-profiler",
    "profile-step-start",
    "profile-step-end",
    "use-pytorch-profiler",
    "pytorch-profiler-collect-shapes",
    "pytorch-profiler-collect-callstack",
    "pytorch-profiler-collect-chakra",
    "record-memory-history",
    "memory-snapshot-path",
    "record-shapes",
    "nvtx-ranges",
    "flight-recorder-trace-buffer-size",
    "flight-recorder-dump-on-timeout",
    "flight-recorder-include-stack-trace",
    "flight-recorder-include-only-active",
    "flight-recorder-extra-dump-on-exec",
    "distributed-timeout-minutes",
    "distributed-backend",
}


def _to_cli_key(field_name: str) -> str:
    """Convert snake_case dataclass field to kebab-case CLI flag key."""
    return field_name.replace("_", "-")


def _is_torch_dtype(value: Any) -> bool:
    """Check if value is a torch dtype like torch.float32."""
    # Avoid importing torch if we can, but handle both module attribute and string cases.
    try:
        import torch

        return isinstance(value, torch.dtype)
    except Exception:
        return False


def _torch_dtype_to_str(value: Any) -> str | None:
    """Convert torch.dtype to CLI string: fp32, bf16, fp16, etc."""
    import torch

    mapping = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        torch.float8_e4m3fn: "fp8",
        torch.float8_e5m2: "fp8",
    }
    return mapping.get(value, str(value).split(".")[-1])


def _should_skip_field(field: dataclasses.Field, value: Any) -> bool:
    """Determine whether a field should be omitted from the YAML output."""
    if not field.init:
        return True
    if field.name in _DERIVED_FIELDS:
        return True
    if field.name.startswith("_"):
        return True
    if value is None:
        if field.default is None:
            return True
        if isinstance(field.default, dataclasses.Field) and field.default.default is None:
            return True
    return False


def _convert_value(value: Any) -> Any:
    """Convert a Python value to a YAML-friendly value."""
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


def _gather_fields_from_dataclass(
    obj: Any, prefix: str = "", collected: dict[str, tuple[str, Any]] | None = None
) -> dict[str, tuple[str, Any]]:
    """
    Recursively walk a dataclass instance and collect (source_path, value) keyed by CLI flag name.

    Returns a dict mapping CLI key -> (source_path, value).
    """
    if collected is None:
        collected = {}
    if obj is None:
        return collected

    # Handle dataclass instances
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for field in dataclasses.fields(obj):
            value = getattr(obj, field.name, None)
            if _should_skip_field(field, value):
                continue
            cli_key = _to_cli_key(field.name)
            source = f"{prefix}.{field.name}" if prefix else field.name
            converted = _convert_value(value)

            # Nested dataclass -> recurse
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                _gather_fields_from_dataclass(value, prefix=source, collected=collected)
            else:
                # If key already exists, apply source priority
                if cli_key in collected:
                    existing_source, existing_value = collected[cli_key]
                    priority_source = _SOURCE_PRIORITY.get(field.name)
                    if priority_source and priority_source in source:
                        # Prefer the prioritized source
                        collected[cli_key] = (source, converted)
                    # Otherwise keep the first occurrence (model fields win)
                else:
                    collected[cli_key] = (source, converted)
        return collected

    return collected


def _build_simulon_config(cfg: Any) -> dict[str, Any]:
    """Flatten a ConfigContainer into a simulon workload.config dict."""
    collected: dict[str, tuple[str, Any]] = {}

    for sub_path in _SUBCONFIG_PATHS:
        obj = cfg
        for attr in sub_path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            _gather_fields_from_dataclass(obj, prefix=sub_path, collected=collected)

    result = {}
    for cli_key, (_source, value) in collected.items():
        if value is None:
            continue
        if isinstance(value, list) and len(value) == 0:
            continue
        if cli_key in _TRAINING_ONLY_FIELDS:
            continue
        if isinstance(value, list | tuple):
            value = _list_to_cli_string(value)
        result[cli_key] = value

    if "use-distributed-optimizer" in result:
        result["distributed-optimizer"] = result.pop("use-distributed-optimizer")

    result["mock-data"] = True
    result["split"] = "1000,0,0"

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Megatron-Bridge recipe to simulon scenario YAML"
    )
    parser.add_argument(
        "--recipe",
        required=True,
        help=(
            "Dotted module path or file path to the Bridge recipe "
            "(e.g., megatron.bridge.recipes.deepseek.deepseek_v3 or "
            "experiments/usecase_deepseek/deepseek_config.py)"
        ),
    )
    parser.add_argument(
        "--function",
        required=True,
        help="Name of the config function inside the recipe module (e.g., deepseek_v3_pretrain_config)",
    )
    parser.add_argument("--output", required=True, help="Path to write the generated workload YAML")
    parser.add_argument(
        "--megatron-bridge-path",
        default=None,
        help="Path to Megatron-Bridge source (added to PYTHONPATH). If not set, assumes it's installed.",
    )
    args = parser.parse_args()

    # Ensure Bridge is importable
    if args.megatron_bridge_path:
        sys.path.insert(0, args.megatron_bridge_path)

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
