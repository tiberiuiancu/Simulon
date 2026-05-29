import os
import subprocess
import sys
from pathlib import Path

import typer
import yaml

from simulon.cli.utils import _ensure_c4_dataset

trace_app = typer.Typer(help="Trace generation commands.")

_MEGATRON_ENTRYPOINT = (
    Path(__file__).parents[2] / "vendor" / "Megatron-LM-traced" / "pretrain_gpt.py"
)


@trace_app.command("generate")
def generate_trace(
    scenario: str = typer.Argument(..., help="Path to scenario.yaml or workload.yaml"),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", "-o", help="Directory to write trace files (overrides default)"
    ),
    gpu: str | None = typer.Option(
        None, "--gpu", help="GPU name (required for workload.yaml input)"
    ),
    stages: list[int] | None = typer.Option(
        None, "--stage", help="Specific PP stages to trace (default: first, middle, last)"
    ),
    all_ranks: bool = typer.Option(
        False, "--all-ranks", help="Trace every GPU rank (default: first rank per PP stage)"
    ),
    mock_data: bool | None = typer.Option(
        None,
        "--mock-data/--no-mock-data",
        help="Use mock synthetic data (default: from config or True)",
    ),
    data_path: str | None = typer.Option(
        None, "--data-path", help="Path to tokenized dataset directory (required for real data)"
    ),
    tokenizer_type: str | None = typer.Option(
        None,
        "--tokenizer-type",
        help="Tokenizer type for real data (e.g., GPT2BPETokenizer, HuggingFaceTokenizer)",
    ),
    tokenizer_model: str | None = typer.Option(
        None,
        "--tokenizer-model",
        help="Tokenizer model path (required for HuggingFaceTokenizer, SentencePiece, etc.)",
    ),
    vocab_file: str | None = typer.Option(
        None, "--vocab-file", help="Vocab file path (required for GPT2BPETokenizer)"
    ),
    merge_file: str | None = typer.Option(
        None, "--merge-file", help="Merge file path (required for GPT2BPETokenizer)"
    ),
    dataset: str | None = typer.Option(
        None, "--dataset", help="Preset dataset (mock) or custom data path"
    ),
    memory_snapshot: Path | None = typer.Option(
        None,
        "--memory-snapshot",
        help="Dump a PyTorch CUDA memory snapshot to the given path (for OOM debugging)",
    ),
    warmup: int = typer.Option(
        5, "--warmup", help="Number of warmup iterations to run before tracing (default: 5)"
    ),
    force_regenerate: bool = typer.Option(
        False, "--force-regenerate", help="Re-generate traces even if they already exist"
    ),
):
    """Generate per-PP-stage execution traces by running Megatron-LM with fake process groups."""
    from simulon.config.resolve import resolve_gpu_spec, resolve_workload, workload_hash
    from simulon.config.scenario import ScenarioConfig
    from simulon.config.workload import MegatronWorkload

    with open(scenario) as f:
        raw = yaml.safe_load(f)

    # Try scenario first; fall back to pure workload YAML.
    try:
        sc = ScenarioConfig.model_validate(raw)
        is_scenario = True
        workload = sc.workload
        # If workload was given as a file path, resolve it to a MegatronWorkload.
        if isinstance(workload, Path):
            workload = resolve_workload(workload)
    except Exception:
        workload = resolve_workload(scenario)
        is_scenario = False
        if gpu is None:
            raise typer.BadParameter("--gpu is required when input is a workload YAML") from None

    if not isinstance(workload, MegatronWorkload):
        raise typer.BadParameter("Workload must be a Megatron workload")

    # Determine GPU name and compute default trace path if no --output-dir
    if output_dir is None:
        if is_scenario:
            try:
                gpu_spec = resolve_gpu_spec(sc.datacenter, include_profile=False)
                gpu_name = (gpu_spec.name or "default").lower().replace(" ", "-")
            except Exception:
                gpu_name = "default"
        else:
            gpu_name = gpu.lower().replace(" ", "-")

        h = workload_hash(workload)
        output_dir = Path("templates/gpu") / gpu_name / "traces" / h

    derived_args: dict[str, str | int | bool]
    explicitly_set: set[str] = set()

    def _set_arg(flag: str, value: str | int | bool) -> None:
        derived_args[flag] = value
        explicitly_set.add(flag)

    cfg = workload.config
    derived_args = {}
    for key in (
        "tensor-model-parallel-size",
        "pipeline-model-parallel-size",
        "micro-batch-size",
        "global-batch-size",
        "seq-length",
        "expert-model-parallel-size",
        "num-layers",
        "hidden-size",
        "num-attention-heads",
        "ffn-hidden-size",
    ):
        if key in cfg:
            _set_arg(f"--{key}", cfg[key])
    if "--max-position-embeddings" not in derived_args and "--seq-length" in derived_args:
        _set_arg("--max-position-embeddings", derived_args["--seq-length"])
    skip = {"num_gpus", "num-gpus", "num_microbatches", "num-microbatches"}
    for key, value in cfg.items():
        flag = "--use-distributed-optimizer" if key == "distributed-optimizer" else f"--{key}"
        if flag not in derived_args and key not in skip:
            if isinstance(value, bool):
                _set_arg(flag, value)
            else:
                _set_arg(flag, value)

    pp = int(cfg.get("pipeline-model-parallel-size", 1))
    tp = int(cfg.get("tensor-model-parallel-size", 1))
    ep = int(cfg.get("expert-model-parallel-size", 1))
    world_size = cfg.get("num_gpus", cfg.get("num-gpus"))

    _DATASET_PRESETS: dict[str, dict[str, str | int | bool]] = {
        "mock": {"--mock-data": True, "--tokenizer-type": "NullTokenizer", "--vocab-size": 32000},
        "c4": {
            "--mock-data": False,
            "--data-path": "./data/c4_en_llama3",
            "--tokenizer-type": "HuggingFaceTokenizer",
            "--tokenizer-model": "NousResearch/Meta-Llama-3-8B",
            "--vocab-size": 128256,
            "--split": "1000,0,0",
        },
    }
    if dataset is not None:
        if dataset in _DATASET_PRESETS:
            for flag, value in _DATASET_PRESETS[dataset].items():
                _set_arg(flag, value)
        else:
            _set_arg("--mock-data", False)
            _set_arg("--data-path", dataset)
    if mock_data is not None:
        _set_arg("--mock-data", mock_data)
    if data_path is not None:
        _set_arg("--data-path", data_path)
    if tokenizer_type is not None:
        _set_arg("--tokenizer-type", tokenizer_type)
    if tokenizer_model is not None:
        _set_arg("--tokenizer-model", tokenizer_model)
    if vocab_file is not None:
        _set_arg("--vocab-file", vocab_file)
    if merge_file is not None:
        _set_arg("--merge-file", merge_file)

    if dataset == "c4" or (data_path is not None and "c4" in data_path):
        seq_len = derived_args.get("--seq-length", 8192)
        _ensure_c4_dataset(
            str(derived_args.get("--data-path", "./data/c4_en_llama3")), seq_length=int(seq_len)
        )

    _TRACE_DEFAULTS = {
        "--lr": 0.001,
        "--min-lr": 0.0,
        "--eval-interval": 1000000,
        "--eval-iters": 0,
        "--save-interval": 1000000,
        "--log-interval": 1,
        "--mock-data": True,
        "--no-masked-softmax-fusion": True,
        "--no-bias-swiglu-fusion": True,
        "--tokenizer-type": "NullTokenizer",
        "--vocab-size": 32000,
    }
    for flag, value in _TRACE_DEFAULTS.items():
        if flag not in explicitly_set:
            derived_args[flag] = value
    if "--train-iters" not in explicitly_set:
        derived_args["--train-iters"] = warmup + 1
    stages_to_trace = stages if stages is not None else list(range(pp))

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dp = int(  # noqa: F841
        cfg.get(
            "data-parallel-size",
            cfg.get(
                "data_parallel_size",
                world_size // pp // tp // ep if world_size and pp and tp else 1,
            ),
        )
    )
    ranks_per_stage = world_size // pp if world_size and pp else tp

    if all_ranks:
        ranks_to_trace = list(range(world_size))
    else:
        ranks_to_trace = [stage * ranks_per_stage for stage in stages_to_trace]

    for rank in ranks_to_trace:
        trace_file = output_dir / f"trace_rank_{rank}.json"
        if trace_file.exists() and not force_regenerate:
            typer.echo(f"Skipping rank {rank}, trace already exists: {trace_file.name}")
            continue

        cmd: list[str] = [sys.executable, str(_MEGATRON_ENTRYPOINT)]
        for flag, value in derived_args.items():
            if value is True:
                cmd.append(flag)
            elif value is False:
                continue
            elif isinstance(value, list):
                cmd.append(flag)
                cmd.extend(str(v) for v in value)
            else:
                cmd.append(flag)
                cmd.append(str(value))
        cmd.extend(["--fake-process-group", "--rank", str(rank), "--trace-dir", str(output_dir)])
        if memory_snapshot is not None:
            cmd.extend(["--record-memory-history", "--memory-snapshot-path", str(memory_snapshot)])

        typer.echo(f"Tracing rank {rank} ...")
        env = os.environ.copy()
        if world_size is not None:
            env["WORLD_SIZE"] = str(world_size)
        try:
            subprocess.run(cmd, check=True, env=env)
        except FileNotFoundError as exc:
            typer.echo(f"Error: could not run Megatron entry point: {exc}", err=True)
            raise typer.Exit(1) from exc
        except subprocess.CalledProcessError as exc:
            typer.echo(
                f"Error: Megatron exited with code {exc.returncode} for rank {rank}", err=True
            )
            raise typer.Exit(1) from exc

    if memory_snapshot is not None:
        typer.echo(f"\nMemory snapshot: {memory_snapshot}")
        visualize_cmd = (
            'python -c "import pickle, torch; '
            f"d=pickle.load(open('{memory_snapshot}','rb')); "
            'torch.cuda.memory._record_memory_history(d)"'
        )
        typer.echo(f"  Visualize with:  {visualize_cmd}")
        typer.echo("  Or load in https://pytorch.org/memory_viz")

    # Save resolved workload alongside traces
    if isinstance(workload, MegatronWorkload):
        workload_yaml_path = output_dir / "workload.yaml"
        with open(workload_yaml_path, "w") as f:
            yaml.dump(
                workload.model_dump(by_alias=False, exclude_none=True),
                f,
                default_flow_style=False,
                sort_keys=False,
            )

    trace_files = sorted(output_dir.glob("trace_rank_*.json"))
    typer.echo(f"\nTrace generation complete. {len(trace_files)} file(s) in {output_dir}:")
    for tf in trace_files:
        typer.echo(f"  {tf.name}")


@trace_app.command("list")
def list_traces(
    all: bool = typer.Option(False, "--all", "-a", help="Show all traces (no pagination)"),
    n: int = typer.Option(5, "-n", help="Number of traces to show"),
    offset: int = typer.Option(0, "--offset", help="Number of traces to skip"),
    gpu: str | None = typer.Option(None, "--gpu", help="Filter by GPU name (case-insensitive)"),
):
    """List resolved workloads from generated trace directories."""
    from simulon.config.resolve import resolve_workload, workload_hash
    from simulon.config.workload import MegatronWorkload

    base = Path("templates/gpu")
    if not base.exists():
        typer.echo("No trace directories found.")
        return

    entries: list[tuple[Path, str]] = []
    for gpu_dir in sorted(base.iterdir()):
        if not gpu_dir.is_dir():
            continue
        gpu_name = gpu_dir.name
        if gpu is not None and gpu.lower() != gpu_name.lower():
            continue
        traces_dir = gpu_dir / "traces"
        if not traces_dir.is_dir():
            continue
        for trace_dir in sorted(traces_dir.iterdir()):
            if not trace_dir.is_dir():
                continue
            workload_yaml = trace_dir / "workload.yaml"
            if workload_yaml.exists():
                entries.append((trace_dir, gpu_name))

    if not entries:
        typer.echo("No traces found.")
        return

    # Sort by modification time (newest first)
    entries.sort(key=lambda e: e[0].stat().st_mtime, reverse=True)

    # Apply pagination
    count = len(entries) if all else n
    sliced = entries[offset : offset + count]

    for i, (trace_dir, gpu_name) in enumerate(sliced):
        if i > 0:
            typer.echo("---")
        trace_hash = trace_dir.name
        workload_yaml = trace_dir / "workload.yaml"
        try:
            workload = resolve_workload(str(workload_yaml))
            h = workload_hash(workload)
        except Exception:
            h = trace_hash
            workload = None

        typer.echo(f"[{gpu_name}] hash={h}")
        if isinstance(workload, MegatronWorkload):
            cfg = workload.config
            pp = cfg.get("pipeline-model-parallel-size", cfg.get("pp", "?"))
            tp = cfg.get("tensor-model-parallel-size", cfg.get("tp", "?"))
            seq = cfg.get("seq-length", cfg.get("seq_length", cfg.get("sequence_length", "?")))
            hidden = cfg.get("hidden-size", cfg.get("hidden_size", "?"))
            layers = cfg.get("num-layers", cfg.get("num_layers", "?"))
            heads = cfg.get("num-attention-heads", cfg.get("num_attention_heads", "?"))
            typer.echo(f"  tp={tp} pp={pp} seq={seq} hidden={hidden} layers={layers} heads={heads}")
