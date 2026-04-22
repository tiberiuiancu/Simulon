#!/usr/bin/env python3
"""Wrapper to run Megatron-LM GPT pretraining with torch profiler chrome trace export."""

from __future__ import annotations

import argparse
import contextlib
import gzip
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
MEGATRON_DIR = SCRIPT_DIR / "megatron-lm"
RESULTS_DIR = SCRIPT_DIR / "results"


def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _build_argv(config: dict) -> list[str]:
    argv = []
    for key, value in config.items():
        arg = f"--{key}"
        if isinstance(value, bool):
            if value:
                argv.append(arg)
        else:
            argv.extend([arg, str(value)])
    return argv


def _find_chrome_trace(tb_dir: Path) -> Path | None:
    profile_dir = tb_dir.parent / "torch_profile"
    if not profile_dir.exists():
        return None
    candidates = sorted(profile_dir.glob("rank-*.json.gz"))
    return candidates[0] if candidates else None


def _copy_chrome_trace(src: Path, dst: Path) -> None:
    shutil.copy(src, dst)
    json_path = dst.with_suffix("")
    with gzip.open(dst, "rt") as f_in, open(json_path, "w") as f_out:
        f_out.write(f_in.read())


def _run_subprocess(argv: list[str], log_path: Path) -> Path | None:
    env = {**os.environ, "PYTHONPATH": f"{MEGATRON_DIR}:{os.environ.get('PYTHONPATH', '')}"}
    cmd = [sys.executable, str(MEGATRON_DIR / "pretrain_gpt.py")] + argv
    with open(log_path, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, check=True)
    tb_dir = Path(argv[argv.index("--tensorboard-dir") + 1]) if "--tensorboard-dir" in argv else Path("tensorboard")
    return _find_chrome_trace(tb_dir)


def _setup_megatron_path() -> None:
    megatron_path = str(MEGATRON_DIR)
    if megatron_path not in sys.path:
        sys.path.insert(0, megatron_path)
    os.environ["PYTHONPATH"] = f"{megatron_path}:{os.environ.get('PYTHONPATH', '')}"


def _run_real(config: dict, log_path: Path) -> Path | None:
    _setup_megatron_path()

    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from functools import partial

    from megatron.training import pretrain, set_startup_timestamps, inprocess_restart
    from megatron.training.arguments import parse_and_validate_args
    from megatron.core.enums import ModelType
    from gpt_builders import gpt_builder
    from model_provider import model_provider
    import pretrain_gpt as pg

    vocab_size = int(config.get("vocab-size", 32000))
    seq_length = int(config.get("seq-length", 8192))
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    pad_id = tokenizer.eos_token_id or 0

    class C4Iterator:
        def __init__(self, split: str, num_samples: int) -> None:
            self.dataset = load_dataset("c4", "en", split=split, streaming=True)
            self.num_samples = num_samples
            self.count = 0
            self.stream = iter(self.dataset)

        def __next__(self) -> dict[str, torch.Tensor]:
            if self.count >= self.num_samples:
                raise StopIteration
            self.count += 1
            example = next(self.stream)
            tokens = tokenizer.encode(example["text"], add_special_tokens=False)
            if len(tokens) > seq_length:
                tokens = tokens[:seq_length]
            tokens = [min(t, vocab_size - 1) for t in tokens]
            actual_len = len(tokens)
            if actual_len < seq_length:
                tokens = tokens + [pad_id] * (seq_length - actual_len)
            tokens_t = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
            return {
                "tokens": tokens_t,
                "labels": tokens_t.clone(),
                "loss_mask": torch.ones_like(tokens_t, dtype=torch.float),
                "attention_mask": torch.ones_like(tokens_t, dtype=torch.long),
                "position_ids": torch.arange(seq_length, dtype=torch.long).unsqueeze(0),
            }

        def __iter__(self):
            return self

    def c4_provider(train_val_test_num_samples, vp_stage=None):
        train_n, valid_n, test_n = train_val_test_num_samples
        return (
            C4Iterator("train", train_n),
            C4Iterator("validation", valid_n),
            C4Iterator("validation", test_n),
        )

    c4_provider.is_distributed = True

    original_argv = sys.argv.copy()
    sys.argv = ["pretrain_gpt.py"] + _build_argv(config)

    trace_path = None
    try:
        _PROGRAM_START_TIME = time.time()
        _MAIN_ENTRY_TIME = time.time()
        set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=_MAIN_ENTRY_TIME)

        pretrain_fn, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

        extra = pg.add_modelopt_args if getattr(pg, "has_nvidia_modelopt", False) else None
        parse_and_validate_args(
            extra_args_provider=extra,
            args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
        )

        with open(log_path, "w") as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                pretrain_fn(
                    c4_provider,
                    partial(model_provider, gpt_builder),
                    ModelType.encoder_or_decoder,
                    pg.forward_step,
                    store=store,
                    get_embedding_ranks=pg.get_embedding_ranks,
                )

        tb_dir = Path(config.get("tensorboard-dir", "tensorboard"))
        trace_path = _find_chrome_trace(tb_dir)
    finally:
        sys.argv = original_argv

    return trace_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Megatron-LM training with profiler")
    parser.add_argument("--mode", choices=["synthetic", "real"], required=True)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config_path = SCRIPT_DIR / f"gpt_oss_1b_{args.mode}.yaml"
    config = _load_config(config_path)

    log_path = RESULTS_DIR / f"megatron_{args.mode}.log"

    if args.mode == "synthetic":
        trace_path = _run_subprocess(_build_argv(config), log_path)
    else:
        trace_path = _run_real(config, log_path)

    if trace_path and trace_path.exists():
        dst = RESULTS_DIR / f"chrome_trace_{args.mode}.json.gz"
        _copy_chrome_trace(trace_path, dst)
        print(f"Chrome trace saved to {dst}")
        print(f"Decompressed JSON at {dst.with_suffix('')}")
    else:
        print(f"Warning: No chrome trace found for mode={args.mode}")

    print(f"Logs saved to {log_path}")


if __name__ == "__main__":
    main()
