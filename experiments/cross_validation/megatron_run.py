#!/usr/bin/env python3
"""Parse a single oellm-autoexp Megatron run directory.

An oellm-autoexp run directory (e.g. ``output/qwen3-32b/qwen3_32b_16_ep1_pp2_..``)
contains the Slurm stdout log Megatron writes.  Two things are extracted:

1. **What actually ran** — parsed from the Megatron "arguments" dump that every
   run prints at startup (``sequence_parallel ........ True``, ``world_size .. 64``
   …).  This is authoritative: it reflects the resolved config the framework
   executed, including settings (like sequence-parallel) that the directory name
   does not encode.

2. **Measured throughput** — parsed from the per-iteration ``log_throughput``
   lines.  The fastest iteration (after a warmup skip) is taken, matching how the
   Simulon ``reference.yaml`` files were produced.

The result is a :class:`MeasuredRun` that the cross-reference tool joins against
a Simulon simulation of the same config.
"""

from __future__ import annotations

import argparse
import glob
import re
from dataclasses import dataclass, field
from pathlib import Path

# A Megatron per-iteration log line (one physical line, pipe-delimited fields).
_ITER_RE = re.compile(
    r"iteration\s+(\d+)/\s*\d+.*?"
    r"elapsed time per iteration \(ms\):\s*([\d.]+).*?"
    r"throughput per GPU \(TFLOP/s/GPU\):\s*([\d.]+).*?"
    r"Tokens per second per GPU \(Tok/s/GPU\):\s*([\d.]+)"
)

# A line from Megatron's startup "arguments" dump, e.g.
#   [default0]:  sequence_parallel ............................... True
_ARG_RE = re.compile(r"^(?:\[[^\]]*\]:)?\s*([a-zA-Z0-9_]+)\s\.{3,}\s*(.+?)\s*$")

# Arguments we care about for matching a Simulon config.
_WANTED_ARGS = {
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "micro_batch_size",
    "global_batch_size",
    "sequence_parallel",
    "num_layers",
    "seq_length",
    "num_layers_per_virtual_pipeline_stage",
    "recompute_activations",
    "recompute_granularity",
    "world_size",
}


def _coerce(val: str):
    v = val.strip()
    if v in ("True", "False"):
        return v == "True"
    if v in ("None", "null", ""):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


@dataclass
class MeasuredRun:
    run_dir: Path
    log_path: Path | None
    args: dict = field(default_factory=dict)

    # measured metrics (fastest iteration)
    iter_time_ms: float | None = None
    per_gpu_tflops: float | None = None
    per_gpu_tps: float | None = None
    fastest_iter: int | None = None
    num_iters_seen: int = 0

    # ---- convenience accessors over the parsed Megatron args ----
    @property
    def tp(self) -> int | None:
        return self.args.get("tensor_model_parallel_size")

    @property
    def pp(self) -> int | None:
        return self.args.get("pipeline_model_parallel_size")

    @property
    def mbs(self) -> int | None:
        return self.args.get("micro_batch_size")

    @property
    def gbs(self) -> int | None:
        return self.args.get("global_batch_size")

    @property
    def sp(self) -> bool:
        return bool(self.args.get("sequence_parallel"))

    @property
    def num_gpus(self) -> int | None:
        return self.args.get("world_size")

    @property
    def vpp(self) -> int | None:
        return self.args.get("num_layers_per_virtual_pipeline_stage")

    @property
    def recompute(self) -> bool:
        return bool(self.args.get("recompute_activations")) or bool(
            self.args.get("recompute_granularity")
        )

    @property
    def seq_length(self) -> int | None:
        return self.args.get("seq_length")

    @property
    def throughput_tps(self) -> float | None:
        if self.per_gpu_tps is None or self.num_gpus is None:
            return None
        return self.per_gpu_tps * self.num_gpus

    @property
    def config_key(self) -> tuple:
        """Canonical (tp, pp, sp, recompute, mbs, gbs, vpp, num_gpus) identity."""
        return (self.tp, self.pp, self.sp, self.recompute, self.mbs,
                self.gbs, self.vpp, self.num_gpus)

    def label(self) -> str:
        vpp_s = f"vp{self.vpp}" if self.vpp else "vpNone"
        sp_s = "sp" if self.sp else "no-sp"
        rc_s = "rc" if self.recompute else "no-rc"
        nodes = self.num_gpus // 4 if self.num_gpus else "?"
        return (
            f"tp{self.tp}-pp{self.pp}-{sp_s}-{rc_s}-mbs{self.mbs}-gbs{self.gbs}-{vpp_s}-{nodes}n"
        )


def _candidate_logs(run_dir: Path) -> list[Path]:
    """Return readable log files for a run, newest Slurm job id first."""
    cands = []
    for p in glob.glob(str(run_dir / "*.log")):
        path = Path(p)
        if path.is_file():  # skip dangling current.log symlinks
            cands.append(path)

    def score(path: Path) -> tuple[int, int]:
        m = re.search(r"slurm-(\d+)\.log$", path.name)
        job_id = int(m.group(1)) if m else -1
        return (job_id, path.stat().st_size)

    return sorted(cands, key=score, reverse=True)


def parse_run(run_dir: Path, warmup_iters: int = 1) -> MeasuredRun:
    """Parse one run directory into a :class:`MeasuredRun`.

    Tries each candidate log (newest job first) until one yields both the
    arguments dump and at least one iteration line.
    """
    run_dir = Path(run_dir)
    result = MeasuredRun(run_dir=run_dir, log_path=None)

    for log_path in _candidate_logs(run_dir):
        args: dict = {}
        iters: list[tuple[int, float, float, float]] = []
        try:
            text = log_path.read_text(errors="replace")
        except OSError:
            continue

        for line in text.splitlines():
            am = _ARG_RE.match(line)
            if am and am.group(1) in _WANTED_ARGS and am.group(1) not in args:
                args[am.group(1)] = _coerce(am.group(2))
            im = _ITER_RE.search(line)
            if im:
                iters.append(
                    (int(im.group(1)), float(im.group(2)), float(im.group(3)), float(im.group(4)))
                )

        if not iters:
            continue

        # Skip warmup iterations, then take the fastest (min elapsed) iteration.
        usable = [it for it in iters if it[0] > warmup_iters] or iters
        fastest = min(usable, key=lambda it: it[1])

        result.log_path = log_path
        result.args = args
        result.num_iters_seen = len(iters)
        result.fastest_iter = fastest[0]
        result.iter_time_ms = fastest[1]
        result.per_gpu_tflops = fastest[2]
        result.per_gpu_tps = fastest[3]
        return result

    return result  # no usable log found


def discover_runs(base_dir: Path) -> list[Path]:
    """Return immediate subdirectories of *base_dir* that look like runs."""
    base = Path(base_dir)
    if not base.is_dir():
        return []
    return sorted(d for d in base.iterdir() if d.is_dir() and any(d.glob("*.log")))


def _main() -> None:
    ap = argparse.ArgumentParser(description="Parse oellm-autoexp Megatron run dirs.")
    ap.add_argument("paths", nargs="+", type=Path,
                    help="Run directories, or a base dir to discover runs under.")
    ap.add_argument("--warmup", type=int, default=1, help="Iterations to skip (default 1)")
    args = ap.parse_args()

    runs: list[Path] = []
    for p in args.paths:
        if (p / "").is_dir() and not any(p.glob("*.log")):
            runs.extend(discover_runs(p))
        else:
            runs.append(p)

    print(f"{'config':<52}  {'iter ms':>10}  {'tps/GPU':>9}  {'TF/s/GPU':>9}  {'iter#':>6}")
    print("─" * 96)
    for rd in runs:
        r = parse_run(rd, warmup_iters=args.warmup)
        if r.iter_time_ms is None:
            print(f"{rd.name:<52}  {'NO DATA':>10}")
            continue
        print(f"{r.label():<52}  {r.iter_time_ms:>10.1f}  {r.per_gpu_tps:>9.1f}  "
              f"{r.per_gpu_tflops:>9.1f}  {r.fastest_iter:>6}")


if __name__ == "__main__":
    _main()
