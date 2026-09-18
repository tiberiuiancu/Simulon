#!/usr/bin/env python3
"""Pull Megatron's per-iteration component timers out of a run's tensorboard events.

Every oellm-autoexp run logs `log_timers_to_tensorboard`, so each run dir carries the
measured decomposition of an iteration -- forward/backward compute, P2P waits, the
distributed optimizer's grad reduce-scatter and param all-gather, the optimizer step --
which is exactly the split the simulator reports (compute / exposed comm / bubble / host).
Comparing the two per component attributes a residual without a single new job.

Needs `tensorboard` (present in the production container, not on the login node):

    apptainer exec --bind /e/project1 /e/project1/e-sta-openeurollm/container/nemo_26.04.sif \\
        python3 experiments/megatron_timers.py <run_dir>... [--skip 2] [--json out.json]

Values are medians over iterations > --skip of the tag's scalar (ms). Megatron reports
timers as per-iteration max over ranks by default (timing_log_option=minmax logs both).
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path


def load_timers(run_dir: Path, skip: int) -> dict[str, float]:
    from tensorboard.backend.event_processing import event_accumulator as ea
    files = sorted(glob.glob(str(run_dir / "tensorboard" / "events.out.tfevents.*"))
                   + glob.glob(str(run_dir / "**" / "events.out.tfevents.*"), recursive=True))
    if not files:
        return {}
    out: dict[str, list[float]] = {}
    for f in files:
        acc = ea.EventAccumulator(f, size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags()["scalars"]:
            if "time" not in tag and "timer" not in tag:
                continue
            vals = [(e.step, e.value) for e in acc.Scalars(tag)]
            steady = [v for s, v in vals if s > skip] or [v for _, v in vals]
            if steady:
                out.setdefault(tag, []).extend(steady)
    return {tag: statistics.median(v) for tag, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--skip", type=int, default=2)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--all-tags", action="store_true", help="print every timer tag, not just the summary set")
    a = ap.parse_args()

    # The tags that map onto the simulator's decomposition. Names as Megatron writes them
    # (`<timer>-time`); the (min|max) suffixes appear with timing_log_option=minmax.
    KEY = ["iteration-time", "forward-backward-time", "forward-compute-time", "backward-compute-time",
           "forward-recv-time", "forward-send-time", "backward-recv-time", "backward-send-time",
           "forward-send-backward-recv-time", "backward-send-forward-recv-time",
           "layernorm-grads-all-reduce-time", "embedding-grads-all-reduce-time",
           "all-grads-sync-time", "params-all-gather-time", "grads-reduce-scatter-time",
           "optimizer-time", "optimizer-copy-to-main-grad-time", "optimizer-clip-main-grad-time",
           "optimizer-inner-step-time", "batch-generator-time"]
    results = {}
    for d in a.run_dirs:
        t = load_timers(d, a.skip)
        results[str(d)] = t
        print(f"\n== {d.name}  ({len(t)} timer tags)")
        if not t:
            print("   no tensorboard scalars found"); continue
        tags = sorted(t) if a.all_tags else [k for k in t if any(k.startswith(p) for p in KEY)]
        for k in sorted(tags, key=lambda k: -t[k]):
            print(f"   {k:<48} {t[k]:10.1f} ms")
    if a.json:
        a.json.write_text(json.dumps(results, indent=1))
        print(f"\njson -> {a.json}")


if __name__ == "__main__":
    main()
