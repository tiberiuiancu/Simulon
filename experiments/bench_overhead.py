#!/usr/bin/env python3
"""Measure the two fixed per-operation costs the simulator is missing.

Both are CLUSTER + TOOLCHAIN properties (not model properties): measure once per machine,
never per model. They are measured DIRECTLY and in isolation — not fitted to close a
residual, and not derived as a small difference of two large numbers.

  (1) KERNEL LAUNCH cost  — the compute-side fixed cost currently absorbed by the fitted `f`.
  (2) COLLECTIVE EXPOSURE — the comm-side fixed cost that is missing entirely, and the
      reason TP configs are under-predicted by 37-63%.

WHY nccl-tests CANNOT answer (2): it runs collectives BACK-TO-BACK at full throughput —
pipelined, best case. Real training issues them as BLOCKING sync points inside a dependency
chain (14k-29k per iteration at mbs=1), where the next GEMM cannot start until the collective
completes. The gap between those two regimes IS the missing term. So we measure both regimes
here and report the difference.

PRE-REGISTERED PREDICTION (stated before running, from the 1.7B campaign residual):
    exposed cost per 16.8 MB AllReduce ~= 531 us at 2 ranks, ~900 us at 4 ranks.
  * If reproduced -> the diagnosis is confirmed; the value goes straight into the model
    (NcclProfile.launch_latency_ms and its rank-dependent generalisation).
  * If it comes back ~30 us -> the per-collective attribution is WRONG, the gap lives
    elsewhere (most likely the PP bubble model), and the TP story needs rebuilding.
Either outcome is decisive. Internal control: variant C must reproduce nccl-tests
(178.8 us @2 ranks, 105.3 us @4 ranks for 16.78 MB on Jupiter) — if it does not, the
harness itself is wrong and no other number here should be believed.

    torchrun --nproc_per_node=<N> experiments/bench_overhead.py [--bytes 16777216]
"""
from __future__ import annotations

import argparse
import os
import statistics
import time

import torch

# Measured Jupiter nccl-tests reference for 16.777 MB AllReduce (results/nccl_allreduce_*).
NCCL_TESTS_REF_US = {2: 178.8, 4: 105.3}
PREDICTED_EXPOSED_US = {2: 531.0, 4: 900.0}


def _sync():
    torch.cuda.synchronize()


def time_loop(fn, n: int, warmup: int = 20, reps: int = 5) -> float:
    """Median over `reps` of (wall / n) in microseconds, after warmup."""
    for _ in range(warmup):
        fn()
    _sync()
    out = []
    for _ in range(reps):
        _sync()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        _sync()
        out.append((time.perf_counter() - t0) / n * 1e6)
    return statistics.median(out)


# ---------------------------------------------------------------------------
# (1) kernel launch cost — single GPU, no distributed
# ---------------------------------------------------------------------------
def bench_launch(dev) -> dict:
    tiny = torch.ones(1, device=dev)
    # Trivial kernel: GPU work is ~0, so wall/launch is the CPU dispatch + launch cost.
    disp = time_loop(lambda: tiny.add_(1.0), n=2000)

    # A kernel with real GPU duration: wall/iter - pure_kernel_time = the per-kernel gap
    # (what the GPU spends NOT computing between consecutive dependent kernels).
    a = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)
    b = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)

    def gemm():
        torch.mm(a, b)

    wall = time_loop(gemm, n=200)
    # pure device time for the same GEMM, measured with CUDA events back-to-back
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(20):
        gemm()
    _sync()
    ev0.record()
    for _ in range(200):
        gemm()
    ev1.record()
    _sync()
    dev_us = ev0.elapsed_time(ev1) / 200 * 1e3
    return {"dispatch_us": disp, "gemm_wall_us": wall, "gemm_device_us": dev_us,
            "gap_us": wall - dev_us}


# ---------------------------------------------------------------------------
# (2) collective exposure — the key measurement
# ---------------------------------------------------------------------------
def bench_collective(dev, nbytes: int, world: int) -> dict:
    import torch.distributed as dist

    n_el = nbytes // 2  # bf16
    buf = torch.ones(n_el, device=dev, dtype=torch.bfloat16)
    # GEMM sized so its duration is comparable to the compute between two real TP
    # collectives (~1 ms at 1.7B), making the interleaving realistic rather than degenerate.
    a = torch.randn(4096, 2048, device=dev, dtype=torch.bfloat16)
    b = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)
    state = {"x": None}

    def compute_only():
        state["x"] = torch.mm(a, b)

    def compute_plus_ar():
        # TRUE dependency chain: the collective consumes the GEMM output, and the next
        # iteration's GEMM cannot proceed until it lands. This is the TP pattern.
        x = torch.mm(a, b)
        dist.all_reduce(x)
        state["x"] = x

    def ar_only():
        dist.all_reduce(buf)

    dist.barrier()
    a_us = time_loop(compute_only, n=100)
    dist.barrier()
    b_us = time_loop(compute_plus_ar, n=100)
    dist.barrier()
    # variant C: back-to-back collectives = the nccl-tests regime (internal control)
    c_buf_us = time_loop(ar_only, n=100)
    dist.barrier()

    ar_bytes = a.shape[0] * b.shape[1] * 2  # the GEMM output actually all-reduced in B
    return {"compute_only_us": a_us, "compute_plus_ar_us": b_us,
            "exposed_ar_us": b_us - a_us, "backtoback_ar_us": c_buf_us,
            "ar_bytes_in_B": ar_bytes, "ctrl_bytes": nbytes}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bytes", type=int, default=16 * 1024 * 1024 + 777216 // 1)
    args = p.parse_args()
    nbytes = 16777216  # match the real TP AllReduce (4096 x 2048 x bf16)

    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    world = int(os.environ.get("WORLD_SIZE", 1))

    if world == 1:
        r = bench_launch(dev)
        print("=" * 72)
        print("(1) KERNEL LAUNCH COST  [1 GPU]")
        print("=" * 72)
        print(f"  CPU dispatch per trivial kernel : {r['dispatch_us']:8.2f} us")
        print(f"  GEMM wall per iter              : {r['gemm_wall_us']:8.2f} us")
        print(f"  GEMM pure device time           : {r['gemm_device_us']:8.2f} us")
        print(f"  => per-kernel GAP               : {r['gap_us']:8.2f} us")
        print("\n  The gap is the compute-side fixed cost the fitted `f` stands in for.")
        return

    import torch.distributed as dist
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    r = bench_collective(dev, nbytes, world)
    if rank == 0:
        ref = NCCL_TESTS_REF_US.get(world)
        pred = PREDICTED_EXPOSED_US.get(world)
        print("=" * 72)
        print(f"(2) COLLECTIVE EXPOSURE  [{world} ranks, AllReduce {r['ar_bytes_in_B']/1e6:.1f} MB]")
        print("=" * 72)
        print(f"  A  compute only                 : {r['compute_only_us']:9.1f} us/iter")
        print(f"  B  compute + blocking AllReduce : {r['compute_plus_ar_us']:9.1f} us/iter")
        print(f"  => EXPOSED cost per collective  : {r['exposed_ar_us']:9.1f} us   <== the term")
        print()
        print(f"  C  back-to-back AllReduce       : {r['backtoback_ar_us']:9.1f} us "
              f"(nccl-tests regime)")
        if ref:
            d = (r["backtoback_ar_us"] / ref - 1) * 100
            print(f"     nccl-tests reference         : {ref:9.1f} us  -> harness {d:+.0f}% "
                  f"{'OK' if abs(d) < 25 else 'MISMATCH: distrust everything above'}")
        if pred:
            print()
            print(f"  PRE-REGISTERED PREDICTION       : {pred:9.1f} us")
            got = r["exposed_ar_us"]
            if 0.5 * pred <= got <= 1.7 * pred:
                print("  VERDICT: CONFIRMED — per-collective exposure explains the TP gap.")
            elif got < 0.2 * pred:
                print("  VERDICT: REFUTED — exposure is far too small; the TP gap is NOT "
                      "per-collective. Look at the PP/bubble model and the accounting.")
            else:
                print("  VERDICT: PARTIAL — real but not the whole gap; decompose further.")
        print()
        print(f"  exposure beyond raw transfer    : {r['exposed_ar_us']-r['backtoback_ar_us']:9.1f} us")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
