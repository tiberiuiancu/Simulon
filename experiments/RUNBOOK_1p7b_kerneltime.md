# Kernel-time foundation experiment — Qwen3-1.7B mbs axis

GOAL: fix the two defects the native scorecard exposed so "trace a base, extrapolate mbs"
becomes trustworthy (<10%):
  1. mbs=1 ANCHORS over-predict (tp1pp1 +12.6%, tp4pp1 +14.3%) — fake-PG per-microbatch
     launch idle accumulates over the microbatch count.
  2. single-base mbs-EXTRAPOLATION fails (+17..+71%) — the uncorrected span can't see the
     GEMM efficiency gain.
Both stem from the sim using the fake-PG wall-clock SPAN as per-op time. kernel_device_ms
is the clean signal; these traces capture it.

WHY the 32B correction (_apply_gemm_efficiency) doesn't just transfer: it holds the launch
overhead at the tp4/mbs1 reference, ASSUMING that config's span ~= hardware (true at 32B,
+5-7%). At 1.7B the mbs=1 anchors ARE the inflated points, so the reference is wrong. We need
a reference-FREE refounding: compute = kernel_compute + (overhead PER LAUNCH)*launches, with
overhead/launch MEASURED (not fitted). This experiment tests whether overhead/launch is
config-invariant — the precondition for that model.

## STEP 1 — generate the kernel-timing traces (you run; ~1.5h, 1 GPU)

```bash
cd /e/home/jusers/vanosch1/jupiter/Simulon
sbatch experiments/trace_thomas_1p7b_kerneltime.sbatch      # keeps --comment=NVIDIA_GPU_CLOCKS=1980
```
5 cells -> templates/gpu/gh200_jupiter-1p7b-kerneltime/traces:
  tp1pp1-mbs1 (anchor), tp1pp1-mbs2, tp1pp1-mbs4 (extrapolation targets), tp2pp1-mbs1,
  tp2pp1-mbs8 (the mbs8 leg).
Same nemo_26.04.sif as the physical + span traces (toolchain match). Each cell prints
`total_kernel_device_ms` + `slots_with_kernel_ms` on success.

RISK: with the profiler ON, mbs4 / mbs8 may OOM (extra profiler memory on one GPU). If a
cell shows "NO TRACE (likely OOM under profiler)", the mbs1/mbs2 pair alone still answers the
core hypothesis; we handle mbs4/8 separately (e.g. trace at higher TP, as we did physically).

## STEP 2 — the decomposition diagnostic (already written; no GPU)

```bash
PYTHONPATH=. python experiments/analyze_1p7b_kerneltime.py
```
Prints per cell: span_compute, kernel_compute, overhead=span-kernel, launches,
overhead/launch, kernel/microbatch. Then:
  * the CONFIG-INVARIANCE test of overhead/launch across tp1pp1 mbs{1,2,4};
  * a REFOUNDED native preview (kernel_compute + a single global overhead/launch) vs physical.

## STEP 3 — the modeling fork (decided by Step 2's invariance result)

- overhead/launch CONFIG-INVARIANT (spread <15%): refound the DAG compute on
  kernel_device_ms + measured global overhead/launch. This is a change in the sim core
  (trace_tracer.py compute path / a new correction in sweep), reference-free, no fitted
  constant. Then re-run analyze_1p7b.py native + extrapolation; expect the anchor inflation
  and the mbs-extrapolation to both drop.
- overhead/launch mbs-DEPENDENT: the idle gap is kernel-size-dependent; model overhead
  per-op as a function of kernel size (still measured), not a global constant.

## Already in place (no action)
- Physical ground truth: all 10 cells, iter 40, in output/qwen3-1p7b (analyze_1p7b.py).
- Span traces: templates/gpu/gh200_jupiter-1p7b/traces.
- Native + extrapolation scorecard: experiments/results_1p7b.txt.
