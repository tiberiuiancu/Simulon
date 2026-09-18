# Robustness pass — Qwen3-1.7B validation setup

Three defects were found in the setup itself (not in the model being validated). Until they
are cleared, several published deltas are either wrong or unresolvable. Everything below is
implemented and ready; nothing has been run.

| # | defect | consequence | fix |
|---|--------|-------------|-----|
| 1 | KT traces book ALL kernel time on **forward** slots, `bwd = exactly 0` | backward replays as ZERO compute -> every schedule-sensitive result (PP bubble, VPP) is fiction. **The PP f-drift 0.66/0.71 is an artifact, not physics.** PP=1 rows survive (totals ~right). | tracer rewritten to attribute by timestamp |
| 2 | every physical cell is a **single run**; no noise floor | we issue +/-5% / +/-10% verdicts with unknown resolution; GH200 drift is ~9% across sessions | 3 cells x 3 repeats |
| 3 | traces captured across **>=4 sessions**; 6 span traces still at stale `padded_vocab=262144` | `overhead = span - kernel` mixes signal with a ~9% cross-session artifact **of equal size** | one unified back-to-back capture |

## STEP 0 — smoke-test the rewritten attribution FIRST  (~4 min, 1 GPU)

`_kernel_ms_by_slot` has never been executed. Its failure mode is a silent one — a
chrome-trace category mismatch across PyTorch/kineto versions yields ZERO attributed
kernels while the capture still "succeeds". Finding that out after a 1-hour, 20-trace
capture wastes the hour. One cell answers the only question that matters (does `bwd` now
carry non-zero kernel time?):

```bash
sbatch experiments/trace_1p7b_kt_smoke.sbatch     # prints PASS / FAIL
```
Categories are now matched case-insensitively with historical spellings, and a failed or
empty attribution prints `SIMULON WARNING:` to the job log rather than passing silently.
**Only proceed to STEP 1 on PASS.** A FAIL with a large `unattributed` value means the slot
annotations do not bound the backward kernels at all — a different anchor is needed, and
that is worth knowing before spending the hour.

## STEP 1 — re-capture traces with the fixed tracer  (~1 h, 1 node, 1 GPU)

The tracer fix is in
`vendor/Megatron-LM-traced/megatron/core/instrumentation/tracer.py` (`_kernel_ms_by_slot`):
kernels are attributed to slots **by timestamp** from the kineto chrome trace (matching each
kernel to its launch site by correlation id), instead of rolling up a `record_function`
scope's `device_time_total`. The old path was thread-local and lost every backward kernel to
autograd's fwd/bwd linking. A new `unattributed_kernel_device_ms` field reports kernel time
that landed outside all slots, so this failure mode is now visible instead of silent.

```bash
cd /e/home/jusers/vanosch1/jupiter/Simulon
sbatch experiments/trace_1p7b_unified.sbatch     # span + KT, back-to-back, pinned clocks
```
Captures all 10 cells in BOTH forms in ONE session (fixes defects 1 and 3 together).
Resumable — cells already present are skipped, so just resubmit if the walltime is hit.

EXPECT IT NOT TO FINISH IN ONE HOUR. 20 generations at ~3-6 min each is ~70-90 min, so plan
on two submissions. This is deliberate: the cell list is ordered so the six that answer the
open questions (anchor, mbs2, tp2, tp4, pp2, pp4) are captured FIRST and fit comfortably in
the first hour; the remaining four (mbs4, mbs8, tp2pp2, rc) are lower value and land on the
resubmit. A partial run is therefore still a useful run.

Then gate them before believing anything:
```bash
PYTHONPATH=. python experiments/check_trace_health.py \
    templates/gpu/gh200_jupiter-1p7b-unified/traces \
    templates/gpu/gh200_jupiter-1p7b-unified-kt/traces
```
Checks slot counts vs the microbatch schedule, **bwd kernel time non-zero and bwd/fwd in
0.5-3x**, kernel <= span, unattributed fraction, rank coverage vs PP, and padded vocab.
Exits 1 on any FAIL. This is the check that would have caught defect 1 on day one.

## STEP 2 — noise floor  (~9 short jobs, <10 min each)

```bash
bash experiments/submit_1p7b_repeats.sh --dry    # inspect first
bash experiments/submit_1p7b_repeats.sh
```
3 cells x 3 repeats, submitted SEQUENTIALLY (parallel launchers previously produced 3 jobs
each = 9 total, 6 wasted). Cells chosen for what they decide: the refounding **anchor**
(its noise propagates into every predicted cell), a **GOOD** cell (-2.9%), and the worst
**FAIL** (+14.3%) — i.e. is the failure bigger than noise?

```bash
PYTHONPATH=. python experiments/analyze_1p7b_noise.py
```
Reports per-cell spread and re-reads each existing verdict against it, labelling any delta
inside the band as **unresolved** rather than as agreement.

## STEP 3 — re-score on clean traces

```bash
PYTHONPATH=. python experiments/analyze_1p7b_refounded.py   # auto-prefers unified traces
PYTHONPATH=. python experiments/analyze_1p7b.py
```
`analyze_1p7b_refounded.py` now (a) prefers the unified registries and prints which set it
used, (b) refuses to silently reuse pre-fix KT traces — it detects `bwd == 0`, prints a
warning, marks every PP>1 row `<-- ARTIFACT`, and excludes those rows from the f-invariance
statistic.

**Expected outcome:** the PP f-drift should collapse if it was the bwd=0 artifact. The TP
drift (0.445/0.502/0.653, all at PP=1, totals only) should SURVIVE — it is the real
kernel-size effect. If TP drift disappears too, the whole f story needs rethinking.

## What this does NOT fix (still open)

- **mbs4/mbs8 under-scaling** — per-slot fwd kernel scales 1.00/1.87/2.41 for mbs 1/2/4
  where ~1/2/4 is expected. mbs2 reconciles exactly (6386+3125+27 = 9538 vs phys 9542);
  mbs4 leaves ~1800 ms unexplained. Re-check after STEP 1: if it persists with correct
  attribution, it is a real profiler limit at large slots, and mbs>=4 stays unvalidated.
- **Per-kernel model** (`gap = sum max(0, dispatch - kernel_k)`) — the hardcode-free
  endpoint. Needs STEP 1 first: you cannot build a per-kernel model on traces that could
  not separate forward from backward.
- **Coverage**: recompute has 1 physical point; VPP untested at 1.7B; everything is
  single-node so the inter-node comm path is unexercised at this scale.

## Recompute axis — if compute allows (2 short jobs)

Add rc points at **PP=1 only**: `tp1pp1-mbs2-rc` and `tp2pp1-mbs1-rc`. An earlier draft of
this plan used rc@pp2, which was a design error: PP is the axis currently confounded by
defect 1, so an rc@pp2 mismatch could not be attributed to recompute rather than to PP.
Holding PP=1 isolates recompute against the part of the model that is already trusted.
1.7B has NATIVE rc traces (the recomputed backward kernels are measured, not modelled by
the 32B constant 0.0504), so this validates the better foundation directly.

## Explicitly NOT doing (and why)

- **VPP at 1.7B** — with 28 layers, interleaving needs PP>2 (Megatron forbids PP=2 without
  p2p overlap, which the protocol keeps off) -> PP=4 -> 7 layers/stage -> NLVPS must divide
  7 -> only NLVPS=1, i.e. v=7 chunks. The bubble model is validated to v=2 and known to
  over-credit by v=16, so this would probe a regime already known to be wrong. Low
  information per GPU-hour.
- **Anything at 32B** — compute constraint.
- **The per-kernel dispatch model** — needs STEP 0/1 to land first.
