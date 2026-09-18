# Falsifying k(4) — the 4-rank rank-skew constant

## Why

`templates/node/jupiter-gh200-4g.yaml` sets `launch_latency_ms: 0.911` for 4-rank
communicators. It reproduces `tp4pp1-mbs1` to −2.9%, but that is **not evidence** — the
constant was derived from that very cell. One parameter, one observation. It is calibrated
but currently **unfalsifiable**.

`tp4pp1-mbs2` fixes that. It holds the communicator size at 4 (so k(4) still applies) and
**halves the collective count**, because 128 microbatches instead of 256:

|  | tp4pp1-mbs1 | tp4pp1-mbs2 |
|---|--:|--:|
| microbatches (gbs 256, dp 1) | 256 | 128 |
| 4-rank collectives / iteration | 28928 | **14464** |
| skew contribution at k=0.911 | 26.4 s | **13.2 s** |

Same design that settled per-collective vs per-microbatch (tp2pp1 vs tp2pp2), applied to the
one axis still resting on a single point.

## PRE-REGISTERED PREDICTION — written before the run

**Physical iteration time: 26.9 – 28.3 s** (centre 27.6 s), vs 41.4 s at mbs1 — a ~33% drop.

    kernel        10.2 – 11.0 s   (kernel time is near-flat in mbs: tp1pp1 measured
                                   8686/8450/8500 ms at mbs 1/2/4; tp2pp1 −6.7% from
                                   mbs1 → mbs8. Assume flat to −7%.)
    comm + bubble  3.5 –  4.1 s   (total bytes unchanged — half as many, twice as large —
                                   and bigger messages sit higher on the busbw curve)
    skew          14464 × 0.911 ms = 13.2 s
    ------------------------------------------------
    total         26.9 – 28.3 s

### Outcomes

| measured | implied k(4) | verdict |
|---|--:|---|
| ~24.0 s | 616 µs | **REFUTES** 0.911 |
| ~26.0 s | 754 µs | consistent (low edge) |
| ~28.0 s | 892 µs | **CONFIRMS** |
| ~30.0 s | 1030 µs | consistent (high edge) |
| ~34.0 s | 1307 µs | **REFUTES** 0.911 |
| near 41 s | — | the cost is **not per-collective** at 4 ranks; the whole term is wrong |
| near 15 s | ~0 | no skew at mbs2; k(4) was an mbs1 artifact |

Note the independent cross-check already available: the directly measured 4-rank arrival
spread is **>1133 µs** (`experiments/measure_skew.py`, 3 of 4 ranks, so a lower bound). The
exposed cost must come in *below* the raw spread, so anything above ~1133 µs would be
internally inconsistent regardless of what this run says.

## Commands

Both are 1 node, well under an hour.

**1 — physical (val40, 40 iters, analyse 21–40):**
```bash
cd /e/home/jusers/vanosch1/jupiter/oellm-autoexp
source jupiter_init.sh
PYTHONPATH=. python scripts/run_autoexp.py \
    --config-name experiments/thomas/qwen3_1p7b_val40_tp4mbs2
```
Output lands in `output/qwen3-1p7b/v40_qwen3_1p7b_1_pp1_tp4_mbs2_rcFalse_vppNone_spFalse_gbs256`.

**2 — trace (span + kernel-timing, one session, pinned clocks):**
```bash
cd /e/home/jusers/vanosch1/jupiter/Simulon
sbatch experiments/trace_1p7b_tp4mbs2.sbatch
```

**3 — score (local, free):**
```bash
SIMULON_1P7B_REGISTRY=gh200_jupiter-1p7b-unified-kt python3 experiments/analyze_1p7b.py
```
The `tp4 pp1 mbs2` row is already wired in and currently prints `MISSING`; it fills in once
both artifacts exist.

## Checks before believing the result

1. **One log file per run.** Output dirs accumulate logs from many jobs; concatenating them
   fabricates a fake warmup regime. Analyse iterations 21–40 of a single log.
2. **n=1.** The tp4 noise floor is 9%, and the tp4pp1-mbs1 figure of 41449 ms was itself a
   +7.5% outlier against a 5-run mean of 39490. A single tp4pp1-mbs2 run lands inside a wide
   band. If it falls near a decision boundary, repeat it — do not adjudicate on one sample.
3. **Trace health.** `python3 experiments/check_trace_health.py` — expect
   `4 × (28 // 1) = 112` TP AllReduces per microbatch, fwd/bwd balanced, bwd non-zero.

## What each outcome means for the model

- **Inside the band** → k(4) survives out-of-sample, the per-collective law now has 2 free
  parameters against 5 observations, and the TP axis is genuinely validated rather than
  fitted. The remaining known gaps are the launch baseline (tp1pp1 −12.2%) and the PP stall
  (−25.0% / −19.5%).
- **Outside** → k(4) is an mbs1 artifact. The 2-rank constant would still stand (it rests on
  three cells spanning 8× in collective count plus a direct timestamp measurement), but the
  group-size scaling would have to be rebuilt, and no TP=4 sweep result should be trusted
  until it is.
