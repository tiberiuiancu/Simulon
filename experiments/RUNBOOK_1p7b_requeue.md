# Qwen3-1.7B validation — re-queue after the TP=1 embedding-crash fix

## What went wrong on the first attempt (jobs 1021172–1021182 physical, 1021193 trace)

- **8/11 physical cells crashed** with `vectorized_gather_kernel ... index out of bounds`.
  Root cause: NullTokenizer sets EOD token id = `vocab_size` = **262144**, and mock_data
  emits it. The embedding table = `padded_vocab_size` rows. I had pinned
  `padded_vocab_size: 262144`, so valid rows are 0..262143 and **EOD (262144) is one past
  the end**.
  - **TP>=2**: `VocabParallelEmbedding` masks tokens `>= vocab_end` → EOD zeroed → no crash.
    (This is why the 32B val40 grid, always TP>=4, never hit it — the bug was latent.)
  - **TP=1**: that mask path isn't applied → raw gather off the end → crash.
  Confirmed by the split: every TP=1 cell crashed; tp2pp1 & tp2pp2 COMPLETED (iter 40/40).
- **tp4pp1** did NOT crash (0 errors) but **TIMED OUT** at the 15-min walltime — slow init
  + 40 iters didn't fit. Needs more walltime.
- **Trace job** hit intermittent `OSError: Disk quota exceeded` on serialize (tp4pp1, mbs8)
  while the 11 physical jobs were concurrently hammering /e/home with logs. Transient, not
  a hard wall (neighboring cells wrote fine). Physical grid is done now → pressure gone.

## The fix (already applied to the repo)

- `config/experiments/thomas/qwen3_1p7b_val40.yaml`:
  - `padded_vocab_size: 262144` -> **262272** (next mult of 128 >= vocab_size+1; divisible
    by 128*TP for TP in {1,2,4}; gives EOD a valid row so TP=1 no longer crashes).
  - slurm `time: "00:15:00"` -> **"00:30:00"** (tp4pp1 timeout headroom).
- `experiments/thomas_1p7b_workloads/*.yaml` (all 11): `padded-vocab-size` -> **262272**
  (keep the trace vocab-projection GEMM matched to the physical runs).

## Re-queue — STEP 1: physical grid (you run this)

```bash
cd /e/home/jusers/vanosch1/jupiter/oellm-autoexp
source jupiter_init.sh >/dev/null 2>&1 && \
PYTHONPATH=. python scripts/run_autoexp.py \
    --config-name experiments/thomas/qwen3_1p7b_val40
```
11 cells, 1 node each, <=30 min. Expect ALL 11 to reach iteration 40 now (incl. the TP=1
cells). Sanity-check afterwards: `grep -L 'iteration       40/' output/qwen3-1p7b/*/slurm-*.log`
should be empty, and `grep -c 'index out of bounds'` should be 0 everywhere.

## Re-queue — STEP 2: traces (you run this; can run in parallel with step 1)

The 7 existing traces are stale (padded=262144). Wipe and regen all 11 at 262272:
```bash
cd /e/home/jusers/vanosch1/jupiter/Simulon
rm -rf templates/gpu/gh200_jupiter-1p7b/traces
sbatch experiments/trace_thomas_1p7b.sbatch     # keep --comment=NVIDIA_GPU_CLOCKS=1980
```
The sbatch has a per-cell skip guard + an inventory footer. Single GPU, ~20 min for all 11.
If `Disk quota exceeded` recurs (it shouldn't, now that the physical grid isn't co-running),
the traces are tiny (~8.5M total) so it's an inode/user quota — clear old crash logs first:
`find /e/home/jusers/vanosch1/jupiter/oellm-autoexp/output/qwen3-1p7b -name 'slurm-1021*.log' -delete`
(those are the failed first-attempt logs, safe to remove).

## After BOTH complete

Ping me. I'll build the analysis harness: gh200_jupiter-1p7b registry + the scorecard in
both modes — (A) native (each cell's own trace vs physical) and (B) extrapolation (predict
every cell from tp1pp1-mbs1 alone vs physical) — held to the >5% / >10% bar. This finally
tests the end goal (trace one base, sweep the rest) on the axes that OOM at 32B: mbs 4/8 and
TP=1.
