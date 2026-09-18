# Attribute the per-sequence overhead k_seq (~49 ms/seq) — Qwen3-1.7B

WHY. The kernel-time decomposition proved the sim's real per-iteration overhead
(physical - kernel_compute - comm - bubble) scales with SEQUENCES per GPU (gbs/dp) at
~49.3 ms/seq (8% spread across tp1/mbs1, tp1/mbs2, tp2/mbs1). The refounded model
`iter = kernel + comm + bubble + 49.3*(gbs/dp)` predicts the clean cells to +/-1.3%. Before
building on 49.3, ATTRIBUTE it to a real Megatron component (so it's principled, not a fitted
number). Prime suspect: batch-generator (dataloader / per-sample host work).

## STEP 1 — profiling runs with per-component timers (you run; ~3 short jobs, <10 min each)

The existing physical runs used timing_log_level=0 (aggregate only). Re-run the 3 CLEAN cells
with timing_log_level=2 into an ISOLATED group (qwen3-1p7b-profile), so the val40 dirs are
untouched:

```bash
cd /e/home/jusers/vanosch1/jupiter/oellm-autoexp
source jupiter_init.sh >/dev/null 2>&1

# tp1pp1-mbs1  (64 seq/GPU)
PYTHONPATH=. python scripts/run_autoexp.py --config-name experiments/thomas/qwen3_1p7b_val40 \
  aux.experiment_group=qwen3-1p7b-profile backend.megatron.timing_log_level=2 \
  backend.megatron.tensor_model_parallel_size=1 backend.megatron.micro_batch_size=1 \
  'sweep.groups=[]'

# tp1pp1-mbs2  (64 seq/GPU)  -- per-sequence component stays EQUAL to mbs1
PYTHONPATH=. python scripts/run_autoexp.py --config-name experiments/thomas/qwen3_1p7b_val40 \
  aux.experiment_group=qwen3-1p7b-profile backend.megatron.timing_log_level=2 \
  backend.megatron.tensor_model_parallel_size=1 backend.megatron.micro_batch_size=2 \
  'sweep.groups=[]'

# tp2pp1-mbs1  (128 seq/GPU) -- per-sequence component DOUBLES vs tp1
PYTHONPATH=. python scripts/run_autoexp.py --config-name experiments/thomas/qwen3_1p7b_val40 \
  aux.experiment_group=qwen3-1p7b-profile backend.megatron.timing_log_level=2 \
  backend.megatron.tensor_model_parallel_size=2 backend.megatron.micro_batch_size=1 \
  'sweep.groups=[]'
```
timing_log_level=2 prints per-iteration component timers as `name .....: (min, max)` ms.
(train_iters stays 40; the analysis uses iters 21-40. barrier_with_L1_time is already on, so
the component split is accurate; the small barrier cost only perturbs the aggregate, which we
don't use here.)

## STEP 2 — attribute (already written; no GPU)

```bash
cd /e/home/jusers/vanosch1/jupiter/Simulon
PYTHONPATH=. python experiments/analyze_1p7b_timers.py
```
Prints each component's mean ms per run + a PER-SEQUENCE test: the component that is ~EQUAL at
tp1/mbs1 and tp1/mbs2 (both 64 seq/GPU) and ~2x at tp2/mbs1 (128 seq/GPU), and whose tp1pp1-mbs1
value sums to ~3000 ms, IS k_seq. Expected: batch-generator (± loss / embedding host work).

## Outcome -> next
- If a clean per-sequence component ~= 3000 ms is found: k_seq is ATTRIBUTED. Refound the sim
  (compute = kernel_device_ms + comm + bubble + that-component-per-seq * gbs/dp) and re-run
  analyze_1p7b.py. Then complete the KT set (pp2/pp4/tp4 + clean mbs4/mbs8 re-trace) for the
  full refounded scorecard.
- If the overhead is split across components or not per-sequence-clean: it is a genuine
  fake-PG/pipeline artifact without a single hardware home; treat k_seq as an empirical
  per-sequence rate (validated to ~1% across configs) and document it as such.

## Context
- Kernel decomposition + per-sequence law: experiments/analyze_1p7b_kerneltime.py, memory
  fakepg-overhead-vs-microbatch. KT traces: templates/gpu/gh200_jupiter-1p7b-kerneltime.
- Native scorecard: experiments/analyze_1p7b.py / results_1p7b.txt.
