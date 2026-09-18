#!/usr/bin/env bash
# Establish the RUN-TO-RUN NOISE FLOOR for the Qwen3-1.7B physical grid.
#
# WHY. Every cell in the scorecard is a SINGLE run, yet we issue +/-5% (GOOD) and +/-10%
# (ACCEPTABLE) verdicts against it. If run-to-run spread is ~1% those verdicts mean
# something; if it is ~6% then "GOOD vs ACCEPTABLE" is noise and several conclusions are
# unfounded. GH200 clock/thermal variation is documented at ~9% ACROSS SESSIONS, so this is
# not a hypothetical worry. Nothing else in the campaign can be trusted at the stated
# precision until this number exists.
#
# Repeats 2 cells x 3 runs (trimmed for limited compute — 6 short jobs). These two bracket
# every verdict, so the third cell was dropped as redundant:
#   tp1pp1-mbs1  the refounding ANCHOR. f is calibrated from it, so ITS noise propagates
#                into every predicted cell — this is the one that cannot be skipped.
#   tp4pp1-mbs1  the worst claimed FAIL (+14.3% span / -12.1% refounded). If its spread is
#                small, the failure is real; if large, the whole TP verdict is unresolved.
# A mid-range GOOD cell (tp1pp1-mbs2) would only confirm noise similar to the anchor at
# nearly identical cost — add it back via CELLS below if the two disagree wildly.
#
# SEQUENTIAL BY DESIGN: running three launchers in parallel previously caused each to
# submit 3 jobs (9 total, 6 wasted). Each invocation here is given ~150s to submit and is
# then killed — the SLURM job it queued keeps running independently.
#
#   bash experiments/submit_1p7b_repeats.sh          # submit
#   bash experiments/submit_1p7b_repeats.sh --dry    # print what would be submitted
set -uo pipefail

AUTOEXP=/e/home/jusers/vanosch1/jupiter/oellm-autoexp
DRY=0
[[ "${1:-}" == "--dry" ]] && DRY=1

cd "$AUTOEXP" || exit 1
# shellcheck disable=SC1091
source jupiter_init.sh >/dev/null 2>&1

# label:tp:mbs   (add "mbs2:1:2" back for a third cell)
CELLS="${CELLS:-anchor:1:1 tp4:4:1}"
REPS="${REPS:-1 2 3}"

for rep in $REPS; do
  for spec in $CELLS; do
    lbl=${spec%%:*}; rest=${spec#*:}; tp=${rest%%:*}; mbs=${rest##*:}
    grp="qwen3-1p7b-rep${rep}"
    echo "=== rep$rep $lbl (tp=$tp mbs=$mbs) -> group $grp"
    cmd=(python scripts/run_autoexp.py
         --config-name experiments/thomas/qwen3_1p7b_val40
         "aux.experiment_group=${grp}"
         "backend.megatron.tensor_model_parallel_size=${tp}"
         "backend.megatron.micro_batch_size=${mbs}"
         'sweep.groups=[]')
    if [[ $DRY -eq 1 ]]; then
      printf '    PYTHONPATH=. %s\n' "${cmd[*]}"
      continue
    fi
    # submit, then stop babysitting: the launcher's monitor loop is not needed here
    PYTHONPATH=. timeout 150 "${cmd[@]}" 2>&1 | grep -E "Submitted job" || \
      echo "    (no submission line seen — check manually)"
  done
done

echo
echo "All submitted. When they finish:"
echo "  PYTHONPATH=. python experiments/analyze_1p7b_noise.py"
