#!/usr/bin/env bash
# =====================================================================================
# Submit the rebuttal timing run (Task 1) for ONE prior, without editing a tracked file.
#
#     bash submit_timing.sh pmf --time 04:00:00
#     bash submit_timing.sh pmf --smoke --time 03:00:00   # 2-image smoke variant, with checks
#     bash submit_timing.sh sit --time 12:00:00 --dry-run # print the sbatch command only
#     bash submit_timing.sh matched_time --gpu h100:1 --time 03:00:00
#         # Task 5: configs/experiments_rebuttal_timing_matched_time.yaml (pMF and JiT in one
#         # job, one model-only baseline process per prior), run id timing_matched_time
#     bash submit_timing.sh budget_levels --gpu h100:1 --time 03:00:00
#         # Task 6: configs/experiments_rebuttal_timing_budget_levels.yaml (cheap and mid
#         # budget levels, pMF and JiT in one job), run id timing_budget_levels
#     bash submit_timing.sh pmf_exact --gpu h100:1 --time 06:00:00
#         # Task 9: pMF-L/16 with exact JAX peak memory (param_loading: host_once), run id
#         # timing_pmf_exact; its --smoke variant runs WITHOUT run.py's checks
#     bash submit_timing.sh jitl --gpu h100:1 --time 06:00:00
#         # Task 9: JiT-L/16, every method on all five tasks, run id timing_jitl
#
# Account, memory and CPUs come from `cluster.env` exactly as in submit.sh (account
# auto-detected from ~/projects when there is exactly one).  The GPU is NOT taken from
# MPCFLOW_GPU: Table 29 was measured on an A100-SXM4-40GB, so this defaults to `a100:1`
# (Narval).  Override with MPCFLOW_TIMING_GPU in cluster.env or --gpu.  The time limit is
# --time, else MPCFLOW_TIMING_TIME, else MPCFLOW_TIME; get a value per prior from
#     python scripts/run_timing.py --config configs/experiments_rebuttal_timing_<prior>.yaml \
#         --run-id timing_<prior> --estimate
# The run id is stable (timing_<prior>, or timing_<prior>_smoke), so resubmitting the same
# command after a timeout resumes: completed configurations are skipped.
# =====================================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
[ -f cluster.env ] && . ./cluster.env

PRIOR="${1:-}"
case "$PRIOR" in
  pmf|jit|imf|sit|matched_time|budget_levels|pmf_exact|jitl) shift ;;
  *) echo "usage: bash submit_timing.sh pmf|jit|imf|sit|matched_time|budget_levels|pmf_exact|jitl [--smoke] [--time HH:MM:SS] [--gpu G]"
     echo "       [--account A] [--mem M] [--cpus N] [--run-id ID] [--dry-run]"
     exit 2 ;;
esac

ACCOUNT="${MPCFLOW_ACCOUNT:-}"
GPU="${MPCFLOW_TIMING_GPU:-a100:1}"
TIME="${MPCFLOW_TIMING_TIME:-${MPCFLOW_TIME:-03:00:00}}"
MEM="${MPCFLOW_MEM:-64G}"
CPUS="${MPCFLOW_CPUS:-8}"
SMOKE=0
DRY=0
RUN_ID=""
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke)   SMOKE=1; shift ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --gpu)     GPU="$2";     shift 2 ;;
    --time)    TIME="$2";    shift 2 ;;
    --mem)     MEM="$2";     shift 2 ;;
    --cpus)    CPUS="$2";    shift 2 ;;
    --run-id)  RUN_ID="$2";  shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done
if [ -z "$RUN_ID" ]; then
  RUN_ID="timing_${PRIOR}"
  [ $SMOKE -eq 1 ] && RUN_ID="timing_${PRIOR}_smoke"
fi

if [ -z "$ACCOUNT" ] && [ -d "$HOME/projects" ]; then
  mapfile -t found < <(ls -1 "$HOME/projects" 2>/dev/null | grep -E '^(def|rrg|ctb)-')
  [ "${#found[@]}" -eq 1 ] && ACCOUNT="${found[0]}"
fi
if [ -z "$ACCOUNT" ]; then
  echo "ERROR: no allocation account. Set MPCFLOW_ACCOUNT in cluster.env or pass --account."
  exit 2
fi
case "$PRIOR:$GPU" in
  pmf_exact:*|jitl:*)
     # Task 9: the main-paper table needs both checkpoints on ONE GPU type (H100 80GB);
     # summarize_timing.py --main-table refuses mixed GPU types.
     ;;
  *:a100*) ;;
  *) echo "WARNING: GPU '$GPU' is not an A100; Table 29 used A100-SXM4-40GB, and the summary"
     echo "         warns when the priors ran on different GPU types." ;;
esac

CMD=(sbatch --account="$ACCOUNT" --gpus-per-node="$GPU" --cpus-per-task="$CPUS"
     --mem="$MEM" --time="$TIME" --job-name="timing_${PRIOR}"
     --export=ALL,PRIOR="$PRIOR",SMOKE="$SMOKE",TIMING_RUN_ID="$RUN_ID"
     ${EXTRA[@]+"${EXTRA[@]}"} slurm/run_timing.sh)

echo "cluster : ${CC_CLUSTER:-unknown}"
echo "account : $ACCOUNT"
echo "gpu     : $GPU        cpus: $CPUS   mem: $MEM   time: $TIME"
echo "prior   : $PRIOR   smoke: $SMOKE   run id: $RUN_ID (stable: resubmit to resume)"
echo "command : ${CMD[*]}"
echo
if [ $DRY -eq 1 ]; then
  echo "--dry-run: nothing submitted."
  exit 0
fi
mkdir -p logs
"${CMD[@]}"
