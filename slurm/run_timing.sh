#!/usr/bin/env bash
# =====================================================================================
# Rebuttal Task 1: timing and peak memory for ONE prior on ONE exclusive GPU.
#
# Submit it with submit_timing.sh, which reads cluster.env exactly as submit.sh does:
#
#     bash submit_timing.sh pmf --time 04:00:00
#     bash submit_timing.sh pmf --smoke --time 03:00:00     # 2-image smoke, with checks
#
# The #SBATCH lines below are only fallbacks; submit_timing.sh overrides them with flags.
#
# Use the SAME GPU type for all four priors (A100-SXM4-40GB, as in Table 29); every job
# records the GPU name and scripts/summarize_timing.py warns if they differ.  Each
# configuration runs in its own process (scripts/run_timing.py), so its memory statistics
# belong to it alone.
# =====================================================================================
#SBATCH --account=def-CHANGEME
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/timing-%j.out
#SBATCH --error=logs/timing-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source activate_cluster.sh                # written by setup_cluster.sh

PRIOR="${PRIOR:?set PRIOR=pmf|jit|imf|sit|matched_time|budget_levels|pmf_exact|jitl}"
# The run id is STABLE (not the Slurm job id), so a job resubmitted after a timeout lands in
# the same run directory and scripts/run_timing.py skips every configuration already done.
if [ "${SMOKE:-0}" = "1" ]; then
  CONFIG="configs/experiments_rebuttal_timing_${PRIOR}_smoke.yaml"
  RUN_ID="${TIMING_RUN_ID:-timing_${PRIOR}_smoke}"
  EXTRA=(--with-checks)
  # Task 9: run.py's checks allocate GPU memory inside the measured process, which would
  # void the exact JAX peak, so the pmf_exact smoke runs without them.
  if [ "$PRIOR" = "pmf_exact" ]; then EXTRA=(); fi
else
  CONFIG="configs/experiments_rebuttal_timing_${PRIOR}.yaml"
  RUN_ID="${TIMING_RUN_ID:-timing_${PRIOR}}"
  EXTRA=()
fi

echo "=========================================================================="
echo "timing job $SLURM_JOB_ID prior $PRIOR on $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "config: $CONFIG   run id: $RUN_ID"
echo "=========================================================================="

if [ -n "${SLURM_TMPDIR:-}" ] && [ -d "$MPCFLOW_CACHE_ROOT" ]; then
  cp -r "$MPCFLOW_CACHE_ROOT" "$SLURM_TMPDIR/cache"
  RUN_CACHE="$SLURM_TMPDIR/cache"
  export HF_HOME="$RUN_CACHE/huggingface"
  export TORCH_HOME="$RUN_CACHE/torch"
else
  RUN_CACHE="$MPCFLOW_CACHE_ROOT"
fi

python scripts/run_timing.py --config "$CONFIG" --run-id "$RUN_ID" \
  --cache-root "$RUN_CACHE" --output-root "$MPCFLOW_OUTPUT_ROOT" ${EXTRA[@]+"${EXTRA[@]}"}

echo "summarise all four priors with:"
echo "  python scripts/summarize_timing.py $MPCFLOW_OUTPUT_ROOT/timing_pmf_<id> $MPCFLOW_OUTPUT_ROOT/timing_jit_<id> ..."
