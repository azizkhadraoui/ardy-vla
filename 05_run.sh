#!/bin/bash -l
#SBATCH -J ardy_closedloop
#SBATCH -o slurm-%x-%A_%a.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:v100_16GB:1
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 8:00:00
#SBATCH --array=0-17
# Stage 5: closed-loop LIBERO for one checkpoint per array index (same index map as 03_run.sh).
# Default budget ~2.6 h per checkpoint: std on all suites at N_INIT=20, protocols on libero_spatial+libero_10 at N_INIT_PROTO=10.
# Recording (RECORD=1) only happens for seed 0. PROJECT=1 re-runs the same checkpoint with the consistency projection (separate JSON).
set -e
# under sbatch, BASH_SOURCE points into Slurm's spool dir, not the submit dir -- SLURM_SUBMIT_DIR is where env.sh is
source "${ENV_SH:-${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/env.sh}"
VARIANTS=(two_stage_goal one_stage_goal two_stage_inpaint two_stage_guidance two_stage_nohist two_stage_goal_rollout)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
export VARIANT="${VARIANT:-${VARIANTS[$((IDX / 3))]}}"
export SEED="${SEED:-$((IDX % 3))}"
export N_INIT="${N_INIT:-20}" N_INIT_PROTO="${N_INIT_PROTO:-10}" RECORD="${RECORD:-1}" PROJECT="${PROJECT:-0}"
echo "array $IDX -> $VARIANT seed $SEED  N_INIT=$N_INIT N_INIT_PROTO=$N_INIT_PROTO PROJECT=$PROJECT"
$PY 05_eval_closedloop.py
echo "=== CLOSEDLOOP DONE $VARIANT s$SEED ==="
