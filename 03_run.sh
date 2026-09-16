#!/bin/bash -l
#SBATCH -J ardy_train
#SBATCH -o slurm-%x-%A_%a.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:v100_16GB:1
#SBATCH -c 8
#SBATCH --mem 48000MB
#SBATCH --time 6:00:00
#SBATCH --array=0-17
# Stage 3: the training matrix as a SLURM array. Index -> (variant, seed):
#   0-2 two_stage_goal s0-2 | 3-5 one_stage_goal | 6-8 two_stage_inpaint | 9-11 two_stage_guidance | 12-14 two_stage_nohist | 15-17 two_stage_goal_rollout
# ~40 min each at PRESET=long on a V100. Pin one job:  VARIANT=two_stage_goal SEED=0 sbatch --array=0 03_run.sh
# The scale point:  PRESET=scale VARIANT=two_stage_goal SEED=0 sbatch --array=0 --time 8:00:00 03_run.sh
set -e
# under sbatch, BASH_SOURCE points into Slurm's spool dir, not the submit dir -- SLURM_SUBMIT_DIR is where env.sh is
source "${ENV_SH:-${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/env.sh}"
VARIANTS=(two_stage_goal one_stage_goal two_stage_inpaint two_stage_guidance two_stage_nohist two_stage_goal_rollout)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
export VARIANT="${VARIANT:-${VARIANTS[$((IDX / 3))]}}"
export SEED="${SEED:-$((IDX % 3))}"
echo "array $IDX -> $VARIANT seed $SEED preset $PRESET"
$PY 03_train.py
echo "=== TRAIN DONE $VARIANT s$SEED ==="
