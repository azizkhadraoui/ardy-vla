#!/bin/bash -l
#SBATCH -J ardy_data
#SBATCH -o slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:v100_16GB:1
#SBATCH -c 8
#SBATCH --mem 48000MB
#SBATCH --time 4:00:00
# Stage 1: download the four LIBERO suites (~25 GB), extract proprio, define the FK explicit stream,
# run the frozen DINOv2 over ~250k frames x 2 cameras, embed the 40 task strings. Idempotent (skips if done).
set -e
# under sbatch, BASH_SOURCE points into Slurm's spool dir, not the submit dir -- SLURM_SUBMIT_DIR is where env.sh is
source "${ENV_SH:-${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/env.sh}"
$PY 01_prepare_data.py
echo "=== DATA DONE ==="
