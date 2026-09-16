#!/bin/bash -l
#SBATCH -J ardy_latency
#SBATCH -o slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:v100_16GB:1
#SBATCH -c 4
#SBATCH --mem 16000MB
#SBATCH --time 0:30:00
# Stage 6: per-window latency of every variant on the allocated GPU. Minutes.
set -e
# under sbatch, BASH_SOURCE points into Slurm's spool dir, not the submit dir -- SLURM_SUBMIT_DIR is where env.sh is
source "${ENV_SH:-${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/env.sh}"
$PY 06_latency.py
echo "=== LATENCY DONE ==="
