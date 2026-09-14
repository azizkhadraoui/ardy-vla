#!/bin/bash -l
#SBATCH -J ardy_openloop
#SBATCH -o slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:v100_16GB:1
#SBATCH -c 8
#SBATCH --mem 48000MB
#SBATCH --time 4:00:00
# Stage 4: open-loop adherence vs horizon for every checkpoint present (~8 min each, 18 ckpts ~2.5 h). Idempotent per checkpoint.
set -e
source "${ENV_SH:-$(dirname "${BASH_SOURCE[0]}")/env.sh}"
$PY 04_eval_openloop.py
echo "=== OPENLOOP DONE ==="
