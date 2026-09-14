#!/bin/bash -l
#SBATCH -J ardy_tok
#SBATCH -o slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:v100_16GB:1
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 2:00:00
# Stage 2: the FSQ motion tokenizer on all suites, full-episode training. ~30 min. Idempotent.
set -e
source "${ENV_SH:-$(dirname "${BASH_SOURCE[0]}")/env.sh}"
export TOK_STEPS="${TOK_STEPS:-12000}"
$PY 02_tokenizer.py
echo "=== TOKENIZER DONE ==="
