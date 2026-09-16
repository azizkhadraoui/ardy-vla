#!/bin/bash
# submit_all.sh -- the whole experiment as a dependency chain, or one stage at a time.
#
#   bash submit_all.sh check      files present, env.sh sourced, python imports OK. Run this first.
#   bash submit_all.sh data       01 (download + features)          1 GPU, ~1.5 h
#   bash submit_all.sh tokenizer  02                                 1 GPU, ~0.5 h
#   bash submit_all.sh train      03 array 0-17                      18 x ~40 min; 2 GPUs -> ~6 h wall
#   bash submit_all.sh scale      03 for the d=512 point             1 GPU, ~2 h
#   bash submit_all.sh eval       04 + 05 array 0-17 + 06            04 ~2.5 h, 05 18 x ~2.6 h -> ~24 h wall on 2 GPUs, 06 minutes
#   bash submit_all.sh chain      01 -> 02 -> 03 -> (04, 05, 06)     everything, with afterok dependencies
#   bash submit_all.sh gate       01 -> 02 -> two_stage_goal s0 -> 04+05 on it alone  (~5 h; decides whether the rest is worth running)
#
# While jobs run, on the login node:   bash 09_lookups.sh > lookups.txt
# When 04/05 have landed:               $PY 07_aggregate.py   then   $PY 08_visualize.py
set -eu
MODE="${1:-check}"
cd "$(dirname "$0")"
sub () { local dep="${2:-}"; local extra="${3:-}"; echo -n "  $1 ${extra} -> "; sbatch ${dep:+--dependency=afterok:$dep} $extra "$1" | awk '{print $NF}'; }

case "$MODE" in
  check)
    for f in ardy_vla.py 01_prepare_data.py 02_tokenizer.py 03_train.py 04_eval_openloop.py 05_eval_closedloop.py 06_latency.py 07_aggregate.py 08_visualize.py env.sh; do
      [ -f "$f" ] || { echo "missing: $f $( [ $f = env.sh ] && echo '(cp env.sh.example env.sh and edit)')"; exit 1; }; done
    source env.sh
    $PY - <<'PYEOF'
import importlib, sys
for m in ("torch", "transformers", "h5py", "huggingface_hub", "imageio", "matplotlib", "PIL"):
    try: importlib.import_module(m); print(f"  ok  {m}")
    except Exception as e: print(f"  MISSING {m}: {e}")
for m in ("mujoco", "robosuite", "bddl", "easydict", "gym"):
    try: mod = importlib.import_module(m); print(f"  ok  {m} {getattr(mod, '__version__', '')}")
    except Exception as e: print(f"  MISSING {m} (needed by 05/08 only): {e}")
import torch; print(f"  cuda {torch.cuda.is_available()}  bf16 not needed (fp16 AMP)")
import mujoco; v = tuple(int(x) for x in mujoco.__version__.split('.')[:2]); print(f"  mujoco {mujoco.__version__}: {'OK for robosuite 1.4' if v < (3,3) else '!! >= 3.3 removes MjData.qM; pip install mujoco==3.1.6'}")
PYEOF
    if [ "${WANDB:-0}" = "1" ]; then
      $PY -c "import wandb, os; print('  ok  wandb ' + wandb.__version__ + '  project=' + str(os.environ.get('WANDB_PROJECT')) + '  mode=' + os.environ.get('WANDB_MODE', 'online'))" ||
        echo "  MISSING wandb but WANDB=1 (pip install wandb, or set WANDB=0) -- stages log nothing and run on"
    else echo "  wandb off (WANDB=0)"; fi
    echo "  WORK_DIR=$WORK_DIR  DATA_DIR=$DATA_DIR  LIBERO_DIR=$LIBERO_DIR  PRESET=$PRESET"
    ;;
  data)      sub 01_run.sh ;;
  tokenizer) sub 02_run.sh ;;
  train)     sub 03_run.sh ;;
  scale)     PRESET=scale VARIANT=two_stage_goal SEED=0 sub 03_run.sh "" "--array=0 --time 8:00:00 -J ardy_train_scale" ;;
  eval)      sub 04_run.sh; sub 05_run.sh; sub 06_run.sh ;;
  chain)
    d=$(sbatch 01_run.sh | awk '{print $NF}'); echo "  01 -> $d"
    t=$(sbatch --dependency=afterok:$d 02_run.sh | awk '{print $NF}'); echo "  02 -> $t (after $d)"
    a=$(sbatch --dependency=afterok:$t 03_run.sh | awk '{print $NF}'); echo "  03 array -> $a (after $t)"
    for f in 04_run.sh 05_run.sh 06_run.sh; do id=$(sbatch --dependency=afterok:$a "$f" | awk '{print $NF}'); echo "  $f -> $id (after array $a)"; done
    echo; echo "07/08 are run by hand once the evals have landed."
    ;;
  gate)
    d=$(sbatch 01_run.sh | awk '{print $NF}'); t=$(sbatch --dependency=afterok:$d 02_run.sh | awk '{print $NF}')
    a=$(VARIANT=two_stage_goal SEED=0 sbatch --dependency=afterok:$t --array=0 03_run.sh | awk '{print $NF}')
    o=$(VARIANT=two_stage_goal SEED=0 sbatch --dependency=afterok:$a 04_run.sh | awk '{print $NF}')
    c=$(VARIANT=two_stage_goal SEED=0 sbatch --dependency=afterok:$a --array=0 05_run.sh | awk '{print $NF}')
    echo "  01 $d -> 02 $t -> 03[two_stage_goal s0] $a -> 04 $o, 05 $c"
    echo; echo "When it lands, check in order:"
    echo "  1. tokenizer_eval.json: mean < 1 deg, no joint > 1.5 -- else nothing downstream is trustworthy"
    echo "  2. openloop_two_stage_goal_s0.json: inwin_goal_pos_err_cm ~< 1, h8_goal_pos_err_cm ~< 1, h8_goal_fk_err_cm well below h8_nogoal_fk_err_cm"
    echo "  3. closedloop_two_stage_goal_s0.json summary.all.std.success -- if this is < 0.5 the controller/perception loop needs debugging before spending 40 GPU-hours"
    echo "Then: bash submit_all.sh train ; bash submit_all.sh eval"
    ;;
  *) echo "usage: bash submit_all.sh [check|data|tokenizer|train|scale|eval|chain|gate]"; exit 1 ;;
esac
echo; echo "watch: squeue -u \$USER     state: bash 09_lookups.sh"
