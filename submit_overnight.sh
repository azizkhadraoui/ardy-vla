#!/bin/bash
# submit_overnight.sh -- Stages 0-2 of docs/OPTIMIZATION_PLAN.md as one dependency chain.
#
# Every job writes its own result file (TAG), so nothing overwrites the 0.22 baseline or another job's output.
# Stage 0 and Stage 2 are independent and run concurrently; Stage 1's evaluation waits only for its own open-loop
# validation of the fast projection. A job that fails takes its dependents with it (afterok) and leaves the rest.
#
#   bash submit_overnight.sh            # submit everything
#   bash submit_overnight.sh smoke      # only the 5-minute smoke test that exercises every new code path
set -eu
cd "$(dirname "$0")"
MODE="${1:-all}"
mkdir -p jobs

runner () {   # $1 name, $2 time, $3 command
  cat > "jobs/$1.sh" <<EOF
#!/bin/bash -l
#SBATCH -J $1
#SBATCH -o slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem 48000MB
#SBATCH --time $2
set -e
source "\${SLURM_SUBMIT_DIR:-\$PWD}/env.sh"
echo "=== $1 on \$(hostname) ==="
$3
echo "=== DONE $1 ==="
EOF
  chmod +x "jobs/$1.sh"
}

sub () { local n="$1"; shift; local dep="${1:-}"; sbatch ${dep:+--dependency=afterok:$dep} "jobs/$n.sh" | awk '{print $NF}'; }

# ---------------------------------------------------------------- smoke test: every new path, tiny
runner smoke 0:25:00 '
set -x
# 1. the training loop with all Stage-2 corrections, 60 steps, into a throwaway checkpoint
VARIANT=two_stage_goal SEED=99 STEPS=60 FORCE=1 W_GRIP=0.5 BODY_VEL_W=0.3 SNAP_IN_LOSS=1 EMA_DECAY=0.999 \
  VAL_EVERY=30 GOAL_FREE_FRAC=0.5 VIS_FRAME_OFFSET=1 HIST_SHIFT_CM=1.0 $PY -u 03_train.py
# 2. that checkpoint through the closed loop with every Stage-1 knob on, 1 task x 2 inits
VARIANT=two_stage_goal SEED=99 PROTOCOLS=std N_INIT=2 CL_SUITES=libero_spatial TAG=_smoke RECORD=0 \
  GRIP_SRC=head GRIP_GATE_CM=0.5 PROJECT=1 PROJECT_MODE=gn EXEC=4 ENSEMBLE_K=4 SEED_SAMPLER=1 \
  MAXTASKS=1 $PY -u 05_eval_closedloop.py || true
# 3. the Stage-0 ablation paths on the REAL checkpoint, 1 task x 1 init each
VARIANT=two_stage_goal SEED=0 PROTOCOLS=std N_INIT=1 CL_SUITES=libero_spatial TAG=_smoke_oracle RECORD=0 \
  GRIP_SRC=oracle SEED_SAMPLER=1 MAXTASKS=1 $PY -u 05_eval_closedloop.py || true
VARIANT=two_stage_goal SEED=0 PROTOCOLS=std N_INIT=1 CL_SUITES=libero_spatial TAG=_smoke_hist RECORD=0 \
  HIST_SOURCE=demo SEED_SAMPLER=1 MAXTASKS=1 $PY -u 05_eval_closedloop.py || true
# 4. the fast projection against the Adam one, open loop, on the real checkpoint
cp -n $WORK_DIR/results/openloop_two_stage_goal_s0.json $WORK_DIR/results/openloop_two_stage_goal_s0_BASE10.json || true
VARIANT=two_stage_goal SEED=0 FORCE=1 PROJECT_MODE=gn $PY -u 04_eval_openloop.py
mv $WORK_DIR/results/openloop_two_stage_goal_s0.json $WORK_DIR/results/openloop_two_stage_goal_s0_GN.json
cp $WORK_DIR/results/openloop_two_stage_goal_s0_BASE10.json $WORK_DIR/results/openloop_two_stage_goal_s0.json
rm -f $WORK_DIR/ckpt/two_stage_goal_s99.pt
$PY - <<PYEOF
import json, os
b = json.load(open(os.environ["WORK_DIR"] + "/results/openloop_two_stage_goal_s0_BASE10.json"))["metrics"]
g = json.load(open(os.environ["WORK_DIR"] + "/results/openloop_two_stage_goal_s0_GN.json"))["metrics"]
print("SMOKE projection check (Adam 30-step -> Gauss-Newton 3-step):")
for k in ("inwin_goal_fkproj_err_cm", "h8_goal_fkproj_err_cm", "h16_goal_fkproj_err_cm"):
    print(f"  {k:28s} {b[k]:.3f} -> {g[k]:.3f} cm")
print("  steering intact:", g["h8_goal_pos_err_cm"] < 0.6 and g["h16_goal_pos_err_cm"] < 0.7)
PYEOF
'

# ---------------------------------------------------------------- Stage 0
runner s0_probe 1:00:00 '$PY -u s0_probe.py'

runner s0_4step 0:30:00 '
cp -n $WORK_DIR/results/openloop_two_stage_goal_s0.json $WORK_DIR/results/openloop_two_stage_goal_s0_BASE10.json
VARIANT=two_stage_goal SEED=0 FORCE=1 SAMPLE_STEPS=4 $PY -u 04_eval_openloop.py
mv $WORK_DIR/results/openloop_two_stage_goal_s0.json $WORK_DIR/results/openloop_two_stage_goal_s0_4STEP.json
cp $WORK_DIR/results/openloop_two_stage_goal_s0_BASE10.json $WORK_DIR/results/openloop_two_stage_goal_s0.json'

runner s0_kpsweep 3:00:00 '
for KP in 150 300 600; do
  echo "--- JP_KP=$KP ---"
  JP_KP=$KP N_DEMO=3 MODES=a REPLAY_TAG=_kp$KP $PY -u replay_demos.py
done'

runner s0_oracle 2:30:00 '
VARIANT=two_stage_goal SEED=0 PROTOCOLS=std N_INIT=10 CL_SUITES=libero_spatial,libero_object \
  TAG=_s0_oracle RECORD=0 GRIP_SRC=oracle SEED_SAMPLER=1 $PY -u 05_eval_closedloop.py'

runner s0_hist 2:30:00 '
VARIANT=two_stage_goal SEED=0 PROTOCOLS=std N_INIT=10 CL_SUITES=libero_spatial,libero_goal \
  TAG=_s0_hist RECORD=0 HIST_SOURCE=demo SEED_SAMPLER=1 $PY -u 05_eval_closedloop.py'

# a seeded rerun of the baseline configuration: the noise floor every later gate is judged against
runner s0_seeded_base 5:00:00 '
VARIANT=two_stage_goal SEED=0 PROTOCOLS=std N_INIT=20 TAG=_s0_seeded RECORD=0 SEED_SAMPLER=1 $PY -u 05_eval_closedloop.py'

# ---------------------------------------------------------------- Stage 1
runner s1_gripper 5:00:00 '
VARIANT=two_stage_goal SEED=0 PROTOCOLS=std N_INIT=20 TAG=_s1_grip RECORD=0 SEED_SAMPLER=1 \
  GRIP_SRC=hyst GRIP_LO=0.048 GRIP_HI=0.058 GRIP_LATCH=6 GRIP_GATE_CM=0.5 $PY -u 05_eval_closedloop.py'

runner s1_full 7:00:00 '
VARIANT=two_stage_goal SEED=0 PROTOCOLS=std N_INIT=20 TAG=_s1_full RECORD=0 SEED_SAMPLER=1 \
  GRIP_SRC=hyst GRIP_GATE_CM=0.5 PROJECT=1 PROJECT_MODE=gn EXEC=4 ENSEMBLE_K=4 $PY -u 05_eval_closedloop.py'

# ---------------------------------------------------------------- Stage 2
runner s2_train 6:00:00 '
VARIANT=two_stage_goal SEED=10 STEPS=90000 FORCE=1 \
  W_GRIP=0.5 BODY_VEL_W=0.3 SNAP_IN_LOSS=1 EMA_DECAY=0.9995 VAL_EVERY=2000 \
  GOAL_FREE_FRAC=0.5 VIS_FRAME_OFFSET=1 HIST_SHIFT_CM=1.0 $PY -u 03_train.py'

runner s2_openloop 1:00:00 '
VARIANT=two_stage_goal SEED=10 FORCE=1 $PY -u 04_eval_openloop.py'

runner s2_closed 7:00:00 '
VARIANT=two_stage_goal SEED=10 PROTOCOLS=std N_INIT=20 TAG=_s2 RECORD=0 SEED_SAMPLER=1 \
  GRIP_SRC=head GRIP_LATCH=6 GRIP_GATE_CM=0.5 PROJECT=1 PROJECT_MODE=gn EXEC=4 ENSEMBLE_K=4 $PY -u 05_eval_closedloop.py'

if [ "$MODE" = "smoke" ]; then
  s=$(sub smoke); echo "smoke -> $s"; exit 0
fi

# ---------------------------------------------------------------- the DAG
SMOKE=$(sub smoke)
echo "smoke              $SMOKE   (everything below waits on it)"
for j in s0_probe s0_4step s0_kpsweep s0_oracle s0_hist s0_seeded_base s1_gripper s1_full s2_train; do
  id=$(sub "$j" "$SMOKE"); echo "$(printf '%-18s' "$j") $id   after smoke"
  eval "ID_$j=$id"
done
eval "T=\$ID_s2_train"
o=$(sub s2_openloop "$T"); echo "$(printf '%-18s' s2_openloop) $o   after s2_train"
c=$(sub s2_closed "$T");   echo "$(printf '%-18s' s2_closed)   $c   after s2_train"
echo
squeue -u "$USER" -o "%.10i %.16j %.9T %.10M %.12l %.20E"
