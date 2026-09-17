#!/bin/bash
# submit_overnight.sh -- Stages 0-2 of docs/OPTIMIZATION_PLAN.md as one dependency chain.
#
# Every job writes its own tagged result file, so nothing overwrites the 0.220 baseline or another job's output,
# and no job mutates a shared file in place. Stage 0, Stage 1 and Stage 2 are independent and run concurrently.
# A job that fails takes its dependents with it (afterok) and leaves the rest alone.
#
#   bash submit_overnight.sh                  # smoke test, then the whole DAG behind it
#   bash submit_overnight.sh smoke            # only the gate
#   bash submit_overnight.sh after <jobid>    # DAG behind an already-running smoke test ("after none" if it passed)
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

# refuse to queue a second copy of a job that is already pending or running
sub () {
  local n="$1"; shift; local dep="${1:-}"
  # return the EXISTING id, not a word: the caller feeds this straight into --dependency=afterok:
  local have; have=$(squeue -u "$USER" -h -n "$n" -o "%i %T" 2>/dev/null | awk '$2=="PENDING"||$2=="RUNNING"{print $1; exit}')
  if [ -n "$have" ]; then echo "$have"; return 0; fi
  sbatch ${dep:+--dependency=afterok:$dep} "jobs/$n.sh" | awk '{print $NF}'
}

# ---------------------------------------------------------------- the gate
# Every code path the night depends on, small, and it FAILS if any of them fails: the point of a gate is to stop
# the DAG, so nothing here is allowed to swallow an error.
runner smoke 0:45:00 '
rc=0
run () { echo "--- SMOKE: $* ---"; if ! "$@"; then echo "SMOKE STEP FAILED: $*"; rc=1; fi; }

# 1. the training loop with every Stage-2 correction, 60 steps, into a throwaway checkpoint
run env VARIANT=two_stage_goal SEED=99 STEPS=60 FORCE=1 W_GRIP=0.1 BODY_VEL_W=0.3 SNAP_IN_LOSS=1 EMA_DECAY=0.999 \
  VAL_EVERY=30 GOAL_FREE_FRAC=0.5 VIS_FRAME_OFFSET=1 HIST_SHIFT_CM=1.0 $PY -u 03_train.py

# 2. that checkpoint through the closed loop with the Stage-2 configuration (trained gripper head)
run env VARIANT=two_stage_goal SEED=99 PROTOCOLS=std N_INIT=2 CL_SUITES=libero_spatial TAG=_smoke FORCE=1 RECORD=0 \
  GRIP_SRC=head GRIP_GATE_CM=0.5 PROJECT=1 PROJECT_MODE=gn EXEC=4 ENSEMBLE_K=4 SEED_SAMPLER=1 MAXTASKS=1 $PY -u 05_eval_closedloop.py

# 3. the Stage-1 configuration -- the hysteresis state machine and the tracking-gated close, which two
#    multi-hour jobs depend on and which nothing else here exercises
run env VARIANT=two_stage_goal SEED=0 PROTOCOLS=std N_INIT=2 CL_SUITES=libero_spatial TAG=_smoke_hyst FORCE=1 RECORD=0 \
  GRIP_SRC=hyst GRIP_LO=0.048 GRIP_HI=0.058 GRIP_LATCH=6 GRIP_GATE_CM=0.5 PROJECT=1 PROJECT_MODE=gn EXEC=4 \
  ENSEMBLE_K=4 SEED_SAMPLER=1 MAXTASKS=1 $PY -u 05_eval_closedloop.py

# 4. the two Stage-0 ablation paths
run env VARIANT=two_stage_goal SEED=0 PROTOCOLS=std N_INIT=1 CL_SUITES=libero_spatial TAG=_smoke_oracle FORCE=1 RECORD=0 \
  GRIP_SRC=oracle SEED_SAMPLER=1 MAXTASKS=1 $PY -u 05_eval_closedloop.py
run env VARIANT=two_stage_goal SEED=0 PROTOCOLS=std N_INIT=1 CL_SUITES=libero_spatial TAG=_smoke_hist FORCE=1 RECORD=0 \
  HIST_SOURCE=demo SEED_SAMPLER=1 MAXTASKS=1 $PY -u 05_eval_closedloop.py

# 5. the fast projection against the 30-step Adam one, open loop. RES_TAG keeps it out of the canonical file.
run env VARIANT=two_stage_goal SEED=0 FORCE=1 PROJECT_MODE=gn RES_TAG=_GN $PY -u 04_eval_openloop.py
rm -f $WORK_DIR/ckpt/two_stage_goal_s99.pt $WORK_DIR/ckpt/two_stage_goal_s99.tmp
$PY - <<PYEOF || rc=1
import json, os
R = os.environ["WORK_DIR"] + "/results/"
b = json.load(open(R + "openloop_two_stage_goal_s0.json"))["metrics"]
g = json.load(open(R + "openloop_two_stage_goal_s0_GN.json"))["metrics"]
print("projection, 30-step Adam -> 3-step Gauss-Newton:")
for k in ("inwin_goal_fkproj_err_cm", "h8_goal_fkproj_err_cm", "h16_goal_fkproj_err_cm"):
    print(f"  {k:28s} {b[k]:.3f} -> {g[k]:.3f} cm")
    assert g[k] <= b[k] + 0.15, f"{k}: the fast projection is materially worse ({g[k]:.3f} vs {b[k]:.3f})"
assert g["h8_goal_pos_err_cm"] < 0.6 and g["h16_goal_pos_err_cm"] < 0.7, "steering regressed"
print("  steering intact, projection within tolerance")
PYEOF
if [ $rc -ne 0 ]; then echo "SMOKE FAILED"; exit 1; fi
echo "SMOKE PASSED"
'

# ---------------------------------------------------------------- Stage 0: diagnostics
runner s0_probe 1:30:00 '$PY -u s0_probe.py'

runner s0_4step 0:30:00 '
env VARIANT=two_stage_goal SEED=0 FORCE=1 SAMPLE_STEPS=4 RES_TAG=_4STEP $PY -u 04_eval_openloop.py'

runner s0_kpsweep 3:00:00 '
for KP in 150 300 600; do
  echo "--- JP_KP=$KP ---"
  env JP_KP=$KP N_DEMO=3 MODES=a REPLAY_TAG=_kp$KP $PY -u replay_demos.py
done'

runner s0_oracle 3:00:00 '
env VARIANT=two_stage_goal SEED=0 PROTOCOLS=std N_INIT=10 CL_SUITES=libero_spatial,libero_object \
  TAG=_s0_oracle RECORD=0 GRIP_SRC=oracle SEED_SAMPLER=1 $PY -u 05_eval_closedloop.py'

runner s0_hist 3:00:00 '
env VARIANT=two_stage_goal SEED=0 PROTOCOLS=std N_INIT=10 CL_SUITES=libero_spatial,libero_goal \
  TAG=_s0_hist RECORD=0 HIST_SOURCE=demo SEED_SAMPLER=1 $PY -u 05_eval_closedloop.py'

# the seeded baseline: the control every Stage-1 number below is compared against, same seeds, same inits
runner s0_seeded_base 6:00:00 '
env VARIANT=two_stage_goal SEED=0 PROTOCOLS=std N_INIT=20 TAG=_s0_seeded RECORD=0 SEED_SAMPLER=1 $PY -u 05_eval_closedloop.py'

# ---------------------------------------------------------------- Stage 1: inference-only, one change at a time
runner s1_gripper 6:00:00 '
env VARIANT=two_stage_goal SEED=0 PROTOCOLS=std N_INIT=20 TAG=_s1_grip RECORD=0 SEED_SAMPLER=1 \
  GRIP_SRC=hyst GRIP_LO=0.048 GRIP_HI=0.058 GRIP_LATCH=6 GRIP_GATE_CM=0.5 $PY -u 05_eval_closedloop.py'

# the projection alone, so a win for s1_full can be attributed: this differs from s1_gripper by ONE change
runner s1_proj 6:00:00 '
env VARIANT=two_stage_goal SEED=0 PROTOCOLS=std N_INIT=20 TAG=_s1_proj RECORD=0 SEED_SAMPLER=1 \
  GRIP_SRC=hyst GRIP_GATE_CM=0.5 PROJECT=1 PROJECT_MODE=gn $PY -u 05_eval_closedloop.py'

runner s1_full 8:00:00 '
env VARIANT=two_stage_goal SEED=0 PROTOCOLS=std N_INIT=20 TAG=_s1_full RECORD=0 SEED_SAMPLER=1 \
  GRIP_SRC=hyst GRIP_GATE_CM=0.5 PROJECT=1 PROJECT_MODE=gn EXEC=4 ENSEMBLE_K=4 $PY -u 05_eval_closedloop.py'

# ---------------------------------------------------------------- Stage 2: the corrected retrain
runner s2_train 8:00:00 '
env VARIANT=two_stage_goal SEED=10 STEPS=90000 FORCE=1 \
  W_GRIP=0.1 BODY_VEL_W=0.3 SNAP_IN_LOSS=1 EMA_DECAY=0.9995 VAL_EVERY=2000 \
  GOAL_FREE_FRAC=0.5 VIS_FRAME_OFFSET=1 HIST_SHIFT_CM=1.0 $PY -u 03_train.py'

runner s2_openloop 1:00:00 '
env VARIANT=two_stage_goal SEED=10 FORCE=1 $PY -u 04_eval_openloop.py'

# the same inference configuration as s1_full, so s2_closed vs s1_full isolates the retrain
runner s2_closed 8:00:00 '
env VARIANT=two_stage_goal SEED=10 PROTOCOLS=std N_INIT=20 TAG=_s2 RECORD=0 SEED_SAMPLER=1 \
  GRIP_SRC=head GRIP_LATCH=6 GRIP_GATE_CM=0.5 PROJECT=1 PROJECT_MODE=gn EXEC=4 ENSEMBLE_K=4 $PY -u 05_eval_closedloop.py'

if [ "$MODE" = "smoke" ]; then
  echo "smoke -> $(sub smoke)"; exit 0
fi

# ---------------------------------------------------------------- the DAG
# "after <jobid>" hangs the DAG off a smoke test already running or already passed, so a launch interrupted by a
# dropped connection does not have to re-run the gate. sub() refuses to queue a job that is already there.
if [ "$MODE" = "after" ]; then
  SMOKE="${2:?usage: submit_overnight.sh after <smoke-jobid>|none}"
  [ "$SMOKE" = "none" ] && SMOKE=""
  echo "DAG behind smoke ${SMOKE:-<already passed>}"
else
  SMOKE=$(sub smoke); echo "$(printf '%-18s' smoke) $SMOKE"
fi
for j in s0_probe s0_4step s0_kpsweep s0_oracle s0_hist s0_seeded_base s1_gripper s1_proj s1_full s2_train; do
  id=$(sub "$j" "$SMOKE"); echo "$(printf '%-18s' "$j") $id"
  eval "ID_$j=\$id"
done
eval "T=\$ID_s2_train"
echo "$(printf '%-18s' s2_openloop) $(sub s2_openloop "$T")"
echo "$(printf '%-18s' s2_closed)   $(sub s2_closed "$T")"
echo
squeue -u "$USER" -o "%.10i %.16j %.9T %.10M %.12l %.20E"
