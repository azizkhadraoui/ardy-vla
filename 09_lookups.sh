#!/bin/bash
# 09_lookups.sh -- state of the run, no GPU. On the login node:  bash 09_lookups.sh > lookups.txt
set -u
source "${ENV_SH:-$(dirname "${BASH_SOURCE[0]}")/env.sh}" 2>/dev/null || true
W="${WORK_DIR:?set WORK_DIR}"
echo "==================== 1. WHAT EXISTS ===================="
for d in data tokenizer ckpt results figures episodes; do printf "  %-10s %5s files  %s\n" "$d" "$(ls "$W/$d" 2>/dev/null | wc -l)" "$(du -sh "$W/$d" 2>/dev/null | cut -f1)"; done
echo; echo "  checkpoints:"; ls -1 "$W/ckpt" 2>/dev/null | sed 's/^/    /'
echo; echo "  results:";     ls -1 "$W/results" 2>/dev/null | sed 's/^/    /'
echo; echo "==================== 2. MATRIX COVERAGE ===================="
for v in two_stage_goal one_stage_goal two_stage_inpaint two_stage_guidance two_stage_nohist two_stage_goal_rollout; do
  line="  $(printf '%-24s' $v)"
  for s in 0 1 2; do
    c=$([ -f "$W/ckpt/${v}_s$s.pt" ] && echo C || echo .); o=$([ -f "$W/results/openloop_${v}_s$s.json" ] && echo O || echo .); l=$([ -f "$W/results/closedloop_${v}_s$s.json" ] && echo L || echo .)
    line="$line  s$s:$c$o$l"
  done; echo "$line"
done
echo "  (C checkpoint, O open-loop json, L closed-loop json)"
echo; echo "==================== 3. TOKENIZER ===================="
[ -f "$W/tokenizer/tokenizer_eval.json" ] && python -c "import json;e=json.load(open('$W/tokenizer/tokenizer_eval.json'));print('  per-joint deg',e['per_joint_rmse_deg'],'mean',e['mean_rmse_deg'])" || echo "  not trained"
echo; echo "==================== 4. GPU HOURS (sacct, since the pack was compiled) ===================="
sacct -u "$USER" -S 2026-09-14 -X --state=COMPLETED --format=JobName%18,Elapsed,State -n 2>/dev/null | head -60
sacct -u "$USER" -S 2026-09-14 -X --state=COMPLETED --format=Elapsed -n 2>/dev/null | awk -F: '{s+=$1*3600+$2*60+$3} END {printf "  total: %.1f GPU-hours across %d jobs\n", s/3600, NR}'
echo; echo "==================== 5. LAST LINES OF RUNNING / RECENT JOBS ===================="
for f in $(ls -t slurm-*.out 2>/dev/null | head -8); do echo "  --- $f"; tail -2 "$f" | sed 's/^/      /'; done
