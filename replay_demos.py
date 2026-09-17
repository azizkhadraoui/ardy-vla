#!/usr/bin/env python
"""The actuation ceiling: replay each demo's OWN joint trajectory through the exact closed-loop actuation path.

WHY THIS EXISTS
    Closed-loop success is 0.220 and the plan->achieved tracking error (2.04 cm) is identical in successes and
    failures, so tracking alone does not explain which episodes work. This removes the policy entirely: the demo's
    own joints are pushed through `LiberoTask.step_to` (joint-position PD, kp=JP_KP, +-0.15 rad clamp) and the
    benchmark's own checker decides success. Whatever this does NOT reach is a ceiling no policy change can pass.

MODES (env MODES, default "a,b,c")
    a  the demo's own gripper COMMAND (actions[:, -1]), joint targets as recorded  -- the actuation ceiling
    b  the same, but targeting q[t+LEAD] to pre-empt the controller's ~0.16 s lag  -- does lead compensation help?
    c  the demo's own trajectory with the POLICY's gripper rule (close iff width < gripper_threshold)
       -- the cost of the width-threshold rule alone, with everything else perfect

    WORK_DIR=... [JP_KP=150] [N_DEMO=5] [MODES=a] [REPLAY_TAG=_kp150] python replay_demos.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import ardy_vla as A
from ardy_vla import log

N_DEMO = int(os.environ.get("N_DEMO", 5))
LEAD = int(os.environ.get("LEAD", 3))
MODES = os.environ.get("MODES", "a,b,c").split(",")
TAG = os.environ.get("REPLAY_TAG", "")
MAXTASKS = int(os.environ.get("MAXTASKS", 0))
MAX_STEPS = dict(libero_spatial=220, libero_object=280, libero_goal=300, libero_10=520)

meta = json.load(open(A.DATA / "meta.json")); pr = np.load(A.DATA / "proprio.npz")
q_all, g_all, act = pr["joint_pos"], pr["gripper"], pr["actions"]
ep_start, ep_len, ep_task = pr["episode_start"], pr["episode_len"], pr["episode_task"]
thr = meta["gripper_threshold_m"]
D = A.SimpleNamespace(meta=meta, ep_task=ep_task)

rows = []
tasks = list(enumerate(meta["tasks"]))[: MAXTASKS or None]
for ti, info in tasks:
    task = A.LiberoTask(D, ti); eps = np.where(ep_task == ti)[0]
    res = {m: [] for m in MODES}; fails = []; t0 = time.time()
    for k in range(min(N_DEMO, len(eps))):
        e = int(eps[k]); s, T = int(ep_start[e]), int(ep_len[e])
        q = q_all[s:s + T]; cmd = act[s:s + T, -1] > 0; width = g_all[s:s + T, 0] - g_all[s:s + T, 1]
        for mode in MODES:
            task.reset(k); ok = False
            for t in range(min(T, MAX_STEPS[info["suite"]])):
                tgt = q[min(t + LEAD, T - 1)] if mode == "b" else q[t]
                closed = (width[t] < thr) if mode == "c" else bool(cmd[t])
                _, done, _ = task.step_to(tgt, closed)
                if done: ok = True; break
            res[mode].append(ok)
            if mode == "a" and not ok: fails.append(k)          # demos the actuation path cannot reproduce
    task.close()
    r = {m: float(np.mean(v)) for m, v in res.items()}
    rows.append(dict(task=ti, suite=info["suite"], failed_demos=fails, **r))
    log(f"task {ti:2d} {info['suite']:14s} replay: " + "  ".join(f"{m} {r[m]:.2f}" for m in MODES) + f"   ({time.time()-t0:.0f}s)")

summary = {}
for s in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
    rs = [r for r in rows if r["suite"] == s]
    if rs: summary[s] = {m: float(np.mean([r[m] for r in rs])) for m in MODES}
if rows: summary["overall"] = {m: float(np.mean([r[m] for r in rows])) for m in MODES}
dst = A.RES_DIR / f"replay_ceiling{TAG}.json"
json.dump(dict(n_demo=N_DEMO, lead=LEAD, modes=MODES, jp_kp=float(os.environ.get("JP_KP", 150)), rows=rows, summary=summary), open(dst, "w"), indent=1)
print(f"\nREPLAY CEILING (JP_KP={os.environ.get('JP_KP', 150)}) -> {dst}")
for s, v in summary.items():
    print(f"  {s:16s} " + "  ".join(f"{m} {v[m]:.3f}" for m in MODES))
