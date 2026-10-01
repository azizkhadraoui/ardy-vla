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

    OSC MODES (need CONTROLLER=OSC_POSE; the space the demos were actually teleoperated in)
    d  the demo's own recorded 7-D action verbatim -- the purest possible replay. Whatever this does not reach
       is a property of the simulator and the saved init states, not of any controller choice we made.
    e  a pose-tracking OSC controller driven by the demo's EE trajectory, anchored at the arm's real starting
       pose and following the demo's DISPLACEMENTS -- which is what our policy's explicit stream could emit.
       Position deltas are frame-invariant here (the LIBERO bases are translations of the world frame), so this
       needs no FK fit and inherits none of its error.

    The point of d and e: joint-position replay tops out at 0.592 while the best policy scores 0.624, i.e. the
    policy is already at the ceiling of ITS actuation path. If OSC replay is far higher, the controller -- not
    the architecture -- is what stands between this model and competitive success.

    WORK_DIR=... [JP_KP=150] [N_DEMO=5] [MODES=a] [REPLAY_TAG=_kp150] python replay_demos.py
    WORK_DIR=... CONTROLLER=OSC_POSE MODES=d,e N_DEMO=5 REPLAY_TAG=_osc python replay_demos.py
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

CONTROLLER = os.environ.get("CONTROLLER", "JOINT_POSITION")
OSC_POS_SCALE = float(os.environ.get("OSC_POS_SCALE", 0.05))   # robosuite OSC_POSE default output_max, metres
OSC_ROT_SCALE = float(os.environ.get("OSC_ROT_SCALE", 0.5))    # ...and radians
if any(m in ("d", "e") for m in MODES) and CONTROLLER != "OSC_POSE":
    raise SystemExit("modes d and e need CONTROLLER=OSC_POSE")

meta = json.load(open(A.DATA / "meta.json")); pr = np.load(A.DATA / "proprio.npz")
q_all, g_all, act = pr["joint_pos"], pr["gripper"], pr["actions"]
ee_all, r6_all = pr["ee_pos"], pr["ee_rot6d"]
ep_start, ep_len, ep_task = pr["episode_start"], pr["episode_len"], pr["episode_task"]
thr = meta["gripper_threshold_m"]
D = A.SimpleNamespace(meta=meta, ep_task=ep_task)


def rot6d_to_mat(v):
    a, b = v[:3], v[3:6]
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b - a * (a @ b); b = b / (np.linalg.norm(b) + 1e-8)
    return np.stack([a, b, np.cross(a, b)], -1)


def quat_to_mat(q):
    """robosuite reports eef orientation as xyzw."""
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def axis_angle(R):
    """Rotation matrix -> axis-angle vector, the increment OSC_POSE expects."""
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0); th = float(np.arccos(c))
    if th < 1e-6: return np.zeros(3)
    return th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


rows = []
tasks = list(enumerate(meta["tasks"]))[: MAXTASKS or None]
for ti, info in tasks:
    task = A.LiberoTask(D, ti); eps = np.where(ep_task == ti)[0]
    res = {m: [] for m in MODES}; fails = []; t0 = time.time()
    for k in range(min(N_DEMO, len(eps))):
        e = int(eps[k]); s, T = int(ep_start[e]), int(ep_len[e])
        q = q_all[s:s + T]; cmd = act[s:s + T, -1] > 0; width = g_all[s:s + T, 0] - g_all[s:s + T, 1]
        for mode in MODES:
            obs = task.reset(k); ok = False
            if mode in ("d", "e"):
                p0 = np.asarray(obs["robot0_eef_pos"], np.float64)      # anchor on where the arm ACTUALLY starts
                R0_act = quat_to_mat(np.asarray(obs["robot0_eef_quat"], np.float64))
                dp = ee_all[s:s + T].astype(np.float64); dr = r6_all[s:s + T].astype(np.float64)
                R0_demo = rot6d_to_mat(dr[0])
                for t in range(min(T, MAX_STEPS[info["suite"]])):
                    if mode == "d":
                        a = act[s + t].astype(np.float64)               # verbatim: exactly what was teleoperated
                    else:
                        tgt_p = p0 + (dp[t] - dp[0])                    # demo displacement from its own start
                        now_p = np.asarray(obs["robot0_eef_pos"], np.float64)
                        ap = np.clip((tgt_p - now_p) / OSC_POS_SCALE, -1, 1)
                        # OSC takes an increment on the CURRENT orientation, so compose the demo's rotation-from-
                        # its-own-start onto the arm's real starting pose, then ask for the remaining delta.
                        R_tgt = (rot6d_to_mat(dr[t]) @ R0_demo.T) @ R0_act
                        R_now = quat_to_mat(np.asarray(obs["robot0_eef_quat"], np.float64))
                        ar = np.clip(axis_angle(R_tgt @ R_now.T) / OSC_ROT_SCALE, -1, 1)
                        a = np.concatenate([ap, ar, [1.0 if bool(cmd[t]) else -1.0]])
                    obs, done, _ = task.step_raw(a)
                    if done: ok = True; break
            else:
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
json.dump(dict(n_demo=N_DEMO, lead=LEAD, modes=MODES, controller=CONTROLLER, jp_kp=float(os.environ.get("JP_KP", 150)), rows=rows, summary=summary), open(dst, "w"), indent=1)
print(f"\nREPLAY CEILING (JP_KP={os.environ.get('JP_KP', 150)}) -> {dst}")
for s, v in summary.items():
    print(f"  {s:16s} " + "  ".join(f"{m} {v[m]:.3f}" for m in MODES))
