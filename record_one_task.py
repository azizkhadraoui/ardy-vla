#!/usr/bin/env python
"""Record one success AND one failure of a single task under one configuration, for side-by-side viewing.

Unlike record_compare.py (which contrasts two configurations on the same episode) this contrasts the two
OUTCOMES of the same configuration, so a hard task can be shown honestly: what it looks like when it works and
what it looks like when it does not.

    WORK_DIR=... VIS_MODE=cnn CKPT_SEED=20 TASK=32 TRIES=12 python record_one_task.py
"""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "egl")
import ardy_vla as A
from ardy_vla import log, DEVICE

VARIANT = os.environ.get("VARIANT", "two_stage_goal")
SEED = int(os.environ.get("CKPT_SEED", 20))
TASK = int(os.environ.get("TASK", 32))
TRIES = int(os.environ.get("TRIES", 12))
EXEC = int(os.environ.get("EXEC", 4))
PROJECT = os.environ.get("PROJECT", "1") == "1"
MAX_STEPS = dict(libero_spatial=220, libero_object=280, libero_goal=300, libero_10=520)

D = A.load_data(vision=False)
ck = torch.load(A.CKPT_DIR / f"{VARIANT}_s{SEED}.pt", map_location=DEVICE, weights_only=False)
model = A.HybridDenoiser(D, ck["variant"], d=ck.get("d_model", A.D_MODEL), layers=ck.get("layers", A.LAYERS),
                         w_grip=ck.get("w_grip", 0.0)).to(DEVICE)
model.load_state_dict(ck["state_dict"]); model.eval()
enc = A.OnlineEncoder(D.meta)
info = D.meta["tasks"][TASK]
log(f"task {TASK}: {info['suite']} / {info['language']}")
log(f"checkpoint {VARIANT}_s{SEED} (vis_mode={ck.get('vis_mode', 'feat')}, step {ck.get('step')})")

task = A.LiberoTask(D, TASK, record_cams=("agentview", "frontview", "birdview"))
pol = A.Policy(D, model, A.VARIANTS[VARIANT], enc, TASK, exec_frames=EXEC, project=PROJECT)
want = {True: 1, False: 1}; kept = {}
for init_idx in range(TRIES):
    if not want: break
    pol.seed_base = 1000003 * TASK + 10007 * init_idx
    r = A.run_episode(task, pol, init_idx, max_steps=MAX_STEPS[info["suite"]], record=True)
    ok = bool(r["success"])
    log(f"  init {init_idx}: {'SUCCESS' if ok else 'fail':7s} ({r['steps']} steps)")
    if want.get(ok):
        tag = "success" if ok else "failure"
        d = A.EPI_DIR / f"{tag}_s0"; d.mkdir(parents=True, exist_ok=True)
        w = lambda X: A.to_world(D, TASK, np.asarray(X, np.float32))
        np.savez_compressed(d / f"{info['suite']}_t{TASK}_i{init_idx}_std.npz",
            ee_path=w(r["ee_path"]), grip=r["grip"], success=ok, protocol="std",
            goal_pos=np.array([np.nan] * 3, np.float32), goal_frame=-1,
            box_center=np.array([np.nan] * 3, np.float32),
            plans=w(np.array([p[1] for p in r["plans"]])), plan_t=np.array([p[0] for p in r["plans"]]),
            language=info["language"], **{f"frames_{c}": np.stack(v) for c, v in r["frames"].items() if len(v)})
        kept[tag] = (init_idx, r["steps"]); del want[ok]
        log(f"  -> kept as {tag}")
task.close()
log(f"kept: {kept}")
