#!/usr/bin/env python
"""Record the SAME task and the SAME initial state under the baseline configuration and the best one, so the
difference on screen is the configuration and nothing else.

    baseline : the 0.220 setup -- width-threshold gripper, joint targets from the decoded FSQ latent, replan
               every 0.4 s, fresh noise each time.
    best     : the 0.448 setup -- hysteresis gripper, explicit stream executed through the Gauss-Newton
               projection, replan every 0.2 s with temporal ensembling over the last 4 plans.

Both write episodes 08_visualize.py can render side by side (VIS_VARIANTS=base,best).

    WORK_DIR=... TASKS=30,11,20 N_INIT=6 python record_compare.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch
import ardy_vla as A
from ardy_vla import log, DEVICE

VARIANT, SEED = "two_stage_goal", 0
TASKS = [int(x) for x in os.environ.get("TASKS", "30,11,20").split(",")]
N_INIT = int(os.environ.get("N_INIT", 6))
WANT = int(os.environ.get("WANT", 2))          # demonstrative pairs to keep per task
MAX_STEPS = dict(libero_spatial=220, libero_object=280, libero_goal=300, libero_10=520)

D = A.load_data(vision=False)
ck = torch.load(A.CKPT_DIR / f"{VARIANT}_s{SEED}.pt", map_location=DEVICE, weights_only=False)
model = A.HybridDenoiser(D, ck["variant"], d=ck.get("d_model", A.D_MODEL), layers=ck.get("layers", A.LAYERS),
                         w_grip=ck.get("w_grip", 0.0)).to(DEVICE)
model.load_state_dict(ck["state_dict"]); model.eval()
enc = A.OnlineEncoder(D.meta)
vcfg = A.VARIANTS[VARIANT]

CONFIGS = {
    "base": dict(grip="width", gate=0.0, project=False, exec_frames=8, ens=1, ctx=0),
    "best": dict(grip="hyst", gate=0.0, project=True, exec_frames=4, ens=4, ctx=0),
}


def apply(cfg):
    """The policy reads these as module globals at call time, so one process can run both configurations."""
    A.GRIP_SRC, A.GRIP_GATE_CM, A.ENSEMBLE_K, A.DECODE_CTX = cfg["grip"], cfg["gate"], cfg["ens"], cfg["ctx"]
    A.PROJECT_MODE = "gn"


kept = 0
for ti in TASKS:
    info = D.meta["tasks"][ti]
    task = A.LiberoTask(D, ti, record_cams=("agentview", "frontview", "birdview"))
    got = 0
    for init_idx in range(N_INIT):
        if got >= WANT: break
        runs = {}
        for name, cfg in CONFIGS.items():
            apply(cfg)
            pol = A.Policy(D, model, vcfg, enc, ti, exec_frames=cfg["exec_frames"], project=cfg["project"])
            pol.seed_base = 1000003 * ti + 10007 * init_idx        # same noise for both, so it is not luck
            runs[name] = A.run_episode(task, pol, init_idx, max_steps=MAX_STEPS[info["suite"]], record=True)
            log(f"  task {ti} init {init_idx} {name:5s}: {'SUCCESS' if runs[name]['success'] else 'fail':7s} "
                f"({runs[name]['steps']} steps)")
        if not (runs["best"]["success"] and not runs["base"]["success"]):
            continue                                              # keep only the pairs that show the difference
        for name, r in runs.items():
            d = A.EPI_DIR / f"{name}_s{SEED}"; d.mkdir(parents=True, exist_ok=True)
            w = lambda X: A.to_world(D, ti, np.asarray(X, np.float32))
            np.savez_compressed(d / f"{info['suite']}_t{ti}_i{init_idx}_std.npz",
                ee_path=w(r["ee_path"]), grip=r["grip"], success=r["success"], protocol="std",
                goal_pos=np.array([np.nan] * 3, np.float32), goal_frame=-1,
                box_center=np.array([np.nan] * 3, np.float32),
                plans=w(np.array([p[1] for p in r["plans"]])), plan_t=np.array([p[0] for p in r["plans"]]),
                language=info["language"], **{f"frames_{c}": np.stack(v) for c, v in r["frames"].items() if len(v)})
        got += 1; kept += 1
        log(f"  -> kept pair task {ti} init {init_idx}")
    task.close()
log(f"kept {kept} demonstrative pairs -> {A.EPI_DIR}/base_s{SEED} and best_s{SEED}")
