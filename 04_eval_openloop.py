#!/usr/bin/env python
"""
04_eval_openloop.py — constraint adherence versus goal horizon, open loop, no simulator.

WHAT IT SETTLES
    The cleanest figure in the paper: for a goal placed 0.8 / 1.6 / 2.4 / 3.2 s ahead, how close does the
    policy land at that instant, on the explicit stream, on the decoded body (FK), and on the body after the
    inference-time consistency projection — against the same rollout with no goal. Per variant, per seed.

PROTOCOL (identical to the Kaggle runs so the numbers are comparable)
    Teacher-forced ground-truth history at window starts 8 / 16 / 24 patches into every held-out demo (200 demos),
    the goal is the demo's own pose + gripper at the target frame, rollouts are autoregressive with the model's
    own tokens as history and ground-truth camera features at each window start (there is no renderer here).
    Inpainting and guidance can only act once the goal frame is inside the current window; that is the point.

OUTPUT   $WORK_DIR/results/openloop_{variant}_s{seed}.json
    Keys:  {inwin|h4|h8|h12|h16}_{goal|nogoal}_{pos|fk|fkproj}_err_cm, *_goal_rot_err_deg, *_goal_jump_excess_cm,
           nogoal_ee_err_cm, fk_consistency_cm, joint_rmse_deg.

    WORK_DIR=... python 04_eval_openloop.py                # every checkpoint found in $WORK_DIR/ckpt
    VARIANT=two_stage_goal SEED=0 python 04_eval_openloop.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import ardy_vla as A
from ardy_vla import log

D = A.load_data()
todo = [(os.environ["VARIANT"], int(os.environ.get("SEED", 0)))] if "VARIANT" in os.environ else \
       [(p.stem.rsplit("_s", 1)[0], int(p.stem.rsplit("_s", 1)[1])) for p in sorted(A.CKPT_DIR.glob("*_s*.pt"))]
for variant, seed in todo:
    dst = A.RES_DIR / f"openloop_{variant}_s{seed}.json"
    if dst.exists() and os.environ.get("FORCE", "0") != "1": log(f"{dst.name} exists; skipping"); continue
    name = variant.replace("_scale", ""); vcfg = A.VARIANTS[name]
    ck = torch.load(A.CKPT_DIR / f"{variant}_s{seed}.pt", map_location=A.DEVICE, weights_only=False)
    model = A.HybridDenoiser(D, ck["variant"], d=ck.get("d_model", A.D_MODEL), layers=ck.get("layers", A.LAYERS)).to(A.DEVICE); model.load_state_dict(ck["state_dict"]); model.eval()
    A.seed_all(1000 + seed); t0 = time.time()
    r = A.evaluate_openloop(D, model, vcfg, D.val_eps)
    # per-suite breakdown of the two headline numbers
    per_suite = {}
    for suite in D.meta["suites"]:
        eps = np.array([e for e in D.val_eps if D.meta["tasks"][D.ep_task[e]]["suite"] == suite])
        if len(eps) >= 5:
            rs = A.evaluate_openloop(D, model, vcfg, eps, horizons=[A.OUT_HORIZON], windows=[16], project=False)
            per_suite[suite] = {k: rs[k] for k in rs if k.startswith(("inwin_goal_pos", "inwin_goal_fk", "outwin_goal_pos", "outwin_goal_fk", "outwin_nogoal"))}
    out = dict(variant=variant, seed=seed, steps=ck.get("step"), d_model=ck.get("d_model"), layers=ck.get("layers"), metrics=r, per_suite=per_suite, secs=round(time.time() - t0))
    json.dump(out, open(dst, "w"), indent=2)
    log(f"{variant} s{seed}: in-goal {r.get('inwin_goal_pos_err_cm'):.2f} / in-FK {r.get('inwin_goal_fk_err_cm'):.2f} | h8 goal {r.get('h8_goal_pos_err_cm'):.2f} FK {r.get('h8_goal_fk_err_cm'):.2f} "
        f"FKproj {r.get('h8_goal_fkproj_err_cm', float('nan')):.2f} | h8 no-goal FK {r.get('h8_nogoal_fk_err_cm'):.2f} | jump {r.get('h8_goal_jump_excess_cm'):+.2f}  ({out['secs']}s) -> {dst.name}")
    del model; torch.cuda.empty_cache()
