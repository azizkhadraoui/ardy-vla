#!/usr/bin/env python
"""Why are libero_10 tasks 34 and 35 at exactly 0.00 in every configuration?

Six configurations spanning different vision encoders, chunk lengths, execution strides and training recipes all
score 0.00 on t34 and t35, while the structurally identical t36 ("put both X and Y in the basket") reaches 0.80.
An exact zero invariant to every intervention has three times in this project turned out to be a harness bug
rather than a capability limit, so this checks the plumbing before anyone concludes the model cannot do it.

Per long-suite task it reports:
  file / language   what our dataset thinks the task is
  resolved          the LIBERO task the evaluator actually loads, and its BDDL -- a mismatch here means the model
                    was trained on one task and is evaluated on another, which would pin success at zero
  demos, frames     whether the training data for that task is present and of sane length
  fk_cm             residual of the per-task FK base fit; if this is large the explicit EE stream is nonsense
  replay            does the demo's own joint trajectory satisfy the benchmark's checker at all
  rollout           what the policy actually does: does the gripper ever close, does the EE reach the objects

    WORK_DIR=... VARIANT=two_stage_goal SEED=123 VIS_SUFFIX=_ft VIS_CKPT=... python t34_diag.py
"""
import os, sys, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch
import ardy_vla as A
from ardy_vla import log, DEVICE

TASKS = [int(x) for x in os.environ.get("DIAG_TASKS", "34,35,36,30").split(",")]
VARIANT = os.environ.get("VARIANT", "two_stage_goal"); SEED = int(os.environ.get("SEED", 123))
N_INIT = int(os.environ.get("DIAG_INIT", 3))

D = A.load_data(vision=False)
pr = D.pr if hasattr(D, "pr") else np.load(A.DATA / "proprio.npz")
A.ensure_libero_repo()
from libero.libero import benchmark, get_libero_path

rows = []
for ti in TASKS:
    info = D.meta["tasks"][ti]
    stem = re.sub(r"_demo\.hdf5$", "", info["file"])
    suite = benchmark.get_benchmark_dict()[info["suite"]]()
    norm = lambda s: re.sub(r"[^a-z]", "", s.lower())
    lut = {norm(suite.get_task(i).language): i for i in range(suite.n_tasks)}
    tid = lut.get(norm(info["language"]))
    resolved = suite.get_task(tid) if tid is not None else None
    eps = np.where(D.ep_task == ti)[0]
    lens = D.ep_len[eps]
    # FK residual: how well the per-task base fit explains this task's recorded EE positions
    q = torch.from_numpy(pr["joint_pos"][D.ep_start[eps[0]]:D.ep_start[eps[0]] + int(lens[0])].astype(np.float32)).to(DEVICE)
    pos, _ = A.fk_with(D.fk_params, q)
    tgt = torch.from_numpy(pr["ee_pos"][D.ep_start[eps[0]]:D.ep_start[eps[0]] + int(lens[0])].astype(np.float32)).to(DEVICE)
    fk_cm = float((pos - tgt).norm(dim=-1).mean()) * 100
    row = dict(task=ti, suite=info["suite"], stem=stem, language=info["language"], n_demos=len(eps),
               mean_len=float(lens.mean()), fk_cm=fk_cm,
               resolved_name=(resolved.name if resolved else None), resolved_bddl=(resolved.bddl_file if resolved else None),
               name_matches=bool(resolved and resolved.name == stem))
    log(f"t{ti} {info['suite']}")
    log(f"   our file    {stem}")
    log(f"   resolved    {row['resolved_name']}   MATCH={row['name_matches']}")
    log(f"   bddl        {row['resolved_bddl']}")
    log(f"   demos {len(eps)}  mean length {lens.mean():.0f}  FK fit residual {fk_cm:.3f} cm")
    rows.append(row)

# ---- does the benchmark's own checker ever fire for these tasks, and what does the policy do?
ck = torch.load(A.CKPT_DIR / f"{VARIANT}_s{SEED}.pt", map_location=DEVICE, weights_only=False)
model = A.HybridDenoiser(D, ck["variant"], d=ck.get("d_model", A.D_MODEL), layers=ck.get("layers", A.LAYERS),
                         w_grip=ck.get("w_grip", 0.0)).to(DEVICE)
A.load_compat(model, ck["state_dict"]); model.eval(); enc = A.OnlineEncoder(D.meta)
log("")
for ti in TASKS:
    info = D.meta["tasks"][ti]; task = A.LiberoTask(D, ti)
    eps = np.where(D.ep_task == ti)[0]
    # 1. the demo's own joints through the controller: does the checker fire for this task at all?
    e = int(eps[0]); s0, T = int(D.ep_start[e]), int(D.ep_len[e])
    q = pr["joint_pos"][s0:s0 + T]; cmd = pr["actions"][s0:s0 + T, -1] > 0
    task.reset(0); ok_replay = False
    for t in range(min(T, 520)):
        _, done, _ = task.step_to(q[t], bool(cmd[t]))
        if done: ok_replay = True; break
    # 2. the policy
    pol = A.Policy(D, model, A.VARIANTS[VARIANT], enc, ti, exec_frames=int(os.environ.get("EXEC", 32)), project=True)
    succ = 0; closes = []; mind = []
    for init in range(N_INIT):
        pol.seed_base = 1000003 * ti + 10007 * init
        r = A.run_episode(task, pol, init, max_steps=520)
        succ += int(r["success"])
        g = np.asarray(r["grip"]); closes.append(int((np.diff((g < 0.05).astype(int)) > 0).sum()))
        mind.append(float(np.abs(np.diff(r["ee_path"], axis=0)).sum()))
    task.close()
    log(f"t{ti}: demo replay success {ok_replay} | policy {succ}/{N_INIT} | gripper closes per episode {closes} | EE path length {np.round(mind, 2)}")
    for r in rows:
        if r["task"] == ti: r.update(demo_replay=ok_replay, policy_succ=succ / N_INIT, closes=closes)
json.dump(rows, open(A.RES_DIR / "t34_diag.json", "w"), indent=1)
log("")
log("READING: name_matches=False means the model and the evaluator disagree about which task this is.")
log("         demo_replay=False with the checker never firing means the task is unsatisfiable as we set it up.")
log("         closes=[0,0,0] means the policy never even attempts a grasp.")
