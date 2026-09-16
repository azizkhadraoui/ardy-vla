#!/usr/bin/env python
"""
05_eval_closedloop.py — closed-loop LIBERO: standard success and the four steering protocols, for ONE checkpoint.

WHY THIS IS THE LOAD-BEARING SCRIPT
    Open-loop adherence says the sampler can hit a pose; it does not say the arm does the task while doing so, nor
    that perception is real. Here the policy runs at 20 Hz in robosuite with a joint-position controller, DINOv2 on
    the rendered cameras at every replan (every EXEC frames), the measured proprio history encoded through the
    frozen tokenizer, the standard 50 init states per task, and success from the benchmark's own checker.

CONDITIONS (per task, per init state)
    std      no constraint. Success rate is the number that goes next to the published ones.
    P4 pose  at t0 = 1 s the demo's own pose + gripper 1.6 s ahead is given as a timed goal (demo k <-> init state k).
             Metrics: success, EE error at the goal frame (cm).
    P1 wp    same, but the waypoint is lifted 8 cm and a 6 cm virtual box sits on the demo pose. Metrics: success,
             waypoint error (cm), box "collision" (EE inside the box within +-0.5 s of the waypoint time).
    P2 grip  gripper-only keyframe 1.6 s ahead. Metrics: success, finger-width error at the keyframe (mm).
    P3 rec   the arm is displaced by +-0.15 rad on joints 1-4 at t = 1 s. (a) no correction, (b) a corrective pose goal
             1.5 s ahead. Metrics: success for both; the gap is what the steering buys under a real disturbance.
    Inpainting and guidance variants receive every constraint but can act only when its frame is inside the window.

BUDGET (V100, 8 cores)  ~5 s per episode. Defaults: std on all suites with N_INIT=20 inits per task (800 episodes,
    ~1.2 h); protocols on PROTO_SUITES=libero_spatial,libero_10 with N_INIT_PROTO=10 (1,000 episodes, ~1.4 h).
    Full 50-init standard eval: N_INIT=50 (3 h).

OUTPUT  $WORK_DIR/results/closedloop_{variant}_s{seed}.json  (per-episode records + per-condition summaries)
        $WORK_DIR/episodes/{variant}_s{seed}/*.npz             (recorded episodes for 08_visualize.py, RECORD=1)

    VARIANT=two_stage_goal SEED=0 WORK_DIR=... LIBERO_DIR=... MUJOCO_GL=egl python 05_eval_closedloop.py
    knobs: N_INIT, N_INIT_PROTO, CL_SUITES, PROTO_SUITES, PROTOCOLS=std,P4,P1,P2,P3, EXEC (frames per replan, 8), PROJECT=0|1, RECORD=0|1
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch
import ardy_vla as A
from ardy_vla import log, DEVICE

VARIANT = os.environ["VARIANT"]; SEED = int(os.environ.get("SEED", 0)); vcfg = A.VARIANTS[VARIANT.replace("_scale", "")]
N_INIT, N_INIT_PROTO = int(os.environ.get("N_INIT", 20)), int(os.environ.get("N_INIT_PROTO", 10))
CL_SUITES = os.environ.get("CL_SUITES", ",".join(A.SUITES)).split(","); PROTO_SUITES = os.environ.get("PROTO_SUITES", "libero_spatial,libero_10").split(",")
PROTOCOLS = os.environ.get("PROTOCOLS", "std,P4,P1,P2,P3").split(","); EXEC = int(os.environ.get("EXEC", 8)); PROJECT = os.environ.get("PROJECT", "0") == "1"
RECORD = os.environ.get("RECORD", "0") == "1"; RECORD_TASKS, RECORD_N = int(os.environ.get("RECORD_TASKS", 2)), int(os.environ.get("RECORD_N", 2))
MAX_STEPS = dict(libero_spatial=220, libero_object=280, libero_goal=300, libero_10=520)
T0, HORIZON, LIFT, BOX = 20, 32, 0.08, 0.03                    # protocol constants: 1 s, 1.6 s, 8 cm, 6 cm box
dst = A.RES_DIR / f"closedloop_{VARIANT}_s{SEED}{'_proj' if PROJECT else ''}.json"
if dst.exists() and os.environ.get("FORCE", "0") != "1": log(f"{dst.name} exists; skipping"); sys.exit(0)

D = A.load_data(vision=False)
ck = torch.load(A.CKPT_DIR / f"{VARIANT}_s{SEED}.pt", map_location=DEVICE, weights_only=False)
model = A.HybridDenoiser(D, ck["variant"], d=ck.get("d_model", A.D_MODEL), layers=ck.get("layers", A.LAYERS)).to(DEVICE); model.load_state_dict(ck["state_dict"]); model.eval()
enc = A.OnlineEncoder(D.meta)
records = []; epi_dir = A.EPI_DIR / f"{VARIANT}_s{SEED}"; epi_dir.mkdir(exist_ok=True)
A.wandb_init("closedloop", VARIANT, SEED, config=dict(ckpt_step=ck.get("step"), n_init=N_INIT, n_init_proto=N_INIT_PROTO,
             protocols=PROTOCOLS, cl_suites=CL_SUITES, proto_suites=PROTO_SUITES, exec_frames=EXEC, project=PROJECT))


def demo_goal(D, ep, frame, mask_kind):
    """Normalised explicit values of demo episode `ep` at `frame`, with a mask for full / pos / gripper constraints."""
    frame = int(min(frame, D.ep_len[ep] - 1)); val = D.exp_pad[D.pf_start_t[ep] + frame][None].clone()
    m = torch.zeros(1, D.EXP_F, device=DEVICE)
    if mask_kind == "full": m[:] = 1
    elif mask_kind == "pos": m[:, :3] = 1
    elif mask_kind == "grip": m[:, 9:] = 1
    return val, m, frame


def run_condition(task, policy, init_idx, proto, record):
    D = task.D; ep = int(task.episodes[init_idx]) if init_idx < len(task.episodes) else int(task.episodes[-1]); rec = dict(protocol=proto, init=init_idx)
    goal_fn = perturb_fn = None; f_goal = None; box_center = None
    if proto in ("P4", "P1", "P2", "P3b"):
        kind = dict(P4="full", P1="pos", P2="grip", P3b="full")[proto]; f_goal = T0 + (HORIZON if proto != "P3b" else 30)
        val, m, f_goal = demo_goal(D, ep, f_goal, kind)
        if proto == "P1":
            box_center = A.unnorm_pos(D, val[0, :3]).cpu().numpy().copy(); val = val.clone(); val[0, 2] += LIFT / D.pos_s_t[2]
        rec["goal_pos"] = A.unnorm_pos(D, val[0, :3]).cpu().numpy().tolist(); rec["goal_frame"] = f_goal
        def goal_fn(t, val=val, m=m, f_goal=f_goal):
            if t < T0 or t > f_goal: return None
            return dict(val=val, mask=m, t_frames=f_goal - t)
    if proto in ("P3a", "P3b"):
        rng = np.random.default_rng(1000 * init_idx + 7); delta = np.zeros(7); delta[:4] = rng.choice([-1, 1], 4) * 0.15; done = {"v": False}
        def perturb_fn(task, t, delta=delta, done=done):
            if t == T0 and not done["v"]:
                jidx = task.robot._ref_joint_pos_indexes; task.env.sim.data.qpos[jidx] = task.env.sim.data.qpos[jidx] + delta; task.env.sim.forward(); done["v"] = True
        rec["perturbation_rad"] = delta.tolist()
    out = A.run_episode(task, policy, init_idx, goal_fn=goal_fn, max_steps=MAX_STEPS[task.info["suite"]], perturb_fn=perturb_fn, record=record)
    rec.update(success=bool(out["success"]), steps=int(out["steps"]))
    ee = out["ee_path"]
    if f_goal is not None and ee.shape[0] > f_goal:
        gp = np.array(rec["goal_pos"]); rec["adherence_cm"] = float(np.linalg.norm(ee[f_goal] - gp) * 100)
        if proto == "P2":
            width_goal = float(((val[0, 9:] * D.gr_s_t + D.gr_m_t)[0] - (val[0, 9:] * D.gr_s_t + D.gr_m_t)[1]).item()); rec["grip_err_mm"] = float(abs(out["grip"][f_goal] - width_goal) * 1000)
    if box_center is not None and ee.shape[0] > T0:
        lo, hi = max(0, f_goal - 10), min(ee.shape[0], f_goal + 10); rec["collision"] = bool((np.abs(ee[lo:hi] - box_center) < BOX).all(-1).any()); rec["box_center"] = box_center.tolist()
    if ee.shape[0] > 3: rec["accel_cm"] = float(np.linalg.norm(np.diff(ee, 2, axis=0), axis=-1).mean() * 100)
    if record:
        # everything the model works in is the robot base frame; the videos in 08 draw into camera images,
        # so the recorded geometry is put back into world coordinates here, once, with the task's base transform
        w = lambda X: A.to_world(D, task.task_index, X)
        np.savez_compressed(epi_dir / f"{task.info['suite']}_t{task.task_index}_i{init_idx}_{proto}.npz", ee_path=w(ee), grip=out["grip"], success=out["success"], protocol=proto,
                            goal_pos=w(np.array(rec.get("goal_pos", [np.nan] * 3))), goal_frame=f_goal if f_goal is not None else -1,
                            box_center=w(np.array(box_center if box_center is not None else [np.nan] * 3)),
                            plans=w(np.array([p[1] for p in out["plans"]])), plan_t=np.array([p[0] for p in out["plans"]]), language=task.info["language"],
                            **{f"frames_{c}": np.stack(v) for c, v in out["frames"].items() if len(v)})
    return rec


t_all = time.time()
for ti, info in enumerate(D.meta["tasks"]):
    protos = [p for p in PROTOCOLS if p == "std" and info["suite"] in CL_SUITES] + \
             [q for p in PROTOCOLS if p != "std" and info["suite"] in PROTO_SUITES for q in (["P3a", "P3b"] if p == "P3" else [p])]
    if not protos: continue
    rec_cams = ("agentview", "frontview", "birdview") if (RECORD and SEED == 0 and ti % (len(D.meta["tasks"]) // max(RECORD_TASKS, 1)) == 0) else ()
    task = A.LiberoTask(D, ti, record_cams=rec_cams); policy = A.Policy(D, model, vcfg, enc, ti, exec_frames=EXEC, project=PROJECT)
    n_std, n_pro = min(N_INIT, len(task.init_states)), min(N_INIT_PROTO, len(task.init_states)); t0 = time.time(); done_here = []
    for proto in protos:
        for init_idx in range(n_std if proto == "std" else n_pro):
            rec = run_condition(task, policy, init_idx, proto, record=bool(rec_cams) and init_idx < RECORD_N); rec.update(task=ti, suite=info["suite"], language=info["language"])
            records.append(rec); done_here.append(rec)
    task.close()
    summ = {p: np.mean([r["success"] for r in done_here if r["protocol"] == p]) for p in dict.fromkeys(r["protocol"] for r in done_here)}
    log(f"task {ti:2d} {info['suite']:14s} {info['language'][:50]:50s} " + " ".join(f"{p}={v:.2f}" for p, v in summ.items()) + f"  ({time.time()-t0:.0f}s)")
    # running view of a 2.6 h job: per-task success as it lands, and the running mean over every task finished so far
    A.wandb_log({"closedloop/step": ti, "closedloop/task_secs": time.time() - t0,
                 **{f"closedloop/task/{p}_success": v for p, v in summ.items()},
                 **{f"closedloop/running/{p}_success": float(np.mean([r["success"] for r in records if r["protocol"] == p]))
                    for p in dict.fromkeys(r["protocol"] for r in records)}})
    json.dump(dict(variant=VARIANT, seed=SEED, project=PROJECT, exec_frames=EXEC, n_init=N_INIT, n_init_proto=N_INIT_PROTO, records=records, partial=True), open(dst, "w"))

# ---- summaries -----------------------------------------------------------------------------------------
def summarise(recs):
    out = {}
    for proto in dict.fromkeys(r["protocol"] for r in recs):
        rs = [r for r in recs if r["protocol"] == proto]; s = dict(n=len(rs), success=float(np.mean([r["success"] for r in rs])))
        for k in ("adherence_cm", "grip_err_mm", "accel_cm"):
            v = [r[k] for r in rs if k in r]; s[k] = float(np.mean(v)) if v else None
        if any("collision" in r for r in rs): s["collision"] = float(np.mean([r["collision"] for r in rs if "collision" in r]))
        out[proto] = s
    return out
summary = dict(all=summarise(records), per_suite={s: summarise([r for r in records if r["suite"] == s]) for s in dict.fromkeys(r["suite"] for r in records)})
json.dump(dict(variant=VARIANT, seed=SEED, project=PROJECT, exec_frames=EXEC, n_init=N_INIT, n_init_proto=N_INIT_PROTO, records=records, summary=summary, secs=round(time.time() - t_all), partial=False), open(dst, "w"), indent=1)
print("\n" + "=" * 100); print(f" {VARIANT} seed {SEED}{' +projection' if PROJECT else ''}   ({(time.time()-t_all)/3600:.1f} h)")
print(f" {'condition':10s}{'n':>6s}{'success':>10s}{'adherence cm':>14s}{'grip mm':>10s}{'collision':>11s}{'accel cm':>10s}"); print("-" * 100)
for p, s in summary["all"].items():
    f = lambda v, w: f"{v:>{w}.2f}" if isinstance(v, float) else f"{'-':>{w}s}"
    print(f" {p:10s}{s['n']:>6d}{s['success']:>10.3f}{f(s.get('adherence_cm'), 14)}{f(s.get('grip_err_mm'), 10)}{f(s.get('collision'), 11)}{f(s.get('accel_cm'), 10)}")
print("=" * 100); print("READING: std is the number next to published LIBERO results. P4/P1/P2 adherence is whether the constraint was met in closed loop;\n"
                        "P1 collision is whether the lift actually cleared the box. P3b - P3a is the recovery the steering buys under a real disturbance.")
A.wandb_summary(summary["all"], prefix="closedloop/")
A.wandb_summary(summary["per_suite"], prefix="closedloop/suite/")
A.wandb_summary(dict(hours=round((time.time() - t_all) / 3600, 2), n_episodes=len(records)), prefix="closedloop/")
A.wandb_table("closedloop/protocols", [dict(protocol=pr, **su_) for pr, su_ in summary["all"].items()])
A.wandb_table("closedloop/per_suite", [dict(suite=sn, protocol=pr, **su_)
                                       for sn, d_ in summary["per_suite"].items() for pr, su_ in d_.items()])
A.wandb_finish()
log(f"raw results -> {dst}")
