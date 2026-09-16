#!/usr/bin/env python
"""
01_prepare_data.py — the four standard LIBERO suites as one dataset.

WHAT IT SETTLES
    Nothing by itself; it removes two objections at once. "One suite, 450 demos" becomes 40 tasks and
    2,000 demos, and the explicit end-effector stream is DEFINED as base ∘ FK(joints) ∘ tool, so the
    consistency loss used by every variant is exact rather than fitted (the dataset's own ee_pos is a
    rigid transform of the flange only to ~0.3 cm; that offset is stored, not used).

WHAT IT PRODUCES ($WORK_DIR/data)
    proprio.npz          joint_pos, joint_vel, gripper, ee_pos, ee_rot6d (FK-defined), ee_pos_dataset,
                         actions, episode_start/len/task, fk_* (the fitted base/tool parameters)
    vision_*.npy         frozen DINOv2-S tokens (CLS + 4x4 pooled patches) per frame and camera, fp16
    text_emb.npy         one mean-pooled T5 vector per task string
    meta.json            tasks [{suite, task_id, language}], encoder names, gripper threshold, FK residuals

CHECKS TO READ IN THE LOG
    "explicit EE stream defined from FK (dataset EE differs by ~0.3 cm)"  — anything above 1 cm means the
    joint/EE pairing is wrong for a suite and nothing downstream is trustworthy.
    "gripper threshold": the finger-width midpoint between the demos' open and close commands. It should
    sit around 4-5 cm; it is what the closed-loop policy uses to turn a predicted width into a command.

    WORK_DIR=... SUITES=libero_spatial,libero_object,libero_goal,libero_10 python 01_prepare_data.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
import ardy_vla as A
from ardy_vla import log

if (A.DATA / "meta.json").exists() and os.environ.get("FORCE", "0") != "1":
    log("data already prepared; set FORCE=1 to redo"); sys.exit(0)

import h5py
from transformers import AutoModel, AutoTokenizer, T5EncoderModel
files_by_suite = A.download_suites()

KEYS = dict(agentview=["agentview_rgb", "agentview_image"], wrist=["eye_in_hand_rgb", "robot0_eye_in_hand_image"], joint_pos=["joint_states", "robot0_joint_pos"],
            gripper=["gripper_states", "robot0_gripper_qpos"], ee_pos=["ee_pos", "robot0_eef_pos"], ee_ori=["ee_ori", "robot0_eef_quat"])
def key(g, cands):
    for c in cands:
        if c in g: return c
    raise KeyError(f"none of {cands} in {list(g.keys())}")
def language_of(f, path):
    try:
        info = json.loads(f["data"].attrs["problem_info"])
        for k in ("language_instruction", "language", "instruction"):
            if k in info: return info[k]
    except Exception: pass
    return os.path.basename(path).replace("_demo.hdf5", "").replace("_", " ")

# ---- pass 1: proprio for every suite -------------------------------------------------------------
tasks, ep_start, ep_len, ep_task, ep_index = [], [], [], [], []
JP, GR, EP_, ER6, ACT = [], [], [], [], []; n = 0
for suite, files in files_by_suite.items():
    for path in files:
        with h5py.File(path, "r") as f:
            lang = language_of(f, path); ti = len(tasks); tasks.append(dict(suite=suite, language=lang, file=os.path.basename(path)))   # LIBERO task id resolved by language in 05
            data = f["data"]
            for dn in sorted(data.keys(), key=lambda k: int(k.split("_")[1])):
                d = data[dn]; obs = d["obs"]
                q = obs[key(obs, KEYS["joint_pos"])][()].astype(np.float32); g = obs[key(obs, KEYS["gripper"])][()].astype(np.float32)
                p = obs[key(obs, KEYS["ee_pos"])][()].astype(np.float32); o = obs[key(obs, KEYS["ee_ori"])][()].astype(np.float64)
                R = A.axis_angle_to_matrix(o) if o.shape[-1] == 3 else A.quat_to_matrix(o); T = q.shape[0]
                JP.append(q); GR.append(g); EP_.append(p); ER6.append(np.concatenate([R[:, :, 0], R[:, :, 1]], -1).astype(np.float32)); ACT.append(d["actions"][()].astype(np.float32))
                ep_start.append(n); ep_len.append(T); ep_task.append(ti); ep_index.append((path, dn)); n += T
    log(f"{suite}: {len(files)} tasks, {sum(1 for i in ep_task if tasks[i]['suite'] == suite)} episodes so far")
ep_start, ep_len, ep_task = np.array(ep_start), np.array(ep_len), np.array(ep_task)
joint_pos, gripper, ee_pos_ds, ee_rot6d_ds, actions = [np.concatenate(a) for a in (JP, GR, EP_, ER6, ACT)]
log(f"pass 1: {len(tasks)} tasks, {len(ep_len)} episodes, {n} frames, mean len {ep_len.mean():.0f}")

# ---- FK: one base per task, one shared tool; the explicit stream is the ROBOT-BASE-FRAME EE ------
# LIBERO puts the robot base at a scene-dependent world position (libero_10 alone spans KITCHEN, LIVING_ROOM
# and STUDY scenes), so no single base transform maps FK(joints) onto the dataset's world-frame ee_pos: pooled
# it lands 30-50 cm off while each scene alone fits to well under a millimetre. The fix is not a looser
# threshold. The base offset is a property of the scene, carries no information about the arm, and would make
# the same motion look different in each scene, so the stream the model sees is defined WITHOUT it:
#     explicit stream := FK(joints) o tool        (the EE in the robot's own base frame, scene-independent)
# The per-task base transforms are fitted anyway, kept for putting a trajectory back into world coordinates
# (05 records world-frame paths for the videos in 08), and used for the data-integrity check below.
frame_task = np.repeat(ep_task, ep_len)
idx = np.random.default_rng(0).choice(n, min(65536, n), replace=False)
q_t, ee_t = torch.tensor(joint_pos, device=A.DEVICE), torch.tensor(ee_pos_ds, device=A.DEVICE); R_t = A.rot6d_to_mat(torch.tensor(ee_rot6d_ds, device=A.DEVICE))
gid_t = torch.tensor(frame_task, device=A.DEVICE, dtype=torch.long)
base_r6, base_t, tool_r6, tool_t = A.fit_fk_grouped(q_t[idx], ee_t[idx], R_t[idx], gid_t[idx], len(tasks), iters=4000)
params = [torch.tensor([1., 0, 0, 0, 1, 0], device=A.DEVICE), torch.zeros(3, device=A.DEVICE), tool_r6, tool_t]   # base-frame stream
pos_l, R_l, res_l = [], [], []
with torch.no_grad():
    for s0 in range(0, n, 16384):
        sl = slice(s0, min(s0 + 16384, n))
        pp, RR = A.fk_with(params, q_t[sl]); pos_l.append(pp.cpu().numpy()); R_l.append(A.mat_to_rot6d(RR).cpu().numpy())
        wp, _ = A.fk_grouped(base_r6, base_t, tool_r6, tool_t, q_t[sl], gid_t[sl])          # the same poses back in world frame
        res_l.append((wp - ee_t[sl]).norm(dim=-1).cpu().numpy())
ee_pos, ee_rot6d = np.concatenate(pos_l).astype(np.float32), np.concatenate(R_l).astype(np.float32)
res_cm = np.concatenate(res_l) * 100
# the gate, now per task rather than per suite: a wrong joint/EE pairing shows up as a task that will not fit
task_res = np.array([float(res_cm[frame_task == i].mean()) for i in range(len(tasks))])
fk_res = {s: float(task_res[[i for i, t in enumerate(tasks) if t["suite"] == s]].max()) for s in files_by_suite}
worst = int(task_res.argmax())
log(f"explicit EE stream defined from FK in the robot base frame; base+tool reproduces the dataset's world EE to "
    f"(worst task per suite, cm): { {k: round(v, 3) for k, v in fk_res.items()} }")
log(f"  worst task overall: {tasks[worst]['suite']}/{tasks[worst]['file']} at {task_res[worst]:.3f} cm; mean over all frames {res_cm.mean():.3f} cm")
log(f"  fitted base translations span {np.abs(base_t.cpu().numpy() - base_t.cpu().numpy().mean(0)).max()*100:.1f} cm across tasks "
    f"(that spread is the scene offset the old single-base fit was trying to average away)")
assert task_res.max() < 1.0, (f"FK does not reproduce the dataset EE for task {tasks[worst]['file']} "
                              f"({task_res[worst]:.2f} cm): check the joint/EE key pairing")
joint_vel = np.concatenate([np.gradient(joint_pos[ep_start[e]:ep_start[e] + ep_len[e]], axis=0) * A.FPS for e in range(len(ep_len))]).astype(np.float32)

# ---- gripper threshold: finger width midpoint between demo open (-1) and close (+1) commands -------
width = gripper[:, 0] - gripper[:, 1]; a = actions[:, -1]
thr = float((width[a > 0].mean() + width[a < 0].mean()) / 2)
log(f"gripper threshold {thr*100:.2f} cm  (closed mean {width[a>0].mean()*100:.2f}, open mean {width[a<0].mean()*100:.2f})")

np.savez_compressed(A.DATA / "proprio.npz", joint_pos=joint_pos, joint_vel=joint_vel, gripper=gripper, ee_pos=ee_pos, ee_rot6d=ee_rot6d, ee_pos_dataset=ee_pos_ds, ee_rot6d_dataset=ee_rot6d_ds,
                    actions=actions, episode_start=ep_start, episode_len=ep_len, episode_task=ep_task,
                    fk_base_r6=base_r6.cpu().numpy(), fk_base_t=base_t.cpu().numpy(),          # per task (n_tasks, 6) / (n_tasks, 3): base-frame -> world
                    fk_tool_r6=tool_r6.cpu().numpy(), fk_tool_t=tool_t.cpu().numpy(),            # shared flange -> EE
                    fk_task_residual_cm=task_res.astype(np.float32))

# ---- frozen vision features -----------------------------------------------------------------------
VIS_NAME = "facebook/dinov2-small" if A.ENCODER == "dinov2" else "google/siglip-base-patch16-224"
MEAN, STD = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) if A.ENCODER == "dinov2" else ((0.5,) * 3, (0.5,) * 3)
vis = AutoModel.from_pretrained(VIS_NAME); vis = getattr(vis, "vision_model", vis).to(A.DEVICE).eval().half()
mean_t, std_t = [torch.tensor(v, device=A.DEVICE).view(1, 3, 1, 1).half() for v in (MEAN, STD)]
@torch.no_grad()
def encode(u8):
    x = torch.from_numpy(np.ascontiguousarray(u8)).to(A.DEVICE)
    if A.FLIP_180: x = torch.flip(x, dims=(1, 2))
    x = x.permute(0, 3, 1, 2).half().div_(255.0); x = F.interpolate(x, size=(A.IMG_RES, A.IMG_RES), mode="bilinear", align_corners=False, antialias=True)
    out = vis(pixel_values=(x - mean_t) / std_t); h = out.last_hidden_state
    cls, patches = (h[:, :1], h[:, 1:]) if A.ENCODER == "dinov2" else (out.pooler_output[:, None], h)
    g_ = int(round(patches.shape[1] ** 0.5)); patches = patches.transpose(1, 2).reshape(patches.shape[0], -1, g_, g_)
    return torch.cat([cls, F.adaptive_avg_pool2d(patches, A.POOL).flatten(2).transpose(1, 2)], 1).cpu()
DV = encode(np.zeros((1, 128, 128, 3), np.uint8)).shape[-1]; NT = 1 + A.POOL * A.POOL
va = np.lib.format.open_memmap(A.DATA / "vision_agentview.npy", mode="w+", dtype=np.float16, shape=(n, NT, DV))
vw = np.lib.format.open_memmap(A.DATA / "vision_wrist.npy", mode="w+", dtype=np.float16, shape=(n, NT, DV))
t0 = time.time()
for e, (path, dn) in enumerate(ep_index):
    s, T = ep_start[e], ep_len[e]
    with h5py.File(path, "r") as f:
        obs = f["data"][dn]["obs"]; ka, kw = key(obs, KEYS["agentview"]), key(obs, KEYS["wrist"])
        for b in range(0, T, 256):
            sl = slice(b, min(b + 256, T)); va[s + sl.start:s + sl.stop] = encode(obs[ka][sl]).numpy(); vw[s + sl.start:s + sl.stop] = encode(obs[kw][sl]).numpy()
        if e == 0:
            from PIL import Image; img = obs[ka][0]; Image.fromarray(np.ascontiguousarray(img[::-1, ::-1] if A.FLIP_180 else img)).save(A.FIG_DIR / "preview_agentview_frame0.png")
    if e % 200 == 0: log(f"  vision {e}/{len(ep_index)} episodes, {time.time()-t0:.0f}s")
va.flush(); vw.flush(); del va, vw

# ---- text ----------------------------------------------------------------------------------------
tk = AutoTokenizer.from_pretrained(A.TEXT_MODEL); t5 = T5EncoderModel.from_pretrained(A.TEXT_MODEL).to(A.DEVICE).eval()
with torch.no_grad():
    enc = tk([t["language"] for t in tasks], return_tensors="pt", padding=True).to(A.DEVICE); hs = t5(**enc).last_hidden_state; m = enc.attention_mask[..., None].float()
    text_emb = ((hs * m).sum(1) / m.sum(1)).float().cpu().numpy()
np.save(A.DATA / "text_emb.npy", text_emb)
json.dump(dict(suites=list(files_by_suite), fps=A.FPS, num_frames=int(n), num_episodes=len(ep_len), tasks=tasks, vision_encoder=VIS_NAME, vision_tokens=NT, vision_dim=int(DV),
               text_model=A.TEXT_MODEL, flip_180=A.FLIP_180, ee_from_fk=True, ee_frame="robot_base", fk_dataset_offset_cm=fk_res, fk_task_residual_cm=[round(v, 4) for v in task_res.tolist()], gripper_threshold_m=thr), open(A.DATA / "meta.json", "w"), indent=2)
log(f"stage 1 done -> {A.DATA}  (figures/preview_agentview_frame0.png: the table must be at the BOTTOM of the image)")
