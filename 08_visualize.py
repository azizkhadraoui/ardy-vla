#!/usr/bin/env python
"""
08_visualize.py — the examples: side-by-side closed-loop videos and trajectory figures from the episodes 05 recorded.

WHAT IT MAKES ($WORK_DIR/figures)
    {suite}_t{task}_i{init}__{variantA}_vs_{variantB}__{protocol}.mp4
        one row per camera (agentview / frontview / birdview), one column per variant; the measured EE path is projected
        into every camera (traversed part bold, the plan at each replan thin), the constraint is a red circle at its
        frame, the P1 box is drawn, success/failure is stamped.
    {suite}_t{task}_i{init}__{protocol}_trajectories.png
        3-D and top-down EE paths of every recorded variant, the goal, the box, the perturbation instant.
    steering_gallery.png
        one goal-frame snapshot per protocol for the method, for the paper's figure 1.

No GPU. Needs ffmpeg (imageio[ffmpeg]) and the robosuite camera matrices, which are re-derived from the LIBERO env
(so LIBERO_DIR and MUJOCO_GL=egl must be set; it builds each task env once, renders nothing itself).

    WORK_DIR=... LIBERO_DIR=... MUJOCO_GL=egl python 08_visualize.py
    knobs: VIS_VARIANTS=two_stage_goal,two_stage_inpaint  VIS_SEED=0  VIS_MAX=12 (episodes)
"""
import os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, imageio
from PIL import Image, ImageDraw
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import ardy_vla as A
from ardy_vla import log

VIS_VARIANTS = os.environ.get("VIS_VARIANTS", "two_stage_goal,two_stage_inpaint").split(","); VIS_SEED = int(os.environ.get("VIS_SEED", 0)); VIS_MAX = int(os.environ.get("VIS_MAX", 12))
COL = [(255, 90, 60), (80, 160, 255), (120, 220, 120), (255, 200, 60)]
meta = json.load(open(A.DATA / "meta.json"))
epi = {v: {os.path.basename(p)[:-4]: p for p in glob.glob(str(A.EPI_DIR / f"{v}_s{VIS_SEED}" / "*.npz"))} for v in VIS_VARIANTS}
common = sorted(set.intersection(*[set(d) for d in epi.values()])) if all(epi.values()) else sorted(epi[VIS_VARIANTS[0]])
log(f"{len(common)} recorded episodes shared by {VIS_VARIANTS}")
A.wandb_init("visualize", name="visualize", config=dict(vis_variants=VIS_VARIANTS, vis_seed=VIS_SEED, vis_max=VIS_MAX, n_episodes=len(common)))

# camera matrices per task (built once per task from the env, no rendering)
A.ensure_libero_repo(); A.patch_robosuite()
from robosuite.utils.camera_utils import get_camera_transform_matrix, project_points_from_world_to_camera
cam_cache = {}
def cam_mats(task_index, cams, res):
    if task_index not in cam_cache:
        D = A.SimpleNamespace(meta=meta, ep_task=np.load(A.DATA / "proprio.npz")["episode_task"]); t = A.LiberoTask(D, task_index)
        cam_cache[task_index] = {c: get_camera_transform_matrix(t.env.sim, c, res, res) for c in cams if c in list(t.env.sim.model.camera_names)}; t.close()
    return cam_cache[task_index]
def proj(pts, M, res): return project_points_from_world_to_camera(np.asarray(pts, np.float64).reshape(-1, 3), M, res, res)

gallery = []
for name in common[:VIS_MAX]:
    runs = {v: dict(np.load(epi[v][name], allow_pickle=True)) for v in VIS_VARIANTS if name in epi[v]}
    suite, tpart, ipart, proto = name.split("_t")[0], name.split("_t")[1].split("_i")[0], name.split("_i")[1].split("_")[0], name.rsplit("_", 1)[1]
    ti = int(tpart); cams = [k[7:] for k in next(iter(runs.values())) if k.startswith("frames_")]
    if not cams: continue
    res = next(iter(runs.values()))[f"frames_{cams[0]}"].shape[1]; mats = cam_mats(ti, cams, res)
    T = max(r[f"frames_{cams[0]}"].shape[0] for r in runs.values())
    frames = []
    for t in range(T):
        rows = []
        for c in cams:
            panels = []
            for vi, (v, r) in enumerate(runs.items()):
                fr = r[f"frames_{c}"]; img = Image.fromarray(fr[min(t, fr.shape[0] - 1)]); d = ImageDraw.Draw(img); col = COL[vi % len(COL)]
                if c in mats:
                    M = mats[c]; path = r["ee_path"]
                    if len(path) > 1:
                        px = proj(path, M, res); pts = [(float(p[1]), float(p[0])) for p in px]; d.line(pts, fill=tuple(int(x * 0.4) for x in col), width=1); d.line(pts[:min(t, len(pts) - 1) + 1] or pts[:1], fill=col, width=3)
                    gf = int(r["goal_frame"])
                    if gf >= 0 and not np.isnan(r["goal_pos"]).any():
                        g = proj(r["goal_pos"], M, res)[0]; rr = 9 if abs(t - gf) <= 2 else 5; d.ellipse([g[1] - rr, g[0] - rr, g[1] + rr, g[0] + rr], outline=(255, 40, 40), width=2)
                    if not np.isnan(r["box_center"]).any():
                        bc = r["box_center"]; corners = np.array([[bc[0] + sx * 0.03, bc[1] + sy * 0.03, bc[2] + sz * 0.03] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]); pc = proj(corners, M, res)
                        for i in range(8):
                            for j in range(i + 1, 8):
                                if np.sum(corners[i] != corners[j]) == 1: d.line([(pc[i][1], pc[i][0]), (pc[j][1], pc[j][0])], fill=(255, 120, 0), width=1)
                d.rectangle([0, 0, res, 16], fill=(0, 0, 0)); d.text((3, 2), f"{c} | {v} | {proto} | t={t/A.FPS:4.1f}s | {'SUCCESS' if bool(r['success']) else 'fail'}", fill=(255, 255, 255)); panels.append(np.asarray(img))
            rows.append(np.concatenate(panels, 1))
        frames.append(np.concatenate(rows, 0))
    vp = A.FIG_DIR / f"{name.rsplit('_', 1)[0]}__{'_vs_'.join(runs)}__{proto}.mp4"; imageio.mimwrite(vp, frames, fps=A.FPS, quality=8)
    r0 = next(iter(runs.values())); gf = int(r0["goal_frame"])
    if gf >= 0 and gf < len(frames): gallery.append((proto, frames[gf]))
    # trajectory figure
    fig = plt.figure(figsize=(11, 4.5)); ax = fig.add_subplot(1, 2, 1, projection="3d"); ax2 = fig.add_subplot(1, 2, 2)
    for vi, (v, r) in enumerate(runs.items()):
        e = r["ee_path"]; col = tuple(x / 255 for x in COL[vi % len(COL)]); ax.plot(e[:, 0], e[:, 1], e[:, 2], color=col, label=f"{v} ({'success' if bool(r['success']) else 'fail'})"); ax2.plot(e[:, 0], e[:, 1], color=col)
    if not np.isnan(r0["goal_pos"]).any(): ax.scatter(*r0["goal_pos"], c="r", marker="*", s=140, label="constraint"); ax2.scatter(r0["goal_pos"][0], r0["goal_pos"][1], c="r", marker="*", s=140)
    if not np.isnan(r0["box_center"]).any(): ax.scatter(*r0["box_center"], c="orange", marker="s", s=120, label="box (6 cm)"); ax2.scatter(r0["box_center"][0], r0["box_center"][1], c="orange", marker="s", s=120)
    ax.set_title(f"{suite} task {ti} init {ipart} — {proto}: {str(r0['language'])[:60]}", fontsize=8); ax.legend(fontsize=7); ax2.set_aspect("equal"); ax2.grid(alpha=0.3); ax2.set_title("top-down")
    plt.tight_layout(); plt.savefig(A.FIG_DIR / f"{name.rsplit('_', 1)[0]}__{proto}_trajectories.png", dpi=130); plt.close()
    A.wandb_video(f"videos/{proto}/{vp.stem}", vp)
    A.wandb_images({f"trajectories/{proto}/{name.rsplit('_', 1)[0]}": A.FIG_DIR / f"{name.rsplit('_', 1)[0]}__{proto}_trajectories.png"})
    log(f"{vp.name}  ({T} frames, cameras {cams})")
if gallery:
    seen = {}; [seen.setdefault(p, f) for p, f in gallery]
    Image.fromarray(np.concatenate([np.asarray(Image.fromarray(f).resize((f.shape[1] // 2, f.shape[0] // 2))) for f in seen.values()], 0)).save(A.FIG_DIR / "steering_gallery.png")
    A.wandb_images({"figures/steering_gallery": A.FIG_DIR / "steering_gallery.png"})
    log(f"steering_gallery.png: {list(seen)}")
A.wandb_finish()
