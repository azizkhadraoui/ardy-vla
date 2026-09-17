#!/usr/bin/env python
"""Stage 0, the checks that need no simulator: what the frozen vision tokens actually encode, and whether the
tokenizer behaves in the mode it is deployed in.

WHY
    The 0.22 is dominated by tasks whose target is ~4 cm (libero_object 0.020) while ~12 cm bowls run at 0.468.
    Two candidate causes live here:
      (a) localisation -- each of the 4x4 pooled DINOv2 cells covers ~17-23 cm of table, so a grocery item is a
          fraction of one cell. A ridge probe from the stored tokens to the demo's own EE / object position says
          how much position survives the pooling, and the same probe on the UNPOOLED 16x16 grid says how much
          the pooling threw away. That difference is the ceiling any resampler or trained CNN could recover.
      (b) the tokenizer is trained on full episodes and 48-frame crops but DEPLOYED on 4 isolated patches
          (03_train.py's l_body and Policy.plan both decode a 4-patch window with no preceding context). If the
          isolated-decode RMSE is far above the 0.863 deg the gate measured, the executed stream is worse than
          the gate ever reported.

    WORK_DIR=... python s0_probe.py            # writes $WORK_DIR/results/s0_probe.json
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import ardy_vla as A
from ardy_vla import log

out = {}
D = A.load_data(vision=True)
pr = D.pr

# ---------------------------------------------------------------- 1. vision probe
# Target: the EE position at the frame each vision token was taken from, and (as a proxy for object position that
# needs no BDDL parsing) the EE position at the moment the gripper first closes in that demo -- i.e. where the
# object is. A probe that cannot find the grasp point cannot be guiding the arm to it.
log("vision probe: building frame index")
grip_cmd = (pr["actions"][:, -1] > 0).astype(np.int8)
rows = []
for e in range(D.E):
    s, T = int(D.ep_start[e]), int(D.ep_len[e])
    c = np.where(grip_cmd[s:s + T] > 0)[0]
    if len(c) == 0: continue
    rows.append((e, s, T, s + int(c[0])))                       # first close = the grasp frame
log(f"  {len(rows)} episodes with a grasp frame")

rng = np.random.default_rng(0)
sel = rng.permutation(len(rows))
tr_rows = [rows[i] for i in sel[: int(0.8 * len(sel))]]
va_rows = [rows[i] for i in sel[int(0.8 * len(sel)):]]


def feats(rs, which, sample_per_ep=4):
    """Stored (pooled) tokens at a few frames per episode, with the two targets."""
    X, Y_now, Y_grasp = [], [], []
    for (e, s, T, gf) in rs:
        fr = np.unique(np.clip(np.linspace(0, T - 1, sample_per_ep).astype(int), 0, T - 1)) + s
        v = which[fr].float().reshape(len(fr), -1).numpy()
        X.append(v); Y_now.append(pr["ee_pos"][fr]); Y_grasp.append(np.repeat(pr["ee_pos"][gf][None], len(fr), 0))
    return np.concatenate(X), np.concatenate(Y_now), np.concatenate(Y_grasp)


def ridge(Xtr, Ytr, Xva, Yva, lam=1.0):
    Xtr = np.concatenate([Xtr, np.ones((len(Xtr), 1), np.float32)], 1)
    Xva = np.concatenate([Xva, np.ones((len(Xva), 1), np.float32)], 1)
    A_ = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1], dtype=np.float64)
    W = np.linalg.solve(A_, Xtr.T @ Ytr)
    pred = Xva @ W
    return float(np.linalg.norm(pred - Yva, axis=-1).mean() * 100)          # cm


for cam, tokens in (("agentview", D.vis_a), ("wrist", D.vis_w)):
    Xtr, Ytr, Gtr = feats(tr_rows, tokens); Xva, Yva, Gva = feats(va_rows, tokens)
    out[f"probe_{cam}_ee_now_cm"] = ridge(Xtr, Ytr, Xva, Yva)
    out[f"probe_{cam}_grasp_point_cm"] = ridge(Xtr, Gtr, Xva, Gva)
    log(f"  {cam}: EE-now {out[f'probe_{cam}_ee_now_cm']:.2f} cm | grasp-point {out[f'probe_{cam}_grasp_point_cm']:.2f} cm  (n_val={len(Xva)})")

# chance level: predict the training mean
_, Ytr, Gtr = feats(tr_rows, D.vis_a); _, Yva, Gva = feats(va_rows, D.vis_a)
out["chance_ee_now_cm"] = float(np.linalg.norm(Yva - Ytr.mean(0), axis=-1).mean() * 100)
out["chance_grasp_point_cm"] = float(np.linalg.norm(Gva - Gtr.mean(0), axis=-1).mean() * 100)
log(f"  chance (predict the mean): EE-now {out['chance_ee_now_cm']:.2f} cm | grasp-point {out['chance_grasp_point_cm']:.2f} cm")

# the same probe from the UNPOOLED grid, on a subset of frames re-encoded on the fly: what pooling costs
try:
    import h5py
    enc = A.OnlineEncoder(D.meta)
    vis = enc.vis
    sub_tr, sub_va = tr_rows[:40], va_rows[:20]

    @torch.no_grad()
    def full_grid(u8):
        x = torch.from_numpy(np.ascontiguousarray(u8)).to(A.DEVICE)
        if A.FLIP_180: x = torch.flip(x, dims=(1, 2))
        x = x.permute(0, 3, 1, 2).half().div_(255.0)
        x = torch.nn.functional.interpolate(x, size=(A.IMG_RES, A.IMG_RES), mode="bilinear", align_corners=False, antialias=True)
        h = vis(pixel_values=(x - enc.mean) / enc.std).last_hidden_state[:, 1:]      # (B, 256, 384) unpooled
        return h.float().cpu().numpy()

    files = {}
    def frames_of(e, fr):
        path, dn = D.meta["tasks"][int(D.ep_task[e])]["file"], None
        suite = D.meta["tasks"][int(D.ep_task[e])]["suite"]
        full = [p for p in A.libero_files(suite) if os.path.basename(p) == path][0]
        f = files.setdefault(full, h5py.File(full, "r"))
        demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
        k = int(np.where(np.where(D.ep_task == D.ep_task[e])[0] == e)[0][0])
        return f["data"][demos[k]]["obs"]["agentview_rgb"][fr]

    def feats_full(rs, n=3):
        X, Y = [], []
        for (e, s, T, gf) in rs:
            fr = np.unique(np.clip(np.linspace(0, T - 1, n).astype(int), 0, T - 1))
            X.append(full_grid(frames_of(e, fr)).reshape(len(fr), -1)); Y.append(pr["ee_pos"][s + fr])
        return np.concatenate(X), np.concatenate(Y)

    t0 = time.time(); Xtr, Ytr = feats_full(sub_tr); Xva, Yva = feats_full(sub_va)
    out["probe_agentview_FULLGRID_ee_now_cm"] = ridge(Xtr, Ytr, Xva, Yva, lam=10.0)
    # the pooled probe on exactly the same subset, so the comparison is like for like
    Xtr2, Ytr2, _ = feats(sub_tr, D.vis_a, sample_per_ep=3); Xva2, Yva2, _ = feats(sub_va, D.vis_a, sample_per_ep=3)
    out["probe_agentview_POOLED_same_subset_cm"] = ridge(Xtr2, Ytr2, Xva2, Yva2, lam=10.0)
    log(f"  agentview on the same subset: pooled 4x4 {out['probe_agentview_POOLED_same_subset_cm']:.2f} cm "
        f"vs full 16x16 {out['probe_agentview_FULLGRID_ee_now_cm']:.2f} cm  ({time.time()-t0:.0f}s)")
    for f in files.values(): f.close()
except Exception as ex:
    log(f"  full-grid probe skipped: {type(ex).__name__}: {ex}")
    out["probe_full_grid_error"] = f"{type(ex).__name__}: {ex}"

# ---------------------------------------------------------------- 2. tokenizer in deployment mode
log("tokenizer: full-episode context vs the isolated 4-patch decode the policy actually runs")
std_jp = D.body_std[:D.NJ].cpu().numpy(); deg = 180.0 / np.pi
errs = {"full_episode": [], "isolated_4patch": [], "crop_128f": []}
with torch.no_grad():
    for e in D.val_eps[:60]:
        e = int(e); s, T = int(D.ep_start[e]), int(D.ep_len[e])
        body = D.body_pad[D.pf_start[e]: D.pf_start[e] + D.pf_len[e]][None]
        with torch.autocast(**A.AMP):
            zq, _ = D.tok.encode(body); rec_full = D.tok.decode(zq).float()
        errs["full_episode"].append((((rec_full[0, :T, :D.NJ] - body[0, :T, :D.NJ]) * D.body_std[:D.NJ]) ** 2).mean().sqrt().item() * deg)
        # isolated 4-patch windows, exactly as 03_train.py's l_body and Policy.plan decode them
        n_p = D.p_len[e]
        for w in range(0, max(1, n_p - A.C), max(1, (n_p - A.C) // 4 or 1)):
            with torch.autocast(**A.AMP):
                rec = D.tok.decode(zq[:, w:w + A.C]).float()
            tgt = body[:, w * A.P:(w + A.C) * A.P]
            if rec.shape[1] != tgt.shape[1]: continue
            errs["isolated_4patch"].append((((rec[0, :, :D.NJ] - tgt[0, :, :D.NJ]) * D.body_std[:D.NJ]) ** 2).mean().sqrt().item() * deg)
        # a 128-frame crop re-based to position 0, as Policy._history_tokens encodes online
        if T > 140:
            crop = body[:, 64:192]
            with torch.autocast(**A.AMP):
                zc, _ = D.tok.encode(crop); rec_c = D.tok.decode(zc).float()
            errs["crop_128f"].append((((rec_c[0, :, :D.NJ] - crop[0, :, :D.NJ]) * D.body_std[:D.NJ]) ** 2).mean().sqrt().item() * deg)
for k, v in errs.items():
    out[f"tok_{k}_rmse_deg"] = float(np.mean(v)) if v else None
    log(f"  {k:16s} {out[f'tok_{k}_rmse_deg']}")
out["tok_gate_isolated_under_1p5"] = bool(out["tok_isolated_4patch_rmse_deg"] is not None and out["tok_isolated_4patch_rmse_deg"] < 1.5)

json.dump(out, open(A.RES_DIR / "s0_probe.json", "w"), indent=1)
log(f"stage 0 probe -> {A.RES_DIR / 's0_probe.json'}")
print(json.dumps(out, indent=1))
