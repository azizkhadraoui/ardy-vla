#!/usr/bin/env python
"""The vision experiment: does finer spatial pooling of the frozen DINOv2 features recover object localisation?

WHY
    s0_probe.json: from the stored 4x4-pooled agentview tokens a ridge probe places the end-effector to 3.40 cm
    but the demo's own GRASP POINT -- where the object is -- only to 11.08 cm, against 18.25 cm for predicting the
    mean. libero_object items are ~4 cm. That is the object suite's 0.020 in one number.
    The unpooled comparison in that run is not usable: 98,304 features against 120 training samples.

WHAT THIS DOES
    mode=probe    re-encodes a large, properly-powered sample of frames ONCE and probes the grasp point at
                  1x1, 2x2, 4x4 (the current setting), 8x8 and 16x16 pooling from the same features, so the
                  comparison isolates pooling and nothing else.
    mode=extract  writes vision_{agentview,wrist}_p{POOL}.npy for the whole dataset at the chosen POOL, in the
                  same layout load_data expects (VIS_SUFFIX selects them).

    WORK_DIR=... MODE=probe [N_EP=400] python vision_experiment.py
    WORK_DIR=... MODE=extract POOL=8 python vision_experiment.py
"""
import os, sys, json, time, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
import ardy_vla as A
from ardy_vla import log

MODE = os.environ.get("MODE", "probe")
meta = json.load(open(A.DATA / "meta.json"))
pr = np.load(A.DATA / "proprio.npz")
ep_start, ep_len, ep_task = pr["episode_start"], pr["episode_len"], pr["episode_task"]
E = len(ep_len)

enc = A.OnlineEncoder(meta)


@torch.no_grad()
def grid_tokens(u8):
    """CLS + the full 16x16 patch grid, before any pooling. (B, 1+256, 384)."""
    x = torch.from_numpy(np.ascontiguousarray(u8)).to(A.DEVICE)
    if x.dim() == 3: x = x[None]
    if A.FLIP_180: x = torch.flip(x, dims=(1, 2))
    x = x.permute(0, 3, 1, 2).half().div_(255.0)
    x = F.interpolate(x, size=(A.IMG_RES, A.IMG_RES), mode="bilinear", align_corners=False, antialias=True)
    h = enc.vis(pixel_values=(x - enc.mean) / enc.std).last_hidden_state
    return h[:, :1], h[:, 1:]                       # cls, patches (B, 256, D)


def pool_to(patches, k):
    """(B,256,D) -> (B, k*k, D) by adaptive average pooling on the 16x16 grid, exactly as 01 does."""
    B, N, Dv = patches.shape; g = int(round(N ** 0.5))
    p = patches.transpose(1, 2).reshape(B, Dv, g, g)
    return F.adaptive_avg_pool2d(p, k).flatten(2).transpose(1, 2)


def demo_frames(e, fr):
    """Raw agentview frames of episode e at frame offsets fr, from the HDF5 the dataset was built from."""
    import h5py
    t = meta["tasks"][int(ep_task[e])]
    path = [p for p in A.libero_files(t["suite"]) if os.path.basename(p) == t["file"]][0]
    k = int(np.where(np.where(ep_task == ep_task[e])[0] == e)[0][0])
    with h5py.File(path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
        return f["data"][demos[k]]["obs"]["agentview_rgb"][fr]


if MODE == "probe":
    N_EP = int(os.environ.get("N_EP", 400)); PER_EP = int(os.environ.get("PER_EP", 6))
    grip = (pr["actions"][:, -1] > 0).astype(np.int8)
    rows = []
    for e in range(E):
        s, T = int(ep_start[e]), int(ep_len[e])
        c = np.where(grip[s:s + T] > 0)[0]
        if len(c): rows.append((e, s, T, s + int(c[0])))
    rng = np.random.default_rng(0); rows = [rows[i] for i in rng.permutation(len(rows))[:N_EP]]
    log(f"probe: {len(rows)} episodes x {PER_EP} frames, one DINOv2 pass, pooled five ways")

    CLS, PATCH, Y_ee, Y_grasp = [], [], [], []
    t0 = time.time()
    for n, (e, s, T, gf) in enumerate(rows):
        fr = np.unique(np.clip(np.linspace(0, T - 1, PER_EP).astype(int), 0, T - 1))
        cls, pat = grid_tokens(demo_frames(e, fr))
        CLS.append(cls.float().cpu().numpy()); PATCH.append(pat.float().cpu().numpy())
        Y_ee.append(pr["ee_pos"][s + fr]); Y_grasp.append(np.repeat(pr["ee_pos"][gf][None], len(fr), 0))
        if n % 100 == 0: log(f"  {n}/{len(rows)} episodes  ({time.time()-t0:.0f}s)")
    CLS = np.concatenate(CLS); PATCH = np.concatenate(PATCH)
    Y_ee = np.concatenate(Y_ee).astype(np.float64); Y_grasp = np.concatenate(Y_grasp).astype(np.float64)
    n = len(CLS); ntr = int(0.8 * n); idx = rng.permutation(n); tr, va = idx[:ntr], idx[ntr:]
    log(f"  {n} frames, {ntr} train / {n-ntr} val")

    def ridge(X, Y, lam):
        Xt = np.concatenate([X[tr], np.ones((len(tr), 1))], 1)
        Xv = np.concatenate([X[va], np.ones((len(va), 1))], 1)
        W = np.linalg.solve(Xt.T @ Xt + lam * np.eye(Xt.shape[1]), Xt.T @ Y[tr])
        return float(np.linalg.norm(Xv @ W - Y[va], axis=-1).mean() * 100)

    out = {"n_frames": n, "n_train": ntr,
           "chance_ee_cm": float(np.linalg.norm(Y_ee[va] - Y_ee[tr].mean(0), axis=-1).mean() * 100),
           "chance_grasp_cm": float(np.linalg.norm(Y_grasp[va] - Y_grasp[tr].mean(0), axis=-1).mean() * 100)}
    P = torch.from_numpy(PATCH)
    print(f"\n{'pooling':>10s}{'tokens':>8s}{'features':>10s}{'EE now cm':>12s}{'grasp pt cm':>14s}")
    for k in (1, 2, 4, 8, 16):
        X = np.concatenate([CLS.reshape(n, -1), pool_to(P, k).reshape(n, -1).numpy()], 1)
        lam = max(1.0, X.shape[1] / 50.0)            # scale ridge with dimensionality, else wide X is unfair
        ee, gp = ridge(X, Y_ee, lam), ridge(X, Y_grasp, lam)
        out[f"pool{k}_ee_cm"], out[f"pool{k}_grasp_cm"] = ee, gp
        print(f"{f'{k}x{k}':>10s}{1+k*k:>8d}{X.shape[1]:>10d}{ee:>12.2f}{gp:>14.2f}")
    print(f"{'chance':>10s}{'':>8s}{'':>10s}{out['chance_ee_cm']:>12.2f}{out['chance_grasp_cm']:>14.2f}")
    json.dump(out, open(A.RES_DIR / "vision_pooling_probe.json", "w"), indent=1)
    log(f"-> {A.RES_DIR / 'vision_pooling_probe.json'}")

elif MODE == "extract":
    K = A.POOL
    NT = 1 + K * K
    log(f"extracting vision at POOL={K} ({NT} tokens/camera) for {int(ep_len.sum())} frames")
    import h5py
    n_frames = int(ep_len.sum()); DV = meta["vision_dim"]
    outs = {}
    for cam in ("agentview", "wrist"):
        outs[cam] = np.lib.format.open_memmap(A.DATA / f"vision_{cam}_p{K}.npy", mode="w+",
                                              dtype=np.float16, shape=(n_frames, NT, DV))
    KEY = {"agentview": "agentview_rgb", "wrist": "eye_in_hand_rgb"}
    t0 = time.time(); done = 0
    for ti, t in enumerate(meta["tasks"]):
        path = [p for p in A.libero_files(t["suite"]) if os.path.basename(p) == t["file"]][0]
        eps = np.where(ep_task == ti)[0]
        with h5py.File(path, "r") as f:
            demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
            for k, e in enumerate(eps):
                s, T = int(ep_start[e]), int(ep_len[e]); obs = f["data"][demos[k]]["obs"]
                for cam in ("agentview", "wrist"):
                    arr = obs[KEY[cam]]
                    for b in range(0, T, 256):
                        sl = slice(b, min(b + 256, T))
                        cls, pat = grid_tokens(arr[sl])
                        tok = torch.cat([cls, pool_to(pat, K)], 1).half().cpu().numpy()
                        outs[cam][s + sl.start: s + sl.stop] = tok
                done += 1
        if ti % 5 == 0: log(f"  task {ti}/{len(meta['tasks'])}, {done} episodes, {time.time()-t0:.0f}s")
    for cam in outs: outs[cam].flush()
    meta[f"vision_tokens_p{K}"] = NT
    json.dump(meta, open(A.DATA / "meta.json", "w"), indent=2)
    log(f"wrote vision_*_p{K}.npy ({NT} tokens/camera) in {(time.time()-t0)/60:.0f} min")
else:
    raise SystemExit(f"unknown MODE {MODE}")
