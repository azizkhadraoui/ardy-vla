#!/usr/bin/env python
"""Adapt the vision encoder to this task, then re-extract features for the policy to train on.

WHY NOT END-TO-END
    Fine-tuning DINOv2 inside the policy loop means a 224px forward and backward pass on two cameras for every
    element of a 256-wide batch -- roughly 2 s a step, which is 17 h for 30k steps and not comparable with any
    run we have. This does the same work in the cheap order: adapt the encoder on a small auxiliary objective,
    freeze it, re-extract the pooled tokens once, then train the policy exactly as before at full batch.

WHAT IT ADAPTS TO
    The measured deficiency is localisation: a ridge probe places the demo's own grasp point 5.15 cm from the
    4x4-pooled frozen tokens (vision_pooling_probe.json) against ~4 cm objects, and the error only falls to
    4.37 cm with 8x8 pooling, so the representation is the limit rather than the pooling. So the objective is
    exactly that quantity -- regress the end-effector position (and the gripper width, which says whether the
    hand is holding something) from the image, with the last FT_BLOCKS transformer blocks unfrozen.

    Reported held-out error in cm is directly comparable with the 5.15 cm the frozen encoder gives.

    WORK_DIR=... FT_BLOCKS=2 FT_STEPS=4000 python vis_ft.py            -> ckpt/vis_ft.pt
    WORK_DIR=... FT_EXTRACT=1 python vis_ft.py                         -> data/vision_{agentview,wrist}_ft.npy
    then: VIS_SUFFIX=_ft python 03_train.py
"""
import os, sys, json, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import ardy_vla as A
from ardy_vla import log, DEVICE, POOL, IMG_RES, FLIP_180

FT_BLOCKS = int(os.environ.get("FT_BLOCKS", 2))      # trailing transformer blocks to unfreeze
FT_STEPS = int(os.environ.get("FT_STEPS", 4000)); FT_B = int(os.environ.get("FT_B", 32))
FT_LR = float(os.environ.get("FT_LR", 1e-5)); FT_JITTER = float(os.environ.get("FT_JITTER", 0.2))
EXTRACT = os.environ.get("FT_EXTRACT", "0") == "1"
CK = A.CKPT_DIR / (f"vis_ft_b{FT_BLOCKS}.pt" if FT_BLOCKS != 2 else "vis_ft.pt")

meta = json.load(open(A.DATA / "meta.json")); pr = np.load(A.DATA / "proprio.npz")
ep_start, ep_len, ep_task = pr["episode_start"], pr["episode_len"], pr["episode_task"]
val_mask = np.load(A.TOK_DIR / "patch_index.npz")["val_mask"].astype(bool)
raw_a = np.load(A.DATA / "raw_agentview.npy", mmap_mode="r"); raw_w = np.load(A.DATA / "raw_wrist.npy", mmap_mode="r")
ee, gr = pr["ee_pos"].astype(np.float32), pr["gripper"].astype(np.float32)
tr_frames = np.concatenate([np.arange(ep_start[e], ep_start[e] + ep_len[e]) for e in np.where(~val_mask)[0]])
va_frames = np.concatenate([np.arange(ep_start[e], ep_start[e] + ep_len[e]) for e in np.where(val_mask)[0]])
m_, s_ = ee[tr_frames].mean(0), ee[tr_frames].std(0) + 1e-6
log(f"adapting on {len(tr_frames)} train / {len(va_frames)} val frames; EE std {np.round(s_ * 100, 1)} cm")

from transformers import AutoModel
name = meta["vision_encoder"]
_m = AutoModel.from_pretrained(name); vis = getattr(_m, "vision_model", _m).to(DEVICE).float()
MEAN = torch.tensor((0.485, 0.456, 0.406), device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225), device=DEVICE).view(1, 3, 1, 1)
blocks = vis.encoder.layer
for p_ in vis.parameters(): p_.requires_grad_(False)
# FT_BLOCKS=0 is the control: same head, same schedule, encoder entirely frozen. It separates the encoder
# adaptation from the head capacity -- the 5.15 cm reference is a RIDGE probe, so comparing it with an MLP head
# on adapted features would credit the encoder with whatever the bigger head alone buys. NB blocks[-0:] is
# every block, so this has to be an explicit branch rather than a slice.
for blk in (blocks[-FT_BLOCKS:] if FT_BLOCKS > 0 else []):
    for p_ in blk.parameters(): p_.requires_grad_(True)
n_tr = sum(p_.numel() for p_ in vis.parameters() if p_.requires_grad)
log(f"{name}: {len(blocks)} blocks, last {FT_BLOCKS} trainable ({n_tr / 1e6:.1f}M of {sum(p_.numel() for p_ in vis.parameters()) / 1e6:.1f}M)")


def embed(u8, jitter=0.0):
    x = u8.permute(0, 3, 1, 2).float().div(255.0)
    if FLIP_180: x = torch.flip(x, dims=(2, 3))
    if jitter > 0:
        b = 1 + jitter * (2 * torch.rand(x.shape[0], 1, 1, 1, device=x.device) - 1)
        g = 1 + jitter * (2 * torch.rand(x.shape[0], 3, 1, 1, device=x.device) - 1)
        mu = x.mean(dim=(1, 2, 3), keepdim=True); x = (((x - mu) * b + mu) * g).clamp(0, 1)
    x = F.interpolate(x, size=(IMG_RES, IMG_RES), mode="bilinear", align_corners=False, antialias=True)
    h = vis(pixel_values=(x - MEAN) / STD).last_hidden_state
    cls, patches = h[:, :1], h[:, 1:]
    g_ = int(round(patches.shape[1] ** 0.5)); patches = patches.transpose(1, 2).reshape(patches.shape[0], -1, g_, g_)
    return torch.cat([cls, F.adaptive_avg_pool2d(patches, POOL).flatten(2).transpose(1, 2)], 1)


if not EXTRACT:
    head = nn.Sequential(nn.Linear(2 * (1 + POOL * POOL) * vis.config.hidden_size, 512), nn.SiLU(), nn.Linear(512, 4)).to(DEVICE)
    trainable = [p_ for p_ in vis.parameters() if p_.requires_grad]
    opt = torch.optim.AdamW((trainable if trainable else []) + list(head.parameters()),
                            lr=FT_LR if trainable else max(FT_LR, 3e-4), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 200) * 0.5 * (1 + math.cos(math.pi * min(s, FT_STEPS) / FT_STEPS)))

    def batch(frames, n, jitter=0.0, gen=None):
        idx = np.sort(frames[(np.random.randint(0, len(frames), n) if gen is None else
                              torch.randint(0, len(frames), (n,), generator=gen).numpy())])
        a = torch.from_numpy(np.ascontiguousarray(raw_a[idx])).to(DEVICE)
        w = torch.from_numpy(np.ascontiguousarray(raw_w[idx])).to(DEVICE)
        f = torch.cat([embed(a, jitter).flatten(1), embed(w, jitter).flatten(1)], -1)
        y = torch.from_numpy(np.concatenate([(ee[idx] - m_) / s_, (gr[idx, :1] - gr[idx, 1:2])], -1)).to(DEVICE)
        return f, y

    @torch.no_grad()
    def val(nb=20):
        vis.eval(); head.eval(); gen = torch.Generator().manual_seed(7); errs = []
        for _ in range(nb):
            f, y = batch(va_frames, FT_B, 0.0, gen); p = head(f)
            errs.append(np.linalg.norm(((p[:, :3] - y[:, :3]).cpu().numpy() * s_), axis=-1) * 100)
        vis.train(); head.train(); return float(np.concatenate(errs).mean())

    log(f"baseline (untrained head, frozen blocks): held-out EE error {val():.2f} cm")
    t0 = time.time(); hist = []
    for step in range(1, FT_STEPS + 1):
        f, y = batch(tr_frames, FT_B, FT_JITTER); p = head(f)
        loss = F.smooth_l1_loss(p[:, :3], y[:, :3]) + 0.1 * F.smooth_l1_loss(p[:, 3:], y[:, 3:])
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([p_ for p_ in vis.parameters() if p_.requires_grad] + list(head.parameters()), 1.0)
        opt.step(); sched.step()
        if step % 500 == 0 or step == FT_STEPS:
            e = val(); hist.append(dict(step=step, loss=float(loss), ee_cm=e))
            log(f"step {step:5d}  loss {float(loss):.4f}  held-out EE {e:.2f} cm  ({time.time() - t0:.0f}s)")
    torch.save(dict(vis=vis.state_dict(), ft_blocks=FT_BLOCKS, steps=FT_STEPS, history=hist, ee_cm=hist[-1]["ee_cm"]), CK)
    json.dump(dict(ft_blocks=FT_BLOCKS, steps=FT_STEPS, batch=FT_B, lr=FT_LR, jitter=FT_JITTER, history=hist),
              open(A.RES_DIR / f"vis_ft_b{FT_BLOCKS}.json", "w"), indent=1)
    log(f"saved {CK.name}: held-out EE {hist[-1]['ee_cm']:.2f} cm "
        f"(frozen 4x4 probe gives 5.15 cm to the grasp point; lower is better localisation)")
else:
    ck = torch.load(CK, map_location=DEVICE, weights_only=False)
    vis.load_state_dict(ck["vis"]); vis.eval()
    log(f"re-extracting with the adapted encoder ({ck['steps']} steps, held-out EE {ck['ee_cm']:.2f} cm)")
    n = int(ep_len.sum()); NT = 1 + POOL * POOL; DV = vis.config.hidden_size
    for cam, src in (("agentview", raw_a), ("wrist", raw_w)):
        dst = np.lib.format.open_memmap(A.DATA / f"vision_{cam}_ft.npy", mode="w+", dtype=np.float16, shape=(n, NT, DV))
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, n, 256):
                j = min(i + 256, n)
                x = torch.from_numpy(np.ascontiguousarray(src[i:j])).to(DEVICE)
                dst[i:j] = embed(x).half().cpu().numpy()
                if i % 51200 == 0: log(f"  {cam}: {i}/{n} ({time.time() - t0:.0f}s)")
        dst.flush(); log(f"wrote vision_{cam}_ft.npy ({n} frames, {NT} tokens, {DV} dims)")
