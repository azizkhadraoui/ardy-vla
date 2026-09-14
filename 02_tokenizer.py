#!/usr/bin/env python
"""
02_tokenizer.py — the causal FSQ motion tokenizer for the joint-space stream, trained once on all suites.

WHY FULL EPISODES
    The first tokenizer trained on 48-frame crops and its learned positional embeddings beyond 12 patches
    were never seen; full-episode reconstruction was 7.8 deg while crop-level validation said 1.3. Half of
    every batch is now a full end-padded episode with a loss mask, so every position the denoiser will
    ever ask the decoder to handle has been trained.

WHAT TO READ
    The final line: held-out per-joint RMSE in degrees on FULL episodes. Anything above ~2 deg on a joint
    is a tokenizer problem that every downstream body metric inherits. Target: mean < 1 deg, no joint > 1.5.

WHAT IT PRODUCES ($WORK_DIR/tokenizer)
    tokenizer.pt, latents.npy (patch-level FSQ latents for every episode), patch_index.npz (+ the val split),
    tokenizer_eval.json

    WORK_DIR=... TOK_STEPS=12000 python 02_tokenizer.py
"""
import os, sys, json, math, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import ardy_vla as A
from ardy_vla import log, DEVICE, AMP, P

if (A.TOK_DIR / "latents.npy").exists() and os.environ.get("FORCE", "0") != "1":
    log("tokenizer already trained; set FORCE=1 to redo"); sys.exit(0)
A.seed_all(0)
pr = np.load(A.DATA / "proprio.npz"); body = np.concatenate([pr["joint_pos"], pr["joint_vel"]], 1).astype(np.float32); N, IN = body.shape
ep_start, ep_len, ep_task = pr["episode_start"], pr["episode_len"], pr["episode_task"]; E = len(ep_len)
val_mask = np.zeros(E, bool)
for t in np.unique(ep_task): val_mask[np.where(ep_task == t)[0][-A.HELDOUT_PER_TASK:]] = True     # last 5 demos of every task held out
tr, va = np.where(~val_mask)[0], np.where(val_mask)[0]
trf = np.concatenate([np.arange(ep_start[e], ep_start[e] + ep_len[e]) for e in tr]); mean, std = body[trf].mean(0), body[trf].std(0) + 1e-6
body_t = torch.from_numpy((body - mean) / std).to(DEVICE); JP = slice(0, IN // 2); L_MAX = int(math.ceil(ep_len.max() / P) * P)

def sample(eps, B, full_frac=0.5):
    e = np.random.choice(eps, B); out = torch.zeros(B, L_MAX, IN, device=DEVICE); mask = torch.zeros(B, L_MAX, 1, device=DEVICE)
    for i, ei in enumerate(e):
        s, T = ep_start[ei], ep_len[ei]
        if i < int(B * full_frac) or T <= A.TOK_SEG: seg = body_t[s:s + T]
        else: st = s + np.random.randint(0, T - A.TOK_SEG + 1); seg = body_t[st:st + A.TOK_SEG]
        Ls = seg.shape[0]; Lp = int(math.ceil(Ls / P) * P); out[i, :Lp] = torch.cat([seg, seg[-1:].expand(Lp - Ls, -1)], 0); mask[i, :Ls] = 1.0
    return out, mask

model = A.MotionTokenizer(IN, P, A.LEVELS, A.G, A.TOK_D, A.TOK_LAYERS, A.TOK_HEADS, max_patches=L_MAX // P + 4, dropout=A.TOK_DROPOUT).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=A.TOK_LR, betas=(0.9, 0.99), weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 500) * 0.5 * (1 + math.cos(math.pi * min(s, A.TOK_STEPS) / A.TOK_STEPS)))
scaler = torch.amp.GradScaler("cuda", enabled=A.USE_AMP); std_jp = torch.from_numpy(std[JP]).to(DEVICE)
w = torch.ones(IN, device=DEVICE); w[IN // 2:] = A.TOK_VEL_DIM_W
def losses(x, xh, m):
    rec = (((xh - x) ** 2 * w) * m).sum() / (m.sum() * IN); mv = m[:, 1:] * m[:, :-1]
    vel = (((xh[:, 1:, JP] - xh[:, :-1, JP]) - (x[:, 1:, JP] - x[:, :-1, JP])) ** 2 * mv).sum() / (mv.sum() * (IN // 2)); return rec, vel
n_par = sum(p.numel() for p in model.parameters()) / 1e6
log(f"tokenizer {n_par:.2f}M params, latent {model.latent_dim}, {len(tr)} train / {len(va)} val episodes, L_MAX {L_MAX}")
A.wandb_init("tokenizer", config=dict(params_M=round(n_par, 2), latent_dim=model.latent_dim, n_train_ep=len(tr), n_val_ep=len(va), L_MAX=L_MAX))
hist = []; t0 = time.time(); model.train()
for step in range(1, A.TOK_STEPS + 1):
    x, m = sample(tr, A.TOK_BATCH)
    with torch.autocast(**AMP):
        xh, _, _ = model(x + A.TOK_NOISE * torch.randn_like(x)); rec, vel = losses(x, xh.float(), m); loss = rec + A.TOK_W_VEL * vel
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
    if step % 500 == 0 or step == 1:
        model.eval()
        with torch.no_grad(), torch.autocast(**AMP):
            xv, mv_ = sample(va, 128, full_frac=1.0); xvh, _, _ = model(xv)
            deg = ((((xvh.float()[..., JP] - xv[..., JP]) * std_jp) ** 2 * mv_).sum() / (mv_.sum() * (IN // 2))).sqrt().item() * 180 / math.pi
        model.train(); hist.append(dict(step=step, loss=loss.item(), val_deg=deg)); log(f"  step {step:5d} loss {loss.item():.4f} val joint RMSE {deg:.2f} deg  {time.time()-t0:.0f}s")
        A.wandb_log({"tok/step": step, "tok/loss": loss.item(), "tok/rec": rec.item(), "tok/vel": vel.item(),
                     "tok/val_joint_rmse_deg": deg, "tok/lr": sched.get_last_lr()[0], "tok/secs": time.time() - t0})

model.eval(); lat_l, p_start, p_len, per_joint = [], [], [], []; npch = 0
with torch.no_grad():
    for e in range(E):
        s, T = ep_start[e], ep_len[e]; seg = body_t[s:s + T]; pad = (-T) % P
        if pad: seg = torch.cat([seg, seg[-1:].expand(pad, -1)], 0)
        with torch.autocast(**AMP): zq, _ = model.encode(seg[None]); xh = model.decode(zq)
        if val_mask[e]: per_joint.append((((xh.float()[0, :T, JP] - seg[None][0, :T, JP]) * std_jp) ** 2).mean(0).sqrt() * 180 / math.pi)
        lat_l.append(zq[0].float().cpu().numpy().astype(np.float16)); p_start.append(npch); p_len.append(zq.shape[1]); npch += zq.shape[1]
pj = torch.stack(per_joint).mean(0); ev = dict(per_joint_rmse_deg=[round(v, 3) for v in pj.tolist()], mean_rmse_deg=round(pj.mean().item(), 3), history=hist)
torch.save(dict(state_dict=model.state_dict(), mean=mean, std=std, config=dict(in_dim=IN, P=P, levels=A.LEVELS, groups=A.G, d=A.TOK_D, n_layers=A.TOK_LAYERS, n_heads=A.TOK_HEADS,
                                                                                max_patches=model.encoder.pos.shape[1], dropout=A.TOK_DROPOUT)), A.TOK_DIR / "tokenizer.pt")
np.save(A.TOK_DIR / "latents.npy", np.concatenate(lat_l))
np.savez(A.TOK_DIR / "patch_index.npz", patch_start=np.array(p_start), patch_len=np.array(p_len), episode_task=ep_task, val_mask=val_mask, P=P)
json.dump(ev, open(A.TOK_DIR / "tokenizer_eval.json", "w"), indent=2)
A.wandb_summary(dict(mean_rmse_deg=ev["mean_rmse_deg"], max_joint_rmse_deg=round(pj.max().item(), 3), n_patches=npch,
                     secs=round(time.time() - t0), **{f"joint{j}_rmse_deg": v for j, v in enumerate(ev["per_joint_rmse_deg"])}), prefix="tok/")
A.wandb_table("tok/per_joint_rmse", [dict(joint=j, rmse_deg=v) for j, v in enumerate(ev["per_joint_rmse_deg"])])
A.wandb_finish()
log(f"stage 2 done: held-out FULL-episode per-joint RMSE (deg) {ev['per_joint_rmse_deg']}  mean {ev['mean_rmse_deg']:.2f}   {'OK' if pj.max() < 2 else '!! a joint exceeds 2 deg'}")
