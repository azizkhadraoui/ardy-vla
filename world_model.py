#!/usr/bin/env python
"""Pretrain the WorldMemory GRU as a latent forward model on the demonstrations' own transitions.

WHY
    Nowhere in the pipeline is the model trained on what its actions do to the scene: the history tokens describe
    the arm, the vision tokens describe the present, and the loss only ever asks for the next demo action. A
    forward model is the cheapest place to put that missing supervision. It needs no reward and no simulator:
    every demo is a chain of (scene, action, next scene) transitions -- 338k frames, 85k patches -- and the
    frozen DINOv2 features are a fixed target the policy already consumes.

WHAT IS LEARNED
    z_i     mean-pooled features of both cameras (2 x 384) at the frame before history patch i
    a_i     the explicit EE stream of that patch (4 frames x (pos, rot6d, gripper) = 44), exactly what the
            history token carries and what the denoiser predicts, so the same module can later score the policy's
            own plans
    target  z_{i+1} - z_i in normalised feature space, teacher-forced over the H=32 patches (6.4 s) plus an
            open-loop rollout over the last WM_ROLL patches where the model feeds its own prediction back

HOW IT IS JUDGED (held-out demos)
    r2_1step        1 - MSE / MSE of "nothing changes"; > 0 means the action explains part of the scene change
    r2_roll_{h}     the same at horizon h patches open loop, against "the scene stays as it was"
    action_ratio    MSE with the actions shuffled across the batch / MSE with the true actions: 1.0 means the
                    model ignores the action and only extrapolates the scene; the memory token is then no better
                    than VisionMemory. This is the number that decides whether stage 2 is worth running.

    WORK_DIR=... WM_STEPS=20000 WM_SEED=0 python world_model.py       -> ckpt/wm_s0.pt, results/world_model_s0.json
    then: MEM_WM=1 WM_CKPT=$WORK_DIR/ckpt/wm_s0.pt python 03_train.py
"""
import os, sys, json, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
import ardy_vla as A
from ardy_vla import log, DEVICE, H, P

assert A.VIS_MODE == "feat", "the forward model predicts the frozen features; run with VIS_MODE=feat"
STEPS = int(os.environ.get("WM_STEPS", 20000)); B = int(os.environ.get("WM_B", 256)); SEED = int(os.environ.get("WM_SEED", 0))
LR = float(os.environ.get("WM_LR", 3e-4)); ROLL = int(os.environ.get("WM_ROLL", 8)); ROLL_W = float(os.environ.get("WM_ROLL_W", 1.0))
D_MODEL = int(os.environ.get("WM_D", A.D_MODEL)); VAL_EVERY = int(os.environ.get("WM_VAL_EVERY", 1000)); TAG = os.environ.get("WM_TAG", "")
torch.manual_seed(SEED); np.random.seed(SEED)

D = A.load_data(vision=True)
dst_ck, dst_js = A.CKPT_DIR / f"wm_s{SEED}{TAG}.pt", A.RES_DIR / f"world_model_s{SEED}{TAG}.json"
model = A.WorldMemory(2 * D.DV, D.EXP, D_MODEL).to(DEVICE)
log(f"world model: dz={2 * D.DV} da={D.EXP} d={D_MODEL}, {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, "
    f"{len(D.train_eps)} train / {len(D.val_eps)} val episodes, {STEPS} steps x {B}")


def gather(eps, n, gen=None):
    """A batch of (z (n,H+1,dz), a (n,H,da), pad (n,H)) at a random window start w in [1, Np] of random episodes."""
    e = torch.as_tensor(np.random.choice(eps, n) if gen is None else eps[torch.randint(0, len(eps), (n,), generator=gen, device=DEVICE).cpu().numpy()], device=DEVICE)
    Np = D.p_len_t[e]
    u = torch.rand(n, device=DEVICE) if gen is None else torch.rand(n, device=DEVICE, generator=gen)
    w = 1 + (u * (Np - 1)).long()                       # w in [1, Np-1]: at least one real history patch, current frame inside the episode
    z = A.wm_gather(D, e, w)
    hmax = torch.minimum(torch.full_like(w, H), w); j = torch.arange(H, device=DEVICE)[None]; pad = j < (H - hmax)[:, None]
    ps = D.p_start_t[e]; a = D.hyb[((ps + w)[:, None] - (H - j)).clamp(min=0)][..., :D.EXP] * (~pad)[..., None]
    return z, a, pad


# feature statistics from training frames, stored in the module so every consumer normalises identically
with torch.no_grad():
    zs = torch.cat([gather(D.train_eps, 512)[0].reshape(-1, 2 * D.DV) for _ in range(8)])
    model.z_mean.copy_(zs.mean(0)); model.z_std.copy_(zs.std(0) + 1e-3)
    log(f"feature scale: mean |z| {zs.abs().mean():.3f}, per-dim std {model.z_std.mean():.4f} (min {model.z_std.min():.4f})")


def losses(z, a, pad, roll=True):
    tgt = model.znorm(z[:, 1:])
    pred = model.predict(z, a, pad); valid = (~pad).float()
    l1 = (((pred - tgt) ** 2).mean(-1) * valid).sum() / valid.sum()
    out = dict(l1=l1)
    if roll and ROLL > 0:
        K0 = H - ROLL
        _, h0 = model.run(z[:, :K0 + 1], a[:, :K0], pad[:, :K0])
        zr, _ = model.rollout(z[:, K0], a[:, K0:], h0); v = valid[:, K0:]
        out["lroll"] = (((zr - tgt[:, K0:]) ** 2).mean(-1) * v).sum() / v.sum(); out["zr"] = zr
    return out, tgt, pred


@torch.no_grad()
def validate(n_batches=20):
    model.eval(); gen = torch.Generator(device=DEVICE); gen.manual_seed(999)
    acc = dict(mse=0.0, base=0.0, shuf=0.0, n=0.0); roll = {h: [0.0, 0.0, 0.0] for h in (1, 2, 4, ROLL) if h <= ROLL}
    for _ in range(n_batches):
        z, a, pad = gather(D.val_eps, B, gen); out, tgt, pred = losses(z, a, pad); valid = (~pad).float()
        prev = model.znorm(z[:, :-1])
        acc["mse"] += (((pred - tgt) ** 2).mean(-1) * valid).sum().item(); acc["base"] += (((prev - tgt) ** 2).mean(-1) * valid).sum().item()
        shuf = model.predict(z, a[torch.roll(torch.arange(B, device=DEVICE), 1)], pad)
        acc["shuf"] += (((shuf - tgt) ** 2).mean(-1) * valid).sum().item(); acc["n"] += valid.sum().item()
        K0 = H - ROLL; zr = out["zr"]; z0 = prev[:, K0]
        for h in roll:
            v = valid[:, K0 + h - 1]; roll[h][0] += (((zr[:, h - 1] - tgt[:, K0 + h - 1]) ** 2).mean(-1) * v).sum().item()
            roll[h][1] += (((z0 - tgt[:, K0 + h - 1]) ** 2).mean(-1) * v).sum().item(); roll[h][2] += v.sum().item()
    model.train()
    r = dict(mse_1step=acc["mse"] / acc["n"], mse_nochange=acc["base"] / acc["n"], r2_1step=1 - acc["mse"] / acc["base"],
             action_ratio=acc["shuf"] / acc["mse"])
    for h, (m, b_, n) in roll.items(): r[f"r2_roll_{h}"] = 1 - m / b_; r[f"mse_roll_{h}"] = m / n
    return r


opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 300) * 0.5 * (1 + math.cos(math.pi * min(s, STEPS) / STEPS)))
A.wandb_init("world_model", "wm", SEED, config=dict(steps=STEPS, batch=B, lr=LR, roll=ROLL, roll_w=ROLL_W, d=D_MODEL))
t0 = time.time(); hist = []; val_hist = []
for step in range(1, STEPS + 1):
    z, a, pad = gather(D.train_eps, B)
    out, _, _ = losses(z, a, pad); loss = out["l1"] + ROLL_W * out.get("lroll", 0.0)
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    if step % 100 == 0:
        row = dict(step=step, loss=loss.item(), l1=out["l1"].item(), lroll=float(out.get("lroll", 0.0)), secs=time.time() - t0); hist.append(row)
        A.wandb_log({f"wm/{k}": v for k, v in row.items() if k != "step"}, step=step)
    if step % 500 == 0: log(f"step {step:6d}  loss {loss.item():.4f}  1-step {out['l1'].item():.4f}  roll {float(out.get('lroll', 0)):.4f}  ({time.time() - t0:.0f}s)")
    if step % VAL_EVERY == 0 or step == STEPS:
        v = validate(); v["step"] = step; val_hist.append(v)
        log(f"  val: R2 1-step {v['r2_1step']:.3f}  roll " + " ".join(f"h{h}={v[f'r2_roll_{h}']:.3f}" for h in (1, 2, 4, ROLL) if f"r2_roll_{h}" in v)
            + f"  action ratio {v['action_ratio']:.3f}")
        A.wandb_log({f"wm/val/{k}": x for k, x in v.items() if k != "step"}, step=step)

val = val_hist[-1]
torch.save(dict(state_dict=model.state_dict(), step=STEPS, val=val, dz=2 * D.DV, da=D.EXP, d=D_MODEL, H=H, P=P, roll=ROLL, seed=SEED), dst_ck)
json.dump(dict(seed=SEED, steps=STEPS, batch=B, lr=LR, roll=ROLL, roll_w=ROLL_W, d=D_MODEL, secs=round(time.time() - t0),
               val=val, val_history=val_hist, train_history=hist), open(dst_js, "w"), indent=1)
log(f"saved {dst_ck.name}; 1-step R2 {val['r2_1step']:.3f}, roll-{ROLL} R2 {val.get(f'r2_roll_{ROLL}', float('nan')):.3f}, "
    f"action ratio {val['action_ratio']:.3f} ({'uses the actions' if val['action_ratio'] > 1.05 else 'IGNORES the actions'})")
A.wandb_summary(val, prefix="wm/"); A.wandb_finish()
