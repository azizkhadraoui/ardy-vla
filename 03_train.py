#!/usr/bin/env python
"""
03_train.py — train ONE (variant, seed) of the hybrid autoregressive denoiser. 03_run.sh maps a SLURM array
index onto the 6 x 3 matrix; a single job can also be pinned with VARIANT=... SEED=....

THE MATRIX AND WHAT EACH ROW IS FOR
    two_stage_goal          the method: explicit-EE stage -> latent-joint stage, masked goal tokens, variable history
    one_stage_goal          architecture ablation: one denoiser over the whole hybrid token, depth doubled to match params
    two_stage_inpaint       steering baseline (DiSCo-style): trained without goals, constraints written into x0 at sampling
    two_stage_guidance      steering baseline (classifier-guidance-style): trained without goals, gradient on goal error at sampling
    two_stage_nohist        history ablation: H = 0
    two_stage_goal_rollout  the method, with a quarter of training histories replaced by the model's own rollouts, refreshed
                            every ROLLOUT_REFRESH steps — the DAgger-style fix for body drift under self-generated history

LOSSES
    hybrid MSE on both streams  +  W_GOAL * (goal loss on the explicit head + goal loss on FK(decoded body))
    + W_BODY * decoded-body MSE  +  W_CONSIST * |FK(decoded body) - explicit|^2       (exact: the explicit stream IS FK)
    10 % condition dropout on text / vision / goals (the unconditional branch used by CFG at sampling),
    history perturbation (noise + constant EE offset on a third of samples) on every variant.

    VARIANT=two_stage_goal SEED=0 WORK_DIR=... PRESET=long python 03_train.py
    PRESET=scale gives the d=512 / 8-layer / 60k-step point (one seed is enough).
"""
import os, sys, math, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
import ardy_vla as A
from ardy_vla import log, DEVICE, AMP, C, P

VARIANT = os.environ["VARIANT"]; SEED = int(os.environ.get("SEED", 0)); vcfg = A.VARIANTS[VARIANT]
dst = A.ckpt_path(VARIANT, SEED) if A.PRESET != "scale" else A.CKPT_DIR / f"{VARIANT}_scale_s{SEED}.pt"
if dst.exists() and os.environ.get("FORCE", "0") != "1":
    log(f"{dst.name} exists; set FORCE=1 to retrain"); sys.exit(0)
A.seed_all(SEED); D = A.load_data()
model = A.HybridDenoiser(D, vcfg).to(DEVICE); n_par = sum(p.numel() for p in model.parameters()) / 1e6
opt = torch.optim.AdamW(model.parameters(), lr=A.LR, betas=(0.9, 0.99), weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 500) * 0.5 * (1 + math.cos(math.pi * min(s, A.STEPS) / A.STEPS)))
scaler = torch.amp.GradScaler("cuda", enabled=A.USE_AMP)
log(f"=== {VARIANT} seed {SEED}: {n_par:.2f}M params, d={A.D_MODEL} layers={A.LAYERS} steps={A.STEPS} batch={A.BATCH}  {vcfg}")
A.wandb_init("train", VARIANT, SEED, config=dict(params_M=round(n_par, 2), ckpt=dst.name, n_train_ep=len(D.train_eps), n_val_ep=len(D.val_eps)))

rollout_pool = []; step_now = [0]
def refresh_rollout_pool(n_batches=8):
    """Self-generated histories: roll the current model forward ROLLOUT_HIST patches from GT history (no goal), then
    use the generated tokens as the recent history of a training window whose target is still ground truth."""
    global rollout_pool; rollout_pool = []; model.eval()
    with torch.no_grad():
        for _ in range(n_batches):
            b = A.make_batch(D, D.train_eps, A.BATCH, goals=None)
            back = A.ROLLOUT_HIST; ok = (b["w"] >= back + 1) & (b["h"] >= back)
            if ok.sum() < 8: continue
            bb = {k: (v[ok] if torch.is_tensor(v) and v.shape[0] == A.BATCH else v) for k, v in b.items()}
            # window that ends where the training window starts: generate its last `back` patches
            start = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in bb.items()}; start["w"] = bb["w"] - back
            start["hist"] = torch.cat([torch.zeros(len(start["w"]), back, D.TOK, device=DEVICE), bb["hist"][:, :A.H - back]], 1)  # shift GT history back by `back` patches
            start["hist_pad"] = torch.cat([torch.ones(len(start["w"]), back, dtype=torch.bool, device=DEVICE), bb["hist_pad"][:, :A.H - back]], 1)
            f0 = D.orig_idx_t[D.pf_start_t[start["e"]] + start["w"] * P].cpu(); start["va"], start["vw"] = D.vis_a[f0].to(DEVICE), D.vis_w[f0].to(DEVICE)
            gen = A.rollout(D, model, vcfg, start, back // C, goal=None)                      # (B, back, TOK)
            bb["hist"] = torch.cat([bb["hist"][:, :A.H - back], gen], 1)                        # replace the most recent `back` GT patches with generated ones
            rollout_pool.append(bb)
    model.train(); log(f"  rollout pool refreshed: {len(rollout_pool)} batches")
    A.wandb_log({"train/step": step_now[0], "train/rollout_pool_batches": len(rollout_pool)})

hist = []; t0 = time.time(); model.train()
for step in range(1, A.STEPS + 1):
    step_now[0] = step
    if vcfg["rollout"] and (step == 1 or step % A.ROLLOUT_REFRESH == 0): refresh_rollout_pool()
    if vcfg["rollout"] and rollout_pool and np.random.rand() < A.ROLLOUT_FRAC:
        b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in rollout_pool[np.random.randint(len(rollout_pool))].items()}
        A.clear_goals(b); B = b["x0"].shape[0]
        if vcfg["use_goals"]:                                   # fresh random goals on the buffered windows
            n_g = torch.randint(0, A.MAX_GOALS + 1, (B,), device=DEVICE); Np = D.p_len_t[b["e"]]
            for k in range(A.MAX_GOALS):
                hi = torch.minimum(Np, b["w"] + C + A.FUT); gp = b["w"] + (torch.rand(B, device=DEVICE) * (hi - b["w"])).long(); gf = torch.randint(0, P, (B,), device=DEVICE)
                typ = torch.randint(0, 3, (B,), device=DEVICE); m = torch.zeros(B, D.EXP_F, device=DEVICE); m[typ == 0] = 1.0; m[typ == 1, :3] = 1.0; m[typ == 2, 9:] = 1.0
                A.set_goal(D, b, k, k < n_g, gp, gf, m)
    else:
        b = A.make_batch(D, D.train_eps, A.BATCH, goals="random" if vcfg["use_goals"] else None, perturb=True); B = A.BATCH
    t = torch.randint(0, A.T_DIFF, (B,), device=DEVICE); x_t = A.q_sample(b["x0"], t, torch.randn_like(b["x0"]))
    with torch.autocast(**AMP):
        cond, cpad = model.cond_tokens(b, drop=True); x0_hat = model(x_t, t, cond, cpad).float()
        E_hat, L_hat = x0_hat[..., :D.EXP], x0_hat[..., D.EXP:]; E_, L_ = b["x0"][..., :D.EXP], b["x0"][..., D.EXP:]
        l_hyb = F.mse_loss(E_hat, E_) + F.mse_loss(L_hat, L_); l_goal = ((E_hat - E_) ** 2 * b["g_win"]).sum() / b["g_win"].sum().clamp(min=1.0)
        body_hat = D.tok.decode(L_hat).float(); l_body = F.mse_loss(body_hat, b["body_tgt"])
    fk_pos, fk_r6 = A.fk_from_body(D, body_hat); Ef = E_hat.reshape(B, C * P, D.EXP_F); Eg = E_.reshape(B, C * P, D.EXP_F); fk_pos_n = (fk_pos - D.pos_m_t) / D.pos_s_t
    l_con = F.mse_loss(fk_pos_n, Ef[..., :3]) + F.mse_loss(fk_r6, Ef[..., 3:9])
    gm = b["g_win"].reshape(B, C * P, D.EXP_F)
    l_goal_body = (((fk_pos_n - Eg[..., :3]) ** 2 * gm[..., :3]).sum() + ((fk_r6 - Eg[..., 3:9]) ** 2 * gm[..., 3:9]).sum()) / gm[..., :9].sum().clamp(min=1.0)
    loss = l_hyb + A.W_GOAL * (l_goal + l_goal_body) + A.W_BODY * l_body + A.W_CONSIST * l_con
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
    if step % 250 == 0 or step == 1:
        hist.append(dict(step=step, loss=loss.item(), hyb=l_hyb.item(), goal=l_goal.item(), goal_body=l_goal_body.item(), body=l_body.item(), consist=l_con.item()))
        A.wandb_log({"train/step": step, "train/loss": loss.item(), "train/hyb": l_hyb.item(), "train/goal": l_goal.item(),
                     "train/goal_body": l_goal_body.item(), "train/body": l_body.item(), "train/consist": l_con.item(),
                     "train/lr": sched.get_last_lr()[0], "train/grad_scale": scaler.get_scale(),
                     "train/steps_per_s": step / max(time.time() - t0, 1e-6), "train/secs": time.time() - t0})
        if step % 2000 == 0 or step == 1: log(f"  step {step:6d} loss {loss.item():.4f} | hyb {l_hyb.item():.4f} goal {l_goal.item():.4f} goal_body {l_goal_body.item():.4f} body {l_body.item():.4f} consist {l_con.item():.4f} | {time.time()-t0:.0f}s")
    if step % 5000 == 0: torch.save(dict(state_dict=model.state_dict(), variant=vcfg, name=VARIANT, seed=SEED, d_model=A.D_MODEL, layers=A.LAYERS, step=step, history=hist), dst)
torch.save(dict(state_dict=model.state_dict(), variant=vcfg, name=VARIANT, seed=SEED, d_model=A.D_MODEL, layers=A.LAYERS, step=A.STEPS, history=hist, preset=A.PRESET, secs=round(time.time() - t0)), dst)
A.wandb_summary(dict(params_M=round(n_par, 2), final_loss=hist[-1]["loss"], final_hyb=hist[-1]["hyb"], final_body=hist[-1]["body"],
                     final_goal=hist[-1]["goal"], final_goal_body=hist[-1]["goal_body"], final_consist=hist[-1]["consist"],
                     minutes=round((time.time() - t0) / 60, 1)), prefix="train/")
A.wandb_finish()
log(f"saved {dst}  ({(time.time()-t0)/60:.0f} min)")
