"""
ardy_vla.py — shared library for the ARDY-style hybrid autoregressive-diffusion VLA experiments.

Everything the numbered scripts need lives here so that the training code, the open-loop evaluator
and the closed-loop evaluator use IDENTICAL definitions of the model, the sampler, the hybrid token,
the FK chain and the metrics. Scripts import it by relative path (they sit beside it).

Layout under $WORK_DIR (all scripts read/write here):
    data/          proprio.npz, meta.json, vision_agentview.npy, vision_wrist.npy, text_emb.npy
    tokenizer/     tokenizer.pt, latents.npy, patch_index.npz, tokenizer_eval.json
    ckpt/          {variant}_s{seed}.pt   (+ _history in the same file)
    results/       openloop_{variant}_s{seed}.json, closedloop_{variant}_s{seed}.json, latency.json,
                   tables.md, summary.json
    figures/       *.png, *.mp4
    episodes/      recorded closed-loop episodes (npz + mp4) for 08_visualize.py

Environment knobs (read at import; every script accepts the same ones):
    WORK_DIR        output root                                  (required)
    DATA_DIR        where the LIBERO HDF5 demos are / go          (default $WORK_DIR/libero_hdf5)
    LIBERO_DIR      clone of Lifelong-Robot-Learning/LIBERO        (needed by 05 and 08 only)
    SUITES          comma list, default libero_spatial,libero_object,libero_goal,libero_10
    ENCODER         dinov2 | siglip     (frozen vision encoder)
    PRESET          long | quick         (model size / steps)
    D_MODEL, LAYERS, STEPS, BATCH, LR   override the preset
    SEED, VARIANT   set by 03/04/05 launchers

The hybrid token for one 4-frame patch is [ explicit 4x(ee_pos 3, rot6d 6, gripper 2) = 44 | FSQ latent ]
where the explicit EE stream is DEFINED as base ∘ FK(joints) ∘ tool, so it is exactly consistent
with the latent (joint) stream and the consistency loss is exact by construction.
"""
import os, sys, json, glob, math, time, re, subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

# ============================================================================ config
WORK_DIR = Path(os.environ.get("WORK_DIR", "./work")).expanduser()
DATA_DIR = Path(os.environ.get("DATA_DIR", WORK_DIR / "libero_hdf5")).expanduser()
LIBERO_DIR = Path(os.environ.get("LIBERO_DIR", WORK_DIR / "LIBERO")).expanduser()
for sub in ("data", "tokenizer", "ckpt", "results", "figures", "episodes"):
    (WORK_DIR / sub).mkdir(parents=True, exist_ok=True)
DATA, TOK_DIR, CKPT_DIR, RES_DIR, FIG_DIR, EPI_DIR = [WORK_DIR / s for s in ("data", "tokenizer", "ckpt", "results", "figures", "episodes")]

SUITES = os.environ.get("SUITES", "libero_spatial,libero_object,libero_goal,libero_10").split(",")
HF_REPO = "yifengzhu-hf/LIBERO-datasets"          # the mirror the official LIBERO downloader uses
ENCODER = os.environ.get("ENCODER", "dinov2")
TEXT_MODEL = "t5-base"
POOL, IMG_RES, FPS, FLIP_180 = 4, 224, 20, True
HELDOUT_PER_TASK = 5

# tokenizer
P, LEVELS, G = 4, [8, 8, 8, 5, 5, 5], 3
TOK_D, TOK_LAYERS, TOK_HEADS, TOK_SEG, TOK_BATCH, TOK_LR = 256, 4, 4, 48, 256, 3e-4
TOK_STEPS = int(os.environ.get("TOK_STEPS", 12000))
TOK_W_VEL, TOK_VEL_DIM_W, TOK_NOISE, TOK_DROPOUT = 1.0, 0.3, 0.02, 0.1

# denoiser
C, H, FUT, MAX_GOALS = 4, 32, 12, 3
T_DIFF, SAMPLE_STEPS, N_HEADS = 100, int(os.environ.get("SAMPLE_STEPS", 10)), 8
PRESET = os.environ.get("PRESET", "long")
_p = dict(long=dict(D_MODEL=384, LAYERS=6, STEPS=30000), quick=dict(D_MODEL=256, LAYERS=4, STEPS=8000), scale=dict(D_MODEL=512, LAYERS=8, STEPS=60000))[PRESET]
D_MODEL = int(os.environ.get("D_MODEL", _p["D_MODEL"])); LAYERS = int(os.environ.get("LAYERS", _p["LAYERS"])); STEPS = int(os.environ.get("STEPS", _p["STEPS"]))
BATCH = int(os.environ.get("BATCH", 128)); LR = float(os.environ.get("LR", 3e-4)); COND_DROP = 0.10
W_GOAL, W_BODY, W_CONSIST = 2.0, 1.0, 5.0
HIST_NOISE, HIST_SHIFT_P, HIST_SHIFT_CM = 0.1, 0.33, 3.0
ROLLOUT_FRAC, ROLLOUT_REFRESH, ROLLOUT_HIST = 0.25, 2000, 8    # rollout-history variant: share of batches drawn from self-generated histories
GOAL_CFG = float(os.environ.get("GOAL_CFG", 2.0))
GUIDE_SCALE = float(os.environ.get("GUIDE_SCALE", 0.3))       # gradient-guidance baseline strength
SNAP_LATENTS = True
PROJECT_BODY_STEPS, PROJECT_BODY_LAM = 30, 0.05
VAL_WINDOWS, HORIZONS, OUT_HORIZON = [8, 16, 24], [4, 8, 12, 16], 8

# variants: two_stage / use_goals / use_vision / use_hist / steering mode / rollout-history training
VARIANTS = {
    "two_stage_goal":         dict(two_stage=True,  use_goals=True,  use_vision=True,  use_hist=True,  steer="token",    rollout=False),
    "one_stage_goal":         dict(two_stage=False, use_goals=True,  use_vision=True,  use_hist=True,  steer="token",    rollout=False),
    "two_stage_inpaint":      dict(two_stage=True,  use_goals=False, use_vision=True,  use_hist=True,  steer="inpaint",  rollout=False),
    "two_stage_guidance":     dict(two_stage=True,  use_goals=False, use_vision=True,  use_hist=True,  steer="guidance", rollout=False),
    "two_stage_nohist":       dict(two_stage=True,  use_goals=True,  use_vision=True,  use_hist=False, steer="token",    rollout=False),
    "two_stage_goal_rollout": dict(two_stage=True,  use_goals=True,  use_vision=True,  use_hist=True,  steer="token",    rollout=True),
}
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"
AMP = dict(device_type="cuda", dtype=torch.float16, enabled=USE_AMP)   # V100: fp16, no bf16


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ============================================================================ weights & biases
# Opt-in: WANDB=1 in env.sh. Every helper below is a no-op when it is off, when the package is missing, or when
# anything inside wandb raises -- a logging failure must never take down a 2.6 h evaluation. Compute nodes without
# outbound network should set WANDB_MODE=offline and `wandb sync $WORK_DIR/wandb/offline-*` from the login node.
WANDB = os.environ.get("WANDB", "0") == "1"
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "ardy-vla")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY") or None
WANDB_GROUP = os.environ.get("WANDB_GROUP") or None        # default group: the variant, so its seeds sit together
_wb = None            # the module, once imported
_wb_run = None
_wb_quiet = False     # set after the first failed log, so a broken run does not print 30k warnings


def wandb_id(variant, seed):
    """One run per checkpoint. Stages 3/4/5 all attach to the same id, so a variant's training curve, its open-loop
    adherence and its closed-loop success live on one run instead of three that have to be joined by hand."""
    return re.sub(r"[^A-Za-z0-9_.-]", "-", f"{variant}_s{seed}" + ("" if PRESET == "long" else f"_{PRESET}"))


def wandb_init(job_type, variant=None, seed=None, config=None, name=None):
    """Start (or re-attach to) the run for this stage. Returns the run, or None when logging is off."""
    global _wb, _wb_run, _wb_quiet
    wandb_finish()
    if not WANDB:
        return None
    try:
        import wandb
    except ImportError:
        log("WANDB=1 but the wandb package is not installed (pip install wandb); continuing without it")
        return None
    try:
        cfg = dict(preset=PRESET, suites=SUITES, encoder=ENCODER, d_model=D_MODEL, layers=LAYERS, steps=STEPS,
                   batch=BATCH, lr=LR, cond_drop=COND_DROP, w_goal=W_GOAL, w_body=W_BODY, w_consist=W_CONSIST,
                   patch=P, chunk=C, hist=H, fut=FUT, max_goals=MAX_GOALS, t_diff=T_DIFF, sample_steps=SAMPLE_STEPS,
                   goal_cfg=GOAL_CFG, guide_scale=GUIDE_SCALE, tok_levels=LEVELS, tok_groups=G, tok_steps=TOK_STEPS,
                   slurm_job=os.environ.get("SLURM_JOB_ID"), slurm_array=os.environ.get("SLURM_ARRAY_TASK_ID"),
                   gpu=torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu", torch_version=torch.__version__)
        if variant is not None:
            cfg["variant"] = variant
            cfg.update({f"v_{k}": v for k, v in VARIANTS.get(variant.replace("_scale", ""), {}).items()})
        if seed is not None:
            cfg["seed"] = seed
        cfg.update(config or {})
        per_ckpt = variant is not None and seed is not None
        _wb_run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, job_type=job_type,
                             group=WANDB_GROUP or (variant if variant is not None else job_type),
                             id=wandb_id(variant, seed) if per_ckpt else None, resume="allow" if per_ckpt else None,
                             name=name or (f"{variant}_s{seed}" if per_ckpt else job_type),
                             # allow_val_change: stages 4/5 resume the training run from a different SLURM job, often a different GPU
                             config=cfg, allow_val_change=True, dir=str(WORK_DIR), settings=wandb.Settings(silent=True))
        _wb, _wb_quiet = wandb, False
        # own x-axis per stage: a re-run (FORCE=1) overlays its curve instead of having its steps dropped as non-monotonic
        for p in ("train", "tok", "closedloop"):
            wandb.define_metric(f"{p}/step"); wandb.define_metric(f"{p}/*", step_metric=f"{p}/step")
        log(f"wandb: {_wb_run.name} [{job_type}] {getattr(_wb_run, 'url', '') or '(offline)'}")
    except Exception as e:
        log(f"wandb init failed ({type(e).__name__}: {e}); continuing without it"); _wb_run = None
    return _wb_run


def _wb_fail(e):
    global _wb_quiet
    if not _wb_quiet:
        log(f"wandb logging failed ({type(e).__name__}: {e}); further wandb warnings suppressed"); _wb_quiet = True


def wandb_log(d, **kw):
    if _wb_run is None: return
    try: _wb.log({k: v for k, v in d.items() if v is not None}, **kw)
    except Exception as e: _wb_fail(e)


def _flat(d, prefix=""):
    """Nested result dicts -> flat scalar keys, the shape run.summary wants."""
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict): out.update(_flat(v, key + "/"))
        elif isinstance(v, (int, float, bool, str)) and not (isinstance(v, float) and math.isnan(v)): out[key] = v
    return out


def wandb_summary(d, prefix=""):
    """Final numbers of a stage: they land in run.summary, which is what the runs table and the reports read."""
    if _wb_run is None: return
    try: _wb_run.summary.update(_flat(d, prefix))
    except Exception as e: _wb_fail(e)


def wandb_table(key, rows):
    """A list of uniform dicts as a wandb.Table (latency rows, per-suite breakdowns, aggregated tables)."""
    if _wb_run is None or not rows: return
    try:
        cols = list(dict.fromkeys(k for r in rows for k in r))
        _wb.log({key: _wb.Table(columns=cols, data=[[r.get(c) for c in cols] for r in rows])})
    except Exception as e: _wb_fail(e)


def wandb_images(mapping):
    """{key: path} -> logged images. Missing files are skipped."""
    if _wb_run is None: return
    try:
        d = {k: _wb.Image(str(p)) for k, p in mapping.items() if Path(p).exists()}
        if d: _wb.log(d)
    except Exception as e: _wb_fail(e)


def wandb_video(key, path, fps=None):
    """A rendered episode (mp4) onto the run, next to the numbers it illustrates."""
    if _wb_run is None or not Path(path).exists(): return
    try: _wb.log({key: _wb.Video(str(path), fps=fps or FPS, format="mp4")})
    except Exception as e: _wb_fail(e)


def wandb_save(*paths):
    """Attach result files (tables.md, summary.json, ...) to the run."""
    if _wb_run is None: return
    try:
        for p in paths:
            if Path(p).exists(): _wb.save(str(p), base_path=str(Path(p).parent), policy="now")
    except Exception as e: _wb_fail(e)


def wandb_finish():
    global _wb_run
    if _wb_run is None: return
    try: _wb.finish()
    except Exception as e: _wb_fail(e)
    _wb_run = None


def seed_all(seed):
    torch.manual_seed(seed); np.random.seed(seed)


# ============================================================================ geometry / FK
def axis_angle_to_matrix(aa):
    theta = np.linalg.norm(aa, axis=-1, keepdims=True); k = aa / np.maximum(theta, 1e-8)
    K = np.zeros((aa.shape[0], 3, 3)); K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]; K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]; K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    s, c = np.sin(theta)[..., None], np.cos(theta)[..., None]
    return np.eye(3)[None] + s * K + (1 - c) * (K @ K)


def quat_to_matrix(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True); x, y, z, w = q.T
    return np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w), 2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w), 2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], -1).reshape(-1, 3, 3)


def rot6d_to_mat(r6):
    a1, a2 = r6[..., :3], r6[..., 3:]; b1 = F.normalize(a1, dim=-1); b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    return torch.stack([b1, b2, torch.cross(b1, b2, dim=-1)], -1)


def mat_to_rot6d(R): return torch.cat([R[..., :, 0], R[..., :, 1]], -1)


def rot_angle_deg(R1, R2):
    c = ((R1.transpose(-1, -2) @ R2).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.rad2deg(torch.acos(c.clamp(-1, 1)))


_DH = [(0.0, 0.333, 0.0), (0.0, 0.0, -math.pi/2), (0.0, 0.316, math.pi/2), (0.0825, 0.0, math.pi/2), (-0.0825, 0.384, -math.pi/2), (0.0, 0.0, math.pi/2), (0.088, 0.0, math.pi/2)]


def _mdh(a, d, alpha, theta):
    ct, st = torch.cos(theta), torch.sin(theta); ca, sa = math.cos(alpha), math.sin(alpha); z, o = torch.zeros_like(ct), torch.ones_like(ct)
    return torch.stack([torch.stack([ct, -st, z, a * o], -1), torch.stack([st * ca, ct * ca, -sa * o, -d * sa * o], -1),
                        torch.stack([st * sa, ct * sa, ca * o, d * ca * o], -1), torch.stack([z, z, z, o], -1)], -2)


def panda_fk(q):
    """q (M,7) rad -> flange pose (M,4,4), Craig-convention DH of the Franka Panda."""
    T = torch.eye(4, device=q.device).expand(q.shape[0], 4, 4)
    for i, (a, d, al) in enumerate(_DH): T = T @ _mdh(a, d, al, q[:, i])
    return T @ _mdh(0.0, 0.107, 0.0, torch.zeros_like(q[:, 0]))


def se3(r6, t):
    T = torch.eye(4, device=r6.device).clone(); T[:3, :3] = rot6d_to_mat(r6); T[:3, 3] = t; return T


def fk_with(params, q):
    Tb, Tt = se3(params[0], params[1]), se3(params[2], params[3]); T = Tb[None] @ panda_fk(q) @ Tt[None]
    return T[:, :3, 3], T[:, :3, :3]


def se3_batch(r6, t):
    """(G,6),(G,3) -> (G,4,4)."""
    T = torch.eye(4, device=r6.device).repeat(r6.shape[0], 1, 1); T[:, :3, :3] = rot6d_to_mat(r6); T[:, :3, 3] = t; return T


def fk_grouped(base_r6, base_t, tool_r6, tool_t, q, gid):
    """base o FK(q) o tool, with the base chosen per row by `gid`. Returns world position and rotation."""
    Tb = se3_batch(base_r6, base_t)[gid]; Tt = se3(tool_r6, tool_t)
    T = Tb @ panda_fk(q) @ Tt[None]
    return T[:, :3, 3], T[:, :3, :3]


def fit_fk_grouped(q, ee, R, gid, n_groups, iters=4000, w_rot=0.1):
    """Fit ONE base SE(3) per group and ONE tool SE(3) shared by all of them.

    LIBERO places the robot base at a scene-dependent world position, so a single base transform cannot map
    FK(joints) onto the dataset's world-frame ee_pos across scenes (it lands 30-50 cm off), while each scene
    alone fits to well under a millimetre. The tool transform is the flange->EE offset of the robot itself and
    is genuinely shared. Groups are tasks here: tasks in one scene simply converge to the same base.
    """
    base_r6 = nn.Parameter(torch.tensor([1., 0, 0, 0, 1, 0], device=q.device).repeat(n_groups, 1))
    base_t = nn.Parameter(torch.zeros(n_groups, 3, device=q.device))
    tool_r6 = nn.Parameter(torch.tensor([1., 0, 0, 0, 1, 0], device=q.device))
    tool_t = nn.Parameter(torch.zeros(3, device=q.device))
    params = [base_r6, base_t, tool_r6, tool_t]
    opt = torch.optim.Adam(params, lr=1e-2)
    for it in range(iters):
        pos, Rp = fk_grouped(base_r6, base_t, tool_r6, tool_t, q, gid); loss = F.mse_loss(pos, ee) + w_rot * F.mse_loss(Rp, R)
        opt.zero_grad(); loss.backward(); opt.step()
        if it == int(iters * 0.6):
            for g_ in opt.param_groups: g_["lr"] = 1e-3
    return [p.detach() for p in params]


def fit_fk(q, ee, R, iters=3000):
    """Fit base + tool SE(3) so that base ∘ FK(q) ∘ tool matches the dataset's (ee, R). Used once, in 01, to define the explicit stream."""
    params = [nn.Parameter(torch.tensor([1., 0, 0, 0, 1, 0], device=q.device)), nn.Parameter(torch.zeros(3, device=q.device)),
              nn.Parameter(torch.tensor([1., 0, 0, 0, 1, 0], device=q.device)), nn.Parameter(torch.zeros(3, device=q.device))]
    opt = torch.optim.Adam(params, lr=1e-2)
    for it in range(iters):
        pos, Rp = fk_with(params, q); loss = F.mse_loss(pos, ee) + 0.1 * F.mse_loss(Rp, R)
        opt.zero_grad(); loss.backward(); opt.step()
        if it == int(iters * 0.6):
            for g_ in opt.param_groups: g_["lr"] = 1e-3
    return [p.detach() for p in params]


# ============================================================================ data download
def libero_files(suite):
    return sorted(glob.glob(str(DATA_DIR / "**" / suite / "*.hdf5"), recursive=True))


def download_suites(suites=SUITES):
    """Original LIBERO HDF5 demos (they carry joint states, the LeRobot/RLDS mirrors do not). Resumable, rate-limit aware."""
    from huggingface_hub import snapshot_download
    want = {"libero_spatial": 10, "libero_object": 10, "libero_goal": 10, "libero_10": 10, "libero_90": 90}
    token = os.environ.get("HF_TOKEN")
    for suite in suites:
        last = None
        for attempt in range(10):
            have = len(libero_files(suite))
            if have >= want[suite]: break
            try:
                snapshot_download(HF_REPO, repo_type="dataset", allow_patterns=[f"{suite}/*.hdf5"], local_dir=str(DATA_DIR), max_workers=2, token=token)
                continue                                   # re-count at the top of the loop, not here
            except Exception as ex:
                last = ex
                # print what actually went wrong: the type alone (LocalProtocolError) says nothing and cost a session
                log(f"{suite}: attempt {attempt+1}/10, {have}/{want[suite]} files, {type(ex).__name__}: {str(ex)[:300]}")
                if attempt >= 4:                           # snapshot_download keeps failing -> resume file by file
                    try:
                        from huggingface_hub import hf_hub_download, list_repo_files
                        names = [f for f in list_repo_files(HF_REPO, repo_type="dataset", token=token) if f.startswith(f"{suite}/") and f.endswith(".hdf5")]
                        log(f"{suite}: falling back to per-file download of {len(names)} files")
                        for fn in names:
                            hf_hub_download(HF_REPO, fn, repo_type="dataset", local_dir=str(DATA_DIR), token=token)
                        continue
                    except Exception as ex2:
                        last = ex2; log(f"{suite}: per-file download also failed, {type(ex2).__name__}: {str(ex2)[:300]}")
                m = re.search(r"Retry after (\d+)", str(ex)); wait = int(m.group(1)) + 10 if m else 90
                log(f"{suite}: waiting {wait}s"); time.sleep(wait)
        have = len(libero_files(suite))
        if have < want[suite]:                             # do not hide the cause behind a bare assert
            raise RuntimeError(f"{suite}: only {have}/{want[suite]} files under {DATA_DIR} after 10 attempts; "
                               f"last error was {type(last).__name__}: {last}") from last
        log(f"{suite}: {len(libero_files(suite))} task files")
    return {s: libero_files(s) for s in suites}


def ensure_libero_repo():
    if not LIBERO_DIR.exists():
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/Lifelong-Robot-Learning/LIBERO", str(LIBERO_DIR)], check=True)
    if str(LIBERO_DIR) not in sys.path: sys.path.insert(0, str(LIBERO_DIR))
    cfg_dir = Path.home() / ".libero"; cfg_dir.mkdir(exist_ok=True); lib = LIBERO_DIR / "libero" / "libero"
    (cfg_dir / "config.yaml").write_text(f"benchmark_root: {lib}\nbddl_files: {lib / 'bddl_files'}\ninit_states: {lib / 'init_files'}\ndatasets: {DATA_DIR}\nassets: {lib / 'assets'}\n")


# ============================================================================ tokenizer
class FSQ(nn.Module):
    def __init__(self, levels):
        super().__init__()
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.float32))
        self.register_buffer("basis", torch.cumprod(torch.tensor([1] + levels[:-1], dtype=torch.long), dim=0))
        self.dim, self.num_codes = len(levels), int(math.prod(levels))

    def bound(self, z):
        L = self.levels; half = (L - 1) / 2; offset = torch.where(L % 2 == 0, torch.full_like(L, 0.5), torch.zeros_like(L))
        return torch.tanh(z + torch.atanh(offset / half)) * half - offset

    def forward(self, z):
        with torch.autocast(device_type=z.device.type, enabled=False):
            zb = self.bound(z.float().clamp(-20, 20)); zr = torch.round(zb); zq = zb + (zr - zb).detach()
            hw = torch.div(self.levels, 2, rounding_mode="floor")
            return (zq / hw).to(z.dtype), ((zr + hw).long() * self.basis).sum(-1)


class CausalTransformer(nn.Module):
    def __init__(self, d, n_layers, n_heads, max_len=256, dropout=0.0):
        super().__init__()
        self.blocks = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout, activation="gelu", batch_first=True, norm_first=True), n_layers, enable_nested_tensor=False)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d)); nn.init.normal_(self.pos, std=0.02); self.norm = nn.LayerNorm(d)

    def forward(self, x):
        T = x.shape[1]; mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        return self.norm(self.blocks(x + self.pos[:, :T], mask=mask, is_causal=True))


class MotionTokenizer(nn.Module):
    """Causal encoder -> FSQ (G groups) -> causal decoder over P-frame patches of the body vector [joint_pos, joint_vel]."""
    def __init__(self, in_dim, P=4, levels=(8, 8, 8, 5, 5, 5), groups=3, d=256, n_layers=4, n_heads=4, max_patches=256, dropout=0.0):
        super().__init__()
        self.in_dim, self.P, self.G, self.fsq = in_dim, P, groups, FSQ(list(levels)); self.latent_dim = groups * len(levels)
        self.enc_in, self.encoder, self.to_lat = nn.Linear(P * in_dim, d), CausalTransformer(d, n_layers, n_heads, max_patches, dropout), nn.Linear(d, self.latent_dim)
        self.from_lat, self.decoder, self.dec_out = nn.Linear(self.latent_dim, d), CausalTransformer(d, n_layers, n_heads, max_patches, dropout), nn.Linear(d, P * in_dim)

    def encode(self, x):
        B, T, Cc = x.shape; h = self.encoder(self.enc_in(x.reshape(B, T // self.P, self.P * Cc)))
        zq, codes = self.fsq(self.to_lat(h).reshape(B, h.shape[1], self.G, -1)); return zq.reshape(B, h.shape[1], -1), codes

    def decode(self, zq):
        y = self.dec_out(self.decoder(self.from_lat(zq))); return y.reshape(y.shape[0], y.shape[1] * self.P, self.in_dim)

    def forward(self, x): zq, codes = self.encode(x); return self.decode(zq), zq, codes


def load_tokenizer():
    ck = torch.load(TOK_DIR / "tokenizer.pt", map_location=DEVICE, weights_only=False); cfg = ck["config"]
    tok = MotionTokenizer(cfg["in_dim"], cfg["P"], cfg["levels"], cfg["groups"], cfg["d"], cfg["n_layers"], cfg["n_heads"], cfg["max_patches"], cfg.get("dropout", 0.0)).to(DEVICE)
    tok.load_state_dict(ck["state_dict"]); tok.eval()
    for p in tok.parameters(): p.requires_grad_(False)
    return tok, ck


# ============================================================================ dataset in memory (namespace D)
def load_data(vision=True):
    """Everything the denoiser needs, as one namespace. Vision features stay on the CPU (pinned) and are gathered per batch:
    four suites are ~6.5 GB of fp16 tokens, too much for a 16 GB V100 next to a training run."""
    D = SimpleNamespace(); D.meta = json.load(open(DATA / "meta.json")); pr = np.load(DATA / "proprio.npz"); pidx = np.load(TOK_DIR / "patch_index.npz")
    D.tok, ck = load_tokenizer(); cfg = ck["config"]
    D.LAT, D.EXP_F = D.tok.latent_dim, 11; D.EXP = P * D.EXP_F; D.TOK = D.EXP + D.LAT; D.NJ = cfg["in_dim"] // 2
    D.body_mean, D.body_std = torch.tensor(ck["mean"], device=DEVICE), torch.tensor(ck["std"], device=DEVICE)
    D.ep_start, D.ep_len, D.ep_task = pr["episode_start"], pr["episode_len"], pr["episode_task"]
    D.p_start, D.p_len, D.val_mask = pidx["patch_start"], pidx["patch_len"], pidx["val_mask"].astype(bool); D.E = len(D.ep_len)
    D.train_eps, D.val_eps = np.where(~D.val_mask)[0], np.where(D.val_mask)[0]
    D.tr_frames = np.concatenate([np.arange(D.ep_start[e], D.ep_start[e] + D.ep_len[e]) for e in D.train_eps])
    D.pos_m, D.pos_s = pr["ee_pos"][D.tr_frames].mean(0), pr["ee_pos"][D.tr_frames].std(0) + 1e-6
    D.gr_m, D.gr_s = pr["gripper"][D.tr_frames].mean(0), pr["gripper"][D.tr_frames].std(0) + 1e-6
    exp_frame = np.concatenate([(pr["ee_pos"] - D.pos_m) / D.pos_s, pr["ee_rot6d"], (pr["gripper"] - D.gr_m) / D.gr_s], 1).astype(np.float32)
    body_frame = ((np.concatenate([pr["joint_pos"], pr["joint_vel"]], 1) - ck["mean"]) / ck["std"]).astype(np.float32)
    D.pf_len = D.p_len * P; D.pf_start = np.concatenate([[0], np.cumsum(D.pf_len)[:-1]])
    orig_idx = np.concatenate([D.ep_start[e] + np.minimum(np.arange(D.pf_len[e]), D.ep_len[e] - 1) for e in range(D.E)])
    D.exp_pad, D.body_pad, D.orig_idx_t = torch.from_numpy(exp_frame[orig_idx]).to(DEVICE), torch.from_numpy(body_frame[orig_idx]).to(DEVICE), torch.from_numpy(orig_idx).to(DEVICE)
    D.hyb = torch.cat([D.exp_pad.reshape(-1, D.EXP), torch.from_numpy(np.load(TOK_DIR / "latents.npy").astype(np.float32)).to(DEVICE)], 1)
    if vision:
        D.vis_a = torch.from_numpy(np.load(DATA / "vision_agentview.npy")).pin_memory() if DEVICE == "cuda" else torch.from_numpy(np.load(DATA / "vision_agentview.npy"))
        D.vis_w = torch.from_numpy(np.load(DATA / "vision_wrist.npy")).pin_memory() if DEVICE == "cuda" else torch.from_numpy(np.load(DATA / "vision_wrist.npy"))
        D.DV = D.vis_a.shape[2]
    else:
        D.vis_a = D.vis_w = None; D.DV = D.meta["vision_dim"]
    D.text = torch.from_numpy(np.load(DATA / "text_emb.npy")).to(DEVICE); D.DT = D.text.shape[1]
    D.pos_m_t, D.pos_s_t, D.gr_m_t, D.gr_s_t = [torch.tensor(a, device=DEVICE) for a in (D.pos_m, D.pos_s, D.gr_m, D.gr_s)]
    D.p_start_t, D.p_len_t, D.pf_start_t, D.pf_len_t, D.ep_start_t, D.ep_len_t, D.ep_task_t = [torch.from_numpy(a).to(DEVICE) for a in (D.p_start, D.p_len, D.pf_start, D.pf_len, D.ep_start, D.ep_len, D.ep_task)]
    D.pr = pr
    # the explicit stream is the EE in the ROBOT BASE frame (01 defines it that way: the world offset is
    # scene-dependent and carries nothing about the arm), so fk_params has an identity base and the shared tool.
    D.fk_tool_r6, D.fk_tool_t = [torch.tensor(pr[k], device=DEVICE) for k in ("fk_tool_r6", "fk_tool_t")]
    D.fk_params = [torch.tensor([1., 0, 0, 0, 1, 0], device=DEVICE), torch.zeros(3, device=DEVICE), D.fk_tool_r6, D.fk_tool_t]
    # per task, base frame -> world; used only to draw recorded trajectories in camera images (05 -> 08)
    D.fk_base_r6, D.fk_base_t = [torch.tensor(pr[k], device=DEVICE) for k in ("fk_base_r6", "fk_base_t")]
    D.gripper_threshold = float(D.meta["gripper_threshold_m"])
    log(f"data: {D.E} episodes ({len(D.train_eps)} train / {len(D.val_eps)} val), {D.hyb.shape[0]} patches, token {D.TOK} = {D.EXP} explicit + {D.LAT} latent, {len(D.meta['tasks'])} tasks")
    return D


def unnorm_pos(D, x): return x * D.pos_s_t + D.pos_m_t


def to_world(D, task_index, X):
    """Base-frame positions (..., 3) -> world, via the task's fitted base transform. Drawing only: every metric
    and every goal lives in the base frame, but 08 projects paths into camera images, which are world-framed."""
    X = np.asarray(X, np.float32)
    if X.size == 0: return X
    T = se3(D.fk_base_r6[task_index], D.fk_base_t[task_index])
    x = torch.from_numpy(X.reshape(-1, 3)).to(T.device)
    return ((x @ T[:3, :3].T) + T[:3, 3]).cpu().numpy().reshape(X.shape)


def fk_from_body(D, body_n):
    q = body_n[..., :D.NJ] * D.body_std[:D.NJ] + D.body_mean[:D.NJ]; pos, R = fk_with(D.fk_params, q.reshape(-1, D.NJ))
    return pos.reshape(*body_n.shape[:-1], 3), mat_to_rot6d(R).reshape(*body_n.shape[:-1], 6)


def fk_pos_of_q(D, q):
    pos, _ = fk_with(D.fk_params, q.reshape(-1, D.NJ)); return pos.reshape(*q.shape[:-1], 3)


def snap_latents(D, z):
    hw = torch.div(D.tok.fsq.levels, 2, rounding_mode="floor").repeat(D.tok.G); L = D.tok.fsq.levels.repeat(D.tok.G)
    return torch.round(z * hw).clamp(-hw, L - 1 - hw) / hw


def project_body(D, body_n, E_n, steps=PROJECT_BODY_STEPS, lam=PROJECT_BODY_LAM):
    """Inference-time consistency projection: q minimising |FK(q) - explicit EE|^2 + lam |q - q_decoded|^2. Reported as a SEPARATE row."""
    q0 = (body_n[..., :D.NJ] * D.body_std[:D.NJ] + D.body_mean[:D.NJ]).reshape(-1, D.NJ).detach(); E = E_n.reshape(-1, D.EXP_F).detach()
    if steps <= 0: return q0.reshape(*body_n.shape[:-1], D.NJ)
    q = q0.clone().requires_grad_(True); opt = torch.optim.Adam([q], lr=0.01)
    with torch.enable_grad():
        for _ in range(steps):
            pos, R = fk_with(D.fk_params, q); pos_n = (pos - D.pos_m_t) / D.pos_s_t
            loss = ((pos_n - E[:, :3]) ** 2).sum(-1).mean() + ((mat_to_rot6d(R) - E[:, 3:9]) ** 2).sum(-1).mean() + lam * ((q - q0) ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return q.detach().reshape(*body_n.shape[:-1], D.NJ)


# ============================================================================ batches
def make_batch(D, eps, B, win_starts=None, hist_len=None, goals="random", perturb=False):
    e = torch.as_tensor(np.random.choice(eps, B) if win_starts is None else eps, device=DEVICE); Np = D.p_len_t[e]
    w = torch.as_tensor(win_starts, device=DEVICE) if win_starts is not None else (torch.rand(B, device=DEVICE) * (Np - C + 1)).long()
    hmax = torch.minimum(torch.full_like(w, H), w); h = hmax if hist_len == "max" else (torch.rand(B, device=DEVICE) * (hmax + 1)).long()
    ps = D.p_start_t[e]; x0 = D.hyb[(ps + w)[:, None] + torch.arange(C, device=DEVICE)]
    j = torch.arange(H, device=DEVICE)[None]; hist_pad = j < (H - h)[:, None]; hist = D.hyb[((ps + w)[:, None] - (H - j)).clamp(min=0)] * (~hist_pad)[..., None]
    if perturb:   # steering augmentation: future target stays ground truth, past gets noise and (on a third) a constant EE offset
        hist = hist + HIST_NOISE * torch.randn_like(hist) * (torch.rand(B, 1, 1, device=DEVICE) < 0.5) * (~hist_pad)[..., None]
        sel = (torch.rand(B, device=DEVICE) < HIST_SHIFT_P).float()[:, None]
        off_n = (torch.randn(B, 3, device=DEVICE) * (HIST_SHIFT_CM / 100.0) / D.pos_s_t) * sel
        shift = off_n[:, None, None, :].expand(B, 1, P, 3).reshape(B, 1, P * 3)
        idx = (torch.arange(P, device=DEVICE)[:, None] * D.EXP_F + torch.arange(3, device=DEVICE)[None]).reshape(-1)
        hist[:, :, idx] = hist[:, :, idx] + shift * (~hist_pad)[..., None]
    f0 = D.orig_idx_t[D.pf_start_t[e] + w * P]; fr = (D.pf_start_t[e] + w * P)[:, None] + torch.arange(C * P, device=DEVICE)
    f0c = f0.cpu()
    va = D.vis_a[f0c].to(DEVICE, non_blocking=True) if D.vis_a is not None else torch.zeros(B, 1 + POOL * POOL, D.DV, device=DEVICE, dtype=torch.float16)
    vw = D.vis_w[f0c].to(DEVICE, non_blocking=True) if D.vis_w is not None else torch.zeros_like(va)
    b = dict(e=e, w=w, h=h, x0=x0, hist=hist, hist_pad=hist_pad, va=va, vw=vw, tx=D.text[D.ep_task_t[e]], body_tgt=D.body_pad[fr],
             g_val=torch.zeros(B, MAX_GOALS, D.EXP_F, device=DEVICE), g_mask=torch.zeros(B, MAX_GOALS, D.EXP_F, device=DEVICE),
             g_t=torch.zeros(B, MAX_GOALS, dtype=torch.long, device=DEVICE), g_pad=torch.ones(B, MAX_GOALS, dtype=torch.bool, device=DEVICE), g_win=torch.zeros(B, C, D.EXP, device=DEVICE))
    if goals == "random":
        n_g = torch.randint(0, MAX_GOALS + 1, (B,), device=DEVICE)
        for k in range(MAX_GOALS):
            hi = torch.minimum(Np, w + C + FUT); gp = w + (torch.rand(B, device=DEVICE) * (hi - w)).long(); gf = torch.randint(0, P, (B,), device=DEVICE)
            typ = torch.randint(0, 3, (B,), device=DEVICE); m = torch.zeros(B, D.EXP_F, device=DEVICE); m[typ == 0] = 1.0; m[typ == 1, :3] = 1.0; m[typ == 2, 9:] = 1.0
            set_goal(D, b, k, k < n_g, gp, gf, m)
    return b


def set_goal(D, b, k, active, gp, gf, m):
    e, w = b["e"], b["w"]; val = D.exp_pad[D.pf_start_t[e] + gp * P + gf]
    b["g_val"][:, k] = val * m; b["g_mask"][:, k] = m * active[:, None]; b["g_t"][:, k] = ((gp - w) * P + gf).clamp(0, (C + FUT) * P - 1); b["g_pad"][:, k] = ~active
    rows = torch.where(active & (gp < w + C))[0]
    if rows.numel():
        pi, off = gp[rows] - w[rows], gf[rows] * D.EXP_F
        for d in range(D.EXP_F): b["g_win"][rows, pi, off + d] = torch.maximum(b["g_win"][rows, pi, off + d], m[rows, d])


def clear_goals(b):
    b["g_val"].zero_(); b["g_mask"].zero_(); b["g_t"].zero_(); b["g_pad"].fill_(True); b["g_win"].zero_()


def put_goal(D, b, slot, val, mask, t_frames):
    """Write an arbitrary goal (normalised explicit values (B,11), mask (B,11), time in frames from window start (int or (B,))) into slot."""
    B = val.shape[0]; b["g_val"][:, slot] = val * mask; b["g_mask"][:, slot] = mask; b["g_pad"][:, slot] = False
    t = torch.as_tensor(t_frames, device=DEVICE).expand(B) if not torch.is_tensor(t_frames) or t_frames.dim() == 0 else t_frames
    b["g_t"][:, slot] = t.clamp(0, (C + FUT) * P - 1)


# ============================================================================ denoiser
def timestep_embedding(t, dim=128):
    half = dim // 2; a = t.float()[:, None] * torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)[None]
    return torch.cat([torch.cos(a), torch.sin(a)], -1)


class Transformer(nn.Module):
    def __init__(self, d, layers, heads, dropout=0.1):
        super().__init__()
        self.blocks = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, activation="gelu", batch_first=True, norm_first=True), layers, enable_nested_tensor=False); self.norm = nn.LayerNorm(d)

    def forward(self, x, pad): return self.norm(self.blocks(x, src_key_padding_mask=pad))


class HybridDenoiser(nn.Module):
    TXT, VIS, HIS, GOL, WIN = range(5)

    def __init__(self, D, vcfg, d=D_MODEL, layers=LAYERS, heads=N_HEADS):
        super().__init__(); self.v = dict(vcfg); self.two_stage, self.use_goals, self.use_vision, self.use_hist = vcfg["two_stage"], vcfg["use_goals"], vcfg["use_vision"], vcfg["use_hist"]
        layers = layers * (1 if self.two_stage else 2)
        self.text_proj, self.vis_proj, self.cam_emb = nn.Linear(D.DT, d), nn.Linear(D.DV, d), nn.Embedding(2, d)
        self.hist_proj, self.hist_pos = nn.Linear(D.TOK, d), nn.Embedding(H, d)
        self.goal_proj, self.goal_time = nn.Linear(2 * D.EXP_F, d), nn.Embedding((C + FUT) * P, d)
        self.type_emb, self.win_pos = nn.Embedding(5, d), nn.Embedding(C, d); self.t_mlp = nn.Sequential(nn.Linear(128, d), nn.SiLU(), nn.Linear(d, d))
        self.in1, self.tf1, self.out1 = nn.Linear(D.TOK, d), Transformer(d, layers, heads), nn.Linear(d, D.EXP if self.two_stage else D.TOK)
        if self.two_stage: self.in2, self.tf2, self.out2 = nn.Linear(D.LAT + D.EXP, d), Transformer(d, layers, heads), nn.Linear(d, D.LAT)
        self.EXP = D.EXP

    def cond_tokens(self, b, drop):
        B, dev = b["x0"].shape[0], b["x0"].device
        dm = lambda: (torch.rand(B, device=dev) < COND_DROP) if drop else torch.zeros(B, dtype=torch.bool, device=dev)
        toks, pads = [self.text_proj(b["tx"])[:, None] + self.type_emb.weight[self.TXT]], [dm()[:, None]]
        if self.use_vision:
            v = torch.cat([self.vis_proj(b["va"].float()) + self.cam_emb.weight[0], self.vis_proj(b["vw"].float()) + self.cam_emb.weight[1]], 1)
            toks.append(v + self.type_emb.weight[self.VIS]); pads.append(dm()[:, None].expand(-1, v.shape[1]))
        if self.use_hist:
            toks.append(self.hist_proj(b["hist"]) + self.hist_pos.weight[None] + self.type_emb.weight[self.HIS]); pads.append(b["hist_pad"])
        if self.use_goals:   # goal tokens are always the LAST MAX_GOALS conditioning tokens (CFG relies on this)
            toks.append(self.goal_proj(torch.cat([b["g_val"] * b["g_mask"], b["g_mask"]], -1)) + self.goal_time(b["g_t"]) + self.type_emb.weight[self.GOL]); pads.append(b["g_pad"] | dm()[:, None])
        return torch.cat(toks, 1), torch.cat(pads, 1)

    def _run(self, tf, out, tokens, temb, cond, cpad):
        B = tokens.shape[0]; wtok = tokens + self.win_pos.weight[None] + self.type_emb.weight[self.WIN] + temb[:, None]
        pad = torch.cat([cpad, torch.zeros(B, 1 + C, dtype=torch.bool, device=tokens.device)], 1)
        return out(tf(torch.cat([cond, temb[:, None], wtok], 1), pad)[:, -C:])

    def forward(self, x_t, t, cond, cpad):
        temb = self.t_mlp(timestep_embedding(t))
        if not self.two_stage: return self._run(self.tf1, self.out1, self.in1(x_t), temb, cond, cpad)
        E_hat = self._run(self.tf1, self.out1, self.in1(x_t), temb, cond, cpad)
        L_hat = self._run(self.tf2, self.out2, self.in2(torch.cat([x_t[..., self.EXP:], E_hat], -1)), temb, cond, cpad)
        return torch.cat([E_hat, L_hat], -1)


def ckpt_path(variant, seed): return CKPT_DIR / f"{variant}_s{seed}.pt"


def load_model(D, variant, seed):
    ck = torch.load(ckpt_path(variant, seed), map_location=DEVICE, weights_only=False)
    model = HybridDenoiser(D, ck["variant"], d=ck.get("d_model", D_MODEL), layers=ck.get("layers", LAYERS)).to(DEVICE); model.load_state_dict(ck["state_dict"]); model.eval()
    return model, ck


# ============================================================================ diffusion / sampling
_tt = torch.arange(T_DIFF + 1, device=DEVICE) / T_DIFF; _ab = torch.cos((_tt + 0.008) / 1.008 * math.pi / 2) ** 2; AB = (_ab / _ab[0])[1:].clamp(1e-5, 0.9999)


def q_sample(x0, t, noise): return AB[t].sqrt()[:, None, None] * x0 + (1 - AB[t]).sqrt()[:, None, None] * noise


def ddim_sample(D, model, b, inpaint=None, guide=None, cfg=1.0, steps=SAMPLE_STEPS):
    """x0-prediction DDIM. inpaint: dict(mask,val) replacing explicit entries of x0_hat (DiSCo-style baseline).
    guide: dict(mask,val) gradient guidance on the explicit goal error through the network (classifier-guidance baseline).
    cfg: classifier-free guidance on goal tokens (goal tokens are the last MAX_GOALS conditioning tokens)."""
    B = b["x0"].shape[0]
    with torch.no_grad():
        cond, cpad = model.cond_tokens(b, drop=False)
        use_cfg = cfg != 1.0 and model.use_goals and bool((~b["g_pad"]).any())
        if use_cfg: cpad_u = cpad.clone(); cpad_u[:, -MAX_GOALS:] = True
    x = torch.randn(B, C, D.TOK, device=DEVICE); ts = torch.linspace(T_DIFF - 1, 0, steps, device=DEVICE).round().long()
    for i, t in enumerate(ts):
        if guide is not None:
            x = x.detach().requires_grad_(True)
            with torch.enable_grad(), torch.autocast(**AMP):
                x0_hat = model(x, t.expand(B), cond, cpad).float()
                L = (((x0_hat[..., :D.EXP] - guide["val"]) ** 2) * guide["mask"]).sum() / guide["mask"].sum().clamp(min=1.0)
                g = torch.autograd.grad(L, x)[0]
            with torch.no_grad():
                x0_hat = x0_hat.detach() - GUIDE_SCALE * L.detach().sqrt() * g / (g.norm() + 1e-8) * math.sqrt(g.numel() / max(B, 1))
            x = x.detach()
        else:
            with torch.no_grad(), torch.autocast(**AMP):
                x0_hat = model(x, t.expand(B), cond, cpad).float()
                if use_cfg: x0_u = model(x, t.expand(B), cond, cpad_u).float(); x0_hat = x0_u + cfg * (x0_hat - x0_u)
        with torch.no_grad():
            x0_hat[..., D.EXP:] = x0_hat[..., D.EXP:].clamp(-1, 1)
            if inpaint is not None: x0_hat[..., :D.EXP] = torch.where(inpaint["mask"] > 0, inpaint["val"], x0_hat[..., :D.EXP])
            if i == len(ts) - 1:
                if SNAP_LATENTS: x0_hat[..., D.EXP:] = snap_latents(D, x0_hat[..., D.EXP:])
                return x0_hat
            eps = (x - AB[t].sqrt() * x0_hat) / (1 - AB[t]).sqrt(); tp = ts[i + 1]; x = AB[tp].sqrt() * x0_hat + (1 - AB[tp]).sqrt() * eps


def steering_inputs(D, vcfg, cur, goal, k):
    """For window k of a rollout, express `goal` in the mechanism this variant supports. Returns (inpaint, guide, cfg)."""
    clear_goals(cur); inpaint = guide = None; cfg = 1.0
    if goal is None: return inpaint, guide, cfg
    off = goal["gp_rel"] - k * C; B = goal["val"].shape[0]
    if vcfg["steer"] == "token":
        if off >= 0: put_goal(D, cur, 0, goal["val"], goal["m"], off * P + goal["gf"]); cfg = GOAL_CFG
    elif 0 <= off < C:                                     # inpainting / guidance can only act once the goal frame is inside the window
        mask = torch.zeros(B, C, D.EXP, device=DEVICE); val = torch.zeros(B, C, D.EXP, device=DEVICE); sl = slice(goal["gf"] * D.EXP_F, (goal["gf"] + 1) * D.EXP_F)
        mask[:, off, sl] = goal["m"]; val[:, off, sl] = goal["val"]
        if vcfg["steer"] == "inpaint": inpaint = dict(mask=mask, val=val)
        else: guide = dict(mask=mask, val=val)
    return inpaint, guide, cfg


def rollout(D, model, vcfg, b, n_win, goal=None):
    """Autoregressive generation of n_win windows from b['w'], own outputs as history, vision teacher-forced at each window start."""
    B = b["x0"].shape[0]; gen = []; cur = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
    for k in range(n_win):
        inpaint, guide, cfg = steering_inputs(D, vcfg, cur, goal, k)
        x = ddim_sample(D, model, cur, inpaint=inpaint, guide=guide, cfg=cfg); gen.append(x)
        cur["hist"] = torch.cat([cur["hist"][:, C:], x], 1); cur["hist_pad"] = torch.cat([cur["hist_pad"][:, C:], torch.zeros(B, C, dtype=torch.bool, device=DEVICE)], 1)
        cur["w"] = cur["w"] + C; f0 = D.orig_idx_t[D.pf_start_t[cur["e"]] + torch.minimum(cur["w"] * P, D.pf_len_t[cur["e"]] - 1)].cpu()
        if D.vis_a is not None: cur["va"], cur["vw"] = D.vis_a[f0].to(DEVICE), D.vis_w[f0].to(DEVICE)
    return torch.cat(gen, 1)


# ============================================================================ open-loop evaluation
@torch.no_grad()
def evaluate_openloop(D, model, vcfg, eps, horizons=HORIZONS, windows=VAL_WINDOWS, project=True):
    """Teacher-forced history at window starts `windows`; goal at the last frame of the window (inwin) and at every horizon (rollout).
    Per horizon: explicit error, decoded-body FK error, projected-body FK error, approach discontinuity, and the no-goal references."""
    model.eval(); acc = {}; add = lambda k, v: acc.setdefault(k, []).append(v.detach().float().cpu())
    for w0 in windows:
        ok0 = np.array([e for e in eps if D.p_len[e] >= w0 + C])
        if len(ok0) < 5: continue
        B = len(ok0); b = make_batch(D, ok0, B, win_starts=np.full(B, w0), hist_len="max", goals=None); gt = b["x0"]
        x = ddim_sample(D, model, b); Ef, Eg = x[..., :D.EXP].reshape(B, C * P, D.EXP_F), gt[..., :D.EXP].reshape(B, C * P, D.EXP_F)
        add("nogoal_ee_err_cm", (unnorm_pos(D, Ef[..., :3]) - unnorm_pos(D, Eg[..., :3])).norm(dim=-1).mean() * 100)
        body_hat = D.tok.decode(x[..., D.EXP:]).float()
        add("joint_rmse_deg", torch.rad2deg(((body_hat[..., :D.NJ] - b["body_tgt"][..., :D.NJ]) * D.body_std[:D.NJ]).pow(2).mean().sqrt()))
        fk_pos, _ = fk_from_body(D, body_hat); add("fk_consistency_cm", (fk_pos - unnorm_pos(D, Ef[..., :3])).norm(dim=-1).mean() * 100)
        for gp_rel in [C - 1] + list(horizons):
            ok = np.array([e for e in eps if D.p_len[e] >= w0 + gp_rel + 1])
            if len(ok) < 5: continue
            Bh = len(ok); bh = make_batch(D, ok, Bh, win_starts=np.full(Bh, w0), hist_len="max", goals=None)
            tags = ["inwin"] if gp_rel == C - 1 else [f"h{gp_rel}"] + (["outwin"] if gp_rel == OUT_HORIZON else [])
            gp = bh["w"] + gp_rel; gf = P - 1; val = D.exp_pad[D.pf_start_t[bh["e"]] + gp * P + gf]; n_win = gp_rel // C + 1; fi = gp_rel * P + gf
            goal = dict(gp_rel=gp_rel, gf=gf, m=torch.ones(Bh, D.EXP_F, device=DEVICE), val=val)
            gen = rollout(D, model, vcfg, bh, n_win, goal=goal); pred = gen[:, gp_rel, gf * D.EXP_F:(gf + 1) * D.EXP_F]
            Eg_all = gen[..., :D.EXP].reshape(Bh, n_win * C * P, D.EXP_F); body_g = D.tok.decode(gen[..., D.EXP:]).float(); fk_g, _ = fk_from_body(D, body_g)
            gt_prev = D.exp_pad[D.pf_start_t[bh["e"]] + gp * P + gf - 1]
            jump_pred = (unnorm_pos(D, Eg_all[:, fi, :3]) - unnorm_pos(D, Eg_all[:, fi - 1, :3])).norm(dim=-1); jump_gt = (unnorm_pos(D, val[:, :3]) - unnorm_pos(D, gt_prev[:, :3])).norm(dim=-1)
            gen0 = rollout(D, model, vcfg, bh, n_win, goal=None); pred0 = gen0[:, gp_rel, gf * D.EXP_F:(gf + 1) * D.EXP_F]; body_0 = D.tok.decode(gen0[..., D.EXP:]).float(); fk_0, _ = fk_from_body(D, body_0)
            if project:
                fk_gp = fk_pos_of_q(D, project_body(D, body_g, Eg_all)); fk_0p = fk_pos_of_q(D, project_body(D, body_0, gen0[..., :D.EXP].reshape(Bh, n_win * C * P, D.EXP_F)))
            for tag in tags:
                add(f"{tag}_goal_pos_err_cm", (unnorm_pos(D, pred[:, :3]) - unnorm_pos(D, val[:, :3])).norm(dim=-1).mean() * 100)
                add(f"{tag}_goal_rot_err_deg", rot_angle_deg(rot6d_to_mat(pred[:, 3:9]), rot6d_to_mat(val[:, 3:9])).mean())
                add(f"{tag}_goal_grip_err_mm", ((pred[:, 9:] - val[:, 9:]) * D.gr_s_t).abs().mean() * 1000)
                add(f"{tag}_goal_fk_err_cm", (fk_g[:, fi] - unnorm_pos(D, val[:, :3])).norm(dim=-1).mean() * 100)
                add(f"{tag}_goal_jump_excess_cm", (jump_pred - jump_gt).mean() * 100)
                add(f"{tag}_nogoal_pos_err_cm", (unnorm_pos(D, pred0[:, :3]) - unnorm_pos(D, val[:, :3])).norm(dim=-1).mean() * 100)
                add(f"{tag}_nogoal_fk_err_cm", (fk_0[:, fi] - unnorm_pos(D, val[:, :3])).norm(dim=-1).mean() * 100)
                if project:
                    add(f"{tag}_goal_fkproj_err_cm", (fk_gp[:, fi] - unnorm_pos(D, val[:, :3])).norm(dim=-1).mean() * 100)
                    add(f"{tag}_nogoal_fkproj_err_cm", (fk_0p[:, fi] - unnorm_pos(D, val[:, :3])).norm(dim=-1).mean() * 100)
    return {k: round(torch.stack(v).mean().item(), 4) for k, v in acc.items()}


# ============================================================================ closed-loop LIBERO runner
class OnlineEncoder:
    """Frozen vision encoder + text embeddings, identical to 01_prepare_data.py, for online use."""
    def __init__(self, meta):
        from transformers import AutoModel
        name = meta["vision_encoder"]; self.kind = "dinov2" if "dinov2" in name else "siglip"
        m = AutoModel.from_pretrained(name); self.vis = getattr(m, "vision_model", m).to(DEVICE).eval().half()
        MEAN, STD = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) if self.kind == "dinov2" else ((0.5,) * 3, (0.5,) * 3)
        self.mean, self.std = [torch.tensor(v, device=DEVICE).view(1, 3, 1, 1).half() for v in (MEAN, STD)]

    @torch.no_grad()
    def __call__(self, u8):
        x = torch.from_numpy(np.ascontiguousarray(u8)).to(DEVICE)
        if x.dim() == 3: x = x[None]
        if FLIP_180: x = torch.flip(x, dims=(1, 2))
        x = x.permute(0, 3, 1, 2).half().div_(255.0); x = F.interpolate(x, size=(IMG_RES, IMG_RES), mode="bilinear", align_corners=False, antialias=True)
        out = self.vis(pixel_values=(x - self.mean) / self.std); h = out.last_hidden_state
        cls, patches = (h[:, :1], h[:, 1:]) if self.kind == "dinov2" else (out.pooler_output[:, None], h)
        g_ = int(round(patches.shape[1] ** 0.5)); patches = patches.transpose(1, 2).reshape(patches.shape[0], -1, g_, g_)
        return torch.cat([cls, F.adaptive_avg_pool2d(patches, POOL).flatten(2).transpose(1, 2)], 1)


def patch_robosuite():
    """robosuite 1.4 compares numpy ints with mujoco enums, which fails on MuJoCo >= 3. Int-safe joint address lookups."""
    import mujoco, robosuite.utils.binding_utils as bu
    FREE, BALL = int(mujoco.mjtJoint.mjJNT_FREE), int(mujoco.mjtJoint.mjJNT_BALL)
    def _qpos(self, name):
        jid = self.joint_name2id(name); jt, addr = int(self.jnt_type[jid]), int(self.jnt_qposadr[jid]); nd = {FREE: 7, BALL: 4}.get(jt, 1); return addr if nd == 1 else (addr, addr + nd)
    def _qvel(self, name):
        jid = self.joint_name2id(name); jt, addr = int(self.jnt_type[jid]), int(self.jnt_dofadr[jid]); nd = {FREE: 6, BALL: 3}.get(jt, 1); return addr if nd == 1 else (addr, addr + nd)
    bu.MjModel.get_joint_qpos_addr = _qpos; bu.MjModel.get_joint_qvel_addr = _qvel


class LiberoTask:
    """One LIBERO task with a joint-position controller, the standard 50 init states, and the matching demos (for goal sampling)."""
    def __init__(self, D, task_index, camera_res=128, record_cams=()):
        os.environ.setdefault("MUJOCO_GL", "egl"); ensure_libero_repo(); patch_robosuite()
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        import h5py
        self.D, self.info = D, D.meta["tasks"][task_index]; self.task_index = task_index
        suite = benchmark.get_benchmark_dict()[self.info["suite"]]()
        norm = lambda s: re.sub(r"[^a-z]", "", s.lower()); tid = {norm(suite.get_task(i).language): i for i in range(suite.n_tasks)}[norm(self.info["language"])]
        self.task, self.task_id = suite.get_task(tid), tid; self.init_states = suite.get_task_init_states(tid)
        self.env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), self.task.problem_folder, self.task.bddl_file),
                                      camera_heights=camera_res, camera_widths=camera_res, controller="JOINT_POSITION")
        self.env.seed(0); self.record_cams = list(record_cams); self.res = camera_res
        eps = np.where(D.ep_task == task_index)[0]; self.episodes = eps           # demo k <-> init state k (LIBERO collected one demo per init state)
        self.h5 = None
        for p in libero_files(self.info["suite"]):
            with h5py.File(p, "r") as f:
                try: lang = json.loads(f["data"].attrs["problem_info"]).get("language_instruction", "")
                except Exception: lang = Path(p).stem.replace("_demo", "").replace("_", " ")
            if re.sub(r"[^a-z]", "", lang.lower()) == re.sub(r"[^a-z]", "", self.info["language"].lower()): self.h5 = p; break

    def reset(self, init_idx):
        self.env.reset()
        try: self.env.set_init_state(self.init_states[init_idx])
        except Exception as ex: log(f"set_init_state failed ({ex}); using env.reset() placement")
        robot = self.env.env.robots[0]; ctrl = robot.controller
        ctrl.output_max = np.full(7, 0.15); ctrl.output_min = np.full(7, -0.15); ctrl.action_scale = None; ctrl.kp = np.full(7, float(os.environ.get("JP_KP", 150)))
        ctrl.kd = 2 * np.sqrt(ctrl.kp) * ctrl.damping_ratio if hasattr(ctrl, "damping_ratio") else ctrl.kd
        for _ in range(5): obs, *_ = self.env.step(np.zeros(8))
        self.robot = robot; return obs

    def step_to(self, q_target, grip_closed):
        """One env step towards joint targets (rad) with a delta joint-position command; gripper +1 closes, -1 opens."""
        q_now = self.robot._joint_positions; act = np.clip((q_target - q_now) / 0.15, -1, 1)
        obs, rew, done, info = self.env.step(np.concatenate([act, [1.0 if grip_closed else -1.0]]))
        try: succ = bool(self.env.check_success())          # robosuite's `done` is horizon-based unless the wrapper folds success in; ask the benchmark checker explicitly
        except Exception: succ = bool(done)
        return obs, succ, info

    def render(self):
        return {c: self.env.sim.render(camera_name=c, width=256, height=256)[::-1].copy() for c in self.record_cams}

    def demo_states(self, k):
        import h5py
        with h5py.File(self.h5, "r") as f:
            demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1])); return f["data"][demos[k]]["states"][()]

    def close(self): self.env.close()


class Policy:
    """MPC-style closed-loop policy: measured proprio history -> hybrid tokens through the frozen tokenizer encoder,
    online vision features, optional goal, sample one window, execute EXEC frames, replan."""
    def __init__(self, D, model, vcfg, enc, task_index, exec_frames=8, project=False):
        self.D, self.model, self.v, self.enc, self.project = D, model, vcfg, enc, project
        self.exec = exec_frames; self.tx = D.text[task_index][None]; self.reset()

    def reset(self): self.q_hist, self.g_hist, self.pending, self.t = [], [], [], 0

    def observe(self, obs):
        self.q_hist.append(np.asarray(obs["robot0_joint_pos"], np.float32)); self.g_hist.append(np.asarray(obs["robot0_gripper_qpos"], np.float32)[:2])
        self.obs = obs

    def _history_tokens(self):
        D = self.D; q = np.stack(self.q_hist); g = np.stack(self.g_hist); T = q.shape[0]
        n = min(H * P, (T // P) * P)
        if n == 0: return torch.zeros(1, H, D.TOK, device=DEVICE), torch.ones(1, H, dtype=torch.bool, device=DEVICE)
        q, g = q[T - n:], g[T - n:]; qv = np.gradient(q, axis=0) * FPS
        body = ((np.concatenate([q, qv], 1) - D.body_mean.cpu().numpy()) / D.body_std.cpu().numpy()).astype(np.float32)
        with torch.autocast(**AMP): lat, _ = D.tok.encode(torch.from_numpy(body)[None].to(DEVICE))
        lat = lat.float()[0]
        pos, R = fk_with(D.fk_params, torch.from_numpy(q).to(DEVICE)); r6 = mat_to_rot6d(R)
        ex = torch.cat([(pos - D.pos_m_t) / D.pos_s_t, r6, (torch.from_numpy(g).to(DEVICE) - D.gr_m_t) / D.gr_s_t], 1).reshape(-1, D.EXP)
        hyb = torch.cat([ex, lat], 1); n_p = hyb.shape[0]
        hist = torch.zeros(1, H, D.TOK, device=DEVICE); pad = torch.ones(1, H, dtype=torch.bool, device=DEVICE)
        hist[0, H - n_p:] = hyb; pad[0, H - n_p:] = False; return hist, pad

    def plan(self, goal=None):
        """goal: dict(val (1,11) normalised, mask (1,11), t_frames int from now) or None. Returns joint targets (C*P,7) and gripper flags."""
        D = self.D; hist, pad = self._history_tokens()
        va = self.enc(self.obs["agentview_image"]); vw = self.enc(self.obs["robot0_eye_in_hand_image"])
        b = dict(x0=torch.zeros(1, C, D.TOK, device=DEVICE), hist=hist, hist_pad=pad, va=va, vw=vw, tx=self.tx,
                 g_val=torch.zeros(1, MAX_GOALS, D.EXP_F, device=DEVICE), g_mask=torch.zeros(1, MAX_GOALS, D.EXP_F, device=DEVICE),
                 g_t=torch.zeros(1, MAX_GOALS, dtype=torch.long, device=DEVICE), g_pad=torch.ones(1, MAX_GOALS, dtype=torch.bool, device=DEVICE))
        inpaint = guide = None; cfg = 1.0
        if goal is not None:
            tf = int(goal["t_frames"])
            if self.v["steer"] == "token":
                if tf >= 0: put_goal(D, b, 0, goal["val"], goal["mask"], tf); cfg = GOAL_CFG
            elif 0 <= tf < C * P:
                mask = torch.zeros(1, C, D.EXP, device=DEVICE); val = torch.zeros(1, C, D.EXP, device=DEVICE); pi, f = tf // P, tf % P
                mask[:, pi, f * D.EXP_F:(f + 1) * D.EXP_F] = goal["mask"]; val[:, pi, f * D.EXP_F:(f + 1) * D.EXP_F] = goal["val"]
                if self.v["steer"] == "inpaint": inpaint = dict(mask=mask, val=val)
                else: guide = dict(mask=mask, val=val)
        x = ddim_sample(D, self.model, b, inpaint=inpaint, guide=guide, cfg=cfg)
        body = D.tok.decode(x[..., D.EXP:]).float(); ex = x[0, :, :D.EXP].reshape(-1, D.EXP_F)
        q = project_body(D, body, ex[None]) if self.project else (body[..., :D.NJ] * D.body_std[:D.NJ] + D.body_mean[:D.NJ])
        q = q[0].cpu().numpy(); width = ((ex[:, 9:] * D.gr_s_t + D.gr_m_t)[:, 0] - (ex[:, 9:] * D.gr_s_t + D.gr_m_t)[:, 1]).cpu().numpy()
        return q, width < D.gripper_threshold, unnorm_pos(D, ex[:, :3]).cpu().numpy()


def run_episode(task, policy, init_idx, goal_fn=None, max_steps=500, perturb_fn=None, record=False):
    """Run one closed-loop episode. goal_fn(t) -> goal dict or None (called at every replan). perturb_fn(task, t) may modify the sim.
    Returns dict(success, steps, ee_path (T,3), grip (T,), plan_paths [list of (16,3)], frames {cam: [..]})."""
    obs = task.reset(init_idx); policy.reset(); policy.observe(obs)
    ee, grip, plans, frames = [], [], [], {c: [] for c in task.record_cams}
    success, t = False, 0
    while t < max_steps and not success:
        goal = goal_fn(t) if goal_fn else None
        q_plan, g_plan, ee_plan = policy.plan(goal); plans.append((t, ee_plan))
        for k in range(policy.exec):
            if perturb_fn: perturb_fn(task, t)
            obs, done, _ = task.step_to(q_plan[k], bool(g_plan[k])); policy.observe(obs)
            pos, _ = fk_with(task.D.fk_params, torch.from_numpy(np.asarray(obs["robot0_joint_pos"], np.float32))[None].to(DEVICE)); ee.append(pos[0].cpu().numpy())
            grip.append(float(obs["robot0_gripper_qpos"][0] - obs["robot0_gripper_qpos"][1]))
            if record:
                for c, im in task.render().items(): frames[c].append(im)
            t += 1
            if done: success = True; break
            if t >= max_steps: break
    return dict(success=success, steps=t, ee_path=np.array(ee), grip=np.array(grip), plans=plans, frames=frames)
