#!/usr/bin/env python
"""
06_latency.py — wall-clock per generated window (0.8 s of motion) for every variant, on the GPU this runs on.

WHAT IT SETTLES
    Whether the steering mechanisms are real-time. ARDY reports 33 ms for humans at 4 steps; a reviewer will ask the
    same of a manipulator policy. Reported per variant for 4 and 10 DDIM steps, with / without CFG, with / without
    the consistency projection, plus the online DINOv2 cost per replan and the tokenizer encode of a full history.

OUTPUT  $WORK_DIR/results/latency.json     (GPU name recorded; run once per GPU type you want to quote)

    WORK_DIR=... python 06_latency.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import ardy_vla as A
from ardy_vla import log, DEVICE

D = A.load_data(vision=False); rows = []
def timeit(fn, n=30, warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1000
for variant in A.VARIANTS:
    p = A.ckpt_path(variant, A.SEEDS[0])
    if not p.exists(): continue
    model, ck = A.load_model(D, variant, A.SEEDS[0]); vcfg = A.VARIANTS[variant]
    b = dict(x0=torch.zeros(1, A.C, D.TOK, device=DEVICE), hist=torch.randn(1, A.H, D.TOK, device=DEVICE), hist_pad=torch.zeros(1, A.H, dtype=torch.bool, device=DEVICE),
             va=torch.randn(1, 1 + A.POOL ** 2, D.DV, device=DEVICE).half(), vw=torch.randn(1, 1 + A.POOL ** 2, D.DV, device=DEVICE).half(), tx=D.text[:1],
             g_val=torch.zeros(1, A.MAX_GOALS, D.EXP_F, device=DEVICE), g_mask=torch.zeros(1, A.MAX_GOALS, D.EXP_F, device=DEVICE),
             g_t=torch.zeros(1, A.MAX_GOALS, dtype=torch.long, device=DEVICE), g_pad=torch.ones(1, A.MAX_GOALS, dtype=torch.bool, device=DEVICE))
    if vcfg["use_goals"]: A.put_goal(D, b, 0, torch.zeros(1, D.EXP_F, device=DEVICE), torch.ones(1, D.EXP_F, device=DEVICE), 20)
    for steps in (4, 10):
        for cfg in ((1.0, A.GOAL_CFG) if vcfg["use_goals"] else (1.0,)):
            ms = timeit(lambda: A.ddim_sample(D, model, b, cfg=cfg, steps=steps))
            body = D.tok.decode(A.ddim_sample(D, model, b, cfg=cfg, steps=steps)[..., D.EXP:]).float()
            ms_dec = timeit(lambda: D.tok.decode(torch.zeros(1, A.C, D.LAT, device=DEVICE)))
            ms_proj = timeit(lambda: A.project_body(D, body, torch.zeros(1, A.C * A.P, D.EXP_F, device=DEVICE)), n=10)
            rows.append(dict(variant=variant, steps=steps, cfg=cfg, sample_ms=round(ms, 2), decode_ms=round(ms_dec, 2), projection_ms=round(ms_proj, 2), params_M=round(sum(p.numel() for p in model.parameters()) / 1e6, 2)))
            log(f"{variant:24s} steps={steps:2d} cfg={cfg:.1f}  sample {ms:6.1f} ms  decode {ms_dec:5.1f} ms  projection {ms_proj:6.1f} ms")
    model = None; torch.cuda.empty_cache()
enc = A.OnlineEncoder(D.meta); img = np.zeros((128, 128, 3), np.uint8)
ms_vis = timeit(lambda: enc(img)); ms_tok = timeit(lambda: D.tok.encode(torch.zeros(1, A.H * A.P, 2 * D.NJ, device=DEVICE)))
out = dict(gpu=torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu", torch=torch.__version__, rows=rows, vision_encode_ms_per_image=round(ms_vis, 2), history_tokenize_ms=round(ms_tok, 2),
           window_seconds=A.C * A.P / A.FPS, note="one window = 0.8 s of motion; a replan every 8 frames (0.4 s) needs sample + 2 x vision + tokenize < 400 ms to be real-time")
json.dump(out, open(A.RES_DIR / "latency.json", "w"), indent=2); log(f"vision {ms_vis:.1f} ms/image, history tokenize {ms_tok:.1f} ms -> {A.RES_DIR/'latency.json'}")
