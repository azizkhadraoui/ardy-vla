#!/usr/bin/env python
"""Dump the raw camera frames the trained vision encoder needs, in the dataset's own frame order.

The precomputed DINOv2 tokens cannot be used to train an encoder -- they ARE the frozen encoder's output. A
from-scratch CNN has to see pixels, and it has to see them augmented differently on every epoch, which a
precomputed feature file cannot provide.

    338,575 frames x 128 x 128 x 3 uint8 = 16.6 GB per camera, indexed exactly like proprio.npz's frames, so
    make_batch can gather them with the same indices it uses for everything else.

    WORK_DIR=... DATA_DIR=... python dump_frames.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, h5py
import ardy_vla as A
from ardy_vla import log

meta = json.load(open(A.DATA / "meta.json"))
pr = np.load(A.DATA / "proprio.npz")
ep_start, ep_len, ep_task = pr["episode_start"], pr["episode_len"], pr["episode_task"]
n_frames = int(ep_len.sum())
KEY = {"agentview": "agentview_rgb", "wrist": "eye_in_hand_rgb"}
RES = int(os.environ.get("RAW_RES", 128))

outs = {}
for cam in KEY:
    p = A.DATA / f"raw_{cam}.npy"
    if p.exists() and os.environ.get("FORCE", "0") != "1":
        log(f"{p.name} exists; set FORCE=1 to redo"); sys.exit(0)
    outs[cam] = np.lib.format.open_memmap(p, mode="w+", dtype=np.uint8, shape=(n_frames, RES, RES, 3))

t0 = time.time(); done = 0
for ti, t in enumerate(meta["tasks"]):
    path = [p for p in A.libero_files(t["suite"]) if os.path.basename(p) == t["file"]][0]
    eps = np.where(ep_task == ti)[0]
    with h5py.File(path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
        for k, e in enumerate(eps):
            s, T = int(ep_start[e]), int(ep_len[e]); obs = f["data"][demos[k]]["obs"]
            for cam in KEY:
                arr = obs[KEY[cam]][()]
                assert arr.shape[1] == RES and arr.shape[2] == RES, f"{cam} is {arr.shape}, expected {RES}"
                outs[cam][s:s + T] = arr[:T]
            done += 1
    if ti % 5 == 0:
        log(f"  task {ti}/{len(meta['tasks'])}, {done} episodes, {time.time()-t0:.0f}s")
for cam in outs: outs[cam].flush()
log(f"wrote raw_agentview.npy and raw_wrist.npy ({n_frames} frames, {RES}px) in {(time.time()-t0)/60:.1f} min")
