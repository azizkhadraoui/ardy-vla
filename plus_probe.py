#!/usr/bin/env python
"""Does each LIBERO-plus perturbation actually reach the policy's inputs?

Three things can silently no-op and would leave a category measuring nothing:
  robot   the perturbation is a MountedPanda{N} class differing only in init_qpos, but our reset loads the
          ORIGINAL task's saved sim state, which contains the original robot joints. Checks that the arm is
          where the variant wants it after reset, i.e. that PlusTask._post_set_state does its job.
  camera  the view angle is parsed from the task name by the plus env. Checks the agentview image differs.
  noise   applied in the wrapper's step() to agentview only. Checks the image differs after a step.
  lang    checks the instruction really is a rewrite of the original.

    LIBERO_DIR=.../LIBERO-plus python plus_probe.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "egl")
from pathlib import Path
import numpy as np, torch
import ardy_vla as A
from ardy_vla import log

SUITE = os.environ.get("PROBE_SUITE", "libero_10")
D = A.load_data(vision=False)
A.ensure_libero_repo()
from libero.libero import benchmark
bench = benchmark.get_benchmark_dict()[SUITE]()
by_name = {bench.get_task(i).name: i for i in range(bench.n_tasks)}
# the classification lives only in the plus checkout; phase 1 runs against the original LIBERO and never needs it
_cls_path = Path(os.environ.get("PLUS_DIR", A.LIBERO_DIR)) / "libero" / "libero" / "benchmark" / "task_classification.json"
cls = json.load(open(_cls_path))[SUITE] if _cls_path.exists() else []
log(f"{SUITE}: {bench.n_tasks} tasks registered, {len(cls)} classified variants")
import re
stems = {re.sub(r"_demo\.hdf5$", "", t["file"]): ti for ti, t in enumerate(D.meta["tasks"]) if t["suite"] == SUITE}


def orig_of(name):
    c = [s for s in stems if name == s or name.startswith(s + "_")]
    return max(c, key=len) if c else None


# The unperturbed reference has to come from the ORIGINAL LIBERO: the plus task map contains only perturbed
# variants, so LiberoTask's language lookup finds nothing there. Two processes, one npz between them.
REF = os.environ.get("REF_NPZ", "")
base_name = sorted(stems)[0]; ti = stems[base_name]
if os.environ.get("REF_WRITE"):
    log(f"writing reference for task {ti}: {base_name}")
    ref = A.LiberoTask(D, ti); obs_ref = ref.reset(0)
    np.savez(REF, q=np.asarray(obs_ref["robot0_joint_pos"], np.float64),
             img=obs_ref["agentview_image"].astype(np.int16), task=ti, name=base_name)
    log(f"  ref joints {np.round(np.asarray(obs_ref['robot0_joint_pos'], np.float64), 3)}")
    ref.close(); sys.exit(0)
z = np.load(REF); q_ref = z["q"]; img_ref = z["img"]
assert int(z["task"]) == ti, f"reference is for task {z['task']}, this run wants {ti}"
log(f"reference task {ti}: {base_name}  joints {np.round(q_ref, 3)}")

rows = []
for cat in ["Robot Initial States", "Camera Viewpoints", "Sensor Noise", "Language Instructions", "Light Conditions", "Background Textures", "Objects Layout"]:
    cand = [r["name"] for r in cls if r["category"] == cat and r["name"] in by_name and orig_of(r["name"]) == base_name]
    if not cand: log(f"{cat}: no variant of the reference task; skipped"); continue
    name = sorted(cand)[0]; pid = by_name[name]
    try:
        t = A.PlusTask(D, ti, bench, pid); obs = t.reset(0)
        q = np.asarray(obs["robot0_joint_pos"], np.float64)
        img = obs["agentview_image"].astype(np.int16)
        obs2, _, _ = t.step_to(np.asarray(obs["robot0_joint_pos"], np.float32), False)
        img2 = obs2["agentview_image"].astype(np.int16)
        row = dict(category=cat, name=name[-70:], joint_delta=float(np.abs(q - q_ref).max()),
                   img_mad=float(np.abs(img - img_ref).mean()), img_mad_after_step=float(np.abs(img2 - img_ref).mean()),
                   language=t.task.language, lang_changed=t.task.language.strip().lower() != D.meta["tasks"][ti]["language"].strip().lower())
        t.close()
    except Exception as ex:
        row = dict(category=cat, name=name[-70:], error=f"{type(ex).__name__}: {str(ex)[:200]}")
    rows.append(row); log(f"{cat:22s} " + json.dumps({k: v for k, v in row.items() if k != "category"})[:220])

log("")
log("VERDICT (what must be non-zero for the category to measure anything)")
for r in rows:
    if "error" in r: log(f"  {r['category']:22s} ERROR {r['error'][:100]}"); continue
    if r["category"] == "Robot Initial States": ok = r["joint_delta"] > 1e-3; what = f"joint delta {r['joint_delta']:.4f} rad"
    elif r["category"] == "Language Instructions": ok = r["lang_changed"]; what = f"instruction {'rewritten' if r['lang_changed'] else 'IDENTICAL'}"
    elif r["category"] == "Sensor Noise": ok = r["img_mad_after_step"] > 1.0; what = f"image MAD after step {r['img_mad_after_step']:.2f}"
    else: ok = r["img_mad"] > 1.0; what = f"image MAD {r['img_mad']:.2f}"
    log(f"  {r['category']:22s} {'OK  ' if ok else 'NO-OP'} {what}")
json.dump(rows, open(A.RES_DIR / f"plus_probe_{SUITE}.json", "w"), indent=1)
