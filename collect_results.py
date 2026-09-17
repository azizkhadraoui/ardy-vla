#!/usr/bin/env python
"""One table of everything the overnight chain has produced so far.

    WORK_DIR=... python collect_results.py
"""
import os, sys, json, glob, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import ardy_vla as A

R = A.RES_DIR
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def closed(path):
    d = json.load(open(path)); r = d.get("records", [])
    if not r: return None
    out = {"n": len(r), "partial": d.get("partial", False), "knobs": d.get("knobs", {})}
    k = sum(x["success"] for x in r); out["succ"] = k / len(r); out["lo"], out["hi"] = wilson(k, len(r))
    for s in SUITES:
        v = [x["success"] for x in r if x["suite"] == s]
        out[s] = (sum(v) / len(v), len(v)) if v else None
    return out


print("=" * 108)
print("CLOSED LOOP  (std success; suite columns show mean and n)")
print("=" * 108)
print(f"{'run':30s}{'n':>6s}{'success':>10s}{'95% CI':>16s}   " + "".join(f"{s.replace('libero_',''):>10s}" for s in SUITES))
rows = []
for p in sorted(glob.glob(str(R / "closedloop_*.json"))):
    name = os.path.basename(p)[len("closedloop_"):-len(".json")]
    try: c = closed(p)
    except Exception as e: print(f"{name:30s} unreadable: {e}"); continue
    if c is None: continue
    ci = f"[{c['lo']:.3f},{c['hi']:.3f}]"
    cells = "".join((f"{c[s][0]:>7.3f}/{c[s][1]:<3d}" if c[s] else f"{'-':>10s}") for s in SUITES)
    flag = "" if not c["partial"] else "  (partial)"
    print(f"{name:30s}{c['n']:>6d}{c['succ']:>10.3f}{ci:>16s}   {cells}{flag}")
    rows.append((name, c))

base = next((c for n, c in rows if n.endswith("two_stage_goal_s0")), None)
if base:
    print("\ndeltas against the unseeded baseline (%.3f):" % base["succ"])
    for n, c in rows:
        if n.endswith("two_stage_goal_s0"): continue
        if c["n"] < 200: continue
        print(f"  {n:34s} {c['succ']:.3f}   {c['succ'] - base['succ']:+.3f}")

print("\n" + "=" * 108)
print("OPEN LOOP  (cm at the goal frame)")
print("=" * 108)
keys = ["inwin_goal_pos_err_cm", "h8_goal_pos_err_cm", "h16_goal_pos_err_cm", "h8_goal_fk_err_cm",
        "h8_goal_fkproj_err_cm", "h16_goal_fkproj_err_cm", "fk_consistency_cm", "joint_rmse_deg"]
print(f"{'run':34s}" + "".join(f"{k.replace('_err_cm','').replace('_goal','')[:11]:>12s}" for k in keys))
for p in sorted(glob.glob(str(R / "openloop_*.json"))):
    name = os.path.basename(p)[len("openloop_"):-len(".json")]
    m = json.load(open(p)).get("metrics", {})
    print(f"{name:34s}" + "".join(f"{m[k]:>12.3f}" if k in m else f"{'-':>12s}" for k in keys))

print("\n" + "=" * 108)
print("REPLAY CEILING  (demo trajectories through the actuation path, no model)")
print("=" * 108)
for p in sorted(glob.glob(str(R / "replay_ceiling*.json"))):
    d = json.load(open(p)); tag = os.path.basename(p)[len("replay_ceiling"):-len(".json")] or " (default)"
    print(f"{tag:12s} kp={d.get('jp_kp')}  modes={d.get('modes')}  n_demo={d.get('n_demo')}")
    for s, v in d.get("summary", {}).items():
        print(f"    {s:16s} " + "  ".join(f"{m} {v[m]:.3f}" for m in sorted(v)))

pp = R / "s0_probe.json"
if pp.exists():
    print("\n" + "=" * 108); print("STAGE-0 PROBE"); print("=" * 108)
    for k, v in json.load(open(pp)).items():
        print(f"  {k:44s} {v if not isinstance(v, float) else round(v, 3)}")

for name, f in (("tokenizer", "tokenizer/tokenizer_eval.json"),):
    p = A.WORK_DIR / f
    if p.exists():
        e = json.load(open(p)); print(f"\n{name}: mean {e['mean_rmse_deg']} deg, per joint {e['per_joint_rmse_deg']}")

ck = sorted(glob.glob(str(A.CKPT_DIR / "*.pt")))
print("\ncheckpoints: " + ", ".join(os.path.basename(c) for c in ck))
