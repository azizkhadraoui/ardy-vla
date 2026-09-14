#!/usr/bin/env python
"""
07_aggregate.py — every table and figure, from the JSONs alone. No GPU, no simulator; login node or laptop.

TABLES ($WORK_DIR/results/tables.md, and summary.json for the paper's plotting scripts)
    T1  closed-loop success per suite, mean +- std over seeds, with a 95 % bootstrap CI over episodes (pooled seeds),
        next to the published numbers you fill into PUBLISHED below.
    T2  steering protocols: success, adherence, collision, smoothness per condition; PAIRED bootstrap differences of
        two_stage_goal against every other variant on the same (task, init, protocol) episodes.
    T3  open-loop adherence vs horizon (explicit / body / projected body / no-goal reference), mean +- std over seeds.
    T4  ablation summary (two-stage, history, rollout training, projection, CFG).
    T5  latency.
FIGURES ($WORK_DIR/figures)
    adherence_vs_horizon.png, success_std.png, steering_protocols.png, training_curves.png

    WORK_DIR=... python 07_aggregate.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import ardy_vla as A
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

RES, FIG = A.RES_DIR, A.FIG_DIR; rng = np.random.default_rng(0)
PUBLISHED = {"OpenVLA-OFT": dict(libero_spatial=0.977, libero_object=0.985, libero_goal=0.976, libero_10=0.947),   # fill / verify from the papers before use
             "pi0": dict(libero_spatial=0.968, libero_object=0.986, libero_goal=0.958, libero_10=0.852),
             "Diffusion Policy": dict(libero_spatial=0.788, libero_object=0.925, libero_goal=0.686, libero_10=0.508)}
lines = []; W = lines.append
def boot_ci(x, n=2000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0: return (np.nan, np.nan)
    m = np.array([rng.choice(x, len(x)).mean() for _ in range(n)]); return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))
def msd(v): v = [x for x in v if x is not None and not np.isnan(x)]; return (np.mean(v), np.std(v)) if v else (np.nan, np.nan)
def fmt(m, s=None, d=2): return "-" if np.isnan(m) else (f"{m:.{d}f}" if s is None or np.isnan(s) else f"{m:.{d}f} ± {s:.{d}f}")

# ---------------- load ----------------
ol = {}; cl = {}
for p in sorted(RES.glob("openloop_*.json")):
    j = json.load(open(p)); ol.setdefault(j["variant"], {})[j["seed"]] = j
for p in sorted(RES.glob("closedloop_*.json")):
    j = json.load(open(p))
    if j.get("partial"): continue
    cl.setdefault(j["variant"] + ("+proj" if j.get("project") else ""), {})[j["seed"]] = j
variants = [v for v in list(A.VARIANTS) + [v + "_scale" for v in A.VARIANTS] if v in ol or v in cl or v + "+proj" in cl]
summary = dict(openloop={}, closedloop={}, paired={}, latency=None)

# ---------------- T1 standard success ----------------
W("# Results\n\n## T1 — Closed-loop LIBERO success (standard, no constraint)\n")
suites = A.SUITES
W("| policy | " + " | ".join(suites) + " | mean |"); W("|" + "---|" * (len(suites) + 2))
for name, r in PUBLISHED.items(): W(f"| {name} (published) | " + " | ".join(f"{r.get(s, np.nan):.3f}" for s in suites) + f" | {np.mean([r.get(s, np.nan) for s in suites]):.3f} |")
for v in cl:
    cells = []
    for s in suites:
        per_seed = [j["summary"]["per_suite"].get(s, {}).get("std", {}).get("success", np.nan) for j in cl[v].values()]
        pooled = [r["success"] for j in cl[v].values() for r in j["records"] if r["suite"] == s and r["protocol"] == "std"]
        lo, hi = boot_ci(pooled); m, sd = msd(per_seed); cells.append(f"{fmt(m, sd, 3)} [{lo:.2f}, {hi:.2f}]" if pooled else "-")
        summary["closedloop"].setdefault(v, {}).setdefault("std", {})[s] = dict(mean=m, std=sd, ci=[lo, hi], n=len(pooled))
    allm = msd([j["summary"]["all"].get("std", {}).get("success", np.nan) for j in cl[v].values()])
    W(f"| {v} ({len(cl[v])} seeds) | " + " | ".join(cells) + f" | {fmt(*allm, 3)} |")
W("\nCells: mean ± std over seeds, [95 % bootstrap CI over episodes, seeds pooled]. Published rows are from the respective papers at 50 inits/task; ours use N_INIT inits/task (see JSON).\n")

# ---------------- T2 steering protocols + paired differences ----------------
W("## T2 — Steering protocols (closed loop)\n")
protos = ["P4", "P1", "P2", "P3a", "P3b"]; labels = dict(P4="timed pose", P1="lifted waypoint + box", P2="gripper keyframe", P3a="disturbance, no correction", P3b="disturbance + corrective goal")
W("| variant | condition | n | success | adherence cm | grip err mm | collision | accel cm |"); W("|---|---|---|---|---|---|---|---|")
for v in cl:
    for p in protos:
        per = [j["summary"]["all"].get(p) for j in cl[v].values() if p in j["summary"]["all"]]
        if not per: continue
        row = [fmt(*msd([x["success"] for x in per]), 3), fmt(*msd([x.get("adherence_cm") for x in per])), fmt(*msd([x.get("grip_err_mm") for x in per])), fmt(*msd([x.get("collision") for x in per])), fmt(*msd([x.get("accel_cm") for x in per]), 3)]
        W(f"| {v} | {p} {labels[p]} | {sum(x['n'] for x in per)} | " + " | ".join(row) + " |")
        summary["closedloop"].setdefault(v, {})[p] = dict(success=msd([x["success"] for x in per]), adherence_cm=msd([x.get("adherence_cm") for x in per]), collision=msd([x.get("collision") for x in per]))
if "two_stage_goal" in cl:
    W("\n### Paired differences, two_stage_goal minus baseline, same (task, init, protocol, seed) episodes, 95 % bootstrap CI\n")
    W("| baseline | condition | Δ success | Δ adherence cm | Δ collision |"); W("|---|---|---|---|---|")
    def keyed(j): return {(r["task"], r["init"], r["protocol"]): r for r in j["records"]}
    for v in cl:
        if v == "two_stage_goal": continue
        for p in protos + ["std"]:
            ds, da, dc = [], [], []
            for seed in cl["two_stage_goal"]:
                if seed not in cl[v]: continue
                a, b = keyed(cl["two_stage_goal"][seed]), keyed(cl[v][seed])
                for k in a.keys() & b.keys():
                    if k[2] != p: continue
                    ds.append(a[k]["success"] - b[k]["success"])
                    if "adherence_cm" in a[k] and "adherence_cm" in b[k]: da.append(a[k]["adherence_cm"] - b[k]["adherence_cm"])
                    if "collision" in a[k] and "collision" in b[k]: dc.append(float(a[k]["collision"]) - float(b[k]["collision"]))
            if not ds: continue
            c = lambda x: (f"{np.mean(x):+.3f} [{boot_ci(x)[0]:+.3f}, {boot_ci(x)[1]:+.3f}]" if x else "-")
            W(f"| {v} | {p} | {c(ds)} | {c(da)} | {c(dc)} |"); summary["paired"].setdefault(v, {})[p] = dict(d_success=(np.mean(ds), boot_ci(ds)), d_adherence=(np.mean(da), boot_ci(da)) if da else None, n=len(ds))
W("\nAn interval excluding zero is the sentence 'X steers better than Y'; one including zero is 'no measured difference'. Do not use stronger wording than the interval supports.\n")

# ---------------- T3 open-loop adherence vs horizon ----------------
W("## T3 — Open-loop constraint adherence vs goal horizon (held-out demos, mean ± std over seeds, cm at the goal frame)\n")
tags = ["inwin"] + [f"h{h}" for h in A.HORIZONS]; secs = [f"{(A.C-1)*A.P/A.FPS:.1f}s"] + [f"{h*A.P/A.FPS:.1f}s" for h in A.HORIZONS]
for stream, key in (("explicit stream", "goal_pos_err_cm"), ("decoded body (FK)", "goal_fk_err_cm"), ("body after projection", "goal_fkproj_err_cm"), ("no goal, body (reference)", "nogoal_fk_err_cm")):
    W(f"\n**{stream}**\n\n| variant | " + " | ".join(secs) + " |"); W("|" + "---|" * (len(secs) + 1))
    for v in ol:
        cells = [fmt(*msd([j["metrics"].get(f"{t}_{key}", np.nan) for j in ol[v].values()])) for t in tags]; W(f"| {v} ({len(ol[v])} seeds) | " + " | ".join(cells) + " |")
        summary["openloop"].setdefault(v, {})[key] = [msd([j["metrics"].get(f"{t}_{key}", np.nan) for j in ol[v].values()]) for t in tags]
W("\n**smoothness**: approach jump into the goal frame beyond the demo (cm; positive = teleport)\n\n| variant | " + " | ".join(secs) + " |"); W("|" + "---|" * (len(secs) + 1))
for v in ol: W(f"| {v} | " + " | ".join(fmt(*msd([j["metrics"].get(f"{t}_goal_jump_excess_cm", np.nan) for j in ol[v].values()])) for t in tags) + " |")

# ---------------- T4 ablations ----------------
W("\n## T4 — What each component buys (h8 = 1.6 s, open loop; std = closed-loop success)\n")
W("| comparison | metric | A | B | Δ |"); W("|---|---|---|---|---|")
def ol_m(v, k): return msd([j["metrics"].get(k, np.nan) for j in ol.get(v, {}).values()])[0]
pairs = [("two-stage vs one-stage", "two_stage_goal", "one_stage_goal"), ("history vs none", "two_stage_goal", "two_stage_nohist"), ("rollout-history training", "two_stage_goal_rollout", "two_stage_goal"),
         ("goal tokens vs inpainting", "two_stage_goal", "two_stage_inpaint"), ("goal tokens vs guidance", "two_stage_goal", "two_stage_guidance"), ("scale d=512", "two_stage_goal_scale", "two_stage_goal")]
for name, a, b in pairs:
    for k in ("h8_goal_pos_err_cm", "h8_goal_fk_err_cm", "h8_goal_jump_excess_cm"):
        x, y = ol_m(a, k), ol_m(b, k)
        if not (np.isnan(x) or np.isnan(y)): W(f"| {name} | {k} | {x:.2f} | {y:.2f} | {x-y:+.2f} |")
x = ol_m("two_stage_goal", "h8_goal_fk_err_cm"); y = ol_m("two_stage_goal", "h8_goal_fkproj_err_cm")
if not np.isnan(x) and not np.isnan(y): W(f"| consistency projection | h8 body cm (before / after) | {x:.2f} | {y:.2f} | {y-x:+.2f} |")

# ---------------- T5 latency ----------------
if (RES / "latency.json").exists():
    L = json.load(open(RES / "latency.json")); summary["latency"] = L
    W(f"\n## T5 — Latency on {L['gpu']} (ms per 0.8 s window)\n\n| variant | steps | CFG | sample | decode | projection |"); W("|---|---|---|---|---|---|")
    for r in L["rows"]: W(f"| {r['variant']} | {r['steps']} | {r['cfg']} | {r['sample_ms']} | {r['decode_ms']} | {r['projection_ms']} |")
    W(f"\nvision encoder {L['vision_encode_ms_per_image']} ms/image, history tokenizer {L['history_tokenize_ms']} ms. {L['note']}\n")

open(RES / "tables.md", "w").write("\n".join(lines)); json.dump(summary, open(RES / "summary.json", "w"), indent=1, default=float)
print("\n".join(lines))

# ---------------- figures ----------------
if ol:
    xs = [(A.C - 1) * A.P / A.FPS] + [h * A.P / A.FPS for h in A.HORIZONS]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, key, ttl in zip(axes, ("goal_pos_err_cm", "goal_fk_err_cm", "goal_fkproj_err_cm"), ("explicit stream", "decoded body (FK)", "body after consistency projection")):
        for v in ol:
            m = np.array([msd([j["metrics"].get(f"{t}_{key}", np.nan) for j in ol[v].values()]) for t in tags]); ax.errorbar(xs, m[:, 0], yerr=m[:, 1], marker="o", capsize=3, label=v)
        ref = "two_stage_goal" if "two_stage_goal" in ol else list(ol)[0]; rk = "nogoal_fk_err_cm" if key != "goal_pos_err_cm" else "nogoal_pos_err_cm"
        ax.plot(xs, [msd([j["metrics"].get(f"{t}_{rk}", np.nan) for j in ol[ref].values()])[0] for t in tags], "k--", label="no goal (reference)")
        ax.set_xlabel("goal horizon (s)"); ax.set_ylabel("error at goal frame (cm)"); ax.set_title(ttl); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG / "adherence_vs_horizon.png", dpi=140); plt.close()
if cl:
    vs = list(cl); x = np.arange(len(suites)); wdt = 0.8 / max(len(vs) + len(PUBLISHED), 1); fig, ax = plt.subplots(figsize=(11, 4))
    for i, (name, r) in enumerate(PUBLISHED.items()): ax.bar(x + i * wdt, [r.get(s, np.nan) for s in suites], wdt, alpha=0.4, label=f"{name} (published)")
    for i, v in enumerate(vs):
        g = lambda s, k: summary["closedloop"].get(v, {}).get("std", {}).get(s, {}).get(k, np.nan)
        m = [g(s, "mean") for s in suites]; e = [g(s, "std") for s in suites]; ax.bar(x + (i + len(PUBLISHED)) * wdt, m, wdt, yerr=e, capsize=2, label=v)
    ax.set_xticks(x + 0.4); ax.set_xticklabels(suites); ax.set_ylabel("success rate"); ax.set_title("closed-loop LIBERO, no constraint"); ax.legend(fontsize=7, ncol=2); plt.tight_layout(); plt.savefig(FIG / "success_std.png", dpi=140); plt.close()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4)); x = np.arange(len(protos)); wdt = 0.8 / len(vs)
    for i, v in enumerate(vs):
        s = [summary["closedloop"][v].get(p, {}).get("success", (np.nan, np.nan)) for p in protos]; a = [summary["closedloop"][v].get(p, {}).get("adherence_cm", (np.nan, np.nan)) for p in protos]; c = [summary["closedloop"][v].get(p, {}).get("collision", (np.nan, np.nan)) for p in protos]
        axes[0].bar(x + i * wdt, [m for m, _ in s], wdt, yerr=[e for _, e in s], capsize=2, label=v); axes[1].bar(x + i * wdt, [m for m, _ in a], wdt, yerr=[e for _, e in a], capsize=2); axes[2].bar(x + i * wdt, [m for m, _ in c], wdt, yerr=[e for _, e in c], capsize=2)
    for ax, ttl in zip(axes, ("task success", "constraint error at its frame (cm)", "box collision rate (P1)")): ax.set_xticks(x + 0.4); ax.set_xticklabels([f"{p}\n{labels[p]}" for p in protos], fontsize=7); ax.set_title(ttl); ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(FIG / "steering_protocols.png", dpi=140); plt.close()
hists = {}
for p in sorted(A.CKPT_DIR.glob("*_s*.pt")):
    try:
        import torch; ck = torch.load(p, map_location="cpu", weights_only=False); hists[p.stem] = ck.get("history", [])
    except Exception: pass
if hists:
    fig, ax = plt.subplots(figsize=(7, 4))
    for n, h in hists.items():
        if h: ax.plot([r["step"] for r in h], [r["hyb"] + r["body"] for r in h], label=n, lw=0.8)
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("hybrid + body loss"); ax.legend(fontsize=5, ncol=2); plt.tight_layout(); plt.savefig(FIG / "training_curves.png", dpi=140); plt.close()
# the paper's own artefacts on one run: the tables as text, summary.json attached, every figure as an image
A.wandb_init("aggregate", name="aggregate", config=dict(n_openloop=sum(len(v) for v in ol.values()), n_closedloop=sum(len(v) for v in cl.values())))
A.wandb_summary(summary)
A.wandb_images({f"figures/{f.stem}": f for f in sorted(FIG.glob("*.png"))})
A.wandb_save(RES / "tables.md", RES / "summary.json")
A.wandb_finish()
print(f"\ntables -> {RES/'tables.md'}   summary -> {RES/'summary.json'}   figures -> {FIG}")
