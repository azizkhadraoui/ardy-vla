#!/usr/bin/env python
"""
09_eval_plus.py -- generalisation on LIBERO-plus (Fu et al. 2025): the same 40 tasks under seven perturbation
dimensions, one init state per perturbed variant, LIBERO's own success checker.

WHY A SEPARATE BENCHMARK
    Standard LIBERO evaluates on the 50 init states the demos were collected from; a model can score well by
    memorising scene -> trajectory. LIBERO-plus keeps the tasks and changes what a memoriser relies on: camera
    pose, lighting, background texture, sensor noise, distractor objects and target displacement, the robot's
    initial state, and the wording of the instruction. Published VLAs drop from ~95% to 15-80% here; camera and
    initial-state changes are where they break. It is the right place to ask whether a memory token trained to
    predict the consequences of actions (MEM_WM) generalises better than one that only summarises frames (MEM_K).

PROTOCOL
    LIBERO_DIR points at the LIBERO-plus checkout (a drop-in replacement for the LIBERO package). Per suite,
    PLUS_N variants are drawn per perturbation category with a fixed seed (the same variants for every checkpoint
    compared), each run once from its single init state with the policy configured exactly as in the std eval
    (EXEC=4, projection, ensembling, hysteresis gripper). Language variants get their rewritten instruction
    embedded online with the same T5 recipe as text_emb.npy; every other variant keeps the original embedding.

OUTPUT  $WORK_DIR/results/plus_{variant}_s{seed}{TAG}.json: per-episode records, success per category x suite,
        and, when REF_RESULT names the std closed-loop json of the same checkpoint, the drop from the clean score.

    LIBERO_DIR=.../LIBERO-plus VARIANT=two_stage_goal SEED=51 MEM_K=8 LONG_W=3 PLUS_N=30 PLUS_SUITES=libero_10 python 09_eval_plus.py
    knobs: PLUS_N (variants per category per suite, 30), PLUS_SUITES, PLUS_CATS, PLUS_SEED (variant draw, 0), EXEC (4), PROJECT (1), TAG, MAXTASKS
"""
import os, sys, json, time, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch
import ardy_vla as A
from ardy_vla import log, DEVICE

VARIANT = os.environ["VARIANT"]; SEED = int(os.environ.get("SEED", 0)); vcfg = A.VARIANTS[VARIANT.replace("_scale", "")]
PLUS_N = int(os.environ.get("PLUS_N", 30)); PLUS_SEED = int(os.environ.get("PLUS_SEED", 0))
PLUS_SUITES = os.environ.get("PLUS_SUITES", ",".join(A.SUITES)).split(",")
CATS = ["Camera Viewpoints", "Robot Initial States", "Language Instructions", "Light Conditions", "Background Textures", "Sensor Noise", "Objects Layout"]
PLUS_CATS = os.environ.get("PLUS_CATS", "|".join(CATS)).split("|")
EXEC = int(os.environ.get("EXEC", 4)); PROJECT = os.environ.get("PROJECT", "1") == "1"; TAG = os.environ.get("TAG", "")
MAXTASKS = int(os.environ.get("MAXTASKS", 0)); REF_RESULT = os.environ.get("REF_RESULT", "")
MAX_STEPS = dict(libero_spatial=220, libero_object=280, libero_goal=300, libero_10=520)
dst = A.RES_DIR / f"plus_{VARIANT}_s{SEED}{TAG}.json"
if dst.exists() and os.environ.get("FORCE", "0") != "1": log(f"{dst.name} exists; skipping"); sys.exit(0)

cls_path = A.LIBERO_DIR / "libero" / "libero" / "benchmark" / "task_classification.json"
assert cls_path.exists(), f"LIBERO_DIR={A.LIBERO_DIR} is not a LIBERO-plus checkout (no {cls_path.name})"
classification = json.load(open(cls_path))

D = A.load_data(vision=False)
ck = torch.load(A.CKPT_DIR / f"{VARIANT}_s{SEED}.pt", map_location=DEVICE, weights_only=False)
if ck.get("mem_wm", 0) != A.MEM_WM: raise RuntimeError(f"checkpoint was trained with MEM_WM={ck.get('mem_wm', 0)}; set MEM_WM accordingly")
model = A.HybridDenoiser(D, ck["variant"], d=ck.get("d_model", A.D_MODEL), layers=ck.get("layers", A.LAYERS), w_grip=ck.get("w_grip", 0.0)).to(DEVICE)
A.load_compat(model, ck["state_dict"]); model.eval(); enc = A.OnlineEncoder(D.meta)
log(f"checkpoint {VARIANT}_s{SEED}: step {ck.get('step')}, vis_mode {ck.get('vis_mode', 'feat')}, mem_wm {ck.get('mem_wm', 0)}")

# the online T5 must reproduce the stored embeddings, or the language variants would measure the encoder, not the policy
orig = [t["language"] for t in D.meta["tasks"]]
cos = torch.nn.functional.cosine_similarity(A.embed_text(orig), D.text, dim=-1)
log(f"online T5 vs text_emb.npy: cosine min {cos.min():.4f} mean {cos.mean():.4f}")
assert cos.min() > 0.98, "online text embedding does not match the training embeddings"

A.ensure_libero_repo()
from libero.libero import benchmark
records = []; t_all = time.time()
A.wandb_init("plus", VARIANT, SEED, config=dict(ckpt_step=ck.get("step"), plus_n=PLUS_N, plus_seed=PLUS_SEED, suites=PLUS_SUITES, exec_frames=EXEC, project=PROJECT))


def original_of(name, stems):
    """The original task a variant name extends: the longest original stem it starts with."""
    c = [s for s in stems if name == s or name.startswith(s + "_")]
    return max(c, key=len) if c else None


n_done = 0
for suite in PLUS_SUITES:
    bench = benchmark.get_benchmark_dict()[suite](); by_name = {bench.get_task(i).name: i for i in range(bench.n_tasks)}
    stems = {re.sub(r"_demo\.hdf5$", "", t["file"]): ti for ti, t in enumerate(D.meta["tasks"]) if t["suite"] == suite}
    rng = np.random.default_rng(PLUS_SEED + 17 * A.SUITES.index(suite))
    picked = []
    for cat in PLUS_CATS:
        names = sorted(r["name"] for r in classification[suite] if r["category"] == cat and r["name"] in by_name)
        take = rng.choice(len(names), min(PLUS_N, len(names)), replace=False)
        picked += [(cat, names[i], {r["name"]: r for r in classification[suite]}[names[i]].get("difficulty_level")) for i in sorted(take)]
    log(f"{suite}: {bench.n_tasks} variants, {len(picked)} drawn ({PLUS_N} per category)")
    for cat, name, level in picked:
        if MAXTASKS and n_done >= MAXTASKS: break
        stem = original_of(name, stems)
        if stem is None: log(f"  no original for {name}; skipped"); continue
        ti = stems[stem]; pid = by_name[name]; t0 = time.time()
        try:
            task = A.PlusTask(D, ti, bench, pid)
        except Exception as ex:
            log(f"  {name}: env failed ({type(ex).__name__}: {str(ex)[:120]})"); records.append(dict(suite=suite, category=cat, level=level, name=name, task=ti, plus_id=pid, error=str(ex)[:300], success=False, steps=0)); continue
        policy = A.Policy(D, model, vcfg, enc, ti, exec_frames=EXEC, project=PROJECT)
        lang = task.task.language
        if cat == "Language Instructions": policy.tx = A.embed_text([lang])
        policy.seed_base = (1000003 * ti + 10007 * pid + 7 * SEED) if A.SEED_SAMPLER else None
        out = A.run_episode(task, policy, 0, max_steps=MAX_STEPS[suite]); task.close()
        rec = dict(suite=suite, category=cat, level=level, name=name, task=ti, plus_id=pid, language=lang, success=bool(out["success"]), steps=int(out["steps"]), secs=round(time.time() - t0, 1))
        records.append(rec); n_done += 1
        log(f"  {suite:14s} {cat[:8]:8s} L{level} t{ti:2d} {'ok  ' if rec['success'] else 'fail'} {rec['steps']:3d} steps  {name[-60:]}  ({rec['secs']}s)")
        if n_done % 10 == 0:
            json.dump(dict(variant=VARIANT, seed=SEED, records=records, partial=True), open(dst, "w"))
            A.wandb_log({"plus/done": n_done, "plus/running_success": float(np.mean([r["success"] for r in records]))})

# ---- summaries -------------------------------------------------------------------------------------------
def rate(rs): return dict(n=len(rs), success=float(np.mean([r["success"] for r in rs]))) if rs else dict(n=0, success=None)
suites_seen = list(dict.fromkeys(r["suite"] for r in records)); cats_seen = list(dict.fromkeys(r["category"] for r in records))
summary = dict(all=rate(records),
               per_category={c: rate([r for r in records if r["category"] == c]) for c in cats_seen},
               per_suite={s: rate([r for r in records if r["suite"] == s]) for s in suites_seen},
               per_suite_category={s: {c: rate([r for r in records if r["suite"] == s and r["category"] == c]) for c in cats_seen} for s in suites_seen},
               per_level={str(l): rate([r for r in records if r.get("level") == l]) for l in sorted({r.get("level") for r in records}, key=lambda x: (x is None, x))},
               errors=sum("error" in r for r in records))
if REF_RESULT and os.path.exists(REF_RESULT):
    ref = json.load(open(REF_RESULT)); ref_suite = {s: v["std"]["success"] for s, v in ref.get("summary", {}).get("per_suite", {}).items() if "std" in v}
    summary["clean_reference"] = dict(file=os.path.basename(REF_RESULT), per_suite=ref_suite,
                                      drop_per_suite={s: ref_suite[s] - summary["per_suite"][s]["success"] for s in suites_seen if s in ref_suite})
json.dump(dict(variant=VARIANT, seed=SEED, project=PROJECT, exec_frames=EXEC, plus_n=PLUS_N, plus_seed=PLUS_SEED, mem_wm=A.MEM_WM, mem_k=A.MEM_K,
               records=records, summary=summary, secs=round(time.time() - t_all), partial=False), open(dst, "w"), indent=1)
log(f"LIBERO-plus {summary['all']['n']} episodes: success {summary['all']['success']:.3f}  ({(time.time() - t_all) / 3600:.2f} h)")
for c in cats_seen: log(f"  {c:22s} {summary['per_category'][c]['success']:.3f}  (n={summary['per_category'][c]['n']})")
for s in suites_seen: log(f"  {s:22s} {summary['per_suite'][s]['success']:.3f}" + (f"  clean {summary['clean_reference']['per_suite'][s]:.3f}" if "clean_reference" in summary and s in summary["clean_reference"]["per_suite"] else ""))
A.wandb_summary(summary["all"], prefix="plus/"); A.wandb_summary({c: v["success"] for c, v in summary["per_category"].items()}, prefix="plus/category/")
A.wandb_summary({s: v["success"] for s, v in summary["per_suite"].items()}, prefix="plus/suite/"); A.wandb_finish()
