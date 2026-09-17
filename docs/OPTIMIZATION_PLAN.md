# ardy-vla — plan to approach SOTA LIBERO success at small latency

Compiled 17 September 2026 from the runs on Panther (Tesla V100-PCIE-16GB, repo commit `0dd98c5`), a
four-reader / five-plan / critic / judge study of the code and results, and three diagnostics run afterwards.
Every number is measured unless marked *est.* Result files are under `runs/results/` of the repo.

---

## 1. Where we stand

### The numbers that matter

| what | value | source |
|---|---|---|
| Closed-loop std success, 4-suite, 800 episodes (20 inits x 40 tasks) | **0.220** [Wilson 0.193, 0.250] | `closedloop_two_stage_goal_s0.json` |
| per suite: spatial / object / goal / libero_10 | 0.400 / **0.020** / 0.425 / 0.035 | same |
| bowl pick-and-place (~12 cm target) vs small grocery item (~4 cm), same task template | 0.468 (131/280) vs **0.023** (5/220) — disjoint CIs | analyst report |
| Open-loop goal adherence, explicit stream, 0.6 -> 3.2 s horizon | 0.58 / 0.49 / 0.49 / 0.50 / **0.51 cm**, flat | `openloop_two_stage_goal_s0.json` |
| same, decoded body (what is executed) | 1.43 / 2.24 / 4.32 / 5.52 / 5.90 cm | same |
| same, no goal (reference) | 3.11 / 3.72 / 6.23 / 7.97 / 8.33 cm | same |
| `fk_consistency_cm` (executed stream vs own explicit stream, in-window) | 1.61 cm | same |
| `joint_rmse_deg` (rolled-out body vs demo) vs tokenizer floor | 3.07 deg vs 0.863 deg | same, `tokenizer_eval.json` |
| Latency per 0.8 s window, V100: 4 DDIM steps / 10 steps (cfg 1.0) | **38.5 ms** / 95.9 ms; projection +235 ms | `latency.json` |
| Training | 22.08M params, 30k steps x batch 128, 47 min, one seed | `03_train.py`, checkpoint |

### What the results already prove — and must be kept

With a goal token the explicit EE stream holds **0.49-0.58 cm at every horizon to 3.2 s**, against 2.9-4.4 cm
without one. It is measured on held-out demos and it lives entirely in stage 1 of the denoiser
(`HybridDenoiser`, `ardy_vla.py`). Nothing below touches that pathway; every gate carries
`h8_goal_pos_err_cm <= 0.6` and `h16_goal_pos_err_cm <= 0.7` so the claim cannot be traded away for success rate.

### Diagnosis — what is, and is not, the bottleneck

Ordered by strength of evidence. Items 1-3 are direct closed-loop experiments run for this plan, not analogies.

**1. The gripper decision rule is a hard cap on small objects.** *Replay ceiling* (`replay_demos.py`): each demo's
own joint trajectory pushed through the identical actuation path (`LiberoTask.step_to`, joint-position PD kp=150,
+-0.15 rad clamp), with no model in the loop.

<!-- REPLAY_TABLE -->
| suite (5 demos/task) | demo gripper command, targets as-is | + lead target q[t+3] | **policy's rule: close iff width < 5.28 cm** |
|---|---|---|---|
| libero_spatial | 0.64 | 0.64 | 0.64 |
| libero_object | 0.84 | 0.82 | **0.30** |
| libero_goal | *pending* | | |
| libero_10 | *pending* | | |
<!-- /REPLAY_TABLE -->

On *perfect* object-suite trajectories the width-threshold rule alone drops success from 0.84 to 0.30: wide items
(milk, orange juice: 1.00 -> 0.00) are held at a width above the global 5.28 cm midpoint and read as "open"; the
close is timed off a lagging measurement. `Policy.plan` (`ardy_vla.py`, `width < D.gripper_threshold`) inherits
exactly this rule, and the demo command it should be predicting is already stored (`actions[:, -1]` in
`proprio.npz`, written by `01_prepare_data.py`) and never learned. Closed-loop evidence agrees: failures close to
0.10 cm on air; in the P2 protocol the gripper-width error at the keyframe is 46.5 mm and the only two episodes
with the gripper in the right state are the two successes.

**2. The actuation path is itself a ceiling on spatial.** Replay as-is reaches only 0.64 on libero_spatial
(task 9 "bowl on the wooden cabinet" 0.00) even with the demo's own gripper command: re-tracking OSC-collected
demos through a joint-position PD with a ~0.16 s lag (analytic, kp=150 -> omega_n 12.25 rad/s, zeta 1) loses
the precise picks. The policy's 0.40 on spatial is 62% of that ceiling. This is tunable *without the model*:
the replay harness measures the ceiling of any controller setting in ~2 min per suite.

**3. Executing the accurate stream is not sufficient for small objects.** Re-running libero_object with the
explicit stream executed through the consistency projection (`PROJECT=1`, `closedloop_two_stage_goal_s0_proj.json`,
n=200): **0.060 vs 0.020**. The plan itself is wrong for a 4 cm item, so the ~1.6 cm stream inconsistency and the
~1 cm tokenizer floor are second-order there. They still matter for bowl precision (see Stage 1).

**4. Perception cannot localise a 4 cm object.** Each camera is frozen DINOv2-S, 16x16 patches mean-pooled to
**4x4 cells of 32x32 native px** — *est.* 17-23 cm of table per cell (`01_prepare_data.py` `encode`,
`OnlineEncoder`). A grocery item is ~1/25 of a cell's area; sub-cell position survives only if DINOv2 happens to
encode it, and mean pooling erases it. No augmentation is possible because features are precomputed; one snapshot
per replan. Every published system at >= 0.94 either fine-tunes its encoder (OpenVLA: frozen vision -22.7 pts)
or trains a small CNN from scratch with crop/shift augmentation (MINERVA: 0.54M params, 95.05, 8 ms; Diffusion
Policy; robomimic: removing shift augmentation = 35-47% relative drop). Published models find object the *easiest*
suite (OFT 0.984, pi0 0.988); here it is the hardest — the ordering inverts exactly where localisation binds.

**5. Under-trained relative to every from-scratch system that reaches 0.95.** 30k steps (3.84M windows) versus
MINERVA 90-120k, OFT 50-150k. Loss was still falling at 30k (`training_curves.png`).

**6. Replanning discards half of every plan with fresh noise and no ensembling.** `EXEC=8`: 16 frames planned,
8 executed, next plan from `torch.randn` with no warm start. MINERVA replans every step with ACT-style temporal
ensembling; its chunk-8 ablation costs -3.2 pts, SmolVLA's chunk ablations agree.

**7. Code-verified train/inference mismatches** (all cheap to fix, all in one retrain):
`03_train.py` decodes the *continuous* `L_hat` for `l_body`/`l_con` while `ddim_sample` clamps and grid-snaps the
latent before `Policy.plan` decodes it; `cond_tokens` drops text/vision/goals independently at 10% but CFG only
ever masks goal tokens, so vision/text dropout trains a pathway inference never uses; the training image is the
window's first frame while inference uses the last history frame (50 ms skew); `l_body` weights the 7 velocity
dims (never executed) equal to positions; history augmentation adds a 3 cm offset to the explicit history only,
teaching the model the explicit pose is unreliable to +-3 cm; the FSQ decoder is trained on full episodes and
48-frame crops but deployed on 4 isolated patches (never measured in that mode).

**8. Measurement noise.** fp16 sampling, unseeded: the same checkpoint on the same task gave 9/20 and 4/20;
per-init agreement across reruns is at chance. n=20/task is +-0.17 per task; +-0.03 pooled over 800.

### The gap, attributed (analyst, from per-task records; inference)

~65-70% precision (gripper rule + localisation + executed-stream error, dominating the 13 small-object / contact
tasks and most of libero_10's constituents), ~15% long-horizon compounding, ~10% spatial-relation disambiguation.

---

## 2. Target and constraints

### Where SOTA sits, and which class we are in (verified in the SOTA report; U = unverified)

| system | spatial / object / goal / long -> mean | params | vision | executed action | latency |
|---|---|---|---|---|---|
| OpenVLA-OFT (+wrist +proprio) | 97.6 / 98.4 / 97.9 / 94.5 -> **97.1** | 7B | DINOv2+SigLIP, LoRA-FT | continuous L1 chunk K=8 | 112 ms A100 |
| pi0 (OFT repro) | 96.8 / 98.8 / 95.8 / 85.2 -> 94.2 (U) | 3.3B | SigLIP, FT | flow matching, H=50 | 31-73 ms 4090 |
| VLA-Adapter | 97.8 / 99.2 / 97.2 / 95.0 -> 97.3 | 0.5B | Prismatic, frozen ok | L1 chunk | **36.5 ms** |
| **MINERVA** (from scratch) | 94.4 / 99.6 / 96.4 / 89.8 -> **95.05** | **0.54M** | scratch CNN + spatial softmax, random crop | flow, H=16, replan every step + temporal ensembling | **8 ms** |
| Diffusion Policy (scratch) | 78.3 / 92.5 / 68.3 / 50.5 -> 72.4 | ~ResNet-18 | scratch, random crop | DDIM-10, EE delta, K=8 | 100 ms 3080 |
| **this run** | 40.0 / 2.0 / 42.5 / 3.5 -> **22.0** | 22M | frozen DINOv2-S pooled 4x4 | FSQ latent -> joints, K=8 | 38.5 ms (4 steps) |

We are a small from-scratch model: the fair peers are Diffusion Policy and MINERVA, not the 3-7B VLAs. MINERVA
is the existence proof that the target is reachable in this class at a latency below ours — and every one of
its ingredients (trained vision with crop augmentation, continuous executed actions, per-step replanning with
ensembling, 90k+ steps, both cameras + proprio) is something this pipeline lacks.

### Targets (4-suite mean, 50 inits/task, 3 seeds)

| programme | content | projected | GPU-h |
|---|---|---|---|
| A — safe (Stages 0-2) | diagnostics, gripper fix, execution-path tuning, correctness retrain | 0.35-0.50, central **0.42** | ~30 |
| B — this plan (Stages 0-4) | A + trained vision + ensembling + long-horizon fix + confirmation | 0.70-0.85, central **0.78** | ~95 |
| C — SOTA-class (add Stage 5) | B + a continuous executed action head alongside the FSQ stream | 0.85-0.93 *est.* | ~120 |

Programme B's number is a projection by analogy to MINERVA/DP after removing the caps measured in §1; it is
gated stage by stage so spending stops the moment a gate fails. Programme C is the only route to >= 0.9 that the
evidence supports: every published system at >= 0.94 executes a continuous action chunk, and this pipeline's
executed stream passes through a 0.86-bit-per-value FSQ codebook (0.863 deg floor) plus a 3.07 deg denoiser
error. It is a design decision — the hybrid token stays for conditioning, history and steering; only *what is
executed* changes — and it is placed last so it is taken with data, not on faith.

### Latency budget (per replan, V100)

Target **<= 60 ms at 4 DDIM steps** (state it as ~16 Hz replanning; the env is 20 Hz) and **<= 125 ms at 10 steps**.
Reference points: MINERVA 8 ms, VLA-Adapter 36.5, OFT 73-112, DP 100. The 4-step sampler is 38.5 ms but its
*accuracy* has never been measured — Stage 0 measures it open-loop before it is allowed into any configuration.
Nothing here needs compilation or distillation; the projection as implemented (235 ms) is the one term that
breaks the budget and is replaced in Stage 1.

### Non-negotiables

Explicit EE stream and goal tokens stay; `l_goal`, `l_goal_body`, `l_con` are never zeroed; open-loop
`h8/h16_goal_pos_err_cm` gated at every retrain; comparability protocol (§4) for any reported number.

---

## 3. The plan, staged

Gate metrics are the keys written by `04_eval_openloop.py` (`metrics.*`) and `05_eval_closedloop.py`
(`summary.all.std.success`, `summary.per_suite.<suite>.std.success`). Sample sizes follow the analyst's power
table: a +0.10 change over 800 episodes is detectable at ~0.8 power when broad across tasks; +0.05 needs 30
inits/task; object-suite changes of +0.05 need ~27 inits/task because its baseline is 0.02. Per-task outcomes are
sampler-dominated, so re-running the same inits twice is as informative as 40 inits once.

### Stage 0 — diagnostics, no training (~5 GPU-h; ~1.5 done)

Purpose: turn every remaining "why" into a number, and fix the noise floor the later gates are judged against.

Done: replay ceiling (§1.1-1.2), `PROJECT=1` on object (§1.3).

1. **Seed the sampler.** `torch.manual_seed` per (task, init, replan) before `torch.randn` in `ddim_sample`;
   record `q_plan` and `FK(q_plan)` next to `ee_plan` in `run_episode` so the 2.04 cm plan->achieved error splits
   into stream inconsistency vs controller lag (currently inseparable). 0.3 GPU-h.
2. **4-step accuracy.** `SAMPLE_STEPS=4 FORCE=1 04_eval_openloop.py` (copy the 10-step JSON aside first — same
   output path). Gate for using 4 steps anywhere: `h8_goal_pos_err_cm <= 0.6`, `h16 <= 0.7`, `nogoal_ee_err_cm`
   within 0.3 cm of the 10-step value. 77 s.
3. **Execution-path tuning by replay.** Sweep `JP_KP` in {150, 300, 600} and, separately, OSC_POSE executing the
   demo's EE pose (what the demos were collected with), through `replay_demos.py`. Gate: replay ceiling
   >= 0.95 on every suite with the demo gripper command. ~1 GPU-h. Whatever wins becomes the execution path.
4. **Oracle-gripper ablation.** Policy motion, gripper from the demo command (or a close-when-within-2-cm trigger),
   spatial + object, 10 inits. Separates "arm gets there, gripper wrong" from "arm never gets there". ~1 GPU-h.
5. **Teacher-forced-history closed loop.** History tokens from demo k instead of measured proprio
   (`Policy._history_tokens`), vision live; spatial + goal, 10 inits. If success rises toward the 0.85 of easy bowl
   tasks, own-history drift drives the long-horizon term and Stage 4's rollout fix is justified. ~1 GPU-h.
6. **Vision probe.** Ridge regression from the stored 4x4 tokens (and from the full 16x16 grid on 2 tasks) to the
   demo's EE position at frame 0 and to object position where the BDDL init states give it. A probe error >= 2 cm
   from pooled tokens but < 1 cm from the full grid is the direct measurement behind Stage 3. Also the
   render-domain check: HDF5 frame 0 vs live `OffScreenRenderEnv` render (pixel MAD, DINOv2 CLS cosine) — a
   trained vision path assumes these match. ~1 GPU-h.
7. **Tokenizer in deployment mode.** Isolated 4-patch decode RMSE and 128-frame-crop history encoding RMSE
   against the 0.863 deg full-episode figure; FSQ per-level code usage. Trigger a tokenizer retrain only if
   isolated-decode RMSE > 1.5 deg. 0.2 GPU-h.

Gate: numbers recorded; the replay ceiling >= 0.95 (item 3) is required before Stage 1's closed-loop numbers mean
anything. On failure of item 3: the controller is the cap — fix it first (it is model-free and cheap) and rescale
every later target to the measured ceiling.

### Stage 1 — inference-only fixes (~8 GPU-h)

1. **Gripper: predict the command, and latch.** Until the retrain: hysteresis (close < 4.8 cm, open > 5.8 cm) with
   a 4-8 frame latch, and gate the close on `|FK(q_measured) - explicit pos| < 0.5 cm` so lag becomes a delay, not
   a miss. Zero retrain, 0 ms. Expected: recovers a large share of the replay gap on object (0.33 -> 0.84 upper
   bound is model-free; the policy's own share is measured by item 0.4).
2. **Execute the explicit stream cheaply.** Replace the 30-step Adam `project_body` with 3 damped Gauss-Newton
   iterations (Jacobian via `torch.func.jacrev`, `lam=0.05` pull to the decoded q): the start error is ~1.6 cm /
   ~3 deg, well inside the linear regime. *est.* 5-15 ms vs 235. Validate on all five open-loop horizons, goal and
   no-goal (`h*_goal_fkproj_err_cm` must match the Adam values 0.56-0.83). Keep as an A/B against OSC_POSE from
   Stage 0.3 — the replay decides the default, not intuition.
3. **`EXEC=4` with temporal ensembling.** Keep the last K plans in absolute time in `Policy`, average overlapping
   frames with exponential weights (ACT m=0.01). Halves the open-loop stretch and removes the fresh-noise jump
   between plans. Requires a seeded sampler. 0 ms extra per replan, 2x replans per episode.

Gate (800 episodes, 20 inits): `summary.all.std.success >= 0.29` (Wilson lower bound clears the baseline's upper
bound), `per_suite.libero_object >= 0.10`, `fk_consistency_cm <= 0.5` with the new projection, steering metrics
unchanged. On failure: keep 1 (it is free and evidenced independently), drop whichever of 2/3 did not move its
open-loop metric, continue.

### Stage 2 — one retrain with the correctness fixes, the gripper head and a real schedule (~12 GPU-h)

Bundled into one run (they touch the same loop at zero marginal cost); ablate only if the bundle regresses:

- **Gripper command head**: one extra output per frame from stage 1 of the denoiser, BCE on the demo command
  `actions[:, -1]`; executed as `sigmoid > 0.5` with the Stage-1 latch. The width dims stay for the P2 protocol.
- **Loss corrections**: decode the straight-through grid-snapped latent for `l_body`/`l_con`/`l_goal_body`;
  weight velocity dims 0.3 in `l_body`; condition dropout on goals only (or add vision/text CFG at inference —
  pick one, they must match); vision frame = last history frame; drop the 3 cm explicit-only history offset (keep
  the noise); goal/no-goal mix 50/50 to match the std eval.
- **Schedule**: `STEPS=90000` (2.4 h at the measured rate) with EMA 0.9995 and a held-out diffusion loss every
  2k steps; checkpoint every 15k; choose the step by validation, not by budget. Keep d=384 — MINERVA's capacity
  ablation says the short-horizon suites are not capacity-bound.
- **Data**: drop no-op frames and demos that fail replay (OpenVLA calls no-op filtering "crucial"; OFT unfiltered
  vs filtered +2.6). The replay run already labels the failing demos.

Gate: open-loop `fk_consistency_cm <= 1.0`, `joint_rmse_deg <= 2.5`, steering intact; closed-loop 800 episodes
`summary.all.std.success >= 0.34`, object >= 0.20. On failure: bisect (loss corrections first), keep the gripper
head and data filtering regardless.

### Stage 3 — trained vision (~20 GPU-h)

The judge's preferred first variant — a learned resampler over the full 16x16 DINOv2 grid — is impractical here:
the full-grid features are 133 GB in fp16 and computing DINOv2 online during training costs 1.6 s per step. The
practical path is the one MINERVA, Diffusion Policy and robomimic validated:

- Dump raw frames once: `uint8` memmaps, 338,575 x 128 x 128 x 3 per camera = 16.6 GB each (`01_prepare_data.py`
  already streams every frame).
- A from-scratch CNN per camera (depthwise-separable, 5-6 stages, ~1M params) with spatial-softmax keypoints
  (K=32) -> tokens that replace the 34 DINOv2 tokens in `cond_tokens`; trained end-to-end with the denoiser.
- Augmentation: random shift +-8 px (pad and crop) at train time, none at eval; colour jitter 0.3 as in the LIBERO
  baseline config. Both cameras.
- *est.* 2-3 ms per camera on V100, i.e. -6 ms against DINOv2. Retrain at the Stage-2 recipe.

Gate (n=30 on object = 300 episodes, then full 800): `per_suite.libero_object.std.success >= 0.40`,
`summary.all.std.success >= Stage-2 result + 0.10`, steering intact, render-domain check from Stage 0 passed.
On failure with a good probe result: the executed stream is the cap — go to Stage 5 before spending on vision
again. On failure with a bad probe result: the CNN/augmentation choice is wrong; try ResNet-18 first-4-layers
(QueST/DP) before abandoning.

### Stage 4 — long-horizon fix and confirmation (~35 GPU-h)

- **History under own control.** Train `two_stage_goal_rollout` (exists, untrained) or, better, the analyst's
  closer surrogate: decode sampled latents to q, apply the controller's lag filter (omega_n 12.25, zeta 1),
  re-encode with the tokenizer and recompute the explicit stream by FK — that is what closed-loop history actually
  looks like. Only if Stage 0.5 showed history drift matters. ~4 GPU-h.
- **Confirmation** at the comparability protocol (§4): 3 seeds x 40 tasks x 50 inits = 6000 episodes at the
  measured 15.75 s/episode = 26 GPU-h, preceded by an 800-episode screen per candidate.

Gate: this is the reported row. Stop condition: if the 50-init mean is below 0.60, do not fund Stage 5 as an
add-on — it becomes the main line.

### Stage 5 — optional: a continuous executed action head (~25 GPU-h)

Add a third head that predicts the 16x7 joint chunk (or the EE delta chunk) *continuously* (L1, or flow) from the
same stage-2 context, trained with the same FK-consistency and goal-body losses, and execute it instead of the
FSQ decode. The tokenizer keeps encoding history; goal tokens keep steering the explicit stream; the hybrid token
stays the model's interface. What changes is only the executed representation, which removes the 0.863 deg
codebook floor and the 3.07 deg latent error at once. OFT measures continuous vs discrete at +5 pts and MINERVA
reports L1 ~ flow. Gate: `joint_rmse_deg <= 1.5` open-loop, object >= 0.70, 4-suite >= 0.85 at n=50.

---

## 4. Comparison protocol

To produce a row comparable to the tables above: all four suites, **50 init states per task** (LIBERO's official
init files, indices 0-49), **3 training seeds**, success from the benchmark checker, `SAMPLE_STEPS` as validated
in Stage 0, report per-suite mean +- std over seeds and pooled Wilson 95% CIs. Screen candidates at 20 inits
(3.5 GPU-h) before any 50-init run (8.75 GPU-h per seed). Because init k <-> demo k and inits 0-44 have training
demos, also report inits 45-49 separately as a held-out-init row — a reviewer will ask. The published rows in §2
are verified against the papers except pi0's (typed from OFT's reproduction; confirm before use); the
`PUBLISHED` dict in `07_aggregate.py` still carries the old unverified digits and must be replaced with §2's.

## 5. Latency budget, final configuration (V100, per replan)

| term | ms | source |
|---|---|---|
| DDIM sample, 4 steps, cfg 1.0 | 38.5 | `latency.json` (accuracy gated in Stage 0.2) |
| vision, 2 cameras: DINOv2 12.8 -> scratch CNN *est.* 6 | 6 | Stage 3 |
| history tokenize | 2.3 | `latency.json` |
| latent decode | 1.6 | `latency.json` |
| explicit-stream execution: 3-step Gauss-Newton *est.* (OSC_POSE: ~0) | 10 | Stage 1.2 |
| gripper head, ensembling | <1 | |
| **total at 4 steps** | **~58 ms** (17 Hz) | target <= 60 |
| total at 10 steps (cfg 1.0) | ~116 ms | target <= 125 |

With CFG on goals (steered protocols): +32 ms at 4 steps, +78 at 10. The old projection (235 ms) never appears.

## 6. Risks, and what would make us stop

- **The replay ceiling stays below 0.95 after the controller sweep** (Stage 0.3): then the actuation path caps
  every later number; OSC_POSE execution of the explicit stream is the remaining option, and targets scale down.
- **Vision probe shows the pooled tokens already localise to < 2 cm**: Stage 3 is then not the lever; its budget
  moves to Stage 5.
- **Stage 3 passes the probe but not the success gate**: the executed stream is the cap; Stage 5 becomes the main
  line, not an option.
- **Noise**: nothing below 30 inits/task is confirmatory; no family-wise correction is applied across gates, so a
  borderline pass is replicated, never promoted.
- **libero_10** compounds single-step skills; at Stage-2 skill rates it stays < 0.15 regardless. Its number moves
  only after Stages 3-4 lift the constituents.
- **Honest projection**: without Stage 5, 0.70-0.85; ~0.9+ requires Stage 5 and it remains *est.* until Stage 3's
  gate result is in. If Stage 2 fails its gate even after bisection, the FSQ executed stream is the limit and the
  compute goes to Stage 5 directly.

## 7. GPU-hour ledger (two V100s; wall-clock roughly half)

| stage | GPU-h | cumulative |
|---|---|---|
| 0 diagnostics (1.5 already spent) | 5 | 5 |
| 1 inference-only fixes + 800-episode gate | 8 | 13 |
| 2 correctness retrain (90k steps) + gates | 12 | 25 |
| 3 trained vision + gates | 20 | 45 |
| 4 long-horizon + 3-seed confirmation | 35 | 80 |
| contingency (reruns against noise) | 15 | 95 |
| 5 continuous executed head (optional) | 25 | 120 |

Engineering, non-GPU: seeding/logging and the replay sweep 1 day; gripper head + loss corrections 1 day;
raw-frame dump + CNN path 2-3 days; ensembling 0.5 day; Stage 5 head 2 days.

---

*Files referenced:* `ardy_vla.py` (`cond_tokens`, `ddim_sample`, `project_body`, `Policy.plan`,
`Policy._history_tokens`, `LiberoTask.reset/step_to`, `OnlineEncoder`, `make_batch`), `03_train.py`,
`02_tokenizer.py`, `01_prepare_data.py`, `04_eval_openloop.py`, `05_eval_closedloop.py`, `06_latency.py`,
`replay_demos.py`; results `runs/results/*.json`.
