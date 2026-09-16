# ARDY-style hybrid autoregressive-diffusion VLA — full experiment pack

Everything needed to go from raw LIBERO demos to the tables, intervals and videos of the paper, packaged to run on
the cluster in one sitting. Compiled 14 September 2026 from the Kaggle proof-of-concept runs (7 Sept: goal tokens hit
constraints at 0.6–0.8 cm at every horizon to 3.2 s, one-stage 2× worse, inpainting teleports; body drift under
self-generated history was the open problem). This pack is designed to answer the five reviewer objections that
run could not: one suite, one seed, open loop only, teacher-forced perception, no baselines beyond inpainting.

**Two V100 16 GB, ~40 GPU-hours total, ~20 h wall-clock if both GPUs are used throughout.** The decision gate
(`gate` mode) costs ~5 GPU-hours and tells you whether the other 35 are worth spending.

---

## Install

```bash
scp -r ardy_vla_experiments kaziz@panther-login:~/ardy_vla/
cd ~/ardy_vla/ardy_vla_experiments
cp env.sh.example env.sh          # edit WORK_DIR, DATA_DIR, LIBERO_DIR, conda activation

# environment (once). robosuite 1.4 needs MuJoCo <= 3.2 (MjData.qM is gone in 3.3); there are py3.10-3.12 wheels for 3.1.6.
conda create -n ardy python=3.11 -y && conda activate ardy
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.40" huggingface_hub sentencepiece h5py "imageio[ffmpeg]" matplotlib pillow scipy
pip install mujoco==3.1.6 robosuite==1.4.1 bddl easydict cloudpickle
pip install wandb                 # optional, only used when WANDB=1 -- see "Weights & Biases" below
git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO $LIBERO_DIR   # 01 also does this if absent

bash submit_all.sh check          # refuses with a clear message if a file or a module is missing
```

Headless rendering: the compute nodes need EGL (`MUJOCO_GL=egl`, set in env.sh). If `check` passes but 05 fails at
`OffScreenRenderEnv`, `export MUJOCO_GL=osmesa` is the slow fallback.

Optional: put a Hugging Face token in `HF_TOKEN` in env.sh — the unauthenticated download of the four suites (25 GB,
40 files) hits the hub's rate limit; the script retries with the server's back-off but a token makes it one pass.

---

## What is here

| File | Stage | What it settles | GPU | Time |
|---|---|---|---|---|
| `ardy_vla.py` | — | Shared library: data, tokenizer, denoiser, samplers (CFG / inpainting / guidance), FK, projection, open-loop metrics, closed-loop LIBERO runner, and the optional W&B helpers. Every script imports it so training and both evaluators share one definition of everything. | — | — |
| `01_prepare_data.py` + `01_run.sh` | 1 | Four suites, 40 tasks, 2 000 demos as one dataset. Explicit EE stream **defined** as FK(joints) in the robot's own base frame (see *The EE frame* below); frozen DINOv2/T5 features; gripper command threshold. | yes | ~1.5 h |
| `02_tokenizer.py` + `02_run.sh` | 2 | FSQ motion tokenizer on all suites, trained on full episodes (the crop-only bug is what made the first tokenizer read 7.8°). | yes | ~0.5 h |
| `03_train.py` + `03_run.sh` | 3 | The matrix: 6 variants × 3 seeds as a SLURM array, d=384 / 6 layers / 30k steps. `scale` mode adds the d=512 point. | yes | 18 × ~40 min |
| `04_eval_openloop.py` + `04_run.sh` | 4 | **T3.** Adherence vs horizon (0.8–3.2 s) on explicit stream, decoded body, projected body, against the no-goal reference. Per checkpoint, per suite. | yes | 18 × ~8 min |
| `05_eval_closedloop.py` + `05_run.sh` | 5 | **T1, T2.** Closed-loop LIBERO with a joint-position controller and online DINOv2: standard success on all suites + the four steering protocols (timed pose, lifted waypoint with a virtual box, gripper keyframe, disturbance recovery). Records episodes for 08. | yes | 18 × ~2.6 h |
| `06_latency.py` + `06_run.sh` | 6 | **T5.** ms per window for every variant at 4/10 steps, with/without CFG and projection; online vision and tokenizer cost. | yes | minutes |
| `07_aggregate.py` | 7 | All tables (mean ± std over seeds, bootstrap CIs over episodes, **paired** differences on identical episodes) and figures. | no | seconds |
| `08_visualize.py` | 8 | Side-by-side multi-camera videos of recorded closed-loop episodes with projected EE paths, constraints and boxes drawn; trajectory figures; a goal-frame gallery. | no* | minutes |
| `09_lookups.sh` | — | State of the run: matrix coverage, tokenizer verdict, GPU hours from `sacct`, tails of recent job logs. | no | seconds |
| `submit_all.sh` | — | Submission in eight modes, including the full dependency chain and the decision gate. | — | — |

\* 08 builds each task env once to read camera matrices, so it needs the simulator installed, not a GPU.

---

## The EE frame, and why the FK check is per task

LIBERO places the robot base at a **scene-dependent world position**. Fitting one base+tool SE(3) so that
`base o FK(joints) o tool` reproduces the dataset's world-frame `ee_pos` gives 0.03 cm on `libero_object`
alone and 0.03 cm on `libero_goal` alone -- and **35-40 cm pooled**, because those suites sit in different
scenes. `libero_10` fails the same way *on its own* (24 cm): it is the one suite whose ten tasks span
KITCHEN, LIVING_ROOM and STUDY. The fitted base translations differ by about 0.9 m in z between suites.

The rotation target is not the cause: refitting with identity `R`, or with the rotation term dropped
entirely, moves the pooled residual by less than a centimetre (39.8 -> 39.1 -> 38.7 cm).

So 01 fits **one base per task and one tool shared by all of them** (`fit_fk_grouped`) and then defines the
stream the model sees *without* the base:

```
explicit stream := FK(joints) o tool        # the EE in the robot's own base frame
```

The base offset is a property of the scene, not of the arm. It carries nothing a policy can use, and in world
frame it makes the same reach look different in every scene. Goals, adherence, the P1 box and the closed-loop
history all live in this one frame, so everything stays self-consistent. The per-task base transforms are
stored in `proprio.npz` (`fk_base_r6`, `fk_base_t`) and used in exactly one place: 05 puts recorded
trajectories back into world coordinates so 08 can draw them into camera images.

The data-integrity gate is kept and tightened -- the **max over tasks** of the per-task residual must be
under 1 cm, where it used to be a per-suite mean. A task that will not fit means the joint/EE pairing is
wrong for it and nothing downstream is trustworthy. Raising the threshold is never the fix.

---

## The matrix, and what each row is for

| variant | steering at test time | what it isolates |
|---|---|---|
| `two_stage_goal` | masked goal tokens (+ CFG 2.0) | **the method** |
| `one_stage_goal` | goal tokens | is the explicit/latent two-stage split doing the work? (depth doubled to match parameters) |
| `two_stage_inpaint` | x0 inpainting (DiSCo-style) | the standard diffusion-steering baseline; acts only once the goal frame is inside the window |
| `two_stage_guidance` | gradient guidance on goal error | the classifier-guidance baseline; same limitation |
| `two_stage_nohist` | goal tokens, H = 0 | does variable-length history matter? (stands in for a memory benchmark) |
| `two_stage_goal_rollout` | goal tokens | training on the model's own rollouts as history — the DAgger-style fix for body drift under self-generated history |

Every variant is trained with the FK consistency loss, the body-level goal loss, 10 % condition dropout and history
perturbation, so the only thing that differs between rows is the thing the row is named after.

---

## How to run

```bash
bash submit_all.sh check        # first
bash submit_all.sh gate         # 01 -> 02 -> two_stage_goal seed 0 -> its 04 and 05.  ~5 GPU-hours.
```

**Read the gate in order** (all under `$WORK_DIR`):

1. `tokenizer/tokenizer_eval.json` — held-out full-episode RMSE: mean < 1°, no joint > 1.5°. If not, the joint
   stream is wrong and every body metric below inherits it. Fix here before anything else.
2. `results/openloop_two_stage_goal_s0.json` — `inwin_goal_pos_err_cm` and `h8_goal_pos_err_cm` should be ≲ 1 cm
   (Kaggle: 0.78 / 0.62), `h8_goal_fk_err_cm` clearly below `h8_nogoal_fk_err_cm`. If the explicit numbers hold but
   the body gap is unchanged, the `rollout` row is the one to watch in the full run.
3. `results/closedloop_two_stage_goal_s0.json` → `summary.all.std.success`. **This is the one number the Kaggle work
   never produced.** Below ~0.5 means the controller/perception loop needs debugging (see "If the closed loop fails")
   before spending the remaining 35 GPU-hours. Between 0.5 and 0.8 is expected for a 22 M-parameter policy trained
   from scratch on 2 000 demos; the steering results are the claim, not the raw success rate.

Then:

```bash
bash submit_all.sh train        # 03 array 0-17 (17 remaining checkpoints)
bash submit_all.sh scale        # optional d=512 point, one seed
bash submit_all.sh eval         # 04, 05 array, 06 — all depend on nothing; submit once train has landed
bash 09_lookups.sh > lookups.txt   # on the login node, any time
$PY 07_aggregate.py             # when 04/05 JSONs exist; re-run as more land
MUJOCO_GL=egl $PY 08_visualize.py
```

`chain` submits the whole thing with `afterok` dependencies if you would rather not gate. Every script is idempotent
(skips when its output exists; `FORCE=1` to redo), so a killed array can simply be resubmitted.

**Predictions to record before running**, as the plan asks: (a) explicit-stream adherence stays flat at ≲ 1 cm to
3.2 s for `two_stage_goal` and is ~2× worse for `one_stage_goal`; (b) inpainting and guidance show a positive
approach jump (teleport) and no body improvement over no-goal; (c) `two_stage_goal_rollout` is the only row whose
h8 body error drops clearly below the no-goal reference without projection; (d) on P1, `two_stage_goal` clears the
box in the large majority of episodes and inpainting does not; (e) P3b − P3a > 0 with a CI excluding zero.
If (c) fails, the paper reports the projection as the mechanism that couples the body, and says so.

---

## What comes out

`results/tables.md`, regenerated by 07:

- **T1** closed-loop success per suite, mean ± std over seeds, 95 % bootstrap CI over episodes, next to the
  published rows in `PUBLISHED` (verify those numbers against the papers before use — they are placeholders typed
  from memory).
- **T2** steering protocols: success, constraint error at its frame, box collision, path smoothness, per condition;
  then **paired** differences of `two_stage_goal` against every other row on identical (task, init, protocol, seed)
  episodes. An interval that excludes zero is the sentence "X steers better than Y"; one that includes zero is not.
- **T3** adherence vs horizon, four streams (explicit / body / projected body / no-goal), plus the smoothness row.
- **T4** what each component buys, as h8 deltas.
- **T5** latency.

`figures/`: `adherence_vs_horizon.png`, `success_std.png`, `steering_protocols.png`, `training_curves.png`, and from
08 one `.mp4` and one `_trajectories.png` per recorded episode plus `steering_gallery.png`.

Send me `results/*.json`, `tables.md`, `lookups.txt` and the figures directory and I can write the results section
and the honest limitations paragraph in one pass.

---

## Weights & Biases (optional)

Off by default: with `WANDB=0` nothing is imported and every logging call in the pack is a no-op, so the JSON files
above stay the source of truth. To turn it on, once on the login node:

```bash
pip install wandb && wandb login
# in env.sh
export WANDB=1
export WANDB_PROJECT=ardy-vla
export WANDB_ENTITY=your-team        # optional
```

**One run per checkpoint.** Stages 3, 4 and 5 attach to the same run id `{variant}_s{seed}` (`_quick` / `_scale`
suffixed off the `long` preset), grouped by variant. So one row of the runs table carries a variant's training curve,
its open-loop adherence *and* its closed-loop success, and the 18 rows of the matrix sort and filter by
`config.variant` / `config.seed` without any joining by hand. Stages 2, 6, 7 and 8 get their own runs.

| Stage | Logged live | In the run summary |
|---|---|---|
| 02 tokenizer | `tok/loss`, `tok/rec`, `tok/vel`, `tok/val_joint_rmse_deg`, `tok/lr` every 500 steps | per-joint held-out RMSE + the mean and max, as summary keys and a table |
| 03 train | every 250 steps: `train/loss` and each term (`hyb`, `goal`, `goal_body`, `body`, `consist`), `train/lr`, `train/grad_scale`, `train/steps_per_s`; rollout-pool refreshes for the rollout variant | final losses, params, minutes |
| 04 open loop | — | every `openloop/*` metric, the per-suite breakdown, and the adherence-vs-horizon curve as a table |
| 05 closed loop | per task as the 2.6 h job runs: `closedloop/task/{proto}_success` and the running mean over the tasks finished so far — you see the success rate forming instead of waiting for the JSON | per-protocol and per-suite summaries, episode count, hours |
| 06 latency | — | the full row table, plus `sample_ms` per (variant, steps, cfg) |
| 07 aggregate | — | `summary.json` flattened, `tables.md` + `summary.json` attached, every figure as an image |
| 08 visualize | the side-by-side `.mp4` per episode and its trajectory figure | — |

Metrics use per-stage x-axes (`train/step`, `tok/step`, `closedloop/step`), so re-running a stage with `FORCE=1`
overlays a second curve rather than having its steps dropped as non-monotonic.

**Compute nodes without outbound network:** `export WANDB_MODE=offline` in env.sh, then from the login node
`wandb sync $WORK_DIR/wandb/offline-run-*`. Offline runs of stages 3/4/5 are separate directories that sync into the
one run id. `submit_all.sh check` reports whether wandb imports and which mode it will use.

Nothing about this is load-bearing: a wandb failure — no package, no network, an expired key, a broken run — is
caught, printed once, and the stage runs on and writes its JSON exactly as before.

---

## Budget knobs

| knob | default | effect |
|---|---|---|
| `N_INIT` (05) | 20 | init states per task for standard success. 50 = the published protocol, 3 h/checkpoint. |
| `N_INIT_PROTO` (05) | 10 | init states per task per steering protocol on `PROTO_SUITES`. |
| `PROTO_SUITES` (05) | spatial, libero_10 | where the steering protocols run. |
| `PROJECT=1` (05) | off | re-run a checkpoint with the consistency projection; lands in a separate `_proj.json` row. |
| `SEEDS` | 0,1,2 | trims the matrix in 06/07 only; the arrays are fixed at 0-17. |
| `PRESET` | long | `quick` (8k, d=256) reproduces the Kaggle scale in 10 min/variant for debugging. |
| `GOAL_CFG`, `GUIDE_SCALE` | 2.0, 0.3 | test-time strengths; sweep on the gate checkpoint if the baseline looks mis-tuned. |

If the queue only gives you one GPU, `chain` completes in ~40 h; nothing needs both.

---

## If the closed loop fails

The closed-loop runner (`ardy_vla.LiberoTask`, `Policy`, `run_episode`) is the one component that has never been
executed — the Kaggle work stopped at kinematic playback. Where it can go wrong, in likelihood order:

1. **`OffScreenRenderEnv(..., controller="JOINT_POSITION")`** — if LIBERO's wrapper rejects the kwarg, build the
   robosuite controller config yourself: `load_controller_config(default_controller="JOINT_POSITION")` and pass
   `controller_configs=` through `**kwargs`.
2. **Tracking lag.** The policy emits joint targets 20 Hz; `step_to` sends the clipped delta with `output_max=0.15`
   rad/step and `kp=150` (`JP_KP` env). If the arm lags the plan, raise `JP_KP`; if it oscillates, lower it. The
   recorded episodes show this immediately (measured path vs plan).
3. **Gripper.** Commands are ±1 from the predicted finger width against `gripper_threshold_m` (fitted in 01 from the
   demos' own commands). If grasps fail systematically, print the predicted width around grasp time.
4. **Perception orientation.** `FLIP_180` is applied identically to HDF5 frames (01) and live `obs` frames (Policy).
   `figures/preview_agentview_frame0.png` must show the table at the bottom.
5. **Init states.** `set_init_state` needs the sim state vector to match the recording; on a version mismatch it
   falls back to `env.reset()` placement and says so in the log. Success is still valid; the demo-derived goals are
   then slightly off the scene, which affects every variant equally.

`PRESET=quick` with `N_INIT=3 PROTOCOLS=std CL_SUITES=libero_spatial` is a 10-minute smoke test of the whole
closed-loop path on a single checkpoint.

---

## Not included, and why

**DROID / real-robot data.** The pack is sim-only by design; a DROID subset would need its own 01 and only produces
open-loop metrics. Worth doing for a camera-ready, not for the decision.

**A memory benchmark (RoboMME / RMBench).** Different simulators, different observation stacks; the `nohist` row is
the cheap stand-in for the history claim.

**Real-time human steering.** The protocols use scripted constraints so every variant gets identical inputs. A
mouse-driven demo is a video for the website, not a table.
