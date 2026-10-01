# ardy-vla — where the project stands

Status as of **1 October 2026**. Every number here is from a completed run of `05_eval_closedloop.py` on the four
standard LIBERO suites using the benchmark's own success checker, 20 initial states per task unless stated
(800 episodes; standard error ≈ 0.017). Partial runs are excluded on purpose — partial means moved by 0.05–0.20
against us three separate times in this project.

Result files are `runs/results/closedloop_two_stage_goal_s<seed>_proj<tag>.json`; `docs/CLUSTER.md` covers how to run
them.

---

## 1. Headline

| | spatial | object | goal | long | mean |
|---|---|---|---|---|---|
| **best configuration** (adapted encoder, 32-frame chunk) | 0.935 | 0.485 | **0.770** | 0.505 | **0.674** |
| same, with contact-triggered adaptive stride | **0.955** | 0.495 | 0.745 | 0.500 | **0.674** |
| 50-init publication protocol (older config) | 0.792 | 0.562 | 0.630 | 0.468 | 0.613 |
| starting point, as inherited | 0.400 | 0.020 | 0.425 | 0.035 | 0.220 |
| OpenVLA-OFT (7B) | 0.976 | 0.984 | 0.979 | 0.945 | 0.971 |
| MINERVA (0.54M, from scratch) | 0.944 | 0.996 | 0.964 | 0.898 | 0.951 |
| Diffusion Policy (from scratch) | 0.788 | 0.925 | 0.686 | 0.508 | 0.727 |

0.220 → 0.674 over the project. **Spatial is at the frontier** (0.955, and 0.970 in one variant, against MINERVA's
0.944). The remaining deficit is concentrated: against OFT, spatial −0.021, goal −0.234, long −0.445,
**object −0.499**. Object and long are 79% of the gap.

The 0.674 figures are 20-init screens. The one 50-init run we have landed 0.011 below its screen, so treat 20-init
numbers as roughly unbiased but unvalidated for this configuration.

### Robustness (LIBERO-plus, 7 perturbation dimensions)

| configuration | clean | LIBERO-plus | retention |
|---|---|---|---|
| wrist-camera-only | 0.599 | **0.377** | 63% |
| baseline (frozen features) | 0.542 | 0.257 | 44% |
| stack_a (best on clean at the time) | 0.624 | **0.199** | 31% |
| OpenVLA-OFT, for reference | 0.971 | 0.696 | 72% |
| OpenVLA-7B, for reference | ~0.77 | 0.156 | 20% |

**The clean ranking inverts.** The configuration that led on standard LIBERO is the *worst* under perturbation.
Within one architecture, one dataset and one pipeline, with levers added one at a time — a causal demonstration of
something the robustness literature argues correlationally. Background texture scores **0.00** for every model we
have; camera viewpoint 0.00–0.03 for anything using the third-person camera.

---

## 2. What worked

| intervention | effect | notes |
|---|---|---|
| Inference-side execution fixes | **+0.238** | 0.220 → 0.458 with the checkpoint *unchanged*: 5 Hz replanning, ACT-style temporal ensembling, Gauss–Newton consistency projection, gripper hysteresis |
| Executing the whole predicted chunk | **+0.278** | same weights, stride 4 → 32: 0.357 → 0.635 |
| Adapted vision encoder | +0.039 | last 2 DINOv2 blocks tuned on a localisation objective, features re-extracted, policy trained at full batch — ~1 GPU-hour instead of 17 end-to-end |
| Contact-triggered adaptive stride | +0.025 | over fixed stride, at **⅓ the replanning cost** (12.9 vs ~37 replans/episode) |
| Gauss–Newton projection replacing 30-step Adam | 235 ms → 10 ms | *and* more accurate: h16 adherence 0.830 → 0.503 cm |
| Frequency-weighted DCT auxiliary loss | +0.066 | on its own baseline only; conflicts with the fuller recipe (0.485) |
| Recurrent visual memory + long-suite oversampling | +0.072 | best clean result at the time (0.624) — **and the worst robustness** (0.199) |

---

## 3. What didn't, with numbers

**Negative results that are properly controlled and worth reporting:**

- **Action-conditioned world-model memory.** A latent forward model pretrained on 338k demo transitions
  demonstrably learns consequences — prediction error **53% worse under action shuffling**, rollout R² 0.67 at
  1.6 s — yet supplying its recurrent state to the policy *hurts*: **0.585** from random init, **0.459** from the
  pretrained model, against 0.614 for the memory it replaced. The random-init control separates architecture from
  pretraining.
- **End-effector-space control is worse than joint-position here.** Replaying demonstrations: joint-position
  0.565, OSC with the demos' own recorded actions 0.490, EE pose tracking 0.380. Counterintuitive, and it killed a
  planned rework.
- **Executing the explicit stream instead of the FSQ latent: 0.539** against 0.624. The Gauss–Newton projection
  was already absorbing the latent's decode error, so a tokenizer retrain would have been wasted.
- **Longer chunks are worse**: `CHUNK=16` (64 frames) 0.580, against 0.635 for 32 frames.
- **The levers don't compose with chunking**: 0.557–0.618 across free strides, all below plain `CHUNK=8`'s 0.659.
- **Adapted features and adaptive stride are redundant, not additive**: each worth ~0.04 alone, 0.674 together —
  exactly the encoder number. Both fix the same grasp failures by different routes.
- **More encoder adaptation is worse**: 4 blocks localises better (0.97 cm vs 1.36 cm) and scores lower (0.657 vs
  0.674).
- **Data cleaning** (dropping demos the actuation path cannot reproduce, 696 of 2000): 0.435.
- **Second observation frame**: 0.500. **Both encoders together**: 0.575. **Trained gripper head**: 0.451 at
  50 inits against 0.613 for hysteresis on identical initial states.
- **Auxiliary grasp-point head: 0.229**, cause not identified. Training converged normally (final loss 0.104,
  comparable to the 0.674 run) and the recorded evaluation knobs are identical, yet the policy collapsed. Needs a
  `W_GRASP` sweep before the idea is dismissed; 1.0 may simply be far too strong on the shared stage-1 features.

---

## 4. Measured ceilings and diagnostics

- **The action representation is the quantified weak point.** The FSQ tokenizer reconstructs held-out joints at
  **0.903° with full-episode context but 6.199° decoded as the four isolated patches the policy actually emits** —
  roughly 6 cm at the end effector, on every executed joint.
- **The vision bottleneck is the readout, not the encoder.** Grasp-point localisation: 5.15 cm from a ridge probe
  on frozen 4×4 features, **1.44 cm from the same frozen features with a 2-layer MLP head**, 1.36 cm with 2 blocks
  unfrozen, 0.97 cm with 4. The information is already there. And closing it from 5.15 to 1.36 cm moved the object
  suite by **−0.005**.
- **Replaying a demonstration is not an upper bound on a closed-loop policy.** Replay succeeds 0.565 overall while
  our policy scores 0.674; task 30 is unreplayable yet the policy solves it 3/3. An earlier conclusion built on
  replay as a "ceiling" was wrong and is retracted.
- **Demonstrations are not reproducible under current simulator versions.** Replaying each demo's own recorded
  actions through the controller they were teleoperated with succeeds only ~0.49–0.57, with end-effector drift of
  just 1.4–1.5 cm — the trajectory tracks and the task still fails, at contact.
- **Long-suite tasks 34 and 35 are genuine capability failures, not harness bugs.** Verified: task resolution
  matches, 50 demos each, FK base fit exact to 0.000 cm, and t34's demo replay satisfies the checker. On t35 the
  policy **never closes the gripper at all**; failing episodes travel ~2× further than successful ones.
- **The sampler is nearly deterministic.** Across-sample positional spread never exceeds 2 cm over a 1.6 s window,
  which is why uncertainty-based chunk triggering cannot work here (see §5) and suggests the diffusion machinery
  may be unnecessary.
- **Latency**: ~65 ms per replan on a V100 at 4 DDIM steps (38.5 ms sampling, 6.4 ms/image vision, 10 ms
  projection), one third of a 200 ms budget — and now at 12.9 replans per episode rather than ~37.

---

## 5. Adaptive chunking: contact beats uncertainty

Adaptive inference-time chunking is a crowded subfield — [AAC](https://arxiv.org/abs/2604.04161) (CVPR 2026),
[BID](https://arxiv.org/abs/2408.17355) (ICLR 2025), [RTC](https://arxiv.org/html/2506.07339),
[Continue-or-Replan](https://arxiv.org/pdf/2608.03483), GeoAAC, and others. We are not first, and AAC independently
reports the same phase behaviour (large chunks for transport, small for manipulation).

What we add is a **mechanism and a contradiction**. A grasp needs a *sustained* close command, because the fingers
take several frames to shut. Our first trigger cut the stride at the gripper transition "to replan with contact in
view" and scored **0.000**: the close was commanded, two frames executed, the plan reproposed, and the arm hovered
for 22 replans at 0.074 m finger width instead of 0.003. Inverting it — run long *through* contact, be reactive in
free space — gives 0.659.

Reimplementing AAC's entropy trigger on the same checkpoints:

| trigger | mean stride | mean success | chunk samples/episode |
|---|---|---|---|
| entropy, 0.25 cm tolerance | 5.5 | 0.163 | 512 |
| entropy, 0.5 cm | 14.3 | 0.497 | 168 |
| entropy, 1.0 cm | 27.7 | 0.619 | 168 |
| entropy, 2.0 cm | 32.0 (never binds) | 0.635 | 168 |
| **contact trigger (ours)** | 16.3 | **0.659** | **16** |

The entropy threshold improves monotonically as it is loosened and peaks **exactly where it stops doing anything**.
At matched mean stride — 14.3 vs 16.3 — the scores are 0.497 and 0.659. So it is not stride *length* but stride
*placement*: uncertainty peaks at contact, which is precisely where a sustained command is required, so an
uncertainty signal shortens at the worst possible moment. Our trigger also needs no extra sampling at all.

This predicts the scope condition: AAC's rule should be fine for policies without a latched, per-frame gripper
command.

---

## 6. Methodological lessons (each one cost real time)

- **An intervention must reach the model identically in training and evaluation.** Camera masking zeroed the
  *feature tensor* in training and the *image* at evaluation, then encoded the black image — handing the model a
  constant it had never seen, and producing a clean-looking **0.000 across 700 episodes**.
- **A policy trained on adapted features must be evaluated with the adapted encoder.** Training read
  `vision_*_ft.npy`; the evaluator re-encoded with the original pretrained weights. Scored **0.079** and looked
  like "adapted features hurt". The evaluator now refuses the mismatch.
- **Verify a third-party benchmark's perturbation actually applies.** LIBERO-plus implements robot-initial-state
  perturbation as a robot class differing only in `init_qpos`, then hands you the *original* task's saved
  simulator state — which silently overwrites it. `plus_probe.py` asserts all seven dimensions reach the policy
  (robot 0.197 rad, camera/light/background MAD 47–67, layout 11.6, noise 7.9).
- **Exact zeros invariant to configuration are usually bugs** — three times here — **but not always**: t34/t35
  survived the same scrutiny and are real.
- **Never trust a partial result's mean.** Suite-ordered evaluation biases it toward whichever suite runs first.
- **Seed anything you will compare.** Unseeded fp16 sampling gave 9/20 and 4/20 on the same checkpoint and task.
- **Guard new code paths on a 2-minute dev-node job**, not a 20-hour submission. Four submissions were burned on
  trivial errors before this became habit.

---

## 7. Open questions

1. **Why did the grasp-point head collapse to 0.229?** Normal training loss, identical evaluation knobs. A
   `W_GRASP` sweep (0.01, 0.1) decides whether explicit spatial supervision is viable — the literature's answer for
   high-precision tasks ([PRISM](https://www.emergentmind.com/topics/libero-object), AimBot).
2. **Does one-pass L1 regression match the diffusion head?** [MINERVA](https://arxiv.org/abs/2609.03715) reports
   flow matching buys nothing at 3.8× the speed, and our sampler's near-determinism agrees. Training is done;
   evaluation in flight.
3. **Object is stuck at 0.485–0.585** after six interventions. If L1 and the grasp head both fail to move it, the
   limitation is the action representation or capacity, and further vision work is wasted.
4. **Statistical floor for publication**: everything is single-seed, and only two configurations have 50-init
   numbers.

---

## 8. Reading the code

| file | what it is |
|---|---|
| `ardy_vla.py` | the library: tokenizer, denoiser, FK/projection, policy, LIBERO wrappers. Behaviour is env-gated — `EXEC_ADAPT`, `ENT_ADAPT`, `VIS_SUFFIX`, `W_GRASP`, `L1_HEAD`, `CHUNK`, `CAM_MASK`, `MEM_K`, `MEM_WM` |
| `03_train.py` / `04_eval_openloop.py` / `05_eval_closedloop.py` | train, open-loop adherence, closed-loop success |
| `09_eval_plus.py`, `plus_probe.py` | LIBERO-plus generalisation, and the probe that verifies its perturbations bite |
| `world_model.py` | the latent forward model (§3, negative) |
| `vis_ft.py` | encoder adaptation + feature re-extraction (§2) |
| `replay_demos.py` | actuation-path diagnostics, joint-position and OSC |
| `t34_diag.py` | the long-suite zero investigation |
| `docs/CLUSTER.md` | how to run any of this on Panther |
