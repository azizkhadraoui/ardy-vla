# Running experiments on Panther

Everything here was learned by running ~40 jobs on this cluster over the ardy-vla project. It is written so
you can start a *separate* experiment alongside the ones already running, without tripping over the same
things twice.

---

## 1. Getting on

```bash
ssh kaziz@10.2.19.87
```

Requires the VPN. When the VPN drops you get `Connection timed out` — jobs keep running regardless, SLURM does
not care whether you are connected. Nothing is lost by disconnecting mid-run.

For scripted access, a helper that pipes a heredoc to a login shell avoids re-authenticating per command:

```bash
# /tmp/px
#!/bin/bash
timeout "${PXT:-120}" ssh -4 -o BatchMode=yes -o ServerAliveInterval=20 -o ServerAliveCountMax=10 \
  kaziz@10.2.19.87 'bash -l -s' < "${1:-/dev/stdin}"
```

Use `bash -l` (login shell) or your conda paths will not be set.

---

## 2. Layout

```
/export/home/kaziz/ardy_vla/
├── code/            the repo; job scripts in code/jobs/; SLURM logs land HERE as slurm-<name>-<id>.out
├── runs/            $WORK_DIR — everything an experiment produces
│   ├── ckpt/        model checkpoints
│   ├── results/     JSON results, one file per run
│   ├── data/        preprocessed tensors (proprio.npz, vision_*.npy, raw_*.npy)
│   ├── tokenizer/   frozen FSQ tokenizer
│   ├── episodes/    recorded rollouts (.npz)
│   └── figures/     rendered videos
├── libero_hdf5/     36 GB of LIBERO demonstrations
├── LIBERO/          upstream LIBERO checkout
├── LIBERO-plus/     the robustness benchmark (drop-in replacement for the above)
└── hf_cache/        HF_HOME — DINOv2, T5 weights
```

`runs/` is 51 GB, the shared filesystem is 83% full with 19 TB free. There is no per-user quota, but it is NFS,
so **many small files are slow** — unzipping 457k files took over an hour.

Put your experiment's outputs under a different `WORK_DIR` if you want them cleanly separated, or reuse
`runs/` and give every result file a distinct `TAG` (see §6).

---

## 3. The environment

```bash
source /export/home/kaziz/ardy_vla/code/env.sh
```

That sets `PY` (the interpreter — always call `$PY`, never `python`), `WORK_DIR`, `DATA_DIR`, `LIBERO_DIR`,
`HF_HOME`, and `MUJOCO_GL=egl`. Copy it and edit rather than modifying it in place if your experiment needs
different paths.

The conda env is `/export/home/kaziz/motion/miniconda3/envs/ardy` (Python 3.11). Already installed: torch,
robosuite 1.4.1, mujoco 3.1.6, bddl, transformers, gym 0.26.2, wandb, ImageMagick + wand, scikit-image, opencv.

**Never run Python on the login node.** It has little memory and the OOM killer takes anything that imports
torch or robosuite — you get a bare `Killed` with no traceback. Use `gpu-dev` or a small `gpu-short` job even
for a one-line import check.

---

## 4. Partitions and limits

| partition | time limit | hardware |
|---|---|---|
| `gpu-all` | unlimited | 25 nodes, mostly V100 16/32 GB, some P100 16 GB |
| `gpu-A100` | 3 days | A100 80 GB, 2 nodes |
| `gpu-H200` | 3 days | H200, 3 nodes |
| `gpu-short` | 2 h | same nodes as `gpu-all` |
| `gpu-dev` | 2 h | 1 node — use for interactive debugging |
| `cpu-all` | unlimited | no GPU |

Your QOS is `gpulimit`: **10 concurrent jobs and 10 GPUs**. Beyond that jobs sit in
`QOSMaxJobsPerUserLimit` — they are queued, not rejected. There are separate `a100_qos` and `h200_qos`, each
capped at 4 GPUs, so the A100/H200 partitions do not consume the same allowance.

The ardy-vla runs use `gpu-all` with V100s. **If your experiment is compute-bound rather than long, try
`gpu-A100` first** — it is usually free and much faster, at the cost of a 3-day ceiling.

**Nodes are shared.** Two of our jobs landing on the same node roughly halved each one's throughput. If timing
matters, add `#SBATCH --exclusive`, but expect to wait longer for the allocation.

---

## 5. A job script that works

```bash
#!/bin/bash -l
#SBATCH -J myexp                 # also names the log: slurm-myexp-<jobid>.out
#SBATCH -o slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem 64000MB
#SBATCH --time 12:00:00
set -e                            # without this a failing step is silently skipped
source "${SLURM_SUBMIT_DIR:-$PWD}/env.sh"

env FOO=1 BAR=2 $PY -u train.py   # -u so the log updates live instead of at exit
```

Submit from `code/` so `$SLURM_SUBMIT_DIR` finds `env.sh`:

```bash
cd /export/home/kaziz/ardy_vla/code
sbatch jobs/myexp.sh
```

Memory: 48 GB is enough for evaluation, 64–80 GB for training that memory-maps the raw frame arrays. Asking
for much more just delays scheduling.

**Time limits are hard.** A job at its limit is killed mid-step with no grace period. One of our runs
(`stack_b`) needed 13.8 h against a 14 h limit and was cut partway through its evaluation — the training
checkpoint survived, so only the evaluation had to be resubmitted. **Save checkpoints periodically** and make
evaluation a separate resumable step, and you can never lose more than one stage.

Arrays for sweeps:

```bash
#SBATCH --array=0-3
S=(a b c d); THIS=${S[$SLURM_ARRAY_TASK_ID]}
```

---

## 6. Conventions worth copying

- **Tag every result file.** Our scripts take `TAG=_myrun` and write
  `results/closedloop_<variant>_s<seed><TAG>.json`. Without it, a second run silently overwrites the first.
- **`FORCE=1` to overwrite**, otherwise scripts skip when the output already exists. That default is what lets
  you resubmit a partially-completed pipeline without redoing finished stages.
- **Write partial results as you go.** Our evaluation dumps JSON every task with `partial: true`, so a job
  killed at its time limit still yields usable data. Beware of reading those partials as if complete — a
  partial run ordered by suite is biased toward whichever suite runs first. Two numbers I reported early from
  partials moved by 0.05 once the runs finished.
- **Seed anything you will compare.** Unseeded fp16 sampling gave 9/20 and 4/20 on the same checkpoint and
  task — larger than most effects we were trying to measure.
- **Log to W&B** (`wandb` is installed and `WANDB_*` is honoured) so you can watch a 12 h job without ssh.

---

## 7. Traps that cost us time

**CRLF line endings.** Editing a `.sh` on Windows and copying it over produces
`bash: $'\r': command not found` or stranger failures. Always:

```bash
scp myjob.sh kaziz@10.2.19.87:.../jobs/ && ssh ... "sed -i 's/\r$//' .../jobs/myjob.sh"
```

**`|| true` hides failures.** A smoke test that swallowed non-zero exits reported success while two of its
steps were broken. Collect return codes explicitly instead.

**Check a train/eval mismatch before believing an ablation.** Our camera ablation zeroed the *feature tensor*
during training but the *image* at evaluation, then encoded that black image — handing the model a constant it
had never seen. It produced a clean-looking 0.000 across 700 episodes that meant nothing. If an ablation gives
a suspiciously round or extreme number, verify that the intervention reaches the model identically in both
paths.

**Verify a benchmark's perturbation actually applies.** LIBERO-plus implements its robot-initial-state
perturbation as a robot class with a different `init_qpos`, but hands you the *original* task's saved
simulator state, which contains the original joint positions. Loading it silently undoes the perturbation. We
had to re-apply the robot's own `init_qpos` afterwards. Assume nothing about a third-party benchmark until you
have measured that its intervention changes the model's input.

**Third-party dependency chains.** LIBERO-plus needs `wand` → the ImageMagick *shared library* (pip does not
ship it; `conda install -c conda-forge imagemagick` plus `MAGICK_HOME` fixes it) → `scikit-image` → more. Write
the install as a job that loops on `ModuleNotFoundError` rather than discovering them one submission at a time.

**Rendering needs `MUJOCO_GL=egl`** and `PYOPENGL_PLATFORM=egl`, both already in `env.sh`. Without them,
offscreen rendering fails on a headless node.

---

## 8. Monitoring

```bash
squeue -u kaziz -o "%.8i %.13j %.3t %.9M %.9L %R"     # running: id, name, state, elapsed, time LEFT, node
sacct -u kaziz -S 2026-09-22 -o JobID%9,JobName%14,State%13,Elapsed,End%17 | grep -v '\.ba\|\.ex'
tail -f slurm-myexp-<jobid>.out
scancel <jobid>
scontrol show job <jobid> | grep StdOut                # where a running job is writing
```

`TIME_LEFT` in `squeue` is the single most useful column — it tells you immediately whether a job will finish
its current stage before the wall clock kills it.

`sinfo -p gpu-all -o "%.12P %.6t %N"` shows what is idle before you choose a partition.

---

## 9. What is currently running

As of 26 Sep, jobs `403578`–`403582` occupy 4–5 of your 10 slots on `gpu-all` for roughly 16 h. That leaves
**5–6 GPU slots free**, plus the A100 and H200 partitions which are on separate allowances and mostly idle.

If you submit to `gpu-A100`, you will not compete with these at all.
