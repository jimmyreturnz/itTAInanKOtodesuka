"""
notebooks/build_notebook.py

Generates kaggle_train.ipynb. Editing a notebook's JSON by hand is miserable
and produces unreviewable diffs, so the notebook is built from this file and
this file is what gets edited.

    python notebooks/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = "https://github.com/jimmyreturnz/itTAInanKOtodesuka.git"
BRANCH = "main"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


CELLS = [
md(f"""
# tAIkoMapper — Kaggle training

Trains the osu!taiko chart generator on 2x T4.

**Before running this**, on your own PC:

1. `python scripts/pack_dataset.py --scan "D:/osu!/Songs"`
2. Upload `data/processed/shards/` as a Kaggle Dataset named **taiko-shards**
3. Attach it to this notebook (Add Data, right-hand panel)

**Settings** (right-hand panel):

- Accelerator: **GPU T4 x2**
- Internet: **On** (needed to clone the repo)
- Persistence: **Files only**

Sessions cap at about 12 hours and this needs far more, so run this notebook
repeatedly. Section 4 restores the previous session's checkpoints and every stage resumes from them.
"""),

md("## 1. Setup"),
code(f"""
import os, sys, subprocess, time
from pathlib import Path

REPO_DIR = Path("/kaggle/working/taiko")

if not REPO_DIR.exists():
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", "{BRANCH}", "{REPO}", str(REPO_DIR)],
        check=True,
    )
else:
    subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=False)

os.chdir(REPO_DIR)
sys.path.insert(0, str(REPO_DIR))
print("repo:", REPO_DIR)
subprocess.run(["git", "-C", str(REPO_DIR), "log", "--oneline", "-1"])
"""),

code("""
import torch
print("torch", torch.__version__, "| CUDA", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i}  {p.name}  {p.total_memory / 1024**3:.1f} GB")

if torch.cuda.device_count() < 2:
    print("\\nWARNING: expected 2 GPUs. Set Accelerator to 'GPU T4 x2'.")
"""),

md("""
## 2. Find the dataset

Kaggle mounts attached datasets read-only under `/kaggle/input/`, usually as a
symlink into `/kaggle/input/datasets/<owner>/<slug>/`. Before Python 3.13
`Path.rglob` refuses to descend into symlinked directories, so this walks with
`followlinks=True` -- otherwise the dataset is right there and invisible.
"""),
code("""
import os

SHARDS = None
for root, _dirs, files in os.walk("/kaggle/input", followlinks=True):
    if "index.json" in files and "mels.dat" in files:
        SHARDS = Path(root)
        break

if SHARDS is None:
    for root, _dirs, files in os.walk("/kaggle/input", followlinks=True):
        if files:
            print("  mounted:", root, sorted(files)[:6])
    raise SystemExit(
        "No packed dataset found under /kaggle/input.\\n"
        "  Attach the dataset you uploaded (Add Data in the right-hand panel).\\n"
        "  It must contain mels.dat, charts.npz and index.json."
    )

print("shards:", SHARDS)
for f in sorted(SHARDS.iterdir()):
    print(f"  {f.name:<14s} {f.stat().st_size / 1024**2:>9.1f} MB")
"""),

code("""
from taiko.data.shards import ShardReader
from taiko.data.preprocessed_dataset import split_indices, print_split_stats
from taiko.data.frames import describe

print(describe())
# "read", not the default "auto". mels.dat is 6.68 GB and the auto threshold is
# a quarter of RAM -- about 8 GB here -- so auto maps the file, and this reader
# stays alive in the kernel for the whole session alongside the training
# subprocesses. It is a wildcard worth removing on the machine whose memory is
# the thing that keeps ending these runs.
reader = ShardReader(SHARDS, mel_io="read")
train_idx, val_idx = split_indices(reader, val_ratio=0.05)
print_split_stats(reader, train_idx, "Train")
print_split_stats(reader, val_idx, "Val")
"""),

md("""
### Sanity check: is the audio aligned with the charts?

This is the one check worth doing before spending any GPU time. If a chart
window and its mel window describe different parts of the song, the loss still
falls and the model still trains -- it just never learns to follow the music.
"""),
code("""
import numpy as np
from taiko.data.preprocessed_dataset import WindowedDataset

# Comparing note frames against non-note frames does not work here: the mel is
# log-scaled and referenced to the clip's own peak, so a blip in a quiet gap
# shows a larger relative rise than a real hit inside a dense stream. Sweeping a
# lag does work -- whatever the absolute numbers, the flux must peak on the
# charted frame. A peak parked at a nonzero lag is a genuine misalignment.
N = 40
probe = WindowedDataset(reader, train_idx, window_frames=1536,
                        random_window=True, augment=False,
                        samples_per_epoch=N, seed=0)

LAGS = range(-5, 6)
acc = {lag: [] for lag in LAGS}
for i in range(N):
    s = probe[i]
    flux = np.maximum(0.0, np.diff(s["mel"].numpy(), axis=1)).sum(0)
    frames = np.flatnonzero(s["chart"].numpy()[:4].sum(0) > 0.5)
    if len(frames) < 20:
        continue
    for lag in LAGS:
        idx = frames + lag
        idx = idx[(idx >= 0) & (idx < len(flux))]
        if len(idx):
            acc[lag].append(flux[idx].mean())

means = {lag: float(np.mean(v)) for lag, v in acc.items() if v}
peak = max(means, key=means.get)
for lag, v in sorted(means.items()):
    print(f"  lag {lag:+d} ({lag * 20:+4d} ms)  onset flux {v:7.3f}"
          + ("   <-- peak" if lag == peak else ""))

assert abs(peak) <= 1, (
    f"onset energy peaks {peak} frames ({peak * 20} ms) away from the charted "
    "notes -- the audio and the charts describe different milliseconds. Stop "
    "and run tests/test_dataset.py.")
print(f"\\nPeak at lag {peak:+d}: audio and charts agree to within one 20 ms frame.")
"""),

md("""
## 3. Run configuration

Every later cell reads these, so they live here rather than inside the stage 1
cell -- otherwise skipping stage 1 takes `CKPT` and `PROFILE` down with it.
"""),
code("""
# D4 called for a full rehearsal on "tiny" before committing to p1, because
# nothing had run on real data. Most of what it was meant to shake out is now
# proven -- mel cache, memmap on Kaggle disk, DataParallel, alignment, Gate A --
# and p1 measures 23.7 samples/s against tiny's 29, so the rehearsal no longer
# saves much. p1 keeps its weights; tiny throws them away. Set "tiny" here if
# you would still rather test the resume loop on weights you can afford to lose.
PROFILE = "p1"                         # "tiny" = rehearsal, "p1" = the real run

CKPT = Path("/kaggle/working/checkpoints")
CKPT.mkdir(parents=True, exist_ok=True)
AE_BEST = CKPT / "autoencoder" / "best.pt"

# Kaggle's cap, and half an hour held back so the zip cell can still run. Every
# stage below is given what is left of this at the moment it starts, not a
# fixed number of hours: the setup cells above cost time, a restarted stage
# costs more, and a stage that believes it has 10.5 hours when the session has
# 4 left is stopped mid-step by Kaggle rather than stopping itself and saving.
SESSION_HOURS = 12.0
SESSION_RESERVE_HOURS = 0.5
SESSION_START = time.time()


def session_hours_left():
    spent = (time.time() - SESSION_START) / 3600
    return max(0.0, SESSION_HOURS - SESSION_RESERVE_HOURS - spent)


print(f"profile {PROFILE}  checkpoints {CKPT}")
print(f"autoencoder {'present' if AE_BEST.exists() else 'not yet trained'}")
print(f"{session_hours_left():.2f} h of session budget left")
"""),

md("""
## 4. Resume from an earlier session

Kaggle deletes `/kaggle/working` when a session ends, so a later session starts
with nothing. Attach the checkpoint dataset you saved and run this: it copies
the read-only files into the writable working directory, where stage 1 sees an
autoencoder to skip and stage 2 sees a `last.pt` to resume from.

Files that are not checkpoints are **not copied**. An attached dataset holding
a damaged checkpoint would otherwise restore it every session, so leaving the
dataset attached for the sake of one good stage keeps re-importing the broken
one. Screening happens on the first four bytes, so it costs nothing even for a
500 MB file, and the stage whose checkpoint was rejected simply starts fresh.

Nothing restored is the correct output on a first run.
"""),
code("""
# Copy read-only checkpoints from /kaggle/input into the writable working dir,
# skipping anything that is not actually a checkpoint.
import os, shutil
from taiko.train.session import describe_file

CKPT = Path("/kaggle/working/checkpoints")
CKPT.mkdir(parents=True, exist_ok=True)

restored, rejected = 0, 0
newest = {}

# Same symlink trap as the dataset search: os.walk, not rglob.
for root, _dirs, files in os.walk("/kaggle/input", followlinks=True):
    for name in files:
        source = Path(root) / name
        if not name.endswith(".pt"):
            continue
        # Substring on the whole path, not exact path parts: a dataset may
        # arrive as autoencoder/best.pt, as taiko-autoencoder/best.pt, or
        # flattened. Matching parts only worked for the first, and failed
        # silently by restoring nothing.
        where = str(source).lower()
        if "autoencoder" not in where and "diffusion" not in where:
            continue
        stage = "autoencoder" if "autoencoder" in where else "diffusion"

        # A torch checkpoint is a zip; the header is enough to reject an
        # archive, an HTML page or a truncated file without reading 500 MB.
        description, container = describe_file(source)
        if not description.startswith(("zip archive", "legacy pickle")):
            rejected += 1
            print(f"  SKIPPED {stage}/{name}: {description}")
            if container:
                print(f"          a {container} container. To recover it, copy it in"
                      f" by hand and run:")
                print(f"          python scripts/rescue_checkpoint.py {CKPT / stage} --write")
            print(f"          not copied, so {stage} will start fresh rather than"
                  f" stop on it.")
            continue

        # Several datasets may carry the same filename; keep the newest.
        key = (stage, name)
        stamp = source.stat().st_mtime
        if key in newest and newest[key][0] >= stamp:
            continue
        newest[key] = (stamp, source)

for (stage, name), (_stamp, source) in sorted(newest.items()):
    target = CKPT / stage / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    restored += 1
    print(f"  restored {stage}/{name}  ({target.stat().st_size / 1024**2:.0f} MB)")

if restored or rejected:
    print(f"\\n{restored} restored, {rejected} skipped")
else:
    print("\\nNothing restored from /kaggle/input.")
    print("  Correct on a first run. On any later run it means the previous")
    print("  session's checkpoints were saved but never attached: stage 1 will")
    print("  retrain the autoencoder and stage 2 will start at step 0. See")
    print("  section 8 -- saving a version is not the same as attaching the")
    print("  dataset, and only the attachment puts files in /kaggle/input.")
"""),

md("""
Now check they survived the round trip, before any GPU time goes into them.

A checkpoint can arrive wrapped in an archive, truncated by an unfinished
transfer, or replaced by an error page, and none of that is visible until
`--resume` tries to open it -- which, without this cell, happens after the
session has already built the model and is ready to train.
"""),
code("""
import subprocess

verify = subprocess.run(
    ["python", "scripts/verify_checkpoints.py", str(CKPT)],
    capture_output=True, text=True,
)
print(verify.stdout)
if verify.stderr.strip():
    print(verify.stderr)

# A stage is only resumable if its own checkpoints verified. Anything that did
# not is renamed aside, so the training cells below start that stage cleanly
# instead of stopping on a file they cannot read.
RESUMABLE = {"autoencoder": False, "diffusion": False}
for stage in RESUMABLE:
    last = CKPT / stage / "last.pt"
    if not last.exists():
        continue
    probe = subprocess.run(
        ["python", "-c",
         "import torch,sys; torch.load(sys.argv[1], map_location='cpu', "
         "weights_only=False)", str(last)],
        capture_output=True, text=True,
    )
    if probe.returncode == 0:
        RESUMABLE[stage] = True
    else:
        broken = last.with_suffix(".pt.unreadable")
        last.rename(broken)
        print(f"{stage}: last.pt will not load; moved to {broken.name}")
        print(f"  try: python scripts/rescue_checkpoint.py {CKPT / stage}")
        print(f"  otherwise this stage restarts from zero, which is fine --")
        print(f"  the other stage is a separate file and is unaffected.")

print(f"\\nresumable: {RESUMABLE}")
"""),

md("""
## 5. A supervisor for the training stages

A training run on Kaggle does not fail by raising; it fails by being killed.
The previous version of this notebook launched stage 2 with `!python ...`,
which discards the exit code -- so when the trainer died 46 minutes into an
11-hour session, the notebook printed nothing wrong, zipped the checkpoints,
ran the evaluation cell and finished green. The remaining ten hours of GPU were
paid for and idle, and the only sign was a log that stopped mid-epoch.

The checkpoint at the moment of death was fine. Two minutes of training had
been lost, not a session. What was missing was anything to pick it back up.

So both stages now go through `supervise`, which:

- gives the stage whatever is left of the session, not a fixed `--max-hours`,
  so a restart does not extend the run past the cap;
- passes `--resume` whenever there is a `last.pt` to resume from, so an attempt
  continues rather than restarting;
- restarts on an out-of-memory death (`--min-free-gb` stopping cleanly, or a
  SIGKILL from the kernel), each time with fewer dataloader workers and without
  pinned memory, because those are the two things that cost host RAM and are
  safe to give up;
- **stops** on any other non-zero exit, because a real error repeated twenty
  times is just a slower way to waste the session;
- raises at the end if the stage never completed, so a committed run shows red
  instead of green.
"""),
code("""
import subprocess, sys, time
from taiko.train import EXIT_LOW_MEMORY

# Killed by SIGKILL: -9 from subprocess, 137 through a shell. This is what the
# OOM killer looks like, and it is the case --min-free-gb exists to pre-empt.
OOM_CODES = {EXIT_LOW_MEMORY, -9, 137}


def _set_arg(args, flag, value):
    args = list(args)
    if flag in args:
        args[args.index(flag) + 1] = str(value)
    else:
        args += [flag, str(value)]
    return args


def _get_arg(args, flag, default):
    return args[args.index(flag) + 1] if flag in args else default


def supervise(script, args, out_dir, budget_hours, label):
    \"\"\"Run a training stage to completion, restarting it when the host runs out.\"\"\"
    out_dir = Path(out_dir)
    deadline = time.time() + budget_hours * 3600
    workers = int(_get_arg(args, "--num-workers", "4"))
    pin = True
    attempt = 0
    futile = 0

    while True:
        remaining = (deadline - time.time()) / 3600
        if remaining <= 0.05:
            print(f"\\n{label}: out of session time.")
            return False

        attempt += 1
        argv = _set_arg(args, "--max-hours", f"{remaining:.3f}")
        argv = _set_arg(argv, "--num-workers", workers)
        if not pin:
            argv = argv + ["--no-pin-memory"]
        last = out_dir / "last.pt"
        if last.exists() and "--resume" not in argv:
            argv = argv + ["--resume", str(last)]

        print(f"\\n{'=' * 70}")
        print(f"{label}: attempt {attempt}, {remaining:.2f} h left, "
              f"{workers} workers, pin_memory={pin}")
        print(f"{'=' * 70}", flush=True)

        started = time.time()
        result = subprocess.run([sys.executable, script, *argv])
        ran = time.time() - started

        if result.returncode == 0:
            print(f"\\n{label}: finished cleanly after {attempt} attempt(s).")
            return True

        if result.returncode not in OOM_CODES:
            raise RuntimeError(
                f"{label} exited {result.returncode} after {ran / 60:.1f} min. "
                f"That is not an out-of-memory death, so restarting it would "
                f"just spend the session repeating it. Read the traceback above."
            )

        # An attempt that dies almost immediately is not making progress, and
        # the loop must not spend eleven hours discovering that.
        futile = futile + 1 if ran < 90 else 0
        if futile >= 3:
            raise RuntimeError(
                f"{label} ran out of memory three times in under 90 seconds "
                f"each. It cannot start on this machine at these settings -- "
                f"lower --batch-size or --window-frames rather than retrying.")

        print(f"\\n{label}: out of host memory after {ran / 60:.1f} min "
              f"(exit {result.returncode}). The checkpoint is current; "
              f"restarting from it with less to hold.", flush=True)
        if workers > 0:
            workers = workers // 2
        else:
            pin = False
        if workers == 0 and not pin:
            print(f"{label}: already at the smallest loader. One more attempt.")
"""),

md("""
## 6. Stage 1 — autoencoder

Roughly 6-10 GPU-hours. Compresses charts into the latent space the diffusion
model works in.

**Gate A: onset F1 must reach 0.98.** Not validation loss -- a loss of 0.01
says nothing about whether notes came back on the right frames. If the gate
will not clear at 16x, drop one entry from `--channel-mult` for 8x and retrain.
A first stage that loses notes caps everything downstream permanently.
"""),
code("""
import subprocess

AE_ARGS = [
    "--shards", str(SHARDS),
    "--out", str(CKPT / "autoencoder"),
    "--window-frames", "1536",
    "--batch-size", "16",
    # 8 epochs (5,500 steps, ~9 min on one T4) cleared Gate A at onset F1
    # 0.9998, flat across every threshold from 0.70 to 0.99. 60 was a guess.
    "--epochs", "8",
    "--samples-per-epoch", "20000",
    "--channel-mult", "1", "1", "2", "2", "4",   # 16x compression
    "--num-workers", "2",
    "--val-every", "500",
    # A memmap's touched pages are resident pages, so sampling random windows
    # walks RSS up by the whole size of mels.dat. "read" preads each window and
    # holds nothing.
    "--mel-io", "read",
    # The clock, not the step count, is what bounds an OOM kill's blast radius.
    "--save-every-min", "10",
    "--min-free-gb", "3",
]
# --max-hours is set per attempt by the supervisor; see section 5.

if RESUMABLE["autoencoder"]:
    AE_ARGS += ["--resume", str(CKPT / "autoencoder" / "last.pt")]
    print("resuming the autoencoder")

# Restored from a checkpoint dataset, or trained earlier in this session: there
# is nothing to do. Gate A is a property of the file, not of this run.
if AE_BEST.exists():
    print(f"{AE_BEST} already exists -- skipping stage 1.")
    print("Delete it, or the whole checkpoints/autoencoder folder, to retrain.")
else:
    # Stage 1 is short, but it is short only if it finishes. Run it under the
    # same supervisor, so an out-of-memory death costs a restart rather than
    # the session -- and so a real error stops the notebook here instead of
    # letting stage 2 fail on a missing autoencoder eleven hours later.
    if not supervise("scripts/train_autoencoder.py", AE_ARGS,
                     CKPT / "autoencoder", session_hours_left(), "stage 1"):
        raise RuntimeError(
            "Stage 1 did not finish this session. Zip the checkpoints, save a "
            "version, attach the output as a dataset, and run again -- it will "
            "resume.")
"""),

md("""
## 7. Stage 2 — diffusion

The long one: 150-250 GPU-hours, so about 15-25 sessions. The supervisor above
gives it whatever is left of `SESSION_HOURS` and puts it back on its feet if
the host runs out of memory.

Start with `--profile p1`. Prove the pipeline first with `tiny` if you want a
fast end-to-end run.
"""),
code("""
assert AE_BEST.exists(), "No autoencoder. Run stage 1, or restore one (section 4)."

import torch
gate_a = torch.load(AE_BEST, map_location="cpu", weights_only=False).get("best_f1", 0)
print(f"Autoencoder Gate A: onset F1 {gate_a:.4f}")
if gate_a < 0.98:
    print("  Below 0.98. Whatever the autoencoder loses is a ceiling on the")
    print("  diffusion model, and more diffusion training will not recover it.")
    print("  Consider retraining stage 1 at 8x before continuing.")
"""),

code("""
# D5: effective batch was 4 (2/GPU x 2 GPUs) -- the smallest in any working
# diffusion recipe. 32/GPU x 2 GPUs reaches an effective 64 without touching
# the window, so the 30 s structural horizon long songs need is kept.
#
# If 32/GPU OOMs, halve it and set GRAD_ACCUM to "2" -- effective batch is what
# has to stay at 64, not the split.
#
# This applies to both profiles. Measured on 2x T4 at this window, p1
# runs 1.8 samples/s at the profile's own 2/GPU and 23.7 at 32/GPU, for 1.3 GiB
# of 15 and step time barely moving (2.22 s -> 2.7 s while the batch grew 16x).
# Small batches leave these models launch-overhead bound, so per_gpu_batch is a
# floor that fits anywhere rather than a recommendation, and checkpointing
# trades compute for memory that is sitting unused.
BATCH_ARGS = ["--batch-size", "32", "--no-grad-checkpoint"]
GRAD_ACCUM = "1"                               # 32/GPU x 2 GPUs = effective 64

# What is left of the session, measured rather than guessed. SESSION_HOURS is
# Kaggle's cap; the cells above have already spent part of it, and a hard-coded
# "10.5" silently becomes 12.5 when the setup cells take two hours, at which
# point Kaggle stops the session mid-step instead of the trainer stopping
# itself and saving.

DIFF_ARGS = [
    "--ae", str(AE_BEST),
    "--shards", str(SHARDS),
    "--out", str(CKPT / "diffusion"),
    "--profile", PROFILE,
    "--window-frames", "1536",
    *BATCH_ARGS,
    "--grad-accum", GRAD_ACCUM,
    "--ranked-only",                           # D2; a no-op on ranked-only shards
    "--epochs", "200",
    "--samples-per-epoch", "20000",
    # Two, not four, on a four-core box. The loader was measured at ~1,200
    # windows/s on four workers against a demand of 25/s, so two is still an
    # order of magnitude more than the GPUs can consume -- and every worker is
    # a fork of a process holding a CUDA context, which is not free.
    "--num-workers", "2",
    "--val-every", "1000",
    # Host RAM, not GPU, ended the two-hour run. See --mel-io in stage 1.
    "--mel-io", "read",
    # The run that died grew 23.4 MB every single step, dead linear from step
    # zero, which is almost exactly one batch of mel (64 x 1536 x 128 x 2 B =
    # 24.0 MB). It is not the mel memmap -- that would saturate at 6.7 GB and
    # visibly flatten, and it did not flatten across 425 steps -- and the same
    # loop on CPU is flat over 2,018 steps. Page-locked batches are the largest
    # thing that only exists on the CUDA path, so they go first. The H2D cost
    # is 25 MB per 2.6 s step, which is nothing here.
    #
    # This is a hypothesis, and the log now tests it either way: the `lock`
    # and trend fields say what is growing within a hundred steps. Drop this
    # line to put pinning back and compare.
    "--no-pin-memory",
    "--val-workers", "0",                      # a loader used once per 1000 steps
    "--prefetch-factor", "2",
    # 250 steps and 10 minutes, whichever comes first. At 2.6 s/step, saving
    # only every 500 steps makes a kill cost 22 minutes -- which is exactly what
    # the run that died at step 550 lost.
    "--save-every", "250",
    "--save-every-min", "10",
    # Stop and save while there is still room to write 541 MB, instead of being
    # SIGKILLed with the session's remaining ten hours unspent.
    "--min-free-gb", "3",
]
# --max-hours is deliberately absent: the supervisor sets it per attempt from
# what is left of the session, so a restart cannot extend the run past the cap.

if RESUMABLE["diffusion"]:
    DIFF_ARGS += ["--resume", str(CKPT / "diffusion" / "last.pt")]
    print("resuming diffusion training")

ok = supervise("scripts/train_diffusion.py", DIFF_ARGS,
               CKPT / "diffusion", session_hours_left(), "stage 2")
if not ok:
    raise RuntimeError(
        "Stage 2 did not finish this session. The checkpoint is current -- run "
        "the next cell to zip it, save a version, and attach the output as a "
        "dataset for the next session.")
"""),

md("""
## 8. Save the checkpoints — and make the next session able to find them

Nothing under `/kaggle/working` survives once the session ends unless you save
it, and **saving it is not the same as attaching it**. This is the step that
decides whether the next session continues the run or starts from step zero,
and it is the one that has quietly cost whole sessions:

1. Run the cell below. It writes `checkpoints.zip` into the session output.
2. *Save Version* (or download the zip from the Output panel).
3. Create or update a Kaggle **Dataset** from that output.
4. **Attach that dataset to this notebook** (Add Data, right-hand panel).

Step 4 is the one that gets skipped. Section 4 restores from `/kaggle/input`,
and a saved output that has not been attached is not in `/kaggle/input` -- so
section 4 prints "Nothing restored", stage 1 retrains the autoencoder from
scratch, stage 2 starts at step 0, and the previous session's eleven hours buy
nothing. If section 4 said "Nothing restored" and this is not your first run,
stop and fix the attachment before spending the GPU.
"""),
code("""
import shutil
archive = shutil.make_archive("/kaggle/working/checkpoints", "zip", str(CKPT))
print(f"{archive}  ({Path(archive).stat().st_size / 1024**2:.0f} MB)")
for f in sorted(CKPT.rglob("*.pt")):
    print(f"  {f.relative_to(CKPT)}  {f.stat().st_size / 1024**2:.0f} MB")

# What the next session will actually be able to pick up, said in the terms
# that matter -- a step number, not a file size.
import torch
last = CKPT / "diffusion" / "last.pt"
if last.exists():
    state = torch.load(last, map_location="cpu", weights_only=False)
    print(f"\\ndiffusion last.pt is at step {state['step']}, "
          f"epoch {state['epoch'] + 1}. Attaching this as a dataset is what "
          f"makes the next session start there instead of at zero.")
else:
    print("\\nNo diffusion checkpoint yet.")
"""),

md("""
## 9. Gate B — is it listening to the music?

Onset F1 above 0.40 against held-out audio. This is what separates a model
following the song from one emitting plausible taiko rhythms; nothing else in
the repo can tell those apart. A model that fails here is not fixed by more
steps.
"""),
code("""
!python scripts/evaluate.py \\
    --diffusion {CKPT / "diffusion" / "best.pt"} \\
    --ae {AE_BEST} \\
    --shards {SHARDS} \\
    --n-maps 30 --steps 30
"""),

md("""
## 10. Generate a map

Supply `--bpm` and `--offset` when you know them. Tempo is an input to the
model now, and getting the grid right is most of getting the chart right.
"""),
code("""
AUDIO = "/kaggle/input/your-song/song.mp3"   # <- change this

!python scripts/generate.py \\
    --audio "{AUDIO}" \\
    --diffusion {CKPT / "diffusion" / "best.pt"} \\
    --ae {AE_BEST} \\
    --difficulty 5.5 \\
    --preset standard \\
    --steps 50 \\
    --out /kaggle/working/outputs
"""),
]


def checkable(source: str) -> str:
    """
    Cell source with IPython's shell escapes blanked out, so it can be compiled.

    `!python ...` is not Python, but everything around it is, and skipping a
    whole cell because it contains one shell line would skip most of the cells
    worth checking.
    """
    lines = source.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        if stripped[:1] in ("!", "%"):
            out.append(" " * (len(line) - len(stripped)) + "pass")
            while line.rstrip().endswith("\\") and i + 1 < len(lines):
                i += 1
                line = lines[i]
                out.append("")
            i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def check(cells: list[dict]) -> None:
    """
    Refuse to write a notebook holding a cell Python cannot parse.

    Cell bodies live inside ordinary triple-quoted strings in this file, so a
    `\\n` intended for the notebook is expanded when *this* module is imported
    and lands in the notebook as a real newline -- splitting the string literal
    it was written inside. It has to be `\\\\n` here to arrive as `\\n` there.

    That is not hypothetical. It shipped, and it broke exactly two cells: the
    one that restores checkpoints from /kaggle/input and the one that decides
    whether each stage can resume. Both die at parse time, before a single
    statement runs, so nothing was ever restored and no stage ever resumed --
    while the diff read perfectly and the failure looked like a mysterious
    "starts from the beginning again". A generated notebook that is never
    parsed until it is on Kaggle is a notebook whose syntax errors cost GPU
    hours, so the build parses it here.
    """
    broken = []
    for i, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        try:
            compile(checkable(source), f"cell{i}", "exec")
        except SyntaxError as exc:
            broken.append(f"  cell {i}, line {exc.lineno}: {exc.msg}\n"
                          f"    {(exc.text or '').strip()}")
    if broken:
        raise SystemExit(
            "refusing to write a notebook with unparseable cells:\n"
            + "\n".join(broken)
            + "\n\n  A backslash-n meant for the notebook needs doubling here."
        )


def main() -> None:
    check(CELLS)
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = Path(__file__).parent / "kaggle_train.ipynb"
    out.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"wrote {out}  ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
