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
BRANCH = "claude/osu-taiko-chart-generation-4tkieb"


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
repeatedly. Section 6 resumes from the previous session's checkpoint.
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

Kaggle mounts attached datasets read-only under `/kaggle/input/`. This locates
the shard folder wherever you put it.
"""),
code("""
SHARDS = None
for candidate in Path("/kaggle/input").rglob("index.json"):
    if (candidate.parent / "mels.dat").exists():
        SHARDS = candidate.parent
        break

if SHARDS is None:
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
reader = ShardReader(SHARDS)
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

probe = WindowedDataset(reader, train_idx, window_frames=1536,
                        random_window=True, augment=False,
                        samples_per_epoch=8, seed=0)

for i in range(4):
    s = probe[i]
    mel, chart = s["mel"].numpy(), s["chart"].numpy()
    # Onset energy in the audio: positive spectral flux, summed over mel bins.
    flux = np.maximum(0, np.diff(mel, axis=1)).sum(0)
    onsets = (chart[:4].sum(0) > 0.5)[1:]
    if onsets.sum() < 5:
        continue
    ratio = flux[onsets].mean() / max(flux[~onsets].mean(), 1e-6)
    print(f"  window {i}: audio energy at notes is {ratio:.2f}x the energy elsewhere "
          f"({int(onsets.sum())} notes)")

print("\\nAbove 1.0 means notes land on audio events. Near 1.0 means they do not,")
print("and something is misaligned -- stop and run tests/test_dataset.py.")
"""),

md("""
## 3. Stage 1 — autoencoder

Roughly 6-10 GPU-hours. Compresses charts into the latent space the diffusion
model works in.

**Gate A: onset F1 must reach 0.98.** Not validation loss -- a loss of 0.01
says nothing about whether notes came back on the right frames. If the gate
will not clear at 16x, drop one entry from `--channel-mult` for 8x and retrain.
A first stage that loses notes caps everything downstream permanently.
"""),
code("""
CKPT = Path("/kaggle/working/checkpoints")
CKPT.mkdir(parents=True, exist_ok=True)

AE_ARGS = [
    "--shards", str(SHARDS),
    "--out", str(CKPT / "autoencoder"),
    "--window-frames", "1536",
    "--batch-size", "16",
    "--epochs", "60",
    "--samples-per-epoch", "20000",
    "--channel-mult", "1", "1", "2", "2", "4",   # 16x compression
    "--num-workers", "2",
    "--val-every", "500",
]

if (CKPT / "autoencoder" / "last.pt").exists():
    AE_ARGS += ["--resume", str(CKPT / "autoencoder" / "last.pt")]
    print("resuming the autoencoder")

!python scripts/train_autoencoder.py {" ".join(AE_ARGS)}
"""),

md("""
## 4. Stage 2 — diffusion

The long one: 150-250 GPU-hours, so about 15-25 sessions. `--max-hours 11`
stops cleanly and saves before Kaggle cuts the session off.

Start with `--profile p1`. Prove the pipeline first with `tiny` if you want a
fast end-to-end run.
"""),
code("""
AE_BEST = CKPT / "autoencoder" / "best.pt"
assert AE_BEST.exists(), "Run stage 1 first."

import torch
gate_a = torch.load(AE_BEST, map_location="cpu", weights_only=False).get("best_f1", 0)
print(f"Autoencoder Gate A: onset F1 {gate_a:.4f}")
if gate_a < 0.98:
    print("  Below 0.98. Whatever the autoencoder loses is a ceiling on the")
    print("  diffusion model, and more diffusion training will not recover it.")
    print("  Consider retraining stage 1 at 8x before continuing.")
"""),

code("""
DIFF_ARGS = [
    "--ae", str(AE_BEST),
    "--shards", str(SHARDS),
    "--out", str(CKPT / "diffusion"),
    "--profile", "p1",
    "--window-frames", "1536",
    "--epochs", "200",
    "--samples-per-epoch", "20000",
    "--num-workers", "2",
    "--val-every", "1000",
    "--save-every", "500",
    "--max-hours", "11",
]

if (CKPT / "diffusion" / "last.pt").exists():
    DIFF_ARGS += ["--resume", str(CKPT / "diffusion" / "last.pt")]
    print("resuming diffusion training")

!python scripts/train_diffusion.py {" ".join(DIFF_ARGS)}
"""),

md("""
## 5. Save the checkpoints

Nothing under `/kaggle/working` survives once the session ends unless you save
it. Do this **before** the session times out, or you lose the run.

Run the cell, then use the notebook's *Save Version* button, or download
`checkpoints.zip` from the Output panel and re-upload it as a Dataset for the
next session.
"""),
code("""
import shutil
archive = shutil.make_archive("/kaggle/working/checkpoints", "zip", str(CKPT))
print(f"{archive}  ({Path(archive).stat().st_size / 1024**2:.0f} MB)")
for f in sorted(CKPT.rglob("*.pt")):
    print(f"  {f.relative_to(CKPT)}  {f.stat().st_size / 1024**2:.0f} MB")
"""),

md("""
## 6. Resuming in a later session

Attach the checkpoint dataset you saved, then run this before sections 3 and 4.
"""),
code("""
# Copy read-only checkpoints from /kaggle/input into the writable working dir.
import shutil

CKPT = Path("/kaggle/working/checkpoints")
CKPT.mkdir(parents=True, exist_ok=True)

restored = 0
for source in Path("/kaggle/input").rglob("*.pt"):
    if "autoencoder" in source.parts or "diffusion" in source.parts:
        stage = "autoencoder" if "autoencoder" in source.parts else "diffusion"
        target = CKPT / stage / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        restored += 1
        print(f"  restored {stage}/{source.name}")

print(f"{restored} checkpoints restored" if restored else
      "Nothing restored -- attach your checkpoint dataset first.")
"""),

md("""
## 7. Gate B — is it listening to the music?

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
## 8. Generate a map

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


def main() -> None:
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
