"""
scripts/train_autoencoder.py

Phase 2 — Train the beatmap autoencoder.

Goal: learn to compress [7, T] beatmap tensors into a latent space
that the diffusion model will later learn to generate.

Usage:
    python scripts/train_autoencoder.py
    python scripts/train_autoencoder.py --resume checkpoints/autoencoder/best.pt
"""

from __future__ import annotations
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from taiko.data.osu_parser import OsuTaikoParser
from taiko.data.tensor_repr import (
    beatmap_to_tensor, FRAME_MS, N_CHANNELS, MAX_FRAMES
)
from taiko.model.autoencoder import BeatmapAutoencoder, AutoencoderConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CACHE_FILE   = "taiko_files_filtered.json"
CKPT_DIR     = Path("checkpoints/autoencoder")
LOG_DIR      = "runs/autoencoder"

BATCH_SIZE   = 8
LR           = 1e-4
MAX_EPOCHS   = 100
VAL_EVERY    = 500       # steps
SAVE_EVERY   = 500
LOG_EVERY    = 20
MAX_STEPS    = None
WARMUP_STEPS = 500
VAL_BATCHES  = 30
KEEP_LAST_N  = 3
VAL_RATIO    = 0.05

# Pad all tensors to this length for batching
# 6 min song @ 20ms = 18000 frames — use slightly more
PAD_FRAMES   = 18_000


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BeatmapTensorDataset(Dataset):
    """
    Loads .osu files on-the-fly, converts to tensor.
    No pre-processing needed — tensors are fast to compute.
    """

    def __init__(self, osu_files: list[Path], pad_frames: int = PAD_FRAMES):
        self.files     = osu_files
        self.pad_frames = pad_frames
        self.parser    = OsuTaikoParser()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        for _ in range(5):  # retry on error
            try:
                import signal, os
                path = self.files[idx]
                bm = self.parser.parse_file(path)
                if bm.note_count == 0:
                    idx = random.randint(0, len(self.files) - 1)
                    continue
                tensor = beatmap_to_tensor(bm, pad_to=None)  # [7, T_raw]

                T = tensor.shape[1]

                # Valid mask: 1 for real frames, 0 for padding
                valid_len  = min(T, self.pad_frames)
                valid_mask = np.zeros(self.pad_frames, dtype=np.float32)
                valid_mask[:valid_len] = 1.0

                # Pad or truncate to pad_frames
                if T < self.pad_frames:
                    pad = np.zeros((N_CHANNELS, self.pad_frames - T), dtype=np.float32)
                    tensor = np.concatenate([tensor, pad], axis=1)
                else:
                    tensor = tensor[:, :self.pad_frames]

                return {
                    "tensor":     torch.from_numpy(tensor),      # [7, pad_frames]
                    "valid_mask": torch.from_numpy(valid_mask),  # [pad_frames]
                    "note_count": bm.note_count,
                    "duration_ms": bm.duration_ms,
                }
            except Exception:
                idx = random.randint(0, len(self.files) - 1)

        # Fallback empty sample
        return {
            "tensor":     torch.zeros(N_CHANNELS, self.pad_frames),
            "valid_mask": torch.zeros(self.pad_frames),
            "note_count": 0,
            "duration_ms": 0,
        }


def load_split(cache_file: str, val_ratio: float = VAL_RATIO):
    files = [Path(p) for p in json.loads(Path(cache_file).read_text())]
    random.seed(42)
    random.shuffle(files)
    n_val = max(1, int(len(files) * val_ratio))
    return files[n_val:], files[:n_val]


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup: int, max_steps: int, lr: float, lr_min: float = 1e-6) -> float:
    if step < warmup:
        return lr * step / max(warmup, 1)
    if max_steps is None or step >= max_steps:
        return lr_min
    t = (step - warmup) / (max_steps - warmup)
    return lr_min + (lr - lr_min) * 0.5 * (1 + math.cos(math.pi * t))


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_ckpt(path: Path, model, optimizer, step, epoch, best_val):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step":      step,
        "epoch":     epoch,
        "best_val":  best_val,
    }, path)
    print(f"  Saved: {path}")


def cleanup(ckpt_dir: Path, keep: int):
    ckpts = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    for old in ckpts[:-keep]:
        old.unlink()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, device, max_batches):
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x    = batch["tensor"].to(device)
        mask = batch["valid_mask"].to(device)
        loss, _ = model.training_loss(x, mask)
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(resume_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- Data ----------------------------------------------------------- #
    if not Path(CACHE_FILE).exists():
        print("ERROR: Run fast_scan.py first.")
        return

    train_files, val_files = load_split(CACHE_FILE)
    print(f"Train: {len(train_files)} | Val: {len(val_files)}")

    train_ds = BeatmapTensorDataset(train_files)
    val_ds   = BeatmapTensorDataset(val_files)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0)

    # ---- Model ---------------------------------------------------------- #
    config = AutoencoderConfig(
        x_channels=7,
        middle_channels=32,
        z_channels=16,
        channel_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        num_groups=8,
        kl_weight=1e-6,
    )
    model = BeatmapAutoencoder(config).to(device)
    params = model.count_parameters()
    print(f"Model: enc={params['encoder']} dec={params['decoder']} total={params['total']}")
    print(f"Compression: {model.compression_ratio}x  "
          f"({PAD_FRAMES} → {PAD_FRAMES // model.compression_ratio} latent frames)")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    writer    = SummaryWriter(LOG_DIR)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    step, start_epoch, best_val = 0, 0, float("inf")
    max_steps = MAX_STEPS or MAX_EPOCHS * len(train_loader)

    # ---- Resume --------------------------------------------------------- #
    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step        = ckpt["step"]
        start_epoch = ckpt["epoch"]
        best_val    = ckpt["best_val"]
        print(f"Resumed from step {step}")

    # ---- Training loop -------------------------------------------------- #
    print(f"\nStarting autoencoder training")
    print(f"TensorBoard: tensorboard --logdir {LOG_DIR}\n")

    model.train()
    t0 = time.time()

    for epoch in range(start_epoch, MAX_EPOCHS):
        for batch in train_loader:
            # LR update
            lr = get_lr(step, WARMUP_STEPS, max_steps, LR)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            x    = batch["tensor"].to(device)
            mask = batch["valid_mask"].to(device)

            optimizer.zero_grad()
            loss, log_dict = model.training_loss(x, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1

            # Log
            if step % LOG_EVERY == 0:
                writer.add_scalar("train/loss",       log_dict["total_loss"], step)
                writer.add_scalar("train/recon_loss", log_dict["recon_loss"], step)
                writer.add_scalar("train/kl_loss",    log_dict["kl_loss"],    step)
                writer.add_scalar("train/lr",         lr,                     step)
                print(
                    f"epoch {epoch+1:3d} | step {step:6d} | "
                    f"loss {log_dict['total_loss']:.4f} | "
                    f"recon {log_dict['recon_loss']:.4f} | "
                    f"kl {log_dict['kl_loss']:.6f} | "
                    f"lr {lr:.2e} | {time.time()-t0:.0f}s"
                )

            # Validate
            if step % VAL_EVERY == 0:
                val_loss = validate(model, val_loader, device, VAL_BATCHES)
                writer.add_scalar("val/loss", val_loss, step)
                print(f"  → val loss: {val_loss:.4f}  (best: {best_val:.4f})")
                if val_loss < best_val:
                    best_val = val_loss
                    save_ckpt(CKPT_DIR / "best.pt", model, optimizer, step, epoch, best_val)

            # Save
            if step % SAVE_EVERY == 0:
                save_ckpt(CKPT_DIR / f"step_{step:07d}.pt", model, optimizer, step, epoch, best_val)
                cleanup(CKPT_DIR, KEEP_LAST_N)

            if MAX_STEPS and step >= MAX_STEPS:
                print("Max steps reached.")
                writer.close()
                return

        print(f"--- End of epoch {epoch+1} ---\n")

    writer.close()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    train(args.resume)
