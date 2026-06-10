"""
scripts/train_autoencoder.py

Phase 2 — Train the beatmap autoencoder with windowed PreprocessedDataset.

Changes from previous version:
  - Uses PreprocessedDataset (pre-computed tensors) instead of on-the-fly .osu parsing
  - 20-second windows instead of full 6-min padded tensors
  - Much larger effective batch size (window is 1000 frames vs 18000)
  - Rate + freq mask augmentation
  - Reads from colab_index.jsonl

Usage:
    python scripts/train_autoencoder.py
    python scripts/train_autoencoder.py --resume checkpoints/autoencoder/best.pt
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from taiko.data.preprocessed_dataset import (
    WindowedDataset,
    load_index,
    print_index_stats,
)
from taiko.model.autoencoder import (
    BeatmapAutoencoder,
    AutoencoderConfig,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INDEX_FILE   = "data/processed/colab_index.jsonl"
DATA_ROOT    = Path("data/processed")
CKPT_DIR     = Path("checkpoints/autoencoder")
LOG_DIR      = "runs/autoencoder"

BATCH_SIZE   = 32
LR           = 1e-4
MAX_EPOCHS   = 100
VAL_EVERY    = 500
SAVE_EVERY   = 500
LOG_EVERY    = 20
WARMUP_STEPS = 1000
VAL_BATCHES  = 50
KEEP_LAST_N  = 3
VAL_RATIO    = 0.05

# With 10k maps, 4 windows per map = 40k samples/epoch
SAMPLES_PER_EPOCH = 40_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_lr(step, warmup, max_steps, lr, lr_min=1e-6):
    if step < warmup:
        return lr * max(step, 1) / max(warmup, 1)

    if max_steps is None or step >= max_steps:
        return lr_min

    t = (step - warmup) / (max_steps - warmup)

    return lr_min + (lr - lr_min) * 0.5 * (
        1 + math.cos(math.pi * t)
    )


def save_ckpt(path, model, optimizer, step, epoch, best_val):
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "best_val": best_val,
        },
        path,
    )

    print(f"  Saved: {path}")


def cleanup(ckpt_dir, keep):
    ckpts = sorted(
        ckpt_dir.glob("step_*.pt"),
        key=lambda p: int(p.stem.split("_")[1])
    )

    for old in ckpts[:-keep]:
        old.unlink()


@torch.no_grad()
def validate(model, loader, device, max_batches):
    model.eval()

    total = 0.0
    n = 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break

        x = batch["tensor"].to(device)
        mask = batch["valid_mask"].to(device)

        loss, _ = model.training_loss(x, mask)

        total += loss.item()
        n += 1

    model.train()

    return total / max(n, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(resume_path=None):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    n_gpus = torch.cuda.device_count()

    print(f"Device: {device}")

    if device.type == "cuda":
        for i in range(n_gpus):
            vram = (
                torch.cuda.get_device_properties(i).total_memory
                / 1024**3
            )

            print(
                f"  GPU {i}: "
                f"{torch.cuda.get_device_name(i)} "
                f"({vram:.1f} GB)"
            )

    # -------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------

    train_records, val_records = load_index(
        INDEX_FILE,
        val_ratio=VAL_RATIO,
    )

    print_index_stats(train_records, "Train")
    print_index_stats(val_records, "Val")

    train_ds = WindowedDataset(
        train_records,
        DATA_ROOT,
        augment=True,
        samples_per_epoch=SAMPLES_PER_EPOCH,
    )

    val_ds = WindowedDataset(
        val_records,
        DATA_ROOT,
        augment=False,
        samples_per_epoch=max(1000, len(val_records) * 4),
    )

    effective_batch = BATCH_SIZE * max(n_gpus, 1)

    num_workers = (
        min(4, n_gpus * 2)
        if n_gpus > 0 else 0
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=effective_batch,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=effective_batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    # -------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------

    config = AutoencoderConfig(
        x_channels=7,
        middle_channels=32,
        z_channels=16,
        channel_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        num_groups=8,
        kl_weight=1e-6,
    )

    model = BeatmapAutoencoder(config)

    if n_gpus > 1:
        model = nn.DataParallel(model)

    model = model.to(device)

    m = (
        model.module
        if isinstance(model, nn.DataParallel)
        else model
    )

    params = m.count_parameters()

    print(
        f"Model: "
        f"enc={params['encoder']} "
        f"dec={params['decoder']} "
        f"total={params['total']}"
    )

    print(f"Compression: {m.compression_ratio}x")

    print(
        f"Window: {train_ds.window_frames} frames = "
        f"{train_ds.window_frames * 20 / 1000:.0f}s"
    )

    print(f"Effective batch: {effective_batch}")

    optimizer = torch.optim.AdamW(
        m.parameters(),
        lr=LR,
        weight_decay=1e-4,
    )

    writer = SummaryWriter(LOG_DIR)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    step = 0
    start_epoch = 0
    best_val = float("inf")

    max_steps = MAX_EPOCHS * len(train_loader)

    # -------------------------------------------------------------------
    # Resume
    # -------------------------------------------------------------------

    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(
            resume_path,
            map_location=device,
        )

        m.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

        step = ckpt["step"]
        start_epoch = ckpt["epoch"]
        best_val = ckpt["best_val"]

        print(
            f"Resumed from step {step} "
            f"(val_loss={best_val:.4f})"
        )

    # -------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------

    print("\nAutoencoder training started (windowed, 20s crops)")
    print(f"TensorBoard: tensorboard --logdir {LOG_DIR}\n")

    model.train()

    t0 = time.time()

    for epoch in range(start_epoch, MAX_EPOCHS):

        for batch in train_loader:

            lr = get_lr(
                step,
                WARMUP_STEPS,
                max_steps,
                LR,
            )

            for pg in optimizer.param_groups:
                pg["lr"] = lr

            x = batch["tensor"].to(device)
            mask = batch["valid_mask"].to(device)

            optimizer.zero_grad()

            loss, log_dict = m.training_loss(x, mask)

            loss.backward()

            nn.utils.clip_grad_norm_(
                m.parameters(),
                1.0,
            )

            optimizer.step()

            step += 1

            # -----------------------------------------------------------
            # Logging
            # -----------------------------------------------------------

            if step % LOG_EVERY == 0:

                writer.add_scalar(
                    "train/loss",
                    log_dict["total_loss"],
                    step,
                )

                writer.add_scalar(
                    "train/recon_loss",
                    log_dict["recon_loss"],
                    step,
                )

                writer.add_scalar(
                    "train/kl_loss",
                    log_dict["kl_loss"],
                    step,
                )

                writer.add_scalar(
                    "train/lr",
                    lr,
                    step,
                )

                print(
                    f"epoch {epoch+1:3d} | "
                    f"step {step:6d} | "
                    f"loss {log_dict['total_loss']:.4f} | "
                    f"recon {log_dict['recon_loss']:.4f} | "
                    f"kl {log_dict['kl_loss']:.6f} | "
                    f"lr {lr:.2e} | "
                    f"{time.time()-t0:.0f}s"
                )

            # -----------------------------------------------------------
            # Validation
            # -----------------------------------------------------------

            if step % VAL_EVERY == 0:

                val_loss = validate(
                    model,
                    val_loader,
                    device,
                    VAL_BATCHES,
                )

                writer.add_scalar(
                    "val/loss",
                    val_loss,
                    step,
                )

                print(
                    f"  → val loss: "
                    f"{val_loss:.4f} "
                    f"(best: {best_val:.4f})"
                )

                if val_loss < best_val:
                    best_val = val_loss

                    save_ckpt(
                        CKPT_DIR / "best.pt",
                        model,
                        optimizer,
                        step,
                        epoch,
                        best_val,
                    )

            # -----------------------------------------------------------
            # Periodic save
            # -----------------------------------------------------------

            if step % SAVE_EVERY == 0:

                save_ckpt(
                    CKPT_DIR / f"step_{step:07d}.pt",
                    model,
                    optimizer,
                    step,
                    epoch,
                    best_val,
                )

                cleanup(CKPT_DIR, KEEP_LAST_N)

        print(f"--- End of epoch {epoch+1} ---\n")

    writer.close()

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--resume",
        default=None,
    )

    args = parser.parse_args()

    train(args.resume)