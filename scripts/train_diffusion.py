"""
scripts/train_diffusion.py

Phase 3 — Train the diffusion model with DDP (DistributedDataParallel).
Autoencoder is frozen. Trains audio encoder + U-Net.

Single GPU:
    python scripts/train_diffusion.py
    python scripts/train_diffusion.py --resume checkpoints/diffusion/best.pt

Multi GPU (Kaggle T4x2):
    torchrun --nproc_per_node=2 scripts/train_diffusion.py
    torchrun --nproc_per_node=2 scripts/train_diffusion.py --resume checkpoints/diffusion/best.pt
"""

from __future__ import annotations
import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from taiko.data.preprocessed_dataset import PreprocessedDataset, load_index, print_index_stats
from taiko.model.diffusion import TaikoDiffusion


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AUTOENCODER_CKPT = "checkpoints/autoencoder/best.pt"
INDEX_FILE       = "data/processed/colab_index.jsonl"
DATA_ROOT        = Path("data/processed")
CKPT_DIR         = Path("checkpoints/diffusion")
LOG_DIR          = "runs/diffusion"

BATCH_SIZE       = 4       # per GPU
LR               = 1e-4
MAX_EPOCHS       = 100
VAL_EVERY        = 500
SAVE_EVERY       = 500
LOG_EVERY        = 20
WARMUP_STEPS     = 1000
VAL_BATCHES      = 20
KEEP_LAST_N      = 3
VAL_RATIO        = 0.05


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp():
    """Initialize DDP if launched with torchrun, else single GPU."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
        return rank, world_size
    return 0, 1


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank): return rank == 0


def unwrap(model):
    if isinstance(model, DDP):
        return model.module
    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_lr(step, warmup, max_steps, lr, lr_min=1e-6):
    if step < warmup:
        return lr * step / max(warmup, 1)
    if max_steps is None or step >= max_steps:
        return lr_min
    t = (step - warmup) / (max_steps - warmup)
    return lr_min + (lr - lr_min) * 0.5 * (1 + math.cos(math.pi * t))


def save_ckpt(path, model, optimizer, step, epoch, best_val):
    path.parent.mkdir(parents=True, exist_ok=True)
    m = unwrap(model)
    torch.save({
        "unet":      m.unet_model.state_dict(),
        "wave":      m.wave_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step":      step,
        "epoch":     epoch,
        "best_val":  best_val,
    }, path)
    print(f"  Saved: {path}")


def cleanup_old(ckpt_dir, keep):
    ckpts = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    for old in ckpts[:-keep]:
        old.unlink()


@torch.no_grad()
def validate(model, loader, device, max_batches):
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        loss, _ = unwrap(model).training_loss(
            batch["tensor"].to(device),
            batch["mel"].to(device),
            batch["difficulty"].to(device),
            batch["style"].to(device),
            batch["valid_mask"].to(device),
        )
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(resume_path=None):
    rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if is_main(rank):
        print(f"World size : {world_size}")
        for i in range(torch.cuda.device_count()):
            vram = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  ({vram:.1f} GB)")

    if not Path(AUTOENCODER_CKPT).exists():
        print(f"ERROR: Autoencoder checkpoint not found: {AUTOENCODER_CKPT}")
        cleanup_ddp()
        return

    # ---- Data ----------------------------------------------------------- #
    train_records, val_records = load_index(INDEX_FILE, val_ratio=VAL_RATIO)
    if is_main(rank):
        print_index_stats(train_records, "Train")
        print_index_stats(val_records,   "Val")

    train_ds = PreprocessedDataset(train_records, DATA_ROOT, augment=True)
    val_ds   = PreprocessedDataset(val_records,   DATA_ROOT, augment=False)

    # DistributedSampler splits data across GPUs
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                       rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size,
                                       rank=rank, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_ds,
        batch_size  = BATCH_SIZE,
        sampler     = train_sampler,
        shuffle     = (train_sampler is None),
        num_workers = 2,
        pin_memory  = True,
        drop_last   = True,
        persistent_workers = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = BATCH_SIZE,
        sampler     = val_sampler,
        shuffle     = False,
        num_workers = 2,
        pin_memory  = True,
        persistent_workers = True,
    )

    # ---- Model ---------------------------------------------------------- #
    model = TaikoDiffusion(
        autoencoder_ckpt    = AUTOENCODER_CKPT,
        timesteps           = 1000,
        beta_schedule       = "linear",
        z_channels          = 16,
        n_mels              = 128,
        audio_base_channels = 64,
        audio_channel_mult  = [1, 1, 2, 2],
        unet_base_channels  = 64,
        unet_channel_mult   = [1, 2, 4],
        unet_num_res_blocks = 2,
        n_styles            = 4,
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)
        if is_main(rank):
            print(f"DDP enabled across {world_size} GPUs")

    m = unwrap(model)
    if is_main(rank):
        unet_p = sum(p.numel() for p in m.unet_model.parameters()) / 1e6
        wave_p = sum(p.numel() for p in m.wave_model.parameters()) / 1e6
        eff_bs = BATCH_SIZE * world_size
        print(f"U-Net: {unet_p:.1f}M | Audio enc: {wave_p:.1f}M")
        print(f"Effective batch size: {eff_bs} ({BATCH_SIZE} × {world_size} GPU)")

    optimizer = torch.optim.AdamW(
        list(m.unet_model.parameters()) +
        list(m.wave_model.parameters()),
        lr=LR, weight_decay=1e-4,
    )

    writer = SummaryWriter(LOG_DIR) if is_main(rank) else None
    if is_main(rank):
        CKPT_DIR.mkdir(parents=True, exist_ok=True)

    step, start_epoch, best_val = 0, 0, float("inf")
    max_steps = MAX_EPOCHS * len(train_loader)

    # ---- Resume --------------------------------------------------------- #
    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location=device)
        m.unet_model.load_state_dict(ckpt["unet"])
        m.wave_model.load_state_dict(ckpt["wave"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step        = ckpt["step"]
        start_epoch = ckpt["epoch"]
        best_val    = ckpt["best_val"]
        if is_main(rank):
            print(f"Resumed from step {step} (val_loss={best_val:.4f})")

    # ---- Training ------------------------------------------------------- #
    if is_main(rank):
        print(f"\nDiffusion training started")
        print(f"TensorBoard: tensorboard --logdir {LOG_DIR}\n")

    model.train()
    t0 = time.time()

    for epoch in range(start_epoch, MAX_EPOCHS):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for batch in train_loader:
            lr = get_lr(step, WARMUP_STEPS, max_steps, LR)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            loss, log_dict = m.training_loss(
                batch["tensor"].to(device),
                batch["mel"].to(device),
                batch["difficulty"].to(device),
                batch["style"].to(device),
                batch["valid_mask"].to(device),
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(m.unet_model.parameters()) +
                list(m.wave_model.parameters()),
                1.0
            )
            optimizer.step()
            step += 1

            if is_main(rank) and step % LOG_EVERY == 0:
                writer.add_scalar("train/loss",       log_dict["loss"],       step)
                writer.add_scalar("train/mae",        log_dict["mae"],        step)
                writer.add_scalar("train/mask_ratio", log_dict["mask_ratio"], step)
                writer.add_scalar("train/lr",         lr,                     step)
                print(
                    f"epoch {epoch+1:3d} | step {step:6d} | "
                    f"loss {log_dict['loss']:.4f} | "
                    f"mae {log_dict['mae']:.4f} | "
                    f"mask {log_dict['mask_ratio']:.2f} | "
                    f"t_mean {log_dict['t_mean']:.0f} | "
                    f"lr {lr:.2e} | {time.time()-t0:.0f}s"
                )

            if is_main(rank) and step % VAL_EVERY == 0:
                val_loss = validate(model, val_loader, device, VAL_BATCHES)
                writer.add_scalar("val/loss", val_loss, step)
                print(f"  → val loss: {val_loss:.4f}  (best: {best_val:.4f})")
                if val_loss < best_val:
                    best_val = val_loss
                    save_ckpt(CKPT_DIR / "best.pt", model, optimizer, step, epoch, best_val)

            if is_main(rank) and step % SAVE_EVERY == 0:
                save_ckpt(CKPT_DIR / f"step_{step:07d}.pt", model, optimizer,
                          step, epoch, best_val)
                cleanup_old(CKPT_DIR, KEEP_LAST_N)

        if is_main(rank):
            print(f"--- End of epoch {epoch+1} ---\n")

    if writer:
        writer.close()
    cleanup_ddp()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    train(args.resume)