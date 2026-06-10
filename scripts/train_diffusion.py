"""
scripts/train_diffusion.py

Phase 3 — Train the diffusion model (frozen autoencoder).

Changes from previous version:
  - Uses WindowedDataset (30s windows) instead of PreprocessedDataset (full map)
  - Passes avg_nps, peak_nps, motif from batch to training_loss()
  - Profile-driven init via model_config.DiffusionProfile
  - --window-frames flag to control window size (default 1500 = 30s)

Profiles:
  p2      — 10-level audio encoder, S4, motif+nps conditioning (recommended)
  p1      — 4-level audio encoder, S4, motif+nps conditioning
  legacy  — small model, no S4, no motif/nps

Multi-GPU (Kaggle T4 x2):
  Uses nn.DataParallel BEFORE .to(device).
  --batch-size-per-gpu 2  →  total batch 4 on 2 GPUs.

Usage:
    python scripts/train_diffusion.py --profile p2
    python scripts/train_diffusion.py --profile p2 --resume checkpoints/diffusion/best.pt
    python scripts/train_diffusion.py --profile p2 --window-frames 1500
    python scripts/train_diffusion.py --profile p1 --batch-size-per-gpu 4
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
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from taiko.data.preprocessed_dataset import (
    WindowedDataset,
    load_index,
    print_index_stats,
    WINDOW_FRAMES_DEFAULT,
)
from taiko.model.diffusion import TaikoDiffusion
from taiko.model.model_config import PROFILES, get_profile


# ---------------------------------------------------------------------------
# Paths / schedule
# ---------------------------------------------------------------------------

AUTOENCODER_CKPT = "checkpoints/autoencoder/best.pt"
INDEX_FILE       = "data/processed/colab_index.jsonl"
DATA_ROOT        = Path("data/processed")
CKPT_DIR         = Path("checkpoints/diffusion")
LOG_DIR          = "runs/diffusion"

MAX_EPOCHS       = 200
VAL_EVERY        = 500
SAVE_EVERY       = 500
LOG_EVERY        = 20
WARMUP_STEPS     = 2000
VAL_BATCHES      = 20
KEEP_LAST_N      = 3
VAL_RATIO        = 0.05
USE_FP16         = False
LOG_DECODE_EVERY = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_lr(step, warmup, max_steps, lr, lr_min=1e-7):
    if step < warmup:
        return lr * max(step, 1) / max(warmup, 1)
    if max_steps is None or step >= max_steps:
        return lr_min
    t = (step - warmup) / (max_steps - warmup)
    return lr_min + (lr - lr_min) * 0.5 * (1 + math.cos(math.pi * t))


def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def setup_gpus(force_single: bool = False) -> tuple[torch.device, int, list[int]]:
    if not torch.cuda.is_available():
        return torch.device("cpu"), 0, []
    n   = torch.cuda.device_count()
    if force_single:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        return torch.device("cuda:0"), 1, [0]
    ids = list(range(n))
    return torch.device("cuda"), n, ids


def save_ckpt(path, model, optimizer, scaler, step, epoch, best_val, profile: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    m = unwrap(model)
    torch.save({
        "profile":   profile,
        "unet":      m.unet_model.state_dict(),
        "wave":      m.wave_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler":    scaler.state_dict(),
        "step":      step,
        "epoch":     epoch,
        "best_val":  best_val,
    }, path)
    print(f"  Saved: {path}")


def cleanup_old(ckpt_dir, keep):
    ckpts = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    for old in ckpts[:-keep]:
        old.unlink()


def load_weights(model, ckpt_path, device, profile: str):
    ckpt        = torch.load(ckpt_path, map_location=device)
    m           = unwrap(model)
    ckpt_profile = ckpt.get("profile", "legacy")
    strict       = ckpt_profile == profile
    u_miss = m.unet_model.load_state_dict(ckpt["unet"], strict=strict)
    w_miss = m.wave_model.load_state_dict(ckpt["wave"], strict=strict)
    if not strict:
        print(f"  Checkpoint profile={ckpt_profile}, training profile={profile} — partial load")
        if u_miss.missing_keys:
            print(f"  U-Net missing keys: {len(u_miss.missing_keys)}")
        if w_miss.missing_keys:
            print(f"  Wave missing keys: {len(w_miss.missing_keys)}")
    return ckpt


@torch.no_grad()
def validate(model, loader, device, max_batches, use_fp16):
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        with autocast(enabled=use_fp16):
            loss, _ = unwrap(model).training_loss(
                batch["tensor"].to(device),
                batch["mel"].to(device),
                batch["difficulty"].to(device),
                batch["style"].to(device),
                batch["valid_mask"].to(device),
                avg_nps  = batch["avg_nps"].to(device),
                peak_nps = batch["peak_nps"].to(device),
                motif    = batch["motif"].to(device),
            )
        total += loss.item()
        n     += 1
    model.train()
    return total / max(n, 1)


@torch.no_grad()
def log_decode_sample(model, batch, device, writer, step):
    m  = unwrap(model)
    x  = batch["tensor"][:1].to(device)
    z  = m.encode(x)
    pred = torch.sigmoid(m.first_stage_model.decode(z))
    onset_pred   = (pred[0, :4] > 0.5).float().sum().item()
    onset_target = (x[0, :4]   > 0.5).float().sum().item()
    writer.add_scalar("val/decode_onsets_pred",   onset_pred,   step)
    writer.add_scalar("val/decode_onsets_target", onset_target, step)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(
    resume_path:         str | None = None,
    profile_name:        str        = "p2",
    batch_size_per_gpu:  int | None = None,
    force_single_gpu:    bool       = False,
    num_workers:         int | None = None,
    window_frames:       int        = WINDOW_FRAMES_DEFAULT,
):
    profile = get_profile(profile_name)
    if batch_size_per_gpu is None:
        batch_size_per_gpu = profile.per_gpu_batch

    device, n_gpus, device_ids = setup_gpus(force_single_gpu)
    use_fp16 = USE_FP16 and device.type == "cuda"

    print(f"Profile      : {profile.name}")
    print(f"Window       : {window_frames} frames ({window_frames * 20 / 1000:.0f}s)")
    print(f"Device       : {device}")
    print(f"GPUs         : {n_gpus}  (ids={device_ids})")
    print(f"fp16         : {use_fp16}")
    print(f"use_nps      : {profile.use_nps}")
    print(f"use_motif    : {profile.use_motif}")
    for i in device_ids:
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}  ({props.total_memory / 1024**3:.1f} GB)")

    if not Path(AUTOENCODER_CKPT).exists():
        print(f"ERROR: Autoencoder checkpoint not found: {AUTOENCODER_CKPT}")
        return

    # ---- Dataset --------------------------------------------------------- #
    train_records, val_records = load_index(INDEX_FILE, val_ratio=VAL_RATIO)
    print_index_stats(train_records, "Train")
    print_index_stats(val_records,   "Val")

    train_ds = WindowedDataset(
        train_records,
        DATA_ROOT,
        window_frames=window_frames,
        random_window=True,
        augment=True,
    )
    val_ds = WindowedDataset(
        val_records,
        DATA_ROOT,
        window_frames=window_frames,
        random_window=False,   # deterministic for fair val comparison
        augment=False,
    )

    total_batch = batch_size_per_gpu * max(n_gpus, 1)
    if n_gpus > 1 and batch_size_per_gpu < 2:
        print("WARNING: batch-size-per-gpu < 2 with DataParallel is inefficient")

    nw = num_workers if num_workers is not None else min(4, max(n_gpus, 1) * 2)

    train_loader = DataLoader(
        train_ds,
        batch_size=total_batch,
        shuffle=True,
        num_workers=nw,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(nw > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=total_batch,
        shuffle=False,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(nw > 0),
    )

    # ---- Model ----------------------------------------------------------- #
    model = TaikoDiffusion(
        autoencoder_ckpt = AUTOENCODER_CKPT,
        profile          = profile_name,
        timesteps        = 1000,
        beta_schedule    = "linear",
        z_channels       = 16,
        n_mels           = 128,
        n_styles         = 4,
    )

    if n_gpus > 1:
        print(f"Wrapping nn.DataParallel on device_ids={device_ids}")
        model = nn.DataParallel(model, device_ids=device_ids)

    model = model.to(device)

    m = unwrap(model)
    print(f"U-Net: {m.unet_model.count_parameters()} | "
          f"Audio: {sum(p.numel() for p in m.wave_model.parameters())/1e6:.1f}M")
    print(f"Batch: {total_batch} total ({batch_size_per_gpu} per GPU × {max(n_gpus,1)} GPU)")

    # ---- Optimiser ------------------------------------------------------- #
    optimizer = torch.optim.AdamW(
        list(m.unet_model.parameters()) + list(m.wave_model.parameters()),
        lr=profile.lr,
        weight_decay=1e-4,
    )
    scaler = GradScaler(enabled=use_fp16)
    writer = SummaryWriter(LOG_DIR)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    step, start_epoch, best_val = 0, 0, float("inf")
    max_steps = MAX_EPOCHS * len(train_loader)

    # ---- Resume ---------------------------------------------------------- #
    if resume_path and Path(resume_path).exists():
        ckpt = load_weights(model, resume_path, device, profile.name)
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        step        = ckpt["step"]
        start_epoch = ckpt["epoch"]
        best_val    = ckpt["best_val"]
        print(f"Resumed from step {step} (val_loss={best_val:.5f})")

    print(f"\nTraining  profile={profile.name}  lr={profile.lr}  "
          f"s4={profile.use_s4}  window={window_frames}f")
    print(f"TensorBoard: tensorboard --logdir {LOG_DIR}\n")

    model.train()
    t0       = time.time()
    val_iter = iter(val_loader)

    for epoch in range(start_epoch, MAX_EPOCHS):
        for batch in train_loader:

            lr = get_lr(step, WARMUP_STEPS, max_steps, profile.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # ---- Forward ------------------------------------------------- #
            with autocast(enabled=use_fp16):
                loss, log_dict = unwrap(model).training_loss(
                    batch["tensor"].to(device),
                    batch["mel"].to(device),
                    batch["difficulty"].to(device),
                    batch["style"].to(device),
                    batch["valid_mask"].to(device),
                    avg_nps  = batch["avg_nps"].to(device)   if profile.use_nps   else None,
                    peak_nps = batch["peak_nps"].to(device)  if profile.use_nps   else None,
                    motif    = batch["motif"].to(device)      if profile.use_motif else None,
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                list(m.unet_model.parameters()) + list(m.wave_model.parameters()),
                1.0,
            )
            scaler.step(optimizer)
            scaler.update()
            step += 1

            # ---- Logging ------------------------------------------------- #
            if step % LOG_EVERY == 0:
                writer.add_scalar("train/loss",       log_dict["loss"],       step)
                writer.add_scalar("train/mae",        log_dict["mae"],        step)
                writer.add_scalar("train/mask_ratio", log_dict["mask_ratio"], step)
                writer.add_scalar("train/lr",         lr,                     step)
                print(
                    f"epoch {epoch+1:3d} | step {step:6d} | "
                    f"loss {log_dict['loss']:.4f} | mae {log_dict['mae']:.4f} | "
                    f"mask {log_dict['mask_ratio']:.2f} | lr {lr:.2e} | "
                    f"{time.time()-t0:.0f}s"
                )

            # ---- Validation ---------------------------------------------- #
            if step % VAL_EVERY == 0:
                val_loss = validate(model, val_loader, device, VAL_BATCHES, use_fp16)
                writer.add_scalar("val/loss", val_loss, step)
                print(f"  → val loss: {val_loss:.5f}  (best: {best_val:.5f})")
                if val_loss < best_val:
                    best_val = val_loss
                    save_ckpt(CKPT_DIR / "best.pt", model, optimizer, scaler,
                              step, epoch, best_val, profile.name)

            # ---- Decode sanity check ------------------------------------- #
            if step % LOG_DECODE_EVERY == 0:
                try:
                    vb = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    vb = next(val_iter)
                log_decode_sample(unwrap(model), vb, device, writer, step)

            # ---- Checkpoint ---------------------------------------------- #
            if step % SAVE_EVERY == 0:
                save_ckpt(CKPT_DIR / f"step_{step:07d}.pt", model, optimizer,
                          scaler, step, epoch, best_val, profile.name)
                cleanup_old(CKPT_DIR, KEEP_LAST_N)

        print(f"--- End of epoch {epoch+1} ---\n")

    writer.close()
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume",             default=None)
    ap.add_argument("--profile",            default="p2", choices=list(PROFILES.keys()))
    ap.add_argument("--batch-size-per-gpu", type=int, default=None,
                    help="Per-GPU batch size (default: from profile)")
    ap.add_argument("--single-gpu",         action="store_true",
                    help="Force cuda:0 only")
    ap.add_argument("--num-workers",        type=int, default=None)
    ap.add_argument("--window-frames",      type=int, default=WINDOW_FRAMES_DEFAULT,
                    help=f"Window size in tensor frames (default: {WINDOW_FRAMES_DEFAULT} = "
                         f"{WINDOW_FRAMES_DEFAULT*20//1000}s)")
    args = ap.parse_args()

    train(
        resume_path        = args.resume,
        profile_name       = args.profile,
        batch_size_per_gpu = args.batch_size_per_gpu,
        force_single_gpu   = args.single_gpu,
        num_workers        = args.num_workers,
        window_frames      = args.window_frames,
    )