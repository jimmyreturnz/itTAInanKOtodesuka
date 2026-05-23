"""
scripts/train_diffusion.py

Phase 3 — Diffusion model training, Kaggle T4x2 edition.

Features:
  - fp16 mixed precision (GradScaler)
  - DataParallel for T4x2
  - Loads from Kaggle input datasets (read-only)
  - Saves checkpoints to /kaggle/working

Usage on Kaggle:
    python scripts/train_diffusion.py
    python scripts/train_diffusion.py --resume /kaggle/working/checkpoints/diffusion/best.pt
"""

from __future__ import annotations
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from taiko.model.diffusion import TaikoDiffusion


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

IS_KAGGLE = os.path.exists("/kaggle/working")

if IS_KAGGLE:
    REPO_DIR   = Path("/kaggle/working/itTAInanKOtodesuka")
    DATA_DIR   = Path("/kaggle/input/datasets/jimmyreturnz/taiko-dataset")
    CKPT_INPUT = Path("/kaggle/input/datasets/jimmyreturnz/autoencoder")
    WORK_DIR   = Path("/kaggle/working")
else:
    REPO_DIR   = Path(".")
    DATA_DIR   = Path("data/processed")
    CKPT_INPUT = Path("checkpoints/autoencoder")
    WORK_DIR   = Path(".")

AUTOENCODER_CKPT = CKPT_INPUT / "best.pt"
INDEX_FILE       = DATA_DIR   / "colab_index.jsonl"
MELS_DIR         = DATA_DIR   / "mels"
TENSORS_DIR      = DATA_DIR   / "tensors"
CKPT_DIR         = WORK_DIR   / "checkpoints" / "diffusion"
LOG_DIR          = str(WORK_DIR / "runs" / "diffusion")


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

BATCH_SIZE       = 4        # per GPU — with 2x T4 effective = 8
LR               = 1e-4
MAX_EPOCHS       = 100
VAL_EVERY        = 500
SAVE_EVERY       = 500
LOG_EVERY        = 20
WARMUP_STEPS     = 1000
VAL_BATCHES      = 20
KEEP_LAST_N      = 3
VAL_RATIO        = 0.05
PAD_FRAMES       = 18_000   # beatmap + mel frames @ 20ms hop
USE_FP16         = True


# ---------------------------------------------------------------------------
# Dataset — loads precomputed .npz files only, no .osu needed
# ---------------------------------------------------------------------------

class PreprocessedDataset(Dataset):
    """
    Loads paired (mel.npz, tensor.npz) from precomputed files.
    No .osu files needed — everything is precomputed.
    """

    def __init__(
        self,
        records: list[dict],
        mels_dir: Path,
        tensors_dir: Path,
        pad_frames: int = PAD_FRAMES,
        augment: bool = False,
    ):
        self.records     = records
        self.mels_dir    = mels_dir
        self.tensors_dir = tensors_dir
        self.pad_frames  = pad_frames
        self.augment     = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        for _ in range(5):
            try:
                rec = self.records[idx]

                # Load mel
                mel_path = self.mels_dir / Path(rec["mel_path"]).name
                mel = np.load(str(mel_path))["mel"].astype(np.float32)  # [128, T]

                # Load beatmap tensor
                tensor_path = self.tensors_dir / Path(rec["tensor_path"]).name
                tensor = np.load(str(tensor_path))["tensor"].astype(np.float32)  # [7, T]

                T_tensor = tensor.shape[1]
                T_mel    = mel.shape[1]

                # Valid mask from tensor length
                valid_len  = min(T_tensor, self.pad_frames)
                valid_mask = np.zeros(self.pad_frames, dtype=np.float32)
                valid_mask[:valid_len] = 1.0

                # Pad / truncate tensor
                if T_tensor < self.pad_frames:
                    pad = np.zeros((tensor.shape[0], self.pad_frames - T_tensor), dtype=np.float32)
                    tensor = np.concatenate([tensor, pad], axis=1)
                else:
                    tensor = tensor[:, :self.pad_frames]

                # Pad / truncate mel — same frame count as tensor (20ms hop)
                if T_mel < self.pad_frames:
                    pad = np.zeros((mel.shape[0], self.pad_frames - T_mel), dtype=np.float32)
                    mel = np.concatenate([mel, pad], axis=1)
                else:
                    mel = mel[:, :self.pad_frames]

                # Augmentation: random time shift
                if self.augment and random.random() < 0.5:
                    shift = random.randint(0, max(0, valid_len - 1000))
                    tensor = np.roll(tensor, -shift, axis=1)
                    mel    = np.roll(mel,    -shift, axis=1)
                    tensor[:, -shift:] = 0
                    mel[:, -shift:]    = 0
                    valid_mask = np.roll(valid_mask, -shift)
                    valid_mask[-shift:] = 0

                difficulty = float(rec.get("star_rating", rec.get("difficulty", 5.0)))
                style      = int(rec.get("style", 0))

                return {
                    "mel":        torch.from_numpy(mel),
                    "tensor":     torch.from_numpy(tensor),
                    "valid_mask": torch.from_numpy(valid_mask),
                    "difficulty": torch.tensor(difficulty, dtype=torch.float32),
                    "style":      torch.tensor(style,      dtype=torch.long),
                }

            except Exception as e:
                idx = random.randint(0, len(self.records) - 1)

        return {
            "mel":        torch.zeros(128,          self.pad_frames),
            "tensor":     torch.zeros(7,            self.pad_frames),
            "valid_mask": torch.zeros(              self.pad_frames),
            "difficulty": torch.tensor(5.0,         dtype=torch.float32),
            "style":      torch.tensor(0,           dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Index loader
# ---------------------------------------------------------------------------

def load_index(index_file: Path, val_ratio: float = VAL_RATIO):
    records = []
    with open(index_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    random.seed(42)
    random.shuffle(records)
    n_val = max(1, int(len(records) * val_ratio))
    return records[n_val:], records[:n_val]


def print_stats(records: list[dict], name: str):
    diffs = [r.get("star_rating", r.get("difficulty", 0)) for r in records]
    styles = [r.get("style", 0) for r in records]
    style_names = {0: "standard", 1: "stream", 2: "speed", 3: "tech"}
    style_counts = {v: styles.count(k) for k, v in style_names.items()}
    print(f"{name}: {len(records)} maps | "
          f"SR {min(diffs):.1f}-{max(diffs):.1f} avg {sum(diffs)/len(diffs):.1f} | "
          f"styles: {style_counts}")


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


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def save_ckpt(path: Path, model, optimizer, scaler, step, epoch, best_val):
    path.parent.mkdir(parents=True, exist_ok=True)
    m = unwrap(model)
    torch.save({
        "unet":      m.unet_model.state_dict(),
        "wave":      m.wave_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler":    scaler.state_dict() if scaler else None,
        "step":      step,
        "epoch":     epoch,
        "best_val":  best_val,
    }, path)
    print(f"  Saved: {path}")


def cleanup_old(ckpt_dir: Path, keep: int):
    ckpts = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    for old in ckpts[:-keep]:
        old.unlink()


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
            )
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(resume_path=None):
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus   = torch.cuda.device_count()
    use_fp16 = USE_FP16 and device.type == "cuda"

    print(f"{'='*60}")
    print(f"Device  : {device}")
    print(f"GPUs    : {n_gpus}")
    print(f"fp16    : {use_fp16}")
    print(f"Kaggle  : {IS_KAGGLE}")
    for i in range(n_gpus):
        vram = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  ({vram:.1f} GB)")
    print(f"{'='*60}")

    # Verify paths
    for name, path in [
        ("Autoencoder", AUTOENCODER_CKPT),
        ("Index",       INDEX_FILE),
        ("Mels dir",    MELS_DIR),
        ("Tensors dir", TENSORS_DIR),
    ]:
        status = "✓" if Path(path).exists() else "✗ MISSING"
        print(f"  {name}: {path} {status}")

    if not AUTOENCODER_CKPT.exists():
        print("ERROR: Autoencoder checkpoint not found.")
        return
    if not INDEX_FILE.exists():
        print("ERROR: Index file not found.")
        return

    # ---- Data ----------------------------------------------------------- #
    train_records, val_records = load_index(INDEX_FILE)
    print_stats(train_records, "Train")
    print_stats(val_records,   "Val")

    train_ds = PreprocessedDataset(train_records, MELS_DIR, TENSORS_DIR, augment=True)
    val_ds   = PreprocessedDataset(val_records,   MELS_DIR, TENSORS_DIR, augment=False)

    # num_workers=2 works on Kaggle
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, drop_last=True,
        pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True,
    )

    # ---- Model ---------------------------------------------------------- #
    model = TaikoDiffusion(
        autoencoder_ckpt    = str(AUTOENCODER_CKPT),
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
    )

    if n_gpus > 1:
        print(f"Using DataParallel with {n_gpus} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)

    m = unwrap(model)
    unet_p = sum(p.numel() for p in m.unet_model.parameters()) / 1e6
    wave_p = sum(p.numel() for p in m.wave_model.parameters()) / 1e6
    print(f"U-Net: {unet_p:.1f}M | Audio enc: {wave_p:.1f}M")
    print(f"Batch: {BATCH_SIZE} per GPU × {n_gpus} GPU = {BATCH_SIZE * n_gpus} effective")

    optimizer = torch.optim.AdamW(
        list(m.unet_model.parameters()) +
        list(m.wave_model.parameters()),
        lr=LR, weight_decay=1e-4,
    )
    scaler   = GradScaler(enabled=use_fp16)
    writer   = SummaryWriter(LOG_DIR)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    step, start_epoch, best_val = 0, 0, float("inf")
    max_steps = MAX_EPOCHS * len(train_loader)

    # ---- Resume --------------------------------------------------------- #
    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location=device)
        m.unet_model.load_state_dict(ckpt["unet"])
        m.wave_model.load_state_dict(ckpt["wave"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") and scaler:
            scaler.load_state_dict(ckpt["scaler"])
        step        = ckpt["step"]
        start_epoch = ckpt["epoch"]
        best_val    = ckpt["best_val"]
        print(f"Resumed from step {step} (val_loss={best_val:.4f})")

    # ---- Training ------------------------------------------------------- #
    print(f"\nStarting diffusion training — {MAX_EPOCHS} epochs")
    print(f"TensorBoard: tensorboard --logdir {LOG_DIR}\n")

    model.train()
    t0 = time.time()

    for epoch in range(start_epoch, MAX_EPOCHS):
        for batch in train_loader:
            lr = get_lr(step, WARMUP_STEPS, max_steps, LR)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            with autocast(enabled=use_fp16):
                loss, log_dict = m.training_loss(
                    batch["tensor"].to(device),
                    batch["mel"].to(device),
                    batch["difficulty"].to(device),
                    batch["style"].to(device),
                    batch["valid_mask"].to(device),
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                list(m.unet_model.parameters()) +
                list(m.wave_model.parameters()),
                1.0,
            )
            scaler.step(optimizer)
            scaler.update()
            step += 1

            if step % LOG_EVERY == 0:
                writer.add_scalar("train/loss",       log_dict["loss"],       step)
                writer.add_scalar("train/mae",        log_dict["mae"],        step)
                writer.add_scalar("train/mask_ratio", log_dict["mask_ratio"], step)
                writer.add_scalar("train/lr",         lr,                     step)
                writer.add_scalar("train/grad_scale", scaler.get_scale(),     step)
                print(
                    f"epoch {epoch+1:3d} | step {step:6d} | "
                    f"loss {log_dict['loss']:.4f} | "
                    f"mae {log_dict['mae']:.4f} | "
                    f"mask {log_dict['mask_ratio']:.2f} | "
                    f"t {log_dict['t_mean']:.0f} | "
                    f"scale {scaler.get_scale():.0f} | "
                    f"lr {lr:.2e} | {time.time()-t0:.0f}s"
                )

            if step % VAL_EVERY == 0:
                val_loss = validate(model, val_loader, device, VAL_BATCHES, use_fp16)
                writer.add_scalar("val/loss", val_loss, step)
                print(f"  → val loss: {val_loss:.4f}  (best: {best_val:.4f})")
                if val_loss < best_val:
                    best_val = val_loss
                    save_ckpt(CKPT_DIR / "best.pt", model, optimizer, scaler,
                              step, epoch, best_val)

            if step % SAVE_EVERY == 0:
                save_ckpt(CKPT_DIR / f"step_{step:07d}.pt", model, optimizer, scaler,
                          step, epoch, best_val)
                cleanup_old(CKPT_DIR, KEEP_LAST_N)

        print(f"--- End of epoch {epoch+1} ---\n")

    writer.close()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    train(args.resume)