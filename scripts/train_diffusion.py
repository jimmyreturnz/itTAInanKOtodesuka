"""
scripts/train_diffusion.py

Phase 3 — Train the diffusion model.
Autoencoder is frozen. Trains audio encoder + U-Net.

Usage:
    python scripts/train_diffusion.py
    python scripts/train_diffusion.py --resume checkpoints/diffusion/best.pt
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
from taiko.data.audio import MelExtractor
from taiko.data.tensor_repr import beatmap_to_tensor, FRAME_MS, N_CHANNELS, MAX_FRAMES
from taiko.model.diffusion import TaikoDiffusion
from taiko.model.autoencoder import AutoencoderConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AUTOENCODER_CKPT = "checkpoints/autoencoder/best.pt"
CACHE_FILE       = "taiko_files_filtered.json"
CKPT_DIR         = Path("checkpoints/diffusion")
LOG_DIR          = "runs/diffusion"

BATCH_SIZE       = 4       # smaller than autoencoder due to audio encoder
LR               = 1e-4
MAX_EPOCHS       = 100
VAL_EVERY        = 500
SAVE_EVERY       = 500
LOG_EVERY        = 20
WARMUP_STEPS     = 1000
VAL_BATCHES      = 20
KEEP_LAST_N      = 3
VAL_RATIO        = 0.05
PAD_FRAMES       = 18_000  # beatmap tensor frames
COMPRESSION      = 16      # autoencoder compression ratio

# Style mapping
STYLE_MAP = {"standard": 0, "stream": 1, "speed": 2, "tech": 3}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DiffusionDataset(Dataset):
    """
    Returns paired (mel, beatmap_tensor, difficulty, style) samples.
    Loads audio and .osu on-the-fly.
    """

    def __init__(self, osu_files: list[Path], pad_frames: int = PAD_FRAMES):
        self.files      = osu_files
        self.pad_frames = pad_frames
        self.parser     = OsuTaikoParser()
        self.extractor  = MelExtractor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        for _ in range(5):
            try:
                osu_path = self.files[idx]
                bm = self.parser.parse_file(osu_path)
                if bm.note_count < 20 or not bm.audio_filename:
                    raise ValueError("invalid map")

                # Find audio
                audio_path = osu_path.parent / bm.audio_filename
                if not audio_path.exists():
                    # Try case-insensitive search
                    candidates = list(osu_path.parent.glob("*.mp3")) + \
                                 list(osu_path.parent.glob("*.ogg"))
                    if not candidates:
                        raise FileNotFoundError("no audio")
                    audio_path = candidates[0]

                # Mel spectrogram
                mel = self.extractor.extract(audio_path)   # [128, T_audio]

                # Pad/truncate mel to match pad_frames
                T_mel = mel.shape[1]
                if T_mel < self.pad_frames:
                    mel = np.concatenate([
                        mel,
                        np.zeros((128, self.pad_frames - T_mel), dtype=np.float32)
                    ], axis=1)
                else:
                    mel = mel[:, :self.pad_frames]

                # Beatmap tensor
                tensor = beatmap_to_tensor(bm, pad_to=None)
                T = tensor.shape[1]
                valid_len = min(T, self.pad_frames)
                valid_mask = np.zeros(self.pad_frames, dtype=np.float32)
                valid_mask[:valid_len] = 1.0

                if T < self.pad_frames:
                    pad = np.zeros((N_CHANNELS, self.pad_frames - T), dtype=np.float32)
                    tensor = np.concatenate([tensor, pad], axis=1)
                else:
                    tensor = tensor[:, :self.pad_frames]

                # Difficulty and style
                difficulty = float(bm.star_rating) if bm.star_rating > 0 else float(bm.overall_difficulty)
                style_name = _infer_style(bm)
                style      = STYLE_MAP[style_name]

                return {
                    "mel":        torch.from_numpy(mel),
                    "tensor":     torch.from_numpy(tensor),
                    "valid_mask": torch.from_numpy(valid_mask),
                    "difficulty": torch.tensor(difficulty, dtype=torch.float32),
                    "style":      torch.tensor(style,      dtype=torch.long),
                }
            except Exception:
                idx = random.randint(0, len(self.files) - 1)

        return {
            "mel":        torch.zeros(128, PAD_FRAMES),
            "tensor":     torch.zeros(N_CHANNELS, PAD_FRAMES),
            "valid_mask": torch.zeros(PAD_FRAMES),
            "difficulty": torch.tensor(5.0),
            "style":      torch.tensor(0, dtype=torch.long),
        }


def _infer_style(bm) -> str:
    """Infer map style from statistics."""
    nps = bm.notes_per_second
    if nps > 10:
        return "stream"
    elif nps > 8:
        return "speed"
    elif bm.big_ratio > 0.15:
        return "tech"
    return "standard"


def load_split(cache_file: str, val_ratio: float = VAL_RATIO):
    files = [Path(p) for p in json.loads(Path(cache_file).read_text())]
    random.seed(42)
    random.shuffle(files)
    n_val = max(1, int(len(files) * val_ratio))
    return files[n_val:], files[:n_val]


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
    torch.save({
        "unet":      model.unet_model.state_dict(),
        "wave":      model.wave_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step":      step,
        "epoch":     epoch,
        "best_val":  best_val,
    }, path)
    print(f"  Saved: {path}")


def cleanup(ckpt_dir, keep):
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
        loss, _ = model.training_loss(
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:  {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    if not Path(AUTOENCODER_CKPT).exists():
        print(f"ERROR: Autoencoder checkpoint not found: {AUTOENCODER_CKPT}")
        return

    # ---- Data ----------------------------------------------------------- #
    train_files, val_files = load_split(CACHE_FILE)
    print(f"Train: {len(train_files)} | Val: {len(val_files)}")

    train_ds = DiffusionDataset(train_files)
    val_ds   = DiffusionDataset(val_files)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0)

    # ---- Model ---------------------------------------------------------- #
    model = TaikoDiffusion(
        autoencoder_ckpt=AUTOENCODER_CKPT,
        timesteps=1000,
        beta_schedule="linear",
        z_channels=16,
        n_mels=128,
        audio_base_channels=64,
        audio_channel_mult=[1, 1, 2, 2],
        unet_base_channels=64,
        unet_channel_mult=[1, 2, 4],
        unet_num_res_blocks=2,
        n_styles=4,
    ).to(device)

    unet_params  = sum(p.numel() for p in model.unet_model.parameters()) / 1e6
    wave_params  = sum(p.numel() for p in model.wave_model.parameters()) / 1e6
    print(f"U-Net: {unet_params:.1f}M | Audio encoder: {wave_params:.1f}M")

    # Only train U-Net + audio encoder (autoencoder is frozen)
    optimizer = torch.optim.AdamW(
        list(model.unet_model.parameters()) +
        list(model.wave_model.parameters()),
        lr=LR, weight_decay=1e-4,
    )

    writer = SummaryWriter(LOG_DIR)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    step, start_epoch, best_val = 0, 0, float("inf")
    max_steps = MAX_EPOCHS * len(train_loader)

    # ---- Resume --------------------------------------------------------- #
    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location=device)
        model.unet_model.load_state_dict(ckpt["unet"])
        model.wave_model.load_state_dict(ckpt["wave"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step        = ckpt["step"]
        start_epoch = ckpt["epoch"]
        best_val    = ckpt["best_val"]
        print(f"Resumed from step {step}")

    # ---- Training ------------------------------------------------------- #
    print(f"\nStarting diffusion training")
    print(f"TensorBoard: tensorboard --logdir {LOG_DIR}\n")

    model.train()
    t0 = time.time()

    for epoch in range(start_epoch, MAX_EPOCHS):
        for batch in train_loader:
            lr = get_lr(step, WARMUP_STEPS, max_steps, LR)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            loss, log_dict = model.training_loss(
                batch["tensor"].to(device),
                batch["mel"].to(device),
                batch["difficulty"].to(device),
                batch["style"].to(device),
                batch["valid_mask"].to(device),
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(model.unet_model.parameters()) +
                list(model.wave_model.parameters()),
                1.0
            )
            optimizer.step()
            step += 1

            if step % LOG_EVERY == 0:
                writer.add_scalar("train/loss", log_dict["loss"], step)
                writer.add_scalar("train/mae",  log_dict["mae"],  step)
                writer.add_scalar("train/lr",   lr,               step)
                print(
                    f"epoch {epoch+1:3d} | step {step:6d} | "
                    f"loss {log_dict['loss']:.4f} | "
                    f"mae {log_dict['mae']:.4f} | "
                    f"t_mean {log_dict['t_mean']:.0f} | "
                    f"lr {lr:.2e} | {time.time()-t0:.0f}s"
                )

            if step % VAL_EVERY == 0:
                val_loss = validate(model, val_loader, device, VAL_BATCHES)
                writer.add_scalar("val/loss", val_loss, step)
                print(f"  → val loss: {val_loss:.4f}  (best: {best_val:.4f})")
                if val_loss < best_val:
                    best_val = val_loss
                    save_ckpt(CKPT_DIR / "best.pt", model, optimizer, step, epoch, best_val)

            if step % SAVE_EVERY == 0:
                save_ckpt(CKPT_DIR / f"step_{step:07d}.pt", model, optimizer, step, epoch, best_val)
                cleanup(CKPT_DIR, KEEP_LAST_N)

        print(f"--- End of epoch {epoch+1} ---\n")

    writer.close()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    train(args.resume)
