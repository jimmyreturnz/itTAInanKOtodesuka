"""
scripts/train.py

Phase 3 — Training loop.

Features:
  - Gradient accumulation (effective batch size = batch_size * grad_accumulation)
  - Cosine LR schedule with linear warmup
  - Gradient clipping
  - Checkpointing (save + resume)
  - TensorBoard logging
  - Validation loop

Usage:
    python scripts/train.py --config configs/base.yaml
    python scripts/train.py --config configs/base.yaml --resume checkpoints/step_1000.pt
"""

from __future__ import annotations
import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from taiko.data.dataset import TaikoDataset, taiko_collate_fn
from taiko.data.tokenizer import TaikoVocabulary
from taiko.model.model import TaikoMapper, TaikoModelConfig


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_nested(cfg: dict, *keys, default=None):
    for k in keys:
        if not isinstance(cfg, dict):
            return default
        cfg = cfg.get(k, default)
    return cfg


# ---------------------------------------------------------------------------
# LR scheduler: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup_steps: int, max_steps: int, lr: float, lr_min: float) -> float:
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    if max_steps is None or step >= max_steps:
        return lr_min
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    cosine   = 0.5 * (1 + math.cos(math.pi * progress))
    return lr_min + (lr - lr_min) * cosine


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: TaikoMapper,
    optimizer: torch.optim.Optimizer,
    scheduler_step: int,
    epoch: int,
    step: int,
    best_val_loss: float,
    config: dict,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_step":  scheduler_step,
        "epoch":           epoch,
        "step":            step,
        "best_val_loss":   best_val_loss,
        "config":          config,
    }, path)
    print(f"  Saved checkpoint: {path}")


def load_checkpoint(path: Path, model: TaikoMapper, optimizer: torch.optim.Optimizer):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"  Resumed from: {path} (step {ckpt['step']}, epoch {ckpt['epoch']})")
    return ckpt


def cleanup_old_checkpoints(ckpt_dir: Path, keep_last_n: int):
    ckpts = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    for old in ckpts[:-keep_last_n]:
        old.unlink()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: TaikoMapper,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        loss = forward_batch(model, batch, device)
        total_loss += loss.item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


# ---------------------------------------------------------------------------
# Single batch forward
# ---------------------------------------------------------------------------

def forward_batch(model: TaikoMapper, batch: dict, device: torch.device) -> torch.Tensor:
    mel        = batch["mel"].to(device)
    token_ids  = batch["token_ids"].to(device)
    cond_ids   = batch["conditioning"].to(device)
    token_mask = batch["token_mask"].to(device)
    return model(mel, token_ids, cond_ids, token_mask)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config_path: str, resume_path: str | None = None):
    cfg = load_config(config_path)

    # ---- Setup ---------------------------------------------------------- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ---- Vocabulary ----------------------------------------------------- #
    vocab = TaikoVocabulary()
    print(f"Vocabulary size: {len(vocab)}")

    # ---- Dataset -------------------------------------------------------- #
    t_cfg = cfg["training"]
    d_cfg = cfg["data"]

    train_ds = TaikoDataset(
        d_cfg["train_index"],
        vocab=vocab,
        window_ms=d_cfg["window_ms"],
        max_seq_len=d_cfg["max_seq_len"],
        training=True,
    )
    val_ds = TaikoDataset(
        d_cfg["val_index"],
        vocab=vocab,
        window_ms=d_cfg["window_ms"],
        max_seq_len=d_cfg["max_seq_len"],
        training=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=t_cfg["batch_size"],
        shuffle=True,
        num_workers=d_cfg["num_workers"],
        collate_fn=taiko_collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=t_cfg["batch_size"],
        shuffle=False,
        num_workers=d_cfg["num_workers"],
        collate_fn=taiko_collate_fn,
        drop_last=False,
    )
    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # ---- Model ---------------------------------------------------------- #
    m_cfg = cfg["model"]
    model_config = TaikoModelConfig(
        vocab_size=len(vocab),
        n_mels=m_cfg["n_mels"],
        encoder_d_model=m_cfg["encoder_d_model"],
        d_model=m_cfg["d_model"],
        nhead=m_cfg["nhead"],
        num_layers=m_cfg["num_layers"],
        dim_feedforward=m_cfg["dim_feedforward"],
        dropout=m_cfg["dropout"],
        max_seq_len=m_cfg["max_seq_len"],
        n_cond_tokens=m_cfg["n_cond_tokens"],
        cfg_dropout=m_cfg["cfg_dropout"],
    )
    model = TaikoMapper(model_config).to(device)
    params = model.count_parameters()
    print(f"Model: encoder={params['encoder']}  decoder={params['decoder']}  total={params['total']}")

    # ---- Optimizer ------------------------------------------------------ #
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=t_cfg["lr"],
        betas=tuple(t_cfg["betas"]),
        eps=t_cfg["eps"],
        weight_decay=t_cfg["weight_decay"],
    )

    # ---- State ---------------------------------------------------------- #
    start_epoch    = 0
    global_step    = 0
    best_val_loss  = float("inf")
    grad_accum     = t_cfg["grad_accumulation"]
    max_steps      = t_cfg.get("max_steps")
    max_epochs     = t_cfg["max_epochs"]

    if max_steps is None:
        max_steps = max_epochs * math.ceil(len(train_loader) / grad_accum)

    # ---- Resume --------------------------------------------------------- #
    if resume_path:
        ckpt = load_checkpoint(Path(resume_path), model, optimizer)
        start_epoch   = ckpt["epoch"]
        global_step   = ckpt["step"]
        best_val_loss = ckpt["best_val_loss"]

    # ---- TensorBoard ---------------------------------------------------- #
    writer = SummaryWriter(log_dir=t_cfg["tensorboard_dir"])

    # ---- Checkpoint dir ------------------------------------------------- #
    ckpt_dir = Path(t_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- Training loop -------------------------------------------------- #
    print(f"\nStarting training — {max_epochs} epochs, {max_steps} steps")
    print(f"Effective batch size: {t_cfg['batch_size']} × {grad_accum} = {t_cfg['batch_size'] * grad_accum}")
    print(f"TensorBoard: tensorboard --logdir {t_cfg['tensorboard_dir']}\n")

    model.train()
    optimizer.zero_grad()
    running_loss   = 0.0
    accum_count    = 0
    t_epoch_start  = time.time()

    for epoch in range(start_epoch, max_epochs):
        for batch in train_loader:
            # ---- LR update ---------------------------------------------- #
            lr = get_lr(global_step, t_cfg["warmup_steps"], max_steps, t_cfg["lr"], t_cfg["lr_min"])
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # ---- Forward ------------------------------------------------ #
            loss = forward_batch(model, batch, device)
            loss = loss / grad_accum
            loss.backward()

            running_loss += loss.item()
            accum_count  += 1

            # ---- Optimizer step (every grad_accum batches) -------------- #
            if accum_count == grad_accum:
                nn.utils.clip_grad_norm_(model.parameters(), t_cfg["grad_clip"])
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                accum_count  = 0

                train_loss = running_loss  # already divided by grad_accum above
                running_loss = 0.0

                # ---- Logging -------------------------------------------- #
                if global_step % t_cfg["log_every_steps"] == 0:
                    writer.add_scalar("train/loss", train_loss, global_step)
                    writer.add_scalar("train/lr",   lr,         global_step)
                    elapsed = time.time() - t_epoch_start
                    print(
                        f"epoch {epoch+1:3d} | step {global_step:6d} | "
                        f"loss {train_loss:.4f} | lr {lr:.2e} | "
                        f"{elapsed:.0f}s"
                    )

                # ---- Validation ----------------------------------------- #
                if global_step % t_cfg["val_every_steps"] == 0:
                    val_loss = validate(model, val_loader, device, t_cfg["val_batches"])
                    writer.add_scalar("val/loss", val_loss, global_step)
                    print(f"  → val loss: {val_loss:.4f}  (best: {best_val_loss:.4f})")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(
                            ckpt_dir / "best.pt",
                            model, optimizer, global_step, epoch,
                            global_step, best_val_loss, cfg,
                        )

                # ---- Checkpoint ----------------------------------------- #
                if global_step % t_cfg["save_every_steps"] == 0:
                    ckpt_path = ckpt_dir / f"step_{global_step:07d}.pt"
                    save_checkpoint(
                        ckpt_path,
                        model, optimizer, global_step, epoch,
                        global_step, best_val_loss, cfg,
                    )
                    cleanup_old_checkpoints(ckpt_dir, t_cfg["keep_last_n"])

                # ---- Max steps ------------------------------------------ #
                if max_steps and global_step >= max_steps:
                    print(f"\nReached max_steps={max_steps}. Training complete.")
                    writer.close()
                    return

        t_epoch_start = time.time()
        print(f"\n--- End of epoch {epoch+1} ---\n")

    print("Training complete.")
    writer.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    train(args.config, args.resume)
