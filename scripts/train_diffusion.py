"""
scripts/train_diffusion.py

Stage 2. Trains the latent diffusion model against a frozen autoencoder.

    python scripts/train_diffusion.py --ae checkpoints/autoencoder/best.pt
    python scripts/train_diffusion.py --resume checkpoints/diffusion/last.pt

Built for interruption. A Kaggle session caps at about 12 hours and this stage
wants 150-250 GPU-hours, so it will be resumed roughly twenty times; every
checkpoint carries the optimiser, the scaler, the EMA and the step count, and
--resume picks up mid-epoch rather than restarting one.

Gate B: onset F1 above 0.40 against held-out audio. That is the alignment gate
-- it is what distinguishes a model listening to the music from one emitting
plausible rhythms. Run scripts/evaluate.py to measure it; a diffusion model
that fails Gate B is not fixed by more steps.

Multi-GPU: nn.DataParallel, which works inside a Kaggle notebook where
torchrun does not. The loss lives in TaikoDiffusion.forward(), so the wrapper
actually scatters -- the previous loop called unwrap(model).training_loss()
and reached through it, leaving the second T4 idle for every run ever done.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from taiko.data.frames import describe
from taiko.data.preprocessed_dataset import (
    WINDOW_FRAMES_DEFAULT, WindowedDataset, print_split_stats, split_indices,
)
from taiko.data.shards import ShardReader
from taiko.model.diffusion import EMA, TaikoDiffusion
from taiko.model.model_config import PROFILES, get_profile


def unwrap(model: nn.Module) -> TaikoDiffusion:
    return model.module if isinstance(model, nn.DataParallel) else model


def lr_at(step: int, warmup: int, total: int, peak: float, floor: float = 1e-6) -> float:
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    if total is None or step >= total:
        return floor
    progress = (step - warmup) / max(total - warmup, 1)
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def batch_to(batch: dict, device: torch.device) -> dict:
    """Only the tensors the model consumes, moved once."""
    keys = ("chart", "mel", "timing", "difficulty", "style", "valid_mask",
            "avg_nps", "peak_nps", "motif", "motif_mask")
    return {k: batch[k].to(device, non_blocking=True) for k in keys}


@torch.no_grad()
def validate(model, loader, device, max_batches: int, use_fp16: bool) -> float:
    model.eval()
    total, seen = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        with torch.amp.autocast("cuda", enabled=use_fp16):
            loss, _ = model(**batch_to(batch, device))
        total += float(loss.mean())
        seen += 1
    model.train()
    return total / max(seen, 1)


def save(path: Path, model, optimizer, scaler, ema, step, epoch, best, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    inner = unwrap(model)
    torch.save({
        "unet": inner.unet_model.state_dict(),
        "wave": inner.wave_model.state_dict(),
        "ema": ema.state_dict() if ema else None,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step, "epoch": epoch, "best_val": best,
        "profile": args.profile,
        "window_frames": args.window_frames,
        "prediction_type": args.prediction_type,
        "autoencoder_ckpt": str(args.ae),
    }, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae", type=Path, default=Path("checkpoints/autoencoder/best.pt"),
                    help="frozen autoencoder checkpoint from stage 1")
    ap.add_argument("--shards", type=Path, default=Path("data/processed/shards"))
    ap.add_argument("--out", type=Path, default=Path("checkpoints/diffusion"))
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--profile", default="p1", choices=sorted(PROFILES))
    ap.add_argument("--window-frames", type=int, default=WINDOW_FRAMES_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=None, help="per GPU; default from profile")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--samples-per-epoch", type=int, default=20_000)
    ap.add_argument("--lr", type=float, default=None, help="default from profile")
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--ema-decay", type=float, default=0.9995)
    ap.add_argument("--prediction-type", default="v", choices=["v", "epsilon"])
    ap.add_argument("--ranked-only", action="store_true",
                    help="train only on ranked maps (recommended for the final run)")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop cleanly before a session limit, saving first")
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--fp32", dest="fp16", action="store_false")
    ap.add_argument("--single-gpu", action="store_true")
    ap.add_argument("--no-grad-checkpoint", action="store_true",
                    help="disable gradient checkpointing. It buys memory with "
                         "recomputation, which is a bad trade when the card is "
                         "empty: p1 at 8/GPU uses under 1 GiB of 15.")
    args = ap.parse_args()

    profile = get_profile(args.profile)
    if args.no_grad_checkpoint:
        profile = replace(profile, use_checkpoint=False)
    batch_per_gpu = args.batch_size or profile.per_gpu_batch
    lr = args.lr or profile.lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Every batch is the same shape, so let cuDNN pick its fastest kernels
    # once instead of re-deciding per call. 5-15% on a conv-dominated run.
    torch.backends.cudnn.benchmark = True
    n_gpus = 0 if device.type == "cpu" else (1 if args.single_gpu else torch.cuda.device_count())
    use_fp16 = args.fp16 and device.type == "cuda"

    print(describe())
    print(f"Device: {device}  GPUs: {n_gpus}  fp16: {use_fp16}")
    for i in range(n_gpus):
        p = torch.cuda.get_device_properties(i)
        print(f"  cuda:{i}  {p.name}  {p.total_memory / 1024**3:.1f} GB")
    print(f"Profile: {profile.summary()}")

    if not args.ae.exists():
        print(f"ERROR: no autoencoder checkpoint at {args.ae}")
        print("  Run scripts/train_autoencoder.py first and clear Gate A.")
        return 1

    ae_ckpt = torch.load(args.ae, map_location="cpu", weights_only=False)
    gate_a = ae_ckpt.get("best_f1", 0.0)
    print(f"Autoencoder Gate A: onset F1 {gate_a:.4f}")
    if gate_a < 0.98:
        print("  WARNING: this autoencoder did not clear Gate A (0.98). Whatever")
        print("  it loses is a ceiling on the diffusion model, and no amount of")
        print("  diffusion training recovers it.")

    # ---- data ------------------------------------------------------------ #
    reader = ShardReader(args.shards)
    train_idx, val_idx = split_indices(reader, val_ratio=0.05, ranked_only=args.ranked_only)
    print_split_stats(reader, train_idx, "Train")
    print_split_stats(reader, val_idx, "Val")

    train_ds = WindowedDataset(
        reader, train_idx, window_frames=args.window_frames,
        random_window=True, augment=True, samples_per_epoch=args.samples_per_epoch,
    )
    val_ds = WindowedDataset(
        reader, val_idx, window_frames=args.window_frames,
        random_window=False, augment=False,
    )

    total_batch = batch_per_gpu * max(n_gpus, 1)
    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, batch_size=total_batch, shuffle=True,
                              drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=total_batch, shuffle=False, **loader_kwargs)

    # ---- model ------------------------------------------------------------ #
    model = TaikoDiffusion(
        autoencoder_ckpt=str(args.ae),
        profile=profile,          # the object, not the name -- --no-grad-checkpoint
                                  # modifies it and passing args.profile would
                                  # silently re-fetch the unmodified original
        prediction_type=args.prediction_type,
    )
    model.first_stage.check_window(args.window_frames)
    model = model.to(device)

    inner = unwrap(model)
    ema = EMA(inner.trainable_parameters(), decay=args.ema_decay)

    if n_gpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(n_gpus)))
        print(f"DataParallel across {n_gpus} GPUs")

    trainable = sum(p.numel() for p in inner.trainable_parameters())
    print(f"Trainable: {trainable / 1e6:.1f}M | "
          f"batch {total_batch} ({batch_per_gpu}/GPU x {max(n_gpus, 1)}) "
          f"x {args.grad_accum} accum = {total_batch * args.grad_accum} effective")

    optimizer = torch.optim.AdamW(inner.trainable_parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    step, start_epoch, best_val = 0, 0, float("inf")
    if args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if ckpt.get("profile") != args.profile:
            print(f"ERROR: checkpoint is profile {ckpt.get('profile')!r}, "
                  f"you asked for {args.profile!r}. Shapes will not match.")
            return 1
        inner.unet_model.load_state_dict(ckpt["unet"])
        inner.wave_model.load_state_dict(ckpt["wave"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        if ckpt.get("ema"):
            ema.load_state_dict(ckpt["ema"])
        step, start_epoch, best_val = ckpt["step"], ckpt["epoch"], ckpt["best_val"]
        print(f"Resumed: step {step}, epoch {start_epoch}, best val {best_val:.5f}")

    total_steps = args.epochs * len(train_loader) // args.grad_accum
    print(f"\n{len(train_loader)} batches/epoch, ~{total_steps} optimiser steps total")
    if args.max_hours:
        print(f"Will stop cleanly after {args.max_hours:.1f} hours")
    print()

    t0 = time.time()
    model.train()
    stop = False

    for epoch in range(start_epoch, args.epochs):
        if stop:
            break
        for i, batch in enumerate(train_loader):
            lr_now = lr_at(step, args.warmup, total_steps, lr)
            for group in optimizer.param_groups:
                group["lr"] = lr_now

            with torch.amp.autocast("cuda", enabled=use_fp16):
                loss, metrics = model(**batch_to(batch, device))
                # DataParallel returns one row per replica; averaging is what
                # makes the reported numbers mean the same thing on 1 GPU and 2.
                loss = loss.mean()
                metrics = metrics.reshape(-1, 4).mean(0)
                scaled = loss / args.grad_accum

            scaler.scale(scaled).backward()

            if (i + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(inner.trainable_parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(inner.trainable_parameters())
                step += 1

                if step % args.log_every == 0:
                    elapsed = time.time() - t0
                    print(f"epoch {epoch + 1:3d} step {step:7d}  "
                          f"loss {float(metrics[0]):.4f}  mae {float(metrics[1]):.4f}  "
                          f"|g| {float(grad_norm):.2f}  lr {lr_now:.2e}  "
                          f"{elapsed / 60:.1f}min")

                if step % args.val_every == 0:
                    # Validate on the EMA weights: those are what generation
                    # uses, so they are the ones whose quality matters.
                    ema.store(inner.trainable_parameters())
                    ema.copy_to(inner.trainable_parameters())
                    val_loss = validate(model, val_loader, device, args.val_batches, use_fp16)
                    ema.restore(inner.trainable_parameters())

                    marker = ""
                    if val_loss < best_val:
                        best_val = val_loss
                        save(args.out / "best.pt", model, optimizer, scaler, ema,
                             step, epoch, best_val, args)
                        marker = "  <- best"
                    print(f"  val {val_loss:.5f} (best {best_val:.5f}){marker}")

                if step % args.save_every == 0:
                    save(args.out / "last.pt", model, optimizer, scaler, ema,
                         step, epoch, best_val, args)

                if args.max_hours and (time.time() - t0) / 3600 >= args.max_hours:
                    print(f"\nReached {args.max_hours}h; saving and stopping.")
                    stop = True
                    break

    save(args.out / "last.pt", model, optimizer, scaler, ema, step, epoch, best_val, args)
    print(f"\n{'=' * 60}")
    print(f"Stopped at step {step}, best val {best_val:.5f}")
    print(f"Checkpoints: {args.out / 'best.pt'}  {args.out / 'last.pt'}")
    print("\nNext, measure Gate B (onset F1 > 0.40 on held-out audio):")
    print(f"  python scripts/evaluate.py --diffusion {args.out / 'best.pt'} --ae {args.ae}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
