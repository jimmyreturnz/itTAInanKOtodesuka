"""
scripts/train_autoencoder.py

Stage 1. Trains the chart autoencoder and calibrates its latent scale.

    python scripts/train_autoencoder.py --shards data/processed/shards
    python scripts/train_autoencoder.py --resume checkpoints/autoencoder/last.pt

Gate A: onset F1 at +/- 1 frame must reach 0.98 before training the diffusion
model. Validation loss is not the gate and never was -- a loss of 0.01 says
nothing about whether onsets came back on the right frames, and onsets on the
right frames is the whole product. The gate is evaluated and printed on every
validation pass so it is impossible to finish a run without knowing.

If the gate will not clear at 16x compression, drop one entry from
--channel-mult for 8x and retrain. A first stage that loses notes puts a
ceiling on everything downstream that no amount of diffusion training removes.
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

from taiko.data.frames import describe
from taiko.data.preprocessed_dataset import (
    WINDOW_FRAMES_DEFAULT, WindowedDataset, print_split_stats, split_indices,
)
from taiko.data.shards import ShardReader
from taiko.data.tensor_repr import CHART_CHANNEL_NAMES, ONSET_CHANNELS
from taiko.model.autoencoder import AutoencoderConfig, ChartAutoencoder

GATE_A_F1 = 0.98


def unwrap(model: nn.Module) -> ChartAutoencoder:
    return model.module if isinstance(model, nn.DataParallel) else model


def lr_at(step: int, warmup: int, total: int, peak: float, floor: float = 1e-6) -> float:
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    if total is None or step >= total:
        return floor
    progress = (step - warmup) / max(total - warmup, 1)
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * progress))


# Candidate decision thresholds, swept on validation.
#
# The loss weights positives about 40:1 to stop the model collapsing to silence
# against a corpus that is 99.5% zeros. That weighting deliberately breaks
# probability calibration: the model is trained to be right about the reweighted
# distribution, so at a fixed 0.5 threshold it over-predicts by roughly the
# weight factor -- recall pins to 1.0 while precision sits near 0.5, which is a
# chart with twice as many notes as it should have.
#
# The fix is to pick the operating point rather than assume it. The threshold
# is measured here and stored in the checkpoint, so decoding uses the same
# value the gate was measured at.
THRESHOLDS = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)


@torch.no_grad()
def onset_reconstruction_f1(
    model: ChartAutoencoder,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    tolerance_frames: int = 1,
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> dict:
    """
    Gate A: onset F1 at the best decision threshold.

    A predicted onset counts as correct when a true onset sits within
    `tolerance_frames`. Tolerance is not generosity -- at 20 ms frames a note
    can already be quantised half a frame from where it was played, so
    demanding an exact frame would measure the grid rather than the model.

    Dilation by max-pooling is what makes "within k frames" a GPU operation
    rather than a Python loop over every note.
    """
    model.eval()
    kernel = 2 * tolerance_frames + 1

    counts = {t: [0, 0, 0] for t in thresholds}
    per_channel = {t: {name: [0, 0, 0] for name in CHART_CHANNEL_NAMES} for t in thresholds}

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        chart = batch["chart"].to(device)
        mask = batch["valid_mask"].to(device).unsqueeze(1)

        recon = model.reconstruct(chart)
        actual = (chart > 0.5).float() * mask
        near_actual = torch.max_pool1d(actual, kernel, stride=1, padding=tolerance_frames)

        for threshold in thresholds:
            predicted = (recon > threshold).float() * mask
            near_predicted = torch.max_pool1d(
                predicted, kernel, stride=1, padding=tolerance_frames
            )
            for ch in ONSET_CHANNELS:
                name = CHART_CHANNEL_NAMES[ch]
                hits = int((predicted[:, ch] * near_actual[:, ch]).sum())
                spurious = int((predicted[:, ch] * (1 - near_actual[:, ch])).sum())
                misses = int((actual[:, ch] * (1 - near_predicted[:, ch])).sum())
                per_channel[threshold][name][0] += hits
                per_channel[threshold][name][1] += spurious
                per_channel[threshold][name][2] += misses
                counts[threshold][0] += hits
                counts[threshold][1] += spurious
                counts[threshold][2] += misses

    model.train()

    def score(threshold: float) -> tuple[float, float, float]:
        tp, fp, fn = counts[threshold]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        return 2 * precision * recall / max(precision + recall, 1e-9), precision, recall

    best_threshold = max(thresholds, key=lambda t: score(t)[0])
    f1, precision, recall = score(best_threshold)
    tp, fp, fn = counts[best_threshold]

    return {
        "f1": f1, "precision": precision, "recall": recall,
        "threshold": best_threshold,
        "tp": tp, "fp": fp, "fn": fn,
        "per_channel": per_channel[best_threshold],
        "sweep": {t: score(t)[0] for t in thresholds},
    }


@torch.no_grad()
def validate(model, loader, device, max_batches: int) -> float:
    model.eval()
    total, seen = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        loss, _ = unwrap(model).training_loss(
            batch["chart"].to(device), batch["valid_mask"].to(device)
        )
        total += float(loss)
        seen += 1
    model.train()
    return total / max(seen, 1)


def save(path: Path, model, optimizer, scaler, step, epoch, best, config,
         threshold: float = 0.5):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": unwrap(model).state_dict(),
        "config": vars(config),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step, "epoch": epoch, "best_f1": best,
        # Decoding must use the threshold the gate was measured at, or the
        # generated chart has a different note count than the score implies.
        "onset_threshold": threshold,
    }, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, default=Path("data/processed/shards"))
    ap.add_argument("--out", type=Path, default=Path("checkpoints/autoencoder"))
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--window-frames", type=int, default=WINDOW_FRAMES_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=16, help="per GPU")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--samples-per-epoch", type=int, default=20_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--middle-channels", type=int, default=64)
    ap.add_argument("--z-channels", type=int, default=16)
    ap.add_argument("--channel-mult", type=int, nargs="+", default=[1, 1, 2, 2, 4],
                    help="one entry per level; compression is 2^(len-1)")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--fp32", dest="fp16", action="store_false")
    ap.add_argument("--single-gpu", action="store_true")
    args = ap.parse_args()

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

    reader = ShardReader(args.shards)
    train_idx, val_idx = split_indices(reader, val_ratio=0.05)
    print_split_stats(reader, train_idx, "Train")
    print_split_stats(reader, val_idx, "Val")

    # The autoencoder only models chart syntax, so it learns from every map --
    # ranked filtering belongs to the diffusion stage, where taste matters.
    train_ds = WindowedDataset(
        reader, train_idx, window_frames=args.window_frames,
        random_window=True, augment=True, samples_per_epoch=args.samples_per_epoch,
    )
    val_ds = WindowedDataset(
        reader, val_idx, window_frames=args.window_frames,
        random_window=False, augment=False,
    )

    batch = args.batch_size * max(n_gpus, 1)
    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, **loader_kwargs)

    config = AutoencoderConfig(
        middle_channels=args.middle_channels,
        z_channels=args.z_channels,
        channel_mult=list(args.channel_mult),
    )
    model = ChartAutoencoder(config)
    model.check_window(args.window_frames)
    print(f"Autoencoder: {model.count_parameters()['total']} params, "
          f"{model.compression}x compression, "
          f"{args.window_frames} -> {args.window_frames // model.compression} latent frames")

    model = model.to(device)
    if n_gpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(n_gpus)))
        print(f"DataParallel across {n_gpus} GPUs, effective batch {batch}")

    optimizer = torch.optim.AdamW(unwrap(model).parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    step, start_epoch, best_f1, best_threshold = 0, 0, 0.0, 0.5
    if args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        unwrap(model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        step, start_epoch = ckpt["step"], ckpt["epoch"]
        best_f1 = ckpt.get("best_f1", 0.0)
        best_threshold = ckpt.get("onset_threshold", 0.5)
        print(f"Resumed from {args.resume}: step {step}, best F1 {best_f1:.4f}")

    total_steps = args.epochs * len(train_loader)
    print(f"\n{len(train_loader)} steps/epoch, {total_steps} total\n")

    t0 = time.time()
    model.train()
    for epoch in range(start_epoch, args.epochs):
        for batch_data in train_loader:
            lr = lr_at(step, args.warmup, total_steps, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

            chart = batch_data["chart"].to(device, non_blocking=True)
            mask = batch_data["valid_mask"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_fp16):
                loss, log = unwrap(model).training_loss(chart, mask)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unwrap(model).parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            step += 1

            if step % 50 == 0:
                print(f"epoch {epoch + 1:3d} step {step:7d}  "
                      f"loss {log['total_loss']:.4f}  "
                      f"don {log['recall_don']:.2f} kat {log['recall_kat']:.2f} "
                      f"big {log['recall_big_don']:.2f} roll {log['recall_roll']:.2f}  "
                      f"lr {lr:.2e}  {time.time() - t0:.0f}s")

            if step % args.val_every == 0:
                val_loss = validate(model, val_loader, device, args.val_batches)
                gate = onset_reconstruction_f1(
                    unwrap(model), val_loader, device, args.val_batches
                )
                status = "PASS" if gate["f1"] >= GATE_A_F1 else "not yet"
                print(
                    f"  val loss {val_loss:.4f} | "
                    f"onset F1 {gate['f1']:.4f} "
                    f"(P {gate['precision']:.3f} R {gate['recall']:.3f} "
                    f"@ threshold {gate['threshold']:.2f}) | "
                    f"Gate A {status}"
                )
                print("      sweep " + "  ".join(
                    f"{t:.2f}:{f:.3f}" for t, f in gate["sweep"].items()
                ))
                for name, (hits, spurious, misses) in gate["per_channel"].items():
                    if hits + misses:
                        print(f"      {name:<9s} recall {hits / max(hits + misses, 1):.3f}"
                              f"  ({hits} hit, {misses} missed, {spurious} spurious)")

                if gate["f1"] > best_f1:
                    best_f1 = gate["f1"]
                    best_threshold = gate["threshold"]
                    save(args.out / "best.pt", model, optimizer, scaler,
                         step, epoch, best_f1, config, best_threshold)
                    print(f"      new best -> {args.out / 'best.pt'}")

                save(args.out / "last.pt", model, optimizer, scaler,
                     step, epoch, best_f1, config, gate["threshold"])

    # ---- calibrate the latent scale on the final weights ------------------ #
    print("\nCalibrating latent scale ...")
    ckpt_path = args.out / "best.pt"
    if ckpt_path.exists():
        unwrap(model).load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=False)["model"]
        )
    calib_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)
    scale = unwrap(model).calibrate_scale(
        ({"chart": b["chart"]} for b in calib_loader), device=device, max_batches=64,
    )
    print(f"  latent scale {scale:.4f}  (latent std was {1 / scale:.4f})")
    save(ckpt_path, model, optimizer, scaler, step, args.epochs, best_f1,
         config, best_threshold)

    print(f"\n{'=' * 60}")
    print(f"Best onset F1: {best_f1:.4f} at threshold {best_threshold:.2f}   "
          f"Gate A ({GATE_A_F1}): "
          f"{'PASSED' if best_f1 >= GATE_A_F1 else 'FAILED'}")
    if best_f1 < GATE_A_F1:
        print("\nDo not start diffusion training on this checkpoint.")
        print(f"  Current compression is {unwrap(model).compression}x. Retry with one")
        print(f"  fewer entry in --channel-mult, e.g. "
              f"--channel-mult {' '.join(str(m) for m in args.channel_mult[:-1])}")
        return 1

    print(f"\nNext: python scripts/train_diffusion.py --ae {ckpt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
