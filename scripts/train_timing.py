"""
scripts/train_timing.py

Train TimingNet (taiko/timing/model.py) on the packed shards: beat and
downbeat probability per 20 ms frame, labelled by the ranked maps' red lines.

    python scripts/train_timing.py --shards data/processed/shards --out checkpoints/timing

Small and quick next to the chart model -- about 1.3M parameters -- so one
Kaggle session is plenty. Validation reports beat F-measure at +/-70 ms (the
MIREX tolerance) on held-out songs, split the same way as the chart model's
validation, so the two never share a song.

Once checkpoints/timing/best.pt exists, taiko.timing.detect_timing uses it
automatically; check it against the other sources with
scripts/benchmark_timing.py before trusting it.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from taiko.data.preprocessed_dataset import WindowedDataset, split_indices
from taiko.data.shards import MEL_IO_MODES, ShardReader
from taiko.timing.model import (
    TimingNet, TimingNetConfig, beat_targets, save_timingnet, timing_loss,
)


def peaks(x: np.ndarray, thresh: float = 0.4) -> np.ndarray:
    ok = (x[1:-1] >= x[:-2]) & (x[1:-1] > x[2:]) & (x[1:-1] > thresh)
    return np.flatnonzero(ok) + 1


def f_measure(pred: np.ndarray, ref: np.ndarray, tol_frames: float = 3.5) -> tuple[int, int, int]:
    """(true positives, predicted, reference) with one-to-one matching."""
    used = np.zeros(len(ref), dtype=bool)
    tp = 0
    for p in pred:
        if len(ref) == 0:
            break
        d = np.abs(ref - p).astype(np.float64)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol_frames:
            used[j] = True
            tp += 1
    return tp, len(pred), len(ref)


@torch.no_grad()
def validate(model, loader, device, max_batches: int) -> dict:
    model.eval()
    totals = {"beat": [0, 0, 0], "down": [0, 0, 0]}
    loss_sum, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        mel = batch["mel"].to(device)
        timing = batch["timing"].to(device)
        valid = batch["valid_mask"].to(device)
        logits = model(mel)
        targets = beat_targets(timing)
        loss_sum += float(timing_loss(logits, targets, valid))
        n += 1
        prob = torch.sigmoid(logits).cpu().numpy()
        tgt = (targets >= 1.0).cpu().numpy()
        vm = valid.cpu().numpy() > 0
        for b in range(prob.shape[0]):
            for k, name in ((0, "beat"), (1, "down")):
                ref = np.flatnonzero(tgt[b, k] & vm[b])
                pred = peaks(prob[b, k] * vm[b])
                for j, v in enumerate(f_measure(pred, ref)):
                    totals[name][j] += v
    model.train()

    def f(t):
        tp, p, r = t
        prec, rec = tp / max(p, 1), tp / max(r, 1)
        return 2 * prec * rec / max(prec + rec, 1e-9)

    return {"loss": loss_sum / max(n, 1), "beat_f": f(totals["beat"]), "down_f": f(totals["down"])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shards", type=Path, default=Path("data/processed/shards"))
    ap.add_argument("--out", type=Path, default=Path("checkpoints/timing"))
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--window-frames", type=int, default=1536)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--mel-io", default="read", choices=list(MEL_IO_MODES))
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-batches", type=int, default=30)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ranked-only", action="store_true")
    ap.add_argument("--max-hours", type=float, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    reader = ShardReader(args.shards, mel_io=args.mel_io)
    train_idx, val_idx = split_indices(reader, val_ratio=0.05, ranked_only=args.ranked_only)
    print(f"Train {len(train_idx)} maps, val {len(val_idx)} maps")

    # Motif dropout is irrelevant here; rate and frequency-mask augmentation
    # are what matter, and the targets follow the augmented timing stream.
    train_ds = WindowedDataset(reader, train_idx, window_frames=args.window_frames,
                               random_window=True, augment=True,
                               samples_per_epoch=args.steps * args.batch_size,
                               rate_p=0.5, rate_range=(0.85, 1.15))
    val_ds = WindowedDataset(reader, val_idx, window_frames=args.window_frames,
                             random_window=False, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, drop_last=True,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = TimingNet(TimingNetConfig()).to(device)
    print(f"TimingNet: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    step, best = 0, -1.0
    if args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        step, best = int(ckpt.get("step", 0)), float(ckpt.get("best_f", -1.0))
        print(f"Resumed at step {step}, best beat F {best:.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model.train()
    for batch in train_loader:
        if step >= args.steps:
            break
        if args.max_hours and (time.time() - t0) / 3600 > args.max_hours:
            print("Time budget reached")
            break

        lr = args.lr * min(1.0, (step + 1) / args.warmup) * \
            0.5 * (1 + math.cos(math.pi * min(step / args.steps, 1.0)))
        for g in opt.param_groups:
            g["lr"] = lr

        mel = batch["mel"].to(device, non_blocking=True)
        timing = batch["timing"].to(device, non_blocking=True)
        valid = batch["valid_mask"].to(device, non_blocking=True)
        loss = timing_loss(model(mel), beat_targets(timing), valid)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1

        if step % args.log_every == 0:
            print(f"step {step:6d}  loss {float(loss):.4f}  lr {lr:.2e}  "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)

        if step % args.val_every == 0 or step == args.steps:
            v = validate(model, val_loader, device, args.val_batches)
            marker = ""
            if v["beat_f"] > best:
                best = v["beat_f"]
                save_timingnet(model, args.out / "best.pt", step=step, best_f=best)
                marker = "  new best"
            save_timingnet(model, args.out / "last.pt", step=step, best_f=best,
                           optimizer=opt.state_dict())
            print(f"  val loss {v['loss']:.4f}  beat F {v['beat_f']:.4f}  "
                  f"downbeat F {v['down_f']:.4f}{marker}", flush=True)

    save_timingnet(model, args.out / "last.pt", step=step, best_f=best,
                   optimizer=opt.state_dict())
    print(f"Done at step {step}; best beat F {best:.4f} -> {args.out / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
