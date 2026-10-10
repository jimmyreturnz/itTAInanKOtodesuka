"""
scripts/probe_rhythm.py

Rhythm-head probe (RETRAIN_PLAN, revised training plan step 1): does the
trained U-Net already know which snap each note sits on?

A fresh rhythm head is trained on the *frozen* checkpoint's last decoder
features to predict, per latent frame, which snaps (1/1 1/2 1/3 1/4 1/6 1/8)
its notes sit on. Labels are the .osu's own ms (note_times.npz divisors).
Then, on held-out windows at fixed noise levels, it reports average precision
per snap against that snap's base rate (what a head that knew nothing would
score). AP far above base rate at low noise means the snap is in the features;
near base rate means the U-Net does not represent it, and the warm start's
auxiliary loss (--rhythm-weight) has to teach it.

    python scripts/probe_rhythm.py --steps 3000              # Kaggle, ~1.5 GPU-h
    python scripts/probe_rhythm.py --steps 20 --val-batches 2 # smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from taiko.data.conditioning import DIVISORS
from taiko.data.preprocessed_dataset import WindowedDataset, split_indices
from taiko.data.shards import ShardReader
from taiko.model.diffusion import load_diffusion

T_LEVELS = (50, 250, 500, 750)
KEYS = ("chart", "mel", "timing", "difficulty", "style", "valid_mask", "avg_nps",
        "peak_nps", "motif", "motif_mask", "window_nps", "profile", "snap_labels")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diffusion", type=Path, default=Path("checkpoints/diffusion/best.pt"))
    ap.add_argument("--ae", type=Path, default=Path("checkpoints/autoencoder/best.pt"))
    ap.add_argument("--shards", type=Path, default=Path("data/processed/shards"))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-batches", type=int, default=40)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("outputs/probe_rhythm.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, ckpt = load_diffusion(args.diffusion, args.ae, device, verbose=True)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    model.cfg_dropout = 0.0                 # always the real condition
    model.rhythm_weight = 1.0
    head = model.unet_model.add_rhythm_head().to(device)
    head.train()

    reader = ShardReader(args.shards)
    train_idx, val_idx = split_indices(reader)
    window = int(ckpt.get("window_frames", 1536))
    common = dict(window_frames=window, rhythm_labels=True, map_profile=model.features >= 3)
    train = DataLoader(WindowedDataset(reader, train_idx, augment=False,
                                       samples_per_epoch=args.steps * args.batch, **common),
                       batch_size=args.batch, num_workers=args.workers)
    val = DataLoader(WindowedDataset(reader, val_idx, augment=False, seed=1,
                                     samples_per_epoch=args.val_batches * args.batch, **common),
                     batch_size=args.batch, num_workers=args.workers)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    t0 = time.time()
    for step, batch in enumerate(train, 1):
        batch = {k: batch[k].to(device) for k in KEYS}
        # The diffusion term is constant here (everything else is frozen), so
        # the total's gradient is the rhythm loss's, and it reaches only the head.
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            total, metrics = model(**batch)
        bce = metrics[4]
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
        if step % 100 == 0 or step == 1:
            print(f"step {step:5d}  rhythm bce {float(bce):.4f}  {(time.time() - t0) / 60:.1f} min",
                  flush=True)

    head.eval()
    scores = {t: [] for t in T_LEVELS}
    labels = {t: [] for t in T_LEVELS}
    with torch.no_grad():
        for batch in val:
            batch = {k: batch[k].to(device) for k in KEYS}
            B = batch["chart"].shape[0]
            for t in T_LEVELS:
                logits = rhythm_logits(model, batch, torch.full((B,), t, device=device))
                lab = F.adaptive_max_pool1d(batch["snap_labels"], logits.shape[-1])
                keep = F.adaptive_max_pool1d(batch["valid_mask"][:, None], logits.shape[-1])[:, 0] > 0
                scores[t].append(logits.permute(0, 2, 1)[keep].float().cpu().numpy())
                labels[t].append(lab.permute(0, 2, 1)[keep].cpu().numpy())

    report = {"checkpoint": str(args.diffusion), "step": ckpt.get("step"),
              "train_steps": args.steps, "divisors": [f"1/{d}" for d in DIVISORS], "by_t": {}}
    y_all = np.concatenate(labels[T_LEVELS[0]])
    base = y_all.mean(axis=0)
    print(f"\n{'':>8}" + "".join(f"{f'1/{d}':>14}" for d in DIVISORS))
    print(f"{'base':>8}" + "".join(f"{b:14.3f}" for b in base))
    for t in T_LEVELS:
        y, s = np.concatenate(labels[t]), np.concatenate(scores[t])
        aps = [float(average_precision_score(y[:, k], s[:, k])) if y[:, k].any() else float("nan")
               for k in range(len(DIVISORS))]
        report["by_t"][t] = aps
        print(f"{f't={t}':>8}" + "".join(f"{a:8.3f} ({a / max(b, 1e-9):4.1f}x)"
                                        for a, b in zip(aps, base)))
    report["base_rate"] = base.tolist()
    report["positives"] = y_all.sum(axis=0).astype(int).tolist()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nAP per snap (x base rate). Wrote {args.out}")
    return 0


def rhythm_logits(model, batch: dict, t: torch.Tensor) -> torch.Tensor:
    """The head's logits at a fixed noise level, via the training forward."""
    captured = {}
    hook = model.unet_model.rhythm_head.register_forward_hook(
        lambda _m, _i, out: captured.setdefault("logits", out))
    try:
        model(**batch, t=t)
    finally:
        hook.remove()
    return captured["logits"]


if __name__ == "__main__":
    raise SystemExit(main())
