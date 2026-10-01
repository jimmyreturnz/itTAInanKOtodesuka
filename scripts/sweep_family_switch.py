"""
scripts/sweep_family_switch.py

The decoder's snap-family switch penalty (decode.FAMILY_SWITCH), on the
model's real cached samples. It was set to 0 on synthetic charts, where any
penalty cost 0.8% exact placement; blind round 3 then judged the model's
snaps as "following noise" -- 129 and 421 fast-snap changes where the ranked
maps had 10 and 24. For each penalty: snap changes per 100 hits against the
ranked maps', with exact snap and onset F1 against the mapper's own ms as the
guard that the notes still land on the right lines.

    python scripts/sweep_family_switch.py --probs-cache outputs/eval_cache_v2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from taiko.data.decode import decode_on_grid
from taiko.data.grid import Grid
from taiko.data.repair import repair
from taiko.data.shards import ShardReader, decode_timing_points, load_note_times
from taiko.eval.metrics import exact_snap, onset_f1, snap_switch_rate
from taiko.data.osu_parser import TaikoNote

BANDS = ((0, 4, "<4*"), (4, 5.5, "4-5.5*"), (5.5, 99, "5.5*+"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, default=Path("data/processed/shards"))
    ap.add_argument("--probs-cache", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--penalties", type=float, nargs="+", default=[0, 0.5, 1, 2, 3, 5, 8])
    args = ap.parse_args()
    if args.threshold is None:
        import torch
        from evaluate import load_model
        _, args.threshold, _ = load_model(Path("checkpoints/diffusion/best.pt"),
                                          Path("checkpoints/autoencoder/best.pt"),
                                          torch.device("cpu"), True)
    reader = ShardReader(args.shards)
    real = load_note_times(reader)
    samples = []
    for f in sorted(args.probs_cache.glob("*.npy")):
        idx = int(f.stem.split("_")[1])
        rec = reader.records[idx]
        points = decode_timing_points(rec["timing_points"])
        frames = np.load(f).shape[-1]
        ref_ms = real[idx][real[idx] < frames * 20]
        band = next(lbl for lo, hi, lbl in BANDS if lo <= rec["difficulty"] < hi)
        samples.append((band, np.load(f), points, ref_ms))

    ranked = {}
    for band, _, points, ref_ms in samples:
        ranked.setdefault(band, []).append(snap_switch_rate(
            [TaikoNote(time=int(t), note_type="don", end_time=int(t)) for t in ref_ms], Grid(points)))
    print("snap changes per 100 hits (ranked: " + ", ".join(
        f"{b} {np.mean(v):.2f}" for b, v in ranked.items()) + ")")
    print(f"{'penalty':>8s}" + "".join(f"{b + ' sw':>13s}" for _, _, b in BANDS)
          + f"{'exact snap':>12s}{'onset F1':>10s}")
    for pen in args.penalties:
        sw, ex, f1 = {}, [], []
        for band, probs, points, ref_ms in samples:
            grid = Grid(points)
            bm = decode_on_grid(probs, points, threshold=args.threshold, family_switch=pen)
            repair(bm, grid)
            hits = [n for n in bm.notes if not n.is_long]
            sw.setdefault(band, []).append(snap_switch_rate(hits, grid))
            e = exact_snap([n.time for n in hits], ref_ms, grid)
            if e.n_matched:
                ex.append(e.exact)
            ref_notes = [TaikoNote(time=int(t), note_type="don", end_time=int(t)) for t in ref_ms]
            f1.append(onset_f1(hits, ref_notes).f1)
        print(f"{pen:>8g}" + "".join(f"{np.mean(sw.get(b, [np.nan])):>13.2f}" for _, _, b in BANDS)
              + f"{np.mean(ex):>12.4f}{np.mean(f1):>10.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
