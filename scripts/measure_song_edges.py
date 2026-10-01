"""
scripts/measure_song_edges.py

Where a generated chart starts and ends, against its ranked map, on whole
held-out songs. The MultiDiffusion taper gave the song's first and last latent
frame weight exactly 0 from the only window covering it, so those frames were
never denoised; this counts the notes they produce and how far the first and
last notes sit from the ranked map's.

    python scripts/measure_song_edges.py --n-maps 10            # current sampler
    python scripts/measure_song_edges.py --n-maps 10 --old-blend  # the taper before the fix

Same seeds either way, so the two runs differ only in the blend.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

import taiko.model.sampling as sampling
from evaluate import dominant_tempo, load_model
from taiko.data.conditioning import (STYLE_NULL, normalise_avg_nps, normalise_difficulty,
                                     normalise_peak_nps)
from taiko.data.decode import decode_on_grid
from taiko.data.frames import FRAME_MS
from taiko.data.grid import Grid
from taiko.data.preprocessed_dataset import WINDOW_FRAMES_DEFAULT, split_indices
from taiko.data.repair import repair
from taiko.data.shards import ShardReader, decode_timing_points
from taiko.data.tensor_repr import build_timing_stream, tensor_to_beatmap


def _old_blend_weights(length: int, ramp: int, device, dtype) -> torch.Tensor:
    w = torch.ones(length, device=device, dtype=dtype)
    if ramp > 0:
        t = torch.linspace(0, 1, ramp, device=device, dtype=dtype)
        taper = 0.5 * (1 - torch.cos(torch.pi * t))
        w[:ramp] = taper
        w[-ramp:] = taper.flip(0)
    return w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diffusion", type=Path, default=Path("checkpoints/diffusion/best.pt"))
    ap.add_argument("--ae", type=Path, default=Path("checkpoints/autoencoder/best.pt"))
    ap.add_argument("--shards", type=Path, default=Path("data/processed/shards"))
    ap.add_argument("--n-maps", type=int, default=10)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg-scale", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--old-blend", action="store_true")
    args = ap.parse_args()

    if args.old_blend:
        sampling._blend_weights = _old_blend_weights

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, threshold, ckpt = load_model(args.diffusion, args.ae, device, True)
    window = ckpt.get("window_frames", WINDOW_FRAMES_DEFAULT)
    edge_ms = model.compression * FRAME_MS   # one latent frame

    reader = ShardReader(args.shards)
    _, val_idx = split_indices(reader, val_ratio=0.05)
    chosen = np.random.default_rng(args.seed).permutation(val_idx)[:args.n_maps]

    print(f"{'old' if args.old_blend else 'new'} blend, edge = first/last {edge_ms:.0f} ms")
    print(f"{'map':44s} {'AI 1st':>7s} {'ref 1st':>7s} {'AI last':>8s} {'ref last':>8s}"
          f" {'head':>4s} {'tail':>4s}")
    heads, tails, d_first, d_last = [], [], [], []
    for idx in map(int, chosen):
        record = reader.records[idx]
        frames = min(reader.chart_length(idx), reader.mel_length(idx))
        if frames < window:
            continue
        song_ms = frames * FRAME_MS
        points = decode_timing_points(record["timing_points"])
        mel = torch.from_numpy(reader.mel_window(idx, 0, frames)).unsqueeze(0)
        timing = torch.from_numpy(build_timing_stream(points, frames, start_frame=0)).unsqueeze(0)
        nps = float(record.get("avg_nps", 0.0))
        probs = sampling.generate_song(
            model, mel=mel, timing=timing,
            difficulty=normalise_difficulty(float(record.get("difficulty", 5.0))),
            style=int(record.get("style", STYLE_NULL)),
            avg_nps=normalise_avg_nps(nps) if nps else None,
            peak_nps=normalise_peak_nps(float(record.get("peak_nps", 0.0))) or None,
            window_frames=window, overlap_frames=window // 2,
            ddim_steps=args.steps, cfg_scale=args.cfg_scale, progress=False,
            generator=torch.Generator(device=device).manual_seed(args.seed + idx),
        )[0].cpu().numpy()

        chart = decode_on_grid(probs, points, threshold=threshold)
        repair(chart, Grid(points))
        ai = sorted(n.time for n in chart.notes)
        bpm, offset, meter = dominant_tempo(points)
        ref = sorted(n.time for n in tensor_to_beatmap(
            reader.chart_window(idx, 0, frames), bpm=bpm, offset_ms=offset,
            meter=meter, timing_points=points).notes)
        if not ai or not ref:
            continue
        head = sum(t < edge_ms for t in ai)
        tail = sum(t >= song_ms - edge_ms for t in ai)
        heads.append(head); tails.append(tail)
        d_first.append(ai[0] - ref[0]); d_last.append(ai[-1] - ref[-1])
        name = f"{record.get('title', '?')} [{record.get('version', '?')}]"[:44]
        print(f"{name:44s} {ai[0]:7d} {ref[0]:7d} {ai[-1]:8d} {ref[-1]:8d} {head:4d} {tail:4d}")

    if not heads:
        print("no maps")
        return 1
    print(f"\n{len(heads)} maps: notes in first latent frame {sum(heads)}, "
          f"in last {sum(tails)}; maps with any {sum(h > 0 for h in heads)}/{len(heads)} head, "
          f"{sum(t > 0 for t in tails)}/{len(tails)} tail")
    print(f"first note minus ranked: median {statistics.median(d_first):+.0f} ms; "
          f"last note minus ranked: median {statistics.median(d_last):+.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
