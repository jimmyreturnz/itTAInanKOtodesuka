"""
scripts/measure_repetition.py

Does a chart map a phrase the same way when the music repeats it? Ranked
mappers reuse their patterns; a model that sees one 30.7 s window at a time
has no way to. For each bar (its red line's meter), the pattern is the
positions of its hits in 1/12 beats plus their colour and size; the score is
the share of bars whose exact pattern occurs in another bar of the same
chart. "Rhythm only" drops colour and size.

    python scripts/measure_repetition.py --probs-cache outputs/eval_cache_v2

2026-10-02, eval pool, best.pt step 58968 (ranked / model):
  rhythm+colour  <2* 0.44/0.23  2-4* 0.37/0.04  4-5.5* 0.18/0.01  5.5-7* 0.19/0.00  7*+ 0.28/0.01
  rhythm only    <2* 0.82/0.47  2-4* 0.63/0.23  4-5.5* 0.44/0.09  5.5-7* 0.47/0.06  7*+ 0.54/0.08
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from taiko.data.decode import decode_on_grid
from taiko.data.grid import Grid
from taiko.data.repair import repair
from taiko.data.shards import ShardReader, decode_timing_points

BANDS = ((0, 2, "<2*"), (2, 4, "2-4*"), (4, 5.5, "4-5.5*"), (5.5, 7, "5.5-7*"), (7, 99, "7*+"))


def bar_patterns(notes, grid: Grid) -> list[tuple]:
    bars = defaultdict(list)
    for n in sorted(notes, key=lambda n: n.time):
        if n.is_long:
            continue
        s = grid.section_at(float(n.time))
        bar = s.ms_per_beat * s.meter
        k = int((n.time - s.offset_ms) // bar)
        pos = round((n.time - s.offset_ms - k * bar) / s.ms_per_beat * 12)
        bars[(s.offset_ms, k)].append((pos, "k" if "kat" in n.note_type else "d",
                                       n.note_type.startswith("big")))
    return [tuple(v) for v in bars.values()]


def repeat_share(bars: list[tuple]) -> float:
    counts = Counter(bars)
    return sum(counts[b] > 1 for b in bars) / len(bars) if bars else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, default=Path("data/processed/shards"))
    ap.add_argument("--probs-cache", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()
    if args.threshold is None:
        import torch
        from evaluate import load_model
        _, args.threshold, _ = load_model(Path("checkpoints/diffusion/best.pt"),
                                          Path("checkpoints/autoencoder/best.pt"),
                                          torch.device("cpu"), True)

    reader = ShardReader(args.shards)
    res = defaultdict(lambda: defaultdict(list))
    seen = set()
    for f in sorted(args.probs_cache.glob("*.npy")):
        idx = int(f.stem.split("_")[1])
        rec = reader.records[idx]
        points = decode_timing_points(rec["timing_points"])
        grid = Grid(points)
        band = next(lbl for lo, hi, lbl in BANDS if lo <= rec["difficulty"] < hi)
        sides = [("model", np.load(f), args.threshold)]
        if idx not in seen:      # the ranked chart once per map, through the same decoder
            seen.add(idx)
            sides.append(("ranked", reader.chart_window(idx, 0, reader.chart_length(idx)), 0.5))
        for side, chart, thr in sides:
            bm = decode_on_grid(chart, points, threshold=thr)
            if side == "model":
                repair(bm, grid)
            bars = bar_patterns(bm.notes, grid)
            res[band][f"{side} full"].append(repeat_share(bars))
            res[band][f"{side} rhythm"].append(repeat_share([tuple(p for p, _, _ in b) for b in bars]))

    print("share of bars whose exact pattern recurs elsewhere in the chart")
    print(f"{'SR':<8s}{'rhythm+colour  ranked':>22s}{'model':>7s}{'rhythm only  ranked':>22s}{'model':>7s}")
    for _, _, band in BANDS:
        if band in res:
            m = {k: float(np.nanmean(v)) for k, v in res[band].items()}
            print(f"{band:<8s}{m['ranked full']:>22.2f}{m['model full']:>7.2f}"
                  f"{m['ranked rhythm']:>22.2f}{m['model rhythm']:>7.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
