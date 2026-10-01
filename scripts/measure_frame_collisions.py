"""
scripts/measure_frame_collisions.py

What the 20 ms chart frame costs a perfect model. Each ranked map is encoded
exactly as training sees it (beatmap_to_tensor) and decoded straight back
through the shipped grid decoder, then every hit is checked against the
mapper's own millisecond:

  exact   a decoded hit within EXACT_MS -- the right line
  moved   nearest decoded hit within MATCH_MS but not exact -- a wrong line,
          what the frame cannot tell apart (1/4 vs 1/6 vs 1/8 at speed)
  lost    nothing within MATCH_MS -- collapsed into a neighbour

Also counted straight from the .osu: hits sharing a frame with the previous
one, and hits one frame after it (the peak finder keeps one of two equal
neighbouring frames, so those collapse too).

Grouped by BPM band and SR band. If moved + lost is material where the hard
maps live, a sub-frame offset channel (autoencoder retrain) is worth it.

    python scripts/measure_frame_collisions.py              # every ranked map
    python scripts/measure_frame_collisions.py --limit 500  # quick look
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from pack_dataset import DEFAULT_SCAN_CACHE, safe_name
from taiko.data.decode import decode_on_grid
from taiko.data.frames import ms_to_frame
from taiko.data.osu_parser import OsuTaikoParser
from taiko.data.shards import ShardReader
from taiko.data.tensor_repr import beatmap_to_tensor

EXACT_MS = 2
MATCH_MS = 25
BPM_BANDS = (0, 150, 180, 210, 240, 999)
SR_BANDS = (0, 2, 4, 5.5, 7, 99)


def band(value: float, edges) -> str:
    for lo, hi in zip(edges, edges[1:]):
        if lo <= value < hi:
            return f"{lo:g}-{hi:g}" if hi != edges[-1] else f"{lo:g}+"
    return "?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, default=Path("data/processed/shards"))
    ap.add_argument("--scan-cache", type=Path, default=DEFAULT_SCAN_CACHE)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    reader = ShardReader(args.shards)
    osu_by_key: dict[str, list[Path]] = defaultdict(list)
    for p in map(Path, json.loads(args.scan_cache.read_text(encoding="utf-8"))):
        osu_by_key[safe_name(p.parent.name)].append(p)

    records = list(reader.records)
    if args.limit:
        records = [records[i] for i in np.random.default_rng(0).permutation(len(records))[:args.limit]]

    parser = OsuTaikoParser()
    # tallies[group][band] = [hits, exact, moved, lost, same_frame, next_frame]
    tallies = {"bpm": defaultdict(lambda: np.zeros(6, int)),
               "sr": defaultdict(lambda: np.zeros(6, int))}
    measured = 0
    for n, rec in enumerate(records):
        if n % 500 == 0:
            print(f"  {n}/{len(records)} maps", flush=True)
        bm = None
        for p in osu_by_key.get(rec["mel_key"], []):
            try:
                cand = parser.parse_file(p)
            except Exception:                           # noqa: BLE001
                continue
            if cand.version == rec.get("version"):
                bm = cand
                break
        if bm is None:
            continue
        hits = np.array(sorted(x.time for x in bm.notes if not x.is_long), dtype=np.int64)
        if hits.size < 2:
            continue

        chart = beatmap_to_tensor(bm)
        got = np.array(sorted(x.time for x in decode_on_grid(chart, bm.timing_points,
                                                              threshold=0.5).notes
                              if not x.is_long), dtype=np.int64)
        if got.size:
            pos = np.searchsorted(got, hits)
            after = got[np.minimum(pos, got.size - 1)]
            before = got[np.maximum(pos - 1, 0)]
            near = np.minimum(np.abs(after - hits), np.abs(before - hits))
        else:
            near = np.full(hits.size, 10 ** 9)
        exact = int((near <= EXACT_MS).sum())
        moved = int(((near > EXACT_MS) & (near <= MATCH_MS)).sum())
        lost = int((near > MATCH_MS).sum())

        frames = np.array([ms_to_frame(t) for t in hits])
        step = np.diff(frames)
        row = np.array([hits.size, exact, moved, lost, int((step == 0).sum()), int((step == 1).sum())])
        tallies["bpm"][band(float(rec.get("bpm") or 0), BPM_BANDS)] += row
        tallies["sr"][band(float(rec.get("difficulty", 0)), SR_BANDS)] += row
        measured += 1

    print(f"\n{measured} ranked maps, every hit against the mapper's own ms")
    for group, title in (("bpm", "BPM band"), ("sr", "SR band")):
        print(f"\n{title:9s} {'hits':>8s} {'exact':>7s} {'moved':>7s} {'lost':>7s}"
              f" {'same fr':>8s} {'next fr':>8s}")
        for key in sorted(tallies[group], key=lambda k: float(k.split('-')[0].rstrip('+'))):
            h, e, m, l, s, x = tallies[group][key]
            print(f"{key:9s} {h:8d} {e / h:7.2%} {m / h:7.2%} {l / h:7.2%} {s / h:8.2%} {x / h:8.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
