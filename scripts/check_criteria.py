"""
scripts/check_criteria.py

How often ranked maps, and the model's charts, break the osu!taiko ranking
criteria for the difficulty their name asks for (taiko/eval/criteria.py).

Ranked maps go first: a ranked Kantan should almost never break a Kantan
*rule*, so a rule that fires often on ranked maps is a misreading (or needs
the wiki's BPM scaling), not a finding. Only then is the model measured.

    python scripts/check_criteria.py                                  # ranked corpus
    python scripts/check_criteria.py --probs-cache outputs/eval_cache_v2   # + the model
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from pack_dataset import DEFAULT_SCAN_CACHE, safe_name
from taiko.data.decode import decode_on_grid
from taiko.data.grid import Grid
from taiko.data.osu_parser import OsuTaikoParser
from taiko.data.repair import repair
from taiko.data.shards import ShardReader, decode_timing_points
from taiko.eval.criteria import LEVELS, check, level_of

BPM_BANDS = ((0, 150, "<150"), (150, 200, "150-200"), (200, 999, "200+"))


def table(title: str, fired: dict[str, list[Counter]]) -> None:
    """Per level: share of maps where each check fires at least once."""
    print(f"\n== {title}")
    for level in LEVELS:
        maps = fired.get(level, [])
        if not maps:
            continue
        keys = Counter()
        for c in maps:
            keys.update({k: 1 for k in c})
        clean = sum(not any(k.startswith("rule") for k in c) for c in maps)
        print(f"  {level}: {len(maps)} maps, {clean / len(maps):.1%} break no rule")
        for k, n in sorted(keys.items(), key=lambda kv: (not kv[0].startswith("rule"), -kv[1])):
            print(f"    {n / len(maps):6.1%}  {k}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, default=Path("data/processed/shards"))
    ap.add_argument("--scan-cache", type=Path, default=DEFAULT_SCAN_CACHE)
    ap.add_argument("--probs-cache", type=Path, default=None,
                    help="evaluate.py's sample cache: score the model's charts too")
    ap.add_argument("--threshold", type=float, default=None,
                    help="hit threshold for decoding cached samples (default: the autoencoder's)")
    ap.add_argument("--by-bpm", action="store_true", help="split the ranked table by BPM band")
    args = ap.parse_args()

    reader = ShardReader(args.shards)
    osu_by_key: dict[str, list[Path]] = defaultdict(list)
    for p in map(Path, json.loads(args.scan_cache.read_text(encoding="utf-8"))):
        osu_by_key[safe_name(p.parent.name)].append(p)
    parser = OsuTaikoParser()

    def ranked_notes(rec):
        for p in osu_by_key.get(rec["mel_key"], []):
            try:
                bm = parser.parse_file(p)
            except Exception:                           # noqa: BLE001
                continue
            if bm.version == rec.get("version"):
                return bm
        return None

    fired: dict[str, list[Counter]] = defaultdict(list)
    by_bpm: dict[str, dict[str, list[Counter]]] = defaultdict(lambda: defaultdict(list))
    ranked_by_idx: dict[int, Counter] = {}
    for n, rec in enumerate(reader.records):
        if n % 2000 == 0:
            print(f"  {n}/{len(reader.records)}", flush=True)
        level = level_of(rec.get("version", ""))
        if level is None:
            continue
        bm = ranked_notes(rec)
        if bm is None:
            continue
        c = check(bm.notes, Grid(bm.timing_points), level)
        fired[level].append(c)
        ranked_by_idx[n] = c
        bpm = float(rec.get("bpm") or 0)
        band = next(lbl for lo, hi, lbl in BPM_BANDS if lo <= bpm < hi)
        by_bpm[band][level].append(c)
    table("ranked maps, by the level their name asks for", fired)
    if args.by_bpm:
        for band, f in by_bpm.items():
            table(f"ranked maps at {band} BPM", f)

    if args.probs_cache:
        if args.threshold is None:
            import torch
            from evaluate import load_model
            _, args.threshold, _ = load_model(Path("checkpoints/diffusion/best.pt"),
                                              Path("checkpoints/autoencoder/best.pt"),
                                              torch.device("cpu"), True)
        ai, same_ranked = defaultdict(list), defaultdict(list)
        for f in sorted(args.probs_cache.glob("*.npy")):
            idx = int(f.stem.split("_")[1])
            rec = reader.records[idx]
            level = level_of(rec.get("version", ""))
            if level is None:
                continue
            points = decode_timing_points(rec["timing_points"])
            chart = decode_on_grid(np.load(f), points, threshold=args.threshold)
            repair(chart, Grid(points))
            ai[level].append(check(chart.notes, Grid(points), level))
            if idx in ranked_by_idx:
                same_ranked[level].append(ranked_by_idx[idx])
        table(f"the model's charts ({args.probs_cache}, every seed)", ai)
        table("the ranked maps of those same songs", same_ranked)
    return 0


if __name__ == "__main__":
    sys.exit(main())
