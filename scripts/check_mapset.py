"""
scripts/check_mapset.py

The beatmapset-level ranking checks (taiko/eval/mapset.py) on ranked sets,
and on the model's charts placed on their ranked map's own timing. As with
check_criteria.py, ranked sets go first: a check that fires on ranked sets
is a misreading until shown otherwise.

    python scripts/check_mapset.py
    python scripts/check_mapset.py --probs-cache outputs/eval_cache_v2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
from taiko.eval.criteria import enforce
from taiko.eval.mapset import Chart, chart_findings, fix_chart, set_findings


def share(title: str, found: list[Counter], unit: str) -> None:
    keys = Counter()
    for c in found:
        keys.update({k: 1 for k in c})
    clean = sum(not any(k.split(":")[0] in ("problem", "warning") for k in c) for c in found)
    print(f"\n== {title}: {len(found)} {unit}, {clean / max(len(found), 1):.1%} with no problem or warning")
    order = {"problem": 0, "warning": 1, "minor": 2}
    for k, n in sorted(keys.items(), key=lambda kv: (order[kv[0].split(":")[0]], -kv[1])):
        print(f"  {n / len(found):6.1%}  {k}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, default=Path("data/processed/shards"))
    ap.add_argument("--scan-cache", type=Path, default=DEFAULT_SCAN_CACHE)
    ap.add_argument("--probs-cache", type=Path, default=None)
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    reader = ShardReader(args.shards)
    osu_by_key: dict[str, list[Path]] = defaultdict(list)
    for p in map(Path, json.loads(args.scan_cache.read_text(encoding="utf-8"))):
        osu_by_key[safe_name(p.parent.name)].append(p)
    parser = OsuTaikoParser()

    by_set: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(reader.records):
        by_set[rec["mel_key"]].append(i)

    t0 = time.time()
    chart_found, set_found, ranked_chart = [], [], {}
    for n, (key, idxs) in enumerate(sorted(by_set.items())):
        if n % 500 == 0:
            print(f"  {n}/{len(by_set)} sets  {time.time() - t0:.0f}s", flush=True)
        parsed = {}
        for p in osu_by_key.get(key, []):
            try:
                bm = parser.parse_file(p)
            except Exception:                           # noqa: BLE001
                continue
            parsed.setdefault(bm.version, bm)
        charts = []
        for i in idxs:
            rec = reader.records[i]
            bm = parsed.get(rec.get("version"))
            if bm is None:
                continue
            chart = Chart.of(bm, sr=float(rec.get("difficulty", 0)))
            charts.append(chart)
            ranked_chart[i] = (chart, bm)
            chart_found.append(chart_findings(chart))
        if charts:
            set_found.append(set_findings(charts))
    share("ranked difficulties", chart_found, "difficulties")
    share("ranked beatmapsets", set_found, "sets")

    if args.probs_cache:
        if args.threshold is None:
            import torch
            from evaluate import load_model
            _, args.threshold, _ = load_model(Path("checkpoints/diffusion/best.pt"),
                                              Path("checkpoints/autoencoder/best.pt"),
                                              torch.device("cpu"), True)
        ai, same = [], []
        for f in sorted(args.probs_cache.glob("*.npy")):
            idx = int(f.stem.split("_")[1])
            if idx not in ranked_chart:
                continue
            ranked, bm = ranked_chart[idx]
            # The .osu's own timing, fractional offsets included: the packed
            # timing is whole ms, which alone puts a line 1 ms off.
            points = bm.timing_points
            out = decode_on_grid(np.load(f), points, threshold=args.threshold)
            repair(out, Grid(points))
            notes, _ = enforce(out.notes, Grid(points), ranked.level)
            notes, _ = fix_chart(notes, bm.timing_points)
            # On the ranked map's own timing, effects and fractional offsets included.
            ai.append(chart_findings(Chart(ranked.version, notes, bm.timing_points, ranked.sr,
                                           bm.slider_multiplier, ranked.level)))
            same.append(chart_findings(ranked))
        share(f"the model's charts ({args.probs_cache}, after enforce)", ai, "charts")
        share("the ranked difficulties of those songs", same, "difficulties")
    return 0


if __name__ == "__main__":
    sys.exit(main())
