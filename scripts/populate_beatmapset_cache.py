"""
scripts/populate_beatmapset_cache.py

Fill beatmapset_cache.json for all beatmapsets in taiko_files_filtered.json.
Ranked status is per beatmapset — every difficulty in a ranked set gets ranked=True
when preprocess reads the cache.

Run this before preprocess_for_colab.py --no-api if ranked counts look too low.

Usage:
    python scripts/populate_beatmapset_cache.py
    python scripts/populate_beatmapset_cache.py --limit 100   # test run
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from dotenv import load_dotenv

from scripts.preprocess_for_colab import (
    CACHE_FILE,
    BEATMAPSET_CACHE,
    API_DELAY,
    OSU_API_KEY,
    load_beatmapset_cache,
    save_beatmapset_cache,
    fetch_beatmapset,
)

load_dotenv(Path(__file__).parent.parent / ".env")


def read_beatmapset_id(osu_path: Path) -> int:
    """Read BeatmapSetID from [General] without full parse."""
    try:
        with open(osu_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("BeatmapSetID:"):
                    v = line.split(":", 1)[1].strip()
                    return int(v) if v.isdigit() else 0
    except OSError:
        pass
    return 0


def collect_set_ids(files: list[Path]) -> set[int]:
    ids: set[int] = set()
    for i, fp in enumerate(files):
        if i and i % 2000 == 0:
            print(f"  scanned {i}/{len(files)} paths...")
        sid = read_beatmapset_id(fp)
        if sid > 0:
            ids.add(sid)
    return ids


def main(limit: int | None = None):
    if not OSU_API_KEY:
        print("ERROR: OSU_API_KEY not set in .env")
        return

    if not Path(CACHE_FILE).exists():
        print(f"ERROR: {CACHE_FILE} not found")
        return

    files = [Path(p) for p in json.loads(Path(CACHE_FILE).read_text(encoding="utf-8"))]
    print(f"Scanning {len(files)} .osu paths for beatmapset IDs...")
    set_ids = sorted(collect_set_ids(files))
    print(f"Unique beatmapsets: {len(set_ids)}")

    cache   = load_beatmapset_cache()
    missing = [sid for sid in set_ids if str(sid) not in cache]
    print(f"Already in cache: {len(set_ids) - len(missing)}")
    print(f"Missing (will fetch): {len(missing)}")

    if limit is not None:
        missing = missing[:limit]
        print(f"Limited to first {len(missing)} fetches")

    if not missing:
        print("Cache is complete for filtered files.")
        return

    session = requests.Session()
    t0      = time.time()
    fetched = 0

    for i, sid in enumerate(missing):
        fetch_beatmapset(sid, cache, session)
        fetched += 1

        if fetched % 50 == 0:
            save_beatmapset_cache(cache)
            elapsed = time.time() - t0
            eta = elapsed / fetched * (len(missing) - fetched)
            print(f"  [{fetched}/{len(missing)}] "
                  f"{elapsed/60:.1f}min elapsed, ETA {eta/60:.1f}min")

        time.sleep(API_DELAY)

    save_beatmapset_cache(cache)

    ranked_sets = sum(
        1 for sid in set_ids
        if isinstance(cache.get(str(sid)), dict) and cache[str(sid)].get("ranked")
    )
    print(f"\nDone. Fetched {fetched} beatmapsets in {(time.time()-t0)/60:.1f} min")
    print(f"Cache size: {len(cache)} entries")
    print(f"Ranked sets (in filtered files): {ranked_sets} / {len(set_ids)}")
    print("Re-run: python scripts/preprocess_for_colab.py --no-api")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Only fetch this many missing sets (for testing)")
    args = ap.parse_args()
    main(limit=args.limit)
