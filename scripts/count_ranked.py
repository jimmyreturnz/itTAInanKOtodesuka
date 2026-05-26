"""
scripts/count_ranked.py

Print ranked vs unranked counts from beatmapset_cache, colab_index, and filtered .osu list.

Usage:
    python scripts/count_ranked.py
"""

from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

CACHE_FILE = "taiko_files_filtered.json"
BEATMAPSET_CACHE = Path("data/processed/beatmapset_cache.json")
INDEX_PATH = Path("data/processed/colab_index.jsonl")


def read_beatmapset_id(osu_path: Path) -> int:
    try:
        with open(osu_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("BeatmapSetID:"):
                    v = line.split(":", 1)[1].strip()
                    return int(v) if v.isdigit() else 0
    except OSError:
        pass
    return 0


def main():
    # --- beatmapset_cache ---
    if BEATMAPSET_CACHE.exists():
        cache = json.loads(BEATMAPSET_CACHE.read_text(encoding="utf-8"))
        sets_ranked = sets_unranked = sets_null = 0
        diffs_ranked = diffs_unranked = 0
        for v in cache.values():
            if not isinstance(v, dict):
                sets_null += 1
                continue
            n = len(v.get("difficulties") or {})
            if v.get("ranked"):
                sets_ranked += 1
                diffs_ranked += n
            else:
                sets_unranked += 1
                diffs_unranked += n

        print("=== beatmapset_cache.json (per beatmapset) ===")
        print(f"Beatmapsets total     : {len(cache)}")
        print(f"  ranked sets         : {sets_ranked}")
        print(f"  unranked sets       : {sets_unranked}")
        print(f"  null / API failed   : {sets_null}")
        print(f"Taiko diffs in cache  : {diffs_ranked} ranked, {diffs_unranked} unranked")
    else:
        print("beatmapset_cache.json not found")
        cache = {}

    # --- colab_index ---
    if INDEX_PATH.exists():
        recs = [
            json.loads(l)
            for l in INDEX_PATH.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        r = sum(1 for x in recs if x.get("ranked"))
        print()
        print("=== colab_index.jsonl (per .osu in training index) ===")
        print(f"Records total         : {len(recs)}")
        print(f"  ranked              : {r}")
        print(f"  unranked            : {len(recs) - r}")
        print("  approved_str:", dict(Counter(x.get("approved_str", "?") for x in recs).most_common(8)))
        if cache and r < 4000:
            print("  (index may be stale — re-run preprocess_for_colab.py after filling cache)")

    # --- filtered .osu + cache ---
    filt = Path(CACHE_FILE)
    if filt.exists() and cache:
        files = json.loads(filt.read_text(encoding="utf-8"))
        ranked_d = unranked_d = unknown_d = 0
        ranked_sets: set[int] = set()
        for fp in files:
            sid = read_beatmapset_id(Path(fp))
            if sid <= 0:
                unknown_d += 1
                continue
            entry = cache.get(str(sid))
            if not isinstance(entry, dict):
                unknown_d += 1
                continue
            if entry.get("ranked"):
                ranked_d += 1
                ranked_sets.add(sid)
            else:
                unranked_d += 1

        print()
        print(f"=== {CACHE_FILE} + cache (per .osu file) ===")
        print(f".osu files total       : {len(files)}")
        print(f"  ranked                : {ranked_d}")
        print(f"  unranked              : {unranked_d}")
        print(f"  unknown (no cache)    : {unknown_d}")
        print(f"Unique ranked sets      : {len(ranked_sets)}")


if __name__ == "__main__":
    main()
