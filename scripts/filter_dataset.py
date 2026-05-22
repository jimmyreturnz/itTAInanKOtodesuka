"""
scripts/filter_dataset.py

Pre-filters taiko_files_cache.json and writes a clean
taiko_files_filtered.json ready for training.

Filters out:
  - Files over 500KB (marathons, gimmick maps)
  - Files under 5KB (empty/broken)
  - Files with fewer than 50 notes
  - Files with no valid timing points
  - Files that fail to parse

Usage:
    python scripts/filter_dataset.py
    python scripts/filter_dataset.py --cache taiko_files_cache.json --output taiko_files_filtered.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from taiko.data.osu_parser import OsuTaikoParser


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache",      default="taiko_files_cache.json")
    parser.add_argument("--output",     default="taiko_files_filtered.json")
    parser.add_argument("--max-kb",     type=int, default=500)
    parser.add_argument("--min-kb",     type=int, default=5)
    parser.add_argument("--min-notes",  type=int, default=50)
    parser.add_argument("--max-notes",  type=int, default=4000)
    args = parser.parse_args()

    cache_path = Path(args.cache)
    if not cache_path.exists():
        print(f"ERROR: {cache_path} not found. Run fast_scan.py first.")
        return

    files = [Path(p) for p in json.loads(cache_path.read_text())]
    print(f"Loaded {len(files)} files from cache.")

    osu_parser = OsuTaikoParser()
    kept, skipped = [], []
    reasons = {
        "too_large":       0,
        "too_small":       0,
        "parse_error":     0,
        "too_few_notes":   0,
        "too_many_notes":  0,
        "no_timing":       0,
        "too_many_timing_points":   0,
        "insane_bpm":               0,
}
    

    for i, f in enumerate(files):
        if i % 500 == 0:
            print(f"  [{i}/{len(files)}] kept={len(kept)} skipped={len(skipped)}...")

        # File size check
        try:
            size_kb = os.path.getsize(f) / 1024
        except OSError:
            reasons["parse_error"] += 1
            skipped.append(str(f))
            continue

        if size_kb > args.max_kb:
            reasons["too_large"] += 1
            skipped.append(str(f))
            continue

        if size_kb < args.min_kb:
            reasons["too_small"] += 1
            skipped.append(str(f))
            continue

        # Parse check
        try:
            bm = osu_parser.parse_file(f)
        except Exception:
            reasons["parse_error"] += 1
            skipped.append(str(f))
            continue

        if bm.note_count < args.min_notes:
            reasons["too_few_notes"] += 1
            skipped.append(str(f))
            continue

        if bm.note_count > args.max_notes:
            reasons["too_many_notes"] += 1
            skipped.append(str(f))
            continue

        if not any(tp.uninherited for tp in bm.timing_points):
            reasons["no_timing"] += 1
            skipped.append(str(f))
            continue

        # Skip SV gimmick maps with insane timing point counts
        if len(bm.timing_points) > 1000:
            reasons["too_many_timing_points"] += 1
            skipped.append(str(f))
            continue

        # Skip maps with insane BPM (< 1ms half-beat)
        for tp in bm.timing_points:
            if tp.uninherited and tp.beat_length > 0:
                bpm = 60000 / tp.beat_length
                if bpm > 10000:
                    reasons["insane_bpm"] = reasons.get("insane_bpm", 0) + 1
                    skipped.append(str(f))
                    break
        else:
            kept.append(str(f))
            continue
        # if we broke out of the for loop, file was skipped

    # Save filtered list
    output_path = Path(args.output)
    output_path.write_text(json.dumps(kept, indent=2), encoding="utf-8")

    print(f"\n{'='*50}")
    print(f"Results:")
    print(f"  Original:  {len(files)}")
    print(f"  Kept:      {len(kept)}")
    print(f"  Skipped:   {len(skipped)}")
    print(f"\nSkip reasons:")
    for reason, count in reasons.items():
        if count > 0:
            print(f"  {reason}: {count}")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
