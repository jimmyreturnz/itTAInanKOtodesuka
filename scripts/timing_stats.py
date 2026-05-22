"""
scripts/timing_stats.py

Analyze timing point distribution across all filtered taiko maps.
Shows red line (BPM) and green line (SV) statistics.

Usage:
    python scripts/timing_stats.py
"""

import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from taiko.data.osu_parser import OsuTaikoParser


def main():
    cache = Path("taiko_files_filtered.json")
    if not cache.exists():
        cache = Path("taiko_files_cache.json")
    
    files = [Path(p) for p in json.loads(cache.read_text())]
    print(f"Analyzing {len(files)} maps...\n")

    parser = OsuTaikoParser()

    red_counts  = []
    green_counts = []
    total_counts = []
    errors = 0

    for i, f in enumerate(files):
        if i % 500 == 0:
            print(f"  [{i}/{len(files)}]...")
        try:
            bm = parser.parse_file(f)
            red   = sum(1 for tp in bm.timing_points if tp.uninherited)
            green = sum(1 for tp in bm.timing_points if not tp.uninherited)
            red_counts.append(red)
            green_counts.append(green)
            total_counts.append(red + green)
        except Exception:
            errors += 1

    def stats(lst, name):
        if not lst:
            return
        lst_sorted = sorted(lst)
        n = len(lst)
        print(f"\n── {name} ──────────────────────────")
        print(f"  Min:    {min(lst)}")
        print(f"  Max:    {max(lst)}")
        print(f"  Mean:   {sum(lst)/n:.1f}")
        print(f"  Median: {lst_sorted[n//2]}")
        print(f"  p90:    {lst_sorted[int(n*0.90)]}")
        print(f"  p95:    {lst_sorted[int(n*0.95)]}")
        print(f"  p99:    {lst_sorted[int(n*0.99)]}")

        # Bucket distribution
        buckets = [(0,0), (1,1), (2,5), (6,10), (11,50),
                   (51,100), (101,500), (501,1000), (1001,5000), (5001,99999)]
        print(f"  Distribution:")
        for lo, hi in buckets:
            count = sum(1 for x in lst if lo <= x <= hi)
            bar = "█" * (count * 30 // max(n, 1))
            print(f"    {lo:5d}-{hi:<5d}: {count:5d} ({count/n:.1%}) {bar}")

    print(f"\n{'='*55}")
    print(f"  TIMING POINT STATISTICS ({len(red_counts)} maps, {errors} errors)")
    print(f"{'='*55}")

    stats(red_counts,   "Red lines (BPM changes)")
    stats(green_counts, "Green lines (SV)")
    stats(total_counts, "Total timing points")

    # Maps that would be caught by various thresholds
    print(f"\n── SV filter thresholds ─────────────────────────")
    for threshold in [100, 500, 1000, 2000, 5000]:
        caught = sum(1 for x in total_counts if x > threshold)
        print(f"  > {threshold:5d} total TPs: {caught:4d} maps ({caught/len(total_counts):.1%})")


if __name__ == "__main__":
    main()
