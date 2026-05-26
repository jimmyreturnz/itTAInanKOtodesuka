"""
scripts/export_unranked_csv.py

Breaks down beatmapset_cache.json into a CSV for manual review.
You open the CSV, set include=1 for unranked maps you want to keep,
then run apply_selection.py to:
  - delete mels for excluded unranked maps
  - keep mels for ranked + selected unranked

Usage:
    python scripts/export_unranked_csv.py
    # → opens data/unranked_review.csv
    # Edit CSV: set include column to 1 for maps you want
    python scripts/apply_selection.py
"""

import json
import csv
import sys
from pathlib import Path

from librosa import cache

sys.path.insert(0, str(Path(__file__).parent.parent))

CACHE_FILE = "data/processed/beatmapset_cache.json"
OUTPUT_CSV = "data/unranked_review.csv"
MELS_DIR = Path("data/processed/mels")


def safe_name(s: str, max_len: int = 120) -> str:
    import re

    cleaned = re.sub(r'[\[\]\(\)\'"!@#$%^&*+=|\\/<>?:;,~`]', '', s)
    cleaned = cleaned.replace('.', '')
    cleaned = re.sub(r'[ _]{2,}', ' ', cleaned).strip()

    return cleaned[:max_len]


def main():
    cache_path = Path(CACHE_FILE)

    if not cache_path.exists():
        print(f"ERROR: {CACHE_FILE} not found.")
        return

    cache = json.loads(cache_path.read_text(encoding="utf-8"))

    print(f"Loaded {len(cache)} beatmapsets from cache.")

    # Separate ranked vs unranked
    ranked_sets = {
        k: v
        for k, v in cache.items()
            if isinstance(v, dict) and v.get("ranked", False)
    }

    unranked_sets = {
        k: v
        for k, v in cache.items()
        if isinstance(v, dict) and not v.get("ranked", False)
    }

    print(f"Ranked:   {len(ranked_sets)} beatmapsets")
    print(f"Unranked: {len(unranked_sets)} beatmapsets")

    # Build CSV rows
    rows = []

    for bms_id, bms_data in sorted(unranked_sets.items(), key=lambda x: x[0]):

        approved_str = bms_data.get("approved_str", "unknown")
        difficulties = bms_data.get("difficulties", {})

        # No difficulty data
        if not difficulties:
            rows.append({
                "beatmapset_id": bms_id,
                "beatmap_id": "",
                "approved_str": approved_str,
                "version": "",
                "star_rating": "",
                "include": 0,
                "notes": "",
            })

        else:
            for bm_id, diff_data in difficulties.items():

                rows.append({
                    "beatmapset_id": bms_id,
                    "beatmap_id": bm_id,
                    "approved_str": approved_str,
                    "version": diff_data.get("version", ""),
                    "star_rating": diff_data.get("sr", ""),
                    "include": 0,
                    "notes": "",
                })

    # Write CSV
    output_path = Path(OUTPUT_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "beatmapset_id",
        "beatmap_id",
        "approved_str",
        "version",
        "star_rating",
        "include",
        "notes",
    ]

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV written: {output_path}")
    print(
        f"Total rows: {len(rows)} difficulties across "
        f"{len(unranked_sets)} unranked beatmapsets"
    )

    print("\nInstructions:")
    print(f"  1. Open {OUTPUT_CSV} in Excel or any CSV editor")
    print("  2. Set 'include' column to 1 for beatmapsets you want to keep")
    print("     (setting any difficulty row to 1 keeps the whole beatmapset)")
    print("  3. Save the CSV")
    print("  4. Run: python scripts/apply_selection.py")


if __name__ == "__main__":
    main()