"""
scripts/apply_selection.py

Reads unranked_review.csv selections and:
  1. Determines which beatmapsets to KEEP (ranked + selected unranked)
  2. Deletes mels for unselected unranked beatmapsets
  3. Tensors are NOT touched here — preprocess_for_colab.py will only
     generate tensors for kept beatmapsets

Usage:
    python scripts/apply_selection.py
    python scripts/apply_selection.py --dry-run   # preview without deleting
"""

import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

CACHE_FILE = "data/processed/beatmapset_cache.json"
CSV_FILE = "data/unranked_review.csv"
MELS_DIR = Path("data/processed/mels")
CACHE_FILE_F = "taiko_files_filtered.json"


def safe_name(s: str, max_len: int = 120) -> str:
    cleaned = re.sub(r'[\[\]\(\)\'"!@#$%^&*+=|\\/<>?:;,~`]', '', s)
    cleaned = cleaned.replace('.', '')
    cleaned = re.sub(r'[ _]{2,}', ' ', cleaned).strip()

    return cleaned[:max_len]


def main(dry_run: bool = False):

    # ---- Load cache -------------------------------------------------- #
    cache = json.loads(
        Path(CACHE_FILE).read_text(encoding="utf-8")
    )

    # ---- Load CSV selections ----------------------------------------- #
    csv_path = Path(CSV_FILE)

    if not csv_path.exists():
        print(
            f"ERROR: {CSV_FILE} not found. "
            f"Run export_unranked_csv.py first."
        )
        return

    # Collect selected unranked beatmapset IDs
    selected_unranked: set[str] = set()

    with open(csv_path, "r", encoding="utf-8-sig") as f:

        reader = csv.DictReader(f)

        for row in reader:

            if str(row.get("include", "0")).strip() == "1":
                selected_unranked.add(
                    str(row["beatmapset_id"]).strip()
                )

    print(f"Selected unranked beatmapsets: {len(selected_unranked)}")

    # ---- Determine keep set ------------------------------------------ #
    ranked_ids = {
        k for k, v in cache.items()
        if isinstance(v, dict) and v.get("ranked", False)
    }

    unranked_ids = {
        k for k, v in cache.items()
        if isinstance(v, dict) and not v.get("ranked", False)
    }

    keep_ids = ranked_ids | selected_unranked
    drop_ids = unranked_ids - selected_unranked

    print(f"Ranked beatmapsets:            {len(ranked_ids)}")
    print(f"Unranked total:                {len(unranked_ids)}")
    print(f"Unranked selected (keep):      {len(selected_unranked)}")
    print(f"Unranked dropped:              {len(drop_ids)}")
    print(f"Total kept:                    {len(keep_ids)}")

    # ---- Load filtered osu cache ------------------------------------- #
    osu_files = [
        Path(p)
        for p in json.loads(
            Path(CACHE_FILE_F).read_text(encoding="utf-8")
        )
    ]

    # Map folder name -> beatmapset_id
    def folder_to_bmsid(folder_name: str) -> str:
        m = re.match(r"^(\d+)", folder_name.strip())
        return m.group(1) if m else ""

    # Build set of mel names to drop
    drop_mel_names: set[str] = set()
    seen_folders: set[Path] = set()

    for osu_path in osu_files:

        folder = osu_path.parent

        if folder in seen_folders:
            continue

        seen_folders.add(folder)

        bms_id = folder_to_bmsid(folder.name)

        if bms_id in drop_ids:
            drop_mel_names.add(
                safe_name(folder.name)
            )

    print(f"\nMel files to delete: {len(drop_mel_names)}")

    # ---- Delete mels ------------------------------------------------- #
    deleted = 0
    not_found = 0

    for mel_name in sorted(drop_mel_names):

        mel_path = MELS_DIR / f"{mel_name}.npz"

        if mel_path.exists():

            if dry_run:
                print(f"  [DRY] would delete: {mel_path.name}")

            else:
                mel_path.unlink()
                deleted += 1

        else:
            not_found += 1

    # ---- Summary ----------------------------------------------------- #
    print(f"\n{'=' * 50}")

    if dry_run:
        print("DRY RUN — nothing deleted")
        print(f"Would delete: {len(drop_mel_names)} mel files")

    else:
        print(f"Deleted:   {deleted} mel files")
        print(
            f"Not found: {not_found} "
            f"(already missing or never extracted)"
        )

    remaining = list(MELS_DIR.glob("*.npz"))

    print(f"Remaining mels: {len(remaining)}")

    print("\nNext step: run preprocess_for_colab.py")
    print("  python scripts/preprocess_for_colab.py")


if __name__ == "__main__":

    import argparse

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without deleting"
    )

    args = ap.parse_args()

    main(dry_run=args.dry_run)