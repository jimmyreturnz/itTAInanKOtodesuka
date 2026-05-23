"""
scripts/rename_mels.py

Renames mel AND tensor .npz files to remove Kaggle-forbidden characters,
then updates colab_index.jsonl so all paths stay in sync.

Forbidden chars removed: [ ] ( ) ' "
Collision handling: appends _1, _2 etc. if cleaned name already exists.

Usage:
    python scripts/rename_mels.py --dry-run   # preview only
    python scripts/rename_mels.py             # actually rename + update index
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path


MELS_DIR    = Path("data/processed/mels")
TENSORS_DIR = Path("data/processed/tensors")
INDEX_PATH  = Path("data/processed/colab_index.jsonl")


def clean_name(name: str) -> str:
    # Remove forbidden/problematic characters
    cleaned = re.sub(r'[\[\]\(\)\'"!@#$%^&*+=|\\/<>?:;,~`]', '', name)
    # Replace dots except in extension (dots in stems can cause issues)
    cleaned = cleaned.replace('.', '')
    # Collapse multiple spaces/underscores
    cleaned = re.sub(r'[ _]{2,}', ' ', cleaned).strip()
    return cleaned


def build_rename_map(folder: Path) -> dict[Path, Path]:
    """
    For every .npz in folder, compute old_path -> new_path.
    Handles collisions by appending _1, _2 etc.
    Returns only entries that actually need renaming.
    """
    rename_map: dict[Path, Path] = {}
    # Track names that will exist after renaming (to detect collisions)
    taken: set[str] = set()

    # First pass: collect files that don't need renaming (already clean)
    for p in sorted(folder.glob("*.npz")):
        new_stem = clean_name(p.stem)
        if new_stem == p.stem:
            taken.add(p.stem.lower())   # Windows is case-insensitive

    # Second pass: assign new names for files that need renaming
    for p in sorted(folder.glob("*.npz")):
        new_stem = clean_name(p.stem)
        if new_stem == p.stem:
            continue   # already clean

        # Resolve collision
        candidate = new_stem
        counter   = 1
        while candidate.lower() in taken:
            candidate = f"{new_stem}_{counter}"
            counter  += 1

        taken.add(candidate.lower())
        rename_map[p] = folder / f"{candidate}.npz"

    return rename_map


def apply_renames(rename_map: dict[Path, Path], dry_run: bool) -> dict[str, str]:
    """
    Execute renames. Returns {old_stem: new_stem} for index updating.
    """
    stem_map: dict[str, str] = {}
    errors = 0

    for old_path, new_path in rename_map.items():
        stem_map[old_path.stem] = new_path.stem
        if dry_run:
            continue
        try:
            old_path.rename(new_path)
        except Exception as e:
            print(f"  ERROR renaming {old_path.name}: {e}")
            errors += 1

    return stem_map, errors


def update_index(mel_stem_map: dict[str, str],
                 tensor_stem_map: dict[str, str],
                 dry_run: bool) -> tuple[int, int]:
    """
    Update colab_index.jsonl using the stem rename maps.
    Returns (mel_updates, tensor_updates).
    """
    if not INDEX_PATH.exists():
        print("No colab_index.jsonl found — skipping index update")
        return 0, 0

    records = [
        json.loads(line)
        for line in INDEX_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    mel_updates    = 0
    tensor_updates = 0

    for rec in records:
        # Update mel_path
        mel_path = rec.get("mel_path", "")
        if mel_path:
            old_stem = Path(mel_path).stem
            if old_stem in mel_stem_map:
                new_stem      = mel_stem_map[old_stem]
                rec["mel_path"] = str(Path(mel_path).parent / f"{new_stem}.npz")
                mel_updates  += 1

        # Update tensor_path
        tensor_path = rec.get("tensor_path", "")
        if tensor_path:
            old_stem = Path(tensor_path).stem
            if old_stem in tensor_stem_map:
                new_stem          = tensor_stem_map[old_stem]
                rec["tensor_path"] = str(Path(tensor_path).parent / f"{new_stem}.npz")
                tensor_updates   += 1

    if not dry_run:
        with open(INDEX_PATH, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return mel_updates, tensor_updates


def verify_index():
    """
    After renaming, verify every path in the index actually exists.
    Prints any broken entries.
    """
    if not INDEX_PATH.exists():
        return

    records = [
        json.loads(line)
        for line in INDEX_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    broken_mel    = 0
    broken_tensor = 0
    base = Path("data/processed")

    for rec in records:
        if not (base / rec.get("mel_path", "")).exists():
            broken_mel += 1
        if not (base / rec.get("tensor_path", "")).exists():
            broken_tensor += 1

    total = len(records)
    print(f"\n── Index verification ──────────────────")
    print(f"  Total records  : {total}")
    print(f"  Broken mel     : {broken_mel}")
    print(f"  Broken tensor  : {broken_tensor}")
    if broken_mel == 0 and broken_tensor == 0:
        print(f"  ✓ All paths verified OK")
    else:
        print(f"  ✗ Fix broken paths before uploading to Kaggle")


def main(dry_run: bool):
    tag = "[DRY RUN] " if dry_run else ""

    # ── Mels ────────────────────────────────────────────────────────────── #
    mel_map = build_rename_map(MELS_DIR) if MELS_DIR.exists() else {}
    print(f"Mels   — files to rename: {len(mel_map)}")
    if dry_run:
        for old, new in list(mel_map.items())[:5]:
            print(f"  {old.name}")
            print(f"  -> {new.name}")
        if len(mel_map) > 5:
            print(f"  ... and {len(mel_map)-5} more")

    mel_stem_map, mel_errors = apply_renames(mel_map, dry_run)

    # ── Tensors ─────────────────────────────────────────────────────────── #
    tensor_map = build_rename_map(TENSORS_DIR) if TENSORS_DIR.exists() else {}
    print(f"Tensors — files to rename: {len(tensor_map)}")
    if dry_run:
        for old, new in list(tensor_map.items())[:5]:
            print(f"  {old.name}")
            print(f"  -> {new.name}")
        if len(tensor_map) > 5:
            print(f"  ... and {len(tensor_map)-5} more")

    tensor_stem_map, tensor_errors = apply_renames(tensor_map, dry_run)

    # ── Index ────────────────────────────────────────────────────────────── #
    mel_upd, tensor_upd = update_index(mel_stem_map, tensor_stem_map, dry_run)
    print(f"\n{tag}Index updates — mel: {mel_upd}, tensor: {tensor_upd}")

    if not dry_run:
        print(f"Rename errors  — mel: {mel_errors}, tensor: {tensor_errors}")
        verify_index()
        print("\nDone. Re-zip mels/ + tensors/ + colab_index.jsonl and upload to Kaggle.")
    else:
        print("\nRun without --dry-run to apply changes.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    main(dry_run=args.dry_run)