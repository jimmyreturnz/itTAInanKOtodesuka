"""
scripts/extract_mels.py

Extract mel spectrograms from all beatmapset folders in taiko_files_filtered.json.
Saves one .npz per beatmapset folder (shared across all difficulties in that folder).

Filenames use clean safe_name() — strips Kaggle-forbidden characters so
rename_mels.py is not needed after running this.

Usage:
    python scripts/extract_mels.py
    python scripts/extract_mels.py --cache taiko_files_filtered.json
    python scripts/extract_mels.py --recompute   # recompute existing mels
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from taiko.data.audio import MelExtractor, save_mel, load_mel


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CACHE_FILE = "taiko_files_filtered.json"
MELS_DIR   = Path("data/processed/mels")


# ---------------------------------------------------------------------------
# Filename helpers — same logic as preprocess_for_colab.py
# ---------------------------------------------------------------------------

def safe_name(s: str, max_len: int = 120) -> str:
    """Strip all Kaggle-forbidden and problematic characters."""
    cleaned = re.sub(r'[\[\]\(\)\'"!@#$%^&*+=|\\/<>?:;,~`]', '', s)
    cleaned = cleaned.replace('.', '')
    cleaned = re.sub(r'[ _]{2,}', ' ', cleaned).strip()
    return cleaned[:max_len]


def find_audio(folder: Path) -> Path | None:
    """Find the first audio file in a folder."""
    for ext in (".mp3", ".ogg", ".wav", ".flac"):
        found = list(folder.glob(f"*{ext}"))
        if found:
            return found[0]
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cache_file: str, recompute: bool):
    if not Path(cache_file).exists():
        print(f"ERROR: {cache_file} not found. Run fast_scan.py first.")
        return

    files = [Path(p) for p in json.loads(Path(cache_file).read_text(encoding="utf-8"))]
    print(f"Loaded {len(files)} filtered maps.")

    # Group by parent folder (one mel per beatmapset)
    by_folder: dict[Path, list[Path]] = defaultdict(list)
    for f in files:
        by_folder[f.parent].append(f)
    print(f"Grouped into {len(by_folder)} beatmapset folders.")

    MELS_DIR.mkdir(parents=True, exist_ok=True)

    extractor     = MelExtractor()
    total_folders = len(by_folder)
    extracted     = 0
    skipped       = 0
    errors        = []
    t_start       = time.time()

    for i, (folder, _) in enumerate(sorted(by_folder.items())):
        if i % 200 == 0:
            elapsed = time.time() - t_start
            eta     = elapsed / max(i, 1) * (total_folders - i)
            print(f"  [{i}/{total_folders}] extracted={extracted} "
                  f"skipped={skipped} errors={len(errors)} "
                  f"| {elapsed/60:.1f}min elapsed | ETA {eta/60:.1f}min")

        mel_name = safe_name(folder.name)
        mel_path = MELS_DIR / f"{mel_name}.npz"

        # Skip if already extracted and not recomputing
        if mel_path.exists() and not recompute:
            skipped += 1
            continue

        # Find audio
        audio = find_audio(folder)
        if audio is None:
            errors.append(f"No audio: {folder.name}")
            continue

        # Skip very large files (corrupted / video)
        try:
            if os.path.getsize(audio) > 150 * 1024 * 1024:  # 150MB
                errors.append(f"Audio too large: {folder.name}")
                continue
        except OSError:
            continue

        # Extract mel
        try:
            mel = extractor.extract(audio)
            save_mel(mel, mel_path)
            extracted += 1
        except Exception as e:
            errors.append(f"Mel error {folder.name}: {e}")
            continue

    total_time = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"Done in {total_time/60:.1f} minutes")
    print(f"Extracted : {extracted}")
    print(f"Skipped   : {skipped} (already exist)")
    print(f"Errors    : {len(errors)}")

    mel_files = list(MELS_DIR.glob("*.npz"))
    total_mb  = sum(f.stat().st_size for f in mel_files) / 1024**2
    print(f"Total mels: {len(mel_files)} files ({total_mb:.0f} MB)")

    if errors:
        err_path = Path("data/processed/mel_errors.txt")
        err_path.write_text("\n".join(errors), encoding="utf-8")
        print(f"Errors saved to: {err_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache",     default=CACHE_FILE, help="Path to filtered cache JSON")
    ap.add_argument("--recompute", action="store_true", help="Recompute existing mels")
    args = ap.parse_args()
    main(args.cache, args.recompute)
