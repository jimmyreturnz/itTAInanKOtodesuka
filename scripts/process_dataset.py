"""
scripts/process_dataset.py

Converts raw beatmap directories into the processed dataset format.
Uses taiko_files_cache.json (from fast_scan.py) instead of scanning.

Usage:
    python scripts/process_dataset.py \
        --cache  taiko_files_cache.json \
        --output data/processed \
        --sr-db  data/star_ratings.json
"""

import argparse
import json
import sys
import os
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from taiko.data.audio import MelExtractor, save_mel, load_mel
from taiko.data.tokenizer import TaikoTokenizer
from taiko.data.osu_parser import OsuTaikoParser
from taiko.data.dataset import BeatmapRecord, split_index


def find_audio(folder: Path):
    for ext in (".mp3", ".ogg", ".wav", ".flac"):
        found = list(folder.glob(f"*{ext}"))
        if found:
            return found[0]
    return None


def estimate_sr(bm):
    return min(10.0, bm.overall_difficulty * 0.6 + bm.notes_per_second * 0.2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache",      default="taiko_files_cache.json", help="Path to taiko_files_cache.json")
    parser.add_argument("--output",     default="data/processed",         help="Output directory")
    parser.add_argument("--sr-db",      default=None,                     help="Path to star_ratings.json")
    parser.add_argument("--min-notes",  type=int,   default=50)
    parser.add_argument("--max-notes",  type=int,   default=5000)
    parser.add_argument("--val-ratio",  type=float, default=0.05)
    parser.add_argument("--no-split",   action="store_true")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Load cache
    # ------------------------------------------------------------------ #
    cache_path = Path(args.cache)
    if not cache_path.exists():
        print(f"ERROR: Cache not found at {cache_path}")
        print("Run fast_scan.py first.")
        return

    taiko_files = [Path(p) for p in json.loads(cache_path.read_text())]
    print(f"Loaded {len(taiko_files)} taiko .osu paths from cache.")

    # Group by parent folder (= beatmapset)
    by_folder: dict[Path, list[Path]] = defaultdict(list)
    for f in taiko_files:
        by_folder[f.parent].append(f)
    print(f"Grouped into {len(by_folder)} beatmapset folders.")

    # ------------------------------------------------------------------ #
    # Load star rating DB
    # ------------------------------------------------------------------ #
    sr_db: dict[int, float] = {}
    if args.sr_db and Path(args.sr_db).exists():
        raw = json.loads(Path(args.sr_db).read_text())
        sr_db = {int(k): float(v) for k, v in raw.items()}
        print(f"Loaded {len(sr_db)} star ratings.")

    # ------------------------------------------------------------------ #
    # Setup output
    # ------------------------------------------------------------------ #
    output_path = Path(args.output)
    mel_dir     = output_path / "mels"
    index_path  = output_path / "index.jsonl"
    mel_dir.mkdir(parents=True, exist_ok=True)

    extractor  = MelExtractor()
    tokenizer  = TaikoTokenizer()
    osu_parser = OsuTaikoParser()

    records = []
    errors  = []
    total_folders = len(by_folder)
    t_start = time.time()

    for i, (folder, osu_files) in enumerate(sorted(by_folder.items())):

        # Progress every 100 folders
        if i % 100 == 0:
            elapsed = time.time() - t_start
            eta = (elapsed / max(i, 1)) * (total_folders - i)
            print(f"  [{i}/{total_folders}] {len(records)} records so far | "
                  f"elapsed {elapsed/60:.1f}min | ETA {eta/60:.1f}min")

        # Find audio
        audio = find_audio(folder)
        if audio is None:
            errors.append(f"No audio: {folder.name}")
            continue

        # Skip very large audio files (corrupted/video files)
        try:
            if os.path.getsize(audio) > 100 * 1024 * 1024:  # 100MB
                errors.append(f"Audio too large: {folder.name}")
                continue
        except OSError:
            continue

        # Mel — use folder name as cache key (truncated to 120 chars for Windows)
        safe_name = folder.name[:120].replace("\\", "_").replace("/", "_")
        mel_path  = mel_dir / f"{safe_name}.npz"

        if mel_path.exists():
            try:
                mel = load_mel(mel_path)
            except Exception:
                mel_path.unlink()  # corrupt cache, recompute
                mel = None
        else:
            mel = None

        if mel is None:
            try:
                mel = extractor.extract(audio)
                save_mel(mel, mel_path)
            except Exception as e:
                errors.append(f"Mel error {folder.name}: {e}")
                continue

        # Process each .osu in this folder
        for osu_path in osu_files:
            try:
                bm = osu_parser.parse_file(osu_path)
            except Exception as e:
                errors.append(f"Parse error {osu_path.name}: {e}")
                continue

            # Filter by note count
            if bm.note_count < args.min_notes or bm.note_count > args.max_notes:
                continue

            # Attach star rating
            if sr_db and bm.beatmap_id in sr_db:
                bm.star_rating = sr_db[bm.beatmap_id]
            elif bm.star_rating == 0.0:
                bm.star_rating = estimate_sr(bm)

            # Tokenize
            try:
                tok = tokenizer.encode(bm)
            except Exception as e:
                errors.append(f"Tokenize error {osu_path.name}: {e}")
                continue

            records.append(BeatmapRecord(
                beatmap_id=bm.beatmap_id,
                beatmap_set_id=bm.beatmap_set_id,
                audio_path=str(audio),
                mel_path=str(mel_path),
                conditioning_ids=tok.conditioning_ids,
                token_ids=tok.token_ids,
                beat_length_ms=tok.beat_length_ms,
                offset_ms=tok.offset_ms,
                duration_ms=bm.duration_ms,
                star_rating=bm.star_rating,
                title=bm.title,
                version=bm.version,
                note_count=bm.note_count,
            ))

    # ------------------------------------------------------------------ #
    # Write index
    # ------------------------------------------------------------------ #
    with open(index_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict()) + "\n")

    total_time = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"Done in {total_time/60:.1f} minutes")
    print(f"Records written : {len(records)}")
    print(f"Errors          : {len(errors)}")
    print(f"Index saved to  : {index_path}")

    if errors:
        error_log = output_path / "errors.txt"
        error_log.write_text("\n".join(errors), encoding="utf-8")
        print(f"Errors logged to: {error_log}")

    # ------------------------------------------------------------------ #
    # Train / val split
    # ------------------------------------------------------------------ #
    if not args.no_split and len(records) > 0:
        split_index(index_path, val_ratio=args.val_ratio)


if __name__ == "__main__":
    main()
