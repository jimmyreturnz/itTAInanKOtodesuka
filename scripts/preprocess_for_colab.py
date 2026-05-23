"""
scripts/preprocess_for_colab.py

Run this LOCALLY before training on Colab.
Generates beatmap tensors + metadata index from .osu files.

- Star rating fetched from osu! API v1 (legacy) using beatmap_id
- Style labeled from snap analysis of hit objects vs timing points
- API results cached locally so reruns don't re-fetch

Output (upload these to Google Drive):
  data/processed/tensors/          <- beatmap tensor .npz files
  data/processed/colab_index.jsonl <- metadata per map

Usage:
    python scripts/preprocess_for_colab.py
    python scripts/preprocess_for_colab.py --no-api   # skip star rating fetch
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import requests
from dotenv import load_dotenv

from taiko.data.osu_parser import OsuTaikoParser, TaikoBeatmap, TimingPoint
from taiko.data.tensor_repr import beatmap_to_tensor

load_dotenv(Path(__file__).parent.parent / ".env")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CACHE_FILE   = "taiko_files_filtered.json"
TENSORS_DIR  = Path("data/processed/tensors")
MELS_DIR     = Path("data/processed/mels")
INDEX_PATH   = Path("data/processed/colab_index.jsonl")
SR_CACHE     = Path("data/processed/sr_cache.json")   # beatmap_id -> star_rating

OSU_API_KEY  = os.getenv("OSU_API_KEY")
OSU_API_V1   = "https://osu.ppy.sh/api/get_beatmaps"

# Style labels
STYLE_NAMES  = {0: "standard", 1: "stream", 2: "speed", 3: "tech"}

# Snap fractions to check (in fractions of a beat)
# Key = fraction denominator, Value = label category
SNAP_DENOM = {
    1:  "whole",
    2:  "half",
    4:  "quarter",    # normal / stream
    3:  "third",      # tech
    6:  "sixth",      # tech
    8:  "eighth",     # speed / tech
    5:  "fifth",      # tech
    7:  "seventh",    # tech
    9:  "ninth",      # tech
    12: "twelfth",    # tech
    16: "sixteenth",  # tech
}

# Snap tolerance: how close (in ms) a note must be to a snap grid line
SNAP_TOLERANCE_MS = 8.0

# Style thresholds
STREAM_MIN_CONSECUTIVE = 8     # min consecutive 1/4 notes to count as stream run
SPEED_EFFECTIVE_BPM    = 270   # effective BPM threshold for speed label
TECH_WEIRD_RATIO       = 0.12  # fraction of notes on weird snaps (3/5/6/7/9/12/16)


# ---------------------------------------------------------------------------
# Star rating — osu! API v1
# ---------------------------------------------------------------------------

def load_sr_cache() -> dict[int, float]:
    if SR_CACHE.exists():
        return {int(k): v for k, v in json.loads(SR_CACHE.read_text()).items()}
    return {}


def save_sr_cache(cache: dict[int, float]):
    SR_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SR_CACHE.write_text(json.dumps(cache))


def fetch_star_rating(beatmap_id: int,
                      cache: dict[int, float],
                      session: requests.Session,
                      ) -> float:
    """
    Fetch star rating from osu! API v1.
    Returns 0.0 if not found or API unavailable.
    """
    if beatmap_id in cache:
        return cache[beatmap_id]
    if not OSU_API_KEY or beatmap_id <= 0:
        return 0.0

    try:
        resp = session.get(OSU_API_V1, params={
            "k": OSU_API_KEY,
            "b": beatmap_id,
            "m": 1,   # taiko
        }, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data:
            sr = float(data[0].get("difficultyrating", 0.0))
            cache[beatmap_id] = sr
            return sr
    except Exception:
        pass

    cache[beatmap_id] = 0.0
    return 0.0


# ---------------------------------------------------------------------------
# Snap analysis
# ---------------------------------------------------------------------------

def get_active_timing(timing_points: list[TimingPoint],
                      time_ms: int) -> TimingPoint | None:
    """Return the last uninherited timing point active at time_ms."""
    active = None
    for tp in timing_points:
        if tp.time > time_ms:
            break
        if tp.uninherited:
            active = tp
    return active


def snap_of_note(note_time: int,
                 timing_points: list[TimingPoint],
                 ) -> int | None:
    """
    Determine the snap denominator of a note.
    Returns the denominator (4 = 1/4, 3 = 1/3, etc.) or None if unsnapped.

    Method:
      1. Find active BPM at note time
      2. Compute beat position = (note_time - tp.time) / ms_per_beat
      3. Check which snap grid the fractional beat position falls on
    """
    tp = get_active_timing(timing_points, note_time)
    if tp is None or tp.beat_length <= 0:
        return None

    ms_per_beat  = tp.beat_length
    offset_ms    = (note_time - tp.time) % ms_per_beat   # position within beat [0, ms_per_beat)

    best_denom = None
    best_error = SNAP_TOLERANCE_MS

    for denom in sorted(SNAP_DENOM.keys()):
        grid_ms  = ms_per_beat / denom
        # Find nearest grid line
        nearest  = round(offset_ms / grid_ms) * grid_ms
        error    = abs(offset_ms - nearest)
        if error < best_error:
            best_error = error
            best_denom = denom

    return best_denom


def effective_bpm(timing_points: list[TimingPoint],
                  notes: list,
                  ) -> float:
    """
    Compute the effective BPM — the BPM that covers the most notes,
    weighted by the density of notes in that timing section.
    """
    if not timing_points or not notes:
        return 0.0

    # Find the uninherited timing point that covers the most notes
    uninherited = [tp for tp in timing_points if tp.uninherited and tp.bpm]
    if not uninherited:
        return 0.0

    note_times = [n.time for n in notes if not n.is_long]

    best_bpm   = 0.0
    best_count = 0

    for i, tp in enumerate(uninherited):
        end_time = uninherited[i + 1].time if i + 1 < len(uninherited) else float("inf")
        count    = sum(1 for t in note_times if tp.time <= t < end_time)
        if count > best_count:
            best_count = count
            best_bpm   = tp.bpm

    return best_bpm


def count_stream_runs(notes: list,
                      timing_points: list[TimingPoint],
                      min_run: int = STREAM_MIN_CONSECUTIVE,
                      ) -> int:
    """
    Count the number of stream runs — sequences of consecutive 1/4 notes
    with at least min_run notes each.
    """
    hit_notes  = [n for n in notes if not n.is_long]
    if len(hit_notes) < min_run:
        return 0

    runs       = 0
    run_len    = 1

    for i in range(1, len(hit_notes)):
        prev = hit_notes[i - 1]
        curr = hit_notes[i]

        tp = get_active_timing(timing_points, curr.time)
        if tp is None or tp.beat_length <= 0:
            run_len = 1
            continue

        ms_per_quarter = tp.beat_length / 4.0
        gap            = curr.time - prev.time
        # Allow ±SNAP_TOLERANCE_MS around a 1/4 note gap
        if abs(gap - ms_per_quarter) <= SNAP_TOLERANCE_MS:
            run_len += 1
        else:
            if run_len >= min_run:
                runs += 1
            run_len = 1

    if run_len >= min_run:
        runs += 1

    return runs


def infer_style(bm: TaikoBeatmap) -> int:
    """
    Classify map style from snap analysis.

    Priority order (a map gets the first label that fits):
      tech   → high use of weird snaps (1/3, 1/5, 1/6, 1/7, 1/8, 1/9, 1/12, 1/16)
      speed  → effective BPM >= 270  (e.g. 180bpm 1/8 = 360 effective, or 330bpm song)
      stream → long consecutive 1/4 runs (8+ notes)
      standard → everything else

    Returns int: 0=standard, 1=stream, 2=speed, 3=tech
    """
    hit_notes = [n for n in bm.notes if not n.is_long]
    if not hit_notes or not bm.timing_points:
        return 0  # standard

    # --- Compute snaps for all hit notes ---
    weird_snaps  = {3, 5, 6, 7, 9, 12, 16}
    snap_counts  = {d: 0 for d in SNAP_DENOM}
    weird_count  = 0
    total_snapped = 0

    for note in hit_notes:
        denom = snap_of_note(note.time, bm.timing_points)
        if denom is not None:
            snap_counts[denom] = snap_counts.get(denom, 0) + 1
            total_snapped += 1
            if denom in weird_snaps:
                weird_count += 1

    if total_snapped == 0:
        return 0

    weird_ratio = weird_count / total_snapped

    # --- Tech: high use of weird snaps ---
    if weird_ratio >= TECH_WEIRD_RATIO:
        return 3  # tech

    # --- Speed: high effective BPM ---
    eff_bpm = effective_bpm(bm.timing_points, bm.notes)

    # Also consider 1/8 snap notes — each 1/8 note doubles effective BPM
    eighth_ratio = snap_counts.get(8, 0) / total_snapped
    if eighth_ratio > 0.10:
        # Significant 1/8 usage → effective BPM doubles
        eff_bpm = eff_bpm * 2

    if eff_bpm >= SPEED_EFFECTIVE_BPM:
        return 2  # speed

    # --- Stream: long consecutive 1/4 runs ---
    stream_runs = count_stream_runs(bm.notes, bm.timing_points)
    if stream_runs >= 3:
        return 1  # stream

    return 0  # standard


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

def safe_name(s: str, max_len: int = 120) -> str:
    return s[:max_len].replace("\\", "_").replace("/", "_")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(use_api: bool = True):
    if not Path(CACHE_FILE).exists():
        print(f"ERROR: {CACHE_FILE} not found. Run fast_scan.py first.")
        return

    files = [Path(p) for p in json.loads(Path(CACHE_FILE).read_text())]
    print(f"Loaded {len(files)} filtered maps.")

    TENSORS_DIR.mkdir(parents=True, exist_ok=True)

    parser   = OsuTaikoParser()
    sr_cache = load_sr_cache()
    session  = requests.Session()

    if use_api and not OSU_API_KEY:
        print("WARNING: OSU_API_KEY not found in .env — star ratings will be 0.0")
        use_api = False

    records  = []
    errors   = []
    skipped  = 0
    api_calls = 0
    t_start  = time.time()

    for i, osu_path in enumerate(files):
        if i % 200 == 0:
            elapsed = time.time() - t_start
            eta     = elapsed / max(i, 1) * (len(files) - i)
            print(f"  [{i}/{len(files)}] records={len(records)} "
                  f"errors={len(errors)} skipped={skipped} "
                  f"api_calls={api_calls} "
                  f"| {elapsed/60:.1f}min elapsed | ETA {eta/60:.1f}min")

        # Check mel exists
        folder_safe = safe_name(osu_path.parent.name)
        mel_path    = MELS_DIR / f"{folder_safe}.npz"
        if not mel_path.exists():
            skipped += 1
            continue

        # Tensor path — one per difficulty
        diff_safe   = safe_name(osu_path.stem)
        tensor_path = TENSORS_DIR / f"{folder_safe}__{diff_safe}.npz"

        # Parse beatmap
        try:
            bm = parser.parse_file(osu_path)
        except Exception as e:
            errors.append(f"{osu_path.name}: parse error: {e}")
            continue

        if bm.note_count < 20:
            skipped += 1
            continue

        # Generate tensor if not cached
        if not tensor_path.exists():
            try:
                tensor = beatmap_to_tensor(bm)
                np.savez_compressed(str(tensor_path), tensor=tensor)
            except Exception as e:
                errors.append(f"{osu_path.name}: tensor error: {e}")
                continue

        # Star rating — from API or fallback to overall_difficulty
        if use_api and bm.beatmap_id > 0:
            if bm.beatmap_id not in sr_cache:
                api_calls += 1
                # Save cache every 100 new API calls
                if api_calls % 100 == 0:
                    save_sr_cache(sr_cache)
                # Rate limit: ~1 req/sec to be safe
                time.sleep(1.0)
            star_rating = fetch_star_rating(bm.beatmap_id, sr_cache, session)
        else:
            star_rating = bm.star_rating if bm.star_rating > 0 else 0.0

        # Fallback if API returned 0
        if star_rating == 0.0:
            star_rating = bm.overall_difficulty   # rough proxy

        # Style label
        style = infer_style(bm)

        records.append({
            "mel_path":    str(mel_path.relative_to("data/processed")),
            "tensor_path": str(tensor_path.relative_to("data/processed")),
            "difficulty":  round(star_rating, 2),
            "style":       style,
            "style_name":  STYLE_NAMES[style],
            "note_count":  bm.note_count,
            "duration_ms": bm.duration_ms,
            "title":       bm.title,
            "version":     bm.version,
            "beatmap_id":  bm.beatmap_id,
        })

    # Final cache save
    save_sr_cache(sr_cache)

    # Write index
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Summary
    total_time = time.time() - t_start
    style_dist = {name: 0 for name in STYLE_NAMES.values()}
    for rec in records:
        style_dist[rec["style_name"]] += 1

    print(f"\n{'='*50}")
    print(f"Done in {total_time/60:.1f} minutes")
    print(f"Records  : {len(records)}")
    print(f"Errors   : {len(errors)}")
    print(f"Skipped  : {skipped} (no mel or too few notes)")
    print(f"API calls: {api_calls}")
    print(f"\nStyle distribution:")
    for name, count in style_dist.items():
        pct = count / max(len(records), 1) * 100
        print(f"  {name:10s}: {count:5d}  ({pct:.1f}%)")

    tensor_files = list(TENSORS_DIR.glob("*.npz"))
    total_mb = sum(f.stat().st_size for f in tensor_files) / 1024**2
    print(f"\nTensor folder size: {total_mb:.0f} MB ({total_mb/1024:.2f} GB)")
    print(f"\nFiles to upload to Google Drive:")
    print(f"  data/processed/tensors/          ({len(records)} .npz files)")
    print(f"  data/processed/colab_index.jsonl")

    if errors:
        err_path = Path("data/processed/tensor_errors.txt")
        err_path.write_text("\n".join(errors))
        print(f"\nErrors saved to: {err_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-api", action="store_true",
                    help="Skip osu! API star rating fetch (faster, uses OD as proxy)")
    args = ap.parse_args()
    main(use_api=not args.no_api)
