"""
scripts/preprocess_for_colab.py

Run this LOCALLY before training on Colab/Kaggle.
Generates beatmap tensors + metadata index from .osu files.

New fields added to index rows:
  avg_nps   — global notes-per-second over full map duration
  peak_nps  — max NPS in any 5-second sliding window
  snap_1_4  — fraction of notes on 1/4 snap
  snap_1_6  — fraction of notes on 1/6 snap
  snap_1_8  — fraction of notes on 1/8 snap

- Ranked status + SR: beatmapset_cache.json first, then osu! API v1 on cache miss
- Works with --no-api as long as beatmapset_cache.json is populated
- Style labeled from snap analysis of hit objects vs timing points
- sr_cache.json kept as SR fallback for old beatmap_id entries

Output (upload these to Google Drive/Kaggle):
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
import re
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

CACHE_FILE        = "taiko_files_filtered.json"
TENSORS_DIR       = Path("data/processed/tensors")
MELS_DIR          = Path("data/processed/mels")
INDEX_PATH        = Path("data/processed/colab_index.jsonl")
SR_CACHE          = Path("data/processed/sr_cache.json")
BEATMAPSET_CACHE  = Path("data/processed/beatmapset_cache.json")

OSU_API_KEY       = os.getenv("OSU_API_KEY")
OSU_API_V1        = "https://osu.ppy.sh/api/get_beatmaps"
API_DELAY         = 0.5

APPROVED_MAP = {
    "4": "loved",
    "3": "qualified",
    "2": "approved",
    "1": "ranked",
    "0": "pending",
    "-1": "wip",
    "-2": "graveyard",
}

STYLE_NAMES  = {0: "standard", 1: "stream", 2: "speed", 3: "tech"}

# Snap analysis constants
SNAP_TOLERANCE_MS      = 8.0
STREAM_MIN_CONSECUTIVE = 8
SPEED_EFFECTIVE_BPM    = 270
TECH_WEIRD_RATIO       = 0.12
SNAP_DENOM             = {1, 2, 4, 3, 6, 8, 5, 7, 9, 12, 16}

# NPS peak window
PEAK_NPS_WINDOW_MS = 5_000   # 5-second sliding window


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def load_sr_cache() -> dict[str, float]:
    if SR_CACHE.exists():
        return {str(k): float(v) for k, v in json.loads(SR_CACHE.read_text()).items()}
    return {}


def load_beatmapset_cache() -> dict[str, dict | None]:
    if BEATMAPSET_CACHE.exists():
        return json.loads(BEATMAPSET_CACHE.read_text(encoding="utf-8"))
    return {}


def save_beatmapset_cache(cache: dict):
    BEATMAPSET_CACHE.parent.mkdir(parents=True, exist_ok=True)
    BEATMAPSET_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def lookup_beatmapset(beatmapset_id: int, cache: dict) -> dict | None:
    if beatmapset_id <= 0:
        return None
    entry = cache.get(str(beatmapset_id))
    if not isinstance(entry, dict):
        return None
    return entry


def metadata_from_set_data(set_data: dict, beatmap_id: int) -> tuple[float, str, str, bool]:
    approved     = str(set_data.get("approved", "0"))
    approved_str = set_data.get("approved_str") or APPROVED_MAP.get(approved, "unknown")
    is_ranked    = bool(set_data.get("ranked", approved in ("1", "2", "3", "4")))
    star_rating  = 0.0
    difficulties = set_data.get("difficulties") or {}
    bid_key      = str(beatmap_id)
    if bid_key in difficulties:
        star_rating = float(difficulties[bid_key].get("sr", 0.0))
    return star_rating, approved, approved_str, is_ranked


# ---------------------------------------------------------------------------
# osu! API v1
# ---------------------------------------------------------------------------

def fetch_beatmapset(
    beatmapset_id: int,
    cache: dict,
    session: requests.Session,
) -> dict | None:
    key = str(beatmapset_id)
    if key in cache:
        return cache[key]
    if not OSU_API_KEY or beatmapset_id <= 0:
        return None
    try:
        resp = session.get(OSU_API_V1, params={
            "k": OSU_API_KEY,
            "s": beatmapset_id,
            "m": 1,
        }, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            cache[key] = None
            return None
        approved     = str(data[0].get("approved", "0"))
        approved_str = APPROVED_MAP.get(approved, "unknown")
        is_ranked    = approved in ("1", "2", "3", "4")
        difficulties = {}
        for d in data:
            bid = str(d.get("beatmap_id", ""))
            if bid:
                difficulties[bid] = {
                    "sr":      float(d.get("difficultyrating", 0.0)),
                    "version": d.get("version", ""),
                }
        result = {
            "approved":     approved,
            "approved_str": approved_str,
            "ranked":       is_ranked,
            "difficulties": difficulties,
        }
        cache[key] = result
        return result
    except Exception as e:
        print(f"  [API error] beatmapset {beatmapset_id}: {e}")
        cache[key] = None
        return None


# ---------------------------------------------------------------------------
# NPS helpers
# ---------------------------------------------------------------------------

def compute_avg_nps(bm: TaikoBeatmap) -> float:
    """
    Global average NPS over the actual note span.

    Uses the time between the first and last hit note rather than
    bm.duration_ms, because duration_ms is derived from the last note's
    time/end_time and can be near-zero when end_time is unset on
    rolls/dendens — producing astronomically wrong NPS values.

    Clamps to [0, 30] to catch any remaining edge cases.
    """
    hit_times = sorted(n.time for n in bm.notes if not n.is_long)
    if len(hit_times) < 2:
        return 0.0
    span_sec = (hit_times[-1] - hit_times[0]) / 1000.0
    if span_sec < 1.0:          # less than 1s span → skip (malformed map)
        return 0.0
    nps = len(hit_times) / span_sec
    return round(min(nps, 30.0), 3)


def compute_peak_nps(bm: TaikoBeatmap, window_ms: int = PEAK_NPS_WINDOW_MS) -> float:
    """
    Peak NPS in any sliding window of `window_ms` milliseconds.
    Uses a two-pointer approach over sorted hit note times — O(n).

    The window always has a fixed duration of window_ms, so we count
    notes inside [t_left, t_left + window_ms] as the right pointer advances.
    Clamps to [0, 30].
    """
    hit_times = sorted(n.time for n in bm.notes if not n.is_long)
    if not hit_times:
        return 0.0

    window_sec = window_ms / 1000.0
    peak       = 0.0
    left       = 0

    for right in range(len(hit_times)):
        # Shrink left pointer until window fits within window_ms
        while hit_times[right] - hit_times[left] > window_ms:
            left += 1
        # Count / fixed window duration (not the actual span, which shrinks
        # when notes cluster — that inflates NPS artificially)
        count = right - left + 1
        nps   = count / window_sec
        if nps > peak:
            peak = nps

    return round(min(peak, 30.0), 3)


# ---------------------------------------------------------------------------
# Snap analysis
# ---------------------------------------------------------------------------

def get_active_timing(timing_points, time_ms):
    active = None
    for tp in timing_points:
        if tp.time > time_ms:
            break
        if tp.uninherited:
            active = tp
    return active


def snap_of_note(note_time, timing_points):
    tp = get_active_timing(timing_points, note_time)
    if tp is None or tp.beat_length <= 0:
        return None
    ms_per_beat = tp.beat_length
    offset_ms   = (note_time - tp.time) % ms_per_beat
    best_denom  = None
    best_error  = SNAP_TOLERANCE_MS
    for denom in sorted(SNAP_DENOM):
        grid_ms = ms_per_beat / denom
        nearest = round(offset_ms / grid_ms) * grid_ms
        error   = abs(offset_ms - nearest)
        if error < best_error:
            best_error = error
            best_denom = denom
    return best_denom


def effective_bpm(timing_points, notes):
    if not timing_points or not notes:
        return 0.0
    uninherited = [tp for tp in timing_points if tp.uninherited and tp.bpm]
    if not uninherited:
        return 0.0
    note_times = [n.time for n in notes if not n.is_long]
    best_bpm, best_count = 0.0, 0
    for i, tp in enumerate(uninherited):
        end_time = uninherited[i + 1].time if i + 1 < len(uninherited) else float("inf")
        count    = sum(1 for t in note_times if tp.time <= t < end_time)
        if count > best_count:
            best_count = count
            best_bpm   = tp.bpm
    return best_bpm


def count_stream_runs(notes, timing_points, min_run=STREAM_MIN_CONSECUTIVE):
    hit_notes = [n for n in notes if not n.is_long]
    if len(hit_notes) < min_run:
        return 0
    runs, run_len = 0, 1
    for i in range(1, len(hit_notes)):
        tp = get_active_timing(timing_points, hit_notes[i].time)
        if tp is None or tp.beat_length <= 0:
            if run_len >= min_run:
                runs += 1
            run_len = 1
            continue
        ms_per_quarter = tp.beat_length / 4.0
        gap            = hit_notes[i].time - hit_notes[i - 1].time
        if abs(gap - ms_per_quarter) <= SNAP_TOLERANCE_MS:
            run_len += 1
        else:
            if run_len >= min_run:
                runs += 1
            run_len = 1
    if run_len >= min_run:
        runs += 1
    return runs


def compute_snap_ratios(bm: TaikoBeatmap) -> dict[str, float]:
    """
    Compute per-snap-divisor fraction of total snapped notes.

    Returns a dict with keys:
        snap_1_4, snap_1_6, snap_1_8
    and implicitly covers the rest (1/1, 1/2, 1/3, etc.) via
    the style inference, but only these three are stored as
    conditioning signals.

    Notes that cannot be snapped to any divisor are excluded from
    the denominator (they register as weird/tech snaps).
    """
    hit_notes = [n for n in bm.notes if not n.is_long]
    if not hit_notes or not bm.timing_points:
        return {"snap_1_4": 0.0, "snap_1_6": 0.0, "snap_1_8": 0.0}

    snap_counts: dict[int, int] = {d: 0 for d in SNAP_DENOM}
    total_snapped = 0

    for note in hit_notes:
        denom = snap_of_note(note.time, bm.timing_points)
        if denom is not None:
            snap_counts[denom] = snap_counts.get(denom, 0) + 1
            total_snapped += 1

    if total_snapped == 0:
        return {"snap_1_4": 0.0, "snap_1_6": 0.0, "snap_1_8": 0.0}

    return {
        "snap_1_4": round(snap_counts.get(4, 0) / total_snapped, 4),
        "snap_1_6": round(snap_counts.get(6, 0) / total_snapped, 4),
        "snap_1_8": round(snap_counts.get(8, 0) / total_snapped, 4),
    }


def infer_style_and_snaps(bm: TaikoBeatmap) -> tuple[int, dict[str, float]]:
    """
    Returns (style_int, snap_ratios_dict).

    Refactored from the original infer_style() so that snap computation
    is done once and shared between the style label and the stored ratios.
    The original infer_style() function is preserved below for backward
    compatibility with any external callers.
    """
    hit_notes = [n for n in bm.notes if not n.is_long]
    if not hit_notes or not bm.timing_points:
        return 0, {"snap_1_4": 0.0, "snap_1_6": 0.0, "snap_1_8": 0.0}

    weird_snaps   = {3, 5, 6, 7, 9, 12, 16}
    snap_counts   = {d: 0 for d in SNAP_DENOM}
    weird_count   = 0
    total_snapped = 0

    for note in hit_notes:
        denom = snap_of_note(note.time, bm.timing_points)
        if denom is not None:
            snap_counts[denom] = snap_counts.get(denom, 0) + 1
            total_snapped += 1
            if denom in weird_snaps:
                weird_count += 1

    snap_ratios = {
        "snap_1_4": round(snap_counts.get(4, 0) / max(total_snapped, 1), 4),
        "snap_1_6": round(snap_counts.get(6, 0) / max(total_snapped, 1), 4),
        "snap_1_8": round(snap_counts.get(8, 0) / max(total_snapped, 1), 4),
    }

    if total_snapped == 0:
        return 0, snap_ratios

    weird_ratio = weird_count / total_snapped
    if weird_ratio >= TECH_WEIRD_RATIO:
        return 3, snap_ratios   # tech

    eff_bpm      = effective_bpm(bm.timing_points, bm.notes)
    eighth_ratio = snap_counts.get(8, 0) / total_snapped
    if eighth_ratio > 0.10:
        eff_bpm *= 2
    if eff_bpm >= SPEED_EFFECTIVE_BPM:
        return 2, snap_ratios   # speed

    stream_runs = count_stream_runs(bm.notes, bm.timing_points)
    if stream_runs >= 3:
        return 1, snap_ratios   # stream

    return 0, snap_ratios       # standard


def infer_style(bm: TaikoBeatmap) -> int:
    """Backward-compatible wrapper — returns style int only."""
    style, _ = infer_style_and_snaps(bm)
    return style


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

def read_beatmapset_id_fast(osu_path: Path) -> int:
    try:
        with open(osu_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("BeatmapSetID:"):
                    v = line.split(":", 1)[1].strip()
                    return int(v) if v.isdigit() else 0
    except OSError:
        pass
    return 0


def print_cache_coverage(files: list[Path], cache: dict) -> None:
    set_ids: set[int] = set()
    for fp in files:
        sid = read_beatmapset_id_fast(fp)
        if sid > 0:
            set_ids.add(sid)
    in_cache    = [sid for sid in set_ids if str(sid) in cache]
    ranked_sets = sum(
        1 for sid in in_cache
        if isinstance(cache[str(sid)], dict) and cache[str(sid)].get("ranked")
    )
    missing = len(set_ids) - len(in_cache)
    print(f"Beatmapsets in filter file : {len(set_ids)} unique")
    print(f"  in beatmapset_cache      : {len(in_cache)}")
    print(f"  ranked sets in cache     : {ranked_sets}")
    print(f"  missing from cache       : {missing}")
    if missing and not OSU_API_KEY:
        print("  → Run: python scripts/populate_beatmapset_cache.py")
    print("Index is one row per .osu difficulty; ranked=True applies to ALL")
    print("difficulties when their beatmapset is ranked in the cache.")


def safe_name(s: str, max_len: int = 120) -> str:
    cleaned = re.sub(r'[\[\]\(\)\'"!@#$%^&*+=|\\/<>?:;,~`]', '', s)
    cleaned = cleaned.replace('.', '')
    cleaned = re.sub(r'[ _]{2,}', ' ', cleaned).strip()
    return cleaned[:max_len]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(use_api: bool = True):
    if not Path(CACHE_FILE).exists():
        print(f"ERROR: {CACHE_FILE} not found. Run fast_scan.py first.")
        return

    files = [Path(p) for p in json.loads(Path(CACHE_FILE).read_text(encoding="utf-8"))]
    print(f"Loaded {len(files)} filtered maps.")

    TENSORS_DIR.mkdir(parents=True, exist_ok=True)

    parser           = OsuTaikoParser()
    sr_cache         = load_sr_cache()
    beatmapset_cache = load_beatmapset_cache()
    session          = requests.Session()

    if use_api and not OSU_API_KEY:
        print("WARNING: OSU_API_KEY not found in .env — using beatmapset_cache only")
        use_api = False

    if beatmapset_cache:
        print(f"Loaded beatmapset cache: {len(beatmapset_cache)} entries")
    elif not use_api:
        print("WARNING: beatmapset_cache.json missing and --no-api set")

    print_cache_coverage(files, beatmapset_cache)

    records      = []
    errors       = []
    skipped      = 0
    api_calls    = 0
    cache_hits   = 0
    cache_misses = 0
    t_start      = time.time()

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

        # Tensor path
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

        # ---- Star rating + ranked status --------------------------------- #
        star_rating  = 0.0
        approved     = "0"
        approved_str = "unknown"
        is_ranked    = False

        set_data = lookup_beatmapset(bm.beatmap_set_id, beatmapset_cache)

        if set_data is not None:
            cache_hits += 1
        elif bm.beatmap_set_id > 0:
            cache_misses += 1

        if set_data is None and use_api and bm.beatmap_set_id > 0:
            set_key = str(bm.beatmap_set_id)
            if set_key not in beatmapset_cache:
                api_calls += 1
                if api_calls % 100 == 0:
                    save_beatmapset_cache(beatmapset_cache)
                time.sleep(API_DELAY)
            set_data = fetch_beatmapset(bm.beatmap_set_id, beatmapset_cache, session)

        if set_data:
            star_rating, approved, approved_str, is_ranked = metadata_from_set_data(
                set_data, bm.beatmap_id
            )

        if star_rating == 0.0 and str(bm.beatmap_id) in sr_cache:
            star_rating = sr_cache[str(bm.beatmap_id)]

        if star_rating == 0.0:
            star_rating = bm.overall_difficulty

        # ---- NPS fields -------------------------------------------------- #
        avg_nps  = compute_avg_nps(bm)
        peak_nps = compute_peak_nps(bm)

        # ---- Style + snap ratios ----------------------------------------- #
        style, snap_ratios = infer_style_and_snaps(bm)

        records.append({
            "mel_path":    str(mel_path.relative_to("data/processed")),
            "tensor_path": str(tensor_path.relative_to("data/processed")),
            "difficulty":  round(star_rating, 2),
            "style":       style,
            "style_name":  STYLE_NAMES[style],
            "note_count":  bm.note_count,
            "duration_ms": bm.duration_ms,
            # NPS conditioning
            "avg_nps":     avg_nps,
            "peak_nps":    peak_nps,
            # Snap ratio conditioning
            "snap_1_4":    snap_ratios["snap_1_4"],
            "snap_1_6":    snap_ratios["snap_1_6"],
            "snap_1_8":    snap_ratios["snap_1_8"],
            # Metadata
            "title":        bm.title,
            "version":      bm.version,
            "beatmap_id":   bm.beatmap_id,
            "beatmapset_id": bm.beatmap_set_id,
            "approved":     approved,
            "approved_str": approved_str,
            "ranked":       is_ranked,
            "creator":      bm.creator,
        })

    # Final saves
    save_beatmapset_cache(beatmapset_cache)

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Summary
    total_time     = time.time() - t_start
    style_dist     = {name: 0 for name in STYLE_NAMES.values()}
    ranked_count   = sum(1 for r in records if r["ranked"])
    unranked_count = len(records) - ranked_count

    for rec in records:
        style_dist[rec["style_name"]] += 1

    avg_nps_all  = [r["avg_nps"]  for r in records if r["avg_nps"]  > 0]
    peak_nps_all = [r["peak_nps"] for r in records if r["peak_nps"] > 0]

    print(f"\n{'='*50}")
    print(f"Done in {total_time/60:.1f} minutes")
    print(f"Records  : {len(records)}")
    print(f"Ranked   : {ranked_count}")
    print(f"Unranked : {unranked_count}")
    print(f"Errors   : {len(errors)}")
    print(f"Skipped  : {skipped}")
    print(f"Cache hits  : {cache_hits}")
    print(f"Cache misses: {cache_misses}")
    print(f"API calls   : {api_calls}")

    if avg_nps_all:
        print(f"\nNPS stats (avg / peak):")
        print(f"  avg_nps  — mean: {sum(avg_nps_all)/len(avg_nps_all):.2f}  "
              f"max: {max(avg_nps_all):.2f}")
        print(f"  peak_nps — mean: {sum(peak_nps_all)/len(peak_nps_all):.2f}  "
              f"max: {max(peak_nps_all):.2f}")

    print(f"\nStyle distribution:")
    for name, count in style_dist.items():
        pct = count / max(len(records), 1) * 100
        print(f"  {name:10s}: {count:5d}  ({pct:.1f}%)")

    snap_1_4_all = [r["snap_1_4"] for r in records]
    snap_1_8_all = [r["snap_1_8"] for r in records]
    if snap_1_4_all:
        print(f"\nSnap distribution (mean across index):")
        print(f"  snap_1_4: {sum(snap_1_4_all)/len(snap_1_4_all):.3f}")
        print(f"  snap_1_8: {sum(snap_1_8_all)/len(snap_1_8_all):.3f}")

    tensor_files = list(TENSORS_DIR.glob("*.npz"))
    total_mb     = sum(f.stat().st_size for f in tensor_files) / 1024**2
    print(f"\nTensor folder size: {total_mb:.0f} MB ({total_mb/1024:.2f} GB)")

    if errors:
        err_path = Path("data/processed/tensor_errors.txt")
        err_path.write_text("\n".join(errors), encoding="utf-8")
        print(f"Errors saved to: {err_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-api", action="store_true",
                    help="Skip osu! API fetch; still read ranked/SR from beatmapset_cache.json")
    args = ap.parse_args()
    main(use_api=not args.no_api)