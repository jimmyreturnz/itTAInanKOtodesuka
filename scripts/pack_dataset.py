"""
scripts/pack_dataset.py

LOCAL step. Turns an osu! Songs folder into the packed dataset you upload.

    .osu + audio  ->  data/processed/shards/{mels.dat, charts.npz, index.json}

Run this on the machine that has the maps. It is the only step that needs the
audio files, and it is the slowest part of the whole project -- mel extraction
for 13k maps takes a few hours -- so it is built to be interrupted and resumed.

    python scripts/pack_dataset.py --scan "D:/osu!/Songs"
    python scripts/pack_dataset.py                    # resume with cached scan
    python scripts/pack_dataset.py --limit 200        # a small set to test with

Two passes, because mels are shared between the difficulties of a beatmapset
and are far more expensive than charts. Pass 1 caches one mel per song folder;
pass 2 parses every difficulty and writes the shards. Interrupting during pass 1
loses nothing -- the cache is per song.

Star ratings come from the osu! API when a key is available, and from
beatmapset_cache.json otherwise. Maps with no known star rating are DROPPED
rather than falling back to OverallDifficulty: OD is an accuracy parameter with
no relation to a star rating, and feeding it in as one poisons the single
conditioning signal the user is most likely to reach for.
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

import numpy as np

from taiko.data.audio import MelExtractor
from taiko.data.conditioning import STYLE_NAMES
from taiko.data.frames import FRAME_MS, describe
from taiko.data.osu_parser import OsuTaikoParser
from taiko.data.shards import ShardWriter
from taiko.data.tensor_repr import beatmap_to_tensors

try:
    import requests
except ImportError:
    requests = None

DEFAULT_SCAN_CACHE = Path("taiko_files_filtered.json")
MEL_CACHE_DIR      = Path("data/processed/mel_cache")
SHARD_DIR          = Path("data/processed/shards")
BEATMAPSET_CACHE   = Path("data/processed/beatmapset_cache.json")

OSU_API_V1 = "https://osu.ppy.sh/api/get_beatmaps"
API_DELAY  = 0.5

MIN_NOTES         = 30
MIN_DURATION_MS   = 20_000
MAX_AUDIO_BYTES   = 150 * 1024 * 1024

SNAP_TOLERANCE_MS      = 8.0
STREAM_MIN_CONSECUTIVE = 8
SPEED_EFFECTIVE_BPM    = 270
TECH_WEIRD_RATIO       = 0.12
SNAP_DENOMS            = (1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16)
PEAK_NPS_WINDOW_MS     = 5_000


# --------------------------------------------------------------------------- #
# Scanning
# --------------------------------------------------------------------------- #

def scan_songs(root: Path) -> list[Path]:
    """Find every taiko-mode .osu under a Songs folder."""
    found: list[Path] = []
    for i, path in enumerate(root.rglob("*.osu")):
        if i % 5000 == 0 and i:
            print(f"  scanned {i} files, {len(found)} taiko so far")
        try:
            # Mode appears in the first few lines; no need to read whole files.
            with open(path, encoding="utf-8", errors="replace") as handle:
                head = handle.read(2048)
            if "Mode: 1" in head or "Mode:1" in head:
                found.append(path)
        except OSError:
            continue
    return found


def safe_name(text: str, max_len: int = 120) -> str:
    """A folder name that survives Windows, Linux and Kaggle alike."""
    cleaned = re.sub(r'[\[\]\(\)\'"!@#$%^&*+=|\\/<>?:;,~`]', "", text)
    cleaned = cleaned.replace(".", "")
    cleaned = re.sub(r"[ _]{2,}", " ", cleaned).strip()
    return cleaned[:max_len]


def find_audio(folder: Path) -> Path | None:
    for ext in (".mp3", ".ogg", ".wav", ".flac"):
        matches = sorted(folder.glob(f"*{ext}"))
        if matches:
            return matches[0]
    return None


# --------------------------------------------------------------------------- #
# Derived statistics
# --------------------------------------------------------------------------- #

def compute_avg_nps(bm) -> float:
    hits = sorted(n.time for n in bm.notes if not n.is_long)
    if len(hits) < 2:
        return 0.0
    span = (hits[-1] - hits[0]) / 1000.0
    if span < 1.0:
        return 0.0
    return round(min(len(hits) / span, 30.0), 3)


def compute_peak_nps(bm, window_ms: int = PEAK_NPS_WINDOW_MS) -> float:
    hits = sorted(n.time for n in bm.notes if not n.is_long)
    if not hits:
        return 0.0
    window_sec = window_ms / 1000.0
    peak, left = 0.0, 0
    for right in range(len(hits)):
        while hits[right] - hits[left] > window_ms:
            left += 1
        peak = max(peak, (right - left + 1) / window_sec)
    return round(min(peak, 30.0), 3)


def active_red_line(timing_points, time_ms):
    active = None
    for tp in timing_points:
        if tp.time > time_ms:
            break
        if tp.uninherited and tp.beat_length > 0:
            active = tp
    return active


def snap_of(note_time: int, timing_points) -> int | None:
    tp = active_red_line(timing_points, note_time)
    if tp is None:
        return None
    offset = (note_time - tp.time) % tp.beat_length
    best, best_error = None, SNAP_TOLERANCE_MS
    for denom in SNAP_DENOMS:
        grid = tp.beat_length / denom
        error = abs(offset - round(offset / grid) * grid)
        if error < best_error:
            best, best_error = denom, error
    return best


def dominant_bpm(bm) -> float:
    """BPM of the red line that governs the most notes, not simply the first."""
    reds = [tp for tp in bm.timing_points if tp.uninherited and tp.beat_length > 0]
    if not reds:
        return 0.0
    hits = [n.time for n in bm.notes if not n.is_long]
    best_bpm, best_count = 0.0, -1
    for i, tp in enumerate(reds):
        end = reds[i + 1].time if i + 1 < len(reds) else float("inf")
        count = sum(1 for t in hits if tp.time <= t < end)
        if count > best_count:
            best_bpm, best_count = 60_000.0 / tp.beat_length, count
    return best_bpm


def infer_style_and_snaps(bm) -> tuple[int, dict[str, float]]:
    hits = [n for n in bm.notes if not n.is_long]
    empty = {"snap_1_4": 0.0, "snap_1_6": 0.0, "snap_1_8": 0.0}
    if not hits or not bm.timing_points:
        return 0, empty

    counts: dict[int, int] = {}
    weird = 0
    total = 0
    for note in hits:
        denom = snap_of(note.time, bm.timing_points)
        if denom is None:
            continue
        counts[denom] = counts.get(denom, 0) + 1
        total += 1
        if denom in (3, 5, 6, 7, 9, 12, 16):
            weird += 1

    if total == 0:
        return 0, empty

    snaps = {
        "snap_1_4": round(counts.get(4, 0) / total, 4),
        "snap_1_6": round(counts.get(6, 0) / total, 4),
        "snap_1_8": round(counts.get(8, 0) / total, 4),
    }

    if weird / total >= TECH_WEIRD_RATIO:
        return 3, snaps

    bpm = dominant_bpm(bm)
    if counts.get(8, 0) / total > 0.10:
        bpm *= 2
    if bpm >= SPEED_EFFECTIVE_BPM:
        return 2, snaps

    runs, run = 0, 1
    for i in range(1, len(hits)):
        tp = active_red_line(bm.timing_points, hits[i].time)
        if tp is None:
            run = 1
            continue
        quarter = tp.beat_length / 4.0
        if abs(hits[i].time - hits[i - 1].time - quarter) <= SNAP_TOLERANCE_MS:
            run += 1
        else:
            runs += run >= STREAM_MIN_CONSECUTIVE
            run = 1
    runs += run >= STREAM_MIN_CONSECUTIVE

    return (1 if runs >= 3 else 0), snaps


# --------------------------------------------------------------------------- #
# Star ratings
# --------------------------------------------------------------------------- #

def load_beatmapset_cache() -> dict:
    if BEATMAPSET_CACHE.exists():
        return json.loads(BEATMAPSET_CACHE.read_text(encoding="utf-8"))
    return {}


def save_beatmapset_cache(cache: dict) -> None:
    BEATMAPSET_CACHE.parent.mkdir(parents=True, exist_ok=True)
    BEATMAPSET_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


# osu! API v1 `approved`: 1 ranked, 2 approved, 3 qualified, 4 loved,
# 0 pending, -1 wip, -2 graveyard.
#
# Loved and qualified are deliberately excluded (DIRECTION.md D2). Loved is a
# popularity vote, not a quality bar, and carries gimmick and SV-abuse maps --
# star rating cannot filter them out because star rating is what they inflate:
# a loved map in the first pack scored 18.16* on 8.81 nps, less dense than the
# genuine 11* charts beneath it. Qualified is not final and can be disqualified.
RANKED_APPROVED = ("1", "2")

APPROVED_NAMES = {
    "-2": "graveyard", "-1": "wip", "0": "pending",
    "1": "ranked", "2": "approved", "3": "qualified", "4": "loved",
}


def fetch_beatmapset(set_id: int, cache: dict, session, api_key: str) -> dict | None:
    key = str(set_id)
    if key in cache:
        return cache[key]
    if not api_key or set_id <= 0 or session is None:
        return None
    try:
        response = session.get(
            OSU_API_V1, params={"k": api_key, "s": set_id, "m": 1}, timeout=15,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:                                   # noqa: BLE001
        print(f"  [api] set {set_id}: {exc}")
        return None

    if not data:
        cache[key] = None
        return None

    approved = str(data[0].get("approved", "0"))
    entry = {
        "approved": approved,
        "ranked": approved in RANKED_APPROVED,
        "approved_str": APPROVED_NAMES.get(approved, approved),
        "difficulties": {
            str(d.get("beatmap_id", "")): {
                "sr": float(d.get("difficultyrating", 0.0)),
                "version": d.get("version", ""),
            }
            for d in data if d.get("beatmap_id")
        },
    }
    cache[key] = entry
    time.sleep(API_DELAY)
    return entry


def is_ranked(entry) -> bool:
    """Ranked or approved. Reads `approved` where present, else `approved_str`."""
    if not isinstance(entry, dict):
        return False
    approved = entry.get("approved")
    if approved is not None:
        return str(approved) in RANKED_APPROVED
    return entry.get("approved_str") in ("ranked", "approved")


def _sr_value(raw) -> float:
    """
    Star rating out of either cache layout.

    populate_beatmapset_cache.py writes {"sr": float, "version": str} per
    difficulty; older entries here wrote a bare float. Reading the dict form
    with float() raises TypeError, which is uncaught and kills pass 2 on the
    first cached ranked map, so both layouts are accepted.
    """
    if isinstance(raw, dict):
        raw = raw.get("sr", 0.0)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def star_rating_for(bm, cache: dict, session, api_key: str) -> tuple[float, bool]:
    entry = cache.get(str(bm.beatmap_set_id))
    if entry is None and api_key:
        entry = fetch_beatmapset(bm.beatmap_set_id, cache, session, api_key)
    if not isinstance(entry, dict):
        return 0.0, False
    sr = _sr_value(entry.get("difficulties", {}).get(str(bm.beatmap_id)))
    return sr, is_ranked(entry)


def _set_id_from_header(path) -> int:
    """BeatmapSetID without a full parse -- it sits in [Metadata], near the top."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for _ in range(80):
                line = fh.readline()
                if not line or line.startswith("[HitObjects]"):
                    break
                if line.startswith("BeatmapSetID:"):
                    return int(line.split(":", 1)[1].strip() or -1)
    except Exception:                                          # noqa: BLE001
        pass
    return -1


def drop_unranked_folders(by_folder: dict, cache: dict) -> int:
    """
    Drop folders that contain no ranked map, before mel extraction pays for them.

    Only the cache is consulted, and a folder whose sets it does not know is
    kept -- pass 2 makes the real decision once the map is parsed and the API
    can be asked. This is an optimisation, never the filter of record.
    """
    dropped = 0
    for folder in list(by_folder):
        keep = False
        for path in by_folder[folder]:
            entry = cache.get(str(_set_id_from_header(path)))
            if not isinstance(entry, dict) or is_ranked(entry):
                keep = True          # unknown or ranked -- pass 2 decides
                break
        if not keep:
            del by_folder[folder]
            dropped += 1
    return dropped


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="Pack an osu! Songs folder into training shards")
    ap.add_argument("--scan", type=str, default=None,
                    help="Path to your osu! Songs folder. Omit to reuse the cached scan.")
    ap.add_argument("--scan-cache", type=Path, default=DEFAULT_SCAN_CACHE)
    ap.add_argument("--out", type=Path, default=SHARD_DIR)
    ap.add_argument("--mel-cache", type=Path, default=MEL_CACHE_DIR)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process this many maps. Use a small number first.")
    ap.add_argument("--no-api", action="store_true", help="Never call the osu! API")
    ap.add_argument("--ranked-only", action="store_true",
                    help="pack only ranked maps (D2 -- unranked carries rate-ups, "
                         "which pair one chart with differently-timed audio)")
    ap.add_argument("--keep-unrated", action="store_true",
                    help="Keep maps with no known star rating (not recommended)")
    args = ap.parse_args()

    print(describe())

    # ---- file list ------------------------------------------------------- #
    if args.scan:
        root = Path(args.scan)
        if not root.is_dir():
            print(f"ERROR: not a directory: {root}")
            return 1
        print(f"Scanning {root} ...")
        files = scan_songs(root)
        args.scan_cache.write_text(
            json.dumps([str(p) for p in files], ensure_ascii=False), encoding="utf-8"
        )
        print(f"Found {len(files)} taiko difficulties -> {args.scan_cache}")
    else:
        if not args.scan_cache.exists():
            print(f"ERROR: no scan cache at {args.scan_cache}.")
            print('  Run once with --scan "D:/osu!/Songs"')
            return 1
        files = [Path(p) for p in json.loads(args.scan_cache.read_text(encoding="utf-8"))]
        print(f"Loaded {len(files)} maps from {args.scan_cache}")

    missing = [p for p in files[:50] if not p.exists()]
    if len(missing) == min(50, len(files)):
        print("ERROR: none of the cached paths exist on this machine.")
        print('  The cache was built elsewhere. Re-run with --scan "<your Songs folder>".')
        return 1

    if args.limit:
        files = files[:args.limit]
        print(f"Limited to {len(files)} maps")

    by_folder: dict[Path, list[Path]] = defaultdict(list)
    for path in files:
        by_folder[path.parent].append(path)
    print(f"{len(by_folder)} beatmapset folders")

    # populate_beatmapset_cache.py reads .env; this did not, so a key kept there
    # was silently ignored and every uncached set was dropped as "no star rating".
    if not args.no_api and not os.environ.get("OSU_API_KEY"):
        try:
            from dotenv import load_dotenv
            load_dotenv(Path(__file__).parent.parent / ".env")
        except ImportError:
            pass

    api_key = None if args.no_api else os.environ.get("OSU_API_KEY")
    session = requests.Session() if (api_key and requests) else None
    if api_key and not requests:
        print("WARNING: OSU_API_KEY set but `requests` is not installed")
    cache = load_beatmapset_cache()
    print(f"Beatmapset cache: {len(cache)} entries"
          f"{'  (API enabled)' if api_key else '  (offline)'}")

    if args.ranked_only:
        dropped = drop_unranked_folders(by_folder, cache)
        print(f"ranked-only: {dropped} folders hold no ranked map, skipping their "
              f"mel extraction ({len(by_folder)} folders remain)")

    # ---- pass 1: mels ---------------------------------------------------- #
    args.mel_cache.mkdir(parents=True, exist_ok=True)
    extractor = MelExtractor()
    parser = OsuTaikoParser()

    print(f"\nPass 1/2  mel extraction -> {args.mel_cache}")
    t0 = time.time()
    extracted = cached = failed = 0

    for i, folder in enumerate(sorted(by_folder)):
        if i % 100 == 0 and i:
            rate = (time.time() - t0) / i
            print(f"  [{i}/{len(by_folder)}] new={extracted} cached={cached} "
                  f"failed={failed}  eta {rate * (len(by_folder) - i) / 60:.0f} min")

        mel_path = args.mel_cache / f"{safe_name(folder.name)}.npy"
        if mel_path.exists():
            cached += 1
            continue

        audio = find_audio(folder)
        if audio is None:
            failed += 1
            continue
        try:
            if audio.stat().st_size > MAX_AUDIO_BYTES:
                failed += 1
                continue
            mel = extractor.extract(audio)
            # float16 on disk: the values feed a network training in fp16, so
            # the second byte buys nothing and this halves a large cache.
            np.save(mel_path, mel.astype(np.float16))
            extracted += 1
        except Exception as exc:                               # noqa: BLE001
            print(f"  [mel] {folder.name}: {exc}")
            failed += 1

    print(f"Pass 1 done in {(time.time() - t0) / 60:.1f} min  "
          f"(new {extracted}, cached {cached}, failed {failed})")

    # ---- pass 2: charts + shards ----------------------------------------- #
    print(f"\nPass 2/2  parsing and packing -> {args.out}")
    t0 = time.time()
    written = 0
    skipped: dict[str, int] = defaultdict(int)

    with ShardWriter(args.out) as writer:
        for i, folder in enumerate(sorted(by_folder)):
            if i % 200 == 0 and i:
                print(f"  [{i}/{len(by_folder)}] packed {written} maps")

            mel_key = safe_name(folder.name)
            mel_path = args.mel_cache / f"{mel_key}.npy"
            if not mel_path.exists():
                skipped["no mel"] += len(by_folder[folder])
                continue

            mel = np.load(mel_path).astype(np.float32)

            pending = []
            for osu_path in sorted(by_folder[folder]):
                try:
                    bm = parser.parse_file(osu_path)
                except Exception:                              # noqa: BLE001
                    skipped["parse error"] += 1
                    continue

                if bm.note_count < MIN_NOTES:
                    skipped["too few notes"] += 1
                    continue
                if bm.duration_ms < MIN_DURATION_MS:
                    skipped["too short"] += 1
                    continue
                if not any(tp.uninherited and tp.beat_length > 0 for tp in bm.timing_points):
                    skipped["no timing"] += 1
                    continue

                sr, ranked = star_rating_for(bm, cache, session, api_key)
                if args.ranked_only and not ranked:
                    skipped["not ranked"] += 1
                    continue
                if sr <= 0 and not args.keep_unrated:
                    skipped["no star rating"] += 1
                    continue

                chart, _ = beatmap_to_tensors(bm)

                # The chart must not claim to run past the end of the audio;
                # that is the alignment contract, checked here at the source.
                if chart.shape[1] > mel.shape[1] + 250:
                    skipped["chart longer than audio"] += 1
                    continue

                style, snaps = infer_style_and_snaps(bm)
                pending.append((bm, chart, {
                    "difficulty":  round(sr, 3),
                    "style":       style,
                    "style_name":  STYLE_NAMES[style],
                    "ranked":      ranked,
                    "avg_nps":     compute_avg_nps(bm),
                    "peak_nps":    compute_peak_nps(bm),
                    "bpm":         round(dominant_bpm(bm), 3),
                    "note_count":  bm.note_count,
                    "duration_ms": bm.duration_ms,
                    "title":       bm.title,
                    "version":     bm.version,
                    "creator":     bm.creator,
                    "beatmap_id":  bm.beatmap_id,
                    "beatmapset_id": bm.beatmap_set_id,
                    **snaps,
                }))

            if not pending:
                continue

            writer.add_mel(mel_key, mel)
            for bm, chart, record in pending:
                writer.add_map(record, chart, bm.timing_points, mel_key)
                written += 1

        stats = writer.close()

    save_beatmapset_cache(cache)

    # ---- summary ---------------------------------------------------------- #
    print(f"\nPass 2 done in {(time.time() - t0) / 60:.1f} min")
    print(f"{'=' * 60}")
    print(f"Packed  : {stats['maps']} maps over {stats['songs']} songs")
    print(f"Audio   : {stats['mel_frames']:,} frames "
          f"({stats['mel_frames'] * FRAME_MS / 1000 / 3600:.1f} hours, "
          f"{stats['mel_gb']:.2f} GB)")
    print(f"Events  : {stats['onsets']:,} onsets, {stats['spans']:,} long notes")
    if skipped:
        print("Skipped :")
        for reason, count in sorted(skipped.items(), key=lambda kv: -kv[1]):
            print(f"    {count:>7,}  {reason}")

    total_bytes = sum(f.stat().st_size for f in args.out.iterdir() if f.is_file())
    print(f"\nUpload this folder to Kaggle: {args.out}  ({total_bytes / 1024**3:.2f} GB)")
    for f in sorted(args.out.iterdir()):
        print(f"    {f.name:<14s} {f.stat().st_size / 1024**2:>9.1f} MB")

    if stats["maps"] == 0:
        print("\nNothing was packed. Check the skip reasons above.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
