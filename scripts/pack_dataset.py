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

Ranked means the osu! API says ranked AND the local .osu's md5 is the ranked
file's md5 (RETRAIN_PLAN 10h): folders also hold rate edits and cut versions
that keep the ranked map's BeatmapID. One record per beatmap ID; the audio is
the AudioFilename of that verified .osu. The star rating is osu!.db's stable
SR of exactly that file (NM, plus DT and HT for rate augmentation). Every
candidate that is not packed is written to pack_report.json with its reason.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from taiko.data.audio import MelExtractor
from taiko.data.conditioning import STYLE_NAMES
from taiko.data import osu_db
from taiko.data.frames import FRAME_MS, describe, ms_to_frame
from taiko.data.osu_parser import OsuTaikoParser
from taiko.data.shards import ShardWriter, write_note_times
from taiko.data.tensor_repr import beatmap_to_tensors

try:
    import requests
except ImportError:
    requests = None

DEFAULT_SCAN_CACHE = Path("taiko_files_filtered.json")
MEL_CACHE_DIR      = Path("data/processed/mel_cache")
SHARD_DIR          = Path("data/processed/shards")
BEATMAPSET_CACHE   = Path("data/processed/beatmapset_cache.json")
PACK_REPORT        = Path("data/processed/pack_report.json")
OSU_DB             = Path("D:/osu!/osu!.db")
VAL_MODULUS        = 20          # a new beatmapset goes to val when crc32(id) % 20 == 0: 5%

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


def named_audio(osu_path: Path) -> str:
    """The AudioFilename an .osu names, read from its header."""
    with open(osu_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("AudioFilename:"):
                return line.split(":", 1)[1].strip()
            if line.startswith("[Metadata]"):
                break
    return ""


def find_audio(folder: Path) -> Path | None:
    """
    The audio the folder's taiko difficulties are timed to: the AudioFilename
    most of its taiko .osu files name. Taking the first .mp3 instead packed 8
    songs against a rate-edited copy in the same folder ("audio 0.650x
    withDT.mp3"), so their charts trained on audio playing at another speed.
    A vote, not the first .osu, because the first can be the rate edit itself.
    """
    votes: Counter[str] = Counter()
    for osu in folder.glob("*.osu"):
        try:
            with open(osu, encoding="utf-8", errors="replace") as handle:
                head = handle.read(2048)
        except OSError:
            continue
        if "Mode: 1" in head or "Mode:1" in head:
            votes[named_audio(osu)] += 1
    for name, _ in votes.most_common():
        if name and (folder / name).is_file():
            return folder / name
    for ext in (".mp3", ".ogg", ".wav", ".flac"):
        matches = sorted(folder.glob(f"*{ext}"))
        if matches:
            return matches[0]
    return None


def legacy_audio(folder: Path) -> Path | None:
    """What the Aug 30 pack took as a folder's audio: its first .mp3. Its mel
    cache is keyed by folder alone, so it is reused only when this matches."""
    for ext in (".mp3", ".ogg", ".wav", ".flac"):
        matches = sorted(folder.glob(f"*{ext}"))
        if matches:
            return matches[0]
    return None


def mel_key_for(audio: Path) -> str:
    """One mel per audio file. Keyed by folder when it is the file the old
    cache holds, so its mels are reused, and by folder + file otherwise."""
    old = legacy_audio(audio.parent)
    if old is not None and old.name.lower() == audio.name.lower():
        return safe_name(audio.parent.name)
    return safe_name(f"{audio.parent.name} {audio.stem}")


def assign_split(set_id: int, beatmap_ids, old: dict) -> str:
    """
    Train or val per beatmapset, so two folders of one set land on one side.
    A set the previous index trained on stays train and its val stays val, so
    a model warm-started from it is never scored on songs it learned; a new
    set goes by a stable hash. Matched by set ID and by beatmap ID, because
    old headers can carry no set ID.
    """
    import zlib
    for side in ("train", "val"):
        if set_id in old[side + "_sets"] or any(b in old[side + "_ids"] for b in beatmap_ids):
            return side
    return "val" if zlib.crc32(str(set_id).encode()) % VAL_MODULUS == 0 else "train"


def old_split(index_path: Path) -> dict:
    """Set and beatmap IDs on each side of the index being replaced."""
    from types import SimpleNamespace
    from taiko.data.preprocessed_dataset import split_indices
    old = {"train_sets": set(), "val_sets": set(), "train_ids": set(), "val_ids": set()}
    if not index_path.exists():
        return old
    records = json.loads(index_path.read_text(encoding="utf-8"))["records"]
    train, val = split_indices(SimpleNamespace(records=records), val_ratio=0.05)
    for side, idx in (("train", train), ("val", val)):
        for i in idx:
            r = records[i]
            if int(r.get("beatmapset_id", 0)) > 0:
                old[side + "_sets"].add(int(r["beatmapset_id"]))
            if int(r.get("beatmap_id", 0)) > 0:
                old[side + "_ids"].add(int(r["beatmap_id"]))
    return old


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
# Ranked and nothing else (DIRECTION.md D2). Loved is a popularity vote, not a
# quality bar, and carries gimmick and SV-abuse maps -- star rating cannot
# filter them out because star rating is what they inflate: a loved map in the
# first pack scored 18.16* on 8.81 nps, less dense than the genuine 11* charts
# beneath it. Qualified is not final and can be disqualified. Approved is the
# legacy marathon status; it is ranked-grade, but "ranked" is the line drawn.
RANKED_APPROVED = ("1",)

APPROVED_NAMES = {
    "-2": "graveyard", "-1": "wip", "0": "pending",
    "1": "ranked", "2": "approved", "3": "qualified", "4": "loved",
}


def fetch_beatmapset(set_id: int, cache: dict, session, api_key: str) -> dict | None:
    key = str(set_id)
    if key in cache and _has_md5(cache[key]):
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
                "md5": d.get("file_md5", ""),
                "approved": str(d.get("approved", approved)),
            }
            for d in data if d.get("beatmap_id")
        },
    }
    cache[key] = entry
    time.sleep(API_DELAY)
    return entry


def _has_md5(entry) -> bool:
    """Entries written before 10h carry no file_md5 and must be fetched again."""
    if entry is None:
        return True          # the API knows no such set; asking again will not help
    diffs = entry.get("difficulties", {}) if isinstance(entry, dict) else {}
    return bool(diffs) and all(isinstance(d, dict) and "md5" in d for d in diffs.values())


def ranked_beatmap_id(entry, md5: str) -> int | None:
    """The beatmap ID whose ranked file has this md5, or None."""
    if not isinstance(entry, dict):
        return None
    for bid, d in entry.get("difficulties", {}).items():
        if isinstance(d, dict) and d.get("md5") == md5:
            return int(bid) if d.get("approved", entry.get("approved")) in RANKED_APPROVED else None
    return None


def is_ranked(entry) -> bool:
    """Ranked only. Reads `approved` where present, else `approved_str`."""
    if not isinstance(entry, dict):
        return False
    approved = entry.get("approved")
    if approved is not None:
        return str(approved) in RANKED_APPROVED
    return entry.get("approved_str") == "ranked"


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


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def verify(files: list[Path], db: dict, cache: dict, session, api_key: str,
           skips: list[dict]) -> list[dict]:
    """
    The ranked, md5-verified candidates, one per beatmap ID. Everything else
    goes to `skips` with its reason. Rate edits and cut versions keep the
    ranked map's BeatmapID, so the header's ID proves nothing; the md5 does.
    """
    def skip(path, reason, **extra):
        skips.append({"folder": path.parent.name, "osu_file": path.name, "reason": reason, **extra})

    by_id: dict[int, list[dict]] = defaultdict(list)
    for i, path in enumerate(files):
        if i % 2000 == 0 and i:
            print(f"  [{i}/{len(files)}] verified, {len(by_id)} ranked beatmap IDs")
            save_beatmapset_cache(cache)      # a killed run keeps its API fetches
        md5 = osu_db.file_md5(path)
        entry = db.get(md5) if md5 else None
        if entry is None:
            skip(path, "not in osu!.db (edited after osu! last scanned it, or unreadable)")
            continue
        set_id = entry.set_id if entry.set_id > 0 else _set_id_from_header(path)
        if set_id <= 0:
            skip(path, "no beatmapset ID (never submitted)")
            continue
        api = fetch_beatmapset(set_id, cache, session, api_key)
        if api is None or not _has_md5(api):
            skip(path, "API has no md5 for this set" if api else "set unknown to the API (or offline)")
            continue
        bid = ranked_beatmap_id(api, md5)
        if bid is None:
            skip(path, "md5 is not a ranked file (rate edit, cut, old version or unranked set)")
            continue
        if osu_db.NM not in entry.taiko_stars:
            skip(path, "no taiko SR in osu!.db")
            continue
        by_id[bid].append({"path": path, "md5": md5, "db": entry,
                           "beatmap_id": bid, "beatmapset_id": set_id})

    chosen = []
    for bid, copies in by_id.items():
        # A second folder holding the same ranked file is packed once: the copy
        # osu!.db lists, else the first by path.
        copies.sort(key=lambda c: (c["path"].parent.name != c["db"].folder, str(c["path"])))
        chosen.append(copies[0])
        for c in copies[1:]:
            skip(c["path"], "duplicate of a packed beatmap ID", beatmap_id=bid,
                 packed=f"{copies[0]['path'].parent.name}/{copies[0]['path'].name}")
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser(description="Pack an osu! Songs folder into training shards")
    ap.add_argument("--scan", type=str, default=None,
                    help="Path to your osu! Songs folder. Omit to reuse the cached scan.")
    ap.add_argument("--scan-cache", type=Path, default=DEFAULT_SCAN_CACHE)
    ap.add_argument("--out", type=Path, default=SHARD_DIR)
    ap.add_argument("--mel-cache", type=Path, default=MEL_CACHE_DIR)
    ap.add_argument("--osu-db", type=Path, default=OSU_DB)
    ap.add_argument("--keep-split-from", type=Path, default=None,
                    help="index.json whose train/val sides carry over (default: the one in --out)")
    ap.add_argument("--report", type=Path, default=PACK_REPORT)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process this many maps. Use a small number first.")
    ap.add_argument("--no-api", action="store_true", help="Never call the osu! API")
    ap.add_argument("--ranked-only", action="store_true",
                    help="no-op: every pack is ranked-only (10h); kept for old commands")
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

    db = osu_db.read(args.osu_db)
    print(f"osu!.db: {len(db)} beatmaps")

    # Read before the writer truncates the old shards.
    old = old_split(args.keep_split_from or args.out / "index.json")
    print(f"Split carried over: {len(old['train_sets'])} train / {len(old['val_sets'])} val sets")

    # ---- pass 0: which files are the ranked ones ------------------------- #
    print("\nPass 0/2  md5 verification")
    skips: list[dict] = []
    try:
        chosen = verify(files, db, cache, session, api_key, skips)
    finally:
        save_beatmapset_cache(cache)
    print(f"{len(chosen)} ranked beatmap IDs, {len(skips)} files skipped")

    by_mel: dict[str, list[dict]] = defaultdict(list)
    audio_of: dict[str, Path] = {}
    for c in chosen:
        audio = c["path"].parent / named_audio(c["path"])
        if not audio.is_file():
            skips.append({"folder": c["path"].parent.name, "osu_file": c["path"].name,
                          "beatmap_id": c["beatmap_id"], "reason": "its AudioFilename is not in the folder"})
            continue
        key = mel_key_for(audio)
        audio_of[key] = audio
        by_mel[key].append(c)
    print(f"{len(by_mel)} audio files")

    # ---- pass 1: mels ---------------------------------------------------- #
    args.mel_cache.mkdir(parents=True, exist_ok=True)
    extractor = MelExtractor()
    parser = OsuTaikoParser()

    print(f"\nPass 1/2  mel extraction -> {args.mel_cache}")
    t0 = time.time()
    extracted = cached = failed = 0

    for i, key in enumerate(sorted(by_mel)):
        if i % 100 == 0 and i:
            rate = (time.time() - t0) / i
            print(f"  [{i}/{len(by_mel)}] new={extracted} cached={cached} "
                  f"failed={failed}  eta {rate * (len(by_mel) - i) / 60:.0f} min")

        mel_path = args.mel_cache / f"{key}.npy"
        if mel_path.exists():
            cached += 1
            continue
        audio = audio_of[key]
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
            print(f"  [mel] {audio}: {exc}")
            failed += 1

    print(f"Pass 1 done in {(time.time() - t0) / 60:.1f} min  "
          f"(new {extracted}, cached {cached}, failed {failed})")

    # ---- splits, per beatmapset ------------------------------------------ #
    ids_of_set: dict[int, list[int]] = defaultdict(list)
    for c in chosen:
        ids_of_set[c["beatmapset_id"]].append(c["beatmap_id"])
    split_of = {sid: assign_split(sid, ids, old) for sid, ids in ids_of_set.items()}

    # ---- pass 2: charts + shards ----------------------------------------- #
    print(f"\nPass 2/2  parsing and packing -> {args.out}")
    t0 = time.time()
    written = 0
    note_times: list[list[int]] = []
    records: list[dict] = []

    def skip(c, reason):
        skips.append({"folder": c["path"].parent.name, "osu_file": c["path"].name,
                      "beatmap_id": c["beatmap_id"], "reason": reason})

    with ShardWriter(args.out) as writer:
        for i, key in enumerate(sorted(by_mel)):
            if i % 200 == 0 and i:
                print(f"  [{i}/{len(by_mel)}] packed {written} maps")

            mel_path = args.mel_cache / f"{key}.npy"
            if not mel_path.exists():
                for c in by_mel[key]:
                    skip(c, "no mel (audio too large or failed to decode)")
                continue
            mel = np.load(mel_path).astype(np.float32)

            pending = []
            for c in sorted(by_mel[key], key=lambda c: str(c["path"])):
                try:
                    bm = parser.parse_file(c["path"])
                except Exception:                              # noqa: BLE001
                    skip(c, "parse error")
                    continue
                if bm.note_count < MIN_NOTES:
                    skip(c, "too few notes")
                    continue
                if bm.duration_ms < MIN_DURATION_MS:
                    skip(c, "too short")
                    continue
                if not any(tp.uninherited and tp.beat_length > 0 for tp in bm.timing_points):
                    skip(c, "no timing")
                    continue

                # The chart must not claim to run past the end of the audio;
                # that is the alignment contract, checked here at the source.
                last = max(n.end_time if n.is_long else n.time for n in bm.notes)
                if ms_to_frame(last) > mel.shape[1] + 200:
                    skip(c, "chart longer than audio")
                    continue

                # The chart runs as long as the song, so an empty outro is in
                # the loss and windows can start there (10e).
                chart, _ = beatmap_to_tensors(bm, pad_to=mel.shape[1])

                stars = c["db"].taiko_stars
                style, snaps = infer_style_and_snaps(bm)
                pending.append((bm, chart, {
                    "difficulty":  round(stars[osu_db.NM], 3),
                    "sr_dt":       round(stars.get(osu_db.DT, 0.0), 3),
                    "sr_ht":       round(stars.get(osu_db.HT, 0.0), 3),
                    "style":       style,
                    "style_name":  STYLE_NAMES[style],
                    "ranked":      True,
                    "split":       split_of[c["beatmapset_id"]],
                    "avg_nps":     compute_avg_nps(bm),
                    "peak_nps":    compute_peak_nps(bm),
                    "bpm":         round(dominant_bpm(bm), 3),
                    "note_count":  bm.note_count,
                    "duration_ms": bm.duration_ms,
                    "title":       bm.title,
                    "version":     bm.version,
                    "creator":     bm.creator,
                    "beatmap_id":  c["beatmap_id"],
                    "beatmapset_id": c["beatmapset_id"],
                    "md5":         c["md5"],
                    "folder":      c["path"].parent.name,
                    "osu_file":    c["path"].name,
                    **snaps,
                }))

            if not pending:
                continue

            writer.add_mel(key, mel)
            for bm, chart, record in pending:
                writer.add_map(record, chart, bm.timing_points, key)
                records.append(record)
                note_times.append(sorted(n.time for n in bm.notes if not n.is_long))
                written += 1

        stats = writer.close()
    write_note_times(args.out, records, note_times)

    # ---- every ranked taiko map osu!.db knows is accounted for ----------- #
    seen = {s["osu_file"] for s in skips} | {r["osu_file"] for r in records}
    for e in db.values():
        if e.mode == 1 and e.status == osu_db.RANKED and e.osu_file not in seen:
            skips.append({"folder": e.folder, "osu_file": e.osu_file, "beatmap_id": e.beatmap_id,
                          "reason": "ranked in osu!.db but not in the scan (rescan with --scan)"})

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(skips, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- summary ---------------------------------------------------------- #
    print(f"\nPass 2 done in {(time.time() - t0) / 60:.1f} min")
    print(f"{'=' * 60}")
    n_val = sum(r["split"] == "val" for r in records)
    print(f"Packed  : {stats['maps']} maps over {stats['songs']} songs "
          f"({len(records) - n_val} train / {n_val} val)")
    print(f"Audio   : {stats['mel_frames']:,} frames "
          f"({stats['mel_frames'] * FRAME_MS / 1000 / 3600:.1f} hours, "
          f"{stats['mel_gb']:.2f} GB)")
    print(f"Events  : {stats['onsets']:,} onsets, {stats['spans']:,} long notes")
    print(f"Skipped : {len(skips)} files -> {args.report}")
    for reason, count in Counter(s["reason"] for s in skips).most_common():
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
