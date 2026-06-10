"""
scripts/analyze_motifs.py

Extracts motif feature vectors from all maps in colab_index.jsonl
and writes them back as a "motif" field per record.

Motif vector (16 floats, all normalized to [0, 1]):
  [0]  quarter_ratio      — fraction of notes on 1/4 snap
  [1]  eighth_ratio       — fraction of notes on 1/8 snap
  [2]  third_ratio        — fraction of notes on 1/3 snap
  [3]  sixth_ratio        — fraction of notes on 1/6 snap
  [4]  other_ratio        — fraction on any other snap
  [5]  alt_ratio          — fraction of consecutive don/kat alternations
  [6]  run2_ratio         — fraction of notes in runs of 2+
  [7]  run4_ratio         — fraction of notes in runs of 4+
  [8]  run8_ratio         — fraction of notes in runs of 8+
  [9]  don_ratio          — fraction of don notes (vs kat)
  [10] big_ratio          — fraction of big notes
  [11] roll_ratio         — fraction of roll frames
  [12] denden_ratio       — fraction of denden frames
  [13] pattern_entropy    — Shannon entropy of 2-gram don/kat patterns (norm to [0,1])
  [14] nps_norm           — notes per second normalized to [0, 1] (max=20nps)
  [15] density_variance   — variance of note density across 4-second windows (norm)

Usage:
    python scripts/analyze_motifs.py
    python scripts/analyze_motifs.py --index data/processed/colab_index.jsonl
"""

from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from taiko.data.osu_parser import OsuTaikoParser, TaikoBeatmap, TimingPoint

INDEX_PATH = Path("data/processed/colab_index.jsonl")
CACHE_FILE = "taiko_files_filtered.json"

SNAP_TOLERANCE_MS = 8.0
MOTIF_DIM         = 16


# ---------------------------------------------------------------------------
# Snap helpers (reused from preprocess_for_colab.py)
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
    best_denom, best_error = None, SNAP_TOLERANCE_MS
    for denom in [1, 2, 3, 4, 6, 8, 5, 7, 9, 12, 16]:
        grid_ms = ms_per_beat / denom
        nearest = round(offset_ms / grid_ms) * grid_ms
        error   = abs(offset_ms - nearest)
        if error < best_error:
            best_error = error
            best_denom = denom
    return best_denom


# ---------------------------------------------------------------------------
# Motif feature extraction
# ---------------------------------------------------------------------------

def extract_motif(bm: TaikoBeatmap) -> list[float]:
    """
    Returns a 16-float motif vector normalized to [0, 1].
    Returns zeros if the map has no notes.
    """
    zeros = [0.0] * MOTIF_DIM

    hit_notes = [n for n in bm.notes if not n.is_long]
    if not hit_notes or not bm.timing_points:
        return zeros

    N = len(hit_notes)

    # ---- Snap distribution -------------------------------------------- #
    snap_counts = {4: 0, 8: 0, 3: 0, 6: 0, "other": 0}
    total_snapped = 0
    for note in hit_notes:
        d = snap_of_note(note.time, bm.timing_points)
        if d is None:
            continue
        total_snapped += 1
        if d == 4:
            snap_counts[4] += 1
        elif d == 8:
            snap_counts[8] += 1
        elif d == 3:
            snap_counts[3] += 1
        elif d == 6:
            snap_counts[6] += 1
        else:
            snap_counts["other"] += 1

    denom = max(total_snapped, 1)
    quarter_ratio = snap_counts[4]       / denom
    eighth_ratio  = snap_counts[8]       / denom
    third_ratio   = snap_counts[3]       / denom
    sixth_ratio   = snap_counts[6]       / denom
    other_ratio   = snap_counts["other"] / denom

    # ---- Don/kat sequence analysis ------------------------------------ #
    # Build sequence of note types: 0=don/big_don, 1=kat/big_kat
    seq = []
    for n in hit_notes:
        if n.note_type in ("don", "big_don"):
            seq.append(0)
        elif n.note_type in ("kat", "big_kat"):
            seq.append(1)

    # Alternation ratio: consecutive different color
    alt_count = sum(
        1 for i in range(1, len(seq)) if seq[i] != seq[i - 1]
    )
    alt_ratio = alt_count / max(len(seq) - 1, 1)

    # Run length analysis
    runs = []
    if seq:
        run = 1
        for i in range(1, len(seq)):
            if seq[i] == seq[i - 1]:
                run += 1
            else:
                runs.append(run)
                run = 1
        runs.append(run)

    total_in_runs = sum(runs) or 1
    run2_ratio = sum(r for r in runs if 2 <= r < 4) / total_in_runs
    run4_ratio = sum(r for r in runs if 4 <= r < 8) / total_in_runs
    run8_ratio = sum(r for r in runs if r >= 8) / total_in_runs

    # ---- Note type ratios -------------------------------------------- #
    don_count    = sum(1 for n in hit_notes if n.note_type in ("don", "big_don"))
    kat_count    = sum(1 for n in hit_notes if n.note_type in ("kat", "big_kat"))
    big_count    = sum(1 for n in bm.notes  if n.note_type in ("big_don", "big_kat"))
    roll_frames  = sum(max(0, n.end_time - n.time) for n in bm.notes if n.note_type == "roll")
    denden_frames= sum(max(0, n.end_time - n.time) for n in bm.notes if n.note_type == "denden")
    dur_ms       = max(bm.duration_ms, 1)

    don_ratio    = don_count / max(don_count + kat_count, 1)
    big_ratio    = big_count / max(N, 1)
    roll_ratio   = min(1.0, roll_frames   / dur_ms)
    denden_ratio = min(1.0, denden_frames / dur_ms)

    # ---- Pattern entropy (2-gram Shannon entropy) --------------------- #
    bigrams: dict[tuple, int] = {}
    for i in range(len(seq) - 1):
        bg = (seq[i], seq[i + 1])
        bigrams[bg] = bigrams.get(bg, 0) + 1
    total_bg = sum(bigrams.values()) or 1
    entropy = 0.0
    for c in bigrams.values():
        p = c / total_bg
        if p > 0:
            entropy -= p * math.log2(p)
    # Max entropy for 2-grams over {0,1} is log2(4) = 2.0
    pattern_entropy = min(1.0, entropy / 2.0)

    # ---- NPS normalized ---------------------------------------------- #
    nps      = bm.note_count / max(dur_ms / 1000, 1)
    nps_norm = min(1.0, nps / 20.0)

    # ---- Density variance over 4-second windows ---------------------- #
    window_ms = 4000.0
    n_windows = max(1, int(dur_ms / window_ms))
    counts    = [0] * n_windows
    for n in hit_notes:
        w = min(int(n.time / window_ms), n_windows - 1)
        counts[w] += 1
    mean_c   = sum(counts) / n_windows
    variance = sum((c - mean_c) ** 2 for c in counts) / n_windows
    # Normalize: typical variance ~25, cap at 100
    density_variance = min(1.0, variance / 100.0)

    return [
        quarter_ratio,
        eighth_ratio,
        third_ratio,
        sixth_ratio,
        other_ratio,
        alt_ratio,
        run2_ratio,
        run4_ratio,
        run8_ratio,
        don_ratio,
        big_ratio,
        roll_ratio,
        denden_ratio,
        pattern_entropy,
        nps_norm,
        density_variance,
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(index_path: Path, cache_file: str):
    if not index_path.exists():
        print(f"ERROR: {index_path} not found")
        return

    records = [
        json.loads(l)
        for l in index_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    print(f"Loaded {len(records)} records from index")

    # Build path → osu_path lookup from cache
    cache_path = Path(cache_file)
    if not cache_path.exists():
        print(f"ERROR: {cache_file} not found — run fast_scan.py first")
        return

    osu_files = [Path(p) for p in json.loads(cache_path.read_text(encoding="utf-8"))]
    # Index by beatmap_id for fast lookup
    by_id: dict[int, Path] = {}
    parser = OsuTaikoParser()
    print(f"Building beatmap_id index from {len(osu_files)} .osu files...")

    # Build id→path map quickly by reading only [Metadata] section
    import re
    id_re = re.compile(r"^BeatmapID\s*:\s*(\d+)", re.MULTILINE)
    for p in osu_files:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            m    = id_re.search(text)
            if m:
                by_id[int(m.group(1))] = p
        except Exception:
            continue
    print(f"Indexed {len(by_id)} beatmap IDs")

    updated = 0
    errors  = 0
    t_start = time.time()

    for i, rec in enumerate(records):
        if i % 500 == 0:
            elapsed = time.time() - t_start
            eta     = elapsed / max(i, 1) * (len(records) - i)
            print(f"  [{i}/{len(records)}] updated={updated} errors={errors} "
                  f"| {elapsed/60:.1f}min | ETA {eta/60:.1f}min")

        # Skip if already has motif
        if "motif" in rec:
            continue

        bid = rec.get("beatmap_id", 0)
        osu_path = by_id.get(bid)
        if osu_path is None:
            rec["motif"] = [0.0] * MOTIF_DIM
            errors += 1
            continue

        try:
            bm     = parser.parse_file(osu_path)
            motif  = extract_motif(bm)
            rec["motif"] = [round(v, 4) for v in motif]
            updated += 1
        except Exception as e:
            rec["motif"] = [0.0] * MOTIF_DIM
            errors += 1

    # Write back
    with open(index_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total_time = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"Done in {total_time/60:.1f} minutes")
    print(f"Updated : {updated}")
    print(f"Errors  : {errors} (zero vector used)")
    print(f"Index saved to: {index_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index",  default=str(INDEX_PATH), help="Path to colab_index.jsonl")
    ap.add_argument("--cache",  default=CACHE_FILE,      help="Path to taiko_files_filtered.json")
    args = ap.parse_args()
    main(Path(args.index), args.cache)