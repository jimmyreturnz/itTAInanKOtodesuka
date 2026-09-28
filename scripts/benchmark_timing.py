"""
scripts/benchmark_timing.py

How good is automatic timing on the music we care about? Ranked red lines are
the answer key: every held-out song in the corpus has hand-checked timing.

    python scripts/benchmark_timing.py --shards data/processed/shards \
        --songs "D:/osu!/Songs" --backends onset beat_this timingnet --n-songs 60

Songs are the chart model's validation split, so a TimingNet trained with
scripts/train_timing.py has never heard them. For each song and backend:

    bpm exact      every ranked section's BPM found within 0.01
    octave         a section found at half or double the BPM
    offset <=2/5ms the grid lands within 2 / 5 ms of the ranked grid,
                   measured mid-section, where a tempo error has the least
                   leverage
    sections       detected sections vs ranked sections with distinct BPMs

The median *signed* offset error across songs is printed as the recommended
DECODER_BIAS_MS (taiko/timing/__init__.py): a constant difference between
our audio decoder's clock and osu!'s shows up there and nowhere else.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from taiko.data.grid import Grid
from taiko.data.preprocessed_dataset import split_indices
from taiko.data.shards import ShardReader, decode_timing_points
from taiko.data.tensor_repr import red_lines
from taiko.timing import detect_timing
from taiko.timing.activations import load_mono


def safe_name(name: str) -> str:
    from scripts.pack_dataset import safe_name as pack_safe_name   # same key as packing
    return pack_safe_name(name)


def distinct_bpm_sections(points) -> list:
    out = []
    for tp in red_lines(points):
        if out and abs(out[-1].beat_length - tp.beat_length) < 1e-6:
            continue
        out.append(tp)
    return out


def score_song(truth_points, detected_points, duration_ms: float) -> dict:
    truth = Grid(truth_points)
    found = Grid(detected_points)
    bounds = [s.offset_ms for s in truth.sections] + [duration_ms]
    exact, octave, offs = 0, 0, []
    for i, sec in enumerate(truth.sections):
        lo, hi = max(bounds[i], 0.0), bounds[i + 1]
        if hi - lo < 4 * sec.ms_per_beat:
            continue
        mid = lo + (hi - lo) / 2
        det = found.section_at(mid)
        ratio = det.bpm / sec.bpm
        if abs(det.bpm - sec.bpm) <= 0.01:
            exact += 1
        elif min(abs(ratio - 2), abs(ratio - 0.5)) < 0.01:
            octave += 1
        # Nearest ranked beat to mid-section, and the detected grid's nearest
        # line to it (at the coarser of the two periods, so half/double time
        # still yields a meaningful phase error).
        k = round((mid - sec.offset_ms) / sec.ms_per_beat)
        t_true = sec.offset_ms + k * sec.ms_per_beat
        period = max(det.ms_per_beat, sec.ms_per_beat)
        rel = (t_true - det.offset_ms) / period
        offs.append(-(rel - round(rel)) * period)       # detected minus truth
    n = len([1 for i, s in enumerate(truth.sections)
             if bounds[i + 1] - max(bounds[i], 0.0) >= 4 * s.ms_per_beat])
    return {
        "sections_true": len(distinct_bpm_sections(truth_points)),
        "sections_found": len(distinct_bpm_sections(detected_points)),
        "bpm_exact": exact / max(n, 1),
        "octave": octave / max(n, 1),
        "offset_signed_ms": float(np.median(offs)) if offs else float("nan"),
        "offset_abs_ms": float(np.median(np.abs(offs))) if offs else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shards", type=Path, default=Path("data/processed/shards"))
    ap.add_argument("--songs", type=Path, required=True, help="osu! Songs folder")
    ap.add_argument("--backends", nargs="+", default=["onset", "beat_this", "timingnet"])
    ap.add_argument("--timingnet", type=Path, default=None)
    ap.add_argument("--n-songs", type=int, default=60)
    ap.add_argument("--passes", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("outputs/timing_benchmark.json"))
    args = ap.parse_args()

    from scripts.pack_dataset import find_audio

    reader = ShardReader(args.shards, mel_io="read")
    _, val_idx = split_indices(reader, val_ratio=0.05)
    by_key: dict[str, int] = {}
    for i in val_idx:
        by_key.setdefault(reader.records[i]["mel_key"], i)

    folders = {safe_name(p.name): p for p in args.songs.iterdir() if p.is_dir()}
    songs = [(k, folders[k], by_key[k]) for k in sorted(by_key) if k in folders]
    print(f"{len(by_key)} held-out songs, {len(songs)} found under {args.songs}")
    songs = songs[:args.n_songs]

    results: dict[str, list[dict]] = {b: [] for b in args.backends}
    for n, (key, folder, idx) in enumerate(songs):
        audio = find_audio(folder)
        if audio is None:
            continue
        truth = decode_timing_points(reader.records[idx]["timing_points"])
        y = load_mono(audio)
        duration = len(y) / 22.05
        for backend in args.backends:
            t0 = time.time()
            try:
                r = detect_timing(y, backend=backend, timingnet=args.timingnet,
                                  passes=args.passes, device=args.device)
                row = score_song(truth, r.timing_points, duration)
            except Exception as exc:                            # noqa: BLE001
                row = {"error": f"{type(exc).__name__}: {exc}"}
            row.update(song=folder.name, seconds=round(time.time() - t0, 1))
            results[backend].append(row)
            if "error" in row:
                print(f"  [{n + 1}/{len(songs)}] {backend:<10s} {row['error'][:80]}")
            else:
                print(f"  [{n + 1}/{len(songs)}] {backend:<10s} bpm {row['bpm_exact']:.2f}  "
                      f"oct {row['octave']:.2f}  off {row['offset_signed_ms']:+6.1f} ms  "
                      f"sections {row['sections_found']}/{row['sections_true']}  {folder.name[:40]}")

    print(f"\n{'backend':<11s} {'songs':>5s} {'bpm exact':>9s} {'octave':>7s} "
          f"{'off<=2ms':>8s} {'off<=5ms':>8s} {'sections ok':>11s} {'bias ms':>8s}")
    summary = {}
    for backend, rows in results.items():
        ok = [r for r in rows if "error" not in r and r["offset_abs_ms"] == r["offset_abs_ms"]]
        if not ok:
            print(f"{backend:<11s} {'-':>5s}  (no successful runs)")
            continue
        s = {
            "songs": len(ok),
            "bpm_exact": statistics.fmean(r["bpm_exact"] for r in ok),
            "octave": statistics.fmean(r["octave"] for r in ok),
            "off_2ms": statistics.fmean(r["offset_abs_ms"] <= 2 for r in ok),
            "off_5ms": statistics.fmean(r["offset_abs_ms"] <= 5 for r in ok),
            "sections_ok": statistics.fmean(r["sections_found"] == r["sections_true"] for r in ok),
            "bias_ms": float(np.median([r["offset_signed_ms"] for r in ok])),
        }
        summary[backend] = s
        print(f"{backend:<11s} {s['songs']:>5d} {s['bpm_exact']:>9.2%} {s['octave']:>7.2%} "
              f"{s['off_2ms']:>8.2%} {s['off_5ms']:>8.2%} {s['sections_ok']:>11.2%} "
              f"{s['bias_ms']:>+8.1f}")

    print("\nA consistent non-zero bias across backends is the decoder offset; set it as "
          "DECODER_BIAS_MS (negated sign: detected minus truth is what is printed).")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "songs": results}, indent=1))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
