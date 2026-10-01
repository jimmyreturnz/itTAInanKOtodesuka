"""
scripts/measure_audio_offset.py

Is the audio the model trained on aligned with its charts the same way for
every song? MP3 decoders disagree about encoder delay (~25 ms for LAME at
44.1 kHz, about one 20 ms frame), so a corpus decoded through two paths would
carry per-song label noise at exactly the scale that separates 1/4 from 1/6.

Per song folder in the shards, three measurements:

  lag      where the stored mel's onset flux best lines up with the .osu's real
           note times, searched at 1 ms over +-LAG_REACH_MS by interpolating
           the 20 ms flux. This is what training saw. Its absolute value
           carries the mel's own frame centring; the spread and any second
           mode are the point.
  path     which decoder load_audio takes on this file today (torchaudio, or
           the librosa fallback), and, with --decode N, that decode's
           sample-exact lag against ffmpeg on N songs.
  file     whether find_audio (first .mp3 in the folder) is the file the .osu
           names in AudioFilename. A mismatch pairs the chart with other audio.

    python scripts/measure_audio_offset.py                 # lag + file, every song
    python scripts/measure_audio_offset.py --decode 150    # + decoder paths
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from pack_dataset import DEFAULT_SCAN_CACHE, find_audio, safe_name
from taiko.data.frames import FRAME_MS
from taiko.data.osu_parser import OsuTaikoParser
from taiko.data.shards import ShardReader

LAG_REACH_MS = 80
DECODE_SECONDS = 30


def flux_of(mel: np.ndarray) -> np.ndarray:
    mel = mel.astype(np.float32)
    f = np.zeros(mel.shape[1], dtype=np.float32)
    f[1:] = np.maximum(mel[:, 1:] - mel[:, :-1], 0.0).mean(axis=0)
    return (f - f.mean()) / (f.std() + 1e-9)


def chart_lag(flux: np.ndarray, note_ms: np.ndarray) -> tuple[int, float]:
    """Lag (ms) at which the flux, read at note time + lag, is highest; and its peak height."""
    frame_t = np.arange(flux.size) * FRAME_MS
    lags = np.arange(-LAG_REACH_MS, LAG_REACH_MS + 1)
    score = np.array([np.interp(note_ms + l, frame_t, flux).mean() for l in lags])
    k = int(score.argmax())
    return int(lags[k]), float(score[k])


def which_path(path: Path) -> str:
    try:
        import torchaudio
        torchaudio.load(str(path))
        return "torchaudio"
    except Exception:                                   # noqa: BLE001
        return "librosa"


def decode_lag_vs_ffmpeg(path: Path) -> int | None:
    """Samples the pack's decode is behind ffmpeg's (positive = pack later)."""
    from taiko.data.audio import load_audio
    try:
        ours, sr = load_audio(path)
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-t", str(DECODE_SECONDS),
             "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
            capture_output=True, check=True).stdout
    except Exception:                                   # noqa: BLE001
        return None
    ref = np.frombuffer(raw, dtype=np.float32)
    n = min(ref.size, ours.size, sr * DECODE_SECONDS)
    a, b = ours[:n].astype(np.float64), ref[:n].astype(np.float64)
    size = 1 << (2 * n - 1).bit_length()
    xc = np.fft.irfft(np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size)), size)
    reach = sr // 10                                   # +-100 ms
    window = np.concatenate([xc[-reach:], xc[:reach + 1]])
    return int(window.argmax()) - reach


def histogram(values, width=5, lo=-LAG_REACH_MS, hi=LAG_REACH_MS) -> str:
    counts = Counter((v - lo) // width for v in values)
    peak = max(counts.values()) if counts else 1
    rows = []
    for b in range((hi - lo) // width + 1):
        c = counts.get(b, 0)
        if c:
            rows.append(f"  {lo + b * width:+4d}..{lo + b * width + width - 1:+4d} ms "
                        f"{c:5d} {'#' * max(1, round(40 * c / peak))}")
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, default=Path("data/processed/shards"))
    ap.add_argument("--scan-cache", type=Path, default=DEFAULT_SCAN_CACHE)
    ap.add_argument("--decode", type=int, default=0,
                    help="also check the decoder path and its lag against ffmpeg on this many songs")
    ap.add_argument("--out", type=Path, default=Path("data/audio_offset.json"))
    args = ap.parse_args()

    reader = ShardReader(args.shards)
    by_key: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(reader.records):
        by_key[rec["mel_key"]].append(i)

    folders: dict[str, Path] = {}
    osu_by_folder: dict[Path, list[Path]] = defaultdict(list)
    for p in map(Path, json.loads(args.scan_cache.read_text(encoding="utf-8"))):
        folders.setdefault(safe_name(p.parent.name), p.parent)
        osu_by_folder[p.parent].append(p)

    parser = OsuTaikoParser()
    rows = []
    for n, (key, idxs) in enumerate(sorted(by_key.items())):
        if n % 250 == 0:
            print(f"  {n}/{len(by_key)} songs", flush=True)
        folder = folders.get(key)
        if folder is None or not folder.exists():
            continue
        rec = max((reader.records[i] for i in idxs), key=lambda r: r.get("note_count", 0))
        bm = None
        for p in osu_by_folder[folder]:
            try:
                cand = parser.parse_file(p)
            except Exception:                           # noqa: BLE001
                continue
            if cand.version == rec.get("version"):
                bm = cand
                break
        if bm is None or len(bm.notes) < 50:
            continue
        idx = idxs[[reader.records[i] is rec for i in idxs].index(True)]
        flux = flux_of(reader.mel_window(idx, 0, reader.mel_length(idx)))
        lag, height = chart_lag(flux, np.array([x.time for x in bm.notes], dtype=np.float64))
        picked = find_audio(folder)
        rows.append({
            "song": key, "lag_ms": lag, "peak": round(height, 3),
            "ext": picked.suffix.lower() if picked else None,
            "file_matches": bool(picked and picked.name.lower() == bm.audio_filename.lower()),
            "osu_audio": bm.audio_filename, "picked": picked.name if picked else None,
            "audio_path": str(picked) if picked else None,
        })

    rng = np.random.default_rng(0)
    for row in [rows[i] for i in rng.permutation(len(rows))[:args.decode]]:
        if row["audio_path"]:
            row["path"] = which_path(Path(row["audio_path"]))
            row["vs_ffmpeg"] = decode_lag_vs_ffmpeg(Path(row["audio_path"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    # A weak peak means the flux barely tracks the notes and the lag is noise.
    solid = [r for r in rows if r["peak"] >= 0.5]
    lags = np.array([r["lag_ms"] for r in solid])
    print(f"\n{len(rows)} songs measured, {len(solid)} with a clear peak (flux z >= 0.5)")
    print(f"chart lag: median {np.median(lags):+.0f} ms, IQR "
          f"{np.percentile(lags, 25):+.0f}..{np.percentile(lags, 75):+.0f}, "
          f"|lag - median| > 10 ms: {np.mean(np.abs(lags - np.median(lags)) > 10):.1%}")
    print(histogram(lags.tolist()))
    for ext in sorted({r["ext"] for r in solid if r["ext"]}):
        sub = [r["lag_ms"] for r in solid if r["ext"] == ext]
        print(f"  {ext}: {len(sub)} songs, median {np.median(sub):+.0f} ms, "
              f"IQR {np.percentile(sub, 25):+.0f}..{np.percentile(sub, 75):+.0f}")

    wrong = [r for r in rows if not r["file_matches"]]
    print(f"\nfind_audio picked a different file from AudioFilename: {len(wrong)}/{len(rows)}")
    for r in wrong[:15]:
        print(f"  {r['song'][:50]:50s} osu={r['osu_audio']}  picked={r['picked']}")

    decoded = [r for r in rows if "path" in r]
    if decoded:
        print(f"\ndecoder paths on {len(decoded)} songs: {dict(Counter(r['path'] for r in decoded))}")
        for path in sorted({r["path"] for r in decoded}):
            for ext in sorted({r["ext"] for r in decoded if r["path"] == path}):
                sub = [r for r in decoded if r["path"] == path and r["ext"] == ext]
                vs = Counter(r["vs_ffmpeg"] for r in sub)
                chart = [r["lag_ms"] for r in sub if r["peak"] >= 0.5]
                print(f"  {path:10s} {ext}: {len(sub)} songs, samples vs ffmpeg {dict(vs.most_common(5))}"
                      + (f", chart lag median {np.median(chart):+.0f} ms" if chart else ""))
    print(f"\nper-song rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
