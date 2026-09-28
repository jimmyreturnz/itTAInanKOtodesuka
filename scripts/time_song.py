"""
scripts/time_song.py

Time a song: find its BPMs and offsets, BPM changes included, and write them
as an .osu you can open in the editor -- no notes, just the red lines.

    python scripts/time_song.py --audio song.mp3
    python scripts/time_song.py --audio song.mp3 --backend beat_this --bpm-range 150 300
    python scripts/time_song.py --audio song.mp3 --title SUPERNOVA --artist USAO

Then either fix anything that is off in the editor and generate from it:

    python scripts/generate.py --audio song.mp3 --timing-from "outputs/<name>.osu"

or generate straight away (generate.py runs the same detection by default).
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from taiko.data.osu_parser import TaikoBeatmap
from taiko.data.osu_writer import OsuTaikoSerializer
from taiko.timing import detect_timing


def main() -> int:
    ap = argparse.ArgumentParser(description="Detect red lines for a song")
    ap.add_argument("--audio", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--backend", default="auto", choices=["auto", "timingnet", "beat_this", "onset"])
    ap.add_argument("--timingnet", type=Path, default=None, help="TimingNet checkpoint")
    ap.add_argument("--passes", type=int, default=8,
                    help="shifted-audio passes averaged for neural backends")
    ap.add_argument("--bpm-range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="restrict BPM, e.g. to settle half/double time")
    ap.add_argument("--leniency", type=float, default=12.0,
                    help="ms a beat may sit off a rounded BPM's grid")
    ap.add_argument("--meter", type=int, default=4)
    ap.add_argument("--title", default=None)
    ap.add_argument("--artist", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--osz", action="store_true", help="also package the audio as an .osz")
    args = ap.parse_args()

    if not args.audio.exists():
        print(f"ERROR: audio not found: {args.audio}")
        return 1

    kwargs = {}
    if args.bpm_range:
        kwargs["bpm_range"] = tuple(args.bpm_range)
        kwargs["prior_bpm"] = None
    result = detect_timing(args.audio, backend=args.backend, timingnet=args.timingnet,
                           passes=args.passes, leniency_ms=args.leniency, meter=args.meter,
                           device=args.device, verbose=True, **kwargs)
    print(result.describe())

    bm = TaikoBeatmap()
    bm.title = args.title or args.audio.stem
    bm.artist = args.artist
    bm.creator = "TaikoAI timing"
    bm.version = "Timing"
    bm.audio_filename = args.audio.name
    bm.timing_points = result.timing_points
    bm.notes = []

    args.out.mkdir(parents=True, exist_ok=True)
    safe = "".join(c for c in bm.title if c.isalnum() or c in " -_")[:40].strip() or "song"
    osu_path = args.out / f"{safe} [Timing].osu"
    text = OsuTaikoSerializer().serialize(bm, args.audio.name)
    osu_path.write_text(text, encoding="utf-8")
    print(f"\nWrote {osu_path}")

    if args.osz:
        osz = args.out / f"{safe} [Timing].osz"
        with zipfile.ZipFile(osz, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(osu_path.name, text)
            archive.write(str(args.audio), args.audio.name)
        print(f"Wrote {osz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
