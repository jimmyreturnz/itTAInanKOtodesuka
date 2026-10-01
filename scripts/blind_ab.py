"""
scripts/blind_ab.py

A blind A/B round for a player: each song gets the AI's chart and its own
ranked map at the same SR, under neutral names in shuffled order, so the only
thing left to judge is how it plays. The metrics can say a chart matches the
ranked distributions; only someone playing it can say whether it is fun.

    python scripts/blind_ab.py make --round 1             # writes the pack
    python scripts/blind_ab.py score --round 1            # after playing

`make` picks 6 held-out songs, one per SR band, that osu!.db says you have
never played (every difficulty in the folder unplayed) and that no earlier
round used (outputs/blind_ab/used.json). Each song becomes a folder with
the audio and two difficulties, [A] and [B]: copy the folders into osu!'s
Songs folder (or pass --songs-dir) and press F5 in song select.

Both charts are written by the same serializer, with the ranked map's red
and green lines, SliderMultiplier, OD and HP, and nothing else: no kiai, no
custom hitsounds. The AI chart scrolls with the ranked map's SV. Anything one
chart has and the other cannot is a tell, and a tell is not blind.

osu! lists a set's difficulties by star rating, not by name, so [A] may be
listed second -- read the name, not the position. The star rating osu! shows
is itself a small tell when the two charts' SR differ; it cannot be hidden.

Then fill in ratings.txt in the round folder (A or B per song, and one line
of why) and run `score`. The key is in key.txt, base64 so a glance does not
give it away.
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

from evaluate import SR_BANDS, load_model
from pack_dataset import DEFAULT_SCAN_CACHE, safe_name
from taiko.data import osu_db
from taiko.data.conditioning import (STYLE_NULL, normalise_avg_nps, normalise_difficulty,
                                     normalise_peak_nps)
from taiko.data.decode import decode_on_grid
from taiko.data.grid import Grid
from taiko.data.osu_parser import OsuTaikoParser, TaikoBeatmap
from taiko.data.osu_writer import OsuTaikoSerializer
from taiko.data.preprocessed_dataset import WINDOW_FRAMES_DEFAULT, split_indices
from taiko.data.repair import repair
from taiko.data.shards import ShardReader, decode_timing_points
from taiko.data.tensor_repr import build_timing_stream
from taiko.eval.criteria import enforce, level_of
from taiko.eval.mapset import fix_chart
from taiko.model.sampling import generate_song

ROOT = Path("outputs/blind_ab")
USED_FILE = ROOT / "used.json"
RATINGS = "ratings.txt"
KEY = "key.txt"


def folder_index(scan_cache: Path) -> dict[str, list[Path]]:
    by_key: dict[str, list[Path]] = {}
    for p in map(Path, json.loads(scan_cache.read_text(encoding="utf-8"))):
        by_key.setdefault(safe_name(p.parent.name), []).append(p)
    return by_key


def never_played(folder: Path, by_folder: dict) -> bool:
    """Every ranked difficulty osu!.db lists in the folder is unplayed. Unranked
    ones (rate edits, cut versions) are never read."""
    entries = [b for b in by_folder.get(folder.name.lower(), []) if b.status == osu_db.RANKED]
    return bool(entries) and all(b.unplayed for b in entries)


def ranked_osu(rec: dict, osus: list[Path], parser: OsuTaikoParser) -> tuple[Path, TaikoBeatmap] | None:
    for p in osus:
        try:
            bm = parser.parse_file(p)
        except Exception:                               # noqa: BLE001
            continue
        if bm.version == rec.get("version"):
            return p, bm
    return None


def pick_songs(reader, val_idx, folders, db, slots, taken, rng) -> dict[int, int]:
    """Slot (SR band index) -> record index, for the slots asked for."""
    picked = {}
    for slot in slots:
        lo, hi, label = SR_BANDS[slot]
        pool = [i for i in val_idx
                if lo <= float(reader.records[i].get("difficulty", 0.0)) < hi
                and reader.records[i]["mel_key"] not in taken]
        rng.shuffle(pool)
        for i in pool:
            osus = folders.get(reader.records[i]["mel_key"], [])
            if osus and never_played(osus[0].parent, db):
                picked[slot] = i
                taken.add(reader.records[i]["mel_key"])
                break
        else:
            print(f"  no never-played held-out song in {label}; that slot stays empty")
    return picked


def ai_chart(model, threshold, window, reader, idx, rec, points, device, seed) -> list:
    frames = min(reader.chart_length(idx), reader.mel_length(idx))
    mel = torch.from_numpy(reader.mel_window(idx, 0, frames)).unsqueeze(0)
    timing = torch.from_numpy(build_timing_stream(points, frames, start_frame=0)).unsqueeze(0)
    nps = float(rec.get("avg_nps", 0.0))
    probs = generate_song(
        model, mel=mel, timing=timing,
        difficulty=normalise_difficulty(float(rec.get("difficulty", 5.0))),
        style=int(rec.get("style", STYLE_NULL)),
        avg_nps=normalise_avg_nps(nps) if nps else None,
        peak_nps=normalise_peak_nps(float(rec.get("peak_nps", 0.0))) or None,
        window_frames=window, overlap_frames=window // 2, progress=False,
        generator=torch.Generator(device=device).manual_seed(seed),
    )[0].cpu().numpy()
    chart = decode_on_grid(probs, points, threshold=threshold)
    repair(chart, Grid(points))
    # As generate.py --level ships it: the level the ranked map is held to.
    level = level_of(rec.get("version", ""), float(rec.get("difficulty", 5.0)))
    notes, _ = enforce(chart.notes, Grid(points), level)
    notes, _ = fix_chart(notes, points)
    return notes


def neutral(ranked: TaikoBeatmap, notes: list, title: str, version: str, audio: str) -> TaikoBeatmap:
    return TaikoBeatmap(
        title=title, artist="Blind A/B", creator="?", version=version, audio_filename=audio,
        hp_drain=ranked.hp_drain, overall_difficulty=ranked.overall_difficulty,
        slider_multiplier=ranked.slider_multiplier, slider_tick_rate=ranked.slider_tick_rate,
        timing_points=list(ranked.timing_points), notes=sorted(notes, key=lambda n: n.time),
    )


def make(args) -> int:
    reader = ShardReader(args.shards)
    _, val_idx = split_indices(reader, val_ratio=0.05)
    folders = folder_index(args.scan_cache)
    db: dict = {}
    for b in osu_db.read(args.osu_db).values():
        db.setdefault(b.folder.lower(), []).append(b)
    rng = random.Random(args.round)

    # Every round is 6 songs never used before: once played, a ranked chart is
    # remembered, and a remembered chart is not blind. (Jimmy, after round 1.)
    used = set(json.loads(USED_FILE.read_text(encoding="utf-8"))) if USED_FILE.exists() else set()
    for legacy in (ROOT / "retired.json", ROOT / "songs.json"):   # the first rounds' bookkeeping
        if legacy.exists():
            data = json.loads(legacy.read_text(encoding="utf-8"))
            used |= set(data) if isinstance(data, list) else \
                {reader.records[v["record"]]["mel_key"] for v in data.values()}
    songs = {slot: {"record": idx} for slot, idx in
             pick_songs(reader, val_idx, folders, db, range(len(SR_BANDS)), used, rng).items()}
    ROOT.mkdir(parents=True, exist_ok=True)
    USED_FILE.write_text(json.dumps(sorted(used), ensure_ascii=False), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, threshold, ckpt = load_model(args.diffusion, args.ae, device, True)
    window = ckpt.get("window_frames", WINDOW_FRAMES_DEFAULT)
    out = ROOT / f"round_{args.round}"
    songs_out = Path(args.songs_dir) if args.songs_dir else out / "songs"
    parser, writer = OsuTaikoParser(), OsuTaikoSerializer()
    key, sheet = {}, ["# Blind A/B round %d. For each song: A or B, whichever was better to play," % args.round,
                      "# then | and one line of why. Example:  3: B | the 1/6 runs land on the vocals", ""]
    for n, slot in enumerate(sorted(songs), start=1):
        idx = songs[slot]["record"]
        rec = reader.records[idx]
        found = ranked_osu(rec, folders.get(rec["mel_key"], []), parser)
        if found is None:
            print(f"  {n}: no .osu for {rec['mel_key']}; skipped")
            continue
        osu_path, ranked = found
        audio_src = osu_path.parent / ranked.audio_filename
        # The ranked .osu's own timing, as the pack writes it out: the packed
        # timing is whole ms, and decoding on it put lines up to 1 ms off.
        points = ranked.timing_points
        print(f"  {n}: {SR_BANDS[slot][2]}  generating ...", flush=True)
        notes = ai_chart(model, threshold, window, reader, idx, rec, points, device, args.round * 1000 + n)

        ai_is_a = rng.random() < 0.5
        title = f"Blind AB r{args.round} #{n}"
        folder = songs_out / title
        folder.mkdir(parents=True, exist_ok=True)
        audio = "audio" + audio_src.suffix.lower()
        shutil.copyfile(audio_src, folder / audio)
        for version, chart in (("A", notes if ai_is_a else ranked.notes),
                               ("B", ranked.notes if ai_is_a else notes)):
            bm = neutral(ranked, chart, title, version, audio)
            (folder / f"{title} [{version}].osu").write_text(writer.serialize(bm, audio), encoding="utf-8")
        key[n] = {"ai": "A" if ai_is_a else "B", "map": f"{rec.get('title')} [{rec.get('version')}]",
                  "sr": rec.get("difficulty"), "band": SR_BANDS[slot][2]}
        sheet.append(f"{n}: ")

    out.mkdir(parents=True, exist_ok=True)
    (out / KEY).write_text(base64.b64encode(json.dumps(
        {"checkpoint": str(args.diffusion), "step": ckpt.get("step"), "songs": key}).encode()).decode())
    ratings = out / RATINGS
    if not ratings.exists():
        ratings.write_text("\n".join(sheet) + "\n", encoding="utf-8")
    print(f"\n{len(key)} songs -> {songs_out}")
    if not args.songs_dir:
        print("  copy those folders into osu!'s Songs folder and press F5 in song select")
    print(f"  then fill in {ratings} and run: python scripts/blind_ab.py score --round {args.round}")
    return 0


def score(args) -> int:
    out = ROOT / f"round_{args.round}"
    key = json.loads(base64.b64decode((out / KEY).read_text()))
    picks = {}
    for line in (out / RATINGS).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#") or ":" not in line:
            continue
        n, rest = line.split(":", 1)
        choice, _, why = rest.partition("|")
        if choice.strip().upper() in ("A", "B"):
            picks[n.strip()] = (choice.strip().upper(), why.strip())

    print(f"round {args.round}: {key['checkpoint']} (step {key['step']})\n")
    ai_wins = 0
    for n, info in key["songs"].items():
        if n not in picks:
            print(f"  {n}: not rated  ({info['band']})")
            continue
        choice, why = picks[n]
        won = "AI" if choice == info["ai"] else "ranked"
        ai_wins += won == "AI"
        print(f"  {n}: preferred {won:6s} {info['band']:<17s} {info['map'][:45]}  -- {why}")
    rated = sum(n in picks for n in key["songs"])
    print(f"\nAI preferred on {ai_wins}/{rated}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("make", "score"))
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--diffusion", type=Path, default=Path("checkpoints/diffusion/best.pt"))
    ap.add_argument("--ae", type=Path, default=Path("checkpoints/autoencoder/best.pt"))
    ap.add_argument("--shards", type=Path, default=Path("data/processed/shards"))
    ap.add_argument("--scan-cache", type=Path, default=DEFAULT_SCAN_CACHE)
    ap.add_argument("--osu-db", type=Path, default=Path("D:/osu!/osu!.db"))
    ap.add_argument("--songs-dir", default=None,
                    help="write the song folders straight into this folder (osu!'s Songs)")
    args = ap.parse_args()
    return make(args) if args.action == "make" else score(args)


if __name__ == "__main__":
    sys.exit(main())
