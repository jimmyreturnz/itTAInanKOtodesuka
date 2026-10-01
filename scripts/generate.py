"""
scripts/generate.py

Generate a playable .osz from an audio file.

    python scripts/generate.py --audio song.mp3 --difficulty 5.5
    python scripts/generate.py --audio song.mp3 --difficulty 7 --style stream
    python scripts/generate.py --audio song.mp3 --timing-from "my timed map.osu"
    python scripts/generate.py --audio song.mp3 --preset tech --bpm 180 --offset 317
    python scripts/generate.py --audio song.mp3 --reference "some map.osu"

Tempo is an input, not something the model invents, and getting the grid right
is most of getting the chart right. In order of preference:

    --timing-from map.osu   red lines from a map you already timed (every one,
                            so BPM changes survive)
    --bpm B --offset O      one tempo you know
    (neither)               detected: super timing (taiko/timing), which finds
                            BPM changes and fits each section to about 1 ms.
                            Its red lines are printed; check them, and pass
                            --timing-from next time if one needs fixing.

Density: when --avg-nps / --peak-nps are left out, the typical density for the
requested star rating is used, from the table fitted on the training data
(stored in the checkpoint, or nps_prior.json beside it). Passing nothing used
to mean asking for zero notes per second.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from taiko.data.audio import MelExtractor
from taiko.data.audio_activity import Activity, activity, activity_score, gate_notes
from taiko.data.conditioning import (
    STYLE_NULL, normalise_avg_nps, normalise_difficulty, normalise_peak_nps,
    style_to_int,
)
from taiko.data.decode import calibrate_threshold, decode_on_grid
from taiko.data.frames import describe, frames_to_sec
from taiko.data.grid import Grid
from taiko.data.motif import (
    MOTIF_NAMES, PRESETS, beat_frames_from_bpm, compute_motif, describe_motif,
    get_preset,
)
from taiko.data.nps_prior import load_prior, lookup
from taiko.data.osu_parser import OsuTaikoParser, TimingPoint
from taiko.data.osu_writer import OsuTaikoSerializer
from taiko.eval.criteria import NAMES, check, enforce
from taiko.eval.mapset import Chart, chart_findings, fix_chart
from taiko.data.repair import repair
from taiko.data.tensor_repr import (
    beatmap_to_tensors, build_timing_stream, red_lines, tensor_to_beatmap,
)
from taiko.data.timing_refine import apply_timing_refinement
from taiko.model.diffusion import load_diffusion
from taiko.model.sampling import generate_song, plan_windows


def load_model(diffusion_ckpt: Path, ae_ckpt: Path, device: torch.device):
    model, threshold, ckpt = load_diffusion(diffusion_ckpt, ae_ckpt, device, verbose=True)
    if ckpt.get("ema"):
        print(f"Using EMA weights ({ckpt['ema']['step']} updates)")
    else:
        print("WARNING: checkpoint has no EMA weights; sample quality will suffer")
    print(f"Model: profile {ckpt.get('profile', 'p1')}, features {model.features}, "
          f"step {ckpt.get('step', '?')}, val {ckpt.get('best_val', float('nan')):.5f}, "
          f"onset threshold {threshold}")
    return model, threshold, ckpt


def resolve_motif(args, parser: OsuTaikoParser) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Turn --preset / --reference / --motif into a vector and its mask."""
    if args.reference:
        reference = Path(args.reference)
        if not reference.exists():
            raise FileNotFoundError(f"reference map not found: {reference}")
        bm = parser.parse_file(reference)
        chart, _ = beatmap_to_tensors(bm)
        bpm = 0.0
        for tp in bm.timing_points:
            if tp.uninherited and tp.beat_length > 0:
                bpm = 60_000.0 / tp.beat_length
                break
        motif = compute_motif(chart, beat_frames_from_bpm(bpm))
        print(f"Style extracted from {reference.name}:")
        print(describe_motif(motif))
        return motif, np.ones_like(motif)

    if args.preset:
        motif = get_preset(args.preset)
        print(f"Style preset {args.preset!r}:")
        print(describe_motif(motif))
        return motif, np.ones_like(motif)

    if args.motif:
        motif = np.asarray(args.motif, dtype=np.float32)
        if motif.size != len(MOTIF_NAMES):
            raise ValueError(f"--motif needs {len(MOTIF_NAMES)} values, got {motif.size}")
        return motif, np.ones_like(motif)

    # Nothing requested. An all-zero mask means "unspecified", which is not the
    # same as asking for a chart with zero of everything.
    return None, None


def resolve_timing(args, total_frames: int) -> tuple[list[TimingPoint], str]:
    """Red lines to generate against, and where they came from."""
    if args.timing_from:
        bm = OsuTaikoParser().parse_file(Path(args.timing_from))
        reds = red_lines(bm.timing_points)
        if not reds:
            raise ValueError(f"{args.timing_from} has no red lines")
        return reds, f"imported from {Path(args.timing_from).name}"

    if args.bpm is not None:
        tp = TimingPoint(time=int(round(args.offset or 0.0)), beat_length=60_000.0 / args.bpm,
                         meter=args.meter, uninherited=True)
        return [tp], "supplied"

    from taiko.timing import detect_timing
    extra = {} if args.bias_ms is None else {"bias_ms": args.bias_ms}
    result = detect_timing(args.audio, verbose=True, **extra)
    return result.timing_points, f"detected ({result.method})"


def resolve_density(args, ckpt: dict, features: int) -> tuple[float | None, float | None, str]:
    """(avg_nps, peak_nps) in real units, or None for "unspecified"."""
    prior = ckpt.get("nps_prior")
    source = "checkpoint"
    if prior is None:
        beside = Path(args.diffusion).parent / "nps_prior.json"
        if beside.exists():
            prior, source = load_prior(beside), str(beside)

    if args.avg_nps is not None or args.peak_nps is not None:
        avg, peak = args.avg_nps, args.peak_nps
        # One given, the other missing: keep the corpus's peak/average ratio
        # for this difficulty rather than sending the missing one as zero.
        if prior is not None:
            typ_avg, typ_peak = lookup(prior, args.difficulty)
            ratio = typ_peak / max(typ_avg, 1e-6)
        else:
            ratio = 1.6
        if peak is None:
            peak = avg * ratio
        if avg is None:
            avg = peak / ratio
        return avg, peak, "supplied"

    if prior is not None:
        avg, peak = lookup(prior, args.difficulty)
        return avg, peak, f"typical for {args.difficulty:.1f}* ({source})"

    if features >= 2:
        return None, None, "unspecified (the model decides)"
    raise SystemExit(
        "This checkpoint predates the density table, and without one the model\n"
        "would be asked for zero notes per second. Either pass --avg-nps and\n"
        "--peak-nps, or make the table once:\n"
        "    python scripts/fit_nps_prior.py --shards <shards> "
        f"--out {Path(args.diffusion).parent / 'nps_prior.json'}"
    )


def window_densities(mel: np.ndarray, avg_nps: float, windows, act: Activity) -> list[float]:
    """
    Per-window density targets that rise and fall with the music: the map's
    average scaled by how busy each window's audio is relative to the song.
    Clipped so no window is asked for less than 30% or more than 160% of it.
    """
    song = float(act.flux.mean()) + 1e-6
    out = []
    for start, end in windows:
        busy = float(act.flux[start:end].mean()) / song
        out.append(normalise_avg_nps(avg_nps * float(np.clip(busy, 0.3, 1.6))))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate an osu!taiko map from audio")
    ap.add_argument("--audio", required=True, type=Path)
    ap.add_argument("--diffusion", type=Path, default=Path("checkpoints/diffusion/best.pt"))
    ap.add_argument("--ae", type=Path, default=Path("checkpoints/autoencoder/best.pt"))
    ap.add_argument("--out", type=Path, default=Path("outputs"))

    ap.add_argument("--difficulty", type=float, default=None,
                    help="target star rating (default: --level's ranked median, else 5.0)")
    ap.add_argument("--level", default=None, choices=list(NAMES),
                    help="a taiko difficulty name: the chart is made to obey that level's "
                         "ranking criteria (every problem and warning), named after it, and "
                         "given its ranked median SR, OD and HP unless --difficulty is set")
    ap.add_argument("--style", default=None,
                    choices=["standard", "stream", "speed", "tech"])
    ap.add_argument("--preset", default=None, choices=sorted(PRESETS),
                    help="named motif preset")
    ap.add_argument("--reference", default=None,
                    help="an .osu file to copy the style of")
    ap.add_argument("--motif", type=float, nargs="+", default=None,
                    help="16 raw motif values (advanced)")
    ap.add_argument("--avg-nps", type=float, default=None,
                    help="default: typical for --difficulty, from the training data")
    ap.add_argument("--peak-nps", type=float, default=None)
    ap.add_argument("--window-density", choices=["off", "auto"], default="off",
                    help="features-2 models only. 'auto' asks each window for a "
                         "density that follows how busy its audio is; 'off' "
                         "leaves that to the model, which learned it from the "
                         "audio")

    timing = ap.add_argument_group("timing")
    timing.add_argument("--timing-from", default=None,
                        help="an .osu whose red lines to use (all of them)")
    timing.add_argument("--bpm", type=float, default=None, help="one known tempo")
    timing.add_argument("--offset", type=float, default=None, help="first beat, ms")
    timing.add_argument("--meter", type=int, default=4)
    timing.add_argument("--bias-ms", type=float, default=None,
                        help="shift detected red lines by this many ms "
                             "(default: DECODER_BIAS_MS)")

    ap.add_argument("--cfg-scale", type=float, default=4.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--window-frames", type=int, default=None,
                    help="default: whatever the checkpoint trained with")
    ap.add_argument("--overlap", type=int, default=None)
    ap.add_argument("--batch-windows", type=int, default=16,
                    help="windows per U-Net forward; lower it on a small GPU")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the checkpoint's onset threshold")
    ap.add_argument("--decode", choices=["grid", "legacy"], default="grid",
                    help="grid: place notes only on legal subdivisions of each "
                         "section's tempo (default). legacy: decode 20 ms frames, "
                         "then snap")
    ap.add_argument("--no-repair", action="store_true",
                    help="skip the playability repair (too-fast hits, big notes "
                         "in streams, overlapping or empty long notes)")
    ap.add_argument("--no-refine", action="store_true",
                    help="skip the post-generation grid snap")
    ap.add_argument("--density-lock", action="store_true",
                    help="move the hit threshold until the chart's NPS matches the "
                         "requested density (--avg-nps, or the typical one for "
                         "--difficulty). Measured on held-out maps: density error "
                         "8.3%% -> 6.1%%, onset F1 -0.009, so it is for when the chart "
                         "comes out clearly denser or sparser than asked")
    ap.add_argument("--quiet-gate", action="store_true",
                    help="drop isolated notes in the song's quietest passages "
                         "that have no attack under them (streams are kept)")
    args = ap.parse_args()
    if args.difficulty is None:
        args.difficulty = NAMES[args.level][1] if args.level else 5.0

    print(describe())

    for path, what in ((args.audio, "audio"), (args.diffusion, "diffusion checkpoint"),
                       (args.ae, "autoencoder checkpoint")):
        if not Path(path).exists():
            print(f"ERROR: {what} not found: {path}")
            return 1
    if args.timing_from and not Path(args.timing_from).exists():
        print(f"ERROR: timing map not found: {args.timing_from}")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt_threshold, ckpt = load_model(args.diffusion, args.ae, device)
    threshold = args.threshold if args.threshold is not None else ckpt_threshold

    window = args.window_frames or ckpt.get("window_frames", 1536)
    overlap = args.overlap if args.overlap is not None else window // 2

    # ---- audio -------------------------------------------------------- #
    print(f"\nExtracting mel from {args.audio.name} ...")
    mel = MelExtractor().extract(args.audio)
    total_frames = mel.shape[1]
    print(f"  {total_frames} frames = {frames_to_sec(total_frames):.1f}s")
    act = activity(mel)

    # ---- tempo -------------------------------------------------------- #
    points, timing_source = resolve_timing(args, total_frames)
    grid = Grid(points)
    print(f"\nTiming ({timing_source}):")
    for sec in grid.sections[:12]:
        print(f"  {sec.offset_ms:9.0f} ms  {sec.bpm:8.3f} BPM  {sec.meter}/4")
    if len(grid.sections) > 12:
        print(f"  ... {len(grid.sections) - 12} more red lines")
    timing = build_timing_stream(points, total_frames)

    # ---- conditioning -------------------------------------------------- #
    parser = OsuTaikoParser()
    motif, motif_mask = resolve_motif(args, parser)
    style = style_to_int(args.style) if args.style else STYLE_NULL
    avg_nps, peak_nps, density_source = resolve_density(args, ckpt, model.features)

    win_nps = None
    if args.window_density == "auto":
        if model.features < 2:
            print("  (--window-density needs a features-2 model; ignored)")
        elif avg_nps is None:
            print("  (--window-density auto needs a map density; ignored)")
        else:
            win_nps = window_densities(mel, avg_nps, plan_windows(total_frames, window, overlap), act)

    print(f"\nGenerating:")
    print(f"  difficulty  {args.difficulty}*")
    print(f"  density     " + (f"{avg_nps:.2f} avg / {peak_nps:.2f} peak nps"
                               if avg_nps is not None and peak_nps is not None
                               else "unspecified") + f"  ({density_source})")
    print(f"  style       {args.style or 'unspecified'}")
    print(f"  guidance    {args.cfg_scale}   steps {args.steps}")
    print(f"  window      {window} frames, overlap {overlap}")

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    chart = generate_song(
        model,
        mel=torch.from_numpy(mel).unsqueeze(0),
        timing=torch.from_numpy(timing).unsqueeze(0),
        difficulty=normalise_difficulty(args.difficulty),
        style=style,
        avg_nps=normalise_avg_nps(avg_nps) if avg_nps is not None else None,
        peak_nps=normalise_peak_nps(peak_nps) if peak_nps is not None else None,
        motif=motif,
        motif_mask=motif_mask,
        window_nps=win_nps,
        window_frames=window,
        overlap_frames=overlap,
        ddim_steps=args.steps,
        cfg_scale=args.cfg_scale,
        eta=args.eta,
        generator=generator,
        batch_windows=args.batch_windows,
    )[0].cpu().numpy()

    # ---- decode -------------------------------------------------------- #
    style_label = args.preset or args.style or "AI"
    version = args.level or f"{style_label.capitalize()} {args.difficulty:.1f}"
    meta = dict(
        title=args.audio.stem, artist="",
        version=version,
        audio_filename=args.audio.name,
        overall_difficulty=NAMES[args.level][2] if args.level else min(10.0, args.difficulty),
    )
    if args.decode == "grid":
        # Every note on a legal subdivision of its own section's tempo.
        bm = decode_on_grid(chart, points, threshold=threshold, meter=args.meter, **meta)
        if args.density_lock and avg_nps:
            hit_th, got = calibrate_threshold(chart, points, avg_nps, threshold=threshold)
            print(f"\nDensity lock: hit threshold {threshold} -> {hit_th:.3f}, "
                  f"{got:.2f} nps for {avg_nps:.2f} requested")
            bm = decode_on_grid(chart, points, threshold=threshold, hit_threshold=hit_th,
                                meter=args.meter, **meta)
    else:
        bm = tensor_to_beatmap(
            chart, bpm=grid.sections[0].bpm, offset_ms=grid.sections[0].offset_ms,
            threshold=threshold, meter=args.meter, timing_points=points, **meta,
        )
        if not args.no_refine and bm.note_count > 0:
            print("\nSnapping to the beat grid ...")
            apply_timing_refinement(bm, timing_points=points, verbose=True)

    if not args.no_repair and bm.notes:
        fixes = repair(bm, grid)
        print(f"\nPlayability repair: {fixes.summary()}")

    if args.level and bm.notes:
        # After repair, which can change note types. The quiet gate below only
        # drops notes, and dropping one cannot break a criterion.
        bm.notes, fixes = enforce(bm.notes, grid, NAMES[args.level][0])
        bm.hp_drain = NAMES[args.level][3]
        bm.compute_stats()
        print(f"\nRanking criteria ({args.level}): "
              + (", ".join(f"{v} {k}" for k, v in fixes.most_common()) or "nothing to fix"))
        left = [k for k in check(bm.notes, grid, NAMES[args.level][0])
                if k.split(":")[0] in ("problem", "warning") and "rest" not in k]
        if left:
            print(f"  still breaks: {left}")

    if bm.notes:
        # The beatmapset checks a generator answers for, on the timing it was given.
        bm.notes, moved = fix_chart(bm.notes, points)
        if moved:
            print(f"\nMapset: {', '.join(f'{v} {k}' for k, v in moved.items())}")
        flagged = [k for k in chart_findings(Chart(bm.version, bm.notes, points))
                   if k.split(":")[0] in ("problem", "warning")]
        if flagged:
            print(f"  ranking checks still flag: {flagged}")

    if args.quiet_gate and bm.notes:
        kept, dropped = gate_notes(bm.notes, act, grid)
        bm.notes = kept
        bm.compute_stats()
        print(f"\nQuiet gate: dropped {len(dropped)} isolated note(s) in quiet passages")

    score = activity_score(bm.notes, act, grid)
    print(f"\nResult:")
    print(f"  notes     {bm.note_count}")
    print(f"  nps       {bm.notes_per_second:.2f}")
    print(f"  don/kat   {bm.don_ratio:.0%} / {1 - bm.don_ratio:.0%}")
    print(f"  big       {bm.big_ratio:.1%}")
    print(f"  rolls     {bm.roll_count}   dendens {bm.denden_count}")
    print(f"  duration  {bm.duration_ms / 1000:.1f}s")
    print(f"  quiet-section notes    {score.quiet_note_rate:.1%} of notes")
    print(f"  strong onsets left empty  {len(score.missed_times_ms)}/{score.strong_onsets}"
          + (f"  e.g. at {', '.join(_mmss(t) for t in score.missed_times_ms[:6])}"
             if score.missed_times_ms else ""))

    if bm.note_count == 0:
        print("\nThe map is empty. Try a lower --threshold or a lower --cfg-scale;")
        print("an undertrained model also produces this.")
        return 1

    # ---- package -------------------------------------------------------- #
    args.out.mkdir(parents=True, exist_ok=True)
    safe = "".join(c for c in args.audio.stem if c.isalnum() or c in " -_")[:40].strip()
    name = f"{safe} [{version}]"
    osz_path = args.out / f"{name}.osz"

    with zipfile.ZipFile(osz_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{name}.osu", OsuTaikoSerializer().serialize(bm, args.audio.name))
        archive.write(str(args.audio), args.audio.name)

    print(f"\nSaved {osz_path}")
    print("Open it with osu! to import.")
    return 0


def _mmss(ms: int) -> str:
    return f"{ms // 60000}:{(ms // 1000) % 60:02d}.{ms % 1000:03d}"


if __name__ == "__main__":
    raise SystemExit(main())
