"""
scripts/evaluate.py

Measures whether the model is any good, against held-out maps it never saw.

    python scripts/evaluate.py --diffusion checkpoints/diffusion/best.pt
    python scripts/evaluate.py --diffusion ... --n-maps 50 --steps 30

Gate B is onset F1 above 0.40. It is the alignment gate: a model that ignores
the audio and emits plausible taiko rhythms scores near zero here however good
its loss curve looks, because matching the reference chart requires matching the
song. Nothing else in this repository can tell those two situations apart.

Difficulty and NPS controllability are measured by asking for values and seeing
what comes back, which is the only honest way to test a control -- a model can
condition on difficulty perfectly in the loss and still ignore it at sampling
time if guidance is misconfigured.

Every chart is generated from the reference map's own red lines -- all of them,
so BPM changes are included -- which measures the model rather than a tempo
detector.

Two audio-agreement numbers sit beside F1, each shown against the ranked map's
own value for the same song (see taiko/data/audio_activity.py):

    quiet notes   share of notes placed in the song's quietest 20% of frames
    onset misses  share of strong on-grid attacks with no note within 40 ms

--grid-probe re-generates each map with its grid shifted a third of a beat and
reports how often the notes follow the shifted grid rather than the original.
1.0 means the model places notes from the beat grid it is given; 0.0 means it
ignores the grid and follows the audio's own beat; 0.5 means neither. It is
the direct test that the timing input reaches the model at all.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from taiko.data.audio_activity import ON_GRID_MS, activity, activity_score
from taiko.data.conditioning import (
    STYLE_NULL, normalise_avg_nps, normalise_difficulty, normalise_peak_nps,
)
from taiko.data.frames import FRAME_MS, describe
from taiko.data.grid import Grid
from taiko.data.motif import beat_frames_from_timing, compute_motif
from taiko.data.osu_parser import TimingPoint
from taiko.data.preprocessed_dataset import WINDOW_FRAMES_DEFAULT, split_indices
from taiko.data.shards import ShardReader, decode_timing_points
from taiko.data.tensor_repr import build_timing_stream, tensor_to_beatmap
from taiko.eval.metrics import (
    note_statistics, onset_f1, pattern_divergence, snap_validity, unplayability,
)
from taiko.model.diffusion import load_diffusion
from taiko.model.sampling import generate_song

GATE_B_F1 = 0.40

TARGETS = {
    "onset_f1":       (">", 0.55),
    "snap_validity":  (">", 0.95),
    "sr_correlation": (">", 0.85),
    "nps_error":      ("<", 1.00),
    "unplayability":  ("<", 0.005),
}


def load_model(diffusion_ckpt: Path, ae_ckpt: Path, device, use_ema: bool = True):
    return load_diffusion(diffusion_ckpt, ae_ckpt, device, use_ema=use_ema)


def dominant_tempo(points: list[TimingPoint]) -> tuple[float, float, int]:
    reds = [tp for tp in points if tp.uninherited and tp.beat_length > 0]
    if not reds:
        return 150.0, 0.0, 4
    tp = reds[0]
    return 60_000.0 / tp.beat_length, float(tp.time), max(1, tp.meter)


def grid_follow_fraction(sample, points: list[TimingPoint], frames: int,
                         threshold: float) -> float:
    """
    Generate against the map's grid shifted by a third of a beat, then ask
    which grid the notes sit on.

    A third of a beat lands on neither grid's binary positions (1/1, 1/2, 1/4),
    so each note on one of those can be attributed unambiguously. Returns
    on_shifted / (on_shifted + on_original).
    """
    shifted = [TimingPoint(time=int(round(tp.time + tp.beat_length / 3.0)),
                           beat_length=tp.beat_length, meter=tp.meter,
                           uninherited=tp.uninherited)
               if tp.uninherited else tp for tp in points]
    timing = torch.from_numpy(build_timing_stream(shifted, frames)).unsqueeze(0)
    chart = sample(timing)
    notes = tensor_to_beatmap(chart, bpm=120, offset_ms=0, threshold=threshold,
                              timing_points=shifted).notes
    if not notes:
        return float("nan")
    times = np.asarray([n.time for n in notes], dtype=np.float64)
    binary = [1, 2, 4]
    on_shift = Grid(shifted).distances_ms(times, binary) <= ON_GRID_MS
    on_orig = Grid(points).distances_ms(times, binary) <= ON_GRID_MS
    a, b = int((on_shift & ~on_orig).sum()), int((on_orig & ~on_shift).sum())
    return a / (a + b) if a + b else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diffusion", type=Path, default=Path("checkpoints/diffusion/best.pt"))
    ap.add_argument("--ae", type=Path, default=Path("checkpoints/autoencoder/best.pt"))
    ap.add_argument("--shards", type=Path, default=Path("data/processed/shards"))
    ap.add_argument("--out", type=Path, default=Path("outputs/evaluation.json"))
    ap.add_argument("--n-maps", type=int, default=40)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg-scale", type=float, default=4.0)
    ap.add_argument("--window-frames", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=6000,
                    help="cap each song's length to keep evaluation quick")
    ap.add_argument("--use-reference-motif", action="store_true",
                    help="condition on the reference chart's own motif. This "
                         "inflates every score and exists to detect leakage: "
                         "a big gap between this and the default run means the "
                         "model is reading the answer off its conditioning.")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--grid-probe", action="store_true",
                    help="also generate against a grid shifted by 1/3 beat and "
                         "measure whether the notes follow it (doubles the cost)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(describe())

    for path, what in ((args.diffusion, "diffusion checkpoint"), (args.ae, "autoencoder")):
        if not path.exists():
            print(f"ERROR: {what} not found: {path}")
            return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, threshold, ckpt = load_model(args.diffusion, args.ae, device, not args.no_ema)
    window = args.window_frames or ckpt.get("window_frames", WINDOW_FRAMES_DEFAULT)
    print(f"Model: profile {ckpt.get('profile')}, step {ckpt.get('step')}, "
          f"threshold {threshold}, window {window}")

    reader = ShardReader(args.shards)
    _, val_idx = split_indices(reader, val_ratio=0.05)
    print(f"Held-out pool: {len(val_idx)} maps")
    if not val_idx:
        print("ERROR: validation split is empty")
        return 1

    rng = np.random.default_rng(args.seed)
    chosen = rng.permutation(val_idx)[:args.n_maps]

    rows = []
    for n, idx in enumerate(chosen):
        idx = int(idx)
        record = reader.records[idx]
        frames = min(reader.chart_length(idx), reader.mel_length(idx), args.max_frames)
        if frames < window:
            continue

        points = decode_timing_points(record["timing_points"])
        bpm, offset, meter = dominant_tempo(points)

        mel = torch.from_numpy(reader.mel_window(idx, 0, frames)).unsqueeze(0)
        timing_np = build_timing_stream(points, frames, start_frame=0)
        timing = torch.from_numpy(timing_np).unsqueeze(0)

        reference_chart = reader.chart_window(idx, 0, frames)
        reference = tensor_to_beatmap(reference_chart, bpm=bpm, offset_ms=offset, meter=meter)

        motif = motif_mask = None
        if args.use_reference_motif:
            motif = compute_motif(reference_chart, beat_frames_from_timing(timing_np))
            motif_mask = np.ones_like(motif)

        requested_sr = float(record.get("difficulty", 5.0))
        requested_nps = float(record.get("avg_nps", 0.0))

        def sample(timing_tensor: torch.Tensor) -> np.ndarray:
            return generate_song(
                model, mel=mel, timing=timing_tensor,
                difficulty=normalise_difficulty(requested_sr),
                style=int(record.get("style", STYLE_NULL)),
                avg_nps=normalise_avg_nps(requested_nps) if requested_nps else None,
                peak_nps=normalise_peak_nps(float(record.get("peak_nps", 0.0))) or None,
                motif=motif, motif_mask=motif_mask,
                window_frames=window, overlap_frames=window // 2,
                ddim_steps=args.steps, cfg_scale=args.cfg_scale,
                progress=False,
                generator=torch.Generator(device=device).manual_seed(args.seed + idx),
            )[0].cpu().numpy()

        generated = tensor_to_beatmap(
            sample(timing), bpm=bpm, offset_ms=offset,
            threshold=threshold, meter=meter, timing_points=points,
        )

        grid = Grid(points)
        act = activity(mel[0].numpy())
        span = (0.0, frames * FRAME_MS)
        gen_act = activity_score(generated.notes, act, grid, span_ms=span)
        ref_act = activity_score(reference.notes, act, grid, span_ms=span)

        follows = float("nan")
        if args.grid_probe:
            follows = grid_follow_fraction(sample, points, frames, threshold)

        f1 = onset_f1(generated.notes, reference.notes)
        snap = snap_validity(generated.notes, generated.timing_points)
        play = unplayability(generated.notes)
        stats = note_statistics(generated)
        reference_stats = note_statistics(reference)

        rows.append({
            "map": f"{record.get('title', '?')} [{record.get('version', '?')}]",
            "onset_f1": f1.f1,
            "onset_precision": f1.precision,
            "onset_recall": f1.recall,
            "onset_mae_ms": f1.mean_abs_error_ms,
            "snap_validity": snap.valid_fraction,
            "unplayability": play.rate,
            "pattern_kl": pattern_divergence(generated.notes, reference.notes),
            "requested_sr": requested_sr,
            "requested_nps": requested_nps,
            "realised_nps": stats.avg_nps,
            "reference_nps": reference_stats.avg_nps,
            "generated_notes": stats.n_notes,
            "reference_notes": reference_stats.n_notes,
            "don_ratio": stats.don_ratio,
            "big_ratio": stats.big_ratio,
            "quiet_note_rate": gen_act.quiet_note_rate,
            "ref_quiet_note_rate": ref_act.quiet_note_rate,
            "onset_miss_rate": gen_act.strong_onset_miss_rate,
            "ref_onset_miss_rate": ref_act.strong_onset_miss_rate,
            "strong_onsets": gen_act.strong_onsets,
            "grid_follow": follows,
        })

        print(f"  [{n + 1}/{len(chosen)}] F1 {f1.f1:.3f}  "
              f"snap {snap.valid_fraction:.3f}  "
              f"notes {stats.n_notes} vs {reference_stats.n_notes}  "
              f"quiet {gen_act.quiet_note_rate:.2f}/{ref_act.quiet_note_rate:.2f}  "
              f"miss {gen_act.strong_onset_miss_rate:.2f}/{ref_act.strong_onset_miss_rate:.2f}  "
              f"{rows[-1]['map'][:44]}")

    if not rows:
        print("ERROR: no map was long enough to evaluate")
        return 1

    # ---- aggregate -------------------------------------------------------- #
    def mean(key: str) -> float:
        return float(statistics.fmean(r[key] for r in rows))

    requested = [r["requested_sr"] for r in rows]
    realised = [r["realised_nps"] for r in rows]
    sr_correlation = (
        float(np.corrcoef(requested, realised)[0, 1])
        if len(rows) > 2 and statistics.pstdev(requested) > 1e-6
           and statistics.pstdev(realised) > 1e-6
        else float("nan")
    )
    nps_error = float(statistics.fmean(
        abs(r["realised_nps"] - r["requested_nps"]) for r in rows if r["requested_nps"] > 0
    )) if any(r["requested_nps"] > 0 for r in rows) else float("nan")

    summary = {
        "n_maps": len(rows),
        "onset_f1": mean("onset_f1"),
        "onset_precision": mean("onset_precision"),
        "onset_recall": mean("onset_recall"),
        "onset_mae_ms": mean("onset_mae_ms"),
        "snap_validity": mean("snap_validity"),
        "unplayability": mean("unplayability"),
        "pattern_kl": mean("pattern_kl"),
        "sr_correlation": sr_correlation,
        "nps_error": nps_error,
        "don_ratio": mean("don_ratio"),
        "big_ratio": mean("big_ratio"),
        "note_ratio": mean("generated_notes") / max(mean("reference_notes"), 1e-6),
        "quiet_note_rate": mean("quiet_note_rate"),
        "ref_quiet_note_rate": mean("ref_quiet_note_rate"),
        "onset_miss_rate": mean("onset_miss_rate"),
        "ref_onset_miss_rate": mean("ref_onset_miss_rate"),
        "grid_follow": (float(np.nanmean([r["grid_follow"] for r in rows]))
                        if args.grid_probe else float("nan")),
    }

    print(f"\n{'=' * 62}")
    print(f"{len(rows)} held-out maps, {args.steps} steps, guidance {args.cfg_scale}")
    print(f"{'=' * 62}")

    for key, (direction, target) in TARGETS.items():
        value = summary[key]
        if value != value:                       # NaN
            verdict = "n/a"
        else:
            ok = value > target if direction == ">" else value < target
            verdict = "PASS" if ok else "below target" if direction == ">" else "over target"
        print(f"  {key:<16s} {value:>8.4f}   target {direction} {target:<7.3f}  {verdict}")

    print(f"\n  {'onset precision':<16s} {summary['onset_precision']:>8.4f}")
    print(f"  {'onset recall':<16s} {summary['onset_recall']:>8.4f}")
    print(f"  {'onset MAE (ms)':<16s} {summary['onset_mae_ms']:>8.2f}")
    print(f"  {'pattern KL':<16s} {summary['pattern_kl']:>8.4f}")
    print(f"  {'note count ratio':<16s} {summary['note_ratio']:>8.4f}   "
          f"(1.0 = same density as the reference)")
    print(f"  {'don ratio':<16s} {summary['don_ratio']:>8.4f}")

    print(f"\n  audio agreement           model   ranked map")
    print(f"  {'quiet-section notes':<24s}{summary['quiet_note_rate']:>7.3f}   "
          f"{summary['ref_quiet_note_rate']:>7.3f}   share of notes in the quietest 20%")
    print(f"  {'missed strong onsets':<24s}{summary['onset_miss_rate']:>7.3f}   "
          f"{summary['ref_onset_miss_rate']:>7.3f}   share of loud on-grid attacks left empty")
    if args.grid_probe:
        print(f"  {'grid follow':<24s}{summary['grid_follow']:>7.3f}   "
              f"(1 = follows the given grid, 0 = ignores it)")

    gate_b = summary["onset_f1"] > GATE_B_F1
    print(f"\n  GATE B  onset F1 > {GATE_B_F1}: "
          f"{'PASSED' if gate_b else 'FAILED'}  ({summary['onset_f1']:.4f})")
    if not gate_b:
        print("\n  The model is not following the audio. More steps will not fix")
        print("  this. Check that training windows pair a chart with the same")
        print("  frames of mel (tests/test_dataset.py), and that the audio")
        print("  encoder levels land on the U-Net's resolutions.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "maps": rows}, indent=2))
    print(f"\nWrote {args.out}")
    return 0 if gate_b else 1


if __name__ == "__main__":
    raise SystemExit(main())
