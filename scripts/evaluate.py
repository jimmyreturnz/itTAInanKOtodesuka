"""
scripts/evaluate.py

Measures whether the model is any good, against held-out maps it never saw.

    python scripts/evaluate.py --diffusion checkpoints/diffusion/best.pt
    python scripts/evaluate.py --diffusion ... --n-maps 50 --steps 30 --seeds 3

Gate B is onset F1 above 0.40. It is the alignment gate: a model that ignores
the audio and emits plausible taiko rhythms scores near zero here however good
its loss curve looks, because matching the reference chart requires matching the
song. Nothing else in this repository can tell those two situations apart.

Difficulty and NPS controllability are measured by asking for values and seeing
what comes back, which is the only honest way to test a control.

Every chart is generated from the reference map's own red lines -- all of them,
so BPM changes are included -- which measures the model rather than a tempo
detector.

What is scored, and why there are several versions of each chart
-----------------------------------------------------------------
The chart model works on 20 ms frames, so a note decoded straight from its
output sits on a frame boundary, up to 10 ms from where it belongs. The
step-53k evaluation scored those raw frame times against the beat grid at a
5 ms tolerance and reported snap validity 0.566. A chart placed *perfectly*
and passed through the same frames scores about 0.5 on that measure -- it was
reading frame quantisation, not the model. So each map is now decoded two
ways and scored before and after post-processing:

    legacy raw     frame decode, as before           (kept for comparison)
    legacy final   + grid snap, what generate.py shipped until now
    grid raw       decode_on_grid: every note on a legal subdivision
    grid final     + repair: the playability fixes, counted   <- the gate

and the ranked map, decoded through the same frames, is the control: its
"old measure" snap validity shows the ceiling the representation imposes.

The headline metrics (TARGETS, Gate B) are on "grid final", which is what
generate.py now produces. Every variant is in the JSON.

Two audio-agreement numbers sit beside F1, each shown against the ranked map's
own value for the same song (see taiko/data/audio_activity.py). --grid-probe
re-generates each map with its grid shifted a third of a beat and reports how
often notes follow the shifted grid: 1.0 follows the grid it is given, 0.0
ignores it.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from taiko.data.audio_activity import ON_GRID_MS, activity, activity_score
from taiko.data.conditioning import (
    STYLE_NULL, normalise_avg_nps, normalise_difficulty, normalise_peak_nps,
)
from taiko.data.decode import calibrate_threshold, decode_on_grid
from taiko.data.frames import FRAME_MS, describe
from taiko.data.grid import Grid
from taiko.data.motif import beat_frames_from_timing, compute_motif
from taiko.data.osu_parser import TimingPoint
from taiko.data.preprocessed_dataset import WINDOW_FRAMES_DEFAULT, split_indices
from taiko.data.repair import repair
from taiko.data.shards import ShardReader, decode_timing_points, load_note_times
from taiko.data.tensor_repr import build_timing_stream, tensor_to_beatmap
from taiko.data.timing_refine import apply_timing_refinement
from taiko.eval.metrics import (
    EXACT_SNAP_DIVISORS, FEEL_FAMILIES, exact_snap, feel_counts, js_divergence,
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

# Half a frame: the most any frame-decoded time can be off its true position.
RAW_TOLERANCE_MS = FRAME_MS / 2

VARIANTS = ("legacy_raw", "legacy_final", "grid_raw", "grid_final")
SHIPPED = "grid_final"

SR_BANDS = [(0, 2.0, "kantan <2*"), (2.0, 3.0, "futsuu 2-3*"), (3.0, 4.0, "muzukashii 3-4*"),
            (4.0, 5.5, "oni 4-5.5*"), (5.5, 7.0, "inner 5.5-7*"), (7.0, 99, "extreme 7*+")]
BPM_BANDS = [(0, 140, "<140"), (140, 180, "140-180"), (180, 220, "180-220"), (220, 999, "220+")]


def load_model(diffusion_ckpt: Path, ae_ckpt: Path, device, use_ema: bool = True):
    return load_diffusion(diffusion_ckpt, ae_ckpt, device, use_ema=use_ema)


def checkpoint_header(path: Path, ckpt: dict) -> dict:
    """
    Which file was evaluated and how it relates to the newest one.

    The step-53k handover could only guess that "model step 53,664" meant
    best.pt while training had reached 54,925 in last.pt. Say it outright.
    """
    header = {"evaluated": str(path), "step": ckpt.get("step"),
              "best_val": ckpt.get("best_val"), "features": ckpt.get("features", 1)}
    last = path.parent / "last.pt"
    if last.exists() and last.resolve() != path.resolve():
        try:
            other = torch.load(last, map_location="cpu", weights_only=False, mmap=True)
            header["latest_training_step"] = other.get("step")
        except Exception as exc:                                  # noqa: BLE001
            header["latest_training_step"] = f"unreadable ({type(exc).__name__})"
    return header


def dominant_tempo(points: list[TimingPoint]) -> tuple[float, float, int]:
    reds = [tp for tp in points if tp.uninherited and tp.beat_length > 0]
    if not reds:
        return 150.0, 0.0, 4
    tp = reds[0]
    return 60_000.0 / tp.beat_length, float(tp.time), max(1, tp.meter)


def decode_variants(probs: np.ndarray, points, grid: Grid, threshold: float,
                    bpm: float, offset: float, meter: int) -> dict:
    legacy_raw = tensor_to_beatmap(probs, bpm=bpm, offset_ms=offset, threshold=threshold,
                                   meter=meter, timing_points=points)
    legacy_final = copy.deepcopy(legacy_raw)
    if legacy_final.notes:
        apply_timing_refinement(legacy_final, timing_points=points, verbose=False)
    grid_raw = decode_on_grid(probs, points, threshold=threshold)
    grid_final = copy.deepcopy(grid_raw)
    fixes = repair(grid_final, grid)
    return {"legacy_raw": legacy_raw, "legacy_final": legacy_final,
            "grid_raw": grid_raw, "grid_final": grid_final, "_repair": fixes}


# Decode-side harness candidates, each through repair like the shipped chart.
# "lock" calibrates the hit threshold to the requested NPS. An onset bias
# (hit evidence up on strong attacks, down on quiet frames) was measured here
# and dropped: missed strong onsets 0.211 -> 0.209, quiet-section notes 0.200
# -> 0.085 against the ranked 0.180 -- it deleted notes rather than finding any.
HARNESS = ("lock",)


def harness_rows(probs: np.ndarray, points, grid: Grid, threshold: float, target_nps: float,
                 act, reference, span) -> dict:
    ref_nps = note_statistics(reference).avg_nps
    row = {}
    for name in HARNESS:
        th = threshold
        if "lock" in name and target_nps > 0:
            th, _ = calibrate_threshold(probs, points, target_nps, threshold=threshold)
        raw = decode_on_grid(probs, points, threshold=threshold, hit_threshold=th)
        bm = copy.deepcopy(raw)
        fixes = repair(bm, grid)
        a = activity_score(bm.notes, act, grid, span_ms=span)
        nps = note_statistics(bm).avg_nps
        row.update({
            f"h_{name}_onset_f1": onset_f1(bm.notes, reference.notes).f1,
            f"h_{name}_pattern_kl": pattern_divergence(bm.notes, reference.notes),
            f"h_{name}_nps_rel_error": abs(nps - ref_nps) / max(ref_nps, 1e-6),
            f"h_{name}_notes": len(bm.notes),
            f"h_{name}_quiet_note_rate": a.quiet_note_rate,
            f"h_{name}_onset_miss_rate": a.strong_onset_miss_rate,
            f"h_{name}_raw_too_fast": unplayability(raw.notes).too_fast,
            f"h_{name}_repair_rate": fixes.rate,
            f"h_{name}_threshold": th,
        })
    return row


def violation_counts(play) -> dict:
    return {"too_fast": play.too_fast, "big_note_streams": play.big_note_streams,
            "overlapping_longs": play.overlapping_longs,
            "zero_length_longs": play.zero_length_longs}


def hit_times(bm) -> list[int]:
    return [n.time for n in bm.notes if not n.is_long]


def exact_snap_fields(pre: str, gen_ms, real_ms, grid: Grid, per_divisor: bool) -> dict:
    """exact_snap as flat row fields. Per-divisor counts stay counts, so their
    mean over maps divides back to the pooled share."""
    s = exact_snap(gen_ms, real_ms, grid)
    row = {f"{pre}exact_snap": s.exact if s.n_matched else float("nan")}
    if per_divisor:
        row["unsnapped_share"] = s.n_unsnapped / max(len(real_ms), 1)
        for d in EXACT_SNAP_DIVISORS:
            e, m = s.per_divisor.get(d, (0, 0))
            row[f"xs_hit_{d}"], row[f"xs_n_{d}"] = e, m
    return row


def score_variants(variants: dict, reference, act, grid: Grid, span, real_ms=None) -> dict:
    """
    Flat metrics for one generated sample, every variant. `real_ms` is the
    .osu's own hit times (note_times.npz); without it there is no exact snap.
    """
    row: dict = {}
    points = reference.timing_points
    for name in VARIANTS:
        bm = variants[name]
        f1 = onset_f1(bm.notes, reference.notes)
        play = unplayability(bm.notes)
        pre = "" if name == SHIPPED else f"{name}_"
        row[f"{pre}onset_f1"] = f1.f1
        row[f"{pre}snap_validity"] = snap_validity(bm.notes, points).valid_fraction
        row[f"{pre}unplayability"] = play.rate
        if real_ms is not None:
            row.update(exact_snap_fields(pre, hit_times(bm), real_ms, grid, name == SHIPPED))
        for k, v in violation_counts(play).items():
            row[f"{pre}{k}"] = v
        if name == SHIPPED:
            stats = note_statistics(bm)
            act_score = activity_score(bm.notes, act, grid, span_ms=span)
            row.update({
                "onset_precision": f1.precision, "onset_recall": f1.recall,
                "onset_mae_ms": f1.mean_abs_error_ms,
                "pattern_kl": pattern_divergence(bm.notes, reference.notes),
                "realised_nps": stats.avg_nps, "generated_notes": stats.n_notes,
                "don_ratio": stats.don_ratio, "big_ratio": stats.big_ratio,
                "quiet_note_rate": act_score.quiet_note_rate,
                "onset_miss_rate": act_score.strong_onset_miss_rate,
            })
    # The old measure, kept so the new numbers can be read against the handover.
    raw = variants["legacy_raw"]
    row["legacy_snap_old"] = row.pop("legacy_raw_snap_validity")
    row["legacy_snap_raw"] = snap_validity(raw.notes, points,
                                           tolerance_ms=RAW_TOLERANCE_MS).valid_fraction
    fixes = variants["_repair"]
    row["repair_rate"] = fixes.rate
    row["repair_fixes"] = fixes.total
    return row


def average_rows(rows: list[dict]) -> dict:
    """Mean over seeds; the spread of the key metrics is kept as *_sd."""
    out = {}
    for k in rows[0]:
        vals = [r[k] for r in rows]
        out[k] = float(np.mean(vals))
    if len(rows) > 1:
        for k in ("onset_f1", "snap_validity", "unplayability"):
            out[f"{k}_sd"] = float(np.std([r[k] for r in rows]))
    return out


def summarise(rows: list[dict]) -> dict:
    def mean(key: str) -> float:
        vals = [r[key] for r in rows if key in r and r[key] == r[key]]
        return float(statistics.fmean(vals)) if vals else float("nan")

    requested = [r["requested_sr"] for r in rows]
    realised = [r["realised_nps"] for r in rows]
    sr_correlation = (
        float(np.corrcoef(requested, realised)[0, 1])
        if len(rows) > 2 and statistics.pstdev(requested) > 1e-6
           and statistics.pstdev(realised) > 1e-6
        else float("nan")
    )
    with_nps = [r for r in rows if r["requested_nps"] > 0]
    nps_error = (float(statistics.fmean(abs(r["realised_nps"] - r["requested_nps"])
                                        for r in with_nps)) if with_nps else float("nan"))

    keys = sorted({k for r in rows for k in r if isinstance(r[k], (int, float))})
    summary = {k: mean(k) for k in keys if k not in ("requested_sr", "requested_nps", "bpm")}
    summary.update({
        "n_maps": len(rows),
        "sr_correlation": sr_correlation,
        "nps_error": nps_error,
        "note_ratio": mean("generated_notes") / max(mean("reference_notes"), 1e-6),
        "groups": {
            "difficulty": group(rows, "requested_sr", SR_BANDS),
            "bpm": group(rows, "bpm", BPM_BANDS),
        },
    })
    return summary


def sr_band(sr: float) -> str:
    return next(label for lo, hi, label in SR_BANDS if lo <= sr < hi)


def add_feel(pool: dict, band: str, side: str, notes, grid: Grid) -> None:
    """Pool feel counts per SR band and side ('model' or 'ranked'), over maps and seeds."""
    slot = pool.setdefault(band, {}).setdefault(side, {f: Counter() for f in FEEL_FAMILIES})
    for family, counts in feel_counts(notes, grid).items():
        slot[family].update(counts)


def feel_table(pool: dict) -> dict:
    """JS divergence, model against ranked, per band and family, plus all bands pooled."""
    out, total = {}, {side: {f: Counter() for f in FEEL_FAMILIES} for side in ("model", "ranked")}
    for band, sides in pool.items():
        if len(sides) < 2:
            continue
        out[band] = {f: js_divergence(sides["model"][f], sides["ranked"][f]) for f in FEEL_FAMILIES}
        for side in total:
            for f in FEEL_FAMILIES:
                total[side][f].update(sides[side][f])
    out["all"] = {f: js_divergence(total["model"][f], total["ranked"][f]) for f in FEEL_FAMILIES}
    return out


def nanmean(vals) -> float:
    vals = [v for v in vals if v == v]
    return float(np.mean(vals)) if vals else float("nan")


def group(rows: list[dict], key: str, bands) -> dict:
    out = {}
    for lo, hi, label in bands:
        sel = [r for r in rows if lo <= r[key] < hi]
        if not sel:
            continue
        out[label] = {
            "maps": len(sel),
            "onset_f1": float(np.mean([r["onset_f1"] for r in sel])),
            "exact_snap": nanmean([r.get("exact_snap", float("nan")) for r in sel]),
            "snap_validity": float(np.mean([r["snap_validity"] for r in sel])),
            "unplayability": float(np.mean([r["unplayability"] for r in sel])),
            "note_ratio": float(np.sum([r["generated_notes"] for r in sel])
                                / max(np.sum([r["reference_notes"] for r in sel]), 1)),
            "quiet_note_rate": float(np.mean([r["quiet_note_rate"] for r in sel])),
            "onset_miss_rate": float(np.mean([r["onset_miss_rate"] for r in sel])),
        }
    return out


def report(summary: dict, args) -> bool:
    print(f"\n{'=' * 70}")
    print(f"{summary['n_maps']} held-out maps, {args.steps} steps, guidance {args.cfg_scale}, "
          f"{args.seeds} seed(s) each")
    print(f"{'=' * 70}")

    for key, (direction, target) in TARGETS.items():
        value = summary[key]
        if value != value:
            verdict = "n/a"
        else:
            ok = value > target if direction == ">" else value < target
            verdict = "PASS" if ok else "below target" if direction == ">" else "over target"
        print(f"  {key:<16s} {value:>8.4f}   target {direction} {target:<7.3f}  {verdict}")

    print(f"\n  snap validity, by what is measured            model    ranked map")
    print(f"  {'raw frames @5 ms  (the old measure)':<44s}{summary['legacy_snap_old']:>7.3f}"
          f"   {summary['ref_snap_old']:>7.3f}   <- ceiling of the representation")
    print(f"  {'raw frames @10 ms (half a frame)':<44s}{summary['legacy_snap_raw']:>7.3f}"
          f"   {summary['ref_snap_raw']:>7.3f}")
    print(f"  {'legacy decode + snap @5 ms':<44s}{summary['legacy_final_snap_validity']:>7.3f}")
    print(f"  {'grid decode @5 ms':<44s}{summary['grid_raw_snap_validity']:>7.3f}")
    print(f"  {'grid decode + repair @5 ms  (shipped)':<44s}{summary['snap_validity']:>7.3f}")

    print(f"\n  playability            too fast  big streams  long overlap  zero longs    rate")
    for label, pre in (("legacy raw", "legacy_raw_"), ("legacy final", "legacy_final_"),
                       ("grid raw", "grid_raw_"), ("grid + repair", ""), ("ranked map", "ref_")):
        print(f"  {label:<20s}{summary[pre + 'too_fast']:>10.2f}"
              f"{summary[pre + 'big_note_streams']:>13.2f}{summary[pre + 'overlapping_longs']:>14.2f}"
              f"{summary[pre + 'zero_length_longs']:>12.2f}{summary[pre + 'unplayability']:>9.4f}")
    print(f"  repair changed {summary['repair_rate']:.1%} of notes on average "
          f"(high means the model itself still breaks the rules)")

    print(f"\n  onset F1: legacy raw {summary['legacy_raw_onset_f1']:.4f}  "
          f"legacy final {summary['legacy_final_onset_f1']:.4f}  "
          f"grid raw {summary['grid_raw_onset_f1']:.4f}  shipped {summary['onset_f1']:.4f}")
    print(f"  {'onset precision':<16s} {summary['onset_precision']:>8.4f}")
    print(f"  {'onset recall':<16s} {summary['onset_recall']:>8.4f}")
    print(f"  {'onset MAE (ms)':<16s} {summary['onset_mae_ms']:>8.2f}")
    print(f"  {'pattern KL':<16s} {summary['pattern_kl']:>8.4f}")
    print(f"  {'note count ratio':<16s} {summary['note_ratio']:>8.4f}   "
          f"(1.0 = same density as the reference)")
    print(f"  {'nps rel. error':<16s} {summary['nps_rel_error']:>8.4f}   "
          f"(mean |realised - reference| / reference, per map)")
    print(f"  {'don ratio':<16s} {summary['don_ratio']:>8.4f}")
    if args.seeds > 1:
        print(f"  seed spread (sd): F1 {summary.get('onset_f1_sd', float('nan')):.4f}  "
              f"snap {summary.get('snap_validity_sd', float('nan')):.4f}  "
              f"unplay {summary.get('unplayability_sd', float('nan')):.4f}")

    if "exact_snap" in summary:
        print(f"\n  exact snap: matched onsets on the mapper's own line, against the .osu's ms")
        print(f"  shipped {summary['exact_snap']:.3f}   grid raw {summary['grid_raw_exact_snap']:.3f}   "
              f"legacy final {summary['legacy_final_exact_snap']:.3f}   "
              f"ranked map through frames {summary['ref_exact_snap']:.3f}  <- ceiling")
        cells = []
        for d in EXACT_SNAP_DIVISORS:
            n = summary.get(f"xs_n_{d}", 0.0)
            if n > 0:
                cells.append(f"1/{d} {summary[f'xs_hit_{d}'] / n:.3f} ({n * summary['n_maps']:.0f})")
        print("  by the mapper's divisor: " + "   ".join(cells))
        print(f"  reference notes on no scored line, left out: {summary['unsnapped_share']:.2%}")

    if summary.get("feel"):
        print(f"\n  feel, model against ranked maps of the same SR band "
              f"(Jensen-Shannon: 0 same, 1 disjoint)")
        print(f"  {'band':<21s}" + "".join(f"{f:>9s}" for f in FEEL_FAMILIES))
        for band, js in summary["feel"].items():
            print(f"  {band:<21s}" + "".join(f"{js[f]:>9.3f}" for f in FEEL_FAMILIES))

    print(f"\n  audio agreement           model   ranked map")
    print(f"  {'quiet-section notes':<24s}{summary['quiet_note_rate']:>7.3f}   "
          f"{summary['ref_quiet_note_rate']:>7.3f}   share of notes in the quietest 20%")
    print(f"  {'missed strong onsets':<24s}{summary['onset_miss_rate']:>7.3f}   "
          f"{summary['ref_onset_miss_rate']:>7.3f}   share of loud on-grid attacks left empty")
    if args.grid_probe:
        print(f"  {'grid follow':<24s}{summary['grid_follow']:>7.3f}   "
              f"(1 = follows the given grid, 0 = ignores it)")

    for name, table in summary["groups"].items():
        print(f"\n  by {name:<18s} maps   F1     exact  snap   unplay  notes  quiet  miss")
        for label, g in table.items():
            print(f"  {label:<21s}{g['maps']:>4d}  {g['onset_f1']:.3f}  {g['exact_snap']:.3f}  "
                  f"{g['snap_validity']:.3f}  "
                  f"{g['unplayability']:.4f}  {g['note_ratio']:.2f}   {g['quiet_note_rate']:.2f}   "
                  f"{g['onset_miss_rate']:.2f}")

    if args.harness:
        cols = ("onset_f1", "pattern_kl", "nps_rel_error", "quiet_note_rate",
                "onset_miss_rate", "raw_too_fast", "repair_rate", "threshold")
        print("\n  harness            F1   patKL  npsErr  quiet   miss  rawFast  repair  thresh")
        shipped = [summary["onset_f1"], summary["pattern_kl"], summary["nps_rel_error"],
                   summary["quiet_note_rate"], summary["onset_miss_rate"],
                   summary["grid_raw_too_fast"], summary["repair_rate"], float("nan")]
        table = [("shipped", shipped)] + [
            (name, [summary.get(f"h_{name}_{c}", float("nan")) for c in cols]) for name in HARNESS]
        for label, v in table:
            print(f"  {label:<14s}{v[0]:>7.3f}{v[1]:>8.3f}{v[2]:>8.3f}{v[3]:>7.3f}{v[4]:>7.3f}"
                  f"{v[5]:>9.1f}{v[6]:>8.3f}{v[7]:>8.3f}")

    gate_b = summary["onset_f1"] > GATE_B_F1
    print(f"\n  GATE B  onset F1 > {GATE_B_F1}: "
          f"{'PASSED' if gate_b else 'FAILED'}  ({summary['onset_f1']:.4f})")
    if not gate_b:
        print("\n  The model is not following the audio. More steps will not fix")
        print("  this. Check that training windows pair a chart with the same")
        print("  frames of mel (tests/test_dataset.py), and that the audio")
        print("  encoder levels land on the U-Net's resolutions.")
    return gate_b


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
    ap.add_argument("--per-band", type=int, default=None,
                    help="the fixed benchmark pool: this many held-out maps from each SR "
                         "band (fewer where a band has fewer), instead of --n-maps at random. "
                         "The random pool is mostly easy maps, with one 7*+ in 30")
    ap.add_argument("--seeds", type=int, default=1,
                    help="samples per map; metrics are averaged and their spread reported")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg-scale", type=float, default=4.0)
    ap.add_argument("--window-frames", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the checkpoint's onset threshold")
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
                         "measure whether the notes follow it (one extra sample per map)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-sr", type=float, default=0.0,
                    help="only held-out maps at or above this star rating. The default "
                         "pool is mostly easy maps; --min-sr 5.5 is the hard-map benchmark")
    ap.add_argument("--max-sr", type=float, default=99.0)
    ap.add_argument("--harness", action="store_true",
                    help="also score the decode-side harness candidates (%s) side by side "
                         "with the shipped decode" % ", ".join(HARNESS))
    ap.add_argument("--probs-cache", type=Path, default=None,
                    help="keep each sampled chart here and reuse it, so decode-side "
                         "changes can be compared on identical samples without resampling")
    args = ap.parse_args()

    print(describe())

    for path, what in ((args.diffusion, "diffusion checkpoint"), (args.ae, "autoencoder")):
        if not path.exists():
            print(f"ERROR: {what} not found: {path}")
            return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, threshold, ckpt = load_model(args.diffusion, args.ae, device, not args.no_ema)
    if args.threshold is not None:
        threshold = args.threshold
    window = args.window_frames or ckpt.get("window_frames", WINDOW_FRAMES_DEFAULT)
    header = checkpoint_header(args.diffusion, ckpt)
    print(f"Evaluating {header['evaluated']}: step {header['step']}"
          + (f" (latest training step {header['latest_training_step']} is in last.pt)"
             if "latest_training_step" in header else "")
          + f", features {header['features']}, profile {ckpt.get('profile')}, "
            f"threshold {threshold}, window {window}")

    reader = ShardReader(args.shards)
    note_times = load_note_times(reader)
    if note_times is None:
        print("no note_times.npz, so no exact snap (python scripts/build_note_times.py)")
    _, val_idx = split_indices(reader, val_ratio=0.05)
    print(f"Held-out pool: {len(val_idx)} maps")
    if not val_idx:
        print("ERROR: validation split is empty")
        return 1

    val_idx = [i for i in val_idx
               if args.min_sr <= float(reader.records[i].get("difficulty", 0.0)) < args.max_sr]
    print(f"  {len(val_idx)} of them between {args.min_sr}* and {args.max_sr}*")

    rng = np.random.default_rng(args.seed)
    if args.per_band:
        chosen = []
        for lo, hi, label in SR_BANDS:
            band = [i for i in val_idx if lo <= float(reader.records[i].get("difficulty", 0.0)) < hi]
            chosen += list(rng.permutation(band)[:args.per_band])
            print(f"  {label}: {min(len(band), args.per_band)} of {len(band)}")
    else:
        chosen = rng.permutation(val_idx)[:args.n_maps]

    rows = []
    feel_pool: dict = {}
    for n, idx in enumerate(chosen):
        idx = int(idx)
        record = reader.records[idx]
        frames = min(reader.chart_length(idx), reader.mel_length(idx), args.max_frames)
        if frames < window:
            continue

        points = decode_timing_points(record["timing_points"])
        bpm, offset, meter = dominant_tempo(points)
        grid = Grid(points)

        mel = torch.from_numpy(reader.mel_window(idx, 0, frames)).unsqueeze(0)
        timing_np = build_timing_stream(points, frames, start_frame=0)
        timing = torch.from_numpy(timing_np).unsqueeze(0)

        reference_chart = reader.chart_window(idx, 0, frames)
        reference = tensor_to_beatmap(reference_chart, bpm=bpm, offset_ms=offset,
                                      meter=meter, timing_points=points)

        motif = motif_mask = None
        if args.use_reference_motif:
            motif = compute_motif(reference_chart, beat_frames_from_timing(timing_np))
            motif_mask = np.ones_like(motif)

        requested_sr = float(record.get("difficulty", 5.0))
        requested_nps = float(record.get("avg_nps", 0.0))

        def sample(timing_tensor: torch.Tensor, seed: int = 0) -> np.ndarray:
            cached = None
            if args.probs_cache is not None and timing_tensor is timing:
                cached = args.probs_cache / (f"{header['step']}_{idx}_{args.seed}_{seed}_"
                                             f"{args.steps}_{args.cfg_scale}_{frames}.npy")
                if cached.exists():
                    return np.load(cached)
            probs = _sample(timing_tensor, seed)
            if cached is not None:
                cached.parent.mkdir(parents=True, exist_ok=True)
                np.save(cached, probs)
            return probs

        def _sample(timing_tensor: torch.Tensor, seed: int) -> np.ndarray:
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
                generator=torch.Generator(device=device).manual_seed(
                    args.seed + idx + 7919 * seed),
            )[0].cpu().numpy()

        act = activity(mel[0].numpy())
        span = (0.0, frames * FRAME_MS)
        real_ms = None
        if note_times is not None:
            real_ms = note_times[idx][note_times[idx] < span[1]]

        band = sr_band(requested_sr)
        # The ranked side: the ranked chart's own frames through the shipped
        # decoder. `reference` is the legacy frame decode, whose notes sit on
        # 20 ms frame times rather than lines (exact snap ~0.3, gaps off-snap).
        ceiling = decode_on_grid(reference_chart, points, threshold=0.5)
        add_feel(feel_pool, band, "ranked", ceiling.notes, grid)

        per_seed = []
        for seed in range(args.seeds):
            probs = sample(timing, seed)
            variants = decode_variants(probs, points, grid, threshold, bpm, offset, meter)
            add_feel(feel_pool, band, "model", variants[SHIPPED].notes, grid)
            scored = score_variants(variants, reference, act, grid, span, real_ms)
            if args.harness:
                scored.update(harness_rows(probs, points, grid, threshold, requested_nps,
                                           act, reference, span))
            per_seed.append(scored)
        row = average_rows(per_seed)

        ref_play = unplayability(reference.notes)
        if real_ms is not None:
            row.update(exact_snap_fields("ref_", hit_times(ceiling), real_ms, grid, False))
        ref_act = activity_score(reference.notes, act, grid, span_ms=span)
        ref_stats = note_statistics(reference)
        row.update({
            "map": f"{record.get('title', '?')} [{record.get('version', '?')}]",
            "requested_sr": requested_sr,
            "requested_nps": requested_nps,
            "bpm": float(record.get("bpm") or bpm),
            "reference_nps": ref_stats.avg_nps,
            "reference_notes": ref_stats.n_notes,
            "nps_rel_error": abs(row["realised_nps"] - ref_stats.avg_nps) / max(ref_stats.avg_nps, 1e-6),
            "ref_snap_old": snap_validity(reference.notes, points).valid_fraction,
            "ref_snap_raw": snap_validity(reference.notes, points,
                                          tolerance_ms=RAW_TOLERANCE_MS).valid_fraction,
            "ref_unplayability": ref_play.rate,
            **{f"ref_{k}": v for k, v in violation_counts(ref_play).items()},
            "ref_quiet_note_rate": ref_act.quiet_note_rate,
            "ref_onset_miss_rate": ref_act.strong_onset_miss_rate,
            "grid_follow": (grid_follow_fraction(sample, points, frames, threshold)
                            if args.grid_probe else float("nan")),
        })
        rows.append(row)

        print(f"  [{n + 1}/{len(chosen)}] F1 {row['onset_f1']:.3f}  "
              f"snap {row['snap_validity']:.3f} (old measure {row['legacy_snap_old']:.3f}, "
              f"ranked {row['ref_snap_old']:.3f})  unplay {row['unplayability']:.4f}  "
              f"notes {row['generated_notes']:.0f} vs {row['reference_notes']}  "
              f"repair {row['repair_rate']:.1%}  {row['map'][:40]}")

    if not rows:
        print("ERROR: no map was long enough to evaluate")
        return 1

    summary = summarise(rows)
    summary["feel"] = feel_table(feel_pool)
    gate_b = report(summary, args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"checkpoint": header, "summary": summary, "maps": rows},
                                   indent=2, default=str))
    print(f"\nWrote {args.out}")
    return 0 if gate_b else 1


if __name__ == "__main__":
    raise SystemExit(main())
