"""
taiko/data/timing_refine.py

Post-generation timing alignment (Mug-Diffusion mug/data/utils.py, taiko-adapted).

Fits BPM + offset from generated note times, then snaps hits to the beat grid.
Used at inference so charts align with music even when the model drifts in phase.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from taiko.data.beat_snap import SNAP_DIVISORS, detect_bpm
from taiko.data.osu_parser import TaikoBeatmap, TaikoNote, TimingPoint
from taiko.data.tensor_repr import FRAME_MS

EPSILON_MS = 10.0
GRID_DIVISORS = [1, 2, 4, 3, 6, 8, 16, 12]  # Mug order, taiko-relevant snaps


def _normalize_bpm(bpm: float) -> float:
    while bpm < 150:
        bpm *= 2
    while bpm >= 300:
        bpm /= 2
    return bpm


def _test_timing(
    time_list: np.ndarray,
    test_bpm: float,
    test_offset: float,
    div: int = 1,
    refine: bool = False,
) -> tuple[float, np.ndarray, float, float]:
    cur_offset = test_offset
    cur_bpm = test_bpm

    gap = 60_000.0 / (test_bpm * div)
    delta = time_list - test_offset
    meter = delta / gap
    meter_round = np.round(meter)
    timing_error = np.abs(meter - meter_round)
    valid = (timing_error < EPSILON_MS / gap).astype(np.int32)
    valid_count = int(np.sum(valid))

    if valid_count >= 2 and refine:
        x = meter_round.reshape(-1, 1)
        y = time_list
        w = valid.astype(np.float64)
        sw = np.sqrt(w)
        design = np.column_stack([x.ravel() * sw, sw])
        target = y * sw
        coef, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        slope, intercept = float(coef[0]), float(coef[1])
        if slope != 0 and np.isfinite(slope) and np.isfinite(intercept):
            cur_offset = intercept
            cur_bpm = _normalize_bpm(60_000.0 / slope / 4)

    valid_ratio = valid_count / max(test_bpm, 1e-6)
    return valid_ratio, valid, cur_bpm, cur_offset


def fit_timing_from_notes(
    note_times_ms: list[int] | np.ndarray,
    verbose: bool = False,
) -> tuple[float, float]:
    """
    Estimate BPM and offset so note times sit on a beat grid.
    Port of Mug-Diffusion timing().
    """
    times = np.asarray(note_times_ms, dtype=np.float32)
    if len(times) < 2:
        raise ValueError("Need at least 2 note times for timing fit")

    offset = float(times[0])
    best_bpm: Optional[float] = None
    best_offset: Optional[float] = None
    best_valid_ratio = -1.0

    t0 = time.time()
    for test_bpm in np.arange(150, 300, 0.1):
        valid_ratio, _, cur_bpm, cur_offset = _test_timing(
            times, test_bpm, offset, div=1, refine=False
        )
        if valid_ratio > best_valid_ratio:
            valid_ratio, _, cur_bpm, cur_offset = _test_timing(
                times, test_bpm, offset, div=1, refine=True
            )
            best_valid_ratio = valid_ratio
            best_bpm = cur_bpm
            best_offset = cur_offset
            if verbose:
                print(
                    f"[timing] valid={valid_ratio:.3f} test_bpm={test_bpm:.1f} "
                    f"-> bpm={cur_bpm:.2f} offset={cur_offset:.1f}"
                )

            gap = 60_000.0 / cur_bpm
            for test_off in np.arange(best_offset, best_offset - gap, -gap / 4):
                valid_ratio, _, cur_bpm, cur_offset = _test_timing(
                    times, cur_bpm, test_off, div=1, refine=False
                )
                if valid_ratio > best_valid_ratio:
                    valid_ratio, _, cur_bpm, cur_offset = _test_timing(
                        times, cur_bpm, test_off, div=1, refine=True
                    )
                    best_valid_ratio = valid_ratio
                    best_bpm = cur_bpm
                    best_offset = cur_offset
                    if verbose:
                        print(
                            f"[timing] valid={valid_ratio:.3f} offset search "
                            f"-> bpm={cur_bpm:.2f} offset={cur_offset:.1f}"
                        )

    assert best_bpm is not None and best_offset is not None
    _, _, best_bpm, best_offset = _test_timing(times, best_bpm, best_offset, div=16, refine=False)
    _, _, best_bpm, best_offset = _test_timing(times, best_bpm, best_offset, div=6, refine=False)

    if verbose:
        print(f"[timing] done in {time.time() - t0:.2f}s bpm={best_bpm:.2f} offset={best_offset:.1f}")

    return best_bpm, best_offset


def snap_ms_to_grid(time_ms: float, bpm: float, offset_ms: float) -> int:
    """Snap one timestamp to the nearest taiko subdivision (Mug format_time)."""
    for div in GRID_DIVISORS:
        gap = 60_000.0 / (bpm * div)
        meter = (time_ms - offset_ms) / gap
        meter_round = round(meter)
        if abs(meter - meter_round) < EPSILON_MS / gap:
            return int(meter_round * gap + offset_ms)
    return int(time_ms)


def snap_beatmap_notes(bm: TaikoBeatmap, bpm: float, offset_ms: float) -> TaikoBeatmap:
    """Snap all note start/end times to the grid and dedupe within one frame."""
    snapped: list[TaikoNote] = []
    for note in bm.notes:
        t = snap_ms_to_grid(note.time, bpm, offset_ms)
        end = note.end_time
        if note.is_long and end > note.time:
            end = snap_ms_to_grid(end, bpm, offset_ms)
            if end <= t:
                end = t + int(FRAME_MS)
        snapped.append(TaikoNote(time=t, note_type=note.note_type, end_time=end))

    snapped.sort(key=lambda n: n.time)
    deduped: list[TaikoNote] = []
    last_t = -999
    for note in snapped:
        if note.time - last_t >= int(FRAME_MS):
            deduped.append(note)
            last_t = note.time

    bm.notes = deduped
    return bm


def resolve_bpm_offset(
    note_times_ms: list[int],
    audio_bpm: float,
    audio_offset_ms: float,
    min_notes_for_fit: int = 20,
) -> tuple[float, float, str]:
    """
    Choose BPM/offset: fit from notes when enough hits, else librosa defaults.
    Returns (bpm, offset_ms, source_label).
    """
    if len(note_times_ms) < min_notes_for_fit:
        return audio_bpm, audio_offset_ms, "audio"

    try:
        fit_bpm, fit_offset = fit_timing_from_notes(note_times_ms, verbose=False)
        return fit_bpm, fit_offset, "notes"
    except (ValueError, AssertionError):
        return audio_bpm, audio_offset_ms, "audio"


def apply_timing_refinement(
    bm: TaikoBeatmap,
    audio_path: Optional[str] = None,
    audio_bpm: Optional[float] = None,
    audio_offset_ms: Optional[float] = None,
    verbose: bool = True,
) -> TaikoBeatmap:
    """
    Full post-process: audio BPM hint -> fit from notes -> snap -> update timing points.
    """
    if not bm.notes:
        if verbose:
            print("[timing] no notes — skipping refinement")
        return bm

    if audio_bpm is None or audio_offset_ms is None:
        if audio_path is None:
            audio_bpm = audio_bpm or 180.0
            audio_offset_ms = audio_offset_ms or 0.0
        else:
            bpm, beat_times = detect_bpm(audio_path)
            audio_bpm = bpm
            audio_offset_ms = (
                float(beat_times[0] * 1000) if len(beat_times) > 0 else 0.0
            )

    note_times = [n.time for n in bm.notes if not n.is_long]
    if not note_times:
        note_times = [n.time for n in bm.notes]

    bpm, offset_ms, source = resolve_bpm_offset(
        note_times, audio_bpm, audio_offset_ms
    )
    if verbose:
        print(f"[timing] BPM={bpm:.2f} offset={offset_ms:.1f} ms (from {source})")
        if source == "notes":
            print(f"[timing] audio hint was BPM={audio_bpm:.1f} offset={audio_offset_ms:.1f}")

    snap_beatmap_notes(bm, bpm, offset_ms)

    bm.timing_points = [
        TimingPoint(
            time=int(offset_ms),
            beat_length=60_000.0 / bpm,
            meter=4,
            uninherited=True,
        )
    ]
    bm.compute_stats()
    return bm
