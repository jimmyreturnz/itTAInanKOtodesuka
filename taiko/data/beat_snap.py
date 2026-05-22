"""
taiko/data/beat_snap.py

BPM detection from audio and beat-snapping of generated note times.

Two steps:
  1. Detect BPM + beat positions from audio using librosa
  2. Snap generated note times to nearest valid beat subdivision

Beat subdivisions supported (standard taiko snaps):
  1/1, 1/2, 1/3, 1/4, 1/6, 1/8, 1/12, 1/16
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional


# Snap divisors to try, in order of preference
# Higher = finer grid = more note positions available
SNAP_DIVISORS = [1, 2, 3, 4, 6, 8, 12, 16]


# ---------------------------------------------------------------------------
# BPM detection
# ---------------------------------------------------------------------------

def detect_bpm(audio_path: str | Path, sr: int = 22050) -> tuple[float, np.ndarray]:
    """
    Detect BPM and beat positions from audio file.

    Returns:
        bpm:        detected tempo in BPM
        beat_times: array of beat timestamps in seconds
    """
    try:
        import librosa
        y, sr = librosa.load(str(audio_path), sr=sr, mono=True)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        # librosa returns array in newer versions
        if hasattr(tempo, '__len__'):
            tempo = float(tempo[0])
        else:
            tempo = float(tempo)

        return tempo, beat_times

    except ImportError:
        raise ImportError("librosa required for BPM detection: pip install librosa")


def detect_bpm_from_waveform(y: np.ndarray, sr: int = 22050) -> tuple[float, np.ndarray]:
    """Same as detect_bpm but accepts a pre-loaded waveform."""
    import librosa
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    if hasattr(tempo, '__len__'):
        tempo = float(tempo[0])
    else:
        tempo = float(tempo)
    return tempo, beat_times


# ---------------------------------------------------------------------------
# Timing point builder
# ---------------------------------------------------------------------------

def build_timing_points_from_bpm(
    bpm: float,
    offset_ms: float = 0.0,
) -> list[dict]:
    """
    Build a single timing point from detected BPM.

    Args:
        bpm:       detected tempo
        offset_ms: first beat offset in ms (from beat_times[0] * 1000)

    Returns:
        list of timing point dicts compatible with OsuTaikoParser.TimingPoint
    """
    beat_length = 60_000.0 / bpm  # ms per beat
    return [{
        "time":        int(offset_ms),
        "beat_length": beat_length,
        "meter":       4,
        "uninherited": True,
        "bpm":         bpm,
    }]


# ---------------------------------------------------------------------------
# Beat snapping
# ---------------------------------------------------------------------------

def snap_time_to_beat(
    time_ms: float,
    beat_length_ms: float,
    offset_ms: float,
    divisors: list[int] = SNAP_DIVISORS,
    tolerance_ms: float = 15.0,
) -> Optional[int]:
    """
    Snap a note time to the nearest beat subdivision.

    Args:
        time_ms:        note time in ms
        beat_length_ms: ms per beat (60000 / BPM)
        offset_ms:      first beat offset in ms
        divisors:       beat subdivisions to try
        tolerance_ms:   max allowed snap distance in ms

    Returns:
        snapped time in ms, or None if no snap within tolerance
    """
    best_time = None
    best_dist = float("inf")

    for div in divisors:
        grid_ms   = beat_length_ms / div
        relative  = time_ms - offset_ms
        nearest_n = round(relative / grid_ms)
        snapped   = offset_ms + nearest_n * grid_ms
        dist      = abs(time_ms - snapped)

        if dist < best_dist:
            best_dist = dist
            best_time = snapped

    if best_dist <= tolerance_ms:
        return int(round(best_time))
    return None   # too far from any grid point — discard or keep original


def snap_note_events(
    note_events: list[tuple[int, str]],
    bpm: float,
    offset_ms: float,
    divisors: list[int] = SNAP_DIVISORS,
    tolerance_ms: float = 20.0,
    discard_unsnapped: bool = False,
) -> list[tuple[int, str]]:
    """
    Snap all note events to the nearest beat subdivision.

    Args:
        note_events:       list of (time_ms, token_str)
        bpm:               song BPM
        offset_ms:         first beat offset in ms
        tolerance_ms:      max snap distance before discarding
        discard_unsnapped: if True, drop notes that can't be snapped

    Returns:
        list of (snapped_time_ms, token_str), sorted by time
    """
    beat_length_ms = 60_000.0 / bpm
    snapped = []
    discarded = 0

    for time_ms, tok in note_events:
        result = snap_time_to_beat(time_ms, beat_length_ms, offset_ms, divisors, tolerance_ms)
        if result is not None:
            snapped.append((result, tok))
        else:
            if discard_unsnapped:
                discarded += 1
            else:
                # Keep original time if can't snap
                snapped.append((time_ms, tok))

    if discarded > 0:
        print(f"  Discarded {discarded} unsnappable notes")

    # Sort and deduplicate (two notes snapped to same time → keep first)
    snapped.sort(key=lambda x: x[0])
    deduped = []
    last_time = -999
    for t, tok in snapped:
        if t != last_time:
            deduped.append((t, tok))
            last_time = t

    return deduped
