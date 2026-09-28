"""
taiko/data/audio_activity.py

How well a map's notes agree with the audio, as two numbers neither of which
is zero for a human mapper, so both are read against the ranked map's value:

    quiet-section notes   share of notes in the song's quietest 20% of frames
    missed strong onsets  share of loud on-grid attacks with no note in 40 ms

gate_notes() is the optional post-filter that drops isolated notes in quiet
passages with no attack under them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from taiko.data.frames import FRAME_MS
from taiko.data.grid import Grid
from taiko.data.osu_parser import TaikoNote

QUIET_QUANTILE = 0.20
NOTE_WINDOW_MS = 40.0      # a strong onset counts as marked if a note is this close
ON_GRID_MS = 15.0          # a frame is 20 ms, so a frame time can sit 10 ms off its line
ONSET_DIVISORS = (1, 2, 4)
STRONG_STD = 1.5           # strong onset: flux above mean + this many std


@dataclass
class Activity:
    loudness: np.ndarray   # [T] mean mel per frame
    flux: np.ndarray       # [T] positive spectral flux
    quiet: np.ndarray      # [T] bool, the quietest QUIET_QUANTILE of frames
    strong: np.ndarray     # [T] bool, loud local-peak attacks (not yet grid-filtered)
    threshold: float


@dataclass
class ActivityScore:
    quiet_note_rate: float
    strong_onset_miss_rate: float
    strong_onsets: int
    missed_times_ms: list[int] = field(default_factory=list)


def activity(mel: np.ndarray) -> Activity:
    """mel: [n_mels, T], song-normalised."""
    mel = np.asarray(mel, dtype=np.float32)
    loudness = mel.mean(axis=0)
    flux = np.zeros_like(loudness)
    flux[1:] = np.maximum(mel[:, 1:] - mel[:, :-1], 0.0).mean(axis=0)

    quiet = loudness <= np.quantile(loudness, QUIET_QUANTILE)
    threshold = float(flux.mean() + STRONG_STD * flux.std())
    prev = np.concatenate([[-np.inf], flux[:-1]])
    nxt = np.concatenate([flux[1:], [-np.inf]])
    strong = (flux > threshold) & (flux >= prev) & (flux >= nxt) & ~quiet
    return Activity(loudness, flux, quiet, strong, threshold)


def _frame(time_ms: float, n: int) -> int:
    return min(max(int(round(time_ms / FRAME_MS)), 0), n - 1)


def activity_score(notes: Sequence[TaikoNote], act: Activity, grid: Grid,
                   span_ms: Optional[tuple[float, float]] = None) -> ActivityScore:
    n = len(act.loudness)
    lo, hi = span_ms if span_ms is not None else (0.0, n * FRAME_MS)
    times = np.asarray(sorted(nt.time for nt in notes if lo <= nt.time < hi), dtype=np.float64)

    quiet_rate = (float(np.mean([act.quiet[_frame(t, n)] for t in times]))
                  if times.size else 0.0)

    onset_ms = [f * FRAME_MS for f in np.flatnonzero(act.strong)]
    onset_ms = [t for t in onset_ms if lo <= t < hi
                and grid.distance_ms(t, ONSET_DIVISORS) <= ON_GRID_MS]
    missed = []
    for t in onset_ms:
        i = np.searchsorted(times, t)
        near = [abs(times[j] - t) for j in (i - 1, i) if 0 <= j < times.size]
        if not near or min(near) > NOTE_WINDOW_MS:
            missed.append(int(round(t)))
    miss_rate = len(missed) / len(onset_ms) if onset_ms else 0.0
    return ActivityScore(quiet_rate, miss_rate, len(onset_ms), missed)


def gate_notes(notes: Sequence[TaikoNote], act: Activity,
               grid: Grid) -> tuple[list[TaikoNote], list[TaikoNote]]:
    """
    Drop notes that are in a quiet frame, have no attack within 40 ms, and
    are isolated (no neighbour closer than half a beat). Long notes and
    streams are kept. ponytail: isolation is a heuristic; it cannot tell a
    deliberate sparse rhythm from filler.
    """
    n = len(act.loudness)
    ordered = sorted(notes, key=lambda x: x.time)
    reach = int(np.ceil(NOTE_WINDOW_MS / FRAME_MS))
    kept, dropped = [], []
    for i, note in enumerate(ordered):
        f = _frame(note.time, n)
        gaps = [abs(ordered[j].time - note.time) for j in (i - 1, i + 1) if 0 <= j < len(ordered)]
        isolated = min(gaps, default=np.inf) > grid.section_at(note.time).ms_per_beat / 2
        attack = act.flux[max(f - reach, 0): f + reach + 1].max() > act.threshold
        if act.quiet[f] and isolated and not attack and not note.is_long:
            dropped.append(note)
        else:
            kept.append(note)
    return kept, dropped
