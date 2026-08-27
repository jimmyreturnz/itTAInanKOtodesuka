"""Quick tests for Mug-style timing refinement."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from taiko.data.osu_parser import TaikoBeatmap, TaikoNote
from taiko.data.timing_refine import (
    fit_timing_from_notes,
    snap_ms_to_grid,
    apply_timing_refinement,
)


def test_fit_timing_on_regular_grid():
    bpm = 180.0
    offset = 500.0
    beat_ms = 60_000.0 / bpm
    times = [offset + i * beat_ms for i in range(32)]  # whole beats
    fit_bpm, fit_offset = fit_timing_from_notes(times, verbose=False)
    # Mug fit can lock to half/double tempo; accept musically equivalent BPM
    assert min(abs(fit_bpm - bpm), abs(fit_bpm - bpm * 2), abs(fit_bpm - bpm / 2)) < 5.0
    assert abs(fit_offset - offset) < 80.0


def test_snap_to_grid():
    bpm = 180.0
    offset = 0.0
    beat_ms = 60_000.0 / bpm
    t = offset + beat_ms / 4 + 3  # slightly off 1/4
    snapped = snap_ms_to_grid(t, bpm, offset)
    assert abs(snapped - (offset + beat_ms / 4)) < 5


def test_apply_refinement_updates_timing_points():
    bm = TaikoBeatmap()
    bpm = 180.0
    offset = 1000.0
    beat_ms = 60_000.0 / bpm
    for i in range(30):
        bm.notes.append(TaikoNote(time=int(offset + i * beat_ms / 4), note_type="don"))
    bm.timing_points = []
    apply_timing_refinement(
        bm, audio_bpm=bpm, audio_offset_ms=offset, verbose=False
    )
    assert len(bm.timing_points) == 1
    assert bm.timing_points[0].uninherited
    assert bm.note_count == 30
