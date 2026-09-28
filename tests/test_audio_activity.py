"""
tests/test_audio_activity.py

Multi-BPM grids, and the audio-agreement metrics and gate.

    python -m pytest tests/test_audio_activity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from taiko.data.audio_activity import activity, activity_score, gate_notes
from taiko.data.frames import FRAME_MS
from taiko.data.grid import Grid
from taiko.data.osu_parser import TaikoBeatmap, TaikoNote, TimingPoint
from taiko.data.tensor_repr import tensor_to_beatmap
from taiko.data.timing_refine import apply_timing_refinement

TWO_TEMPOS = [
    TimingPoint(time=1000, beat_length=500.0, meter=4, uninherited=True),    # 120
    TimingPoint(time=9000, beat_length=60_000 / 200, meter=4, uninherited=True),
]


def test_grid_uses_the_red_line_in_force():
    g = Grid(TWO_TEMPOS)
    assert g.section_at(0).bpm == 120.0            # before the first: extends it
    assert g.section_at(8999).bpm == 120.0
    assert abs(g.section_at(9000).bpm - 200.0) < 1e-9
    # 9000 + 3 beats at 200 BPM = 9900; on the 120 BPM grid it would be off.
    assert g.snap(9903, [1], 10) == 9900
    assert g.distance_ms(9900, [1]) < 1e-9


def test_multi_bpm_snapping_and_decoding():
    bm = TaikoBeatmap()
    bm.notes = [TaikoNote(time=t, note_type="don") for t in (1004, 1498, 9003, 9302, 9598)]
    apply_timing_refinement(bm, timing_points=TWO_TEMPOS, verbose=False)
    assert [n.time for n in bm.notes] == [1000, 1500, 9000, 9300, 9600]
    assert len(bm.timing_points) == 2

    chart = np.zeros((6, 600), dtype=np.float32)
    chart[0, 50] = 1.0
    out = tensor_to_beatmap(chart, bpm=1, offset_ms=0, timing_points=TWO_TEMPOS)
    assert [tp.time for tp in out.timing_points] == [1000, 9000]


def _mel(frames: int, quiet: slice, hits: list[int]) -> np.ndarray:
    mel = np.full((128, frames), 0.6, dtype=np.float32)
    mel[:, quiet] = -0.8
    for f in hits:
        mel[:, f] = 1.0
    return mel


def test_quiet_notes_and_missed_onsets_are_measured():
    frames = 1000
    hits = list(range(100, 500, 25))               # loud, on a 500 ms grid
    mel = _mel(frames, slice(600, 900), hits)
    act = activity(mel)
    grid = Grid([TimingPoint(time=0, beat_length=500.0, meter=4, uninherited=True)])

    in_quiet = [TaikoNote(time=int(f * FRAME_MS), note_type="don") for f in range(650, 850, 25)]
    on_hits = [TaikoNote(time=int(f * FRAME_MS), note_type="don") for f in hits]

    score = activity_score(on_hits + in_quiet, act, grid)
    assert abs(score.quiet_note_rate - len(in_quiet) / (len(in_quiet) + len(on_hits))) < 1e-9
    # The only unmarked attack is the music coming back in after the quiet
    # section, at frame 900 = 18.0 s, which is on the grid.
    assert score.missed_times_ms == [18000], score.missed_times_ms

    sparse = activity_score(on_hits[::2], act, grid)
    assert 0.3 < sparse.strong_onset_miss_rate < 0.7


def test_gate_drops_isolated_quiet_filler_but_keeps_streams():
    frames = 1000
    mel = _mel(frames, slice(600, 900), hits=[])
    act = activity(mel)
    grid = Grid([TimingPoint(time=0, beat_length=500.0, meter=4, uninherited=True)])

    isolated = [TaikoNote(time=t, note_type="don") for t in (12500, 14000, 15500)]
    stream = [TaikoNote(time=16000 + i * 125, note_type="kat") for i in range(6)]
    loud = [TaikoNote(time=4000, note_type="don")]
    kept, dropped = gate_notes(loud + isolated + stream, act, grid)
    assert [n.time for n in dropped] == [12500, 14000, 15500]
    assert len(kept) == 1 + len(stream)
