"""
tests/test_decode.py

Grid-aware decoding, playability repair, and the snap-validity measurement
the step-53k evaluation was reading.

    python -m pytest tests/test_decode.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from taiko.data.decode import decode_on_grid
from taiko.data.grid import Grid
from taiko.data.osu_parser import TaikoBeatmap, TaikoNote, TimingPoint
from taiko.data.repair import repair
from taiko.data.tensor_repr import beatmap_to_tensors, tensor_to_beatmap
from taiko.data.timing_refine import apply_timing_refinement
from taiko.eval.metrics import snap_validity, unplayability

TWO_TEMPOS = [
    TimingPoint(time=1117, beat_length=60_000.0 / 212, meter=4, uninherited=True),
    TimingPoint(time=31_117, beat_length=60_000.0 / 174, meter=4, uninherited=True),
]


def _mixed_chart(seed: int = 0) -> TaikoBeatmap:
    """
    Phrases of 1/4, 1/6 and 1/3 rhythm at both tempos, the way real charts
    switch snap -- by the phrase, each starting on a beat -- never closer
    than 45 ms.
    """
    rng = np.random.default_rng(seed)
    grid = Grid(TWO_TEMPOS)
    times: list[float] = []
    for i, sec in enumerate(grid.sections):
        end = TWO_TEMPOS[1].time if i == 0 else 60_000
        beat = 0
        while sec.offset_ms + (beat + 4) * sec.ms_per_beat < end - 500:
            div = int(rng.choice([4, 4, 6, 3]))
            start = sec.offset_ms + beat * sec.ms_per_beat
            steps = rng.integers(1, 3, size=4 * div)
            pos = np.cumsum(steps) - steps[0]
            for k in pos[pos < 4 * div]:
                t = start + k * sec.ms_per_beat / div
                if not times or t - times[-1] >= 45:
                    times.append(t)
            beat += 4 + int(rng.integers(0, 2))
    bm = TaikoBeatmap()
    bm.timing_points = TWO_TEMPOS
    kinds = ["don", "kat", "big_don", "kat"]
    bm.notes = [TaikoNote(time=int(round(t)), note_type=kinds[i % 4]) for i, t in enumerate(times)]
    return bm


def test_perfect_chart_scores_half_on_the_old_measure():
    """
    The handover's 0.566 snap validity was measured on raw 20 ms frame times
    at a 5 ms tolerance. A chart placed perfectly scores about half on that --
    the measure reads frame quantisation, not the model.
    """
    bm = _mixed_chart()
    chart, _ = beatmap_to_tensors(bm)
    raw = tensor_to_beatmap(chart, bpm=212, offset_ms=1117, timing_points=TWO_TEMPOS)
    old = snap_validity(raw.notes, TWO_TEMPOS).valid_fraction
    fair = snap_validity(raw.notes, TWO_TEMPOS, tolerance_ms=10.0).valid_fraction
    assert old < 0.75, old
    assert fair > 0.97, fair

    apply_timing_refinement(raw, timing_points=TWO_TEMPOS, verbose=False)
    assert snap_validity(raw.notes, TWO_TEMPOS).valid_fraction > 0.9


def test_grid_decoding_recovers_mixed_snaps_across_a_bpm_change():
    bm = _mixed_chart(seed=1)
    chart, _ = beatmap_to_tensors(bm)
    out = decode_on_grid(chart, TWO_TEMPOS, threshold=0.5)

    assert snap_validity(out.notes, TWO_TEMPOS, tolerance_ms=1.0).valid_fraction == 1.0
    truth = {n.time: n.note_type for n in bm.notes}
    got = {n.time: n.note_type for n in out.notes}
    matched = sum(1 for t, k in got.items()
                  if any(abs(t - u) <= 1 and truth[u] == k for u in truth))
    assert len(got) == len(truth), (len(got), len(truth))
    assert matched / len(truth) > 0.99, matched / len(truth)

    # Better than decoding frames and snapping afterwards, on the same input.
    legacy = tensor_to_beatmap(chart, bpm=212, offset_ms=1117, timing_points=TWO_TEMPOS)
    apply_timing_refinement(legacy, timing_points=TWO_TEMPOS, verbose=False)
    legacy_exact = sum(1 for n in legacy.notes if any(abs(n.time - u) <= 1 for u in truth))
    assert matched >= legacy_exact, (matched, legacy_exact)
    assert [tp.time for tp in out.timing_points] == [1117, 31117]


def test_repair_fixes_every_violation_type_and_counts_them():
    grid = Grid([TimingPoint(time=0, beat_length=500.0, meter=4, uninherited=True)])
    bm = TaikoBeatmap()
    bm.timing_points = grid.timing_points()
    bm.notes = [
        TaikoNote(time=1000, note_type="don"),
        TaikoNote(time=1010, note_type="kat"),            # too fast after 1000
        TaikoNote(time=1500, note_type="big_don"),
        TaikoNote(time=1540, note_type="don"),            # makes 1500 a big-note stream
        TaikoNote(time=2000, note_type="roll", end_time=3000),
        TaikoNote(time=2500, note_type="don"),            # inside the roll
        TaikoNote(time=2900, note_type="denden", end_time=3500),   # overlaps the roll
        TaikoNote(time=4000, note_type="roll", end_time=4000),     # zero length
    ]
    before = unplayability(bm.notes)
    assert before.rate > 0
    rep = repair(bm, grid)
    after = unplayability(bm.notes)
    assert after.rate == 0.0, after
    assert rep.dropped_too_fast == 1 and rep.shrunk_big_notes == 1
    assert rep.dropped_in_longs >= 1 and rep.trimmed_longs == 1 and rep.dropped_zero_longs == 1
    assert [n.time for n in bm.notes if n.note_type == "don"][0] == 1000   # the on-beat one kept
