"""
tests/test_metrics.py

The eval metrics are the only signal that separates "the model learned taiko"
from "the loss went down." A metric that can be gamed -- double-counted
matches, an infinite KL, a two-pointer window that misses the actual burst --
is worse than no metric, because it looks like progress while hiding the
failure it exists to catch.
"""

from __future__ import annotations
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from taiko.data.osu_parser import TaikoBeatmap, TaikoNote, TimingPoint
from taiko.eval.metrics import (
    ChartReport, NoteStats, OnsetScore, SnapScore, UnplayabilityScore,
    evaluate_chart, format_report, note_statistics, onset_f1,
    pattern_divergence, snap_validity, unplayability,
)


def _note(time: int, note_type: str, end_time: int = 0) -> TaikoNote:
    return TaikoNote(time=time, note_type=note_type, end_time=end_time or time)


def _map(notes: list[TaikoNote], timing_points: list[TimingPoint] | None = None) -> TaikoBeatmap:
    bm = TaikoBeatmap()
    bm.notes = notes
    bm.timing_points = timing_points or [TimingPoint(time=0, beat_length=400.0, meter=4, uninherited=True)]
    bm.compute_stats()
    return bm


# --------------------------------------------------------------------------- #
# onset_f1
# --------------------------------------------------------------------------- #

def test_onset_f1_greedy_matching_avoids_double_count():
    """Three generated notes clustered near one reference note must not each
    count as a match -- recall is bounded by the single reference note, and
    only one of the three can win it."""
    reference = [_note(1000, "don")]
    generated = [_note(995, "don"), _note(1005, "don"), _note(1010, "don")]
    score = onset_f1(generated, reference, tolerance_ms=25.0)

    assert score.n_matched == 1, score.n_matched
    assert abs(score.recall - 1.0) < 1e-9, score.recall
    assert abs(score.precision - 1 / 3) < 1e-9, score.precision
    print(f"  greedy match no double-count ok  (P={score.precision:.2f} R={score.recall:.2f})")


def test_onset_f1_perfect_copy_gives_f1_one():
    times = [0, 200, 400, 600, 900, 1300]
    reference = [_note(t, "don") for t in times]
    generated = [_note(t, "don") for t in times]
    score = onset_f1(generated, reference)

    assert score.f1 == 1.0, score.f1
    assert score.n_matched == len(times)
    assert score.mean_abs_error_ms == 0.0
    print("  perfect copy -> f1 == 1.0    ok")


def test_onset_f1_disjoint_gives_zero():
    reference = [_note(0, "don"), _note(1000, "don")]
    generated = [_note(5000, "don"), _note(6000, "don")]
    score = onset_f1(generated, reference, tolerance_ms=25.0)

    assert score.n_matched == 0
    assert score.precision == 0.0 and score.recall == 0.0 and score.f1 == 0.0
    print("  disjoint sets -> zero        ok")


def test_onset_f1_mean_error_over_matched_only():
    reference = [_note(0, "don"), _note(1000, "don")]
    # first note matches with a 10 ms error, second is unmatched entirely
    generated = [_note(10, "don"), _note(5000, "don")]
    score = onset_f1(generated, reference, tolerance_ms=25.0)

    assert score.n_matched == 1
    assert abs(score.mean_abs_error_ms - 10.0) < 1e-9, score.mean_abs_error_ms
    print("  mean error over matched only ok")


# --------------------------------------------------------------------------- #
# snap_validity
# --------------------------------------------------------------------------- #

def test_snap_validity_on_1_4_grid_is_fully_valid():
    """150 BPM -> beat_length 400 ms; offsets 100 and 300 ms sit on the 1/4
    grid but on no coarser grid (1/1, 1/2), so they must classify as 1/4."""
    tp = [TimingPoint(time=0, beat_length=400.0, meter=4, uninherited=True)]
    notes = [_note(t, "don") for t in range(100, 2000, 200)]
    score = snap_validity(notes, tp, tolerance_ms=5.0)

    assert score.valid_fraction == 1.0, score.valid_fraction
    assert score.unsnapped == 0
    assert score.divisor_counts[4] == len(notes), score.divisor_counts
    print(f"  on-grid notes -> valid 1.0   ok  ({score.divisor_counts[4]} at 1/4)")


def test_snap_validity_off_grid_is_mostly_invalid():
    tp = [TimingPoint(time=0, beat_length=400.0, meter=4, uninherited=True)]
    # +12 ms lands at least 12 ms from every standard divisor's grid point
    # (checked against the finest, 1/16 grid, which subsumes 1/2, 1/4 and
    # 1/8) -- well outside the 5 ms tolerance.
    notes = [_note(t + 12, "don") for t in range(0, 2000, 100)]
    score = snap_validity(notes, tp, tolerance_ms=5.0)

    assert score.valid_fraction < 0.2, score.valid_fraction
    print(f"  off-grid notes -> low valid  ok  ({score.valid_fraction:.2f})")


def test_snap_validity_coarsest_divisor_wins():
    """A note exactly on the downbeat matches every divisor; it must be
    classified as 1/1, the simplest description, not the finest grid."""
    tp = [TimingPoint(time=0, beat_length=400.0, meter=4, uninherited=True)]
    notes = [_note(t, "don") for t in range(0, 2000, 400)]
    score = snap_validity(notes, tp, tolerance_ms=5.0)

    assert score.divisor_counts[1] == len(notes), score.divisor_counts
    assert score.divisor_counts[16] == 0, score.divisor_counts
    print("  downbeat -> classed as 1/1   ok")


def test_snap_validity_handles_bpm_change():
    """A note that is on-grid for its own segment's tempo must score valid
    even though it would be off-grid under the map's other tempo."""
    tp = [
        TimingPoint(time=0,     beat_length=400.0, meter=4, uninherited=True),   # 150 BPM
        TimingPoint(time=2000,  beat_length=600.0, meter=4, uninherited=True),   # 100 BPM
    ]
    notes = [
        _note(100, "don"),    # 1/4 of the 400 ms beat -> valid under segment 1
        _note(2150, "don"),   # 1/4 of the 600 ms beat (150 ms) -> valid under segment 2
    ]
    score = snap_validity(notes, tp, tolerance_ms=5.0)

    assert score.valid_fraction == 1.0, score.valid_fraction
    assert score.unsnapped == 0
    print("  mid-map BPM change handled   ok")


def test_snap_validity_before_first_red_line_is_unsnapped():
    tp = [TimingPoint(time=1000, beat_length=400.0, meter=4, uninherited=True)]
    notes = [_note(500, "don")]
    score = snap_validity(notes, tp, tolerance_ms=5.0)
    assert score.unsnapped == 1
    assert score.valid_fraction == 0.0
    print("  before first red line unsnap ok")


# --------------------------------------------------------------------------- #
# unplayability
# --------------------------------------------------------------------------- #

def test_unplayability_flags_too_fast():
    notes = [_note(0, "don"), _note(15, "kat")]   # 15 ms apart, well under 30 ms
    score = unplayability(notes)
    assert score.too_fast == 1
    assert score.big_note_streams == 0
    assert score.overlapping_longs == 0
    assert score.zero_length_longs == 0
    print("  flags too_fast in isolation  ok")


def test_unplayability_flags_big_note_streams():
    notes = [_note(0, "don"), _note(40, "big_don"), _note(80, "don")]   # 40 ms gaps
    score = unplayability(notes)
    assert score.big_note_streams == 1
    assert score.too_fast == 0        # gaps are 40 ms, above the 30 ms hit-speed limit
    assert score.overlapping_longs == 0
    assert score.zero_length_longs == 0
    print("  flags big_note_streams alone ok")


def test_unplayability_flags_overlapping_longs():
    notes = [_note(0, "roll", end_time=500), _note(200, "denden", end_time=700)]
    score = unplayability(notes)
    assert score.overlapping_longs == 2   # both members of the overlapping pair
    assert score.too_fast == 0
    assert score.big_note_streams == 0
    assert score.zero_length_longs == 0
    print("  flags overlapping_longs alone ok")


def test_unplayability_flags_hit_swallowed_by_long():
    notes = [_note(0, "roll", end_time=500), _note(250, "don")]
    score = unplayability(notes)
    assert score.overlapping_longs == 1
    print("  flags hit inside a long      ok")


def test_unplayability_flags_zero_length_longs():
    notes = [_note(0, "roll", end_time=0)]
    score = unplayability(notes)
    assert score.zero_length_longs == 1
    assert score.overlapping_longs == 0   # a degenerate span is excluded from overlap checks
    print("  flags zero_length_longs      ok")


def test_unplayability_clean_chart_is_zero():
    notes = [_note(t, "don" if i % 2 == 0 else "kat") for i, t in enumerate(range(0, 2000, 200))]
    score = unplayability(notes)
    assert score.too_fast == 0 and score.big_note_streams == 0
    assert score.overlapping_longs == 0 and score.zero_length_longs == 0
    assert score.rate == 0.0
    print("  clean chart -> zero rate     ok")


# --------------------------------------------------------------------------- #
# pattern_divergence
# --------------------------------------------------------------------------- #

def _alternating(n: int) -> list[TaikoNote]:
    return [_note(i * 100, "don" if i % 2 == 0 else "kat") for i in range(n)]


def _monotone(n: int, note_type: str = "don") -> list[TaikoNote]:
    return [_note(i * 100, note_type) for i in range(n)]


def test_pattern_divergence_identical_sequences_is_near_zero():
    a = _alternating(40)
    b = _alternating(40)
    kl = pattern_divergence(a, b, n=4)
    assert kl < 0.05, kl
    print(f"  identical sequences -> ~0    ok  (kl={kl:.4f})")


def test_pattern_divergence_opposite_sequences_is_clearly_positive_and_finite():
    alternating = _alternating(60)
    monotone = _monotone(60)
    kl = pattern_divergence(monotone, alternating, n=4)
    assert kl > 1.0, kl
    assert kl < float("inf")
    print(f"  opposite sequences -> >0     ok  (kl={kl:.3f})")


def test_pattern_divergence_ignores_rolls_and_dendens():
    a = _alternating(40) + [_note(50000, "roll", end_time=50500)]
    b = _alternating(40)
    kl = pattern_divergence(a, b, n=4)
    assert kl < 0.05, kl
    print("  rolls/dendens ignored        ok")


def test_pattern_divergence_short_sequences_return_zero():
    assert pattern_divergence(_monotone(2), _alternating(2), n=4) == 0.0
    print("  too-short sequences -> 0.0   ok")


# --------------------------------------------------------------------------- #
# note_statistics / peak_nps
# --------------------------------------------------------------------------- #

def test_note_statistics_basic_counts():
    notes = [
        _note(0, "don"), _note(200, "kat"), _note(400, "big_don"),
        _note(600, "roll", end_time=900), _note(1200, "denden", end_time=1400),
    ]
    stats = note_statistics(notes)
    assert stats.n_notes == 5
    assert stats.roll_count == 1 and stats.denden_count == 1
    assert abs(stats.big_ratio - 1 / 5) < 1e-9
    # 3 hits (don, kat, big_don) over 400 ms span
    assert abs(stats.avg_nps - 3 / 0.4) < 1e-6, stats.avg_nps
    print("  note_statistics counts       ok")


def test_note_statistics_accepts_beatmap():
    bm = _map([_note(0, "don"), _note(100, "kat")])
    stats = note_statistics(bm)
    assert stats.n_notes == 2
    print("  note_statistics(beatmap)     ok")


def test_peak_nps_two_pointer_known_burst():
    """10 notes packed into 1 s, then silence, in a much longer chart. The
    5 s window containing the burst must report exactly 10/5 notes/sec --
    not diluted by scanning the whole chart, and not missed either."""
    burst = [_note(i * 100, "don") for i in range(10)]          # 0..900 ms, 10 notes
    tail = [_note(60_000, "don"), _note(90_000, "don")]         # isolated, far away
    stats = note_statistics(burst + tail)

    assert stats.peak_nps == 10 / 5.0, stats.peak_nps
    print(f"  peak_nps finds the burst     ok  ({stats.peak_nps:.2f} nps)")


def _naive_peak_nps(times: list[int], window_sec: float = 5.0) -> float:
    """O(n^2) reference: the busiest window always starts at some note time,
    so trying each note as a window start is exhaustive."""
    window_ms = window_sec * 1000.0
    best = 0
    for t in times:
        count = sum(1 for u in times if t <= u <= t + window_ms)
        best = max(best, count)
    return best / window_sec


def test_peak_nps_two_pointer_matches_brute_force():
    """Cross-check the two-pointer scan against an O(n^2) reference over many
    random note sets -- catches an off-by-one the hand-picked burst case
    could miss."""
    random.seed(0)
    for _ in range(25):
        n = random.randint(2, 60)
        times = sorted(random.sample(range(0, 20_000), n))
        notes = [_note(t, "don") for t in times]
        stats = note_statistics(notes)
        want = _naive_peak_nps(times)
        assert abs(stats.peak_nps - want) < 1e-9, (stats.peak_nps, want, times)
    print("  two-pointer matches brute force ok")


# --------------------------------------------------------------------------- #
# evaluate_chart / format_report
# --------------------------------------------------------------------------- #

def test_evaluate_chart_without_reference_leaves_reference_metrics_none():
    bm = _map([_note(0, "don"), _note(100, "kat"), _note(200, "don")])
    report = evaluate_chart(bm)
    assert isinstance(report, ChartReport)
    assert report.onset is None and report.pattern_kl is None
    assert isinstance(report.stats, NoteStats)
    assert isinstance(report.unplayability, UnplayabilityScore)
    assert isinstance(report.snap, SnapScore)
    print("  no reference -> ref fields None ok")


def test_evaluate_chart_with_reference_fills_everything():
    generated = _map([_note(0, "don"), _note(100, "kat"), _note(200, "don")])
    reference = _map([_note(0, "don"), _note(100, "kat"), _note(200, "don")])
    report = evaluate_chart(generated, reference)
    assert isinstance(report.onset, OnsetScore)
    assert report.onset.f1 == 1.0
    assert isinstance(report.pattern_kl, float)
    print("  reference supplied -> filled ok")


def test_format_report_is_a_readable_string():
    bm = _map([_note(0, "don"), _note(100, "kat")])
    report = evaluate_chart(bm, bm)
    text = format_report(report)
    assert isinstance(text, str)
    for label in ("notes", "snap valid", "unplayable rate", "onset f1", "pattern KL"):
        assert label in text, label
    print("  format_report readable       ok")


if __name__ == "__main__":
    print("metrics")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all metrics tests passed")
