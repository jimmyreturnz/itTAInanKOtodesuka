"""
taiko/eval/metrics.py

Quantitative scoring for a chart, generated or human-made. Used two places:
as a training-log sanity check after each sampled chart (so a collapsing
model is visible in the log before someone wastes ten minutes listening to
it), and as the acceptance gate for automated evaluation once a model exists
worth gating.

Every metric here answers one question a person would otherwise have to
answer by ear, and each is deliberately blind to the things the others cover:

    onset_f1
        Are the notes in roughly the right places? A raw "count how many
        generated notes are within tolerance of a reference note" inflates
        arbitrarily -- a model that spams three notes around every real
        onset scores perfect apparent recall on that measure while being
        unplayable. Greedy one-to-one matching is what forces that failure
        mode to show up as low precision instead of being hidden.

    snap_validity
        Would a human osu! mapper accept these note times, or do they look
        like raw model output that was never quantised to the beat grid?
        This is orthogonal to onset_f1: a chart can match a reference's
        onsets closely in a coarse sense while still landing at essentially
        random sub-beat offsets, and a chart with no reference at all can
        still be checked against its own timing points.

    unplayability
        Is the chart something a human hand can execute? Density and F1
        both stay silent about a 27 ms double-hit or a big note chained
        into a stream -- these make a chart broken rather than merely
        "different from the reference," and matter even for a chart with
        no reference to compare against.

    pattern_divergence
        Does the don/kat colour sequence read like the reference's
        rhythmic vocabulary, or has generation collapsed onto a narrow,
        repetitive n-gram? This is the mode-collapse detector: two charts
        can agree on density, snap validity and even onset F1 while one
        alternates colour unpredictably and the other just alternates
        don-kat-don-kat forever.

    note_statistics
        Coarse shape of a chart (density, note-type mix, burst intensity)
        for tracking trends across an epoch without needing a reference at
        all.

evaluate_chart / format_report tie these into one call for the training
loop: compute everything that needs no reference, add the metrics that do
when one is supplied, and print a fixed-width summary that reads cleanly
across many lines of a log.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np

from taiko.data.osu_parser import TaikoBeatmap, TaikoNote, TimingPoint

# --------------------------------------------------------------------------- #
# Note-type groupings shared across metrics
# --------------------------------------------------------------------------- #

DON_TYPES  = ("don", "big_don")
KAT_TYPES  = ("kat", "big_kat")
HIT_TYPES  = DON_TYPES + KAT_TYPES     # anything that is a single hit, not a hold
BIG_TYPES  = ("big_don", "big_kat")
LONG_TYPES = ("roll", "denden")

# Below this gap, two hits cannot both be struck by a human -- the drum
# has to be re-struck and no one can do that in under ~30 ms.
MIN_HIT_GAP_MS = 30.0

# A big note needs both drum surfaces at once, so it cannot be chained at
# stream speed even though a same-colour hit could be.
MIN_BIG_GAP_MS = 60.0

DEFAULT_SNAP_DIVISORS = (1, 2, 3, 4, 6, 8, 12, 16)


# --------------------------------------------------------------------------- #
# Onset accuracy
# --------------------------------------------------------------------------- #

@dataclass
class OnsetScore:
    precision: float
    recall: float
    f1: float
    n_generated: int
    n_reference: int
    n_matched: int
    mean_abs_error_ms: float


def onset_f1(
    generated: list[TaikoNote],
    reference: list[TaikoNote],
    tolerance_ms: float = 25.0,
) -> OnsetScore:
    """
    Precision/recall/F1 of generated note *times* against reference times.

    Matching is greedy one-to-one: every candidate (generated, reference)
    pair within tolerance is considered, closest pairs are consumed first,
    and each note on either side can be used at most once. Without the
    one-to-one constraint, several generated notes clustered near a single
    reference note would each count as a hit and double (or triple, or
    more) the apparent recall -- exactly the failure mode this metric
    exists to catch, so it cannot be allowed to hide it.
    """
    gen_times = np.asarray(sorted(n.time for n in generated), dtype=np.float64)
    ref_times = np.asarray(sorted(n.time for n in reference), dtype=np.float64)
    n_gen, n_ref = gen_times.size, ref_times.size

    candidates: list[tuple[float, int, int]] = []
    if n_gen and n_ref:
        # A reference note can only ever match a generated note within
        # `tolerance_ms`, so a searchsorted window around each generated
        # time keeps candidate generation near-linear instead of the full
        # O(n_gen * n_ref) pairwise scan a naive version would need.
        lo = np.searchsorted(ref_times, gen_times - tolerance_ms, side="left")
        hi = np.searchsorted(ref_times, gen_times + tolerance_ms, side="right")
        for gi in range(n_gen):
            for ri in range(int(lo[gi]), int(hi[gi])):
                candidates.append((abs(float(gen_times[gi] - ref_times[ri])), gi, ri))

    candidates.sort(key=lambda c: c[0])

    used_gen = [False] * n_gen
    used_ref = [False] * n_ref
    matched_diffs: list[float] = []
    for diff, gi, ri in candidates:
        if used_gen[gi] or used_ref[ri]:
            continue
        used_gen[gi] = True
        used_ref[ri] = True
        matched_diffs.append(diff)

    n_matched = len(matched_diffs)
    precision = n_matched / n_gen if n_gen else 0.0
    recall    = n_matched / n_ref if n_ref else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_abs_error_ms = float(sum(matched_diffs) / n_matched) if n_matched else 0.0

    return OnsetScore(
        precision=precision, recall=recall, f1=f1,
        n_generated=n_gen, n_reference=n_ref, n_matched=n_matched,
        mean_abs_error_ms=mean_abs_error_ms,
    )


# --------------------------------------------------------------------------- #
# Snap (beat-grid) validity
# --------------------------------------------------------------------------- #

@dataclass
class SnapScore:
    valid_fraction: float
    divisor_counts: dict[int, int]
    unsnapped: int


def snap_validity(
    notes: list[TaikoNote],
    timing_points: list[TimingPoint],
    tolerance_ms: float = 5.0,
    divisors: tuple[int, ...] = DEFAULT_SNAP_DIVISORS,
) -> SnapScore:
    """
    Fraction of notes that land on a legal beat subdivision.

    Each note is judged against the uninherited (red) timing point active
    at its time, exactly as osu!'s own grid works -- a green (inherited)
    line only changes slider velocity, never the beat grid, so it plays no
    part here. A note before the first red line, or with no red line at
    all, has no grid to judge it against and counts as unsnapped rather
    than being scored against a fabricated default tempo.

    Divisors are tested from coarsest to finest (1, 2, 3, ... 16) and the
    first one whose grid a note's offset falls within tolerance of wins
    the classification. This has to run coarse-first: every point on the
    1/4 grid also sits exactly on the 1/8 and 1/16 grids (4 and 16 are
    both multiples of 4), so scanning finest-first would tag nearly every
    ordinarily-snapped note as "1/16" purely because the fine grid
    contains the coarse one as a subset. Coarse-first reports the
    simplest grid that actually explains the note, which is the
    classification a human reading the distribution would expect.
    """
    reds = sorted(
        (tp for tp in timing_points if tp.uninherited and tp.beat_length > 0),
        key=lambda tp: tp.time,
    )
    red_times = [tp.time for tp in reds]
    sorted_divisors = sorted(set(divisors))

    divisor_counts: dict[int, int] = {d: 0 for d in sorted_divisors}
    unsnapped = 0
    valid = 0

    for note in notes:
        idx = bisect.bisect_right(red_times, note.time) - 1
        if idx < 0:
            unsnapped += 1
            continue

        red = reds[idx]
        beat_length = red.beat_length
        offset = (note.time - red.time) % beat_length

        matched: int | None = None
        for d in sorted_divisors:
            step = beat_length / d
            r = offset % step
            dist = min(r, step - r)
            if dist <= tolerance_ms:
                matched = d
                break

        if matched is None:
            unsnapped += 1
        else:
            divisor_counts[matched] += 1
            valid += 1

    total = len(notes)
    # No notes to judge is vacuously fully valid, not zero -- there is no
    # violation to report.
    valid_fraction = valid / total if total else 1.0

    return SnapScore(valid_fraction=valid_fraction, divisor_counts=divisor_counts, unsnapped=unsnapped)


# --------------------------------------------------------------------------- #
# Playability
# --------------------------------------------------------------------------- #

@dataclass
class UnplayabilityScore:
    too_fast: int
    big_note_streams: int
    overlapping_longs: int
    zero_length_longs: int
    rate: float


def _spans_intersect(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    """Strict interior overlap -- one span ending exactly as another begins is legal."""
    return a_start < b_end and b_start < a_end


def unplayability(notes: list[TaikoNote]) -> UnplayabilityScore:
    """
    Count taiko-specific violations a density or onset metric cannot see.

    Each count is a distinct failure mode, checked independently so a
    generator that only breaks one rule shows up as only one nonzero
    field:

      too_fast           two hits (not holds) closer together than a hand
                          can physically re-strike the drum.
      big_note_streams    a big note chained into stream-speed hits either
                          side of it, which needs both drum surfaces at
                          once and so cannot be hit that fast.
      overlapping_longs   a roll/denden whose span overlaps another long's
                          span, or swallows a hit note -- none of which
                          osu! or a human player can resolve.
      zero_length_longs   a hold with no duration at all, which is not a
                          hold.
    """
    ordered = sorted(notes, key=lambda n: n.time)
    hits  = [n for n in ordered if n.note_type in HIT_TYPES]
    longs = [n for n in ordered if n.note_type in LONG_TYPES]

    too_fast = 0
    for a, b in zip(hits, hits[1:]):
        if (b.time - a.time) < MIN_HIT_GAP_MS:
            too_fast += 1

    big_note_streams = 0
    for i, n in enumerate(hits):
        if n.note_type not in BIG_TYPES:
            continue
        too_close = (
            (i > 0 and (n.time - hits[i - 1].time) < MIN_BIG_GAP_MS)
            or (i < len(hits) - 1 and (hits[i + 1].time - n.time) < MIN_BIG_GAP_MS)
        )
        if too_close:
            big_note_streams += 1

    zero_length_longs = sum(1 for n in longs if n.duration <= 0)

    # Degenerate (zero-length) longs have no meaningful span, so they are
    # left out of the overlap check and counted only above.
    live_longs = [n for n in longs if n.duration > 0]
    overlapping_longs = 0
    for i, a in enumerate(live_longs):
        overlap = any(
            _spans_intersect(a.time, a.end_time, b.time, b.end_time)
            for j, b in enumerate(live_longs) if j != i
        )
        if not overlap:
            overlap = any(a.time < h.time < a.end_time for h in hits)
        if overlap:
            overlapping_longs += 1

    total_violations = too_fast + big_note_streams + overlapping_longs + zero_length_longs
    rate = total_violations / max(len(notes), 1)

    return UnplayabilityScore(
        too_fast=too_fast, big_note_streams=big_note_streams,
        overlapping_longs=overlapping_longs, zero_length_longs=zero_length_longs,
        rate=rate,
    )


# --------------------------------------------------------------------------- #
# Pattern (don/kat n-gram) divergence
# --------------------------------------------------------------------------- #

def _colour_sequence(notes: list[TaikoNote]) -> list[int]:
    """Hit notes only, time-ordered, as 0 (don-like) / 1 (kat-like)."""
    ordered = sorted((n for n in notes if n.note_type in HIT_TYPES), key=lambda n: n.time)
    return [1 if n.note_type in KAT_TYPES else 0 for n in ordered]


def _ngram_counts(colours: list[int], n: int, n_grams_total: int) -> np.ndarray:
    counts = np.zeros(n_grams_total, dtype=np.float64)
    for i in range(len(colours) - n + 1):
        idx = 0
        for c in colours[i:i + n]:
            idx = (idx << 1) | c
        counts[idx] += 1
    return counts


def pattern_divergence(
    generated: list[TaikoNote],
    reference: list[TaikoNote],
    n: int = 4,
) -> float:
    """
    KL(generated || reference) in bits, over don/kat n-gram distributions.

    Add-one (Laplace) smoothing is applied over *all* 2**n possible
    n-grams, not just the ones actually observed on either side. Without
    it, any n-gram present in the generated chart but absent from the
    reference sends the KL term to infinity, which makes the metric
    useless for exactly the case it exists to catch (a generator that
    plays a pattern the reference never uses).
    """
    gen_colours = _colour_sequence(generated)
    ref_colours = _colour_sequence(reference)
    if len(gen_colours) < n or len(ref_colours) < n:
        return 0.0

    n_grams_total = 2 ** n
    gen_counts = _ngram_counts(gen_colours, n, n_grams_total)
    ref_counts = _ngram_counts(ref_colours, n, n_grams_total)

    gen_probs = (gen_counts + 1.0) / (gen_counts.sum() + n_grams_total)
    ref_probs = (ref_counts + 1.0) / (ref_counts.sum() + n_grams_total)

    return float(np.sum(gen_probs * np.log2(gen_probs / ref_probs)))


# --------------------------------------------------------------------------- #
# Descriptive statistics
# --------------------------------------------------------------------------- #

@dataclass
class NoteStats:
    n_notes: int
    don_ratio: float
    kat_ratio: float
    big_ratio: float
    roll_count: int
    denden_count: int
    avg_nps: float
    peak_nps: float


def _peak_nps(hit_times: list[int], window_sec: float = 5.0) -> float:
    """
    Maximum notes-per-second in any `window_sec`-wide window, via a
    two-pointer scan over sorted onset times.

    A per-second-bucket histogram would hide a burst that straddles a
    bucket boundary (10 notes split 5-and-5 across a 1 s boundary reads as
    two ordinary seconds, not the burst it is); sliding a continuous
    window catches it regardless of where it falls. Peak NPS is reported
    as (largest window occupancy) / window_sec, i.e. the average rate
    inside the busiest `window_sec` stretch of the chart.
    """
    if not hit_times:
        return 0.0
    window_ms = window_sec * 1000.0
    left = 0
    peak = 0
    for right in range(len(hit_times)):
        while hit_times[right] - hit_times[left] > window_ms:
            left += 1
        peak = max(peak, right - left + 1)
    return peak / window_sec


def note_statistics(
    bm_or_notes: TaikoBeatmap | list[TaikoNote],
    duration_ms: float | None = None,
) -> NoteStats:
    """
    Coarse shape of a chart: density, note-type mix, burst intensity.

    Args:
        bm_or_notes: a parsed beatmap, or a bare note list for callers
            (e.g. a sliding training window) that never built one.
        duration_ms: only used as an NPS fallback when there are fewer
            than two hit notes to measure a span between -- otherwise NPS
            is always the span between the first and last hit note, since
            that is what a player actually experiences as the chart's
            pace, not the nominal length of the audio around it.
    """
    notes = bm_or_notes.notes if isinstance(bm_or_notes, TaikoBeatmap) else bm_or_notes

    n_notes  = len(notes)
    don_c    = sum(1 for n in notes if n.note_type in DON_TYPES)
    kat_c    = sum(1 for n in notes if n.note_type in KAT_TYPES)
    big_c    = sum(1 for n in notes if n.note_type in BIG_TYPES)
    roll_c   = sum(1 for n in notes if n.note_type == "roll")
    denden_c = sum(1 for n in notes if n.note_type == "denden")

    hits_total = don_c + kat_c
    don_ratio  = don_c / hits_total if hits_total else 0.5
    kat_ratio  = kat_c / hits_total if hits_total else 0.5
    big_ratio  = big_c / n_notes if n_notes else 0.0

    hit_times = sorted(n.time for n in notes if n.note_type in HIT_TYPES)
    if len(hit_times) >= 2:
        span_sec = (hit_times[-1] - hit_times[0]) / 1000.0
        avg_nps  = len(hit_times) / span_sec if span_sec > 0 else 0.0
    elif duration_ms:
        avg_nps = len(hit_times) / (duration_ms / 1000.0)
    else:
        avg_nps = 0.0

    peak_nps = _peak_nps(hit_times)

    return NoteStats(
        n_notes=n_notes, don_ratio=don_ratio, kat_ratio=kat_ratio, big_ratio=big_ratio,
        roll_count=roll_c, denden_count=denden_c, avg_nps=avg_nps, peak_nps=peak_nps,
    )


# --------------------------------------------------------------------------- #
# Aggregate report
# --------------------------------------------------------------------------- #

@dataclass
class ChartReport:
    stats: NoteStats
    unplayability: UnplayabilityScore
    snap: SnapScore
    onset: OnsetScore | None
    pattern_kl: float | None


def evaluate_chart(generated: TaikoBeatmap, reference: TaikoBeatmap | None = None) -> ChartReport:
    """
    Run every metric that applies. `stats`, `unplayability` and `snap` need
    only the generated chart; `onset` and `pattern_kl` need a reference and
    stay None without one, so a caller can tell "no reference supplied"
    apart from "scored zero".
    """
    stats     = note_statistics(generated)
    unplay    = unplayability(generated.notes)
    snap      = snap_validity(generated.notes, generated.timing_points)

    onset: OnsetScore | None = None
    pattern_kl: float | None = None
    if reference is not None:
        onset      = onset_f1(generated.notes, reference.notes)
        pattern_kl = pattern_divergence(generated.notes, reference.notes)

    return ChartReport(stats=stats, unplayability=unplay, snap=snap, onset=onset, pattern_kl=pattern_kl)


def format_report(report: ChartReport) -> str:
    """Fixed-width plain-text summary, one field per line, for a training log."""
    s, u, p = report.stats, report.unplayability, report.snap
    lines = [
        f"  {'notes':<17s} {s.n_notes}",
        f"  {'don / kat':<17s} {s.don_ratio:.2f} / {s.kat_ratio:.2f}",
        f"  {'big fraction':<17s} {s.big_ratio:.2f}",
        f"  {'roll / denden':<17s} {s.roll_count} / {s.denden_count}",
        f"  {'avg / peak nps':<17s} {s.avg_nps:.2f} / {s.peak_nps:.2f}",
        f"  {'snap valid':<17s} {p.valid_fraction:.2f}  (unsnapped {p.unsnapped})",
        f"  {'unplayable rate':<17s} {u.rate:.3f}  "
        f"(fast {u.too_fast}, big-stream {u.big_note_streams}, "
        f"overlap {u.overlapping_longs}, zero-len {u.zero_length_longs})",
    ]
    if report.onset is not None:
        o = report.onset
        lines.append(
            f"  {'onset f1':<17s} {o.f1:.2f}  "
            f"(P {o.precision:.2f} R {o.recall:.2f}, err {o.mean_abs_error_ms:.1f} ms)"
        )
    if report.pattern_kl is not None:
        lines.append(f"  {'pattern KL':<17s} {report.pattern_kl:.3f} bits")
    return "\n".join(lines)
