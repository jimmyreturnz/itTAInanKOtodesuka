"""
taiko/data/decode.py

Chart probabilities -> notes that sit on the beat grid by construction.

Why not decode frames and snap afterwards
-----------------------------------------
The chart lives on a 20 ms frame grid, so a frame-decoded note is up to 10 ms
from where it belongs. Snapping afterwards moves each note to the first
subdivision within tolerance, one note at a time. Where two subdivisions'
lines are close together it can pick the wrong one: at 212 BPM a 1/4 line and
a 1/3 line can be 24 ms apart, so one frame lies within 10 ms of both. A note
that is more than 10 ms from every line is kept exactly where it fell, off
the grid.

Here the question is turned around. Every legal position is enumerated first:
each subdivision of each red line's own tempo. Each onset peak in the
probabilities then chooses among the lines near it, so every note is on a line
by construction. A mild prior prefers common snaps (1/1, 1/2, 1/4) over rare
ones (1/8, 1/12). When a frame could belong to either of two nearby lines, the
probability and the prior decide together, not whichever divisor happens to be
tried first.

Peaks first, not lines first
----------------------------
This decoder used to ask every line for the probability at its own frame. On
30 held-out 5.5*+ maps that split one onset into two notes 22-27 ms apart 147
times a map (one onset spread over three frames lit a 1/4 line and a 1/3 or
1/6 line beside it), and lost onsets whose peak sat a frame off their line --
12.2% of the model's peaks are more than 10 ms off, against 2.0% for ranked
charts. Deciding peaks first, on the same samples: onset F1 0.670 -> 0.728,
pattern KL 0.484 -> 0.183, note count 1.37x -> 1.09x the reference, missed
strong onsets 0.211 -> 0.163 (ranked 0.152), and repair touches 0.2% of notes
instead of 5.0%.

On synthetic charts that change snap by the phrase (1/4, 1/6, 1/3 across a
212 -> 174 BPM change), this put 100% of notes on the exact line they were
written on; decode-then-snap put 97.5-99%.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from taiko.data.frames import FRAME_MS
from taiko.data.grid import Grid
from taiko.data.osu_parser import TaikoBeatmap, TaikoNote, TimingPoint
from taiko.eval.metrics import MIN_HIT_GAP_MS
from taiko.data.tensor_repr import (
    CH_BIG_DON, CH_BIG_KAT, CH_DON, CH_KAT, tensor_to_beatmap,
)

# Allowed subdivisions and how much each is preferred when two compete for
# the same onset. Values are multipliers on the onset probability.
SNAP_PRIOR = {1: 1.0, 2: 1.0, 4: 1.0, 3: 0.9, 6: 0.85, 8: 0.8, 12: 0.7}

# A hit peak this close to a long note belongs to the long note.
NMS_MS = 25.0

HIT_CHANNELS = ((CH_DON, "don"), (CH_KAT, "kat"), (CH_BIG_DON, "big_don"), (CH_BIG_KAT, "big_kat"))


@dataclass
class Candidate:
    time_ms: float
    divisor: int


def grid_candidates(grid: Grid, end_ms: float, divisors=tuple(SNAP_PRIOR)) -> list[Candidate]:
    """Every legal position up to `end_ms`, tagged with its coarsest divisor."""
    divisors = sorted(set(divisors))
    L = int(np.lcm.reduce(divisors))
    starts = [s.offset_ms for s in grid.sections] + [end_ms]
    out: list[Candidate] = []
    for i, sec in enumerate(grid.sections):
        lo = 0.0 if i == 0 else starts[i]
        hi = starts[i + 1]
        if hi <= lo:
            continue
        step = sec.ms_per_beat / L
        k0 = int(np.ceil((lo - sec.offset_ms) / step))
        k1 = int(np.floor((hi - 1e-6 - sec.offset_ms) / step))
        for k in range(k0, k1 + 1):
            level = next((d for d in divisors if k % (L // d) == 0), None)
            if level is not None:
                out.append(Candidate(sec.offset_ms + k * step, level))
    return out


def decode_on_grid(
    chart: np.ndarray,
    timing_points: list[TimingPoint],
    threshold: float = 0.5,
    divisors=tuple(SNAP_PRIOR),
    family_switch: float | None = None,
    hit_threshold: float | None = None,
    **meta,
) -> TaikoBeatmap:
    """
    [6, T] chart probabilities -> a beatmap whose hits all sit on the grid.

    Long notes come from the same region decoding as tensor_to_beatmap, with
    their ends moved to the nearest grid line. `meta` is passed through to
    tensor_to_beatmap for title, version and so on.

    `hit_threshold` gates hits only (default: `threshold`, which also finds
    the long notes), so density can be tuned without growing or shrinking
    rolls.
    """
    grid = Grid(timing_points)
    first = grid.sections[0]
    bm = tensor_to_beatmap(chart, bpm=first.bpm, offset_ms=first.offset_ms,
                           threshold=threshold, timing_points=timing_points, **meta)
    longs = [n for n in bm.notes if n.is_long]
    for n in longs:
        n.time = int(round(_nearest(grid, n.time, divisors)))
        n.end_time = max(n.time + int(FRAME_MS), int(round(_nearest(grid, n.end_time, divisors))))

    end_ms = chart.shape[1] * FRAME_MS
    cands = grid_candidates(grid, end_ms, divisors)
    if not cands:
        bm.notes = longs
        bm.compute_stats()
        return bm
    t = np.array([c.time_ms for c in cands])
    div = np.array([c.divisor for c in cands])
    thr = threshold if hit_threshold is None else hit_threshold
    hits = _decode_peaks(chart, t, div, grid, thr, longs, family_switch)
    bm.notes = sorted(longs + hits, key=lambda n: (n.time, 0 if n.is_long else 1))
    bm.compute_stats()
    return bm



# How far a peak may sit from the line it is assigned to. The model's peaks
# land more than 10 ms off their line 12.2% of the time (ranked charts: 2.0%);
# a line that reads only its own frame sees the tail of such an onset rather
# than its peak, and the onset is lost.
PEAK_REACH_MS = 1.5 * FRAME_MS
# A line whose own frame is the peak's competes on the snap prior alone, as in
# line-first decoding; a line one frame over is a fallback at this weight. A
# distance-weighted claim instead handed 1/3 notes to a 1/8 line 3 ms from
# their frame (94.8% exact on the mixed-snap test, against >99%). Swept on the
# mixed-snap test: exact up to 0.7, 97.8% at 0.85, 66.7% at 1.0. Hard-map F1
# rises over the same sweep (0.722 -> 0.758), but only because the eval's
# reference notes sit on 20 ms frames, so a note pulled onto the wrong line
# toward its frame matches them better -- each peak emits one note whatever
# the weight, so the weight changes placement and nothing else.
NEIGHBOUR_WEIGHT = 0.7


def _decode_peaks(chart, t, div, grid, thr, longs, family_switch) -> list[TaikoNote]:
    """
    One note per onset peak, placed on the line the peak most plausibly means.

    Line-first decoding asked every grid line for the probability at its own
    frame. An onset spread over two or three frames then lit several lines, and
    an onset whose peak sat a frame off its line lit none. Here the peaks are
    found first -- one per onset, like the frame decoder -- and each becomes a
    cluster of the lines within PEAK_REACH_MS, weighted by distance. The
    Viterbi picks the line, so neighbouring notes still vote on snap family.
    """
    ch = [c for c, _ in HIT_CHANNELS]
    env = chart[ch].max(axis=0).astype(np.float64)
    prev = np.concatenate([[-np.inf], env[:-1]])
    nxt = np.concatenate([env[1:], [-np.inf]])
    peaks = np.flatnonzero((env > prev) & (env >= nxt) & (env > thr))
    peak_ms = peaks * FRAME_MS
    peaks = [k for k, x in zip(peaks, peak_ms)
             if not any(n.time - NMS_MS < x < n.end_time + NMS_MS for n in longs)]

    pt, pp, pd, pc, clusters = [], [], [], [], []
    for k in peaks:
        x = k * FRAME_MS
        lo, hi = np.searchsorted(t, x - PEAK_REACH_MS), np.searchsorted(t, x + PEAK_REACH_MS, "right")
        idx = range(lo, hi) if hi > lo else [int(np.argmin(np.abs(t - x)))]
        members = []
        for i in idx:
            members.append(len(pt))
            pt.append(t[i]); pd.append(div[i]); pc.append(int(chart[ch, k].argmax()))
            f = t[i] / FRAME_MS
            own = int(np.round(f)) == k or (abs(f - np.floor(f) - 0.5) < 0.05
                                             and k in (int(np.floor(f)), int(np.ceil(f))))
            pp.append(env[k] * (1.0 if own else NEIGHBOUR_WEIGHT))
        clusters.append(members)
    if not clusters:
        return []
    pt, pp, pd = np.array(pt), np.array(pp), np.array(pd)
    chosen = _viterbi(clusters, pt, pp, pd, grid, family_switch)
    return [TaikoNote(time=int(round(pt[i])), note_type=HIT_CHANNELS[pc[i]][1]) for i in chosen]


def realised_nps(bm: TaikoBeatmap) -> float:
    """Hits per second over the span they cover -- the measure avg_nps is packed with."""
    hits = sorted(n.time for n in bm.notes if not n.is_long)
    span = (hits[-1] - hits[0]) / 1000.0 if len(hits) > 1 else 0.0
    return len(hits) / span if span > 0 else 0.0


def calibrate_threshold(chart: np.ndarray, timing_points: list[TimingPoint], target_nps: float,
                        lo: float = 0.5, hi: float = 0.99, iters: int = 10,
                        **decode_kwargs) -> tuple[float, float]:
    """
    Hit threshold whose decode lands on `target_nps`, by bisection.

    The model's density tracks the requested NPS only loosely -- measured at
    1.89x the reference on oni maps and 0.67x of a 7.5* request by hand -- but
    the ranking of candidates inside a chart is what it is good at. Moving the
    threshold keeps that ranking and fixes only how many are kept. [lo, hi]
    bounds how far it may go from the autoencoder's own threshold, so a target
    the probabilities cannot honestly support stops at the bound rather than
    admitting noise. Returns (threshold, realised nps).
    """
    # Density moves in steps (a threshold either passes a candidate or not), so
    # the bisection's last midpoint can sit on the far side of the step nearest
    # the target. Every threshold tried is kept and the closest one returned.
    best: tuple[float, float] | None = None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        got = realised_nps(decode_on_grid(chart, timing_points, hit_threshold=mid, **decode_kwargs))
        if best is None or abs(got - target_nps) < abs(best[1] - target_nps):
            best = (mid, got)
        if got > target_nps:
            lo = mid
        else:
            hi = mid
    return best


BINARY = {4, 8}
TERNARY = {3, 6, 12}
# Log-penalty for changing snap family (1/4 vs 1/6) between notes less than a
# beat apart. Measured on synthetic charts that switch family by the phrase:
# 0 placed 100% of notes on the exact line, 1.5 and 3.0 placed 99.2% -- the
# penalty drags a new phrase's first note onto the old family. Kept as a knob
# for A/B on real model output, whose frames are noisier than these.
FAMILY_SWITCH = 0.0

# Log-penalty for picking two lines closer than a hand can re-strike. Two
# peaks two frames apart can each reach a line within PEAK_REACH_MS, and
# choosing freely the Viterbi would put them on lines 22.7 ms apart at 220 BPM
# (the 1/4 and 1/3 lines after a beat) and leave repair to guess which to drop.
# Large enough that a collision is only taken when no other pair of lines
# exists, so repair still sees those.
COLLISION = 50.0


def _family(d: int) -> int:
    """0 neutral (1/1, 1/2 belong to both), 1 binary, 2 ternary."""
    return 1 if d in BINARY else 2 if d in TERNARY else 0


def _viterbi(clusters: list[list[int]], t: np.ndarray, p: np.ndarray, div: np.ndarray,
             grid: Grid, family_switch: float | None = None) -> list[int]:
    """
    Pick one candidate per onset cluster.

    Emission: log(probability x snap prior). Transition: a penalty when two
    notes less than a beat apart sit on different snap families -- a 1/4
    note followed by a 1/6 note. Real charts do switch between 1/4 and 1/6,
    but by the phrase, not note by note; a frame that is ambiguous between
    the two lines is almost always continuing whatever its neighbours are
    doing. Neutral positions (1/1, 1/2) switch for free. Two picks closer
    than MIN_HIT_GAP_MS cost COLLISION.
    """
    if not clusters:
        return []
    switch = FAMILY_SWITCH if family_switch is None else family_switch
    emit = [np.log(np.maximum(p[c], 1e-6) * np.array([SNAP_PRIOR.get(int(div[i]), 0.7) for i in c]))
            for c in clusters]
    score = emit[0].copy()
    # Each state carries the family of the last non-neutral note, so a 1/2
    # between two 1/6s does not reset the phrase.
    fam = [_family(int(div[i])) for i in clusters[0]]
    back: list[np.ndarray] = []
    for k in range(1, len(clusters)):
        prev, cur = clusters[k - 1], clusters[k]
        new_score = np.empty(len(cur))
        new_back = np.empty(len(cur), dtype=int)
        new_fam = []
        for j, b in enumerate(cur):
            fb = _family(int(div[b]))
            beat = grid.section_at(t[b]).ms_per_beat
            best, arg, arg_fam = -np.inf, 0, 0
            for i, a in enumerate(prev):
                fa = fam[i]
                close = (t[b] - t[a]) < beat
                pen = switch if (close and fa and fb and fa != fb) else 0.0
                if t[b] - t[a] < MIN_HIT_GAP_MS:
                    pen += COLLISION
                s = score[i] - pen
                if s > best:
                    best, arg, arg_fam = s, i, (fb or fa if close else fb)
            new_score[j] = best + emit[k][j]
            new_back[j] = arg
            new_fam.append(arg_fam)
        score, fam = new_score, new_fam
        back.append(new_back)
    j = int(np.argmax(score))
    path = [j]
    for bk in reversed(back):
        j = int(bk[j])
        path.append(j)
    path.reverse()
    return [clusters[k][j] for k, j in enumerate(path)]


def _nearest(grid: Grid, time_ms: float, divisors) -> float:
    sec = grid.section_at(time_ms)
    return min((sec.nearest(time_ms, d) for d in divisors), key=lambda x: abs(x - time_ms))
