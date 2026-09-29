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
each subdivision of each red line's own tempo. Each one then gets a score
from the frames around it, and positions are kept by non-maximum suppression,
so every note is on a line by construction. A mild prior prefers common
snaps (1/1, 1/2, 1/4) over rare ones (1/8, 1/12). When a frame could belong
to either of two nearby lines, the probability and the prior decide together,
not whichever divisor happens to be tried first.

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
from taiko.data.tensor_repr import (
    CH_BIG_DON, CH_BIG_KAT, CH_DON, CH_KAT, tensor_to_beatmap,
)

# Allowed subdivisions and how much each is preferred when two compete for
# the same onset. Values are multipliers on the onset probability.
SNAP_PRIOR = {1: 1.0, 2: 1.0, 4: 1.0, 3: 0.9, 6: 0.85, 8: 0.8, 12: 0.7}

# Two hits closer than this are one hit seen on two lines. A frame is 20 ms,
# and no playable taiko rhythm puts hits closer than ~30 ms.
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


def _evidence(channel: np.ndarray, t_ms: np.ndarray, slack: float = 0.05) -> np.ndarray:
    """
    Onset probability available to a position: the frame a note at that time
    would have been written to, round(t / 20). Only that frame -- reading
    neighbours too lets one onset be claimed by grid lines 30 ms apart. A
    position within `slack` frames of a frame boundary reads both frames,
    since rounding could have sent it either way.
    """
    f = t_ms / FRAME_MS
    n = len(channel)
    near = np.clip(np.round(f).astype(int), 0, n - 1)
    best = channel[near].astype(np.float64)
    frac = f - np.floor(f)
    edge = np.abs(frac - 0.5) < slack
    other = np.clip(np.where(near > f, near - 1, near + 1), 0, n - 1)
    best[edge] = np.maximum(best[edge], channel[other[edge]])
    return best


def decode_on_grid(
    chart: np.ndarray,
    timing_points: list[TimingPoint],
    threshold: float = 0.5,
    divisors=tuple(SNAP_PRIOR),
    family_switch: float | None = None,
    **meta,
) -> TaikoBeatmap:
    """
    [6, T] chart probabilities -> a beatmap whose hits all sit on the grid.

    Long notes come from the same region decoding as tensor_to_beatmap, with
    their ends moved to the nearest grid line. `meta` is passed through to
    tensor_to_beatmap for title, version and so on.
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
    prior = np.array([SNAP_PRIOR.get(c.divisor, 0.7) for c in cands])

    probs = np.stack([_evidence(chart[ch], t) for ch, _ in HIT_CHANNELS])     # [4, N]
    best_ch = probs.argmax(axis=0)
    p = probs.max(axis=0)
    div = np.array([c.divisor for c in cands])
    live = np.flatnonzero((p > threshold)
                          & ~np.array([any(n.time - NMS_MS < x < n.end_time + NMS_MS
                                           for n in longs) for x in t]))

    # Candidates within NMS_MS of each other are one onset seen on several
    # lines. Group them, then choose one line per onset for the whole song at
    # once, so neighbouring notes can vote.
    clusters: list[list[int]] = []
    for i in live:
        if clusters and t[i] - t[clusters[-1][0]] < NMS_MS:
            clusters[-1].append(int(i))
        else:
            clusters.append([int(i)])
    chosen = _viterbi(clusters, t, p, div, grid, family_switch)

    hits = [TaikoNote(time=int(round(t[i])), note_type=HIT_CHANNELS[best_ch[i]][1])
            for i in chosen]
    bm.notes = sorted(longs + hits, key=lambda n: (n.time, 0 if n.is_long else 1))
    bm.compute_stats()
    return bm


BINARY = {4, 8}
TERNARY = {3, 6, 12}
# Log-penalty for changing snap family (1/4 vs 1/6) between notes less than a
# beat apart. Measured on synthetic charts that switch family by the phrase:
# 0 placed 100% of notes on the exact line, 1.5 and 3.0 placed 99.2% -- the
# penalty drags a new phrase's first note onto the old family. Kept as a knob
# for A/B on real model output, whose frames are noisier than these.
FAMILY_SWITCH = 0.0


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
    doing. Neutral positions (1/1, 1/2) switch for free.
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
