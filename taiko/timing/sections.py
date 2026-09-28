"""
taiko/timing/sections.py

Beat times -> osu! red lines.

Precision comes from fitting, not from any single beat. A beat tracker's
individual beats are off by several ms; one straight line through a hundred of
them pins the tempo to hundredths of a BPM and the offset to about a
millisecond. This is the same insight Mapperatorinator's super timing rests on
(it averages 20 shifted runs, then fits whole sections); the fitting here is
rewritten for beat lists rather than its token histograms.

1. Segment: optimal piecewise-linear fit of time against beat index (dynamic
   programming, a penalty per extra section), so BPM changes become section
   boundaries and a steady song stays one section.
2. Fit each section robustly: least squares, then again without outliers.
3. Round the BPM like a mapper would -- integer, then .5, .1, .01 -- keeping the
   first that still fits every beat within the leniency (Mapperatorinator's
   "human rounding").
4. Merge neighbours that ended up with the same BPM on a continuous grid.
5. Polish each offset against a ~1 ms onset envelope, if one is given.
6. Place each red line on a downbeat, from downbeat evidence if available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FittedSection:
    first_beat: int          # index into the beat list
    last_beat: int           # inclusive
    offset_ms: float         # time of beat `first_beat` on the fitted grid
    ms_per_beat: float
    rms_ms: float
    max_ms: float
    meter: int = 4
    downbeat_phase: int = 0  # which beat (from first_beat) is a downbeat
    # Where the section starts and ends, in beats of its own grid counted from
    # offset_ms. Default to the tracked beats; refine_boundaries moves them
    # when the activation says the tempo changed a beat or two elsewhere.
    k_start: int = 0
    k_end: int | None = None

    @property
    def bpm(self) -> float:
        return 60_000.0 / self.ms_per_beat

    @property
    def n_beats(self) -> int:
        return self.last_beat - self.first_beat + 1

    @property
    def start_ms(self) -> float:
        return self.offset_ms + self.k_start * self.ms_per_beat

    @property
    def end_ms(self) -> float:
        k = self.n_beats - 1 if self.k_end is None else self.k_end
        return self.offset_ms + k * self.ms_per_beat


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #

def _noise_ms(times: np.ndarray) -> float:
    """Robust per-beat jitter: MAD of second differences, scaled for a line."""
    if len(times) < 5:
        return 5.0
    d2 = np.diff(times, 2)
    mad = np.median(np.abs(d2 - np.median(d2)))
    return max(1.4826 * mad / np.sqrt(6.0), 1.0)


def segment_beats(times: np.ndarray, min_beats: int = 8,
                  penalty: float | None = None) -> list[tuple[int, int]]:
    """
    Split beats into runs of constant tempo.

    Cost of a run is the SSE of a straight line through (index, time); each
    run costs `penalty` extra, so a split must buy back more error than it
    costs. The default penalty scales with the measured jitter, so a noisy
    activation does not shatter a steady song into pieces.
    """
    n = len(times)
    if n < 2 * min_beats:
        return [(0, n - 1)]
    sigma = _noise_ms(times)
    if penalty is None:
        penalty = 30.0 * sigma ** 2 * np.log(n)

    x = np.arange(n, dtype=np.float64)
    y = np.asarray(times, dtype=np.float64)
    cx, cy = np.concatenate([[0], np.cumsum(x)]), np.concatenate([[0], np.cumsum(y)])
    cxx = np.concatenate([[0], np.cumsum(x * x)])
    cxy = np.concatenate([[0], np.cumsum(x * y)])
    cyy = np.concatenate([[0], np.cumsum(y * y)])

    def sse(i: np.ndarray, j: int) -> np.ndarray:
        m = j - i
        sx, sy = cx[j] - cx[i], cy[j] - cy[i]
        sxx, sxy, syy = cxx[j] - cxx[i], cxy[j] - cxy[i], cyy[j] - cyy[i]
        vx = sxx - sx * sx / m
        cov = sxy - sx * sy / m
        vy = syy - sy * sy / m
        return np.maximum(vy - np.where(vx > 0, cov * cov / np.maximum(vx, 1e-12), 0.0), 0.0)

    best = np.full(n + 1, np.inf)
    best[0] = 0.0
    prev = np.zeros(n + 1, dtype=np.int64)
    for j in range(min_beats, n + 1):
        i = np.arange(0, j - min_beats + 1)
        i = i[np.isfinite(best[i])]
        if i.size == 0:
            continue
        cost = best[i] + sse(i, j) + penalty
        k = int(np.argmin(cost))
        best[j], prev[j] = cost[k], i[k]

    if not np.isfinite(best[n]):
        return [(0, n - 1)]
    runs, j = [], n
    while j > 0:
        i = int(prev[j])
        runs.append((i, j - 1))
        j = i
    return runs[::-1]


# --------------------------------------------------------------------------- #
# Fitting and rounding
# --------------------------------------------------------------------------- #

def _fit(times: np.ndarray, idx: np.ndarray) -> tuple[float, float]:
    A = np.stack([idx, np.ones_like(idx)], axis=1)
    slope, intercept = np.linalg.lstsq(A, times, rcond=None)[0]
    return float(slope), float(intercept)


def robust_fit(times: np.ndarray) -> tuple[float, float, np.ndarray]:
    """(ms_per_beat, time of index 0, inlier mask). Two passes, 3-sigma clip."""
    idx = np.arange(len(times), dtype=np.float64)
    mpb, t0 = _fit(times, idx)
    resid = times - (t0 + mpb * idx)
    sigma = max(1.4826 * np.median(np.abs(resid - np.median(resid))), 1.0)
    inlier = np.abs(resid) <= 3.0 * sigma
    if inlier.sum() >= max(4, len(times) // 2):
        mpb, t0 = _fit(times[inlier], idx[inlier])
    return mpb, t0, inlier


def human_round(times: np.ndarray, mpb: float, t0: float, inlier: np.ndarray,
                leniency_ms: float) -> tuple[float, float]:
    """
    The roundest BPM that still fits.

    Tries integer, .5, .1 and .01 BPM in that order; for each, the offset is
    refitted as the mean residual, and the candidate is accepted if every
    inlier beat stays within `leniency_ms` and the RMS error grows by less than
    a millisecond. Falls back to the unrounded fit.
    """
    idx = np.arange(len(times), dtype=np.float64)
    base = times[inlier] - (t0 + mpb * idx[inlier])
    base_rms = float(np.sqrt(np.mean(base ** 2))) if base.size else 0.0
    bpm = 60_000.0 / mpb
    for step in (1.0, 0.5, 0.1, 0.01):
        cand_bpm = round(bpm / step) * step
        if cand_bpm <= 0:
            continue
        cand_mpb = 60_000.0 / cand_bpm
        cand_t0 = float(np.mean(times[inlier] - cand_mpb * idx[inlier]))
        r = times[inlier] - (cand_t0 + cand_mpb * idx[inlier])
        if r.size and np.max(np.abs(r)) <= leniency_ms and \
                np.sqrt(np.mean(r ** 2)) <= base_rms + 1.0:
            return cand_mpb, cand_t0
    return mpb, t0


def fit_sections(times: np.ndarray, runs: list[tuple[int, int]],
                 leniency_ms: float = 12.0) -> list[FittedSection]:
    out = []
    for a, b in runs:
        seg = np.asarray(times[a:b + 1], dtype=np.float64)
        mpb, t0, inlier = robust_fit(seg)
        mpb, t0 = human_round(seg, mpb, t0, inlier, leniency_ms)
        r = seg - (t0 + mpb * np.arange(len(seg)))
        out.append(FittedSection(a, b, t0, mpb, float(np.sqrt(np.mean(r[inlier] ** 2))),
                                 float(np.max(np.abs(r[inlier]))) if inlier.any() else 0.0))
    return out


def merge_sections(sections: list[FittedSection], times: np.ndarray,
                   leniency_ms: float = 12.0) -> list[FittedSection]:
    """
    Join neighbours that ended up on the same grid.

    Segmentation can split a steady song where the tracker wobbled; if both
    halves round to the same BPM and the second half's beats still sit on the
    first half's grid, they were one section all along -- and one red line is
    what a mapper would write.
    """
    if not sections:
        return sections
    merged = [sections[0]]
    for sec in sections[1:]:
        last = merged[-1]
        if abs(last.bpm - sec.bpm) < 0.005:
            seg = np.asarray(times[last.first_beat:sec.last_beat + 1], dtype=np.float64)
            idx = np.arange(len(seg))
            r = seg - (last.offset_ms + last.ms_per_beat * idx)
            if np.percentile(np.abs(r), 90) <= leniency_ms:
                t0 = float(np.mean(seg - last.ms_per_beat * idx))
                r = seg - (t0 + last.ms_per_beat * idx)
                merged[-1] = FittedSection(last.first_beat, sec.last_beat, t0,
                                           last.ms_per_beat, float(np.sqrt(np.mean(r ** 2))),
                                           float(np.max(np.abs(r))))
                continue
        merged.append(sec)
    return merged


# --------------------------------------------------------------------------- #
# Offset polish and downbeats
# --------------------------------------------------------------------------- #

def polish_offsets(sections: list[FittedSection], envelope: np.ndarray, env_fps: float,
                   search_ms: float = 8.0, step_ms: float = 0.5) -> list[float]:
    """
    Nudge each section's offset to where the onset envelope lines up best.

    The score is the envelope summed over the section's beat and half-beat
    positions. Returns the shift applied to each section, in ms. A shift is
    kept only if it improves the score by more than 2%, so a flat envelope
    (a pad, a vocal intro) does not drag the grid around.
    """
    shifts = []
    grid_t = np.arange(len(envelope)) * 1000.0 / env_fps
    for sec in sections:
        k = np.arange(0, sec.n_beats, 0.5)
        weights = np.where(k % 1 == 0, 1.0, 0.5)
        base = sec.offset_ms + k * sec.ms_per_beat

        def score(d: float) -> float:
            return float(np.sum(weights * np.interp(base + d, grid_t, envelope,
                                                    left=0.0, right=0.0)))

        deltas = np.arange(-search_ms, search_ms + 1e-9, step_ms)
        scores = np.array([score(d) for d in deltas])
        best = float(deltas[int(np.argmax(scores))])
        s0 = score(0.0)
        if scores.max() > s0 * 1.02 + 1e-9:
            sec.offset_ms += best
            shifts.append(best)
        else:
            shifts.append(0.0)
    return shifts


def choose_downbeats(sections: list[FittedSection], downbeat_at_beat: np.ndarray | None,
                     meter: int = 4) -> None:
    """Pick each section's downbeat phase from per-beat downbeat evidence."""
    for sec in sections:
        sec.meter = meter
        if downbeat_at_beat is None:
            sec.downbeat_phase = 0
            continue
        ev = np.asarray(downbeat_at_beat[sec.first_beat:sec.last_beat + 1], dtype=np.float64)
        if ev.size < meter:
            sec.downbeat_phase = 0
            continue
        means = [ev[p::meter].mean() for p in range(meter)]
        sec.downbeat_phase = int(np.argmax(means))


def absorb_short_sections(sections: list[FittedSection], times: np.ndarray,
                          min_beats: int = 16, leniency_ms: float = 12.0) -> list[FittedSection]:
    """
    Fold short sections into a neighbour whose grid already covers them.

    The tracker's first few beats wobble while it locks on, and the last few
    can drift as the music fades; segmentation turns each wobble into an
    8-beat "section" with an odd BPM. If the longer neighbour's grid, extended
    over those beats, still fits them within the leniency, they were never a
    tempo change. A short section nothing can absorb is kept -- a real
    two-bar ritardando looks exactly like that.
    """
    secs = list(sections)
    changed = True
    while changed and len(secs) > 1:
        changed = False
        order = sorted(range(len(secs)), key=lambda i: secs[i].n_beats)
        for i in order:
            s = secs[i]
            if s.n_beats >= min_beats:
                break
            neighbours = [j for j in (i - 1, i + 1) if 0 <= j < len(secs)]
            neighbours.sort(key=lambda j: -secs[j].n_beats)
            for j in neighbours:
                nb = secs[j]
                idx = np.arange(s.first_beat, s.last_beat + 1) - nb.first_beat
                r = times[s.first_beat:s.last_beat + 1] - (nb.offset_ms + nb.ms_per_beat * idx)
                if np.percentile(np.abs(r), 75) <= leniency_ms:
                    first = min(nb.first_beat, s.first_beat)
                    offset = nb.offset_ms + (first - nb.first_beat) * nb.ms_per_beat
                    secs[j] = FittedSection(first, max(nb.last_beat, s.last_beat), offset,
                                            nb.ms_per_beat, nb.rms_ms, nb.max_ms, nb.meter,
                                            nb.downbeat_phase)
                    del secs[i]
                    changed = True
                    break
            if changed:
                break
    return secs


def drop_ragged_edges(sections: list[FittedSection], max_rms_ms: float = 4.0,
                      min_beats: int = 16) -> list[FittedSection]:
    """Short, badly fitting sections at either end are tracker noise, not music."""
    secs = list(sections)
    while len(secs) > 1 and secs[0].n_beats < min_beats and secs[0].rms_ms > max_rms_ms:
        secs.pop(0)
    while len(secs) > 1 and secs[-1].n_beats < min_beats and secs[-1].rms_ms > max_rms_ms:
        secs.pop()
    return secs


def refine_boundaries(sections: list[FittedSection], act: np.ndarray, fps: float,
                      search_beats: int = 6) -> None:
    """
    Decide exactly where each tempo change happens, from the activation.

    The tracker follows a smoothed tempo curve, so around a change it glides:
    a few beats come out between the two grids, and segmentation files them
    with whichever tempo they happen to fit. On a click track that moved a
    150 -> 200 BPM change 1.2 s early, re-timing three real 150 BPM beats.

    Both sections' grids are already fitted to hundreds of beats, so each
    candidate change point can be scored directly: activation summed over the
    old grid up to it plus the new grid after it. The best candidate sets
    where the first section ends and the second begins.
    """
    if len(sections) < 2:
        return
    a = np.asarray(act, dtype=np.float64)
    a = a - np.median(a)
    sd = a.std()
    a = a / sd if sd > 1e-9 else a
    # Tolerate a frame of misalignment either way.
    a = np.maximum(a, np.maximum(np.roll(a, 1), np.roll(a, -1)))
    t_axis = np.arange(len(a)) * 1000.0 / fps

    def at(t: np.ndarray) -> np.ndarray:
        return np.interp(t, t_axis, a, left=0.0, right=0.0)

    for A, B in zip(sections[:-1], sections[1:]):
        lo = min(A.end_ms, B.start_ms) - search_beats * A.ms_per_beat
        hi = max(A.end_ms, B.start_ms) + search_beats * B.ms_per_beat
        ka = np.arange(np.ceil((lo - A.offset_ms) / A.ms_per_beat),
                       np.floor((hi - A.offset_ms) / A.ms_per_beat) + 1).astype(int)
        kb = np.arange(np.ceil((lo - B.offset_ms) / B.ms_per_beat),
                       np.floor((hi - B.offset_ms) / B.ms_per_beat) + 1).astype(int)
        if ka.size == 0 or kb.size == 0:
            continue
        ta = A.offset_ms + ka * A.ms_per_beat
        tb = B.offset_ms + kb * B.ms_per_beat
        va, vb = at(ta), at(tb)
        gap = 0.4 * min(A.ms_per_beat, B.ms_per_beat)
        best, best_i, best_j = -np.inf, None, None
        for i in range(len(ta)):
            j = int(np.searchsorted(tb, ta[i] + gap))
            if j >= len(tb):
                continue
            s = va[:i + 1].sum() + vb[j:].sum()
            if s > best:
                best, best_i, best_j = s, i, j
        if best_i is None:
            continue
        A.k_end = int(ka[best_i])
        B.k_start = int(kb[best_j])


def red_line_times(sections: list[FittedSection], times: np.ndarray | None = None) -> list[float]:
    """
    Where each red line goes.

    On the section's first downbeat where that is possible: the red line may
    move back from the section's first beat to the downbeat before it, but
    never onto or before the previous section's last beat -- that beat
    belongs to the previous tempo, and a red line ahead of it would re-time
    it. The first red line is pulled back by whole measures to the earliest
    downbeat at or after 0 ms, the way songs are usually timed.
    """
    out: list[float] = []
    for i, sec in enumerate(sections):
        first = sec.start_ms
        measure = sec.meter * sec.ms_per_beat
        t = first
        # Beat k is a downbeat when k = downbeat_phase (mod meter).
        back = (sec.k_start - sec.downbeat_phase) % sec.meter
        if back:
            candidate = first - back * sec.ms_per_beat
            prev_last = sections[i - 1].end_ms if i > 0 else -np.inf
            if candidate > prev_last + 0.25 * sec.ms_per_beat:
                t = candidate
        if i == 0:
            # osu! extends the first red line's grid backwards, so its exact
            # position only has to be a downbeat; keep it at or after 0 ms.
            while t - measure >= 0:
                t -= measure
            while t < 0:
                t += measure
        out.append(t)
    return out
