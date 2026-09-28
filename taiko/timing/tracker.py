"""
taiko/timing/tracker.py

Beat activation -> a beat time for every beat of the song.

Works on any per-frame activation at any frame rate: a neural beat
probability (beat_this, TimingNet) or a percussive onset envelope. Three steps:

1. Global tempo from the activation's autocorrelation, weighted by a
   log-normal prior on BPM. The prior is centred where taiko maps live
   (~170 BPM) rather than librosa's 120, because the half/double ambiguity is
   the commonest way a tracker gets a rhythm-game song wrong, and it should be
   resolved the way a taiko mapper would.
2. Local tempo in sliding windows, folded to within 1.5x of the global tempo
   (as Mapperatorinator folds its BPM estimates), so a BPM change is followed
   but an octave flip is not.
3. Dynamic-programming beat tracking (Ellis 2007) against that local period:
   each beat is placed to maximise activation while keeping inter-beat
   intervals close to the local period. Beats are then refined below one
   frame by parabolic interpolation on the activation.
"""

from __future__ import annotations

import numpy as np

PRIOR_BPM = 170.0
PRIOR_OCTAVES = 0.9
BPM_RANGE = (60.0, 330.0)


def _normalise(act: np.ndarray) -> np.ndarray:
    act = np.asarray(act, dtype=np.float64)
    act = act - np.median(act)
    sd = act.std()
    return act / sd if sd > 1e-9 else act


def _autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    n = len(x)
    size = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(x, size)
    ac = np.fft.irfft(spec * np.conj(spec), size)[: max_lag + 1]
    return ac / max(ac[0], 1e-9)


def _tempo_scores(act: np.ndarray, fps: float, bpm_lo: float, bpm_hi: float,
                  prior_bpm: float | None, prior_octaves: float):
    max_lag = int(np.ceil(fps * 60.0 / bpm_lo)) + 2
    ac = _autocorr(act, max_lag)
    lags = np.arange(1, max_lag + 1)
    bpms = 60.0 * fps / lags
    ok = (bpms >= bpm_lo) & (bpms <= bpm_hi)
    # Reward periods whose double is also periodic: a true beat usually has
    # energy at 2x its period too, a spurious sub-beat does not.
    doubled = np.array([ac[min(2 * l, max_lag)] for l in lags])
    score = ac[lags] + 0.5 * doubled
    if prior_bpm:
        score = score * np.exp(-0.5 * (np.log2(bpms / prior_bpm) / prior_octaves) ** 2)
    score[~ok] = -np.inf
    return lags, score


def _refine_peak(lags: np.ndarray, score: np.ndarray, i: int) -> float:
    if 0 < i < len(score) - 1 and np.isfinite(score[i - 1]) and np.isfinite(score[i + 1]):
        a, b, c = score[i - 1], score[i], score[i + 1]
        denom = a - 2 * b + c
        if abs(denom) > 1e-12:
            return lags[i] + 0.5 * (a - c) / denom
    return float(lags[i])


def global_tempo(act: np.ndarray, fps: float, bpm_range=BPM_RANGE,
                 prior_bpm: float | None = PRIOR_BPM,
                 prior_octaves: float = PRIOR_OCTAVES) -> float:
    lags, score = _tempo_scores(_normalise(act), fps, bpm_range[0], bpm_range[1],
                                prior_bpm, prior_octaves)
    i = int(np.argmax(score))
    return 60.0 * fps / _refine_peak(lags, score, i)


def local_periods(act: np.ndarray, fps: float, global_bpm: float,
                  window_s: float = 8.0, hop_s: float = 1.0) -> np.ndarray:
    """
    Beat period in frames, per frame. Each window's tempo is searched only
    within 1.5x of the global tempo, with a mild pull towards it, so a real
    tempo change moves the estimate but a half-time bar does not.
    """
    act = _normalise(act)
    n = len(act)
    win = int(window_s * fps)
    hop = max(1, int(hop_s * fps))
    lo, hi = global_bpm / 1.5, global_bpm * 1.5
    centres, periods = [], []
    for c in range(0, n, hop):
        a, b = max(0, c - win // 2), min(n, c + win // 2)
        if b - a < win // 2:
            continue
        lags, score = _tempo_scores(act[a:b], fps, lo, hi, global_bpm, 0.35)
        if not np.isfinite(score).any():
            continue
        i = int(np.argmax(score))
        centres.append(c)
        periods.append(_refine_peak(lags, score, i))
    period_global = 60.0 * fps / global_bpm
    if not centres:
        return np.full(n, period_global)
    periods = np.asarray(periods)
    # Median over 5 windows: a one-window blip is noise, not a tempo change.
    if len(periods) >= 5:
        padded = np.pad(periods, 2, mode="edge")
        periods = np.median(np.stack([padded[i:i + len(periods)] for i in range(5)]), axis=0)
    return np.interp(np.arange(n), centres, periods)


def track_beats(act: np.ndarray, fps: float, periods: np.ndarray,
                tightness: float = 100.0) -> np.ndarray:
    """
    Beat frames by dynamic programming (Ellis 2007) with a time-varying period.

    score[t] = act[t] + max over predecessors p of
               score[p] - tightness * log((t - p) / period[t])^2
    """
    act = _normalise(act)
    n = len(act)
    score = act.copy()
    back = np.full(n, -1, dtype=np.int64)
    for t in range(n):
        P = periods[t]
        lo, hi = int(t - 2.0 * P), int(t - 0.5 * P)
        if hi <= 0:
            continue
        lo = max(lo, 0)
        cand = np.arange(lo, hi)
        penalty = tightness * np.log((t - cand) / P) ** 2
        vals = score[lo:hi] - penalty
        j = int(np.argmax(vals))
        score[t] = act[t] + vals[j]
        back[t] = lo + j

    # End on the best-scoring frame within the last period.
    tail = int(max(1, periods[-1]))
    end = n - tail + int(np.argmax(score[n - tail:]))
    beats = [end]
    while back[beats[-1]] >= 0:
        beats.append(int(back[beats[-1]]))
    return trim_weak_ends(act, np.asarray(beats[::-1], dtype=np.int64))


def trim_weak_ends(act: np.ndarray, beats: np.ndarray, rel: float = 0.3,
                   run: int = 4) -> np.ndarray:
    """
    Drop the beats the DP invented in silence before the music starts and
    after it ends.

    A beat is "real" if the activation under it reaches `rel` of the median
    beat's; the song starts at the first run of `run` real beats in a row and
    ends at the last one. Beats inside a break stay: they are where a mapper's
    grid runs through the break too.
    """
    if len(beats) < 2 * run:
        return beats
    strength = np.array([act[max(0, b - 1):b + 2].max() for b in beats])
    ref = np.median(strength[strength > 0]) if (strength > 0).any() else 0.0
    real = strength >= rel * ref
    runs = np.convolve(real.astype(int), np.ones(run, dtype=int), mode="valid") == run
    if not runs.any():
        return beats
    first = int(np.flatnonzero(runs)[0])
    last = int(np.flatnonzero(runs)[-1]) + run - 1
    return beats[first:last + 1]


def refine_subframe(act: np.ndarray, frames: np.ndarray, search: int = 2) -> np.ndarray:
    """Beat positions in fractional frames: nearest local max, then a parabola."""
    act = np.asarray(act, dtype=np.float64)
    n = len(act)
    out = np.empty(len(frames), dtype=np.float64)
    for k, f in enumerate(frames):
        a, b = max(1, f - search), min(n - 1, f + search + 1)
        i = a + int(np.argmax(act[a:b])) if b > a else int(f)
        i = min(max(i, 1), n - 2)
        y0, y1, y2 = act[i - 1], act[i], act[i + 1]
        denom = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
        out[k] = i + float(np.clip(delta, -0.5, 0.5))
    return out


def beat_times_ms(act: np.ndarray, fps: float, bpm_range=BPM_RANGE,
                  prior_bpm: float | None = PRIOR_BPM) -> tuple[np.ndarray, float]:
    """(beat times in ms, global BPM)."""
    bpm = global_tempo(act, fps, bpm_range, prior_bpm)
    periods = local_periods(act, fps, bpm)
    frames = track_beats(act, fps, periods)
    return refine_subframe(act, frames) * 1000.0 / fps, bpm
