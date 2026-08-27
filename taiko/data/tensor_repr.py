"""
taiko/data/tensor_repr.py

Beatmap <-> tensor representation.

A map becomes TWO arrays on the shared time grid from `taiko.data.frames`:

    chart  [6, T]   what the model generates
    timing [3, T]   what the model is told

    chart channels          timing channels
    ---------------------   -------------------------------------
    0  don                  0  sin(2*pi * beat_phase)
    1  kat                  1  cos(2*pi * beat_phase)
    2  big_don              2  downbeat proximity, (1+cos(measure))/2
    3  big_kat
    4  roll     (sustained)
    5  denden   (sustained)

Why the beat grid is an input, not an output
--------------------------------------------
The previous representation had the beat grid as channel 6 of the generated
tensor, so the model had to hallucinate a tempo out of noise and inference then
reverse-engineered BPM from whatever notes came back. But tempo is *known* at
generation time -- the user supplies it, or `beat_snap.detect_bpm` finds it.
Feeding it in means notes land on the grid by construction instead of being
snapped back onto it afterwards.

Continuous phase rather than a pulse train matters here. A pulse can only say
"a beat is somewhere in this 20 ms frame"; sin/cos phase says exactly where
inside the frame the beat falls, which is the sub-frame precision taiko needs
at 1/4 and 1/6 snaps.

True BPM is used, not a normalised one. Normalising into a 150-300 band (as a
pulse-density representation must) would make measure boundaries meaningless,
and a continuous phase has no density problem to solve.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional

from taiko.data.frames import (
    FRAME_MS,
    frame_to_ms,
    ms_to_frame,
)
from taiko.data.osu_parser import TaikoBeatmap, TaikoNote, TimingPoint


# --------------------------------------------------------------------------- #
# Channel layout
# --------------------------------------------------------------------------- #

N_CHART_CHANNELS  = 6
N_TIMING_CHANNELS = 3

CH_DON     = 0
CH_KAT     = 1
CH_BIG_DON = 2
CH_BIG_KAT = 3
CH_ROLL    = 4
CH_DENDEN  = 5

CHART_CHANNEL_NAMES = ["don", "kat", "big_don", "big_kat", "roll", "denden"]

TM_SIN      = 0
TM_COS      = 1
TM_DOWNBEAT = 2

ONSET_CHANNELS = (CH_DON, CH_KAT, CH_BIG_DON, CH_BIG_KAT)
HOLD_CHANNELS  = (CH_ROLL, CH_DENDEN)

MAX_FRAMES = 45_000     # 900 s -- a hard ceiling, not an expected length

# Retained for callers that still import the old name.
N_CHANNELS = N_CHART_CHANNELS

DEFAULT_MS_PER_BEAT = 500.0     # osu!'s own fallback: 120 BPM
DEFAULT_METER       = 4


# --------------------------------------------------------------------------- #
# Timing stream
# --------------------------------------------------------------------------- #

def _red_line_segments(
    timing_points: list[TimingPoint],
    total_ms: float,
) -> list[tuple[float, float, float, int]]:
    """
    Split the map into spans of constant tempo.

    Returns a list of (start_ms, end_ms, ms_per_beat, meter). Green lines carry
    slider velocity only and never move the beat grid, so they are ignored here.
    Timing points are assumed already sorted by time.
    """
    reds = [
        tp for tp in timing_points
        if tp.uninherited and tp.beat_length > 0
    ]

    if not reds:
        return [(0.0, total_ms, DEFAULT_MS_PER_BEAT, DEFAULT_METER)]

    segments: list[tuple[float, float, float, int]] = []

    # Audio before the first red line still needs a grid; extend the first
    # tempo backwards rather than leaving a dead zone at the start of the map.
    if reds[0].time > 0:
        segments.append((
            0.0, float(reds[0].time),
            float(reds[0].beat_length), max(1, reds[0].meter),
        ))

    for i, tp in enumerate(reds):
        end = float(reds[i + 1].time) if i + 1 < len(reds) else total_ms
        if end <= tp.time:
            continue
        segments.append((
            float(tp.time), end,
            float(tp.beat_length), max(1, tp.meter),
        ))

    return segments


def build_timing_stream(
    timing_points: list[TimingPoint],
    n_frames: int,
) -> np.ndarray:
    """
    Build the [3, T] conditioning stream from a map's timing points.

    Phase is measured from each red line's own offset, so a mid-map tempo
    change restarts the grid exactly where osu! restarts it.
    """
    stream = np.zeros((N_TIMING_CHANNELS, n_frames), dtype=np.float32)
    if n_frames <= 0:
        return stream

    total_ms = n_frames * FRAME_MS
    frame_ms = np.arange(n_frames, dtype=np.float64) * FRAME_MS

    for start_ms, end_ms, ms_per_beat, meter in _red_line_segments(timing_points, total_ms):
        lo = max(0, int(np.floor(start_ms / FRAME_MS)))
        hi = min(n_frames, int(np.ceil(end_ms / FRAME_MS)))
        if hi <= lo or ms_per_beat <= 0:
            continue

        t = frame_ms[lo:hi] - start_ms

        beat_phase = (t / ms_per_beat) % 1.0
        stream[TM_SIN, lo:hi] = np.sin(2 * np.pi * beat_phase)
        stream[TM_COS, lo:hi] = np.cos(2 * np.pi * beat_phase)

        measure_phase = (t / (ms_per_beat * meter)) % 1.0
        stream[TM_DOWNBEAT, lo:hi] = 0.5 * (1.0 + np.cos(2 * np.pi * measure_phase))

    return stream


def timing_stream_from_bpm(
    bpm: float,
    offset_ms: float,
    n_frames: int,
    meter: int = DEFAULT_METER,
) -> np.ndarray:
    """
    Build the timing stream at inference, from a single detected or
    user-supplied tempo. Equivalent to a map with one red line.
    """
    if bpm <= 0:
        bpm = 60_000.0 / DEFAULT_MS_PER_BEAT
    tp = TimingPoint(
        time=int(round(offset_ms)),
        beat_length=60_000.0 / bpm,
        meter=max(1, meter),
        uninherited=True,
    )
    return build_timing_stream([tp], n_frames)


# --------------------------------------------------------------------------- #
# Beatmap -> tensors
# --------------------------------------------------------------------------- #

def beatmap_to_tensors(
    bm: TaikoBeatmap,
    max_frames: int = MAX_FRAMES,
    pad_to: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a parsed beatmap into (chart [6, T], timing [3, T]), both float32.

    Long notes mark their onset frame in the sustain channel too, so a roll and
    a denden are each fully described by one channel rather than needing a
    separate onset channel to disambiguate them.
    """
    if bm.notes:
        last_time = max(n.end_time if n.is_long else n.time for n in bm.notes)
    else:
        last_time = 0

    n_frames = min(ms_to_frame(last_time) + 50, max_frames)
    if pad_to is not None:
        n_frames = pad_to
    n_frames = max(n_frames, 1)

    chart = np.zeros((N_CHART_CHANNELS, n_frames), dtype=np.float32)

    for note in bm.notes:
        sf = ms_to_frame(note.time)
        if sf >= n_frames or sf < 0:
            continue

        if note.note_type == "don":
            chart[CH_DON, sf] = 1.0
        elif note.note_type == "kat":
            chart[CH_KAT, sf] = 1.0
        elif note.note_type == "big_don":
            chart[CH_BIG_DON, sf] = 1.0
        elif note.note_type == "big_kat":
            chart[CH_BIG_KAT, sf] = 1.0
        elif note.note_type == "roll":
            ef = min(ms_to_frame(note.end_time), n_frames - 1)
            chart[CH_ROLL, sf:max(ef, sf) + 1] = 1.0
        elif note.note_type == "denden":
            ef = min(ms_to_frame(note.end_time), n_frames - 1)
            chart[CH_DENDEN, sf:max(ef, sf) + 1] = 1.0

    timing = build_timing_stream(bm.timing_points, n_frames)
    return chart, timing


def beatmap_to_tensor(bm: TaikoBeatmap, **kwargs) -> np.ndarray:
    """Chart channels only, for callers that do not need the timing stream."""
    chart, _ = beatmap_to_tensors(bm, **kwargs)
    return chart


# --------------------------------------------------------------------------- #
# Tensors -> beatmap
# --------------------------------------------------------------------------- #

def tensor_to_beatmap(
    chart: np.ndarray,
    bpm: float,
    offset_ms: float,
    threshold: float = 0.5,
    min_long_frames: int = 3,
    title: str = "AI Generated",
    artist: str = "",
    version: str = "AI",
    audio_filename: str = "audio.mp3",
    overall_difficulty: float = 5.0,
    meter: int = DEFAULT_METER,
) -> TaikoBeatmap:
    """Convert a [6, T] chart back into a beatmap."""
    if chart.shape[0] < N_CHART_CHANNELS:
        raise ValueError(
            f"expected {N_CHART_CHANNELS} chart channels, got {chart.shape[0]}"
        )

    bm = TaikoBeatmap()
    bm.title              = title
    bm.artist             = artist
    bm.creator            = "TaikoAI"
    bm.version            = version
    bm.audio_filename     = audio_filename
    bm.overall_difficulty = overall_difficulty
    bm.hp_drain           = min(10.0, overall_difficulty * 0.8)
    bm.slider_multiplier  = 1.4
    bm.slider_tick_rate   = 1.0
    bm.approach_rate      = overall_difficulty

    bm.timing_points = [TimingPoint(
        time=int(round(offset_ms)),
        beat_length=60_000.0 / max(bpm, 1e-6),
        meter=max(1, meter),
        uninherited=True,
    )]

    notes: list[TaikoNote] = []

    # Long notes first: their spans mask out any hit onsets that fall inside.
    long_spans: list[tuple[int, int]] = []
    for ch, note_type in ((CH_ROLL, "roll"), (CH_DENDEN, "denden")):
        for sf, ef in _find_regions(chart[ch], threshold):
            if ef - sf < min_long_frames:
                continue
            long_spans.append((sf, ef))
            notes.append(TaikoNote(
                time=int(round(frame_to_ms(sf))),
                note_type=note_type,
                end_time=int(round(frame_to_ms(ef))),
            ))

    def inside_long(frame: int) -> bool:
        return any(sf <= frame <= ef for sf, ef in long_spans)

    for ch, note_type in (
        (CH_DON,     "don"),
        (CH_KAT,     "kat"),
        (CH_BIG_DON, "big_don"),
        (CH_BIG_KAT, "big_kat"),
    ):
        for frame in _find_onsets(chart[ch], threshold):
            if inside_long(frame):
                continue
            notes.append(TaikoNote(
                time=int(round(frame_to_ms(frame))),
                note_type=note_type,
            ))

    notes.sort(key=lambda n: (n.time, 0 if n.is_long else 1))

    # One hittable object per frame. Taiko cannot express two simultaneous hits.
    deduped: list[TaikoNote] = []
    last_t = -10**9
    for note in notes:
        if note.time - last_t >= int(FRAME_MS):
            deduped.append(note)
            last_t = note.time

    bm.notes = deduped
    bm.compute_stats()
    return bm


def _find_onsets(ch: np.ndarray, threshold: float) -> list[int]:
    """Rising edges: frames where the channel crosses the threshold upward."""
    above = ch > threshold
    if not above.any():
        return []
    prev = np.concatenate([[False], above[:-1]])
    return np.flatnonzero(above & ~prev).tolist()


def _find_regions(ch: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Contiguous runs above the threshold, as inclusive (start, end) frames."""
    above = ch > threshold
    if not above.any():
        return []
    padded = np.concatenate([[False], above, [False]])
    edges  = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends   = np.flatnonzero(edges == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


# --------------------------------------------------------------------------- #
# Round trip
# --------------------------------------------------------------------------- #

def _count_within(query: list[int], reference: list[int], tol: float) -> int:
    """How many `query` times have a `reference` time within +/- tol."""
    if not query or not reference:
        return 0
    ref = np.asarray(reference, dtype=np.float64)
    q   = np.asarray(query, dtype=np.float64)
    idx = np.searchsorted(ref, q)
    left  = np.abs(q - ref[np.clip(idx - 1, 0, len(ref) - 1)])
    right = np.abs(q - ref[np.clip(idx,     0, len(ref) - 1)])
    return int(np.sum(np.minimum(left, right) <= tol))


def round_trip_accuracy(bm: TaikoBeatmap) -> dict:
    """
    Encode and decode a map, then measure what survived. This is the ceiling on
    everything downstream: the model can never beat the representation.
    """
    bpm, offset_ms, meter = 120.0, 0.0, DEFAULT_METER
    for tp in bm.timing_points:
        if tp.uninherited and tp.beat_length > 0:
            bpm      = 60_000.0 / tp.beat_length
            offset_ms = float(tp.time)
            meter    = max(1, tp.meter)
            break

    chart, timing = beatmap_to_tensors(bm)
    bm2 = tensor_to_beatmap(
        chart, bpm, offset_ms,
        title=bm.title, version=bm.version,
        overall_difficulty=bm.overall_difficulty, meter=meter,
    )

    orig_times  = sorted({n.time for n in bm.notes})
    recon_times = sorted({n.time for n in bm2.notes})

    # Frame quantisation moves a note by up to half a frame in each direction,
    # so matching has to be a tolerance window. One frame either way covers it.
    tol = FRAME_MS
    recovered     = _count_within(orig_times, recon_times, tol)
    matched_recon = _count_within(recon_times, orig_times, tol)

    return {
        "original_notes":      len(orig_times),
        "reconstructed_notes": len(recon_times),
        "recovered":           recovered,
        "recall":              recovered / max(len(orig_times), 1),
        "precision":           matched_recon / max(len(recon_times), 1),
        "false_positives":     len(recon_times) - matched_recon,
        "chart_shape":         chart.shape,
        "timing_shape":        timing.shape,
        "chart_kb":            chart.nbytes / 1024,
    }


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #

def save_tensors(chart: np.ndarray, timing: np.ndarray, path: str | Path) -> None:
    np.savez_compressed(str(path), chart=chart, timing=timing)


def load_tensors(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a chart/timing pair, tolerating the pre-split 7-channel format where
    channel 6 was a beat pulse train.
    """
    data = np.load(str(path))

    if "chart" in data:
        chart  = data["chart"].astype(np.float32)
        timing = (
            data["timing"].astype(np.float32)
            if "timing" in data
            else np.zeros((N_TIMING_CHANNELS, chart.shape[1]), dtype=np.float32)
        )
        return chart, timing

    legacy = (data["tensor"] if "tensor" in data else data["arr_0"]).astype(np.float32)
    return legacy[:N_CHART_CHANNELS], np.zeros(
        (N_TIMING_CHANNELS, legacy.shape[1]), dtype=np.float32
    )


def save_tensor(tensor: np.ndarray, path: str | Path) -> None:
    np.savez_compressed(str(path), chart=tensor)


def load_tensor(path: str | Path) -> np.ndarray:
    return load_tensors(path)[0]
