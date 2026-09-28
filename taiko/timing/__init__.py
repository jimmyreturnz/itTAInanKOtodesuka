"""
taiko/timing -- super timing: red lines from audio, BPM changes included.

    from taiko.timing import detect_timing
    result = detect_timing("song.mp3")
    result.timing_points      # osu! red lines, ready for generation or export

Pipeline: beat evidence (activations.py) -> a beat for every beat of the song
(tracker.py) -> constant-tempo sections, fitted, human-rounded, merged and
polished against the beats' averaged attack (sections.py) -> red lines on downbeats.

Timing is still worth checking by ear. What this removes is the work of
finding the BPMs and offsets; a section that needs a nudge is a one-line edit
in the editor, and --timing-from takes the corrected map back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from taiko.data.osu_parser import TimingPoint
from taiko.timing.activations import (
    Activations, beat_this_activations, load_mono,
    onset_activations, timingnet_activations,
)
from taiko.timing.sections import (
    FittedSection, absorb_short_sections, choose_downbeats, drop_ragged_edges,
    fit_sections, merge_sections, polish_offsets, red_line_times, refine_boundaries,
    segment_beats,
)
from taiko.timing.tracker import BPM_RANGE, PRIOR_BPM, beat_times_ms

DEFAULT_TIMINGNET = Path("checkpoints/timing/best.pt")

# Constant offset between our decoder's clock and osu!'s, in ms, added to every
# red line. Measured by scripts/benchmark_timing.py against ranked maps (it
# prints the recommended value as the median signed offset error). Zero until
# measured -- a guess here would be worse than none.
#
# One data point so far: USAO - SUPERNOVA timed by ear in the osu! editor
# (global and local offset 0) at 394 ms, where the beats' attacks begin at
# 408 ms in our decode (libsndfile/mpg123, gapless trimming). That suggests
# about -14 ms, but one song timed by ear is not a calibration.
DECODER_BIAS_MS = 0.0


@dataclass
class TimingResult:
    timing_points: list[TimingPoint]
    method: str
    sections: list[FittedSection]
    beats_ms: np.ndarray
    global_bpm: float
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        lines = [f"method: {self.method}   beats tracked: {len(self.beats_ms)}   "
                 f"global tempo estimate: {self.global_bpm:.2f} BPM"]
        for tp, sec in zip(self.timing_points, self.sections):
            lines.append(
                f"  {tp.time:>8d} ms  {60_000.0 / tp.beat_length:9.3f} BPM  "
                f"{tp.meter}/4   {sec.n_beats:4d} beats  fit rms {sec.rms_ms:4.1f} ms  "
                f"max {sec.max_ms:4.1f} ms"
            )
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def get_activations(y: np.ndarray, backend: str = "auto", timingnet: str | Path | None = None,
                    passes: int = 8, device: str = "cpu", verbose: bool = False) -> Activations:
    tried = []
    if backend in ("auto", "timingnet"):
        ckpt = Path(timingnet) if timingnet else DEFAULT_TIMINGNET
        if ckpt.exists():
            return timingnet_activations(y, ckpt, passes=passes, device=device)
        tried.append(f"timingnet: no checkpoint at {ckpt}")
        if backend == "timingnet":
            raise FileNotFoundError(tried[-1])
    if backend in ("auto", "beat_this"):
        try:
            return beat_this_activations(y, passes=passes, device=device)
        except Exception as exc:                                  # noqa: BLE001
            tried.append(f"beat_this: {type(exc).__name__}: {str(exc)[:120]}")
            if backend == "beat_this":
                raise
    if verbose and tried:
        for t in tried:
            print(f"  [timing] skipped {t}")
    return onset_activations(y)


def _evidence_at(act: np.ndarray | None, fps: float, times_ms: np.ndarray) -> np.ndarray | None:
    if act is None:
        return None
    frames = np.round(times_ms * fps / 1000.0).astype(np.int64)
    out = np.empty(len(frames))
    for i, f in enumerate(frames):
        a, b = max(0, f - 2), min(len(act), f + 3)
        out[i] = act[a:b].max() if b > a else 0.0
    return out


def detect_timing(
    audio: str | Path | np.ndarray,
    backend: str = "auto",
    timingnet: str | Path | None = None,
    passes: int = 8,
    bpm_range: tuple[float, float] = BPM_RANGE,
    prior_bpm: float | None = PRIOR_BPM,
    leniency_ms: float = 12.0,
    polish: bool = True,
    meter: int = 4,
    bias_ms: float = DECODER_BIAS_MS,
    device: str = "cpu",
    verbose: bool = False,
) -> TimingResult:
    """
    Red lines for a song.

    Args:
        audio:      a path, or a mono waveform at 22050 Hz
        backend:    "auto" (timingnet if a checkpoint exists, else beat_this if
                    it loads, else onset), or one of those by name
        bpm_range:  allowed BPMs. Narrow it to settle a half/double-time
                    ambiguity you already know the answer to.
        prior_bpm:  where the half/double-time choice leans. None for no lean.
        leniency_ms: how far any beat may sit off a rounded BPM's grid before a
                    rounder BPM is rejected.
    """
    y = load_mono(audio) if not isinstance(audio, np.ndarray) else audio
    acts = get_activations(y, backend, timingnet, passes, device, verbose)
    if verbose:
        print(f"  [timing] activations from {acts.source} at {acts.fps:.0f} fps")

    beats, gbpm = beat_times_ms(acts.beat, acts.fps, bpm_range, prior_bpm)
    notes: list[str] = []
    if len(beats) < 8:
        raise ValueError(f"only {len(beats)} beats found; is there music in this file?")

    runs = segment_beats(beats)
    sections = fit_sections(beats, runs, leniency_ms)
    sections = merge_sections(sections, beats, leniency_ms)
    sections = absorb_short_sections(sections, beats, leniency_ms=leniency_ms)
    sections = merge_sections(sections, beats, leniency_ms)
    sections = drop_ragged_edges(sections)

    if polish:
        shifts = polish_offsets(sections, y, 22_050)
        moved = [s for s in shifts if s != 0.0]
        if moved:
            notes.append(f"onset polish moved {len(moved)} section(s) by "
                         f"{', '.join(f'{s:+.1f}' for s in moved[:6])} ms")

    refine_boundaries(sections, acts.beat, acts.fps)
    choose_downbeats(sections, _evidence_at(acts.downbeat, acts.fps, beats), meter)
    if acts.source == "onset":
        notes.append("downbeats from kick-drum onsets only; check the barlines by ear")

    times = red_line_times(sections, beats)
    points = [
        TimingPoint(time=int(round(t + bias_ms)), beat_length=sec.ms_per_beat,
                    meter=sec.meter, uninherited=True)
        for t, sec in zip(times, sections)
    ]
    return TimingResult(points, acts.source, sections, beats, gbpm, notes)
