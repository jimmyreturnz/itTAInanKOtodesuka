"""
taiko/data/grid.py

A beat grid built from every red line, so a song with BPM changes snaps and
measures each section against its own tempo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from taiko.data.osu_parser import TimingPoint
from taiko.data.tensor_repr import red_lines


@dataclass(frozen=True)
class Section:
    offset_ms: float
    ms_per_beat: float
    meter: int = 4

    @property
    def bpm(self) -> float:
        return 60_000.0 / self.ms_per_beat

    def distance_ms(self, time_ms: float, divisor: int) -> float:
        gap = self.ms_per_beat / divisor
        rel = (time_ms - self.offset_ms) / gap
        return abs(rel - round(rel)) * gap

    def nearest(self, time_ms: float, divisor: int) -> float:
        gap = self.ms_per_beat / divisor
        return self.offset_ms + round((time_ms - self.offset_ms) / gap) * gap


class Grid:
    """Red lines in time order. Before the first one, the first one extends back."""

    def __init__(self, timing_points: Sequence[TimingPoint]):
        reds = red_lines(list(timing_points))
        if not reds:
            raise ValueError("a grid needs at least one red line")
        self.sections = [Section(float(tp.time), float(tp.beat_length), max(1, tp.meter))
                         for tp in reds]
        self._starts = np.asarray([s.offset_ms for s in self.sections])

    @classmethod
    def single(cls, bpm: float, offset_ms: float, meter: int = 4) -> "Grid":
        return cls([TimingPoint(time=int(round(offset_ms)), beat_length=60_000.0 / bpm,
                                meter=meter, uninherited=True)])

    def section_at(self, time_ms: float) -> Section:
        i = int(np.searchsorted(self._starts, time_ms, side="right")) - 1
        return self.sections[max(i, 0)]

    def snap(self, time_ms: float, divisors: Sequence[int], epsilon_ms: float) -> Optional[float]:
        """First divisor (in the order given) with a line within epsilon; None if none."""
        sec = self.section_at(time_ms)
        for div in divisors:
            if sec.distance_ms(time_ms, div) < epsilon_ms:
                return sec.nearest(time_ms, div)
        return None

    def distance_ms(self, time_ms: float, divisors: Sequence[int]) -> float:
        sec = self.section_at(time_ms)
        return min(sec.distance_ms(time_ms, d) for d in divisors)

    def distances_ms(self, times_ms, divisors: Sequence[int]) -> np.ndarray:
        return np.asarray([self.distance_ms(float(t), divisors) for t in times_ms])

    def timing_points(self) -> list[TimingPoint]:
        return [TimingPoint(time=int(round(s.offset_ms)), beat_length=s.ms_per_beat,
                            meter=s.meter, uninherited=True) for s in self.sections]
