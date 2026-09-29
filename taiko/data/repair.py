"""
taiko/data/repair.py

Deterministic fixes for charts a human could not play.

One rule per unplayability type that taiko/eval/metrics.py counts:

    too_fast           hits closer than MIN_HIT_GAP_MS: keep the one on the
                       coarser subdivision (the likelier real note), drop the other
    big_note_streams   a big note with a neighbour closer than MIN_BIG_GAP_MS
                       becomes the small note of the same colour
    overlapping_longs  a long that runs into the next long is cut short before
                       it; hits inside a long are removed
    zero_length_longs  removed

Every fix is counted and reported. A chart that needs a lot of repair is a
sign the model has more to learn, and the counts are what show that; repair
exists so each generated map is playable, not to hide the problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from taiko.data.frames import FRAME_MS
from taiko.data.grid import Grid
from taiko.data.osu_parser import TaikoBeatmap, TaikoNote
from taiko.eval.metrics import BIG_TYPES, LONG_TYPES, MIN_BIG_GAP_MS, MIN_HIT_GAP_MS

SMALL_OF = {"big_don": "don", "big_kat": "kat"}
_COARSE = (1, 2, 4, 3, 6, 8, 12)


@dataclass
class RepairReport:
    dropped_too_fast: int = 0
    shrunk_big_notes: int = 0
    trimmed_longs: int = 0
    dropped_in_longs: int = 0
    dropped_zero_longs: int = 0
    notes_before: int = 0

    @property
    def total(self) -> int:
        return (self.dropped_too_fast + self.shrunk_big_notes + self.trimmed_longs
                + self.dropped_in_longs + self.dropped_zero_longs)

    @property
    def rate(self) -> float:
        return self.total / max(self.notes_before, 1)

    def summary(self) -> str:
        return (f"{self.total} fixes on {self.notes_before} notes ({self.rate:.1%}): "
                f"{self.dropped_too_fast} too-fast dropped, {self.shrunk_big_notes} big->small, "
                f"{self.trimmed_longs} longs trimmed, {self.dropped_in_longs} hits inside longs, "
                f"{self.dropped_zero_longs} zero-length longs")


def _coarseness(grid: Grid | None, t: int) -> int:
    """Index of the coarsest subdivision `t` sits on (lower = coarser)."""
    if grid is None:
        return 0
    sec = grid.section_at(t)
    for i, d in enumerate(_COARSE):
        if sec.distance_ms(t, d) <= 2.0:
            return i
    return len(_COARSE)


def repair(bm: TaikoBeatmap, grid: Grid | None = None) -> RepairReport:
    """Fix `bm` in place and report what changed."""
    rep = RepairReport(notes_before=len(bm.notes))
    notes = sorted(bm.notes, key=lambda n: n.time)

    longs = []
    for n in (n for n in notes if n.note_type in LONG_TYPES):
        if n.duration <= 0:
            rep.dropped_zero_longs += 1
            continue
        longs.append(n)
    for a, b in zip(longs, longs[1:]):
        if a.end_time >= b.time:
            a.end_time = max(a.time + int(FRAME_MS), b.time - int(FRAME_MS))
            rep.trimmed_longs += 1
    longs = [n for n in longs if n.end_time > n.time]

    hits = []
    for n in (n for n in notes if n.note_type not in LONG_TYPES):
        if any(l.time <= n.time <= l.end_time for l in longs):
            rep.dropped_in_longs += 1
            continue
        hits.append(n)

    kept: list[TaikoNote] = []
    for n in hits:
        if kept and n.time - kept[-1].time < MIN_HIT_GAP_MS:
            rep.dropped_too_fast += 1
            if _coarseness(grid, n.time) < _coarseness(grid, kept[-1].time):
                kept[-1] = n
            continue
        kept.append(n)

    for i, n in enumerate(kept):
        if n.note_type not in BIG_TYPES:
            continue
        near = ((i > 0 and n.time - kept[i - 1].time < MIN_BIG_GAP_MS)
                or (i + 1 < len(kept) and kept[i + 1].time - n.time < MIN_BIG_GAP_MS))
        if near:
            n.note_type = SMALL_OF[n.note_type]
            rep.shrunk_big_notes += 1

    bm.notes = sorted(longs + kept, key=lambda n: (n.time, 0 if n.is_long else 1))
    bm.compute_stats()
    return rep
