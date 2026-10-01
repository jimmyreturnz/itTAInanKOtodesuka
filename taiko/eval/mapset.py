"""
taiko/eval/mapset.py

The checks a ranked osu!taiko beatmapset is held to beyond what each
difficulty's own criteria say (those are taiko/eval/criteria.py): snapping,
timing lines, barlines, scroll speed, and the set's spread. Rules:
https://osu.ppy.sh/wiki/en/Ranking_criteria (General, osu!taiko).

Severities as in criteria.py: problem (unrankable), warning (fix it or have a
reason), minor (worth a look). `chart_findings` takes one difficulty,
`set_findings` the difficulties of one set.

Packaging (metadata, backgrounds, audio and hitsound files) is not here: it
is not something a chart generator gets right or wrong.
"""

from __future__ import annotations

import bisect
from collections import Counter
from dataclasses import dataclass, field

from taiko.data.osu_parser import TaikoBeatmap, TaikoNote, TimingPoint
from taiko.eval.criteria import LEVELS, level_of

SNAP_GRIDS = (16, 12, 9, 7, 5)       # between them, every divisor a ranked map may use
CONCURRENT_MS = 10                   # closer than this, two objects read as one
SV_NEAR_BARLINE_MS = 5               # a green line this far before a barline leaves it at the old speed
KIAI_UNSNAP_WARN_MS = 5
SCROLL_TOLERANCE = 1.0               # effective scroll (SV x BPM) that still counts as the same
LOW_SCROLL_LEVELS = ("Kantan", "Futsuu")
# Lowest difficulty a set may have, by drain (+ up to 30 s of break on the top
# difficulty): under 2:30 nothing above Futsuu, under 3:15 above Muzukashii,
# under 4:00 above Oni.
SPREAD = ((150_000, "Futsuu"), (195_000, "Muzukashii"), (240_000, "Oni"))
BREAK_LENIENCY_MS = 30_000


@dataclass
class Chart:
    version: str
    notes: list[TaikoNote]
    timing_points: list[TimingPoint]
    sr: float | None = None
    slider_multiplier: float = 1.4
    level: str | None = field(default=None)

    @classmethod
    def of(cls, bm: TaikoBeatmap, sr: float | None = None) -> "Chart":
        return cls(bm.version, list(bm.notes), list(bm.timing_points), sr, bm.slider_multiplier)

    def __post_init__(self):
        self.notes = sorted(self.notes, key=lambda n: n.time)
        self.timing_points = sorted(self.timing_points, key=lambda t: (t.exact_time, not t.uninherited))
        if self.level is None:
            self.level = level_of(self.version, self.sr)
        # Binary-search tables: a gimmick map has thousands of green lines, and
        # a walk per note over them is the whole cost of the check.
        self._line_times = [t.exact_time for t in self.timing_points]
        self._reds = [t for t in self.timing_points if t.uninherited and t.beat_length > 0]
        self._red_times = [t.exact_time for t in self._reds]
        sv, svs = 1.0, []
        for t in self.timing_points:
            sv = 1.0 if t.uninherited else (-100.0 / t.beat_length if t.beat_length < 0 else sv)
            svs.append(sv)
        self._svs = svs


def _end(n: TaikoNote) -> int:
    return n.end_time if n.is_long else n.time


def _reds(chart: Chart) -> list[TimingPoint]:
    return chart._reds


def _red_in(chart: Chart, time: float) -> TimingPoint | None:
    """The red line in force at `time`; before the first, the first extends back."""
    if not chart._reds:
        return None
    return chart._reds[max(bisect.bisect_right(chart._red_times, time) - 1, 0)]


def unsnap_ms(time: float, red: TimingPoint) -> float:
    """
    How far `time` sits from the nearest line on any ranked grid, as osu!
    stores a snapped time: the exact line position truncated to a whole ms.
    """
    best = None
    for divisor in SNAP_GRIDS:
        step = red.beat_length / divisor
        snapped = red.exact_time + round((time - red.exact_time) / step) * step
        u = time - int(snapped)
        if best is None or abs(u) < abs(best):
            best = u
    return best


def drain_ms(notes: list[TaikoNote]) -> float:
    return (_end(notes[-1]) - notes[0].time) if notes else 0.0


def barlines(chart: Chart, until: float) -> list[float]:
    """Barline times: every measure from each red line to the next, the first omitted if it says so."""
    reds = _reds(chart)
    out = []
    for i, r in enumerate(reds):
        stop = reds[i + 1].exact_time if i + 1 < len(reds) else until
        # A .osu is whole ms, so barlines under 1 ms apart are one barline as far
        # as any check can tell -- and a gimmick line's 0.001 ms beat would
        # otherwise list billions of them.
        measure = max(r.beat_length * r.meter, 1.0)
        t, k = r.exact_time, 0
        while t < stop - 1e-6:
            if not (k == 0 and r.omits_barline):
                out.append(t)
            k += 1
            t = r.exact_time + k * measure
    return out


def _sv_at(chart: Chart, time: float) -> float:
    """The SV multiplier in force: a red line resets it to 1, a green line sets it."""
    i = bisect.bisect_right(chart._line_times, time) - 1
    return chart._svs[i] if i >= 0 else 1.0


def chart_findings(chart: Chart) -> Counter:
    out: Counter = Counter()
    notes, reds = chart.notes, _reds(chart)
    if not chart.timing_points:
        out["problem: no timing lines"] += 1
        return out
    first = chart.timing_points[0]
    if not first.uninherited:
        out["problem: first timing line is inherited"] += 1
    elif first.kiai:
        out["warning: first timing line toggles kiai"] += 1

    # Snapping: every object's head, and a long note's end.
    for n in notes:
        for t in ((n.time, n.end_time) if n.is_long else (n.time,)):
            red = _red_in(chart, t)
            if red is None:
                continue
            u = abs(unsnap_ms(t, red))
            if u >= 2:
                out["problem: unsnapped object"] += 1
            elif u >= 1:
                out["minor: object 1 ms off its line"] += 1
        if reds and n.time < reds[0].exact_time:
            out["warning: object before the first red line"] += 1

    # Concurrent: on top of each other, or too close to read as two.
    for a, b in zip(notes, notes[1:]):
        apart = b.time - _end(a)
        if apart <= 0:
            out["problem: concurrent objects"] += 1
        elif apart <= CONCURRENT_MS and b.note_type != "denden":
            out["warning: almost concurrent objects"] += 1

    if notes and drain_ms(notes) < 30_000:
        out["problem: drain time under 30 s"] += 1

    # Timing lines on one millisecond.
    for a, b in zip(chart.timing_points, chart.timing_points[1:]):
        if abs(a.exact_time - b.exact_time) < 1e-6 and a.uninherited == b.uninherited:
            out["problem: two timing lines of one kind on one ms"] += 1

    # An object 0-5 ms before a line that changes its scroll speed: it moves at the wrong one.
    lines = chart.timing_points
    for n in notes:
        j = bisect.bisect_right(chart._line_times, n.time)
        nxt = lines[j] if j < len(lines) else None
        if nxt is None or nxt.exact_time - n.time > 5:
            continue
        red = _red_in(chart, n.time)
        red_next = _red_in(chart, nxt.exact_time)
        if red is None or red_next is None or abs(unsnap_ms(n.time, red)) > 1:
            continue
        before = _sv_at(chart, n.time) * 60_000 / red.beat_length
        after = _sv_at(chart, nxt.exact_time) * 60_000 / red_next.beat_length
        if abs(before - after) > 1:
            out["warning: object just before a scroll speed change"] += 1

    if notes:
        end = _end(notes[-1])
        bars = barlines(chart, end + 60_000)
        nxt = next((b for b in bars if b >= end), None)
        if nxt is not None and -2.0 < end - nxt <= -1.0:
            out["problem: last note 1 ms before a barline, hiding it" if not notes[-1].is_long
                else "minor: last long note ends 1 ms before a barline"] += 1

        # A red line cutting the previous section's last measure short.
        for a, b in zip(reds, reds[1:]):
            measure = a.beat_length * a.meter
            distance = b.exact_time - a.exact_time
            if b.omits_barline or (a.omits_barline and distance <= measure):
                continue
            rest = distance % measure
            if measure - rest <= 2 or rest <= 1e-9:
                continue
            out["problem: barline close to the next red line" if 0.5 <= rest <= measure / 2
                else "warning: barline close to the next red line"] += 1

        # An SV change a few ms before a barline drags the barline to the new
        # speed. Barlines are whole ms in osu!stable, like a snapped note; a
        # green line that changes only volume or kiai moves nothing. Measured
        # on 1389 ranked maps: exact barlines and every green line flagged
        # 85.4% of them, truncated barlines 41.9%.
        whole_bars = sorted({int(b) for b in bars})
        for i, t in enumerate(chart.timing_points):
            if t.uninherited or (i > 0 and abs(chart._svs[i] - chart._svs[i - 1]) < 1e-9):
                continue
            j = bisect.bisect_left(whole_bars, t.exact_time)
            if j < len(whole_bars) and -SV_NEAR_BARLINE_MS <= t.exact_time - whole_bars[j] < 0:
                out["warning: green line just before a barline"] += 1

    # Kiai edges off the grid.
    for i, t in enumerate(chart.timing_points):
        if not t.kiai or (i > 0 and chart.timing_points[i - 1].kiai):
            continue
        red = _red_in(chart, t.exact_time)
        if red is None:
            continue
        u = abs(unsnap_ms(t.exact_time, red))
        if u >= KIAI_UNSNAP_WARN_MS:
            out["warning: kiai starts off the grid"] += 1
        elif u >= 1:
            out["minor: kiai starts 1 ms off the grid"] += 1

    # Kantan and Futsuu keep one scroll speed.
    if chart.level in LOW_SCROLL_LEVELS and notes:
        end = _end(notes[-1])
        offsets = sorted({t.exact_time for t in chart.timing_points}) + [end]
        weight: Counter = Counter()
        segments = []
        note_times = [n.time for n in notes]
        green_times = {t.exact_time for t in chart.timing_points if not t.uninherited}
        for a, b in zip(offsets, offsets[1:]):
            k = bisect.bisect_left(note_times, a)
            if b <= a or k >= len(note_times) or note_times[k] >= b:
                continue
            red = _red_in(chart, a)
            speed = round(_sv_at(chart, a) * 60_000 / red.beat_length, 2)
            weight[speed] += b - a
            green = a in green_times
            segments.append((speed, green))
        if weight:
            dominant = weight.most_common(1)[0][0]
            for speed, green in segments:
                if green and abs(speed - dominant) > SCROLL_TOLERANCE:
                    out["warning: scroll speed change in a low difficulty"] += 1
    return out


def set_findings(charts: list[Chart]) -> Counter:
    out: Counter = Counter()
    charts = [c for c in charts if c.notes]
    if not charts:
        return out
    ordered = sorted(charts, key=lambda c: (LEVELS.index(c.level) if c.level in LEVELS else 99,
                                            c.sr or 0))

    # Spread: how hard the lowest difficulty may be, for the drain.
    lowest = ordered[0]
    for c in ordered:
        effective = drain_ms(c.notes)
        if effective >= BREAK_LENIENCY_MS:
            played = _end(c.notes[-1]) - c.notes[0].time
            breaks = max(played - effective, 0)
            effective += min(breaks, BREAK_LENIENCY_MS) if c is ordered[-1] else breaks
        for limit, allowed in SPREAD:
            if effective < limit and lowest.level in LEVELS and \
                    LEVELS.index(lowest.level) > LEVELS.index(allowed):
                out[f"problem: lowest difficulty above {allowed} for its drain time"] += 1
                break

    # Base SV moves one way across the spread.
    if len(ordered) >= 3:
        values = [round(c.slider_multiplier, 2) for c in ordered]
        up = sum(b < a for a, b in zip(values, values[1:]))
        down = sum(b > a for a, b in zip(values, values[1:]))
        ascending = up <= down
        out["warning: base SV goes against the spread"] += sum(
            (b < a) if ascending else (b > a) for a, b in zip(values, values[1:]))
        if not out["warning: base SV goes against the spread"]:
            del out["warning: base SV goes against the spread"]

    # The same red line omits its barline in every difficulty, or in none.
    ref = ordered[0]
    for c in ordered[1:]:
        for r in _reds(ref):
            other = next((o for o in _reds(c) if abs(o.exact_time - r.exact_time) < 1.0), None)
            if other is not None and other.omits_barline != r.omits_barline:
                out["problem: barlines differ between difficulties"] += 1

    # Kiai sections: the same in every difficulty.
    def kiai_starts(c: Chart) -> tuple:
        return tuple(round(t.exact_time) for i, t in enumerate(c.timing_points)
                     if t.kiai and not (i > 0 and c.timing_points[i - 1].kiai))
    if len({kiai_starts(c) for c in ordered}) > 1:
        out["minor: kiai differs between difficulties"] += 1
    return out


def fix_chart(notes: list[TaikoNote], timing_points: list[TimingPoint]) -> tuple[list[TaikoNote], Counter]:
    """
    Fix what `chart_findings` flags that a generator, not the timing, is to
    blame for: an object 1-5 ms before a line that changes its scroll speed
    moves onto the line, which is where a mapper puts a note and its green
    line -- truncated to whole ms, a grid note at 1234 sat under a green line
    at 1235 in 12.8% of the model's charts against 3.3% of ranked ones. It
    stays within 1 ms of its snap.
    """
    chart = Chart("", notes, timing_points, level="Inner Oni")
    lines, fixes, out = chart.timing_points, Counter(), []
    for n in chart.notes:
        j = bisect.bisect_right(chart._line_times, n.time)
        nxt = lines[j] if j < len(lines) else None
        if nxt is not None and 0 < nxt.exact_time - n.time <= 5:
            red, red_next = _red_in(chart, n.time), _red_in(chart, nxt.exact_time)
            before = _sv_at(chart, n.time) * 60_000 / red.beat_length
            after = _sv_at(chart, nxt.exact_time) * 60_000 / red_next.beat_length
            if abs(before - after) > 1 and abs(unsnap_ms(int(nxt.exact_time), red)) <= 1:
                shift = int(nxt.exact_time) - n.time
                n = TaikoNote(time=n.time + shift, note_type=n.note_type,
                              end_time=n.end_time + shift if n.is_long else n.time + shift)
                fixes["moved onto the scroll speed change"] += 1
        out.append(n)
    return out, fixes
