"""
taiko/eval/criteria.py

The osu!taiko ranking criteria, difficulty-specific part, as a checker.
Source: https://osu.ppy.sh/wiki/en/Ranking_criteria/osu!taiko (fetched
2026-10-02). A chart that hits a Kantan's star rating but plays 1/4 is not a
Kantan, and a player sees that in seconds -- round 1 of the blind A/B picked
the human chart 6/6.

Patterns are read in the map's own beat, capped at a 180 BPM beat (see
SCALE_BPM): a "1/4 pattern" is a chain of hits each 1/4 beat (or less) after
the last. Long notes break a chain. Every check reports how often it fires, and whether the wiki calls it
a rule (must) or a guideline (should); scripts/check_criteria.py measures
both on ranked maps before anything is enforced, because the wiki's numbers
are written for ~180 BPM and scale with tempo.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, replace

from taiko.data.osu_parser import TaikoNote

SLACK = 0.03          # a gap is "1/4" within 3% of a quarter beat (ms truncation)
# The criteria are written for ~180 BPM and relax below it: "double the
# density of existing guidelines" at 90 BPM, and the Easy recovery example
# goes 4 beats at 180 -> 3 at 120 -> 2 at 90 (Ranking_criteria/Scaling_BPM).
# Both are the same thing: a beat counts for no more than a 180 BPM beat.
# Measured on 1651-2078 ranked maps per level: read in the map's own beat,
# rule breaks were 1.2-3.5% at 150+ BPM and 12-19% below 150.
SCALE_BPM = 180.0
LEVELS = ("Kantan", "Futsuu", "Muzukashii", "Oni", "Inner Oni")


@dataclass(frozen=True)
class Level:
    min_gap: float                  # rule: notes at least this many beats apart
    rest: float                     # a rest moment is a gap of at least this many beats
    chain_beats: float              # guideline: a rest at least every this many beats
    spinner_gap: float              # guideline: beats between a spinner and its preceding note
    # (fraction of a beat, max notes, "rule"/"guideline") -- chains at or faster than the snap
    max_notes: tuple[tuple[float, int, str], ...] = ()
    finisher_free: float | None = None      # rule: no finisher in chains this fast or faster
    plain: tuple[tuple[float, str], ...] = ()  # (snap, kind): no colour change / finisher there


SPEC = {
    "Kantan": Level(min_gap=1 / 2, rest=3, chain_beats=36, spinner_gap=1 / 2,
                    max_notes=((1, 7, "guideline"),), plain=((1 / 2, "rule"),)),
    "Futsuu": Level(min_gap=1 / 3, rest=2, chain_beats=36, spinner_gap=1 / 2,
                    max_notes=((1 / 3, 2, "guideline"), (1 / 2, 7, "guideline")),
                    plain=((1 / 3, "rule"),)),
    "Muzukashii": Level(min_gap=1 / 6, rest=3 / 2, chain_beats=36, spinner_gap=1 / 2,
                        max_notes=((1 / 4, 5, "rule"), (1 / 6, 4, "guideline")),
                        finisher_free=1 / 4),
    "Oni": Level(min_gap=1 / 8, rest=1, chain_beats=20, spinner_gap=1 / 4,
                 max_notes=((1 / 4, 9, "guideline"), (1 / 8, 2, "guideline")),
                 finisher_free=1 / 6),
    "Inner Oni": Level(min_gap=0.0, rest=1, chain_beats=float("inf"), spinner_gap=1 / 4),
}


def level_of(version: str) -> str | None:
    """The criteria level a difficulty name asks for; None for a custom name."""
    v = version.lower()
    if re.search(r"\b(hell|ura|inner)\b", v):
        return "Inner Oni"          # the criteria stop at Inner Oni; Ura and Hell are above it
    if re.search(r"\boni\b", v):
        return "Oni"
    if "muzukashii" in v or re.search(r"\bmuzu\b", v):
        return "Muzukashii"
    if "futsuu" in v or "futsu" in v:
        return "Futsuu"
    if "kantan" in v:
        return "Kantan"
    return None


def _beat(grid, time_ms: float) -> float:
    """The beat, in ms, that patterns are measured in: the red line's, capped at 180 BPM's."""
    return min(grid.section_at(float(time_ms)).ms_per_beat, 60_000.0 / SCALE_BPM)


def _is_big(n: TaikoNote) -> bool:
    return n.note_type.startswith("big_")


def _colour(n: TaikoNote) -> str:
    return "k" if "kat" in n.note_type else "d"


def _chains(hits: list[TaikoNote], beats: list[float], snap: float) -> list[list[int]]:
    """Index runs whose every gap is at most `snap` beats (within SLACK)."""
    runs, cur = [], [0]
    for i in range(1, len(hits)):
        if beats[i] <= snap * (1 + SLACK) and beats[i] > 0:
            cur.append(i)
        else:
            if len(cur) > 1:
                runs.append(cur)
            cur = [i]
    if len(cur) > 1:
        runs.append(cur)
    return runs


def check(notes: list[TaikoNote], grid, level: str) -> Counter:
    """
    Counter of fired checks, keyed "rule: ..." or "guideline: ...". Hits in a
    chain are split at long notes, and every gap is in beats of the red line
    in force at the later note.
    """
    spec = SPEC[level]
    out: Counter = Counter()
    ordered = sorted(notes, key=lambda n: n.time)

    # Split at long notes: a roll or a spinner is not part of a hit chain.
    segments, cur = [], []
    for n in ordered:
        if n.is_long:
            if cur:
                segments.append(cur)
            cur = []
        else:
            cur.append(n)
    if cur:
        segments.append(cur)

    for hits in segments:
        beats = [0.0] + [(hits[i].time - hits[i - 1].time)
                         / _beat(grid, hits[i].time)
                         for i in range(1, len(hits))]
        for i in range(1, len(hits)):
            if 0 < beats[i] < spec.min_gap * (1 - SLACK):
                out[f"rule: notes closer than 1/{round(1 / spec.min_gap)}"] += 1

        for snap, limit, kind in spec.max_notes:
            for run in _chains(hits, beats, snap):
                if len(run) > limit:
                    out[f"{kind}: 1/{round(1 / snap)} pattern over {limit} notes"] += 1

        for snap, kind in spec.plain:
            for run in _chains(hits, beats, snap):
                notes_ = [hits[i] for i in run]
                if len({_colour(n) for n in notes_}) > 1:
                    out[f"{kind}: colour change in a 1/{round(1 / snap)} pattern"] += 1
                if any(_is_big(n) for n in notes_):
                    out[f"{kind}: finisher in a 1/{round(1 / snap)} pattern"] += 1

        if spec.finisher_free is not None:
            for run in _chains(hits, beats, spec.finisher_free):
                if any(_is_big(hits[i]) for i in run):
                    out[f"rule: finisher in a 1/{round(1 / spec.finisher_free)} or faster pattern"] += 1

        if level == "Oni":
            # A 1/4 pattern's finisher only at its end, opposite colour to the note before.
            for run in _chains(hits, beats, 1 / 4):
                for k, i in enumerate(run):
                    if _is_big(hits[i]) and (k != len(run) - 1
                                              or _colour(hits[i]) == _colour(hits[run[k - 1]])):
                        out["rule: finisher inside a 1/4 pattern, or same colour as the note before"] += 1

        if level == "Muzukashii":
            for run in _chains(hits, beats, 1 / 4):
                if len(run) > 3:
                    cols = [_colour(hits[i]) for i in run]
                    changes = [k for k in range(1, len(cols)) if cols[k] != cols[k - 1]]
                    if len(changes) > 1 or (changes and changes[0] not in (1, len(cols) - 1)):
                        out["guideline: 1/4 pattern over 3 notes with a colour change inside"] += 1

    # Rest moments, across everything: the time between rests, in beats.
    if spec.chain_beats != float("inf") and len(ordered) > 1:
        start, slow = ordered[0].time, 0
        for prev, nxt in zip(ordered, ordered[1:]):
            beat = _beat(grid, nxt.time)
            end = prev.end_time if prev.is_long else prev.time
            gap = (nxt.time - end) / beat
            # Muzukashii's second kind of rest: three consecutive 1/1 notes.
            slow = slow + 1 if gap >= 1 - SLACK else 0
            if gap >= spec.rest * (1 - SLACK) or (level == "Muzukashii" and slow >= 2):
                slow = 0
                if (end - start) / beat > spec.chain_beats * (1 + SLACK):
                    out[f"guideline: no rest ({spec.rest:g}/1) within {spec.chain_beats:g} beats"] += 1
                start = nxt.time
        if (ordered[-1].time - start) / _beat(grid, ordered[-1].time) \
                > spec.chain_beats * (1 + SLACK):
            out[f"guideline: no rest ({spec.rest:g}/1) within {spec.chain_beats:g} beats"] += 1

    for prev, n in zip(ordered, ordered[1:]):
        if n.note_type == "denden":
            gap = (n.time - (prev.end_time if prev.is_long else prev.time)) \
                / _beat(grid, n.time)
            if gap < spec.spinner_gap * (1 - SLACK):
                out[f"guideline: spinner closer than 1/{round(1 / spec.spinner_gap)} to the note before"] += 1
    return out


# --------------------------------------------------------------------------- #
# Enforcing the rules on a generated chart
# --------------------------------------------------------------------------- #

# A difficulty name a user can ask for: (criteria level, star rating, OD, HP).
# SR is the median of the ranked maps carrying that name (11106 maps); OD and
# HP the median of 250 of them -- which sit on the wiki's bounds. Ura and Hell
# Oni are above the criteria's last level, so they are checked as Inner Oni.
NAMES = {
    "Kantan":     ("Kantan", 1.36, 3.0, 8.0),
    "Futsuu":     ("Futsuu", 2.25, 4.0, 7.0),
    "Muzukashii": ("Muzukashii", 3.23, 5.0, 6.0),
    "Oni":        ("Oni", 4.19, 5.5, 5.5),
    "Inner Oni":  ("Inner Oni", 5.41, 6.0, 6.0),
    "Ura Oni":    ("Inner Oni", 5.95, 6.5, 5.5),
    "Hell Oni":   ("Inner Oni", 6.90, 7.0, 5.6),
}
STRENGTH_DIVISORS = (1, 2, 3, 4, 6, 8, 12, 16)


def level_for_sr(sr: float) -> str:
    """For a custom name: the level whose ranked SR band holds `sr`, split
    halfway between neighbouring levels' medians (1.80, 2.74, 3.71, 4.80)."""
    levels = [k for k in NAMES if NAMES[k][0] == k]          # the five criteria levels
    for lo, hi in zip(levels, levels[1:]):
        if sr < (NAMES[lo][1] + NAMES[hi][1]) / 2:
            return lo
    return levels[-1]


def _strength(n: TaikoNote, grid) -> int:
    """Metrical weight as a divisor: 1 on the beat, 2 on the half, ... lower is stronger."""
    sec = grid.section_at(float(n.time))
    return next((d for d in STRENGTH_DIVISORS if sec.distance_ms(float(n.time), d) <= 2.0), 99)


def _small(n: TaikoNote) -> TaikoNote:
    return replace(n, note_type=n.note_type.replace("big_", ""))


def enforce(notes: list[TaikoNote], grid, level: str,
            guidelines: bool = True) -> tuple[list[TaikoNote], Counter]:
    """
    Make a chart obey `level`'s rules, and its pattern guidelines unless
    `guidelines` is False, and count each fix. The rest-moment guidelines are
    never enforced: ranked maps break them 10-32% of the time, and forcing a
    rest deletes a phrase. Every other guideline here ranked maps follow
    90-99% of the time -- and after the rules were enforced, these were the
    gap left: Muzukashii 1/4 colour changes 53% against ranked 4.7%, Oni 1/8
    triples 38% against 5.6%.

    There are no probabilities left at this point, so every choice goes by
    metre: of two notes too close together, or the note that splits a run
    that is too long, the one on the weaker beat goes; a pattern that must
    be one colour takes the colour of its strongest note. ponytail: metre
    is a proxy for what the model was surest of; carrying the decoder's
    peak heights through would choose better.
    """
    spec = SPEC[level]
    fixes: Counter = Counter()
    out = sorted(notes, key=lambda n: n.time)

    def gap(a: TaikoNote, b: TaikoNote) -> float:
        return (b.time - a.time) / _beat(grid, b.time)

    # 1. Minimum gap between hits.
    if spec.min_gap > 0:
        kept: list[TaikoNote] = []
        for n in out:
            prev = kept[-1] if kept else None
            if (prev is not None and not n.is_long and not prev.is_long
                    and 0 < gap(prev, n) < spec.min_gap * (1 - SLACK)):
                fixes["dropped: too close"] += 1
                if _strength(n, grid) < _strength(prev, grid):
                    kept[-1] = n
                continue
            kept.append(n)
        out = kept

    # 2. 1/4 runs too long (Muzukashii's rule): drop the weakest note among
    # the first limit+1, which splits the run there; repeat until none is.
    for snap, limit, kind in spec.max_notes:
        if kind != "rule" and not guidelines:
            continue
        while True:
            hits = [n for n in out if not n.is_long]
            beats = [0.0] + [gap(hits[i - 1], hits[i]) for i in range(1, len(hits))]
            long_run = next((r for r in _chains(hits, beats, snap) if len(r) > limit), None)
            if long_run is None:
                break
            window = long_run[1:limit + 1]
            victim = max(window, key=lambda i: (_strength(hits[i], grid), i))
            out = [n for n in out if n is not hits[victim]]
            fixes[f"dropped: 1/{round(1 / snap)} run over {limit}"] += 1

    # 2b. A spinner needs room after the note before it: that note goes.
    if guidelines:
        kept = []
        for n in out:
            # A loop: with the note before gone, the one before that can be too close too.
            while (n.note_type == "denden" and kept and
                    (n.time - (kept[-1].end_time if kept[-1].is_long else kept[-1].time))
                    / _beat(grid, n.time) < spec.spinner_gap * (1 - SLACK)):
                kept.pop()
                fixes["dropped: too close before a spinner"] += 1
            kept.append(n)
        out = kept

    def runs(snap: float) -> tuple[list[TaikoNote], list[list[int]]]:
        hits = [n for n in out if not n.is_long]
        beats = [0.0] + [gap(hits[i - 1], hits[i]) for i in range(1, len(hits))]
        return hits, _chains(hits, beats, snap)

    swap: dict[int, TaikoNote] = {}     # id(original) -> replacement

    def put(n: TaikoNote, new: TaikoNote, why: str) -> None:
        if new.note_type != n.note_type:
            swap[id(n)] = new
            fixes[why] += 1

    # 3. Patterns that must stay plain: one colour, no finishers.
    for snap, kind in spec.plain:
        if kind != "rule":
            continue
        hits, chains = runs(snap)
        for run in chains:
            strongest = min(run, key=lambda i: (_strength(hits[i], grid), i))
            colour = "kat" if _colour(hits[strongest]) == "k" else "don"
            for i in run:
                put(hits[i], replace(hits[i], note_type=colour), "recoloured / unfinished: plain pattern")
        out = [swap.get(id(n), n) for n in out]
        swap.clear()

    # 4. Finishers in patterns too fast for them.
    if spec.finisher_free is not None:
        hits, chains = runs(spec.finisher_free)
        for run in chains:
            for i in run:
                if _is_big(hits[i]):
                    put(hits[i], _small(hits[i]), "unfinished: fast pattern")
        out = [swap.get(id(n), n) for n in out]
        swap.clear()
    if level == "Oni":
        hits, chains = runs(1 / 4)
        for run in chains:
            for k, i in enumerate(run):
                if _is_big(hits[i]) and (k != len(run) - 1
                                          or _colour(hits[i]) == _colour(hits[run[k - 1]])):
                    put(hits[i], _small(hits[i]), "unfinished: 1/4 pattern")
        out = [swap.get(id(n), n) for n in out]
        swap.clear()

    # 5. Muzukashii: a 1/4 pattern over 3 notes changes colour once at most,
    # at its start or end -- recoloured to one colour, its strongest note's.
    if level == "Muzukashii" and guidelines:
        hits, chains = runs(1 / 4)
        for run in chains:
            if len(run) <= 3:
                continue
            cols = [_colour(hits[i]) for i in run]
            changes = [k for k in range(1, len(cols)) if cols[k] != cols[k - 1]]
            if len(changes) > 1 or (changes and changes[0] not in (1, len(cols) - 1)):
                strongest = min(run, key=lambda i: (_strength(hits[i], grid), i))
                colour = _colour(hits[strongest])
                for i in run:
                    if _colour(hits[i]) != colour:
                        new = hits[i].note_type.replace("kat", "don") if colour == "d"                             else hits[i].note_type.replace("don", "kat")
                        put(hits[i], replace(hits[i], note_type=new), "recoloured: 1/4 pattern")
        out = [swap.get(id(n), n) for n in out]
    return out, fixes
