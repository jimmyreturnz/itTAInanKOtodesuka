"""
taiko/eval/criteria.py

The osu!taiko ranking criteria for a single difficulty, as checks a ranked
beatmapset is held to, and a pass that makes a generated chart meet them.
Rules: https://osu.ppy.sh/wiki/en/Ranking_criteria/osu!taiko and
https://osu.ppy.sh/wiki/en/Ranking_criteria/Scaling_BPM.

A chart that hits a Kantan's star rating but plays 1/4 is not a Kantan, and
a player sees it in seconds: round 1 of the blind A/B picked the human chart
6/6, and the model broke a rule in 37-61% of its charts.

Every finding has a severity, the way a ranking review weighs it:

  problem   unrankable as it stands
  warning   should be fixed, or have a reason
  minor     at the limit; worth a look

Snaps are read against a *folded* beat (`folded_beat_ms`): the red line's
tempo doubled or halved into the band the criteria are written for, so a
90 BPM 1/4 counts as the 180 BPM 1/2 it plays like.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, replace

from taiko.data.osu_parser import TaikoNote

EPS_MS = 1.0                     # timestamps are whole ms; equal gaps can differ by one

# Levels, easiest first. Expert and Ultra are Inner Oni and Hell Oni.
LEVELS = ("Kantan", "Futsuu", "Muzukashii", "Oni", "Inner Oni", "Hell Oni")

# A difficulty name a user can ask for: (level, star rating, OD, HP). SR is the
# median of the ranked maps carrying that name (11106 maps); OD and HP the
# median of 250 of them, which sit on the criteria's recommended values.
NAMES = {
    "Kantan":     ("Kantan", 1.36, 3.0, 8.0),
    "Futsuu":     ("Futsuu", 2.25, 4.0, 7.0),
    "Muzukashii": ("Muzukashii", 3.23, 5.0, 6.0),
    "Oni":        ("Oni", 4.19, 5.5, 5.5),
    "Inner Oni":  ("Inner Oni", 5.41, 6.0, 6.0),
    "Ura Oni":    ("Inner Oni", 5.95, 6.5, 5.5),
    "Hell Oni":   ("Hell Oni", 6.90, 7.0, 5.6),
}
_BY_NAME = {"kantan": "Kantan", "futsuu": "Futsuu", "muzukashii": "Muzukashii", "oni": "Oni",
            "inner oni": "Inner Oni", "ura oni": "Inner Oni", "hell oni": "Hell Oni"}
# A name that is no level is placed by star rating.
SR_EDGES = ((2.0, "Kantan"), (2.7, "Futsuu"), (4.0, "Muzukashii"), (5.3, "Oni"), (6.5, "Inner Oni"))


def level_for_sr(sr: float) -> str:
    return next((lvl for edge, lvl in SR_EDGES if sr < edge), "Hell Oni")


def level_of(version: str, sr: float | None = None) -> str | None:
    """
    The level a difficulty is held to. Only an exact level name counts, once
    an owner ("Kayoko's ") and "collab" are stripped: "Ono's Taiko Muzukashii"
    is a custom name. A custom name goes by `sr`, or is None without one. An
    "Oni" whose SR is Inner Oni's or above is held to that.
    """
    name = re.sub(r"^\s*\w+'s\s+", "", version.lower())
    name = re.sub(r"\bcollab\b", "", name)
    name = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", name)).strip()
    level = _BY_NAME.get(name)
    if level is None:
        return level_for_sr(sr) if sr is not None else None
    if level == "Oni" and sr is not None and LEVELS.index(level_for_sr(sr)) > LEVELS.index("Oni"):
        return level_for_sr(sr)
    return level


def folded_beat_ms(beat_ms: float) -> float:
    """The red line's beat folded into 130-270 BPM: x2 below 110, x1.5 at 110-130, /2 above 270."""
    while beat_ms <= 60_000 / 270:
        beat_ms *= 2
    while beat_ms >= 60_000 / 110:
        beat_ms /= 2
    while beat_ms >= 60_000 / 130:
        beat_ms /= 1.5
    return beat_ms


@dataclass(frozen=True)
class Spec:
    min_gap: float | None = None                         # warning: hits closer than this
    patterns: tuple[tuple[float, int], ...] = ()         # (snap, max notes): over is a warning, at is minor
    plain: float | None = None                           # warning: colour change in a pattern this fast
    finisher: float | None = None                        # problem: finisher in a pattern this fast
    finisher_mid: float | None = None                    # Oni: at this speed, only last and colour-changed
    finisher_warn: float | None = None                   # Muzukashii: warning at this speed
    finisher_warn_mono: float | None = None              # Inner/Hell: warning, same colour or not last
    rest: tuple[tuple[int, float], ...] = ()             # (consecutive gaps, beats each) that make a rest
    chain: tuple[float, float] | None = None             # (minor, warning) beats without a rest
    spinner_gap: float | None = None                     # minor: beats from the note before a spinner
    od: float | None = None
    hp: tuple[float, float, float, float] | None = None  # by drain: <=1:00, <3:45, <4:45, longer
    # Measured on ranked maps rather than written in the criteria (blind A/B
    # round 3 caught both by eye):
    own_beat_run: int | None = None                      # warning: 1/1 run, in the song's own beat
    straight: bool = False                               # warning: a note on a triplet-only line


SPEC = {
    # own_beat_run: the fold reads a 128 BPM beat as 192 BPM's, so a Kantan's
    # 1/1 never counted as one; in the song's own beat 2.3% of 1860 ranked
    # Kantans run past 7 (Futsuu 31.8%, Muzukashii 27.6%: theirs is normal).
    # straight: 96.0% of ranked Kantans and 88.2% of Futsuus put no note on a
    # triplet-only line; the rest are swing songs.
    "Kantan": Spec(min_gap=1 / 2, patterns=((1, 7), (1 / 2, 2)), plain=1 / 2, finisher=1 / 2,
                   rest=((1, 3.0),), chain=(36, 44), spinner_gap=1 / 2, od=3, hp=(9, 8, 7, 6),
                   own_beat_run=7, straight=True),
    "Futsuu": Spec(min_gap=1 / 3, patterns=((1 / 2, 7), (1 / 3, 2)), plain=1 / 3, finisher=1 / 3,
                   rest=((1, 2.0),), chain=(36, 44), spinner_gap=1 / 2, od=4, hp=(8, 7, 6, 5),
                   straight=True),
    "Muzukashii": Spec(min_gap=1 / 6, patterns=((1 / 4, 5), (1 / 6, 4)), finisher=1 / 4,
                       finisher_warn=1 / 3, rest=((1, 1.5), (3, 1.0)), chain=(36, 44),
                       spinner_gap=1 / 2, od=5, hp=(7, 6, 5, 4)),
    "Oni": Spec(min_gap=1 / 6, patterns=((1 / 4, 9), (1 / 6, 4), (1 / 8, 2)), finisher=1 / 6,
                finisher_mid=1 / 4, rest=((1, 1.0),), chain=(20, 32), spinner_gap=1 / 4,
                od=5.5, hp=(7, 5.5, 5, 4)),
    "Inner Oni": Spec(finisher_warn_mono=1 / 3, spinner_gap=1 / 4),
    "Hell Oni": Spec(finisher_warn_mono=1 / 3),
}


def _beat(grid, time_ms: float) -> float:
    return folded_beat_ms(grid.section_at(float(time_ms)).ms_per_beat)


def _is_big(n: TaikoNote) -> bool:
    return n.note_type.startswith("big_")


def _colour(n: TaikoNote) -> str:
    return "k" if "kat" in n.note_type else "d"


def _end(n: TaikoNote) -> int:
    return n.end_time if n.is_long else n.time


def _own_beat(grid, time_ms: float) -> float:
    return grid.section_at(float(time_ms)).ms_per_beat


def _triplet_only(n: TaikoNote, grid) -> bool:
    """On a 1/3, 1/6 or 1/12 line and on no 1/16 line."""
    sec = grid.section_at(float(n.time))
    return sec.distance_ms(float(n.time), 12) <= 2 and sec.distance_ms(float(n.time), 16) > 2


def _chains(hits: list[TaikoNote], grid, snap: float, beat=None) -> list[list[int]]:
    """Runs of hits whose every gap is at most `snap` beats (folded, unless `beat` says otherwise)."""
    beat = beat or _beat
    runs, cur = [], [0] if hits else []
    for i in range(1, len(hits)):
        if hits[i].time - hits[i - 1].time <= snap * beat(grid, hits[i - 1].time) + EPS_MS:
            cur.append(i)
        else:
            if len(cur) > 1:
                runs.append(cur)
            cur = [i]
    if len(cur) > 1:
        runs.append(cur)
    return runs


def _pattern_place(objs: list[TaikoNote], k: int) -> tuple[bool, bool, float]:
    """
    (first in its pattern, last in its pattern, spacing ms) for the hit at
    objs[k]. A pattern is consecutive hits: a hit opens one when the gap
    after it is shorter than the gap before, closes one when the gap before
    is shorter, and sits in the middle when they are equal. Its spacing is
    the gap that ties it to its pattern, the shorter one in the middle. A hit
    with no hit on either side is in no pattern.
    """
    prev = objs[k - 1] if k > 0 and not objs[k - 1].is_long else None
    nxt = objs[k + 1] if k + 1 < len(objs) and not objs[k + 1].is_long else None
    n = objs[k]
    if prev is None and nxt is None:
        return True, True, 0.0
    if prev is None:
        return True, False, nxt.time - n.time
    if nxt is None:
        return False, True, n.time - prev.time
    before, after = n.time - prev.time, nxt.time - n.time
    if after < before:
        return True, False, after
    if before < after:
        return False, True, before
    return False, False, before


def _finisher_finding(objs: list[TaikoNote], k: int, grid, spec: Spec) -> str | None:
    n = objs[k]
    first, last, spacing = _pattern_place(objs, k)
    in_pattern = not (first and last)
    beat = _beat(grid, n.time)
    prev = objs[k - 1] if k > 0 else None
    nxt = objs[k + 1] if k + 1 < len(objs) else None
    mono_before = prev is not None and not prev.is_long and _colour(prev) == _colour(n)
    mono_after = nxt is not None and not nxt.is_long and _colour(nxt) == _colour(n)

    def within(snap):
        return snap is not None and in_pattern and spacing <= snap * beat + EPS_MS

    if within(spec.finisher) or (within(spec.finisher_mid) and ((mono_before and not first) or not last)):
        return "problem: finisher in a pattern too fast for it"
    if within(spec.finisher_warn) or (within(spec.finisher_warn_mono) and
                                      ((mono_before and not first) or (mono_after and not last) or not last)):
        return "warning: finisher inside a fast pattern"
    return None


def check(notes: list[TaikoNote], grid, level: str, drain_ms: float | None = None,
          od: float | None = None, hp: float | None = None) -> Counter:
    """Findings for a chart held to `level`, keyed "problem: ...", "warning: ..." or "minor: ..."."""
    spec = SPEC[level]
    out: Counter = Counter()
    objs = sorted(notes, key=lambda n: n.time)
    hits = [n for n in objs if not n.is_long]

    if spec.min_gap:
        for a, b in zip(hits, hits[1:]):
            if b.time - a.time < spec.min_gap * _beat(grid, a.time) - EPS_MS:
                out[f"warning: notes closer than 1/{round(1 / spec.min_gap)}"] += 1

    for snap, limit in spec.patterns:
        for run in _chains(hits, grid, snap):
            if len(run) > limit:
                out[f"warning: 1/{round(1 / snap)} pattern over {limit} notes"] += 1
            elif len(run) == limit:
                out[f"minor: 1/{round(1 / snap)} pattern of {limit} notes"] += 1

    if spec.own_beat_run:
        for run in _chains(hits, grid, 1, beat=_own_beat):
            if len(run) > spec.own_beat_run:
                out[f"warning: 1/1 run over {spec.own_beat_run} notes in the song's own beat"] += 1
    if spec.straight:
        n_triplet = sum(_triplet_only(n, grid) for n in hits)
        if n_triplet:
            out["warning: triplet-only note in a straight low difficulty"] += n_triplet

    if spec.plain:
        for run in _chains(hits, grid, spec.plain):
            if len({_colour(hits[i]) for i in run}) > 1:
                out[f"warning: colour change in a 1/{round(1 / spec.plain)} pattern"] += 1

    for k, n in enumerate(objs):
        if not n.is_long and _is_big(n):
            finding = _finisher_finding(objs, k, grid, spec)
            if finding:
                out[finding] += 1

    if spec.rest and spec.chain and len(objs) > 1:
        gaps = [objs[i + 1].time - _end(objs[i]) for i in range(len(objs) - 1)]
        start, within_chain = objs[0].time, False
        for i, n in enumerate(objs):
            beat = _beat(grid, n.time)
            opens = i == 0 or any(i - c >= 0 and min(gaps[i - c:i]) + EPS_MS >= b * beat
                                  for c, b in spec.rest)
            closes = i == len(objs) - 1 or any(i + c <= len(gaps) and min(gaps[i:i + c]) + EPS_MS >= b * beat
                                               for c, b in spec.rest)
            if opens:
                within_chain, start = True, n.time
            if closes and within_chain:
                within_chain = False
                beats = (_end(n) - start + EPS_MS) // grid.section_at(float(n.time)).ms_per_beat
                if beats > spec.chain[1]:
                    out["warning: too long without a rest"] += 1
                elif beats > spec.chain[0]:
                    out["minor: long without a rest"] += 1

    if spec.spinner_gap:
        for a, b in zip(objs, objs[1:]):
            if b.note_type == "denden" and b.time - _end(a) < spec.spinner_gap * _beat(grid, a.time):
                out[f"minor: spinner closer than 1/{round(1 / spec.spinner_gap)} to the note before"] += 1

    if spec.od is not None and od is not None and od != spec.od:
        out[f"{'warning' if abs(od - spec.od) > 0.5 else 'minor'}: OD off the recommended value"] += 1
    want = recommended_hp(level, drain_ms) if drain_ms is not None else None
    if want is not None and hp is not None and hp != want:
        out[f"{'warning' if abs(hp - want) > 1 else 'minor'}: HP off the recommended value"] += 1
    return out


def recommended_hp(level: str, drain_ms: float) -> float | None:
    hp = SPEC[level].hp
    if hp is None:
        return None
    return hp[0 if drain_ms <= 60_000 else 3 if drain_ms >= 285_000 else 2 if drain_ms >= 225_000 else 1]


# --------------------------------------------------------------------------- #
# Making a generated chart meet them
# --------------------------------------------------------------------------- #

STRENGTH_DIVISORS = (1, 2, 3, 4, 6, 8, 12, 16)


def _strength(n: TaikoNote, grid) -> int:
    """Metrical weight as a divisor: 1 on the beat, 2 on the half, ... lower is stronger."""
    sec = grid.section_at(float(n.time))
    return next((d for d in STRENGTH_DIVISORS if sec.distance_ms(float(n.time), d) <= 2.0), 99)


def _with_colour(n: TaikoNote, colour: str) -> TaikoNote:
    t = n.note_type
    return replace(n, note_type=t.replace("kat", "don") if colour == "d" else t.replace("don", "kat"))


def enforce(notes: list[TaikoNote], grid, level: str) -> tuple[list[TaikoNote], Counter]:
    """
    Fix every problem and warning `check` raises about the notes, and count
    each fix. Minors are left: they are at the limit, not over it. Rest
    moments are never forced: ranked maps run long without one 10-32% of the
    time, and a forced rest deletes a phrase.

    There are no probabilities left at this point, so every choice goes by
    metre: of two notes too close together, or the note that splits a run
    that is too long, the one on the weaker beat goes; a pattern that must
    be one colour takes the colour of its strongest note; a finisher that
    may not stand there becomes a small note. ponytail: metre is a proxy
    for what the model was surest of; carrying the decoder's peak heights
    through would choose better.
    """
    spec = SPEC[level]
    fixes: Counter = Counter()
    out = sorted(notes, key=lambda n: n.time)

    # Triplet-only notes in a straight low difficulty: onto the nearest 1/4
    # line, truncated as osu! stores it -- first, so the gap pass below
    # judges the moved note where it now stands.
    if spec.straight:
        moved = []
        for n in out:
            if not n.is_long and _triplet_only(n, grid):
                sec = grid.section_at(float(n.time))
                t = int(sec.nearest(float(n.time), 4) + 1e-6)
                n = replace(n, time=t, end_time=t)
                fixes["moved: triplet onto a 1/4 line"] += 1
            moved.append(n)
        out = sorted(moved, key=lambda n: n.time)

    if spec.min_gap:
        kept: list[TaikoNote] = []
        for n in out:
            prev = kept[-1] if kept else None
            if (prev is not None and not n.is_long and not prev.is_long
                    and n.time - prev.time < spec.min_gap * _beat(grid, prev.time) - EPS_MS):
                fixes["dropped: too close"] += 1
                if _strength(n, grid) < _strength(prev, grid):
                    kept[-1] = n
                continue
            kept.append(n)
        out = kept

    # A run too long: drop the weakest of its first limit+1 notes, which splits it.
    runs_to_split = [(snap, limit, _beat) for snap, limit in spec.patterns]
    if spec.own_beat_run:
        runs_to_split.append((1, spec.own_beat_run, _own_beat))
    for snap, limit, beat in runs_to_split:
        while True:
            hits = [n for n in out if not n.is_long]
            run = next((r for r in _chains(hits, grid, snap, beat=beat) if len(r) > limit), None)
            if run is None:
                break
            victim = max(run[1:limit + 1], key=lambda i: (_strength(hits[i], grid), i))
            out = [n for n in out if n is not hits[victim]]
            fixes[f"dropped: 1/{round(1 / snap)} pattern over {limit}"] += 1

    if spec.plain:
        hits = [n for n in out if not n.is_long]
        swap = {}
        for run in _chains(hits, grid, spec.plain):
            colour = _colour(hits[min(run, key=lambda i: (_strength(hits[i], grid), i))])
            for i in run:
                if _colour(hits[i]) != colour:
                    swap[id(hits[i])] = _with_colour(hits[i], colour)
                    fixes["recoloured: plain pattern"] += 1
        out = [swap.get(id(n), n) for n in out]

    # Finishers last: the passes above change who neighbours whom. Shrinking
    # one changes nothing about its neighbours' places, so one pass suffices.
    for k, n in enumerate(out):
        if not n.is_long and _is_big(n) and _finisher_finding(out, k, grid, spec):
            out[k] = replace(n, note_type=n.note_type.replace("big_", ""))
            fixes["shrunk: finisher where it may not stand"] += 1

    if spec.spinner_gap:
        kept = []
        for n in out:
            # A loop: with the note before gone, the one before that can be too close too.
            while (n.note_type == "denden" and kept
                   and n.time - _end(kept[-1]) < spec.spinner_gap * _beat(grid, kept[-1].time)):
                kept.pop()
                fixes["dropped: too close before a spinner"] += 1
            kept.append(n)
        out = kept
    return out, fixes
