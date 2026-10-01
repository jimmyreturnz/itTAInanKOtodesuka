"""Run: python tests/test_criteria.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from taiko.data.grid import Grid
from taiko.data.osu_parser import TaikoNote, TimingPoint
from taiko.eval.criteria import check, level_of


def _grid(bpm: float) -> Grid:
    return Grid([TimingPoint(time=0, beat_length=60_000 / bpm, meter=4, uninherited=True)])


def _notes(times, types):
    return [TaikoNote(time=int(t), note_type=k, end_time=int(t)) for t, k in zip(times, types)]


def test_a_kantan_1_2_colour_change_breaks_the_rule_at_180_but_not_at_90():
    # At 180 BPM a 1/2 is 167 ms; at 90 BPM it is 333 ms, which reads as a
    # 180 BPM 1/1, so the wiki allows the colour change there.
    for bpm, broken in ((180, True), (90, False)):
        half = 60_000 / bpm / 2
        notes = _notes([0, half], ["don", "kat"])
        fired = check(notes, _grid(bpm), "Kantan")
        assert ("rule: colour change in a 1/2 pattern" in fired) == broken, (bpm, fired)


def test_muzukashii_1_4_runs_and_finishers():
    quarter = 60_000 / 180 / 4
    six = _notes([i * quarter for i in range(6)], ["don"] * 6)
    assert "rule: 1/4 pattern over 5 notes" in check(six, _grid(180), "Muzukashii")
    five = _notes([i * quarter for i in range(5)], ["don"] * 4 + ["big_don"])
    fired = check(five, _grid(180), "Muzukashii")
    assert "rule: 1/4 pattern over 5 notes" not in fired
    assert "rule: finisher in a 1/4 or faster pattern" in fired


def test_oni_finisher_only_at_the_end_of_a_1_4_and_opposite_colour():
    quarter = 60_000 / 180 / 4
    t = [i * quarter for i in range(4)]
    ok = check(_notes(t, ["don", "don", "kat", "big_don"]), _grid(180), "Oni")
    same = check(_notes(t, ["don", "don", "don", "big_don"]), _grid(180), "Oni")
    inside = check(_notes(t, ["don", "big_kat", "don", "kat"]), _grid(180), "Oni")
    key = "rule: finisher inside a 1/4 pattern, or same colour as the note before"
    assert key not in ok and key in same and key in inside


def test_level_of_reads_custom_names():
    assert level_of("Kayoko's Oni") == "Oni"
    assert level_of("Inner Oni") == "Inner Oni"
    assert level_of("Hell Oni") == "Inner Oni"
    assert level_of("Ono's Taiko Muzukashii") == "Muzukashii"
    assert level_of("Chocolate from Rinami") is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name}  ok")
    print("all criteria tests passed")
