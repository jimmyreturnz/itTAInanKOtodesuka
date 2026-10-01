"""Run: python tests/test_criteria.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from taiko.data.grid import Grid
from taiko.data.osu_parser import TaikoNote, TimingPoint
from taiko.eval.criteria import LEVELS, check, enforce, folded_beat_ms, level_of


def _grid(bpm: float) -> Grid:
    return Grid([TimingPoint(time=0, beat_length=60_000 / bpm, meter=4, uninherited=True)])


def _notes(times, types):
    return [TaikoNote(time=int(t), note_type=k, end_time=int(t)) for t, k in zip(times, types)]


def _fixable(found) -> list[str]:
    return [k for k in found if k.split(":")[0] in ("problem", "warning") and "rest" not in k]


def test_the_beat_folds_into_130_270_bpm():
    for bpm, folded in ((90, 180), (100, 200), (120, 180), (180, 180), (300, 150)):
        assert abs(60_000 / folded_beat_ms(60_000 / bpm) - folded) < 1e-6, bpm


def test_a_kantan_1_2_colour_change_is_flagged_at_180_but_not_at_90():
    # At 90 BPM the beat folds to 180's, so its 1/2 reads as 180's 1/1.
    for bpm, flagged in ((180, True), (90, False)):
        half = 60_000 / bpm / 2
        found = check(_notes([0, half], ["don", "kat"]), _grid(bpm), "Kantan")
        assert ("warning: colour change in a 1/2 pattern" in found) == flagged, (bpm, found)


def test_pattern_length_is_a_warning_over_the_limit_and_minor_at_it():
    quarter = 60_000 / 180 / 4
    six = check(_notes([i * quarter for i in range(6)], ["don"] * 6), _grid(180), "Muzukashii")
    five = check(_notes([i * quarter for i in range(5)], ["don"] * 5), _grid(180), "Muzukashii")
    assert "warning: 1/4 pattern over 5 notes" in six
    assert "minor: 1/4 pattern of 5 notes" in five and not _fixable(five)


def test_oni_finisher_only_at_the_end_of_a_1_4_and_after_a_colour_change():
    quarter = 60_000 / 180 / 4
    t = [i * quarter for i in range(4)] + [4 * quarter + 1000]
    key = "problem: finisher in a pattern too fast for it"
    ok = check(_notes(t, ["don", "don", "kat", "big_don", "don"]), _grid(180), "Oni")
    same = check(_notes(t, ["don", "don", "don", "big_don", "don"]), _grid(180), "Oni")
    inside = check(_notes(t, ["don", "big_kat", "don", "kat", "don"]), _grid(180), "Oni")
    assert key not in ok and key in same and key in inside, (ok, same, inside)


def test_level_of_takes_exact_names_and_falls_back_to_star_rating():
    assert level_of("Kayoko's Oni") == "Oni"
    assert level_of("Inner Oni") == "Inner Oni"
    assert level_of("Ura Oni") == "Inner Oni"
    assert level_of("Hell Oni") == "Hell Oni"
    assert level_of("Ono's Taiko Muzukashii") is None              # a custom name
    assert level_of("Ono's Taiko Muzukashii", 3.2) == "Muzukashii"
    assert level_of("Oni", 5.6) == "Inner Oni"                     # an Oni rated as an Inner Oni
    assert level_of("Kantan", 2.5) == "Kantan"                     # names below Oni are trusted


def test_enforce_recolours_a_kantan_1_2_by_its_stronger_note():
    half = 60_000 / 180 / 2
    out, _ = enforce(_notes([0, half], ["kat", "don"]), _grid(180), "Kantan")
    assert [n.note_type for n in out] == ["kat", "kat"], out     # the downbeat's colour wins
    assert not _fixable(check(out, _grid(180), "Kantan"))


def test_enforce_leaves_nothing_to_fix_on_random_dense_charts():
    import random
    rng = random.Random(0)
    kinds = ["don", "kat", "big_don", "big_kat"]
    for bpm in (95, 150, 180, 230, 300):
        step = 60_000 / bpm / 8
        for _ in range(16):
            times = sorted({round(rng.randrange(400) * step) for _ in range(250)})
            notes = _notes(times, [rng.choice(kinds) for _ in times])
            for level in LEVELS:
                out, _ = enforce(notes, _grid(bpm), level)
                left = _fixable(check(out, _grid(bpm), level))
                assert not left, (bpm, level, left)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name}  ok")
    print("all criteria tests passed")
