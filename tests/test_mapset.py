"""Run: python tests/test_mapset.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from taiko.data.osu_parser import TaikoNote, TimingPoint
from taiko.eval.mapset import Chart, chart_findings, fix_chart, set_findings, unsnap_ms

BEAT = 60_000 / 180          # 333.33 ms


def red(t, beat=BEAT, meter=4, effects=0):
    return TimingPoint(int(t), beat, meter, True, effects, float(t))


def green(t, sv, effects=0):
    return TimingPoint(int(t), -100 / sv, 4, False, effects, float(t))


def hits(times, kind="don"):
    return [TaikoNote(time=int(t), note_type=kind, end_time=int(t)) for t in times]


def song(n=200):          # a minute of 1/1 notes on the beat
    return [int(i * BEAT) for i in range(n)]


def test_a_note_on_its_line_is_truncated_like_osu_and_counts_as_snapped():
    r = red(0)
    assert unsnap_ms(int(BEAT / 4 * 3), r) == 0         # 250.0 -> 250
    assert unsnap_ms(int(BEAT / 3), r) == 0             # 111.11 -> 111
    assert unsnap_ms(int(BEAT / 3) + 1, r) == 1         # rounded instead: 1 ms off
    assert abs(unsnap_ms(300, r)) >= 2                  # on no ranked grid


def test_unsnapped_concurrent_and_short_charts_are_problems():
    c = Chart("Oni", hits(song() + [int(BEAT * 10) + 40]), [red(0)])
    found = chart_findings(c)
    assert "problem: unsnapped object" in found
    assert "problem: concurrent objects" in chart_findings(Chart("Oni", hits([0, 0] + song()[1:]), [red(0)]))
    assert "problem: drain time under 30 s" in chart_findings(Chart("Oni", hits(song(50)), [red(0)]))
    assert not [k for k in chart_findings(Chart("Oni", hits(song()), [red(0)])) if not k.startswith("minor")]


def test_a_red_line_cutting_a_measure_short_needs_its_barline_omitted():
    cut = int(BEAT * 4 * 8 + BEAT)              # one beat into a measure
    notes = hits(song())
    assert "problem: barline close to the next red line" in chart_findings(
        Chart("Oni", notes, [red(0), red(cut, BEAT)]))
    assert not any("barline close" in k for k in chart_findings(
        Chart("Oni", notes, [red(0), red(cut, BEAT, effects=8)])))


def test_the_last_note_1_ms_before_a_barline_hides_it():
    bar = int(BEAT * 4 * 50)
    found = chart_findings(Chart("Oni", hits(song(150) + [bar - 1]), [red(0)]))
    assert "problem: last note 1 ms before a barline, hiding it" in found


def test_an_sv_change_just_before_a_barline_is_flagged_and_a_volume_line_is_not():
    bar = int(BEAT * 4 * 10)
    sv = Chart("Oni", hits(song()), [red(0), green(bar - 3, 1.5)])
    vol = Chart("Oni", hits(song()), [red(0), green(bar - 3, 1.0)])
    assert "warning: green line just before a barline" in chart_findings(sv)
    assert "warning: green line just before a barline" not in chart_findings(vol)


def test_a_kantan_keeps_one_scroll_speed():
    lines = [red(0), green(int(BEAT * 40), 1.5), green(int(BEAT * 60), 1.0)]   # a 20-beat speed-up
    assert "warning: scroll speed change in a low difficulty" in chart_findings(Chart("Kantan", hits(song()), lines))
    assert "warning: scroll speed change in a low difficulty" not in chart_findings(Chart("Oni", hits(song()), lines))


def test_a_short_song_needs_a_low_lowest_difficulty():
    long_minute = song(400)                   # 2:13 of drain: lowest at most Futsuu
    oni = Chart("Oni", hits(long_minute), [red(0)])
    inner = Chart("Inner Oni", hits(long_minute), [red(0)])
    assert "problem: lowest difficulty above Futsuu for its drain time" in set_findings([oni, inner])
    kantan = Chart("Kantan", hits(long_minute), [red(0)])
    assert not set_findings([kantan, oni, inner])


def test_fix_chart_moves_a_note_onto_the_scroll_change_just_after_it():
    t = int(BEAT * 20)                        # 6666: a grid note truncated from 6666.67
    lines = [red(0), green(t + 1, 1.5)]       # the green line, as written, at 6667
    before = chart_findings(Chart("Oni", hits(song()), lines))
    assert "warning: object just before a scroll speed change" in before
    fixed, moves = fix_chart(hits(song()), lines)
    assert moves["moved onto the scroll speed change"] == 1
    assert "warning: object just before a scroll speed change" not in chart_findings(Chart("Oni", fixed, lines))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name}  ok")
    print("all mapset tests passed")
