"""
tests/test_serializer.py

Everything the model generates leaves through the serializer, so a bug here
degrades every output no matter how good the model is -- and it degrades them
silently, since the file still opens in osu!.
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from taiko.data.osu_parser import OsuTaikoParser, TaikoBeatmap, TaikoNote, TimingPoint
from taiko.data.tensor_repr import (
    CH_DENDEN, CH_DON, CH_ROLL, N_CHART_CHANNELS, tensor_to_beatmap,
)
from taiko.data.osu_writer import OsuTaikoSerializer


def _beatmap(notes, bpm: float = 150.0) -> TaikoBeatmap:
    bm = TaikoBeatmap()
    bm.title, bm.version, bm.audio_filename = "Test", "Test", "audio.mp3"
    bm.slider_multiplier = 1.4
    bm.timing_points = [TimingPoint(0, 60_000.0 / bpm, 4, True)]
    bm.notes = notes
    bm.compute_stats()
    return bm


def _round_trip(bm: TaikoBeatmap) -> TaikoBeatmap:
    text = OsuTaikoSerializer().serialize(bm, "audio.mp3")
    return OsuTaikoParser().parse_text(text)


def test_every_note_type_survives():
    """
    big_kat is the one that used to break. It was written as hitsound 10
    (WHISTLE|CLAP), which is a kat with no FINISH bit -- so it came back as an
    ordinary kat and every big kat in a generated map was silently downgraded.
    """
    notes = [
        TaikoNote(1000, "don"),
        TaikoNote(1400, "kat"),
        TaikoNote(1800, "big_don"),
        TaikoNote(2200, "big_kat"),
    ]
    back = _round_trip(_beatmap(notes))
    assert [n.note_type for n in back.notes] == [n.note_type for n in notes], \
        [n.note_type for n in back.notes]
    assert [n.time for n in back.notes] == [n.time for n in notes]
    print("  all four hit types        ok")


def test_drumroll_duration_survives():
    """
    osu! stores a drumroll as a pixel length and derives duration from slider
    velocity. Writing milliseconds there gives a roll off by a factor of
    ms_per_beat -- at 150 BPM a 400 ms roll came back as 1143 ms.
    """
    for bpm in (120.0, 150.0, 180.0, 240.0):
        ms_per_beat = 60_000.0 / bpm
        for beats in (0.5, 1.0, 2.0, 4.0):
            duration = int(round(ms_per_beat * beats))
            bm = _beatmap([TaikoNote(1000, "roll", end_time=1000 + duration)], bpm=bpm)
            back = _round_trip(bm)
            assert len(back.notes) == 1, back.notes
            got = back.notes[0].duration
            assert abs(got - duration) <= 2, \
                f"{bpm} BPM, {beats} beats: wrote {duration} ms, read {got} ms"
    print("  drumroll duration         ok  (16 tempo/length combinations)")


def test_denden_duration_survives():
    bm = _beatmap([TaikoNote(1000, "denden", end_time=3500)])
    back = _round_trip(bm)
    assert back.notes[0].note_type == "denden"
    assert back.notes[0].duration == 2500
    print("  denden duration           ok")


def test_timing_points_survive():
    bm = _beatmap([TaikoNote(1000, "don")], bpm=187.5)
    back = _round_trip(bm)
    reds = [tp for tp in back.timing_points if tp.uninherited]
    assert len(reds) == 1
    assert abs(60_000.0 / reds[0].beat_length - 187.5) < 0.01
    print("  timing points             ok")


def test_generated_chart_round_trips():
    """The real path: a decoded chart out and back with nothing lost."""
    chart = np.zeros((N_CHART_CHANNELS, 3000), dtype=np.float32)
    chart[CH_DON, 100:2000:25] = 1.0
    chart[1, 110:2000:50] = 1.0
    chart[2, 300:2000:200] = 1.0
    chart[3, 400:2000:400] = 1.0
    chart[CH_ROLL, 2100:2300] = 1.0
    chart[CH_DENDEN, 2400:2700] = 1.0
    chart[:4, 2100:2700] = 0.0

    bm = tensor_to_beatmap(chart, bpm=180.0, offset_ms=0.0)
    back = _round_trip(bm)

    assert back.note_count == bm.note_count, (bm.note_count, back.note_count)
    assert [n.note_type for n in back.notes] == [n.note_type for n in bm.notes]
    assert [n.time for n in back.notes] == [n.time for n in bm.notes]

    for a, b in zip(bm.notes, back.notes):
        if a.is_long:
            assert abs(a.duration - b.duration) <= 2, (a, b)
    print(f"  generated chart           ok  ({bm.note_count} notes)")


def test_output_is_taiko_mode():
    text = OsuTaikoSerializer().serialize(_beatmap([TaikoNote(1000, "don")]), "a.mp3")
    assert "Mode: 1" in text
    assert text.startswith("osu file format v14")
    print("  taiko mode header         ok")


if __name__ == "__main__":
    print("serializer")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all serializer tests passed")
