"""
tests/test_osu_parser.py

Parser correctness, focused on the things that silently corrupt training data
rather than the things that raise.
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from taiko.data.osu_parser import OsuTaikoParser, _timing_at


def _map(hit_objects: str, timing_points: str = "0,250,4,1,0,100,1,0",
         slider_multiplier: float = 1.4) -> str:
    """Minimal taiko .osu. 250 ms per beat = 240 BPM."""
    return f"""osu file format v14

[General]
AudioFilename: audio.mp3
Mode: 1

[Metadata]
Title:Test
Version:Test

[Difficulty]
HPDrainRate:5
OverallDifficulty:5
SliderMultiplier:{slider_multiplier}
SliderTickRate:1

[TimingPoints]
{timing_points}

[HitObjects]
{hit_objects}
"""


def test_hit_types():
    """don / kat / big / big-kat come from the hitsound bit flags."""
    bm = OsuTaikoParser().parse_text(_map(
        "\n".join([
            "256,192,1000,1,0,0:0:0:0:",    # normal      -> don
            "256,192,1250,1,8,0:0:0:0:",    # clap        -> kat
            "256,192,1500,1,4,0:0:0:0:",    # finish      -> big_don
            "256,192,1750,1,12,0:0:0:0:",   # finish+clap -> big_kat
        ])
    ))
    assert [n.note_type for n in bm.notes] == ["don", "kat", "big_don", "big_kat"]
    print("  hit types                 ok")


def test_roll_duration_one_beat():
    """
    140 px at SliderMultiplier 1.4 and 1.0x velocity is exactly one beat.
    At 250 ms per beat that is a 250 ms drumroll.

    The old parser read parts[7] -- the pixel length -- as an absolute end
    time, producing end_time = 140 ms, i.e. a roll that ends 860 ms before it
    starts.
    """
    bm = OsuTaikoParser().parse_text(_map(
        "256,192,1000,2,0,L|300:192,1,140"
    ))
    roll = bm.notes[0]
    assert roll.note_type == "roll", roll.note_type
    assert roll.time == 1000
    assert roll.duration == 250, f"expected 250 ms, got {roll.duration}"
    print(f"  roll 1 beat @240bpm       ok  ({roll.duration} ms)")


def test_roll_duration_respects_slides():
    """A repeating drumroll lasts `slides` times as long."""
    bm = OsuTaikoParser().parse_text(_map(
        "256,192,1000,2,0,L|300:192,3,140"
    ))
    assert bm.notes[0].duration == 750, bm.notes[0].duration
    print(f"  roll x3 slides            ok  ({bm.notes[0].duration} ms)")


def test_roll_duration_respects_slider_velocity():
    """A -50 green line means 2.0x velocity, so the same length takes half as long."""
    bm = OsuTaikoParser().parse_text(_map(
        "256,192,1000,2,0,L|300:192,1,140",
        timing_points="0,250,4,1,0,100,1,0\n500,-50,4,1,0,100,0,0",
    ))
    assert bm.notes[0].duration == 125, bm.notes[0].duration
    print(f"  roll under 2.0x SV        ok  ({bm.notes[0].duration} ms)")


def test_red_line_clears_green_line():
    """osu! resets slider velocity to 1.0x at every uninherited line."""
    bm = OsuTaikoParser().parse_text(_map(
        "256,192,2000,2,0,L|300:192,1,140",
        timing_points=(
            "0,250,4,1,0,100,1,0\n"
            "500,-50,4,1,0,100,0,0\n"      # 2.0x
            "1500,250,4,1,0,100,1,0"       # red line -> back to 1.0x
        ),
    ))
    assert bm.notes[0].duration == 250, bm.notes[0].duration

    _, sv_before = _timing_at(bm.timing_points, 1000)
    _, sv_after  = _timing_at(bm.timing_points, 2000)
    assert sv_before == 2.0 and sv_after == 1.0
    print("  red line clears green SV  ok")


def test_slider_velocity_scales_with_bpm_change():
    """Duration follows the red line in force at the drumroll, not the first one."""
    bm = OsuTaikoParser().parse_text(_map(
        "256,192,2000,2,0,L|300:192,1,140",
        timing_points="0,250,4,1,0,100,1,0\n1500,500,4,1,0,100,1,0",   # 240 -> 120 BPM
    ))
    assert bm.notes[0].duration == 500, bm.notes[0].duration
    print("  roll follows BPM change   ok")


def test_denden_uses_explicit_end_time():
    """Spinners carry a real end time at index 5 -- that path was already correct."""
    bm = OsuTaikoParser().parse_text(_map("256,192,1000,8,0,3000,0:0:0:0:"))
    assert bm.notes[0].note_type == "denden"
    assert bm.notes[0].duration == 2000
    print("  denden end time           ok")


def test_duration_uses_max_not_last_note():
    """A drumroll that starts early and ends late still sets the map duration."""
    bm = OsuTaikoParser().parse_text(_map(
        "256,192,1000,2,0,L|300:192,8,140\n"     # 1000 -> 3000
        "256,192,1500,1,0,0:0:0:0:"              # a hit inside the roll
    ))
    assert bm.duration_ms == 3000, bm.duration_ms
    print("  duration_ms = max end     ok")


def test_malformed_slider_falls_back_to_one_beat():
    """A slider with no usable geometry stays representable instead of collapsing."""
    bm = OsuTaikoParser().parse_text(_map("256,192,1000,2,0,L|300:192"))
    assert bm.notes[0].duration == 250, bm.notes[0].duration
    print("  malformed slider fallback ok")


def test_rejects_non_taiko_mode():
    try:
        OsuTaikoParser().parse_text(_map("256,192,1000,1,0").replace("Mode: 1", "Mode: 0"))
    except ValueError:
        print("  rejects non-taiko mode    ok")
        return
    raise AssertionError("expected ValueError for Mode: 0")


if __name__ == "__main__":
    print("osu_parser")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all parser tests passed")
