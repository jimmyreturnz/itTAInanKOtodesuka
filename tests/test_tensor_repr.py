"""
tests/test_tensor_repr.py

The representation is the ceiling on the whole system: whatever the round trip
loses here, no amount of training can recover.
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tempfile
import numpy as np

from taiko.data.frames import FRAME_MS, ms_to_frame
from taiko.data.osu_parser import OsuTaikoParser, TimingPoint
from taiko.data.tensor_repr import (
    CH_BIG_DON, CH_DENDEN, CH_DON, CH_KAT, CH_ROLL,
    N_CHART_CHANNELS, N_TIMING_CHANNELS,
    TM_COS, TM_DOWNBEAT, TM_SIN,
    beatmap_to_tensors, build_timing_stream, round_trip_accuracy,
    tensor_to_beatmap, timing_stream_from_bpm,
)


def _map(hit_objects: str, timing_points: str = "0,250,4,1,0,100,1,0") -> str:
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
SliderMultiplier:1.4
SliderTickRate:1

[TimingPoints]
{timing_points}

[HitObjects]
{hit_objects}
"""


def test_shapes_and_split():
    bm = OsuTaikoParser().parse_text(_map("256,192,1000,1,0,0:0:0:0:"))
    chart, timing = beatmap_to_tensors(bm)
    assert chart.shape[0] == N_CHART_CHANNELS == 6
    assert timing.shape[0] == N_TIMING_CHANNELS == 3
    assert chart.shape[1] == timing.shape[1]
    print(f"  chart {chart.shape} / timing {timing.shape}   ok")


def test_notes_land_on_the_right_frames():
    bm = OsuTaikoParser().parse_text(_map(
        "256,192,1000,1,0,0:0:0:0:\n"
        "256,192,1500,1,8,0:0:0:0:\n"
        "256,192,2000,1,4,0:0:0:0:"
    ))
    chart, _ = beatmap_to_tensors(bm)
    assert chart[CH_DON,     ms_to_frame(1000)] == 1.0
    assert chart[CH_KAT,     ms_to_frame(1500)] == 1.0
    assert chart[CH_BIG_DON, ms_to_frame(2000)] == 1.0
    assert chart[CH_DON].sum() == 1.0
    print("  onsets on exact frames    ok")


def test_long_notes_are_sustained_from_their_onset():
    """A roll occupies its own channel for its whole span, onset frame included."""
    bm = OsuTaikoParser().parse_text(_map("256,192,1000,2,0,L|300:192,1,140"))
    chart, _ = beatmap_to_tensors(bm)
    s, e = ms_to_frame(1000), ms_to_frame(1250)
    assert chart[CH_ROLL, s] == 1.0, "roll must be active on its onset frame"
    assert chart[CH_ROLL, e] == 1.0
    assert chart[CH_ROLL, s - 1] == 0.0
    assert chart[CH_ROLL, e + 1] == 0.0
    assert chart[CH_DON].sum() == 0.0, "a roll must not also fire the don channel"
    print("  roll sustain span         ok")


def test_timing_phase_is_zero_on_the_beat():
    """sin=0 and cos=1 exactly on a beat; a quarter later sin=1."""
    # 400 ms per beat = 150 BPM, chosen so that the beat, its quarter and its
    # half all fall on exact 20 ms frames.
    tp = [TimingPoint(time=0, beat_length=400.0, meter=4, uninherited=True)]
    stream = build_timing_stream(tp, n_frames=200)

    on_beat = ms_to_frame(800)
    assert abs(stream[TM_SIN, on_beat]) < 1e-5, stream[TM_SIN, on_beat]
    assert abs(stream[TM_COS, on_beat] - 1.0) < 1e-5

    quarter = ms_to_frame(900)          # a quarter of a beat later
    assert abs(stream[TM_SIN, quarter] - 1.0) < 1e-5, stream[TM_SIN, quarter]

    half = ms_to_frame(1000)            # half a beat later
    assert abs(stream[TM_COS, half] + 1.0) < 1e-5, stream[TM_COS, half]
    print("  beat phase alignment      ok")


def test_downbeat_peaks_at_the_measure():
    tp = [TimingPoint(time=0, beat_length=500.0, meter=4, uninherited=True)]
    stream = build_timing_stream(tp, n_frames=400)
    measure_ms = 500.0 * 4
    assert stream[TM_DOWNBEAT, ms_to_frame(0)] > 0.999
    assert stream[TM_DOWNBEAT, ms_to_frame(measure_ms)] > 0.999
    assert stream[TM_DOWNBEAT, ms_to_frame(measure_ms / 2)] < 0.001
    print("  downbeat proximity        ok")


def test_timing_follows_a_mid_map_tempo_change():
    """Phase restarts from the new red line, exactly as osu! restarts it."""
    # 120 BPM then 300 BPM. Both beat lengths are whole multiples of the 20 ms
    # frame, so every assertion below lands on an exact frame rather than
    # between two of them.
    tp = [
        TimingPoint(time=0,    beat_length=500.0, meter=4, uninherited=True),
        TimingPoint(time=4000, beat_length=200.0, meter=4, uninherited=True),
    ]
    stream = build_timing_stream(tp, n_frames=400)
    assert abs(stream[TM_COS, ms_to_frame(3500)] - 1.0) < 1e-5, "beat at the old tempo"
    assert abs(stream[TM_COS, ms_to_frame(4000)] - 1.0) < 1e-5, "red line is a beat"
    assert abs(stream[TM_COS, ms_to_frame(4200)] - 1.0) < 1e-5, "beat at the new tempo"
    assert abs(stream[TM_COS, ms_to_frame(4100)] + 1.0) < 1e-5, "half a beat at the new tempo"
    print("  tempo change restarts     ok")


def test_green_lines_do_not_move_the_beat_grid():
    from_red_only = build_timing_stream(
        [TimingPoint(time=0, beat_length=500.0, meter=4, uninherited=True)], 400
    )
    with_green = build_timing_stream(
        [
            TimingPoint(time=0,    beat_length=500.0, meter=4, uninherited=True),
            TimingPoint(time=1000, beat_length=-50.0, meter=4, uninherited=False),
        ],
        400,
    )
    assert np.allclose(from_red_only, with_green)
    print("  green lines ignored       ok")


def test_inference_timing_matches_training_timing():
    """
    timing_stream_from_bpm must produce exactly what build_timing_stream would
    for the same tempo -- otherwise the model meets a different conditioning
    distribution at generation time than it trained on.
    """
    a = timing_stream_from_bpm(bpm=180.0, offset_ms=317.0, n_frames=500, meter=4)
    b = build_timing_stream(
        [TimingPoint(time=317, beat_length=60_000.0 / 180.0, meter=4, uninherited=True)],
        500,
    )
    assert np.allclose(a, b), "train/inference timing streams diverge"
    print("  train == inference timing ok")


def test_round_trip_recovers_every_note():
    hits = []
    for i in range(64):
        t = 1000 + i * 125            # 1/2 notes at 240 BPM
        sound = 8 if i % 3 == 0 else 0
        hits.append(f"256,192,{t},1,{sound},0:0:0:0:")
    bm = OsuTaikoParser().parse_text(_map("\n".join(hits)))

    m = round_trip_accuracy(bm)
    assert m["recall"] == 1.0, m
    assert m["precision"] == 1.0, m
    print(f"  round trip {m['original_notes']} notes     ok  "
          f"(recall {m['recall']:.0%}, precision {m['precision']:.0%})")


def test_round_trip_preserves_note_types():
    bm = OsuTaikoParser().parse_text(_map(
        "256,192,1000,1,0,0:0:0:0:\n"
        "256,192,1250,1,8,0:0:0:0:\n"
        "256,192,1500,1,4,0:0:0:0:\n"
        "256,192,1750,1,12,0:0:0:0:"
    ))
    chart, _ = beatmap_to_tensors(bm)
    back = tensor_to_beatmap(chart, bpm=240.0, offset_ms=0.0)
    assert [n.note_type for n in back.notes] == ["don", "kat", "big_don", "big_kat"]
    print("  round trip note types     ok")


def test_hits_inside_a_roll_are_dropped():
    """
    A hit cannot coexist with a drumroll. The decoder must resolve the overlap
    the same way every time, or generated maps get unhittable stacked objects.
    """
    chart = np.zeros((N_CHART_CHANNELS, 200), dtype=np.float32)
    chart[CH_ROLL, 50:100] = 1.0
    chart[CH_DON, 60] = 1.0
    chart[CH_DON, 150] = 1.0
    back = tensor_to_beatmap(chart, bpm=240.0, offset_ms=0.0)
    types = [n.note_type for n in back.notes]
    assert types == ["roll", "don"], types
    print("  hits inside rolls dropped ok")


def test_short_long_notes_are_discarded():
    chart = np.zeros((N_CHART_CHANNELS, 200), dtype=np.float32)
    chart[CH_DENDEN, 50:52] = 1.0        # 2 frames, below min_long_frames
    back = tensor_to_beatmap(chart, bpm=240.0, offset_ms=0.0)
    assert back.notes == []
    print("  sub-threshold longs cut   ok")


def test_legacy_seven_channel_load():
    """Old 7-channel tensors must still load, minus the retired beat channel."""
    from taiko.data.tensor_repr import load_tensors
    legacy = np.random.rand(7, 300).astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        tmp = str(Path(d) / "legacy.npz")
        np.savez_compressed(tmp, tensor=legacy)
        chart, timing = load_tensors(tmp)
    assert chart.shape == (6, 300)
    assert timing.shape == (3, 300)
    assert np.allclose(chart, legacy[:6])
    print("  legacy 7ch load           ok")


if __name__ == "__main__":
    print("tensor_repr")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all representation tests passed")
