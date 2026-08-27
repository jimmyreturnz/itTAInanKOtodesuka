"""
tests/test_shards.py

Packing is lossless where it matters: every note, every span, every frame of
audio must come back exactly where it went in. A silent off-by-one here would
misalign the whole corpus.
"""

from __future__ import annotations
import sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from taiko.data.osu_parser import TimingPoint
from taiko.data.shards import MEL_BINS, ShardReader, ShardWriter, encode_chart, decode_chart_window
from taiko.data.tensor_repr import (
    CH_DENDEN, CH_DON, CH_KAT, CH_ROLL, N_CHART_CHANNELS, build_timing_stream,
)


def _make_chart(width=3000, seed=0):
    rng = np.random.default_rng(seed)
    chart = np.zeros((N_CHART_CHANNELS, width), dtype=np.float32)
    for f in rng.choice(width, size=400, replace=False):
        chart[rng.integers(0, 4), f] = 1.0
    chart[CH_ROLL, 500:640] = 1.0
    chart[CH_ROLL, 1800:1900] = 1.0
    chart[CH_DENDEN, 2200:2400] = 1.0
    # A roll and an onset cannot share a frame in real data; clear the overlap
    # so the round-trip comparison is against a legal chart.
    for ch in (CH_ROLL, CH_DENDEN):
        active = chart[ch] > 0.5
        chart[0:4, active] = 0.0
    return chart


def test_sparse_encode_decode_is_lossless():
    chart = _make_chart()
    ev = encode_chart(chart)
    back = decode_chart_window(
        ev["onset_frames"], ev["onset_channels"],
        ev["span_starts"], ev["span_ends"], ev["span_channels"],
        0, chart.shape[1],
    )
    assert np.array_equal(back, chart), "sparse round trip lost data"
    print(f"  sparse round trip         ok  ({ev['onset_frames'].size} onsets, "
          f"{ev['span_starts'].size} spans)")


def test_window_decode_matches_slice():
    chart = _make_chart()
    ev = encode_chart(chart)
    for start, width in ((0, 500), (450, 300), (1750, 400), (2900, 200)):
        got = decode_chart_window(
            ev["onset_frames"], ev["onset_channels"],
            ev["span_starts"], ev["span_ends"], ev["span_channels"],
            start, width,
        )
        want = np.zeros((N_CHART_CHANNELS, width), dtype=np.float32)
        hi = min(start + width, chart.shape[1])
        want[:, :hi - start] = chart[:, start:hi]
        assert np.array_equal(got, want), (start, width)
    print("  window == slice           ok")


def test_span_crossing_window_start_stays_held():
    """A drumroll that began before the window must read as held, not restart."""
    chart = np.zeros((N_CHART_CHANNELS, 1000), dtype=np.float32)
    chart[CH_ROLL, 100:600] = 1.0
    ev = encode_chart(chart)
    win = decode_chart_window(
        ev["onset_frames"], ev["onset_channels"],
        ev["span_starts"], ev["span_ends"], ev["span_channels"],
        300, 200,
    )
    assert win[CH_ROLL].all(), "roll should cover the entire window"
    print("  span clipped, not dropped ok")


def test_full_pack_round_trip():
    chart_a = _make_chart(3000, seed=1)
    chart_b = _make_chart(2000, seed=2)
    mel = (np.random.default_rng(3).random((MEL_BINS, 3500)).astype(np.float32) * 2 - 1)
    points = [
        TimingPoint(time=0, beat_length=400.0, meter=4, uninherited=True),
        TimingPoint(time=20_000, beat_length=300.0, meter=3, uninherited=True),
        TimingPoint(time=25_000, beat_length=-50.0, meter=4, uninherited=False),  # ignored
    ]

    with tempfile.TemporaryDirectory() as tmp:
        with ShardWriter(tmp) as w:
            w.add_mel("song", mel)
            w.add_map({"difficulty": 5.2, "style": 1}, chart_a, points, "song")
            w.add_map({"difficulty": 3.1, "style": 0}, chart_b, points, "song")
            stats = w.close()

        assert stats["maps"] == 2 and stats["songs"] == 1

        r = ShardReader(tmp)
        assert len(r) == 2
        assert r.chart_length(0) == 3000 and r.chart_length(1) == 2000
        assert r.records[0]["difficulty"] == 5.2

        for idx, chart in ((0, chart_a), (1, chart_b)):
            got = r.chart_window(idx, 0, chart.shape[1])
            assert np.array_equal(got, chart), f"chart {idx} corrupted by packing"

        # float16 is the only lossy step, and only in the mantissa.
        got_mel = r.mel_window(0, 100, 400)
        assert got_mel.shape == (MEL_BINS, 400)
        assert np.allclose(got_mel, mel[:, 100:500], atol=1e-3), "mel misaligned"

        # Past the end of the audio must be silence, not garbage or a wrap.
        tail = r.mel_window(0, 3400, 400)
        assert np.allclose(tail[:, 100:], 0.0), "no zero padding past end of audio"
        assert np.allclose(tail[:, :100], mel[:, 3400:3500], atol=1e-3)

        timing = r.timing_window(0, 250, 300)
        want   = build_timing_stream(points, 300, start_frame=250)
        assert np.allclose(timing, want), "timing stream diverged"
        assert r.onset_count(0) > 0
    print("  full pack round trip      ok")


def test_reader_rejects_a_stale_frame_rate():
    """A dataset packed on a different time grid must refuse to load."""
    import json
    with tempfile.TemporaryDirectory() as tmp:
        with ShardWriter(tmp) as w:
            w.add_mel("s", np.zeros((MEL_BINS, 100), dtype=np.float32))
            w.add_map({}, np.zeros((N_CHART_CHANNELS, 100), dtype=np.float32), [], "s")
            w.close()
        p = Path(tmp) / "index.json"
        data = json.loads(p.read_text())
        data["frame_ms"] = 10.0
        p.write_text(json.dumps(data))
        try:
            ShardReader(tmp)
        except ValueError as e:
            assert "repack" in str(e)
            print("  stale frame rate refused  ok")
            return
    raise AssertionError("reader accepted a mismatched frame rate")


def test_missing_dataset_says_what_to_run():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            ShardReader(tmp)
        except FileNotFoundError as e:
            assert "pack_dataset" in str(e)
            print("  missing dataset message   ok")
            return
    raise AssertionError("expected FileNotFoundError")


if __name__ == "__main__":
    print("shards")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all shard tests passed")
