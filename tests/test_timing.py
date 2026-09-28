"""
tests/test_timing.py

Super timing on synthetic audio with exactly known red lines, and TimingNet's
targets.

    python -m pytest tests/test_timing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from taiko.data.osu_parser import TimingPoint
from taiko.data.tensor_repr import build_timing_stream, timing_stream_from_bpm
from taiko.timing import detect_timing
from taiko.timing.model import TimingNet, TimingNetConfig, beat_targets
from taiko.timing.sections import fit_sections, human_round, robust_fit, segment_beats

SR = 22_050


def _clicks(sections, dur_s: float, seed: int = 0) -> np.ndarray:
    """Kick on beats (accented every 4th), hat on half-beats, faint noise."""
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 0.01, int(dur_s * SR)).astype(np.float32)
    n = np.arange(int(0.03 * SR))

    def burst(amp, f):
        return amp * np.sin(2 * np.pi * f * n / SR) * np.exp(-n / (0.005 * SR))

    for start, bpm, count in sections:
        mpb = 60_000.0 / bpm
        for k in range(count):
            for t, b in ((start + k * mpb, burst(1.0 if k % 4 == 0 else 0.6, 60 if k % 4 == 0 else 180)),
                         (start + (k + 0.5) * mpb, burst(0.25, 5000))):
                i = int(round(t / 1000 * SR))
                if i + len(b) < len(y):
                    y[i:i + len(b)] += b
    return y


def _phase_error_ms(tp: TimingPoint, true_offset: float, bpm: float) -> float:
    mpb = 60_000.0 / bpm
    rel = (tp.time - true_offset) / mpb
    return abs(rel - round(rel)) * mpb


def test_steady_tempo_exact_bpm_and_offset():
    y = _clicks([(1234.0, 174.0, 110)], 42)
    r = detect_timing(y, backend="onset")
    assert len(r.timing_points) == 1, r.describe()
    tp = r.timing_points[0]
    assert abs(60_000.0 / tp.beat_length - 174.0) < 1e-6, r.describe()
    assert _phase_error_ms(tp, 1234.0, 174.0) <= 1.0, r.describe()


def test_bpm_change_becomes_a_second_red_line():
    first = (800.0, 150.0, 60)
    second_start = 800.0 + 60 * 400.0
    y = _clicks([first, (second_start, 200.0, 80)], 52)
    r = detect_timing(y, backend="onset")
    bpms = [round(60_000.0 / tp.beat_length, 3) for tp in r.timing_points]
    assert bpms == [150.0, 200.0], r.describe()
    assert _phase_error_ms(r.timing_points[1], second_start, 200.0) <= 1.0, r.describe()
    # The second red line must not re-time the first section's last beat.
    assert r.timing_points[1].time > second_start - 400.0


def test_segmentation_and_human_rounding():
    rng = np.random.default_rng(0)
    a = 500 + np.arange(64) * 400.0                    # 150 BPM
    b = a[-1] + 300 + np.arange(64) * 300.0           # 200 BPM
    times = np.concatenate([a, b]) + rng.normal(0, 2.0, 128)
    runs = segment_beats(times)
    # The last 150 BPM beat also lies on the 200 BPM line (the gap between the
    # sections is one 200 BPM period), so either side may claim it.
    assert len(runs) == 2 and runs[1][0] in (63, 64), runs
    secs = fit_sections(times, runs)
    assert [round(s.bpm, 6) for s in secs] == [150.0, 200.0]

    # A genuinely fractional BPM is not forced onto an integer.
    t = 100 + np.arange(200) * (60_000.0 / 173.37)
    mpb, t0, inlier = robust_fit(t)
    mpb2, _ = human_round(t, mpb, t0, inlier, leniency_ms=12.0)
    assert abs(60_000.0 / mpb2 - 173.37) < 1e-6


def test_timingnet_targets_follow_the_grid():
    stream = torch.from_numpy(timing_stream_from_bpm(150.0, 130.0, 1000))[None]
    tgt = beat_targets(stream)[0]
    beats = torch.nonzero(tgt[0] >= 1.0).flatten().numpy()
    expected = np.round((130.0 + np.arange(0, 60) * 400.0) / 20.0).astype(int)
    expected = expected[expected < 1000]
    assert set(beats) == set(expected[expected > 0]), (beats[:8], expected[:8])
    downs = torch.nonzero(tgt[1] >= 1.0).flatten().numpy()
    assert set(downs) == set(expected[expected > 0][::4]) or \
        set(downs) == set(expected[4::4]) | set(expected[:1][expected[:1] > 0])

    # Multi-BPM: beats restart at the second red line.
    pts = [TimingPoint(time=0, beat_length=500.0, meter=4, uninherited=True),
           TimingPoint(time=10_010, beat_length=300.0, meter=4, uninherited=True)]
    s = torch.from_numpy(build_timing_stream(pts, 800))[None]
    b = torch.nonzero(beat_targets(s)[0, 0] >= 1.0).flatten().numpy()
    # 10010 ms and 10310 ms are frames 500.5 and 515.5; torch rounds half to even.
    assert 475 in b and 500 in b and 516 in b and 525 not in b, b
    net = TimingNet(TimingNetConfig(channels=16, n_blocks=3))
    assert net(torch.randn(2, 128, 300)).shape == (2, 2, 300)
