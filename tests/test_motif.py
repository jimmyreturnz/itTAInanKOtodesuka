"""
tests/test_motif.py

The motif vector is the style control. If it does not separate styles, the
product feature does not exist; if it is too precise, the model reads the
answer off it and generation collapses.
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from taiko.data.motif import (
    MOTIF_DIM, MOTIF_NAMES, N_BUCKETS, PRESETS,
    beat_frames_from_bpm, beat_frames_from_timing,
    compute_motif, corrupt_motif, get_preset, quantise_motif,
)
from taiko.data.tensor_repr import (
    CH_BIG_DON, CH_DON, CH_KAT, CH_ROLL, N_CHART_CHANNELS,
    timing_stream_from_bpm,
)


def _chart(width: int = 1500) -> np.ndarray:
    return np.zeros((N_CHART_CHANNELS, width), dtype=np.float32)


def _place(chart, frames, channel=CH_DON):
    for f in frames:
        chart[channel, int(f)] = 1.0


def test_beat_frames_recovered_from_timing():
    """The phasor increment must give back the tempo it was built from."""
    for bpm in (120.0, 150.0, 180.0, 240.0):
        timing = timing_stream_from_bpm(bpm, offset_ms=0.0, n_frames=2000)
        got    = beat_frames_from_timing(timing)
        want   = beat_frames_from_bpm(bpm)
        assert abs(got - want) < 0.05 * want, f"{bpm} BPM: got {got}, want {want}"
    print("  beat length recovery      ok")


def test_beat_frames_unknown_returns_zero():
    """An absent timing stream must report unknown, not guess a tempo."""
    assert beat_frames_from_timing(np.zeros((3, 500), dtype=np.float32)) == 0.0
    print("  unknown tempo -> 0        ok")


def test_ioi_histogram_finds_the_right_bin():
    """A pure 1/4 stream at 150 BPM must load the 1/4 bin and nothing else."""
    bpm = 150.0
    beat = beat_frames_from_bpm(bpm)          # 400 ms / 20 ms = 20 frames
    chart = _chart()
    _place(chart, np.arange(100, 1400, beat / 4))
    m = compute_motif(chart, beat, quantise=False)

    quarter = MOTIF_NAMES.index("ioi_1_4")
    assert m[quarter] > 0.9, m[quarter]
    for name in ("ioi_1_8", "ioi_1_2", "ioi_1_1", "ioi_1_3"):
        assert m[MOTIF_NAMES.index(name)] < 0.1, name
    print(f"  1/4 stream -> ioi_1_4     ok  ({m[quarter]:.2f})")


def test_colour_mix_and_change_rate():
    """Strict don/kat alternation is a full colour-change rate and half kat."""
    beat = beat_frames_from_bpm(150.0)
    chart = _chart()
    for i, f in enumerate(np.arange(100, 1400, beat / 4)):
        _place(chart, [f], CH_KAT if i % 2 else CH_DON)
    m = compute_motif(chart, beat, quantise=False)

    assert abs(m[MOTIF_NAMES.index("kat_frac")] - 0.5) < 0.02
    assert m[MOTIF_NAMES.index("colour_change")] > 0.98
    assert m[MOTIF_NAMES.index("pattern_entropy")] > 0.4
    print("  colour mix / change rate  ok")


def test_monotone_pattern_has_no_entropy():
    beat = beat_frames_from_bpm(150.0)
    chart = _chart()
    _place(chart, np.arange(100, 1400, beat / 4), CH_DON)
    m = compute_motif(chart, beat, quantise=False)
    assert m[MOTIF_NAMES.index("pattern_entropy")] < 0.01
    assert m[MOTIF_NAMES.index("colour_change")] < 0.01
    print("  monotone -> zero entropy  ok")


def test_burst_fraction_separates_stream_from_spread():
    """
    Two windows with identical note counts -- one in dense runs, one evenly
    spaced. Density cannot tell them apart; burst_frac must.
    """
    beat = beat_frames_from_bpm(150.0)
    burst = MOTIF_NAMES.index("burst_frac")
    density = MOTIF_NAMES.index("note_density")

    streamy = _chart()
    for start in range(100, 1300, 200):
        _place(streamy, [start + i * beat / 4 for i in range(8)])

    spread = _chart()
    _place(spread, np.linspace(100, 1400, 48))

    a = compute_motif(streamy, beat, quantise=False)
    b = compute_motif(spread,  beat, quantise=False)

    assert abs(a[density] - b[density]) < 0.05, "densities should be comparable"
    assert a[burst] > 0.9 and b[burst] < 0.1, (a[burst], b[burst])
    print(f"  burst vs spread           ok  ({a[burst]:.2f} vs {b[burst]:.2f})")


def test_big_and_roll_dimensions():
    beat = beat_frames_from_bpm(150.0)
    chart = _chart()
    _place(chart, np.arange(100, 500, beat / 2), CH_DON)
    _place(chart, np.arange(100 + beat / 4, 500, beat / 2), CH_BIG_DON)
    chart[CH_ROLL, 800:1100] = 1.0
    m = compute_motif(chart, beat, quantise=False)

    assert abs(m[MOTIF_NAMES.index("big_frac")] - 0.5) < 0.1
    assert abs(m[MOTIF_NAMES.index("roll_density")] - 300 / 1500) < 0.02
    print("  big / roll dimensions     ok")


def test_density_variance_detects_uneven_sections():
    beat = beat_frames_from_bpm(150.0)
    even = _chart()
    _place(even, np.linspace(50, 1450, 60))

    uneven = _chart()
    _place(uneven, np.linspace(50, 400, 55))       # everything in one slice
    _place(uneven, np.linspace(900, 1400, 5))

    v = MOTIF_NAMES.index("density_variance")
    a = compute_motif(even,   beat, quantise=False)
    b = compute_motif(uneven, beat, quantise=False)
    assert b[v] > a[v] + 0.3, (a[v], b[v])
    print(f"  density variance          ok  ({a[v]:.2f} vs {b[v]:.2f})")


def test_empty_window_is_all_zero():
    m = compute_motif(_chart(), beat_frames_from_bpm(150.0))
    assert np.all(m == 0.0)
    print("  empty window              ok")


def test_no_tempo_leaves_ioi_dims_empty():
    """Without a tempo the IOI bins must stay zero, not be computed against a guess."""
    chart = _chart()
    _place(chart, np.arange(100, 1400, 5))
    m = compute_motif(chart, beat_frames=0.0, quantise=False)
    assert np.all(m[:7] == 0.0)
    assert m[MOTIF_NAMES.index("note_density")] > 0, "density does not need a tempo"
    print("  no tempo -> no IOI bins   ok")


def test_quantisation_snaps_to_buckets():
    raw = np.linspace(0, 1, MOTIF_DIM).astype(np.float32)
    q = quantise_motif(raw)
    levels = np.round(q * (N_BUCKETS - 1))
    assert np.allclose(q, levels / (N_BUCKETS - 1))
    assert q.max() <= 1.0 and q.min() >= 0.0
    print(f"  quantised to {N_BUCKETS} buckets    ok")


def test_compute_motif_quantises_by_default():
    beat = beat_frames_from_bpm(150.0)
    chart = _chart()
    _place(chart, np.arange(100, 1400, beat / 4))
    m = compute_motif(chart, beat)
    assert np.allclose(m, quantise_motif(m)), "default path must be quantised"
    print("  default path quantised    ok")


def test_corruption_drops_dimensions_and_reports_a_mask():
    rng = np.random.default_rng(0)
    motif = np.full(MOTIF_DIM, 0.6, dtype=np.float32)

    dropped_any = False
    for _ in range(40):
        out, mask = corrupt_motif(motif, rng, dim_dropout=0.3, jitter=0.0)
        assert out.shape == mask.shape == (MOTIF_DIM,)
        assert np.all(out[mask == 0] == 0.0), "dropped dims must be zeroed"
        assert np.all(out[mask == 1] == 0.6), "surviving dims must be untouched"
        dropped_any |= bool((mask == 0).any())
    assert dropped_any, "dropout never fired"
    print("  per-dim dropout + mask    ok")


def test_corruption_dropout_rate_is_roughly_right():
    rng = np.random.default_rng(1)
    motif = np.full(MOTIF_DIM, 0.5, dtype=np.float32)
    kept = np.mean([
        corrupt_motif(motif, rng, dim_dropout=0.3, jitter=0.0)[1].mean()
        for _ in range(500)
    ])
    assert 0.62 < kept < 0.78, kept
    print(f"  dropout rate              ok  (kept {kept:.2f})")


def test_presets_are_distinct_and_well_formed():
    for name, vec in PRESETS.items():
        assert vec.shape == (MOTIF_DIM,), name
        assert vec.min() >= 0.0 and vec.max() <= 1.0, name
        assert np.allclose(vec, quantise_motif(vec)), f"{name} must be quantised"

    names = sorted(PRESETS)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = float(np.abs(PRESETS[a] - PRESETS[b]).sum())
            assert d > 0.3, f"presets {a} and {b} are too close ({d:.2f})"
    print(f"  {len(PRESETS)} presets distinct       ok")


def test_stream_preset_is_burstier_than_simple():
    """Sanity: the presets have to actually encode what their names claim."""
    burst = MOTIF_NAMES.index("burst_frac")
    density = MOTIF_NAMES.index("note_density")
    assert get_preset("stream")[burst] > get_preset("simple")[burst]
    assert get_preset("speed")[density] > get_preset("standard")[density]
    assert get_preset("big_heavy")[MOTIF_NAMES.index("big_frac")] > \
           get_preset("stream")[MOTIF_NAMES.index("big_frac")]
    print("  preset semantics          ok")


if __name__ == "__main__":
    print("motif")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all motif tests passed")
