"""
tests/test_dataset.py

The alignment test in here is the one that matters most in the repository. If
it passes, the model can learn to follow the music; if it fails, nothing
downstream can compensate.
"""

from __future__ import annotations
import sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from taiko.data.conditioning import STYLE_NULL
from taiko.data.frames import FRAME_MS
from taiko.data.motif import MOTIF_DIM
from taiko.data.osu_parser import TimingPoint
from taiko.data.preprocessed_dataset import WindowedDataset, split_indices
from taiko.data.shards import MEL_BINS, ShardReader, ShardWriter
from taiko.data.tensor_repr import (
    CH_DON, CH_ROLL, N_CHART_CHANNELS, N_TIMING_CHANNELS,
)

W = 384          # a multiple of 64, as WindowedDataset requires

# ponytail: TemporaryDirectory(ignore_cleanup_errors=True) everywhere below --
# ShardReader holds an np.memmap on mels.dat and never releases it, so Windows
# refuses to delete the temp dir. Give ShardReader a close()/context manager if
# anything ever needs to repack shards in a live process.


def _decode_frames(mel: np.ndarray) -> np.ndarray:
    """Read the absolute frame index back out of the probe mel."""
    high = np.round(mel[0] * 100).astype(np.int64)
    low  = np.round(mel[1] * 100).astype(np.int64)
    return high * 64 + low


def _build(tmp: str, n_songs: int = 4, diffs_per_song: int = 3, length: int = 4000):
    """
    A corpus whose mel encodes its own frame index, so any misalignment between
    audio and chart is directly readable from the data.

    The index is split across two channels as (t // 64, t % 64), each scaled by
    1/100. Mels are stored as float16, which cannot resolve a 1/4000 ramp --
    two coarse channels stay exact where one fine one would not.
    """
    with ShardWriter(tmp) as writer:
        for s in range(n_songs):
            key = f"song{s}"
            t = np.arange(length)
            mel = np.zeros((MEL_BINS, length), dtype=np.float32)
            mel[0] = (t // 64) / 100.0
            mel[1] = (t % 64) / 100.0
            mel[2] = s / 100.0
            writer.add_mel(key, mel)

            for d in range(diffs_per_song):
                chart = np.zeros((N_CHART_CHANNELS, length), dtype=np.float32)
                chart[CH_DON, ::37] = 1.0
                chart[CH_ROLL, 1000:1200] = 1.0
                chart[CH_DON, 1000:1200] = 0.0
                writer.add_map(
                    {
                        "difficulty": 3.0 + d,
                        "style": d % 4,
                        "avg_nps": 4.0 + d,
                        "peak_nps": 8.0 + d,
                        "ranked": s % 2 == 0,
                    },
                    chart,
                    [TimingPoint(time=0, beat_length=400.0, meter=4, uninherited=True)],
                    key,
                )
    return ShardReader(tmp)


def test_mel_and_chart_describe_the_same_frames():
    """
    THE alignment test.

    mel[0] is a ramp of absolute frame index. Reading the ramp back tells us
    which frames of audio the sample actually contains; it must be exactly the
    frames the chart window covers. The old `start * 2` bug fails this by a
    factor of two.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        reader = _build(tmp)
        ds = WindowedDataset(reader, window_frames=W, random_window=True,
                             augment=False, samples_per_epoch=64, seed=7)

        for i in range(64):
            s = ds[i]
            mel = s["mel"].numpy()
            frames = _decode_frames(mel)

            assert frames[-1] - frames[0] == W - 1, (
                f"mel window spans {frames[-1] - frames[0] + 1} frames, "
                f"chart window spans {W} -- the two grids disagree"
            )
            assert np.array_equal(frames, np.arange(frames[0], frames[0] + W)), (
                "mel frames are not consecutive -- the window was resampled or "
                "strided rather than sliced"
            )
            first_frame = frames[0]

            chart = s["chart"].numpy()
            for f in np.flatnonzero(chart[CH_DON] > 0.5):
                absolute = first_frame + int(f)
                assert absolute % 37 == 0, (
                    f"a don landed at absolute frame {absolute}, which is not "
                    f"where the chart put one -- audio and chart are offset"
                )
        print(f"  mel/chart same frames     ok  ({64} windows)")


def test_shapes_and_dtypes():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        ds = WindowedDataset(_build(tmp), window_frames=W, samples_per_epoch=4)
        s = ds[0]
        assert s["mel"].shape    == (MEL_BINS, W)
        assert s["chart"].shape  == (N_CHART_CHANNELS, W)
        assert s["timing"].shape == (N_TIMING_CHANNELS, W)
        assert s["valid_mask"].shape == (W,)
        assert s["motif"].shape == s["motif_mask"].shape == (MOTIF_DIM,)
        assert s["style"].dtype == torch.long
        for key in ("mel", "chart", "timing", "valid_mask", "motif"):
            assert s[key].dtype == torch.float32, key
        print("  shapes and dtypes         ok")


def test_timing_stream_tracks_the_window():
    """The beat grid must describe the window's own frames, not frame zero."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        reader = _build(tmp)
        ds = WindowedDataset(reader, window_frames=W, random_window=True,
                             augment=False, samples_per_epoch=32, seed=3)
        from taiko.data.motif import beat_frames_from_timing
        for i in range(32):
            s = ds[i]
            beat = beat_frames_from_timing(s["timing"].numpy())
            assert abs(beat - 20.0) < 1.0, f"recovered beat {beat}, expected 20 frames"
        print("  timing follows window     ok")


def test_samples_per_epoch_controls_length():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        reader = _build(tmp)
        assert len(WindowedDataset(reader, window_frames=W)) == len(reader)
        assert len(WindowedDataset(reader, window_frames=W, samples_per_epoch=5000)) == 5000
        print("  samples_per_epoch         ok")


def test_eval_mode_is_deterministic():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        reader = _build(tmp)
        ds = WindowedDataset(reader, window_frames=W, random_window=False, augment=False)
        for i in (0, 3, 7):
            assert torch.equal(ds[i]["chart"], ds[i]["chart"])
            assert torch.equal(ds[i]["mel"], ds[i]["mel"])
        print("  eval determinism          ok")


def test_training_mode_moves_the_window():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        reader = _build(tmp)
        ds = WindowedDataset(reader, window_frames=W, random_window=True,
                             augment=False, samples_per_epoch=40, seed=11)
        firsts = {float(ds[i]["mel"][0][0]) for i in range(40)}
        assert len(firsts) > 10, f"only {len(firsts)} distinct window starts"
        print(f"  window randomisation      ok  ({len(firsts)} distinct starts)")


def test_rate_augmentation_keeps_everything_in_step():
    """Augmentation must resample audio, chart and beat grid by the same factor."""
    from taiko.data.preprocessed_dataset import _rate_augment
    from taiko.data.motif import beat_frames_from_timing
    from taiko.data.tensor_repr import timing_stream_from_bpm

    mel = np.zeros((MEL_BINS, 1000), dtype=np.float32)
    mel[0] = np.arange(1000) / 1000.0
    chart = np.zeros((N_CHART_CHANNELS, 1000), dtype=np.float32)
    chart[CH_DON, ::20] = 1.0
    timing = timing_stream_from_bpm(150.0, 0.0, 1000)
    valid = np.ones(1000, dtype=np.float32)

    rate = 1.25
    m2, c2, t2, v2 = _rate_augment(mel, chart, timing, valid, rate)
    assert m2.shape == (MEL_BINS, 1000)
    assert c2.shape == (N_CHART_CHANNELS, 1000)

    # 150 BPM is 20 frames per beat; sped up by 1.25 it becomes 16.
    beat = beat_frames_from_timing(t2[:, :int(1000 / rate)])
    assert abs(beat - 20.0 / rate) < 1.5, f"beat became {beat}, expected {20.0/rate}"

    # Notes were every 20 frames; they should now be every 16.
    gaps = np.diff(np.flatnonzero(c2[CH_DON] > 0.5))
    assert abs(float(np.median(gaps)) - 20.0 / rate) < 1.5, np.median(gaps)
    print(f"  rate augmentation in step ok  (beat {beat:.1f} frames)")


def test_valid_mask_marks_the_tail():
    # A chart shorter than the window is the only case that can overrun: for
    # longer charts the sampler bounds the start so the window always fits.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        reader = _build(tmp, length=300)
        ds = WindowedDataset(reader, window_frames=W, random_window=True,
                             augment=False, samples_per_epoch=20, seed=5)
        for i in range(20):
            mask = ds[i]["valid_mask"].numpy()
            cut = int(mask.sum())
            assert cut == 300, f"expected 300 valid frames, got {cut}"
            assert mask[:cut].all() and not mask[cut:].any(), "mask must be a prefix"
        print("  valid mask tail           ok")


def test_window_never_starts_past_the_last_note():
    """Sampling into the silent tail wastes a training step on nothing."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        reader = _build(tmp, length=4000)
        ds = WindowedDataset(reader, window_frames=W, random_window=True,
                             augment=False, samples_per_epoch=50, seed=9)
        for i in range(50):
            s = ds[i]
            assert s["valid_mask"].sum() == W, "window ran past the chart"
            assert s["chart"].sum() > 0, "sampled an empty window"
        print("  windows stay in the chart ok")


def test_motif_is_corrupted_only_when_augmenting():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        reader = _build(tmp)

        clean = WindowedDataset(reader, window_frames=W, augment=False, samples_per_epoch=10)
        for i in range(10):
            assert clean[i]["motif_mask"].sum() == MOTIF_DIM

        dirty = WindowedDataset(reader, window_frames=W, augment=True,
                                samples_per_epoch=60, motif_dropout=0.3, seed=2)
        kept = np.mean([float(dirty[i]["motif_mask"].mean()) for i in range(60)])
        assert 0.55 < kept < 0.85, kept
        print(f"  motif dropout             ok  (kept {kept:.2f})")


def test_split_is_by_song_not_by_map():
    """
    Difficulties of one song share audio; letting them straddle the split turns
    validation into a memorisation check.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        reader = _build(tmp, n_songs=10, diffs_per_song=3)
        train, val = split_indices(reader, val_ratio=0.3, seed=1)

        train_songs = {reader.records[i]["mel_key"] for i in train}
        val_songs   = {reader.records[i]["mel_key"] for i in val}
        assert not (train_songs & val_songs), "a song appears in both splits"
        assert len(train) + len(val) == len(reader)
        print(f"  song-level split          ok  ({len(train_songs)}/{len(val_songs)} songs)")


def test_ranked_only_filter():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        reader = _build(tmp, n_songs=10)
        train, val = split_indices(reader, val_ratio=0.2, ranked_only=True)
        for i in train + val:
            assert reader.records[i]["ranked"]
        print("  ranked-only filter        ok")


def test_conditioning_is_normalised():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        ds = WindowedDataset(_build(tmp), window_frames=W, augment=False, samples_per_epoch=12)
        for i in range(12):
            s = ds[i]
            for key in ("difficulty", "avg_nps", "peak_nps"):
                v = float(s[key])
                assert 0.0 <= v <= 1.5, f"{key} = {v} is not normalised"
            assert 0 <= int(s["style"]) <= STYLE_NULL
        print("  conditioning normalised   ok")


def test_batches_collate():
    from torch.utils.data import DataLoader
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        ds = WindowedDataset(_build(tmp), window_frames=W, augment=True, samples_per_epoch=16)
        batch = next(iter(DataLoader(ds, batch_size=4, num_workers=0)))
        assert batch["mel"].shape == (4, MEL_BINS, W)
        assert batch["chart"].shape == (4, N_CHART_CHANNELS, W)
        assert batch["timing"].shape == (4, N_TIMING_CHANNELS, W)
        print("  dataloader collate        ok")


if __name__ == "__main__":
    print("dataset")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all dataset tests passed")
