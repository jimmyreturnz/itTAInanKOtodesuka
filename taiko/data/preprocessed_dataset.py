"""
taiko/data/preprocessed_dataset.py

Windowed dataset over the packed corpus.

THE ALIGNMENT CONTRACT
----------------------
For a window starting at frame `s`, every array describes frames [s, s+W):

    mel    [128, W]     the audio
    chart  [6,   W]     what happened in the chart
    timing [3,   W]     where the beats are

One frame is one frame is one frame. The previous version cropped the mel at
`start * 2`, on the belief that mel ran at twice the chart rate -- it does not;
both are 20 ms. Every training sample therefore paired a chart with a different
span of the song, and the model could not learn alignment no matter how long it
trained, while the loss fell exactly as it would if everything were fine.

`taiko.data.frames.assert_aligned` is called on load so a regression of that
class stops the run instead of quietly wasting a week of GPU time.

Augmentation
------------
Rate augmentation resamples mel, chart and timing together. Stretching time
stretches the beat grid with it, so the timing stream stays truthful about the
augmented audio -- resampling the sin/cos phasor is right once its radius is
restored (see _rate_augment), which is one
more reason to carry phase rather than a pulse train.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from taiko.data.conditioning import (
    MOTIF_DIM_DROPOUT, MOTIF_JITTER, STYLE_NULL,
    normalise_avg_nps, normalise_difficulty, normalise_peak_nps,
)
from taiko.data.frames import FRAME_MS, assert_aligned
from taiko.data.motif import (
    MOTIF_DIM, beat_frames_from_timing, compute_motif, corrupt_motif,
)
from taiko.data.shards import MEL_BINS, ShardReader
from taiko.data.tensor_repr import (
    N_CHART_CHANNELS, N_TIMING_CHANNELS, ONSET_CHANNELS, TM_COS, TM_SIN,
)

# Window sizes in frames (20 ms each).
#
# Every value here is a multiple of 64, so a window round-trips exactly through
# an autoencoder compressing by 8, 16, 32 or 64. A window that is not a whole
# number of latent frames loses its remainder on the way back out -- 1500
# frames returns as 1488 at 16x -- and the missing tail lands on the loss as
# phantom error at the end of every single sample.
WINDOW_FRAMES_MIN     = 1_024    # 20.5 s
WINDOW_FRAMES_MAX     = 2_048    # 41.0 s
WINDOW_FRAMES_DEFAULT = 1_536    # 30.7 s

WINDOW_ALIGNMENT = 64


class WindowedDataset(Dataset):
    """
    Random fixed-size windows over the packed corpus.

    Args:
        reader:            an open ShardReader
        indices:           which maps this split may draw from
        window_frames:     window size; must stay divisible by the autoencoder's
                           compression ratio
        random_window:     True to draw a random start per call (training),
                           False to start at the first note (evaluation)
        augment:           rate and frequency-mask augmentation
        samples_per_epoch: how many windows make an epoch. Without this an
                           "epoch" is one window per map, so a 13k-map corpus
                           yields 13k windows and the learning-rate schedule is
                           calibrated against a near-meaningless unit.
        motif_dropout:     per-dimension motif dropout. See taiko/data/motif.py
                           -- this is what stops the model reading the answer
                           off its own conditioning vector.
    """

    def __init__(
        self,
        reader:            ShardReader,
        indices:           Sequence[int] | None = None,
        window_frames:     int   = WINDOW_FRAMES_DEFAULT,
        random_window:     bool  = True,
        augment:           bool  = False,
        samples_per_epoch: Optional[int] = None,
        motif_dropout:     float = MOTIF_DIM_DROPOUT,
        motif_jitter:      float = MOTIF_JITTER,
        rate_p:            float = 0.2,
        rate_range:        tuple[float, float] = (0.9, 1.1),
        freq_mask_p:       float = 0.15,
        freq_mask_bands:   int   = 12,
        seed:              int   = 0,
    ):
        self.reader        = reader
        self.indices       = list(indices) if indices is not None else list(range(len(reader)))
        self.window_frames = int(window_frames)
        self.random_window = random_window
        self.augment       = augment
        self.motif_dropout = motif_dropout
        self.motif_jitter  = motif_jitter
        self.rate_p        = rate_p
        self.rate_range    = rate_range
        self.freq_mask_p   = freq_mask_p
        self.freq_mask_bands = freq_mask_bands
        self.seed          = seed

        if not self.indices:
            raise ValueError("WindowedDataset got an empty split")

        if self.window_frames % WINDOW_ALIGNMENT != 0:
            raise ValueError(
                f"window_frames must be a multiple of {WINDOW_ALIGNMENT} so it "
                f"round-trips through the autoencoder; got {self.window_frames}. "
                f"Nearest valid: "
                f"{round(self.window_frames / WINDOW_ALIGNMENT) * WINDOW_ALIGNMENT}"
            )

        self._length = int(samples_per_epoch) if samples_per_epoch else len(self.indices)
        self._checked_alignment = False

    def __len__(self) -> int:
        return self._length

    # ---------------------------------------------------------------- #

    def __getitem__(self, i: int) -> dict:
        # Each worker and each epoch needs its own stream, or every worker
        # draws the same windows.
        rng = np.random.default_rng(
            (self.seed, i, torch.initial_seed() & 0xFFFF_FFFF)
        )
        py_rng = random.Random(int(rng.integers(0, 2**31 - 1)))

        if self._length == len(self.indices) and not self.random_window:
            idx = self.indices[i % len(self.indices)]
        else:
            idx = self.indices[int(rng.integers(0, len(self.indices)))]

        for _ in range(4):
            try:
                return self._load(idx, rng, py_rng)
            except Exception as exc:                       # noqa: BLE001
                print(f"[WindowedDataset] map {idx} failed: {exc!r}")
                idx = self.indices[int(rng.integers(0, len(self.indices)))]

        return self._empty()

    # ---------------------------------------------------------------- #

    def _load(self, idx: int, rng: np.random.Generator, py_rng: random.Random) -> dict:
        reader = self.reader
        W      = self.window_frames

        chart_len = reader.chart_length(idx)
        mel_len   = reader.mel_length(idx)

        if not self._checked_alignment:
            assert_aligned(mel_len, chart_len, context=f"map {idx}")
            self._checked_alignment = True

        # The chart may legitimately end before the audio; never sample a window
        # that starts past the last note.
        max_start = max(0, chart_len - W)
        start = int(rng.integers(0, max_start + 1)) if (self.random_window and max_start > 0) else 0

        # One start, three reads. This is the whole alignment contract.
        mel    = reader.mel_window(idx,    start, W)
        chart  = reader.chart_window(idx,  start, W)
        timing = reader.timing_window(idx, start, W)

        valid_len  = max(0, min(W, chart_len - start))
        valid_mask = np.zeros(W, dtype=np.float32)
        valid_mask[:valid_len] = 1.0

        if self.augment and py_rng.random() < self.rate_p:
            rate = py_rng.uniform(*self.rate_range)
            mel, chart, timing, valid_mask = _rate_augment(mel, chart, timing, valid_mask, rate)

        if self.augment and py_rng.random() < self.freq_mask_p:
            bands = py_rng.randint(1, self.freq_mask_bands)
            lo    = py_rng.randint(0, MEL_BINS - bands)
            mel[lo:lo + bands, :] = 0.0

        record = reader.records[idx]

        difficulty = float(record.get("difficulty", 0.0))
        avg_nps    = float(record.get("avg_nps", 0.0))
        peak_nps   = float(record.get("peak_nps", 0.0))
        style      = int(record.get("style", STYLE_NULL))

        if self.augment:
            # Small jitter so the model does not treat these as exact lookups.
            difficulty = max(0.0, difficulty + py_rng.gauss(0, 0.1))
            avg_nps    = max(0.0, avg_nps    + py_rng.gauss(0, 0.2))
            peak_nps   = max(0.0, peak_nps   + py_rng.gauss(0, 0.3))

        # The motif is measured on the window the model must generate, so it is
        # corrupted before the model ever sees it.
        beat_frames = beat_frames_from_timing(timing)
        motif = compute_motif(chart, beat_frames, quantise=True)
        if self.augment and self.motif_dropout > 0:
            motif, motif_mask = corrupt_motif(
                motif, rng, dim_dropout=self.motif_dropout, jitter=self.motif_jitter
            )
        else:
            motif_mask = np.ones(MOTIF_DIM, dtype=np.float32)

        onsets     = sum(int((chart[c] > 0.5).sum()) for c in ONSET_CHANNELS)
        window_sec = max(valid_mask.sum() * FRAME_MS / 1000.0, 1e-6)

        return {
            "mel":         torch.from_numpy(mel),
            "chart":       torch.from_numpy(chart),
            "timing":      torch.from_numpy(timing),
            "valid_mask":  torch.from_numpy(valid_mask),
            "difficulty":  torch.tensor(normalise_difficulty(difficulty), dtype=torch.float32),
            "style":       torch.tensor(style, dtype=torch.long),
            "avg_nps":     torch.tensor(normalise_avg_nps(avg_nps),  dtype=torch.float32),
            "peak_nps":    torch.tensor(normalise_peak_nps(peak_nps), dtype=torch.float32),
            "motif":       torch.from_numpy(motif),
            "motif_mask":  torch.from_numpy(motif_mask),
            "local_nps":   torch.tensor(onsets / window_sec, dtype=torch.float32),
            "map_index":   torch.tensor(idx, dtype=torch.long),
        }

    def _empty(self) -> dict:
        W = self.window_frames
        return {
            "mel":        torch.zeros(MEL_BINS, W),
            "chart":      torch.zeros(N_CHART_CHANNELS, W),
            "timing":     torch.zeros(N_TIMING_CHANNELS, W),
            "valid_mask": torch.zeros(W),
            "difficulty": torch.tensor(0.0),
            "style":      torch.tensor(STYLE_NULL, dtype=torch.long),
            "avg_nps":    torch.tensor(0.0),
            "peak_nps":   torch.tensor(0.0),
            "motif":      torch.zeros(MOTIF_DIM),
            "motif_mask": torch.zeros(MOTIF_DIM),
            "local_nps":  torch.tensor(0.0),
            "map_index":  torch.tensor(-1, dtype=torch.long),
        }


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #

def _rate_augment(
    mel: np.ndarray,
    chart: np.ndarray,
    timing: np.ndarray,
    valid_mask: np.ndarray,
    rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample all four arrays by the same factor, then pad or crop back to width.

    Chart uses nearest-neighbour so onsets stay crisp single frames; mel and
    timing use linear because both are continuous signals.

    The timing phasor then needs renormalising. Linear interpolation between two
    points on the unit circle traces the chord rather than the arc, so the radius
    collapses toward the origin between samples -- by cos(dtheta/2), which at
    150 BPM is a 2.5% error every frame. That silently gave the model a beat grid
    of wobbling magnitude on the 20% of samples this fires for, while inference
    always supplies an exact one. The angle survives interpolation to second
    order; only the length has to be put back.
    """
    W = mel.shape[1]
    new_w = max(32, int(round(W / rate)))

    def resample(arr: np.ndarray, mode: str) -> np.ndarray:
        t = torch.from_numpy(arr).unsqueeze(0)
        if mode == "linear":
            out = F.interpolate(t, size=new_w, mode="linear", align_corners=False)
        else:
            out = F.interpolate(t, size=new_w, mode="nearest")
        return out[0].numpy()

    mel    = resample(mel,    "linear")
    chart  = resample(chart,  "nearest")
    timing = resample(timing, "linear")
    radius = np.sqrt(timing[TM_SIN] ** 2 + timing[TM_COS] ** 2)
    ok = radius > 1e-6
    timing[TM_SIN] = np.where(ok, timing[TM_SIN] / np.where(ok, radius, 1.0), 0.0)
    timing[TM_COS] = np.where(ok, timing[TM_COS] / np.where(ok, radius, 1.0), 0.0)

    valid  = resample(valid_mask[None, :], "nearest")[0]

    def fit(arr: np.ndarray) -> np.ndarray:
        if arr.shape[-1] >= W:
            return np.ascontiguousarray(arr[..., :W])
        pad = np.zeros(arr.shape[:-1] + (W - arr.shape[-1],), dtype=arr.dtype)
        return np.ascontiguousarray(np.concatenate([arr, pad], axis=-1))

    return fit(mel), fit(chart), fit(timing), fit(valid)


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #

def split_indices(
    reader: ShardReader,
    val_ratio: float = 0.05,
    seed: int = 42,
    ranked_only: bool = False,
) -> tuple[list[int], list[int]]:
    """
    Split by *song*, not by map.

    A beatmapset's difficulties share one audio file. Splitting by map would put
    the Muzukashii of a song in train and its Oni in validation, so validation
    would measure memorisation of songs the model has already heard rather than
    generalisation to new ones -- and would report a val loss far better than
    the real one.
    """
    by_song: dict[str, list[int]] = {}
    for idx, record in enumerate(reader.records):
        if ranked_only and not record.get("ranked", False):
            continue
        by_song.setdefault(record["mel_key"], []).append(idx)

    songs = sorted(by_song)
    rng = random.Random(seed)
    rng.shuffle(songs)

    n_val = max(1, int(len(songs) * val_ratio))
    val_songs = set(songs[:n_val])

    train_idx = [i for s in songs if s not in val_songs for i in by_song[s]]
    val_idx   = [i for s in songs if s in val_songs     for i in by_song[s]]
    return train_idx, val_idx


def print_split_stats(reader: ShardReader, indices: Sequence[int], label: str) -> None:
    from taiko.data.conditioning import style_to_name

    if not indices:
        print(f"{label}: empty")
        return

    styles: dict[str, int] = {}
    diffs, npss, songs = [], [], set()

    for i in indices:
        r = reader.records[i]
        name = style_to_name(r.get("style", STYLE_NULL))
        styles[name] = styles.get(name, 0) + 1
        diffs.append(float(r.get("difficulty", 0.0)))
        if r.get("avg_nps", 0):
            npss.append(float(r["avg_nps"]))
        songs.add(r["mel_key"])

    ranked = sum(1 for i in indices if reader.records[i].get("ranked"))
    print(f"{label}: {len(indices)} maps over {len(songs)} songs  ({ranked} ranked)")
    for name, count in sorted(styles.items(), key=lambda kv: -kv[1]):
        print(f"    {name:<10s} {count:>6d}  ({count / len(indices) * 100:5.1f}%)")
    if diffs:
        print(f"    SR  {min(diffs):.1f} - {max(diffs):.1f}  (mean {sum(diffs)/len(diffs):.2f})")
    if npss:
        print(f"    NPS {min(npss):.1f} - {max(npss):.1f}  (mean {sum(npss)/len(npss):.2f})")


def load_reader(shard_dir: str | Path = "data/processed/shards") -> ShardReader:
    return ShardReader(shard_dir)
