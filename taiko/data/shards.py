"""
taiko/data/shards.py

Packed dataset format.

Thirteen thousand individually-compressed .npz files, decompressed once per
sample, is a CPU bottleneck that starves the GPU -- and on Kaggle it is also a
slow, fragile upload. This module packs the corpus into three files:

    mels.dat        raw float16 memmap, [total_frames, 128]
    charts.npz      sparse note events, concatenated with per-map offsets
    index.json      metadata and offsets for every map

Three decisions worth stating, because each is a large multiplier:

  float16 mels. Mel values live in [-1, 1] and feed a network that trains in
  fp16 anyway, so the second byte buys nothing. Halves both the file and the
  bytes read per batch.

  Sparse charts. A taiko chart is about 99.5% zeros. Storing 6 x T floats per
  map wastes gigabytes to encode silence; storing (frame, channel) events and
  (start, end, channel) spans costs a few kilobytes per map, and the dense
  window is rebuilt on the fly in microseconds.

  Timing points, not timing streams. A dense [3, T] stream per map would be
  larger than the chart it accompanies. Storing the handful of red lines and
  rebuilding the stream for the requested window is exact and effectively free.

Time-major mel layout ([frames, mels], not [mels, frames]) so that reading a
window is one contiguous slice rather than 128 strided reads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

from taiko.data.frames import FRAME_MS
from taiko.data.osu_parser import TimingPoint
from taiko.data.tensor_repr import (
    HOLD_CHANNELS,
    N_CHART_CHANNELS,
    ONSET_CHANNELS,
    build_timing_stream,
)

MEL_BINS      = 128
MEL_FILENAME   = "mels.dat"
CHART_FILENAME = "charts.npz"
INDEX_FILENAME = "index.json"

SHARD_FORMAT_VERSION = 1


# --------------------------------------------------------------------------- #
# Sparse chart encoding
# --------------------------------------------------------------------------- #

def encode_chart(chart: np.ndarray) -> dict[str, np.ndarray]:
    """
    Dense [6, T] chart -> sparse events.

    Onset channels contribute one event per active frame -- not per rising
    edge. Two notes of the same colour on consecutive frames is a 50 NPS
    absurdity that real charts never contain, but storage that silently drops
    data it considers implausible is storage that cannot be trusted. Merging
    adjacent onsets is a decoding decision and lives in `tensor_to_beatmap`.
    """
    onset_frames:  list[int] = []
    onset_channels: list[int] = []
    span_starts:   list[int] = []
    span_ends:     list[int] = []
    span_channels: list[int] = []

    for ch in ONSET_CHANNELS:
        frames = np.flatnonzero(chart[ch] > 0.5)
        if frames.size == 0:
            continue
        onset_frames.extend(frames.tolist())
        onset_channels.extend([ch] * len(frames))

    for ch in HOLD_CHANNELS:
        above = chart[ch] > 0.5
        if not above.any():
            continue
        padded = np.concatenate([[False], above, [False]])
        edges  = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(edges == 1)
        ends   = np.flatnonzero(edges == -1) - 1
        span_starts.extend(starts.tolist())
        span_ends.extend(ends.tolist())
        span_channels.extend([ch] * len(starts))

    return {
        "onset_frames":   np.asarray(onset_frames,   dtype=np.int32),
        "onset_channels": np.asarray(onset_channels, dtype=np.int8),
        "span_starts":    np.asarray(span_starts,    dtype=np.int32),
        "span_ends":      np.asarray(span_ends,      dtype=np.int32),
        "span_channels":  np.asarray(span_channels,  dtype=np.int8),
    }


def decode_chart_window(
    onset_frames:   np.ndarray,
    onset_channels: np.ndarray,
    span_starts:    np.ndarray,
    span_ends:      np.ndarray,
    span_channels:  np.ndarray,
    start: int,
    width: int,
) -> np.ndarray:
    """
    Rebuild a dense [6, width] window from sparse events.

    Spans are clipped to the window rather than dropped, so a drumroll that
    began before the window still reads as held throughout it -- otherwise the
    model would see a roll appear from nowhere partway through.
    """
    out = np.zeros((N_CHART_CHANNELS, width), dtype=np.float32)
    stop = start + width

    if onset_frames.size:
        sel = (onset_frames >= start) & (onset_frames < stop)
        if sel.any():
            out[onset_channels[sel].astype(np.int64),
                (onset_frames[sel] - start).astype(np.int64)] = 1.0

    if span_starts.size:
        sel = (span_ends >= start) & (span_starts < stop)
        for s, e, ch in zip(span_starts[sel], span_ends[sel], span_channels[sel]):
            lo = max(int(s) - start, 0)
            hi = min(int(e) - start, width - 1)
            if hi >= lo:
                out[int(ch), lo:hi + 1] = 1.0

    return out


# --------------------------------------------------------------------------- #
# Timing point packing
# --------------------------------------------------------------------------- #

def encode_timing_points(timing_points: list[TimingPoint]) -> list[list[float]]:
    """Red lines only, as [time_ms, beat_length_ms, meter] triples."""
    return [
        [float(tp.time), float(tp.beat_length), float(max(1, tp.meter))]
        for tp in timing_points
        if tp.uninherited and tp.beat_length > 0
    ]


def decode_timing_points(packed: list[list[float]]) -> list[TimingPoint]:
    return [
        TimingPoint(
            time=int(t), beat_length=float(bl),
            meter=int(m), uninherited=True,
        )
        for t, bl, m in packed
    ]


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #

class ShardWriter:
    """
    Builds a packed dataset incrementally.

    Mels are shared between the difficulties of one beatmapset, so
    `add_mel` is keyed by beatmapset and `add_map` references that key.

    Usage:
        with ShardWriter(out_dir) as w:
            w.add_mel("Some Song Folder", mel)          # [128, T] float32
            w.add_map(record, chart, timing_points, mel_key="Some Song Folder")
    """

    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._mel_file   = open(self.out_dir / MEL_FILENAME, "wb")
        self._mel_frames = 0
        self._mel_spans: dict[str, list[int]] = {}

        self._records: list[dict] = []
        self._chart_arrays: dict[str, list[np.ndarray]] = {
            key: [] for key in
            ("onset_frames", "onset_channels", "span_starts", "span_ends", "span_channels")
        }
        self._onset_offsets = [0]
        self._span_offsets  = [0]

    # -- mels ------------------------------------------------------------- #

    def has_mel(self, key: str) -> bool:
        return key in self._mel_spans

    def add_mel(self, key: str, mel: np.ndarray) -> None:
        """Append a [128, T] mel. Writing the same key twice is a no-op."""
        if key in self._mel_spans:
            return
        if mel.ndim != 2 or mel.shape[0] != MEL_BINS:
            raise ValueError(f"expected mel [{MEL_BINS}, T], got {mel.shape}")

        # Time-major on disk so a window read is contiguous.
        payload = np.ascontiguousarray(mel.T, dtype=np.float16)
        self._mel_file.write(payload.tobytes())

        self._mel_spans[key] = [self._mel_frames, mel.shape[1]]
        self._mel_frames += mel.shape[1]

    # -- charts ----------------------------------------------------------- #

    def add_map(
        self,
        record: dict,
        chart: np.ndarray,
        timing_points: list[TimingPoint],
        mel_key: str,
    ) -> None:
        if mel_key not in self._mel_spans:
            raise KeyError(f"mel {mel_key!r} must be added before its maps")

        events = encode_chart(chart)
        for key, arr in events.items():
            self._chart_arrays[key].append(arr)
        self._onset_offsets.append(self._onset_offsets[-1] + events["onset_frames"].size)
        self._span_offsets.append(self._span_offsets[-1] + events["span_starts"].size)

        row = dict(record)
        row["mel_key"]       = mel_key
        row["chart_frames"]  = int(chart.shape[1])
        row["timing_points"] = encode_timing_points(timing_points)
        self._records.append(row)

    # -- finish ----------------------------------------------------------- #

    def close(self) -> dict:
        self._mel_file.close()

        def cat(key: str, dtype) -> np.ndarray:
            parts = self._chart_arrays[key]
            if not parts:
                return np.empty(0, dtype=dtype)
            return np.concatenate(parts).astype(dtype)

        np.savez_compressed(
            self.out_dir / CHART_FILENAME,
            onset_frames   = cat("onset_frames",   np.int32),
            onset_channels = cat("onset_channels", np.int8),
            span_starts    = cat("span_starts",    np.int32),
            span_ends      = cat("span_ends",      np.int32),
            span_channels  = cat("span_channels",  np.int8),
            onset_offsets  = np.asarray(self._onset_offsets, dtype=np.int64),
            span_offsets   = np.asarray(self._span_offsets,  dtype=np.int64),
        )

        index = {
            "version":     SHARD_FORMAT_VERSION,
            "frame_ms":    FRAME_MS,
            "mel_bins":    MEL_BINS,
            "mel_frames":  self._mel_frames,
            "mel_spans":   self._mel_spans,
            "records":     self._records,
        }
        (self.out_dir / INDEX_FILENAME).write_text(
            json.dumps(index, ensure_ascii=False), encoding="utf-8"
        )

        return {
            "maps":       len(self._records),
            "songs":      len(self._mel_spans),
            "mel_frames": self._mel_frames,
            "mel_gb":     self._mel_frames * MEL_BINS * 2 / 1024 ** 3,
            "onsets":     self._onset_offsets[-1],
            "spans":      self._span_offsets[-1],
        }

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc) -> None:
        if not self._mel_file.closed:
            self.close()


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #

class ShardReader:
    """
    Random-access reader over a packed dataset.

    Safe to construct before forking dataloader workers: the memmap is opened
    lazily per process, so workers do not share a file handle.
    """

    def __init__(self, shard_dir: str | Path):
        self.dir = Path(shard_dir)
        index_path = self.dir / INDEX_FILENAME
        if not index_path.exists():
            raise FileNotFoundError(
                f"no packed dataset at {self.dir}\n"
                f"  Run: python scripts/pack_dataset.py"
            )

        index = json.loads(index_path.read_text(encoding="utf-8"))
        if index.get("version") != SHARD_FORMAT_VERSION:
            raise ValueError(
                f"shard format v{index.get('version')} but this build expects "
                f"v{SHARD_FORMAT_VERSION}; repack the dataset"
            )
        if abs(index.get("frame_ms", 0) - FRAME_MS) > 1e-9:
            raise ValueError(
                f"packed at {index.get('frame_ms')} ms/frame but this build "
                f"uses {FRAME_MS} ms/frame; repack the dataset"
            )

        self.records    = index["records"]
        self.mel_spans  = index["mel_spans"]
        self.mel_frames = index["mel_frames"]

        charts = np.load(self.dir / CHART_FILENAME)
        self._onset_frames   = charts["onset_frames"]
        self._onset_channels = charts["onset_channels"]
        self._span_starts    = charts["span_starts"]
        self._span_ends      = charts["span_ends"]
        self._span_channels  = charts["span_channels"]
        self._onset_offsets  = charts["onset_offsets"]
        self._span_offsets   = charts["span_offsets"]

        self._mel: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.records)

    @property
    def mel(self) -> np.ndarray:
        if self._mel is None:
            self._mel = np.memmap(
                self.dir / MEL_FILENAME, dtype=np.float16, mode="r",
                shape=(self.mel_frames, MEL_BINS),
            )
        return self._mel

    def mel_window(self, idx: int, start: int, width: int) -> np.ndarray:
        """
        [128, width] float32 mel window, zero-padded past the end of the audio.
        """
        offset, length = self.mel_spans[self.records[idx]["mel_key"]]
        out = np.zeros((width, MEL_BINS), dtype=np.float32)

        lo = max(start, 0)
        hi = min(start + width, length)
        if hi > lo:
            chunk = self.mel[offset + lo: offset + hi]
            out[lo - start: hi - start] = chunk.astype(np.float32)

        return np.ascontiguousarray(out.T)

    def mel_length(self, idx: int) -> int:
        return self.mel_spans[self.records[idx]["mel_key"]][1]

    def chart_window(self, idx: int, start: int, width: int) -> np.ndarray:
        o0, o1 = self._onset_offsets[idx], self._onset_offsets[idx + 1]
        s0, s1 = self._span_offsets[idx],  self._span_offsets[idx + 1]
        return decode_chart_window(
            self._onset_frames[o0:o1],
            self._onset_channels[o0:o1],
            self._span_starts[s0:s1],
            self._span_ends[s0:s1],
            self._span_channels[s0:s1],
            start, width,
        )

    def timing_window(self, idx: int, start: int, width: int) -> np.ndarray:
        points = decode_timing_points(self.records[idx]["timing_points"])
        return build_timing_stream(points, width, start_frame=start)

    def chart_length(self, idx: int) -> int:
        return int(self.records[idx]["chart_frames"])

    def onset_count(self, idx: int) -> int:
        return int(self._onset_offsets[idx + 1] - self._onset_offsets[idx])

    def iter_records(self) -> Iterator[dict]:
        return iter(self.records)
