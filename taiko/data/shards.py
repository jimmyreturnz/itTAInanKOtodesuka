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

Reading mels: mmap or pread
---------------------------
`ShardReader` can serve windows either way, and on a memory-capped machine the
choice decides whether the run finishes.

A memmap's touched pages are resident pages: they count in RSS and against a
container's memory limit exactly like an allocation does. Sampling random
windows walks the whole file eventually, so RSS climbs by the full size of
mels.dat over the first hour or two of training and then the run dies -- slowly
enough to look like a leak, and with nothing in the Python heap to blame.
Measured on a 732 MB file: 20k random windows via mmap grew RSS by 732 MB; the
same windows via `os.pread` grew it by 3 MB.

So `mel_io="read"` issues one pread per window -- the layout above makes that a
single contiguous read -- and holds no pages at all. `mel_io="mmap"` is faster
when the corpus comfortably fits in RAM, which is the case locally and is not
the case on Kaggle. `"auto"` picks by comparing the file against the memory the
machine actually has.

"read" also advises the kernel to drop each window's pages once it has them.
Nothing is resident either way, but the pages still pass through the page
cache, and inside a container that cache is charged to the cgroup: random
windows walk the whole corpus, so a 6.7 GB mel file settles 6.7 GB of cache
against a limit under 30 GB. It is reclaimable rather than lost, but it is
still the difference between a memory figure that describes the run and one
that describes the file, and random access over a corpus that size gets no
benefit from caching it.
"""

from __future__ import annotations

import json
import os
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

MEL_IO_MODES = ("auto", "mmap", "read")

# Above this share of total RAM, "auto" stops mapping the mel file. A memmap
# that fits in a fraction of memory is free speed; one that approaches the
# limit is the run's cause of death.
MEL_MMAP_RAM_FRACTION = 0.25


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

def _total_ram_bytes() -> int:
    """Total system RAM, or 0 where it cannot be read."""
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return 0


class ShardReader:
    """
    Random-access reader over a packed dataset.

    Safe to construct before forking dataloader workers: the mel file is opened
    lazily per process, so workers do not share a handle or a mapping.

    Args:
        mel_io: how mel windows are read -- see the module docstring.
            "read" issues one pread per window and holds no pages; "mmap" maps
            the file and is faster while it fits in RAM; "auto" maps only when
            the file is small relative to the machine.
    """

    def __init__(self, shard_dir: str | Path, mel_io: str = "auto",
                 drop_page_cache: bool | None = None):
        self.dir = Path(shard_dir)
        if mel_io not in MEL_IO_MODES:
            raise ValueError(f"mel_io must be one of {MEL_IO_MODES}, got {mel_io!r}")
        self.mel_io_requested = mel_io
        self._drop_page_cache_requested = drop_page_cache
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
        self._mel_fd: int | None = None
        self._owner_pid: int | None = None
        self.mel_io = self._resolve_mel_io(mel_io)
        # "read" is chosen precisely when the corpus does not comfortably fit in
        # memory, which is also exactly when caching it is pointless. Default the
        # advice on in that mode and leave it off under mmap, where the page
        # cache *is* the mechanism.
        self.drop_page_cache = (
            self.mel_io == "read" if drop_page_cache is None else bool(drop_page_cache)
        )

    def __len__(self) -> int:
        return len(self.records)

    # -- mel access -------------------------------------------------------- #

    @property
    def mel_bytes(self) -> int:
        return self.mel_frames * MEL_BINS * 2

    def _resolve_mel_io(self, requested: str) -> str:
        if requested != "auto":
            return requested
        ram = _total_ram_bytes()
        if ram and self.mel_bytes <= ram * MEL_MMAP_RAM_FRACTION:
            return "mmap"
        return "read"

    def describe_mel_io(self) -> str:
        ram = _total_ram_bytes()
        ram_text = f"{ram / 1024 ** 3:.1f} GB RAM" if ram else "RAM unknown"
        note = ("resident pages grow to the size of the file"
                if self.mel_io == "mmap" else
                "one pread per window, no resident pages"
                + (", page cache dropped after each read" if self.drop_page_cache else ""))
        return (f"mel I/O: {self.mel_io} ({self.mel_bytes / 1024 ** 3:.2f} GB mels.dat, "
                f"{ram_text}) -- {note}")

    def _reset_if_forked(self) -> None:
        """Drop handles inherited across a fork so each worker opens its own."""
        pid = os.getpid()
        if self._owner_pid == pid:
            return
        self._mel = None
        self._mel_fd = None
        self._owner_pid = pid

    @property
    def mel(self) -> np.ndarray:
        self._reset_if_forked()
        if self._mel is None:
            self._mel = np.memmap(
                self.dir / MEL_FILENAME, dtype=np.float16, mode="r",
                shape=(self.mel_frames, MEL_BINS),
            )
        return self._mel

    @property
    def _fd(self) -> int:
        self._reset_if_forked()
        if self._mel_fd is None:
            self._mel_fd = os.open(self.dir / MEL_FILENAME, os.O_RDONLY)
        return self._mel_fd

    def _read_frames(self, frame_start: int, count: int) -> np.ndarray:
        """[count, 128] float32, read without mapping the file."""
        row_bytes = MEL_BINS * 2
        want = count * row_bytes
        offset = frame_start * row_bytes
        chunks, got = [], 0
        while got < want:
            block = os.pread(self._fd, want - got, offset + got)
            if not block:
                break                              # Short file; caller zero-pads.
            chunks.append(block)
            got += len(block)
        if self.drop_page_cache and got:
            self._forget(offset, got)
        raw = chunks[0] if len(chunks) == 1 else b"".join(chunks)
        frames = len(raw) // row_bytes
        return np.frombuffer(raw, dtype=np.float16, count=frames * MEL_BINS) \
                 .reshape(frames, MEL_BINS).astype(np.float32)

    def _forget(self, offset: int, length: int) -> None:
        """
        Tell the kernel the pages just read will not be wanted again.

        pread keeps nothing resident, but the pages it reads through still
        land in the page cache -- and inside a container that cache is charged
        to the cgroup. Sampling random windows walks the whole corpus, so a
        6.7 GB mel file eventually parks 6.7 GB of cache against a limit of
        under 30, on the one machine that cannot spare it. The kernel will
        reclaim it rather than kill anything, but it also drives the memory
        figure a supervisor reads, and it leaves that much less room for the
        allocations that genuinely cannot be reclaimed.

        Caching buys nothing here anyway: windows are drawn at random from a
        corpus far larger than the cache, so a page read now is unlikely to be
        wanted before it is evicted. Failures are ignored -- this is an
        optimisation, and a kernel that will not take the advice is not a
        reason to stop training.
        """
        try:
            os.posix_fadvise(self._fd, offset, length, os.POSIX_FADV_DONTNEED)
        except (OSError, AttributeError, ValueError):
            self.drop_page_cache = False           # Not supported here; stop asking.

    def mel_window(self, idx: int, start: int, width: int) -> np.ndarray:
        """
        [128, width] float32 mel window, zero-padded past the end of the audio.
        """
        offset, length = self.mel_spans[self.records[idx]["mel_key"]]
        out = np.zeros((width, MEL_BINS), dtype=np.float32)

        lo = max(start, 0)
        hi = min(start + width, length)
        if hi > lo:
            if self.mel_io == "mmap":
                chunk = self.mel[offset + lo: offset + hi].astype(np.float32)
            else:
                chunk = self._read_frames(offset + lo, hi - lo)
            out[lo - start: lo - start + chunk.shape[0]] = chunk

        return np.ascontiguousarray(out.T)

    def __getstate__(self) -> dict:
        """Never pickle an open handle to a dataloader worker."""
        state = dict(self.__dict__)
        state["_mel"] = None
        state["_mel_fd"] = None
        state["_owner_pid"] = None
        return state

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
