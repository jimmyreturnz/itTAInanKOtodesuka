"""
taiko/data/tensor_repr.py

Beatmap <-> Tensor representation for diffusion training.

Tensor shape: [7, T]
  Channel 0: don
  Channel 1: kat
  Channel 2: big_don
  Channel 3: big_kat
  Channel 4: roll         (1.0 for full duration)
  Channel 5: denden       (1.0 for full duration)
  Channel 6: beat         (beat grid from timing points, like Mug-Diffusion)

Frame resolution: FRAME_MS = 20ms
  - 6 min song = 18000 frames (manageable for U-Net)
  - 180 BPM 1/4 note = 83ms = 4 frames apart (safe)
  - 180 BPM 1/8 note = 41ms = 2 frames apart (acceptable)

Audio alignment:
  audio.py must use HOP_LENGTH = 441 samples @ 22050 Hz = 20ms per frame
  so mel frame i == beatmap tensor frame i exactly.

Beat channel follows Mug-Diffusion convertor.py timing_to_array():
  - walks all timing points (red + green lines)
  - normalizes BPM to 150-300 range
  - marks every half-beat
  - stores sub-frame offset for precision
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional

from taiko.data.osu_parser import TaikoBeatmap, TaikoNote, TimingPoint


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRAME_MS   = 20.0      # ms per frame — must match audio.py
N_CHANNELS = 7
MAX_FRAMES = 45_000    # 900 seconds @ 20ms

CH_DON     = 0
CH_KAT     = 1
CH_BIG_DON = 2
CH_BIG_KAT = 3
CH_ROLL    = 4
CH_DENDEN  = 5
CH_BEAT    = 6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ms_to_frame(ms: float) -> int:
    return int(round(ms / FRAME_MS))

def frame_to_ms(frame: int) -> float:
    return frame * FRAME_MS


# ---------------------------------------------------------------------------
# Beat channel — Mug-Diffusion approach
# ---------------------------------------------------------------------------

def build_beat_channel(
    timing_points: list,
    n_frames: int,
) -> np.ndarray:
    """
    Build beat grid channel from timing points.
    Replicates Mug-Diffusion timing_to_array() logic.
    Optimized: only uses red lines (BPM changes), ignores green lines (SV).
    Green lines dont change the beat grid, only SV — safe to skip.
    """
    channel = np.zeros(n_frames, dtype=np.float32)

    if not timing_points:
        return channel

    # Only use red lines (uninherited) — ignore green SV lines entirely
    red_lines = [
        (float(tp.time), 60_000.0 / tp.beat_length)
        for tp in timing_points
        if tp.uninherited and tp.beat_length > 0
    ]

    if not red_lines:
        return channel

    for i, (start_ms, bpm) in enumerate(red_lines):
        end_ms = red_lines[i + 1][0] if i < len(red_lines) - 1 else n_frames * FRAME_MS

        # Normalize BPM to 150-300 range
        normalized_bpm = bpm
        while normalized_bpm < 150:
            normalized_bpm *= 2
        while normalized_bpm >= 300:
            normalized_bpm /= 2

        half_beat_ms = 60_000.0 / normalized_bpm / 2
        if half_beat_ms < 5.0:  # guard: skip BPM > 6000
            continue

        # Use numpy arange instead of while loop — much faster
        beat_positions = np.arange(start_ms, end_ms, half_beat_ms)
        frame_indices  = (beat_positions / FRAME_MS).astype(int)
        offsets        = (beat_positions / FRAME_MS) - frame_indices
        values         = (1.0 - offsets * 0.5).astype(np.float32)

        valid = (frame_indices >= 0) & (frame_indices < n_frames)
        np.maximum.at(channel, frame_indices[valid], values[valid])

    return channel


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def beatmap_to_tensor(
    bm: TaikoBeatmap,
    max_frames: int = MAX_FRAMES,
    pad_to: Optional[int] = None,
) -> np.ndarray:
    """
    Convert TaikoBeatmap to float32 tensor [7, T].

    Returns:
        np.ndarray [7, T], float32, values in [0, 1]
    """
    if bm.notes:
        last_time = max(n.end_time if n.is_long else n.time for n in bm.notes)
    else:
        last_time = 0

    n_frames = min(ms_to_frame(last_time) + 50, max_frames)
    if pad_to is not None:
        n_frames = pad_to

    tensor = np.zeros((N_CHANNELS, n_frames), dtype=np.float32)

    # ---- Note channels -------------------------------------------------- #
    for note in bm.notes:
        sf = ms_to_frame(note.time)
        if sf >= n_frames:
            continue

        if note.note_type == "don":
            tensor[CH_DON, sf] = 1.0

        elif note.note_type == "kat":
            tensor[CH_KAT, sf] = 1.0

        elif note.note_type == "big_don":
            tensor[CH_BIG_DON, sf] = 1.0

        elif note.note_type == "big_kat":
            tensor[CH_BIG_KAT, sf] = 1.0

        elif note.note_type == "roll":
            tensor[CH_DON, sf] = 1.0
            ef = min(ms_to_frame(note.end_time), n_frames - 1)
            tensor[CH_ROLL, sf:ef + 1] = 1.0

        elif note.note_type == "denden":
            tensor[CH_DON, sf] = 1.0
            ef = min(ms_to_frame(note.end_time), n_frames - 1)
            tensor[CH_DENDEN, sf:ef + 1] = 1.0

    # ---- Beat channel (Mug-Diffusion approach) -------------------------- #
    tensor[CH_BEAT] = build_beat_channel(bm.timing_points, n_frames)

    return tensor


def tensor_to_beatmap(
    tensor: np.ndarray,
    bpm: float,
    offset_ms: float,
    threshold: float = 0.5,
    min_long_frames: int = 3,
    title: str = "AI Generated",
    artist: str = "",
    version: str = "AI",
    audio_filename: str = "audio.mp3",
    overall_difficulty: float = 5.0,
) -> TaikoBeatmap:
    """Convert tensor [7, T] back to TaikoBeatmap."""
    bm = TaikoBeatmap()
    bm.title             = title
    bm.artist            = artist
    bm.creator           = "TaikoAI"
    bm.version           = version
    bm.audio_filename    = audio_filename
    bm.overall_difficulty = overall_difficulty
    bm.hp_drain          = min(10.0, overall_difficulty * 0.8)
    bm.slider_multiplier = 1.4
    bm.slider_tick_rate  = 1.0
    bm.approach_rate     = overall_difficulty

    bm.timing_points = [TimingPoint(
        time=int(offset_ms),
        beat_length=60_000.0 / bpm,
        meter=4,
        uninherited=True,
    )]

    n_frames = tensor.shape[1]
    notes    = []

    # Simple note channels
    for ch, note_type in [
        (CH_DON,     "don"),
        (CH_KAT,     "kat"),
        (CH_BIG_DON, "big_don"),
        (CH_BIG_KAT, "big_kat"),
    ]:
        for frame in _find_onsets(tensor[ch], threshold):
            if tensor[CH_ROLL, frame] > threshold or tensor[CH_DENDEN, frame] > threshold:
                continue
            notes.append(TaikoNote(time=int(frame_to_ms(frame)), note_type=note_type))

    # Roll regions
    for sf, ef in _find_regions(tensor[CH_ROLL], threshold):
        if ef - sf < min_long_frames:
            continue
        notes.append(TaikoNote(
            time=int(frame_to_ms(sf)),
            note_type="roll",
            end_time=int(frame_to_ms(ef)),
        ))

    # Denden regions
    for sf, ef in _find_regions(tensor[CH_DENDEN], threshold):
        if ef - sf < min_long_frames:
            continue
        notes.append(TaikoNote(
            time=int(frame_to_ms(sf)),
            note_type="denden",
            end_time=int(frame_to_ms(ef)),
        ))

    notes.sort(key=lambda n: n.time)

    # Deduplicate within one frame
    deduped, last_t = [], -999
    for note in notes:
        if note.time - last_t >= int(FRAME_MS):
            deduped.append(note)
            last_t = note.time

    bm.notes = deduped
    bm.compute_stats()
    return bm


def _find_onsets(ch: np.ndarray, threshold: float) -> list[int]:
    above = ch > threshold
    return [i for i in range(len(above)) if above[i] and (i == 0 or not above[i-1])]


def _find_regions(ch: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    regions, above = [], ch > threshold
    in_r, start = False, 0
    for i in range(len(above)):
        if above[i] and not in_r:
            start, in_r = i, True
        elif not above[i] and in_r:
            regions.append((start, i - 1))
            in_r = False
    if in_r:
        regions.append((start, len(above) - 1))
    return regions


# ---------------------------------------------------------------------------
# Round-trip accuracy
# ---------------------------------------------------------------------------

def round_trip_accuracy(bm: TaikoBeatmap) -> dict:
    # Get primary BPM from timing points
    bpm, offset_ms = 120.0, 0.0
    for tp in bm.timing_points:
        if tp.uninherited and tp.beat_length > 0:
            bpm = 60_000.0 / tp.beat_length
            offset_ms = float(tp.time)
            break

    tensor = beatmap_to_tensor(bm)
    bm2    = tensor_to_beatmap(tensor, bpm, offset_ms,
                                title=bm.title, version=bm.version,
                                overall_difficulty=bm.overall_difficulty)

    orig_times  = set(n.time for n in bm.notes)
    recon_times = set(n.time for n in bm2.notes)

    recovered = sum(1 for t in orig_times if any(abs(t-r) <= FRAME_MS for r in recon_times))
    false_pos = sum(1 for r in recon_times if not any(abs(r-t) <= FRAME_MS for t in orig_times))

    return {
        "original_notes":      len(orig_times),
        "reconstructed_notes": len(recon_times),
        "recovered":           recovered,
        "recall":              recovered / max(len(orig_times), 1),
        "precision":           (len(recon_times) - false_pos) / max(len(recon_times), 1),
        "false_positives":     false_pos,
        "tensor_shape":        tensor.shape,
        "tensor_kb":           tensor.nbytes / 1024,
    }


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_tensor(tensor: np.ndarray, path: str | Path):
    np.savez_compressed(str(path), tensor=tensor)

def load_tensor(path: str | Path) -> np.ndarray:
    return np.load(str(path))["tensor"]
