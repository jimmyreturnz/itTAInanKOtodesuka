"""
taiko/data/preprocessed_dataset.py

Two dataset classes:

  PreprocessedDataset  — original full-map class (kept for compatibility)
  WindowedDataset      — NEW: samples random 20-40s windows per __getitem__

WindowedDataset differences from PreprocessedDataset:
  - Random window start each call (training) or fixed (eval)
  - Tensor and mel are cropped to the window, not padded to full map size
  - valid_mask is recomputed for the window (all 1s unless at the map tail)
  - local_nps computed on the fly from onset counts in the window
  - section_pos = window_start / map_total_frames  (float 0-1)
  - motif computed from onset + beat channels in the window (16-dim vector)
  - avg_nps / peak_nps / snap ratios read from the index row

Motif vector (16 dims):
  Bins 0-7:  IOI histogram — inter-onset-interval distribution in beat fractions
             (1/8, 1/6, 1/4, 1/3, 3/8, 1/2, 3/4, 1/1)
  Bin  8:    don fraction  (don / total onsets in window)
  Bin  9:    kat fraction
  Bin  10:   big fraction  (big_don + big_kat) / total
  Bin  11:   roll fraction (roll onset frames / window)
  Bin  12:   local_nps normalised by avg_nps  (density relative to map mean)
  Bin  13:   beat regularity — std of beat channel values (lower = more regular)
  Bin  14:   snap_1_4 in this window
  Bin  15:   snap_1_8 in this window

All values are float32 in [0, 1].

Expected index format (one JSON line per map, produced by preprocess_for_colab.py):
  {
    "mel_path":    "mels/FolderName.npz",
    "tensor_path": "tensors/FolderName__DiffName.npz",
    "difficulty":  5.2,
    "style":       1,
    "avg_nps":     7.3,
    "peak_nps":    12.1,
    "snap_1_4":    0.72,
    "snap_1_6":    0.05,
    "snap_1_8":    0.14,
    "note_count":  677,
    "duration_ms": 171400
  }
"""

from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from taiko.data.tensor_repr import N_CHANNELS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAD_FRAMES  = 18_000   # full-map tensor pad (360s @ 20ms)
MEL_FRAMES  = 36_000   # full-map mel pad    (360s @ 10ms — 2× tensor)

# Window sizes in tensor frames (20ms per frame)
WINDOW_FRAMES_MIN = 1_000   # 20s
WINDOW_FRAMES_MAX = 2_000   # 40s
WINDOW_FRAMES_DEFAULT = 1_500  # 30s

MOTIF_DIM   = 16
FRAME_MS    = 20.0

# Tensor channel indices (must match tensor_repr.py)
CH_DON     = 0
CH_KAT     = 1
CH_BIG_DON = 2
CH_BIG_KAT = 3
CH_ROLL    = 4
CH_DENDEN  = 5
CH_BEAT    = 6

# IOI histogram bin edges in beat fractions (relative to beat_length)
# Each bin captures notes spaced approximately that fraction of a beat apart
IOI_BINS = [1/8, 1/6, 1/4, 1/3, 3/8, 1/2, 3/4, 1.0]
IOI_TOL  = 0.04   # ±4% tolerance


# ---------------------------------------------------------------------------
# Motif computation
# ---------------------------------------------------------------------------

def _compute_motif(
    window: np.ndarray,   # [7, W] float32
    avg_nps: float,
    snap_1_4: float,
    snap_1_8: float,
    frame_ms: float = FRAME_MS,
) -> np.ndarray:
    """
    Compute a 16-dimensional motif vector from a tensor window.
    All values are in [0, 1].
    """
    W         = window.shape[1]
    motif     = np.zeros(MOTIF_DIM, dtype=np.float32)

    # ---- Onset channels -------------------------------------------------- #
    # An onset is a frame where a channel transitions from 0→1
    def onsets(ch: int) -> np.ndarray:
        above = window[ch] > 0.5
        if above.sum() == 0:
            return np.array([], dtype=np.int32)
        # First frame counts as onset if active
        starts = np.where(above & ~np.concatenate([[False], above[:-1]]))[0]
        return starts.astype(np.int32)

    don_on     = onsets(CH_DON)
    kat_on     = onsets(CH_KAT)
    big_don_on = onsets(CH_BIG_DON)
    big_kat_on = onsets(CH_BIG_KAT)
    roll_on    = onsets(CH_ROLL)

    all_on = np.sort(np.concatenate([don_on, kat_on, big_don_on, big_kat_on]))
    total  = max(len(all_on), 1)

    # ---- Beat length estimate from beat channel (CH6) -------------------- #
    # Beat channel has pulses at each half-beat; estimate spacing from IBI
    beat_ch    = window[CH_BEAT]
    beat_peaks = np.where(beat_ch > 0.5)[0]
    if len(beat_peaks) >= 2:
        ibi         = np.diff(beat_peaks)       # inter-beat-interval in frames
        median_ibi  = float(np.median(ibi))
        beat_frames = max(median_ibi * 2, 1.0)  # half-beat → full beat in frames
    else:
        # Fallback: estimate beat length from avg_nps
        # avg_nps notes/sec at 1/4 density → beat_ms = 1000 / (avg_nps / 4)
        beat_ms_fallback = 1000.0 / max(avg_nps / 4.0, 0.5)
        beat_frames = beat_ms_fallback / frame_ms  # convert ms → frames

    # ---- IOI histogram (bins 0-7) ---------------------------------------- #
    if len(all_on) >= 2:
        ioi_frames = np.diff(all_on).astype(float)
        tol = max(1.0, IOI_TOL * beat_frames)   # tolerance in frames
        for b, frac in enumerate(IOI_BINS):
            target  = frac * beat_frames
            matches = int(np.sum(np.abs(ioi_frames - target) <= tol))
            motif[b] = min(matches / max(len(ioi_frames), 1), 1.0)

    # ---- Note type fractions (bins 8-11) --------------------------------- #
    motif[8]  = min(len(don_on)     / total, 1.0)   # don fraction
    motif[9]  = min(len(kat_on)     / total, 1.0)   # kat fraction
    motif[10] = min(
        (len(big_don_on) + len(big_kat_on)) / total, 1.0
    )  # big fraction
    roll_frames = float((window[CH_ROLL] > 0.5).sum())
    motif[11] = min(roll_frames / max(W, 1), 1.0)   # roll density

    # ---- Relative density (bin 12) --------------------------------------- #
    window_sec = W * frame_ms / 1000.0
    local_nps  = len(all_on) / max(window_sec, 1.0)
    motif[12]  = min(local_nps / max(avg_nps * 2.0, 1.0), 1.0)

    # ---- Beat regularity (bin 13) ---------------------------------------- #
    # Low std of beat channel values → more regular beat grid → higher value
    if beat_ch.max() > 0:
        regularity = 1.0 - float(np.std(beat_ch[beat_ch > 0.1]))
        motif[13]  = max(0.0, min(regularity, 1.0))

    # ---- Window-local snap ratios (bins 14-15) --------------------------- #
    # Approximate from IOI distribution: 1/4 snap ≈ bin index 2, 1/8 ≈ bin 0
    motif[14] = float(motif[2])   # snap_1_4 proxy
    motif[15] = float(motif[0])   # snap_1_8 proxy

    return motif


# ---------------------------------------------------------------------------
# WindowedDataset
# ---------------------------------------------------------------------------

class WindowedDataset(Dataset):
    """
    Samples random fixed-size windows from pre-computed mel + tensor pairs.

    Each __getitem__ returns:
        mel        [128, W*2]   — mel crop (2× tensor resolution)
        tensor     [7,   W]     — beatmap tensor crop
        valid_mask [W]          — 1=real frame, 0=past map end
        difficulty float        — star rating
        style      int          — 0-3 global style label
        avg_nps    float        — global NPS (from index)
        peak_nps   float        — peak 5s-window NPS (from index)
        local_nps  float        — NPS in this specific window
        section_pos float       — window start / map length (0-1)
        motif      [16]         — local rhythm fingerprint vector

    Args:
        records      — list of index dicts from load_index()
        data_root    — Path to data/processed/
        window_frames — window size in tensor frames (default 1500 = 30s)
        random_window — if True, randomise start each call (training mode)
                        if False, use fixed start=0 (eval mode)
        augment      — apply rate + freq-mask augmentation (training only)
    """

    def __init__(
        self,
        records:       list[dict],
        data_root:     Path,
        window_frames: int   = WINDOW_FRAMES_DEFAULT,
        random_window: bool  = True,
        augment:       bool  = False,
        rate_p:        float = 0.2,
        rate_range:    tuple = (0.9, 1.1),
        freq_mask_p:   float = 0.15,
        freq_mask_bands: int = 12,
    ):
        self.records        = records
        self.data_root      = Path(data_root)
        self.window_frames  = window_frames
        self.random_window  = random_window
        self.augment        = augment
        self.rate_p         = rate_p
        self.rate_range     = rate_range
        self.freq_mask_p    = freq_mask_p
        self.freq_mask_bands = freq_mask_bands

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        for attempt in range(5):
            rec = self.records[idx]
            try:
                return self._load(rec)
            except Exception as e:
                print(f"[WindowedDataset] failed idx={idx} attempt={attempt}: {e}")
                idx = random.randint(0, len(self.records) - 1)

        # Fallback empty sample
        W = self.window_frames
        return {
            "mel":         torch.zeros(128, W * 2),
            "tensor":      torch.zeros(N_CHANNELS, W),
            "valid_mask":  torch.zeros(W),
            "difficulty":  torch.tensor(5.0),
            "style":       torch.tensor(0, dtype=torch.long),
            "avg_nps":     torch.tensor(5.0),
            "peak_nps":    torch.tensor(8.0),
            "local_nps":   torch.tensor(5.0),
            "section_pos": torch.tensor(0.0),
            "motif":       torch.zeros(MOTIF_DIM),
        }

    def _load(self, rec: dict) -> dict:
        W = self.window_frames

        # ---- Load raw files ---------------------------------------------- #
        mel_path    = self.data_root / rec["mel_path"].replace("\\", "/")
        tensor_path = self.data_root / rec["tensor_path"].replace("\\", "/")

        mel_npz    = np.load(mel_path)
        tensor_npz = np.load(tensor_path)

        mel    = (mel_npz["mel"]    if "mel"    in mel_npz    else mel_npz["arr_0"]).astype(np.float32)
        tensor = (tensor_npz["tensor"] if "tensor" in tensor_npz else tensor_npz["arr_0"]).astype(np.float32)

        T_tensor = tensor.shape[1]   # actual map length in frames
        T_mel    = mel.shape[1]      # actual mel length (≈ 2× T_tensor)

        # ---- Pick window start ------------------------------------------- #
        # Leave at least W frames from the start; allow partial windows at tail
        max_start = max(0, T_tensor - W)

        if self.random_window and max_start > 0:
            start = random.randint(0, max_start)
        else:
            start = 0

        end = start + W

        # ---- Crop tensor -------------------------------------------------- #
        if end <= T_tensor:
            t_crop     = tensor[:, start:end].copy()
            valid_mask = np.ones(W, dtype=np.float32)
        else:
            # Partial window at map tail — pad with zeros
            t_crop     = np.zeros((N_CHANNELS, W), dtype=np.float32)
            valid_len  = T_tensor - start
            t_crop[:, :valid_len] = tensor[:, start:T_tensor]
            valid_mask = np.zeros(W, dtype=np.float32)
            valid_mask[:valid_len] = 1.0

        # ---- Crop mel (2× resolution) ------------------------------------- #
        mel_start = start * 2
        mel_end   = end   * 2

        if mel_end <= T_mel:
            m_crop = mel[:, mel_start:mel_end].copy()
        else:
            m_crop = np.zeros((128, W * 2), dtype=np.float32)
            avail  = max(0, T_mel - mel_start)
            if avail > 0:
                m_crop[:, :avail] = mel[:, mel_start:mel_start + avail]

        # ---- Rate augmentation (applied jointly to tensor + mel) ---------- #
        if self.augment and random.random() < self.rate_p:
            rate  = random.uniform(*self.rate_range)
            new_W = max(32, int(W / rate))
            new_M = new_W * 2

            t_t = torch.from_numpy(t_crop).unsqueeze(0)
            m_t = torch.from_numpy(m_crop).unsqueeze(0)

            t_crop = F.interpolate(t_t, size=new_W, mode="nearest")[0].numpy()
            m_crop = F.interpolate(m_t, size=new_M, mode="linear", align_corners=False)[0].numpy()

            # Re-pad / truncate back to W and W*2
            if new_W < W:
                pad_t  = np.zeros((N_CHANNELS, W - new_W), dtype=np.float32)
                t_crop = np.concatenate([t_crop, pad_t], axis=1)
                pad_m  = np.zeros((128, W * 2 - new_M), dtype=np.float32)
                m_crop = np.concatenate([m_crop, pad_m], axis=1)
                vm_len = int(valid_mask.sum() / rate)
                valid_mask = np.zeros(W, dtype=np.float32)
                valid_mask[:min(vm_len, W)] = 1.0
            else:
                t_crop = t_crop[:, :W]
                m_crop = m_crop[:, :W * 2]

        # ---- Freq-mask augmentation (mel only) ---------------------------- #
        if self.augment and random.random() < self.freq_mask_p:
            f  = random.randint(1, self.freq_mask_bands)
            f0 = random.randint(0, 127 - f)
            m_crop[f0:f0 + f, :] = 0.0

        # ---- Conditioning fields ----------------------------------------- #
        difficulty = float(rec["difficulty"])
        style      = int(rec["style"])
        avg_nps    = float(rec.get("avg_nps",  0.0))
        peak_nps   = float(rec.get("peak_nps", 0.0))

        if self.augment:
            difficulty = max(0.0, min(10.0, difficulty + random.gauss(0, 0.1)))
            avg_nps    = max(0.0, avg_nps + random.gauss(0, 0.2))
            peak_nps   = max(0.0, peak_nps + random.gauss(0, 0.3))

        # ---- Per-window signals ------------------------------------------ #
        # local_nps: count onset frames in window onset channels
        onset_active = (t_crop[CH_DON] > 0.5) | (t_crop[CH_KAT] > 0.5) | \
                       (t_crop[CH_BIG_DON] > 0.5) | (t_crop[CH_BIG_KAT] > 0.5)
        # Only count transitions (0→1), not sustained frames
        onset_count = int(np.sum(onset_active & ~np.concatenate([[False], onset_active[:-1]])))
        window_sec  = float(valid_mask.sum()) * FRAME_MS / 1000.0
        local_nps   = onset_count / max(window_sec, 1.0)

        # section_pos: normalised position in map (0=start, 1=end)
        section_pos = start / max(T_tensor - 1, 1)

        # motif vector
        motif = _compute_motif(t_crop, avg_nps=avg_nps, snap_1_4=rec.get("snap_1_4", 0.0),
                               snap_1_8=rec.get("snap_1_8", 0.0))

        return {
            "mel":         torch.from_numpy(m_crop),
            "tensor":      torch.from_numpy(t_crop),
            "valid_mask":  torch.from_numpy(valid_mask),
            "difficulty":  torch.tensor(difficulty,  dtype=torch.float32),
            "style":       torch.tensor(style,        dtype=torch.long),
            "avg_nps":     torch.tensor(avg_nps,      dtype=torch.float32),
            "peak_nps":    torch.tensor(peak_nps,     dtype=torch.float32),
            "local_nps":   torch.tensor(local_nps,    dtype=torch.float32),
            "section_pos": torch.tensor(section_pos,  dtype=torch.float32),
            "motif":       torch.from_numpy(motif),
        }


# ---------------------------------------------------------------------------
# PreprocessedDataset (original — kept for backward compatibility)
# ---------------------------------------------------------------------------

class PreprocessedDataset(Dataset):
    """
    Original full-map dataset. Pads everything to PAD_FRAMES / MEL_FRAMES.
    Kept for backward compatibility and autoencoder training.
    """

    def __init__(
        self,
        records:     list[dict],
        data_root:   Path,
        pad_frames:  int   = PAD_FRAMES,
        mel_frames:  int   = MEL_FRAMES,
        augment:     bool  = False,
        rate_p:      float = 0.2,
        rate_range:  tuple = (0.9, 1.1),
        freq_mask_p: float = 0.15,
        freq_mask_bands: int = 12,
    ):
        self.records         = records
        self.data_root       = Path(data_root)
        self.pad_frames      = pad_frames
        self.mel_frames      = mel_frames
        self.augment         = augment
        self.rate_p          = rate_p
        self.rate_range      = rate_range
        self.freq_mask_p     = freq_mask_p
        self.freq_mask_bands = freq_mask_bands

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        for _ in range(5):
            rec = self.records[idx]
            try:
                mel_path    = self.data_root / rec["mel_path"].replace("\\", "/")
                tensor_path = self.data_root / rec["tensor_path"].replace("\\", "/")

                mel_npz    = np.load(mel_path)
                tensor_npz = np.load(tensor_path)

                mel    = (mel_npz["mel"]       if "mel"    in mel_npz    else mel_npz["arr_0"]).astype(np.float32)
                tensor = (tensor_npz["tensor"] if "tensor" in tensor_npz else tensor_npz["arr_0"]).astype(np.float32)

                T          = tensor.shape[1]
                valid_len  = min(T, self.pad_frames)
                valid_mask = np.zeros(self.pad_frames, dtype=np.float32)
                valid_mask[:valid_len] = 1.0

                if T < self.pad_frames:
                    tensor = np.concatenate(
                        [tensor, np.zeros((N_CHANNELS, self.pad_frames - T), dtype=np.float32)], axis=1
                    )
                else:
                    tensor = tensor[:, :self.pad_frames]

                T_mel = mel.shape[1]
                if T_mel < self.mel_frames:
                    mel = np.concatenate(
                        [mel, np.zeros((128, self.mel_frames - T_mel), dtype=np.float32)], axis=1
                    )
                else:
                    mel = mel[:, :self.mel_frames]

                if self.augment:
                    if random.random() < self.rate_p:
                        rate  = random.uniform(*self.rate_range)
                        t_mel = max(64, int(mel.shape[1] / rate))
                        t_bm  = max(32, int(tensor.shape[1] / rate))
                        mel_t = torch.from_numpy(mel).unsqueeze(0)
                        mel   = F.interpolate(mel_t, size=t_mel, mode="linear", align_corners=False)[0].numpy()
                        ten_t = torch.from_numpy(tensor).unsqueeze(0)
                        tensor = F.interpolate(ten_t, size=t_bm, mode="nearest")[0].numpy()
                        valid_len = min(int(valid_len / rate), self.pad_frames)
                        valid_mask = np.zeros(self.pad_frames, dtype=np.float32)
                        valid_mask[:valid_len] = 1.0
                        if mel.shape[1] < self.mel_frames:
                            mel = np.concatenate([mel, np.zeros((128, self.mel_frames - mel.shape[1]), np.float32)], axis=1)
                        else:
                            mel = mel[:, :self.mel_frames]
                        if tensor.shape[1] < self.pad_frames:
                            tensor = np.concatenate([tensor, np.zeros((N_CHANNELS, self.pad_frames - tensor.shape[1]), np.float32)], axis=1)
                        else:
                            tensor = tensor[:, :self.pad_frames]
                    if random.random() < self.freq_mask_p:
                        f  = random.randint(1, self.freq_mask_bands)
                        f0 = random.randint(0, 127 - f)
                        mel[f0:f0 + f, :] = 0.0

                difficulty = float(rec["difficulty"])
                style      = int(rec["style"])

                if self.augment:
                    difficulty = max(0.0, min(10.0, difficulty + random.gauss(0, 0.1)))

                return {
                    "mel":        torch.from_numpy(mel),
                    "tensor":     torch.from_numpy(tensor),
                    "valid_mask": torch.from_numpy(valid_mask),
                    "difficulty": torch.tensor(difficulty, dtype=torch.float32),
                    "style":      torch.tensor(style, dtype=torch.long),
                    "avg_nps":    torch.tensor(float(rec.get("avg_nps", 0.0)),  dtype=torch.float32),
                    "peak_nps":   torch.tensor(float(rec.get("peak_nps", 0.0)), dtype=torch.float32),
                }
            except Exception as e:
                print(f"[dataset] failed idx={idx} ({rec.get('tensor_path', '?')}): {e}")
                idx = random.randint(0, len(self.records) - 1)

        return {
            "mel":        torch.zeros(128, self.mel_frames),
            "tensor":     torch.zeros(N_CHANNELS, self.pad_frames),
            "valid_mask": torch.zeros(self.pad_frames),
            "difficulty": torch.tensor(5.0),
            "style":      torch.tensor(0, dtype=torch.long),
            "avg_nps":    torch.tensor(0.0),
            "peak_nps":   torch.tensor(0.0),
        }


# ---------------------------------------------------------------------------
# Index loader
# ---------------------------------------------------------------------------

def load_index(
    index_path: str | Path,
    val_ratio:  float = 0.05,
    seed:       int   = 42,
) -> tuple[list[dict], list[dict]]:
    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")
    records = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rng = random.Random(seed)
    rng.shuffle(records)
    n_val = max(1, int(len(records) * val_ratio))
    return records[n_val:], records[:n_val]


def print_index_stats(records: list[dict], label: str = "dataset"):
    style_names  = {0: "standard", 1: "stream", 2: "speed", 3: "tech"}
    style_dist   = {}
    difficulties = []
    nps_vals     = []

    for r in records:
        s = r.get("style_name", style_names.get(r.get("style", 0), "?"))
        style_dist[s] = style_dist.get(s, 0) + 1
        difficulties.append(r.get("difficulty", 0.0))
        if r.get("avg_nps", 0) > 0:
            nps_vals.append(r["avg_nps"])

    print(f"{label}: {len(records)} maps")
    for s, c in sorted(style_dist.items()):
        pct = c / max(len(records), 1) * 100
        print(f"  {s:10s}: {c:5d}  ({pct:.1f}%)")
    if difficulties:
        print(f"  SR range  : {min(difficulties):.1f} – {max(difficulties):.1f}"
              f"  (mean {sum(difficulties)/len(difficulties):.1f})")
    if nps_vals:
        print(f"  NPS range : {min(nps_vals):.1f} – {max(nps_vals):.1f}"
              f"  (mean {sum(nps_vals)/len(nps_vals):.1f})")