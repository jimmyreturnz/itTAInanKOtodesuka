"""
taiko/data/preprocessed_dataset.py

Dataset class for Kaggle/Colab training using pre-computed tensors + mels.
Reads from colab_index.jsonl — no .osu parsing, no audio extraction.

Expected index format (one JSON line per map):
  {
    "mel_path":    "mels/1000 Song.npz",
    "tensor_path": "tensors/1000 Song__Oni.npz",
    "difficulty":  5.2,
    "style":       1,
    "note_count":  677,
    "duration_ms": 171400
  }

All paths in the index are relative to data/processed/.
"""

from __future__ import annotations
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from taiko.data.tensor_repr import N_CHANNELS

# ---------------------------------------------------------------------------
# Constants — must match preprocess_for_colab.py and train_diffusion.py
# ---------------------------------------------------------------------------

PAD_FRAMES = 18_000     # beatmap frames  (18000 × 20ms = 360s)
MEL_FRAMES = 36_000     # mel frames      (36000 × 10ms = 360s)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PreprocessedDataset(Dataset):
    """
    Loads pre-computed mel + beatmap tensor .npz pairs.
    Much faster than on-the-fly parsing — just two np.load() calls per item.
    """

    def __init__(self,
                 records:    list[dict],
                 data_root:  Path,
                 pad_frames: int = PAD_FRAMES,
                 mel_frames: int = MEL_FRAMES,
                 augment:    bool = False,
                 ):
        self.records    = records
        self.data_root  = Path(data_root)
        self.pad_frames = pad_frames
        self.mel_frames = mel_frames
        self.augment    = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        for _ in range(5):
            rec = self.records[idx]
            try:
                # ---- Load mel ------------------------------------------ #
                mel_path = self.data_root / rec["mel_path"]
                mel_npz  = np.load(mel_path)
                # Support both key names
                mel = mel_npz["mel"] if "mel" in mel_npz else mel_npz["arr_0"]
                mel = mel.astype(np.float32)                # [128, T_mel]

                # Pad / truncate to mel_frames
                T_mel = mel.shape[1]
                if T_mel < self.mel_frames:
                    mel = np.concatenate([
                        mel,
                        np.zeros((128, self.mel_frames - T_mel), dtype=np.float32)
                    ], axis=1)
                else:
                    mel = mel[:, :self.mel_frames]

                # ---- Load tensor --------------------------------------- #
                tensor_path = self.data_root / rec["tensor_path"]
                tensor_npz  = np.load(tensor_path)
                tensor = tensor_npz["tensor"] if "tensor" in tensor_npz else tensor_npz["arr_0"]
                tensor = tensor.astype(np.float32)           # [7, T_raw]

                # Build valid mask before padding
                T         = tensor.shape[1]
                valid_len = min(T, self.pad_frames)
                valid_mask = np.zeros(self.pad_frames, dtype=np.float32)
                valid_mask[:valid_len] = 1.0

                # Pad / truncate tensor to pad_frames
                if T < self.pad_frames:
                    pad    = np.zeros((N_CHANNELS, self.pad_frames - T), dtype=np.float32)
                    tensor = np.concatenate([tensor, pad], axis=1)
                else:
                    tensor = tensor[:, :self.pad_frames]

                # ---- Conditioning -------------------------------------- #
                difficulty = float(rec["difficulty"])
                style      = int(rec["style"])

                # Optional: mild difficulty jitter during training
                if self.augment:
                    difficulty = difficulty + random.gauss(0, 0.1)
                    difficulty = max(0.0, min(10.0, difficulty))

                return {
                    "mel":        torch.from_numpy(mel),
                    "tensor":     torch.from_numpy(tensor),
                    "valid_mask": torch.from_numpy(valid_mask),
                    "difficulty": torch.tensor(difficulty, dtype=torch.float32),
                    "style":      torch.tensor(style,      dtype=torch.long),
                }

            except Exception as e:
                print(f"[dataset] failed idx={idx} ({rec.get('tensor_path', '?')}): {e}")
                idx = random.randint(0, len(self.records) - 1)

        # Fallback — should rarely happen with pre-computed files
        print(f"[dataset] WARNING: returning empty sample after 5 retries")
        return {
            "mel":        torch.zeros(128, self.mel_frames),
            "tensor":     torch.zeros(N_CHANNELS, self.pad_frames),
            "valid_mask": torch.zeros(self.pad_frames),
            "difficulty": torch.tensor(5.0),
            "style":      torch.tensor(0, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Index loader
# ---------------------------------------------------------------------------

def load_index(index_path: str | Path,
               val_ratio:  float = 0.05,
               seed:       int   = 42,
               ) -> tuple[list[dict], list[dict]]:
    """
    Load colab_index.jsonl and split into train/val.
    Returns (train_records, val_records).
    """
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
    """Print a summary of the index records."""
    style_names = {0: "standard", 1: "stream", 2: "speed", 3: "tech"}
    style_dist  = {n: 0 for n in style_names.values()}
    difficulties = []

    for r in records:
        s = r.get("style_name", style_names.get(r.get("style", 0), "?"))
        style_dist[s] = style_dist.get(s, 0) + 1
        difficulties.append(r.get("difficulty", 0.0))

    print(f"{label}: {len(records)} maps")
    for s, c in sorted(style_dist.items()):
        pct = c / max(len(records), 1) * 100
        print(f"  {s:10s}: {c:5d}  ({pct:.1f}%)")
    if difficulties:
        print(f"  SR range  : {min(difficulties):.1f} – {max(difficulties):.1f}"
              f"  (mean {sum(difficulties)/len(difficulties):.1f})")