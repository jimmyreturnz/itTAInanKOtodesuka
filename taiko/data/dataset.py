"""
taiko/data/dataset.py

PyTorch Dataset that pairs mel spectrograms with tokenized beatmaps.

Each sample:
    - mel:            [128, T_audio] float32  — full song mel
    - conditioning:   [N_cond]       int64    — conditioning token IDs
    - token_ids:      [T_seq]        int64    — SOS + events + EOS
    - token_mask:     [T_seq]        bool     — True = real token, False = PAD

During training we chunk the mel into windows aligned to note events,
so the model sees a local audio context rather than the full song at once.
This is necessary for GPU memory and for learning local rhythm patterns.

Window design:
    - WINDOW_MS = 8192ms  (~8 seconds of audio context per training sample)
    - Randomly sampled from the song during training, deterministic during eval
    - Tokens within the window are extracted with their time offsets re-zeroed
"""

from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Optional
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object  # fallback for type hints without torch

from taiko.data.audio import MelExtractor, save_mel, load_mel, SAMPLE_RATE, HOP_LENGTH
from taiko.data.tokenizer import TaikoTokenizer, TaikoVocabulary, TokenizedBeatmap
from taiko.data.osu_parser import OsuTaikoParser, TaikoBeatmap


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW_MS      = 8_192    # audio context window in ms (~8 seconds)
MAX_SEQ_LEN    = 512      # max token sequence length per window
TIME_QUANT_MS  = 10       # must match tokenizer.py


# ---------------------------------------------------------------------------
# Processed sample record (stored on disk as JSON)
# ---------------------------------------------------------------------------

class BeatmapRecord:
    """
    Metadata record for one processed beatmap, stored as a JSON entry
    in the dataset index. Points to the mel .npz and stores token IDs.
    """
    # v2: added beat_length_ms, offset_ms
    __slots__ = [
        "beatmap_id", "beatmap_set_id", "audio_path", "mel_path",
        "conditioning_ids", "token_ids", "beat_length_ms", "offset_ms", "duration_ms", "star_rating",
        "title", "version", "note_count",
    ]

    def __init__(self, **kwargs):
        for k in self.__slots__:
            setattr(self, k, kwargs.get(k))

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__slots__}

    @classmethod
    def from_dict(cls, d: dict) -> "BeatmapRecord":
        return cls(**d)


# ---------------------------------------------------------------------------
# Dataset index builder
# ---------------------------------------------------------------------------

class DatasetBuilder:
    """
    Processes raw beatmap directories into a flat dataset index.

    Directory structure expected:
        data/raw/
            <beatmapset_id>/
                audio.mp3          (or .ogg)
                <diff_name>.osu    (one per difficulty, all taiko mode 1)

    Output:
        data/processed/
            mels/
                <beatmap_id>.npz   (precomputed mel spectrograms)
            index.jsonl            (one JSON record per difficulty)
    """

    def __init__(
        self,
        raw_dir: str | Path,
        processed_dir: str | Path,
        mel_extractor: Optional[MelExtractor] = None,
        tokenizer: Optional[TaikoTokenizer] = None,
    ):
        self.raw_dir       = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.mel_dir       = self.processed_dir / "mels"
        self.index_path    = self.processed_dir / "index.jsonl"
        self.extractor     = mel_extractor or MelExtractor()
        self.tokenizer     = tokenizer or TaikoTokenizer()

        self.mel_dir.mkdir(parents=True, exist_ok=True)

    def build(
        self,
        star_rating_db: Optional[dict[int, float]] = None,
        skip_existing: bool = True,
        min_notes: int = 50,
        max_notes: int = 5000,
    ) -> int:
        """
        Process all beatmap sets in raw_dir.

        Args:
            star_rating_db: dict mapping beatmap_id → star_rating
                            (from osu! API scrape). If None, uses OD as proxy.
            skip_existing:  skip beatmaps whose mel already exists
            min_notes:      discard maps with fewer notes (too sparse)
            max_notes:      discard maps with more notes (anomalies)

        Returns:
            Number of successfully processed beatmaps.
        """
        parser  = OsuTaikoParser()
        records = []
        errors  = []

        beatmapset_dirs = [d for d in self.raw_dir.iterdir() if d.is_dir()]
        print(f"Found {len(beatmapset_dirs)} beatmapset directories.")

        for bms_dir in sorted(beatmapset_dirs):
            # Find audio file
            audio_path = self._find_audio(bms_dir)
            if audio_path is None:
                errors.append(f"No audio in {bms_dir.name}")
                continue

            # Compute mel once per beatmapset (all diffs share the same audio)
            mel_cache_path = self.mel_dir / f"{bms_dir.name}.npz"
            if skip_existing and mel_cache_path.exists():
                mel = load_mel(mel_cache_path)
            else:
                try:
                    mel = self.extractor.extract(audio_path)
                    save_mel(mel, mel_cache_path)
                except Exception as e:
                    errors.append(f"Mel error {bms_dir.name}: {e}")
                    continue

            # Process each .osu file in this set
            for osu_path in bms_dir.glob("*.osu"):
                try:
                    bm = parser.parse_file(osu_path)
                except Exception as e:
                    errors.append(f"Parse error {osu_path.name}: {e}")
                    continue

                # Filter
                if bm.note_count < min_notes or bm.note_count > max_notes:
                    continue

                # Attach star rating from API db if available
                if star_rating_db and bm.beatmap_id in star_rating_db:
                    bm.star_rating = star_rating_db[bm.beatmap_id]
                elif bm.star_rating == 0.0:
                    # Fallback: rough SR estimate from OD + NPS
                    bm.star_rating = self._estimate_sr(bm)

                # Tokenize
                try:
                    tok = self.tokenizer.encode(bm)
                except Exception as e:
                    errors.append(f"Tokenize error {osu_path.name}: {e}")
                    continue

                # Handle AudioLeadIn offset
                lead_in_ms = self._get_lead_in(osu_path)

                record = BeatmapRecord(
                    beatmap_id=bm.beatmap_id,
                    beatmap_set_id=bm.beatmap_set_id,
                    audio_path=str(audio_path),
                    mel_path=str(mel_cache_path),
                    conditioning_ids=tok.conditioning_ids,
                    token_ids=tok.token_ids,
                    duration_ms=bm.duration_ms,
                    star_rating=bm.star_rating,
                    title=bm.title,
                    version=bm.version,
                    note_count=bm.note_count,
                )
                records.append(record)

        # Write index
        with open(self.index_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec.to_dict()) + "\n")

        print(f"Processed: {len(records)} beatmaps | Errors: {len(errors)}")
        if errors:
            print("First 10 errors:")
            for e in errors[:10]:
                print(f"  {e}")

        return len(records)

    def _find_audio(self, bms_dir: Path) -> Optional[Path]:
        for ext in (".mp3", ".ogg", ".wav", ".flac"):
            matches = list(bms_dir.glob(f"*{ext}"))
            if matches:
                return matches[0]
        return None

    def _get_lead_in(self, osu_path: Path) -> int:
        """Extract AudioLeadIn value from .osu file."""
        try:
            for line in osu_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("AudioLeadIn:"):
                    return int(line.split(":")[1].strip())
        except Exception:
            pass
        return 0

    def _estimate_sr(self, bm: TaikoBeatmap) -> float:
        """Very rough SR estimate when API data is unavailable."""
        return min(10.0, bm.overall_difficulty * 0.6 + bm.notes_per_second * 0.2)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class TaikoDataset(Dataset):
    """
    PyTorch Dataset for taiko beatmap generation training.

    Each __getitem__ returns a windowed sample:
        mel_window:      [128, W_frames]  float32  — audio context
        conditioning:    [N_cond]         int64    — conditioning IDs
        token_ids:       [T_seq]          int64    — padded token sequence
        token_mask:      [T_seq]          bool     — True = real token

    Window sampling:
        During training, a random window start is chosen from the song.
        Notes within [window_start, window_start + WINDOW_MS] are extracted.
        If a window has < MIN_NOTES_PER_WINDOW notes, resample.
    """

    MIN_NOTES_PER_WINDOW = 3
    N_COND_TOKENS        = 5   # must match tokenizer.conditioning_ids() length

    def __init__(
        self,
        index_path: str | Path,
        vocab: Optional[TaikoVocabulary] = None,
        window_ms: int = WINDOW_MS,
        max_seq_len: int = MAX_SEQ_LEN,
        training: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.index_path  = Path(index_path)
        self.vocab       = vocab or TaikoVocabulary()
        self.window_ms   = window_ms
        self.max_seq_len = max_seq_len
        self.training    = training

        # Frames per window
        self.window_frames = int(window_ms / 1000.0 * SAMPLE_RATE / HOP_LENGTH)

        # Load index
        self.records: list[BeatmapRecord] = []
        with open(self.index_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(BeatmapRecord.from_dict(json.loads(line)))

        if max_samples:
            self.records = self.records[:max_samples]

        print(f"TaikoDataset: {len(self.records)} beatmaps loaded.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        import torch
        record = self.records[idx]

        # Load mel
        mel = load_mel(record.mel_path)           # [128, T_full]
        total_frames = mel.shape[1]
        total_ms     = total_frames * (HOP_LENGTH / SAMPLE_RATE * 1000.0)

        # Choose window start
        max_start_ms = max(0.0, total_ms - self.window_ms)
        if self.training:
            start_ms = random.uniform(0, max_start_ms)
        else:
            start_ms = 0.0  # deterministic eval: start of song

        end_ms = start_ms + self.window_ms

        # Crop mel window
        start_frame = int(start_ms / 1000.0 * SAMPLE_RATE / HOP_LENGTH)
        end_frame   = start_frame + self.window_frames
        mel_window  = mel[:, start_frame:end_frame]

        # Pad mel if shorter than window (end of song)
        if mel_window.shape[1] < self.window_frames:
            pad = np.zeros((mel.shape[0], self.window_frames - mel_window.shape[1]), dtype=np.float32)
            mel_window = np.concatenate([mel_window, pad], axis=1)

        # Extract tokens within this window from the full token sequence
        beat_length_ms = getattr(record, 'beat_length_ms', None) or 500.0
        offset_ms_val  = getattr(record, 'offset_ms', None) or 0.0
        token_ids = self._extract_window_tokens(
            record.token_ids, start_ms, end_ms,
            beat_length_ms=beat_length_ms,
            offset_ms=offset_ms_val,
        )

        # Pad / truncate token sequence
        token_ids, token_mask = self._pad_tokens(token_ids)

        return {
            "mel":          torch.from_numpy(mel_window),              # [128, W]
            "conditioning": torch.tensor(record.conditioning_ids, dtype=torch.long),
            "token_ids":    torch.tensor(token_ids,   dtype=torch.long),
            "token_mask":   torch.tensor(token_mask,  dtype=torch.bool),
            # Debug info
            "star_rating":  torch.tensor(record.star_rating, dtype=torch.float32),
            "beatmap_id":   record.beatmap_id or 0,
        }

    def _extract_window_tokens(
        self,
        full_token_ids: list[int],
        start_ms: float,
        end_ms: float,
        beat_length_ms: float = 500.0,
        offset_ms: float = 0.0,
    ) -> list[int]:
        """
        Walk the full token sequence (beat-relative v2) and extract notes in [start_ms, end_ms].
        Re-zeros beat deltas relative to window start.

        Returns a new token list: [SOS, BEAT_d1, NOTE_1, ..., EOS]
        """
        from taiko.data.tokenizer import ms_to_steps, SUBDIVISIONS_PER_BEAT, MAX_BEAT_DELTA, SILENCE_THRESHOLD
        vocab = self.vocab
        tokens_out = [vocab.SOS_ID]

        abs_steps         = 0
        prev_emitted_steps = ms_to_steps(start_ms, beat_length_ms, offset_ms)
        i = 0

        while i < len(full_token_ids):
            tid = full_token_ids[i]

            if tid in (vocab.SOS_ID, vocab.PAD_ID):
                i += 1; continue
            if tid == vocab.EOS_ID:
                break
            if tid == vocab.SILENCE_ID:
                i += 1; continue

            if vocab.is_beat_token(tid):
                delta      = vocab.beat_token_to_steps(tid)
                abs_steps += delta

                if i + 1 >= len(full_token_ids):
                    i += 1; continue

                note_tid = full_token_ids[i + 1]

                # Convert steps back to ms to check window bounds
                from taiko.data.tokenizer import steps_to_ms
                abs_ms = steps_to_ms(abs_steps, beat_length_ms, offset_ms)

                if start_ms <= abs_ms < end_ms:
                    rel_steps = abs_steps - prev_emitted_steps
                    rel_steps = max(0, rel_steps)
                    # Emit SILENCE if gap too large
                    if rel_steps > MAX_BEAT_DELTA:
                        tokens_out.append(vocab.SILENCE_ID)
                        rel_steps = rel_steps % MAX_BEAT_DELTA
                    tokens_out.append(vocab.steps_to_beat_token(rel_steps))
                    tokens_out.append(note_tid)
                    prev_emitted_steps = abs_steps

                i += 2; continue
            i += 1

        tokens_out.append(vocab.EOS_ID)
        return tokens_out

    def _pad_tokens(self, token_ids: list[int]) -> tuple[list[int], list[bool]]:
        """Pad or truncate to max_seq_len. Returns (ids, mask)."""
        vocab = self.vocab
        if len(token_ids) > self.max_seq_len:
            # Truncate but keep EOS
            token_ids = token_ids[:self.max_seq_len - 1] + [vocab.EOS_ID]

        mask = [True] * len(token_ids)
        pad_len = self.max_seq_len - len(token_ids)
        token_ids = token_ids + [vocab.PAD_ID] * pad_len
        mask      = mask      + [False]        * pad_len
        return token_ids, mask


# ---------------------------------------------------------------------------
# Collate function for DataLoader
# ---------------------------------------------------------------------------

def taiko_collate_fn(batch: list[dict]) -> dict:
    """Stack batch items into tensors."""
    import torch
    return {
        "mel":          torch.stack([b["mel"]          for b in batch]),
        "conditioning": torch.stack([b["conditioning"] for b in batch]),
        "token_ids":    torch.stack([b["token_ids"]    for b in batch]),
        "token_mask":   torch.stack([b["token_mask"]   for b in batch]),
        "star_rating":  torch.stack([b["star_rating"]  for b in batch]),
    }


# ---------------------------------------------------------------------------
# Train / val split
# ---------------------------------------------------------------------------

def split_index(
    index_path: str | Path,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[Path, Path]:
    """
    Split index.jsonl into train.jsonl and val.jsonl.
    Stratified by star rating bucket to keep difficulty distribution balanced.
    """
    index_path = Path(index_path)
    out_dir    = index_path.parent

    with open(index_path, "r") as f:
        records = [json.loads(l) for l in f if l.strip()]

    rng = random.Random(seed)
    rng.shuffle(records)

    # Stratify by SR bucket
    buckets: dict[int, list[dict]] = {}
    for rec in records:
        bucket = int(rec.get("star_rating", 0))
        buckets.setdefault(bucket, []).append(rec)

    train_recs, val_recs = [], []
    for bucket_recs in buckets.values():
        n_val = max(1, int(len(bucket_recs) * val_ratio))
        val_recs.extend(bucket_recs[:n_val])
        train_recs.extend(bucket_recs[n_val:])

    train_path = out_dir / "train.jsonl"
    val_path   = out_dir / "val.jsonl"

    with open(train_path, "w") as f:
        for r in train_recs:
            f.write(json.dumps(r) + "\n")
    with open(val_path, "w") as f:
        for r in val_recs:
            f.write(json.dumps(r) + "\n")

    print(f"Split: {len(train_recs)} train | {len(val_recs)} val")
    return train_path, val_path
