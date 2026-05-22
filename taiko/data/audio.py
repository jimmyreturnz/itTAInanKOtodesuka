"""
taiko/data/audio.py

Converts audio files (.mp3, .ogg, .wav) → mel spectrogram tensors.

Design choices:
  - 22,050 Hz sample rate (sufficient for rhythm; reduces compute vs 44.1kHz)
  - 128 mel bins  (matches Whisper's encoder input width)
  - 10ms hop length → 1 frame = 10ms, aligns with TIME token quantization
  - Window: 25ms (Whisper standard)
  - Output shape: [128, T] float32, log-scaled, normalized to [-1, 1]

The 10ms hop is intentional: each mel frame corresponds to exactly one
TIME_1 token step, making audio-token alignment trivial during training.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np

# ---------------------------------------------------------------------------
# Constants — must stay in sync with tokenizer.py TIME_QUANTIZE_MS = 10
# ---------------------------------------------------------------------------

SAMPLE_RATE     = 22_050          # Hz
HOP_LENGTH      = 441             # samples = 20ms @ 22050 Hz
WIN_LENGTH      = 882             # samples ≈ 40ms @ 22050 Hz
N_FFT           = 1024            # FFT size (next power of 2 above win_length)
N_MELS          = 128             # mel bins (matches Whisper)
F_MIN           = 20.0            # Hz
F_MAX           = 8_000.0         # Hz  (taiko hi-hat/snare energy is < 8kHz)
TOP_DB          = 80.0            # dynamic range clamp (dB)


# ---------------------------------------------------------------------------
# Backend: try torchaudio first, fall back to librosa
# ---------------------------------------------------------------------------

def _load_torchaudio(path: str | Path) -> tuple[np.ndarray, int]:
    import torchaudio
    waveform, sr = torchaudio.load(str(path))
    # torchaudio returns [channels, samples]; mix to mono
    waveform = waveform.mean(dim=0).numpy()
    return waveform, sr


def _load_librosa(path: str | Path) -> tuple[np.ndarray, int]:
    import librosa
    waveform, sr = librosa.load(str(path), sr=None, mono=True)
    return waveform, sr


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Load audio file → (mono float32 waveform, sample_rate)."""
    try:
        return _load_torchaudio(path)
    except ImportError:
        return _load_librosa(path)


def resample(waveform: np.ndarray, orig_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    if orig_sr == target_sr:
        return waveform
    try:
        import torchaudio.functional as F
        import torch
        t = torch.from_numpy(waveform).unsqueeze(0)
        t = F.resample(t, orig_sr, target_sr)
        return t.squeeze(0).numpy()
    except ImportError:
        import librosa
        return librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)


# ---------------------------------------------------------------------------
# Mel spectrogram computation
# ---------------------------------------------------------------------------

def compute_mel_spectrogram(
    waveform: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    hop_length: int = HOP_LENGTH,
    win_length: int = WIN_LENGTH,
    n_fft: int = N_FFT,
    f_min: float = F_MIN,
    f_max: float = F_MAX,
    top_db: float = TOP_DB,
) -> np.ndarray:
    """
    Compute log-mel spectrogram.

    Returns:
        np.ndarray of shape [n_mels, T], float32
        Values normalized to [-1, 1] via (log_mel / (top_db/2)) + 1
    """
    try:
        return _mel_torchaudio(waveform, sr, n_mels, hop_length, win_length, n_fft, f_min, f_max, top_db)
    except ImportError:
        return _mel_librosa(waveform, sr, n_mels, hop_length, win_length, n_fft, f_min, f_max, top_db)


def _mel_torchaudio(waveform, sr, n_mels, hop_length, win_length, n_fft, f_min, f_max, top_db):
    import torch
    import torchaudio.transforms as T

    mel_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        power=2.0,
        norm="slaney",
        mel_scale="slaney",
    )
    amp_to_db = T.AmplitudeToDB(stype="power", top_db=top_db)

    t = torch.from_numpy(waveform).float().unsqueeze(0)  # [1, samples]
    mel = mel_transform(t)          # [1, n_mels, T]
    mel_db = amp_to_db(mel)         # [1, n_mels, T], range ≈ [-top_db, 0]
    mel_db = mel_db.squeeze(0)      # [n_mels, T]

    # Normalize to [-1, 1]: shift so max≈0 → divide by (top_db/2) → +1
    mel_norm = (mel_db / (top_db / 2.0)) + 1.0
    mel_norm = mel_norm.clamp(-1.0, 1.0)
    return mel_norm.numpy().astype(np.float32)


def _mel_librosa(waveform, sr, n_mels, hop_length, win_length, n_fft, f_min, f_max, top_db):
    import librosa

    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
        power=2.0,
        norm="slaney",
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=top_db)  # [-top_db, 0]
    mel_norm = (mel_db / (top_db / 2.0)) + 1.0
    mel_norm = np.clip(mel_norm, -1.0, 1.0)
    return mel_norm.astype(np.float32)


# ---------------------------------------------------------------------------
# High-level: audio file → mel tensor
# ---------------------------------------------------------------------------

class MelExtractor:
    """
    Main interface for the pipeline.

    Usage:
        extractor = MelExtractor()
        mel = extractor.extract("song.mp3")   # shape: [128, T]
        # T ≈ audio_duration_ms / 10
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_mels: int = N_MELS,
        hop_length: int = HOP_LENGTH,
        win_length: int = WIN_LENGTH,
        n_fft: int = N_FFT,
        f_min: float = F_MIN,
        f_max: float = F_MAX,
        top_db: float = TOP_DB,
    ):
        self.sample_rate = sample_rate
        self.n_mels      = n_mels
        self.hop_length  = hop_length
        self.win_length  = win_length
        self.n_fft       = n_fft
        self.f_min       = f_min
        self.f_max       = f_max
        self.top_db      = top_db

    def extract(self, audio_path: str | Path) -> np.ndarray:
        """
        Load audio file and return mel spectrogram.

        Returns:
            np.ndarray, shape [128, T], float32, values in [-1, 1]
            T = ceil(num_samples / hop_length)
        """
        waveform, sr = load_audio(audio_path)
        if sr != self.sample_rate:
            waveform = resample(waveform, sr, self.sample_rate)
        return compute_mel_spectrogram(
            waveform,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_fft=self.n_fft,
            f_min=self.f_min,
            f_max=self.f_max,
            top_db=self.top_db,
        )

    def extract_waveform(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """For when you've already loaded the waveform (e.g. from cache)."""
        if sr != self.sample_rate:
            waveform = resample(waveform, sr, self.sample_rate)
        return compute_mel_spectrogram(waveform, sr=self.sample_rate,
            n_mels=self.n_mels, hop_length=self.hop_length,
            win_length=self.win_length, n_fft=self.n_fft,
            f_min=self.f_min, f_max=self.f_max, top_db=self.top_db)

    def frames_to_ms(self, n_frames: int) -> float:
        """Convert frame count to milliseconds."""
        return n_frames * self.hop_length / self.sample_rate * 1000.0

    def ms_to_frames(self, ms: float) -> int:
        """Convert milliseconds to frame index."""
        return int(ms / 1000.0 * self.sample_rate / self.hop_length)

    @property
    def ms_per_frame(self) -> float:
        """How many ms each mel frame represents — should be ~10ms."""
        return self.hop_length / self.sample_rate * 1000.0


# ---------------------------------------------------------------------------
# Cache: save/load precomputed mel spectrograms
# ---------------------------------------------------------------------------

def save_mel(mel: np.ndarray, path: str | Path):
    """Save mel spectrogram as compressed numpy file (.npz)."""
    np.savez_compressed(str(path), mel=mel)


def load_mel(path: str | Path) -> np.ndarray:
    """Load a saved mel spectrogram."""
    data = np.load(str(path))
    return data["mel"]


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

def verify_alignment(mel: np.ndarray, beatmap_duration_ms: float) -> dict:
    """
    Check that mel frame count roughly matches the beatmap duration.
    Returns a dict with alignment stats.
    """
    extractor = MelExtractor()
    mel_duration_ms = extractor.frames_to_ms(mel.shape[1])
    diff_ms = abs(mel_duration_ms - beatmap_duration_ms)
    return {
        "mel_frames": mel.shape[1],
        "mel_duration_ms": mel_duration_ms,
        "beatmap_duration_ms": beatmap_duration_ms,
        "diff_ms": diff_ms,
        "aligned": diff_ms < 1000,  # within 1 second is fine
    }