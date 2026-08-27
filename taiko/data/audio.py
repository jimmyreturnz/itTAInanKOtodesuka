"""
taiko/data/audio.py

Audio file -> log-mel spectrogram on the shared time grid.

The hop length comes from `taiko.data.frames`, which is the single source of
truth for the time grid. It is not repeated here, and nothing in this module
may override it: one mel frame must equal one chart frame, or training pairs
charts with the wrong span of audio and the model never learns to follow the
music. See frames.py for why that failure is invisible in the loss curve.

    22050 Hz, hop 441 -> 20 ms per frame, 50 frames per second
    128 mel bins, 40 ms window
    log scale, per-clip peak reference, normalised to [-1, 1]

Both backends reference the per-clip maximum. torchaudio's AmplitudeToDB uses a
fixed reference of 1.0 by default, which makes its output depend on the mix
loudness -- so a quiet song and a loud one would land in different parts of the
range for identical musical content. librosa's power_to_db(ref=np.max) is
per-clip already; this module makes torchaudio match rather than the other way
round, since the chart should not depend on how hot the master was.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np

from taiko.data.frames import FRAME_MS, HOP_LENGTH, SAMPLE_RATE

# ---------------------------------------------------------------------------
# Constants -- the time grid comes from frames.py and is not redefined here
# ---------------------------------------------------------------------------

WIN_LENGTH = 882             # samples, about 40 ms at 22050 Hz
N_FFT      = 1024            # next power of two above win_length
N_MELS     = 128
F_MIN      = 20.0            # Hz
F_MAX      = 8_000.0         # Hz; taiko-relevant percussive energy sits below this
TOP_DB     = 80.0            # dynamic range clamp


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

    t = torch.from_numpy(waveform).float().unsqueeze(0)   # [1, samples]
    mel = mel_transform(t)                                # [1, n_mels, T]
    mel_db = amp_to_db(mel).squeeze(0)                    # [n_mels, T]

    # Re-reference to the clip's own peak. AmplitudeToDB uses a fixed reference
    # of 1.0, so without this the result tracks mix loudness rather than
    # musical content, and disagrees with the librosa path.
    mel_db = mel_db - mel_db.max()
    mel_db = mel_db.clamp(min=-top_db)

    mel_norm = ((mel_db / (top_db / 2.0)) + 1.0).clamp(-1.0, 1.0)
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
            T frames of FRAME_MS each
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
        return n_frames * self.ms_per_frame

    def ms_to_frames(self, ms: float) -> int:
        return int(round(ms / self.ms_per_frame))

    @property
    def ms_per_frame(self) -> float:
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