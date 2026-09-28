"""
taiko/timing/activations.py

Where the beat evidence comes from. Every source returns the same thing:

    Activations(beat, downbeat, fps)   per-frame evidence, any frame rate

Sources, best first:

    timingnet  our own network, trained on ranked red lines -- the same music
               it will be asked to time (taiko/timing/model.py). Needs a
               checkpoint from scripts/train_timing.py.
    beat_this  CPJKU's pretrained tracker (pip install beat-this). General
               music; its weights download on first use.
    onset      a percussive onset envelope. No model, always available, fine
               for steady electronic music, weakest on sparse or rubato
               material, and it has no real notion of downbeats.

Neural sources are run several times on audio shifted by a random fraction
of a frame and averaged on a 4 ms grid. Each pass quantises time at a
different phase, so the average resolves beat positions finer than the
network's own 20 ms frames -- Mapperatorinator's super-timing trick, applied to
frame activations instead of generated tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 22_050
ENSEMBLE_FPS = 250.0


@dataclass
class Activations:
    beat: np.ndarray
    downbeat: np.ndarray | None
    fps: float
    source: str


def load_mono(path: str | Path, sr: int = SR) -> np.ndarray:
    """The same decoder the training data went through, resampled to `sr`."""
    from taiko.data.audio import load_audio, resample
    y, file_sr = load_audio(path)
    y = np.asarray(y, dtype=np.float32)
    if file_sr != sr:
        y = resample(y, file_sr, sr)
    return y


# --------------------------------------------------------------------------- #
# Onset envelope (no model)
# --------------------------------------------------------------------------- #

def onset_activations(y: np.ndarray, sr: int = SR, hop: int = 110) -> Activations:
    """
    Percussive onset strength at ~200 fps as beat evidence, and low-band
    (kick) onset strength as a weak downbeat hint.
    """
    import librosa
    perc = librosa.effects.percussive(y, margin=2.0)
    beat = librosa.onset.onset_strength(y=perc, sr=sr, hop_length=hop, n_fft=1024)
    low = librosa.onset.onset_strength(y=perc, sr=sr, hop_length=hop, n_fft=2048,
                                       n_mels=24, fmax=180.0)
    n = min(len(beat), len(low))
    return Activations(beat[:n].astype(np.float64), low[:n].astype(np.float64),
                       sr / hop, "onset")


def fine_envelope(y: np.ndarray, sr: int = SR, hop: int = 22,
                  win_ms: float = 2.0) -> tuple[np.ndarray, float]:
    """
    ~1 ms attack envelope, for polishing offsets.

    Rising log-energy of the percussive, high-passed signal in 2 ms windows.
    A spectral-flux envelope was tried first and sat 2 ms late on click tracks
    with exactly known onsets: its 23 ms analysis window peaks after the
    attack has begun. Measured in the time domain the error was 0.0 ms, which
    is what the last step of timing needs -- everything before it is only as
    good as this.
    """
    import librosa
    import scipy.signal as sg
    b, a = sg.butter(2, 150.0 / (sr / 2), "high")
    x = sg.lfilter(b, a, np.asarray(y, dtype=np.float64)).astype(np.float32)
    perc = librosa.effects.percussive(x, margin=2.0).astype(np.float64)
    win = max(1, int(win_ms / 1000.0 * sr))
    energy = np.convolve(perc ** 2, np.ones(win) / win, mode="same")[::hop]
    log_e = np.log(energy + 1e-8)
    env = np.maximum(np.diff(log_e, prepend=log_e[0]), 0.0)
    return env, sr / hop


# --------------------------------------------------------------------------- #
# Shifted ensembles for frame-level networks
# --------------------------------------------------------------------------- #

def _ensemble(run, y: np.ndarray, sr: int, net_fps: float, passes: int,
              seed: int = 0) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Average `run(signal) -> (beat, downbeat)` over `passes` sub-frame shifts,
    resampled onto a common ENSEMBLE_FPS grid.
    """
    rng = np.random.default_rng(seed)
    hop_samples = sr / net_fps
    duration = len(y) / sr
    grid = np.arange(0, duration, 1.0 / ENSEMBLE_FPS)
    beat_sum = np.zeros_like(grid)
    down_sum = np.zeros_like(grid)
    have_down = False
    for p in range(passes):
        shift = 0 if p == 0 else int(rng.integers(1, int(hop_samples)))
        padded = np.concatenate([np.zeros(shift, dtype=y.dtype), y])
        beat, down = run(padded)
        t = np.arange(len(beat)) / net_fps - shift / sr
        beat_sum += np.interp(grid, t, beat, left=0.0, right=0.0)
        if down is not None:
            down_sum += np.interp(grid, t, down, left=0.0, right=0.0)
            have_down = True
    return beat_sum / passes, (down_sum / passes if have_down else None)


def beat_this_activations(y: np.ndarray, sr: int = SR, passes: int = 8,
                          device: str = "cpu", checkpoint: str = "final0") -> Activations:
    import torch
    from beat_this.inference import Audio2Frames

    frames = Audio2Frames(checkpoint_path=checkpoint, device=device)

    def run(signal):
        with torch.no_grad():
            b, d = frames(signal.astype(np.float64), sr)
        return torch.sigmoid(b).cpu().numpy(), torch.sigmoid(d).cpu().numpy()

    beat, down = _ensemble(run, y, sr, 50.0, passes)
    return Activations(beat, down, ENSEMBLE_FPS, f"beat_this x{passes}")


def timingnet_activations(y: np.ndarray, checkpoint: str | Path, sr: int = SR,
                          passes: int = 8, device: str = "cpu") -> Activations:
    from taiko.timing.model import load_timingnet, run_timingnet
    model = load_timingnet(checkpoint, device)

    def run(signal):
        return run_timingnet(model, signal, sr, device)

    beat, down = _ensemble(run, y, sr, 50.0, passes)
    return Activations(beat, down, ENSEMBLE_FPS, f"timingnet x{passes}")
