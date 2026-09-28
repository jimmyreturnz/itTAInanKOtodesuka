"""
taiko/model/timing_encoder.py

The beat grid, carried to every U-Net level without being averaged away.

What was wrong
--------------
The U-Net used to receive the [3, T] timing stream average-pooled to latent
resolution. One latent frame is 16 chart frames, 320 ms -- about one beat at
taiko tempos -- and the mean of a phasor over one full turn is zero. Measured on
a clean grid, the magnitude that survived pooling was

    120 BPM  0.45      180 BPM  0.04      187.5 BPM  0.00      240 BPM  0.19
                                                                (sign flipped)

so across the tempos taiko charts most often use, the model was told almost
nothing about where the beats are, and above ~190 BPM it was told the wrong
thing. The notes it placed came from the audio and from its rhythm prior
alone -- which is also why it could fill a quiet section with a plausible
stream, or skip a sound that sat squarely on the beat.

What this does instead
----------------------
Features are built at full rate from the stream the dataset already provides
(so the shards do not change), then brought down to latent resolution by
*learned* strided convolution, the same way the mel stem does it. A strided
conv can learn a different weight for each sub-frame offset, so where inside
a 320 ms latent frame the beats fall survives the trip; an average cannot.

Per frame, at 20 ms:

    sin/cos(2 pi k phase)  k = 1, 2, 3, 4, 6   beat, 1/2, 1/3, 1/4, 1/6 grids
    downbeat proximity                          straight from the stream
    log2(bpm / 180)                             from the phase's rate of change
    grid present                                 0 in the zero-padded tail

The harmonics matter: a note on the 1/4 grid is at a phase where the k = 4
phasor is at its peak, whatever the tempo, so the network gets snap positions
without having to derive them from the fundamental.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from taiko.data.frames import FPS
from taiko.data.tensor_repr import TM_COS, TM_DOWNBEAT, TM_SIN

HARMONICS = (1, 2, 3, 4, 6)
N_TIMING_FEATURES = 2 * len(HARMONICS) + 3
REFERENCE_BPM = 180.0


def Normalize(channels: int, num_groups: int = 8) -> nn.GroupNorm:
    while channels % num_groups != 0:
        num_groups //= 2
    return nn.GroupNorm(num_groups, channels, eps=1e-6, affine=True)


def timing_features(timing: torch.Tensor) -> torch.Tensor:
    """
    [B, 3, T] timing stream -> [B, N_TIMING_FEATURES, T] at the same rate.

    Pure function of the stream, no parameters: it is cheap, it is the same at
    training and inference by construction, and it can be tested directly.
    """
    s, c = timing[:, TM_SIN], timing[:, TM_COS]
    radius = torch.sqrt(s * s + c * c)
    present = (radius > 0.5).to(timing.dtype)
    theta = torch.atan2(s, c)                                   # [-pi, pi]

    feats = []
    for k in HARMONICS:
        feats.append(torch.sin(k * theta) * present)
        feats.append(torch.cos(k * theta) * present)
    feats.append(timing[:, TM_DOWNBEAT] * present)

    # Tempo from how fast the phase advances. Wrapped difference in turns per
    # frame is exact for anything slower than 25 beats per second; the frame a
    # red line lands on can jump, so the estimate is median-filtered over a
    # few frames before use.
    dtheta = theta[:, 1:] - theta[:, :-1]
    dtheta = torch.remainder(dtheta + math.pi, 2 * math.pi) - math.pi
    turns = torch.remainder(dtheta / (2 * math.pi), 1.0)
    turns = F.pad(turns, (1, 0), mode="replicate")
    turns = _median3(_median3(turns))
    bpm = turns * FPS * 60.0
    log_tempo = torch.log2(bpm.clamp(min=30.0) / REFERENCE_BPM).clamp(-2.0, 2.0)
    feats.append(log_tempo * present)

    feats.append(present)
    return torch.stack(feats, dim=1)


def _median3(x: torch.Tensor) -> torch.Tensor:
    padded = F.pad(x.unsqueeze(1), (1, 1), mode="replicate").squeeze(1)
    window = torch.stack([padded[:, :-2], padded[:, 1:-1], padded[:, 2:]], dim=-1)
    return window.median(dim=-1).values


class TimingEncoder(nn.Module):
    """
    Full-rate timing features -> one small feature map per U-Net level.

    Output i has `channels` channels at T / (compression * 2**i) frames,
    matching MelEncoder1D's level layout so the U-Net can take both side by side.
    """

    def __init__(self, compression: int, n_levels: int, channels: int = 32,
                 num_groups: int = 8):
        super().__init__()
        if compression < 1 or (compression & (compression - 1)) != 0:
            raise ValueError(f"compression must be a power of two, got {compression}")

        self.compression = compression
        self.n_levels = n_levels
        self.channels = channels

        layers: list[nn.Module] = [nn.Conv1d(N_TIMING_FEATURES, channels, 3, padding=1)]
        for _ in range(int(math.log2(compression))):
            layers += [
                Normalize(channels, num_groups),
                nn.SiLU(),
                nn.Conv1d(channels, channels, 4, stride=2, padding=1),
            ]
        self.stem = nn.Sequential(*layers)

        self.levels = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(n_levels):
            self.levels.append(nn.Sequential(
                Normalize(channels, num_groups), nn.SiLU(),
                nn.Conv1d(channels, channels, 3, padding=1),
            ))
            self.downsamples.append(
                nn.Conv1d(channels, channels, 4, stride=2, padding=1)
                if i < n_levels - 1 else None
            )

    def forward(self, timing: torch.Tensor) -> list[torch.Tensor]:
        h = self.stem(timing_features(timing))
        out: list[torch.Tensor] = []
        for i in range(self.n_levels):
            h = h + self.levels[i](h)
            out.append(h)
            if self.downsamples[i] is not None:
                h = self.downsamples[i](h)
        return out
