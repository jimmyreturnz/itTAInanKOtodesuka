"""
taiko/timing/model.py

TimingNet: per-frame beat and downbeat probability from our own mel.

Why train one when beat_this exists: the worry about general-purpose trackers
on rhythm-game music is fair -- J-core, speedcore and breakcore at 200+ BPM,
fake-out breaks, tempo ramps -- and it is exactly the music the ranked corpus
is made of. Every ranked map's red lines are hand-checked beat and barline
labels for its song, so the corpus is a free, in-domain training set of
~2,800 songs. That is Mapperatorinator's insight (its timing model trains on
red lines); this is a far smaller model, because it only has to find beats,
not write a map.

Architecture: a dilated temporal convolution network in the style of Böck and
Davies' TCN beat trackers -- non-causal, receptive field about 20 s -- on the
same 20 ms mel frames as the chart model, so it reads the packed shards with
no extra preprocessing. About 1.3M parameters; it exports to ONNX with
nothing unusual in it.

Targets come from the dataset's own timing stream, *after* augmentation, so a
rate-stretched window gets correspondingly stretched beats:

    beat      frames where the beat phase wraps (sub-frame crossing rounded)
    downbeat  those beats where the downbeat channel is at its peak

Neighbouring frames get half weight, which tolerates the +/-10 ms rounding of
a beat to a frame without teaching the network to smear its output.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from taiko.data.tensor_repr import TM_COS, TM_DOWNBEAT, TM_SIN


@dataclass
class TimingNetConfig:
    n_mels: int = 128
    channels: int = 128
    n_blocks: int = 10          # dilations 1, 2, 4, ... 512 frames
    kernel: int = 5
    dropout: float = 0.1


class _Block(nn.Module):
    def __init__(self, ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(ch, ch, kernel, padding=pad * 2, dilation=dilation * 2)
        self.norm = nn.GroupNorm(8, ch)
        self.drop = nn.Dropout(dropout)
        self.mix = nn.Conv1d(2 * ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = F.elu(self.conv1(x))
        b = F.elu(self.conv2(x))
        h = self.mix(self.drop(torch.cat([a, b], dim=1)))
        return self.norm(x + h)


class TimingNet(nn.Module):
    def __init__(self, config: TimingNetConfig | None = None):
        super().__init__()
        self.config = config or TimingNetConfig()
        c = self.config
        self.front = nn.Sequential(
            nn.Conv1d(c.n_mels, c.channels, 3, padding=1), nn.ELU(),
            nn.Conv1d(c.channels, c.channels, 3, padding=1), nn.ELU(),
        )
        self.blocks = nn.ModuleList(
            [_Block(c.channels, c.kernel, 2 ** i, c.dropout) for i in range(c.n_blocks)]
        )
        self.head = nn.Conv1d(c.channels, 2, 1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """[B, 128, T] mel -> [B, 2, T] logits (beat, downbeat)."""
        h = self.front(mel)
        for block in self.blocks:
            h = block(h)
        return self.head(h)


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #

def beat_targets(timing: torch.Tensor) -> torch.Tensor:
    """
    [B, 3, T] timing stream -> [B, 2, T] soft targets (beat, downbeat).

    Pure function of the stream, so it follows every augmentation the stream
    went through.
    """
    s, c = timing[:, TM_SIN], timing[:, TM_COS]
    present = (s * s + c * c) > 0.25
    phase = torch.remainder(torch.atan2(s, c) / (2 * np.pi), 1.0)

    prev = phase[:, :-1]
    cur = phase[:, 1:]
    wrap = (cur < prev - 0.5) & present[:, 1:] & present[:, :-1]
    # Where between frame t-1 and t the phase crossed 1.0.
    frac = (1.0 - prev) / ((1.0 - prev) + cur).clamp(min=1e-6)
    B, T = phase.shape
    beat = torch.zeros(B, T, device=timing.device)
    down = torch.zeros(B, T, device=timing.device)
    bi, ti = torch.nonzero(wrap, as_tuple=True)
    frame = (ti + frac[bi, ti]).round().long().clamp(0, T - 1)   # ti is t-1
    is_down = timing[bi, TM_DOWNBEAT, frame] > 0.9
    beat[bi, frame] = 1.0
    down[bi[is_down], frame[is_down]] = 1.0

    def widen(x: torch.Tensor) -> torch.Tensor:
        nb = F.max_pool1d(x.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
        return torch.maximum(x, 0.5 * nb)

    return torch.stack([widen(beat), widen(down)], dim=1)


def timing_loss(logits: torch.Tensor, targets: torch.Tensor,
                valid: torch.Tensor | None = None, pos_weight: float = 8.0) -> torch.Tensor:
    w = torch.tensor([pos_weight], device=logits.device)
    per = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=w, reduction="none")
    if valid is not None:
        per = per * valid.unsqueeze(1)
        return per.sum() / (valid.sum() * logits.shape[1]).clamp(min=1.0)
    return per.mean()


# --------------------------------------------------------------------------- #
# Save / load / run
# --------------------------------------------------------------------------- #

def save_timingnet(model: TimingNet, path: str | Path, **extra) -> None:
    from taiko.train import atomic_save
    atomic_save({"model": model.state_dict(), "config": asdict(model.config), **extra},
                Path(path))


def load_timingnet(path: str | Path, device: str = "cpu") -> TimingNet:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = TimingNet(TimingNetConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


@torch.no_grad()
def run_timingnet(model: TimingNet, signal: np.ndarray, sr: int,
                  device: str = "cpu", chunk: int = 6000) -> tuple[np.ndarray, np.ndarray]:
    """Waveform -> (beat, downbeat) probabilities at 50 fps."""
    from taiko.data.audio import MelExtractor
    mel = MelExtractor().extract_waveform(np.asarray(signal, dtype=np.float32), sr)
    x = torch.from_numpy(mel).unsqueeze(0).to(device)
    T = x.shape[-1]
    ctx = 1024                                           # receptive-field margin
    out = torch.zeros(2, T, device=device)
    for a in range(0, T, chunk):
        lo, hi = max(0, a - ctx), min(T, a + chunk + ctx)
        y = torch.sigmoid(model(x[:, :, lo:hi]))[0]
        out[:, a:min(a + chunk, T)] = y[:, a - lo:a - lo + min(chunk, T - a)]
    out = out.cpu().numpy()
    return out[0], out[1]
