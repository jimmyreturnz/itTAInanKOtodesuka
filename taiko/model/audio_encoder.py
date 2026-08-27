"""
taiko/model/audio_encoder.py

Multi-scale mel encoder whose levels land on the U-Net's latent resolutions.

The resolution problem this solves
----------------------------------
The previous encoder emitted its first level at full mel resolution -- 1500
frames for a 30 s window -- while U-Net level 0 operates on the latent grid,
1500/16 = 93 frames. The two were reconciled inside the U-Net with

    F.interpolate(audio, size=x.shape[2], mode="linear")

which *point-samples*: about 15 of every 16 mel frames were discarded, and the
survivors were picked by an arbitrary phase. Onset energy -- the single most
important signal for deciding where a note goes -- was thrown away at exactly
the level where the U-Net decides where notes go.

The fix is structural, not a better interpolation. A strided stem takes the mel
down to latent resolution with *learned* pooling, so every input frame
contributes to the features the U-Net actually reads. From there one encoder
level per U-Net level, each halving as the U-Net halves:

    mel [128, T]                         T frames at 20 ms
      stem: log2(compression) strided convs
        -> [C, T/16]                     latent resolution
      level 0 -> f0 [C0, T/16 ]          U-Net level 0
      level 1 -> f1 [C1, T/32 ]          U-Net level 1
      level 2 -> f2 [C2, T/64 ]          U-Net level 2
      level 3 -> f3 [C3, T/128]          U-Net level 3

The stem keeps a parallel onset path. Learned strided convolution alone tends
to smooth over the transients that matter most here, so a fixed spectral-flux
branch -- positive first difference across mel bins, the classic onset detection
function -- is computed at full resolution and max-pooled down. Max pooling is
the point: a transient anywhere inside a 16-frame span survives to the latent
frame instead of being averaged into the noise floor.

Why the old p2 profile was strictly worse
-----------------------------------------
It built ten encoder levels, but the U-Net read `audio_features[0..3]`. Levels
4-9 received no gradient and contributed nothing -- pure wasted FLOPs -- and its
used levels carried fewer channels than p1's. `n_levels` is now derived from the
U-Net's own depth so an encoder level that nothing consumes cannot be built.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def Normalize(channels: int, num_groups: int = 8) -> nn.GroupNorm:
    while channels % num_groups != 0:
        num_groups //= 2
    return nn.GroupNorm(num_groups, channels, eps=1e-6, affine=True)


# --------------------------------------------------------------------------- #
# Onset path
# --------------------------------------------------------------------------- #

class SpectralFlux(nn.Module):
    """
    Fixed spectral-flux onset function, computed before any downsampling.

    flux(t) = sum over mel bins of max(0, mel[b, t] - mel[b, t-1])

    Not learned, and deliberately so: it costs nothing, it is exactly the signal
    a strided convolution is worst at preserving, and it gives the network a
    clean transient track from the first layer rather than something it has to
    discover through a 16x bottleneck.
    """

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        diff = mel[:, :, 1:] - mel[:, :, :-1]
        flux = F.relu(diff).sum(dim=1, keepdim=True)
        flux = F.pad(flux, (1, 0))

        # Per-clip normalisation: absolute flux scales with mix loudness, which
        # is not something the chart should depend on.
        peak = flux.amax(dim=2, keepdim=True).clamp(min=1e-5)
        return flux / peak


class OnsetStem(nn.Module):
    """
    Carries transients down to latent resolution by max pooling.

    Average pooling would dilute a 1-frame transient by the pooling factor;
    max pooling keeps it. Alongside the max, the mean gives the network a sense
    of how busy the span was overall.
    """

    def __init__(self, compression: int, out_channels: int):
        super().__init__()
        self.compression = compression
        self.flux = SpectralFlux()
        self.proj = nn.Conv1d(2, out_channels, kernel_size=3, padding=1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        flux = self.flux(mel)
        pooled_max  = F.max_pool1d(flux, self.compression, ceil_mode=True)
        pooled_mean = F.avg_pool1d(flux, self.compression, ceil_mode=True)
        return self.proj(torch.cat([pooled_max, pooled_mean], dim=1))


# --------------------------------------------------------------------------- #
# Blocks
# --------------------------------------------------------------------------- #

class ResBlock1D(nn.Module):
    def __init__(self, channels: int, dilation: int = 1, num_groups: int = 8):
        super().__init__()
        self.norm1 = Normalize(channels, num_groups)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm2 = Normalize(channels, num_groups)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class EncoderLevel(nn.Module):
    """Two dilated residual blocks. Alternating dilations widen the receptive
    field without extra downsampling."""

    def __init__(self, channels: int, index: int, num_groups: int = 8):
        super().__init__()
        d1, d2 = (1, 2) if index % 2 == 0 else (4, 8)
        self.blocks = nn.Sequential(
            ResBlock1D(channels, d1, num_groups),
            ResBlock1D(channels, d2, num_groups),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class SelfAttention1D(nn.Module):
    """Pre-norm self-attention over time, residual."""

    def __init__(self, channels: int, num_heads: int | None = None):
        super().__init__()
        if num_heads is None:
            num_heads = max(1, min(8, channels // 64))
        while channels % num_heads != 0:
            num_heads -= 1
        self.norm = Normalize(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x).transpose(1, 2)
        h, _ = self.attn(h, h, h, need_weights=False)
        return x + h.transpose(1, 2)


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #

class MelEncoder1D(nn.Module):
    """
    mel [B, 128, T] -> one feature map per U-Net level.

    Output i has shape [B, out_channels[i], T / (compression * 2**i)], which is
    exactly the resolution U-Net level i works at.

    Args:
        compression:  autoencoder temporal compression, so the stem knows how
                      far to stride. Must be a power of two.
        n_levels:     number of U-Net levels to feed. One feature map each; no
                      level is produced that nothing consumes.
        attention_levels: which levels get self-attention. Defaults to the
                      coarsest half, where sequences are short enough for
                      quadratic cost to be irrelevant.
    """

    def __init__(
        self,
        n_mels:           int = 128,
        base_channels:    int = 128,
        channel_mult:     list[int] | tuple[int, ...] | None = None,
        compression:      int = 16,
        n_levels:         int | None = None,
        num_groups:       int = 8,
        attention_levels: set[int] | None = None,
        onset_channels:   int = 32,
    ):
        super().__init__()

        if channel_mult is None:
            channel_mult = (1, 1, 2, 4)
        channel_mult = tuple(channel_mult)

        if n_levels is None:
            n_levels = len(channel_mult)
        if n_levels > len(channel_mult):
            raise ValueError(
                f"need a channel multiplier per level: {n_levels} levels but "
                f"channel_mult has {len(channel_mult)} entries"
            )
        channel_mult = channel_mult[:n_levels]

        if compression < 1 or (compression & (compression - 1)) != 0:
            raise ValueError(f"compression must be a power of two, got {compression}")

        self.n_levels    = n_levels
        self.compression = compression
        self.channel_mult = channel_mult

        stem_out = base_channels * channel_mult[0]

        # --- stem: mel resolution -> latent resolution --------------------- #
        n_strides = int(math.log2(compression))
        layers: list[nn.Module] = [nn.Conv1d(n_mels, base_channels, 3, padding=1)]
        ch = base_channels
        for _ in range(n_strides):
            layers += [
                Normalize(ch, num_groups),
                nn.SiLU(),
                nn.Conv1d(ch, min(ch * 2, stem_out), 4, stride=2, padding=1),
            ]
            ch = min(ch * 2, stem_out)
        if ch != stem_out:
            layers.append(nn.Conv1d(ch, stem_out, 1))
        self.stem = nn.Sequential(*layers)

        self.onset_stem = OnsetStem(compression, onset_channels)
        self.merge = nn.Conv1d(stem_out + onset_channels, stem_out, 1)

        # --- one level per U-Net level ------------------------------------- #
        if attention_levels is None:
            attention_levels = set(range(n_levels // 2, n_levels))
        self.attention_levels = attention_levels

        self.projections = nn.ModuleList()
        self.levels      = nn.ModuleList()
        self.attentions  = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        in_ch = stem_out
        for i, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            self.projections.append(
                nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
            )
            self.levels.append(EncoderLevel(out_ch, i, num_groups))
            self.attentions.append(
                SelfAttention1D(out_ch) if i in attention_levels else None
            )
            self.downsamples.append(
                nn.Conv1d(out_ch, out_ch, 4, stride=2, padding=1)
                if i < n_levels - 1 else None
            )
            in_ch = out_ch

        self.out_channels = [base_channels * m for m in channel_mult]

    def forward(self, mel: torch.Tensor) -> list[torch.Tensor]:
        h = self.stem(mel)
        onset = self.onset_stem(mel)

        # ceil_mode pooling and strided convolution can disagree by one frame.
        if onset.shape[2] != h.shape[2]:
            onset = F.interpolate(onset, size=h.shape[2], mode="nearest")

        h = self.merge(torch.cat([h, onset], dim=1))

        features: list[torch.Tensor] = []
        for i in range(self.n_levels):
            h = self.projections[i](h)
            h = self.levels[i](h)
            if self.attentions[i] is not None:
                h = self.attentions[i](h)
            features.append(h)
            if self.downsamples[i] is not None:
                h = self.downsamples[i](h)

        return features

    def latent_frames(self, mel_frames: int) -> int:
        """
        Latent length the stem produces for a given mel length.

        Matches Conv1d(kernel=4, stride=2, padding=1) exactly:
        out = floor((in + 2*pad - kernel) / stride) + 1. Guessing with a
        ceiling divide is off by one on odd lengths, which is enough to
        misalign the audio features against the VAE latent.
        """
        n = mel_frames
        for _ in range(int(math.log2(self.compression))):
            n = (n + 2 * 1 - 4) // 2 + 1
        return n

    def count_parameters(self) -> str:
        return f"{sum(p.numel() for p in self.parameters()) / 1e6:.1f}M"
