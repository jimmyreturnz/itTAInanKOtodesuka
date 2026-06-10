"""
taiko/model/audio_encoder.py

MelSpectrogramScaleEncoder1D — adapted from Mug-Diffusion's wave.py.

Changes from previous version:
  - attention_resolutions now defaults to last 40% of levels instead of
    hardcoded {2,3} — works correctly for both 4-level and 10-level encoders
  - num_heads scales with channel count (out_ch // 32, min 1)
  - Everything else unchanged — arbitrary channel_mult length already worked

Input:  mel [B, 128, T_audio]
Output: list of tensors at each resolution level
        [B, C, T], [B, C, T/2], [B, C, T/4], ...
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def Normalize(channels: int, num_groups: int = 8) -> nn.GroupNorm:
    while channels % num_groups != 0:
        num_groups //= 2
    return nn.GroupNorm(num_groups, channels, eps=1e-6, affine=True)


class ResBlock1D(nn.Module):
    def __init__(self, channels: int, dilation: int = 1, num_groups: int = 8):
        super().__init__()
        self.norm1 = Normalize(channels, num_groups)
        self.conv1 = nn.Conv1d(channels, channels, 3,
                               padding=dilation, dilation=dilation)
        self.norm2 = Normalize(channels, num_groups)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class AudioEncoderLevel(nn.Module):
    def __init__(self, channels: int, i_block: int, num_groups: int = 8):
        super().__init__()
        dilations = (1, 2) if i_block % 2 == 0 else (4, 8)
        self.blocks = nn.Sequential(
            ResBlock1D(channels, dilations[0], num_groups),
            ResBlock1D(channels, dilations[1], num_groups),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class MelEncoder1D(nn.Module):
    """
    Multi-scale mel spectrogram encoder.

    attention_resolutions:
      If None, defaults to the last 40% of levels (coarser scales only).
      For 4 levels  → {2, 3}      (same as before)
      For 10 levels → {6,7,8,9}   (coarser half)
      Pass an explicit set to override.

    Input:  [B, 128, T]
    Output: list of feature tensors, one per level
    """

    def __init__(
        self,
        n_mels:               int       = 128,
        base_channels:        int       = 64,
        channel_mult:         list[int] = None,
        num_groups:           int       = 8,
        attention_resolutions: set      = None,
    ):
        super().__init__()
        if channel_mult is None:
            channel_mult = [1, 1, 2, 2]

        n_levels = len(channel_mult)

        # Default: attention on the last 40% of levels (coarse scales)
        if attention_resolutions is None:
            attn_start = int(n_levels * 0.6)
            attention_resolutions = set(range(attn_start, n_levels))

        self.conv_in = nn.Conv1d(n_mels, base_channels, kernel_size=3, padding=1)

        self.levels    = nn.ModuleList()
        self.downsamps = nn.ModuleList()
        self.attns     = nn.ModuleList()

        in_ch = base_channels
        for i, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            proj   = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
            level  = nn.ModuleDict({
                "proj":  proj,
                "block": AudioEncoderLevel(out_ch, i, num_groups),
            })
            self.levels.append(level)

            if i in attention_resolutions:
                self.attns.append(nn.MultiheadAttention(
                    out_ch,
                    num_heads=max(1, out_ch // 32),
                    batch_first=True,
                    dropout=0.0,
                ))
            else:
                self.attns.append(None)

            if i < n_levels - 1:
                self.downsamps.append(
                    nn.Conv1d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)
                )
            else:
                self.downsamps.append(None)

            in_ch = out_ch

        self.out_channels = [base_channels * m for m in channel_mult]

    def forward(self, mel: torch.Tensor) -> list[torch.Tensor]:
        h  = self.conv_in(mel)
        hs = []

        for i, level in enumerate(self.levels):
            h = level["proj"](h)
            h = level["block"](h)

            if self.attns[i] is not None:
                B, C, T = h.shape
                h_t     = h.permute(0, 2, 1)
                h_t, _  = self.attns[i](h_t, h_t, h_t)
                h       = h_t.permute(0, 2, 1)

            hs.append(h)

            if self.downsamps[i] is not None:
                h = self.downsamps[i](h)

        return hs