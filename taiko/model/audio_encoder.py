"""
taiko/model/audio_encoder.py

MelSpectrogramScaleEncoder1D — directly adapted from Mug-Diffusion's wave.py.

Key design (from wave.py):
  - Treats mel as 1D: [B, 128, T] not [B, 1, 128, T]
  - Dilated convolutions: (1,2) and (4,8) alternating
  - Attention at coarser resolutions only
  - Returns multi-scale features for U-Net skip connections

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
    """1D ResNet block with dilated convolutions — from Mug-Diffusion wave.py."""

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
    """One resolution level: 2 ResBlocks with alternating dilations."""

    def __init__(self, channels: int, i_block: int, num_groups: int = 8):
        super().__init__()
        # Alternating dilation pattern from Mug-Diffusion
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
    Adapted from Mug-Diffusion MelspectrogramScaleEncoder1D.

    Returns features at each scale for U-Net cross-attention.
    Smaller model than Mug-Diffusion to fit 4GB VRAM.

    Input:  [B, 128, T]
    Output: list of [B, C, T], [B, C, T//2], [B, C, T//4], [B, C, T//8]
    """

    def __init__(
        self,
        n_mels: int = 128,
        base_channels: int = 64,
        channel_mult: list[int] = None,
        num_groups: int = 8,
        attention_resolutions: set = None,
    ):
        super().__init__()
        if channel_mult is None:
            channel_mult = [1, 1, 2, 2]
        if attention_resolutions is None:
            attention_resolutions = {2, 3}  # only at coarser scales

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
                    out_ch, num_heads=max(1, out_ch // 32),
                    batch_first=True, dropout=0.0,
                ))
            else:
                self.attns.append(None)

            if i < len(channel_mult) - 1:
                self.downsamps.append(
                    nn.Conv1d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)
                )
            else:
                self.downsamps.append(None)

            in_ch = out_ch

        self.out_channels = [base_channels * m for m in channel_mult]

    def forward(self, mel: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            mel: [B, 128, T]
        Returns:
            list of feature tensors at each scale
        """
        h = self.conv_in(mel)
        hs = []

        for i, level in enumerate(self.levels):
            h = level["proj"](h)
            h = level["block"](h)

            # Attention at coarse scales
            if self.attns[i] is not None:
                B, C, T = h.shape
                h_t = h.permute(0, 2, 1)          # [B, T, C]
                h_t, _ = self.attns[i](h_t, h_t, h_t)
                h = h_t.permute(0, 2, 1)           # [B, C, T]

            hs.append(h)

            if self.downsamps[i] is not None:
                h = self.downsamps[i](h)

        return hs
