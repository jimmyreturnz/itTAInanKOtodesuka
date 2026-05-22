"""
taiko/model/audio_encoder.py

Convolutional audio encoder: mel spectrogram [128, T] → context vectors [T', D]

Architecture:
    - Stack of Conv2d blocks compressing the 128 mel bins down to 1
    - Temporal downsampling by 4x (10ms frames → 40ms context steps)
    - Linear projection to model dimension
    - Sinusoidal positional encoding added

Why CNN not transformer here:
    - Local receptive field is ideal for onset detection
    - Much cheaper than self-attention over long audio sequences
    - The decoder handles long-range reasoning via cross-attention
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Conv block
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv2d → GroupNorm → GELU, with optional stride for downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] = (3, 3),
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (1, 1),
        groups: int = 8,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # GroupNorm: num_groups must divide out_channels
        num_groups = min(groups, out_channels)
        while out_channels % num_groups != 0:
            num_groups //= 2
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


# ---------------------------------------------------------------------------
# Audio encoder
# ---------------------------------------------------------------------------

class AudioEncoder(nn.Module):
    """
    Encodes mel spectrogram into a sequence of context vectors.

    Input:  [B, 1, 128, T]   (unsqueeze channel dim before passing)
    Output: [B, T//4, d_model]

    The 4x temporal downsampling means each output frame covers 40ms,
    giving ~200 frames for an 8-second window — manageable for cross-attention.

    Architecture:
        Block 1: [1, 128, T]  → [32, 64, T]    (halve mel bins)
        Block 2: [32, 64, T]  → [64, 32, T]    (halve mel bins)
        Block 3: [64, 32, T]  → [128, 16, T]   (halve mel bins)
        Block 4: [128, 16, T] → [256, 8, T//2] (halve mel bins + temporal 2x)
        Block 5: [256, 8, T//2] → [512, 1, T//4] (collapse mel + temporal 2x)
        Reshape: [B, 512, 1, T//4] → [B, T//4, 512]
        Project: [B, T//4, 512] → [B, T//4, d_model]
    """

    def __init__(
        self,
        n_mels: int = 128,
        d_model: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.blocks = nn.Sequential(
            # Freq downsampling only
            ConvBlock(1,   32,  kernel_size=(3,3), stride=(2,1), padding=(1,1)),   # [32, 64, T]
            ConvBlock(32,  64,  kernel_size=(3,3), stride=(2,1), padding=(1,1)),   # [64, 32, T]
            ConvBlock(64,  128, kernel_size=(3,3), stride=(2,1), padding=(1,1)),   # [128, 16, T]
            # Freq + temporal downsampling
            ConvBlock(128, 256, kernel_size=(3,3), stride=(2,2), padding=(1,1)),   # [256, 8, T//2]
            # Collapse freq, temporal downsampling
            ConvBlock(256, 512, kernel_size=(8,3), stride=(8,2), padding=(0,1)),   # [512, 1, T//4]
        )

        self.proj    = nn.Linear(512, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        # Lightweight transformer encoder for global context
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-norm (more stable)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [B, 128, T]  float32, values in [-1, 1]
        Returns:
            context: [B, T//4, d_model]
        """
        x = mel.unsqueeze(1)          # [B, 1, 128, T]
        x = self.blocks(x)            # [B, 512, 1, T//4]
        x = x.squeeze(2)              # [B, 512, T//4]
        x = x.permute(0, 2, 1)        # [B, T//4, 512]
        x = self.proj(x)              # [B, T//4, d_model]
        x = self.pos_enc(x)           # [B, T//4, d_model]
        x = self.transformer(x)       # [B, T//4, d_model]
        return x
