"""
taiko/model/unet.py

1D Diffusion U-Net for taiko beatmap generation.
Conditioned on:
  - audio features (cross-attention at each level, multi-scale)
  - timestep embedding
  - difficulty + style (added to timestep embedding)

Architecture follows Mug-Diffusion's UNet1DModel but adapted for 1D taiko latents.

Input:  noisy latent [B, z_channels, T_latent]
Output: predicted noise [B, z_channels, T_latent]
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


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------

class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding + MLP projection."""

    def __init__(self, d_model: int, dim: int):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Linear(d_model, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        emb  = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.proj(emb)


# ---------------------------------------------------------------------------
# Conditioning embedding (difficulty + style)
# ---------------------------------------------------------------------------

class CondEmbedding(nn.Module):
    """Embeds difficulty (float) and style (int) into a vector."""

    def __init__(self, n_styles: int, dim: int):
        super().__init__()
        self.diff_proj  = nn.Linear(1, dim // 2)
        self.style_emb  = nn.Embedding(n_styles, dim // 2)
        self.out_proj   = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        difficulty: torch.Tensor,   # [B] float, 0-10
        style: torch.Tensor,        # [B] int
    ) -> torch.Tensor:
        d = self.diff_proj(difficulty.unsqueeze(-1) / 10.0)
        s = self.style_emb(style)
        return self.out_proj(torch.cat([d, s], dim=-1))


# ---------------------------------------------------------------------------
# U-Net building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """ResNet block conditioned on timestep + conditioning embedding."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int,
                 num_groups: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm1   = Normalize(in_ch,  num_groups)
        self.conv1   = nn.Conv1d(in_ch,  out_ch, 3, padding=1)
        self.emb_proj= nn.Linear(emb_dim, out_ch)
        self.norm2   = Normalize(out_ch, num_groups)
        self.dropout = nn.Dropout(dropout)
        self.conv2   = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.skip    = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(F.silu(emb))[:, :, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class CrossAttention1D(nn.Module):
    """Cross-attention to audio features at matching scale."""

    def __init__(self, channels: int, context_dim: int,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm    = Normalize(channels)
        self.attn    = nn.MultiheadAttention(
            channels, n_heads, dropout=dropout, batch_first=True,
            kdim=context_dim, vdim=context_dim,
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        x:       [B, C, T_latent]
        context: [B, C_audio, T_audio]
        """
        B, C, T = x.shape
        h = self.norm(x).permute(0, 2, 1)                    # [B, T, C]
        ctx = context.permute(0, 2, 1)                        # [B, T_audio, C_audio]
        h, _ = self.attn(h, ctx, ctx)
        return x + h.permute(0, 2, 1)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 4, stride=2, padding=1)
    def forward(self, x): return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, padding=1)
    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class TaikoDiffusionUNet(nn.Module):
    """
    1D Diffusion U-Net for taiko latent generation.

    Sized for GTX 1650 (4GB VRAM):
      z_channels=16, base_channels=64, channel_mult=[1,2,4]
      ~15M parameters
    """

    def __init__(
        self,
        z_channels:     int        = 16,
        base_channels:  int        = 64,
        channel_mult:   list[int]  = None,
        num_res_blocks: int        = 2,
        num_groups:     int        = 8,
        dropout:        float      = 0.1,
        timestep_dim:   int        = 256,
        n_styles:       int        = 4,    # standard, stream, speed, tech
        audio_channels: list[int]  = None, # output channels from audio encoder
        cfg_dropout:    float      = 0.1,
    ):
        super().__init__()
        if channel_mult is None:
            channel_mult = [1, 2, 4]
        if audio_channels is None:
            audio_channels = [64, 64, 128, 128]  # matches MelEncoder1D with base_channels=64
        # Always cast to int to avoid PyTorch type errors
        audio_channels = [int(c) for c in audio_channels]

        self.cfg_dropout = cfg_dropout

        # Timestep + conditioning embedding
        emb_dim = timestep_dim
        self.time_emb = TimestepEmbedding(d_model=128, dim=emb_dim)
        self.cond_emb = CondEmbedding(n_styles=n_styles, dim=emb_dim)

        # Input projection
        self.conv_in = nn.Conv1d(z_channels, base_channels, 3, padding=1)

        # Encoder path
        self.enc_blocks  = nn.ModuleList()
        self.enc_attns   = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.skip_channels = []

        in_ch = base_channels
        for i, mult in enumerate(channel_mult):
            out_ch   = base_channels * mult
            audio_ch = int(audio_channels[min(i, len(audio_channels) - 1)])

            for _ in range(num_res_blocks):
                self.enc_blocks.append(ResBlock(in_ch, out_ch, emb_dim, num_groups, dropout))
                self.enc_attns.append(CrossAttention1D(out_ch, audio_ch))
                self.skip_channels.append(out_ch)
                in_ch = out_ch

            if i < len(channel_mult) - 1:
                self.downsamples.append(Downsample(in_ch))
                self.downsamples.append(None)  # placeholder for level tracking
            else:
                self.downsamples.append(None)

        # Middle
        self.mid_block1 = ResBlock(in_ch, in_ch, emb_dim, num_groups)
        self.mid_attn   = CrossAttention1D(in_ch, audio_channels[-1])
        self.mid_block2 = ResBlock(in_ch, in_ch, emb_dim, num_groups)

        # Decoder path
        self.dec_blocks  = nn.ModuleList()
        self.dec_attns   = nn.ModuleList()
        self.upsamples   = nn.ModuleList()

        skip_ch_list = list(reversed(self.skip_channels))
        skip_idx = 0

        for i, mult in enumerate(reversed(channel_mult)):
            out_ch   = base_channels * mult
            audio_ch = audio_channels[min(len(channel_mult) - 1 - i, len(audio_channels) - 1)]

            for j in range(num_res_blocks):
                skip_ch = skip_ch_list[skip_idx] if skip_idx < len(skip_ch_list) else 0
                self.dec_blocks.append(ResBlock(in_ch + skip_ch, out_ch, emb_dim, num_groups, dropout))
                self.dec_attns.append(CrossAttention1D(out_ch, audio_ch))
                in_ch = out_ch
                skip_idx += 1

            if i < len(channel_mult) - 1:
                self.upsamples.append(Upsample(in_ch))
            else:
                self.upsamples.append(None)

        # Output
        self.norm_out = Normalize(in_ch, num_groups)
        self.conv_out = nn.Conv1d(in_ch, z_channels, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(
        self,
        x: torch.Tensor,                    # [B, z_channels, T_latent]
        t: torch.Tensor,                    # [B] int timesteps
        audio_features: list[torch.Tensor], # multi-scale from MelEncoder1D
        difficulty: torch.Tensor,           # [B] float
        style: torch.Tensor,                # [B] int
        drop_cond: bool = False,            # for CFG: drop conditioning
    ) -> torch.Tensor:

        # CFG dropout during training
        if self.training and not drop_cond:
            mask = torch.rand(x.shape[0], device=x.device) < self.cfg_dropout
            difficulty = torch.where(mask, torch.zeros_like(difficulty), difficulty)
            style      = torch.where(mask, torch.zeros_like(style),      style)

        # Embeddings
        emb = self.time_emb(t) + self.cond_emb(difficulty, style)

        h = self.conv_in(x)

        # Encoder
        skips = []
        audio_idx = 0
        down_idx  = 0
        block_idx = 0

        for i in range(len(self.enc_blocks)):
            h = self.enc_blocks[i](h, emb)
            # Interpolate audio features to match latent length
            af = audio_features[min(audio_idx, len(audio_features) - 1)]
            af = F.interpolate(af, size=h.shape[2], mode="linear", align_corners=False)
            h = self.enc_attns[i](h, af)
            skips.append(h)

            # Downsample at end of each level (every num_res_blocks blocks)
            if (i + 1) % 2 == 0 and down_idx < len(self.downsamples):
                if self.downsamples[down_idx] is not None:
                    h = self.downsamples[down_idx](h)
                    audio_idx = min(audio_idx + 1, len(audio_features) - 1)
                down_idx += 1

        # Middle
        h = self.mid_block1(h, emb)
        af = audio_features[-1]
        af = F.interpolate(af, size=h.shape[2], mode="linear", align_corners=False)
        h = self.mid_attn(h, af)
        h = self.mid_block2(h, emb)

        # Decoder
        up_idx    = 0
        audio_idx = len(audio_features) - 1

        for i in range(len(self.dec_blocks)):
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = self.dec_blocks[i](h, emb)
            af = audio_features[min(audio_idx, len(audio_features) - 1)]
            af = F.interpolate(af, size=h.shape[2], mode="linear", align_corners=False)
            h = self.dec_attns[i](h, af)

            if (i + 1) % 2 == 0 and up_idx < len(self.upsamples):
                if self.upsamples[up_idx] is not None:
                    h = self.upsamples[up_idx](h)
                    audio_idx = max(audio_idx - 1, 0)
                up_idx += 1

        h = self.conv_out(F.silu(self.norm_out(h)))
        return h

    def count_parameters(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return f"{n/1e6:.1f}M"