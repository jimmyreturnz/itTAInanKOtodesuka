"""
taiko/model/unet.py  (v2 — fixed audio channel tracking)

1D Diffusion U-Net for taiko beatmap generation.
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
# Timestep + conditioning embeddings
# ---------------------------------------------------------------------------

class TimestepEmbedding(nn.Module):
    def __init__(self, d_model: int, dim: int):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Linear(d_model, dim), nn.SiLU(), nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half  = self.d_model // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args  = t[:, None].float() * freqs[None]
        emb   = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.proj(emb)


class CondEmbedding(nn.Module):
    def __init__(self, n_styles: int, dim: int):
        super().__init__()
        self.diff_proj = nn.Linear(1, dim // 2)
        self.style_emb = nn.Embedding(n_styles, dim // 2)
        self.out_proj  = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, difficulty: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        d = self.diff_proj(difficulty.unsqueeze(-1) / 10.0)
        s = self.style_emb(style)
        return self.out_proj(torch.cat([d, s], dim=-1))


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int,
                 num_groups: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm1    = Normalize(in_ch,  num_groups)
        self.conv1    = nn.Conv1d(in_ch,  out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.norm2    = Normalize(out_ch, num_groups)
        self.dropout  = nn.Dropout(dropout)
        self.conv2    = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.skip     = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(F.silu(emb))[:, :, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class CrossAttention1D(nn.Module):
    """Cross-attention: latent [B,C,T] attends to audio [B,C_audio,T_audio]."""

    def __init__(self, channels: int, context_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # Ensure channels divisible by n_heads
        while channels % n_heads != 0:
            n_heads = max(1, n_heads // 2)
        self.norm = Normalize(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
            kdim=int(context_dim),
            vdim=int(context_dim),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h   = self.norm(x).permute(0, 2, 1)      # [B, T, C]
        ctx = context.permute(0, 2, 1)            # [B, T_audio, C_audio]
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
    1D Diffusion U-Net.
    audio_channels MUST match the actual output channels of MelEncoder1D.
    Use diffusion.py which derives this automatically from wave_model.out_channels.
    """

    def __init__(
        self,
        z_channels:     int       = 16,
        base_channels:  int       = 64,
        channel_mult:   list[int] = None,
        num_res_blocks: int       = 2,
        num_groups:     int       = 8,
        dropout:        float     = 0.1,
        timestep_dim:   int       = 256,
        n_styles:       int       = 4,
        audio_channels: list[int] = None,
        cfg_dropout:    float     = 0.1,
    ):
        super().__init__()
        if channel_mult is None:
            channel_mult = [1, 2, 4]
        if audio_channels is None:
            audio_channels = [64, 64, 128, 128]
        # Cast to int — always
        audio_channels = [int(c) for c in audio_channels]

        self.channel_mult   = channel_mult
        self.num_res_blocks = num_res_blocks
        self.cfg_dropout    = cfg_dropout
        self.n_levels       = len(channel_mult)

        emb_dim = timestep_dim
        self.time_emb = TimestepEmbedding(d_model=128, dim=emb_dim)
        self.cond_emb = CondEmbedding(n_styles=n_styles, dim=emb_dim)
        self.conv_in  = nn.Conv1d(z_channels, base_channels, 3, padding=1)

        # ------------------------------------------------------------------
        # Build encoder: track which audio level each block uses
        # enc_audio_level[i] = index into audio_features list
        # ------------------------------------------------------------------
        self.enc_blocks     = nn.ModuleList()
        self.enc_attns      = nn.ModuleList()
        self.enc_downsamples= nn.ModuleList()  # one per level (None for last)
        self.enc_audio_level = []   # which audio feature index each enc block uses
        self.skip_channels  = []

        in_ch = base_channels
        for level_idx, mult in enumerate(channel_mult):
            out_ch   = base_channels * mult
            audio_ch = audio_channels[min(level_idx, len(audio_channels) - 1)]

            for _ in range(num_res_blocks):
                self.enc_blocks.append(ResBlock(in_ch, out_ch, emb_dim, num_groups, dropout))
                self.enc_attns.append(CrossAttention1D(out_ch, audio_ch))
                self.enc_audio_level.append(level_idx)
                self.skip_channels.append(out_ch)
                in_ch = out_ch

            # Downsample between levels (not after last)
            if level_idx < self.n_levels - 1:
                self.enc_downsamples.append(Downsample(in_ch))
            else:
                self.enc_downsamples.append(None)

        # Middle — uses deepest audio features
        mid_audio_ch = audio_channels[min(self.n_levels - 1, len(audio_channels) - 1)]
        self.mid_block1 = ResBlock(in_ch, in_ch, emb_dim, num_groups)
        self.mid_attn   = CrossAttention1D(in_ch, mid_audio_ch)
        self.mid_block2 = ResBlock(in_ch, in_ch, emb_dim, num_groups)

        # ------------------------------------------------------------------
        # Build decoder: mirror of encoder
        # ------------------------------------------------------------------
        self.dec_blocks     = nn.ModuleList()
        self.dec_attns      = nn.ModuleList()
        self.dec_upsamples  = nn.ModuleList()
        self.dec_audio_level= []

        skip_ch_reversed = list(reversed(self.skip_channels))
        skip_idx = 0

        for level_idx, mult in enumerate(reversed(channel_mult)):
            audio_level = self.n_levels - 1 - level_idx
            audio_ch    = audio_channels[min(audio_level, len(audio_channels) - 1)]
            out_ch      = base_channels * mult

            for _ in range(num_res_blocks):
                skip_ch = skip_ch_reversed[skip_idx] if skip_idx < len(skip_ch_reversed) else 0
                self.dec_blocks.append(ResBlock(in_ch + skip_ch, out_ch, emb_dim, num_groups, dropout))
                self.dec_attns.append(CrossAttention1D(out_ch, audio_ch))
                self.dec_audio_level.append(audio_level)
                in_ch = out_ch
                skip_idx += 1

            if level_idx < self.n_levels - 1:
                self.dec_upsamples.append(Upsample(in_ch))
            else:
                self.dec_upsamples.append(None)

        self.norm_out = Normalize(in_ch, num_groups)
        self.conv_out = nn.Conv1d(in_ch, z_channels, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def _get_audio(self, audio_features: list[torch.Tensor], level: int, target_len: int) -> torch.Tensor:
        """Get audio feature at given level, interpolated to target length."""
        af = audio_features[min(level, len(audio_features) - 1)]
        if af.shape[2] != target_len:
            af = F.interpolate(af, size=target_len, mode="linear", align_corners=False)
        return af

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        audio_features: list[torch.Tensor],
        difficulty: torch.Tensor,
        style: torch.Tensor,
        drop_cond: bool = False,
    ) -> torch.Tensor:

        # CFG dropout
        if self.training and not drop_cond:
            mask       = torch.rand(x.shape[0], device=x.device) < self.cfg_dropout
            difficulty = torch.where(mask, torch.zeros_like(difficulty), difficulty)
            style      = torch.where(mask, torch.zeros_like(style),      style)

        emb = self.time_emb(t) + self.cond_emb(difficulty, style)
        h   = self.conv_in(x)

        # Encoder
        skips = []
        block_idx = 0
        for level_idx in range(self.n_levels):
            for _ in range(self.num_res_blocks):
                h = self.enc_blocks[block_idx](h, emb)
                af = self._get_audio(audio_features, self.enc_audio_level[block_idx], h.shape[2])
                h = self.enc_attns[block_idx](h, af)
                skips.append(h)
                block_idx += 1

            if self.enc_downsamples[level_idx] is not None:
                h = self.enc_downsamples[level_idx](h)

        # Middle
        af = self._get_audio(audio_features, self.n_levels - 1, h.shape[2])
        h  = self.mid_block1(h, emb)
        h  = self.mid_attn(h, af)
        h  = self.mid_block2(h, emb)

        # Decoder
        block_idx = 0
        up_idx    = 0
        for level_idx in range(self.n_levels):
            for _ in range(self.num_res_blocks):
                skip = skips.pop()
                # Match spatial size before concat (off-by-one from strided conv)
                if h.shape[2] != skip.shape[2]:
                    h = F.interpolate(h, size=skip.shape[2], mode="nearest")
                h    = torch.cat([h, skip], dim=1)
                h    = self.dec_blocks[block_idx](h, emb)
                af   = self._get_audio(audio_features, self.dec_audio_level[block_idx], h.shape[2])
                h    = self.dec_attns[block_idx](h, af)
                block_idx += 1

            if self.dec_upsamples[up_idx] is not None:
                h = self.dec_upsamples[up_idx](h)
            up_idx += 1

        return self.conv_out(F.silu(self.norm_out(h)))

    def count_parameters(self) -> str:
        return f"{sum(p.numel() for p in self.parameters())/1e6:.1f}M"