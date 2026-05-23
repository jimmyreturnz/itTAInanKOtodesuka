"""
taiko/model/unet.py

1D Diffusion U-Net for taiko beatmap generation.
Conditioned on:
  - audio features (cross-attention at each level, multi-scale)
  - timestep embedding
  - difficulty + style (added to timestep embedding)

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
# Conditioning embedding
# ---------------------------------------------------------------------------

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
# U-Net building blocks
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
    def __init__(self, channels: int, context_dim: int,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = Normalize(channels)
        self.attn = nn.MultiheadAttention(
            channels, n_heads, dropout=dropout, batch_first=True,
            kdim=context_dim, vdim=context_dim,
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        x:       [B, C,       T_latent]
        context: [B, C_audio, T_audio]  — interpolated to T_latent before call
        """
        h = self.norm(x).permute(0, 2, 1)       # [B, T, C]
        ctx = context.permute(0, 2, 1)           # [B, T_audio, C_audio]
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

    FIX: encoder/decoder level structure is now explicit — one entry per
    *level* (not per block), so audio_idx advances exactly once per level
    and always matches the correct audio_channels entry.
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
        n_styles:       int        = 4,
        audio_channels: list[int]  = None,  # must match MelEncoder1D.out_channels
        cfg_dropout:    float      = 0.1,
    ):
        super().__init__()
        if channel_mult is None:
            channel_mult  = [1, 2, 4]
        if audio_channels is None:
            audio_channels = [64, 64, 128, 128]

        self.num_levels    = len(channel_mult)
        self.num_res_blocks = num_res_blocks
        self.cfg_dropout   = cfg_dropout

        emb_dim = timestep_dim
        self.time_emb = TimestepEmbedding(d_model=128, dim=emb_dim)
        self.cond_emb = CondEmbedding(n_styles=n_styles, dim=emb_dim)
        self.conv_in  = nn.Conv1d(z_channels, base_channels, 3, padding=1)

        # ------------------------------------------------------------------
        # Encoder — one ModuleList of lists per level
        # enc_res[lvl]  : nn.ModuleList of ResBlocks for that level
        # enc_attn[lvl] : CrossAttention1D for that level
        # enc_down[lvl] : Downsample (or None for last level)
        # ------------------------------------------------------------------
        self.enc_res  = nn.ModuleList()
        self.enc_attn = nn.ModuleList()
        self.enc_down = nn.ModuleList()
        self.skip_channels = []   # out_ch at each level (for decoder)

        in_ch = base_channels
        for lvl, mult in enumerate(channel_mult):
            out_ch   = base_channels * mult
            aud_ch   = audio_channels[min(lvl, len(audio_channels) - 1)]

            res_list = nn.ModuleList()
            cur_in   = in_ch
            for _ in range(num_res_blocks):
                res_list.append(ResBlock(cur_in, out_ch, emb_dim, num_groups, dropout))
                cur_in = out_ch
            self.enc_res.append(res_list)
            self.enc_attn.append(CrossAttention1D(out_ch, aud_ch, n_heads=max(1, out_ch // 64)))

            if lvl < self.num_levels - 1:
                self.enc_down.append(Downsample(out_ch))
            else:
                self.enc_down.append(None)   # last level: no downsample

            self.skip_channels.append(out_ch)
            in_ch = out_ch

        # ------------------------------------------------------------------
        # Middle
        # ------------------------------------------------------------------
        self.mid_block1 = ResBlock(in_ch, in_ch, emb_dim, num_groups)
        self.mid_attn   = CrossAttention1D(in_ch, audio_channels[min(self.num_levels - 1,
                                                                       len(audio_channels) - 1)],
                                           n_heads=max(1, in_ch // 64))
        self.mid_block2 = ResBlock(in_ch, in_ch, emb_dim, num_groups)

        # ------------------------------------------------------------------
        # Decoder — mirrors encoder in reverse
        # dec_res[lvl]  : nn.ModuleList of ResBlocks (first takes skip concat)
        # dec_attn[lvl] : CrossAttention1D
        # dec_up[lvl]   : Upsample (or None for last level = finest)
        # ------------------------------------------------------------------
        self.dec_res  = nn.ModuleList()
        self.dec_attn = nn.ModuleList()
        self.dec_up   = nn.ModuleList()

        for lvl, mult in enumerate(reversed(channel_mult)):
            enc_lvl  = self.num_levels - 1 - lvl          # corresponding encoder level
            out_ch   = base_channels * mult
            skip_ch  = self.skip_channels[enc_lvl]
            aud_ch   = audio_channels[min(enc_lvl, len(audio_channels) - 1)]

            res_list = nn.ModuleList()
            cur_in   = in_ch + skip_ch                     # first block gets skip concat
            for i in range(num_res_blocks):
                res_list.append(ResBlock(cur_in, out_ch, emb_dim, num_groups, dropout))
                cur_in = out_ch
            self.dec_res.append(res_list)
            self.dec_attn.append(CrossAttention1D(out_ch, aud_ch, n_heads=max(1, out_ch // 64)))

            if lvl < self.num_levels - 1:
                self.dec_up.append(Upsample(out_ch))
            else:
                self.dec_up.append(None)

            in_ch = out_ch

        # Output
        self.norm_out = Normalize(in_ch, num_groups)
        self.conv_out = nn.Conv1d(in_ch, z_channels, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def _interpolate_audio(self, af: torch.Tensor, t_len: int) -> torch.Tensor:
        if af.shape[2] == t_len:
            return af
        return F.interpolate(af, size=t_len, mode="linear", align_corners=False)

    def forward(
        self,
        x:              torch.Tensor,
        t:              torch.Tensor,
        audio_features: list[torch.Tensor],
        difficulty:     torch.Tensor,
        style:          torch.Tensor,
        drop_cond:      bool = False,
    ) -> torch.Tensor:

        # CFG dropout
        if self.training and not drop_cond:
            mask       = torch.rand(x.shape[0], device=x.device) < self.cfg_dropout
            difficulty = torch.where(mask, torch.zeros_like(difficulty), difficulty)
            style      = torch.where(mask, torch.zeros_like(style), style)

        emb = self.time_emb(t) + self.cond_emb(difficulty, style)
        h   = self.conv_in(x)

        # ---- Encoder ----
        skips = []
        for lvl in range(self.num_levels):
            for blk in self.enc_res[lvl]:
                h = blk(h, emb)
            af = self._interpolate_audio(
                audio_features[min(lvl, len(audio_features) - 1)], h.shape[2]
            )
            h = self.enc_attn[lvl](h, af)
            skips.append(h)
            if self.enc_down[lvl] is not None:
                h = self.enc_down[lvl](h)

        # ---- Middle ----
        h = self.mid_block1(h, emb)
        af = self._interpolate_audio(audio_features[-1], h.shape[2])
        h = self.mid_attn(h, af)
        h = self.mid_block2(h, emb)

        # ---- Decoder ----
        for lvl in range(self.num_levels):
            enc_lvl = self.num_levels - 1 - lvl
            skip = skips[enc_lvl]
            # Align T dim in case of odd-length sequences
            if h.shape[2] != skip.shape[2]:
                h = F.pad(h, (0, skip.shape[2] - h.shape[2]))
            h = torch.cat([h, skip], dim=1)
            for blk in self.dec_res[lvl]:
                h = blk(h, emb)
            af = self._interpolate_audio(
                audio_features[min(enc_lvl, len(audio_features) - 1)], h.shape[2]
            )
            h = self.dec_attn[lvl](h, af)
            if self.dec_up[lvl] is not None:
                h = self.dec_up[lvl](h)

        return self.conv_out(F.silu(self.norm_out(h)))

    def count_parameters(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return f"{n/1e6:.1f}M"