"""
taiko/model/unet.py

1D Diffusion U-Net for taiko beatmap generation.

Changes from previous version:
  - Audio conditioning: concat (AudioConcatBlock) instead of cross-attention
    Matches Mug-Diffusion's approach — simpler, faster, less memory
  - Gradient checkpointing: use_checkpoint flag on all ResBlocks
  - scale-shift norm for better conditioning injection
  - Audio features interpolated to match latent length before concat

Input:  noisy latent [B, z_channels, T_latent]
Output: predicted noise [B, z_channels, T_latent]
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from taiko.model.s4_block import S4StyleBlock


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
# Conditioning embedding (difficulty + style)
# ---------------------------------------------------------------------------

class CondEmbedding(nn.Module):
    def __init__(self, n_styles: int, dim: int, cfg_dropout: float = 0.5):
        super().__init__()
        self.cfg_dropout = cfg_dropout
        self.diff_proj   = nn.Linear(1, dim // 2)
        self.style_emb   = nn.Embedding(n_styles, dim // 2)
        self.out_proj    = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, difficulty: torch.Tensor, style: torch.Tensor,
                drop_cond: bool = False) -> torch.Tensor:
        # CFG dropout during training
        if self.training and not drop_cond:
            mask = torch.rand(difficulty.shape[0], device=difficulty.device) < self.cfg_dropout
            difficulty = torch.where(mask, torch.zeros_like(difficulty), difficulty)
            style      = torch.where(mask, torch.zeros_like(style), style)
        d = self.diff_proj(difficulty.unsqueeze(-1) / 10.0)
        s = self.style_emb(style)
        return self.out_proj(torch.cat([d, s], dim=-1))


# ---------------------------------------------------------------------------
# ResNet block with scale-shift conditioning + gradient checkpointing
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int,
                 num_groups: int = 8, dropout: float = 0.1,
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1    = Normalize(in_ch,  num_groups)
        self.conv1    = nn.Conv1d(in_ch,  out_ch, 3, padding=1)
        self.emb_proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_ch * 2))
        self.norm2    = Normalize(out_ch, num_groups)
        self.dropout  = nn.Dropout(dropout)
        self.conv2    = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip     = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def _forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        # Scale-shift conditioning (more expressive than additive)
        ss    = self.emb_proj(emb).unsqueeze(-1)          # [B, 2*out_ch, 1]
        scale, shift = ss.chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return grad_checkpoint(self._forward, x, emb, use_reentrant=False)
        return self._forward(x, emb)


# ---------------------------------------------------------------------------
# Audio concat block — Mug-Diffusion style
# Audio features are interpolated to match latent length then concatenated
# ---------------------------------------------------------------------------

class AudioConcatBlock(nn.Module):
    """
    Concatenates interpolated audio features to the latent channel dimension.
    Much simpler and faster than cross-attention.
    """
    def forward(self, x: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """
        x:     [B, C_lat, T_lat]
        audio: [B, C_aud, T_aud]  — will be interpolated to T_lat
        """
        if audio.shape[2] != x.shape[2]:
            audio = F.interpolate(audio, size=x.shape[2],
                                  mode="linear", align_corners=False)
        return torch.cat([x, audio], dim=1)


# ---------------------------------------------------------------------------
# Downsample / Upsample
# ---------------------------------------------------------------------------

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
    1D Diffusion U-Net with audio concat conditioning.

    Audio features are concatenated (not cross-attended) at each level,
    matching Mug-Diffusion's AudioConcatBlock approach.

    Gradient checkpointing available via use_checkpoint flag.
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
        audio_channels: list[int]  = None,
        cfg_dropout:    float      = 0.5,
        use_checkpoint: bool       = False,
        use_s4:         bool       = False,
    ):
        super().__init__()
        if channel_mult is None:
            channel_mult  = [1, 2, 4]
        if audio_channels is None:
            audio_channels = [64, 64, 128, 128]

        self.num_levels     = len(channel_mult)
        self.num_res_blocks = num_res_blocks
        self.use_checkpoint = use_checkpoint
        self.use_s4         = use_s4

        emb_dim = timestep_dim
        self.time_emb = TimestepEmbedding(d_model=128, dim=emb_dim)
        self.cond_emb = CondEmbedding(n_styles=n_styles, dim=emb_dim,
                                       cfg_dropout=cfg_dropout)

        # Input projection (before audio concat, so z_channels only)
        self.conv_in = nn.Conv1d(z_channels, base_channels, 3, padding=1)

        # ------------------------------------------------------------------
        # Encoder — AudioConcatBlock then ResBlocks at each level
        # After concat: in_ch = base_ch + audio_ch at that level
        # ------------------------------------------------------------------
        self.enc_concat = nn.ModuleList()   # AudioConcatBlock per level
        self.enc_proj   = nn.ModuleList()   # projection after concat
        self.enc_res    = nn.ModuleList()   # ResBlocks per level
        self.enc_s4     = nn.ModuleList()   # optional S4 per level
        self.enc_down   = nn.ModuleList()   # Downsample (or None)
        self.skip_channels = []

        in_ch = base_channels
        for lvl, mult in enumerate(channel_mult):
            out_ch  = base_channels * mult
            aud_ch  = audio_channels[min(lvl, len(audio_channels) - 1)]
            cat_ch  = in_ch + aud_ch       # channels after concat

            self.enc_concat.append(AudioConcatBlock())
            # Project concat channels down to out_ch
            self.enc_proj.append(nn.Conv1d(cat_ch, out_ch, 1))

            res_list = nn.ModuleList()
            for _ in range(num_res_blocks):
                res_list.append(ResBlock(out_ch, out_ch, emb_dim,
                                          num_groups, dropout, use_checkpoint))
            self.enc_res.append(res_list)
            if use_s4:
                self.enc_s4.append(S4StyleBlock(out_ch, use_checkpoint=use_checkpoint))
            else:
                self.enc_s4.append(None)
            self.skip_channels.append(out_ch)

            if lvl < self.num_levels - 1:
                self.enc_down.append(Downsample(out_ch))
            else:
                self.enc_down.append(None)

            in_ch = out_ch

        # ------------------------------------------------------------------
        # Bottleneck
        # ------------------------------------------------------------------
        self.mid_block1 = ResBlock(in_ch, in_ch, emb_dim, num_groups,
                                    use_checkpoint=use_checkpoint)
        self.mid_block2 = ResBlock(in_ch, in_ch, emb_dim, num_groups,
                                    use_checkpoint=use_checkpoint)

        # ------------------------------------------------------------------
        # Decoder — AudioConcatBlock + skip concat + ResBlocks
        # ------------------------------------------------------------------
        self.dec_concat = nn.ModuleList()
        self.dec_proj   = nn.ModuleList()
        self.dec_res    = nn.ModuleList()
        self.dec_s4     = nn.ModuleList()
        self.dec_up     = nn.ModuleList()

        for lvl in range(self.num_levels):
            enc_lvl = self.num_levels - 1 - lvl
            out_ch  = base_channels * channel_mult[enc_lvl]
            aud_ch  = audio_channels[min(enc_lvl, len(audio_channels) - 1)]
            skip_ch = self.skip_channels[enc_lvl]

            # in_ch (from prev decoder) + audio + skip
            cat_ch  = in_ch + aud_ch + skip_ch

            self.dec_concat.append(AudioConcatBlock())
            self.dec_proj.append(nn.Conv1d(cat_ch, out_ch, 1))

            res_list = nn.ModuleList()
            for _ in range(num_res_blocks):
                res_list.append(ResBlock(out_ch, out_ch, emb_dim,
                                          num_groups, dropout, use_checkpoint))
            self.dec_res.append(res_list)
            if use_s4 and lvl < self.num_levels - 1:
                self.dec_s4.append(S4StyleBlock(out_ch, use_checkpoint=use_checkpoint))
            else:
                self.dec_s4.append(None)

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

    def forward(
        self,
        x:              torch.Tensor,
        t:              torch.Tensor,
        audio_features: list[torch.Tensor],
        difficulty:     torch.Tensor,
        style:          torch.Tensor,
        drop_cond:      bool = False,
    ) -> torch.Tensor:

        emb = self.time_emb(t) + self.cond_emb(difficulty, style, drop_cond)
        h   = self.conv_in(x)

        # ---- Encoder ----
        skips = []
        for lvl in range(self.num_levels):
            af = audio_features[min(lvl, len(audio_features) - 1)]
            h  = self.enc_concat[lvl](h, af)    # concat audio
            h  = self.enc_proj[lvl](h)           # project to out_ch
            for blk in self.enc_res[lvl]:
                h = blk(h, emb)
            if self.enc_s4[lvl] is not None:
                h = self.enc_s4[lvl](h)
            skips.append(h)
            if self.enc_down[lvl] is not None:
                h = self.enc_down[lvl](h)

        # ---- Bottleneck ----
        h = self.mid_block1(h, emb)
        h = self.mid_block2(h, emb)

        # ---- Decoder ----
        for lvl in range(self.num_levels):
            enc_lvl = self.num_levels - 1 - lvl
            skip    = skips[enc_lvl]
            af      = audio_features[min(enc_lvl, len(audio_features) - 1)]

            # Align T dim for skip connection
            if h.shape[2] != skip.shape[2]:
                h = F.pad(h, (0, skip.shape[2] - h.shape[2]))

            h = self.dec_concat[lvl](h, af)                         # concat audio
            h = torch.cat([h, skip], dim=1)                         # concat skip
            h = self.dec_proj[lvl](h)                               # project
            for blk in self.dec_res[lvl]:
                h = blk(h, emb)
            if self.dec_s4[lvl] is not None:
                h = self.dec_s4[lvl](h)
            if self.dec_up[lvl] is not None:
                h = self.dec_up[lvl](h)

        return self.conv_out(F.silu(self.norm_out(h)))

    def count_parameters(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return f"{n/1e6:.1f}M"