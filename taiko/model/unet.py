"""
taiko/model/unet.py

Updated CondEmbedding with avg_nps + peak_nps as separate signals.

Dimension allocation (emb_dim=256):
  difficulty : 48   (~19%)  Linear(1, 48)
  avg_nps    : 32   (~12%)  Linear(1, 32)   normalized /20
  peak_nps   : 32   (~12%)  Linear(1, 32)   normalized /30
  style      : 48   (~19%)  Embedding(n, 48)
  motif      : 96   (~37%)  MLP(16→96)
  total      : 256  → out_proj(256, emb_dim)

NPS normalization:
  avg_nps  / 20.0
  peak_nps / 30.0
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint as grad_checkpoint

from taiko.model.s4_block import S4StyleBlock

MOTIF_DIM = 16


def Normalize(
    channels: int,
    num_groups: int = 8
) -> nn.GroupNorm:

    while channels % num_groups != 0:
        num_groups //= 2

    return nn.GroupNorm(
        num_groups,
        channels,
        eps=1e-6,
        affine=True,
    )


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------

class TimestepEmbedding(nn.Module):

    def __init__(
        self,
        d_model: int,
        dim: int,
    ):
        super().__init__()

        self.d_model = d_model

        self.proj = nn.Sequential(
            nn.Linear(d_model, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        t: torch.Tensor
    ) -> torch.Tensor:

        half = self.d_model // 2

        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(
                half,
                device=t.device
            )
            / (half - 1)
        )

        args = t[:, None].float() * freqs[None]

        emb = torch.cat(
            [
                torch.cos(args),
                torch.sin(args),
            ],
            dim=-1,
        )

        return self.proj(emb)


# ---------------------------------------------------------------------------
# Conditioning embedding
# ---------------------------------------------------------------------------

class CondEmbedding(nn.Module):
    """
    Conditioning embedding:
      difficulty + avg_nps + peak_nps + style + motif
    """

    def __init__(
        self,
        n_styles: int,
        dim: int,
        cfg_dropout: float = 0.5,
        use_motif: bool = True,
        use_nps: bool = True,
    ):
        super().__init__()

        self.cfg_dropout = cfg_dropout
        self.use_motif = use_motif
        self.use_nps = use_nps

        # ---------------------------------------------------------------
        # Dimension allocation
        # ---------------------------------------------------------------

        if use_motif and use_nps:

            d_diff = max(8, int(dim * 0.19))
            d_anps = max(8, int(dim * 0.12))
            d_pnps = max(8, int(dim * 0.12))
            d_style = max(8, int(dim * 0.19))

            d_motif = (
                dim
                - d_diff
                - d_anps
                - d_pnps
                - d_style
            )

        elif use_motif:

            d_diff = max(8, int(dim * 0.25))
            d_anps = 0
            d_pnps = 0
            d_style = max(8, int(dim * 0.25))

            d_motif = (
                dim
                - d_diff
                - d_style
            )

        elif use_nps:

            d_diff = max(8, int(dim * 0.25))
            d_anps = max(8, int(dim * 0.15))
            d_pnps = max(8, int(dim * 0.15))

            d_style = (
                dim
                - d_diff
                - d_anps
                - d_pnps
            )

            d_motif = 0

        else:

            d_diff = max(8, dim // 2)
            d_anps = 0
            d_pnps = 0
            d_style = dim - d_diff
            d_motif = 0

        # ---------------------------------------------------------------
        # Main embeddings
        # ---------------------------------------------------------------

        self.diff_proj = nn.Linear(1, d_diff)

        self.style_emb = nn.Embedding(
            n_styles,
            d_style,
        )

        if use_nps:

            self.anps_proj = nn.Linear(1, d_anps)

            self.pnps_proj = nn.Linear(1, d_pnps)

        else:

            self.anps_proj = None
            self.pnps_proj = None

        if use_motif:

            self.motif_proj = nn.Sequential(
                nn.Linear(MOTIF_DIM, d_motif),
                nn.SiLU(),
                nn.Linear(d_motif, d_motif),
            )

        else:

            self.motif_proj = None

        total_in = (
            d_diff
            + d_style
            + (
                d_anps + d_pnps
                if use_nps else 0
            )
            + (
                d_motif
                if use_motif else 0
            )
        )

        self.out_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(total_in, dim),
        )

    def forward(
        self,
        difficulty: torch.Tensor,
        style: torch.Tensor,
        avg_nps: torch.Tensor | None = None,
        peak_nps: torch.Tensor | None = None,
        motif: torch.Tensor | None = None,
        drop_cond: bool = False,
    ) -> torch.Tensor:

        B = difficulty.shape[0]

        # ---------------------------------------------------------------
        # Classifier-free guidance dropout
        # ---------------------------------------------------------------

        if self.training and not drop_cond:

            mask = (
                torch.rand(
                    B,
                    device=difficulty.device
                )
                < self.cfg_dropout
            )

            difficulty = torch.where(
                mask,
                torch.zeros_like(difficulty),
                difficulty,
            )

            style = torch.where(
                mask,
                torch.zeros_like(style),
                style,
            )

            if avg_nps is not None:
                avg_nps = avg_nps * (~mask).float()

            if peak_nps is not None:
                peak_nps = peak_nps * (~mask).float()

            if motif is not None:
                motif = (
                    motif
                    * (~mask).float().unsqueeze(-1)
                )

        # ---------------------------------------------------------------
        # Build embedding parts
        # ---------------------------------------------------------------

        parts = [
            self.diff_proj(
                difficulty.unsqueeze(-1) / 10.0
            ),

            self.style_emb(style),
        ]

        if self.use_nps and self.anps_proj is not None:

            anps = (
                avg_nps
                if avg_nps is not None
                else torch.zeros(
                    B,
                    device=difficulty.device,
                )
            )

            pnps = (
                peak_nps
                if peak_nps is not None
                else torch.zeros(
                    B,
                    device=difficulty.device,
                )
            )

            parts.append(
                self.anps_proj(
                    anps.unsqueeze(-1) / 20.0
                )
            )

            parts.append(
                self.pnps_proj(
                    pnps.unsqueeze(-1) / 30.0
                )
            )

        if self.use_motif and self.motif_proj is not None:

            m = (
                motif
                if motif is not None
                else torch.zeros(
                    B,
                    MOTIF_DIM,
                    device=difficulty.device,
                )
            )

            parts.append(
                self.motif_proj(m)
            )

        return self.out_proj(
            torch.cat(parts, dim=-1)
        )


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
        emb_dim,
        num_groups=8,
        dropout=0.1,
        use_checkpoint=False,
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.norm1 = Normalize(
            in_ch,
            num_groups,
        )

        self.conv1 = nn.Conv1d(
            in_ch,
            out_ch,
            3,
            padding=1,
        )

        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_ch * 2),
        )

        self.norm2 = Normalize(
            out_ch,
            num_groups,
        )

        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_ch,
            out_ch,
            3,
            padding=1,
        )

        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

        self.skip = (
            nn.Conv1d(in_ch, out_ch, 1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def _forward(self, x, emb):

        h = self.conv1(
            F.silu(
                self.norm1(x)
            )
        )

        ss = self.emb_proj(emb).unsqueeze(-1)

        scale, shift = ss.chunk(2, dim=1)

        h = self.norm2(h) * (1 + scale) + shift

        h = self.conv2(
            self.dropout(
                F.silu(h)
            )
        )

        return h + self.skip(x)

    def forward(self, x, emb):

        if self.use_checkpoint and self.training:

            return grad_checkpoint(
                self._forward,
                x,
                emb,
                use_reentrant=False,
            )

        return self._forward(x, emb)


# ---------------------------------------------------------------------------
# Audio concat block
# ---------------------------------------------------------------------------

class AudioConcatBlock(nn.Module):

    def forward(self, x, audio):

        if audio.shape[2] != x.shape[2]:

            audio = F.interpolate(
                audio,
                size=x.shape[2],
                mode="linear",
                align_corners=False,
            )

        return torch.cat([x, audio], dim=1)


# ---------------------------------------------------------------------------
# Sampling layers
# ---------------------------------------------------------------------------

class Downsample(nn.Module):

    def __init__(self, ch):
        super().__init__()

        self.conv = nn.Conv1d(
            ch,
            ch,
            4,
            stride=2,
            padding=1,
        )

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):

    def __init__(self, ch):
        super().__init__()

        self.conv = nn.Conv1d(
            ch,
            ch,
            3,
            padding=1,
        )

    def forward(self, x):

        x = F.interpolate(
            x,
            scale_factor=2.0,
            mode="nearest",
        )

        return self.conv(x)


# ---------------------------------------------------------------------------
# Main UNet
# ---------------------------------------------------------------------------

class TaikoDiffusionUNet(nn.Module):

    def __init__(
        self,
        z_channels: int = 16,
        base_channels: int = 128,
        channel_mult: list[int] = None,
        num_res_blocks: int = 2,
        num_groups: int = 8,
        dropout: float = 0.1,
        timestep_dim: int = 256,
        n_styles: int = 4,
        audio_channels: list[int] = None,
        cfg_dropout: float = 0.5,
        use_checkpoint: bool = True,
        use_s4: bool = True,
        use_motif: bool = True,
        use_nps: bool = True,
    ):
        super().__init__()

        if channel_mult is None:
            channel_mult = [1, 2, 3, 4]

        if audio_channels is None:
            audio_channels = [128, 128, 128, 128]

        self.num_levels = len(channel_mult)
        self.num_res_blocks = num_res_blocks

        self.use_s4 = use_s4
        self.use_motif = use_motif
        self.use_nps = use_nps

        emb_dim = timestep_dim

        self.time_emb = TimestepEmbedding(
            d_model=128,
            dim=emb_dim,
        )

        self.cond_emb = CondEmbedding(
            n_styles=n_styles,
            dim=emb_dim,
            cfg_dropout=cfg_dropout,
            use_motif=use_motif,
            use_nps=use_nps,
        )

        self.conv_in = nn.Conv1d(
            z_channels,
            base_channels,
            3,
            padding=1,
        )

        # ---------------------------------------------------------------
        # Encoder
        # ---------------------------------------------------------------

        self.enc_concat = nn.ModuleList()
        self.enc_proj = nn.ModuleList()
        self.enc_res = nn.ModuleList()
        self.enc_s4 = nn.ModuleList()
        self.enc_down = nn.ModuleList()

        self.skip_channels = []

        in_ch = base_channels

        for lvl, mult in enumerate(channel_mult):

            out_ch = base_channels * mult

            aud_ch = audio_channels[
                min(lvl, len(audio_channels) - 1)
            ]

            self.enc_concat.append(
                AudioConcatBlock()
            )

            self.enc_proj.append(
                nn.Conv1d(
                    in_ch + aud_ch,
                    out_ch,
                    1,
                )
            )

            self.enc_res.append(
                nn.ModuleList(
                    [
                        ResBlock(
                            out_ch,
                            out_ch,
                            emb_dim,
                            num_groups,
                            dropout,
                            use_checkpoint,
                        )
                        for _ in range(num_res_blocks)
                    ]
                )
            )

            self.enc_s4.append(
                S4StyleBlock(
                    out_ch,
                    use_checkpoint=use_checkpoint,
                )
                if use_s4 else None
            )

            self.skip_channels.append(out_ch)

            self.enc_down.append(
                Downsample(out_ch)
                if lvl < self.num_levels - 1
                else None
            )

            in_ch = out_ch

        # ---------------------------------------------------------------
        # Middle
        # ---------------------------------------------------------------

        self.mid_block1 = ResBlock(
            in_ch,
            in_ch,
            emb_dim,
            num_groups,
            use_checkpoint=use_checkpoint,
        )

        self.mid_block2 = ResBlock(
            in_ch,
            in_ch,
            emb_dim,
            num_groups,
            use_checkpoint=use_checkpoint,
        )

        # ---------------------------------------------------------------
        # Decoder
        # ---------------------------------------------------------------

        self.dec_concat = nn.ModuleList()
        self.dec_proj = nn.ModuleList()
        self.dec_res = nn.ModuleList()
        self.dec_s4 = nn.ModuleList()
        self.dec_up = nn.ModuleList()

        for lvl in range(self.num_levels):

            enc_lvl = self.num_levels - 1 - lvl

            out_ch = (
                base_channels
                * channel_mult[enc_lvl]
            )

            aud_ch = audio_channels[
                min(enc_lvl, len(audio_channels) - 1)
            ]

            skip_ch = self.skip_channels[enc_lvl]

            self.dec_concat.append(
                AudioConcatBlock()
            )

            self.dec_proj.append(
                nn.Conv1d(
                    in_ch + aud_ch + skip_ch,
                    out_ch,
                    1,
                )
            )

            self.dec_res.append(
                nn.ModuleList(
                    [
                        ResBlock(
                            out_ch,
                            out_ch,
                            emb_dim,
                            num_groups,
                            dropout,
                            use_checkpoint,
                        )
                        for _ in range(num_res_blocks)
                    ]
                )
            )

            self.dec_s4.append(
                S4StyleBlock(
                    out_ch,
                    use_checkpoint=use_checkpoint,
                )
                if use_s4 and lvl < self.num_levels - 1
                else None
            )

            self.dec_up.append(
                Upsample(out_ch)
                if lvl < self.num_levels - 1
                else None
            )

            in_ch = out_ch

        # ---------------------------------------------------------------
        # Output
        # ---------------------------------------------------------------

        self.norm_out = Normalize(
            in_ch,
            num_groups,
        )

        self.conv_out = nn.Conv1d(
            in_ch,
            z_channels,
            3,
            padding=1,
        )

        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(
        self,
        x,
        t,
        audio_features,
        difficulty,
        style,
        avg_nps=None,
        peak_nps=None,
        motif=None,
        drop_cond=False,
    ):

        emb = self.time_emb(t) + self.cond_emb(
            difficulty,
            style,
            avg_nps=(
                avg_nps
                if self.use_nps else None
            ),
            peak_nps=(
                peak_nps
                if self.use_nps else None
            ),
            motif=(
                motif
                if self.use_motif else None
            ),
            drop_cond=drop_cond,
        )

        h = self.conv_in(x)

        skips = []

        # ---------------------------------------------------------------
        # Encoder
        # ---------------------------------------------------------------

        for lvl in range(self.num_levels):

            af = audio_features[
                min(
                    lvl,
                    len(audio_features) - 1
                )
            ]

            h = self.enc_concat[lvl](h, af)

            h = self.enc_proj[lvl](h)

            for blk in self.enc_res[lvl]:
                h = blk(h, emb)

            if self.enc_s4[lvl] is not None:
                h = self.enc_s4[lvl](h)

            skips.append(h)

            if self.enc_down[lvl] is not None:
                h = self.enc_down[lvl](h)

        # ---------------------------------------------------------------
        # Middle
        # ---------------------------------------------------------------

        h = self.mid_block1(h, emb)
        h = self.mid_block2(h, emb)

        # ---------------------------------------------------------------
        # Decoder
        # ---------------------------------------------------------------

        for lvl in range(self.num_levels):

            enc_lvl = self.num_levels - 1 - lvl

            skip = skips[enc_lvl]

            af = audio_features[
                min(
                    enc_lvl,
                    len(audio_features) - 1
                )
            ]

            if h.shape[2] != skip.shape[2]:

                h = F.pad(
                    h,
                    (
                        0,
                        skip.shape[2] - h.shape[2]
                    )
                )

            h = self.dec_concat[lvl](h, af)

            h = torch.cat([h, skip], dim=1)

            h = self.dec_proj[lvl](h)

            for blk in self.dec_res[lvl]:
                h = blk(h, emb)

            if self.dec_s4[lvl] is not None:
                h = self.dec_s4[lvl](h)

            if self.dec_up[lvl] is not None:
                h = self.dec_up[lvl](h)

        return self.conv_out(
            F.silu(
                self.norm_out(h)
            )
        )

    def count_parameters(self) -> str:

        n = sum(
            p.numel()
            for p in self.parameters()
        )

        return f"{n / 1e6:.1f}M"