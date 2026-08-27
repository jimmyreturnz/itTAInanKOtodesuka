"""
taiko/model/unet.py

Conditional 1D U-Net that denoises chart latents.

Inputs at every level:
    z            the noisy latent being denoised
    audio        one feature map per level, already at that level's resolution
    timing       the beat grid, downsampled to each level
    emb          timestep + conditioning, injected through FiLM

Two fixes to how conditioning works
-----------------------------------
1. A real null token. The previous unconditional branch was built by setting
   `style = 0` -- but style 0 is "standard", a real class. Guidance therefore
   pushed samples away from standard style rather than away from
   unconditionality, so asking for a style did something close to the opposite
   of what it should. Conditioning now carries a learned null embedding and
   STYLE_NULL is a dedicated index.

2. A mask, not a zero. Dropping a motif dimension by zeroing it says "this
   dimension is zero", which is a specific request -- zero big notes, zero
   density -- and not the same thing as "unspecified". Each dimension carries
   a companion mask bit so the two are distinguishable.

The timing stream enters as its own conditioning channel rather than as
something the model must infer, and is pooled to each level with average
pooling: phase is continuous, so averaging is meaningful where it would not be
for a pulse train.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from taiko.data.conditioning import N_STYLES, STYLE_NULL
from taiko.data.motif import MOTIF_DIM
from taiko.data.tensor_repr import N_TIMING_CHANNELS
from taiko.model.s4_block import S4StyleBlock


def Normalize(channels: int, num_groups: int = 8) -> nn.GroupNorm:
    while channels % num_groups != 0:
        num_groups //= 2
    return nn.GroupNorm(num_groups, channels, eps=1e-6, affine=True)


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #

class TimestepEmbedding(nn.Module):
    def __init__(self, d_model: int, dim: int):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Linear(d_model, dim), nn.SiLU(), nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = t[:, None].float() * freqs[None]
        return self.proj(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))


class ConditionEmbedding(nn.Module):
    """
    difficulty + avg_nps + peak_nps + style + motif -> one embedding vector.

    Scalars arrive already normalised by `taiko.data.conditioning`, so nothing
    here rescales them -- a normalisation constant that lives in two places is
    a train/inference skew waiting to happen.
    """

    def __init__(self, dim: int, n_styles: int = N_STYLES):
        super().__init__()

        d_scalar = max(16, dim // 8)
        d_style  = max(16, dim // 6)
        d_motif  = max(32, dim // 3)

        self.diff_proj = nn.Linear(1, d_scalar)
        self.anps_proj = nn.Linear(1, d_scalar)
        self.pnps_proj = nn.Linear(1, d_scalar)
        self.style_emb = nn.Embedding(n_styles, d_style)

        # The mask is concatenated with the values so the network can tell
        # "unspecified" from "requested zero".
        self.motif_proj = nn.Sequential(
            nn.Linear(MOTIF_DIM * 2, d_motif), nn.SiLU(), nn.Linear(d_motif, d_motif),
        )

        total = d_scalar * 3 + d_style + d_motif
        self.out_proj = nn.Sequential(nn.SiLU(), nn.Linear(total, dim))

        # The learned unconditional embedding. Guidance extrapolates away from
        # this point, so it must mean "nothing requested" and nothing else.
        self.null_embedding = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.null_embedding, std=0.02)

    def forward(
        self,
        difficulty: torch.Tensor,
        style: torch.Tensor,
        avg_nps: torch.Tensor | None = None,
        peak_nps: torch.Tensor | None = None,
        motif: torch.Tensor | None = None,
        motif_mask: torch.Tensor | None = None,
        drop_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            drop_mask: [B] bool. True replaces that sample's conditioning with
                the learned null embedding. Callers own this -- the layer does
                not sample it -- so the training loop and the guidance path use
                exactly the same mechanism.
        """
        B = difficulty.shape[0]
        device = difficulty.device

        def scalar(x: torch.Tensor | None) -> torch.Tensor:
            if x is None:
                return torch.zeros(B, 1, device=device, dtype=difficulty.dtype)
            return x.reshape(B, 1).to(difficulty.dtype)

        if motif is None:
            motif = torch.zeros(B, MOTIF_DIM, device=device, dtype=difficulty.dtype)
        if motif_mask is None:
            motif_mask = torch.ones_like(motif)

        parts = [
            self.diff_proj(scalar(difficulty)),
            self.anps_proj(scalar(avg_nps)),
            self.pnps_proj(scalar(peak_nps)),
            self.style_emb(style.reshape(B)),
            self.motif_proj(torch.cat([motif, motif_mask], dim=-1)),
        ]
        emb = self.out_proj(torch.cat(parts, dim=-1))

        if drop_mask is not None:
            # A blend by a 0/1 float rather than torch.where. The two are
            # numerically identical, but the blend keeps null_embedding in the
            # autograd graph on every step, with coefficient zero when nothing
            # was dropped. Under DistributedDataParallel a parameter that gets
            # no gradient in some steps aborts the reduction, and this batch
            # can legitimately contain no dropped samples.
            d = drop_mask.reshape(B, 1).to(emb.dtype)
            null = self.null_embedding.to(emb.dtype).expand_as(emb)
            emb = emb * (1.0 - d) + null * d

        return emb

    def unconditional(self, batch: int, device, dtype=torch.float32) -> torch.Tensor:
        """The null embedding directly, for the guidance branch."""
        return self.null_embedding.to(device=device, dtype=dtype).expand(batch, -1)


# --------------------------------------------------------------------------- #
# Blocks
# --------------------------------------------------------------------------- #

class ResBlock(nn.Module):
    """Residual block with FiLM conditioning from the timestep/condition vector."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int,
                 num_groups: int = 8, dropout: float = 0.1,
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.norm1 = Normalize(in_ch, num_groups)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_ch * 2))
        self.norm2 = Normalize(out_ch, num_groups)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)

        # Zero-init the residual branch so the block starts as identity.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def _forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb_proj(emb).unsqueeze(-1).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return grad_checkpoint(self._forward, x, emb, use_reentrant=False)
        return self._forward(x, emb)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


def _match_length(x: torch.Tensor, length: int) -> torch.Tensor:
    """Trim or edge-pad to an exact length. Odd sequence lengths make a level's
    up and down paths disagree by one frame."""
    if x.shape[-1] == length:
        return x
    if x.shape[-1] > length:
        return x[..., :length]
    return F.pad(x, (0, length - x.shape[-1]), mode="replicate")


# --------------------------------------------------------------------------- #
# U-Net
# --------------------------------------------------------------------------- #

class TaikoDiffusionUNet(nn.Module):
    """
    Predicts the diffusion target for a chart latent.

    Audio features must already be at each level's resolution -- that is the
    audio encoder's job now, and the reason it has a strided stem. What arrives
    here is aligned, so nothing has to interpolate a 16x resolution gap and
    discard most of the onset information doing it.
    """

    def __init__(
        self,
        z_channels:     int = 16,
        base_channels:  int = 128,
        channel_mult:   tuple[int, ...] | list[int] | None = None,
        num_res_blocks: int = 2,
        num_groups:     int = 8,
        dropout:        float = 0.1,
        emb_dim:        int = 256,
        n_styles:       int = N_STYLES,
        audio_channels: list[int] | None = None,
        timing_channels: int = N_TIMING_CHANNELS,
        use_checkpoint: bool = True,
        use_s4:         bool = True,
    ):
        super().__init__()

        channel_mult = tuple(channel_mult or (1, 2, 3, 4))
        self.num_levels = len(channel_mult)
        self.num_res_blocks = num_res_blocks
        self.timing_channels = timing_channels

        if audio_channels is None:
            audio_channels = [base_channels] * self.num_levels
        if len(audio_channels) != self.num_levels:
            raise ValueError(
                f"audio_channels has {len(audio_channels)} entries but the U-Net "
                f"has {self.num_levels} levels; every level takes exactly one "
                f"audio feature map"
            )
        self.audio_channels = list(audio_channels)

        self.time_emb = TimestepEmbedding(d_model=128, dim=emb_dim)
        self.cond_emb = ConditionEmbedding(dim=emb_dim, n_styles=n_styles)

        self.conv_in = nn.Conv1d(z_channels, base_channels, 3, padding=1)

        # ---- encoder ------------------------------------------------------ #
        self.enc_proj = nn.ModuleList()
        self.enc_res  = nn.ModuleList()
        self.enc_s4   = nn.ModuleList()
        self.enc_down = nn.ModuleList()
        self.skip_channels: list[int] = []

        in_ch = base_channels
        for lvl, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            self.enc_proj.append(
                nn.Conv1d(in_ch + self.audio_channels[lvl] + timing_channels, out_ch, 1)
            )
            self.enc_res.append(nn.ModuleList([
                ResBlock(out_ch, out_ch, emb_dim, num_groups, dropout, use_checkpoint)
                for _ in range(num_res_blocks)
            ]))
            self.enc_s4.append(
                S4StyleBlock(out_ch, use_checkpoint=use_checkpoint) if use_s4 else None
            )
            self.skip_channels.append(out_ch)
            self.enc_down.append(Downsample(out_ch) if lvl < self.num_levels - 1 else None)
            in_ch = out_ch

        # ---- middle ------------------------------------------------------- #
        self.mid_block1 = ResBlock(in_ch, in_ch, emb_dim, num_groups,
                                   dropout, use_checkpoint)
        # Self-attention at the coarsest level. The sequence is short there --
        # about 12 frames for a 30 s window -- so this buys whole-window
        # structure for almost nothing.
        self.mid_attn = _CoarseAttention(in_ch)
        self.mid_block2 = ResBlock(in_ch, in_ch, emb_dim, num_groups,
                                   dropout, use_checkpoint)

        # ---- decoder ------------------------------------------------------ #
        self.dec_proj = nn.ModuleList()
        self.dec_res  = nn.ModuleList()
        self.dec_s4   = nn.ModuleList()
        self.dec_up   = nn.ModuleList()

        for lvl in range(self.num_levels):
            enc_lvl = self.num_levels - 1 - lvl
            out_ch  = base_channels * channel_mult[enc_lvl]
            self.dec_proj.append(nn.Conv1d(
                in_ch + self.audio_channels[enc_lvl] + timing_channels
                + self.skip_channels[enc_lvl],
                out_ch, 1,
            ))
            self.dec_res.append(nn.ModuleList([
                ResBlock(out_ch, out_ch, emb_dim, num_groups, dropout, use_checkpoint)
                for _ in range(num_res_blocks)
            ]))
            self.dec_s4.append(
                S4StyleBlock(out_ch, use_checkpoint=use_checkpoint)
                if use_s4 and lvl < self.num_levels - 1 else None
            )
            self.dec_up.append(Upsample(out_ch) if lvl < self.num_levels - 1 else None)
            in_ch = out_ch

        self.norm_out = Normalize(in_ch, num_groups)
        self.conv_out = nn.Conv1d(in_ch, z_channels, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    # ---------------------------------------------------------------- #

    def _timing_pyramid(self, timing: torch.Tensor, lengths: list[int]) -> list[torch.Tensor]:
        """
        Downsample the timing stream to each level.

        Average pooling is correct here only because phase is continuous; a
        pulse train would need max pooling to survive, which is one of the
        reasons the timing stream carries sin/cos rather than pulses.
        """
        out = []
        for length in lengths:
            if timing.shape[-1] == length:
                out.append(timing)
            else:
                out.append(F.adaptive_avg_pool1d(timing, length))
        return out

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        audio_features: list[torch.Tensor],
        timing: torch.Tensor,
        difficulty: torch.Tensor,
        style: torch.Tensor,
        avg_nps: torch.Tensor | None = None,
        peak_nps: torch.Tensor | None = None,
        motif: torch.Tensor | None = None,
        motif_mask: torch.Tensor | None = None,
        drop_mask: torch.Tensor | None = None,
        cond_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            audio_features: one per level, already at that level's resolution.
            timing:         [B, 3, T_latent] beat grid on the latent grid.
            cond_emb:       a precomputed conditioning embedding, which lets the
                            guidance path reuse one null embedding instead of
                            rebuilding it every step.
        """
        if len(audio_features) != self.num_levels:
            raise ValueError(
                f"got {len(audio_features)} audio feature maps for "
                f"{self.num_levels} U-Net levels"
            )

        if cond_emb is None:
            cond_emb = self.cond_emb(
                difficulty, style, avg_nps, peak_nps, motif, motif_mask, drop_mask
            )
        emb = self.time_emb(t) + cond_emb

        h = self.conv_in(z)

        level_lengths = []
        length = h.shape[-1]
        for lvl in range(self.num_levels):
            level_lengths.append(length)
            length = (length + 2 - 4) // 2 + 1
        timings = self._timing_pyramid(timing, level_lengths)

        skips: list[torch.Tensor] = []
        for lvl in range(self.num_levels):
            audio = _match_length(audio_features[lvl], h.shape[-1])
            tm    = _match_length(timings[lvl], h.shape[-1])
            h = self.enc_proj[lvl](torch.cat([h, audio, tm], dim=1))
            for block in self.enc_res[lvl]:
                h = block(h, emb)
            if self.enc_s4[lvl] is not None:
                h = self.enc_s4[lvl](h)
            skips.append(h)
            if self.enc_down[lvl] is not None:
                h = self.enc_down[lvl](h)

        h = self.mid_block2(self.mid_attn(self.mid_block1(h, emb)), emb)

        for lvl in range(self.num_levels):
            enc_lvl = self.num_levels - 1 - lvl
            skip = skips[enc_lvl]
            h = _match_length(h, skip.shape[-1])
            audio = _match_length(audio_features[enc_lvl], h.shape[-1])
            tm    = _match_length(timings[enc_lvl], h.shape[-1])
            h = self.dec_proj[lvl](torch.cat([h, audio, tm, skip], dim=1))
            for block in self.dec_res[lvl]:
                h = block(h, emb)
            if self.dec_s4[lvl] is not None:
                h = self.dec_s4[lvl](h)
            if self.dec_up[lvl] is not None:
                h = self.dec_up[lvl](h)

        h = _match_length(h, z.shape[-1])
        return self.conv_out(F.silu(self.norm_out(h)))

    def count_parameters(self) -> str:
        return f"{sum(p.numel() for p in self.parameters()) / 1e6:.1f}M"


class _CoarseAttention(nn.Module):
    """Self-attention over the whole window at the coarsest latent level."""

    def __init__(self, channels: int):
        super().__init__()
        heads = max(1, min(8, channels // 64))
        while channels % heads != 0:
            heads -= 1
        self.norm = Normalize(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x).transpose(1, 2)
        h, _ = self.attn(h, h, h, need_weights=False)
        return x + h.transpose(1, 2)
