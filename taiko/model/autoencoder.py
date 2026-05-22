"""
taiko/model/autoencoder.py

Beatmap Autoencoder (KL-regularized VAE) — closely follows Mug-Diffusion's AutoencoderKL.

Architecture:
    Encoder: Conv1d in → ResNet blocks → downsample × N → Conv1d out (z_channels * 2)
    Decoder: Conv1d in → ResNet blocks → upsample × N → Conv1d out (x_channels)

Input:  beatmap tensor [B, 7, T]   (7 channels: don/kat/big_don/big_kat/roll/denden/beat)
Latent: [B, z_channels, T // 2^(N-1)]
Output: [B, 7, T]

Key differences from Mug-Diffusion (mania 16ch → taiko 7ch):
    x_channels: 16 → 7
    middle_channels: 64 → 32  (smaller for 4GB VRAM)
    z_channels: 32 → 16
    channel_mult: [1,1,2,2,4,4] → [1,1,2,2,4] (one fewer level)
    num_groups: 32 → 8 (fewer channels need fewer groups)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def Normalize(channels: int, num_groups: int = 8) -> nn.GroupNorm:
    # Ensure num_groups divides channels
    while channels % num_groups != 0:
        num_groups //= 2
    return nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6, affine=True)


class ResnetBlock(nn.Module):
    """1D ResNet block with GroupNorm + SiLU. No time embedding needed for autoencoder."""

    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm1 = Normalize(in_channels, num_groups)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = Normalize(out_channels, num_groups)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        self.skip = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class Downsample(nn.Module):
    """Strided Conv1d downsampling by 2x."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbor upsample + Conv1d (avoids checkerboard artifacts)."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    Encodes beatmap tensor [B, x_channels, T] → latent params [B, z_channels*2, T//scale].
    Output has 2*z_channels for mean + logvar (reparameterization trick).
    """

    def __init__(
        self,
        x_channels: int,
        middle_channels: int,
        z_channels: int,
        channel_mult: list[int],
        num_res_blocks: int,
        num_groups: int = 8,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.num_resolutions = len(channel_mult)
        self.num_res_blocks   = num_res_blocks

        self.conv_in = nn.Conv1d(x_channels, middle_channels, kernel_size=3, padding=1)

        # Downsampling path
        self.down = nn.ModuleList()
        block_in = middle_channels
        for i_level in range(self.num_resolutions):
            block_out = middle_channels * channel_mult[i_level]
            blocks    = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResnetBlock(block_in, block_out, num_groups, dropout))
                block_in = block_out
            level = nn.Module()
            level.block = blocks
            if i_level != self.num_resolutions - 1:
                level.downsample = Downsample(block_in)
            self.down.append(level)

        # Middle
        self.mid_block1 = ResnetBlock(block_in, block_in, num_groups)
        self.mid_block2 = ResnetBlock(block_in, block_in, num_groups)

        # Output: project to z_channels * 2 (mean + logvar)
        self.norm_out = Normalize(block_in, num_groups)
        self.conv_out = nn.Conv1d(block_in, z_channels * 2, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for block in self.down[i_level].block:
                h = block(h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)
        h = self.mid_block1(h)
        h = self.mid_block2(h)
        h = self.conv_out(F.silu(self.norm_out(h)))
        return h


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """
    Decodes latent [B, z_channels, T//scale] → beatmap tensor [B, x_channels, T].
    """

    def __init__(
        self,
        x_channels: int,
        middle_channels: int,
        z_channels: int,
        channel_mult: list[int],
        num_res_blocks: int,
        num_groups: int = 8,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.num_resolutions = len(channel_mult)
        self.num_res_blocks   = num_res_blocks

        block_in = middle_channels * channel_mult[-1]
        self.conv_in = nn.Conv1d(z_channels, block_in, kernel_size=3, padding=1)

        # Middle
        self.mid_block1 = ResnetBlock(block_in, block_in, num_groups)
        self.mid_block2 = ResnetBlock(block_in, block_in, num_groups)

        # Upsampling path
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block_out = middle_channels * channel_mult[i_level]
            blocks    = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                blocks.append(ResnetBlock(block_in, block_out, num_groups, dropout))
                block_in = block_out
            level = nn.Module()
            level.block = blocks
            if i_level != 0:
                level.upsample = Upsample(block_in)
            self.up.insert(0, level)

        # Output
        self.norm_out = Normalize(block_in, num_groups)
        self.conv_out = nn.Conv1d(block_in, x_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.mid_block1(h)
        h = self.mid_block2(h)
        for i_level in reversed(range(self.num_resolutions)):
            for block in self.up[i_level].block:
                h = block(h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        h = self.conv_out(F.silu(self.norm_out(h)))
        return h


# ---------------------------------------------------------------------------
# Diagonal Gaussian Distribution (reparameterization)
# ---------------------------------------------------------------------------

class DiagonalGaussianDistribution:
    """
    Parameterizes a diagonal Gaussian from encoder output.
    encoder outputs [mean, logvar] concatenated along channel dim.
    """

    def __init__(self, parameters: torch.Tensor, scale: float = 1.0):
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -10.0, 20.0)
        self.std    = torch.exp(0.5 * self.logvar)
        self.var    = torch.exp(self.logvar)
        self.scale  = scale

    def sample(self) -> torch.Tensor:
        eps = torch.randn_like(self.mean)
        return (self.mean + self.std * eps) * self.scale

    def mode(self) -> torch.Tensor:
        return self.mean * self.scale

    def kl(self) -> torch.Tensor:
        """KL divergence to standard normal N(0,1)."""
        return 0.5 * torch.mean(
            self.mean.pow(2) + self.var - 1.0 - self.logvar
        )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class TaikoReconstructLoss(nn.Module):
    """
    Reconstruction loss for taiko beatmap tensor.

    Channel weights (tuned for taiko sparsity):
        don/kat onset channels:   high weight — getting onsets right is critical
        big_don/big_kat:          high weight — rare but important
        roll/denden duration:     medium weight — duration matters less than onset
        beat channel:             low weight — mostly for reference, not generated
    """

    CHANNEL_WEIGHTS = [
        3.0,   # don
        3.0,   # kat
        4.0,   # big_don  (rare, upweight)
        4.0,   # big_kat
        2.0,   # roll
        2.0,   # denden
        0.5,   # beat     (reference channel, low weight)
    ]

    def __init__(self, label_smoothing: float = 0.001):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.register_buffer(
            "weights",
            torch.tensor(self.CHANNEL_WEIGHTS, dtype=torch.float32).view(1, 7, 1)
        )

    def forward(
        self,
        target: torch.Tensor,        # [B, 7, T]
        prediction: torch.Tensor,    # [B, 7, T]
        valid_mask: torch.Tensor,    # [B, T] 1=valid 0=padding
    ) -> tuple[torch.Tensor, dict]:

        # Label smoothing
        if self.label_smoothing > 0:
            target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # BCE loss per channel
        loss = F.binary_cross_entropy_with_logits(
            prediction,
            target,
            reduction="none",
        )  # [B, 7, T]

        # Apply channel weights
        loss = loss * self.weights

        # Apply valid mask (ignore padding)
        mask = valid_mask.unsqueeze(1)   # [B, 1, T]
        loss = (loss * mask).sum() / (mask.sum() * 7 + 1e-8)

        # Per-channel losses for logging
        with torch.no_grad():
            ch_names = ["don", "kat", "big_don", "big_kat", "roll", "denden", "beat"]
            log_dict = {}
            for i, name in enumerate(ch_names):
                ch_loss = (F.binary_cross_entropy_with_logits(
                    prediction[:, i], target[:, i], reduction="none"
                ) * valid_mask).sum() / (valid_mask.sum() + 1e-8)
                log_dict[f"loss_{name}"] = ch_loss.item()

        return loss, log_dict


# ---------------------------------------------------------------------------
# Full Autoencoder
# ---------------------------------------------------------------------------

@dataclass
class AutoencoderConfig:
    x_channels:     int        = 7       # taiko channels
    middle_channels: int       = 32      # base channel width (small for 4GB VRAM)
    z_channels:     int        = 16      # latent channels
    channel_mult:   list       = None    # multipliers per level
    num_res_blocks: int        = 2
    num_groups:     int        = 8
    dropout:        float      = 0.0
    kl_weight:      float      = 1e-6    # very small — prioritize reconstruction
    scale:          float      = 1.0

    def __post_init__(self):
        if self.channel_mult is None:
            self.channel_mult = [1, 1, 2, 2, 4]


class BeatmapAutoencoder(nn.Module):
    """
    Full KL-VAE autoencoder for taiko beatmap tensors.
    Follows Mug-Diffusion's AutoencoderKL closely.

    Compression ratio: 2^(len(channel_mult)-1) = 2^4 = 16x temporal downsampling
    So a 9000-frame tensor → 562-frame latent.
    """

    def __init__(self, config: AutoencoderConfig = None):
        super().__init__()
        self.config = config or AutoencoderConfig()
        cfg = self.config

        self.encoder = Encoder(
            x_channels=cfg.x_channels,
            middle_channels=cfg.middle_channels,
            z_channels=cfg.z_channels,
            channel_mult=cfg.channel_mult,
            num_res_blocks=cfg.num_res_blocks,
            num_groups=cfg.num_groups,
            dropout=cfg.dropout,
        )
        self.decoder = Decoder(
            x_channels=cfg.x_channels,
            middle_channels=cfg.middle_channels,
            z_channels=cfg.z_channels,
            channel_mult=cfg.channel_mult,
            num_res_blocks=cfg.num_res_blocks,
            num_groups=cfg.num_groups,
            dropout=cfg.dropout,
        )
        self.loss = TaikoReconstructLoss()
        self.scale = cfg.scale
        self.kl_weight = cfg.kl_weight

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        h = self.encoder(x)
        return DiagonalGaussianDistribution(h, scale=self.scale)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z / self.scale)

    def forward(
        self,
        x: torch.Tensor,
        sample_posterior: bool = True,
    ) -> tuple[torch.Tensor, DiagonalGaussianDistribution]:
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        return self.decode(z), posterior

    def training_loss(
        self,
        x: torch.Tensor,          # [B, 7, T]
        valid_mask: torch.Tensor,  # [B, T]
    ) -> tuple[torch.Tensor, dict]:
        recon, posterior = self(x, sample_posterior=True)
        recon_loss, log_dict = self.loss(x, recon, valid_mask)
        kl_loss = posterior.kl()
        total   = recon_loss + kl_loss * self.kl_weight
        log_dict["recon_loss"] = recon_loss.item()
        log_dict["kl_loss"]    = kl_loss.item()
        log_dict["total_loss"] = total.item()
        return total, log_dict

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic reconstruction (uses mode, not sample)."""
        recon, _ = self(x, sample_posterior=False)
        return torch.sigmoid(recon)  # [0,1] for visualization

    def count_parameters(self) -> dict:
        enc = sum(p.numel() for p in self.encoder.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
        return {
            "encoder": f"{enc/1e6:.2f}M",
            "decoder": f"{dec/1e6:.2f}M",
            "total":   f"{(enc+dec)/1e6:.2f}M",
        }

    @property
    def compression_ratio(self) -> int:
        return 2 ** (len(self.config.channel_mult) - 1)
