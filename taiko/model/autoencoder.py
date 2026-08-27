"""
taiko/model/autoencoder.py

KL-regularised autoencoder over chart tensors.

    chart [B, 6, T]  ->  z [B, 16, T / compression]  ->  chart [B, 6, T]

The diffusion model works in this latent space, so whatever the autoencoder
cannot reconstruct is a hard ceiling on the whole system. That is why Gate A in
the development plan is a reconstruction gate and not a loss threshold: a val
loss of 0.01 tells you nothing about whether the onsets came back on the right
frames, and onsets on the right frames is the entire product.

Latent scale
------------
Diffusion assumes its input has roughly unit variance -- the noise schedule's
signal-to-noise ratio is calibrated against that. A VAE trained with a small KL
weight produces whatever scale it likes, often several times unit. Stable
Diffusion handles this with a fixed scalar measured after training; the same
applies here, and `calibrate_scale()` measures it. It was previously left at
1.0 and never measured, which silently mis-tunes every timestep of the
diffusion schedule.

Class imbalance
---------------
A taiko chart is about 99.5% zeros. Unweighted BCE is minimised by predicting
silence everywhere, which scores well and reconstructs nothing. The loss below
weights positives heavily and reports per-channel numbers so a channel that has
collapsed to silence is visible in the logs rather than hidden inside an average
that looks fine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from taiko.data.tensor_repr import CHART_CHANNEL_NAMES, N_CHART_CHANNELS


def Normalize(channels: int, num_groups: int = 8) -> nn.GroupNorm:
    while channels % num_groups != 0:
        num_groups //= 2
    return nn.GroupNorm(num_groups, channels, eps=1e-6, affine=True)


# --------------------------------------------------------------------------- #
# Blocks
# --------------------------------------------------------------------------- #

class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 num_groups: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm1 = Normalize(in_channels, num_groups)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.norm2 = Normalize(out_channels, num_groups)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbour upsample then convolve, which avoids the checkerboard
    artefacts a transposed convolution produces."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


# --------------------------------------------------------------------------- #
# Encoder / decoder
# --------------------------------------------------------------------------- #

class Encoder(nn.Module):
    def __init__(self, x_channels: int, middle_channels: int, z_channels: int,
                 channel_mult: list[int], num_res_blocks: int,
                 num_groups: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_resolutions = len(channel_mult)

        self.conv_in = nn.Conv1d(x_channels, middle_channels, 3, padding=1)

        self.down = nn.ModuleList()
        block_in = middle_channels
        for i, mult in enumerate(channel_mult):
            block_out = middle_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResnetBlock(block_in, block_out, num_groups, dropout))
                block_in = block_out
            level = nn.Module()
            level.block = blocks
            if i != self.num_resolutions - 1:
                level.downsample = Downsample(block_in)
            self.down.append(level)

        self.mid_block1 = ResnetBlock(block_in, block_in, num_groups)
        self.mid_block2 = ResnetBlock(block_in, block_in, num_groups)
        self.norm_out = Normalize(block_in, num_groups)
        self.conv_out = nn.Conv1d(block_in, z_channels * 2, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for i in range(self.num_resolutions):
            for block in self.down[i].block:
                h = block(h)
            if i != self.num_resolutions - 1:
                h = self.down[i].downsample(h)
        h = self.mid_block2(self.mid_block1(h))
        return self.conv_out(F.silu(self.norm_out(h)))


class Decoder(nn.Module):
    def __init__(self, x_channels: int, middle_channels: int, z_channels: int,
                 channel_mult: list[int], num_res_blocks: int,
                 num_groups: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_resolutions = len(channel_mult)

        block_in = middle_channels * channel_mult[-1]
        self.conv_in = nn.Conv1d(z_channels, block_in, 3, padding=1)
        self.mid_block1 = ResnetBlock(block_in, block_in, num_groups)
        self.mid_block2 = ResnetBlock(block_in, block_in, num_groups)

        self.up = nn.ModuleList()
        for i in reversed(range(self.num_resolutions)):
            block_out = middle_channels * channel_mult[i]
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                blocks.append(ResnetBlock(block_in, block_out, num_groups, dropout))
                block_in = block_out
            level = nn.Module()
            level.block = blocks
            if i != 0:
                level.upsample = Upsample(block_in)
            self.up.insert(0, level)

        self.norm_out = Normalize(block_in, num_groups)
        self.conv_out = nn.Conv1d(block_in, x_channels, 3, padding=1)

        # Start biased towards silence. Charts are ~99.5% zeros, so a neutral
        # init spends the first few thousand steps just learning that.
        nn.init.zeros_(self.conv_out.weight)
        nn.init.constant_(self.conv_out.bias, -4.0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.mid_block2(self.mid_block1(self.conv_in(z)))
        for i in reversed(range(self.num_resolutions)):
            for block in self.up[i].block:
                h = block(h)
            if i != 0:
                h = self.up[i].upsample(h)
        return self.conv_out(F.silu(self.norm_out(h)))


# --------------------------------------------------------------------------- #
# Posterior
# --------------------------------------------------------------------------- #

class DiagonalGaussianDistribution:
    """Diagonal Gaussian parameterised by concatenated [mean, logvar]."""

    def __init__(self, parameters: torch.Tensor, scale: float = 1.0):
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -10.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        self.scale = scale

    def sample(self) -> torch.Tensor:
        return (self.mean + self.std * torch.randn_like(self.mean)) * self.scale

    def mode(self) -> torch.Tensor:
        return self.mean * self.scale

    def kl(self) -> torch.Tensor:
        return 0.5 * torch.mean(self.mean.pow(2) + self.var - 1.0 - self.logvar)


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #

class ChartReconstructionLoss(nn.Module):
    """
    Masked, class-balanced BCE over the six chart channels.

    `pos_weight` does the heavy lifting. At roughly 200:1 negative-to-positive,
    unweighted BCE is minimised by predicting silence; the weights make a missed
    note cost far more than a false one, which is the right trade for a first
    stage whose failures are unrecoverable downstream.

    Sustained channels are weighted lower per frame than onsets because they
    span hundreds of frames -- otherwise one long drumroll outweighs every hit
    note in the window.
    """

    POS_WEIGHT = {
        "don":     40.0,
        "kat":     40.0,
        "big_don": 80.0,     # rarer, and visually distinctive when wrong
        "big_kat": 80.0,
        "roll":     8.0,     # sustained: many frames per event
        "denden":   8.0,
    }

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing
        weights = torch.tensor(
            [self.POS_WEIGHT[name] for name in CHART_CHANNEL_NAMES],
            dtype=torch.float32,
        ).view(1, N_CHART_CHANNELS, 1)
        self.register_buffer("pos_weight", weights)

    def forward(
        self,
        target: torch.Tensor,
        logits: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        if self.label_smoothing > 0:
            target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Applying pos_weight by hand rather than through the pos_weight kwarg
        # keeps it per-channel while staying inside the numerically stable
        # logits form.
        per_element = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        weight = 1.0 + (self.pos_weight - 1.0) * target
        per_element = per_element * weight

        mask = valid_mask.unsqueeze(1)
        denom = mask.sum().clamp(min=1.0) * N_CHART_CHANNELS
        loss = (per_element * mask).sum() / denom

        log: dict[str, float] = {}
        with torch.no_grad():
            for i, name in enumerate(CHART_CHANNEL_NAMES):
                ch_mask = valid_mask
                raw = F.binary_cross_entropy_with_logits(
                    logits[:, i], target[:, i], reduction="none"
                )
                log[f"loss_{name}"] = float(
                    (raw * ch_mask).sum() / ch_mask.sum().clamp(min=1.0)
                )
                # Recall per channel is what actually tells you whether a
                # channel has collapsed to silence.
                pred_pos = (logits[:, i] > 0) & (ch_mask > 0)
                true_pos = (target[:, i] > 0.5) & (ch_mask > 0)
                n_true = true_pos.sum().clamp(min=1)
                log[f"recall_{name}"] = float((pred_pos & true_pos).sum() / n_true)

        return loss, log


# --------------------------------------------------------------------------- #
# Autoencoder
# --------------------------------------------------------------------------- #

@dataclass
class AutoencoderConfig:
    x_channels:      int = N_CHART_CHANNELS
    middle_channels: int = 64
    z_channels:      int = 16
    channel_mult:    list[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    num_res_blocks:  int = 2
    num_groups:      int = 8
    dropout:         float = 0.0
    kl_weight:       float = 1e-6
    scale:           float = 1.0

    @property
    def compression(self) -> int:
        return 2 ** (len(self.channel_mult) - 1)


class ChartAutoencoder(nn.Module):
    """
    Compresses a chart by `compression` in time.

    Compression is a real trade. At 16x with 20 ms frames one latent frame spans
    320 ms -- longer than a beat at 190 BPM -- so a single latent vector has to
    carry every note in that span and their exact frames. If Gate A fails at
    16x, drop `channel_mult` by one entry for 8x rather than training the
    diffusion model on a lossy first stage.
    """

    def __init__(self, config: AutoencoderConfig | None = None):
        super().__init__()
        self.config = config or AutoencoderConfig()
        cfg = self.config

        self.encoder = Encoder(
            cfg.x_channels, cfg.middle_channels, cfg.z_channels,
            cfg.channel_mult, cfg.num_res_blocks, cfg.num_groups, cfg.dropout,
        )
        self.decoder = Decoder(
            cfg.x_channels, cfg.middle_channels, cfg.z_channels,
            cfg.channel_mult, cfg.num_res_blocks, cfg.num_groups, cfg.dropout,
        )
        self.loss = ChartReconstructionLoss()

        # Registered as a buffer so calibration travels with the checkpoint. A
        # scale that lives only in a config file gets lost, and a diffusion
        # model trained against the wrong latent scale fails in a way that
        # looks like a modelling problem.
        self.register_buffer("latent_scale", torch.tensor(float(cfg.scale)))

    @property
    def compression(self) -> int:
        return self.config.compression

    @property
    def scale(self) -> float:
        return float(self.latent_scale)

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        return DiagonalGaussianDistribution(self.encoder(x), scale=self.scale)

    def decode(self, z: torch.Tensor, target_length: int | None = None) -> torch.Tensor:
        """
        Latent -> logits. Callers apply sigmoid when they want probabilities.

        `target_length` trims or pads the output to an exact frame count.
        Downsampling discards the remainder when the input is not a multiple of
        the compression ratio, and upsampling cannot invent it back, so a 1500
        frame window would return 1488 frames. Training windows should be
        multiples of `compression` (see `check_window`); this argument exists
        for inference, where song length is not ours to choose.
        """
        out = self.decoder(z / self.scale)
        if target_length is not None and out.shape[-1] != target_length:
            if out.shape[-1] > target_length:
                out = out[..., :target_length]
            else:
                out = F.pad(out, (0, target_length - out.shape[-1]), value=-10.0)
        return out

    def check_window(self, frames: int) -> None:
        """Reject a window size that the encoder/decoder pair cannot round-trip."""
        if frames % self.compression != 0:
            raise ValueError(
                f"window of {frames} frames is not a multiple of the "
                f"{self.compression}x compression ratio; {frames // self.compression}"
                f" latent frames decode back to "
                f"{frames // self.compression * self.compression} frames, losing "
                f"{frames % self.compression}. Use a multiple of "
                f"{self.compression}."
            )

    def forward(self, x: torch.Tensor, sample_posterior: bool = True):
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        return self.decode(z, target_length=x.shape[-1]), posterior

    def training_loss(self, x: torch.Tensor, valid_mask: torch.Tensor):
        logits, posterior = self(x, sample_posterior=True)
        recon, log = self.loss(x, logits, valid_mask)
        kl = posterior.kl()
        total = recon + kl * self.config.kl_weight
        log.update(
            recon_loss=recon.detach().item(),
            kl_loss=kl.detach().item(),
            total_loss=total.detach().item(),
        )
        return total, log

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic reconstruction as probabilities in [0, 1]."""
        logits, _ = self(x, sample_posterior=False)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def calibrate_scale(self, batches, device=None, max_batches: int = 200) -> float:
        """
        Measure the latent standard deviation and set `latent_scale` to 1/std,
        so the diffusion model receives roughly unit-variance input.

        Uses the posterior mode rather than a sample: the diffusion model
        encodes with `.mode()`, so that is the distribution whose scale matters.

        `batches` yields either tensors or dicts containing a "chart" key.
        """
        was_training = self.training
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        # Two-pass-free variance: accumulate sums so this stays O(1) in memory
        # over an arbitrarily large calibration set.
        total = torch.zeros((), dtype=torch.float64, device=device)
        total_sq = torch.zeros((), dtype=torch.float64, device=device)
        count = 0

        previous = float(self.latent_scale)
        self.latent_scale.fill_(1.0)          # measure the raw latent

        for i, batch in enumerate(batches):
            if i >= max_batches:
                break
            x = batch["chart"] if isinstance(batch, dict) else batch
            z = self.encode(x.to(device)).mode().double()
            total += z.sum()
            total_sq += (z * z).sum()
            count += z.numel()

        if count == 0:
            self.latent_scale.fill_(previous)
            if was_training:
                self.train()
            raise ValueError("calibrate_scale got no data")

        mean = total / count
        var = (total_sq / count) - mean * mean
        std = float(torch.sqrt(var.clamp(min=1e-12)))

        scale = 1.0 / max(std, 1e-6)
        self.latent_scale.fill_(scale)
        self.config.scale = scale

        if was_training:
            self.train()
        return scale

    def count_parameters(self) -> dict[str, str]:
        enc = sum(p.numel() for p in self.encoder.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
        return {
            "encoder": f"{enc / 1e6:.2f}M",
            "decoder": f"{dec / 1e6:.2f}M",
            "total":   f"{(enc + dec) / 1e6:.2f}M",
        }


# Retained so existing imports keep working.
BeatmapAutoencoder = ChartAutoencoder
TaikoReconstructLoss = ChartReconstructionLoss
