"""
taiko/model/diffusion.py

Latent diffusion over chart latents, conditioned on audio, tempo and style.

    chart --(frozen VAE)--> z
    mel   --(audio encoder)--> per-level features
    z_t + t + audio + timing + condition --(U-Net)--> v
    sample: DDIM with classifier-free guidance --> z --(VAE)--> chart

DataParallel
------------
`forward()` computes the loss. That is not a stylistic choice: the previous
training loop wrapped the model in `nn.DataParallel` and then called
`unwrap(model).training_loss(...)`, reaching *through* the wrapper. DataParallel
only scatters work on `forward()`, so it never ran -- every batch went to
cuda:0 and the second GPU sat idle for every run. Putting the loss in
`forward()` means calling the wrapped module does the right thing, and calling
the unwrapped one still works on a single device.

Guidance
--------
The unconditional branch uses the U-Net's learned null embedding, computed
once per generation rather than per step. It is a genuinely separate point in
conditioning space from any real style -- see `taiko/data/conditioning.py`.
"""

from __future__ import annotations

import copy
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from taiko.data.conditioning import CFG_DROPOUT, STYLE_NULL
from taiko.data.motif import MOTIF_DIM
from taiko.model.audio_encoder import MelEncoder1D
from taiko.model.autoencoder import AutoencoderConfig, ChartAutoencoder
from taiko.model.model_config import DiffusionProfile, get_profile
from taiko.model.noise_scheduler import NoiseScheduler
from taiko.model.unet import TaikoDiffusionUNet


def _frozen_train(self, mode: bool = True):
    """Keep a module in eval mode however the parent is toggled."""
    return self


class EMA:
    """
    Exponential moving average of the trainable weights.

    Sampling from an EMA of the weights rather than the weights themselves is
    close to mandatory for diffusion sample quality, and it was missing
    entirely. The averaged weights sit nearer the centre of the basin the
    optimiser is bouncing around in, which shows up directly as cleaner
    samples -- not as a better validation loss, which is why it is easy to skip
    without noticing.

    `warmup` ramps the decay in from 0 so the average is not anchored to the
    random initialisation for its first few thousand steps.
    """

    def __init__(self, parameters: Iterable[nn.Parameter], decay: float = 0.9995,
                 warmup: int = 1000):
        self.decay = decay
        self.warmup = warmup
        self.step = 0
        self.shadow = [p.detach().clone().float() for p in parameters]
        self._backup: list[torch.Tensor] | None = None

    def current_decay(self) -> float:
        if self.warmup <= 0:
            return self.decay
        # Standard bias-corrected warmup: early steps track the live weights
        # closely, easing towards `decay` as the average becomes meaningful.
        return min(self.decay, (1 + self.step) / (10 + self.step))

    @torch.no_grad()
    def update(self, parameters: Iterable[nn.Parameter]) -> None:
        self.step += 1
        d = self.current_decay()
        for shadow, param in zip(self.shadow, parameters):
            shadow.mul_(d).add_(param.detach().float(), alpha=1.0 - d)

    @torch.no_grad()
    def copy_to(self, parameters: Iterable[nn.Parameter]) -> None:
        for shadow, param in zip(self.shadow, parameters):
            param.data.copy_(shadow.to(param.dtype))

    @torch.no_grad()
    def store(self, parameters: Iterable[nn.Parameter]) -> None:
        self._backup = [p.detach().clone() for p in parameters]

    @torch.no_grad()
    def restore(self, parameters: Iterable[nn.Parameter]) -> None:
        if self._backup is None:
            raise RuntimeError("restore() without a matching store()")
        for backup, param in zip(self._backup, parameters):
            param.data.copy_(backup)
        self._backup = None

    def state_dict(self) -> dict:
        return {"decay": self.decay, "warmup": self.warmup,
                "step": self.step, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.warmup = state.get("warmup", 0)
        self.step = state["step"]
        self.shadow = [t.clone().float() for t in state["shadow"]]


class TaikoDiffusion(nn.Module):
    def __init__(
        self,
        autoencoder_ckpt: str | None = None,
        profile: str | DiffusionProfile = "p1",
        autoencoder: ChartAutoencoder | None = None,
        timesteps: int = 1000,
        beta_schedule: str = "cosine",
        prediction_type: str = "v",
        cfg_scale: float = 4.0,
        cfg_dropout: float | None = None,
        z_channels: int = 16,
        n_mels: int = 128,
        verbose: bool = True,
    ):
        super().__init__()

        if isinstance(profile, str):
            profile = get_profile(profile)
        self.profile = profile
        self.cfg_scale = cfg_scale
        self.z_channels = z_channels
        self.cfg_dropout = CFG_DROPOUT if cfg_dropout is None else cfg_dropout

        # ---- frozen first stage ------------------------------------------ #
        if autoencoder is not None:
            self.first_stage = autoencoder
        else:
            if autoencoder_ckpt is None:
                raise ValueError("need autoencoder_ckpt or an autoencoder instance")
            ckpt = torch.load(autoencoder_ckpt, map_location="cpu", weights_only=False)
            ae_config = ckpt.get("config") or AutoencoderConfig(z_channels=z_channels)
            if isinstance(ae_config, dict):
                ae_config = AutoencoderConfig(**ae_config)
            self.first_stage = ChartAutoencoder(ae_config)
            self.first_stage.load_state_dict(ckpt["model"])
            if verbose:
                print(
                    f"Autoencoder: {autoencoder_ckpt} "
                    f"(step {ckpt.get('step', '?')}, "
                    f"{self.first_stage.compression}x, "
                    f"latent scale {self.first_stage.scale:.4f})"
                )
                if abs(self.first_stage.scale - 1.0) < 1e-6:
                    print(
                        "  WARNING: latent scale is exactly 1.0. If the "
                        "autoencoder was never calibrated, the diffusion "
                        "schedule is mis-tuned at every timestep. Run "
                        "scripts/calibrate_latent_scale.py."
                    )

        self.first_stage.eval()
        self.first_stage.train = _frozen_train.__get__(self.first_stage)
        for p in self.first_stage.parameters():
            p.requires_grad = False

        self.compression = self.first_stage.compression

        # ---- audio encoder ----------------------------------------------- #
        self.wave_model = MelEncoder1D(
            n_mels=n_mels,
            base_channels=profile.audio_base_channels,
            channel_mult=profile.audio_channel_mult,
            compression=self.compression,
            n_levels=len(profile.unet_channel_mult),
        )

        # ---- U-Net -------------------------------------------------------- #
        self.unet_model = TaikoDiffusionUNet(
            z_channels=z_channels,
            base_channels=profile.unet_base_channels,
            channel_mult=profile.unet_channel_mult,
            num_res_blocks=profile.unet_num_res_blocks,
            audio_channels=self.wave_model.out_channels,
            use_checkpoint=profile.use_checkpoint,
            use_s4=profile.use_s4,
        )

        self.scheduler = NoiseScheduler(
            timesteps=timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
        )

        if verbose:
            print(
                f"Profile {profile.name}: U-Net {self.unet_model.count_parameters()}, "
                f"audio {self.wave_model.count_parameters()}, "
                f"compression {self.compression}x, {prediction_type}-prediction"
            )

    # ------------------------------------------------------------------ #

    def trainable_parameters(self) -> list[nn.Parameter]:
        return list(self.unet_model.parameters()) + list(self.wave_model.parameters())

    @torch.no_grad()
    def encode(self, chart: torch.Tensor) -> torch.Tensor:
        return self.first_stage.encode(chart).mode()

    @torch.no_grad()
    def decode(self, z: torch.Tensor, target_length: int | None = None) -> torch.Tensor:
        return torch.sigmoid(self.first_stage.decode(z, target_length=target_length))

    def downsample_timing(self, timing: torch.Tensor, latent_frames: int) -> torch.Tensor:
        """Timing stream from chart resolution to latent resolution."""
        if timing.shape[-1] == latent_frames:
            return timing
        return F.adaptive_avg_pool1d(timing, latent_frames)

    # ------------------------------------------------------------------ #

    def forward(
        self,
        chart: torch.Tensor,
        mel: torch.Tensor,
        timing: torch.Tensor,
        difficulty: torch.Tensor,
        style: torch.Tensor,
        valid_mask: torch.Tensor,
        avg_nps: torch.Tensor | None = None,
        peak_nps: torch.Tensor | None = None,
        motif: torch.Tensor | None = None,
        motif_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Training loss. Returns (loss, metrics) where metrics is a stacked tensor
        rather than a dict of floats, so DataParallel can gather it across GPUs
        -- a dict of Python floats silently keeps only the first replica's.

        metrics: [loss, mae, mean_t, mask_ratio], each already batch-averaged.
        """
        B = chart.shape[0]
        device = chart.device

        z = self.encode(chart)
        t = torch.randint(0, self.scheduler.timesteps, (B,), device=device).long()
        noise = torch.randn_like(z)
        z_t = self.scheduler.add_noise(z, t, noise)
        target = self.scheduler.target_for(z, noise, t)

        audio_features = self.wave_model(mel)
        timing_latent = self.downsample_timing(timing, z.shape[-1])

        # Classifier-free guidance dropout lives here, not inside the embedding
        # layer, so the training path and the sampling path share one mechanism.
        drop_mask = self._cfg_drop_mask(B, device)

        prediction = self.unet_model(
            z_t, t, audio_features, timing_latent,
            difficulty=difficulty, style=style,
            avg_nps=avg_nps, peak_nps=peak_nps,
            motif=motif, motif_mask=motif_mask,
            drop_mask=drop_mask,
        )

        # The latent mask marks frames whose chart frames were all padding.
        mask = self._latent_mask(valid_mask, z.shape[-1]).unsqueeze(1)

        per_element = F.mse_loss(prediction, target, reduction="none")
        denom = mask.sum().clamp(min=1.0) * prediction.shape[1]
        loss = (per_element * mask).sum() / denom

        with torch.no_grad():
            mae = ((prediction - target).abs() * mask).sum() / denom

        metrics = torch.stack([
            loss.detach(), mae, t.float().mean(), mask.mean(),
        ])
        return loss, metrics

    def _cfg_drop_mask(self, batch: int, device) -> torch.Tensor:
        """
        Choose which samples see the null condition.

        Stratified rather than independent Bernoulli. The per-GPU batch here is
        2, so independent draws at p=0.15 leave most steps with no
        unconditional sample at all and occasionally two -- the unconditional
        branch then trains on a trickle of high-variance gradient, and guidance
        quality follows. This draws exactly floor(B*p) samples plus one more
        with probability frac(B*p), which keeps the marginal rate exact while
        removing most of the variance.
        """
        if self.cfg_dropout <= 0:
            return torch.zeros(batch, dtype=torch.bool, device=device)
        if self.cfg_dropout >= 1:
            return torch.ones(batch, dtype=torch.bool, device=device)

        expected = batch * self.cfg_dropout
        k = int(expected)
        if torch.rand((), device=device) < (expected - k):
            k += 1
        k = min(k, batch)

        mask = torch.zeros(batch, dtype=torch.bool, device=device)
        if k > 0:
            mask[torch.randperm(batch, device=device)[:k]] = True
        return mask

    def _latent_mask(self, valid_mask: torch.Tensor, latent_frames: int) -> torch.Tensor:
        """
        Chart-resolution validity -> latent resolution.

        Max pooling, not striding: a latent frame that overlaps any real chart
        frame is real. Striding by the compression ratio (`valid_mask[:, ::16]`,
        as before) samples one frame in sixteen and both hardcodes the ratio and
        gets the boundary wrong.
        """
        if valid_mask.shape[-1] == latent_frames:
            return valid_mask
        return F.adaptive_max_pool1d(valid_mask.unsqueeze(1), latent_frames).squeeze(1)

    # Retained so single-GPU callers and tests can be explicit.
    def training_loss(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate_latent(
        self,
        mel: torch.Tensor,
        timing: torch.Tensor,
        difficulty: float | torch.Tensor,
        style: int | torch.Tensor,
        latent_frames: int | None = None,
        avg_nps: float | torch.Tensor | None = None,
        peak_nps: float | torch.Tensor | None = None,
        motif: torch.Tensor | None = None,
        motif_mask: torch.Tensor | None = None,
        ddim_steps: int = 50,
        cfg_scale: float | None = None,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
        progress: bool = False,
    ) -> torch.Tensor:
        """
        Sample one latent for a single window.

        For a whole song use `taiko.model.sampling.generate_song`, which tiles
        overlapping windows of the size the model trained on. Running a
        three-minute song through here in one pass is six times longer than any
        window the model has ever seen.
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        cfg_scale = self.cfg_scale if cfg_scale is None else cfg_scale

        B = mel.shape[0]
        if latent_frames is None:
            latent_frames = self.wave_model.latent_frames(mel.shape[-1])

        def as_batch(v, fill, dt=torch.float32):
            if v is None:
                return torch.full((B,), fill, device=device, dtype=dt)
            if isinstance(v, torch.Tensor):
                return v.reshape(-1).to(device=device, dtype=dt).expand(B)
            return torch.full((B,), v, device=device, dtype=dt)

        diff_t  = as_batch(difficulty, 0.5)
        style_t = as_batch(style, STYLE_NULL, torch.long)
        anps_t  = as_batch(avg_nps, 0.0)
        pnps_t  = as_batch(peak_nps, 0.0)

        if motif is None:
            motif_t = torch.zeros(B, MOTIF_DIM, device=device, dtype=dtype)
            mask_t  = torch.zeros(B, MOTIF_DIM, device=device, dtype=dtype)
        else:
            motif_t = torch.as_tensor(motif, device=device, dtype=dtype).reshape(1, -1).expand(B, -1)
            mask_t = (
                torch.ones_like(motif_t) if motif_mask is None
                else torch.as_tensor(motif_mask, device=device, dtype=dtype).reshape(1, -1).expand(B, -1)
            )

        audio_features = self.wave_model(mel.to(device))
        timing_latent = self.downsample_timing(timing.to(device), latent_frames)

        # Both conditioning embeddings are built once. They do not depend on the
        # timestep, so recomputing them 50 times is pure waste.
        cond_emb = self.unet_model.cond_emb(
            diff_t, style_t, anps_t, pnps_t, motif_t, mask_t,
        )
        uncond_emb = self.unet_model.cond_emb.unconditional(B, device, cond_emb.dtype)

        z = torch.randn(
            B, self.z_channels, latent_frames,
            device=device, dtype=dtype, generator=generator,
        )

        sequence = self.scheduler.timestep_sequence(ddim_steps)
        for i, t_val in enumerate(sequence):
            t_prev_val = sequence[i + 1] if i + 1 < len(sequence) else 0
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            t_prev = torch.full((B,), t_prev_val, device=device, dtype=torch.long)

            pred = self.unet_model(
                z, t, audio_features, timing_latent,
                difficulty=diff_t, style=style_t, cond_emb=cond_emb,
            )

            if cfg_scale != 1.0:
                uncond = self.unet_model(
                    z, t, audio_features, timing_latent,
                    difficulty=diff_t, style=style_t, cond_emb=uncond_emb,
                )
                pred = uncond + cfg_scale * (pred - uncond)

            z = self.scheduler.ddim_step(pred, z, t, t_prev, eta=eta)

            if progress and (i % 10 == 0 or i == len(sequence) - 1):
                print(f"  step {i + 1}/{len(sequence)}  (t={t_val})")

        return z

    def make_ema(self, decay: float = 0.9995, warmup: int = 1000) -> EMA:
        return EMA(self.trainable_parameters(), decay=decay, warmup=warmup)
