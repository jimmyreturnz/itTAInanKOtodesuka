"""
taiko/model/diffusion.py

Full diffusion model — wraps autoencoder + audio encoder + U-Net.
Closely follows Mug-Diffusion's DDPM class structure.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional

from taiko.model.autoencoder import BeatmapAutoencoder, AutoencoderConfig
from taiko.model.audio_encoder import MelEncoder1D
from taiko.model.unet import TaikoDiffusionUNet
from taiko.model.noise_scheduler import NoiseScheduler


def disabled_train(self, mode=True):
    """Freeze a module — from Mug-Diffusion."""
    return self


class TaikoDiffusion(nn.Module):
    """
    Full taiko diffusion model.

    Components (matching Mug-Diffusion structure):
      first_stage_model → BeatmapAutoencoder  (frozen)
      wave_model        → MelEncoder1D        (trained)
      unet_model        → TaikoDiffusionUNet  (trained)
    """

    def __init__(
        self,
        autoencoder_ckpt: str,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        parameterization: str = "eps",   # predict noise (eps) like Mug-Diffusion
        cfg_scale: float = 1.5,
        z_channels: int = 16,
        # Audio encoder
        n_mels: int = 128,
        audio_base_channels: int = 64,
        audio_channel_mult: list = None,
        # U-Net
        unet_base_channels: int = 64,
        unet_channel_mult: list = None,
        unet_num_res_blocks: int = 2,
        unet_dropout: float = 0.1,
        n_styles: int = 4,
    ):
        super().__init__()
        self.parameterization = parameterization
        self.cfg_scale        = cfg_scale
        self.z_channels       = z_channels

        # ---- Autoencoder (frozen) --------------------------------------- #
        ae_config = AutoencoderConfig(z_channels=z_channels)
        self.first_stage_model = BeatmapAutoencoder(ae_config)
        ckpt = torch.load(autoencoder_ckpt, map_location="cpu")
        self.first_stage_model.load_state_dict(ckpt["model"])
        self.first_stage_model.eval()
        self.first_stage_model.train = disabled_train.__get__(self.first_stage_model)
        for p in self.first_stage_model.parameters():
            p.requires_grad = False
        print(f"Loaded frozen autoencoder from {autoencoder_ckpt} (step {ckpt['step']})")

        # ---- Audio encoder --------------------------------------------- #
        if audio_channel_mult is None:
            audio_channel_mult = [1, 1, 2, 2]
        self.wave_model = MelEncoder1D(
            n_mels=n_mels,
            base_channels=audio_base_channels,
            channel_mult=audio_channel_mult,
        )
        # Derive audio_channels directly from the encoder — always correct
        audio_out_channels = [int(c) for c in self.wave_model.out_channels]
        print(f"Audio encoder output channels: {audio_out_channels}")

        # ---- U-Net ------------------------------------------------------ #
        if unet_channel_mult is None:
            unet_channel_mult = [1, 2, 4]
        self.unet_model = TaikoDiffusionUNet(
            z_channels=z_channels,
            base_channels=unet_base_channels,
            channel_mult=unet_channel_mult,
            num_res_blocks=unet_num_res_blocks,
            dropout=unet_dropout,
            n_styles=n_styles,
            audio_channels=audio_out_channels,
        )

        # ---- Noise scheduler -------------------------------------------- #
        self.scheduler = NoiseScheduler(timesteps=timesteps, beta_schedule=beta_schedule)

    def to(self, device):
        super().to(device)
        self.scheduler.to(device)
        return self

    def encode(self, beatmap_tensor: torch.Tensor) -> torch.Tensor:
        """Encode beatmap tensor to latent (deterministic mode)."""
        with torch.no_grad():
            return self.first_stage_model.encode(beatmap_tensor).mode()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to beatmap tensor."""
        with torch.no_grad():
            return torch.sigmoid(self.first_stage_model.decode(z))

    def training_loss(
        self,
        beatmap: torch.Tensor,       # [B, 7, T]
        mel: torch.Tensor,           # [B, 128, T_audio]
        difficulty: torch.Tensor,    # [B] float
        style: torch.Tensor,         # [B] int
        valid_mask: torch.Tensor,    # [B, T] not used in diffusion loss but kept for API consistency
    ) -> tuple[torch.Tensor, dict]:

        B = beatmap.shape[0]
        device = beatmap.device

        # Encode to latent (frozen autoencoder)
        z = self.encode(beatmap)                          # [B, 16, T//16]

        # Sample timestep
        t = torch.randint(0, self.scheduler.timesteps, (B,), device=device).long()

        # Add noise
        noise  = torch.randn_like(z)
        z_noisy = self.scheduler.add_noise(z, t, noise)

        # Get audio features
        audio_features = self.wave_model(mel)

        # Predict noise
        noise_pred = self.unet_model(
            z_noisy, t, audio_features, difficulty, style
        )

        # Downsample valid_mask to latent resolution
        # valid_mask: [B, T_beatmap] -> [B, T_latent]
        compression = self.first_stage_model.compression_ratio
        latent_len  = z.shape[2]
        # Average pool mask down to latent length
        mask_down = valid_mask.unsqueeze(1)  # [B, 1, T]
        mask_down = nn.functional.adaptive_avg_pool1d(mask_down, latent_len)
        mask_down = (mask_down.squeeze(1) > 0.5).float()  # [B, T_latent]

        # Masked MSE loss — only compute on real (non-padded) frames
        diff = (noise_pred - noise) ** 2  # [B, z_ch, T_latent]
        mask_expanded = mask_down.unsqueeze(1)  # [B, 1, T_latent]
        loss = (diff * mask_expanded).sum() / (mask_expanded.sum() * z.shape[1] + 1e-8)

        with torch.no_grad():
            mae        = ((noise_pred - noise).abs() * mask_expanded).sum() / (mask_expanded.sum() * z.shape[1] + 1e-8)
            mask_ratio = mask_down.mean().item()

        log_dict = {
            "loss":       loss.item(),
            "mae":        mae.item(),
            "t_mean":     t.float().mean().item(),
            "mask_ratio": mask_ratio,
        }
        return loss, log_dict

    @torch.no_grad()
    def generate(
        self,
        mel: torch.Tensor,           # [1, 128, T_audio]
        difficulty: float,
        style: int,
        latent_length: int,          # T_latent = song_frames // compression_ratio
        ddim_steps: int = 50,
        cfg_scale: float = None,
        eta: float = 0.0,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Generate a beatmap latent from audio.
        Uses DDIM sampling (fast, 50 steps instead of 1000).
        Uses CFG for difficulty/style control.
        """
        if device is None:
            device = next(self.parameters()).device
        if cfg_scale is None:
            cfg_scale = self.cfg_scale

        B = 1
        diff_t  = torch.tensor([difficulty], device=device)
        style_t = torch.tensor([style],      device=device, dtype=torch.long)

        # Audio features — computed once, reused for all steps
        audio_features = self.wave_model(mel)

        # Start from pure noise
        z = torch.randn(B, self.z_channels, latent_length, device=device)

        # DDIM timestep schedule
        step_size  = self.scheduler.timesteps // ddim_steps
        timesteps  = list(reversed(range(0, self.scheduler.timesteps, step_size)))

        for i, t_val in enumerate(timesteps):
            t_prev_val = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            t      = torch.full((B,), t_val,      device=device, dtype=torch.long)
            t_prev = torch.full((B,), t_prev_val, device=device, dtype=torch.long)

            # Conditioned prediction
            noise_cond = self.unet_model(
                z, t, audio_features, diff_t, style_t, drop_cond=False
            )

            if cfg_scale != 1.0:
                # Unconditioned prediction (CFG)
                noise_uncond = self.unet_model(
                    z, t, audio_features,
                    torch.zeros_like(diff_t),
                    torch.zeros_like(style_t),
                    drop_cond=True,
                )
                # CFG interpolation
                noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
            else:
                noise_pred = noise_cond

            z = self.scheduler.ddim_sample(noise_pred, z, t, t_prev, eta=eta)

        return z