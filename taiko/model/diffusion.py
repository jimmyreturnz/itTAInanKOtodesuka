"""
taiko/model/diffusion.py

Full diffusion model — wraps autoencoder + audio encoder + U-Net.

Changes from previous version:
  - smooth_l1 loss (Huber) instead of MSE — more stable at low timesteps
  - masked loss using valid_mask — ignores padding regions
  - CFG dropout moved to CondEmbedding in unet.py
  - add_noise returns z_noisy only (not tuple)
  - mask_ratio logged for monitoring
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from taiko.model.autoencoder import BeatmapAutoencoder, AutoencoderConfig
from taiko.model.audio_encoder import MelEncoder1D
from taiko.model.unet import TaikoDiffusionUNet
from taiko.model.noise_scheduler import NoiseScheduler


def disabled_train(self, mode=True):
    return self


class TaikoDiffusion(nn.Module):

    def __init__(
        self,
        autoencoder_ckpt:    str,
        timesteps:           int   = 1000,
        beta_schedule:       str   = "linear",
        cfg_scale:           float = 1.5,
        z_channels:          int   = 16,
        # Audio encoder
        n_mels:              int   = 128,
        audio_base_channels: int   = 64,
        audio_channel_mult:  list  = None,
        # U-Net
        unet_base_channels:  int   = 64,
        unet_channel_mult:   list  = None,
        unet_num_res_blocks: int   = 2,
        unet_dropout:        float = 0.1,
        n_styles:            int   = 4,
        cfg_dropout:         float = 0.5,
        use_checkpoint:      bool  = False,
        use_s4:              bool  = False,
    ):
        super().__init__()
        self.cfg_scale  = cfg_scale
        self.z_channels = z_channels

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
        audio_out_channels = self.wave_model.out_channels
        print(f"Audio encoder output channels: {audio_out_channels}")

        # ---- U-Net ------------------------------------------------------ #
        if unet_channel_mult is None:
            unet_channel_mult = [1, 2, 4]
        self.unet_model = TaikoDiffusionUNet(
            z_channels     = z_channels,
            base_channels  = unet_base_channels,
            channel_mult   = unet_channel_mult,
            num_res_blocks = unet_num_res_blocks,
            dropout        = unet_dropout,
            n_styles       = n_styles,
            audio_channels = audio_out_channels,
            cfg_dropout    = cfg_dropout,
            use_checkpoint = use_checkpoint,
            use_s4         = use_s4,
        )

        # ---- Noise scheduler -------------------------------------------- #
        self.scheduler = NoiseScheduler(timesteps=timesteps, beta_schedule=beta_schedule)

    def to(self, device):
        super().to(device)
        self.scheduler.to(device)
        return self

    def encode(self, beatmap_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.first_stage_model.encode(beatmap_tensor).mode()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.first_stage_model.decode(z))

    def training_loss(
        self,
        beatmap:    torch.Tensor,   # [B, 7, T]
        mel:        torch.Tensor,   # [B, 128, T_mel]
        difficulty: torch.Tensor,   # [B] float
        style:      torch.Tensor,   # [B] int
        valid_mask: torch.Tensor,   # [B, T] 1=real, 0=padding
    ) -> tuple[torch.Tensor, dict]:

        B      = beatmap.shape[0]
        device = beatmap.device

        # 1. Encode beatmap → latent
        z = self.encode(beatmap)                    # [B, 16, T_lat]

        # 2. Sample timestep + noise
        t     = torch.randint(0, self.scheduler.timesteps, (B,), device=device).long()
        noise = torch.randn_like(z)

        # 3. Forward diffusion
        z_noisy = self.scheduler.add_noise(z, t, noise)

        # 4. Audio features
        audio_features = self.wave_model(mel)

        # 5. Predict noise
        noise_pred = self.unet_model(
            z_noisy, t, audio_features, difficulty, style
        )

        # 6. Masked Smooth L1 loss (Huber) — only on real frames
        # Downsample valid_mask from beatmap resolution to latent resolution
        T_lat = z.shape[2]
        mask  = valid_mask[:, ::16][:, :T_lat].unsqueeze(1)   # [B, 1, T_lat]

        loss_per_element = F.smooth_l1_loss(noise_pred, noise, reduction="none")
        loss = (loss_per_element * mask).sum() / mask.sum().clamp(min=1)

        with torch.no_grad():
            mae = ((noise_pred - noise).abs() * mask).sum() / mask.sum().clamp(min=1)

        log_dict = {
            "loss":       loss.item(),
            "mae":        mae.item(),
            "t_mean":     t.float().mean().item(),
            "mask_ratio": mask.mean().item(),
        }
        return loss, log_dict

    @torch.no_grad()
    def generate(
        self,
        mel:           torch.Tensor,
        difficulty:    float,
        style:         int,
        latent_length: int,
        ddim_steps:    int   = 50,
        cfg_scale:     float = None,
        eta:           float = 0.0,
        device:        torch.device = None,
    ) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        if cfg_scale is None:
            cfg_scale = self.cfg_scale

        B       = 1
        diff_t  = torch.tensor([difficulty], device=device)
        style_t = torch.tensor([style],      device=device, dtype=torch.long)

        audio_features = self.wave_model(mel)
        z = torch.randn(B, self.z_channels, latent_length, device=device)

        step_size = self.scheduler.timesteps // ddim_steps
        timesteps = list(reversed(range(0, self.scheduler.timesteps, step_size)))

        for i, t_val in enumerate(timesteps):
            t_prev_val = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            t      = torch.full((B,), t_val,      device=device, dtype=torch.long)
            t_prev = torch.full((B,), t_prev_val, device=device, dtype=torch.long)

            noise_cond = self.unet_model(
                z, t, audio_features, diff_t, style_t, drop_cond=False
            )

            if cfg_scale != 1.0:
                noise_uncond = self.unet_model(
                    z, t, audio_features,
                    torch.zeros_like(diff_t),
                    torch.zeros_like(style_t),
                    drop_cond=True,
                )
                noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
            else:
                noise_pred = noise_cond

            z = self.scheduler.ddim_sample(noise_pred, z, t, t_prev, eta=eta)

        return z