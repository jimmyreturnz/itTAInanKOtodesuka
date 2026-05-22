"""
scripts/train_diffusion.py

Phase 3 — Train the Diffusion U-Net.

The autoencoder (Phase 2) is frozen. This script trains:
  - MelSpectrogramScaleEncoder1D  (audio → multi-scale features)
  - Diffusion U-Net               (denoise latent conditioned on audio)
  - DDPM noise scheduler          (cosine, 1000 steps)

Conditioning signals:
  - Audio features   → cross-attention at each U-Net scale
  - Difficulty       → embedded + added to timestep embedding
  - Style (SR)       → embedded + added to timestep embedding
  - CFG dropout 10%  → enables classifier-free guidance at inference

Usage:
    python scripts/train_diffusion.py
    python scripts/train_diffusion.py --resume checkpoints/diffusion/best.pt
    python scripts/train_diffusion.py --config configs/base.yaml
"""

from __future__ import annotations
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from matplotlib.pyplot import stem
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from taiko.data.osu_parser import OsuTaikoParser
from taiko.data.tensor_repr import beatmap_to_tensor, FRAME_MS, N_CHANNELS
from taiko.model.autoencoder import BeatmapAutoencoder, AutoencoderConfig


# ────────────────────────────────────────────────────────────────────────────
# Defaults (overridden by configs/base.yaml [diffusion] section)
# ────────────────────────────────────────────────────────────────────────────

DEFAULTS = dict(
    # Paths
    cache_file        = "taiko_files_filtered.json",
    mel_dir           = "data/processed/mels",
    ae_ckpt           = "checkpoints/autoencoder/best.pt",
    ckpt_dir          = "checkpoints/diffusion",
    log_dir           = "runs/diffusion",

    # Data
    pad_frames        = 18_000,   # beatmap frames   (18 000 × 20ms = 6 min)
    val_ratio         = 0.05,
    num_workers       = 0,        # 0 for Windows

    # Autoencoder (must match Phase 2 config)
    ae_x_channels     = 7,
    ae_middle_channels= 32,
    ae_z_channels     = 16,
    ae_channel_mult   = [1, 1, 2, 2, 4],
    ae_num_res_blocks = 2,
    ae_num_groups     = 8,
    ae_kl_weight      = 1e-6,

    # Diffusion
    timesteps         = 1000,
    beta_schedule     = "cosine",
    z_channels        = 16,       # must match ae_z_channels
    cfg_dropout       = 0.10,     # probability of dropping conditioning

    # Audio encoder
    mel_bins          = 128,
    audio_base_ch     = 32,       # keep small for 4 GB VRAM
    audio_levels      = 3,        # number of downsampling levels

    # U-Net
    unet_base_ch      = 64,
    unet_levels       = 3,
    unet_res_blocks   = 2,
    unet_attn_heads   = 4,
    unet_groups       = 8,

    # Conditioning
    diff_emb_dim      = 128,      # timestep / difficulty / style embed size
    n_difficulties    = 10,       # 0–9 difficulty buckets
    n_styles          = 5,        # 0–4 style buckets (e.g. SR ranges)

    # Training
    batch_size        = 4,
    lr                = 1e-4,
    weight_decay      = 0.01,
    grad_clip         = 1.0,
    max_epochs        = 100,
    max_steps         = None,
    warmup_steps      = 1000,
    lr_min            = 1e-6,
    val_every         = 500,
    save_every        = 500,
    log_every         = 20,
    val_batches       = 30,
    keep_last_n       = 3,
    precision         = "fp32",   # GTX 1650: fp32 only
    grad_checkpoint   = False,    # enable if OOM
)


# ────────────────────────────────────────────────────────────────────────────
# Config loader
# ────────────────────────────────────────────────────────────────────────────

def load_config(yaml_path: str | None) -> dict:
    cfg = dict(DEFAULTS)
    if yaml_path and Path(yaml_path).exists():
        raw = yaml.safe_load(Path(yaml_path).read_text())
        diff_cfg = raw.get("diffusion", {})
        cfg.update(diff_cfg)
    return cfg


# ────────────────────────────────────────────────────────────────────────────
# DDPM noise schedule
# ────────────────────────────────────────────────────────────────────────────

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule — more stable than linear for latent diffusion."""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, min=1e-5, max=0.999)


def linear_beta_schedule(timesteps: int,
                         beta_start: float = 1e-4,
                         beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


class DDPMScheduler:
    """
    Forward (noising) process only — used during training.
    Stores all buffers on CPU; move to device as needed.
    """

    def __init__(self, timesteps: int = 1000, schedule: str = "cosine"):
        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            betas = linear_beta_schedule(timesteps)

        alphas            = 1.0 - betas
        alphas_cumprod    = torch.cumprod(alphas, dim=0)

        self.timesteps         = timesteps
        self.betas             = betas
        self.alphas_cumprod    = alphas_cumprod
        self.sqrt_ac           = alphas_cumprod.sqrt()
        self.sqrt_one_minus_ac = (1.0 - alphas_cumprod).sqrt()

    def add_noise(self,
              x_start: torch.Tensor,
              noise:   torch.Tensor,
              t:       torch.Tensor) -> torch.Tensor:
        """
        Forward process: x_t = sqrt(ā_t)·x_0 + sqrt(1-ā_t)·ε
        t : [B] long tensor of timestep indices
        """
        device = x_start.device

        s_ac = self.sqrt_ac.to(device)[t].view(-1, 1, 1)
        s_oac = self.sqrt_one_minus_ac.to(device)[t].view(-1, 1, 1)

        return s_ac * x_start + s_oac * noise


# ────────────────────────────────────────────────────────────────────────────
# Audio encoder — multi-scale mel features
# ────────────────────────────────────────────────────────────────────────────

class ResBlock1D(nn.Module):
    """1-D residual block with two dilated convolutions."""

    def __init__(self, ch: int, dilation: int = 1, groups: int = 8):
        super().__init__()
        pad = dilation
        self.net = nn.Sequential(
            nn.GroupNorm(groups, ch),
            nn.SiLU(),
            nn.Conv1d(ch, ch, 3, padding=pad, dilation=dilation),
            nn.GroupNorm(groups, ch),
            nn.SiLU(),
            nn.Conv1d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class MelScaleEncoder(nn.Module):
    """
    mel [B, 128, T_audio]
      → stem Conv → levels of (ResBlocks + downsample)
      → returns list of feature maps at each resolution
    """

    def __init__(self, mel_bins: int = 128, base_ch: int = 32, levels: int = 3,
                 groups: int = 8):
        super().__init__()
        self.stem = nn.Conv1d(mel_bins, base_ch, kernel_size=3, padding=1)

        self.levels = nn.ModuleList()
        self.downs  = nn.ModuleList()
        ch = base_ch
        for i in range(levels):
            block = nn.Sequential(
                ResBlock1D(ch, dilation=1, groups=groups),
                ResBlock1D(ch, dilation=2, groups=groups),
                ResBlock1D(ch, dilation=4, groups=groups),
            )
            self.levels.append(block)
            next_ch = ch * 2
            self.downs.append(
                nn.Conv1d(ch, next_ch, kernel_size=4, stride=2, padding=1)
            )
            ch = next_ch

        self.out_channels = [base_ch * (2 ** i) for i in range(levels)]

    def forward(self, mel: torch.Tensor) -> list[torch.Tensor]:
        """Returns features at each scale, coarsest last."""
        x = self.stem(mel)
        feats = []
        for block, down in zip(self.levels, self.downs):
            x = block(x)
            feats.append(x)
            x = down(x)
        return feats   # [fine, ..., coarse]


# ────────────────────────────────────────────────────────────────────────────
# Timestep / conditioning embedding
# ────────────────────────────────────────────────────────────────────────────

def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal position embedding for diffusion timesteps."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=timesteps.device) / (half - 1)
    )
    args  = timesteps[:, None].float() * freqs[None]
    emb   = torch.cat([args.sin(), args.cos()], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ConditioningEmbedder(nn.Module):
    """
    Produces a single conditioning vector [B, emb_dim] from:
      - diffusion timestep  t
      - difficulty          (int 0–n_difficulties-1)
      - style               (int 0–n_styles-1)
    """

    def __init__(self, emb_dim: int, n_difficulties: int, n_styles: int):
        super().__init__()
        self.emb_dim = emb_dim

        # Timestep MLP
        self.t_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )
        # Discrete embeddings
        self.diff_emb  = nn.Embedding(n_difficulties + 1, emb_dim)  # +1 for CFG null
        self.style_emb = nn.Embedding(n_styles      + 1, emb_dim)  # +1 for CFG null

        self.null_diff  = n_difficulties
        self.null_style = n_styles

    def forward(self,
                t:          torch.Tensor,   # [B] long
                difficulty: torch.Tensor,   # [B] long
                style:      torch.Tensor,   # [B] long
                ) -> torch.Tensor:          # [B, emb_dim]
        t_emb = sinusoidal_embedding(t, self.emb_dim)
        t_emb = self.t_mlp(t_emb)
        return t_emb + self.diff_emb(difficulty) + self.style_emb(style)


# ────────────────────────────────────────────────────────────────────────────
# 1-D Cross-Attention
# ────────────────────────────────────────────────────────────────────────────

class CrossAttention1D(nn.Module):
    """
    Query from latent [B, C, T_lat], key/value from audio [B, C_aud, T_aud].
    Projects both to d_model, applies MHA, projects back.
    """

    def __init__(self, q_ch: int, kv_ch: int, heads: int = 4, groups: int = 8):
        super().__init__()
        d_model = q_ch
        self.norm_q  = nn.GroupNorm(groups, q_ch)
        self.norm_kv = nn.LayerNorm(kv_ch)
        self.q_proj  = nn.Linear(q_ch,  d_model)
        self.k_proj  = nn.Linear(kv_ch, d_model)
        self.v_proj  = nn.Linear(kv_ch, d_model)
        self.attn    = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.out     = nn.Linear(d_model, q_ch)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """
        x   : [B, q_ch,  T_lat]
        ctx : [B, kv_ch, T_aud]
        """
        B, C, T = x.shape
        xn = self.norm_q(x).permute(0, 2, 1)          # [B, T_lat, C]
        cn = self.norm_kv(ctx.permute(0, 2, 1))        # [B, T_aud, kv_ch]

        q = self.q_proj(xn)
        k = self.k_proj(cn)
        v = self.v_proj(cn)

        out, _ = self.attn(q, k, v)                    # [B, T_lat, d_model]
        out = self.out(out).permute(0, 2, 1)            # [B, q_ch, T_lat]
        return x + out


# ────────────────────────────────────────────────────────────────────────────
# U-Net building blocks
# ────────────────────────────────────────────────────────────────────────────

class ResNetBlock1D(nn.Module):
    """ResNet block with conditioning injection (scale-shift from emb)."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int,
                 groups: int = 8, grad_checkpoint: bool = False):
        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        self.norm1  = nn.GroupNorm(groups, in_ch)
        self.conv1  = nn.Conv1d(in_ch,  out_ch, 3, padding=1)
        self.norm2  = nn.GroupNorm(groups, out_ch)
        self.conv2  = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_ch * 2))
        self.skip   = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act    = nn.SiLU()

    def _forward(self, x, emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # scale-shift conditioning
        ss = self.emb_proj(emb).unsqueeze(-1)          # [B, 2*out_ch, 1]
        scale, shift = ss.chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.act(h)
        h = self.conv2(h)
        return h + self.skip(x)

    def forward(self, x, emb):
        if self.grad_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward, x, emb)
        return self._forward(x, emb)


# ────────────────────────────────────────────────────────────────────────────
# Diffusion U-Net
# ────────────────────────────────────────────────────────────────────────────

class DiffusionUNet1D(nn.Module):
    """
    1-D U-Net for latent diffusion on beatmap sequences.

    Input:  noisy latent [B, z_channels, T_lat]
    Output: predicted noise [B, z_channels, T_lat]

    Conditioned via:
      - emb [B, emb_dim]                    → ResNet scale-shift
      - audio_feats list[[B, C_aud, T_aud]] → cross-attention at each level
    """

    def __init__(self,
                 z_channels:   int,
                 base_ch:      int,
                 levels:       int,
                 res_blocks:   int,
                 emb_dim:      int,
                 audio_chs:    list[int],
                 attn_heads:   int = 4,
                 groups:       int = 8,
                 grad_checkpoint: bool = False,
                 ):
        super().__init__()
        self.levels = levels
        gc = grad_checkpoint

        # --- stem / projection ---
        self.in_proj = nn.Conv1d(z_channels, base_ch, 3, padding=1)

        # --- encoder path ---
        self.enc_blocks = nn.ModuleList()   # ResNet blocks
        self.enc_attns  = nn.ModuleList()   # cross-attention to audio
        self.enc_downs  = nn.ModuleList()   # downsampling convolutions
        ch = base_ch
        enc_channels = []
        for lvl in range(levels):
            aud_ch = audio_chs[min(lvl, len(audio_chs) - 1)]
            blocks = nn.ModuleList([
                ResNetBlock1D(ch, ch, emb_dim, groups, gc) for _ in range(res_blocks)
            ])
            attn = CrossAttention1D(ch, aud_ch, attn_heads, groups)
            self.enc_blocks.append(blocks)
            self.enc_attns.append(attn)
            enc_channels.append(ch)
            if lvl < levels - 1:
                self.enc_downs.append(
                    nn.Conv1d(ch, ch * 2, kernel_size=4, stride=2, padding=1)
                )
                ch *= 2
            else:
                self.enc_downs.append(nn.Identity())   # no-op at bottom

        # --- bottleneck ---
        self.mid_block1 = ResNetBlock1D(ch, ch, emb_dim, groups, gc)
        self.mid_attn   = CrossAttention1D(ch, audio_chs[-1], attn_heads, groups)
        self.mid_block2 = ResNetBlock1D(ch, ch, emb_dim, groups, gc)

        # --- decoder path ---
        self.dec_blocks = nn.ModuleList()
        self.dec_attns  = nn.ModuleList()
        self.dec_ups    = nn.ModuleList()
        for lvl in reversed(range(levels)):
            aud_ch   = audio_chs[min(lvl, len(audio_chs) - 1)]
            skip_ch  = enc_channels[lvl]
            in_ch    = ch + skip_ch          # concat skip connection
            out_ch   = base_ch * (2 ** lvl)
            blocks = nn.ModuleList([
                ResNetBlock1D(in_ch if i == 0 else out_ch, out_ch, emb_dim, groups, gc)
                for i in range(res_blocks)
            ])
            attn = CrossAttention1D(out_ch, aud_ch, attn_heads, groups)
            self.dec_blocks.append(blocks)
            self.dec_attns.append(attn)
            if lvl > 0:
                self.dec_ups.append(
                    nn.ConvTranspose1d(out_ch, out_ch // 2, kernel_size=4, stride=2, padding=1)
                )
                ch = out_ch // 2
            else:
                self.dec_ups.append(nn.Identity())
                ch = out_ch

        # --- output ---
        self.out_norm = nn.GroupNorm(groups, ch)
        self.out_proj = nn.Conv1d(ch, z_channels, 3, padding=1)

    def forward(self,
            x:           torch.Tensor,        # [B, z_ch, T_lat]
            emb:         torch.Tensor,        # [B, emb_dim]
            audio_feats: list[torch.Tensor],  # multi-scale audio
            ) -> torch.Tensor:

        input_len = x.shape[-1]   # <-- remember original latent length

        h = self.in_proj(x)

        # --- encoder ---
        skips = []
        for lvl, (blocks, attn, down) in enumerate(
                zip(self.enc_blocks, self.enc_attns, self.enc_downs)):

            for blk in blocks:
                h = blk(h, emb)

            # Align audio features
            aud = audio_feats[min(lvl, len(audio_feats) - 1)]
            aud = F.interpolate(
                aud,
                size=h.shape[-1],
                mode="linear",
                align_corners=False
            )

            h = attn(h, aud)

            skips.append(h)
            h = down(h)

        # --- bottleneck ---
        h = self.mid_block1(h, emb)

        aud = audio_feats[-1]
        aud = F.interpolate(
            aud,
            size=h.shape[-1],
            mode="linear",
            align_corners=False
        )

        h = self.mid_attn(h, aud)
        h = self.mid_block2(h, emb)

        # --- decoder ---
        for lvl_idx, (blocks, attn, up) in enumerate(
                zip(self.dec_blocks, self.dec_attns, self.dec_ups)):

            skip = skips[-(lvl_idx + 1)]

            # Match temporal length before concat
            min_t = min(h.shape[-1], skip.shape[-1])
            h    = h[..., :min_t]
            skip = skip[..., :min_t]

            h = torch.cat([h, skip], dim=1)

            for blk in blocks:
                h = blk(h, emb)

            lvl_real = self.levels - 1 - lvl_idx

            aud = audio_feats[min(lvl_real, len(audio_feats) - 1)]
            aud = F.interpolate(
                aud,
                size=h.shape[-1],
                mode="linear",
                align_corners=False
            )

            h = attn(h, aud)

            h = up(h)

        h = F.silu(self.out_norm(h))
        h = self.out_proj(h)

        # ------------------------------------------------------------------
        # FINAL LENGTH FIX
        # ------------------------------------------------------------------
        if h.shape[-1] > input_len:
            h = h[..., :input_len]

        elif h.shape[-1] < input_len:
            h = F.pad(h, (0, input_len - h.shape[-1]))

        return h


# ────────────────────────────────────────────────────────────────────────────
# Full diffusion model wrapper
# ────────────────────────────────────────────────────────────────────────────

class TaikoDiffusionModel(nn.Module):
    """
    Wraps:
      - MelScaleEncoder          (audio → multi-scale features)
      - ConditioningEmbedder     (t + difficulty + style → emb vector)
      - DiffusionUNet1D          (predict noise from noisy latent)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.audio_encoder = MelScaleEncoder(
            mel_bins = cfg["mel_bins"],
            base_ch  = cfg["audio_base_ch"],
            levels   = cfg["audio_levels"],
            groups   = cfg["unet_groups"],
        )
        self.cond_embedder = ConditioningEmbedder(
            emb_dim      = cfg["diff_emb_dim"],
            n_difficulties = cfg["n_difficulties"],
            n_styles     = cfg["n_styles"],
        )
        self.unet = DiffusionUNet1D(
            z_channels   = cfg["z_channels"],
            base_ch      = cfg["unet_base_ch"],
            levels       = cfg["unet_levels"],
            res_blocks   = cfg["unet_res_blocks"],
            emb_dim      = cfg["diff_emb_dim"],
            audio_chs    = self.audio_encoder.out_channels,
            attn_heads   = cfg["unet_attn_heads"],
            groups       = cfg["unet_groups"],
            grad_checkpoint = cfg["grad_checkpoint"],
        )

    def forward(self,
                noisy_latent: torch.Tensor,   # [B, 16, T_lat]
                t:            torch.Tensor,   # [B]
                mel:          torch.Tensor,   # [B, 128, T_mel]
                difficulty:   torch.Tensor,   # [B]
                style:        torch.Tensor,   # [B]
                ) -> torch.Tensor:
        audio_feats = self.audio_encoder(mel)
        emb         = self.cond_embedder(t, difficulty, style)
        return self.unet(noisy_latent, emb, audio_feats)

    def count_parameters(self) -> dict:
        def n(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "audio_encoder": n(self.audio_encoder),
            "cond_embedder": n(self.cond_embedder),
            "unet":          n(self.unet),
            "total":         n(self),
        }


# ────────────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────────────

class DiffusionDataset(Dataset):
    """
    For each .osu file:
      - Loads beatmap tensor  [7, T]
      - Finds the pre-computed mel  data/processed/mels/<stem>.npy  [128, T_mel]
      - Extracts difficulty + style from the .osu metadata
      - Pads / truncates both to fixed lengths
    """

    def __init__(self,
                 osu_files:  list[Path],
                 mel_dir:    Path,
                 pad_frames: int = 18_000,
                 mel_frames: int = 9_000,   # mel is ~2× coarser than 20ms frames
                 ):
        self.files      = osu_files
        self.mel_dir    = mel_dir
        self.pad_frames = pad_frames
        self.mel_frames = mel_frames
        self.parser     = OsuTaikoParser()

    def __len__(self):
        return len(self.files)

    def _get_mel(self, osu_path: Path) -> np.ndarray | None:
        stem = osu_path.stem

        candidates = [
            self.mel_dir / f"{stem}.npz",
            self.mel_dir / f"{osu_path.parent.name}.npz",
            self.mel_dir / f"{stem}.npy",
            self.mel_dir / f"{osu_path.parent.name}.npy",
        ]

        for c in candidates:
            if c.exists():
                data = np.load(c)

                if isinstance(data, np.lib.npyio.NpzFile):
                    mel = data[data.files[0]]
                else:
                    mel = data

                return mel.astype(np.float32)

        return None

    def _difficulty_bucket(self, bm) -> int:
        """Map star rating to 0–9 bucket."""
        try:
            sr = float(getattr(bm, "star_rating", 0) or 0)
        except Exception:
            sr = 0.0
        return min(9, max(0, int(sr)))

    def _style_bucket(self, bm) -> int:
        """Map to 0–4 style bucket based on note density."""
        try:
            density = bm.note_count / max(bm.duration_ms / 1000, 1)
        except Exception:
            density = 4.0
        # Rough density buckets: <4, 4-7, 7-10, 10-14, 14+
        for i, thresh in enumerate([4, 7, 10, 14]):
            if density < thresh:
                return i
        return 4

    def __getitem__(self, idx: int) -> dict:
        for _ in range(5):
            try:
                path = self.files[idx]
                bm   = self.parser.parse_file(path)
                if bm.note_count == 0:
                    raise ValueError("empty map")

                # ---- beatmap tensor ----
                tensor = beatmap_to_tensor(bm, pad_to=None)  # [7, T_raw]
                T = tensor.shape[1]
                valid_len  = min(T, self.pad_frames)
                valid_mask = np.zeros(self.pad_frames, dtype=np.float32)
                valid_mask[:valid_len] = 1.0
                if T < self.pad_frames:
                    pad = np.zeros((N_CHANNELS, self.pad_frames - T), dtype=np.float32)
                    tensor = np.concatenate([tensor, pad], axis=1)
                else:
                    tensor = tensor[:, :self.pad_frames]

                # ---- mel spectrogram ----
                mel = self._get_mel(path)
                if mel is None:
                    raise FileNotFoundError(f"mel not found for {path.name}")
                # Pad / truncate mel
                Tm = mel.shape[1]
                if Tm < self.mel_frames:
                    mel = np.pad(mel, ((0, 0), (0, self.mel_frames - Tm)))
                else:
                    mel = mel[:, :self.mel_frames]

                difficulty = self._difficulty_bucket(bm)
                style      = self._style_bucket(bm)

                return {
                    "tensor":     torch.from_numpy(tensor).float(),      # [7, pad_frames]
                    "mel":        torch.from_numpy(mel).float(),          # [128, mel_frames]
                    "valid_mask": torch.from_numpy(valid_mask).float(),   # [pad_frames]
                    "difficulty": torch.tensor(difficulty, dtype=torch.long),
                    "style":      torch.tensor(style,      dtype=torch.long),
                }

            except Exception as e:
                idx = random.randint(0, len(self.files) - 1)

        # Fallback
        return {
            "tensor":     torch.zeros(N_CHANNELS, self.pad_frames),
            "mel":        torch.zeros(128, self.mel_frames if hasattr(self, 'mel_frames') else 9_000),
            "valid_mask": torch.zeros(self.pad_frames),
            "difficulty": torch.tensor(0, dtype=torch.long),
            "style":      torch.tensor(0, dtype=torch.long),
        }


def load_split(cache_file: str, val_ratio: float = 0.05) -> tuple[list, list]:
    files = [Path(p) for p in json.loads(Path(cache_file).read_text())]
    random.seed(42)
    random.shuffle(files)
    n_val = max(1, int(len(files) * val_ratio))
    return files[n_val:], files[:n_val]


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def get_lr(step: int, warmup: int, max_steps: int | None,
           lr: float, lr_min: float) -> float:
    if step < warmup:
        return lr * max(step, 1) / max(warmup, 1)
    if max_steps is None or step >= max_steps:
        return lr_min
    t = (step - warmup) / (max_steps - warmup)
    return lr_min + (lr - lr_min) * 0.5 * (1 + math.cos(math.pi * t))


def save_ckpt(path: Path, model, optimizer, step, epoch, best_val):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step":      step,
        "epoch":     epoch,
        "best_val":  best_val,
    }, path)
    print(f"  Saved: {path}")


def cleanup(ckpt_dir: Path, keep: int):
    ckpts = sorted(ckpt_dir.glob("step_*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))
    for old in ckpts[:-keep]:
        old.unlink()


def load_autoencoder(cfg: dict, device: torch.device) -> BeatmapAutoencoder:
    """Load the frozen Phase-2 autoencoder."""
    ae_cfg = AutoencoderConfig(
        x_channels      = cfg["ae_x_channels"],
        middle_channels  = cfg["ae_middle_channels"],
        z_channels       = cfg["ae_z_channels"],
        channel_mult     = cfg["ae_channel_mult"],
        num_res_blocks   = cfg["ae_num_res_blocks"],
        num_groups       = cfg["ae_num_groups"],
        kl_weight        = cfg["ae_kl_weight"],
    )
    ae = BeatmapAutoencoder(ae_cfg).to(device)
    ckpt_path = Path(cfg["ae_ckpt"])
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Autoencoder checkpoint not found: {ckpt_path}\n"
            f"Run Phase 2 training first."
        )
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    ae.load_state_dict(state, strict=True)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False
    print(f"Autoencoder loaded from {ckpt_path} [FROZEN]")
    return ae


# ────────────────────────────────────────────────────────────────────────────
# Validation
# ────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model:     TaikoDiffusionModel,
             ae:        BeatmapAutoencoder,
             scheduler: DDPMScheduler,
             loader:    DataLoader,
             device:    torch.device,
             cfg:       dict,
             max_batches: int) -> float:
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        tensor = batch["tensor"].to(device)
        mel    = batch["mel"].to(device)
        diff   = batch["difficulty"].to(device)
        style  = batch["style"].to(device)
        B = tensor.shape[0]

        # Encode
        with torch.no_grad():
            z_dist   = ae.encode(tensor)
            z_start  = z_dist.mode()

        t      = torch.randint(0, scheduler.timesteps, (B,), device=device)
        noise  = torch.randn_like(z_start)
        z_t    = scheduler.add_noise(z_start, noise, t)

        pred   = model(z_t, t, mel, diff, style)
        loss   = F.mse_loss(pred, noise)
        total += loss.item()
        n     += 1

    model.train()
    return total / max(n, 1)


# ────────────────────────────────────────────────────────────────────────────
# Training loop
# ────────────────────────────────────────────────────────────────────────────

def train(resume_path: str | None, config_path: str | None):
    cfg    = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM   : {vram:.1f} GB")
        if vram < 5:
            print("⚠  <5 GB VRAM — if OOM, set grad_checkpoint=true in configs/base.yaml")

    # ── Data ──────────────────────────────────────────────────────────────── #
    if not Path(cfg["cache_file"]).exists():
        print("ERROR: taiko_files_filtered.json not found.")
        return

    train_files, val_files = load_split(cfg["cache_file"], cfg["val_ratio"])
    print(f"Train : {len(train_files)} maps  |  Val : {len(val_files)} maps")

    mel_dir    = Path(cfg["mel_dir"])
    mel_frames = cfg["pad_frames"] // 2   # mel is at ~2× lower temporal resolution

    train_ds = DiffusionDataset(train_files, mel_dir, cfg["pad_frames"], mel_frames)
    val_ds   = DiffusionDataset(val_files,   mel_dir, cfg["pad_frames"], mel_frames)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], drop_last=True, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds,   batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=(device.type == "cuda")
    )

    # ── Frozen autoencoder ────────────────────────────────────────────────── #
    ae = load_autoencoder(cfg, device)

    # ── Diffusion model ───────────────────────────────────────────────────── #
    model = TaikoDiffusionModel(cfg).to(device)
    params = model.count_parameters()
    print(f"U-Net params   : {params['unet']:,}")
    print(f"Audio enc      : {params['audio_encoder']:,}")
    print(f"Total trainable: {params['total']:,}")

    # ── Noise scheduler ───────────────────────────────────────────────────── #
    scheduler = DDPMScheduler(
        timesteps = cfg["timesteps"],
        schedule  = cfg["beta_schedule"],
    )

    # ── Optimizer ─────────────────────────────────────────────────────────── #
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg["lr"],
        weight_decay = cfg["weight_decay"],
    )
    writer   = SummaryWriter(cfg["log_dir"])
    ckpt_dir = Path(cfg["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    step, start_epoch, best_val = 0, 0, float("inf")
    max_steps = cfg["max_steps"] or (cfg["max_epochs"] * len(train_loader))

    # ── Resume ────────────────────────────────────────────────────────────── #
    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step        = ckpt["step"]
        start_epoch = ckpt["epoch"]
        best_val    = ckpt["best_val"]
        print(f"Resumed from step {step} (val_loss={best_val:.5f})")

    # ── Training ─────────────────────────────────────────────────────────── #
    print(f"\nDiffusion training started")
    print(f"TensorBoard: tensorboard --logdir {cfg['log_dir']}\n")

    model.train()
    t0 = time.time()

    for epoch in range(start_epoch, cfg["max_epochs"]):
        for batch in train_loader:

            # LR warmup + cosine decay
            lr = get_lr(step, cfg["warmup_steps"], max_steps,
                        cfg["lr"], cfg["lr_min"])
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            tensor = batch["tensor"].to(device)
            mel    = batch["mel"].to(device)
            diff   = batch["difficulty"].to(device)
            style  = batch["style"].to(device)
            B      = tensor.shape[0]

            # 1. Encode beatmap → latent  (frozen AE, no grad)
            with torch.no_grad():
                z_dist  = ae.encode(tensor)
                z_start = z_dist.mode()                    # [B, 16, T_lat]

            # 2. Sample random timestep + noise
            t_idx  = torch.randint(0, scheduler.timesteps, (B,), device=device)
            noise  = torch.randn_like(z_start)

            # 3. Forward diffusion
            z_t = scheduler.add_noise(z_start, noise, t_idx)  # [B, 16, T_lat]

            # 4. Classifier-free guidance dropout
            #    Replace conditioning with null tokens randomly
            drop_mask  = torch.rand(B, device=device) < cfg["cfg_dropout"]
            null_diff  = torch.full_like(diff,  cfg["n_difficulties"])
            null_style = torch.full_like(style, cfg["n_styles"])
            diff_in  = torch.where(drop_mask, null_diff,  diff)
            style_in = torch.where(drop_mask, null_style, style)

            # 5. Predict noise
            optimizer.zero_grad()
            pred = model(z_t, t_idx, mel, diff_in, style_in)   # [B, 16, T_lat]

            # 6. MSE loss on predicted vs actual noise
            loss = F.mse_loss(pred, noise)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            step += 1

            # ── Logging ────────────────────────────────────────────────── #
            if step % cfg["log_every"] == 0:
                elapsed = time.time() - t0
                print(
                    f"epoch {epoch+1:3d} | step {step:6d} | "
                    f"loss {loss.item():.5f} | "
                    f"lr {lr:.2e} | {elapsed:.0f}s"
                )
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/lr",   lr,          step)
                if device.type == "cuda":
                    writer.add_scalar(
                        "train/vram_gb",
                        torch.cuda.memory_allocated() / 1e9, step
                    )

            # ── Validation ────────────────────────────────────────────── #
            if step % cfg["val_every"] == 0:
                val_loss = validate(
                    model, ae, scheduler, val_loader,
                    device, cfg, cfg["val_batches"]
                )
                writer.add_scalar("val/loss", val_loss, step)
                print(f"  → val loss: {val_loss:.5f}  (best: {best_val:.5f})")
                if val_loss < best_val:
                    best_val = val_loss
                    save_ckpt(
                        ckpt_dir / "best.pt",
                        model, optimizer, step, epoch, best_val
                    )

            # ── Checkpoint ────────────────────────────────────────────── #
            if step % cfg["save_every"] == 0:
                save_ckpt(
                    ckpt_dir / f"step_{step:07d}.pt",
                    model, optimizer, step, epoch, best_val
                )
                cleanup(ckpt_dir, cfg["keep_last_n"])

            if cfg["max_steps"] and step >= cfg["max_steps"]:
                print("Max steps reached.")
                writer.close()
                return

        print(f"--- End of epoch {epoch+1} ---\n")

    writer.close()
    print("Done.")


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3 — Diffusion U-Net training")
    parser.add_argument("--resume", default=None,
                        help="Path to a diffusion checkpoint to resume from")
    parser.add_argument("--config", default="configs/base.yaml",
                        help="Path to YAML config (default: configs/base.yaml)")
    args = parser.parse_args()
    train(args.resume, args.config)
