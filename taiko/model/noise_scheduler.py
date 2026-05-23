"""
taiko/model/noise_scheduler.py

DDPM noise scheduler — directly from Mug-Diffusion's register_schedule().
Handles forward (add noise) and reverse (denoise) processes.

Fix: all _extract() calls now move buffers to the input tensor's device,
making it compatible with DataParallel where tensors may be on cuda:0 or cuda:1.
"""

from __future__ import annotations
import numpy as np
import torch


def make_beta_schedule(
    schedule: str = "linear",
    timesteps: int = 1000,
    linear_start: float = 1e-4,
    linear_end: float = 2e-2,
    cosine_s: float = 8e-3,
) -> np.ndarray:
    if schedule == "linear":
        return np.linspace(linear_start, linear_end, timesteps)
    elif schedule == "cosine":
        steps = timesteps + 1
        x     = np.linspace(0, timesteps, steps)
        alphas_cumprod = np.cos(((x / timesteps) + cosine_s) / (1 + cosine_s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0, 0.999)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


class NoiseScheduler:
    """
    DDPM noise scheduler.
    Precomputes all alpha/beta buffers at init (same as Mug-Diffusion).
    All operations automatically move buffers to the device of the input tensor.
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        linear_start: float = 1e-4,
        linear_end: float = 2e-2,
        cosine_s: float = 8e-3,
        device: torch.device = torch.device("cpu"),
    ):
        self.timesteps = timesteps
        betas = make_beta_schedule(
            beta_schedule, timesteps, linear_start, linear_end, cosine_s
        )
        alphas              = 1.0 - betas
        alphas_cumprod      = np.cumprod(alphas)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        def t(x): return torch.tensor(x, dtype=torch.float32, device=device)

        self.betas                         = t(betas)
        self.alphas_cumprod                = t(alphas_cumprod)
        self.alphas_cumprod_prev           = t(alphas_cumprod_prev)
        self.sqrt_alphas_cumprod           = t(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = t(np.sqrt(1.0 - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod     = t(np.sqrt(1.0 / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod   = t(np.sqrt(1.0 / alphas_cumprod - 1))

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = t(posterior_variance)
        self.posterior_log_variance_clipped = t(
            np.log(np.maximum(posterior_variance, 1e-20))
        )
        self.posterior_mean_coef1 = t(
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coef2 = t(
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

    def to(self, device: torch.device):
        for attr in vars(self):
            val = getattr(self, attr)
            if isinstance(val, torch.Tensor):
                setattr(self, attr, val.to(device))
        return self

    def _extract(self, a: torch.Tensor, t: torch.Tensor, shape) -> torch.Tensor:
        """Extract values from buffer, always on the same device as t."""
        out = a.to(t.device).gather(-1, t)
        return out.reshape(t.shape[0], *((1,) * (len(shape) - 1)))

    def add_noise(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward process: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_start)
        t = t.to(x_start.device)
        sqrt_alpha  = self._extract(self.sqrt_alphas_cumprod,            t, x_start.shape)
        sqrt_1alpha = self._extract(self.sqrt_one_minus_alphas_cumprod,  t, x_start.shape)
        return sqrt_alpha * x_start + sqrt_1alpha * noise

    @torch.no_grad()
    def p_sample(
        self,
        model_out: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """Single reverse step: p(x_{t-1} | x_t)"""
        t = t.to(x_t.device)
        sqrt_recip   = self._extract(self.sqrt_recip_alphas_cumprod,   t, x_t.shape)
        sqrt_recipm1 = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        x_recon = sqrt_recip * x_t - sqrt_recipm1 * model_out

        if clip_denoised:
            x_recon = x_recon.clamp(-3.0, 3.0)

        coef1 = self._extract(self.posterior_mean_coef1,           t, x_t.shape)
        coef2 = self._extract(self.posterior_mean_coef2,           t, x_t.shape)
        mean  = coef1 * x_recon + coef2 * x_t

        log_var = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        noise   = torch.randn_like(x_t)
        mask    = (t > 0).float().reshape(-1, *((1,) * (len(x_t.shape) - 1)))
        return mean + mask * (0.5 * log_var).exp() * noise

    @torch.no_grad()
    def ddim_sample(
        self,
        model_out: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """DDIM sampling step — faster inference (50 steps instead of 1000)."""
        t      = t.to(x_t.device)
        t_prev = t_prev.to(x_t.device)
        alpha_t      = self._extract(self.alphas_cumprod, t,      x_t.shape)
        alpha_t_prev = self._extract(self.alphas_cumprod, t_prev, x_t.shape)

        x_0_pred = (x_t - (1 - alpha_t).sqrt() * model_out) / alpha_t.sqrt()
        x_0_pred = x_0_pred.clamp(-3.0, 3.0)

        sigma = eta * ((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)).sqrt()
        noise = torch.randn_like(x_t) if eta > 0 else torch.zeros_like(x_t)

        return (
            alpha_t_prev.sqrt() * x_0_pred
            + (1 - alpha_t_prev - sigma ** 2).clamp(0).sqrt() * model_out
            + sigma * noise
        )
