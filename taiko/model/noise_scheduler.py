"""
taiko/model/noise_scheduler.py

Diffusion schedule and sampling steps.

An nn.Module holding registered buffers, so `.to(device)` moves the schedule
along with the model. The previous version kept plain tensors and hand-rolled a
`.to()` plus per-call device moves, which is exactly the sort of thing that
works until the day it silently does not.

Parameterisation
----------------
v-prediction (Salimans and Ho, "Progressive Distillation") rather than
epsilon-prediction:

    v = sqrt(alpha_bar) * noise - sqrt(1 - alpha_bar) * x_0

Epsilon-prediction degenerates at high noise levels: as alpha_bar goes to zero
the input is nearly pure noise, and predicting the noise becomes trivial and
uninformative while the implied x_0 is wildly sensitive to small errors. That
matters more here than in image diffusion because chart latents are sparse --
most of the signal lives in a few frames, so a target that is unstable at high
noise wastes most of the schedule. v-prediction stays well-conditioned across
the whole range.

Cosine betas for the same reason: a linear schedule spends too much of its
length at noise levels where there is nothing left to learn.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def make_beta_schedule(
    schedule: str = "cosine",
    timesteps: int = 1000,
    linear_start: float = 1e-4,
    linear_end: float = 2e-2,
    cosine_s: float = 8e-3,
) -> np.ndarray:
    if schedule == "linear":
        return np.linspace(linear_start, linear_end, timesteps, dtype=np.float64)

    if schedule == "cosine":
        steps = timesteps + 1
        x = np.linspace(0, timesteps, steps, dtype=np.float64)
        alphas_cumprod = np.cos(((x / timesteps) + cosine_s) / (1 + cosine_s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0, 0.999)

    raise ValueError(f"unknown schedule {schedule!r}; use 'cosine' or 'linear'")


class NoiseScheduler(nn.Module):
    """
    DDPM forward process plus DDIM sampling.

    Args:
        prediction_type: "v" or "epsilon". "v" is the default and what the
            training loop assumes; "epsilon" exists for comparison runs.
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_schedule: str = "cosine",
        prediction_type: str = "v",
        linear_start: float = 1e-4,
        linear_end: float = 2e-2,
        cosine_s: float = 8e-3,
    ):
        super().__init__()
        if prediction_type not in ("v", "epsilon"):
            raise ValueError(f"unknown prediction_type {prediction_type!r}")

        self.timesteps = timesteps
        self.prediction_type = prediction_type
        self.beta_schedule = beta_schedule

        betas = make_beta_schedule(beta_schedule, timesteps, linear_start, linear_end, cosine_s)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        def buf(name: str, value: np.ndarray) -> None:
            self.register_buffer(name, torch.tensor(value, dtype=torch.float32))

        buf("betas", betas)
        buf("alphas_cumprod", alphas_cumprod)
        buf("alphas_cumprod_prev", alphas_cumprod_prev)
        buf("sqrt_alphas_cumprod", np.sqrt(alphas_cumprod))
        buf("sqrt_one_minus_alphas_cumprod", np.sqrt(1.0 - alphas_cumprod))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        buf("posterior_variance", posterior_variance)
        buf("posterior_log_variance_clipped", np.log(np.maximum(posterior_variance, 1e-20)))
        buf("posterior_mean_coef1", betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        buf("posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod))

    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract(buffer: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        out = buffer.gather(-1, t)
        return out.reshape(t.shape[0], *((1,) * (len(shape) - 1)))

    def add_noise(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward process q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start)
        a = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        b = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return a * x_start + b * noise

    def target_for(
        self,
        x_start: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """What the network is asked to predict, under the chosen parameterisation."""
        if self.prediction_type == "epsilon":
            return noise
        a = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        b = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return a * noise - b * x_start

    def to_eps_and_x0(
        self,
        model_out: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        clip: float | None = 4.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert a model prediction into (epsilon, x_0), whichever it produced.

        `clip` bounds x_0. Latents are calibrated to unit variance, so anything
        beyond a few standard deviations is a sampling excursion rather than a
        plausible chart, and letting it through destabilises the remaining
        steps.
        """
        a = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        b = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        if self.prediction_type == "epsilon":
            eps = model_out
            x0 = (x_t - b * eps) / a.clamp(min=1e-8)
        else:
            x0 = a * x_t - b * model_out
            eps = b * x_t + a * model_out

        if clip is not None:
            x0 = x0.clamp(-clip, clip)
            # Keep eps consistent with the clipped x_0, or the DDIM step below
            # mixes a clipped estimate with an unclipped one and drifts.
            eps = (x_t - a * x0) / b.clamp(min=1e-8)

        return eps, x0

    @torch.no_grad()
    def ddim_step(
        self,
        model_out: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        eta: float = 0.0,
        clip: float | None = 4.0,
    ) -> torch.Tensor:
        """One DDIM update from t to t_prev. eta=0 is deterministic."""
        eps, x0 = self.to_eps_and_x0(model_out, x_t, t, clip=clip)

        alpha_prev = self._extract(self.alphas_cumprod, t_prev, x_t.shape)
        alpha_t    = self._extract(self.alphas_cumprod, t,      x_t.shape)

        sigma = eta * torch.sqrt(
            ((1 - alpha_prev) / (1 - alpha_t).clamp(min=1e-8))
            * (1 - alpha_t / alpha_prev.clamp(min=1e-8))
        ).clamp(min=0.0)

        direction = (1 - alpha_prev - sigma ** 2).clamp(min=0.0).sqrt() * eps
        out = alpha_prev.sqrt() * x0 + direction

        if eta > 0:
            out = out + sigma * torch.randn_like(x_t)
        return out

    @torch.no_grad()
    def p_sample(
        self,
        model_out: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        clip: float | None = 4.0,
    ) -> torch.Tensor:
        """One ancestral DDPM step."""
        _, x0 = self.to_eps_and_x0(model_out, x_t, t, clip=clip)
        mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x0
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        log_var = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        nonzero = (t > 0).float().reshape(-1, *((1,) * (x_t.dim() - 1)))
        return mean + nonzero * (0.5 * log_var).exp() * torch.randn_like(x_t)

    def timestep_sequence(self, steps: int) -> list[int]:
        """
        Descending timesteps for DDIM, always ending at 0.

        Built with linspace rather than a fixed stride: a stride that does not
        divide `timesteps` leaves the final step short, which shows up as a
        faint high-frequency residue in the sample.
        """
        steps = max(1, min(steps, self.timesteps))
        seq = np.linspace(self.timesteps - 1, 0, steps).round().astype(int)
        return np.unique(seq)[::-1].tolist()

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        """Signal-to-noise ratio, for loss weighting."""
        a = self.alphas_cumprod.gather(-1, t)
        return a / (1 - a).clamp(min=1e-8)
