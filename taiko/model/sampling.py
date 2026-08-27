"""
taiko/model/sampling.py

Whole-song generation by tiling overlapping windows.

The model trains on 1536-frame windows (30.7 s). A three-minute song is 9000
frames -- six times longer than anything it has seen, through an audio encoder
containing global self-attention whose cost is quadratic and whose statistics
shift completely at that length. Generating a song in one pass is both an OOM
risk and a distribution shift.

The fix is MultiDiffusion (Bar-Tal et al.): denoise every window in parallel and
average their predictions in the overlaps *at each step*, rather than generating
windows independently and stitching afterwards. The distinction matters. Joining
finished windows leaves a seam at every boundary -- two independent samples
disagree about what the music was doing, and no amount of crossfading hides a
drumroll that starts in one window and not the other. Averaging inside the loop
means the windows are denoising one shared latent, so they agree by
construction.

Cost is proportional to coverage: with 50% overlap a song takes twice the
compute of tiling with none, which is a small price for seamlessness.
"""

from __future__ import annotations

import numpy as np
import torch

from taiko.data.conditioning import STYLE_NULL
from taiko.data.motif import MOTIF_DIM


def plan_windows(total: int, window: int, overlap: int) -> list[tuple[int, int]]:
    """
    Cover [0, total) with windows of `window` frames overlapping by `overlap`.

    The last window is pulled back to end exactly at `total` rather than
    running past it, so the end of a song gets the same treatment as the
    middle. A song shorter than one window yields a single window.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if overlap >= window:
        raise ValueError(f"overlap {overlap} must be smaller than window {window}")

    if total <= window:
        return [(0, total)]

    stride = window - overlap
    starts = list(range(0, total - window + 1, stride))
    if starts[-1] + window < total:
        starts.append(total - window)

    return [(s, s + window) for s in starts]


def _blend_weights(length: int, ramp: int, device, dtype) -> torch.Tensor:
    """
    Raised-cosine taper at both ends of a window.

    A rectangular weight would make each frame's contribution jump as windows
    hand over. Tapering means a frame near a boundary is a smooth mixture of
    both neighbours' opinions, which is what makes the overlap invisible rather
    than merely blurry.
    """
    w = torch.ones(length, device=device, dtype=dtype)
    if ramp > 0:
        t = torch.linspace(0, 1, ramp, device=device, dtype=dtype)
        taper = 0.5 * (1 - torch.cos(torch.pi * t))
        w[:ramp] = taper
        w[-ramp:] = taper.flip(0)
    return w


@torch.no_grad()
def generate_song_latent(
    model,
    mel: torch.Tensor,
    timing: torch.Tensor,
    difficulty: float,
    style: int = STYLE_NULL,
    avg_nps: float | None = None,
    peak_nps: float | None = None,
    motif: torch.Tensor | np.ndarray | None = None,
    motif_mask: torch.Tensor | np.ndarray | None = None,
    window_frames: int = 1536,
    overlap_frames: int = 768,
    ddim_steps: int = 50,
    cfg_scale: float | None = None,
    eta: float = 0.0,
    generator: torch.Generator | None = None,
    progress: bool = True,
) -> torch.Tensor:
    """
    Generate the latent for a whole song.

    Args:
        mel:    [1, 128, T] at chart resolution
        timing: [1, 3, T] beat grid over the same frames
        window_frames:  must match training, and be a multiple of the
                        autoencoder compression
        overlap_frames: how much neighbouring windows share. Half a window is a
                        good default; less starts to show at boundaries.

    Returns:
        [1, z_channels, T // compression]
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    scheduler = model.scheduler
    compression = model.compression
    cfg_scale = model.cfg_scale if cfg_scale is None else cfg_scale

    if window_frames % compression != 0:
        raise ValueError(
            f"window_frames {window_frames} must be a multiple of the "
            f"{compression}x compression ratio"
        )

    mel = mel.to(device=device, dtype=dtype)
    timing = timing.to(device=device, dtype=dtype)

    total_frames = mel.shape[-1]
    windows = plan_windows(total_frames, window_frames, overlap_frames)

    latent_total = total_frames // compression
    latent_window = window_frames // compression
    latent_overlap = overlap_frames // compression

    # --- per-window conditioning, built once --------------------------------- #
    def as_batch(value, fill, dt=torch.float32):
        if value is None:
            return torch.full((1,), fill, device=device, dtype=dt)
        return torch.full((1,), float(value), device=device, dtype=dt) if dt.is_floating_point \
            else torch.full((1,), int(value), device=device, dtype=dt)

    diff_t  = as_batch(difficulty, 0.5)
    style_t = as_batch(style, STYLE_NULL, torch.long)
    anps_t  = as_batch(avg_nps, 0.0)
    pnps_t  = as_batch(peak_nps, 0.0)

    if motif is None:
        motif_t = torch.zeros(1, MOTIF_DIM, device=device, dtype=dtype)
        mask_t  = torch.zeros(1, MOTIF_DIM, device=device, dtype=dtype)
    else:
        motif_t = torch.as_tensor(np.asarray(motif), device=device, dtype=dtype).reshape(1, -1)
        mask_t = (
            torch.ones_like(motif_t) if motif_mask is None
            else torch.as_tensor(np.asarray(motif_mask), device=device, dtype=dtype).reshape(1, -1)
        )

    cond_emb = model.unet_model.cond_emb(diff_t, style_t, anps_t, pnps_t, motif_t, mask_t)
    uncond_emb = model.unet_model.cond_emb.unconditional(1, device, cond_emb.dtype)

    # Audio features and timing are deterministic functions of the input, so
    # they are computed once per window instead of once per window per step --
    # 50x less audio encoding for a 50-step sample.
    cached = []
    for start, end in windows:
        window_mel = mel[:, :, start:end]
        features = model.wave_model(window_mel)
        window_timing = model.downsample_timing(timing[:, :, start:end], latent_window)
        cached.append((start // compression, features, window_timing))

    blend = _blend_weights(latent_window, latent_overlap // 2, device, dtype)
    blend = blend.reshape(1, 1, -1)

    z = torch.randn(
        1, model.z_channels, latent_total, device=device, dtype=dtype, generator=generator,
    )

    sequence = scheduler.timestep_sequence(ddim_steps)
    if progress:
        print(
            f"  {len(windows)} windows x {len(sequence)} steps "
            f"({total_frames} frames, {latent_total} latent)"
        )

    for i, t_val in enumerate(sequence):
        t_prev_val = sequence[i + 1] if i + 1 < len(sequence) else 0
        t = torch.full((1,), t_val, device=device, dtype=torch.long)
        t_prev = torch.full((1,), t_prev_val, device=device, dtype=torch.long)

        # Accumulate each window's prediction into a shared canvas, weighted by
        # the taper, then normalise. This is the MultiDiffusion step: the
        # windows never diverge because they never own separate latents.
        numerator = torch.zeros_like(z)
        denominator = torch.zeros(1, 1, latent_total, device=device, dtype=dtype)

        for latent_start, features, window_timing in cached:
            lo = latent_start
            hi = min(lo + latent_window, latent_total)
            span = hi - lo
            z_window = z[:, :, lo:hi]

            if span < latent_window:
                z_window = torch.nn.functional.pad(z_window, (0, latent_window - span))

            pred = model.unet_model(
                z_window, t, features, window_timing,
                difficulty=diff_t, style=style_t, cond_emb=cond_emb,
            )
            if cfg_scale != 1.0:
                uncond = model.unet_model(
                    z_window, t, features, window_timing,
                    difficulty=diff_t, style=style_t, cond_emb=uncond_emb,
                )
                pred = uncond + cfg_scale * (pred - uncond)

            numerator[:, :, lo:hi] += (pred * blend)[:, :, :span]
            denominator[:, :, lo:hi] += blend[:, :, :span]

        prediction = numerator / denominator.clamp(min=1e-6)
        z = scheduler.ddim_step(prediction, z, t, t_prev, eta=eta)

        if progress and (i % 10 == 0 or i == len(sequence) - 1):
            print(f"    step {i + 1}/{len(sequence)}")

    return z


@torch.no_grad()
def generate_song(
    model,
    mel: torch.Tensor,
    timing: torch.Tensor,
    threshold: float = 0.5,
    **kwargs,
) -> torch.Tensor:
    """
    Generate a whole song and decode it to chart probabilities [1, 6, T].

    Decoding is done in overlapping chunks for the same reason generation is:
    the decoder is convolutional and would happily take the whole song, but
    chunking keeps peak memory flat for arbitrarily long audio.
    """
    z = generate_song_latent(model, mel, timing, **kwargs)
    target_length = mel.shape[-1]

    compression = model.compression
    chunk_latent = kwargs.get("window_frames", 1536) // compression
    overlap_latent = chunk_latent // 4

    total_latent = z.shape[-1]
    out = torch.zeros(1, 6, target_length, device=z.device, dtype=torch.float32)
    weight = torch.zeros(1, 1, target_length, device=z.device, dtype=torch.float32)

    for lo, hi in plan_windows(total_latent, min(chunk_latent, total_latent), overlap_latent):
        chunk = model.decode(z[:, :, lo:hi])
        frame_lo = lo * compression
        frame_hi = min(frame_lo + chunk.shape[-1], target_length)
        span = frame_hi - frame_lo
        out[:, :, frame_lo:frame_hi] += chunk[:, :, :span].float()
        weight[:, :, frame_lo:frame_hi] += 1.0

    return out / weight.clamp(min=1.0)
