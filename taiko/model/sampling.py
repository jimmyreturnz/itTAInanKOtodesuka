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

from typing import Sequence

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

    Sampled at half steps, so no frame weighs exactly 0. With linspace(0, 1)
    the song's first and last latent frame, which only one window covers,
    got 0 from it: the prediction there was 0/0 clamped to 0, so those
    frames were never denoised -- the stray notes in the first 320 ms. Two
    overlapping tapers still sum to exactly 1.
    """
    w = torch.ones(length, device=device, dtype=dtype)
    if ramp > 0:
        t = (torch.arange(ramp, device=device, dtype=dtype) + 0.5) / ramp
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
    window_nps: float | Sequence[float] | None = None,
    window_frames: int = 1536,
    overlap_frames: int = 768,
    ddim_steps: int = 50,
    cfg_scale: float | None = None,
    eta: float = 0.0,
    generator: torch.Generator | None = None,
    progress: bool = True,
    batch_windows: int = 16,
) -> torch.Tensor:
    """
    Generate the latent for a whole song.

    Args:
        mel:    [1, 128, T] at chart resolution
        timing: [1, 3, T] beat grid over the same frames
        avg_nps, peak_nps: normalised map-wide density. None means
                        "unspecified" on models built with features >= 2; on
                        generation-1 models it means zero, which is never what
                        you want -- generate.py fills it from the NPS prior.
        window_nps:     normalised density per window (a list as long as
                        `plan_windows` returns), one value for all windows, or
                        None to leave it to the model. Ignored by generation-1
                        models.
        window_frames:  must match training, and be a multiple of the
                        autoencoder compression
        overlap_frames: how much neighbouring windows share. Half a window is a
                        good default; less starts to show at boundaries.
        batch_windows:  how many windows share one U-Net forward. Windows and
                        both guidance branches are stacked into one batch, so a
                        five-minute song is ~50 forwards per chunk instead of
                        ~1,900 batch-1 calls. Lower it if a long song runs out
                        of memory on a small card.

    Returns:
        [1, z_channels, T // compression]
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    scheduler = model.scheduler
    compression = model.compression
    cfg_scale = model.cfg_scale if cfg_scale is None else cfg_scale
    features = getattr(model, "features", 1)

    if window_frames % compression != 0:
        raise ValueError(
            f"window_frames {window_frames} must be a multiple of the "
            f"{compression}x compression ratio"
        )

    mel = mel.to(device=device, dtype=dtype)
    timing = timing.to(device=device, dtype=dtype)

    total_frames = mel.shape[-1]
    windows = plan_windows(total_frames, window_frames, overlap_frames)
    n_win = len(windows)

    latent_total = total_frames // compression
    latent_window = window_frames // compression
    latent_overlap = overlap_frames // compression

    # --- conditioning, one row per window ------------------------------------ #
    def column(value, fill, dt=torch.float32):
        v = fill if value is None else value
        return torch.full((n_win,), float(v) if dt.is_floating_point else int(v),
                          device=device, dtype=dt)

    diff_t  = column(difficulty, 0.5)
    style_t = column(style, STYLE_NULL, torch.long)
    anps_t  = column(avg_nps, 0.0)
    pnps_t  = column(peak_nps, 0.0)

    if motif is None:
        motif_t = torch.zeros(n_win, MOTIF_DIM, device=device, dtype=dtype)
        mask_t  = torch.zeros(n_win, MOTIF_DIM, device=device, dtype=dtype)
    else:
        motif_t = torch.as_tensor(np.asarray(motif), device=device, dtype=dtype).reshape(1, -1)
        mask_t = (
            torch.ones_like(motif_t) if motif_mask is None
            else torch.as_tensor(np.asarray(motif_mask), device=device, dtype=dtype).reshape(1, -1)
        )
        motif_t, mask_t = motif_t.expand(n_win, -1), mask_t.expand(n_win, -1)

    density = {}
    if features >= 2:
        if window_nps is None:
            wnps_t = torch.zeros(n_win, device=device)
            wknown = torch.zeros(n_win, device=device)
        else:
            values = np.asarray(window_nps, dtype=np.float32).reshape(-1)
            if values.size not in (1, n_win):
                raise ValueError(f"window_nps has {values.size} values for {n_win} windows")
            values = np.broadcast_to(values, (n_win,))
            wnps_t = torch.as_tensor(values.copy(), device=device)
            wknown = torch.ones(n_win, device=device)
        mknown = torch.full((n_win,), 0.0 if avg_nps is None else 1.0, device=device)
        density = dict(window_nps=wnps_t, window_known=wknown, map_nps_known=mknown)

    cond_emb = model.unet_model.cond_emb(
        diff_t, style_t, anps_t, pnps_t, motif_t, mask_t, **density,
    )
    uncond_emb = model.unet_model.cond_emb.unconditional(n_win, device, cond_emb.dtype)

    # --- audio and timing features, per window, computed once ----------------- #
    # They are deterministic functions of the input, so they are computed once
    # instead of once per step -- 50x less encoding for a 50-step sample.
    # Windows are always full width: a song shorter than one window is padded,
    # since every window in a batch must share one shape.
    def window_slice(x: torch.Tensor, start: int, fill: float) -> torch.Tensor:
        piece = x[:, :, start:start + window_frames]
        if piece.shape[-1] < window_frames:
            piece = torch.nn.functional.pad(
                piece, (0, window_frames - piece.shape[-1]), value=fill)
        return piece

    # -1 is silence on the normalised mel scale; 0 would be mid-loudness.
    mel_w = torch.cat([window_slice(mel, s, -1.0) for s, _ in windows], dim=0)
    tim_w = torch.cat([window_slice(timing, s, 0.0) for s, _ in windows], dim=0)

    audio_levels: list[list[torch.Tensor]] = []
    timing_latents, timing_levels = [], []
    for lo in range(0, n_win, batch_windows):
        hi = min(lo + batch_windows, n_win)
        audio_levels.append(model.wave_model(mel_w[lo:hi]))
        timing_latents.append(model.downsample_timing(tim_w[lo:hi], latent_window))
        timing_levels.append(model.encode_timing(tim_w[lo:hi]) if features >= 2 else None)

    starts = [s // compression for s, _ in windows]

    blend = _blend_weights(latent_window, latent_overlap // 2, device, dtype)
    blend = blend.reshape(1, 1, -1)

    z = torch.randn(
        1, model.z_channels, latent_total, device=device, dtype=dtype, generator=generator,
    )

    sequence = scheduler.timestep_sequence(ddim_steps)
    if progress:
        print(
            f"  {n_win} windows x {len(sequence)} steps "
            f"({total_frames} frames, {latent_total} latent)"
        )

    guided = cfg_scale != 1.0

    for i, t_val in enumerate(sequence):
        t_prev_val = sequence[i + 1] if i + 1 < len(sequence) else 0
        t_prev = torch.full((1,), t_prev_val, device=device, dtype=torch.long)

        # Accumulate each window's prediction into a shared canvas, weighted by
        # the taper, then normalise. This is the MultiDiffusion step: the
        # windows never diverge because they never own separate latents.
        numerator = torch.zeros_like(z)
        denominator = torch.zeros(1, 1, latent_total, device=device, dtype=dtype)

        for chunk, lo in enumerate(range(0, n_win, batch_windows)):
            hi = min(lo + batch_windows, n_win)
            n = hi - lo

            z_rows = []
            for w in range(lo, hi):
                piece = z[:, :, starts[w]:starts[w] + latent_window]
                if piece.shape[-1] < latent_window:
                    piece = torch.nn.functional.pad(piece, (0, latent_window - piece.shape[-1]))
                z_rows.append(piece)
            z_batch = torch.cat(z_rows, dim=0)

            audio = audio_levels[chunk]
            tlat = timing_latents[chunk]
            tfeat = timing_levels[chunk]
            emb = cond_emb[lo:hi]

            if guided:
                # Conditional and unconditional in the same forward.
                z_batch = torch.cat([z_batch, z_batch], dim=0)
                audio = [torch.cat([a, a], dim=0) for a in audio]
                tlat = torch.cat([tlat, tlat], dim=0)
                if tfeat is not None:
                    tfeat = [torch.cat([f, f], dim=0) for f in tfeat]
                emb = torch.cat([emb, uncond_emb[lo:hi]], dim=0)

            t = torch.full((z_batch.shape[0],), t_val, device=device, dtype=torch.long)
            pred = model.unet_model(
                z_batch, t, audio, tlat,
                difficulty=diff_t[:1].expand(z_batch.shape[0]),
                style=style_t[:1].expand(z_batch.shape[0]),
                cond_emb=emb, timing_features=tfeat,
            )
            if guided:
                cond, uncond = pred[:n], pred[n:]
                pred = uncond + cfg_scale * (cond - uncond)

            for row, w in enumerate(range(lo, hi)):
                a = starts[w]
                b = min(a + latent_window, latent_total)
                span = b - a
                numerator[:, :, a:b] += (pred[row:row + 1] * blend)[:, :, :span]
                denominator[:, :, a:b] += blend[:, :, :span]

        prediction = numerator / denominator.clamp(min=1e-6)
        t = torch.full((1,), t_val, device=device, dtype=torch.long)
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
