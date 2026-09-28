"""
tests/test_warm_start.py

A features-1 checkpoint widened into a features-2 model must compute exactly
what it computed before -- that is the whole premise of resuming the paused run
instead of starting over. Also covers the timing features themselves: the beat
grid has to survive the trip to latent resolution, which is the defect the
upgrade exists to fix.

    python tests/test_warm_start.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from taiko.data.conditioning import STYLE_NULL
from taiko.data.motif import MOTIF_DIM
from taiko.data.tensor_repr import timing_stream_from_bpm
from taiko.model.autoencoder import AutoencoderConfig, ChartAutoencoder
from taiko.model.diffusion import EMA, TaikoDiffusion
from taiko.model.timing_encoder import HARMONICS, timing_features
from taiko.train.warm_start import warm_start

W = 256


def _model(features: int, ae: ChartAutoencoder, seed: int) -> TaikoDiffusion:
    torch.manual_seed(seed)
    return TaikoDiffusion(autoencoder=ae, profile="tiny", features=features, verbose=False)


def _inputs(B: int = 2):
    g = torch.Generator().manual_seed(0)
    mel = torch.rand(B, 128, W, generator=g) * 2 - 1
    timing = torch.from_numpy(np.stack([timing_stream_from_bpm(bpm, 130, W)
                                        for bpm in (150.0, 200.0)][:B]))
    return mel, timing


def _unet_out(model: TaikoDiffusion, mel, timing, z, t):
    B = mel.shape[0]
    cond = dict(
        difficulty=torch.full((B,), 0.6), style=torch.full((B,), STYLE_NULL),
        avg_nps=torch.full((B,), 0.5), peak_nps=torch.full((B,), 0.4),
        motif=torch.zeros(B, MOTIF_DIM), motif_mask=torch.zeros(B, MOTIF_DIM),
    )
    extra = {}
    if model.features >= 2:
        extra = dict(timing_features=model.encode_timing(timing),
                     window_nps=torch.zeros(B), window_known=torch.zeros(B),
                     map_nps_known=torch.ones(B))
    audio = model.wave_model(mel)
    tlat = model.downsample_timing(timing, z.shape[-1])
    return model.unet_model(z, t, audio, tlat, **cond, **extra)


def test_widened_model_reproduces_the_old_one():
    torch.manual_seed(0)
    ae = ChartAutoencoder(AutoencoderConfig())
    old = _model(1, ae, seed=1).eval()

    # Pretend it trained: perturb every weight, give it Adam state and an EMA.
    with torch.no_grad():
        for p in old.trainable_parameters():
            p.add_(torch.randn_like(p) * 0.05)
    opt = torch.optim.AdamW(old.trainable_parameters(), lr=1e-4)
    for p in old.trainable_parameters():
        p.grad = torch.randn_like(p) * 1e-3
    opt.step()
    ema = EMA(old.trainable_parameters(), decay=0.999, warmup=0)
    ema.update(old.trainable_parameters())
    ckpt = {"unet": old.unet_model.state_dict(), "wave": old.wave_model.state_dict(),
            "optimizer": opt.state_dict(), "ema": ema.state_dict(), "step": 1234}

    new = _model(2, ae, seed=2).eval()
    new_opt = torch.optim.AdamW(new.trainable_parameters(), lr=1e-4)
    new_ema = EMA(new.trainable_parameters(), decay=0.5, warmup=0)
    report = warm_start(new, ckpt, optimizer=new_opt, ema=new_ema)

    assert report.widened, "nothing was widened; the upgrade did not happen"
    assert any(n.startswith("timing.") for n in report.fresh)

    mel, timing = _inputs()
    z = torch.randn(2, 16, W // ae.compression)
    t = torch.tensor([10, 700])
    with torch.no_grad():
        a = _unet_out(old, mel, timing, z, t)
        b = _unet_out(new, mel, timing, z, t)
    diff = (a - b).abs().max().item()
    assert diff < 1e-5, f"widened model differs from the original by {diff}"

    # Adam state: carried for unchanged tensors, dropped for widened ones.
    names = [n for n, _ in new.named_trainable_parameters()]
    state = new_opt.state_dict()["state"]
    for j, name in enumerate(names):
        if name in report.widened or name.startswith("timing."):
            assert j not in state, f"{name} kept moments that no longer fit"
    carried = len(state)
    assert carried == len(report.copied), (carried, len(report.copied))

    # EMA: old averages where they existed, aligned by name.
    old_names = [n for n, _ in old.named_trainable_parameters()]
    old_shadow = dict(zip(old_names, ckpt["ema"]["shadow"]))
    for name, shadow in zip(names, new_ema.shadow):
        if name in old_shadow and name not in report.widened:
            assert torch.equal(shadow, old_shadow[name].float()), name
    assert new_ema.step == ema.step
    print(f"  warm start reproduces the old model ok  (max diff {diff:.2e}, "
          f"{len(report.widened)} widened, {carried} Adam states carried)")


def test_timing_features_keep_the_beat_through_latent_resolution():
    """
    The pooled phasor loses the grid near 187.5 BPM. The harmonic features
    feed a learned strided stem instead; what matters for that stem is that
    the full-rate features distinguish a grid from the same grid shifted by
    half a beat -- which pooled sin/cos at 187.5 BPM cannot do at all.
    """
    bpm = 187.5
    a = torch.from_numpy(timing_stream_from_bpm(bpm, 0, 1536))[None]
    b = torch.from_numpy(timing_stream_from_bpm(bpm, 160, 1536))[None]  # half a beat

    pooled_gap = (F.adaptive_avg_pool1d(a[:, :2], 96)
                  - F.adaptive_avg_pool1d(b[:, :2], 96)).abs().max().item()
    fa, fb = timing_features(a), timing_features(b)
    full_gap = (fa[:, :2] - fb[:, :2]).abs().max().item()
    assert pooled_gap < 0.05, pooled_gap
    assert full_gap > 1.9, full_gap
    print(f"  half-beat shift at {bpm} BPM: pooled gap {pooled_gap:.3f}, "
          f"full-rate gap {full_gap:.2f} ok")


def test_timing_features_report_tempo_and_snaps():
    for bpm in (120.0, 180.0, 240.0):
        f = timing_features(torch.from_numpy(timing_stream_from_bpm(bpm, 0, 1000))[None])[0]
        log_tempo = f[2 * len(HARMONICS) + 1, 10:-10]
        got = 180.0 * 2 ** float(log_tempo.median())
        assert abs(got - bpm) / bpm < 0.02, (bpm, got)
        assert float(f[-1].min()) == 1.0            # grid present everywhere
    print("  tempo channel tracks BPM ok")


if __name__ == "__main__":
    print("warm start")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all warm start tests passed")
