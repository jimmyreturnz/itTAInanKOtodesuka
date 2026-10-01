"""Run: python tests/test_sampling.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from taiko.model.sampling import _blend_weights, plan_windows


def test_no_latent_frame_of_the_song_is_left_unweighted():
    # The song's first and last frame are covered by one window only. A zero
    # weight there left them un-denoised: notes in the first 320 ms.
    window, overlap, total = 96, 48, 1000
    w = _blend_weights(window, overlap // 2, "cpu", torch.float32)
    cover = torch.zeros(total)
    for a, b in plan_windows(total, window, overlap):
        cover[a:b] += w[:b - a]
    assert cover.min() > 0, cover[:3]


def test_overlapping_tapers_still_sum_to_one():
    ramp = 24
    w = _blend_weights(2 * ramp, ramp, "cpu", torch.float64)
    # Window A's falling ramp over window B's rising one.
    torch.testing.assert_close(w[-ramp:] + w[:ramp], torch.ones(ramp, dtype=torch.float64))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name}  ok")
    print("all sampling tests passed")
