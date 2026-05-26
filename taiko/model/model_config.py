"""
Shared model / training profiles (legacy vs Mug-inspired P1).
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class DiffusionProfile:
    name: str
    audio_base_channels: int
    audio_channel_mult: tuple[int, ...]
    unet_base_channels: int
    unet_channel_mult: tuple[int, ...]
    unet_num_res_blocks: int
    use_s4: bool
    use_checkpoint: bool
    per_gpu_batch: int
    lr: float = 1e-5
    cfg_dropout: float = 0.5


PROFILES: dict[str, DiffusionProfile] = {
    "legacy": DiffusionProfile(
        name="legacy",
        audio_base_channels=64,
        audio_channel_mult=(1, 1, 2, 2),
        unet_base_channels=64,
        unet_channel_mult=(1, 2, 4),
        unet_num_res_blocks=2,
        use_s4=False,
        use_checkpoint=True,
        per_gpu_batch=2,
    ),
    "p1": DiffusionProfile(
        name="p1",
        audio_base_channels=128,
        audio_channel_mult=(1, 1, 2, 4),
        unet_base_channels=128,
        unet_channel_mult=(1, 2, 3, 4),
        unet_num_res_blocks=2,
        use_s4=True,
        use_checkpoint=True,
        per_gpu_batch=2,
    ),
}


def get_profile(name: str) -> DiffusionProfile:
    if name not in PROFILES:
        raise KeyError(f"Unknown profile {name!r}. Choose from: {list(PROFILES)}")
    return PROFILES[name]
