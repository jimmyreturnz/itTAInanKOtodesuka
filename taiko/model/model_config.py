"""
taiko/model/model_config.py

Model size profiles.

What went wrong with the old profiles
-------------------------------------
`p2` built a ten-level audio encoder, but the U-Net read `audio_features[0..3]`.
Levels 4-9 received no gradient and contributed nothing -- and its four used
levels carried [128, 128, 128, 128] channels against `p1`'s
[128, 128, 256, 512]. It cost strictly more and delivered strictly less, and it
was the default.

The shape of that bug is now impossible: the audio encoder takes `n_levels`
from the U-Net's depth, so it cannot build a level nothing consumes.
`validate()` re-checks the invariant at construction time anyway, because
config that can drift silently eventually does.

Sizing for two T4s
------------------
A Kaggle T4 has about 15 GB. The numbers below are for a 1536-frame window
(30.7 s) at 16x compression, batch 2 per GPU, fp16 with gradient checkpointing.
Measure before trusting them -- they move with window size.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiffusionProfile:
    name:                str
    audio_base_channels: int
    audio_channel_mult:  tuple[int, ...]
    unet_base_channels:  int
    unet_channel_mult:   tuple[int, ...]
    unet_num_res_blocks: int
    use_s4:              bool
    use_checkpoint:      bool
    per_gpu_batch:       int
    lr:                  float = 1e-4
    note:                str = ""

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        n_levels = len(self.unet_channel_mult)
        if len(self.audio_channel_mult) != n_levels:
            raise ValueError(
                f"profile {self.name!r}: the audio encoder has "
                f"{len(self.audio_channel_mult)} levels but the U-Net has "
                f"{n_levels}. Every U-Net level consumes exactly one audio "
                f"feature map; any extra encoder level would receive no "
                f"gradient and burn FLOPs for nothing (this is the p2 bug)."
            )
        if n_levels < 1:
            raise ValueError(f"profile {self.name!r}: needs at least one level")

    @property
    def audio_out_channels(self) -> list[int]:
        return [self.audio_base_channels * m for m in self.audio_channel_mult]

    @property
    def n_levels(self) -> int:
        return len(self.unet_channel_mult)

    def summary(self) -> str:
        return (
            f"{self.name}: {self.n_levels} levels | "
            f"audio {self.audio_out_channels} | "
            f"unet base {self.unet_base_channels} x {list(self.unet_channel_mult)} | "
            f"s4={self.use_s4} ckpt={self.use_checkpoint} | "
            f"batch/gpu {self.per_gpu_batch} | lr {self.lr:g}"
        )


PROFILES: dict[str, DiffusionProfile] = {

    # Fast enough to iterate on a laptop or a short Kaggle session. Use this to
    # prove the pipeline end to end before spending real quota.
    "tiny": DiffusionProfile(
        name                = "tiny",
        audio_base_channels = 32,
        audio_channel_mult  = (1, 1, 2, 2),
        unet_base_channels  = 32,
        unet_channel_mult   = (1, 2, 2, 4),
        unet_num_res_blocks = 1,
        use_s4              = True,
        use_checkpoint      = False,
        per_gpu_batch       = 8,
        lr                  = 2e-4,
        note                = "smoke tests and pipeline shakedown",
    ),

    # The recommended target. Channels widen towards the coarse levels, where
    # long-range structure lives and the sequences are short.
    "p1": DiffusionProfile(
        name                = "p1",
        audio_base_channels = 128,
        audio_channel_mult  = (1, 1, 2, 4),
        unet_base_channels  = 128,
        unet_channel_mult   = (1, 2, 3, 4),
        unet_num_res_blocks = 2,
        use_s4              = True,
        use_checkpoint      = True,
        per_gpu_batch       = 2,
        lr                  = 1e-4,
        note                = "recommended: ~9 GB per T4 at 1536 frames",
    ),

    # Wider, for a second run once p1 has cleared Gate B. Deeper is not the
    # lever here -- the U-Net already sees the whole window through the
    # coarse-level attention -- so this widens instead.
    "p2": DiffusionProfile(
        name                = "p2",
        audio_base_channels = 128,
        audio_channel_mult  = (1, 2, 4, 4),
        unet_base_channels  = 160,
        unet_channel_mult   = (1, 2, 3, 4),
        unet_num_res_blocks = 2,
        use_s4              = True,
        use_checkpoint      = True,
        per_gpu_batch       = 1,
        lr                  = 8e-5,
        note                = "wider; tight on 15 GB, needs checkpointing",
    ),
}


def get_profile(name: str) -> DiffusionProfile:
    if name not in PROFILES:
        raise KeyError(f"unknown profile {name!r}. Available: {sorted(PROFILES)}")
    return PROFILES[name]


def print_profiles() -> None:
    for profile in PROFILES.values():
        print(f"  {profile.summary()}")
        if profile.note:
            print(f"      {profile.note}")
