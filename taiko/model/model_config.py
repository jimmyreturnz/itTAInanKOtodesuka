"""
taiko/model/model_config.py

Shared model / training profiles.

Profiles:
  legacy  — original small model, no S4, 4-level audio encoder
  p1      — Mug-inspired medium model, S4, 4-level audio encoder @ 128ch
  p2      — Mug-equivalent large model, S4, 10-level audio encoder @ 128ch
             matches Mug-Diffusion's MelspectrogramScaleEncoder1D channel_mult

p1 vs p2:
  Audio encoder:
    p1: base=128, mult=(1,1,2,4)        → 4 levels  → out=[128,128,256,512]
    p2: base=128, mult=(1,1,1,1,2,2,2,4,4,4) → 10 levels → out=[128,128,128,128,256,256,256,512,512,512]

  U-Net:
    p1: base=128, mult=(1,2,3,4)  → 4 levels, audio_channels=[128,128,256,512]
    p2: base=128, mult=(1,2,3,4)  → 4 levels, audio_channels=[128,128,128,128]
        (U-Net only uses first 4 audio levels; deeper encoder levels captured
         in the final feature map via downsampling within the encoder itself)

  Memory (T4 14.6GB, batch=2 per GPU):
    p1: ~9GB  — fits comfortably
    p2: ~13GB — tight, requires use_checkpoint=True

Conditioning (all profiles):
  difficulty  ~19% of emb_dim  (float, normalized 0-10)
  nps         ~25% of emb_dim  (float, normalized 0-20)
  style       ~19% of emb_dim  (int embedding, 4 classes)
  motif       ~37% of emb_dim  (float vector, 16 dims)
"""

from __future__ import annotations
from dataclasses import dataclass, field


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
    lr:                  float = 1e-5
    cfg_dropout:         float = 0.5
    use_motif:           bool  = True
    use_nps:             bool  = True

    @property
    def audio_out_channels(self) -> list[int]:
        """Output channel count at each audio encoder level."""
        return [self.audio_base_channels * m for m in self.audio_channel_mult]

    @property
    def unet_audio_channels(self) -> list[int]:
        """
        Audio channels fed into U-Net at each U-Net level.
        U-Net level i uses audio_out_channels[i] (clamped to available levels).
        """
        n = len(self.unet_channel_mult)
        outs = self.audio_out_channels
        return [outs[min(i, len(outs) - 1)] for i in range(n)]


PROFILES: dict[str, DiffusionProfile] = {

    "legacy": DiffusionProfile(
        name                = "legacy",
        audio_base_channels = 64,
        audio_channel_mult  = (1, 1, 2, 2),
        unet_base_channels  = 64,
        unet_channel_mult   = (1, 2, 4),
        unet_num_res_blocks = 2,
        use_s4              = False,
        use_checkpoint      = True,
        per_gpu_batch       = 2,
        use_motif           = False,
        use_nps             = False,
    ),

    "p1": DiffusionProfile(
        name                = "p1",
        # Audio: 4 levels → out=[128, 128, 256, 512]
        audio_base_channels = 128,
        audio_channel_mult  = (1, 1, 2, 4),
        # U-Net: 4 levels, audio_channels=[128, 128, 256, 512]
        unet_base_channels  = 128,
        unet_channel_mult   = (1, 2, 3, 4),
        unet_num_res_blocks = 2,
        use_s4              = True,
        use_checkpoint      = True,
        per_gpu_batch       = 2,
        use_motif           = True,
        use_nps             = True,
    ),

    "p2": DiffusionProfile(
        name                = "p2",
        # Audio: 10 levels (Mug-equivalent) → out=[128,128,128,128,256,256,256,512,512,512]
        audio_base_channels = 128,
        audio_channel_mult  = (1, 1, 1, 1, 2, 2, 2, 4, 4, 4),
        # U-Net: still 4 levels, but audio_channels=[128,128,128,128]
        # (deeper encoder features are aggregated within the encoder itself)
        unet_base_channels  = 128,
        unet_channel_mult   = (1, 2, 3, 4),
        unet_num_res_blocks = 2,
        use_s4              = True,
        use_checkpoint      = True,
        per_gpu_batch       = 2,
        use_motif           = True,
        use_nps             = True,
    ),

}


def get_profile(name: str) -> DiffusionProfile:
    if name not in PROFILES:
        raise KeyError(f"Unknown profile {name!r}. Available: {list(PROFILES)}")
    return PROFILES[name]


def print_profile(p: DiffusionProfile):
    """Print a human-readable summary of a profile."""
    print(f"\nProfile: {p.name}")
    print(f"  Audio encoder : base={p.audio_base_channels}, mult={p.audio_channel_mult}")
    print(f"  Audio out ch  : {p.audio_out_channels}")
    print(f"  U-Net audio ch: {p.unet_audio_channels}")
    print(f"  U-Net         : base={p.unet_base_channels}, mult={p.unet_channel_mult}")
    print(f"  S4            : {p.use_s4}")
    print(f"  Checkpoint    : {p.use_checkpoint}")
    print(f"  Motif         : {p.use_motif}")
    print(f"  NPS           : {p.use_nps}")
    print(f"  Batch/GPU     : {p.per_gpu_batch}")
    print(f"  LR            : {p.lr}")