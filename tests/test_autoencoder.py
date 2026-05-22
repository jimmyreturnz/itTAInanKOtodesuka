"""
test_autoencoder.py

Sanity check for the beatmap autoencoder.
No training needed — just checks shapes, forward pass, and VRAM.

Run from project root:
    python test_autoencoder.py
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, ".")


def separator(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:  {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    from taiko.model.autoencoder import BeatmapAutoencoder, AutoencoderConfig
    from taiko.data.tensor_repr import N_CHANNELS

    B          = 4       # batch size
    PAD_FRAMES = 18_000  # ~6 min song @ 20ms

    # ------------------------------------------------------------------ #
    separator("Model structure")
    # ------------------------------------------------------------------ #

    config = AutoencoderConfig(
        x_channels=7,
        middle_channels=32,
        z_channels=16,
        channel_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        num_groups=8,
        kl_weight=1e-6,
    )
    model = BeatmapAutoencoder(config).to(device)
    params = model.count_parameters()

    print(f"Encoder:          {params['encoder']}")
    print(f"Decoder:          {params['decoder']}")
    print(f"Total:            {params['total']}")
    print(f"Compression:      {model.compression_ratio}x")
    print(f"Latent frames:    {PAD_FRAMES} → {PAD_FRAMES // model.compression_ratio}")
    print(f"Latent shape:     [B, {config.z_channels}, {PAD_FRAMES // model.compression_ratio}]")

    # ------------------------------------------------------------------ #
    separator("Forward pass — shapes")
    # ------------------------------------------------------------------ #

    x    = torch.randn(B, N_CHANNELS, PAD_FRAMES).to(device)
    mask = torch.ones(B, PAD_FRAMES).to(device)

    recon, posterior = model(x, sample_posterior=True)

    print(f"Input:      {tuple(x.shape)}")
    print(f"Recon:      {tuple(recon.shape)}  (should match input)")
    print(f"Latent mean:{tuple(posterior.mean.shape)}")
    print(f"Latent std: {tuple(posterior.std.shape)}")

    assert recon.shape == x.shape, "Reconstruction shape mismatch!"
    print("Shape check ✓")

    # ------------------------------------------------------------------ #
    separator("Loss computation")
    # ------------------------------------------------------------------ #

    loss, log_dict = model.training_loss(x, mask)
    print(f"Total loss:  {log_dict['total_loss']:.4f}")
    print(f"Recon loss:  {log_dict['recon_loss']:.4f}")
    print(f"KL loss:     {log_dict['kl_loss']:.6f}")
    print(f"Per-channel losses:")
    for k, v in log_dict.items():
        if k.startswith("loss_"):
            print(f"  {k}: {v:.4f}")
    assert loss.item() > 0
    print("Loss ✓")

    # ------------------------------------------------------------------ #
    separator("Backward pass")
    # ------------------------------------------------------------------ #

    loss.backward()
    grads_ok = all(
        p.grad is not None
        for p in model.parameters()
        if p.requires_grad
    )
    print(f"All gradients computed: {grads_ok}")
    assert grads_ok
    print("Backward ✓")

    # ------------------------------------------------------------------ #
    separator("VRAM usage")
    # ------------------------------------------------------------------ #

    if device.type == "cuda":
        alloc   = torch.cuda.memory_allocated(device)  / 1024**3
        reserved= torch.cuda.memory_reserved(device)   / 1024**3
        print(f"Allocated: {alloc:.2f} GB")
        print(f"Reserved:  {reserved:.2f} GB")

        if alloc < 3.0:
            print("VRAM usage OK for GTX 1650 ✓")
        else:
            print("WARNING: High VRAM — may OOM during training with this batch size")
            print("Try reducing BATCH_SIZE in train_autoencoder.py")

    # ------------------------------------------------------------------ #
    separator("Real beatmap round-trip (untrained)")
    # ------------------------------------------------------------------ #

    import json
    from taiko.data.osu_parser import OsuTaikoParser
    from taiko.data.tensor_repr import beatmap_to_tensor, tensor_to_beatmap, FRAME_MS

    cache = Path("taiko_files_cache.json")
    if not cache.exists():
        print("Cache not found — skipping real map test")
        return

    files  = [Path(p) for p in json.loads(cache.read_text())]
    parser = OsuTaikoParser()
    bm     = parser.parse_file(files[0])  # just test the first map in cache

    tensor = beatmap_to_tensor(bm)               # [7, T_raw]
    T      = tensor.shape[1]

    # Pad to multiple of compression_ratio
    ratio  = model.compression_ratio
    T_pad  = math.ceil(T / ratio) * ratio
    pad    = np.zeros((N_CHANNELS, T_pad - T), dtype=np.float32)
    tensor_padded = np.concatenate([tensor, pad], axis=1)

    x_single = torch.from_numpy(tensor_padded).unsqueeze(0).to(device)  # [1, 7, T_pad]

    model.eval()
    with torch.no_grad():
        recon_single = model.reconstruct(x_single)  # [1, 7, T_pad], sigmoid applied

    recon_np = recon_single[0].cpu().numpy()[:, :T]  # strip padding

    print(f"Map: {bm.title} [{bm.version}]")
    print(f"Original notes: {bm.note_count}")
    print(f"Input  channel sums: {tensor.sum(axis=1).round(1).tolist()}")
    print(f"Output channel sums: {recon_np.sum(axis=1).round(1).tolist()}")
    print(f"(Untrained model — values will be near 0.5, not meaningful yet)")
    print("\nRun train_autoencoder.py to train, then retest reconstruction quality.")

    separator("ALL CHECKS PASSED — Autoencoder ready for training")


if __name__ == "__main__":
    import math
    main()
