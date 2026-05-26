"""
scripts/test_reconstruction.py

Tests autoencoder reconstruction quality on real maps.
Shows note-level recall/precision and visual channel comparison.

Run from project root:
    python scripts/test_reconstruction.py
    python scripts/test_reconstruction.py --n-maps 20 --ckpt checkpoints/autoencoder/best.pt
"""

from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from taiko.data.osu_parser import OsuTaikoParser
from taiko.data.tensor_repr import (
    beatmap_to_tensor, tensor_to_beatmap,
    round_trip_accuracy, N_CHANNELS, FRAME_MS
)
from taiko.model.autoencoder import BeatmapAutoencoder, AutoencoderConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AE_CKPT    = "checkpoints/autoencoder/best.pt"
CACHE_FILE = "taiko_files_filtered.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, device: torch.device) -> BeatmapAutoencoder:
    config = AutoencoderConfig(
        x_channels      = 7,
        middle_channels  = 32,
        z_channels       = 16,
        channel_mult     = [1, 1, 2, 2, 4],
        num_res_blocks   = 2,
        num_groups       = 8,
        kl_weight        = 1e-6,
    )
    model = BeatmapAutoencoder(config).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded: {ckpt_path}  (step={ckpt.get('step', '?')}, val_loss={ckpt.get('best_val', '?')})")
    return model


def visualize_channels(original: np.ndarray,
                        reconstructed: np.ndarray,
                        title: str,
                        n_frames: int = 200):
    """Print a simple ASCII visualization of the first n_frames."""
    ch_names = ["don", "kat", "big_don", "big_kat", "roll", "denden", "beat"]
    print(f"\n  {title} — first {n_frames} frames (threshold 0.5)")
    print(f"  {'Channel':<10} {'Original':^{n_frames}} {'Recon':^{n_frames}}")
    print(f"  {'-'*10} {'-'*n_frames} {'-'*n_frames}")

    for i, name in enumerate(ch_names):
        orig  = original[i, :n_frames]
        recon = reconstructed[i, :n_frames]

        orig_str  = "".join("█" if v > 0.5 else "·" for v in orig)
        recon_str = "".join("█" if v > 0.5 else "·" for v in recon)

        match = sum(1 for a, b in zip(orig > 0.5, recon > 0.5) if a == b)
        acc   = match / n_frames * 100

        print(f"  {name:<10} {orig_str} {recon_str}  {acc:.0f}%")


def test_map(model: BeatmapAutoencoder,
             osu_path: Path,
             parser: OsuTaikoParser,
             device: torch.device,
             visualize: bool = True) -> dict | None:
    try:
        bm = parser.parse_file(osu_path)
        if bm.note_count < 20:
            return None

        tensor = beatmap_to_tensor(bm)   # [7, T_raw]
        T      = tensor.shape[1]

        # Pad to multiple of compression ratio
        ratio = model.compression_ratio
        T_pad = ((T + ratio - 1) // ratio) * ratio
        if T_pad > T:
            pad    = np.zeros((N_CHANNELS, T_pad - T), dtype=np.float32)
            tensor_padded = np.concatenate([tensor, pad], axis=1)
        else:
            tensor_padded = tensor

        x = torch.from_numpy(tensor_padded).unsqueeze(0).to(device)  # [1, 7, T_pad]

        with torch.no_grad():
            recon = model.reconstruct(x)   # [1, 7, T_pad]

        recon_np = recon[0].cpu().numpy()[:, :T]   # [7, T_raw]

        # Note-level accuracy
        acc = round_trip_accuracy(bm)

        # Channel-level accuracy (frame by frame, threshold 0.5)
        ch_names  = ["don", "kat", "big_don", "big_kat", "roll", "denden", "beat"]
        ch_accs   = {}
        for i, name in enumerate(ch_names):
            orig_bin  = (tensor[i]    > 0.5).astype(float)
            recon_bin = (recon_np[i]  > 0.5).astype(float)
            ch_accs[name] = float((orig_bin == recon_bin).mean())

        result = {
            "title":        bm.title,
            "version":      bm.version,
            "note_count":   bm.note_count,
            "duration_s":   bm.duration_ms / 1000,
            "recall":       acc["recall"],
            "precision":    acc["precision"],
            "false_pos":    acc["false_positives"],
            "ch_accs":      ch_accs,
        }

        if visualize:
            print(f"\n{'─'*80}")
            print(f"  {bm.title} [{bm.version}]")
            print(f"  Notes: {bm.note_count} | Duration: {bm.duration_ms/1000:.1f}s")
            print(f"  Recall:    {acc['recall']:.1%}")
            print(f"  Precision: {acc['precision']:.1%}")
            print(f"  False positives: {acc['false_positives']}")
            print(f"  Channel frame accuracy:")
            for name, a in ch_accs.items():
                bar = "█" * int(a * 20) + "·" * (20 - int(a * 20))
                print(f"    {name:<10} [{bar}] {a:.1%}")
            visualize_channels(tensor, recon_np, bm.title[:30])

        return result

    except Exception as e:
        print(f"  ERROR: {osu_path.name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(n_maps: int = 10, ckpt: str = AE_CKPT, visualize: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model  = load_model(ckpt, device)
    parser = OsuTaikoParser()

    if not Path(CACHE_FILE).exists():
        print(f"ERROR: {CACHE_FILE} not found")
        return

    files = [Path(p) for p in json.loads(Path(CACHE_FILE).read_text(encoding="utf-8"))]
    random.seed(42)
    sample = random.sample(files, min(n_maps, len(files)))

    print(f"\nTesting {len(sample)} maps...\n")

    results = []
    for path in sample:
        r = test_map(model, path, parser, device, visualize=visualize)
        if r:
            results.append(r)

    if not results:
        print("No valid results.")
        return

    # Summary
    print(f"\n{'='*80}")
    print(f"  SUMMARY — {len(results)} maps")
    print(f"{'='*80}")
    print(f"  Mean recall    : {np.mean([r['recall']    for r in results]):.1%}")
    print(f"  Mean precision : {np.mean([r['precision'] for r in results]):.1%}")
    print(f"  Mean false pos : {np.mean([r['false_pos'] for r in results]):.1f}")
    print(f"\n  Channel frame accuracy (mean):")
    ch_names = ["don", "kat", "big_don", "big_kat", "roll", "denden", "beat"]
    for name in ch_names:
        mean_acc = np.mean([r["ch_accs"][name] for r in results])
        bar = "█" * int(mean_acc * 20) + "·" * (20 - int(mean_acc * 20))
        print(f"    {name:<10} [{bar}] {mean_acc:.1%}")

    # Flag if quality is too low
    mean_recall = np.mean([r["recall"] for r in results])
    if mean_recall < 0.7:
        print(f"\n  ⚠ WARNING: Mean recall {mean_recall:.1%} is low.")
        print(f"    The autoencoder may not be trained well enough.")
        print(f"    This could explain diffusion instability.")
    else:
        print(f"\n  ✓ Autoencoder quality looks OK for diffusion training.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-maps",    type=int, default=10,      help="Number of maps to test")
    ap.add_argument("--ckpt",      default=AE_CKPT,           help="Autoencoder checkpoint")
    ap.add_argument("--no-visual", action="store_true",        help="Skip ASCII visualization")
    args = ap.parse_args()
    main(n_maps=args.n_maps, ckpt=args.ckpt, visualize=not args.no_visual)