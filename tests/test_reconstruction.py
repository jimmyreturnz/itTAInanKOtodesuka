"""
test_reconstruction.py

Tests TRAINED autoencoder reconstruction quality.
Run AFTER training with train_autoencoder.py.

Checks:
  - Note timing accuracy (are notes within 1 frame = 20ms?)
  - Note density preservation (same note count?)
  - Note type preservation (don stays don, kat stays kat?)
  - Roll preservation (start + end time within tolerance?)
  - Big note preservation (big_don / big_kat?)
  - Denden preservation

Usage:
    python test_reconstruction.py
    python test_reconstruction.py --checkpoint checkpoints/autoencoder/best.pt
    python test_reconstruction.py --n-maps 50
"""

import sys
import json
import math
import argparse
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, ".")

import numpy as np
import torch

from taiko.data.osu_parser import OsuTaikoParser
from taiko.data.tensor_repr import (
    beatmap_to_tensor, tensor_to_beatmap,
    N_CHANNELS, FRAME_MS, _find_onsets, _find_regions
)
from taiko.model.autoencoder import BeatmapAutoencoder, AutoencoderConfig


# ---------------------------------------------------------------------------
# Per-map metrics
# ---------------------------------------------------------------------------

@dataclass
class ReconMetrics:
    # Note counts
    orig_total:      int   = 0
    recon_total:     int   = 0

    # Per type
    orig_don:        int   = 0
    orig_kat:        int   = 0
    orig_big_don:    int   = 0
    orig_big_kat:    int   = 0
    orig_rolls:      int   = 0
    orig_dendens:    int   = 0

    recon_don:       int   = 0
    recon_kat:       int   = 0
    recon_big_don:   int   = 0
    recon_big_kat:   int   = 0
    recon_rolls:     int   = 0
    recon_dendens:   int   = 0

    # Timing accuracy
    timing_recall:   float = 0.0   # fraction of orig notes recovered within 1 frame
    timing_precision:float = 0.0   # fraction of recon notes that match an orig note
    false_positives: int   = 0

    # Roll accuracy
    roll_recall:     float = 0.0
    roll_timing_err: float = 0.0   # avg start time error in ms

    title: str = ""
    version: str = ""


def evaluate_map(
    model: BeatmapAutoencoder,
    bm,
    device: torch.device,
    threshold: float = 0.5,
) -> ReconMetrics:
    m = ReconMetrics(title=bm.title, version=bm.version)

    # Build tensor
    tensor = beatmap_to_tensor(bm)
    T      = tensor.shape[1]
    ratio  = model.compression_ratio
    T_pad  = math.ceil(T / ratio) * ratio

    pad = np.zeros((N_CHANNELS, T_pad - T), dtype=np.float32)
    tensor_padded = np.concatenate([tensor, pad], axis=1)

    x = torch.from_numpy(tensor_padded).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        recon = model.reconstruct(x)  # sigmoid applied, [0,1]

    recon_np = recon[0].cpu().numpy()[:, :T]  # strip padding, [7, T]

    # ---- Original note counts ------------------------------------------ #
    m.orig_total   = bm.note_count
    m.orig_don     = bm.don_count
    m.orig_kat     = bm.kat_count
    m.orig_big_don = sum(1 for n in bm.notes if n.note_type == "big_don")
    m.orig_big_kat = sum(1 for n in bm.notes if n.note_type == "big_kat")
    m.orig_rolls   = sum(1 for n in bm.notes if n.note_type == "roll")
    m.orig_dendens = sum(1 for n in bm.notes if n.note_type == "denden")

    # ---- Reconstructed note counts ------------------------------------- #
    from taiko.data.tensor_repr import CH_DON, CH_KAT, CH_BIG_DON, CH_BIG_KAT, CH_ROLL, CH_DENDEN

    don_onsets     = _find_onsets(recon_np[CH_DON],     threshold)
    kat_onsets     = _find_onsets(recon_np[CH_KAT],     threshold)
    big_don_onsets = _find_onsets(recon_np[CH_BIG_DON], threshold)
    big_kat_onsets = _find_onsets(recon_np[CH_BIG_KAT], threshold)
    roll_regions   = _find_regions(recon_np[CH_ROLL],   threshold)
    denden_regions = _find_regions(recon_np[CH_DENDEN], threshold)

    # Filter onsets that are part of rolls/dendens
    roll_mask   = recon_np[CH_ROLL]   > threshold
    denden_mask = recon_np[CH_DENDEN] > threshold

    don_onsets_clean = [f for f in don_onsets
                        if not roll_mask[f] and not denden_mask[f]]
    kat_onsets_clean = [f for f in kat_onsets
                        if not roll_mask[f] and not denden_mask[f]]

    m.recon_don     = len(don_onsets_clean)
    m.recon_kat     = len(kat_onsets_clean)
    m.recon_big_don = len(big_don_onsets)
    m.recon_big_kat = len(big_kat_onsets)
    m.recon_rolls   = len(roll_regions)
    m.recon_dendens = len(denden_regions)
    m.recon_total   = (m.recon_don + m.recon_kat +
                       m.recon_big_don + m.recon_big_kat +
                       m.recon_rolls + m.recon_dendens)

    # ---- Timing accuracy ----------------------------------------------- #
    # All original note onset times in frames
    orig_frames = set()
    for note in bm.notes:
        f = int(round(note.time / FRAME_MS))
        if f < T:
            orig_frames.add(f)

    # All reconstructed onset frames (all channels combined)
    recon_frames = set(
        don_onsets_clean + kat_onsets_clean +
        big_don_onsets + big_kat_onsets +
        [sf for sf, ef in roll_regions] +
        [sf for sf, ef in denden_regions]
    )

    # Recall: orig notes recovered within 1 frame tolerance
    tolerance = 1  # frames = 20ms
    recovered = sum(
        1 for f in orig_frames
        if any(abs(f - r) <= tolerance for r in recon_frames)
    )
    false_pos = sum(
        1 for r in recon_frames
        if not any(abs(r - f) <= tolerance for f in orig_frames)
    )

    m.timing_recall    = recovered / max(len(orig_frames), 1)
    m.timing_precision = (len(recon_frames) - false_pos) / max(len(recon_frames), 1)
    m.false_positives  = false_pos

    # ---- Roll timing accuracy ------------------------------------------ #
    if m.orig_rolls > 0 and m.recon_rolls > 0:
        orig_roll_starts  = sorted(
            int(round(n.time / FRAME_MS))
            for n in bm.notes if n.note_type == "roll"
        )
        recon_roll_starts = sorted(sf for sf, ef in roll_regions)

        matched_errors = []
        for orig_f in orig_roll_starts:
            if not recon_roll_starts:
                break
            closest = min(recon_roll_starts, key=lambda r: abs(r - orig_f))
            matched_errors.append(abs(orig_f - closest) * FRAME_MS)

        m.roll_recall     = min(len(recon_roll_starts), len(orig_roll_starts)) / len(orig_roll_starts)
        m.roll_timing_err = np.mean(matched_errors) if matched_errors else 0.0

    return m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/autoencoder/best.pt")
    parser.add_argument("--n-maps",     type=int, default=30)
    parser.add_argument("--threshold",  type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load model ----------------------------------------------------- #
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        print("Train first: python scripts/train_autoencoder.py")
        return

    ckpt   = torch.load(ckpt_path, map_location=device)
    config = AutoencoderConfig()
    model  = BeatmapAutoencoder(config).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint: step={ckpt['step']}  val_loss={ckpt['best_val']:.4f}")

    # ---- Load maps ------------------------------------------------------ #
    cache = Path("taiko_files_cache.json")
    if not cache.exists():
        print("ERROR: Run fast_scan.py first.")
        return

    import random
    files = [Path(p) for p in json.loads(cache.read_text())]
    random.seed(42)
    random.shuffle(files)
    test_files = files[:args.n_maps]

    osu_parser = OsuTaikoParser()
    all_metrics = []
    errors = 0

    print(f"\nEvaluating {args.n_maps} maps...\n")

    for i, f in enumerate(test_files):
        try:
            bm = osu_parser.parse_file(f)
            if bm.note_count < 20:
                continue
            m = evaluate_map(model, bm, device, args.threshold)
            all_metrics.append(m)

            # Print per-map summary
            status = "✓" if m.timing_recall > 0.95 else "✗"
            print(
                f"{status} [{i+1:2d}] {m.title[:30]:<30} [{m.version[:15]:<15}] "
                f"notes {m.orig_total:4d}→{m.recon_total:4d} | "
                f"recall {m.timing_recall:.0%} | "
                f"prec {m.timing_precision:.0%} | "
                f"FP {m.false_positives:3d}"
            )
        except Exception as e:
            errors += 1
            print(f"  ERR [{i+1}] {f.name}: {e}")

    if not all_metrics:
        print("No maps evaluated.")
        return

    # ---- Aggregate stats ----------------------------------------------- #
    def avg(lst): return sum(lst) / max(len(lst), 1)

    print(f"\n{'='*70}")
    print(f"  RECONSTRUCTION QUALITY SUMMARY ({len(all_metrics)} maps, {errors} errors)")
    print(f"{'='*70}")

    print(f"\n── Timing Accuracy ──────────────────────────────")
    print(f"  Avg Recall:     {avg([m.timing_recall    for m in all_metrics]):.1%}  (target: >95%)")
    print(f"  Avg Precision:  {avg([m.timing_precision for m in all_metrics]):.1%}  (target: >95%)")
    print(f"  Avg False Pos:  {avg([m.false_positives  for m in all_metrics]):.1f}")
    print(f"  Maps >95% recall: {sum(1 for m in all_metrics if m.timing_recall > 0.95)}/{len(all_metrics)}")

    print(f"\n── Note Density ─────────────────────────────────")
    density_ratios = [m.recon_total / max(m.orig_total, 1) for m in all_metrics]
    print(f"  Avg recon/orig ratio: {avg(density_ratios):.2f}  (target: ~1.0)")
    print(f"  Min: {min(density_ratios):.2f}  Max: {max(density_ratios):.2f}")

    print(f"\n── Note Type Preservation ───────────────────────")
    don_ratios = [m.recon_don / max(m.orig_don, 1) for m in all_metrics if m.orig_don > 0]
    kat_ratios = [m.recon_kat / max(m.orig_kat, 1) for m in all_metrics if m.orig_kat > 0]
    big_don_ratios = [m.recon_big_don / max(m.orig_big_don, 1) for m in all_metrics if m.orig_big_don > 0]
    big_kat_ratios = [m.recon_big_kat / max(m.orig_big_kat, 1) for m in all_metrics if m.orig_big_kat > 0]

    print(f"  Don ratio:     {avg(don_ratios):.2f}  (target: ~1.0)")
    print(f"  Kat ratio:     {avg(kat_ratios):.2f}  (target: ~1.0)")
    print(f"  Big don ratio: {avg(big_don_ratios):.2f}  (target: ~1.0)")
    print(f"  Big kat ratio: {avg(big_kat_ratios):.2f}  (target: ~1.0)")

    print(f"\n── Roll / Denden Preservation ───────────────────")
    maps_with_rolls = [m for m in all_metrics if m.orig_rolls > 0]
    maps_with_den   = [m for m in all_metrics if m.orig_dendens > 0]

    if maps_with_rolls:
        roll_counts = [m.recon_rolls / max(m.orig_rolls, 1) for m in maps_with_rolls]
        print(f"  Roll count ratio:   {avg(roll_counts):.2f}  (target: ~1.0)")
        print(f"  Roll timing err:    {avg([m.roll_timing_err for m in maps_with_rolls]):.1f}ms  (target: <20ms)")
        print(f"  Roll recall:        {avg([m.roll_recall for m in maps_with_rolls]):.1%}")
    else:
        print(f"  No rolls in test set")

    if maps_with_den:
        den_counts = [m.recon_dendens / max(m.orig_dendens, 1) for m in maps_with_den]
        print(f"  Denden count ratio: {avg(den_counts):.2f}  (target: ~1.0)")
    else:
        print(f"  No dendens in test set")

    # ---- Overall verdict ----------------------------------------------- #
    avg_recall    = avg([m.timing_recall    for m in all_metrics])
    avg_precision = avg([m.timing_precision for m in all_metrics])
    avg_density   = avg(density_ratios)

    print(f"\n── Verdict ──────────────────────────────────────")
    if avg_recall > 0.95 and avg_precision > 0.95 and 0.9 < avg_density < 1.1:
        print("  ✓ PASS — Autoencoder reconstruction is good enough for diffusion training")
    elif avg_recall > 0.90 and avg_precision > 0.85:
        print("  ~ MARGINAL — Consider more training epochs or lower threshold")
    else:
        print("  ✗ FAIL — Reconstruction quality too low for diffusion training")
        print("    Possible fixes:")
        print("    - Train for more epochs")
        print("    - Reduce kl_weight (e.g. 1e-7)")
        print("    - Increase middle_channels (e.g. 48)")
        print("    - Add attention in mid block")


if __name__ == "__main__":
    main()
