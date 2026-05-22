"""
test_tensor.py

Tests the beatmap <-> tensor round-trip conversion.
Run from project root:
    python test_tensor.py
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, ".")

from taiko.data.osu_parser import OsuTaikoParser
from taiko.data.tensor_repr import (
    beatmap_to_tensor, tensor_to_beatmap,
    round_trip_accuracy, FRAME_MS, N_CHANNELS
)


def separator(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def main():
    cache_path = Path("taiko_files_cache.json")
    if not cache_path.exists():
        print("ERROR: Run fast_scan.py first.")
        return

    taiko_files = [Path(p) for p in json.loads(cache_path.read_text())]
    print(f"Loaded {len(taiko_files)} taiko files from cache.")

    parser = OsuTaikoParser()

    # ------------------------------------------------------------------ #
    separator("STEP 1 — Single map round-trip")
    # ------------------------------------------------------------------ #

    bm = parser.parse_file(taiko_files[0])
    print(f"Map: {bm.title} [{bm.version}]")
    print(f"Notes: {bm.note_count} | NPS: {bm.notes_per_second:.1f} | Duration: {bm.duration_ms/1000:.1f}s")

    tensor = beatmap_to_tensor(bm)
    print(f"\nTensor shape: {tensor.shape}   (expected: [7, ~{int(bm.duration_ms/FRAME_MS)}])")
    print(f"Tensor size:  {tensor.nbytes/1024:.1f} KB")
    print(f"Channel sums: {tensor.sum(axis=1).tolist()}")
    print(f"  CH0 don:     {tensor[0].sum():.0f}")
    print(f"  CH1 kat:     {tensor[1].sum():.0f}")
    print(f"  CH2 big_don: {tensor[2].sum():.0f}")
    print(f"  CH3 big_kat: {tensor[3].sum():.0f}")
    print(f"  CH4 roll:    {tensor[4].sum():.0f}")
    print(f"  CH5 denden:  {tensor[5].sum():.0f}")
    print(f"  CH6 beat:    {tensor[6].sum():.0f}")

    # ------------------------------------------------------------------ #
    separator("STEP 2 — Round-trip accuracy")
    # ------------------------------------------------------------------ #

    metrics = round_trip_accuracy(bm)
    print(f"Original notes:      {metrics['original_notes']}")
    print(f"Reconstructed notes: {metrics['reconstructed_notes']}")
    print(f"Recovered:           {metrics['recovered']}")
    print(f"Recall:              {metrics['recall']:.1%}   (should be ~100%)")
    print(f"Precision:           {metrics['precision']:.1%}   (should be ~100%)")
    print(f"False positives:     {metrics['false_positives']}")

    if metrics['recall'] > 0.99 and metrics['precision'] > 0.99:
        print("\nRound-trip accuracy ✓")
    else:
        print("\nWARNING: Round-trip accuracy is low — check tensor repr logic")

    # ------------------------------------------------------------------ #
    separator("STEP 3 — Batch accuracy across 20 maps")
    # ------------------------------------------------------------------ #

    total_recall    = 0.0
    total_precision = 0.0
    n_tested        = 0
    errors          = 0

    for osu_file in taiko_files[:20]:
        try:
            bm = parser.parse_file(osu_file)
            if bm.note_count < 10:
                continue
            m = round_trip_accuracy(bm)
            total_recall    += m["recall"]
            total_precision += m["precision"]
            n_tested += 1
        except Exception as e:
            errors += 1

    avg_recall    = total_recall    / max(n_tested, 1)
    avg_precision = total_precision / max(n_tested, 1)

    print(f"Tested:        {n_tested} maps ({errors} errors)")
    print(f"Avg Recall:    {avg_recall:.1%}")
    print(f"Avg Precision: {avg_precision:.1%}")

    if avg_recall > 0.99 and avg_precision > 0.99:
        print("\nBatch round-trip ✓ — tensor repr is ready")
    else:
        print("\nWARNING: Some maps have low accuracy")

    # ------------------------------------------------------------------ #
    separator("STEP 4 — Tensor storage estimate")
    # ------------------------------------------------------------------ #

    sample_tensor = beatmap_to_tensor(parser.parse_file(taiko_files[0]))
    bytes_per_frame = N_CHANNELS * 4  # float32
    avg_frames = sample_tensor.shape[1]

    total_maps  = len(taiko_files)
    avg_size_kb = avg_frames * bytes_per_frame / 1024
    total_gb    = avg_size_kb * total_maps / 1024**2

    print(f"Sample tensor: {sample_tensor.shape} = {sample_tensor.nbytes/1024:.1f} KB")
    print(f"Estimated total for {total_maps} maps: {total_gb:.2f} GB")
    print(f"(Compare: mels were ~7MB each)")


if __name__ == "__main__":
    main()
