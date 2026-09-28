"""
scripts/fit_nps_prior.py

Fit the star-rating -> note-density table that generation uses when
--avg-nps / --peak-nps are not given, and write it as JSON.

Checkpoints trained after this change carry the table themselves. This script
is for the ones trained before it:

    python scripts/fit_nps_prior.py --shards data/processed/shards \
        --out checkpoints/diffusion/nps_prior.json

generate.py looks for nps_prior.json next to the diffusion checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from taiko.data.nps_prior import describe, fit_nps_prior, save_prior
from taiko.data.preprocessed_dataset import split_indices
from taiko.data.shards import ShardReader


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shards", type=Path, default=Path("data/processed/shards"))
    ap.add_argument("--out", type=Path, default=Path("checkpoints/diffusion/nps_prior.json"))
    ap.add_argument("--ranked-only", action="store_true")
    args = ap.parse_args()

    reader = ShardReader(args.shards, mel_io="read")
    # The same split training uses, so the table describes what the model saw.
    train_idx, _ = split_indices(reader, val_ratio=0.05, ranked_only=args.ranked_only)
    prior = fit_nps_prior(reader.records[i] for i in train_idx)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_prior(prior, args.out)
    print(describe(prior))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
