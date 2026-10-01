"""
scripts/select_checkpoint.py

Pick the checkpoint to keep from evaluate.py results, by chart quality rather
than val MSE. best.pt is chosen by diffusion val loss, which is averaged over
noise levels and barely tracks the chart: it sat at step 58968 while training
ran on to 62400.

A candidate has to pass three gates:

  silence     notes on silence at most SILENCE_X x the ranked maps'
  too fast    raw too-fast pairs (before repair) at most TOO_FAST_X x ranked
  no band     no SR band's onset F1 more than BAND_F1_DROP below the incumbent's
  worse

and the survivors are ranked by exact snap. Each JSON must come from the same
pool (--per-band, --seeds), or the numbers are not comparable; that is checked.

    python scripts/evaluate.py --diffusion ckpt/step_60000.pt --per-band 10 --seeds 3 \\
        --out outputs/eval/step_60000.json
    python scripts/select_checkpoint.py outputs/eval/*.json --incumbent outputs/eval/best.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SILENCE_X = 1.3
TOO_FAST_X = 2.0
BAND_F1_DROP = 0.02


def gates(summary: dict, incumbent: dict | None) -> list[str]:
    """The gates this result fails, by name; empty when it passes."""
    failed = []
    if summary["silence_rate"] > SILENCE_X * summary["ref_silence_rate"]:
        failed.append(f"silence {summary['silence_rate']:.4f} > {SILENCE_X} x "
                      f"{summary['ref_silence_rate']:.4f}")
    if summary["grid_raw_too_fast"] > TOO_FAST_X * summary["ref_too_fast"]:
        failed.append(f"too fast {summary['grid_raw_too_fast']:.2f} > {TOO_FAST_X} x "
                      f"{summary['ref_too_fast']:.2f}")
    if incumbent is not None:
        for band, g in incumbent["groups"]["difficulty"].items():
            mine = summary["groups"]["difficulty"].get(band)
            if mine is None:
                failed.append(f"no {band} maps")
            elif mine["onset_f1"] < g["onset_f1"] - BAND_F1_DROP:
                failed.append(f"{band} F1 {mine['onset_f1']:.3f} < {g['onset_f1']:.3f} - {BAND_F1_DROP}")
    return failed


def pool_of(result: dict) -> tuple:
    return tuple(sorted(m["map"] for m in result["maps"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path, nargs="+", help="evaluate.py --out JSON, one per checkpoint")
    ap.add_argument("--incumbent", type=Path, default=None,
                    help="the result of the checkpoint currently kept; band F1 must not fall below it")
    args = ap.parse_args()

    results = {p: json.loads(p.read_text(encoding="utf-8")) for p in args.results}
    incumbent = json.loads(args.incumbent.read_text(encoding="utf-8")) if args.incumbent else None
    pools = {pool_of(r) for r in results.values()} | ({pool_of(incumbent)} if incumbent else set())
    if len(pools) > 1:
        print("ERROR: these results were scored on different map pools; rerun with the same "
              "--per-band / --seeds / --seed")
        return 1

    passing = []
    for path, r in results.items():
        s = r["summary"]
        failed = gates(s, incumbent["summary"] if incumbent else None)
        name = r["checkpoint"].get("evaluated", str(path))
        print(f"{name}  step {r['checkpoint'].get('step')}  exact snap {s['exact_snap']:.4f}  "
              f"F1 {s['onset_f1']:.4f}  " + ("PASS" if not failed else "FAIL: " + "; ".join(failed)))
        if not failed:
            passing.append((s["exact_snap"], name, path))

    if not passing:
        print("\nno candidate passes; keep the incumbent")
        return 1
    best = max(passing)
    print(f"\nkeep {best[1]}  (exact snap {best[0]:.4f}, {best[2]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
