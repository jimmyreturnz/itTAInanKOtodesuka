"""
scripts/verify_checkpoints.py

Confirm the checkpoints are what they claim to be, before training starts.

    python scripts/verify_checkpoints.py /kaggle/working/checkpoints

Run this straight after restoring a checkpoint dataset. It takes seconds and
answers the only question that matters at that point: will `--resume` open
these files? The alternative is discovering the answer eleven hours later, when
the session has already spent its budget building a model it cannot resume.

Three levels of check, cheapest first, because each rules out a different
failure:

  size and digest against the manifest, when one travelled with the files.
  Catches a truncated transfer or a wrapped file without reading the payload.

  magic bytes. Catches an archive, an HTML error page, or an empty file
  regardless of whether a manifest exists.

  a real torch.load. The only proof that the pickle inside is intact and
  carries the keys the training script will ask for.

Exits non-zero if anything would fail at resume, so a notebook cell can stop
rather than continue into a session it will lose.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from taiko.train.session import MANIFEST_NAME, describe_file, verify_manifest

# What each stage's checkpoint must contain to be resumable. Checking for the
# keys is what separates "a valid torch file" from "a checkpoint this script
# can actually resume from".
REQUIRED = {
    "autoencoder": ("model", "step"),
    "diffusion":   ("unet", "wave", "step"),
}


def check_file(path: Path, stage: str) -> tuple[bool, str]:
    description, container = describe_file(path)

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:                                       # noqa: BLE001
        message = f"will not load -- {type(exc).__name__}: {exc}\n      it is: {description}"
        if container:
            message += (f"\n      a {container} container; try "
                        f"scripts/rescue_checkpoint.py {path.parent}")
        return False, message

    if not isinstance(payload, dict):
        return False, f"loaded, but it is a {type(payload).__name__}"

    required = REQUIRED.get(stage, ())
    missing = [key for key in required if key not in payload]
    if missing:
        return False, (f"loaded, but missing {missing} -- "
                       f"this is not a {stage} checkpoint")

    step = payload.get("step", "?")
    epoch = payload.get("epoch", "?")
    extras = []
    if payload.get("ema"):
        extras.append(f"EMA at {payload['ema'].get('step', '?')}")
    if "best_f1" in payload:
        extras.append(f"Gate A F1 {payload['best_f1']:.4f}")
    if "best_val" in payload:
        extras.append(f"best val {payload['best_val']:.5f}")
    if payload.get("profile"):
        extras.append(f"profile {payload['profile']}")

    size_mb = path.stat().st_size / 1024 ** 2
    return True, (f"ok -- step {step}, epoch {epoch}, {size_mb:.0f} MB"
                  + (f"  ({', '.join(extras)})" if extras else ""))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Verify checkpoints before spending a training session on them",
    )
    ap.add_argument("root", type=Path, nargs="?", default=Path("checkpoints"))
    ap.add_argument("--require", nargs="*", default=[],
                    help="stages that must be present and valid, "
                         "e.g. --require autoencoder")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"No checkpoint directory at {args.root}")
        return 1 if args.require else 0

    failures = 0
    found_stages: set[str] = set()

    for stage in ("autoencoder", "diffusion"):
        directory = args.root / stage
        if not directory.is_dir():
            continue

        print(f"\n{stage}")

        ok_names, problems = verify_manifest(directory)
        if problems:
            print(f"  manifest ({MANIFEST_NAME}):")
            for problem in problems:
                print(f"    {problem}")
            failures += len(problems)
        elif ok_names:
            print(f"  manifest: {len(ok_names)} file(s) match "
                  f"({', '.join(ok_names)})")
        else:
            print(f"  manifest: none travelled with these files")

        checkpoints = sorted(directory.glob("*.pt"))
        if not checkpoints:
            print("  no .pt files here")
            continue

        for path in checkpoints:
            good, detail = check_file(path, stage)
            print(f"  {path.name:<12s} {detail}")
            if good:
                found_stages.add(stage)
            else:
                failures += 1

    print()
    for stage in args.require:
        if stage not in found_stages:
            print(f"REQUIRED: no usable {stage} checkpoint under {args.root}")
            failures += 1

    if failures:
        print(f"{failures} problem(s). Do not start a training session on these.")
        print("  wrapped files      -> scripts/rescue_checkpoint.py <dir> --write")
        print("  truncated files    -> re-download; the transfer did not finish")
        print("  nothing recoverable-> train that stage again without --resume;")
        print("                        the other stage is independent and unaffected")
        return 1

    print("All checkpoints verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
