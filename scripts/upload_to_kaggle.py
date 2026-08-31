"""
Upload the packed shards to Kaggle, with progress, and check they arrived.

The Kaggle client reports success when its CreateDataset call returns, not when
the files are actually there. One run over an unstable link pushed 12.2 GB for a
6.69 GB mels.dat, lost the final call to SSL errors, retried, and registered a
dataset holding only charts.npz and index.json -- while printing nothing wrong.

So this asks Kaggle what it actually holds afterwards and compares sizes against
the local files. Success here means the bytes are on Kaggle, not that a function
returned.

    python scripts/upload_to_kaggle.py                # upload, then verify
    python scripts/upload_to_kaggle.py --verify-only  # just check what is there

Authentication comes from ~/.kaggle/access_token (new KGAT_ bearer tokens) or
~/.kaggle/kaggle.json (legacy username+key).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Relative to the repo, not the shell's cwd -- otherwise running this from a
# parent directory silently looks for the shards somewhere they cannot be, and
# reports them missing.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SHARDS = REPO_ROOT / "data" / "processed" / "shards"
REQUIRED = ("mels.dat", "charts.npz", "index.json")


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} GB"


def get_api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def remote_files(api, ref: str) -> dict[str, int]:
    """{name: bytes} for what Kaggle actually holds. Empty dict if absent."""
    try:
        listing = api.dataset_list_files(ref)
    except Exception:                                          # noqa: BLE001
        return {}
    out = {}
    for f in getattr(listing, "files", []) or []:
        size = getattr(f, "total_bytes", None) or getattr(f, "size", None) or 0
        out[str(f.name)] = int(size)
    return out


def remote_total(api, ref: str) -> int:
    """Dataset size as Kaggle reports it -- the listing can omit large files."""
    for d in api.dataset_list(mine=True):
        if d.ref == ref:
            return int(getattr(d, "total_bytes", 0) or 0)
    return 0


def verify(api, ref: str, shards: Path) -> bool:
    print(f"\nVerifying {ref}")
    local = {n: (shards / n).stat().st_size for n in REQUIRED}
    want = sum(local.values())

    remote = remote_files(api, ref)
    total = remote_total(api, ref)

    print(f"  local  : {len(local)} files, {human(want)}")
    print(f"  Kaggle : {len(remote)} files listed, {human(total)} reported")

    problems = []
    for name, size in local.items():
        got = remote.get(name)
        if got is None:
            problems.append(f"MISSING  {name}  ({human(size)} expected)")
        elif got and got != size:
            problems.append(f"SIZE     {name}: local {human(size)}, Kaggle {human(got)}")

    # The file listing has been seen to under-report, so the dataset's own byte
    # total is the second opinion -- but only when the listing is incomplete.
    # Kaggle reports the *stored* (compressed) size there, which is legitimately
    # smaller than the raw bytes, so it would false-alarm on a good upload.
    if total and len(remote) < len(local) and total < want * 0.98:
        problems.append(f"TOTAL    dataset is {human(total)}, expected ~{human(want)}")

    for p in problems:
        print(f"  {p}")

    if problems:
        print("\n  INCOMPLETE. Re-run this script; it pushes a new version.")
        return False
    print("\n  OK -- every file is on Kaggle at the right size.")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Upload packed shards to Kaggle")
    ap.add_argument("--shards", type=Path, default=DEFAULT_SHARDS)
    ap.add_argument("--owner", default="jimmyreturnz")
    ap.add_argument("--slug", default="taiko-shards")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--public", action="store_true",
                    help="default is private, which is right for mels of "
                         "copyrighted audio")
    args = ap.parse_args()

    ref = f"{args.owner}/{args.slug}"
    shards = args.shards

    shards = shards.resolve()
    missing = [n for n in REQUIRED if not (shards / n).exists()]
    if missing:
        print(f"ERROR: {shards} is missing {', '.join(missing)}")
        if not shards.exists():
            print("  That folder does not exist. Pack the dataset first:")
            print("    python scripts/pack_dataset.py --ranked-only")
        else:
            print("  Pack the dataset first, or point --shards at it.")
        return 1

    try:
        import kaggle                                          # noqa: F401
    except ImportError:
        print(f"ERROR: the `kaggle` package is not installed for {sys.executable}")
        print("  Either install it here:  pip install kaggle")
        print("  or use the interpreter that has it (your global Python 3.11).")
        return 1

    api = get_api()

    if args.verify_only:
        return 0 if verify(api, ref, shards) else 1

    # The CLI needs this file beside the data, and writes the dataset's identity.
    (shards / "dataset-metadata.json").write_text(json.dumps({
        "title": args.slug,
        "id": ref,
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2), encoding="utf-8")

    exists = bool(remote_files(api, ref)) or any(
        d.ref == ref for d in api.dataset_list(mine=True))

    size = sum((shards / n).stat().st_size for n in REQUIRED)
    print(f"Uploading {human(size)} to {ref}"
          f"  ({'new version' if exists else 'new dataset'}, "
          f"{'public' if args.public else 'private'})")
    print("Watch the per-file bars. If the byte counter climbs past a file's own")
    print("size, chunks are being re-sent -- the link is dropping, and the run")
    print("may not converge. Ctrl+C and retry on a better connection.\n")

    t0 = time.time()
    try:
        # The client builds a temp filename from the folder path, so a path with
        # separators lands in directories it never creates. Run from the parent
        # and pass the bare folder name.
        import os
        parent, name = shards.resolve().parent, shards.resolve().name
        cwd = os.getcwd()
        os.chdir(parent)
        try:
            if exists:
                api.dataset_create_version(name, version_notes="reupload",
                                           quiet=False, dir_mode="skip")
            else:
                api.dataset_create_new(name, public=args.public, quiet=False,
                                       dir_mode="skip")
        finally:
            os.chdir(cwd)
    except KeyboardInterrupt:
        print("\nInterrupted. Nothing is lost locally; re-run to try again.")
        return 130

    print(f"\nTransfer call returned after {(time.time() - t0) / 60:.1f} min.")
    print("That is not proof the files landed -- checking.")

    ok = verify(api, ref, shards)
    print(f"\nhttps://www.kaggle.com/datasets/{ref}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
