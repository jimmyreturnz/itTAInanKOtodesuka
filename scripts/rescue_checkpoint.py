"""
scripts/rescue_checkpoint.py

Work out why a checkpoint will not load, and get the weights back if they are
still there.

    python scripts/rescue_checkpoint.py /kaggle/working/checkpoints/diffusion
    python scripts/rescue_checkpoint.py checkpoints/diffusion --write

`torch.load` reports a damaged checkpoint as "invalid load key, 'x'", which is
pickle quoting the first byte it did not understand. That byte is usually
enough to identify the file: 'P' is a healthy zip, '7' is a 7-Zip archive,
'\\x1f' is gzip, '<' is an HTML error page saved under a .pt name. A file that
turns out to be an archive is not lost -- the checkpoint is inside it.

Two failures look identical in the log and are not the same problem:

  Every checkpoint in a directory fails the same way. Nothing was written
  wrong; something happened to the files afterwards, in the download, the
  upload, or the attach. This is the recoverable case.

  One file fails and its siblings load. That one write was interrupted.
  Checkpoints are written to a temporary file and renamed, so this should no
  longer happen, but an out-of-disk error can still truncate the temporary
  file. Use the sibling; --resume already falls back to best.pt on its own.

Without --write nothing is modified: it reports what it found and what it
would do.
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import lzma
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from taiko.train.session import describe_file

CHECKPOINT_KEYS = ("model", "unet", "wave", "optimizer", "step")


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def _report(message: str, indent: str = "") -> None:
    """Print immediately. Extraction of a large archive is minutes of silence
    otherwise, which is indistinguishable from a hang."""
    print(f"{indent}{message}", flush=True)


def loads_cleanly(path: Path) -> tuple[bool, str]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:                                       # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    if not isinstance(obj, dict):
        return False, f"loaded, but it is a {type(obj).__name__}, not a checkpoint"
    present = [k for k in CHECKPOINT_KEYS if k in obj]
    if not present:
        return False, f"loaded, but has none of {CHECKPOINT_KEYS}"
    step = obj.get("step", "?")
    return True, f"step {step}, keys: {', '.join(sorted(obj)[:8])}"


def _decompress_stream(source: Path, target: Path, opener) -> bool:
    """Single-stream formats: gzip, bzip2, xz. One file in, one file out."""
    try:
        with opener(source, "rb") as raw, open(target, "wb") as out:
            shutil.copyfileobj(raw, out, length=8 * 1024 * 1024)
        return True
    except Exception as exc:                                       # noqa: BLE001
        print(f"    decompression failed: {exc}")
        return False


def _largest_member(directory: Path) -> Path | None:
    """
    The biggest file an archive yielded.

    Size is the right heuristic: a checkpoint is tens to hundreds of megabytes
    and anything else an archive carries alongside it -- a log, a config, a
    macOS resource fork -- is orders of magnitude smaller.
    """
    files = [p for p in directory.rglob("*") if p.is_file()]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_size)


def extract(source: Path, kind: str, workdir: Path) -> Path | None:
    """Unwrap one container. Returns the path of the recovered payload."""
    workdir.mkdir(parents=True, exist_ok=True)

    if kind in ("gzip", "bz2", "xz"):
        opener = {"gzip": gzip.open, "bz2": bz2.open, "xz": lzma.open}[kind]
        target = workdir / source.name
        return target if _decompress_stream(source, target, opener) else None

    if kind == "zip":
        try:
            with zipfile.ZipFile(source) as archive:
                archive.extractall(workdir)
        except Exception as exc:                                   # noqa: BLE001
            print(f"    unzip failed: {exc}")
            return None
        return _largest_member(workdir)

    if kind == "tar":
        try:
            with tarfile.open(source) as archive:
                # filter="data" refuses absolute paths and members escaping the
                # destination; without it, extracting an untrusted archive can
                # write anywhere on the filesystem.
                try:
                    archive.extractall(workdir, filter="data")
                except TypeError:                # Python < 3.12
                    archive.extractall(workdir)
        except Exception as exc:                                   # noqa: BLE001
            print(f"    untar failed: {exc}")
            return None
        return _largest_member(workdir)

    if kind == "7z":
        # The binary first: it streams, where py7zr buffers members in memory.
        # On a checkpoint of several hundred megabytes, on a machine already
        # short of RAM, that difference is the difference between recovering
        # the run and being killed while recovering it.
        binary = shutil.which("7z") or shutil.which("7za") or shutil.which("7zr")
        if binary:
            _report(f"    using {Path(binary).name}")
            result = subprocess.run(
                [binary, "x", str(source), f"-o{workdir}", "-y", "-bso0", "-bsp0"],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                return _largest_member(workdir)
            _report(f"    {Path(binary).name} failed: "
                    f"{(result.stderr or result.stdout).strip()[:200]}")

        try:
            import py7zr                                           # noqa: PLC0415
        except ImportError:
            _report("    no 7-Zip available. Install one and re-run:")
            _report("      pip install py7zr          # or: apt-get install -y p7zip-full")
            return None

        _report("    using py7zr (holds members in memory; the 7z binary is "
                "lighter if you can install it)")
        try:
            with py7zr.SevenZipFile(source, "r") as archive:
                archive.extractall(workdir)
            return _largest_member(workdir)
        except MemoryError:
            _report("    ran out of memory. Install the binary instead:")
            _report("      apt-get install -y p7zip-full")
            return None
        except Exception as exc:                                   # noqa: BLE001
            _report(f"    py7zr failed: {exc}")
            return None

    return None


def inspect(path: Path) -> None:
    """
    Report everything knowable about a file without trying to repair it.

    When extraction fails, the summary line says "unrecoverable" and nothing
    about why. The three causes need different responses and look identical
    from outside: the archive is truncated, the tool to open it is missing, or
    there is nowhere to put what comes out.
    """
    description, container = describe_file(path)
    size = path.stat().st_size if path.exists() else 0
    print(f"\n{path}")
    print(f"  identified as : {description}")
    print(f"  size on disk  : {size:,} bytes ({size / 1024**2:.1f} MB)")

    with open(path, "rb") as handle:
        head = handle.read(32)
        handle.seek(max(0, size - 16))
        tail = handle.read(16)
    print(f"  first 16 bytes: {head[:16].hex(' ')}")
    print(f"  last 16 bytes : {tail.hex(' ')}")

    free = free_bytes(path.parent)
    print(f"  free space    : {free / 1024**3:.1f} GB next to the file")

    if container != "7z":
        if container:
            print(f"  container     : {container}")
        return

    # A 7z file records its total size in the 32-byte header. Comparing that
    # against the file on disk is how a truncated download is distinguished
    # from a corrupt one -- extraction fails the same way for both.
    import struct
    if len(head) >= 32:
        next_offset, next_size = struct.unpack("<QQ", head[12:28])
        expected = 32 + next_offset + next_size
        print(f"  header says   : {expected:,} bytes total")
        if expected > size:
            short = expected - size
            print(f"  TRUNCATED     : {short:,} bytes missing "
                  f"({short / 1024**2:.1f} MB, {short / expected:.1%} of the file)")
            print(f"                  the upload or download did not finish; "
                  f"re-fetch it")
            return
        if expected < size:
            print(f"  note          : {size - expected:,} bytes of trailing data")

    try:
        import py7zr                                               # noqa: PLC0415
    except ImportError:
        print("  contents      : install py7zr to list them "
              "(pip install py7zr)")
        return

    try:
        with py7zr.SevenZipFile(path, "r") as archive:
            members = archive.list()
    except Exception as exc:                                       # noqa: BLE001
        print(f"  contents      : cannot be listed -- {type(exc).__name__}: {exc}")
        print("                  the archive header is damaged, not just the payload")
        return

    total = sum(m.uncompressed for m in members)
    print(f"  contents      : {len(members)} member(s), "
          f"{total / 1024**2:.1f} MB uncompressed")
    for member in members[:10]:
        print(f"      {member.filename}  ({member.uncompressed / 1024**2:.1f} MB)"
              f"{'  [encrypted]' if getattr(member, 'is_encrypted', False) else ''}")
    if len(members) > 10:
        print(f"      ... and {len(members) - 10} more")

    if total > free:
        print(f"  NO ROOM       : needs {total / 1024**3:.1f} GB, "
              f"{free / 1024**3:.1f} GB free")


def rescue(path: Path, write: bool, depth: int = 0) -> bool:
    """Diagnose one checkpoint, unwrapping containers until it loads."""
    indent = "  " + "  " * depth
    description, container = describe_file(path)
    print(f"{indent}{path.name}: {description}")

    ok, detail = loads_cleanly(path)
    if ok:
        print(f"{indent}  loads fine -- {detail}")
        return True

    print(f"{indent}  will not load -- {detail}")

    if not container:
        return False

    if depth >= 3:
        print(f"{indent}  giving up after three layers of wrapping")
        return False

    size_mb = path.stat().st_size / 1024 ** 2
    # Extract beside the file, not into /tmp. On the same filesystem the final
    # move is a rename rather than a copy of several hundred megabytes, and it
    # cannot fill a small /tmp on a machine whose real space is elsewhere.
    workdir = path.parent / f".rescue-{path.name}"

    needed = path.stat().st_size * 3          # archive + payload + the copy
    available = free_bytes(path.parent)
    if available < needed:
        _report(f"  not enough room next to the file: "
                f"{available / 1024**3:.1f} GB free, about "
                f"{needed / 1024**3:.1f} GB needed", indent)
        return False

    _report(f"  it is a {container} container; extracting {size_mb:.0f} MB "
            f"(minutes, not seconds)", indent)

    try:
        recovered = extract(path, container, workdir)
        if recovered is None:
            _report("  nothing came out of it", indent)
            return False

        out_mb = recovered.stat().st_size / 1024 ** 2
        _report(f"  recovered {recovered.name} ({out_mb:.1f} MB)", indent)

        inner_ok, inner_detail = loads_cleanly(recovered)
        if not inner_ok:
            inner_description, inner_container = describe_file(recovered)
            if inner_container:
                staged = path.with_suffix(path.suffix + f".layer{depth + 1}")
                shutil.copy2(recovered, staged)
                try:
                    return rescue(staged, write, depth + 1)
                finally:
                    staged.unlink(missing_ok=True)
            print(f"{indent}  the contents do not load either: {inner_detail}")
            print(f"{indent}  ({inner_description})")
            return False

        print(f"{indent}  the contents ARE a checkpoint -- {inner_detail}")

        if not write:
            _report(f"  re-run with --write to replace {path.name} with it", indent)
            return True

        backup = path.with_suffix(path.suffix + ".broken")
        shutil.move(str(path), str(backup))
        # Same filesystem, so this is a rename: instant, and it cannot leave a
        # half-written checkpoint behind if the process dies mid-copy.
        shutil.move(str(recovered), str(path))
        _report(f"  wrote {path.name}; the unreadable file is now {backup.name}", indent)
        return True
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Diagnose and recover checkpoints that will not load",
    )
    ap.add_argument("target", type=Path,
                    help="a .pt file, or a directory of them")
    ap.add_argument("--inspect", action="store_true",
                    help="report what each file is and what is inside it, "
                         "without attempting any repair")
    ap.add_argument("--write", action="store_true",
                    help="replace a wrapped checkpoint with its contents "
                         "(the original is kept as .pt.broken)")
    args = ap.parse_args()

    if not args.target.exists():
        print(f"ERROR: {args.target} does not exist")
        return 1

    if args.target.is_file():
        targets = [args.target]
    else:
        targets = sorted(args.target.rglob("*.pt"))
        if not targets:
            print(f"No .pt files under {args.target}")
            return 1

    if args.inspect:
        for path in targets:
            inspect(path)
        return 0

    print(f"Checking {len(targets)} checkpoint(s)\n")
    healthy, rescued, lost = [], [], []
    for path in targets:
        before, _ = loads_cleanly(path)
        if rescue(path, args.write):
            (healthy if before else rescued).append(path)
        else:
            lost.append(path)
        print()

    print("=" * 62)
    print(f"  {len(healthy)} already fine")
    print(f"  {len(rescued)} recoverable{'' if args.write else ' (nothing written yet)'}")
    print(f"  {len(lost)} unrecoverable")

    if rescued and not args.write:
        print(f"\nRe-run with --write to repair them:")
        print(f"  python scripts/rescue_checkpoint.py {args.target} --write")

    if lost and not rescued and not healthy:
        print("\nNothing here can be recovered. This does not mean starting over:")
        print("  Stage 1 is a separate checkpoint and is very likely fine -- check it")
        print("  with this same script against checkpoints/autoencoder.")
        print("  Stage 2 restarts by running train_diffusion.py without --resume.")

    return 0 if not lost else 1


if __name__ == "__main__":
    raise SystemExit(main())
