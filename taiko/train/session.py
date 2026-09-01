"""
taiko/train/session.py

Everything a long training run needs to survive being killed.

Both stages run on Kaggle, where a session is preemptible in three different
ways -- the 12-hour cap, an out-of-memory kill, and the user closing the tab --
and only one of them is polite enough to let the process finish. The rules that
follow from that:

  Checkpoints are written atomically. `torch.save` straight to `last.pt` leaves
  a truncated file if the process dies mid-write, and the next session then
  fails to resume at all: the one moment the checkpoint is fragile is exactly
  the moment the session is most likely to be killed. Write to a temporary file
  in the same directory, fsync, then `os.replace` -- rename is atomic, so
  `last.pt` is either the old checkpoint or the new one and never half of each.

  Checkpoints are written on a clock, not only on a step count. `--save-every
  500` at 2.6 s/step is a 22-minute blast radius, and an OOM kill at step 550
  throws away everything since step 500. Wall-clock saving bounds the loss in
  the unit that actually matters.

  SIGTERM saves before exiting. Kaggle's session timeout and the notebook's
  interrupt button both arrive as signals, and a run that ignores them discards
  whatever it did since the last periodic save. SIGKILL (the OOM killer) cannot
  be caught, which is why the periodic save above is the primary defence and
  this is the secondary one.

Memory reporting lives here too, because "your notebook tried to allocate more
memory than is available" is a message about host RAM that arrives with no
indication of what consumed it. Printing RSS and MemAvailable on every log line
turns the next occurrence into a number that can be read off the log rather
than a mystery.
"""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path
from typing import Any, Callable

import torch


# --------------------------------------------------------------------------- #
# Memory reporting
# --------------------------------------------------------------------------- #

_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def _rss_mb(pid: int | None = None) -> float:
    """Resident set size of one process, in MB. 0.0 where /proc is absent."""
    try:
        with open(f"/proc/{pid or os.getpid()}/statm", encoding="ascii") as fh:
            return int(fh.read().split()[1]) * _PAGE_SIZE / 1024 ** 2
    except (OSError, IndexError, ValueError):
        return 0.0


def _child_pids(pid: int) -> list[int]:
    try:
        with open(f"/proc/{pid}/task/{pid}/children", encoding="ascii") as fh:
            return [int(p) for p in fh.read().split()]
    except (OSError, ValueError):
        return []


def memory_mb() -> dict[str, float]:
    """
    Host and device memory, in MB.

    `tree` includes the dataloader workers. They are where a windowed dataset's
    memory actually lives -- each worker holds its own prefetch queue and its
    own copy-on-write image of the index -- so a figure that counts only the
    parent process reports roughly a third of the truth and exonerates the
    component most likely to be at fault.
    """
    pid = os.getpid()
    tree = _rss_mb(pid)
    stack = _child_pids(pid)
    while stack:
        child = stack.pop()
        tree += _rss_mb(child)
        stack.extend(_child_pids(child))

    available = 0.0
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) / 1024
                    break
    except (OSError, IndexError, ValueError):
        pass

    out = {"rss": _rss_mb(pid), "tree": tree, "available": available}
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.memory_allocated() / 1024 ** 2
        out["gpu_reserved"] = torch.cuda.memory_reserved() / 1024 ** 2
    return out


def memory_line() -> str:
    """One compact field for the training log: `ram 3.1/12.4G gpu 8.9G`."""
    m = memory_mb()
    text = f"ram {m['tree'] / 1024:.1f}/{(m['tree'] + m['available']) / 1024:.1f}G"
    if "gpu_reserved" in m:
        text += f" gpu {m['gpu_reserved'] / 1024:.1f}G"
    return text


# --------------------------------------------------------------------------- #
# Stopping cleanly
# --------------------------------------------------------------------------- #

class _StopFlag:
    def __init__(self) -> None:
        self.requested = False
        self.reason = ""

    def __bool__(self) -> bool:
        return self.requested


def install_stop_handlers() -> _StopFlag:
    """
    Turn SIGTERM/SIGINT into a flag the training loop checks.

    The loop then saves and exits through its normal path. Raising from inside
    a signal handler would unwind through whatever CUDA or dataloader call
    happened to be running, which is a good way to lose the checkpoint you were
    trying to protect.

    SIGKILL is not catchable. Periodic saving is what covers that case.
    """
    flag = _StopFlag()

    def handler(signum, _frame):
        if flag.requested:            # A second signal means "now".
            raise KeyboardInterrupt
        flag.requested = True
        flag.reason = signal.Signals(signum).name
        print(f"\n[{flag.reason}] received -- finishing this step, saving, "
              f"then exiting. Send it again to abort immediately.", flush=True)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass                       # Not the main thread; nothing to install.
    return flag


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #

def atomic_save(payload: dict, path: Path) -> None:
    """
    Write a checkpoint that is never observed half-written.

    The temporary file is created in the destination directory so the final
    `os.replace` stays within one filesystem, where it is atomic.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    try:
        with open(tmp, "wb") as fh:
            torch.save(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


class SaveTrigger:
    """
    Should we checkpoint now?

    Three independent reasons, any of which fires: a step count, a wall-clock
    interval, and the end of an epoch. The clock is the one that matters on a
    preemptible machine -- steps are not a unit of risk, minutes are -- but the
    step count stays because it is what makes a run reproducible from its log.
    """

    def __init__(self, every_steps: int = 0, every_minutes: float = 0.0):
        self.every_steps = max(0, int(every_steps))
        self.every_seconds = max(0.0, float(every_minutes) * 60.0)
        self.last_save = time.time()

    def due(self, step: int) -> str:
        if self.every_steps and step % self.every_steps == 0:
            return "step"
        if self.every_seconds and (time.time() - self.last_save) >= self.every_seconds:
            return "clock"
        return ""

    def mark(self) -> None:
        self.last_save = time.time()

    def summary(self) -> str:
        parts = []
        if self.every_steps:
            parts.append(f"every {self.every_steps} steps")
        if self.every_seconds:
            parts.append(f"every {self.every_seconds / 60:.0f} min")
        parts.append("at each epoch end")
        return ", ".join(parts)


class CheckpointSaver:
    """
    Owns `last.pt` and `best.pt` for one stage.

    `build` is a callable returning the payload dict, so a save costs nothing
    until it is actually taken -- and so the caller keeps one definition of what
    a checkpoint contains instead of repeating it at four call sites.
    """

    def __init__(self, out_dir: Path, build: Callable[[], dict[str, Any]],
                 trigger: SaveTrigger, verbose: bool = True):
        self.out = Path(out_dir)
        self.build = build
        self.trigger = trigger
        self.verbose = verbose
        self.saves = 0

    def save(self, name: str = "last.pt", reason: str = "") -> Path:
        path = self.out / name
        atomic_save(self.build(), path)
        if name == "last.pt":
            # Only the resume target resets the clock. best.pt is written
            # whenever validation improves, which is not a reason to let
            # last.pt -- the file --resume actually reads -- go stale.
            self.trigger.mark()
        self.saves += 1
        if self.verbose and reason:
            print(f"  saved {name} ({reason})", flush=True)
        return path

    def maybe_save(self, step: int) -> bool:
        reason = self.trigger.due(step)
        if not reason:
            return False
        self.save("last.pt", reason)
        return True


# Magic numbers, longest first so a prefix cannot shadow a longer match.
# The point is not to support these formats -- it is to say what a file
# actually is when torch refuses it, because "invalid load key, '7'" is pickle
# reporting the first byte it did not understand and nothing more.
_FILE_SIGNATURES: tuple[tuple[bytes, str, str], ...] = (
    (b"\xfd7zXZ\x00",           "xz archive",        "xz"),
    (b"7z\xbc\xaf\x27\x1c",    "7-Zip archive",     "7z"),
    (b"PK\x03\x04",             "zip archive",       "zip"),
    (b"PK\x05\x06",             "empty zip archive", ""),
    (b"\x1f\x8b",               "gzip stream",       "gzip"),
    (b"BZh",                     "bzip2 stream",      "bz2"),
    (b"version https://git-lfs", "Git LFS pointer",   ""),
    (b"<!DOCTYPE",               "HTML page",         ""),
    (b"<html",                   "HTML page",         ""),
    (b"<?xml",                   "XML document",      ""),
)


def describe_file(path: Path) -> tuple[str, str]:
    """
    Identify a file from its first bytes.

    Returns (description, container_kind). `container_kind` is non-empty when
    the file is an archive whose contents could be extracted -- which is the
    difference between "your training is gone" and "your training is wrapped in
    something".
    """
    path = Path(path)
    if not path.exists():
        return "missing", ""

    size = path.stat().st_size
    if size == 0:
        return "empty (0 bytes)", ""

    with open(path, "rb") as handle:
        head = handle.read(512)
        tar_marker = b""
        if size > 262:
            handle.seek(257)
            tar_marker = handle.read(5)

    for signature, description, kind in _FILE_SIGNATURES:
        if head.startswith(signature):
            return f"{description} ({size / 1024**2:.1f} MB)", kind

    if tar_marker == b"ustar":
        return f"tar archive ({size / 1024**2:.1f} MB)", "tar"

    # torch.save without the zip container writes a bare pickle.
    if head[:1] == b"\x80" and len(head) > 1 and head[1] in (2, 3, 4, 5):
        return f"legacy pickle, protocol {head[1]} ({size / 1024**2:.1f} MB)", ""

    printable = sum(32 <= b < 127 or b in (9, 10, 13) for b in head)
    if head and printable / len(head) > 0.9:
        preview = head[:60].decode("utf-8", "replace").replace("\n", " ")
        return f"text, not a checkpoint ({size / 1024**2:.1f} MB): {preview!r}", ""

    return (f"unrecognised, first bytes {head[:8]!r} ({size / 1024**2:.1f} MB)", "")


def load_checkpoint(path: Path, map_location, fallbacks: tuple[str, ...] = ("best.pt",)):
    """
    Load a checkpoint, falling back to a sibling if it will not open.

    A truncated `last.pt` used to end the run before it started -- the file most
    likely to be damaged is the one written most often, and losing the session
    to it wastes the very hours the checkpoint existed to protect. Falling back
    to `best.pt` costs some steps and saves the run.
    """
    path = Path(path)
    candidates = [path] + [path.parent / name for name in fallbacks
                           if (path.parent / name) != path]
    errors: list[str] = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            ckpt = torch.load(candidate, map_location=map_location, weights_only=False)
        except Exception as exc:                                   # noqa: BLE001
            description, container = describe_file(candidate)
            note = f"  {candidate}: {type(exc).__name__}: {exc}\n" \
                   f"      the file on disk is: {description}"
            if container:
                note += (
                    f"\n      it is a {container} container, so the checkpoint "
                    f"inside is probably intact --\n"
                    f"      run: python scripts/rescue_checkpoint.py "
                    f"{candidate.parent}"
                )
            errors.append(note)
            print(f"WARNING: {candidate} could not be loaded ({exc}).")
            print(f"         it is: {description}")
            continue
        if candidate != path:
            print(f"  fell back to {candidate}")
        return ckpt, candidate
    if errors:
        raise RuntimeError(
            "no loadable checkpoint:\n" + "\n".join(errors)
            + "\n\n  If none of these can be rescued, start stage 2 again without"
              "\n  --resume. Stage 1 is unaffected: the autoencoder loaded fine,"
              "\n  or this script would have stopped before building the model."
        )
    return None, None
