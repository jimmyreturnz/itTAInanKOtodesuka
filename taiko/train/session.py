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
indication of what consumed it. The first version of this printed the sum of
RSS over the process and its dataloader workers, which multiplies every page
they share by the number of workers -- the CUDA context, torch's shared
objects, the model, the packed index -- and reported a run at 30 GB whose real
footprint was a fraction of that. A wrong number is worse than none: it sent a
whole session looking for a leak in the data path that measurement then showed
was not there.

So the numbers here are chosen to be attributable. PSS instead of RSS, so the
tree's total is the tree's real footprint. This process and its workers
separately, so growth has an address. The cgroup's own usage and limit, because
that is what a container is actually killed on and /proc/meminfo may be
describing the host. And page cache called out on its own, because streaming a
6.7 GB mel file parks 6.7 GB of it -- reclaimable, not lost, and not the run's
own growth.

Running out is then something the run can act on rather than be killed by:
`headroom_mb` is what the training loops watch, and a stage that sees it fall
too far saves and exits with EXIT_LOW_MEMORY so a supervisor can restart it
with --resume. Two minutes lost instead of a session.
"""

from __future__ import annotations

import hashlib
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

# Exit code meaning "I stopped myself because host memory ran out, my checkpoint
# is current, start me again with --resume". Distinct from 1 (a real error) so a
# supervisor can tell "resume me" from "I am broken" without parsing the log.
EXIT_LOW_MEMORY = 17


def _rss_mb(pid: int | None = None) -> float:
    """Resident set size of one process, in MB. 0.0 where /proc is absent."""
    try:
        with open(f"/proc/{pid or os.getpid()}/statm", encoding="ascii") as fh:
            return int(fh.read().split()[1]) * _PAGE_SIZE / 1024 ** 2
    except (OSError, IndexError, ValueError):
        return 0.0


# What /proc/<pid>/smaps_rollup calls each kind of resident memory, and what
# that kind actually is on a training host. This split is the whole point: 23 MB
# a step is a mystery, 23 MB a step of *locked* memory is the pinned-batch
# allocator and 23 MB a step of *file* memory is a memmap.
_SMAPS_FIELDS = {
    "Pss":       "pss",         # the tree's honest total
    "Pss_Anon":  "anon",        # heap, and cudaHostAlloc'd pinned buffers
    "Pss_File":  "file",        # mapped files -- a mel memmap lands here
    "Pss_Shmem": "shmem",       # /dev/shm: how dataloader workers ship batches
    "Locked":    "locked",      # page-locked, i.e. pin_memory
    "Rss":       "rss",
}


def _smaps_mb(pid: int | None = None) -> dict[str, float]:
    """
    One process's resident memory, split by kind, in MB.

    Falls back to RSS alone where smaps_rollup is unavailable, so this degrades
    to the old (wrong-but-present) number rather than to nothing.
    """
    out = {name: 0.0 for name in _SMAPS_FIELDS.values()}
    try:
        with open(f"/proc/{pid or os.getpid()}/smaps_rollup", encoding="ascii") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                name = _SMAPS_FIELDS.get(key)
                if name:
                    out[name] = int(rest.split()[0]) / 1024
        if out["pss"]:
            return out
    except (OSError, IndexError, ValueError):
        pass
    rss = _rss_mb(pid)
    return {**out, "pss": rss, "rss": rss}


def _pss_mb(pid: int | None = None) -> float:
    """
    Proportional set size of one process, in MB.

    RSS counts a shared page in full in every process that maps it, so summing
    RSS across a parent and its forked dataloader workers multiplies everything
    they share -- the CUDA context, torch's shared objects, the model, the
    packed index -- by the number of workers. That is not a small correction:
    measured here on a four-worker loader, the parent's RSS was 825 MB and the
    RSS sum over the tree was 2,618 MB for a process whose real footprint was
    about a gigabyte. A run reported at 30 GB on that arithmetic may be nowhere
    near 30 GB, which is exactly how two sessions can be spent without learning
    anything.

    PSS divides each shared page by the number of processes mapping it, so the
    sum over a process tree is the tree's real footprint. Falls back to RSS
    where smaps_rollup is unavailable (older kernels, restricted /proc).
    """
    try:
        with open(f"/proc/{pid or os.getpid()}/smaps_rollup", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("Pss:"):
                    return int(line.split()[1]) / 1024
    except (OSError, IndexError, ValueError):
        pass
    return _rss_mb(pid)


def _child_pids(pid: int) -> list[int]:
    try:
        with open(f"/proc/{pid}/task/{pid}/children", encoding="ascii") as fh:
            return [int(p) for p in fh.read().split()]
    except (OSError, ValueError):
        return []


def _descendants(pid: int) -> list[int]:
    out, stack = [], _child_pids(pid)
    while stack:
        child = stack.pop()
        out.append(child)
        stack.extend(_child_pids(child))
    return out


def _read_int(path: str) -> int | None:
    try:
        with open(path, encoding="ascii") as fh:
            text = fh.read().strip()
        return None if text == "max" else int(text)
    except (OSError, ValueError):
        return None


def cgroup_memory() -> dict[str, float]:
    """
    The container's own memory accounting, in MB: `usage`, `limit`, `cache`.

    This is the number a container is actually killed on, and it is not what
    /proc/meminfo reports -- on a machine where /proc is not namespaced,
    MemAvailable describes the host and says nothing about the limit this
    process will die at. Empty dict where no cgroup limit is visible.

    `cache` is page cache charged to the cgroup. Streaming a 6.7 GB mel file
    parks 6.7 GB there; it is reclaimable rather than lost, so it is reported
    separately instead of being mistaken for the run's own growth.
    """
    v2_usage = _read_int("/sys/fs/cgroup/memory.current")
    if v2_usage is not None:
        limit = _read_int("/sys/fs/cgroup/memory.max")
        cache = 0
        try:
            with open("/sys/fs/cgroup/memory.stat", encoding="ascii") as fh:
                for line in fh:
                    if line.startswith("file "):
                        cache = int(line.split()[1])
                        break
        except (OSError, IndexError, ValueError):
            pass
        out = {"usage": v2_usage / 1024 ** 2, "cache": cache / 1024 ** 2}
        if limit:
            out["limit"] = limit / 1024 ** 2
        return out

    v1_usage = _read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    if v1_usage is not None:
        limit = _read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        cache = 0
        try:
            with open("/sys/fs/cgroup/memory/memory.stat", encoding="ascii") as fh:
                for line in fh:
                    if line.startswith("total_cache "):
                        cache = int(line.split()[1])
                        break
        except (OSError, IndexError, ValueError):
            pass
        out = {"usage": v1_usage / 1024 ** 2, "cache": cache / 1024 ** 2}
        # A cgroup with no limit reports something absurd like 8 EB.
        if limit and limit < 1 << 53:
            out["limit"] = limit / 1024 ** 2
        return out

    return {}


def _mem_available_mb() -> float:
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except (OSError, IndexError, ValueError):
        pass
    return 0.0


def headroom_mb() -> float:
    """
    How much more this process tree can allocate before it is killed, in MB.

    Prefers the cgroup limit, because that is what kills a container, and
    discounts page cache, because the kernel reclaims that rather than dying
    on it. Falls back to MemAvailable where no limit is visible.
    """
    cg = cgroup_memory()
    if "limit" in cg:
        anonymous = max(cg["usage"] - cg.get("cache", 0.0), 0.0)
        return max(cg["limit"] - anonymous, 0.0)
    return _mem_available_mb()


def headroom_gb() -> float:
    return headroom_mb() / 1024


def memory_mb() -> dict[str, float]:
    """
    Host and device memory, in MB.

    `self` is this process, `workers` is everything it forked -- the dataloader
    workers, where a windowed dataset's memory actually lives -- and both are
    PSS, so the pages they share are counted once between them rather than once
    each. `tree` is their sum: the real footprint of the run.
    """
    pid = os.getpid()
    children = _descendants(pid)
    mine = _smaps_mb(pid)
    theirs = [_smaps_mb(child) for child in children]

    def total(name: str) -> float:
        return mine[name] + sum(t[name] for t in theirs)

    out = {
        "self": mine["pss"],
        "workers": sum(t["pss"] for t in theirs),
        "worker_count": float(len(children)),
        "tree": total("pss"),
        # Where that total lives. Anonymous is the heap and the pinned-buffer
        # allocator; file is mapped files; shmem is how workers hand batches
        # over; locked is pinned specifically, and is a subset of anonymous.
        "anon": total("anon"),
        "file": total("file"),
        "shmem": total("shmem"),
        "locked": total("locked"),
        "rss": mine["rss"],
        "available": _mem_available_mb(),
        "headroom": headroom_mb(),
    }
    cg = cgroup_memory()
    if cg:
        out["cgroup_usage"] = cg["usage"]
        out["cgroup_cache"] = cg.get("cache", 0.0)
        if "limit" in cg:
            out["cgroup_limit"] = cg["limit"]
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.memory_allocated() / 1024 ** 2
        out["gpu_reserved"] = torch.cuda.memory_reserved() / 1024 ** 2
    return out


def memory_line(snapshot: dict[str, float] | None = None) -> str:
    """
    One compact field for the training log.

    Every term is here because its absence cost a session: how much this run
    holds, how much of that is the workers, how much of the container's usage
    is merely reclaimable page cache, and how much room is actually left.

        ram 4.1+1.2G | cg 18.3/29.0G cache 6.7G | free 10.7G gpu 1.1G
    """
    m = snapshot or memory_mb()
    text = f"ram {m['self'] / 1024:.1f}+{m['workers'] / 1024:.1f}G"
    if m["locked"] > 256:
        text += f" lock {m['locked'] / 1024:.1f}G"
    if "cgroup_limit" in m:
        text += (f" | cg {m['cgroup_usage'] / 1024:.1f}/{m['cgroup_limit'] / 1024:.1f}G"
                 f" cache {m.get('cgroup_cache', 0.0) / 1024:.1f}G")
    text += f" | free {m['headroom'] / 1024:.1f}G"
    if "gpu_reserved" in m:
        text += f" gpu {m['gpu_reserved'] / 1024:.1f}G"
    return text


def memory_report(snapshot: dict[str, float] | None = None) -> str:
    """The same numbers over several lines, for the start and end of a run."""
    m = snapshot or memory_mb()
    lines = [
        f"  this process     {m['self'] / 1024:6.2f} GB   (RSS {m['rss'] / 1024:.2f} GB)",
        f"  {int(m['worker_count'])} child processes {m['workers'] / 1024:6.2f} GB",
        f"  run total        {m['tree'] / 1024:6.2f} GB",
    ]
    if "cgroup_limit" in m:
        lines.append(
            f"  container        {m['cgroup_usage'] / 1024:6.2f} GB of "
            f"{m['cgroup_limit'] / 1024:.2f} GB, of which "
            f"{m.get('cgroup_cache', 0.0) / 1024:.2f} GB is reclaimable page cache")
    else:
        lines.append("  container        no cgroup limit visible; "
                     "falling back to MemAvailable")
    lines.append(f"  room to grow     {m['headroom'] / 1024:6.2f} GB")
    lines.append(f"  of the total:    anonymous {m['anon'] / 1024:.2f} GB "
                 f"(page-locked {m['locked'] / 1024:.2f} GB), "
                 f"mapped files {m['file'] / 1024:.2f} GB, "
                 f"shared {m['shmem'] / 1024:.2f} GB")
    return "\n".join(lines)


class MemoryTrend:
    """
    Watches host memory across a run and says what it is doing, and why.

    A number printed every 25 steps is not a trend, and two sessions were spent
    reading one. What the log actually needed to say was: this is growing, at
    23 MB a step, and at that rate you have 870 steps before you are killed --
    which is knowable at step 150, not at hour one.

    The attribution matters as much as the rate. Growth in `locked` is the
    pinned-batch allocator; in `file`, a memmap; in `shmem`, the pipe the
    dataloader workers ship batches through; in `anon` and nothing else, the
    Python or CUDA heap. Naming the kind turns "find the leak" into a search
    with one place to look.

    A baseline is taken after `warmup_steps`, because the first minute of a run
    is loader spin-up and CUDA context creation and is not a trend.
    """

    def __init__(self, warmup_steps: int = 100, window: int = 20):
        self.warmup_steps = warmup_steps
        self.window = window
        self.baseline: tuple[int, dict[str, float]] | None = None
        self.samples: list[tuple[int, float]] = []
        self._warned = False

    def observe(self, step: int, snapshot: dict[str, float] | None = None) -> None:
        snapshot = snapshot or memory_mb()
        if self.baseline is None:
            if step < self.warmup_steps:
                return
            self.baseline = (step, snapshot)
        self.samples.append((step, snapshot["tree"]))
        if len(self.samples) > self.window:
            del self.samples[0]

    def slope_mb_per_step(self) -> float | None:
        """Least squares over the recent window; None until there is one."""
        if len(self.samples) < 4:
            return None
        n = len(self.samples)
        sx = sum(s for s, _ in self.samples)
        sy = sum(m for _, m in self.samples)
        sxx = sum(s * s for s, _ in self.samples)
        sxy = sum(s * m for s, m in self.samples)
        denominator = n * sxx - sx * sx
        if denominator == 0:
            return None
        return (n * sxy - sx * sy) / denominator

    def steps_left(self) -> float | None:
        slope = self.slope_mb_per_step()
        if not slope or slope <= 0:
            return None
        return headroom_mb() / slope

    def first_warning(self, threshold_mb_per_step: float = 1.0) -> str:
        """
        The alarm, raised once, as soon as growth is unmistakable.

        Once is deliberate: a warning on every log line is a warning nobody
        reads, and the useful moment is the first one -- when there are still
        hours left to act on it rather than minutes.
        """
        if self._warned:
            return ""
        slope = self.slope_mb_per_step()
        if slope is None or slope < threshold_mb_per_step:
            return ""
        self._warned = True
        left = self.steps_left()
        when = (f"about {left:.0f} steps" if left is not None else "an unknown time")
        return ("\n  HOST MEMORY IS GROWING. This run has " + when + " before it is\n"
                "  killed. What is growing:\n" + self.report())

    def compact(self) -> str:
        """`+23.4MB/st ~870 left`, or empty while nothing is growing."""
        slope = self.slope_mb_per_step()
        if slope is None or slope < 0.5:
            return ""
        left = self.steps_left()
        text = f"+{slope:.1f}MB/st"
        return text + (f" ~{left:.0f} left" if left is not None else "")

    def report(self, snapshot: dict[str, float] | None = None) -> str:
        """Where the growth since the baseline actually went."""
        if self.baseline is None:
            return "  memory trend: not enough of the run has passed yet"
        first_step, first = self.baseline
        now = snapshot or memory_mb()
        steps = (self.samples[-1][0] if self.samples else first_step) - first_step
        grown = now["tree"] - first["tree"]
        slope = self.slope_mb_per_step()
        if slope is not None and slope < 0.5:
            return (f"  memory flat since step {first_step} "
                    f"({grown:+.0f} MB over {steps} steps)")

        lines = [f"  since step {first_step}: {grown:+.0f} MB over {steps} steps"
                 + (f" ({grown / steps:+.1f} MB/step)" if steps > 0 else "")]
        for key, label in (("anon", "anonymous (heap, pinned buffers)"),
                           ("locked", "  of which page-locked (pin_memory)"),
                           ("file", "mapped files (a memmap)"),
                           ("shmem", "shared memory (dataloader IPC)")):
            lines.append(f"    {label:<36s} {now[key] - first[key]:+8.0f} MB"
                         f"   (now {now[key] / 1024:.2f} GB)")
        left = self.steps_left()
        if left is not None:
            lines.append(f"    at this rate: about {left:.0f} steps of headroom left")
        return "\n".join(lines)


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

MANIFEST_NAME = "checkpoints.sha256"


def file_digest(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    """sha256 of a file, read in chunks so a 500 MB checkpoint costs no memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block_bytes := handle.read(chunk):
            digest.update(block_bytes)
    return digest.hexdigest()


def read_manifest(directory: Path) -> dict[str, tuple[str, int]]:
    """{name: (digest, size)} from a directory's manifest; empty if absent."""
    manifest = Path(directory) / MANIFEST_NAME
    if not manifest.exists():
        return {}
    out: dict[str, tuple[str, int]] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            digest, size, name = line.split(None, 2)
            out[name] = (digest, int(size))
        except ValueError:
            continue
    return out


def update_manifest(directory: Path, changed: Path | None = None) -> Path:
    """
    Record the size and digest of every checkpoint in a directory.

    Written next to the checkpoints and carried with them. A checkpoint that
    arrives on the next machine 7-Zip-wrapped, truncated, or replaced by an
    HTML error page has a different size and digest, and that is knowable in
    seconds at restore time rather than eleven hours later when --resume
    finally opens it.

    `changed` names the one file just written, and every other entry is carried
    over from the existing manifest. Re-hashing the whole directory on every
    save meant reading best.pt as well as last.pt -- a gigabyte of I/O every
    ten minutes, through the page cache, on the machine least able to spare it,
    to recompute a digest that could not have changed.
    """
    directory = Path(directory)
    known = read_manifest(directory)
    if changed is not None:
        known.pop(Path(changed).name, None)

    lines = []
    for path in sorted(directory.glob("*.pt")):
        cached = known.get(path.name)
        if cached and cached[1] == path.stat().st_size:
            digest, size = cached
        else:
            digest, size = file_digest(path), path.stat().st_size
        lines.append(f"{digest}  {size}  {path.name}")

    manifest = directory / MANIFEST_NAME
    manifest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return manifest


def verify_manifest(directory: Path) -> tuple[list[str], list[str]]:
    """
    Check checkpoints against their manifest.

    Returns (ok, problems). A missing manifest is not a problem -- checkpoints
    from before this existed are still perfectly loadable -- so it reports
    nothing rather than inventing a failure.
    """
    directory = Path(directory)
    manifest = directory / MANIFEST_NAME
    if not manifest.exists():
        return [], []

    ok: list[str] = []
    problems: list[str] = []

    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            expected_digest, expected_size, name = line.split(None, 2)
        except ValueError:
            continue

        path = directory / name
        if not path.exists():
            problems.append(f"{name}: recorded in the manifest but not here")
            continue

        actual_size = path.stat().st_size
        if actual_size != int(expected_size):
            delta = actual_size - int(expected_size)
            description, container = describe_file(path)
            detail = (f"{name}: {actual_size:,} bytes, manifest says "
                      f"{int(expected_size):,} ({delta:+,})")
            detail += f"\n      it is now: {description}"
            if container:
                detail += (f"\n      a {container} container -- something wrapped it "
                           f"in transit; scripts/rescue_checkpoint.py can unwrap it")
            elif actual_size < int(expected_size):
                detail += "\n      smaller than recorded: the transfer did not finish"
            problems.append(detail)
            continue

        if file_digest(path) != expected_digest:
            problems.append(f"{name}: right size, wrong contents -- "
                            f"the bytes changed in transit")
            continue

        ok.append(name)

    return ok, problems


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
        # Refresh after every write. The manifest is only useful if it
        # describes the files as they are now, not as they were at the start
        # of the session -- but only this file changed, so only this file is
        # re-hashed.
        update_manifest(self.out, changed=path)
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
