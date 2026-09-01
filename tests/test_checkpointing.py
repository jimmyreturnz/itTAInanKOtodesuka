"""
tests/test_checkpointing.py

The checkpoint is the only thing a preemptible session leaves behind, so the
properties tested here are the ones that decide whether 200 GPU-hours of work
survives being interrupted twenty times.

Three claims, each of which failed at least once in a real run:

  A partial write is never observed. `torch.save` straight to last.pt leaves a
  truncated file when the process dies mid-write, and the next session then
  cannot start at all.

  A damaged checkpoint does not end the run. Falling back to best.pt costs
  steps; refusing to open anything costs the session.

  Saving is driven by a clock as well as a step count, because the events that
  kill these runs are measured in minutes.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from taiko.train import CheckpointSaver, SaveTrigger, atomic_save, load_checkpoint


def test_atomic_save_leaves_no_partial_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sub" / "last.pt"
        atomic_save({"step": 1, "weights": torch.zeros(1000)}, path)
        assert path.exists() and torch.load(path, weights_only=False)["step"] == 1

        # A second write must replace the first without ever shrinking it in
        # place: the temporary file is the only thing that is ever incomplete.
        atomic_save({"step": 2, "weights": torch.zeros(4000)}, path)
        assert torch.load(path, weights_only=False)["step"] == 2
        assert not list(path.parent.glob("*.tmp*")), "temporary file left behind"
        print("  atomic save              ok")


def test_a_failed_save_leaves_the_previous_checkpoint_intact():
    class Unpicklable:
        def __reduce__(self):
            raise RuntimeError("boom")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "last.pt"
        atomic_save({"step": 7}, path)
        try:
            atomic_save({"step": 8, "bad": Unpicklable()}, path)
        except Exception:                                          # noqa: BLE001
            pass
        assert torch.load(path, weights_only=False)["step"] == 7, \
            "a failed save destroyed the good checkpoint"
        assert not list(path.parent.glob("*.tmp*"))
        print("  failed save is harmless  ok")


def test_load_falls_back_when_last_is_damaged():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        atomic_save({"step": 100}, out / "best.pt")
        atomic_save({"step": 200}, out / "last.pt")

        ckpt, source = load_checkpoint(out / "last.pt", map_location="cpu")
        assert ckpt["step"] == 200 and source.name == "last.pt"

        (out / "last.pt").write_bytes((out / "last.pt").read_bytes()[:64])
        ckpt, source = load_checkpoint(out / "last.pt", map_location="cpu")
        assert ckpt["step"] == 100 and source.name == "best.pt", "no fallback"
        print("  fallback to best.pt      ok")


def test_load_reports_nothing_rather_than_crashing_on_a_fresh_run():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt, source = load_checkpoint(Path(tmp) / "last.pt", map_location="cpu")
        assert ckpt is None and source is None
        print("  missing checkpoint       ok")


def test_load_raises_when_every_candidate_is_damaged():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        for name in ("last.pt", "best.pt"):
            (out / name).write_bytes(b"not a checkpoint")
        try:
            load_checkpoint(out / "last.pt", map_location="cpu")
        except RuntimeError as exc:
            assert "no loadable checkpoint" in str(exc)
            print("  all candidates damaged   ok")
            return
    raise AssertionError("expected a RuntimeError")


def test_trigger_fires_on_steps_and_on_the_clock():
    trigger = SaveTrigger(every_steps=10, every_minutes=0.0)
    assert not trigger.due(9) and trigger.due(10) == "step"

    # The clock is the reason this class exists: a run saving only every N
    # steps loses everything since the last multiple of N, and N steps is a
    # different number of minutes on every machine it runs on.
    trigger = SaveTrigger(every_steps=0, every_minutes=0.02)   # 1.2 s
    assert not trigger.due(1)
    time.sleep(1.3)
    assert trigger.due(2) == "clock"
    trigger.mark()
    assert not trigger.due(3), "mark() did not reset the clock"
    print("  step and clock triggers  ok")


def test_saver_writes_only_when_due():
    with tempfile.TemporaryDirectory() as tmp:
        state = {"step": 0}
        saver = CheckpointSaver(Path(tmp), lambda: dict(state),
                                SaveTrigger(every_steps=5), verbose=False)
        for step in range(1, 11):
            state["step"] = step
            saver.maybe_save(step)
        assert saver.saves == 2, saver.saves
        assert torch.load(Path(tmp) / "last.pt", weights_only=False)["step"] == 10
        print("  saver honours the trigger ok")


def test_manifest_rehashes_only_what_changed():
    """
    Re-hashing the whole directory on every save read best.pt as well as
    last.pt -- a gigabyte of I/O every ten minutes, on the machine least able
    to spare it, to recompute a digest that could not have changed.
    """
    from taiko.train.session import read_manifest, update_manifest

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        torch.save({"a": 1}, out / "best.pt")
        torch.save({"a": 2}, out / "last.pt")
        update_manifest(out)
        first = read_manifest(out)
        assert set(first) == {"best.pt", "last.pt"}

        # Rewrite last.pt only, and claim as much.
        torch.save({"a": 3}, out / "last.pt")
        update_manifest(out, changed=out / "last.pt")
        second = read_manifest(out)
        assert second["best.pt"] == first["best.pt"], "carried entry was disturbed"
        assert second["last.pt"] != first["last.pt"], "changed file was not re-hashed"

        # A file that changed behind our back is still caught, because size is
        # checked before the cached digest is trusted.
        torch.save({"a": 4, "padding": list(range(4096))}, out / "best.pt")
        update_manifest(out, changed=out / "last.pt")
        third = read_manifest(out)
        assert third["best.pt"] != first["best.pt"], "a resized file kept a stale digest"
        print("  manifest rehashes one file ok")


def test_memory_counts_shared_pages_once():
    """
    The old figure summed RSS over the process tree, so every page a forked
    worker shares with its parent was counted twice. It reported a run at 30 GB
    whose footprint was a fraction of that, and a whole session was spent
    hunting a leak that measurement later showed was not there.
    """
    from taiko.train.session import headroom_mb, memory_line, memory_mb

    m = memory_mb()
    assert m["self"] > 0
    assert m["tree"] >= m["self"]
    # PSS never exceeds RSS: shared pages are divided, not multiplied.
    assert m["self"] <= m["rss"] + 1.0, (m["self"], m["rss"])
    assert headroom_mb() >= 0
    line = memory_line()
    assert "ram" in line and "free" in line, line
    print("  memory is PSS, not RSS sum ok")


if __name__ == "__main__":
    print("checkpointing")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all checkpointing tests passed")
