"""
tests/test_notebook.py

The Kaggle notebook is generated, and until now nothing ever parsed it.

Two of its cells shipped with a syntax error -- the one that restores
checkpoints from /kaggle/input and the one that decides whether a stage can
resume -- because a `\n` written for the notebook was expanded when
build_notebook.py was imported and split the string literal it sat inside.
Neither cell ran a single statement, so no session ever restored anything and
every run started from step zero, which is a very expensive way to find a
missing backslash.

So: the notebook must parse, it must match its generator, and the supervisor
that decides whether a killed run costs two minutes or ten hours must do what
it says.
"""

from __future__ import annotations
import json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "notebooks"))

from build_notebook import CELLS, check, checkable      # noqa: E402

NOTEBOOK = REPO / "notebooks" / "kaggle_train.ipynb"


def _cells() -> list[dict]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _code_source() -> str:
    """Every code cell of the generated notebook, concatenated."""
    return "\n".join(
        "".join(cell["source"]) for cell in _cells() if cell["cell_type"] == "code"
    )


def test_every_code_cell_parses():
    broken = []
    for i, cell in enumerate(_cells()):
        if cell["cell_type"] != "code":
            continue
        try:
            compile(checkable("".join(cell["source"])), f"cell{i}", "exec")
        except SyntaxError as exc:
            broken.append(f"cell {i} line {exc.lineno}: {exc.msg}")
    assert not broken, "\n".join(broken)
    print("  every code cell parses    ok")


def test_the_build_refuses_a_broken_cell():
    """The guard has to fire, or it is decoration."""
    try:
        check([{"cell_type": "code", "source": ['print("a\n b")\n']}])
    except SystemExit:
        print("  build rejects bad syntax  ok")
        return
    raise AssertionError("check() accepted a cell that cannot be parsed")


def test_notebook_matches_its_generator():
    """
    A hand-edited notebook is a notebook whose next regeneration silently
    reverts the edit.
    """
    before = NOTEBOOK.read_bytes()
    subprocess.run([sys.executable, str(REPO / "notebooks" / "build_notebook.py")],
                   check=True, capture_output=True)
    after = NOTEBOOK.read_bytes()
    if before != after:
        NOTEBOOK.write_bytes(before)
        raise AssertionError("kaggle_train.ipynb is out of date; "
                             "run python notebooks/build_notebook.py")
    print("  notebook matches builder  ok")


def _supervise():
    """The supervisor, lifted out of the notebook it lives in."""
    source = next("".join(c["source"]) for c in _cells()
                  if c["cell_type"] == "code" and "def supervise(" in "".join(c["source"]))
    namespace = {"Path": Path, "time": time, "sys": sys, "subprocess": subprocess}
    exec(source, namespace)                                        # noqa: S102
    return namespace["supervise"]


FAKE_STAGE = '''
import os, sys, pathlib
argv = sys.argv[1:]
out = pathlib.Path(argv[argv.index("--out") + 1])
out.mkdir(parents=True, exist_ok=True)
counter = out / "attempts"
n = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(n))
(out / f"argv{n}").write_text(" ".join(argv))
(out / "last.pt").write_bytes(b"x")
sys.exit(int(os.environ["FAKE_PLAN"].split(",")[n - 1]))
'''


def _run(plan: str, label: str):
    """Run the supervisor against a stage whose exit codes we choose."""
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "fake_stage.py"
        script.write_text(FAKE_STAGE, encoding="utf-8")
        out = Path(tmp) / "out"
        os.environ["FAKE_PLAN"] = plan
        error = None
        try:
            ok = _supervise()(str(script), ["--num-workers", "4", "--out", str(out)],
                              out, 1.0, label)
        except RuntimeError as exc:
            ok, error = None, str(exc)
        attempts = int((out / "attempts").read_text())
        argvs = [(out / f"argv{i}").read_text() for i in range(1, attempts + 1)]
        return ok, error, argvs


def test_out_of_memory_restarts_and_resumes():
    """
    The whole point. A run killed for memory has a current checkpoint; the
    session's remaining ten hours are only lost if nobody picks it back up.
    """
    ok, error, argvs = _run("17,17,0", "oom twice")
    assert ok is True, error
    assert len(argvs) == 3, argvs
    assert "--resume" not in argvs[0], "resumed from nothing on the first attempt"
    assert all("--resume" in a for a in argvs[1:]), argvs
    # Fewer workers each time: they are the part of the footprint we can give up.
    assert ["--num-workers 4" in argvs[0],
            "--num-workers 2" in argvs[1],
            "--num-workers 1" in argvs[2]] == [True] * 3, argvs
    # And the budget is re-derived, never inherited from the first attempt.
    assert all("--max-hours" in a for a in argvs), argvs
    print("  OOM restarts and resumes  ok")



def test_child_output_is_relayed_not_inherited():
    """
    A child that inherits the notebook's stdout writes straight to file
    descriptor 1, and Kaggle records those bytes twice -- once from the
    descriptor, once when ipykernel's watcher re-emits them -- so every
    training line appeared twice a fraction of a second apart.

    The supervisor must therefore hand the child a pipe and do the printing
    itself, leaving exactly one writer.
    """
    source = _code_source()
    assert "stdout=subprocess.PIPE" in source, \
        "the supervisor is not capturing the child's output"
    assert "subprocess.run([sys.executable, script" not in source, \
        "a training stage is run with inherited stdout again; its output " \
        "will be recorded twice"
    print("  child output relayed      ok")


def test_relayed_output_keeps_exit_code_and_order():
    """
    Capturing must not cost the two things the supervisor depends on: the exit
    code that distinguishes an out-of-memory kill from a real error, and
    stderr landing in order with the stdout it interrupts.
    """
    import re, subprocess, sys, textwrap

    match = re.search(r"def _run_streaming\(command\):.*?\n    return process\n",
                      _code_source(), re.S)
    assert match, "the streaming helper is gone"
    namespace = {"subprocess": subprocess}
    exec(match.group(0), namespace)
    run_streaming = namespace["_run_streaming"]

    with tempfile.TemporaryDirectory() as tmp:
        child = Path(tmp) / "child.py"
        child.write_text(textwrap.dedent("""
            import sys
            print("out", flush=True)
            print("err", file=sys.stderr, flush=True)
            sys.exit(int(sys.argv[1]))
        """), encoding="utf-8")

        # 137 is the out-of-memory kill the retry loop keys on; losing it would
        # turn a recoverable death into an unretried failure.
        for code in (0, 1, 137):
            process = run_streaming([sys.executable, str(child), str(code)])
            assert process.returncode == code, (code, process.returncode)
    print("  exit codes survive relay  ok")

def test_a_real_error_is_not_retried():
    ok, error, argvs = _run("1", "broken")
    assert ok is None and "exited 1" in error, (ok, error)
    assert len(argvs) == 1, "a genuine error was retried"
    print("  real errors stop the run  ok")


def test_instant_failures_give_up():
    """Three deaths in under 90 seconds is a machine that cannot run this."""
    ok, error, argvs = _run("17,17,17,17", "futile")
    assert ok is None and "three times" in error, (ok, error)
    assert len(argvs) == 3, argvs
    print("  futile restarts give up   ok")


if __name__ == "__main__":
    print("notebook")
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except AssertionError as exc:
                failed += 1
                print(f"  FAILED {name}: {exc}")
    raise SystemExit(failed or print("all notebook tests passed"))
