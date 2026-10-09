"""
What makes a test file here runnable on its own: the repository on the path,
a runs folder of the tests' own, and a way to run its tests without pytest.

Every test file imports this before anything of the repository's, and ends

    if __name__ == "__main__":
        sys.exit(runner.main(globals()))

Deliberately dependency-free. A research repository that needs a toolchain
installed before anyone can check it still works is a repository whose tests
do not get run; where pytest is installed, `python3 -m pytest tests/` finds
the same tests.
"""
from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
import time
import traceback

#: The repository, one level up, where the modules under test sit.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Added here rather than left to the caller so that running a file directly
# works from anywhere, which is the whole point of it being runnable directly.
sys.path.insert(0, ROOT)

# A runs folder of the tests' own, read by gol_store when it is imported:
# hence first. Tests point the store at scratch folders as they go, but one
# that forgot would otherwise write into the live runs folder.
os.environ["GOL_RUNS_DIR"] = tempfile.mkdtemp(prefix="gol-tests-")
atexit.register(shutil.rmtree, os.environ["GOL_RUNS_DIR"], True)


def main(namespace: dict) -> int:
    """Run every test_ function in a file's namespace, reporting like pytest would."""
    tests = sorted((name, fn) for name, fn in namespace.items()
                   if name.startswith("test_") and callable(fn))
    failures = []
    started = time.perf_counter()
    for name, fn in tests:
        try:
            fn()
            print(".", end="", flush=True)
        except Exception:
            failures.append((name, traceback.format_exc()))
            print("F", end="", flush=True)
    print(f"\n\n{len(tests) - len(failures)} passed, {len(failures)} failed "
          f"in {time.perf_counter() - started:.1f}s")
    for name, trace in failures:
        print(f"\n--- {name} ---\n{trace}")
    return 1 if failures else 0
