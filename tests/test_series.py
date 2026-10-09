"""
A run's history, as the series and the page's registry of statistics hold it.

    python3 tests/test_series.py

The page once worked out the statistics of a frame a second time, in
JavaScript, and a parity test held the two copies together. They are now
worked out once, in Python (gol_series, served by gol_framestats); what is
left to check is the series a cache holds across a change of code, and that
every statistic gol_series measures has a name, a place and a meaning in the
page's registry. The registry is read with node; without it this says so and
passes, because a missing tool is not a failing test.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# A runs folder of the tests' own, which nothing here should need, but one
# test that forgot would otherwise write into the live runs folder.
os.environ["GOL_RUNS_DIR"] = tempfile.mkdtemp(prefix="gol-tests-")
atexit.register(shutil.rmtree, os.environ["GOL_RUNS_DIR"], True)

import gol_series
from gol_config import SimConfig
from GraphOfLifeSimple import new_world

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _frames(count: int = 6):
    """A handful of frames with decisions recorded, so every branch is hit."""
    cfg = SimConfig(
        total_tokens=4000, n_nodes=70, k_neighbors=6,
        hidden_layers=[16, 12], message_amount=2, random_input_amount=2,
        allow_handover=True, allow_revolutions=True,
        # Every optional mechanic on, so the keys each one adds are actually
        # compared. With gifting off the gift statistics are absent on both
        # sides and agree by not existing, which is not agreement.
        allow_gifting=True, prune_after="reproduction",
        inactive_window="iteration",
        seed=21,
    )
    world = new_world(cfg)
    collected = []
    for _ in range(count):
        collected.extend(world.step(record_decisions=True))
    return collected


def test_a_cache_written_across_a_code_change_keeps_its_new_statistics():
    """
    A run's cache can hold rows from two versions of gol_series at once: the
    rows already stored when a statistic was added keep their old shape, and
    only frames recorded afterwards carry the new key. The series has to expose
    that key regardless.

    This is a regression test. The keys used to be read off the first row, so a
    statistic added partway through a run was computed, stored, and then
    dropped on the way out — every power-law chart reported no data while the
    numbers sat in the cache file.
    """
    old = {"_frame": 0, "nodes": 10, "edges": 20}
    new = {"_frame": 2, "nodes": 12, "edges": 24, "degreeGamma": 2.4, "boxDimension": 1.8}
    keys = gol_series._series_keys([old, old, new, new])

    for key in ("degreeGamma", "boxDimension"):
        assert key in keys, f"{key} was added partway through and then lost"
    assert "_frame" not in keys, "the frame index is bookkeeping, not a statistic"
    assert set(keys) == {"nodes", "edges", "degreeGamma", "boxDimension"}

    # And the rows that predate it report nothing rather than a wrong number.
    series = {k: [row.get(k) for row in [old, old, new, new]] for k in keys}
    assert series["degreeGamma"] == [None, None, 2.4, 2.4]


# What a frame is, rather than what was measured on it. Every row carries
# these, and none of them is a statistic anyone plots, names or explains.
COORDINATES = {"iteration", "phase", "nodes_before"}


def _registry():
    """RunStats: the page's one list of what each run statistic is called and means."""
    script = (
        "const fs = require('fs');"
        "const RunStats = new Function(fs.readFileSync(process.argv[1] + '/web/js/runstats.js', 'utf8')"
        " + '; return RunStats;')();"
        "const keys = RunStats.keys();"
        "console.log(JSON.stringify({ keys, explained: keys.filter(k => RunStats.explain(k)),"
        " shares: [...RunStats.POPULATION_COUNTS],"
        " derived: Object.values(RunStats.DERIVED).flatMap(d => d.needs) }));"
    )
    result = subprocess.run(["node", "-e", script, ROOT], capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise AssertionError(f"could not read the registry:\n{result.stderr[:2000]}")
    return json.loads(result.stdout)


def test_every_statistic_has_a_name_a_section_and_a_meaning():
    """
    The page names, places and explains each run statistic in one registry,
    web/js/runstats.js. A statistic gol_series gains and the registry misses
    turns up in a chart menu under its raw key and with nothing to say about
    it; one left there after gol_series dropped it is a menu entry that plots
    nothing.
    """
    if shutil.which("node") is None:
        print("node is not installed; skipping")
        return

    registry = _registry()
    measured = set(gol_series.frame_stats(_frames(2)[-1])) - COORDINATES
    named = set(registry["keys"])
    assert named == measured, (
        f"measured but not in the registry: {sorted(measured - named)}; "
        f"in the registry but not measured: {sorted(named - measured)}")
    unexplained = named - set(registry["explained"])
    assert not unexplained, f"named but never explained: {sorted(unexplained)}"
    assert set(registry["shares"]) <= named, "a share of the population that is not a statistic"
    assert set(registry["derived"]) <= named | COORDINATES, "a ratio of something not measured"


if __name__ == "__main__":
    import time
    import traceback

    tests = sorted((n, f) for n, f in globals().items()
                   if n.startswith("test_") and callable(f))
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
    sys.exit(1 if failures else 0)
