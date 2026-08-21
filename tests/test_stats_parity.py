"""
The two statistics implementations must agree.

Forty-odd quantities are computed twice: once in Python by gol_series, which
feeds the charts, and once in JavaScript by stats.js, which feeds the panel
under the graph. Same names, same meanings, two bodies of code. Nobody planned
that — the Python came first for the server, and the JavaScript grew because
the viewer needed numbers for the current frame without asking for them.

Two implementations of the same thing drift, and these had: an optimisation to
the JavaScript edge-flow map quietly began skipping agents that were culled
during cleanup, so the panel and the chart reported different totals for the
same frame and neither looked wrong on its own.

This runs both over identical frames and compares every shared key. It needs
node on PATH; without it, it says so and passes, because a missing tool is not
a failing test.

    python3 tests/test_stats_parity.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gol_series
from gol_config import SimConfig
from GraphOfLifeSimple import new_world

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# stats.js names this one in camel case and gol_series in snake case. A cosmetic
# difference, but it has to be spelled out for the comparison to line up.
ALIASES = {"nodesBefore": "nodes_before"}

# Nothing is exempt. The structural quantities — cycle rank, bridges,
# triangles, transitivity, dimension, radius, diameter, mean path — are
# independent implementations on the two sides rather than one calling the
# other, which makes them the most valuable things here to compare, not the
# least. Sampled estimates are compared with a tolerance; see TOLERANCES.
JS_ONLY: set = set()

# Radius, diameter, mean path and dimension are estimates from sampled sweeps,
# and the two sides do not have to pick the same starting points. They are held
# to being close rather than equal; everything else must match exactly.
TOLERANCES = {
    "radius": 0.35, "diameter": 0.35, "meanPathLength": 0.35, "dimension": 0.35,
}

# The bridge into node. Kept here rather than in a file of its own so the test
# is one thing to read.
DRIVER = r"""
const fs = require('fs');
const [framesPath, root] = process.argv.slice(2);

const sources = ['colormaps.js', 'metrics.js', 'graphstats.js', 'stats.js']
  .map(name => fs.readFileSync(`${root}/web/js/${name}`, 'utf8')).join('\n');
const FrameMetrics = new Function('window', sources + '; return FrameMetrics;')(
  { devicePixelRatio: 1 }
);

const settings = {
  nodeColorBy: 'tokens', nodeSizeBy: 'tokens',
  nodeColorLog: false, nodeSizeLog: false,
  edgeColorBy: 'constant', edgeWidthBy: 'constant'
};

const frames = JSON.parse(fs.readFileSync(framesPath, 'utf8'));
// includeStructure so the panel's own expensive statistics are exercised too,
// even though only the shared ones are compared.
const out = frames.map(frame => new FrameMetrics(frame, settings).summary(true));
process.stdout.write(JSON.stringify(out));
"""


def _frames(count: int = 6):
    """A handful of frames with decisions recorded, so every branch is hit."""
    cfg = SimConfig(
        total_tokens=4000, n_nodes=70, k_neighbors=6,
        hidden_layers=[16, 12], message_amount=2, random_input_amount=2,
        allow_rewire=True, allow_handover=True, allow_revolutions=True,
        seed=21,
    )
    world = new_world(cfg)
    collected = []
    for _ in range(count):
        collected.extend(world.step(record_decisions=True))
    return collected


def _javascript_summaries(frames):
    with tempfile.TemporaryDirectory() as work:
        frames_path = os.path.join(work, "frames.json")
        driver_path = os.path.join(work, "driver.js")
        with open(frames_path, "w") as handle:
            json.dump(frames, handle)
        with open(driver_path, "w") as handle:
            handle.write(DRIVER)

        result = subprocess.run(
            ["node", driver_path, frames_path, ROOT],
            capture_output=True, text=True, timeout=120,
        )
    if result.returncode != 0:
        raise AssertionError(f"the JavaScript side failed:\n{result.stderr[:2000]}")
    return json.loads(result.stdout)


def test_python_and_javascript_agree_on_every_shared_statistic():
    if shutil.which("node") is None:
        print("node is not installed; skipping the parity check")
        return

    frames = _frames()
    from_js = _javascript_summaries(frames)
    from_py = [gol_series.frame_stats(frame) for frame in frames]

    disagreements = []
    compared = 0

    for index, (js, py) in enumerate(zip(from_js, from_py)):
        for key, js_value in js.items():
            if key in JS_ONLY:
                continue
            py_key = ALIASES.get(key, key)
            if py_key not in py:
                continue

            py_value = py[py_key]
            compared += 1

            if js_value is None or py_value is None:
                if js_value != py_value:
                    disagreements.append((index, key, js_value, py_value))
                continue

            if isinstance(js_value, (int, float)) and isinstance(py_value, (int, float)):
                relative = TOLERANCES.get(key, 0.0)
                tolerance = max(1e-9 * max(1.0, abs(float(py_value))),
                                relative * max(1.0, abs(float(py_value))))
                if abs(float(js_value) - float(py_value)) > tolerance:
                    disagreements.append((index, key, js_value, py_value))
            elif js_value != py_value:
                disagreements.append((index, key, js_value, py_value))

    assert compared > 200, f"only {compared} values compared; the bridge is not working"

    if disagreements:
        lines = "\n".join(
            f"    frame {i} {key}: javascript={a!r} python={b!r}"
            for i, key, a, b in disagreements[:15]
        )
        raise AssertionError(
            f"{len(disagreements)} of {compared} values disagree:\n{lines}"
        )


def test_the_two_sides_cover_the_same_ground():
    """
    A statistic added to one side and forgotten on the other shows up here
    rather than as a blank in the interface much later.
    """
    if shutil.which("node") is None:
        print("node is not installed; skipping")
        return

    frames = _frames(2)
    js_keys = set(_javascript_summaries(frames)[-1])
    py_keys = set(gol_series.frame_stats(frames[-1]))

    js_keys = {ALIASES.get(k, k) for k in js_keys}
    missing_in_js = py_keys - js_keys
    assert not missing_in_js, (
        f"the charts compute these but the panel does not: {sorted(missing_in_js)}"
    )


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
