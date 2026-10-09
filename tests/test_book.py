#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the book's data layer: how a measure of a run is kept
(book_data), where an iteration's frames are, and the helpers the figures
are drawn with (book_figures, book_graph, book_svg).

    python3 tests/test_book.py

A tiny world is run once, into a runs folder of the tests' own, and every
measure that can be taken of a world that short is taken of it. The others
read a settled world of 3,000 iterations and are listed as such, so that a
new measure has to be put on one list or the other.
"""
from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Nothing here may touch the real runs folder: gol_store reads this on import.
os.environ["GOL_RUNS_DIR"] = tempfile.mkdtemp(prefix="gol-tests-")
atexit.register(shutil.rmtree, os.environ["GOL_RUNS_DIR"], True)

import numpy as np

import book_chapters  # noqa: F401  (registers every measure and chapter)
import book_data as D
import book_figures as F
import book_graph
import book_svg
import gol_run
import gol_store
from gol_config import SimConfig

#: Measures that read a settled world — from iteration 500, often to 3,000 —
#: and so cannot be taken of the tiny one. Every other measure is.
SETTLED_ONLY = {"partners", "mobility", "fragile", "growth", "alike3", "lifedims2"}


def _tiny_run(name: str = "tiny", iterations: int = 30, **overrides) -> str:
    settings = dict(total_tokens=3000, n_nodes=60, k_neighbors=6, hidden_layers=[14, 12],
                    message_amount=2, random_input_amount=2, seed=5, checkpoint_every=10)
    settings.update(overrides)
    run_id = gol_store.create_run(name, SimConfig(**settings))["id"]
    gol_run.advance(run_id, until=iterations)
    return run_id


TINY = None


def tiny() -> str:
    global TINY
    if TINY is None:
        TINY = _tiny_run()
    return TINY


# ---------------------------------------------------------------------------
# Keeping a measure
# ---------------------------------------------------------------------------

def test_a_measure_is_made_once_and_kept_until_the_run_moves_on():
    made = []

    @D.measure("test-count")
    def count(run_id):
        made.append(run_id)
        return {"frames": gol_store.count_frames(run_id)}

    run_id = _tiny_run("kept", 6)
    first = count(run_id)
    assert count(run_id) == first and len(made) == 1, "a kept measure was made again"
    assert first["stamp"] == 6 and first["frames"] == 12

    gol_run.advance(run_id, until=8)
    again = count(run_id)
    assert len(made) == 2 and again["stamp"] == 8 and again["frames"] == 16, \
        "a run that moved on was handed the measure of where it was"
    leftovers = [n for n in os.listdir(os.path.join(gol_store.BASE_DIR, ".book")) if ".part" in n]
    assert not leftovers, f"a write left {leftovers} behind"


def test_a_strict_cache_makes_nothing():
    @D.measure("test-strict")
    def nothing(run_id):
        return {}

    os.environ["GOL_BOOK_CACHE"] = "strict"
    try:
        nothing(tiny())
    except D.NotKept:
        pass
    else:
        raise AssertionError("a strict cache made a measure")
    finally:
        del os.environ["GOL_BOOK_CACHE"]
    assert nothing(tiny())["stamp"] == 30


def test_arrays_are_kept_whole_and_the_stamp_comes_back_a_number():
    @D.measure("test-arrays", file="{run}.test-arrays.npz")
    def arrays(run_id):
        return {"table": np.arange(6, dtype=float).reshape(2, 3)}

    made = arrays(tiny())
    kept = arrays(tiny())
    assert np.array_equal(made["table"], kept["table"]) and D._plain(kept["stamp"]) == 30


def test_what_a_measure_is_asked_with_is_part_of_its_key():
    calls = []

    def want(run_id, window=None):
        return {"iteration": gol_store.load_meta(run_id)["iteration"], "window": window}

    def fits(kept, wanted):
        return kept["iteration"] == wanted["iteration"] and wanted["window"] in (None, kept["window"])

    @D.measure("test-window", field="key", want=want, fits=fits)
    def windowed(run_id, window=None):
        calls.append(window)
        return {"asked": window}

    windowed(tiny(), 3)
    windowed(tiny())                       # answered by the copy made with a window
    windowed(tiny(), 3)
    windowed(tiny(), 4)                    # another window: made again
    assert calls == [3, 4], calls


def test_two_measures_cannot_share_a_name():
    def a(run_id):
        return {}

    def b(run_id):
        return {}

    D.measure("test-twice")(a)
    D.measure("test-twice")(a)             # the same function, seen again: as book_figures run as a script
    try:
        D.measure("test-twice")(b)
    except ValueError:
        return
    raise AssertionError("a second measure took a name already taken")


def test_every_measure_the_book_keeps_can_be_taken_of_a_world_or_says_why_not():
    real = {name: m for name, m in D.MEASURES.items() if not name.startswith("test-")}
    assert SETTLED_ONLY <= set(real), f"no such measures: {SETTLED_ONLY - set(real)}"
    for name, m in sorted(real.items()):
        if name in SETTLED_ONLY:
            continue
        made = m.make(tiny())
        assert isinstance(made, dict) and made, f"{name} made nothing of a world"


def test_a_measure_of_decisions_refuses_a_run_that_kept_none():
    run_id = _tiny_run("silent", 4, export_decisions=False)
    try:
        D.MEASURES["kin"].make(run_id)
    except ValueError:
        return
    raise AssertionError("kinship read a run without decisions as if nobody had staked")


# ---------------------------------------------------------------------------
# Where an iteration's frames are
# ---------------------------------------------------------------------------

def test_frame_at_finds_both_phases_of_an_iteration():
    for t in (0, 7, 29):
        for phase in (1, 2):
            frame = D.frame_at(tiny(), t, phase)
            assert (frame["iteration"], frame["phase"]) == (t, phase)
    assert D.last_iteration(tiny()) == 29


def test_frame_at_refuses_a_frame_that_is_not_the_one_asked_for():
    run_id = _tiny_run("misfiled", 4)
    wrong = gol_store.read_frame(run_id, 2)
    gol_store.write_frame(run_id, 4, wrong)               # iteration 1's frame where 2's should be
    try:
        D.frame_at(run_id, 2, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("frame_at handed out the wrong iteration's frame")

    sparse = _tiny_run("sparse", 6, export_every=2)
    assert D.frame_at(sparse, 4, 2)["iteration"] == 4
    try:
        D.frame_at(sparse, 3, 2)
    except KeyError:
        return
    raise AssertionError("an iteration that was never recorded was read")


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def test_classes_say_where_a_value_falls():
    classes = F.Classes((1, 1, "1"), (2, 4, "2–4"), (5, 10 ** 9, "5+"))
    assert classes.labels == ["1", "2–4", "5+"]
    assert [classes.which(v) for v in (0, 1, 3, 4, 70)] == [None, 0, 1, 1, 2]
    assert classes.label_of(3) == "2–4" and classes.label_of(0) is None
    masks = classes.masks(np.array([1, 2, 9]))
    assert [m.tolist() for m in masks] == [[True, False, False], [False, True, False], [False, False, True]]


def test_a_quantile_band_reads_past_gaps_and_drops_thin_points():
    table = np.array([[1.0, 2.0, np.nan], [3.0, np.nan, np.nan], [5.0, 6.0, 7.0]])
    band = F.quantile_band(table, [10, 20, 30])
    assert band["x"] == [10, 20, 30] and band["y"][0] == 3.0 and band["y"][1] == 4.0
    thin = F.quantile_band(table, [10, 20, 30], min_count=2)
    assert thin["x"] == [10, 20], "a point only one world reached was drawn"


def test_the_graph_helpers_agree_on_a_small_graph():
    frame = {"ids": [1, 2, 3, 4, 5], "edges": [[1, 2], [2, 3], [3, 1], [3, 4], [4, 5], [5, 5]]}
    adj = book_graph.graph(frame)
    assert adj[5] == {4}, "a self-loop became a neighbour"
    assert book_graph.degrees(frame["edges"])[5] == 3, "a frame's own count of a self-loop is two"
    assert book_graph.bfs(adj, 1) == {1: 0, 2: 1, 3: 1, 4: 2, 5: 3}
    assert book_graph.bfs(adj, 1, limit=1) == {1: 0, 2: 1, 3: 1}
    assert dict(book_graph.triangles(adj)) == {1: 1, 2: 1, 3: 1}
    assert set(book_graph.core(adj)) == {1, 2, 3}


def test_a_chart_that_says_what_is_not_drawn_is_refused():
    good = {"title": "t", "x": {"label": "x"}, "y": {"label": "y", "min": 0},
            "series": [F.line([1, 2], [3, 4], "a line", 0), F.bars(["a", "b"], [1, 2], 1),
                       F.area([1, 2], [0, 0], [1, 1], 2), F.cells([(0, 1, 0, 1, 5)])],
            "colourbar": {"map": "viridis", "min": 0, "max": 5}}
    assert book_svg.validate(good) == []
    book_svg.render([good])
    bad = dict(good, series=[{"x": [1], "y": [1], "color": 1}])
    try:
        book_svg.render([bad], name="bad")
    except ValueError as e:
        assert "color" in str(e)
        return
    raise AssertionError("a misspelt key was drawn as nothing, without a word")


def _main() -> int:
    import time
    import traceback

    tests = sorted((name, fn) for name, fn in globals().items()
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


if __name__ == "__main__":
    sys.exit(_main())
