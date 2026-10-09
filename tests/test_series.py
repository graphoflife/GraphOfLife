"""
A run's statistics: what a row of them holds, the history a run is charted
from, the families, the flow, the spectral gap and the curvature, and the
page's registry of every one of them.

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

import hashlib
import json
import math
import shutil
import subprocess
import sys

import runner  # first: the repository on the path, and a runs folder of the tests' own

import gol_record
import gol_series
from gol_config import SimConfig
from GraphOfLifeSimple import new_world
from worlds import adjacency, scratch_runs, small, unrecorded_run, write_frames


# ---------------------------------------------------------------------------
# The cache, and the page's registry
# ---------------------------------------------------------------------------

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
    result = subprocess.run(["node", "-e", script, runner.ROOT], capture_output=True, text=True, timeout=60)
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


# ---------------------------------------------------------------------------
# What a row holds
# ---------------------------------------------------------------------------

#: What a row of statistics holds under the current SERIES_VERSION: every key
#: of a heavy row, and which of them are heavy, as a digest. Caches of rows are
#: kept under the version, and the version is bumped by hand — its own comment
#: records a time it was not. Change what a row holds and this digest moves;
#: the test then asks for the version to move with it.
SERIES_ROW_PIN = (23, "cb770dabf276")


def test_the_series_version_moves_with_what_a_row_holds():
    world = new_world(small(seed=3))
    frames = [frame for _ in range(2) for frame in world.step(record_decisions=True)]
    keys = sorted(gol_series.frame_stats(frames[-1], frames[-2], True))
    text = "\n".join(keys) + "\n--heavy--\n" + "\n".join(sorted(gol_series.HEAVY_KEYS))
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    assert (gol_series.SERIES_VERSION, digest) == SERIES_ROW_PIN, (
        f"a row now holds {len(keys)} statistics, {len(gol_series.HEAVY_KEYS)} of them heavy "
        f"(digest {digest}), under SERIES_VERSION {gol_series.SERIES_VERSION}. If what a row "
        f"holds changed, bump SERIES_VERSION and pin both here.")


def test_a_light_pass_leaves_the_costly_statistics_empty_and_the_rest_filled():
    """
    A chart of the node count should not wait on a box-covering dimension.

    Five sixths of what summarising a frame costs goes on statistics that walk
    the graph — bridges, triangles, distance sweeps, two dimension estimates —
    and most charts plot none of them. A light row has to hold the cheap ones
    and carry the rest as explicit nulls: *present and empty*, because a row
    that simply lacked them would be indistinguishable from a run recorded
    before those statistics existed.
    """
    import gol_series

    world = new_world(SimConfig(total_tokens=600, n_nodes=40, k_neighbors=4,
                                seed=9, hidden_layers=[6]))
    frame = world.step(record_decisions=False)[0]

    light = gol_series.frame_stats(frame, None, heavy=False)
    heavy = gol_series.frame_stats(frame, None, heavy=True)

    assert set(light) == set(heavy), "the two passes must produce the same row shape"
    for key in gol_series.HEAVY_KEYS:
        assert key in light, f"{key} must be present in a light row, as a null"
        assert light[key] is None, f"{key} was computed on a light pass"
    assert heavy["bridges"] is not None, "a heavy pass must actually count bridges"

    # HEAVY_KEYS is what the light pass fills with nulls, and it has to name
    # exactly what the heavy pass produces. Since that pass is one function,
    # the list can be checked against it directly rather than inferred from
    # two rows having the same shape.
    produced = set(gol_series._heavy_stats(
        frame, frame["ids"], frame["edges"], frame["tokens"], None))
    assert produced == set(gol_series.HEAVY_KEYS), (
        f"_heavy_stats and HEAVY_KEYS disagree. "
        f"returned but unlisted: {sorted(produced - set(gol_series.HEAVY_KEYS))}; "
        f"listed but not returned: {sorted(set(gol_series.HEAVY_KEYS) - produced)}")
    assert light["nodes"] == heavy["nodes"], "the cheap statistics must agree"
    assert light["gini"] == heavy["gini"]


def test_a_frame_is_compared_only_with_the_phase_it_started_from():
    """
    A frame recorded before deltas were tracked has its change worked out
    from the frame before, as long as that frame is the state its phase began
    in: (it, 1) before (it, 2), and (it - 1, 2) before (it, 1). On a run that
    records every Nth iteration the frame before a reproduction frame is N
    iterations back. Every caller used to settle this from the run's
    configuration and the frame indices, and the page's Diagrams never did.
    """
    import gol_series

    world = new_world(small(seed=12))
    frames = []
    for _ in range(4):
        frames.extend(world.step(record_decisions=False))
    older = [{k: v for k, v in f.items() if k != "delta"} for f in frames]

    game, its_start = older[5], older[4]          # (2, 2) and (2, 1)
    reproduction, last_game = older[4], older[3]  # (2, 1) and (1, 2)
    for frame, start in ((game, its_start), (reproduction, last_game)):
        assert gol_series.starts(start, frame)
        worked_out = gol_series.frame_stats(frame, start, heavy=False)
        recorded = gol_series.frame_stats(frames[older.index(frame)], None, heavy=False)
        for key in ("gainers", "losers", "maxTokenAdded", "maxTokenLost"):
            assert worked_out[key] == recorded[key] is not None, (frame["phase"], key)

    # Two iterations back is not where anything started.
    assert not gol_series.starts(older[1], game)
    assert gol_series.frame_stats(game, older[1], heavy=False)["gainers"] is None
    assert gol_series.frame_stats(reproduction, older[1], heavy=False)["gainers"] is None


# ---------------------------------------------------------------------------
# Loading a run's history at increasing resolution
# ---------------------------------------------------------------------------

def test_bisection_visits_the_ends_first_and_then_keeps_halving():
    """
    Both ends, then the middle, then the middles of the halves.

    The order has to be a permutation — every sample reached exactly once —
    and it has to *start* wide, because the whole point is that a partial
    answer spans the entire run rather than describing its first few percent.
    """
    import gol_series

    order = gol_series.bisection_order(9)
    assert sorted(order) == list(range(9)), f"not a permutation: {order}"
    assert order[:3] == [0, 8, 4], f"ends then middle, got {order[:3]}"
    assert sorted(order[:5]) == [0, 2, 4, 6, 8], (
        f"five points should be evenly spread, got {sorted(order[:5])}")

    for count in (0, 1, 2, 3, 300):
        got = gol_series.bisection_order(count)
        assert sorted(got) == list(range(count)), f"{count} gave {got}"


def test_every_prefix_of_the_order_contains_the_one_before_it():
    """
    Nesting is what makes the climb free.

    Each pass must only pay for samples the previous pass did not take. If the
    prefixes were not nested, asking for five points after three would recompute
    work already done, and a progressive load would cost more than a single one
    rather than the same.
    """
    import gol_series

    order = gol_series.bisection_order(64)
    for size in range(2, len(order)):
        assert set(order[:size - 1]) < set(order[:size]), (
            f"prefix of {size} does not contain the prefix of {size - 1}")


def test_a_coarse_request_spans_the_whole_run_and_a_finer_one_refines_it():
    """
    Asking for fewer points must not mean asking for less of the run.

    The failure this guards against is the obvious implementation — take the
    first N samples — which draws a chart of the beginning of the run and calls
    it a chart of the run. A coarse pass has to reach the last iteration, and
    every later pass has to be a superset of the earlier one.
    """
    with scratch_runs():
        run_id, _, _ = unrecorded_run(40, seed=3)

        coarse = gol_record.build_series(run_id, points=3)
        finer = gol_record.build_series(run_id, points=5)
        whole = gol_record.build_series(run_id)

        last = whole["series"]["iteration"][-1]
        assert coarse["series"]["iteration"][0] == 0
        assert coarse["series"]["iteration"][-1] == last, (
            "a coarse pass must still reach the end of the run")
        assert not coarse["complete"] and whole["complete"]
        assert coarse["done"] < finer["done"] <= whole["done"]

        of = lambda r: set(r["series"]["iteration"])
        assert of(coarse) < of(finer) <= of(whole), (
            "each pass must contain the one before it")


def test_a_heavy_pass_upgrades_the_rows_a_light_one_left_behind():
    """
    The two passes share one cache, so the second has to fill in the first.

    The failure this guards against is the cache counting a light row as
    already done: the expensive statistics would then never be computed for any
    frame the cheap pass reached first, and the chart would be permanently
    empty with no sign that anything was missing.
    """
    with scratch_runs():
        run_id, _, _ = unrecorded_run(20)

        light = gol_record.build_series(run_id, heavy=False)
        assert light["heavy"] is False
        assert all(v is None for v in light["series"]["bridges"])

        heavy = gol_record.build_series(run_id, heavy=True)
        assert heavy["heavy"] is True
        assert all(v is not None for v in heavy["series"]["bridges"]), (
            "the heavy pass did not upgrade the rows the light pass stored")
        assert heavy["count"] == light["count"], "the upgrade duplicated points"

        # And going back to a light request keeps what the heavy pass found,
        # rather than throwing the expensive work away again.
        back = gol_record.build_series(run_id, heavy=False)
        assert all(v is not None for v in back["series"]["bridges"])

        assert not [k for k in heavy["keys"] if k.startswith("_")], (
            "bookkeeping fields must not travel with the statistics")


def test_a_reply_is_everything_the_history_knows():
    """
    A coarse request on a summarised run hands back all of it.

    Only the samples asked for used to go back, so the page merged replies
    into what it held — and a page with the whole run drawn cheaply that then
    asked for bridges got two points back. The reply is the history now, and
    the page keeps the latest one.
    """
    with scratch_runs():
        run_id, _, _ = unrecorded_run(20)
        whole = gol_record.build_series(run_id, heavy=False)
        coarse = gol_record.build_series(run_id, points=2, heavy=False)
        assert coarse["count"] == whole["count"], (
            f"a coarse request returned {coarse['count']} of {whole['count']} known rows")
        assert coarse["done"] == coarse["totalPoints"] and coarse["complete"], (
            "a summarised run does not say it is finished, so a climb would not stop")

        # The first deep step summarises two samples and hands back the rest.
        deep = gol_record.build_series(run_id, points=2, heavy=True)
        assert deep["count"] == whole["count"], "a deep step dropped the cheap rows"
        filled = sum(v is not None for v in deep["series"]["bridges"])
        assert 0 < filled < deep["count"], f"{filled} bridge counts after one deep step"
        assert all(v is not None for v in deep["series"]["nodes"]), (
            "a deep step blanked the cheap statistics")
        assert deep["done"] == 2 and deep["complete"] and not deep["heavy"]


def test_a_cheap_row_never_blanks_what_a_deep_one_found():
    """
    The page used to guard this itself, when it merged replies. The history
    does it now: a cheap row for a frame whose graph statistics are already
    known keeps them, and keeps whatever only the cheap row carried.
    """
    history = gol_series.History()
    history.add({"_frame": 0, "_heavy": True, "nodes": 10, "bridges": 3})
    history.add({"_frame": 0, "_heavy": False, "nodes": 10, "bridges": None,
                 "cladesInWindow": 2})
    row = history.rows[0]
    assert row["bridges"] == 3 and row["_heavy"], "a cheap row blanked a bridge count"
    assert row["cladesInWindow"] == 2, "what only the cheap row carried was lost"


def test_the_depth_follows_the_statistics_named():
    """
    The page names what it plots, and the one list of what walks the graph
    decides how deep to summarise.
    """
    assert not gol_series.needs_graph(["nodes", "edges", "tokens"])
    assert gol_series.needs_graph(["nodes", "bridges"])
    assert not gol_series.needs_graph([])


def test_a_history_says_how_big_the_run_was():
    """
    A history finished at one size is not finished at the next.

    The page used to trust a finished history until someone told it to forget
    one, and only the Viewer ever did — Diagrams on a run still going kept
    drawing it as it once was. The reply says which run size it describes, so
    a page that knows the run is bigger now can ask again.
    """
    with scratch_runs():
        run_id, world, written = unrecorded_run(10)
        first = gol_record.build_series(run_id, heavy=False)
        assert first["frames"] == written and first["complete"]

        written = write_frames(run_id, world, written, 10)
        grown = gol_record.build_series(run_id, heavy=False)
        assert grown["frames"] == written and grown["complete"]
        assert max(grown["series"]["iteration"]) > max(first["series"]["iteration"]), (
            "the history of a run that grew did not grow with it")


def test_a_run_is_never_summarised_at_more_than_the_cap():
    """
    However long a run is, the chart is capped.

    Summarising one large frame costs over a second, so an uncapped history is
    hours of work for a chart a few hundred pixels wide.
    """
    import gol_series

    for iterations in (299, 300, 301, 5000, 100000):
        stride = gol_series._sample_stride(iterations)
        assert len(range(0, iterations, stride)) <= gol_series.MAX_SAMPLED_ITERATIONS, (
            f"{iterations} iterations at stride {stride} exceeds the cap")


def test_a_cancelled_series_build_keeps_what_it_finished():
    """
    Stopping a summary nobody is waiting for must not throw its work away.

    The server now stops a build when the browser hangs up, which is what keeps
    abandoned requests from piling up behind the one that is wanted. But the
    build is incremental — each request finishes what the last did not — so a
    build that discarded its partial rows on the way out would make navigating
    back and forth start from nothing every time, and a large run would never
    finish summarising at all. The rows computed before the stop have to reach
    the cache, and the next build has to begin after them.
    """
    with scratch_runs():
        run_id, _, written = unrecorded_run(24)

        # Hang up after three frames: part-way through the second iteration,
        # whose other phase must not be forgotten.
        calls = {"n": 0}
        def hung_up():
            calls["n"] += 1
            return calls["n"] > 3          # asked before every frame

        gol_record.build_series(run_id, heavy=False, cancelled=hung_up)
        kept = len(gol_record._load_cache(run_id).get("rows", []))
        assert 0 < kept < written, (
            f"a cancelled build kept {kept} of {written} rows; it should keep "
            f"what it finished and nothing it did not")

        # The next build carries on from there and completes.
        done = gol_record.build_series(run_id, heavy=False)
        assert done["complete"], "the build after a cancelled one did not finish"
        assert len(gol_record._load_cache(run_id)["rows"]) == written


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------

def test_the_forest_keeps_only_the_phase_asked_for():
    """
    The phase filter lives in gol_lineage.forest now. Both backends used to
    filter before calling it and count what was left after, each its own way.
    """
    import gol_lineage
    frames = [{"iteration": i // 2, "phase": 1 + i % 2,
               "brain_ids": [1, 1, 2], "parent_brain_ids": [-1, -1, 1]} for i in range(6)]
    game = gol_lineage.forest(frames, "2")
    assert game["frames"] == 3, f"{game['frames']} frames kept of the three game frames"
    assert [c["phase"] for c in game["columns"]] == [2, 2, 2]
    assert gol_lineage.forest(frames)["frames"] == 6


def test_families_are_counted_from_ancestry_not_from_one_frame():
    """
    How many families the living divide into needs ancestry, and ancestry is a
    chain: it cannot be read off a single frame and it cannot be sampled.

    `distinctParents` — which was called `distinctLineages` and never counted
    lineages — looks one step back and tracks the population. The windowed
    count looks as far back as the window and does not.
    """
    import gol_series

    window = gol_series.CladeWindow(window=2)

    # A founder, then two children of it, then a grandchild of each. Anchored
    # two iterations back from iteration 3, everything alive descends from the
    # single agent that was alive at iteration 1.
    window.observe(0, [1], [-1])
    window.observe(1, [2], [1])
    window.observe(2, [3, 4], [2, 2])
    window.observe(3, [5, 6], [3, 4])
    assert window.families([5, 6], 3) == 1, "both descend from brain 2, alive at 1"

    # Anchored at the present, everything is its own family.
    assert gol_series.CladeWindow(window=0).families([5, 6], 3) == 2

    # Ancestry beyond the window is dropped rather than kept for ever.
    long_window = gol_series.CladeWindow(window=1)
    for i in range(1, 400):
        long_window.observe(i, [i], [i - 1])
    assert len(long_window.parent) < 40, \
        f"the window is holding {len(long_window.parent)} links; it is not forgetting"


def test_a_genotype_that_outlives_the_window_stays_its_own_family():
    """
    A founder and its child, both alive for ever, are two families however
    long they live. Twice the window after it was first seen, a genotype's
    ancestry is let go — and the founder's was, while it was still alive, so
    the next time it was seen it was taken for a newborn, and its child was
    counted into its family.

    It also made the count depend on where it had started: a count taken up
    after a pause had first seen everybody later, and let them go later. The
    rows of a run that was paused were not the rows of one that was not.
    """
    founder, child = 1, 2

    def frames(until):
        yield 0, [founder], [gol_series.NO_PARENT]
        for t in range(1, until):
            yield t, [founder, child], [gol_series.NO_PARENT, founder]

    window = gol_series.CladeWindow(window=2)
    counts = [window.count({"iteration": t, "brain_ids": ids, "parent_brain_ids": up}, t)
              for t, ids, up in frames(30)]
    assert counts[3:] == [2] * 27, counts

    # A count taken up anywhere, warmed on the frames before as the recorder
    # warms it, goes on exactly as one that never stopped.
    for start in range(1, 25):
        resumed = gol_series.CladeWindow(window=2)
        history = [({"iteration": t, "brain_ids": ids, "parent_brain_ids": up}, t)
                   for t, ids, up in frames(30)]
        resumed.warm((index, frame) for frame, index in history[max(0, start - 4):start])
        again = [resumed.count(frame, index) for frame, index in history[start:]]
        assert again == counts[start:], (start, again, counts[start:])


def test_the_family_count_is_absent_when_the_chain_is_broken():
    """
    A run recorded every other iteration has holes where the links were, and a
    family count computed over holes is a guess. Absent is the honest answer,
    and it is the convention the rest of these statistics already follow.
    """
    import gol_store

    def series_for(export_every):
        with scratch_runs():
            cfg = small(seed=7, export_every=export_every, export_decisions=False)
            meta = gol_store.create_run("x", cfg)
            run_id = meta["id"]
            world = new_world(cfg)
            written = 0
            for iteration in range(14):
                frames = world.step(record_decisions=False)
                if iteration % export_every == 0:
                    for frame in frames:
                        gol_store.write_frame(run_id, written, frame)
                        written += 1
            gol_store.update_meta(run_id, frame_count=written,
                                  iteration=world.iteration)
            return gol_record.build_series(run_id)

    whole = series_for(1)
    assert "cladesInWindow" in whole["keys"], \
        "a fully recorded run should have a family count"
    counts = whole["series"]["cladesInWindow"]
    assert all(c >= 1 for c in counts)
    assert max(counts) > 1, "everything in one family from the first frame is suspicious"

    sampled = series_for(2)
    assert "cladesInWindow" not in sampled["keys"], \
        "a sampled run cannot have its ancestry rebuilt and must not pretend to"


# ---------------------------------------------------------------------------
# Lightning: circulating token flow
# ---------------------------------------------------------------------------

def _flow(triples):
    """A frame carrying just the allocations, from (source, target, tokens)."""
    by = {}
    for source, target, tokens in triples:
        by.setdefault(source, {"agent": source, "targets": [], "alloc": []})
        by[source]["targets"].append(target)
        by[source]["alloc"].append(tokens)
    return {"decisions": {"allocations": list(by.values())}}


def test_a_long_loop_is_worth_far_more_than_short_ones():
    """
    The whole point of squaring the hops, checked against the same tokens.

    Ten tokens round one ten-hop ring and ten tokens round three triangles are
    the same amount of flow. If the score did not distinguish them it would be
    measuring volume rather than structure, which is what `cyclingShare`
    already does.
    """
    import gol_lightning

    ring = gol_lightning.lightning(_flow([(i, (i + 1) % 10, 1) for i in range(10)]))
    triangles = []
    for base in (0, 3, 6):
        triangles += [(base, base + 1, 1), (base + 1, base + 2, 1), (base + 2, base, 1)]
    tri = gol_lightning.lightning(_flow(triangles))

    assert ring["lightningScore"] == 100, f"one ten-hop loop is 10², got {ring['lightningScore']}"
    assert tri["lightningScore"] == 27, f"three three-hop loops is 3x3², got {tri['lightningScore']}"
    assert ring["lightningLongest"] == 10 and tri["lightningLongest"] == 3
    # Both are entirely circulating: the difference is shape, not volume.
    assert ring["cyclingShare"] == 1.0 and tri["cyclingShare"] == 1.0


def test_more_tokens_round_the_same_ring_is_more_lightnings():
    """Two tokens on every edge of a ten-ring is two loops, not one heavier."""
    import gol_lightning

    got = gol_lightning.lightning(_flow([(i, (i + 1) % 10, 2) for i in range(10)]))
    assert got["lightningScore"] == 200, got["lightningScore"]
    assert got["lightningLongest"] == 10


def test_one_way_transport_has_no_lightning_and_shows_as_imbalance():
    """
    A line has nothing to find, and says so twice over.

    The score is zero because there is no loop, and the imbalance is above zero
    because conservation forbids one — which is the half of the answer that
    needs no heuristic and is what makes the zero trustworthy.
    """
    import gol_lightning

    got = gol_lightning.lightning(_flow([(i, i + 1, 5) for i in range(6)]))
    assert got["lightningScore"] == 0
    assert got["cyclingShare"] == 0
    assert got["flowImbalance"] > 0, "a line strands flow at both ends"


def test_netting_removes_sloshing_and_keeps_real_circuits():
    """
    The distinction the net measure exists to make.

    Two neighbours trading tokens both ways is a two-hop loop under the gross
    reading and nothing at all once reciprocity is cancelled — which is the
    honest answer, since nothing went anywhere. A genuine ring is untouched.
    And a ring buried in reciprocal noise keeps exactly the ring.
    """
    import gol_lightning

    sloshing = gol_lightning.lightning(_flow([(0, 1, 5), (1, 0, 3)]))
    assert sloshing["lightningScore"] == 12, "three tokens go round and back"
    assert sloshing["netLightningScore"] == 0, "but none of it went anywhere"
    assert sloshing["netFlowShare"] == 0.25, "two of eight tokens survive"

    ring = gol_lightning.lightning(_flow([(i, (i + 1) % 10, 1) for i in range(10)]))
    assert ring["netLightningScore"] == ring["lightningScore"] == 100
    assert ring["netFlowShare"] == 1.0, "nothing to cancel in a one-way ring"

    noisy = _flow([(i, (i + 1) % 10, 3) for i in range(10)]
                  + [((i + 1) % 10, i, 2) for i in range(10)])
    buried = gol_lightning.lightning(noisy)
    assert buried["lightningScore"] == 500, buried["lightningScore"]
    assert buried["netLightningScore"] == 100, (
        "netting should leave the ring and nothing else, got "
        f"{buried['netLightningScore']}")
    assert buried["netLightningLongest"] == 10


def test_the_bracket_never_closes_the_wrong_way_round():
    """
    Circulation cannot exceed what conservation permits.

    `cyclingShare` is a lower bound found by searching and `flowImbalance` an
    exact upper bound found by counting, so the first can never be larger than
    one minus the second. If it were, one of the two would be wrong, and there
    would be no telling which.
    """
    import gol_lightning

    world = new_world(SimConfig(total_tokens=800, n_nodes=40, k_neighbors=4,
                                seed=6, hidden_layers=[6]))
    seen = 0
    for _ in range(12):
        for frame in world.step(record_decisions=True):
            got = gol_lightning.lightning(frame)
            if got["lightningScore"] is None:
                continue          # a reproduction phase moves nothing on links
            seen += 1
            assert got["cyclingShare"] <= 1 - got["flowImbalance"] + 1e-9, (
                f"circulating {got['cyclingShare']} but only "
                f"{1 - got['flowImbalance']} is permitted")
            assert 0 <= got["cyclingShare"] <= 1
            assert got["lightningScore"] >= 0
            # Netting cannot invent circulation, and cannot change a balance,
            # so the same ceiling holds and the net score is never the larger.
            assert got["netCyclingShare"] <= 1 - got["flowImbalance"] + 1e-9
            assert got["netLightningScore"] <= got["lightningScore"]
            assert got["netLightningLongest"] <= got["lightningLongest"]
    assert seen >= 6, f"only {seen} game phases carried any flow"


# ---------------------------------------------------------------------------
# The spectral gap: how hard the graph is to cut in half
# ---------------------------------------------------------------------------

def test_the_spectral_gap_is_exact_on_a_complete_graph():
    """
    The one family with an answer in closed form.

    Every complete graph has normalised-Laplacian eigenvalues 0 and n/(n-1),
    repeated. Anything that is not that is not a spectral gap, and since this
    is found by iteration rather than by solving, it is worth pinning against
    a case where the true value is known to every digit.
    """
    import gol_spectral

    for n in (4, 6, 10, 20):
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        got = gol_spectral.spectral_gap(list(range(n)), adjacency(edges))
        exact = n / (n - 1)
        assert abs(got - exact) < 1e-12, f"K{n} gave {got}, wanted {exact}"


def test_a_ring_matches_its_closed_form_too():
    """
    A cycle of n has λ₂ = 1 - cos(2π/n), which is small and gets smaller.

    Worth having beside the complete graph because it is the opposite end: a
    ring is about as easy to cut as a connected graph gets, so the two together
    pin both ends of the range rather than one point in it.

    A ring is also the case that decided the algorithm. Its eigenvalues crowd
    together as it grows, and power iteration separates them at a rate set by
    the space between — so an 80-ring defeated it entirely while Lanczos is
    exact here to the last bit. That is the same crowding real graphs in this
    substrate have, which is why the slow ones are in this test.
    """
    import gol_spectral

    for n in (12, 20, 40, 80):
        ring = [(i, (i + 1) % n) for i in range(n)]
        got = gol_spectral.spectral_gap(list(range(n)), adjacency(ring))
        exact = 1 - math.cos(2 * math.pi / n)
        assert abs(got - exact) < 1e-12, f"a {n}-cycle gave {got}, wanted {exact}"


def test_a_cheap_cut_shows_up_as_a_small_gap():
    """
    Two cliques joined by one edge against one clique of the same size.

    This is the whole point of the measure: both graphs are dense, both are
    connected, and only one of them can be split without cost. A count of
    bridges says "one" for the dumbbell and nothing about how much that bridge
    is holding together.
    """
    import gol_spectral

    half = [(i, j) for i in range(6) for j in range(i + 1, 6)]
    other = [(i + 6, j + 6) for i, j in half]
    dumbbell = half + other + [(0, 6)]
    whole = [(i, j) for i in range(12) for j in range(i + 1, 12)]

    weak = gol_spectral.spectral_gap(list(range(12)), adjacency(dumbbell))
    strong = gol_spectral.spectral_gap(list(range(12)), adjacency(whole))

    assert weak < 0.2, f"a dumbbell should cut cheaply, got {weak}"
    assert strong > 1.0, f"a complete graph should not cut at all, got {strong}"
    assert weak < strong / 5, "the two should not be anywhere near each other"


def test_the_gap_ignores_a_stray_component_rather_than_reporting_zero():
    """
    Read on the largest component, deliberately.

    The textbook definition gives exactly zero for a disconnected graph, which
    is true and useless here: one pair of agents adrift would answer for the
    whole population and the number would be zero forever. Cleanup keeps only
    the largest component anyway, so this is the reading that means something.
    """
    import gol_spectral

    core = [(i, j) for i in range(8) for j in range(i + 1, 8)]
    ids = list(range(10))
    adrift = adjacency(core + [(8, 9)])
    adrift.setdefault(8, set()).add(9)

    got = gol_spectral.spectral_gap(ids, adrift)
    alone = gol_spectral.spectral_gap(list(range(8)), adjacency(core))
    assert abs(got - alone) < 1e-9, (
        f"the stray pair changed the answer: {got} against {alone}")


# ---------------------------------------------------------------------------
# Curvature: the term the dimension fit was throwing away
# ---------------------------------------------------------------------------

def test_curvature_recovers_the_ricci_scalar_it_was_given():
    """
    Feed the model in, get the parameter back.

    The shell of a ball of radius r in d dimensions with Ricci scalar R goes as
    r^(d-1) (1 - R r^2 / 6d), so log shell is linear in log r and in r^2. Build
    exactly that and the fit has to return the R that built it, or the algebra
    converting the second coefficient is wrong.
    """
    import math
    import gol_series

    d, R = 3.0, 0.6
    rs = [1.0, 2.0, 3.0, 4.0, 5.0]
    log_r = [math.log(r) for r in rs]
    shell = [2.5 + (d - 1) * math.log(r) - R * r * r / (6 * d) for r in rs]

    got = gol_series._ball_curvature(rs, log_r, shell)
    assert abs(got - R) < 1e-9, f"expected {R}, got {got}"


def test_a_ring_is_flat_and_a_tree_is_negatively_curved():
    """
    The two cases with an answer known in advance.

    A ring's shells never change size, so it has no bend at all and reads flat.
    A branching tree's shells grow exponentially — far more room than flat
    space allows — which is negative curvature, and is the same statement as a
    tree being hyperbolic. Sparse expanders are negatively curved as a theorem
    (Salez 2021), so the sign here is also a reading on expansion.
    """
    import gol_series

    ring = [[i, (i + 1) % 60] for i in range(60)]
    flat = gol_series._structure(list(range(60)), ring)["ricciCurvature"]
    assert flat is not None, "a ring gives enough radii to fit"
    assert abs(flat) < 1e-6, f"a ring should read flat, got {flat}"

    # Balanced binary tree, deep enough for the ball to keep growing.
    tree = [[(i - 1) // 2, i] for i in range(1, 255)]
    curved = gol_series._structure(list(range(255)), tree)["ricciCurvature"]
    assert curved is not None and curved < 0, (
        f"a branching tree should read negatively curved, got {curved}")


def test_curvature_refuses_a_fit_with_no_evidence_in_it():
    """
    Three points fit three parameters exactly, which is not a measurement.

    The residual would be zero whatever the data said, so the curvature would
    be a restatement of the input rather than a reading of it.
    """
    import math
    import gol_series

    rs = [1.0, 2.0, 3.0]
    assert gol_series._ball_curvature(rs, [math.log(r) for r in rs],
                                      [1.0, 2.0, 2.5]) is None


if __name__ == "__main__":
    sys.exit(runner.main(globals()))
