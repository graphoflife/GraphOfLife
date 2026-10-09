"""
Invariants the simulation must not break.

These are properties rather than fixed expected values. A simulation whose
whole point is that nobody knows what it will do cannot be tested by writing
down what it should do — but it can be held to the things that must be true
whatever it does. Tokens are conserved. The
result does not depend on the order agents happen to be visited in.

Every one of these was found by hand while chasing a bug. They are here so
that finding them a second time is the test suite's job rather than someone's
afternoon.

    python3 -m pytest tests/          # if you have pytest
    python3 tests/test_engine.py      # if you do not

The statistics a run records are held to theirs in test_series.py, the two
ways the site is served in test_site.py, and the teaching script in
test_explain_minimal.py.
"""

from __future__ import annotations

import dataclasses
import itertools
import os
import re
import sys
import tempfile

import runner  # first: the repository on the path, and a runs folder of the tests' own

import networkx as nx

import gol_series
from gol_config import SimConfig
from GraphOfLifeSimple import GraphOfLife, make_brain, new_world
from GraphOfLifeSimple import _choose_binary as G_choose_binary
from worlds import adjacency, advanced_run, scratch_runs, small, unrecorded_run, write_frames


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

def test_tokens_are_conserved():
    """
    Nothing creates or destroys tokens unless the configuration says so.

    Reproduction moves them from parent to child, the game moves them between
    neighbours, and cleanup redistributes what the dead leave behind. None of
    that changes the total.
    """
    cfg = small()
    world = new_world(cfg)
    expected = sum(world.tokens.values())

    for _ in range(12):
        for frame in world.step(record_decisions=False):
            assert frame["summary"]["tokens"] == expected, (
                f"total moved to {frame['summary']['tokens']} at iteration "
                f"{frame['iteration']}, phase {frame['phase']}"
            )


def test_tokens_grow_only_when_asked():
    cfg = small(tokens_created_per_phase=5)
    world = new_world(cfg)
    start = sum(world.tokens.values())

    frames = world.step(record_decisions=False)
    assert frames[-1]["summary"]["tokens"] == start + 5 * len(frames)


def test_no_agent_is_ever_in_debt():
    cfg = small()
    world = new_world(cfg)
    for _ in range(10):
        for frame in world.step(record_decisions=False):
            assert min(frame["tokens"]) >= 0


def test_a_pool_with_no_heirs_is_not_reported_as_redistributed():
    """
    The pool is only redistributed if somebody was left to receive it.

    When the last agent dies the pool is dropped and a fresh world's worth is
    minted for the resurrected one instead. Reporting it as redistributed put a
    number on a chart — it is shown in the viewer's General panel and plotted
    per run — for tokens that went nowhere.

    Minting is what makes the pool non-empty here: agents that starve are by
    definition holding nothing, so an emptied world's pool is whatever the
    phase created and not one token more.
    """
    cfg = small(tokens_created_per_phase=5)
    world = new_world(cfg)

    # Everyone broke at once, which is the only route to an empty world.
    for u in list(world.G.nodes()):
        world.tokens[u] = 0
    report = world._cleanup_and_redistribute()

    assert report["resurrected"], "a world emptied of agents should resurrect one"
    assert report["redistributed"] == 0, (
        f"reported {report['redistributed']} tokens redistributed with nobody "
        f"alive to receive them")
    assert sum(world.tokens.values()) == cfg.total_tokens, (
        "the resurrected agent should hold the whole world, minted fresh")


def test_a_pool_with_heirs_is_reported_in_full():
    """
    The other half: what is actually shared out is still counted.

    The estate comes from an agent cut off from the largest component rather
    than a starved one, because only the disconnected die with anything left.
    """
    cfg = small()
    world = new_world(cfg)

    stranded = max(world.G.nodes(), key=lambda u: world.tokens[u])
    world.G.remove_edges_from(list(world.G.edges(stranded)))
    estate = world.tokens[stranded]
    assert estate > 0, "the test needs an agent that dies holding something"

    before = sum(world.tokens.values())
    report = world._cleanup_and_redistribute()

    assert not report["resurrected"], "the rest of the world is still standing"
    assert stranded not in world.G, "an agent on its own island should be culled"
    assert report["redistributed"] == estate, (
        f"reported {report['redistributed']} redistributed, but {estate} was recovered")
    assert sum(world.tokens.values()) == before, "tokens are conserved across cleanup"


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

def test_no_self_loops_survive_a_phase():
    cfg = small(allow_handover=True)
    world = new_world(cfg)
    for _ in range(10):
        world.step(record_decisions=False)
        assert not list(nx.selfloop_edges(world.G))


# ---------------------------------------------------------------------------
# Topology: what a cut would cost
# ---------------------------------------------------------------------------

def test_a_path_is_all_bridges_and_the_middle_one_is_the_worst():
    """
    Every edge of a path splits it, and the worst split is the middle.

    0-1-2-3: cutting the outer edges strands one node, cutting the middle one
    strands two of four. So the worst single cut costs half the population,
    which is the largest a cut can ever cost.
    """
    import GraphOfLifeSimple as G

    edges = [(0, 1), (1, 2), (2, 3)]
    ids, adj = [0, 1, 2, 3], adjacency(edges)

    splits = G.bridge_splits(ids, adj)
    assert len(splits) == 3, f"a path of four has three bridges, got {len(splits)}"
    sides = sorted(min(b, 4 - b) for _, _, b in splits)
    assert sides == [1, 1, 2], f"expected splits of 1, 1 and 2, got {sides}"
    assert G.worst_cut_share(ids, adj) == 0.5


def test_a_cycle_has_no_bridges_at_all():
    """Every edge of a cycle lies on a loop, so nothing can be cut in two."""
    import GraphOfLifeSimple as G

    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    ids, adj = [0, 1, 2, 3], adjacency(edges)

    assert G.bridge_splits(ids, adj) == []
    assert G.worst_cut_share(ids, adj) == 0.0
    # And nothing peels: a cycle is its own 2-core.
    assert G.two_core_size(ids, adj) == 4


def test_two_blobs_on_one_edge_is_the_worst_case_the_thesis_is_about():
    """
    Two triangles joined by a single edge: one bridge, half the world behind it.

    This is the shape `Graphs.md` §6 says is a mass extinction waiting to
    happen — cleanup keeps only the largest component, so cutting that edge
    kills three of six. A bridge *count* cannot tell this apart from a triangle
    with three leaves stuck on it, which also has three bridges and costs one
    node each. That is the whole reason for measuring the split.
    """
    import GraphOfLifeSimple as G

    dumbbell = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3)]
    ids, adj = list(range(6)), adjacency(dumbbell)
    assert len(G.bridge_splits(ids, adj)) == 1
    assert G.worst_cut_share(ids, adj) == 0.5

    fringed = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 4), (2, 5)]
    ids, adj = list(range(6)), adjacency(fringed)
    assert len(G.bridge_splits(ids, adj)) == 3, "same bridge count as the dumbbell"
    assert abs(G.worst_cut_share(ids, adj) - 1 / 6) < 1e-12, (
        "three leaves cost one node each, and the count alone cannot say so")


def test_peeling_leaves_leaves_the_core():
    """
    The 2-core is what survives stripping hanging trees, however deep.

    A triangle with a two-node tail loses both tail nodes, not just the leaf —
    peeling the leaf makes its neighbour a leaf in turn.
    """
    import GraphOfLifeSimple as G

    edges = [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4)]
    ids, adj = list(range(5)), adjacency(edges)
    assert G.two_core_size(ids, adj) == 3

    # A tree has no core at all: peeling never stops until nothing is left.
    edges = [(0, 1), (1, 2), (1, 3), (3, 4)]
    ids, adj = list(range(5)), adjacency(edges)
    assert G.two_core_size(ids, adj) == 0


def test_the_cut_risk_is_recorded_before_the_cull_not_after():
    """
    The engine's own reading has to come from the graph the cull has not touched.

    Measured afterwards it is a consequence of the cull, and the thesis it
    exists to test — that fragility predicts the cull — becomes untestable,
    which is exactly how the first attempt at it stalled.
    """
    world = new_world(SimConfig(total_tokens=600, n_nodes=40, k_neighbors=4,
                                seed=5, hidden_layers=[6]))
    seen = 0
    for _ in range(6):
        for frame in world.step(record_decisions=False):
            risk = frame["cleanup"]["cutRiskBefore"]
            assert risk is not None, "the engine did not record it"
            assert 0.0 <= risk <= 0.5, f"a share of the population, got {risk}"
            seen += 1
    assert seen >= 12, f"both phases of six iterations, got {seen}"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_same_seed_gives_the_same_run():
    """
    A seed has to reach the starting graph as well as the brains. It did not
    for a long time: numpy's seed does not touch networkx, which draws the
    Watts-Strogatz graph from the `random` module, so two seeded runs diverged
    at the very first phase.
    """
    def trace():
        world = new_world(small(seed=99))
        return [(f["summary"]["nodes"], f["summary"]["edges"])
                for _ in range(6) for f in world.step(record_decisions=False)]

    assert trace() == trace()


def test_different_seeds_diverge():
    def trace(seed):
        world = new_world(small(seed=seed))
        return [f["summary"]["nodes"]
                for _ in range(6) for f in world.step(record_decisions=False)]

    assert trace(1) != trace(2)


def test_a_checkpoint_restores_the_whole_world():
    """
    Everything the world is made of has to come back: the graph, the tokens,
    the brains, the messages in flight and the random stream.

    Messages were missing for a long time, so a resumed run began deaf for a
    phase while the run it claimed to continue did not.
    """
    cfg = small(seed=31)
    world = new_world(cfg)
    for _ in range(5):
        world.step(record_decisions=False)

    blob = {k: v.copy() for k, v in world.to_checkpoint().items()}
    restored = GraphOfLife.from_checkpoint(blob, cfg)

    assert set(restored.G.nodes()) == set(world.G.nodes())
    assert (set(map(frozenset, restored.G.edges()))
            == set(map(frozenset, world.G.edges())))
    assert restored.tokens == world.tokens
    assert restored.iteration == world.iteration
    assert restored.next_agent_id == world.next_agent_id
    assert restored.messages == world.messages, "messages in flight were lost"

    for node in world.G.nodes():
        for live, back in zip(world.brains[node].weights, restored.brains[node].weights):
            assert (live == back).all(), "a brain came back different"


def test_resuming_continues_the_same_run():
    """
    A resumed run must be the same run, not a plausible one.

    It was not, for a long time, and nothing about the state was to blame:
    the graph, the tokens, the brains and the random stream all came back
    correctly. What did not was the *order* of each node's neighbours.
    networkx keeps adjacency in insertion order, so a graph rebuilt from an
    edge list presents its neighbours differently from the one it was copied
    from — and the engine reads neighbours as the columns of a matrix, so a
    column meant a different agent and every decision naming a neighbour by
    column landed elsewhere.

    Sorting the neighbours fixed it. This checks the whole point of that.
    """
    cfg = small(seed=31)
    world = new_world(cfg)
    for _ in range(5):
        world.step(record_decisions=False)

    blob = {k: v.copy() for k, v in world.to_checkpoint().items()}
    straight_on = [(f["summary"]["nodes"], f["summary"]["edges"], f["summary"]["tokens"])
                   for _ in range(6) for f in world.step(record_decisions=False)]

    restored = GraphOfLife.from_checkpoint(blob, cfg)
    after_restore = [(f["summary"]["nodes"], f["summary"]["edges"], f["summary"]["tokens"])
                     for _ in range(6) for f in restored.step(record_decisions=False)]

    assert straight_on == after_restore, "the resumed run diverged from the original"


def _recorded(world, iterations):
    """Every frame a world records over some iterations, as canonical JSON."""
    import json
    return [json.dumps(frame, sort_keys=True)
            for _ in range(iterations) for frame in world.step(record_decisions=True)]


def test_two_worlds_stepped_in_turn_record_what_each_records_alone():
    """
    The server runs simulations as threads of one process, and the browser
    runs them in one interpreter. Their worlds drew from numpy's one global
    stream, so a run going beside another could not be reproduced from its
    seed — and starting a run reseeded that stream for every run already
    going. Each world draws from a stream of its own now.
    """
    def binary(seed):
        return small(seed=seed, brain_kind="binary", hidden_layers=[24, 16])

    alone_a = _recorded(new_world(small(seed=41)), 5)
    alone_b = _recorded(new_world(binary(42)), 5)

    a, b = new_world(small(seed=41)), new_world(binary(42))
    together_a, together_b = [], []
    for _ in range(5):
        together_a += _recorded(a, 1)
        together_b += _recorded(b, 1)
        new_world(small(seed=43))            # a third run starting meanwhile

    assert together_a == alone_a, "a float world was moved by the worlds beside it"
    assert together_b == alone_b, "a binary world was moved by the worlds beside it"


def test_the_own_generator_draws_the_global_stream():
    """
    A world's stream is a RandomState seeded the way np.random.seed seeded the
    global one, so a run made before each world owned its stream is the same
    run now, and a checkpoint holding the global stream's state resumes into
    the world's. Checked for every kind of draw the engine makes, ending on an
    odd count of normals, whose spare the state has to carry.
    """
    import numpy as np

    def draws(source):
        return [source.random(), source.random((2, 3)),
                source.normal(0.0, 0.5, size=(3, 2)), source.uniform(-2.0, 2.0, size=(2, 2)),
                source.randint(0, 2, size=5),
                source.choice(np.array([-1, 1], dtype=np.int8), size=4),
                source.choice([5, 9, 13]), source.multinomial(10, [0.25] * 4),
                source.standard_normal((3,))]

    np.random.seed(2024)
    global_draws = draws(np.random)
    global_state = np.random.get_state()
    own = np.random.RandomState(2024)
    own_draws = draws(own)
    own_state = own.get_state()

    for theirs, ours in zip(global_draws, own_draws):
        assert np.array_equal(theirs, ours), (theirs, ours)
    assert np.array_equal(global_state[1], own_state[1])
    assert global_state[2:] == own_state[2:], "the half-drawn normal was not carried"


def test_the_engine_calls_no_module_level_random():
    """
    One call to numpy's module-level random functions would put the global
    stream back into every run, and the coupling between runs sharing a
    process with it. Nothing else would say so: a run with nobody beside it
    is unchanged either way.
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("GraphOfLifeSimple.py", "gol_config.py", "gol_series.py",
                 "gol_lineage.py", "gol_spectral.py", "gol_lightning.py"):
        with open(os.path.join(here, name)) as f:
            found = re.findall(r"np\.random\.(?!RandomState\b)\w+", f.read())
        assert not found, f"{name} draws from the global stream: {found}"


def test_a_checkpoint_carries_the_worlds_own_stream():
    """
    A checkpoint holds the world's stream and a restore puts it back into
    that world alone: neither touches numpy's global stream, and nothing drawn
    by anyone else in between reaches the resumed world.
    """
    import numpy as np

    cfg = small(seed=51)
    world = new_world(cfg)
    _recorded(world, 3)
    blob = {k: v.copy() for k, v in world.to_checkpoint().items()}
    straight_on = _recorded(world, 3)

    np.random.seed(1)
    untouched = np.random.get_state()[1].copy()
    restored = GraphOfLife.from_checkpoint(blob, cfg)
    assert np.array_equal(np.random.get_state()[1], untouched), \
        "restoring a world reseeded the global stream"

    new_world(small(seed=52)).step(record_decisions=False)
    np.random.random(1000)
    assert _recorded(restored, 3) == straight_on, \
        "the resumed world drew from something other than its own stream"


def _what_was_recorded(run_id):
    """A run's frames as canonical JSON, and the bytes of every checkpoint array."""
    import json
    import numpy as np
    import gol_store
    frames = [json.dumps(gol_store.read_frame(run_id, i), sort_keys=True)
              for i in range(gol_store.count_frames(run_id))]
    with np.load(gol_store.checkpoint_path(run_id)) as blob:
        arrays = {key: blob[key].tobytes() for key in blob.files}
    return frames, arrays


def test_a_stopped_and_resumed_run_is_the_run_that_never_stopped():
    """
    A run stopped part-way and taken up again from its checkpoint on disk —
    through the store, the truncation and the loop the server and the lab
    share — records exactly what a run left alone records: every frame, and a
    final checkpoint whose every array, the random stream included, is the
    same. This was only ever checked on a few totals, in memory.
    """
    import gol_run
    import gol_store

    cfg = small(seed=61, checkpoint_every=4)
    with scratch_runs():
        straight = gol_store.create_run("straight", cfg)["id"]
        assert gol_run.advance(straight, until=10) == "stopped"

        paused = gol_store.create_run("paused", cfg)["id"]
        gol_run.advance(paused, until=6)
        assert gol_store.load_meta(paused)["checkpoint_iteration"] == 6
        gol_run.advance(paused, until=10)

        assert _what_was_recorded(paused) == _what_was_recorded(straight)


def test_a_run_cut_between_checkpoints_resumes_to_the_same_run():
    """
    A run that dies between checkpoints — the machine switched off, the
    process killed — leaves frames its checkpoint does not account for. Taken
    up again it drops them, goes back to the checkpoint, lives those
    iterations again, and ends exactly where a run never cut off ends.
    """
    import gol_run
    import gol_store

    class PowerCut(Exception):
        pass

    def at_seven(world):
        if world.iteration == 7:
            raise PowerCut
        return False

    cfg = small(seed=62, checkpoint_every=4)
    with scratch_runs():
        straight = gol_store.create_run("straight", cfg)["id"]
        gol_run.advance(straight, until=10)

        cut = gol_store.create_run("cut", cfg)["id"]
        try:
            gol_run.advance(cut, at_seven)
        except PowerCut:
            pass
        meta = gol_store.load_meta(cut)
        assert (meta["iteration"], meta["checkpoint_iteration"]) == (7, 4), meta
        assert gol_store.count_frames(cut) == 14, "the frames past the checkpoint were not there"

        gol_run.advance(cut, until=10)
        assert _what_was_recorded(cut) == _what_was_recorded(straight)


def test_two_runs_in_two_threads_match_each_alone():
    """
    The server advances every run it is asked to from a thread of one process.
    Two going at once record exactly what each records alone.
    """
    import threading
    import gol_run
    import gol_store

    with scratch_runs():
        made = {seed: [gol_store.create_run(f"{seed}", small(seed=seed))["id"]
                       for _ in range(2)] for seed in (71, 72)}
        for seed, (alone, _) in made.items():
            gol_run.advance(alone, until=6)

        threads = [threading.Thread(target=gol_run.advance, args=(together,),
                                    kwargs={"until": 6})
                   for _, together in made.values()]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        for seed, (alone, together) in made.items():
            assert _what_was_recorded(together) == _what_was_recorded(alone), \
                f"seed {seed} recorded something else beside another run"


def _rows_without_costs(run_id):
    """A run's recorded rows, without what the iteration cost, which differs every time."""
    import gol_record
    return [{k: v for k, v in row.items() if k not in ("_seconds", "_cpu", "_peakMB")}
            for row in gol_record.read_stats(run_id)]


def test_the_recorder_summarises_a_frame_as_the_series_does():
    """
    A row recorded as the run goes is the row the charts would make of that
    frame from disk — every statistic, the graph ones on the iterations they
    are due, and the families — so a chapter and a chart never disagree.
    """
    import gol_record

    with scratch_runs():
        run_id = advanced_run(small(seed=81), 6, {"heavy_every": 2})
        rows = gol_record.read_stats(run_id)
        assert [row["_frame"] for row in rows] == list(range(12))
        assert [row["_heavy"] for row in rows] == [row["iteration"] % 2 == 0 for row in rows]
        assert all("_seconds" in row for row in rows if row["phase"] == 2)

        series = gol_record.build_series(run_id)["series"]
        for row in rows:
            frame = row["_frame"]
            for key, value in row.items():
                if key.startswith("_") or (key in gol_series.HEAVY_KEYS and not row["_heavy"]):
                    continue
                assert series[key][frame] == value, (frame, key, series[key][frame], value)


def test_the_recorder_reads_stored_frames_as_it_reads_live_ones():
    """
    A run recorded afterwards from its stored frames — any run made before
    runs recorded themselves — gets the rows it would have got live, apart
    from what the iterations cost, which only a live run can know.
    """
    import gol_record
    import gol_store

    with scratch_runs():
        live = advanced_run(small(seed=82), 6, {"heavy_every": 3}, "live")
        later = advanced_run(small(seed=82), 6, None, "later")
        assert gol_record.read_stats(later) == []

        assert gol_record.record_stored(later, heavy_every=3) == 12
        assert _rows_without_costs(later) == _rows_without_costs(live)
        assert not any("_seconds" in row for row in gol_record.read_stats(later))
        assert gol_record.record_stored(later, heavy_every=3) == 12, "a second pass added rows"

        # A run from before deltas were kept works its token changes out from
        # the frame before, so one summarised in two sittings has to pick that
        # frame up again where the first sitting stopped.
        old = [advanced_run(small(seed=85), 5, None, name) for name in ("once", "twice")]
        for run_id in old:
            for index in range(gol_store.count_frames(run_id)):
                frame = gol_store.read_frame(run_id, index)
                frame.pop("delta")
                gol_store.write_frame(run_id, index, frame)
        gol_record.record_stored(old[0])
        sitting = iter(range(100))
        gol_record.record_stored(old[1], cancelled=lambda: next(sitting) >= 5)
        assert len(gol_record.read_stats(old[1])) == 5
        gol_record.record_stored(old[1])
        assert gol_record.read_stats(old[1]) == gol_record.read_stats(old[0])


def test_resuming_drops_frames_and_stats_rows_past_the_checkpoint():
    """
    A run cut off between checkpoints has recorded rows for iterations it is
    about to live again, and maybe half of one more line. Taken up again, its
    record is cut back with its frames and continues exactly as an uncut run's
    does — families included, which need the iterations before the cut.
    """
    import gol_record
    import gol_run
    import gol_store

    class PowerCut(Exception):
        pass

    def cut_at(iteration):
        def cut(world):
            if world.iteration == iteration:
                raise PowerCut
            return False
        return cut

    cfg = small(seed=83, checkpoint_every=7)
    with scratch_runs():
        straight = advanced_run(cfg, 24, {"heavy_every": 5}, "straight")

        # Cut between checkpoints, and cut just after one: the second leaves no
        # whole row to drop, only the line being written when it died.
        for when, rows in ((19, 38), (14, 28)):
            cut = gol_store.create_run(f"cut at {when}", cfg, record={"heavy_every": 5})["id"]
            try:
                gol_run.advance(cut, cut_at(when))
            except PowerCut:
                pass
            assert len(gol_record.read_stats(cut)) == rows
            with open(gol_record.stats_path(cut), "a") as f:
                f.write(f'{{"_frame": {rows}, "nodes"')

            gol_run.advance(cut, until=24)
            assert _rows_without_costs(cut) == _rows_without_costs(straight), when
            assert _what_was_recorded(cut) == _what_was_recorded(straight), when


def test_a_checkpoint_timeline_ends_where_its_frames_do():
    """
    Both backends drop the frames a resume abandons, from the same rule: two
    frames for every recorded iteration below the checkpoint's.
    """
    every = SimConfig(export_every=1)
    assert [every.frames_before(i) for i in (0, 1, 5)] == [0, 2, 10]
    sparse = SimConfig(export_every=3)
    assert [i for i in range(10) if sparse.records(i)] == [0, 3, 6, 9]
    assert sparse.frames_before(7) == 6 and sparse.frames_before(6) == 4


def test_recording_decisions_does_not_change_the_run():
    """
    What the agents decided is recorded or not as a run is configured, and
    recording it must not change what they decide: the lab keeps it for every
    run, and a run without it has to be the same run.
    """
    import json

    def recorded(keep):
        world = new_world(small(seed=84))
        return [json.dumps({k: v for k, v in frame.items() if k != "decisions"}, sort_keys=True)
                for _ in range(5) for frame in world.step(record_decisions=keep)]

    assert recorded(True) == recorded(False)


def test_neighbour_order_does_not_depend_on_graph_history():
    """
    The same graph must present the same neighbours in the same order however
    it was built, since that order is what the brain's columns refer to.
    """
    import networkx as nx_local

    grown = nx_local.Graph()
    grown.add_edges_from([(3, 1), (3, 7), (1, 7), (7, 2), (2, 3)])

    # The same graph, assembled in a different order.
    rebuilt = nx_local.Graph()
    rebuilt.add_edges_from([(2, 3), (7, 2), (1, 7), (3, 7), (3, 1)])

    for node in grown.nodes():
        assert sorted(grown.neighbors(node)) == sorted(rebuilt.neighbors(node))

    # Insertion order genuinely differs; sorting is what removes the dependence.
    differs = any(list(grown[n]) != list(rebuilt[n]) for n in grown.nodes())
    assert differs, "this test is not exercising what it claims to"


# ---------------------------------------------------------------------------
# The runs folder
# ---------------------------------------------------------------------------

def test_a_held_run_cannot_be_advanced_twice():
    """
    One run, one advancer. A second — Start pressed twice, or the server and
    the lab reaching for the same run — is refused rather than let loose on
    the same frames, and anyone asking sees the run is going.
    """
    import gol_run
    import gol_server
    import gol_store

    with scratch_runs():
        run_id = gol_store.create_run("x", small(seed=73))["id"]
        assert not gol_store.held(run_id)
        with gol_store.hold(run_id):
            assert gol_store.held(run_id)
            try:
                gol_run.advance(run_id, until=2)
                raise AssertionError("a held run was advanced a second time")
            except gol_store.RunBusy:
                pass
            assert not gol_server.POOL.start(run_id), "the server started a held run"
            assert gol_server.Handler._decorate(gol_store.load_meta(run_id))["running"]
        assert not gol_store.held(run_id)


def test_a_lock_dies_with_its_process():
    """
    A run whose advancer was killed must not stay locked, since nobody is left
    to let go. The system releases the lock when its holder dies, so the run
    is free at once — and the runs folder is wherever GOL_RUNS_DIR says.
    """
    import signal
    import subprocess
    import gol_store

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with scratch_runs() as tmp:
        run_id = gol_store.create_run("x", small(seed=74))["id"]
        script = ("import gol_store, time\n"
                  f"with gol_store.hold({run_id!r}):\n"
                  "    print('held', flush=True)\n"
                  "    time.sleep(60)\n")
        holder = subprocess.Popen([sys.executable, "-B", "-c", script], cwd=here,
                                  env={**os.environ, "GOL_RUNS_DIR": tmp},
                                  stdout=subprocess.PIPE, text=True)
        try:
            assert holder.stdout.readline().strip() == "held"
            assert gol_store.held(run_id), "another process's hold was not seen"
            holder.send_signal(signal.SIGKILL)
            holder.wait(timeout=10)
            assert not gol_store.held(run_id), "a dead holder still held the run"
        finally:
            if holder.poll() is None:
                holder.kill()
            holder.stdout.close()


def test_meta_written_from_two_threads_is_never_torn():
    """
    A running run rewrites its metadata every iteration while the page may
    rename it, both by reading the file, changing it and writing it back.
    Unlocked, the later write carried the earlier read and the rename was
    lost; with one temporary name for both, a write could fail outright.
    """
    import threading
    import gol_store

    with scratch_runs():
        run_id = gol_store.create_run("x", small(seed=75))["id"]

        def write(key):
            for i in range(150):
                gol_store.update_meta(run_id, **{key: i})

        threads = [threading.Thread(target=write, args=(key,)) for key in ("iteration", "name")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        meta = gol_store.load_meta(run_id)
        assert (meta["iteration"], meta["name"]) == (149, 149), meta


def test_a_frame_index_survives_growing_past_five_digits():
    """
    Frame names are zero-padded to five digits, but padding is a minimum.

    At index 100000 the name grows a digit. Reading the index from a fixed
    five-character slice turned that into 10000, which is not a harmless
    misreading: truncating a resumed run walks the directory asking whether
    each frame is at or after the cut, and a frame claiming to be 10000 when it
    is really 100000 is stepped straight over. Frames from a timeline the
    resumed world never lived through would stay on disk, which is the one
    thing the store promises cannot happen.
    """
    import gol_store

    for index in (0, 1, 99999, 100000, 123456, 9999999):
        name = os.path.basename(gol_store.frame_path("GOL_00_00_00_n001", index))
        assert gol_store.frame_index(name) == index, name

    for other in ("checkpoint.npz", "meta.json", "frame_.json.gz",
                  "frame_00001.json.gz.tmp", "frame_abc.json.gz"):
        assert gol_store.frame_index(other) is None, other


def test_a_truncated_resume_removes_frames_past_a_hundred_thousand():
    """The same thing, exercised through the call that actually matters."""
    import gol_store

    with tempfile.TemporaryDirectory() as tmp:
        original = gol_store.BASE_DIR
        gol_store.BASE_DIR = tmp
        try:
            run_id = "GOL_00_00_00_n001"
            os.makedirs(gol_store.frames_dir(run_id))
            kept, cut = 99998, 100001
            for index in (kept, cut):
                with open(gol_store.frame_path(run_id, index), "w") as f:
                    f.write("{}")

            gol_store.truncate_frames_from(run_id, 100000)

            assert os.path.exists(gol_store.frame_path(run_id, kept)), \
                "a frame before the cut was deleted"
            assert not os.path.exists(gol_store.frame_path(run_id, cut)), \
                "a frame past the cut survived the truncation"
        finally:
            gol_store.BASE_DIR = original


def test_copying_a_run_forks_it_rather_than_backing_it_up():
    """
    A duplicate is a run in its own right, starting where the original is.

    Its own id, its own directory, its own creation time — but every frame and
    the checkpoint, so it can be resumed and taken somewhere else while the
    original carries on. Whatever the original was doing, the copy is doing
    nothing: nothing is advancing it.
    """
    import gol_store

    with tempfile.TemporaryDirectory() as tmp:
        original = gol_store.BASE_DIR
        gol_store.BASE_DIR = tmp
        try:
            meta = gol_store.create_run("first world", SimConfig())
            run_id = meta["id"]
            for index in range(4):
                gol_store.write_frame(run_id, index, {"ids": [1, 2], "at": index})
            gol_store.update_meta(run_id, status="running", iteration=42,
                                  frame_count=4, error="something went wrong")

            copy = gol_store.copy_run(run_id)

            assert copy["id"] != run_id, "a copy must not share the original's id"
            assert copy["iteration"] == 42, "the copy should start where the original is"
            assert copy["status"] == "idle" and copy["error"] is None, \
                "nothing is advancing the copy, and it did not inherit the failure"
            assert gol_store.count_frames(copy["id"]) == 4, "the frames did not come along"
            assert gol_store.read_frame(copy["id"], 3) == gol_store.read_frame(run_id, 3)
            assert gol_store.load_meta(run_id)["status"] == "running", \
                "copying changed the original"

            # Independent from here on.
            gol_store.write_frame(copy["id"], 4, {"ids": [1], "at": 4})
            assert gol_store.count_frames(run_id) == 4
            assert gol_store.count_frames(copy["id"]) == 5
        finally:
            gol_store.BASE_DIR = original


def test_a_run_size_is_kept_until_the_run_changes():
    """
    The size is cached against the run's folders, so it has to notice a write.
    A stale size would say a run takes less room than it does.
    """
    import gol_store

    with tempfile.TemporaryDirectory() as tmp:
        original = gol_store.BASE_DIR
        gol_store.BASE_DIR = tmp
        try:
            run_id, world, written = unrecorded_run(3)
            before = gol_store.run_size_bytes(run_id)
            assert gol_store.run_size_bytes(run_id) == before
            write_frames(run_id, world, written, 2)
            after = gol_store.run_size_bytes(run_id)
            walked = sum(os.path.getsize(os.path.join(r, f))
                         for r, _, fs in os.walk(gol_store.run_dir(run_id)) for f in fs)
            assert after == walked > before, (before, after, walked)
        finally:
            gol_store.BASE_DIR = original


# ---------------------------------------------------------------------------
# Brains
# ---------------------------------------------------------------------------

def test_every_brain_kind_runs_and_conserves_tokens():
    """
    The three kinds differ in what a weight is, not in the rules. Whatever a
    brain decides, the economy is still closed.
    """
    for kind in ("float", "float16", "binary"):
        cfg = small(brain_kind=kind, hidden_layers=[24, 20], seed=13)
        world = new_world(cfg)
        expected = sum(world.tokens.values())
        for _ in range(6):
            for frame in world.step(record_decisions=False):
                assert frame["summary"]["tokens"] == expected, f"{kind} lost tokens"
        assert world.G.number_of_nodes() > 0, f"{kind} died out immediately"


def test_weights_stay_in_the_shape_their_kind_promises():
    """A binary brain that quietly grew a 0.5 would not be a binary brain."""
    import numpy as np

    cfg = small(brain_kind="binary", hidden_layers=[24, 20])
    world = new_world(cfg)
    for _ in range(4):
        world.step(record_decisions=False)
    for node in list(world.G.nodes())[:20]:
        for W in world.brains[node].weights:
            assert W.dtype == np.int8
            assert set(np.unique(W)).issubset({-1, 0, 1}), "a weight left -1, 0, 1"

    cfg = small(brain_kind="float16", hidden_layers=[24, 20])
    world = new_world(cfg)
    for _ in range(4):
        world.step(record_decisions=False)
    for node in list(world.G.nodes())[:20]:
        for W in world.brains[node].weights:
            assert W.dtype == np.float16


def test_a_checkpoint_keeps_the_brain_kind():
    """
    Weights come back as the type they were saved as, or a binary brain
    resumes as something else entirely.
    """
    import numpy as np

    for kind, dtype in (("float", np.float64), ("float16", np.float16),
                        ("binary", np.int8)):
        cfg = small(brain_kind=kind, hidden_layers=[24, 20], seed=7)
        world = new_world(cfg)
        for _ in range(4):
            world.step(record_decisions=False)

        blob = {k: v.copy() for k, v in world.to_checkpoint().items()}

        # Run the original on before restoring. Restoring puts the random
        # stream back to where the checkpoint was taken, so doing it first
        # would leave the original consuming the stream the copy needs.
        straight_on = [f["summary"]["nodes"]
                       for _ in range(3) for f in world.step(record_decisions=False)]

        restored = GraphOfLife.from_checkpoint(blob, cfg)
        node = next(iter(restored.G.nodes()))
        assert restored.brains[node].weights[0].dtype == dtype, f"{kind} came back wrong"

        resumed = [f["summary"]["nodes"]
                   for _ in range(3) for f in restored.step(record_decisions=False)]
        assert straight_on == resumed, f"{kind} did not resume the same run"


def test_a_tie_is_not_a_no():
    """
    Integer outputs tie constantly where floats never do. Answering no to every
    tie would be a bias a lineage could not evolve out of, so an undetermined
    maximum falls back to a coin.
    """
    import numpy as np

    rng = np.random.RandomState(3)
    # mode says "take the maximum", and the two sides are equal.
    outcomes = {G_choose_binary(5.0, 5.0, 0.0, 1.0, rng) for _ in range(200)}
    assert outcomes == {True, False}, "a tie always fell the same way"

    # A clear preference is still obeyed exactly.
    assert G_choose_binary(5.0, 1.0, 0.0, 1.0, rng) is True
    assert G_choose_binary(1.0, 5.0, 0.0, 1.0, rng) is False


def test_a_binary_brain_with_gifting_reads_the_at_risk_flag_as_a_flag():
    """
    The binary encoder cuts an observation where the input layout says. It
    used to cut one flag in, always; with gifting there are two, so the
    at-risk flag was laddered as a magnitude and the last magnitude went in as
    a raw number — the same count of rows, so no shape check noticed.
    """
    import numpy as np
    cfg = small(brain_kind="binary", allow_gifting=True)
    brain = make_brain(cfg, 0, np.random.RandomState(0))
    kinds = cfg.input_kinds()
    x = np.zeros((cfg.n_inputs(), 1))
    x[kinds["flag"].stop - 1, 0] = 1.0              # the at-risk flag
    x[kinds["magnitude"].stop - 1, 0] = 5.0         # the last magnitude
    encoded = brain.encode(x)[:, 0]
    assert len(encoded) == cfg.binary_rows()
    assert encoded[:kinds["flag"].stop].tolist() == [0, 1], "the flags pass straight through"
    assert encoded.max() <= 1, "something reached the brain that is not a bit"
    magnitudes = kinds["magnitude"].stop - kinds["magnitude"].start
    last = encoded[kinds["flag"].stop + (magnitudes - 1) * cfg.brain_bits:
                   kinds["flag"].stop + magnitudes * cfg.brain_bits]
    assert last.sum() > 1, "the last magnitude is spread over its ladder"


def test_a_binary_checkpoint_says_how_many_flags_its_brains_read():
    """
    A binary+gifting checkpoint written before the encoder read the layout
    holds weights trained on the old cut, and nothing about their shape says
    so: it is refused. A binary checkpoint without gifting was never wrong,
    and still loads.
    """
    with_gifts = small(brain_kind="binary", allow_gifting=True)
    blob = dict(new_world(with_gifts).to_checkpoint())
    assert int(blob["flags"]) == 2
    del blob["flags"]
    try:
        GraphOfLife.from_checkpoint(blob, with_gifts)
    except ValueError as exc:
        assert "flag" in str(exc), str(exc)
    else:
        raise AssertionError("an old binary+gifting checkpoint was resumed")

    plain = small(brain_kind="binary")
    old = dict(new_world(plain).to_checkpoint())
    del old["flags"]
    GraphOfLife.from_checkpoint(old, plain)


def test_the_form_is_told_the_shape_of_the_brain_it_would_build():
    """
    brain_shape reads an unallocated brain, so a built one has to agree with
    it. The new-run form used to do this arithmetic itself, and missed the
    input and six outputs gifting adds.
    """
    import numpy as np
    from GraphOfLifeSimple import brain_shape, make_brain
    for overrides in ({}, {"brain_kind": "float16"}, {"brain_kind": "binary"},
                      {"allow_gifting": False, "allow_revolutions": False,
                       "allow_handover": False}):
        cfg = SimConfig.for_new_run(**overrides)
        shape = brain_shape(cfg)
        brain = make_brain(cfg, 1, np.random.RandomState(1))
        built = sum(W.size for W in brain.weights) + sum(b.size for b in brain.biases)
        assert shape["weights"] == built, (overrides, shape["weights"], built)
        assert shape["firstLayer"] == brain.weights[0].shape[1], overrides
        assert shape["outputs"] == brain.weights[-1].shape[0], overrides
        assert shape["bytesPerWeight"] == brain.weights[0].dtype.itemsize, overrides


def test_every_brain_kind_has_a_preset_that_validates():
    """
    Choosing a brain should not also mean knowing what else to change.

    A binary brain needs wider layers and a gentler mutation rate, and getting
    the second one wrong kills runs rather than merely making them worse. The
    presets live with the engine so the form cannot drift from them.
    """
    kinds = SimConfig.MODES["brain_kind"]
    assert set(SimConfig.BRAIN_PRESETS) == set(kinds), \
        "every brain kind the engine accepts needs a preset"

    for kind in kinds:
        preset = SimConfig.BRAIN_PRESETS[kind]
        cfg = SimConfig(brain_kind=kind, **preset)
        cfg.validate()
        world = new_world(cfg)
        world.step(record_decisions=False)
        assert sum(world.tokens.values()) == cfg.total_tokens

    binary = SimConfig.BRAIN_PRESETS["binary"]
    floaty = SimConfig.BRAIN_PRESETS["float"]
    assert sum(binary["hidden_layers"]) > sum(floaty["hidden_layers"]), \
        "a binary unit carries a bit where a float carries many; it needs the room"
    assert binary["mutation_sparsity"] < floaty["mutation_sparsity"], \
        "a binary brain's smallest move is a whole step, so its rate must be gentler"


def test_a_binary_brain_spends_no_rows_on_things_that_are_already_bits():
    """
    The ladder is for magnitudes. Nothing else should be on it.

    Every input used to be spread across `brain_bits` thresholds spanning
    roughly -2 to 12, which is right for a logged token count and absurd for a
    value that is only ever 0 or 1: fifteen of its sixteen rows could never
    change. Three hundred of the first layer's eight hundred and sixty-four
    rows were permanently zero, carrying weights that were mutated for the
    whole of a run and could never affect anything.

    Now the is-self flag, the message channels and the noise are one row each,
    and every one of those rows does something.
    """
    import numpy as np

    cfg = small(brain_kind="binary", brain_bits=16, hidden_layers=[24, 16],
                message_amount=5, random_input_amount=5, seed=4)
    world = new_world(cfg)
    for _ in range(4):
        world.step(record_decisions=False)

    magnitudes = cfg.input_kinds()["magnitude"]
    assert cfg.binary_rows() == ((magnitudes.stop - magnitudes.start) * cfg.brain_bits
                                 + cfg.bit_inputs())
    brain = next(iter(world.brains.values()))
    assert brain.layer_sizes()[0] == cfg.binary_rows()

    rows = []
    original = world._observe

    def spy(u, candidates, *rest):
        x = world._inputs(u, candidates, *rest)
        rows.append(world.brains[u].encode(x))
        return original(u, candidates, *rest)

    world._observe = spy
    try:
        world.reproduction_phase(record_decisions=False)
    finally:
        world._observe = original

    encoded = np.hstack(rows)
    assert encoded.shape[0] == cfg.binary_rows()

    # Everything after the ladder is a bit that stands for itself.
    kinds = cfg.input_kinds()
    start = kinds["flag"].stop + (kinds["magnitude"].stop - kinds["magnitude"].start) * cfg.brain_bits
    tail = encoded[start:]
    assert tail.shape[0] == 4 * cfg.message_amount + cfg.random_input_amount
    dead = int((tail.max(axis=1) == tail.min(axis=1)).sum())
    assert dead == 0, f"{dead} message or noise rows can never change"
    assert set(np.unique(tail).tolist()) <= {0, 1}


def test_a_binary_checkpoint_refuses_a_ladder_it_was_not_written_for():
    """
    Splitting a magnitude's rows into a band and a place inside it keeps their
    number and changes what each means, so the shape check cannot see it: the
    binary brain writes its split into the checkpoint and checks it on the way
    back. Weights read under the wrong split would be nonsense that loads.
    """
    import numpy as np
    from GraphOfLifeSimple import GraphOfLife

    cfg = small(brain_kind="binary", seed=5)
    world = new_world(cfg)
    world.step(record_decisions=False)
    blob = world.to_checkpoint()
    assert tuple(int(v) for v in blob["ladder"]) == cfg.ladder_split()
    GraphOfLife.from_checkpoint(dict(blob), cfg)             # its own split loads

    for other in ({k: v for k, v in blob.items() if k != "ladder"},        # before the split
                  {**blob, "ladder": np.array([13, 3], dtype=np.int64)}):  # another split
        try:
            GraphOfLife.from_checkpoint(other, cfg)
        except ValueError as err:
            assert "cannot be carried across" in str(err)
        else:
            raise AssertionError("weights written under another ladder were loaded")

    # A float brain has no ladder, and its checkpoint carries none.
    floaty = small(seed=5)
    float_world = new_world(floaty)
    float_world.step(record_decisions=False)
    float_blob = float_world.to_checkpoint()
    assert "ladder" not in float_blob
    GraphOfLife.from_checkpoint(dict(float_blob), floaty)


def test_the_ladder_starts_where_its_values_start():
    """
    Nothing on the ladder can be negative, so the ladder should not be.

    It began at -2 because the noise and message inputs ran a little under
    zero. They are not on it any more — everything left is log1p of a count or
    a quantile of one — and three of its sixteen rungs sat under zero where
    nothing could ever reach them.
    """
    import numpy as np

    cfg = small(brain_kind="binary", brain_bits=16, hidden_layers=[24, 16], seed=8)
    world = new_world(cfg)
    for _ in range(3):
        world.step(record_decisions=False)

    log_deg, _neighs, q_tok, q_deg, log_tok, _risk = world._precompute_features()
    lowest = np.inf
    for u in sorted(world.G.nodes())[:20]:
        seen = world._inputs(u, [u] + sorted(world.G.neighbors(u)),
                             log_deg, q_tok, q_deg, log_tok, _risk)
        span = seen[cfg.input_kinds()["magnitude"]]
        lowest = min(lowest, float(span.min()))
    assert lowest >= 0.0, f"a laddered input went to {lowest}"

    thresholds = make_brain(cfg, 0).thresholds()
    assert thresholds.min() >= 0.0, \
        f"the ladder starts at {thresholds.min()}, below anything that can reach it"


def test_looking_at_many_candidates_is_looking_at_each_in_turn():
    """
    An agent's inputs are filled a row at a time across all its candidates,
    which is faster than building them one candidate at a time. It must still
    sense exactly what a look at each candidate in turn would, down to the
    noise: that is what keeps a seed giving the run it always gave.
    """
    import numpy as np

    for kind in ("float", "binary"):
        world = new_world(small(brain_kind=kind, allow_gifting=True, seed=4))
        for _ in range(3):
            world.step(record_decisions=False)
        log_deg, _neighs, q_tok, q_deg, log_tok, at_risk = world._precompute_features()
        rest = (log_deg, q_tok, q_deg, log_tok, at_risk)
        u = max(world.G.nodes(), key=lambda n: world.G.degree[n])
        candidates = [u] + sorted(world.G.neighbors(u))

        stream = world.rng.get_state()
        together = world._inputs(u, candidates, *rest)
        world.rng.set_state(stream)
        in_turn = np.column_stack([world._inputs(u, [v], *rest)[:, 0] for v in candidates])
        assert np.array_equal(together, in_turn), \
            f"{kind}: looking at every candidate at once sensed something different"


def test_a_binary_world_says_bits_and_hears_bits():
    """
    Its output layer hands back a count, and squashing that through tanh gave
    a value that was neither a bit nor a useful magnitude — eleven distinct
    values across a whole phase, then read back through a ladder that could
    only see the bottom of it. A binary world's messages are bits, and so is
    its noise.
    """

    cfg = small(brain_kind="binary", brain_bits=16, hidden_layers=[24, 16], seed=6)
    world = new_world(cfg)
    for _ in range(4):
        world.step(record_decisions=False)

    sent = [v for notes in world.messages.values() for vec in notes.values() for v in vec]
    assert sent, "wanted some messages to look at"
    assert set(sent) <= {0.0, 1.0}, f"a binary world sent non-bits: {sorted(set(sent))[:6]}"

    log_deg, _neighs, q_tok, q_deg, log_tok, _risk = world._precompute_features()
    u, v = sorted(world.G.nodes())[:2]
    looks = world._inputs(u, [v] * 40, log_deg, q_tok, q_deg, log_tok, _risk)
    seen = set(looks[-cfg.random_input_amount:].ravel().tolist())
    assert seen <= {0.0, 1.0}, f"a binary world's noise was not bits: {sorted(seen)[:6]}"


def test_the_float_brains_do_not_ladder_anything():
    """The ladder belongs to the binary brain; the others read values whole."""
    import numpy as np

    for kind in ("float", "float16"):
        cfg = small(brain_kind=kind)
        brain = new_world(cfg).brains[sorted(new_world(cfg).G.nodes())[0]]
        assert brain.layer_sizes()[0] == cfg.n_inputs(), \
            f"{kind} should take one row per input"

    cfg = small(brain_kind="float", random_input_amount=6, seed=2)
    world = new_world(cfg)
    features = world._precompute_features()
    u, v = sorted(world.G.nodes())[:2]
    vec = world._inputs(u, [v], features[0], features[2], features[3], features[4],
                        features[5])[:, 0]
    noise = vec[-cfg.random_input_amount:]
    assert not set(noise.tolist()) <= {0.0, 1.0}, \
        "a float world's noise should be a spread of magnitudes, not coins"


def test_a_checkpoint_refuses_a_brain_it_does_not_fit():
    """
    Weights are saved; the shape they were for is not. Loading them into a
    different architecture used to succeed and then die inside a matrix
    multiply several steps later, saying nothing about why.
    """
    import io
    import numpy as np

    cfg = small(brain_kind="binary", brain_bits=16, hidden_layers=[24, 16], seed=5)
    world = new_world(cfg)
    world.step(record_decisions=False)

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **world.to_checkpoint())
    buffer.seek(0)

    wider = small(brain_kind="binary", brain_bits=16, hidden_layers=[40, 16], seed=5)
    with np.load(buffer) as blob:
        try:
            GraphOfLife.from_checkpoint(blob, wider)
        except ValueError as exc:
            assert "shape" in str(exc), exc
        else:
            raise AssertionError("a checkpoint loaded into the wrong architecture")

    buffer.seek(0)
    with np.load(buffer) as blob:
        again = GraphOfLife.from_checkpoint(blob, cfg)
    assert sum(again.tokens.values()) == sum(world.tokens.values())


def test_the_ladder_resolves_a_band_and_a_place_inside_it():
    """
    Two monotone fields beat one, because their resolutions multiply.

    One ladder of sixteen rungs resolves fifteen levels across a range of e^12
    in tokens, which makes a level a factor of 2.23 — an agent could not tell a
    hundred tokens from a hundred and eighty. Splitting the same sixteen rows
    into twelve for the band and four for the place inside it resolves
    thirty-six, and a level becomes a factor of about 1.4.

    The constraint the split has to respect is that a binary unit computes a
    sum of weights in -1, 0 and +1 and thresholds it, and that sum cannot
    weight a row by two to its position. Both fields stay monotone in the
    value, so a plain sum can still find the magnitude — which a place-value
    encoding would not allow, however much more it could in principle say.
    """
    import numpy as np

    cfg = SimConfig(brain_kind="binary", brain_bits=16)
    bands, within = cfg.ladder_split()
    assert bands + within == cfg.brain_bits, "the split must not change the width"
    assert within >= 1 and bands > within

    brain = make_brain(cfg, 0)
    edges = brain.thresholds()
    assert len(edges) == bands
    assert np.allclose(np.diff(edges), brain.band_width()), \
        "the reported edges must be the ones the encoder uses"

    def code(value):
        x = np.zeros((cfg.n_inputs(), 1))
        first = cfg.input_kinds()["magnitude"].start
        x[first, 0] = value
        rows = brain.encode(x)[first:first + cfg.brain_bits, 0]
        return tuple(int(v) for v in rows)

    seen = {code(v) for v in np.linspace(0.0, 12.0, 4000)}
    assert len(seen) >= 30, f"only {len(seen)} distinct codes; the split bought nothing"

    # The band field alone is still a ladder: monotone, and never skipping.
    for value in np.linspace(0.0, 12.0, 400):
        band = code(value)[:bands]
        assert list(band) == sorted(band, reverse=True), \
            f"the band field is not a ladder at {value}"

    # Saturates rather than wrapping.
    assert code(40.0) == code(12.0), "a value above the range must pin at the top"
    assert sum(code(0.0)) < sum(code(11.0)), "the bottom must read lower than the top"


def test_a_split_ladder_stays_readable_by_a_ternary_sum():
    """
    The measurement that decided the design, kept so it cannot quietly rot.

    Random units, because that is what evolution starts from: if the magnitude
    is not in reach of a random ternary sum, mutation has to find it with no
    gradient and no head start. A plain ladder scores about 0.53 and a
    place-value code about 0.19; the split has to stay near the ladder.
    """
    import numpy as np

    rng = np.random.default_rng(7)
    cfg = SimConfig(brain_kind="binary", brain_bits=16)
    brain = make_brain(cfg, 0)

    values = rng.uniform(0.0, 12.0, size=2000)
    x = np.zeros((cfg.n_inputs(), values.size))
    first = cfg.input_kinds()["magnitude"].start
    x[first] = values
    rows = brain.encode(x)[first:first + cfg.brain_bits].T.astype(float)

    draw = rng.random((300, cfg.brain_bits))
    weights = np.zeros_like(draw)
    weights[draw < 1 / 6] = -1
    weights[draw > 5 / 6] = 1

    def rank_corr(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        ra -= ra.mean()
        rb -= rb.mean()
        denom = np.sqrt((ra @ ra) * (rb @ rb))
        return 0.0 if denom == 0 else float((ra @ rb) / denom)

    sums = rows @ weights.T
    readable = np.mean([abs(rank_corr(values, sums[:, u])) > 0.3
                        for u in range(sums.shape[1])])
    assert readable > 0.55, (
        f"only {100*readable:.0f}% of random ternary units can read the magnitude; "
        f"a plain ladder manages about 73% and place value about 26%")

    assert abs(rank_corr(values, rows.sum(axis=1))) > 0.9, \
        "the row count should still track the value"


def test_a_checkpoint_knows_how_it_encoded_its_magnitudes():
    """
    The row count did not change when the ladder was split, and the meaning of
    every row did. Weights written under one split are nonsense under another
    and nothing about their shape says so, so the split is written down.
    """
    import io
    import numpy as np

    cfg = small(brain_kind="binary", brain_bits=16, hidden_layers=[24, 16], seed=5)
    world = new_world(cfg)
    world.step(record_decisions=False)

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **world.to_checkpoint())

    buffer.seek(0)
    with np.load(buffer) as blob:
        assert "ladder" in blob, "a binary checkpoint must record its split"
        stale = {k: blob[k] for k in blob.files if k != "ladder"}

    older = io.BytesIO()
    np.savez_compressed(older, **stale)
    older.seek(0)
    with np.load(older) as blob:
        try:
            GraphOfLife.from_checkpoint(blob, cfg)
        except ValueError as exc:
            assert "ladder" in str(exc) or "encoded" in str(exc), exc
        else:
            raise AssertionError("a checkpoint from before the split loaded silently")

    buffer.seek(0)
    with np.load(buffer) as blob:
        again = GraphOfLife.from_checkpoint(blob, cfg)
    assert sum(again.tokens.values()) == sum(world.tokens.values())


def test_a_brain_id_names_a_genotype_not_an_allocation():
    """
    A copy is the same genotype, so it keeps the same name.

    An id used to be handed out on every copy as well as every mutation, which
    made it an allocation counter: two agents holding byte-identical weights
    were recorded as unrelated, and the id linking one recorded brain to the
    next was itself never recorded, because a copy was mutated immediately and
    only the mutation reached a frame. Half the ids a run created never
    appeared in any frame and the genealogy could not be rebuilt from what was
    written down.

    Now an id changes only where the weights change.
    """
    cfg = small(seed=4)
    world = new_world(cfg)
    source = next(iter(world.brains.values()))

    clone = world._copy_brain(source)
    assert clone.brain_id == source.brain_id, "a copy is the same genotype"
    assert clone.parent_brain_id == source.parent_brain_id, "and the same ancestry"
    assert clone is not source and clone.weights[0] is not source.weights[0]
    assert all((a == b).all() for a, b in zip(clone.weights, source.weights))

    before = world.next_brain_id
    world._copy_brain(source)
    assert world.next_brain_id == before, "copying must not allocate an id"

    # Mutation is the only thing that makes a new genotype, and it records
    # where it came from.
    was = clone.brain_id
    while clone.brain_id == was:
        world._mutate_brain(clone)
    assert clone.parent_brain_id == was
    assert world.next_brain_id > before


def test_copies_of_one_genotype_are_counted_once():
    """
    `distinctBrains` counts genotypes now, so it has to be able to fall.

    While an id was handed out per copy it tracked the population almost
    exactly and could not report diversity at all.
    """
    cfg = small(seed=8)
    world = new_world(cfg)
    for _ in range(8):
        world.step(record_decisions=False)

    agents = world.G.number_of_nodes()
    genotypes = len({b.brain_id for b in world.brains.values()})
    assert genotypes <= agents
    assert genotypes < agents, (
        f"{genotypes} genotypes among {agents} agents — no two agents share "
        f"one, which is what the old allocation-per-copy behaviour looked like")


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

def test_frame_arrays_stay_aligned():
    cfg = small()
    world = new_world(cfg)
    for _ in range(6):
        for frame in world.step(record_decisions=True):
            n = len(frame["ids"])
            for key in ("tokens", "brain_ids", "parent_brain_ids", "parent_ids", "delta"):
                assert len(frame[key]) == n, f"{key} is out of step with ids"


def test_delta_is_the_change_across_the_phase():
    cfg = small()
    world = new_world(cfg)
    previous = dict(world.tokens)   # the phase before the first one is the start
    for _ in range(6):
        for frame in world.step(record_decisions=False):
            for i, node in enumerate(frame["ids"]):
                before = previous.get(node, 0)
                assert frame["delta"][i] == frame["tokens"][i] - before
            previous = dict(zip(frame["ids"], frame["tokens"]))


def test_edges_only_reference_present_nodes():
    cfg = small()
    world = new_world(cfg)
    for _ in range(6):
        for frame in world.step(record_decisions=False):
            present = set(frame["ids"])
            for a, b in frame["edges"]:
                assert a in present and b in present


def test_a_frame_records_the_decisions_its_mechanics_say_it_does():
    """
    Every combination of the mechanics that add or take away a decision, a
    few iterations each: every record holds exactly the keys the engine's own
    statement of the contract, decision_keys, says it does.
    """
    from GraphOfLifeSimple import decision_keys
    for gifting, handover, revolutions, prune in itertools.product(
            (False, True), (False, True), (False, True), ("blotto", "reproduction", "both")):
        cfg = small(allow_gifting=gifting, allow_handover=handover,
                    allow_revolutions=revolutions, prune_after=prune, seed=5)
        expected = decision_keys(cfg)
        world = new_world(cfg)
        seen = set()
        for _ in range(4):
            for frame in world.step(record_decisions=True):
                decisions = frame["decisions"]
                if frame["phase"] == 1:
                    records = [("reproduction", decisions)]
                    records += [("birth", b) for b in decisions["births"]]
                else:
                    records = [("game", decisions)]
                    records += [("allocation", a) for a in decisions["allocations"]]
                    records += [("winner", w) for w in decisions["winners"]]
                for kind, record in records:
                    assert set(record) == expected[kind], (
                        f"{cfg.strain_id()}, prune after {prune}: a {kind} record holds "
                        f"{sorted(record)}, not {sorted(expected[kind])}")
                    seen.add(kind)
        assert seen == set(expected), f"{cfg.strain_id()}: never saw {set(expected) - seen}"


def test_a_frame_names_only_ancestors_that_were_themselves_recorded():
    """
    The genealogy has to be rebuildable from what is written down.

    Every parent a frame names should be a genotype that appeared in an
    earlier frame — otherwise the chain breaks there and the agent looks like
    a founder. A few strays are expected: a genotype made and killed inside one
    iteration never reaches a frame at all.
    """
    cfg = small(seed=6)
    world = new_world(cfg)

    seen = set()
    dangling = 0
    total = 0
    for _ in range(12):
        for frame in world.step(record_decisions=False):
            for brain, parent in zip(frame["brain_ids"], frame["parent_brain_ids"]):
                total += 1
                if parent != -1 and parent not in seen:
                    dangling += 1
            seen.update(frame["brain_ids"])

    share = dangling / max(1, total)
    assert share < 0.05, (
        f"{share:.0%} of the parents named in frames were never recorded "
        f"themselves; the genealogy cannot be rebuilt from the frames")


# ---------------------------------------------------------------------------
# Optional mechanics
# ---------------------------------------------------------------------------

def test_a_mechanic_can_be_switched_off():
    """Switching a rule off changes the brain, so the run must still start."""
    for mechanic in ("allow_handover", "allow_revolutions"):
        world = new_world(small(**{mechanic: False}))
        for _ in range(3):
            world.step(record_decisions=True)
        assert world.G.number_of_nodes() > 0, f"a run with {mechanic} off died immediately"


def test_output_layout_matches_the_configuration():
    for handover in (True, False):
        for revolutions in (True, False):
            cfg = small(allow_handover=handover, allow_revolutions=revolutions)
            world = new_world(cfg)
            node = next(iter(world.G.nodes()))
            rows = world.brains[node].weights[-1].shape[0]
            assert rows == cfg.n_outputs()


def test_statistics_absent_rather_than_zero_when_a_rule_is_off():
    """
    "Not part of these rules" and "allowed but nobody did it" are different
    findings, and a zero cannot tell them apart.
    """
    world = new_world(small(allow_revolutions=False))
    _, game = world.step(record_decisions=True)
    stats = gol_series.frame_stats(game)
    assert stats["revolutions"] is None
    assert stats["revoltShare"] is None


def test_every_agent_in_a_phase_reads_the_same_messages():
    """
    A phase must not let its own writes change what it is reading.

    Messages used to be written straight into the store the observation loop
    was reading from, so an agent saw a mixture: some signals from last phase,
    some written moments earlier in this one, and which it got depended on
    where its id fell in the loop. Seventeen per cent of all reads in a phase
    were of values written during that same phase — low ids systematically
    reading stale signals and high ids fresh ones, for no reason anyone chose.

    Writes now go to an outbox delivered once the phase is over.

    The pre-pass adds a delivery partway through, on purpose — that is the
    whole option — so with it on the rule is per pass rather than per phase:
    within any one sweep of the population, nobody's read changes under them.
    Checked both ways, because the property being protected is that a read
    never depends on where an id fell in a loop, and that holds either way.
    """
    import copy

    for prepass in (False, True):
        world = new_world(small(seed=3, message_prepass=prepass))
        for _ in range(3):
            world.step()

        for run in (world.reproduction_phase, world.blotto_phase):
            baseline = {"at": copy.deepcopy(world.messages)}
            changed = []
            original_input = world._inputs
            original_deliver = world._deliver_messages

            def watching(u, candidates, *args, **kwargs):
                for v in candidates:
                    for src, dst in ((u, u), (u, v), (v, u), (v, v)):
                        if world.messages.get(src, {}).get(dst) != baseline["at"].get(src, {}).get(dst):
                            changed.append((src, dst))
                return original_input(u, candidates, *args, **kwargs)

            # A delivery ends one sweep and begins the next, so that is where
            # the comparison is allowed to move on.
            def delivering(outbox):
                original_deliver(outbox)
                baseline["at"] = copy.deepcopy(world.messages)

            world._inputs = watching
            world._deliver_messages = delivering
            try:
                run(record_decisions=False)
            finally:
                world._inputs = original_input
                world._deliver_messages = original_deliver

            assert not changed, (
                f"with message_prepass={prepass}, {len(changed)} reads returned a "
                f"message that had changed mid-sweep, e.g. {changed[:3]}")


def test_a_phase_looks_exactly_as_often_as_it_was_asked_to():
    """
    One pass per agent, or two if a pre-pass was asked for. Never a spare one.

    The game phase used to observe twice unconditionally: once to write
    messages, once to place stakes. That second look is now a choice, and the
    cost of the choice is exactly one extra forward pass per agent per phase —
    so the count is worth pinning down, in both directions.

    Reproduction's acting pass skips agents who cannot afford a child, so it
    looks no more than once each; the pre-pass gives everyone a turn, which is
    why the counts are compared against a ceiling rather than an equality.
    """
    for prepass, passes in ((False, 1), (True, 2)):
        world = new_world(small(seed=3, message_prepass=prepass))
        world.step()

        for name, run in (("reproduction", world.reproduction_phase),
                          ("game", world.blotto_phase)):
            present = world.G.number_of_nodes()
            calls = []
            original = world._observe

            def counting(*args, **kwargs):
                calls.append(1)
                return original(*args, **kwargs)

            world._observe = counting
            try:
                run(record_decisions=False)
            finally:
                world._observe = original

            assert len(calls) <= present * passes, (
                f"the {name} phase made {len(calls)} forward passes for {present} "
                f"agents with message_prepass={prepass}, wanted at most "
                f"{present * passes}")
            if prepass:
                assert len(calls) > present, (
                    f"the {name} phase made {len(calls)} forward passes for "
                    f"{present} agents, which is not enough for a pre-pass")


def test_the_message_prepass_gives_the_acting_pass_messages_from_this_graph():
    """
    The whole point of the option, stated as the thing it changes.

    Without it a phase is one pass — observe, speak, act — so what an agent
    acts on was written by its neighbours a phase ago, before the births,
    deaths and conquests since. With it, everyone speaks first, that is
    delivered, and the pass that acts reads a generation written from the graph
    as it stands.

    Checked by watching what is actually in front of the acting pass rather
    than by counting deliveries, because a delivery that happened is not
    evidence that anybody read it.
    """
    import copy

    def what_the_acting_pass_sees(prepass):
        cfg = SimConfig(total_tokens=4000, message_prepass=prepass, seed=3)
        world = new_world(cfg)
        for _ in range(6):
            world.step(record_decisions=False)

        before = copy.deepcopy(world.messages)
        seen = {"in_prepass": False}

        real_prepass = world._message_prepass
        def prepass_wrap(step, features):
            seen["in_prepass"] = True
            real_prepass(step, features)
            seen["in_prepass"] = False
            seen["after_prepass"] = copy.deepcopy(world.messages)
        world._message_prepass = prepass_wrap

        real_observe = world._observe
        def observe(u, candidates, *rest):
            if "at_act" not in seen and not seen["in_prepass"]:
                seen["at_act"] = copy.deepcopy(world.messages)
            return real_observe(u, candidates, *rest)
        world._observe = observe

        world.reproduction_phase(False)
        return before, seen, copy.deepcopy(world.messages)

    before, seen, after = what_the_acting_pass_sees(False)
    assert seen["at_act"] == before, \
        "without the pre-pass the acting pass should be reading last phase's messages"
    assert after != seen["at_act"], "the acting pass should still write messages"

    before, seen, after = what_the_acting_pass_sees(True)
    assert seen["at_act"] == seen["after_prepass"], \
        "with the pre-pass the acting pass should be reading what the pre-pass just delivered"
    assert seen["at_act"] != before, \
        "with the pre-pass the acting pass should not be reading last phase's messages"
    assert after != seen["at_act"], \
        "the acting pass must keep writing its own messages, not only consume the pre-pass's"


def test_the_message_prepass_speaks_for_everyone_including_the_broke():
    """
    An agent with nothing is still there and can still be seen.

    The reproduction phase's acting pass skips anyone who cannot afford a
    child, so leaving the pre-pass to follow that rule would silence exactly
    the agents whose neighbours most need to know about them.
    """

    cfg = SimConfig(total_tokens=4000, message_prepass=True, seed=7)
    world = new_world(cfg)
    for _ in range(6):
        world.step(record_decisions=False)

    world.tokens[sorted(world.G.nodes())[0]] = 0
    broke = [u for u in world.G.nodes() if world.tokens.get(u, 0) <= 0]
    assert broke, "wanted at least one agent holding nothing"

    spoke = []
    real_emit = world._emit_messages
    world._emit_messages = lambda u, t, Y, o: (spoke.append(u), real_emit(u, t, Y, o))[1]
    world._message_prepass("repro.messages", world._precompute_features())

    for u in broke:
        assert u in spoke, f"agent {u} holds nothing and was not given a turn to speak"


def test_the_message_prepass_changes_nothing_it_should_not():
    """Tokens stay conserved and a seeded run stays reproducible with it on."""

    for prepass in (False, True):
        cfg = SimConfig(total_tokens=3000, message_prepass=prepass, seed=11)
        world = new_world(cfg)
        for _ in range(8):
            world.step(record_decisions=False)
            assert sum(world.tokens.values()) == cfg.total_tokens, \
                f"tokens leaked with message_prepass={prepass}"

    def run():
        world = new_world(SimConfig(total_tokens=3000, message_prepass=True, seed=11))
        for _ in range(8):
            world.step(record_decisions=False)
        return sorted(world.tokens.items())

    assert run() == run(), "a seeded run with the pre-pass is not reproducible"


def test_a_prepass_without_messages_is_refused():
    """
    A pass that exists only to send messages has nothing to do without them.

    Refused rather than quietly ignored: a setting that is accepted and then
    does nothing is worse than one that says why it cannot be had.
    """
    try:
        SimConfig.from_dict({"message_prepass": True, "exchange_messages": False})
    except ValueError:
        pass
    else:
        raise AssertionError("a pre-pass without messages should not validate")

    assert SimConfig.from_dict({"message_prepass": True}).message_prepass
    assert SimConfig.from_dict({}).message_prepass is False, \
        "a run recorded before the option existed must read as having run without it"


def test_the_pre_pass_is_on_for_new_runs_and_off_for_old_ones():
    """
    Turning a default on must not reach backwards.

    A run recorded before the option existed ran one pass per phase. Reading
    its stored configuration as though it had used a pre-pass would change what
    a resumed run does, which is the whole reason LEGACY_WHEN_ABSENT exists.
    """
    assert SimConfig().message_prepass is True, "new runs should get the pre-pass"

    # Read back off disk: a key that is not there says what that run did.
    assert SimConfig.from_dict({"total_tokens": 10000}).message_prepass is False, \
        "a configuration written before the option existed must read as off"
    assert SimConfig.from_dict({"total_tokens": 10000,
                                "message_prepass": True}).message_prepass is True

    # Arriving from outside: a key that is not there says nothing about the
    # past, so it means today's default. Asking the API for a world without
    # naming the pre-pass used to quietly get one without it.
    assert SimConfig.from_dict({"total_tokens": 10000}, stored=False).message_prepass is True, \
        "a fresh request that omits the option should get the current default"
    assert SimConfig.from_dict({"total_tokens": 10000, "message_prepass": False},
                               stored=False).message_prepass is False, \
        "an explicit choice must survive either way"

    # The dangerous direction is the one that is not the default: a stored
    # config read as fresh would change what a resumed run does.
    import inspect
    assert inspect.signature(SimConfig.from_dict).parameters["stored"].default is True, \
        "reading a stored config must be what happens when nobody says otherwise"


# ---------------------------------------------------------------------------
# The control: decisions taken from noise
# ---------------------------------------------------------------------------

def test_a_random_world_never_asks_its_brains_anything():
    """
    The control has to actually bypass the brain, not merely disturb it.

    Checked by making the brain unusable: if anything still calls it, this
    raises. That is a stronger guarantee than comparing trajectories, which
    could match by luck or diverge for reasons that have nothing to do with
    whether the forward pass happened.
    """
    import GraphOfLifeSimple as G

    world = new_world(SimConfig(total_tokens=600, n_nodes=30, k_neighbors=4,
                                seed=3, hidden_layers=[6], random_decisions=True))

    def refuse(self, X):
        raise AssertionError("a control world consulted a brain")

    # Patched on the class: Brain has __slots__, so an instance cannot be given
    # a different method, and newborns would arrive with working ones anyway.
    original = G.Brain.forward
    G.Brain.forward = refuse
    try:
        for _ in range(8):
            world.step(record_decisions=False)
    finally:
        G.Brain.forward = original


def test_the_control_is_a_mechanic_and_says_so_in_the_name():
    """A run taken from noise must not be filed under the same algorithm."""
    assert SimConfig().strain_id() == "gol-1"
    assert SimConfig(random_decisions=True).strain_id() == "gol-1+random_decisions"
    # And it is off unless asked for, or every run ever made would be a control.
    assert SimConfig().random_decisions is False


def test_an_ordinary_world_still_reads_its_inputs():
    """
    The other half of the guarantee, or the test above would pass on a world
    that had stopped using its brains entirely.
    """
    import GraphOfLifeSimple as G

    world = new_world(SimConfig(total_tokens=600, n_nodes=30, k_neighbors=4,
                                seed=3, hidden_layers=[6]))
    asked = []
    original = G.Brain.forward

    def counted(self, X):
        asked.append(1)
        return original(self, X)

    G.Brain.forward = counted
    try:
        world.step(record_decisions=False)
    finally:
        G.Brain.forward = original
    assert asked, "an ordinary world went a whole iteration without a forward pass"


# ---------------------------------------------------------------------------
# Edge upkeep
# ---------------------------------------------------------------------------

def test_a_newborns_links_survive_the_first_accounting():
    """
    A connection counts as used for the phase it was made in.

    Judged on token flow alone a newborn's links carry nothing in the phase
    they are created, so the very next prune would cut them and reproduction
    would build a graph that the next step tears straight back down. Every
    schedule and both windows have to agree about this, since the phase a link
    is born in is a different one under each.
    """
    for when in ("reproduction", "blotto", "both"):
        for window in ("phase", "iteration"):
            world = new_world(small(allow_gifting=True, prune_after=when,
                                    inactive_window=window))
            world.phase_count = 7
            parent = sorted(world.G.nodes())[0]
            child = world.next_agent_id
            world.G.add_node(child)
            world._register_agent(child, 5, world._new_brain(), parent)
            world._add_edge(parent, child)

            stale = world._stale_edges()
            assert world._edge_key(parent, child) not in stale, (
                f"a link made this phase was already stale under "
                f"prune_after={when}, inactive_window={window}")


def test_the_window_decides_how_long_an_unused_link_survives():
    """
    `phase` judges the phase just ended; `iteration` gives it the last two.

    The difference is exactly one phase of grace, and it is what lets a
    connection that carried a game-phase bid still be in credit when the
    reproduction phase settles up.
    """
    for window, still_there in (("phase", False), ("iteration", True)):
        world = new_world(small(inactive_window=window))
        u, v = sorted(world.G.edges())[0]
        world.phase_count = 10
        world.edge_active_at[world._edge_key(u, v)] = 9   # used one phase ago

        stale = set(world._stale_edges())
        assert (world._edge_key(u, v) not in stale) == still_there, (
            f"a link last used one phase ago should "
            f"{'survive' if still_there else 'lapse'} under {window}")


def test_the_schedule_decides_which_phase_settles_up():
    world_repro = new_world(small(prune_after="reproduction"))
    world_game = new_world(small(prune_after="blotto"))
    world_both = new_world(small(prune_after="both"))

    assert world_repro._prunes_after(1) and not world_repro._prunes_after(2)
    assert world_game._prunes_after(2) and not world_game._prunes_after(1)
    assert world_both._prunes_after(1) and world_both._prunes_after(2)


def test_a_gift_keeps_a_link_that_would_otherwise_lapse():
    """
    The whole point of gifting: tokens crossing a connection are use of it.

    Built rather than waited for, because whether an evolved agent chooses to
    give is not something a test can arrange. Two agents, one connection, no
    flow across it for long enough that it is due to be cut — then a gift, and
    it is not.
    """
    world = new_world(small(allow_gifting=True, prune_after="reproduction",
                            inactive_window="iteration"))
    u, v = sorted(world.G.edges())[0]
    key = world._edge_key(u, v)
    world.phase_count = 20
    world.edge_active_at[key] = 5           # idle for fifteen phases

    assert key in set(world._stale_edges()), "the link should be due to be cut"

    world._mark_flow(u, v)                   # a gift crosses it
    assert key not in set(world._stale_edges()), (
        "tokens crossed this link; it should no longer be stale")


def test_gifts_are_paid_from_what_reproduction_left():
    """
    An agent cannot promise the same tokens to a child and to a neighbour.

    The gift budget is a share of the purse *after* reproduction has taken its
    cut, so however generous both heads are the two together can never exceed
    what the agent held.
    """
    import numpy as np

    world = new_world(small(allow_gifting=True))
    u = sorted(world.G.nodes())[0]
    candidates = [u] + sorted(world.G.neighbors(u))

    # Every gift head maximally in favour, so the split is as large as the rule
    # allows rather than as large as this agent happens to want.
    Y = np.zeros((world.cfg.n_outputs(), len(candidates)))
    Y[world.heads["GIFT_FRACTION"], :] = [[10.0], [-10.0]]
    Y[world.heads["GIFT"], :] = [[10.0], [-10.0]]
    Y[world.heads["GIFT_MODE"], :] = [[-10.0], [10.0]]

    purse = 40
    gifts = world._choose_gifts(u, purse, candidates, Y)
    assert gifts, "maximal gift heads should produce at least one gift"
    assert sum(amount for _, _, amount in gifts) <= purse, (
        "gifts together exceeded the purse they were paid from")
    assert all(taker != u for _, taker, _ in gifts), (
        "a gift to oneself crosses no connection and is not a gift")


def test_gifting_off_means_no_gift_heads_and_no_risk_input():
    """
    A mechanic that is off costs the brain nothing.

    Gifting adds three output heads and an input flag, so leaving them in place
    when it is off would change the architecture of every run that does not use
    it — and a checkpoint saved under one shape cannot be resumed under another.
    """
    off, on = small(allow_gifting=False), small(allow_gifting=True)

    assert not [k for k in off.head_layout() if k.startswith("GIFT")]
    assert len([k for k in on.head_layout() if k.startswith("GIFT")]) == 3
    assert on.n_outputs() == off.n_outputs() + 6
    assert on.n_inputs() == off.n_inputs() + 1
    assert off.flag_inputs() == 1 and on.flag_inputs() == 2


def test_the_risk_flag_says_which_links_are_about_to_lapse():
    """
    The input that makes a gift a decision rather than a guess.

    It is the second flag row, and it is set for exactly the neighbours whose
    connection would be cut if nothing more crossed it.
    """
    world = new_world(small(allow_gifting=True, inactive_window="phase"))
    world.phase_count = 12
    u = max(world.G.nodes(), key=lambda n: world.G.degree[n])
    neighbours = sorted(world.G.neighbors(u))

    # Everything idle, then one neighbour's link freshly used.
    for a, b in world.G.edges():
        world.edge_active_at[world._edge_key(a, b)] = 0
    safe = neighbours[0]
    world.edge_active_at[world._edge_key(u, safe)] = 12

    at_risk = set(world._stale_edges())
    features = world._precompute_features()
    log_deg, _n, q_tok, q_deg, log_tok, _ = features

    seen = world._inputs(u, neighbours + [u], log_deg, q_tok, q_deg, log_tok, at_risk)
    for i, v in enumerate(neighbours):
        expected = 0.0 if v == safe else 1.0
        assert seen[1, i] == expected, f"risk flag wrong for the link {u}-{v}"

    # And an agent looking at itself has no connection to be told about.
    assert seen[1, -1] == 0.0


def test_weighted_redistribution_conserves_tokens_and_favours_the_rich():
    """
    `by_tokens` changes who the estate goes to, not how much of it there is.

    Conservation is the part that must hold exactly; the tilt is the part that
    makes it a different mechanic, and over many culls it is what turns the
    cleanup from a leveller into an engine of concentration.
    """

    for mode in ("uniform", "by_tokens"):
        world = new_world(small(redistribution=mode))
        for _ in range(6):
            world.step(record_decisions=False)
        assert sum(world.tokens.values()) == world.cfg.total_tokens, (
            f"{mode} redistribution did not conserve tokens")

    # The tilt itself, on one cleanup with a pool to share and a clear favourite.
    world = new_world(small(redistribution="by_tokens"))
    rich = sorted(world.G.nodes())[0]
    for u in world.G.nodes():
        world.tokens[u] = 1
    world.tokens[rich] = 10_000
    world.cfg.total_tokens = sum(world.tokens.values())

    before = world.tokens[rich]
    world._cleanup_and_redistribute()
    # Nothing died, so nothing was shared; give it something to share.
    world.cfg.tokens_created_per_phase = 5000
    world._cleanup_and_redistribute()
    assert world.tokens[rich] > before, (
        "weighting by holdings should send most of the pool to the largest holder")


def test_the_frozen_upkeep_settings_reproduce_the_old_engine():
    """
    `gol-1` has to keep meaning what it meant.

    These four mechanics change the engine's core loop — when edges are cut,
    what counts as use, an extra transfer step, an extra input row — so the
    frozen path through all of it has to be byte-identical to the path that
    existed before them. Anything else silently rewrites every result already
    recorded.
    """
    frozen = small(allow_gifting=False, prune_after="blotto",
                   inactive_window="phase", redistribution="uniform")
    assert frozen.strain_id() == "gol-1", "the frozen settings are not gol-1"
    assert frozen.n_inputs() == SimConfig(**{**dataclasses.asdict(frozen)}).n_inputs()

    # The engine takes the same route: no gift step, no risk flag, and the cut
    # falling after the game phase on the phase just ended.
    world = new_world(frozen)
    assert not world.cfg.allow_gifting
    assert world._window_phases() == 1
    assert world._prunes_after(2) and not world._prunes_after(1)
    assert "GIFT" not in world.heads


def test_the_head_layout_is_the_only_statement_of_where_the_rows_are():
    """
    build_heads reads cfg.head_layout() rather than counting the offsets again.

    It used to count them again, and the two drifted the moment a head was
    added: the layout gained the gift rows, this did not, and the run died
    reaching for a head that had no slice. One statement, checked here against
    every combination that changes the shape.
    """
    from GraphOfLifeSimple import build_heads

    for gifting in (False, True):
        for handover in (False, True):
            for revolutions in (False, True):
                cfg = small(allow_gifting=gifting, allow_handover=handover,
                            allow_revolutions=revolutions)
                layout = cfg.head_layout()
                heads = build_heads(cfg)

                assert set(heads) == (set(layout) - {"MESSAGE"}) | {"MESSAGE_START"}
                for name, (start, end) in layout.items():
                    if name == "MESSAGE":
                        assert heads["MESSAGE_START"] == start
                    elif name == "BLOTTO":
                        assert heads[name] == start
                    else:
                        assert heads[name] == slice(start, end), name

                # And the heads tile the output rows, each starting where the
                # one before it ends: a head laid over another, or past a gap,
                # would read rows that mean something else or nothing.
                spans = sorted(tuple(span) for span in layout.values())
                assert spans[0][0] == 0, spans
                assert all(a[1] == b[0] for a, b in zip(spans, spans[1:])), spans


def test_the_seed_graph_survives_its_first_accounting():
    """
    A connection cannot be cut before anybody has had a chance to use it.

    The seed graph is built before any phase has run. Stamped with the phase
    counter as it stands, every one of those connections arrives already a
    phase stale, and a run that settles up after reproduction against a
    single-phase window cut *the entire seed graph* at the end of its first
    reproduction phase — before one token had ever crossed it. The population
    then continued on newborn links alone.

    The same applies to a checkpoint written before upkeep was recorded: its
    connections have no history and must not be cut for lacking one.
    """
    for when in ("reproduction", "blotto", "both"):
        for window in ("phase", "iteration"):
            for gifting in (False, True):
                world = new_world(small(allow_gifting=gifting, prune_after=when,
                                        inactive_window=window))
                seeded = world.G.number_of_edges()
                assert seeded > 0

                # The very first accounting, whenever it falls due.
                world.phase_count += 1
                stale = world._stale_edges()
                assert not stale, (
                    f"{len(stale)} of {seeded} seed connections were stale at the "
                    f"first accounting under prune_after={when}, "
                    f"inactive_window={window}, allow_gifting={gifting}")


def test_a_resumed_run_does_not_cut_a_graph_it_has_no_history_for():
    """
    A checkpoint from before upkeep was tracked carries no activity record.

    Read as "never used", the first accounting after the resume would cut every
    connection in it at once.
    """
    world = new_world(small(prune_after="reproduction", inactive_window="phase"))
    for _ in range(2):
        world.step(record_decisions=False)

    blob = dict(world.to_checkpoint())
    del blob["edge_active_at"]          # as an older checkpoint would arrive

    resumed = GraphOfLife.from_checkpoint(blob, world.cfg)
    assert resumed.G.number_of_edges() > 0
    resumed.phase_count += 1
    assert not resumed._stale_edges(), (
        "a resumed run cut connections it simply had no history for")


def test_a_resumed_run_remembers_which_connections_were_used():
    """
    Which connections carried tokens, and when, is world state like any other.

    A resumed run has to know what the game phase before the checkpoint did,
    or its first accounting judges connections on a history it does not have —
    sparing everything, or cutting a graph that was busy moments earlier.
    Messages in flight are checkpointed for exactly this reason; this is the
    same argument about a different book.
    """
    world = new_world(small(allow_gifting=True, prune_after="reproduction",
                            inactive_window="iteration"))
    for _ in range(3):
        world.step(record_decisions=False)

    live = {e: at for e, at in world.edge_active_at.items() if world.G.has_edge(*e)}
    busy = {e for e, at in live.items() if at == world.phase_count}
    assert busy, "the game phase before the checkpoint moved nothing"

    resumed = GraphOfLife.from_checkpoint(dict(world.to_checkpoint()), world.cfg)

    assert resumed.phase_count == world.phase_count
    assert {e: at for e, at in resumed.edge_active_at.items()
            if resumed.G.has_edge(*e)} == live, "the activity record did not survive"
    assert busy == {e for e, at in resumed.edge_active_at.items()
                    if at == resumed.phase_count}, (
        "a resumed run forgot which connections the last game phase used")

    # And so the next accounting falls exactly as it would have.
    assert sorted(resumed._stale_edges()) == sorted(world._stale_edges())


def test_no_legacy_entry_restates_the_frozen_default():
    """
    A legacy entry that matches the frozen default is a line that does nothing.

    An absent key already resolves to the frozen default, so naming a mechanic
    whose pre-existing behaviour *is* that default changes no reading of any
    stored configuration. It is not merely noise: it buries the entries that do
    matter. The list reached seven such lines against two real ones, four of
    them added under a comment claiming they kept an old checkpoint loadable —
    the frozen defaults were what kept it loadable, and those entries were
    never consulted.

    So the rule is the assertion: an entry belongs here only if it differs.
    """
    import gol_config

    pointless = {
        name: value for name, value in SimConfig.LEGACY_WHEN_ABSENT.items()
        if name in gol_config.MECHANICS and gol_config.MECHANICS[name] == value
    }
    assert not pointless, (
        f"these legacy entries restate the frozen default and so do nothing: "
        f"{sorted(pointless)}. Delete them, or the two that matter cannot be "
        f"picked out from the ones that do not.")

    # And every entry has to name something that exists, or it is a typo that
    # silently never applies.
    unknown = set(SimConfig.LEGACY_WHEN_ABSENT) - {f.name for f in dataclasses.fields(SimConfig)}
    assert not unknown, f"legacy entries for fields that do not exist: {sorted(unknown)}"


# ---------------------------------------------------------------------------
# Strains: which algorithm a run is
# ---------------------------------------------------------------------------

def test_every_setting_is_classified_as_one_of_the_three_kinds():
    """
    A new setting has to be declared a mechanic, a parameter or infrastructure.

    This is the guard that keeps the strain scheme honest. A mechanic left out
    of the table changes what a run does without changing its name, so two runs
    of different algorithms compare as the same one — and nothing anywhere else
    would notice.
    """
    import gol_config

    declared = (set(gol_config.MECHANICS)
                | set(gol_config.PARAMETERS)
                | set(gol_config.INFRASTRUCTURE))
    actual = {f.name for f in dataclasses.fields(SimConfig)}

    assert actual - declared == set(), (
        f"settings that are not classified: {sorted(actual - declared)}. Add "
        f"each to MECHANICS, PARAMETERS or INFRASTRUCTURE in gol_config.py — "
        f"see research/Research.md §5.1 for which is which")
    assert declared - actual == set(), (
        f"classified but not a setting: {sorted(declared - actual)}")

    overlap = set(gol_config.MECHANICS) & set(gol_config.PARAMETERS)
    assert not overlap, f"classified twice: {sorted(overlap)}"


def test_a_default_world_is_the_baseline_strain():
    assert SimConfig().strain_id() == "gol-1"
    # Parameters are not part of the algorithm's name; otherwise every seed
    # would be its own strain and nothing could be grouped.
    varied = SimConfig(total_tokens=2500, n_nodes=50, seed=7,
                       hidden_layers=[12, 10], mutation_probability=0.1)
    assert varied.strain_id() == "gol-1"


def test_a_changed_mechanic_shows_up_in_the_name():
    assert SimConfig(brain_kind="binary").strain_id() == "gol-1+brain_kind=binary"
    assert SimConfig(exchange_messages=False).strain_id() == "gol-1+exchange_messages=false"
    assert SimConfig(tokens_created_per_phase=5).strain_id() == "gol-1+tokens_created_per_phase=5"

    # Alphabetical, so the same set of mechanics always spells the same strain
    # whatever order they were passed in.
    one = SimConfig(brain_kind="binary", allow_revolutions=False).strain_id()
    other = SimConfig(allow_revolutions=False, brain_kind="binary").strain_id()
    assert one == other == "gol-1+allow_revolutions=false+brain_kind=binary"


def test_the_frozen_defaults_are_what_a_bare_config_does():
    """
    The scheme only works if a mechanic's frozen default is its actual default.

    Otherwise `gol-1` would name a world nobody can build by asking for
    nothing, and every run would carry a strain listing mechanics it never
    changed.
    """
    import gol_config

    bare = SimConfig()
    for name, frozen in gol_config.MECHANICS.items():
        assert getattr(bare, name) == frozen, (
            f"{name} defaults to {getattr(bare, name)!r} but is frozen at "
            f"{frozen!r}. Changing a frozen default renames every existing "
            f"strain — bump SPEC instead")


def test_a_run_and_its_checkpoint_both_say_which_algorithm_they_are():
    cfg = small(brain_kind="binary")
    world = new_world(cfg)
    blob = world.to_checkpoint()

    assert "strain" in blob, "a checkpoint travels alone and must name its algorithm"
    assert str(blob["strain"]) == cfg.strain_id() == "gol-1+brain_kind=binary"

    # And it does not disturb restoring, which reads the keys it knows.
    restored = GraphOfLife.from_checkpoint(blob, cfg)
    assert set(restored.G.nodes()) == set(world.G.nodes())


if __name__ == "__main__":
    sys.exit(runner.main(globals()))
