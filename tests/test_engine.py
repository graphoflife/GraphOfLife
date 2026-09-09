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

Deliberately dependency-free. A research repository that needs a toolchain
installed before anyone can check it still works is a repository whose tests
do not get run.
"""

from __future__ import annotations

import dataclasses
import math
import os
import random
import re
import sys
import tempfile

# The modules under test sit in the repository root, one level up. Added here
# rather than left to the caller so that running this file directly works from
# anywhere, which is the whole point of it being runnable directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx

import gol_series
from gol_config import SimConfig
from GraphOfLifeSimple import GraphOfLife, make_brain, new_world
from GraphOfLifeSimple import _choose_binary as G_choose_binary


def small(**overrides) -> SimConfig:
    """A world small enough to run many times in a test."""
    settings = dict(
        total_tokens=3000, n_nodes=60, k_neighbors=6,
        hidden_layers=[14, 12], message_amount=2, random_input_amount=2,
        seed=17,
    )
    settings.update(overrides)
    return SimConfig(**settings)


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

    np.random.seed(3)
    # mode says "take the maximum", and the two sides are equal.
    outcomes = {G_choose_binary(5.0, 5.0, 0.0, 1.0) for _ in range(200)}
    assert outcomes == {True, False}, "a tie always fell the same way"

    # A clear preference is still obeyed exactly.
    assert G_choose_binary(5.0, 1.0, 0.0, 1.0) is True
    assert G_choose_binary(1.0, 5.0, 0.0, 1.0) is False


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
            original_input = world._input_vec
            original_deliver = world._deliver_messages

            def watching(u, v, *args, **kwargs):
                for src, dst in ((u, u), (u, v), (v, u), (v, v)):
                    if world.messages.get(src, {}).get(dst) != baseline["at"].get(src, {}).get(dst):
                        changed.append((src, dst))
                return original_input(u, v, *args, **kwargs)

            # A delivery ends one sweep and begins the next, so that is where
            # the comparison is allowed to move on.
            def delivering(outbox):
                original_deliver(outbox)
                baseline["at"] = copy.deepcopy(world.messages)

            world._input_vec = watching
            world._deliver_messages = delivering
            try:
                run(record_decisions=False)
            finally:
                world._input_vec = original_input
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


# ---------------------------------------------------------------------------
# The teaching script
# ---------------------------------------------------------------------------

def test_the_teaching_script_runs_and_conserves_tokens():
    """
    explain_minimal.py is on the site for people to copy and run, and the
    Explanation walks through it line by line. A version of it that crashes, or
    that quietly leaks tokens, would be teaching something false — so it is
    held to the same invariants as the engine it stands in for.
    """
    import random
    import numpy as np
    import explain_minimal as minimal

    random.seed(11)
    np.random.seed(11)
    world = minimal.World()

    for i in range(40):
        world.step()
        assert world.adj, f"the world emptied at iteration {i + 1}"
        assert sum(world.tokens.values()) == minimal.TOKENS, (
            f"tokens were not conserved at iteration {i + 1}")
        assert len(minimal.components(world.adj)) == 1, (
            f"cleanup left more than one piece at iteration {i + 1}")
        for agent, neighbours in world.adj.items():
            assert agent not in neighbours, f"agent {agent} is joined to itself"


def test_the_teaching_script_gives_a_revolution_to_its_strongest_rebel():
    """
    The revolution goes to the strongest staker in the rung that tipped it, not
    to a random member of the crowd. The full engine is held to this too; the
    teaching script has its own copy of the rule, so it gets its own check.
    """
    import explain_minimal as minimal

    # One big spender against four small ones and a larger rebel.
    staked = {1: 20, 2: 1, 3: 2, 4: 3, 5: 4, 6: 14}
    revolt = {2: 1, 3: 2, 4: 3, 5: 4, 6: 14}
    winners = {minimal.resolve(dict(staked), dict(revolt)) for _ in range(200)}
    assert winners == {6}, f"expected the strongest rebel to take it, got {winners}"


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
    import gol_store

    with tempfile.TemporaryDirectory() as tmp:
        original = gol_store.BASE_DIR
        gol_store.BASE_DIR = tmp
        try:
            cfg = SimConfig(total_tokens=400, n_nodes=30, k_neighbors=4,
                            seed=3, hidden_layers=[6], export_decisions=False)
            run_id = gol_store.create_run("x", cfg)["id"]
            world = new_world(cfg)
            written = 0
            for _ in range(40):
                for frame in world.step(record_decisions=False):
                    gol_store.write_frame(run_id, written, frame)
                    written += 1
            gol_store.update_meta(run_id, frame_count=written,
                                  iteration=world.iteration)

            coarse = gol_series.build_series(run_id, points=3)
            finer = gol_series.build_series(run_id, points=5)
            whole = gol_series.build_series(run_id)

            last = whole["series"]["iteration"][-1]
            assert coarse["series"]["iteration"][0] == 0
            assert coarse["series"]["iteration"][-1] == last, (
                "a coarse pass must still reach the end of the run")
            assert not coarse["complete"] and whole["complete"]
            assert coarse["points"] < finer["points"] <= whole["points"]

            of = lambda r: set(r["series"]["iteration"])
            assert of(coarse) < of(finer) <= of(whole), (
                "each pass must contain the one before it")
        finally:
            gol_store.BASE_DIR = original


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
    assert light["nodes"] == heavy["nodes"], "the cheap statistics must agree"
    assert light["gini"] == heavy["gini"]


def test_a_heavy_pass_upgrades_the_rows_a_light_one_left_behind():
    """
    The two passes share one cache, so the second has to fill in the first.

    The failure this guards against is the cache counting a light row as
    already done: the expensive statistics would then never be computed for any
    frame the cheap pass reached first, and the chart would be permanently
    empty with no sign that anything was missing.
    """
    import gol_store, gol_series

    with tempfile.TemporaryDirectory() as tmp:
        original = gol_store.BASE_DIR
        gol_store.BASE_DIR = tmp
        try:
            cfg = SimConfig(total_tokens=400, n_nodes=30, k_neighbors=4,
                            seed=4, hidden_layers=[6], export_decisions=False)
            run_id = gol_store.create_run("x", cfg)["id"]
            world = new_world(cfg)
            written = 0
            for _ in range(20):
                for frame in world.step(record_decisions=False):
                    gol_store.write_frame(run_id, written, frame)
                    written += 1
            gol_store.update_meta(run_id, frame_count=written,
                                  iteration=world.iteration)

            light = gol_series.build_series(run_id, heavy=False)
            assert light["heavy"] is False
            assert all(v is None for v in light["series"]["bridges"])

            heavy = gol_series.build_series(run_id, heavy=True)
            assert heavy["heavy"] is True
            assert all(v is not None for v in heavy["series"]["bridges"]), (
                "the heavy pass did not upgrade the rows the light pass stored")
            assert heavy["count"] == light["count"], "the upgrade duplicated points"

            # And going back to a light request keeps what the heavy pass found,
            # rather than throwing the expensive work away again.
            back = gol_series.build_series(run_id, heavy=False)
            assert all(v is not None for v in back["series"]["bridges"])

            assert not [k for k in heavy["keys"] if k.startswith("_")], (
                "bookkeeping fields must not travel with the statistics")
        finally:
            gol_store.BASE_DIR = original


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


# ---------------------------------------------------------------------------
# Topology: what a cut would cost
# ---------------------------------------------------------------------------

def _adj(edges):
    """Adjacency as the bridge walk wants it, from a list of pairs."""
    out = {}
    for a, b in edges:
        out.setdefault(a, set()).add(b)
        out.setdefault(b, set()).add(a)
    return out


def test_a_path_is_all_bridges_and_the_middle_one_is_the_worst():
    """
    Every edge of a path splits it, and the worst split is the middle.

    0-1-2-3: cutting the outer edges strands one node, cutting the middle one
    strands two of four. So the worst single cut costs half the population,
    which is the largest a cut can ever cost.
    """
    import GraphOfLifeSimple as G

    edges = [(0, 1), (1, 2), (2, 3)]
    ids, adj = [0, 1, 2, 3], _adj(edges)

    splits = G.bridge_splits(ids, adj)
    assert len(splits) == 3, f"a path of four has three bridges, got {len(splits)}"
    sides = sorted(min(b, 4 - b) for _, _, b in splits)
    assert sides == [1, 1, 2], f"expected splits of 1, 1 and 2, got {sides}"
    assert G.worst_cut_share(ids, adj) == 0.5


def test_a_cycle_has_no_bridges_at_all():
    """Every edge of a cycle lies on a loop, so nothing can be cut in two."""
    import GraphOfLifeSimple as G

    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    ids, adj = [0, 1, 2, 3], _adj(edges)

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
    ids, adj = list(range(6)), _adj(dumbbell)
    assert len(G.bridge_splits(ids, adj)) == 1
    assert G.worst_cut_share(ids, adj) == 0.5

    fringed = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 4), (2, 5)]
    ids, adj = list(range(6)), _adj(fringed)
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
    ids, adj = list(range(5)), _adj(edges)
    assert G.two_core_size(ids, adj) == 3

    # A tree has no core at all: peeling never stops until nothing is left.
    edges = [(0, 1), (1, 2), (1, 3), (3, 4)]
    ids, adj = list(range(5)), _adj(edges)
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
        got = gol_spectral.spectral_gap(list(range(n)), _adj(edges))
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
        got = gol_spectral.spectral_gap(list(range(n)), _adj(ring))
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

    weak = gol_spectral.spectral_gap(list(range(12)), _adj(dumbbell))
    strong = gol_spectral.spectral_gap(list(range(12)), _adj(whole))

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
    adrift = _adj(core + [(8, 9)])
    adrift.setdefault(8, set()).add(9)

    got = gol_spectral.spectral_gap(ids, adrift)
    alone = gol_spectral.spectral_gap(list(range(8)), _adj(core))
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


# ---------------------------------------------------------------------------
# The two ways the site is served
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


def test_the_server_and_the_build_ship_the_same_python():
    """
    The page fetches engine files from /py/. build_site.sh copies them there
    when it assembles the static site; gol_server.py serves them from the
    repository root, because when it is serving web/ off the disk there is
    nowhere to copy them to.

    Two lists of the same thing, so they drift. They already did: the
    Explanation fetches the script it walks through, which the build shipped
    and the server did not, so the tab worked once published and reported that
    it could not load on localhost.
    """
    import re
    import gol_server

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = open(os.path.join(root, "build_site.sh")).read()
    match = re.search(r"for f in ([^;]+); do", script)
    assert match, "could not find the copy loop in build_site.sh"
    copied = set(match.group(1).split())

    served = set(gol_server.SHIPPED_PY)
    assert copied == served, (
        f"build_site.sh copies {sorted(copied)} but gol_server serves "
        f"{sorted(served)}")

    for name in served:
        assert os.path.isfile(os.path.join(root, name)), f"{name} does not exist"

    # The same arrangement for documents the page renders, which live outside
    # web/ for the same reason and would fail the same way: rendering on the
    # published site and reporting that it cannot be read on localhost.
    #
    # Checked by name rather than by the whole destination path, because the
    # build copies them in a loop and the full string never appears. The
    # stamping table is checked separately, and is the half that matters — a
    # document that is copied and not stamped is a document a returning visitor
    # keeps the old version of.
    stamped = re.search(r"declare -A stamp_in=\((.*?)\n\)", script, re.S)
    assert stamped, "could not find the stamp_in table in build_site.sh"
    for url, source in gol_server.SHIPPED_DOCS.items():
        assert os.path.isfile(os.path.join(root, source)), f"{source} does not exist"
        assert os.path.basename(source) in script, (
            f"gol_server serves {url} from {source}, and build_site.sh never "
            f"copies it into the site")
        assert url in stamped.group(1), (
            f"build_site.sh copies {url} but never stamps it, so a cached copy "
            f"survives a deploy")


# ---------------------------------------------------------------------------
# Running without pytest
# ---------------------------------------------------------------------------

def test_every_asset_a_script_fetches_by_name_is_cache_stamped():
    """
    A returning visitor must never run new code against an old asset.

    index.html's script and stylesheet tags are stamped wholesale, but anything
    a script fetches by name — a worker, what a worker imports, the teaching
    script, a recording — is invisible to that pass and has to be named in
    build_site.sh. Nothing about adding a new one makes you remember, and the
    failure is silent and only hits people who have been here before: the page
    is new, the file behind it is last week's.

    So the two are checked against each other. Any asset reference in web/js
    that build_site.sh does not stamp fails here rather than in someone's
    cache.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = open(os.path.join(root, "build_site.sh")).read()

    # Only the stamping table counts, not the whole script. Looking for the
    # path anywhere in the file passed for a `cp` line that shipped a document
    # and never stamped it — mentioning an asset is not the same as versioning
    # it, and the whole failure this guards against is a file that is shipped
    # and cached.
    table = re.search(r"declare -A stamp_in=\((.*?)\n\)", script, re.S)
    assert table, "could not find the stamp_in table in build_site.sh"
    build = table.group(1)

    # A quoted path into one of the shipped directories, or a bare file next to
    # the script — which is what importScripts() takes.
    quoted = re.compile(r"""['"]((?:js|py|data|css)/[\w./-]+|[\w-]+\.js)['"]""")
    interesting = (".js", ".py", ".json", ".bin", ".css", ".md")

    def ours(line, at):
        """
        False for a name glued onto a base URL.

        `importScripts(PYODIDE + 'pyodide.js')` is fetched from a CDN that
        versions itself in its own path; there is nothing of ours in it to
        stamp.
        """
        return not line[:at].rstrip().endswith("+")

    unstamped = []
    js_dir = os.path.join(root, "web", "js")
    for name in sorted(os.listdir(js_dir)):
        if not name.endswith(".js"):
            continue
        source = open(os.path.join(js_dir, name)).read()
        for line in source.splitlines():
            # Only where a file is actually being fetched.
            if not re.search(r"importScripts\(|new Worker\(|fetch\(|SOURCE|SCRIPT|RUN:|workerUrl", line):
                continue
            for match in quoted.finditer(line):
                ref = match.group(1)
                if not ref.endswith(interesting) or not ours(line, match.start()):
                    continue
                if ref not in build:
                    unstamped.append(f"{name}: {ref}")

    assert not unstamped, (
        "these are fetched by name but build_site.sh does not stamp them, so a "
        "cached copy will survive a deploy:\n  " + "\n  ".join(unstamped))


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
    import numpy as np

    def what_the_acting_pass_sees(prepass):
        random.seed(3)
        np.random.seed(3)
        cfg = SimConfig(total_tokens=4000, message_prepass=prepass)
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
    import numpy as np

    random.seed(7)
    np.random.seed(7)
    cfg = SimConfig(total_tokens=4000, message_prepass=True)
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
    import numpy as np

    for prepass in (False, True):
        random.seed(11)
        np.random.seed(11)
        cfg = SimConfig(total_tokens=3000, message_prepass=prepass, seed=11)
        world = new_world(cfg)
        for _ in range(8):
            world.step(record_decisions=False)
            assert sum(world.tokens.values()) == cfg.total_tokens, \
                f"tokens leaked with message_prepass={prepass}"

    def run():
        random.seed(11)
        np.random.seed(11)
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


def test_the_interface_offers_every_setting_the_engine_has():
    """
    A field nobody can set is a field nobody knows about.

    Every knob on SimConfig should have somewhere in the form to set it, and
    every checkbox should say what it does — the pre-pass and messages were
    both added without one, and an unexplained checkbox is a checkbox nobody
    touches.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    page = open(os.path.join(root, "web", "index.html")).read()

    # Not offered on purpose: the seed graph's rewire probability and the
    # run-control knobs are set elsewhere or left at their defaults.
    from dataclasses import fields as dataclass_fields
    missing = [f.name for f in dataclass_fields(SimConfig)
               if f'data-cfg="{f.name}"' not in page]
    assert not missing, f"no form field for: {', '.join(missing)}"

    for name in ("exchange_messages", "message_prepass", "allow_handover",
                 "allow_revolutions"):
        block = page.split(f'data-cfg="{name}"')[1].split("</div>")[0]
        assert "<small>" in block, f"the {name} checkbox has no explanation under it"


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


def test_every_brain_kind_has_a_preset_that_validates():
    """
    Choosing a brain should not also mean knowing what else to change.

    A binary brain needs wider layers and a gentler mutation rate, and getting
    the second one wrong kills runs rather than merely making them worse. The
    presets live with the engine so the form cannot drift from them.
    """
    kinds = ("float", "float16", "binary")
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


def test_the_defaults_endpoint_carries_the_brain_presets():
    """The form fills itself in from the engine, so the engine has to say."""
    import gol_server

    payload = gol_server.Handler._defaults()
    assert "brain_presets" in payload, "the form has nowhere to read the presets from"
    assert set(payload["brain_presets"]) == set(SimConfig.BRAIN_PRESETS)


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

    random.seed(4)
    np.random.seed(4)
    cfg = small(brain_kind="binary", brain_bits=16, hidden_layers=[24, 16],
                message_amount=5, random_input_amount=5, seed=4)
    world = new_world(cfg)
    for _ in range(4):
        world.step(record_decisions=False)

    assert cfg.binary_rows() == cfg.MAGNITUDE_INPUTS * cfg.brain_bits + cfg.bit_inputs()
    brain = next(iter(world.brains.values()))
    assert brain.layer_sizes()[0] == cfg.binary_rows()

    rows = []
    original = world._observe

    def spy(u, candidates, *rest):
        x = np.column_stack([world._input_vec(u, v, *rest) for v in candidates])
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
    start = cfg.FLAG_INPUTS + cfg.MAGNITUDE_INPUTS * cfg.brain_bits
    tail = encoded[start:]
    assert tail.shape[0] == 4 * cfg.message_amount + cfg.random_input_amount
    dead = int((tail.max(axis=1) == tail.min(axis=1)).sum())
    assert dead == 0, f"{dead} message or noise rows can never change"
    assert set(np.unique(tail).tolist()) <= {0, 1}


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

    log_deg, _neighs, q_tok, q_deg, log_tok = world._precompute_features()
    lowest = np.inf
    for u in sorted(world.G.nodes())[:20]:
        for v in [u] + sorted(world.G.neighbors(u)):
            vec = world._input_vec(u, v, log_deg, q_tok, q_deg, log_tok)
            span = vec[cfg.FLAG_INPUTS:cfg.FLAG_INPUTS + cfg.MAGNITUDE_INPUTS]
            lowest = min(lowest, float(span.min()))
    assert lowest >= 0.0, f"a laddered input went to {lowest}"

    thresholds = make_brain(cfg, 0).thresholds()
    assert thresholds.min() >= 0.0, \
        f"the ladder starts at {thresholds.min()}, below anything that can reach it"


def test_a_binary_world_says_bits_and_hears_bits():
    """
    Its output layer hands back a count, and squashing that through tanh gave
    a value that was neither a bit nor a useful magnitude — eleven distinct
    values across a whole phase, then read back through a ladder that could
    only see the bottom of it. A binary world's messages are bits, and so is
    its noise.
    """
    import numpy as np

    random.seed(6)
    np.random.seed(6)
    cfg = small(brain_kind="binary", brain_bits=16, hidden_layers=[24, 16], seed=6)
    world = new_world(cfg)
    for _ in range(4):
        world.step(record_decisions=False)

    sent = [v for notes in world.messages.values() for vec in notes.values() for v in vec]
    assert sent, "wanted some messages to look at"
    assert set(sent) <= {0.0, 1.0}, f"a binary world sent non-bits: {sorted(set(sent))[:6]}"

    log_deg, _neighs, q_tok, q_deg, log_tok = world._precompute_features()
    u, v = sorted(world.G.nodes())[:2]
    seen = set()
    for _ in range(40):
        vec = world._input_vec(u, v, log_deg, q_tok, q_deg, log_tok)
        seen.update(vec[-cfg.random_input_amount:].tolist())
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
    vec = world._input_vec(u, v, features[0], features[2], features[3], features[4])
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

    brain = make_brain(cfg, 0, allocate=False)
    edges = brain.thresholds()
    assert len(edges) == bands
    assert np.allclose(np.diff(edges), brain.band_width()), \
        "the reported edges must be the ones the encoder uses"

    def code(value):
        x = np.zeros((cfg.n_inputs(), 1))
        x[cfg.FLAG_INPUTS, 0] = value
        rows = brain.encode(x)[cfg.FLAG_INPUTS:cfg.FLAG_INPUTS + cfg.brain_bits, 0]
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
    brain = make_brain(cfg, 0, allocate=False)

    values = rng.uniform(0.0, 12.0, size=2000)
    x = np.zeros((cfg.n_inputs(), values.size))
    x[cfg.FLAG_INPUTS] = values
    rows = brain.encode(x)[cfg.FLAG_INPUTS:cfg.FLAG_INPUTS + cfg.brain_bits].T.astype(float)

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


def test_families_are_counted_from_ancestry_not_from_one_frame():
    """
    How many families the living divide into needs ancestry, and ancestry is a
    chain: it cannot be read off a single frame and it cannot be sampled.

    `distinctParents` — which was called `distinctLineages` and never counted
    lineages — looks one step back and tracks the population. The windowed
    count looks as far back as the window and does not.
    """
    import tempfile

    import gol_series
    import gol_store

    window = gol_series._CladeWindow(window=2)

    # A founder, then two children of it, then a grandchild of each. Anchored
    # two iterations back from iteration 3, everything alive descends from the
    # single agent that was alive at iteration 1.
    window.observe(0, [1], [-1])
    window.observe(1, [2], [1])
    window.observe(2, [3, 4], [2, 2])
    window.observe(3, [5, 6], [3, 4])
    assert window.families([5, 6], 3) == 1, "both descend from brain 2, alive at 1"

    # Anchored at the present, everything is its own family.
    assert gol_series._CladeWindow(window=0).families([5, 6], 3) == 2

    # Ancestry beyond the window is dropped rather than kept for ever.
    long_window = gol_series._CladeWindow(window=1)
    for i in range(1, 400):
        long_window.observe(i, [i], [i - 1])
    assert len(long_window.parent) < 40, \
        f"the window is holding {len(long_window.parent)} links; it is not forgetting"


def test_the_family_count_is_absent_when_the_chain_is_broken():
    """
    A run recorded every other iteration has holes where the links were, and a
    family count computed over holes is a guess. Absent is the honest answer,
    and it is the convention the rest of these statistics already follow.
    """
    import tempfile

    import gol_series
    import gol_store

    def series_for(export_every):
        with tempfile.TemporaryDirectory() as tmp:
            original = gol_store.BASE_DIR
            gol_store.BASE_DIR = tmp
            try:
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
                return gol_series.build_series(run_id)
            finally:
                gol_store.BASE_DIR = original

    whole = series_for(1)
    assert "cladesInWindow" in whole["keys"], \
        "a fully recorded run should have a family count"
    counts = whole["series"]["cladesInWindow"]
    assert all(c >= 1 for c in counts)
    assert max(counts) > 1, "everything in one family from the first frame is suspicious"

    sampled = series_for(2)
    assert "cladesInWindow" not in sampled["keys"], \
        "a sampled run cannot have its ancestry rebuilt and must not pretend to"


def _main() -> int:
    """Find the tests in this file and run them, reporting like pytest would."""
    import time
    import traceback

    tests = sorted(
        (name, fn) for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    )

    failures = []
    started = time.perf_counter()
    for name, fn in tests:
        try:
            fn()
            print(".", end="", flush=True)
        except Exception:
            failures.append((name, traceback.format_exc()))
            print("F", end="", flush=True)

    elapsed = time.perf_counter() - started
    print(f"\n\n{len(tests) - len(failures)} passed, {len(failures)} failed "
          f"in {elapsed:.1f}s")

    for name, trace in failures:
        print(f"\n--- {name} ---\n{trace}")
    return 1 if failures else 0


if __name__ == "__main__":
    import sys
    sys.exit(_main())
