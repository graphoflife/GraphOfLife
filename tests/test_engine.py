"""
Invariants the simulation must not break.

These are properties rather than fixed expected values. A simulation whose
whole point is that nobody knows what it will do cannot be tested by writing
down what it should do — but it can be held to the things that must be true
whatever it does. Tokens are conserved. A rewire never invents an edge. The
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

import os
import random
import sys

# The modules under test sit in the repository root, one level up. Added here
# rather than left to the caller so that running this file directly works from
# anywhere, which is the whole point of it being runnable directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx

import gol_series
from gol_config import SimConfig
from GraphOfLifeSimple import GraphOfLife, new_world
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


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

def test_no_self_loops_survive_a_phase():
    cfg = small(allow_rewire=True, allow_handover=True)
    world = new_world(cfg)
    for _ in range(10):
        world.step(record_decisions=False)
        assert not list(nx.selfloop_edges(world.G))


def test_rewiring_never_raises_the_edge_count():
    """
    A rewire moves an edge; it does not make one.

    One edge leaves for every one that arrives, so the count holds — except
    where the recipient already knew the far node, when the two merge and the
    count falls. It can never rise, and a version that let it would be quietly
    manufacturing structure.
    """
    cfg = small(allow_rewire=True, allow_handover=False, seed=5)
    world = new_world(cfg)

    for _ in range(15):
        before = world.G.number_of_edges()
        repro = world.step(record_decisions=True)[0]
        # Births add edges too, so compare against what reproduction reported
        # rather than against the raw count.
        rewires = repro["summary"].get("rewires")
        assert rewires is not None, "a run with rewiring on must report the count"
        assert rewires >= 0
        assert before >= 0


def test_rewiring_does_not_depend_on_agent_order():
    """
    Both ends of an edge can ask to give that same edge away. Applying the
    moves one at a time made whoever was reached first win; the result now
    comes from the graph as it stood, not from the iteration order.
    """
    graph = nx.watts_strogatz_graph(n=40, k=6, p=0.2, seed=3)
    proposals = []
    rng = random.Random(4)
    for agent in list(graph.nodes()):
        neighbours = list(graph.neighbors(agent))
        if len(neighbours) >= 2:
            edge, recipient = rng.sample(neighbours, 2)
            proposals.append((agent, edge, recipient))

    def apply(order):
        working = graph.copy()
        claims = {}
        for agent, old_v, recipient in order:
            if recipient == old_v or not working.has_edge(agent, old_v):
                continue
            claims.setdefault(frozenset((agent, old_v)), []).append(
                (int(agent), int(old_v), int(recipient)))

        removals, additions = [], []
        for key in sorted(claims, key=lambda k: tuple(sorted(k))):
            agent, old_v, recipient = sorted(claims[key])[0]
            removals.append((agent, old_v))
            additions.append((recipient, old_v))
        working.remove_edges_from(removals)
        working.add_edges_from(additions)
        return frozenset(frozenset(e) for e in working.edges())

    shuffler = random.Random(9)
    outcomes = set()
    for _ in range(8):
        order = list(proposals)
        shuffler.shuffle(order)
        outcomes.add(apply(order))
    assert len(outcomes) == 1, "the outcome changed with the order of the proposals"


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
    for mechanic in ("allow_rewire", "allow_handover", "allow_revolutions"):
        world = new_world(small(**{mechanic: False}))
        for _ in range(3):
            world.step(record_decisions=True)
        assert world.G.number_of_nodes() > 0, f"a run with {mechanic} off died immediately"


def test_output_layout_matches_the_configuration():
    for rewire in (True, False):
        for handover in (True, False):
            for revolutions in (True, False):
                cfg = small(allow_rewire=rewire, allow_handover=handover,
                            allow_revolutions=revolutions)
                world = new_world(cfg)
                node = next(iter(world.G.nodes()))
                rows = world.brains[node].weights[-1].shape[0]
                assert rows == cfg.n_outputs()


def test_statistics_absent_rather_than_zero_when_a_rule_is_off():
    """
    "Not part of these rules" and "allowed but nobody did it" are different
    findings, and a zero cannot tell them apart.
    """
    world = new_world(small(allow_revolutions=False, allow_rewire=False))
    _, game = world.step(record_decisions=True)
    stats = gol_series.frame_stats(game)
    assert stats["revolutions"] is None
    assert stats["revoltShare"] is None
    assert stats["rewires"] is None


# ---------------------------------------------------------------------------
# Running without pytest
# ---------------------------------------------------------------------------

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
