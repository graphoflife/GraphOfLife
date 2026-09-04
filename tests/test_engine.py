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
    """
    import copy

    world = new_world(small(seed=3))
    for _ in range(3):
        world.step()

    for run in (world.reproduction_phase, world.blotto_phase):
        at_start = copy.deepcopy(world.messages)
        changed = []
        original = world._input_vec

        def watching(u, v, *args, **kwargs):
            for src, dst in ((u, u), (u, v), (v, u), (v, v)):
                if world.messages.get(src, {}).get(dst) != at_start.get(src, {}).get(dst):
                    changed.append((src, dst))
            return original(u, v, *args, **kwargs)

        world._input_vec = watching
        try:
            run(record_decisions=False)
        finally:
            world._input_vec = original

        assert not changed, (
            f"{len(changed)} reads returned a message that had changed since the "
            f"phase began, e.g. {changed[:3]}")


def test_each_phase_looks_once_per_agent():
    """
    The game phase used to observe twice: once to write messages, once to place
    stakes. That is a second forward pass through every brain for the same
    neighbourhood, and it made the game behave unlike reproduction, which has
    only ever looked once.
    """
    world = new_world(small(seed=3))
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

        assert len(calls) == present, (
            f"the {name} phase made {len(calls)} forward passes for {present} agents")


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
# The two ways the site is served
# ---------------------------------------------------------------------------

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
    build = open(os.path.join(root, "build_site.sh")).read()

    # A quoted path into one of the shipped directories, or a bare file next to
    # the script — which is what importScripts() takes.
    quoted = re.compile(r"""['"]((?:js|py|data|css)/[\w./-]+|[\w-]+\.js)['"]""")
    interesting = (".js", ".py", ".json", ".bin", ".css")

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
