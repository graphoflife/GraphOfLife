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
