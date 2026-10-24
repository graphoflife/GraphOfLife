"""
The rules of the author's walkthrough of 24 October 2026, held to what they say.

research/screen/walkthrough.md chose them for the screen's next round of
random worlds: hidden units that pass a change on (hidden_activation), links
that cost a token (link_token) or upkeep a brain chooses (link_upkeep,
upkeep_to), the lottery game (contest), a mate that pays (mate_pays) and an
estate that flows to the nearest of the living (redistribution = local).
Every one is off by default, and the old rules' own tests in test_engine.py
hold that a world without them is the world it always was.

    python3 tests/test_rules.py
"""

from __future__ import annotations

import json
import sys

import runner  # first: the repository on the path, and a runs folder of the tests' own

import networkx as nx
import numpy as np

from gol_config import SimConfig
from GraphOfLifeSimple import GraphOfLife, decision_keys, make_brain, new_world
from worlds import small


def _recorded(world, iterations):
    """Every frame a world records over some iterations, as canonical JSON."""
    return [json.dumps(frame, sort_keys=True)
            for _ in range(iterations) for frame in world.step(record_decisions=True)]


def _refused(cfg_kwargs, why):
    try:
        small(**cfg_kwargs).validate()
    except ValueError as exc:
        assert why in str(exc), (cfg_kwargs, str(exc))
    else:
        raise AssertionError(f"accepted {cfg_kwargs}")


# A world of each new rule, small enough to run many times.
RULES = {
    "tanh": dict(hidden_activation="tanh", hidden_layers=[12, 10, 8]),
    "link_token": dict(link_token=True),
    "link_token, mended": dict(link_token=True, mend_on_death=True, birth_links=2),
    "link_token, handover, reach": dict(link_token=True, allow_handover=True, link_search="reach"),
    "lottery": dict(contest="lottery", allow_revolutions=False, incumbency=1.5),
    "mate_pays": dict(mix_at_birth=True, mate_pays=True, hidden_layers=[16]),
    "upkeep, paid only": dict(link_upkeep="paid_only"),
    "upkeep, paid or used, to the estate": dict(link_upkeep="paid_or_used", upkeep_to="estate"),
    "local estate": dict(redistribution="local"),
    "local estate, mended": dict(redistribution="local", mend_on_death=True, birth_links=2),
    "everything": dict(hidden_activation="tanh", hidden_layers=[12, 10, 8], link_token=True,
                       link_upkeep="paid_or_used", upkeep_to="estate", contest="lottery",
                       allow_revolutions=False, mix_at_birth=True, mix_mates="parents",
                       mate_pays=True, redistribution="local", link_search="reach"),
}


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def test_every_new_rule_is_off_by_default_and_named_when_on():
    import gol_config
    assert SimConfig().strain_id() == "gol-1"
    for label, rule in RULES.items():
        cfg = small(**rule)
        cfg.validate()
        name = cfg.strain_id()
        for key, value in rule.items():
            if key not in gol_config.MECHANICS or value == gol_config.MECHANICS[key]:
                continue
            spelt = key if value is True else f"{key}={str(value).lower() if isinstance(value, bool) else value}"
            assert f"+{spelt}" in name, f"{label}: {spelt} is missing from {name}"


def test_what_each_new_rule_needs():
    _refused(dict(contest="lottery"), "allow_revolutions")
    _refused(dict(mate_pays=True), "mix_at_birth")
    _refused(dict(hidden_activation="tanh", brain_kind="binary", hidden_layers=[24, 16]), "binary")
    _refused(dict(upkeep_to="estate"), "link_upkeep")
    for setting in ("hidden_activation", "contest", "link_upkeep", "upkeep_to"):
        _refused({setting: "no-such-rule"}, setting)


def test_upkeep_adds_one_decision_to_the_brain_and_nothing_else():
    plain, paying = small(), small(link_upkeep="paid_only")
    assert paying.n_outputs() == plain.n_outputs() + 2
    assert "UPKEEP" in paying.head_layout() and "UPKEEP" not in plain.head_layout()
    assert paying.n_inputs() == plain.n_inputs()


# ---------------------------------------------------------------------------
# The brain's curve
# ---------------------------------------------------------------------------

def test_tanh_hidden_units_are_tanh_and_pass_a_change_through_three_layers():
    rng = np.random.default_rng(3)
    gains = {}
    for curve in ("sigmoid", "tanh"):
        cfg = small(hidden_activation=curve, hidden_layers=[16, 16, 16])
        brain = make_brain(cfg, 1, np.random.RandomState(4))
        x = rng.normal(0, 1, (cfg.n_inputs(), 9))
        squash = np.tanh if curve == "tanh" else (lambda z: 1 / (1 + np.exp(-z)))
        a = x
        for i, (W, b) in enumerate(zip(brain.weights, brain.biases)):
            z = W.astype(float) @ a + b.astype(float)
            a = z if i == len(brain.weights) - 1 else squash(z)
        assert np.allclose(brain.forward(x), a), f"a {curve} brain is not the network it says"
        eps = rng.normal(0, 0.1, x.shape)
        d = brain.forward(x + eps) - brain.forward(x)
        gains[curve] = float(np.sqrt((d ** 2).mean()) / np.sqrt((eps ** 2).mean()))
    assert gains["sigmoid"] < 0.05 < 0.2 < gains["tanh"], gains


# ---------------------------------------------------------------------------
# Tokens are conserved under every new rule, and every frame keeps its contract
# ---------------------------------------------------------------------------

def test_every_new_rule_conserves_tokens_with_the_links_counted():
    for label, rule in RULES.items():
        cfg = small(seed=21, **rule)
        world = new_world(cfg)
        held_by_links = cfg.link_token
        for _ in range(15):
            for frame in world.step(record_decisions=False):
                total = frame["summary"]["tokens"] + (frame["summary"]["edges"] if held_by_links else 0)
                assert total == cfg.total_tokens, (
                    f"{label}: {total} tokens at iteration {frame['iteration']}, phase {frame['phase']}")
                assert min(frame["tokens"], default=1) >= 0, f"{label}: an agent in debt"


def test_every_new_rule_keeps_the_frame_contract():
    for label, rule in RULES.items():
        cfg = small(seed=8, **rule)
        expected = decision_keys(cfg)
        world = new_world(cfg)
        for _ in range(4):
            for frame in world.step(record_decisions=True):
                decisions = frame["decisions"]
                if frame["phase"] == 1:
                    records = [("reproduction", decisions)] + [("birth", b) for b in decisions["births"]]
                else:
                    records = [("game", decisions)]
                    records += [("allocation", a) for a in decisions["allocations"]]
                    records += [("winner", w) for w in decisions["winners"]]
                for kind, record in records:
                    assert set(record) == expected[kind], (
                        f"{label}: a {kind} record holds {sorted(record)}, not {sorted(expected[kind])}")


def test_every_new_rule_resumes_to_the_same_run():
    for label, rule in RULES.items():
        cfg = small(seed=31, **rule)
        world = new_world(cfg)
        _recorded(world, 3)
        blob = {k: v.copy() for k, v in world.to_checkpoint().items()}
        straight_on = _recorded(world, 3)
        restored = GraphOfLife.from_checkpoint(blob, cfg)
        assert _recorded(restored, 3) == straight_on, f"{label}: the resumed run diverged"


# ---------------------------------------------------------------------------
# Links that hold a token
# ---------------------------------------------------------------------------

def test_a_link_costs_its_maker_a_token_and_no_token_makes_no_link():
    world = new_world(small(link_token=True))
    a, b = sorted(world.G.nodes())[:2]
    if world.G.has_edge(a, b):
        world._drop_edges([(a, b)])
    world.tokens[a] = 1
    assert world._make_link(a, b) and world.G.has_edge(a, b) and world.tokens[a] == 0
    world._drop_edges([(a, b)])
    assert not world._make_link(a, b) and not world.G.has_edge(a, b), "a maker without a token made a link"


def test_a_lapsing_link_gives_its_token_to_one_of_its_ends():
    world = new_world(small(link_token=True))
    u, v = sorted(world._edge_key(*e) for e in world.G.edges())[0]
    before = world.tokens[u] + world.tokens[v]
    world._lapse([(u, v)])
    assert not world.G.has_edge(u, v)
    assert world.tokens[u] + world.tokens[v] == before + 1


def test_the_links_of_the_dead_go_to_the_agents_they_joined():
    world = new_world(small(link_token=True))
    u = sorted(world.G.nodes())[0]
    neighbours = sorted(world.G.neighbors(u))
    held = {v: world.tokens[v] for v in neighbours}
    world._release_links({u})
    for v in neighbours:
        assert world.tokens[v] == held[v] + 1, "a neighbour of the dead did not get its link's token"


def test_a_world_with_link_tokens_pays_for_its_seed_graph():
    cfg = small(link_token=True)
    world = new_world(cfg)
    assert sum(world.tokens.values()) + world.G.number_of_edges() == cfg.total_tokens


# ---------------------------------------------------------------------------
# The lottery
# ---------------------------------------------------------------------------

def test_the_lottery_draws_the_taker_in_proportion_to_its_stake():
    world = new_world(small(contest="lottery", allow_revolutions=False))
    wins = sum(world._lottery({7: 3.0, 9: 1.0})[0] == 7 for _ in range(4000))
    assert 0.72 < wins / 4000 < 0.78, wins / 4000
    assert world._lottery({5: 2.0})[0] == 5


# ---------------------------------------------------------------------------
# A mate that pays
# ---------------------------------------------------------------------------

def test_a_mate_pays_its_child_and_never_its_last_token():
    cfg = small(mix_at_birth=True, mate_pays=True, hidden_layers=[16], seed=12)
    world = new_world(cfg)
    paid_any = False
    for _ in range(6):
        births = {}
        frame = world.reproduction_phase(record_decisions=True)
        for b in frame["decisions"]["births"]:
            births[b["child"]] = b["invested"]
        for mate, child, amount in frame["decisions"]["mate_paid"]:
            paid_any = True
            assert 1 <= amount <= births[child], "a mate paid more than the parent gave"
        world.blotto_phase(record_decisions=False)
        world.iteration += 1
    assert paid_any, "no mate ever paid"
    world._dues_of_mates = [(m, c, 5) for m, c in [(sorted(world.G.nodes())[0], sorted(world.G.nodes())[1])]]
    m = world._dues_of_mates[0][0]
    world.tokens[m] = 1
    assert world._pay_mates() == [] and world.tokens[m] == 1, "a mate gave its last token"


# ---------------------------------------------------------------------------
# Upkeep
# ---------------------------------------------------------------------------

def test_under_paid_only_every_link_left_was_paid_for_or_is_new():
    cfg = small(link_upkeep="paid_only", seed=14)
    world = new_world(cfg)
    for _ in range(3):
        world.step(record_decisions=False)
    for _ in range(4):
        frame = world.reproduction_phase(record_decisions=True)
        for payer, other in frame["decisions"]["upkeep"]:
            assert world.edge_paid_at.get(world._edge_key(payer, other)) == world.iteration
        world.blotto_phase(record_decisions=False)
        for u, v in world.G.edges():
            assert world.edge_paid_at.get(world._edge_key(u, v)) == world.iteration, (
                "a link nobody paid for outlived the game")
        world.iteration += 1


def test_upkeep_never_takes_an_agents_last_token():
    world = new_world(small(link_upkeep="paid_or_used"))
    u = sorted(world.G.nodes())[0]
    others = sorted(world.G.neighbors(u))
    world.tokens[u] = 2
    paid = world._pay_upkeep([(u, v, 1.0) for v in others])
    assert len(paid) == 1 and world.tokens[u] == 1, "upkeep took more than the agent could spare"


def test_upkeep_to_the_estate_joins_the_pool_of_the_dead():
    world = new_world(small(link_upkeep="paid_only", upkeep_to="estate"))
    u = sorted(world.G.nodes())[0]
    v = sorted(world.G.neighbors(u))[0]
    held_v = world.tokens[v]
    world._pay_upkeep([(u, v, 1.0)])
    assert world._upkeep_pool == 1 and world.tokens[v] == held_v
    world._cleanup_and_redistribute()
    assert world._upkeep_pool == 0


# ---------------------------------------------------------------------------
# The local estate
# ---------------------------------------------------------------------------

def _chain_world(redistribution="local"):
    """A line of five agents, 0 - 1 - 2 - 3 - 4, built by hand."""
    cfg = small(redistribution=redistribution, n_nodes=5, k_neighbors=2, total_tokens=50)
    world = GraphOfLife(None, cfg, _empty=True)
    for u in range(5):
        world.G.add_node(u)
        world._register_agent(u, 10, make_brain(cfg, u + 1, np.random.RandomState(u)), -1)
    for u in range(4):
        world._add_edge(u, u + 1)
    world.next_agent_id, world.next_brain_id = 5, 6
    return world


def test_a_local_estate_flows_step_by_step_to_the_nearest_living():
    world = _chain_world()
    # Agents 3 and 4 die; 3 is next to the living agent 2, and 4 two steps from it.
    given, rest = world._flow_estates({3: 10, 4: 10}, {3, 4})
    assert (given, rest) == (20, 0)
    assert world.tokens[2] == 30 and world.tokens[1] == 10, "the estates did not reach the nearest living agent"


def test_a_piece_cut_off_whole_flows_out_across_the_link_it_lost():
    world = _chain_world()
    # The link 2 - 3 lapses, cutting off 3 and 4; their estates still reach 2.
    world._lapsed = []
    world._lapse([(2, 3)])
    report = world._cleanup_and_redistribute()
    assert report["orphaned"] == 2
    assert sorted(world.G.nodes()) == [0, 1, 2]
    assert world.tokens[2] == 30 and world.tokens[0] == world.tokens[1] == 10


def test_an_estate_with_no_way_to_the_living_goes_to_the_parent():
    world = _chain_world()
    world.G.add_node(9)
    world._register_agent(9, 7, make_brain(world.cfg, 99, np.random.RandomState(9)), 0)
    given, rest = world._flow_estates({9: 7}, {9})
    assert (given, rest) == (7, 0) and world.tokens[0] == 17


def test_a_local_estate_splits_evenly_among_the_nearest():
    world = _chain_world()
    # Agent 2 dies between 1 and 3: each gets half.
    given, rest = world._flow_estates({2: 10}, {2})
    assert given == 10 and world.tokens[1] == world.tokens[3] == 15


if __name__ == "__main__":
    sys.exit(runner.main(globals()))
