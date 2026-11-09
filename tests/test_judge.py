"""
The screen's judge, held to what it says: the author's lines for round 3 on
(research/screen/criteria.py), the numbers they are drawn on
(research/screen/measures.py), the criteria rounds 1 and 2 were judged by
(criteria_1.py), and the author's rule for marking a drawn value bad
(judge.py).

    python3 tests/test_judge.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys

import runner  # first: the repository on the path, and a runs folder of the tests' own

import numpy as np

import gol_plan
import gol_record
import gol_screen
import gol_store
from worlds import advanced_run, scratch_runs, small


def _judge():
    spec = importlib.util.spec_from_file_location(
        "screen_judge", os.path.join(runner.ROOT, "research", "screen", "judge.py"))
    judge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(judge)
    return judge


JUDGE = _judge()
C, C1, M = JUDGE.C, JUDGE.C1, JUDGE.M


def _world(changed, fate="lived", interesting=False, healthy=False):
    return {"round": "R1", "drawn": True, "fate": fate, "set": changed, "sampler": 1,
            "verdict": {"interesting": interesting, "healthy": healthy, "criteria": {"alive": fate == "lived"}}}


def _drawn(n, seed=77, **wanted):
    """n drawn worlds of the first sampler (in order) whose settings have the values asked for."""
    found, i = [], 0
    while len(found) < n:
        i += 1
        changed, _ = gol_screen.draw(1, seed, i)
        full = gol_screen.full(changed)
        if all(full[k] == v for k, v in wanted.items()):
            found.append(changed)
    return found


def _box(X, Y, Z, first=0):
    """The links of a box of agents, each joined to its neighbours along the three axes."""
    def at(x, y, z):
        return first + (x * Y + y) * Z + z
    pairs = []
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                pairs += [(at(x, y, z), at(x + 1, y, z))] if x + 1 < X else []
                pairs += [(at(x, y, z), at(x, y + 1, z))] if y + 1 < Y else []
                pairs += [(at(x, y, z), at(x, y, z + 1))] if z + 1 < Z else []
    return np.array(pairs)


def _every_line_met():
    """Averages a little inside every line."""
    out = {}
    for line in C.LINES:
        nudge = {"at most": -0.01, "below": -0.01, "at least": 0.01, "above": 0.01}[line["passes"]]
        out.setdefault(line["measure"], line["value"] + nudge)
    out["links"] = 6.0
    out["dimension"] = 3.0
    return out


# ---------------------------------------------------------------------------
# Round 3 on: the author's lines
# ---------------------------------------------------------------------------

def test_every_line_passes_on_its_own_side():
    """
    "At most" and "at least" keep their own value, "below" and "above" do not: 80% of the agents with
    the same tokens is not diverse, and neither is a largest holding of exactly 3. A number that could
    not be read passes no line.
    """
    line = {k: next(x for x in C.LINES if x["key"] == k) for k in ("agent_cut", "core", "tokens_alike", "max_tokens")}
    assert C.passes(line["agent_cut"], 0.15) and not C.passes(line["agent_cut"], 0.1501)
    assert C.passes(line["core"], 0.8) and not C.passes(line["core"], 0.79)
    assert not C.passes(line["tokens_alike"], 0.8) and C.passes(line["tokens_alike"], 0.79)
    assert not C.passes(line["max_tokens"], 3) and C.passes(line["max_tokens"], 3.2)
    assert not C.passes(line["core"], None) and not C.passes(line["core"], float("nan"))
    assert C.passes(line["agent_cut"], 0.2, at=0.3)


def test_a_world_is_judged_line_by_line():
    """
    A world inside every line is healthy. One line crossed fails its criterion and no other; two
    criteria missed is still interesting, provisionally, and three is not. A dead world is neither.
    """
    good = _every_line_met()
    got = C.judged(good, alive=True)
    assert got["healthy"] and got["interesting"] and got["misses"] == 0, got
    alike = C.judged({**good, "ages_alike": 0.85}, alive=True)
    assert not alike["lines"]["ages_alike"] and sum(not ok for ok in alike["lines"].values()) == 1
    assert [k for k, ok in alike["criteria"].items() if not ok] == ["diverse"]
    assert alike["interesting"] and not alike["healthy"]
    two = C.judged({**good, "ages_alike": 0.85, "gini": 0.3}, alive=True)
    assert two["misses"] == 2 and two["interesting"]
    three = C.judged({**good, "ages_alike": 0.85, "gini": 0.3, "width": None}, alive=True)
    assert three["misses"] == 3 and not three["interesting"]
    dead = C.judged(good, alive=False)
    assert not dead["healthy"] and not dead["interesting"] and not any(dead["criteria"].values())


def test_the_sensitivity_table_moves_one_line_at_a_time():
    """Each line drawn at each value it tries, the others where they are; dead worlds are not counted."""
    good = _every_line_met()
    worlds = [{"alive": True, "averages": {**good, "gini": 0.40}},
              {"alive": True, "averages": {**good, "gini": 0.50}},
              {"alive": False, "averages": {**good, "gini": 0.70}}]
    rows = {row["value"]: row for row in C.sensitivity(worlds) if row["line"] == "gini"}
    assert [rows[v]["pass"] for v in (0.3, 0.4, 0.46, 0.5, 0.6)] == [2, 2, 1, 1, 0]
    assert rows[0.46]["drawn"] and rows[0.46]["healthy"] == 1 and rows[0.3]["healthy"] == 2
    assert all(row["drawn"] == (row["value"] == C.VALUES[row["line"]]) for row in C.sensitivity(worlds))
    # Every line tries the value it is drawn at.
    assert all(line["value"] in line["tries"] for line in C.LINES)


def test_the_page_says_the_lines_where_they_are():
    """The criteria's words are made from the lines, so the page cannot say one thing and the judge another."""
    asks = {c["key"]: c["asks"] for c in C.CRITERIA}
    assert "5%" in asks["decentralised"] and "3 to 12" in asks["decentralised"]
    assert "80%" in asks["resilient"] and "15%" in asks["resilient"]
    assert "1.3 times" in asks["space"] and "2 to 5" in asks["space"]
    assert "more than 3" in asks["diverse"] and "95%" in asks["diverse"]
    assert "0.46" in asks["uneven"]
    assert {line["criterion"] for line in C.LINES} == set(C.KEYS[1:])
    measured = {line["measure"] for line in C.LINES} | {r["measure"] for r in C.REPORTED}
    assert {"half_cut", "apart"} <= measured


# ---------------------------------------------------------------------------
# What the numbers are made of
# ---------------------------------------------------------------------------

def test_how_alike_the_agents_are():
    """The share holding the most common value; for a yes-or-no choice, the share making the more common one."""
    assert M.alike([4, 4, 4, 9]) == 0.75 and M.alike([7]) == 1.0
    assert M.larger_side(0.3) == 0.7 and M.larger_side(1.0) == 1.0 and M.larger_side(None) is None


def test_the_cost_of_cutting_a_world_in_half():
    """
    A box of 16 by 8 by 8 agents is cut across its long side, through 64 links: an exponent of
    ln 64 / ln 1024 = 0.6. Two boxes joined by one link are cut there, at an exponent of 0. A
    world in pieces is cut in its largest.
    """
    box = M.half_cut(16 * 8 * 8, _box(16, 8, 8))
    assert box["crossing"] == 64 and abs(box["exponent"] - 0.6) < 1e-12, box
    pair = np.vstack([_box(6, 6, 6), _box(6, 6, 6, first=216), [[0, 216]]])
    got = M.half_cut(432, pair)
    assert got["crossing"] == 1 and got["exponent"] == 0.0, got
    apart = np.vstack([_box(6, 6, 6), _box(4, 4, 4, first=216)])
    assert M.half_cut(280, apart)["agents"] == 216


def test_a_world_is_read_from_what_it_recorded():
    """
    A small world read as the judge reads round 3: its statistics every iteration, the frames of its
    heavy games, and its kept worlds, every number averaged from the iteration asked on; the readings
    are kept and found again, and nothing before that iteration counts.
    """
    with scratch_runs():
        cfg = small(total_tokens=1000, n_nodes=40, k_neighbors=4, hidden_layers=[6],
                    archive_every=20, checkpoint_every=20, extinction_threshold=0)
        run = advanced_run(cfg, 40, record={"heavy_every": 5})
        items = M.items(run, 40, settle=10)
        assert items == [(run, "series", 40), (run, "kept", 20), (run, "kept", 40)], items
        have = {}
        for item in items:
            have[item] = M.read(item)
            M.keep(item, have)
        assert M.kept([run]) == {item: json.loads(json.dumps(r)) for item, r in have.items()}, \
            "the readings kept are not the readings made"
        averages = M.averages(have, items, settle=10)
        for key in ("agents", "links", "hub", "max_tokens", "gini", "children", "children_alike", "dimension",
                    "tokens_alike", "ages_alike", "genotypes_alike", "core", "core_links_alike", "agent_cut",
                    "link_cut", "half_cut", "half_cut_twin", "apart", "kinds"):
            assert key in averages, (key, sorted(averages))
        rows = gol_record.read_stats(run)
        games = [r for r in rows if r["phase"] == 2 and r["iteration"] >= 10]
        assert math.isclose(averages["gini"], np.mean([r["gini"] for r in games]))
        heavy = [r for r in games if r["_heavy"]]
        assert [f["t"] for f in have[items[0]]["frames"] if f["t"] >= 10] == [r["iteration"] for r in heavy]
        assert 0 <= averages["apart"] <= 1 and 0 < averages["children_alike"] <= 1
        later = M.averages(have, items, settle=30)
        assert math.isclose(later["gini"], np.mean([r["gini"] for r in games if r["iteration"] >= 30]))


def test_a_round_three_world_carries_its_drawn_founders():
    """
    Round 3 draws the number of founders, which the plan keeps with the world rather than the
    condition: the judge puts it back beside the other drawn settings, so a value is judged where
    it was drawn.
    """
    with scratch_runs():
        worlds = JUDGE.worlds_of(["E40"])
        draws = gol_screen.draws(gol_plan.read_plan("E40")["screen"])
        skipped = gol_plan.read_plan("E40")["screen"].get("skip") or []
        assert len(worlds) == len(draws) == 100 - len(skipped)
        assert not {w["seed"] - 3000 for w in worlds} & set(skipped)
        for w, d in zip(worlds, draws):
            assert w["set"].get("n_nodes") == d["set"].get("n_nodes"), (w["id"], d)
            assert w["fate"] == "not made"
        assert {gol_screen.full(w["set"], 2)["n_nodes"] for w in worlds} == {0, 150, 600}


# ---------------------------------------------------------------------------
# Rounds 1 and 2, and the rule for marking a value bad
# ---------------------------------------------------------------------------

def test_a_world_that_misses_one_criterion_was_still_interesting_in_rounds_one_and_two():
    """The author's room for variability then: alive, and at most one of the other criteria missed."""
    everything = {key: True for key in C1.KEYS}
    assert C1.healthy(everything) and C1.interesting(everything)
    one_off = {**everything, "space": False}
    assert not C1.healthy(one_off) and C1.interesting(one_off)
    two_off = {**one_off, "uneven": False}
    assert not C1.interesting(two_off)
    assert not C1.interesting({**everything, "alive": False})


def test_a_value_is_marked_bad_only_after_ten_worlds_that_all_died_or_were_all_dull():
    """
    The author's rule: tried in at least ten worlds where it could matter, and every one died, or
    none was interesting. Nine are not enough, and one interesting world among ten saves a value.
    """
    # Worlds without a cap on links, every one dead; the others interesting.
    dead = [_world(c, fate="died") for c in _drawn(10, valence=0)]
    living = [_world(c, interesting=True) for c in _drawn(12, seed=78, valence=4)]
    got = JUDGE.marking(dead + living, {})
    bad = {(v["setting"], json.dumps(v["value"])) for v in got["values"] if v["bad"]}
    assert ("valence", "0") in bad, bad
    assert got["forced_next"]["valence"] == [0]
    assert not any(s == "valence" and v != "0" for s, v in bad)
    record = next(v for v in got["values"] if v["setting"] == "valence" and v["value"] == 0)
    assert record["met"] == {"alive": 0}, record
    # Nine dead worlds: not yet.
    got = JUDGE.marking(dead[:9] + living, {})
    assert not any(v["bad"] for v in got["values"] if v["setting"] == "valence" and v["value"] == 0)
    # Ten worlds, none dead but none interesting: marked. One of them interesting: saved.
    dull = [_world(c) for c in _drawn(10, valence=0)]
    got = JUDGE.marking(dull + living, {})
    assert any(v["bad"] for v in got["values"] if v["setting"] == "valence" and v["value"] == 0)
    saved = dull[:9] + [_world(dull[9]["set"], interesting=True)]
    got = JUDGE.marking(saved + living, {})
    assert not any(v["bad"] for v in got["values"] if v["setting"] == "valence" and v["value"] == 0)


# ---------------------------------------------------------------------------
# The proposed order of health, the settings' tests, fertility passed on, and the shortlist (2026-11-07)
# ---------------------------------------------------------------------------

def _screen(name):
    spec = importlib.util.spec_from_file_location(
        f"screen_{name}", os.path.join(runner.ROOT, "research", "screen", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HEALTH, EFFECTS, FAMILIES, FUNNEL, CULLS = (_screen(name) for name in ("health", "effects", "families", "funnel",
                                                                        "culls"))


def _line(key):
    return next(line for line in HEALTH.lines() if line["key"] == key)


def test_a_line_is_read_as_the_share_of_what_it_asks():
    near = lambda a, b: abs(a - b) < 1e-9
    assert near(HEALTH.progress(_line("core"), 0.76), 0.95)            # 76 of the 80 the line asks
    assert HEALTH.progress(_line("core"), 0.85) == 1.0                  # more than enough counts as enough
    assert near(HEALTH.progress(_line("hub"), 0.30), 0.05 / 0.30)       # six times over the line
    assert near(HEALTH.progress(_line("width"), 1.1), 1 / 3)            # counted from the random twin's 1
    assert near(HEALTH.progress(_line("links_min"), 2.5), 0.5)          # from a tree's two links an agent
    assert near(HEALTH.progress(_line("tokens_alike"), 0.9), 0.5)       # 0.1 unlike, of the 0.2 asked
    assert near(HEALTH.progress(_line("richest"), 0.20), 0.5)           # twice the share of the tokens allowed
    assert HEALTH.progress(_line("width"), 0.9) == 0.0                  # below its twin: none of it
    assert HEALTH.progress(_line("core"), None) == 0.0                  # not read: nothing met


def test_health_is_the_sum_of_the_criteria_each_as_its_weakest_line():
    well = {"hub": 0.01, "links": 4.0, "richest": 0.01, "core": 0.9, "agent_cut": 0.05, "link_cut": 0.05,
            "width": 2.0, "dimension": 3.0, "half_cut": 0.6, "tokens_alike": 0.3, "max_tokens": 100.0,
            "core_links_alike": 0.3, "ages_alike": 0.3, "children_alike": 0.7, "apart": 0.8, "gini_random": 1.2}
    got = HEALTH.score(well)
    assert got["health"] == 5.0 and got["met"] == 5, got
    got = HEALTH.score({**well, "core": 0.4})                           # half the 2-core asked: resilient 0.5
    assert abs(got["criteria"]["resilient"] - 0.5) < 1e-9 and abs(got["health"] - 4.5) < 1e-9, got
    got = HEALTH.score({**well, "core": 0.4, "agent_cut": 0.6})         # the weakest line decides: 0.25
    assert abs(got["criteria"]["resilient"] - 0.25) < 1e-9, got
    # The author's own lines: five criteria too, the genotype line and B2's Gini among them.
    author = HEALTH.score({**well, "genotypes_alike": 0.01, "gini": 0.5}, proposed=False)
    assert set(author["criteria"]) == set(C.KEYS[1:]) and author["met"] == 5, author


def test_a_cube_is_harder_to_halve_than_a_plane_and_a_dense_one_than_a_sparse_one():
    exponent = lambda dim, links: M.half_cut(1500, HEALTH.geometric(1500, dim, links, 3))["exponent"]
    plane, cube, dense = exponent(2, 6), exponent(3, 6), exponent(3, 9)
    assert plane < cube < dense, (plane, cube, dense)


def test_tokens_dealt_at_random_have_the_gini_the_line_is_drawn_at():
    rng = np.random.default_rng(3)
    for mean in (1.5, 3.0, 7.0, 20.0):
        x = np.sort(rng.geometric(1.0 / mean, 200000).astype(float))
        n = x.size
        gini = float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))
        assert abs(gini - HEALTH.random_gini(mean)) < 0.01, (mean, gini, HEALTH.random_gini(mean))
    assert HEALTH.random_gini(1.0) == 0.0                               # one token each: no spread at all


def test_shuffling_finds_a_difference_only_where_there_is_one():
    rng = np.random.RandomState(5)
    groups = ["a"] * 30 + ["b"] * 30
    chance, size = EFFECTS.permutation_test(np.array([0] * 25 + [1] * 5 + [0] * 5 + [1] * 25), groups, True, rng)
    assert chance < 0.001 and size > 0.3, (chance, size)
    chance, _ = EFFECTS.permutation_test(rng.permutation(np.array([0, 1] * 30)), groups, True, rng)
    assert chance > 0.05, chance
    chance, size = EFFECTS.permutation_test(np.arange(60, dtype=float), groups, False, rng)
    assert chance < 0.001 and size > 0.5, (chance, size)


def test_false_discovery_rates_follow_benjamini_and_hochberg():
    got = EFFECTS.false_discovery([0.01, 0.04, 0.03, 0.2])
    assert np.allclose(got, [0.04, 0.16 / 3, 0.16 / 3, 0.2]), got


def _tree(heritable):
    """A family tree in which fertility is passed on to children, or drawn afresh for every child."""
    class Tree:
        def __init__(self, run_id, until):
            rng = np.random.RandomState(2)
            size = 1 << 17
            self.parent = np.full(size, -2, dtype=np.int64)
            self.born = np.full(size, -1, dtype=np.int32)
            self.parent[:300], self.born[:300] = -1, 0
            fertile = {i: rng.choice([0.02, 0.06, 0.12]) for i in range(300)}
            new, alive = 300, list(range(300))
            for t in range(1, until):
                born = []
                for a in alive:
                    if rng.rand() < fertile[a]:
                        self.parent[new], self.born[new] = a, t
                        keep = heritable and rng.rand() > 0.3
                        fertile[new] = fertile[a] if keep else rng.choice([0.02, 0.06, 0.12])
                        born.append(new)
                        new += 1
                alive = [a for a in alive if rng.rand() > 0.07] + born
                if len(alive) > 2000:
                    alive = list(rng.choice(alive, 2000, replace=False))
    return Tree


def test_fertility_passed_on_is_told_from_chance():
    real = FAMILIES.K.Tree
    try:
        FAMILIES.K.Tree = _tree(heritable=True)
        passed = FAMILIES.fertility("made-up", 600)
        FAMILIES.K.Tree = _tree(heritable=False)
        drawn = FAMILIES.fertility("made-up", 600)
    finally:
        FAMILIES.K.Tree = real
    assert passed["z"] > 3, passed
    assert drawn["z"] < 2, drawn


def test_a_phase_is_rebuilt_as_it_left_the_network_before_its_cleanup():
    prev = {"ids": [1, 2, 3], "edges": [[1, 2], [1, 3]]}
    # A child of 1, born linked to it, takes over 1's link to 2; a search joins 3 to the child; 1-3 lapses.
    frame = {"decisions": {"births": [{"agent": 1, "child": 9, "links": [1], "handed_over": [2]}],
                           "proposals": [[3, 9, True], [2, 3, False]], "pruned_edges": [[1, 3]]}}
    assert CULLS.before_cleanup(prev, frame) == {1: {9}, 2: {9}, 3: {9}, 9: {1, 2, 3}}


def test_what_an_agent_held_is_read_from_its_stakes_and_its_births():
    # A game: a node holds what was staked on it; under link_token a lapsed link may have paid either end a token.
    prev = {"ids": [1, 2, 3, 4], "tokens": [3, 3, 3, 3], "edges": []}
    game = {"phase": 2, "decisions": {"allocations": [{"targets": [1, 2], "alloc": [2, 1]},
                                                      {"targets": [2, 1, 3], "alloc": [0, 0, 3]}],
                                      "pruned_edges": [[2, 4]]}}
    assert CULLS.holdings(prev, game, {"link_token": True}) == ({1: 2, 2: 1, 3: 3, 4: 0}, {1: 2, 2: 2, 3: 3, 4: 1})
    assert CULLS.holdings(prev, game, {})[1][4] == 0
    # Reproduction: 1 gives its child 9 all it has; 9 pays for its two links under link_token; 3 gives 2 a token.
    birth = {"phase": 1, "decisions": {"births": [{"agent": 1, "tokens_before": 3, "invested": 3, "child": 9,
                                                   "links": [1, 2], "handed_over": []}],
                                       "gifts": [[3, 2, 1]]}}
    least, most = CULLS.holdings(prev, birth, {"link_token": True})
    assert least == most == {1: 0, 2: 4, 3: 2, 4: 3, 9: 1}
    # Searchers may have paid tokens the record leaves out: then nobody is surely broke.
    assert set(CULLS.holdings(prev, birth, {"link_search": "walk"})[1].values()) == {None}


def test_a_piece_culled_counts_the_culled_and_leaves_one_by_one():
    # 1 holds the world together: 2 and 3 hang from it, and 4 hangs from it with two leaves of its own.
    adj = {1: {2, 3, 4}, 2: {1}, 3: {1}, 4: {1, 5, 6}, 5: {4}, 6: {4}}
    held = {4: 0, 5: 3, 6: 2}
    # 4 starved: its two leaves are culled, each alone.
    assert CULLS.pieces(adj, {1, 2, 3}, 1, 2, held, held, mend=False) == (1, 1, False)
    # The link 1-4 lapsed instead: 4 and its leaves are culled as one piece of three.
    adj = {1: {2, 3}, 2: {1}, 3: {1}, 4: {5, 6}, 5: {4}, 6: {4}}
    held = {4: 1, 5: 1, 6: 1}
    assert CULLS.pieces(adj, {1, 2, 3}, 0, 3, held, held, mend=False) == (3, 3, False)
    # A piece cut off whole by a lapse, with an agent that starved inside it: under mending its death splits
    # nothing, so the piece is the two culled; without, they are two pieces of one.
    adj = {1: {2}, 2: {1}, 3: {4}, 4: {3, 5}, 5: {4}}
    held = {3: 2, 4: 0, 5: 2}
    assert CULLS.pieces(adj, {1, 2}, 1, 2, held, held, mend=True) == (2, 2, False)
    assert CULLS.pieces(adj, {1, 2}, 1, 2, held, held, mend=False) == (1, 1, False)
    # Two agents the record leaves open, of whom one starved: the piece is two surely, four at most.
    adj = {1: {2}, 2: {1}, 3: {4}, 4: {3, 5}, 5: {4, 6}, 6: {5}}
    least, most = {3: 2, 4: 0, 5: 2, 6: 0}, {3: 2, 4: None, 5: 2, 6: None}
    assert CULLS.pieces(adj, {1, 2}, 1, 3, least, most, mend=True) == (2, 4, False)
    # The counts settle an open agent when they can: here only one starved, and one was open.
    assert CULLS.pieces(adj, {1, 2}, 1, 3, {**least, 6: 1}, {**most, 6: 1}, mend=True) == (3, 3, False)
    # An agent that surely held a token cannot have been removed beside a survivor.
    adj = {1: {2, 3}, 2: {1}, 3: {1}}
    assert CULLS.pieces(adj, {1, 2}, 0, 1, {3: 2}, {3: 2}, mend=False)[2]
    # A big cull counts once it is more than BIG_CULL of the world, as often as it happens; surely, or at most.
    cut = {"lost": [[150, 1, 0.4, 0.4], [151, 2, 0.1, 0.1], [300, 1, 0.1, 0.2]]}
    assert HEALTH.big_culls(cut, 100, 300) == 0.5 and HEALTH.big_culls(cut, 100, 400) == 2 / 3
    assert HEALTH.big_culls(cut, 100, 400, surely=True) == 1 / 3


def test_the_shortlist_is_every_candidate_no_other_beats_in_both():
    ranks = lambda r: {k: r for k in FUNNEL.EVOLUTION}
    worlds = [{"id": "a", "numbers": {"health": 5.0}, "ranks": ranks(1)},
              {"id": "b", "numbers": {"health": 4.0}, "ranks": ranks(2)},      # a is better in both
              {"id": "c", "numbers": {"health": 5.5}, "ranks": ranks(3)}]      # the healthiest, though less evolving
    assert FUNNEL.pareto(worlds) == ["a", "c"], FUNNEL.pareto(worlds)


if __name__ == "__main__":
    sys.exit(runner.main(globals()))
