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


if __name__ == "__main__":
    sys.exit(runner.main(globals()))
