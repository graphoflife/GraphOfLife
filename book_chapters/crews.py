"""
A crew of raiders (Experiment 33).

Crew raiders, which pool their stakes as revolution tokens on the weakest
stranger next door, planted as 5% of B2's five worlds played on at 2,000,
5,000 and 10,000 in three places, beside lone raiders and controls on the
same patches; and what a crew meets in its first game, read off the worlds as
they were planted (research/oee/strategies/mobs.py).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import numpy as np

import book_figures as F
import gol_plan
import gol_store
from book_figures import chapter, recipe
from book_chapters.common import BLUE, GREY, ORANGE, VIOLET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOBS = os.path.join(ROOT, "GraphOfLifeRuns", ".pilots", "atlas", "e33_mobs.json")
ERAS = [(2000, BLUE), (5000, ORANGE), (10000, VIOLET)]
KINDS = [("raider", "lone raiders", VIOLET), ("crew", "crews", ORANGE), ("resident", "controls", GREY)]
T = 300
RUNS = ("Experiment 33's 135 invasions (`python3 gol_lab.py run E33`): B2's worlds of Experiment 16 played on "
        "to 10,000 iterations as pilots of the engine's own rules (`GraphOfLifeRuns/.pilots/L/B2-long-s001` … "
        "`s005`), at iterations 2,000, 5,000 and 10,000, a patch of 5% on places 1 to 3, given the crew's rule, "
        "the lone raider's ([Strategies written by hand](strategies-written-by-hand.md)'s) or frozen copies of one patch member's brain (the control); the lone "
        "raiders' and the controls' plantings are Experiment 30's own.")
MOB_RUNS = ("No world is played: the worlds and places of Experiment 33's crews, read as they were planted — B2's "
            "worlds played on (`GraphOfLifeRuns/.pilots/L/B2-long-s001` … `s005`) at iterations 2,000, 5,000 and "
            "10,000, places 1 to 3.")
STEPS = ["Load the world of the era; on place p, draw an agent at random and take its nearest neighbours, step by "
         "step, until the patch holds 5% of the world.",
         "Give the patch the rule, or the control; play 300 iterations under the world's own rules, counting the "
         "patch's strategy and its tokens after every iteration.",
         "A planting takes hold when its strategy is then alive, holds more agents than the patch began with, and "
         "more than the control planted in the same place."]


def plantings() -> Dict[tuple, List[Dict[str, int]]]:
    """Each trial's series, by (world, era, strategy, place); empty until every trial is played."""
    found = {}
    for f in gol_plan.trials_of("E33"):
        t = f["trial"]
        path = os.path.join(gol_store.run_dir(gol_plan.trial_id_for(t)), "result.json")
        if not os.path.exists(path):
            return {}
        with open(path) as fh:
            found[(t["run"], t["era"], t["strategy"], t["place"])] = json.load(fh)["result"]["series"]
    return found


def life(series: List[Dict[str, int]]) -> int:
    """The iteration a planting's last agent died in, or T if it lived to the end."""
    return series[-1]["t"] if series[-1]["mine"] == 0 else T


def held(found, run: str, era: int, strategy: str, place: int) -> bool:
    s, c = found[(run, era, strategy, place)], found[(run, era, "resident", place)]
    mine = s[-1]["mine"] if s[-1]["t"] == T else 0
    control = c[-1]["mine"] if c[-1]["t"] == T else 0
    return mine > s[0]["mine"] and mine > control


@chapter
def crews(ch: F.Chapter) -> None:
    found = plantings()
    if not found:
        return
    places = sorted({(run, era, place) for run, era, _, place in found})
    eras = [e for e, _ in ERAS]
    cats = [f"{e:,}" for e in eras]
    took = {kind: [sum(held(found, r, e, kind, p) for r, ee, p in places if ee == e) for e in eras]
            for kind in ("raider", "crew")}
    bars = [{"label": label, "kind": "bars", "x0": [k + off for k in range(len(eras))],
             "x1": [k + off + 0.3 for k in range(len(eras))], "y": took[kind], "colour": colour}
            for (kind, label, colour), off in zip(KINDS[:2], (-0.32, 0.02))]
    dots = []
    for era, colour in ERAS:
        mine = [(life(found[(r, e, "raider", p)]), life(found[(r, e, "crew", p)])) for r, e, p in places if e == era]
        dots.append({"label": f"planted at {era:,}", "x": [a for a, _ in mine], "y": [b for _, b in mine],
                     "points": True, "size": 6, "colour": colour, "alpha": 0.85})
    dots.append({"label": None, "x": [0, T], "y": [0, T], "colour": GREY, "dash": [5, 4], "width": 1.2})
    ch.grid("hold", [
        {"title": "Plantings that took hold", "x": {"categories": cats, "label": "iteration the patch was planted"},
         "y": {"label": "of 15", "min": 0, "max": 15}, "series": bars},
        {"title": "How long each planting lived", "x": {"label": "the lone raiders' last iteration", "min": 0,
                                                        "max": T},
         "y": {"label": "the crew's last iteration", "min": 0, "max": T}, "series": dots}],
        title="Crews and lone raiders, planted in the same places", columns=2,
        caption=("Left: of the fifteen plantings at each era (five worlds, three places), how many took hold, lone "
                 "raiders (violet) and crews (orange). Right: each place once, by the iteration the lone raiders' "
                 "last agent died in and the crew's (300: still alive at the end). Below the dashed line the crew "
                 "died first."),
        recipe=recipe(RUNS, "each trial's `result.json`", STEPS))

    late = [(r, e, p) for r, e, p in places if e >= 5000]
    panels = []
    for key, title, ylabel in (("agents", "Agents of the patch", "share of the patch planted"),
                               ("wealth", "Tokens per agent of the patch", "tokens per agent")):
        series = []
        for kind, label, colour in KINDS:
            ts, ys = [], []
            for t in range(0, 151):
                vals = []
                for r, e, p in late:
                    s = found[(r, e, kind, p)]
                    row = s[t] if t < len(s) else None
                    if key == "agents":
                        vals.append(row["mine"] / s[0]["mine"] if row else 0.0)
                    elif row and row["mine"]:
                        vals.append(row["tokens"] / row["mine"])
                if vals:
                    ts.append(t)
                    ys.append(float(np.median(vals)))
            series.append({"label": label if key == "agents" else None, "x": ts, "y": ys, "colour": colour,
                           "width": 1.6})
        panels.append({"title": title, "x": {"label": "iterations after planting"}, "y": {"label": ylabel, "min": 0},
                       "series": series})
    ch.grid("wealth", panels, title="What became of the patches planted at 5,000 and 10,000", columns=2,
            caption=("The median over the thirty plantings at 5,000 and 10,000 of each kind: crews (orange), lone "
                     "raiders (violet), controls (grey). Left: the patch's strategy's agents as a share of the "
                     "patch planted (0 once it has died out). Right: their tokens per agent, over the plantings "
                     "still alive."),
            recipe=recipe(RUNS, "each trial's `result.json`", STEPS[:2]))

    ch.number("held", {kind: dict(zip(map(str, eras), took[kind])) for kind in took})
    ch.number("life", {str(e): {kind: float(np.median([life(found[(r, ee, kind, p)]) for r, ee, p in places
                                                       if ee == e])) for kind, _, _ in KINDS} for e in eras})
    ch.number("crew_died_first", {str(e): sum(life(found[(r, ee, "crew", p)]) < life(found[(r, ee, "raider", p)])
                                              for r, ee, p in places if ee == e) for e in eras})
    ch.number("wealth_at_10", {kind: float(np.median([found[(r, e, kind, p)][10]["tokens"] /
                                                      found[(r, e, kind, p)][10]["mine"] for r, e, p in late
                                                      if len(found[(r, e, kind, p)]) > 10
                                                      and found[(r, e, kind, p)][10]["mine"]]))
                               for kind, _, _ in KINDS})
    if os.path.exists(MOBS):
        _mobs(ch, json.load(open(MOBS)), eras, cats)


def _mobs(ch: F.Chapter, rows: List[Dict[str, Any]], eras: List[int], cats: List[str]) -> None:
    sizes = {e: {1: 0, 2: 0, 3: 0} for e in eras}
    outcome = {e: {"sure": 0, "maybe": 0, "lost": 0} for e in eras}
    for r in rows:
        for k, n in r["mob_sizes"].items():
            sizes[r["era"]][min(3, int(k))] += n
        for k in outcome[r["era"]]:
            outcome[r["era"]][k] += r["outcome"][k][0]
    width = 0.24
    mob_bars = [{"label": label, "kind": "bars", "x0": [k + off for k in range(len(eras))],
                 "x1": [k + off + width for k in range(len(eras))], "y": [sizes[e][m] for e in eras],
                 "colour": colour}
                for (m, label, colour), off in zip(((1, "one raider", GREY), (2, "two", ORANGE),
                                                    (3, "three or more", VIOLET)), (-0.38, -0.12, 0.14))]
    out_bars = [{"label": label, "kind": "bars", "x0": [k + off for k in range(len(eras))],
                 "x1": [k + off + width for k in range(len(eras))], "y": [outcome[e][key] for e in eras],
                 "colour": colour}
                for (key, label, colour), off in zip((("sure", "sure to win", BLUE), ("maybe", "might win", GREY),
                                                      ("lost", "sure to lose", ORANGE)), (-0.38, -0.12, 0.14))]
    ch.grid("mobs", [
        {"title": "Strangers by the raiders that chose them", "x": {"categories": cats,
                                                                     "label": "iteration the patch was planted"},
         "y": {"label": "strangers chosen, fifteen plantings", "min": 0}, "series": mob_bars},
        {"title": "What each stake could do", "x": {"categories": cats, "label": "iteration the patch was planted"},
         "y": {"label": "strangers chosen, fifteen plantings", "min": 0}, "series": out_bars}],
        title="A crew's first game, as planted", columns=2,
        caption=("Every stranger a crew raider would stake on in its first game, over the fifteen plantings of "
                 "each era. Left: how many raiders of the crew chose it. Right: whether the raiders that chose it "
                 "outweigh it even if it keeps everything at home (sure to win), only if it keeps no more than it "
                 "must (might win), or not even then (sure to lose: their tokens go to the stranger)."),
        recipe=recipe(MOB_RUNS, "the worlds' checkpoints at each era, read by `research/oee/strategies/mobs.py`",
                      ["Draw each patch as the planting draws it.",
                       "For every patch agent with a stranger next door: the weakest such stranger (the first, "
                       "where several are as weak); if the agent holds at least as much and has more than its "
                       "home share, it stakes all but that share there.",
                       "Raiders that chose the same stranger are one mob. Weigh the mob against the stranger's "
                       "stake at home, counted one and a half times: everything it holds (sure to win), or only "
                       "what B2 makes it keep, 30% rounded up and at least one token (might win). What other "
                       "neighbours stake on it is left out."]))
    tot = lambda key: {str(e): sum(r[key] for r in rows if r["era"] == e) for e in eras}
    ch.number("first_game", {"patch_agents": tot("patch"), "one_token": tot("one_token"),
                             "can_breed": tot("can_breed"), "crew_staking": tot("crew_staking"),
                             "lone_staking": tot("lone_raiders_staking"), "targets": tot("targets"),
                             "mob_sizes": {str(e): sizes[e] for e in eras},
                             "outcome": {str(e): outcome[e] for e in eras},
                             "agents_per_stranger": {str(e): float(np.mean([r["patch_agents_per_stranger"]
                                                                            for r in rows if r["era"] == e]))
                                                     for e in eras}})
