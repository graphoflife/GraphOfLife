"""
Raiders that breed (Experiment 34) and hand-written strategies in B1
(Experiment 35).

Patches of 5% given a strategy written by hand, planted at 2,000, 5,000 and
10,000 iterations in three places: in B2's five worlds played on, raiders and
turtles that have a child whenever they can give it the world's least child,
beside the barren raiders and the controls of the same places; and in B1's two
long worlds, every strategy at once. What became of each patch over its 300
iterations, read from each trial's series; and how many agents of the worlds
planted in could afford a child at all, read from the worlds as they were kept.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np

import book_figures as F
import gol_plan
import gol_store
from book_figures import chapter, recipe
from book_chapters.common import GREEN, GREY, ORANGE, RED, VIOLET

T = 300
ERAS = [2000, 5000, 10000]
B2_WORLDS = [f"pilot:L/B2-long-s{s:03d}" for s in range(1, 6)]
B1_WORLDS = ["pilot:L/B1-long-s1", "pilot:L/B1-long-s2"]
B2_KINDS = [("breeding-raider", "breeding raiders", ORANGE), ("raider", "lone raiders", VIOLET),
            ("breeding-turtle", "breeding turtles", GREEN), ("resident", "controls", GREY)]
B1_KINDS = [("crew", "crews", RED)] + B2_KINDS
B2_RUNS = ("Experiment 34's 180 invasions (`python3 gol_lab.py run E34`): B2's worlds of Experiment 16 played on "
           "to 10,000 iterations as pilots of the engine's own rules (`GraphOfLifeRuns/.pilots/L/B2-long-s001` … "
           "`s005`), at iterations 2,000, 5,000 and 10,000, a patch of 5% on places 1 to 3, given the breeding "
           "raider's rule, the breeding turtle's, the lone raider's ([Strategies written by hand]"
           "(strategies-written-by-hand.md)'s) or frozen copies of one patch member's brain (the control); the "
           "lone raiders' and the controls' plantings are those of Experiments 30 and 33.")
B1_RUNS = ("Experiment 35's 90 invasions (`python3 gol_lab.py run E35`): B1's two worlds played on to 10,000 "
           "iterations as pilots (`GraphOfLifeRuns/.pilots/L/B1-long-s1` and `s2`), at iterations 2,000, 5,000 "
           "and 10,000, a patch of 5% on places 1 to 3, given the crew's rule, the breeding raider's, the breeding "
           "turtle's, the lone raider's or frozen copies of one patch member's brain (the control), each rule "
           "reading B1's home share, holder's bonus and least child.")
STEPS = ["Load the world of the era; on place p, draw an agent at random and take its nearest neighbours, step by "
         "step, until the patch holds 5% of the world.",
         "Give the patch the rule, or the control; play 300 iterations under the world's own rules, counting the "
         "patch's strategy's agents and their tokens after every iteration (each trial's `result.json`)."]


def plantings(experiment: str) -> Dict[tuple, List[Dict[str, int]]]:
    """Each trial's series, by (world, era, strategy, place); empty until every trial is played."""
    found = {}
    for f in gol_plan.trials_of(experiment):
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


def at_end(series: List[Dict[str, int]], key: str) -> int:
    return series[-1][key] if series[-1]["t"] == T else 0


def held(found, place: Tuple[str, int, int], kind: str, key: str) -> bool:
    """Alive after T, with more agents (or tokens) than the patch began with and than the control."""
    run, era, p = place
    s, c = found[(run, era, kind, p)], found[(run, era, "resident", p)]
    return at_end(s, key) > s[0][key] and at_end(s, key) > at_end(c, key)


def _places(found) -> List[Tuple[str, int, int]]:
    return sorted({(run, era, p) for run, era, _, p in found})


def _counts(found, kinds, test) -> Dict[str, List[int]]:
    places = _places(found)
    return {kind: [sum(test(pl, kind) for pl in places if pl[1] == e) for e in ERAS] for kind, _, _ in kinds}


def _bars(kinds, values: Dict[str, List[int]]) -> List[dict]:
    width = 0.8 / len(kinds)
    return [{"label": label, "kind": "bars",
             "x0": [k - 0.4 + i * width for k in range(len(ERAS))],
             "x1": [k - 0.4 + (i + 0.85) * width for k in range(len(ERAS))],
             "y": values[kind], "colour": colour}
            for i, (kind, label, colour) in enumerate(kinds)]


def _median_course(found, places, kind: str, key: str, per_agent: bool) -> Tuple[List[int], List[float]]:
    """The median over the plantings, at each iteration after planting, of the patch's share of what it began
    with (0 once it has died out), or of its tokens per agent over the plantings still alive."""
    ts, ys = [], []
    for t in range(T + 1):
        vals = []
        for run, era, p in places:
            s = found[(run, era, kind, p)]
            row = s[t] if t < len(s) else None
            if per_agent:
                if row and row["mine"]:
                    vals.append(row["tokens"] / row["mine"])
            else:
                vals.append(row[key] / s[0][key] if row else 0.0)
        if vals:
            ts.append(t)
            ys.append(float(np.median(vals)))
    return ts, ys


def _able(worlds: List[str], least: int) -> Dict[str, float]:
    """The share of the agents of the worlds planted in that hold enough for the world's least child and one kept."""
    out = {}
    for era in ERAS:
        tokens = np.concatenate([gol_store.kept_world(w, era)["tokens"] for w in worlds])
        out[str(era)] = float(np.mean(tokens >= least + 1))
    return out


def _numbers(ch: F.Chapter, found, kinds) -> None:
    places = _places(found)
    ch.number("held_by_agents", _counts(found, kinds, lambda pl, k: held(found, pl, k, "mine")))
    ch.number("held_by_tokens", _counts(found, kinds, lambda pl, k: held(found, pl, k, "tokens")))
    ch.number("alive", _counts(found, kinds, lambda pl, k: at_end(found[(pl[0], pl[1], k, pl[2])], "mine") > 0))
    ch.number("life", {kind: {str(e): float(np.median([life(found[(r, ee, kind, p)]) for r, ee, p in places
                                                       if ee == e])) for e in ERAS} for kind, _, _ in kinds})
    # The median planting's largest size hides the few that grew: count them too.
    ch.number("grew", _counts(found, kinds, lambda pl, k: max(x["mine"] for x in found[(pl[0], pl[1], k, pl[2])])
                              > found[(pl[0], pl[1], k, pl[2])][0]["mine"]))
    ch.number("peak", {kind: {str(e): float(np.median([max(x["mine"] for x in found[(r, ee, kind, p)])
                                                       / found[(r, ee, kind, p)][0]["mine"]
                                                       for r, ee, p in places if ee == e])) for e in ERAS}
                       for kind, _, _ in kinds})
    ch.number("patch", {str(e): float(np.median([found[(r, ee, "resident", p)][0]["mine"] for r, ee, p in places
                                                 if ee == e])) for e in ERAS})
    ch.number("tokens_per_agent_planted", {str(e): float(np.median([found[(r, ee, "resident", p)][0]["tokens"]
                                                                    / found[(r, ee, "resident", p)][0]["mine"]
                                                                    for r, ee, p in places if ee == e]))
                                           for e in ERAS})


@chapter
def breeders(ch: F.Chapter) -> None:
    found = plantings("E34")
    if not found:
        return
    cats = [f"{e:,}" for e in ERAS]
    took = _counts(found, B2_KINDS[:3], lambda pl, k: held(found, pl, k, "mine"))
    alive = _counts(found, B2_KINDS, lambda pl, k: at_end(found[(pl[0], pl[1], k, pl[2])], "mine") > 0)
    ch.grid("hold", [
        {"title": "Plantings that took hold", "x": {"categories": cats, "label": "iteration the patch was planted"},
         "y": {"label": "of 15", "min": 0, "max": 15}, "series": _bars(B2_KINDS[:3], took)},
        {"title": "Plantings still alive after 300 iterations", "x": {"categories": cats,
                                                                      "label": "iteration the patch was planted"},
         "y": {"label": "of 15", "min": 0, "max": 15}, "series": _bars(B2_KINDS, alive)}],
        title="Breeding raiders, beside barren raiders, breeding turtles and controls", columns=2,
        caption=("Of the fifteen plantings at each era (five worlds, three places). Left: how many took hold — "
                 "alive after 300 iterations, with more agents than the patch began with and than the control "
                 "planted in the same place: breeding raiders (orange), lone raiders (violet), breeding turtles "
                 "(green). Right: how many were still alive at all, the controls (grey) beside them."),
        recipe=recipe(B2_RUNS, "each trial's `result.json`",
                      STEPS + ["A planting takes hold when its strategy is alive after 300 iterations, holds more "
                               "agents than the patch began with, and more than the control planted in the same "
                               "place; count the plantings of each era that took hold, and those alive at all."]))

    late = [pl for pl in _places(found) if pl[1] >= 5000]
    panels = []
    for per_agent, title, ylabel in ((False, "Agents of the patch", "share of the patch planted"),
                                     (True, "Tokens per agent of the patch", "tokens per agent")):
        series = []
        for kind, label, colour in B2_KINDS:
            ts, ys = _median_course(found, late, kind, "mine", per_agent)
            series.append({"label": None if per_agent else label, "x": ts, "y": ys, "colour": colour,
                           "width": 1.6})
        panels.append({"title": title, "x": {"label": "iterations after planting", "min": 0, "max": T},
                       "y": {"label": ylabel, "min": 0}, "series": series})
    ch.grid("patches", panels, title="What became of the patches planted at 5,000 and 10,000", columns=2,
            caption=("The median over the thirty plantings at 5,000 and 10,000 of each kind: breeding raiders "
                     "(orange), lone raiders (violet), breeding turtles (green), controls (grey). Left: the "
                     "strategy's agents as a share of the patch planted (0 once it has died out). Right: their "
                     "tokens per agent, over the plantings still alive."),
            recipe=recipe(B2_RUNS, "each trial's `result.json`",
                          STEPS + ["At each iteration after planting, take the median over the thirty plantings "
                                   "of the strategy's agents over the patch planted, and of its tokens per agent "
                                   "over the plantings still alive."]))
    _numbers(ch, found, B2_KINDS)
    ch.number("able", _able(B2_WORLDS, 10))


@chapter
def b1raiders(ch: F.Chapter) -> None:
    found = plantings("E35")
    if not found:
        return
    cats = [f"{e:,}" for e in ERAS]
    alive = _counts(found, B1_KINDS, lambda pl, k: at_end(found[(pl[0], pl[1], k, pl[2])], "mine") > 0)
    places = _places(found)
    lived = {kind: [float(np.median([life(found[(r, ee, kind, p)]) for r, ee, p in places if ee == e]))
                    for e in ERAS] for kind, _, _ in B1_KINDS}
    ch.grid("hold", [
        {"title": "Plantings still alive after 300 iterations", "x": {"categories": cats,
                                                                      "label": "iteration the patch was planted"},
         "y": {"label": "of 6", "min": 0, "max": 6}, "series": _bars(B1_KINDS, alive)},
        {"title": "How long a planting lived", "x": {"categories": cats, "label": "iteration the patch was planted"},
         "y": {"label": "median iterations (300: to the end)", "min": 0, "max": T},
         "series": _bars(B1_KINDS, lived)}],
        title="Every strategy in B1's two long worlds", columns=2,
        caption=("Of the six plantings at each era (two worlds, three places): crews (red), breeding raiders "
                 "(orange), lone raiders (violet), breeding turtles (green) and the controls (grey). Left: how many "
                 "were still alive after 300 iterations; none of the strategies took hold, by its tokens or by "
                 "its agents. Right: the median iteration a planting's last agent died in."),
        recipe=recipe(B1_RUNS, "each trial's `result.json`",
                      STEPS + ["Count the plantings of each era whose strategy is alive after 300 iterations, "
                               "and take the median of the iteration each planting's last agent died in (300 if "
                               "it lived to the end)."]))
    panels = []
    for key, title in (("tokens", "Tokens of the patch"), ("mine", "Agents of the patch")):
        series = []
        for kind, label, colour in B1_KINDS:
            ts, ys = _median_course(found, places, kind, key, False)
            series.append({"label": label if key == "tokens" else None, "x": ts, "y": ys, "colour": colour,
                           "width": 1.6})
        panels.append({"title": title, "x": {"label": "iterations after planting", "min": 0, "max": T},
                       "y": {"label": "share of what the patch began with", "min": 0}, "series": series})
    ch.grid("patches", panels, title="What became of the patches in B1", columns=2,
            caption=("The median over the eighteen plantings of each kind, at each iteration after planting: the "
                     "strategy's tokens (left) and agents (right) as a share of what the patch began with, 0 once "
                     "it has died out."),
            recipe=recipe(B1_RUNS, "each trial's `result.json`",
                          STEPS + ["At each iteration after planting, take the median over the eighteen plantings "
                                   "of the strategy's tokens over the patch's tokens at planting, and of its agents "
                                   "over the patch planted."]))
    _numbers(ch, found, B1_KINDS)
    ch.number("able", _able(B1_WORLDS, 1))
    ch.number("worlds", {str(e): {w: {"agents": int(len(t)), "median": float(np.median(t)),
                                      "one_token": float(np.mean(t == 1))}
                                  for w in B1_WORLDS for t in [gol_store.kept_world(w, e)["tokens"]]}
                         for e in ERAS})
