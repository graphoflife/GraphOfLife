"""
How large a crowd a raider needs (Experiment 29).

Raider patches of 1 to 100 agents and of 5% of the world, ten places each in worlds 2 and 3 of
Experiment 16 at iteration 2,000, and 5% in five places in the other eight, each with a control:
the same patch given frozen copies of one resident's brain.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np

import book_figures as F
import gol_plan
import gol_store
from book_figures import chapter, recipe
from book_chapters.common import GREY, ORANGE

SIZES = [1, 3, 10, 30, 100, 0.05]
TWO = ["B2-10000-s002", "B2-10000-s003"]
RUNS = ("Experiment 29's 320 invasions (`python3 gol_lab.py run E29`): the ten B2 worlds of Experiment 16, "
        "`B2-10000-s001` … `-s010`, as their archives keep them at iteration 2,000. In worlds 2 and 3, patches "
        "of 1, 3, 10, 30 and 100 agents and of 5% of the world on places 1 to 10; in the other eight, 5% on "
        "places 1 to 5. Every patch once with the raider's rule and once as a control, frozen copies of one "
        "patch member's brain (`gol_strategies.plant`).")
STEPS = ["Load the world of iteration 2,000; on place p, draw an agent at random and take its nearest neighbours, "
         "step by step, until the patch has its size.",
         "Give the patch the raider ([Strategies written by hand](strategies-written-by-hand.md)'s rule), or the control; neither ever mutates.",
         "Play 300 iterations under the world's own rules, counting the patch's agents after every iteration.",
         "A planting takes hold when, after 300 iterations, its raiders are alive, more than the patch began "
         "with, and more than the control holds there."]


def _results() -> Dict[Tuple[str, str, float, int], Dict[str, Any]]:
    out = {}
    for found in gol_plan.trials_of("E29"):
        t = found["trial"]
        path = os.path.join(gol_store.run_dir(gol_plan.trial_id_for(t)), "result.json")
        if os.path.exists(path):
            with open(path) as f:
                out[(t["run"], t["strategy"], t["size"], t["place"])] = json.load(f)["result"]
    return out


def _end(r: Dict[str, Any]) -> Tuple[int, int, int]:
    """(the patch's agents at 300, or 0 if it died; the agents it began with; the world's agents at the end)."""
    s = r["series"]
    return (s[-1]["mine"] if s[-1]["t"] == 300 else 0), s[0]["mine"], s[-1]["agents"]


def held(raider: Dict[str, Any], control: Dict[str, Any]) -> bool:
    mine, start, _ = _end(raider)
    return mine > start and mine > _end(control)[0]


def label(size: float) -> str:
    return f"{size:.0%}" if size < 1 else f"{int(size)}"


@chapter
def crowd(ch: F.Chapter) -> None:
    res = _results()
    if len(res) < 320:
        return
    took, alive, grew = [], [], []
    for size in SIZES:
        pairs = [(res[(w, "raider", size, p)], res[(w, "resident", size, p)]) for w in TWO for p in range(1, 11)]
        took.append(sum(held(r, c) for r, c in pairs))
        alive.append(sum(_end(c)[0] > 0 for _, c in pairs))
        grew.append(sum(_end(c)[0] > _end(c)[1] for _, c in pairs))
    cats = [label(s) for s in SIZES]
    ch.figure("crowd", title="How many plantings took hold, by the size of the patch",
              x={"categories": cats, "label": "agents in the patch (or share of the world)"},
              y={"label": "plantings of 20", "min": 0, "max": 20},
              series=[{"label": "raiders that took hold", "kind": "bars",
                       "x0": [k - 0.32 for k in range(len(cats))], "x1": [k + 0.32 for k in range(len(cats))],
                       "y": took, "colour": ORANGE},
                      {"label": "controls larger after 300 iterations than they began", "x": list(range(len(cats))),
                       "y": grew, "colour": GREY, "points": True, "size": 7}],
              caption=("Bars: of the 20 plantings of each size (ten places in each of worlds 2 and 3), how many "
                       "raider patches took hold: alive after 300 iterations, more than they began with, and more "
                       "than the control planted in the same place. Grey dots: how many of the controls, frozen "
                       "copies of one resident's brain, were larger after 300 iterations than they began."),
              recipe=recipe(RUNS, "each trial's `result.json`", STEPS))
    worlds = sorted({k[0] for k in res})
    rows = []
    for w in worlds:
        places = sorted(k[3] for k in res if k[0] == w and k[1] == "raider" and k[2] == 0.05)
        for p in places:
            r, c = res[(w, "raider", 0.05, p)], res[(w, "resident", 0.05, p)]
            n = max(1, _end(r)[2])
            rows.append({"world": w, "place": p, "held": held(r, c), "raiders": _end(r)[0] / n,
                         "control": _end(c)[0] / max(1, _end(c)[2])})
    wcats = [w[-2:].lstrip("0") for w in worlds]
    series = []
    for kind, colour, key, text in ((True, ORANGE, "raiders", "raiders that took hold"),
                                   (False, ORANGE, "raiders", "raiders that did not"),
                                   (None, GREY, "control", "controls")):
        xs, ys = [], []
        for row in rows:
            if kind is None or row["held"] == kind:
                k = worlds.index(row["world"])
                xs.append(k + (0.18 if kind is None else -0.18) + 0.03 * (row["place"] - 5))
                ys.append(row[key])
        series.append({"label": text, "x": xs, "y": ys, "points": True, "size": 6 if kind else 4,
                       "colour": colour, "alpha": 0.9 if kind else 0.5})
    ch.figure("worlds", title="A patch of 5% in every world",
              x={"categories": wcats, "label": "world (the seed of Experiment 16)"},
              y={"label": "share of the world after 300 iterations", "min": 0},
              series=series,
              caption=("Each dot is one planting of a patch of 5% of the world at iteration 2,000: left of each "
                       "world, the raiders (filled where they took hold), right of it, the control planted in the "
                       "same place. Worlds 2 and 3 have ten places, the others five."),
              recipe=recipe(RUNS, "each trial's `result.json`", STEPS))
    per_world = {w: sum(r["held"] for r in rows if r["world"] == w and r["place"] <= 5) for w in worlds}
    ch.number("crowd", {"sizes": [label(s) for s in SIZES], "took_hold": took, "controls_alive": alive,
                        "controls_grew": grew})
    ch.number("worlds", {"held_in_places_1_to_5": per_world,
                         "held_world_2": sum(r["held"] for r in rows if r["world"] == "B2-10000-s002"),
                         "held_world_3": sum(r["held"] for r in rows if r["world"] == "B2-10000-s003")})
