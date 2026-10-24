"""
Does evolution close the door? (Experiment 30)

Raiders planted, 5% of the world in three places, in B2's five worlds played on to 10,000 iterations, at
500, 2,000, 5,000 and 10,000, each with a control; and what a planting's starting wealth says about
whether it takes hold, over every 5% planting of Experiments 29 and 30.
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
from book_chapters.common import BLUE, GREEN, GREY, ORANGE, VIOLET

ERAS = [(500, GREEN), (2000, BLUE), (5000, ORANGE), (10000, VIOLET)]
RUNS = ("Experiment 30's 120 invasions (`python3 gol_lab.py run E30`): B2's worlds of Experiment 16 played on "
        "to 10,000 iterations as pilots of the engine's own rules (`GraphOfLifeRuns/.pilots/L/B2-long-s001` … "
        "`s005`), at iterations 500, 2,000, 5,000 and 10,000, a patch of 5% on places 1 to 3, once with the "
        "raider's rule and once as a control; and, in the second chart, every patch of 5% of Experiment 29 too "
        "(Experiment 16's ten worlds at 2,000), of which Experiment 30's plantings at 2,000 replay fifteen, counted "
        "once.")
STEPS = ["Load the world of the era; on place p, draw an agent at random and take its nearest neighbours, step by "
         "step, until the patch holds 5% of the world.",
         "Give the patch the raider ([Strategies written by hand](strategies-written-by-hand.md)'s rule), or the control, frozen copies of one patch member's brain.",
         "Play 300 iterations under the world's own rules; a planting takes hold when its raiders are then alive, "
         "more than the patch began with, and more than the control holds.",
         "A patch's wealth is its tokens over its agents when it is planted."]


def _plantings(experiment: str) -> List[Dict[str, Any]]:
    """Each 5% raider planting with its control: its era, starting wealth, whether it took hold."""
    found = {}
    for f in gol_plan.trials_of(experiment):
        t = f["trial"]
        path = os.path.join(gol_store.run_dir(gol_plan.trial_id_for(t)), "result.json")
        if not os.path.exists(path):
            return []
        with open(path) as fh:
            found[(t["run"], t["era"], t["strategy"], t["size"], t["place"])] = json.load(fh)["result"]
    out = []
    for (run, era, strategy, size, place), r in found.items():
        if strategy != "raider" or size != 0.05:
            continue
        c = found[(run, era, "resident", size, place)]
        s, cs = r["series"], c["series"]
        mine = s[-1]["mine"] if s[-1]["t"] == 300 else 0
        ctrl = cs[-1]["mine"] if cs[-1]["t"] == 300 else 0
        out.append({"run": run, "era": era, "place": place, "wealth": s[0]["tokens"] / s[0]["mine"],
                    "held": mine > s[0]["mine"] and mine > ctrl, "share": mine / max(1, s[-1]["agents"])})
    return out


@chapter
def door(ch: F.Chapter) -> None:
    e30, e29 = _plantings("E30"), _plantings("E29")
    if len(e30) < 60 or len(e29) < 60:
        return
    eras = [e for e, _ in ERAS]
    held = [sum(p["held"] for p in e30 if p["era"] == e) for e in eras]
    shares = []
    for k, (era, colour) in enumerate(ERAS):
        mine = [p["share"] for p in e30 if p["era"] == era]
        shares.append({"label": None, "x": list(F.jitter(k, len(mine), 7 + k)), "y": mine, "points": True,
                       "size": 6, "colour": colour, "alpha": 0.85})
    cats = [f"{e:,}" for e in eras]
    ch.grid("door", [
        {"title": "Plantings that took hold", "x": {"categories": cats, "label": "iteration the raiders were planted"},
         "y": {"label": "of 15", "min": 0, "max": 15},
         "series": [{"label": None, "kind": "bars", "x0": [k - 0.32 for k in range(4)],
                     "x1": [k + 0.32 for k in range(4)], "y": held, "colour": ORANGE}]},
        {"title": "Raiders' share after 300 iterations", "x": {"categories": cats,
                                                                           "label": "iteration the raiders were planted"},
         "y": {"label": "share of the agents", "min": 0}, "series": shares}],
        title="The same raid, planted in older worlds", columns=2,
        caption=("Left: of the fifteen plantings at each era (five worlds, three places), how many took hold. "
                 "Right: each planting's raiders as a share of the world after 300 iterations (0 where they died "
                 "out)."),
        recipe=recipe(RUNS, "each trial's `result.json`", STEPS))
    # Experiment 30's plantings at 2,000 replay fifteen of Experiment 29's exactly: the long runs begin from
    # Experiment 16's worlds at 2,000, and the places are the same. Each planting is counted once.
    seen = {(p["wealth"], p["share"]) for p in e29}
    replays = [p for p in e30 if p["era"] == 2000]
    assert all((p["wealth"], p["share"]) in seen for p in replays), "E30's plantings at 2,000 no longer replay E29's"
    both = e29 + [p for p in e30 if p["era"] != 2000]
    series = []
    for era, colour in ERAS:
        mine = [p for p in both if p["era"] == era]
        series.append({"label": f"planted at {era:,}", "x": [p["wealth"] for p in mine], "y": [p["share"] for p in mine],
                       "points": True, "size": 6, "colour": colour, "alpha": 0.8})
    ch.figure("wealth", title="What a raid starts from", x={"label": "the patch's tokens per agent when planted"},
              y={"label": "raiders' share of the world after 300 iterations", "min": 0}, series=series,
              caption=("Each dot is one planting of 5% of a world, from Experiments 29 and 30, each counted once, by the "
                       "patch's tokens per agent when it was planted and the raiders' share of the world 300 iterations "
                       "later."),
              recipe=recipe(RUNS, "each trial's `result.json`", STEPS))
    bins = [(0, 2), (2, 2.5), (2.5, 3), (3, 4), (4, 99)]
    ch.number("by_era", {f"{e}": {"held": h, "wealth_median": float(np.median([p["wealth"] for p in e30 if p["era"] == e]))}
                         for e, h in zip(eras, held)})
    ch.number("by_wealth", [{"from": lo, "to": hi, "held": sum(p["held"] for p in both if lo <= p["wealth"] < hi),
                             "plantings": sum(1 for p in both if lo <= p["wealth"] < hi)} for lo, hi in bins])
