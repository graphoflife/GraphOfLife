"""
Chapter 94 · What a brain can compute.

Chapter 51's raider, set by hand into the weights of a B2 brain (research/oee/strategies/handset.py),
checked against the raider's rule on every situation a planted raider met, and planted in its place.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import book_figures as F
from book_figures import chapter, recipe
from book_chapters.common import BLUE, ORANGE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRATEGIES = os.path.join(ROOT, "GraphOfLifeRuns", ".pilots", "strategies")
WORLDS = ["B2-10000-s001", "B2-10000-s002", "B2-10000-s003"]
REPLAYS = ["replay_s001.json", "replay_s002_s003_seed1.json", "replay_s002_s003_seed2.json"]
RUNS = ("Three B2 worlds of Experiment 16, `B2-10000-s001` … `-s003`, as their archives keep them at "
        "iteration 2,000 (`GraphOfLifeRuns/<run>/archive/`), each with a patch of raiders planted as in "
        "[Chapter 51](strategies-written-by-hand.md) (`research/oee/pilots/manual.py`: the agent drawn on "
        "seed *s* and its nearest neighbours, step by step, until the patch holds 5% of the world).")


def _load(name: str) -> Any:
    path = os.path.join(STRATEGIES, name)
    return json.load(open(path)) if os.path.exists(path) else None


@chapter
def computable(ch: F.Chapter) -> None:
    agreement = _load("agreement.json")
    replays = [r for name in REPLAYS for r in (_load(name) or [])]
    if not agreement or len(replays) < 12:
        return
    by_world = {}
    for name, got in agreement["files"].items():
        world = next(w for w in WORLDS if w in name)
        by_world[world] = got
    ch.number("margin", agreement["margin"])
    ch.number("agreement", by_world)
    ch.number("situations", sum(g["situations"] for g in by_world.values()))

    def share(r: Dict[str, Any]) -> float:
        last = r["series"][-1]
        return last["mine"] / max(1, last["agents"]) if last["t"] == 300 else 0.0

    ch.number("replays", [{"run": r["run"], "seed": r["seed"], "rule": r["rule"], "share_at_300": share(r),
                           "last": r["series"][-1]["t"]} for r in replays])
    series: List[Dict[str, Any]] = []
    for rule, colour, label in ((True, ORANGE, "the raider's rule (manual.py)"),
                                (False, BLUE, "the raider's weights in a B2 brain")):
        mine = sorted((r for r in replays if r["rule"] == rule), key=lambda r: (r["run"], r["seed"]))
        for k, r in enumerate(mine):
            s = r["series"]
            series.append({"label": label if k == 0 else None, "x": [p["t"] for p in s],
                           "y": [p["mine"] / max(1, p["agents"]) for p in s], "colour": colour, "width": 1.5,
                           **({"dash": [6, 3]} if r["seed"] == 2 else {})})
    ch.figure("replays", title="The rule and the brain, planted in the same places",
              x={"label": "iterations after planting"}, y={"label": "share of the agents", "min": 0},
              series=series,
              caption=("One line per planting: three evolved worlds, two seeds each (seed 1 solid, seed 2 "
                       "dashed). Orange: a patch of raiders that follow the rule of Chapter 51; blue: the same "
                       "patch given a B2 brain whose weights were set by hand to play the same rule. Neither "
                       "ever mutates. A line that stops early is a patch that died out."),
              recipe=recipe(RUNS, "the worlds the archives keep",
                            ["Load the world of iteration 2,000 and plant the patch on seed 1 or 2.",
                             "Give every agent of the patch the raider's rule (`manual.py`), or a frozen copy "
                             "of the hand-set brain (`research/oee/strategies/raider_B2.npz`, made by "
                             "`python3 research/oee/strategies/handset.py save`).",
                             "Play 300 iterations under the world's own rules (`handset.py invade "
                             "B2-10000-s001 B2-10000-s002 B2-10000-s003 --seeds 1,2 --T 300`) and count, "
                             "after each, the agents carrying the patch's brain, as a share of all agents."]))
