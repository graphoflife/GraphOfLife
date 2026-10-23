# -*- coding: utf-8 -*-
"""
Who dies, and why: what Experiment 15's sixty worlds did.

The worlds through time (agents, births, the median agent's tokens: bands of the
ten worlds of each condition that lived, from their stats); the measure the
thesis named — the share of agents holding a single token, each world's median
over iterations 300 to 1,000, read from its game frames
(research/oee/checks/e15_thesis.py); and whether evolution goes on under each
rule: the brains of iteration 1,000 against those of 500 over an
agent's lifetime (T = 100; research/oee/checks/rejudge.py), B2's own from its
ten worlds (long_steps.json), the other rules from five worlds each
(lab_rejudge_T100.json).
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np

import book_figures as F
from book_figures import chapter, recipe
from book_chapters.common import BLUE, GREEN, GREY, RED, VIOLET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILOTS = os.path.join(ROOT, "GraphOfLifeRuns", ".pilots")
#: The conditions that lived, by name, with the run family each is and its colour.
CONDITIONS = [("B2", "B2-10000", BLUE), ("the last token kept", "B2-10000-054b50", RED),
              ("half kept", "B2-10000-df2615", VIOLET), ("children of any size", "B2-10000-185eb7", GREEN),
              ("no old age", "B2-10000-3ec24f", GREY)]
RUNS = ("The 60 runs of Experiment 15 — B2 at 10,000 tokens, seeds 1 to 10, 1,000 iterations — in six conditions: "
        "B2, and B2 with one rule of death changed. The ten worlds with nothing kept died out within 50 iterations "
        "and are not drawn. They are made with `python3 gol_lab.py run E15`.")


def _contests() -> Dict[str, List[float]]:
    """Each world's later-era share, 1,000 against 500 (T = 100), by condition: the mean of its contests."""
    by: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    families = {family: name for name, family, _ in CONDITIONS}
    for path, key in ((os.path.join(PILOTS, "lab_rejudge_T100.json"), "world"),
                      (os.path.join(PILOTS, "long_steps.json"), "run")):
        if not os.path.exists(path):
            continue
        for r in json.load(open(path)):
            name = families.get(r[key].rsplit("-s", 1)[0])
            if name and (r["a"], r["b"]) == (500, 1000) and r.get("T", 100) == 100:
                by[name][r[key]].append(r["share"])
    return {name: [float(np.mean(v)) for v in worlds.values()] for name, worlds in by.items()}


@chapter
def whodies(ch: F.Chapter) -> None:
    path = os.path.join(PILOTS, "e15_thesis.json")
    if not os.path.exists(path):
        return
    thesis = json.load(open(path))
    conditions = thesis["conditions"]
    names = [name for name, _, _ in CONDITIONS]
    colours = [c for _, _, c in CONDITIONS]
    shares = [[w["one_token"] for w in conditions[name]["worlds"].values() if w["one_token"] is not None]
              for name in names]
    ch.figure("one_token", title="Agents holding a single token",
              x={"categories": names}, y={"label": "share of the agents (median over iterations 300–1,000)",
                                          "min": 0, "max": 1},
              series=F.dots(names, shares, colours),
              guides=[{"axis": "y", "at": 0.5, "label": "B2's line: below a half"},
                      {"axis": "y", "at": 0.67, "label": "the last token's: above two thirds"}],
              caption=("Each dot is a world: the share of its agents holding exactly one token after each game, its "
                       "median over iterations 300 to 1,000; the bar is the median world. The thesis asked B2's "
                       "worlds to stay below a half and the worlds keeping only the last token to rise above two "
                       "thirds, seven times in ten each."),
              recipe=recipe(RUNS, "Each run's game frames, `GraphOfLifeRuns/<run>/frames/` "
                                  "(`research/oee/checks/e15_thesis.py`)",
                            ["For every iteration from 300 to 1,000, read the frame after the game (frame 2t + 1) "
                             "and count the agents holding exactly one token, as a share of all.",
                             "Take each world's median over those 701 games."]))
    # The worlds through time, each condition a band of its ten worlds.
    # The median agent's tokens from iteration 100, when the founders' first riches are spent.
    for name, stat, phase, title, y, rate, start in (
            ("agents", "nodes", 2, "Agents alive", "agents", False, 0),
            ("births", "births", 1, "Births per hundred agents in a reproduction phase", "births per 100 agents", True, 0),
            ("median", "medianTokens", 2, "The median agent's tokens", "tokens", False, 100)):
        series = []
        for label, family, colour in CONDITIONS:
            runs = []
            for seed in range(1, 11):
                its, values = (F.per_agent if rate else F.series)(f"{family}-s{seed:03d}", stat, phase)
                keep = (its >= start) & (its <= 1000)
                runs.append((its[keep], values[keep]))
            series.append(F.band_series(runs, label, colour))
        ch.figure(name, title=title, x={"label": "iteration"},
                  y={"label": y, "log": True} if rate else {"label": y, "min": 0}, series=series,
                  caption=("Each band is ten worlds: the line their median, the shade the middle half and nine in "
                           "ten. The worlds that kept nothing at home died out within 50 iterations and are not "
                           "drawn." + (" A logarithmic axis: births fall a hundredfold without old age." if rate
                                       else " From iteration 100, once the founders' first riches are spent."
                                       if start else "")),
                  recipe=recipe(RUNS, "From each run's `GraphOfLifeRuns/<run>/stats.jsonl`, one row per recorded "
                                      "phase (see [What a run records](frames-and-stats.md))",
                                [f"Read `{stat}` from the rows of phase {phase}"
                                 + (", divided by the agents that faced the phase (`nodes_before`), times 100"
                                    if rate else "") + f", for iterations {start:,} to 1,000.",
                                 "At each iteration, the median of the ten worlds, the middle half and nine in ten."]))
    contests = _contests()
    ch.figure("evolution", title="Do the brains of 1,000 beat those of 500?",
              x={"categories": names}, y={"label": "the later brains' share of a contested world", "min": 0.3, "max": 1},
              series=F.dots(names, [contests.get(name, []) for name in names], colours),
              guides=[{"axis": "y", "at": 0.5, "label": "no better"}],
              caption=("Each dot is a world: the share of a world its brains of iteration 1,000 hold after 100 "
                       "iterations against those of iteration 500, the mean of two contests, one in each era's world; "
                       "the bar is the median world. B2 from its ten worlds, each other rule from five."),
              recipe=recipe(RUNS, "Each run's archive, `GraphOfLifeRuns/<run>/archive/` "
                                  "(`GraphOfLifeRuns/.pilots/lab_rejudge_T100.json`, `long_steps.json`)",
                            ["Seat the brains of iteration 500 and of iteration 1,000 on alternate agents of each "
                             "era's world and play it on for 100 iterations, the length of an agent's life "
                             "(`research/oee/checks/rejudge.py`).",
                             "Count the agents the later era's brains hold at the end, as a share of all."]))
    for name in names:
        c = conditions[name]
        ch.number(f"one_token:{name}", {"median": c["one_token_median"], "below_half": c["below_half"],
                                        "above_two_thirds": c["above_two_thirds"]})
        if contests.get(name):
            ch.number(f"evolution:{name}", {"median": float(np.median(contests[name])),
                                            "low": min(contests[name]), "high": max(contests[name]),
                                            "worlds": len(contests[name])})
    ch.number("verdict", {**thesis["verdict"], "holds": thesis["holds"]})
