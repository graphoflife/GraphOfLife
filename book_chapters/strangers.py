# -*- coding: utf-8 -*-
"""
Chapter 83 · Links to strangers — what batch K's worlds did with a search.

Batch K (2026-10-16): B2 at 10,000 tokens with each search the engine knows
(link_search = jump, walk or reach), each steering (random, looking) and each
consent (mutual, forced), seeds 1 and 2, 1,000 iterations, as pilot worlds; B2
itself with the same seeds is the control (P/B2-s1, s2, the lab's Experiment 16
worlds again). research/oee/checks/k_links.py played 50 more games from each
world's checkpoint at 1,000 and counted the proposals and the links they made
(GraphOfLifeRuns/.pilots/k_links.json); the lifetime contests (T = 100) are in
batch_K_T100.json and, for B2, long_steps.json.

    use        proposals and links made per hundred agents an iteration, every world
    network    links per agent, path length, agents and the poor, by search
    evolution  the brains of 1,000 against those of 500
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

import book_figures as F
from book_figures import chapter, recipe
from book_chapters.common import BLUE, GREEN, GREY, ORANGE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILOTS = os.path.join(ROOT, "GraphOfLifeRuns", ".pilots")
SEARCHES = [("B2", "no search (B2)", GREY), ("jump", "jump", BLUE), ("walk", "walk", GREEN),
            ("reach", "reach", ORANGE)]
RUNS = ("Batch K's 24 pilot worlds — B2 at 10,000 tokens with `link_search` = jump, walk or reach, "
        "`link_steering` = random or looking and `link_consent` = mutual or forced, seeds 1 and 2, 1,000 "
        "iterations (`GraphOfLifeRuns/.pilots/K/`, `research/oee/pilots/make_jobs_TU.py`) — and B2 itself with the "
        "same seeds (`P/B2-s1`, `-s2`).")


def _search(world: str) -> str:
    return "B2" if world.startswith("P/") else world.split("/")[1].split("-")[0]


def _contests() -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = defaultdict(list)
    path = os.path.join(PILOTS, "batch_K_T100.json")
    by: Dict[str, List[float]] = defaultdict(list)
    for r in json.load(open(path)) if os.path.exists(path) else []:
        if (r["a"], r["b"]) == (500, 1000) and r.get("share") is not None:
            by[r["world"].split(".pilots/")[-1]].append(r["share"])
    for world, shares in by.items():
        out[_search(world)].append(float(np.mean(shares)))
    # B2's own: Experiment 16's ten worlds, the same contests (long_steps.json).
    path = os.path.join(PILOTS, "long_steps.json")
    b2: Dict[str, List[float]] = defaultdict(list)
    for r in json.load(open(path)) if os.path.exists(path) else []:
        if (r["a"], r["b"]) == (500, 1000) and r.get("T", 100) == 100:
            b2[r["run"]].append(r["share"])
    out["B2"] = [float(np.mean(v)) for v in b2.values()]
    return out


@chapter
def strangers(ch: F.Chapter) -> None:
    path = os.path.join(PILOTS, "k_links.json")
    if not os.path.exists(path):
        return
    rows = json.load(open(path))
    by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[_search(r["world"])].append(r)
    names = [label for _, label, _ in SEARCHES]
    colours = [c for _, _, c in SEARCHES]

    # 1. How much they search: a panel for each search, its four steerings and consents side by side.
    kinds = [k for k, _, _ in SEARCHES[1:]]
    ways = [(steer, consent) for steer in ("random", "looking") for consent in ("mutual", "forced")]
    cats = [f"{steer} · {consent}" for steer, consent in ways]
    colour_of = {k: c for k, _, c in SEARCHES}
    charts = []
    for key, what in (("proposed_per_100", "proposed"), ("made_per_100", "made")):
        for k in kinds:
            # A world with none is drawn at the bottom of the axis.
            groups = [[max(r[key], 0.003) for r in rows if r["world"].startswith(f"K/{k}-{steer}-{consent}-")]
                      for steer, consent in ways]
            charts.append({"title": f"{k}: links {what}", "x": {"categories": cats},
                           "y": {"label": "per hundred agents an iteration", "log": True, "min": 0.003, "max": 50},
                           "series": F.dots(cats, groups, [colour_of[k]] * len(cats))})
    ch.grid("use", charts, columns=3, title="How much the agents search",
            caption=("Each dot is a world, read over 50 games played on from its checkpoint at iteration 1,000: "
                     "above, the links its agents proposed to strangers; below, those that were made; per hundred "
                     "agents an iteration, on a logarithmic axis (a world with none is drawn at the bottom). Jump: "
                     "a walk as many steps long as the tokens an agent spends on it; walk: a scout one step further "
                     "each iteration it is paid for; reach: as many free steps as the tokens held would pay. Each "
                     "column of a panel is a steering (random, or looking for the best liked) and a consent (both "
                     "ends agree, or only the one who proposes)."),
            recipe=recipe(RUNS, "`GraphOfLifeRuns/.pilots/k_links.json` (`research/oee/checks/k_links.py`)",
                          ["Play each world on for 50 games from its checkpoint at 1,000, recording them in the "
                           "lab's form.", "Count the proposals in each reproduction phase and those that made a "
                           "link, per agent that faced the phase."]))
    for k in kinds + ["B2"]:
        mine = by.get(k, [])
        ch.number(f"use:{k}", {key: [round(r[key], 3) for r in mine]
                               for key in ("proposed_per_100", "made_per_100")}
                  | {"distance_median": [r["distance_median"] for r in mine],
                     "made_share": [r["made_share"] for r in mine]})

    # 2. What it does to the world.
    charts = []
    for key, title, label in (("meanDegree", "Links per agent", "links per agent"),
                              ("meanPathLength", "Steps between two agents", "mean path length"),
                              ("nodes", "Agents", "agents"),
                              ("share_at_most_three", "The poor", "share with three tokens or fewer")):
        charts.append({"title": title, "x": {"categories": names},
                       "y": {"label": label, "min": 0} | ({"max": 1} if key == "share_at_most_three" else {}),
                       "series": F.dots(names, [[r[key] for r in by.get(k, []) if r.get(key) is not None]
                                                for k, _, _ in SEARCHES], colours)})
        for k, _, _ in SEARCHES:
            ch.number(f"{key}:{k}", [round(r[key], 3) for r in by.get(k, []) if r.get(key) is not None])
    ch.grid("network", charts, columns=2, title="What a search does to a world",
            caption=("Each dot is a world: grey, B2 itself; the others, the eight worlds of each search (two "
                     "steerings, two consents, two seeds). Links per agent and the mean path length are means of "
                     "the network measures every 50 iterations from 500 to 1,000; agents too; the poor are the "
                     "share of the agents holding three tokens or fewer at 1,000."),
            recipe=recipe(RUNS, "Each world's `series.jsonl` and its checkpoint at 1,000 "
                                "(`GraphOfLifeRuns/.pilots/k_links.json`)",
                          ["Average the network rows of iterations 500 to 1,000.",
                           "Count the agents with three tokens or fewer in the checkpoint at 1,000."]))

    # 3. Does evolution go on?
    contests = _contests()
    if sum(len(v) for k, v in contests.items() if k != "B2") >= 12:
        ch.figure("evolution", title="Do the brains of 1,000 beat those of 500?",
                  x={"categories": names}, y={"label": "the later brains' share of a contested world",
                                              "min": 0.2, "max": 1},
                  series=F.dots(names, [contests.get(k, []) for k, _, _ in SEARCHES], colours),
                  guides=[{"axis": "y", "at": 0.5, "label": "no better"}],
                  caption=("Each dot is a world: the share of a world its brains of iteration 1,000 hold after 100 "
                           "iterations against those of 500, the mean of two contests, one in each era's world. "
                           "B2: Experiment 16's ten worlds."),
                  recipe=recipe(RUNS + " B2: Experiment 16's ten runs.",
                                "`GraphOfLifeRuns/.pilots/batch_K_T100.json` and `long_steps.json`",
                                ["Seat the brains of the two eras on alternate agents of each era's world and play "
                                 "100 iterations (`research/oee/checks/rejudge.py`).",
                                 "Count the agents the later era's brains hold, as a share of all."]))
        for k, _, _ in SEARCHES:
            ch.number(f"evolution:{k}", [round(x, 3) for x in contests.get(k, [])])
