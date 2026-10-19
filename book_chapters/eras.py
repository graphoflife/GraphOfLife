"""
Chapter 95 · Every era against three (Experiment 28).

Before the contests: how much an agent could gain by its best reply to the rest, era by era, in every
long world (research/oee/checks/exploitability.py): whether evolution closes in on an equilibrium.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

import book_figures as F
from book_figures import chapter, recipe
from book_chapters.common import BLUE, GREEN, GREY, ORANGE, VIOLET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOM = os.path.join(ROOT, "GraphOfLifeRuns", ".pilots", "exploitability_eras.json")
#: Each family of long worlds: its label, how its runs are named, its colour.
FAMILIES = [("Experiment 18, seed 5 (mixed at birth, to 50,000)", lambda r: r == "B2-10000-4af4a3-s005", ORANGE),
            ("Experiment 18, seed 6", lambda r: r == "B2-10000-4af4a3-s006", VIOLET),
            ("Experiment 19 (parents as mates), ten worlds", lambda r: "bcc1bc" in r, BLUE),
            ("Experiment 17 (mixed at birth), eight worlds", lambda r: "4af4a3" in r, GREEN),
            ("B2, Experiment 16, ten worlds", lambda r: r.startswith("B2-10000-s"), GREY)]
RUNS = ("The lab's long worlds: Experiment 18's two worlds played to 50,000 iterations "
        "(`B2-10000-4af4a3-s005`, `-s006`, every 2,500), Experiment 19's ten to 20,000 "
        "(`B2-10000-bcc1bc-s001` … `-s010`), Experiment 17's to 10,000 (`B2-10000-4af4a3-s001` … `-s010` but "
        "the two above) and Experiment 16's B2 worlds to 2,000 (`B2-10000-s001` … `-s010`), at the iterations "
        "their archives keep.")


@chapter
def eras(ch: F.Chapter) -> None:
    if not os.path.exists(ROOM):
        return
    rows = json.load(open(ROOM))
    series: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}
    for label, member, colour in FAMILIES:
        mine = [r for r in rows if member(r["run"])]
        rows = [r for r in rows if not member(r["run"])]
        by_it: Dict[int, List[float]] = defaultdict(list)
        for r in mine:
            by_it[int(r["iteration"])].append(r["exploitability"])
        its = sorted(by_it)
        med = [float(np.median(by_it[i])) for i in its]
        lo = [float(min(by_it[i])) for i in its]
        hi = [float(max(by_it[i])) for i in its]
        if len({r["run"] for r in mine}) > 1:
            series.append({"label": None, "kind": "area", "x": its, "lo": lo, "hi": hi, "colour": colour,
                           "alpha": 0.18})
        series.append({"label": label, "x": its, "y": med, "colour": colour, "width": 1.6})
        summary[label] = {"iterations": its, "median": med, "lowest": lo, "highest": hi}
    ch.number("room", summary)
    ch.figure("room", title="What a best reply would gain, era by era",
              x={"label": "iteration"}, y={"label": "nodes a game, best reply over own play", "min": 0},
              series=series,
              caption=("Each line is a family of long worlds: for a sample of up to 300 agents of each world "
                       "holding three tokens or more, the nodes it would hold after one game by its best reply "
                       "to the others' play, less what its own play holds, averaged. A shaded band, where a "
                       "family has several worlds, runs from its lowest world to its highest; the line is the "
                       "median. A population at an equilibrium would score 0."),
              recipe=recipe(RUNS, "the worlds the archives keep",
                            ["Load the world at the iteration and play one game, so its messages are fresh.",
                             "Draw the random inputs 8 times and turn every brain's outputs into stakes as the "
                             "engine does: the others' play, as 8 draws.",
                             "For up to 300 agents with three tokens or more, score its own play and the best "
                             "of a set of replies (everything home; the cheapest way to keep its node and take "
                             "the cheapest neighbours; taking without defending) by the nodes it holds after the "
                             "game, averaged over the 8 draws (`python3 research/oee/checks/exploitability.py "
                             "RUN:ITERATION … --draws 8 --sample 300`).",
                             "The gain is the best reply's score less the own play's, averaged over the agents."]))
