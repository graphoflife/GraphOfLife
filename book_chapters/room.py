"""
Room for the steadiest world (Experiment 31).

Experiment 19's world (brains mixed at birth, parents as mates) given each rule that keeps a world's room —
the estate to the newborns, to the parents, no old age, no mending — two seeds each, read at iteration 3,000
by the four measures of the author's picture and by the four parts of the diversity score, beside
Experiment 19's own ten worlds at the same iteration.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import numpy as np

import book_figures as F
import gol_plan
from book_figures import chapter, recipe
from book_chapters.common import BLUE, GREEN, GREY, ORANGE, VIOLET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
READINGS = os.path.join(ROOT, "GraphOfLifeRuns", ".pilots", "atlas", "e31_readings.json")
#: B2's ten worlds played on, read at 3,000: the lines of the author's picture there.
LINES = {"marks": 7, "diversity": 0.543, "wide": 1.5, "gini_tokens": 0.466}
GROUPS = [("parents as mates (Experiment 19)", GREY), ("estate to the newborns", ORANGE),
          ("estate to the parents", VIOLET), ("no old age", BLUE), ("deaths not mended", GREEN)]
RUNS = ("Experiment 31's eight runs (`python3 gol_lab.py run E31`): B2 at 10,000 tokens with brains mixed at birth "
        "and parents as mates, and the estate of the dead paid to the newborns (`B2-10000-06b876-…`) or to the "
        "parents (`-b8e9fc-…`), no old age (`-89f5b1-…`) or deaths not mended (`-c58a16-…`), seeds 1 and 2, to "
        "3,000 iterations; and Experiment 19's ten worlds (`B2-10000-bcc1bc-s001` … `-s010`), all read at "
        "iteration 3,000 from their archives.")
STEPS = ["Read each world as `research/atlas/profile.py` reads every world: the seven marks, the four spreads "
         "ranked among the frozen reference worlds (their mean is the diversity score), the Gini of the tokens.",
         "Its typical distance over its random twin's, as `research/atlas/space.py` measures it.",
         "The lines: every mark; the median diversity, and the least Gini, of B2's ten worlds played on and read "
         "at 3,000; a typical distance of 1.5 times the twin's."]
PARTS = [("spread_tokens", "Tokens"), ("spread_links", "Links"), ("spread_ages", "Ages"),
         ("effective_genotypes_per_agent", "Genotypes")]


def _group(run_id: str, conditions: Dict[str, str]) -> str:
    return conditions.get(run_id, "parents as mates (Experiment 19)")


@chapter
def room(ch: F.Chapter) -> None:
    if not os.path.exists(READINGS):
        return
    sys.path.insert(0, os.path.join(ROOT, "research", "atlas"))
    import profile as P
    ref = P.reference()
    try:
        conditions = {s.run_id: s.condition for s in gol_plan.experiment_runs("E31")}
    except Exception:
        return
    rows = [r for r in json.load(open(READINGS)) if r["at"] == 3000]
    cats = [g for g, _ in GROUPS]
    labels = ["parents as mates", "estate, newborns", "estate, parents", "no old age", "not mended"]

    def panel(title: str, ylabel: str, value, line=None) -> Dict[str, Any]:
        series = []
        for k, (group, colour) in enumerate(GROUPS):
            mine = [value(r) for r in rows if _group(r["id"], conditions) == group]
            series.append({"label": None, "x": list(F.jitter(k, len(mine), 5 + k)), "y": mine, "points": True,
                           "size": 6, "colour": colour, "alpha": 0.85})
        if line is not None:
            series.append({"label": None, "x": [-0.5, len(cats) - 0.5], "y": [line, line], "colour": GREY,
                           "dash": [5, 4], "width": 1.2})
        return {"title": title, "x": {"categories": labels}, "y": {"label": ylabel, "min": 0}, "series": series}

    ch.grid("measures", [panel("Marks of the health bar", "of 7", lambda r: r["marks"], LINES["marks"]),
                         panel("Diversity score", "mean percentile", lambda r: r["diversity"], LINES["diversity"]),
                         panel("Width over the random twin's", "typical distance ratio", lambda r: r["wide"],
                               LINES["wide"]),
                         panel("Gini of the agents' tokens", "Gini", lambda r: r["gini_tokens"],
                               LINES["gini_tokens"])],
            title="Experiment 19's world, given room", columns=2,
            caption=("Each dot is one world at iteration 3,000: Experiment 19's ten (grey) and Experiment 31's two "
                     "for each rule. The dashed line is what the author's picture asks: every mark, a diversity "
                     "score above B2's median world at 3,000 (0.54), a typical distance at least 1.5 times the "
                     "random twin's, a Gini no lower than B2's least even world at 3,000 (0.47)."),
            recipe=recipe(RUNS, "each world's archive at iteration 3,000", STEPS))
    parts = []
    for key, title in PARTS:
        parts.append(panel(title, "percentile among all worlds", lambda r, key=key: P.diversity(r, ref)[0][key]))
    ch.grid("parts", parts, title="What each rule costs of the diversity", columns=2,
            caption=("The four parts of the diversity score, each world's percentile among the frozen reference "
                     "worlds: how many doubling classes its agents' tokens, links and ages fill, and its effective "
                     "number of genotypes per agent. Each dot is one world at iteration 3,000."),
            recipe=recipe(RUNS, "each world's archive at iteration 3,000", STEPS[:1]))
    ch.number("worlds", [{**{k: r.get(k) for k in ("id", "agents", "marks", "diversity", "wide", "gini_tokens",
                                                    "tokens_per_agent", "dimension")},
                          "group": _group(r["id"], conditions),
                          "parts": P.diversity(r, ref)[0]} for r in rows])
