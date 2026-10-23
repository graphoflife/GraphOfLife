"""
The search's best world, in new seeds (Experiment 24).

The settings of the random search's best world, Z/r2-25, played by the lab with the prototype from five
new seeds and from its own: each world read at iterations 1,000 and 2,000 by the four measures of the
author's picture, and the world of seed 2025 compared with the pilot's, game for game.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import numpy as np

import book_figures as F
import gol_store
from book_figures import chapter, recipe
from book_chapters.common import GREY, ORANGE, VIOLET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
READINGS = os.path.join(ROOT, "GraphOfLifeRuns", ".pilots", "atlas", "e24_readings.json")
PILOT = os.path.join(ROOT, "GraphOfLifeRuns", ".pilots", "Z", "r2-25-s2025", "series.jsonl")
OWN = "B2-10000-d2b0d2-s2025"
#: Healthy and rich's lines: every mark, diversity above B2's median world, a space, no more even than B2.
LINES = {"marks": 7, "diversity": 0.49, "wide": 1.5, "gini_tokens": 0.47}
RUNS = ("Experiment 24's six runs (`python3 gol_lab.py run E24`): the settings of `Z/r2-25` played by the "
        "research prototype in the lab (`gol_pilot.py`), seeds 1 to 5 (`B2-10000-d2b0d2-s001` … `-s005`) and "
        "the search's own seed (`B2-10000-d2b0d2-s2025`), each to 2,000 iterations, with a whole world kept "
        "every 100.")


def _twin() -> Dict[str, Any]:
    """How far the lab's world of seed 2025 matches the pilot's agents and links, iteration by iteration."""
    path = os.path.join(gol_store.run_dir(OWN), "series.jsonl")
    if not (os.path.exists(path) and os.path.exists(PILOT)):
        return {}
    lab = [json.loads(line) for line in open(path)]
    pilot = [json.loads(line) for line in open(PILOT)]
    same = [(a["it"], a["nodes"], a["edges"]) == (b["it"], b["nodes"], b["edges"]) for a, b in zip(lab, pilot)]
    first = next((i for i, ok in enumerate(same) if not ok), None)
    return {"compared": len(same), "first_difference": first}


@chapter
def searchbest(ch: F.Chapter) -> None:
    if not os.path.exists(READINGS):
        return
    rows = json.load(open(READINGS))
    twin = _twin()
    if twin:
        ch.number("twin", twin)
    panels = []
    for key, title, ylabel in (("marks", "Marks of the health bar", "of 7"),
                               ("diversity", "Diversity score", "mean percentile"),
                               ("wide", "Width over the random twin's", "typical distance ratio"),
                               ("gini_tokens", "Gini of the agents' tokens", "Gini")):
        series = []
        for at, colour in ((1000, ORANGE), (2000, VIOLET)):
            new = [r for r in rows if r["at"] == at and not r["id"].endswith("s2025")]
            own = [r for r in rows if r["at"] == at and r["id"].endswith("s2025")]
            x0 = 0 if at == 1000 else 1
            series.append({"label": f"new seeds, iteration {at:,}" if key == "marks" else None,
                           "x": list(F.jitter(x0, len(new), 11 + at)), "y": [r[key] for r in new], "points": True,
                           "size": 6, "colour": colour, "alpha": 0.85})
            series.append({"label": f"the search's own seed, iteration {at:,}" if key == "marks" else None,
                           "x": [x0 + 0.3], "y": [r[key] for r in own], "points": True, "size": 9, "colour": colour,
                           "alpha": 0.5})
        series.append({"label": "the line the author's picture asks for" if key == "marks" else None,
                       "x": [-0.5, 1.5], "y": [LINES[key]] * 2, "colour": GREY, "dash": [5, 4], "width": 1.2})
        panels.append({"title": title, "x": {"categories": ["1,000", "2,000"], "label": "iteration"},
                       "y": {"label": ylabel, "min": 0}, "series": series})
    ch.grid("measures", panels, title="The search's best settings, in six worlds", columns=2,
            caption=("Each dot is one world at iteration 1,000 (orange) or 2,000 (violet): the five new seeds small, "
                     "the search's own seed large. The dashed line is what the author's picture asks: every mark of "
                     "the health bar, a diversity score above B2's median world (0.49), a typical distance at least "
                     "1.5 times the random twin's, and a Gini no lower than B2's least even world (0.47)."),
            recipe=recipe(RUNS, "each run's checkpoints at 1,000 and 2,000 (`ckpt_001000.npz`, `ckpt_002000.npz`)",
                          ["Read each world as `research/atlas/profile.py` reads every world: the seven marks, the "
                           "four spreads ranked among the frozen reference worlds, the Gini of the tokens.",
                           "Its typical distance over its random twin's as `research/atlas/space.py` measures it."]))
    ch.number("readings", [{k: r.get(k) for k in ("id", "at", "agents", "marks", "diversity", "wide", "gini_tokens",
                                                  "tokens_per_agent", "dimension", "agent_cut", "link_cut")}
                           for r in rows])
