"""
A mate that pays its share (Experiment 22).

B2 with brains mixed at birth and a mate that pays the child as much as the parent, in four worlds to 5,000
iterations, played by the prototype in the lab; against Experiment 17's worlds of the same seeds, whose mates
pay nothing.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np

import book_figures as F
import gol_store
from book_figures import chapter, recipe
from book_chapters.common import GREY, ORANGE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILOTS = os.path.join(ROOT, "GraphOfLifeRuns", ".pilots", "M")
READINGS = os.path.join(ROOT, "GraphOfLifeRuns", ".pilots", "atlas", "b2long_e22_readings.json")
SEEDS = (1, 2, 3, 4)
RUNS = ("Experiment 22's four runs (`python3 gol_lab.py run E22`): B2 at 10,000 tokens with brains mixed at birth "
        "and the pilots' rule `mate_pays`, seeds 1 to 4, to 5,000 iterations, played by the research prototype in "
        "the lab (`B2-10000-8ff8c5-s001` … `-s004`); against Experiment 17's runs of the same seeds "
        "(`B2-10000-4af4a3-s001` … `-s004`), mixing with a mate that pays nothing.")


def paying(seed: int) -> Tuple[np.ndarray, np.ndarray]:
    path = os.path.join(gol_store.run_dir(f"B2-10000-8ff8c5-s{seed:03d}"), "series.jsonl")
    rows = [json.loads(line) for line in open(path)]
    return np.array([r["it"] for r in rows], float), np.array([r["nodes"] for r in rows], float)


def free(seed: int) -> Tuple[np.ndarray, np.ndarray]:
    its, nodes = [], []
    for line in open(os.path.join(gol_store.run_dir(f"B2-10000-4af4a3-s{seed:03d}"), "stats.jsonl")):
        r = json.loads(line)
        if r.get("phase") == 2:
            its.append(r["iteration"] + 1)
            nodes.append(r["nodes"])
    return np.array(its, float), np.array(nodes, float)


def twin(seed: int) -> Dict[str, Any]:
    lab = [json.loads(line) for line in open(os.path.join(gol_store.run_dir(f"B2-10000-8ff8c5-s{seed:03d}"),
                                                          "series.jsonl"))]
    pilot = [json.loads(line) for line in open(os.path.join(PILOTS, f"mix-matepays-s{seed}", "series.jsonl"))]
    same = [(a["it"], a["nodes"], a["edges"]) == (b["it"], b["nodes"], b["edges"]) for a, b in zip(lab, pilot)]
    return {"compared": len(same), "first_difference": next((i for i, ok in enumerate(same) if not ok), None)}


@chapter
def matepays(ch: F.Chapter) -> None:
    try:
        worlds = {s: (paying(s), free(s)) for s in SEEDS}
    except FileNotFoundError:
        return
    series: List[Dict[str, Any]] = []
    steadiness = {}
    for k, s in enumerate(SEEDS):
        (pi, pn), (fi, fn) = worlds[s]
        for (its, nodes), colour, text in (((pi, pn), ORANGE, "a mate that pays"), ((fi, fn), GREY, "a free mate")):
            # The same stretch for both: some of Experiment 17's worlds were played on (Experiment 18).
            keep = its <= 5000
            x, y = F.thin(its[keep], nodes[keep], 50)
            series.append({"label": text if k == 0 else None, "x": list(x), "y": list(y), "colour": colour,
                           "width": 1.4, "alpha": 0.9})
        a = pn[(pi >= 500) & (pi <= 5000)]
        b = fn[(fi >= 500) & (fi <= 5000)]
        steadiness[s] = {"paying_sd_log": float(np.std(np.log(a))), "free_sd_log": float(np.std(np.log(b))),
                         "paying_min": float(a.min()), "free_min": float(b.min()),
                         "paying_median": float(np.median(a)), "free_median": float(np.median(b))}
    ch.figure("sizes", title="Agents alive, with a mate that pays and with a free mate",
              x={"label": "iteration"}, y={"label": "agents after each game", "log": True},
              series=series,
              caption=("Each line is one world, its agents after each game in means over 50 iterations, on a "
                       "logarithmic axis. Orange: brains mixed at birth with a mate that pays the child as much as "
                       "the parent (seeds 1 to 4); grey: Experiment 17's worlds of the same seeds, whose mates pay "
                       "nothing."),
              recipe=recipe(RUNS, "the pilots' `series.jsonl` and the lab runs' `stats.jsonl`",
                            ["Read the agents after each game: a pilot's row of each iteration, a lab run's row of "
                             "phase 2.",
                             "Average over stretches of 50 iterations."]))
    ch.number("steadiness", steadiness)
    ch.number("twins", {s: twin(s) for s in (1, 2)})
    if os.path.exists(READINGS):
        rows = json.load(open(READINGS))
        ch.number("health", [{k: r.get(k) for k in ("id", "at", "agents", "marks", "diversity", "wide",
                                                    "gini_tokens")} for r in rows])
