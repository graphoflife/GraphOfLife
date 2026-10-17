# -*- coding: utf-8 -*-
"""
Chapter 73 · Who inherits? The estate of the dead shared out three new ways —
to the poorest, to the newborns, to their parents — in B1 and B2 worlds of
10,000 tokens (pilot batch R), beside controls of the same settings and seeds,
and each world among every world the book has made. A fourth way, added on
2026-10-17: each estate to the dead agent's own living children (batch H, the
same worlds' form, in GraphOfLifeRuns/.pilots/R/ as well).

Pilot worlds keep a row of statistics every iteration (series.jsonl); the
lifetime contests between their eras are in GraphOfLifeRuns/.pilots/
batch_R_T100.json (and batch_P_T100.json for the B2 controls); their
statistics beside every other world are in book/results/worlds.json.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import book_figures as F
from book_figures import chapter, recipe
from book_chapters.common import BLUE, GREEN, GREY, ORANGE, RED, VIOLET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILOTS = os.path.join(ROOT, "GraphOfLifeRuns", ".pilots")
RULES = [("uniform", "evenly (control)", GREY), ("poorest", "to the poorest", GREEN),
         ("newborns", "to the newborns", VIOLET), ("parents", "to the parents", RED),
         ("children", "to its own children", BLUE)]
#: Each rule's worlds, by base: the control is the same settings and seeds with the estate shared evenly.
WORLDS = {
    "B2": {"uniform": ["P/B2-s1", "P/B2-s2"],
           **{r: [f"R/B2-{r}-s1", f"R/B2-{r}-s2"] for r, _, _ in RULES[1:]}},
    "B1": {"uniform": ["L/B1-long-s1", "L/B1-long-s2"],
           **{r: [f"R/B1-{r}-s1", f"R/B1-{r}-s2"] for r, _, _ in RULES[1:]}},
}
RUNS = ("Pilot worlds of 10,000 tokens, two seeds each: B2's settings (as in Experiment 16) and B1's, with "
        "`redistribution` = `poorest`, `newborns`, `parents` or `children` (`GraphOfLifeRuns/.pilots/R/`); the "
        "controls are the same settings and seeds with the estate shared evenly, `P/B2-s1`, `-s2` and the first "
        "2,000 iterations of `L/B1-long-s1`, `-s2`.")


def _series(world: str) -> List[Dict[str, Any]]:
    path = os.path.join(PILOTS, world, "series.jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("it", 0) <= 2000:
                rows.append(r)
    return rows


def _smooth(its: np.ndarray, v: np.ndarray, every: int = 20) -> Tuple[List[float], List[float]]:
    """Means over blocks of `every` iterations, so a line through 2,000 noisy rows reads."""
    xs, ys = [], []
    for start in range(int(its.min()), int(its.max()) + 1, every):
        sel = (its >= start) & (its < start + every) & np.isfinite(v)
        if sel.any():
            xs.append(float(start + every / 2))
            ys.append(float(v[sel].mean()))
    return xs, ys


def _value(r: Dict[str, Any], key: str) -> float:
    nodes = r.get("nodes") or np.nan
    if key == "births_per_agent":
        return (r.get("births", np.nan) / nodes) if nodes else np.nan
    if key == "death_age":
        return (r.get("deathAgeSum", np.nan) / r["deaths"]) if r.get("deaths") else np.nan
    v = r.get(key)
    return float(v) if isinstance(v, (int, float)) else np.nan


def _contests() -> Dict[str, Dict[Tuple[int, int], List[float]]]:
    out: Dict[str, Dict[Tuple[int, int], List[float]]] = {}
    for name in ("batch_R_T100.json", "batch_P_T100.json"):
        path = os.path.join(PILOTS, name)
        for c in json.load(open(path)) if os.path.exists(path) else []:
            world = c["world"].split(".pilots/")[-1]
            if c.get("share") is not None:
                out.setdefault(world, {}).setdefault((c["a"], c["b"]), []).append(c["share"])
    return out


@chapter
def inherit(ch: F.Chapter) -> None:
    present = [w for base in WORLDS.values() for ws in base.values() for w in ws if _series(w)]
    if not any(w.startswith("R/") for w in present):
        return

    # Each base's worlds through their 2,000 iterations.
    for base in ("B2", "B1"):
        charts = []
        for key, title, label in (("nodes", "Agents", "agents"),
                                  ("births_per_agent", "Births", "births per agent and iteration"),
                                  ("gini", "Inequality", "Gini coefficient of tokens"),
                                  ("meanAge", "Age", "mean age of the living")):
            series = []
            for rule, name, colour in RULES:
                for k, world in enumerate(WORLDS[base][rule]):
                    rows = _series(world)
                    if not rows:
                        continue
                    its = np.array([r["it"] for r in rows], float)
                    xs, ys = _smooth(its, np.array([_value(r, key) for r in rows], float))
                    series.append(F.line(xs, ys, name if k == 0 else None, colour, width=1.6))
            charts.append({"title": title, "x": {"label": "iteration"}, "y": {"label": label, "min": 0},
                           "series": series})
            for rule, _, _ in RULES:
                late = [np.nanmean([_value(r, key) for r in _series(w) if 1500 <= r["it"] <= 2000])
                        for w in WORLDS[base][rule] if _series(w)]
                if late:
                    ch.number(f"{base}:{rule}:{key}", [round(float(x), 4) for x in late])
        ch.grid(f"life_{base}", charts, columns=2, title=f"{base} worlds, by who inherits",
                caption=(f"{base} worlds of 10,000 tokens, two seeds of each rule, through 2,000 iterations (means "
                         f"over blocks of 20 iterations). Grey: the estate shared evenly, the control; green: to the "
                         f"poorest; violet: to the children born that iteration; red: to their parents; blue "
                         f"(added later): each estate to the dead agent's own living children."),
                recipe=recipe(RUNS, "each world's `series.jsonl`, one row per iteration",
                              ["Read agents, births, the Gini coefficient and mean age every iteration.",
                               "Average over blocks of 20 iterations and draw a line per world."]))

    # Does evolution go on? Lifetime contests between eras.
    contests = _contests()
    charts = []
    for pair, title in (((500, 1000), "1,000 against 500"), ((1000, 2000), "2,000 against 1,000")):
        cats, groups, colours = [], [], []
        for base in ("B2", "B1"):
            for rule, name, colour in RULES:
                vals = [np.mean(contests[w][pair]) for w in WORLDS[base][rule]
                        if w in contests and pair in contests[w]]
                if vals:
                    cats.append(f"{base} {rule}")
                    groups.append(vals)
                    colours.append(colour)
                    ch.number(f"contest:{base}:{rule}:{pair[1]}v{pair[0]}", [round(float(v), 3) for v in vals])
        if cats:
            charts.append({"title": title, "x": {"categories": cats},
                           "y": {"label": "share of nodes the later brains hold", "min": 0, "max": 1},
                           "series": F.dots(cats, groups, colours),
                           "guides": [{"axis": "y", "at": 0.5, "label": "a tie"}]})
    if charts:
        ch.grid("contests", charts, columns=1, title="Does evolution go on?",
                caption=("Lifetime contests (T = 100 iterations): the brains of a later moment against those of an "
                         "earlier one, half and half on the same world, once in each moment's world; the share of "
                         "nodes the later brains hold at the end, averaged over the two. Above a half, the later "
                         "brains are better. Each dot is a world."),
                recipe=recipe(RUNS, "`GraphOfLifeRuns/.pilots/batch_R_T100.json` and `batch_P_T100.json` "
                              "(`research/oee/checks/rejudge.py --T 100`)",
                              ["Load both moments' checkpoints; give half the agents of one world the brains of "
                               "each moment.", "Play 100 iterations; count the nodes each side holds.",
                               "Do it again in the other moment's world and average."]))
