"""
Births as rich as the neighbours (Experiment 32).

Experiment 19's world (brains mixed at birth, parents as mates) with the
estate of the dead paid to the newborns or to the parents, and a child that
must start with at least the median holding of its parent's neighbours; and
the floor alone. Each world read at iteration 3,000 by the four measures of
the author's picture beside the same worlds without the floor (Experiments 31
and 19); then where its poor agents come from: what agents hold at each age,
and how many births keep a world its size.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np

import book_figures as F
import gol_store
from book_figures import chapter, recipe
from book_chapters.common import BLUE, GREY, ORANGE, VIOLET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATLAS = os.path.join(ROOT, "GraphOfLifeRuns", ".pilots", "atlas")
#: B2's ten worlds played on, read at 3,000: the lines of the author's picture there.
LINES = {"marks": 7, "diversity": 0.543, "wide": 1.5, "gini_tokens": 0.466}
#: (the rule, the runs without the floor, the runs with it); seeds 1 and 2 of each.
RULES = [("parents as mates", "B2-10000-bcc1bc", "B2-10000-b289dd"),
         ("estate, newborns", "B2-10000-06b876", "B2-10000-12215f"),
         ("estate, parents", "B2-10000-b8e9fc", "B2-10000-86df9e")]
GROUPS = [("parents as mates, no floor", GREY), ("parents as mates, the floor", BLUE),
          ("an estate, no floor", VIOLET), ("an estate, the floor", ORANGE)]
AT = 3000
AGES = [(a, a + 9) for a in range(1, 100, 10)]
RUNS = ("Experiment 32's six runs (`python3 gol_lab.py run E32`): B2 at 10,000 tokens with brains mixed at birth, "
        "parents as mates and a child never poorer than the median holding of its parent's neighbours, with the "
        "estate of the dead paid to the newborns (`B2-10000-12215f-…`), to the parents (`-86df9e-…`) or neither "
        "(`-b289dd-…`), seeds 1 and 2, to 3,000 iterations; against Experiment 31's estate worlds of the same "
        "seeds (`B2-10000-06b876-…`, `-b8e9fc-…`) and Experiment 19's ten worlds (`B2-10000-bcc1bc-s001` … "
        "`-s010`), which have no floor.")
STEPS = ["Read each world as `research/atlas/profile.py` reads every world: the seven marks, the four spreads "
         "ranked among the frozen reference worlds (their mean is the diversity score), the Gini of the tokens; "
         "and its typical distance over its random twin's, as `research/atlas/space.py` measures it "
         "(`research/atlas/readings.py` does both).",
         "The lines: every mark; the median diversity, and the least Gini, of B2's ten worlds played on and read "
         "at 3,000; a typical distance of 1.5 times the twin's."]


def _group(run_id: str) -> str:
    base = run_id.rsplit("-s", 1)[0]
    for rule, without, with_floor in RULES:
        if base in (without, with_floor):
            kind = "parents as mates" if rule == "parents as mates" else "an estate"
            return f"{kind}, {'the floor' if base == with_floor else 'no floor'}"
    raise KeyError(run_id)


def _readings() -> List[Dict[str, Any]]:
    rows = []
    for name in ("e31_readings.json", "e32_readings.json"):
        path = os.path.join(ATLAS, name)
        if not os.path.exists(path):
            return []
        rows += [r for r in json.load(open(path)) if r.get("at") == AT]
    keep = {b for _, without, with_floor in RULES for b in (without, with_floor)}
    return [r for r in rows if r["id"].rsplit("-s", 1)[0] in keep]


def _ages(run_id: str) -> Tuple[np.ndarray, np.ndarray]:
    z = gol_store.load_archive(run_id, AT)
    return int(z["counters"][2]) - z["born_at"], z["tokens"].astype(float)


def _ledger(run_id: str) -> Tuple[float, float]:
    """Births an iteration and agents alive, means over iterations 2,000 to 3,000."""
    births, agents = [], []
    with open(os.path.join(gol_store.run_dir(run_id), "stats.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("phase") == 1 and 2000 <= r["iteration"] < AT:
                births.append(r["births"])
                agents.append(r["nodes"])
    return float(np.mean(births)), float(np.mean(agents))


def _floor(run_id: str) -> Dict[str, Any]:
    """How many agents could have a child at 3,000, and for how many the floor asks more than min_child."""
    z = gol_store.load_archive(run_id, AT)
    tokens = z["tokens"].astype(float)
    n = len(tokens)
    nb: List[List[int]] = [[] for _ in range(n)]
    for a, b in np.asarray(z["edges"]).reshape(-1, 2):
        if a != b and 0 <= a < n and 0 <= b < n:
            nb[a].append(int(b))
            nb[b].append(int(a))
    median = np.array([np.median(tokens[x]) if x else 0.0 for x in nb])
    able = tokens >= 11                  # a child of 10 (min_child) and one token kept
    age = int(z["counters"][2]) - z["born_at"][able]
    return {"able": int(able.sum()), "floor_above_min_child": int((able & (median > 10)).sum()),
            "median_neighbours": float(np.median(median)),
            "able_median_age": float(np.median(age)), "able_aged_10_or_less": float(np.mean(age <= 10))}


@chapter
def floor(ch: F.Chapter) -> None:
    rows = _readings()
    if len(rows) < 18:
        return
    cats = [rule for rule, _, _ in RULES]

    def panel(title: str, ylabel: str, key: str) -> Dict[str, Any]:
        series = []
        for side, colour, off, label in ((0, GREY, -0.17, "without the floor"), (1, ORANGE, 0.17, "with the floor")):
            xs, ys = [], []
            for k, rule in enumerate(RULES):
                mine = [r[key] for r in rows if r["id"].rsplit("-s", 1)[0] == rule[1 + side]]
                xs += list(F.jitter(k, len(mine), 3 + k + 7 * side, width=0.12) + off)
                ys += mine
            series.append({"label": label if key == "marks" else None, "x": xs, "y": ys, "points": True,
                           "size": 6, "colour": colour, "alpha": 0.85})
        series.append({"label": None, "x": [-0.5, len(cats) - 0.5], "y": [LINES[key]] * 2, "colour": GREY,
                       "dash": [5, 4], "width": 1.2})
        return {"title": title, "x": {"categories": cats}, "y": {"label": ylabel, "min": 0}, "series": series}

    ch.grid("measures", [panel("Marks of the health bar", "of 7", "marks"),
                         panel("Diversity score", "mean percentile", "diversity"),
                         panel("Width over the random twin's", "typical distance ratio", "wide"),
                         panel("Gini of the agents' tokens", "Gini", "gini_tokens")],
            title="The floor, with each estate and without", columns=2,
            caption=("Each dot is one world at iteration 3,000, without the floor (grey: Experiment 19's ten worlds "
                     "for parents as mates alone, Experiment 31's two for each estate) and with it (orange: "
                     "Experiment 32's two for each). The dashed line is what the author's picture asks: every mark, "
                     "a diversity score above B2's median world at 3,000 (0.54), a typical distance at least 1.5 "
                     "times the random twin's, a Gini no lower than B2's least even world at 3,000 (0.47)."),
            recipe=recipe(RUNS, "each world's archive at iteration 3,000", STEPS))

    worlds = [f"{base}-s{s:03d}" for _, without, with_floor in RULES for base in (without, with_floor)
              for s in (1, 2)]
    worlds += [f"B2-10000-bcc1bc-s{s:03d}" for s in range(3, 11)]
    age_series, by_age = [], {}
    for group, colour in GROUPS:
        mine = [w for w in worlds if _group(w) == group]
        held = []
        for w in mine:
            age, tokens = _ages(w)
            held.append([float(tokens[(age >= a) & (age <= b)].mean()) for a, b in AGES])
        by_age[group] = np.mean(held, axis=0).tolist()
        age_series.append({"label": group, "x": [(a + b) / 2 for a, b in AGES], "y": by_age[group],
                           "colour": colour, "width": 1.6})
    ledger = {w: _ledger(w) for w in worlds}
    top = max(b for b, _ in ledger.values()) * 1.1
    led_series = [{"label": None, "x": [ledger[w][0] for w in worlds if _group(w) == group],
                   "y": [ledger[w][1] for w in worlds if _group(w) == group], "points": True, "size": 6,
                   "colour": colour, "alpha": 0.85} for group, colour in GROUPS]
    led_series.append({"label": None, "x": [0, top], "y": [0, 100 * top], "colour": GREY, "dash": [5, 4],
                       "width": 1.2})
    ch.grid("paupers", [
        {"title": "What agents hold at each age", "x": {"label": "age (iterations)", "min": 0, "max": 100},
         "y": {"label": "mean tokens of the agents of that age", "log": True}, "series": age_series},
        {"title": "Births and agents", "x": {"label": "births an iteration", "min": 0},
         "y": {"label": "agents alive", "min": 0}, "series": led_series}],
        title="Where the poor agents come from", columns=2,
        caption=("Left: the mean tokens of the agents of each age, in tens of iterations, at iteration 3,000, "
                 "averaged over the worlds of each kind: Experiment 19's ten, and seeds 1 and 2 of the others. "
                 "Right: each world's births an iteration and agents alive, both averaged over iterations 2,000 to 3,000; the dashed line is a hundred agents "
                 "for every birth an iteration, which a world keeps when everyone lives out a hundred iterations."),
        recipe=recipe(RUNS, "each world's archive at iteration 3,000 and its `stats.jsonl`",
                      ["Left: each agent's age is the iteration less the one it was born in; average the tokens "
                       "of the agents aged 1 to 10, 11 to 20, and so on.",
                       "Right: the births and the agents of each iteration's first phase, averaged over the "
                       "iterations 2,000 to 2,999."]))
    ch.number("worlds", [{**{k: r.get(k) for k in ("id", "agents", "marks", "diversity", "wide", "gini_tokens",
                                                    "tokens_per_agent", "dimension")}, "group": _group(r["id"])}
                         for r in rows])
    ch.number("by_age", by_age)
    ch.number("ledger", {w: {"births": round(b, 1), "agents": round(a), "agents_per_birth": round(a / b, 1)}
                         for w, (b, a) in ledger.items()})
    ch.number("floor_binds", {w: _floor(w) for w in worlds})
