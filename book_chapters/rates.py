# -*- coding: utf-8 -*-
"""
the work on open-ended evolution, How fast should brains change? — Experiment 9.

B2 at five rates of change: a brain copied into a newborn or a won node
changes with probability 0.05, 0.1, 0.2 (B2 itself), 0.4 or 0.8. Two kinds of
contest decide the thesis. Across rates: the brains of iteration 1,500 of two
rates of one seed, played in each of the two worlds (gol_tournament.contest
with brains from two runs). Within a rate: each world's eras, each against the
one before, the last step being iteration 1,500 against 1,000.
"""
from __future__ import annotations

import os
import re
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Tuple

import numpy as np

import book_data as D
import book_figures as F
from book_figures import chapter, describe, recipe, runs_of
from book_chapters.common import BLUE, GREEN, GREY, ORANGE, RED, VIOLET
import gol_store as store
import gol_tournament as tour

RATES: Tuple[Tuple[str, float], ...] = (("mutation 0.05", 0.05), ("mutation 0.1", 0.1), ("B2", 0.2),
                                        ("mutation 0.4", 0.4), ("mutation 0.8", 0.8))
RATE_COLOURS = (BLUE, GREEN, GREY, ORANGE, RED)
END = 1500
STEP_ERAS = (0, 100, 200, 500, 1000, 1500)
T = 20
E09_RUNS = ("The 50 runs of Experiment 9 — B2 at 10,000 tokens, seeds 1 to 10, with the probability "
            "that a brain changes when it is copied at 0.05, 0.1, 0.2 (B2 itself, `B2-10000-s001` … "
            "`-s010`, shared with Experiment 16), 0.4 and 0.8, run to iteration 1,500 at least with a "
            "whole world archived every 100. They are made with `python3 gol_lab.py run E09`; every "
            "setting is listed in the box of the runs behind this page.")
ARCHIVE = ("each run's archive, `GraphOfLifeRuns/<run>/archive/world_<iteration>.npz`, kept every 100 "
           "iterations")


def _jobs() -> int:
    return int(os.environ.get("GOL_BOOK_JOBS", max(1, (os.cpu_count() or 2) // 2)))


def _seed(run_id: str) -> int:
    return int(re.search(r"-s(\d+)$", run_id).group(1))


def _by_seed() -> Dict[int, Dict[float, str]]:
    """Every seed's five runs, by rate."""
    out: Dict[int, Dict[float, str]] = {}
    for name, rate in RATES:
        for spec in runs_of("E09", name):
            out.setdefault(_seed(spec.run_id), {})[rate] = spec.run_id
    return out


def _lived(run_id: str) -> int:
    return store.load_meta(run_id).get("iteration", 0) if os.path.exists(store.meta_path(run_id)) else 0


def _cross_one(task: Tuple[str, str, str, float, float, int]) -> Dict[str, Any]:
    host, run_a, run_b, rate_a, rate_b, seed = task
    config = store.load_meta(host)["config"]
    result = tour.contest(config, store.load_archive(host, END), store.load_archive(run_a, END),
                          store.load_archive(run_b, END), T, seed)
    return {"host": host, "a": rate_a, "b": rate_b, "seed": seed, **result}


@D.measure("rates-cross", field="key", want=lambda run_id: {"end": END, "T": T, "rates": [r for _, r in RATES]})
def rates_cross(run_id: str) -> Dict[str, Any]:
    """For the seed of `run_id` (its B2 run): every pair of rates, at home and away, at iteration 1,500."""
    found = re.search(r"-s(\d+)$", run_id)
    runs = _by_seed().get(int(found.group(1)), {}) if found else {}
    rates = [r for _, r in RATES]
    if set(runs) != set(rates) or any(END not in store.archived(runs[r]) for r in rates):
        return {"contests": [], "missing": f"not every rate of this seed has a world archived at {END}"}
    tasks = [(runs[host], runs[a], runs[b], a, b, 17)
             for i, a in enumerate(rates) for b in rates[i + 1:] for host in (a, b)]
    with ProcessPoolExecutor(_jobs()) as pool:
        contests = list(pool.map(_cross_one, tasks))
    return {"contests": contests}


def _want_steps(run_id: str) -> Dict[str, Any]:
    have = set(store.archived(run_id))
    return {"eras": [e for e in STEP_ERAS if e in have], "T": T, "coins": 1}


@D.measure("tournament-steps", field="key", want=_want_steps)
def tournament_steps(run_id: str) -> Dict[str, Any]:
    """Each era against the one before, at home and away."""
    eras = _want_steps(run_id)["eras"]
    if len(eras) < 2:
        return {"eras": eras, "contests": []}
    played = tour.play(run_id, eras, T=T, coins=1, against="previous", jobs=_jobs())
    return {"eras": played["eras"], "contests": played["contests"]}


@D.measure("clones-1500", field="key", want=lambda run_id: {"at": END})
def clones(run_id: str) -> Dict[str, Any]:
    """Groups of clones in the world of iteration 1,500: neighbours joined when they carry the same brain."""
    if END not in store.archived(run_id):
        return {"missing": f"no world archived at iteration {END}"}
    blob = store.load_archive(run_id, END)
    n = len(blob["ids"])
    brain = blob["brain_ids"]
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in blob["edges"].tolist():
        if brain[a] == brain[b]:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    sizes: Dict[int, int] = {}
    for i in range(n):
        r = find(i)
        sizes[r] = sizes.get(r, 0) + 1
    v = np.array(list(sizes.values()), dtype=float)
    return {"agents": n, "groups": len(v), "largest": int(v.max()),
            "agentMean": float((v * v).sum() / n), "genotypes": int(len(set(brain.tolist())))}


def _pair(contests: List[Dict[str, Any]], a: float, b: float) -> float:
    """b's share against a, the mean over the contests the two played."""
    s = [c["share"] for c in contests if c["a"] == a and c["b"] == b and c["share"] is not None]
    return float(np.mean(s)) if s else float("nan")


def _step(played: Dict[str, Any], a: int, b: int) -> float:
    s = [c["share"] for c in played["contests"] if c["a"] == a and c["b"] == b and c["share"] is not None]
    return float(np.mean(s)) if s else float("nan")


@chapter
def rates(ch: F.Chapter) -> None:
    seeds = _by_seed()
    every = [r for runs in seeds.values() for r in runs.values()]
    if not every or any(_lived(r) < END for r in every):
        return
    rates_ = [r for _, r in RATES]
    names = dict((r, n) for n, r in RATES)
    pairs = [(a, b) for i, a in enumerate(rates_) for b in rates_[i + 1:]]

    # Across rates: a pair's score per seed is the second rate's mean share in its two contests.
    cross = {s: rates_cross(runs[0.2])["contests"] for s, runs in sorted(seeds.items())}
    table = {f"{a}–{b}": [_pair(cross[s], a, b) for s in sorted(cross)] for a, b in pairs}
    labels = list(table)
    from book_chapters.oee import dots
    ch.figure("cross", **dots("cross", labels, table, title="The brains of one rate against another's",
                              y={"label": "the second rate's share of the world", "min": 0.3, "max": 0.7},
                              guides=[{"axis": "y", "at": v, "label": t} for v, t in
                                      ((0.45, "the thesis's band"), (0.55, ""))]),
              caption=("Each column is a pair of rates of change, written lower–higher; each dot one seed: "
                       "the share of a world the brains of iteration 1,500 of the higher rate held "
                       "against those of the lower, the mean of two contests, one in each rate's world. "
                       "The bar is the median seed; the dashed lines the band of the thesis."),
              recipe=recipe(E09_RUNS, ARCHIVE, [
                  "For each seed and each pair of rates, take the worlds of both rates at iteration 1,500. "
                  "Play two contests, one in each of the two worlds: every agent gets a brain drawn at "
                  "random from the agents of one rate's world or of the other's, a coin each, mutation is "
                  "switched off, and 20 iterations are played (`gol_tournament.contest`, brains from two "
                  "runs).",
                  "The higher rate's score is its share of the agents at the end, averaged over the two."]))
    ch.number("cross", {k: describe(v) for k, v in table.items()})

    # Within a rate: each world's steps.
    steps = {r: [tournament_steps(seeds[s][r]) for s in sorted(seeds)] for r in rates_}
    step_pairs = list(zip(STEP_ERAS, STEP_ERAS[1:]))
    step_labels = [f"{a:,}→{b:,}" for a, b in step_pairs]
    dots_ = []
    for k, r in enumerate(rates_):
        rng = np.random.RandomState(5 + k)
        for j, (a, b) in enumerate(step_pairs):
            v = [x for x in (_step(p, a, b) for p in steps[r]) if np.isfinite(x)]
            x0 = j + (k - 2) * 0.16
            dots_.append({"label": names[r] if j == 0 else None,
                          "x": (x0 + rng.uniform(-0.04, 0.04, len(v))).tolist(), "y": v,
                          "points": True, "size": 5, "colour": RATE_COLOURS[k], "alpha": 0.75})
            dots_.append({"label": None, "x": [x0 - 0.07, x0 + 0.07], "y": [float(np.median(v))] * 2,
                          "colour": RATE_COLOURS[k], "width": 3})
    ch.figure("steps", title="Each era against the one before, at five rates of change",
              x={"label": "the step, from one era to the next", "categories": step_labels},
              y={"label": "the later era's share of the world", "min": 0.3, "max": 0.85},
              series=dots_, guides=[{"axis": "y", "at": 0.5, "label": "a tie"}],
              caption=("One dot per world: the later era's share of the agents, the mean of its two "
                       "contests with the era before, one in each era's world. The bars are the medians "
                       "of each rate's ten worlds."),
              recipe=recipe(E09_RUNS, ARCHIVE, [
                  "For each world, play each of the eras 0, 100, 200, 500, 1,000 and 1,500 against the one "
                  "before it, once in each era's world (`gol_tournament.play` with `against=\"previous\"`, "
                  "20 iterations, a coin per agent), and average each pair's two shares."]))
    late = {names[r]: [_step(p, 1000, 1500) for p in steps[r]] for r in rates_}
    ch.number("late_vs_1000", {k: describe(v) for k, v in late.items()})
    ch.number("steps", {names[r]: {lab: describe([_step(p, a, b) for p in steps[r]])
                                   for lab, (a, b) in zip(step_labels, step_pairs)} for r in rates_})

    # What the rates did to the worlds, settled from iteration 500.
    life = {}
    for r in rates_:
        rows = {}
        for s in sorted(seeds):
            run = seeds[s][r]
            rows[run] = {stat: F.mean_over(run, stat, 500, END) for stat in ("nodes", "distinctBrains")}
        life[names[r]] = rows
    genos = {names[r]: [v["distinctBrains"] / v["nodes"] for v in life[names[r]].values()] for r in rates_}
    agents = {names[r]: [v["nodes"] for v in life[names[r]].values()] for r in rates_}
    groups = {names[r]: [clones(seeds[s][r])["agentMean"] for s in sorted(seeds)] for r in rates_}
    panels = [dots("genotypes", list(genos), genos, title="genotypes per agent",
                   y={"label": "distinct brains per agent", "min": 0}),
              dots("groups", list(groups), groups, title="an agent's group of clones at iteration 1,500",
                   y={"label": "agents, on average", "min": 0}),
              dots("agents", list(agents), agents, title="agents", y={"label": "agents", "min": 0})]
    ch.grid("worlds", panels, columns=1, title="What the rate of change does to a world",
            caption=("One dot per world, the bar the median of ten. Genotypes and agents: the mean over "
                     "iterations 500 to 1,500. Groups of clones: in the world of iteration 1,500, "
                     "neighbours that carry the same brain joined together, the size of an agent's group "
                     "averaged over agents."),
            recipe=recipe(E09_RUNS, F.STATS_FILE + " and the archive of iteration 1,500", [
                "For every run take the rows with `phase` = 2 and 500 ≤ `iteration` ≤ 1,500 and average "
                "`distinctBrains` / `nodes` and `nodes`.",
                "Load the world of iteration 1,500 (`gol_store.load_archive`), join every two neighbours "
                "with the same `brain_ids` into one group, and average over agents the size of the group "
                "each belongs to (`rates.clones`)."]))
    ch.number("worlds", {k: {"genotypes_per_agent": describe(genos[k]), "agents": describe(agents[k]),
                             "clone_group": describe(groups[k])} for k in genos})
