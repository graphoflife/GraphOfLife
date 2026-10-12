# -*- coding: utf-8 -*-
"""
Part V: engineering open-ended evolution. Chapter 49 — how to tell a world that
keeps evolving.

The yardstick is a cross-time tournament (gol_tournament): the brains of two
moments of one world play each other in that world, with mutation switched off,
and the share of the world each era ends up holding is its score. A world that
keeps evolving keeps beating its own past; one that has stopped only ties with
it.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np

import book_data as D
import book_figures as F
from book_figures import chapter, describe, recipe, runs_of
from book_chapters.common import BLUE, GREY
import gol_store as store
import gol_tournament as tour

#: The moments a world's brains are played from: the founders, then ever
#: further apart. Every one of them is in the archive of a run made with
#: archive_every = 50; a run without one has none of them.
ERAS = (0, 50, 100, 200, 300, 500, 700, 1000)

#: Iterations a contest lasts. Long enough for most of a contested world to
#: change hands at least once (Chapter 16), short enough to play many.
T = 20


def _jobs() -> int:
    """Contests are played in parallel, on this many processes."""
    return int(os.environ.get("GOL_BOOK_JOBS", max(1, (os.cpu_count() or 2) // 2)))


def _eras(run_id: str) -> List[int]:
    have = set(store.archived(run_id))
    return [e for e in ERAS if e in have]


def _want(run_id: str) -> Dict[str, Any]:
    """A kept tournament holds as long as the same eras were played the same way."""
    return {"eras": _eras(run_id), "T": T, "coins": 1}


@D.measure("tournament", field="key", want=_want)
def tournament(run_id: str) -> Dict[str, Any]:
    """Every pair of eras, at home and away, and one rating per era."""
    eras = _eras(run_id)
    if len(eras) < 2:
        return {"eras": eras, "elo": [0.0] * len(eras), "residual": None, "contests": []}
    played = tour.play(run_id, eras, T=T, coins=1, against="all", jobs=_jobs())
    return {"eras": played["eras"], "elo": played["elo"], "residual": played["residual"],
            "contests": played["contests"]}


def score(played: Dict[str, Any], earlier: int, later: int) -> float:
    """The later era's share against the earlier one, averaged over both worlds it was played in."""
    shares = [c["share"] for c in played["contests"]
              if c["a"] == earlier and c["b"] == later and c["share"] is not None]
    return float(np.mean(shares)) if shares else float("nan")


E12_RUNS = ("The 10 runs of Experiment 12 — B1 at 5,000 tokens, seeds 1 to 10, 1,000 "
            "iterations, a whole world archived every 50 — `B1-5000-s001` … `-s010`. They are made "
            "with `python3 gol_lab.py run E12`; every setting is listed in the box at the top of "
            "the chapter.")
ARCHIVE = ("each run's archive, `GraphOfLifeRuns/<run>/archive/world_<iteration>.npz`: the whole "
           "world as a checkpoint writes it, kept every 50 iterations")
HOW = (f"For every pair of the eras {', '.join(str(e) for e in ERAS)} play two contests, one in "
       f"each era's world: give every agent a brain drawn at random from one era or the other, a "
       f"coin each, clear the messages, switch mutation off and run {T} iterations "
       f"(`gol_tournament.contest`). The later era's score is the share of agents carrying one "
       f"of its brains at the end.")


@chapter
def yardstick(ch: F.Chapter) -> None:
    specs = [s for s in runs_of("E12", "baseline")
             if store.load_meta(s.run_id).get("iteration", 0) >= ERAS[-1]]
    if not specs:
        return
    played = {s.lab["seed"]: tournament(s.run_id) for s in specs}

    # Ratings against the founders, one world at a time.
    series = []
    table = []
    for seed, p in sorted(played.items()):
        series.append({"x": p["eras"], "y": p["elo"], "colour": GREY, "alpha": 0.45, "width": 1})
        table.append(dict(zip(p["eras"], p["elo"])))
    median = [float(np.median([row[e] for row in table if e in row])) for e in ERAS]
    series.append({"x": list(ERAS), "y": median, "colour": BLUE, "width": 3,
                   "label": "the median world"})
    ch.number("elo_median", dict(zip(ERAS, median)))
    ch.figure(
        "elo",
        title="How much better than the founders",
        x={"label": "iteration the brains are taken from"},
        y={"label": "Elo rating against the founders"},
        series=series,
        guides=[{"axis": "y", "at": 0, "label": "the founders"}],
        caption=("**How much better than the founders.** Each grey line is one world; the blue "
                 "line is the median of the ten at each era. A rating says how often an era's "
                 "brains beat the founders': +200 is about three contests in four, +400 ten in "
                 "eleven. The ratings are fitted to every contest between every pair of eras, "
                 "so each point leans on the others."),
        recipe=recipe(E12_RUNS, ARCHIVE, [
            HOW,
            "Read every score *p* as odds, log(*p* / (1 − *p*)), after keeping it between 0.02 and "
            "0.98, and find one rating per era — the founders' at 0 — that fits them best by least "
            "squares, the later era's rating minus the earlier's for each contest "
            "(`gol_tournament.ratings`). Multiply by 400 / ln 10 for Elo points.",
            "Draw each world's ratings, and the median world at each era.",
        ]))

    # Every pair of eras, the median world's score.
    cells = {"kind": "cells", "x0": [], "x1": [], "y0": [], "y1": [], "value": []}
    for i, earlier in enumerate(ERAS):
        for later in ERAS[i + 1:]:
            values = [score(p, earlier, later) for p in played.values()]
            values = [v for v in values if np.isfinite(v)]
            if not values:
                continue
            cells["x0"].append(i - 0.5)
            cells["x1"].append(i + 0.5)
            j = ERAS.index(later)
            cells["y0"].append(j - 0.5)
            cells["y1"].append(j + 0.5)
            cells["value"].append(float(np.median(values)))
    ch.figure(
        "pairs",
        title="Every era against every earlier one",
        x={"label": "the earlier era", "categories": [str(e) for e in ERAS]},
        y={"label": "the later era", "categories": [str(e) for e in ERAS]},
        series=[cells],
        colourbar={"label": "the later era's share", "min": 0.3, "max": 0.9},
        caption=("**Every era against every earlier one.** Each square is one pair of eras: "
                 "the share of a contested world the later era's brains held, in the median "
                 "world. 0.5 is a tie. The bottom row is every era against the founders; a "
                 "world still improving would be darker the further a square is from the "
                 "diagonal."),
        recipe=recipe(E12_RUNS, ARCHIVE, [
            HOW,
            "For each world and each pair, average its two contests.",
            "Take the median over the ten worlds, and colour the pair's square by it.",
        ]))

    late = [score(p, 300, 1000) for p in played.values()]
    early = [score(p, 0, 300) for p in played.values()]
    ch.number("late_vs_300", describe(late))
    ch.number("300_vs_founders", describe(early))
