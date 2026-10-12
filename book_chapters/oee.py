# -*- coding: utf-8 -*-
"""
Part V: engineering open-ended evolution — Chapters 49 to 53, Experiments 12 to 16.

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
    curves = []
    table = []
    for seed, p in sorted(played.items()):
        curves.append({"x": p["eras"], "y": p["elo"], "colour": GREY, "alpha": 0.45, "width": 1})
        table.append(dict(zip(p["eras"], p["elo"])))
    median = [float(np.median([row[e] for row in table if e in row])) for e in ERAS]
    curves.append({"x": list(ERAS), "y": median, "colour": BLUE, "width": 3,
                   "label": "the median world"})
    ch.number("elo_median", dict(zip(ERAS, median)))
    ch.figure(
        "elo",
        title="How much better than the founders",
        x={"label": "iteration the brains are taken from"},
        y={"label": "Elo rating against the founders"},
        series=curves,
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


# ---------------------------------------------------------------------------
# What every chapter of the part reads off a world
# ---------------------------------------------------------------------------

from book_figures import band_series, line, series  # noqa: E402
from book_chapters.common import GREEN, ORANGE, RED, VIOLET, YELLOW  # noqa: E402

COLOURS = (BLUE, RED, GREEN, VIOLET, ORANGE, YELLOW)

#: The health bar of Chapter 49: statistic of the recorded rows, its words,
#: and the range it has to lie in.
BAR = (("coreShare", "agents in the 2-core", 0.8, None),
       ("cutRisk", "largest share one connection cuts off", None, 0.01),
       ("maxDegree/nodes", "best-connected agent's links per agent", None, 0.05),
       ("meanDegree", "connections per agent", 3.0, 12.0),
       ("dimension", "ball dimension", 2.3, 4.0))


def settled_median(run_id: str, stat: str, start: int, stop: int, phase: int = 2) -> float:
    its, values = series(run_id, stat, phase)
    pick = (its >= start) & (its <= stop) & np.isfinite(values)
    return float(np.median(values[pick])) if pick.any() else float("nan")


def lowest(run_id: str, start: int = 50) -> float:
    """The fewest agents after the game of any iteration from `start` on (0 for a world that died out)."""
    its, n = series(run_id, "nodes", 2)
    pick = its >= start
    low = float(np.min(n[pick])) if pick.any() else float("nan")
    return 0.0 if store.load_meta(run_id).get("status") == "extinct" else low


def dots(name: str, conditions: List[str], values: Dict[str, List[float]], *, title: str, y: Dict[str, Any],
         guides: List[Dict[str, Any]] = ()) -> Dict[str, Any]:
    """One column of dots per condition, with its median as a bar."""
    out = []
    for k, cond in enumerate(conditions):
        v = np.array([x for x in values.get(cond, []) if x is not None and np.isfinite(x)], dtype=float)
        if not v.size:
            continue
        out.append({"label": None, "x": F.jitter(k, v.size, 11 + k).tolist(), "y": v.tolist(),
                    "points": True, "size": 6, "colour": COLOURS[k % len(COLOURS)], "alpha": 0.85})
        out.append(line([k - 0.38, k + 0.38], [float(np.median(v))] * 2, None, COLOURS[k % len(COLOURS)], width=3))
    return {"title": title, "x": {"categories": conditions}, "y": y, "series": out,
            "guides": list(guides), "legend": False}


# ---------------------------------------------------------------------------
# Chapter 50 · Brains that see, and worlds that break (Experiment 13)
# ---------------------------------------------------------------------------

E13_CONDITIONS = ["baseline", "one hidden layer", "one hidden layer, change only when copied"]
E13_RUNS = ("The 30 runs of Experiment 13 — at 5,000 tokens, seeds 1 to 10, 1,500 iterations — "
            "in three conditions: the baseline B1 (`B1-5000-s001` … `-s010`, shared with "
            "Experiment 12), its brains with one hidden layer of 16, and those brains changed only "
            "when copied. They are made with `python3 gol_lab.py run E13`.")


@chapter
def seeing(ch: F.Chapter) -> None:
    specs = {c: [s for s in runs_of("E13", c) if store.load_meta(s.run_id).get("iteration", 0) > 0]
             for c in E13_CONDITIONS}
    if not all(specs.values()):
        return
    low = {c: [lowest(s.run_id) for s in specs[c]] for c in E13_CONDITIONS}
    broke = {c: int(sum(1 for v in low[c] if v < 50)) for c in E13_CONDITIONS}
    ch.number("lowest", {c: describe(low[c]) for c in E13_CONDITIONS})
    ch.number("broke", broke)

    bands = [band_series([series(s.run_id, "nodes", 2) for s in specs[c]], c, COLOURS[k])
             for k, c in enumerate(E13_CONDITIONS)]
    ch.figure("agents", title="Agents alive", x={"label": "iteration"}, y={"label": "agents", "min": 0},
              series=bands,
              caption=("**Agents alive.** For each condition the median world at each iteration, "
                       "with the middle half of the worlds (darker) and nine in ten (paler)."),
              recipe=recipe(E13_RUNS, F.STATS_FILE, [
                  "Take `nodes` of the rows with `phase` = 2, one line per run.",
                  "For each condition, the median, quartiles and 5th and 95th percentiles of the runs "
                  "at each iteration (`book_figures.band_series`)."]))
    ch.figure("lowest", **dots("lowest", E13_CONDITIONS, low, title="The fewest agents a world fell to",
                               y={"label": "agents (0: died out)", "min": 0},
                               guides=[{"axis": "y", "at": 50, "label": "50 agents"}]),
              caption=("**The fewest agents a world fell to.** One dot per world: its fewest agents "
                       "after the game of any iteration from 50 on, 0 for a world that died out; the "
                       "bar is the median world. The dashed line is the thesis's 50 agents."),
              recipe=recipe(E13_RUNS, F.STATS_FILE, [
                  "Take the smallest `nodes` of the rows with `phase` = 2 and `iteration` ≥ 50; a run "
                  "that ended extinct counts 0.",
                  "Count the worlds below 50 in each condition."]))
    panels = []
    for stat, words, lo, hi in BAR[:3]:
        values = {c: [settled_median(s.run_id, stat, 200, 1500) for s in specs[c]] for c in E13_CONDITIONS}
        guides = [{"axis": "y", "at": v, "label": "the bar"} for v in (lo, hi) if v is not None]
        panels.append(dots(stat, E13_CONDITIONS, values, title=words, y={"label": words, "min": 0},
                           guides=guides))
    ch.grid("health", panels, columns=3, title="How the network holds",
            caption=("**How the network holds.** One dot per world, at its median over iterations "
                     "200 to 1,500 while it lived; the bar is the median world and the dashed line the "
                     "health bar of Chapter 49."),
            recipe=recipe(E13_RUNS, F.STATS_FILE, [
                "Take `coreShare`, `cutRisk` and `maxDegree` / `nodes` of the rows with `phase` = 2 "
                "and 200 ≤ `iteration` ≤ 1,500, and each world's median of each."]))


# ---------------------------------------------------------------------------
# Chapter 51 · A world that cannot break (Experiment 14)
# ---------------------------------------------------------------------------

E14_CONDITIONS = ["all the rules", "no mending", "any unused link lapses", "no links at birth",
                  "no valence", "handover"]
E14_RUNS = ("The 60 runs of Experiment 14 — B2 at 5,000 tokens, seeds 1 to 10, 1,000 iterations — "
            "in six conditions: all the rules of B2, and B2 with one of them taken away or "
            "changed. They are made with `python3 gol_lab.py run E14`.")


def health(run_id: str, start: int = 200, stop: int = 1000) -> Dict[str, Any]:
    """A world against the health bar: its medians, whether each passes, and whether anyone was cut off."""
    medians = {stat: settled_median(run_id, stat, start, stop) for stat, *_ in BAR}
    passes = {stat: bool(np.isfinite(medians[stat]) and (lo is None or medians[stat] >= lo)
                         and (hi is None or medians[stat] <= hi)) for stat, _, lo, hi in BAR}
    cut_off = 0.0
    for phase in (1, 2):
        its, orphaned = series(run_id, "orphaned", phase)
        pick = (its >= start) & (its <= stop) & np.isfinite(orphaned)
        cut_off += float(np.sum(orphaned[pick]))
    return {"medians": medians, "passes": passes, "cutOff": cut_off,
            "holds": all(passes.values()) and cut_off == 0}


@chapter
def unbreakable(ch: F.Chapter) -> None:
    specs = {c: [s for s in runs_of("E14", c) if store.load_meta(s.run_id).get("iteration", 0) >= 1000]
             for c in E14_CONDITIONS}
    if not all(specs.values()):
        return
    checked = {c: [health(s.run_id) for s in specs[c]] for c in E14_CONDITIONS}
    ch.number("hold", {c: int(sum(h["holds"] for h in checked[c])) for c in E14_CONDITIONS})
    ch.number("worlds", {c: len(checked[c]) for c in E14_CONDITIONS})
    ch.number("medians", {c: {stat: describe([h["medians"][stat] for h in checked[c]]) for stat, *_ in BAR}
                          for c in E14_CONDITIONS})
    panels = []
    for stat, words, lo, hi in BAR:
        values = {c: [h["medians"][stat] for h in checked[c]] for c in E14_CONDITIONS}
        guides = [{"axis": "y", "at": v, "label": "the bar"} for v in (lo, hi) if v is not None]
        panels.append(dots(stat, E14_CONDITIONS, values, title=words, y={"label": words, "min": 0},
                           guides=guides))
    ch.grid("bar", panels, columns=2, title="Each rule against the health bar",
            caption=("**Each rule against the health bar.** One dot per world, at its median over "
                     "iterations 200 to 1,000; the bar is the median world; dashed lines are the "
                     "health bar of Chapter 49. The first column has every rule of B2; each other "
                     "column takes one away."),
            recipe=recipe(E14_RUNS, F.STATS_FILE, [
                "For every run take the rows with `phase` = 2 and 200 ≤ `iteration` ≤ 1,000, and "
                "the median of `coreShare`, `cutRisk`, `maxDegree` / `nodes`, `meanDegree` and "
                "`dimension`.",
                "A world holds if every median is inside the bar and `orphaned` is 0 in every row "
                "of either phase in the same iterations."]))


# ---------------------------------------------------------------------------
# Chapter 52 · Who dies, and why (Experiment 15)
# ---------------------------------------------------------------------------

E15_CONDITIONS = ["B2", "the last token kept", "nothing kept", "half kept", "children of any size",
                  "no old age"]
E15_RUNS = ("The 60 runs of Experiment 15 — B2 at 5,000 tokens, seeds 1 to 10, 1,000 iterations — "
            "in six conditions that change how much an agent keeps at home, how small a child may "
            "be, and whether agents die of age. They are made with `python3 gol_lab.py run E15`.")


@D.measure("onetoken")
def one_token(run_id: str) -> Dict[str, Any]:
    """The share of agents holding a single token after the game, every 50 iterations."""
    last = D.last_iteration(run_id)
    its, shares = [], []
    for t in range(0, last + 1, 50):
        try:
            frame = D.frame_at(run_id, t, 2)
        except (KeyError, ValueError, FileNotFoundError):
            continue
        tokens = np.array(frame["tokens"])
        if tokens.size:
            its.append(t)
            shares.append(float(np.mean(tokens == 1)))
    return {"iterations": its, "shares": shares}


@chapter
def deaths(ch: F.Chapter) -> None:
    specs = {c: [s for s in runs_of("E15", c) if store.load_meta(s.run_id).get("iteration", 0) > 0]
             for c in E15_CONDITIONS}
    if not all(specs.values()):
        return
    single = {}
    for c in E15_CONDITIONS:
        single[c] = []
        for s in specs[c]:
            m = one_token(s.run_id)
            v = [x for t, x in zip(m["iterations"], m["shares"]) if t >= 300]
            single[c].append(float(np.median(v)) if v else float("nan"))
    low = {c: [lowest(s.run_id) for s in specs[c]] for c in E15_CONDITIONS}
    ch.number("one_token", {c: describe(single[c]) for c in E15_CONDITIONS})
    ch.number("lowest", {c: describe(low[c]) for c in E15_CONDITIONS})
    ch.number("starved_worlds", {c: int(sum(1 for v in low[c] if v < 100)) for c in E15_CONDITIONS})
    ch.grid("who", [
        dots("single", E15_CONDITIONS, single, title="Agents holding a single token",
             y={"label": "share of agents", "min": 0, "max": 1},
             guides=[{"axis": "y", "at": 0.5, "label": "half"}]),
        dots("lowest", E15_CONDITIONS, low, title="The fewest agents a world fell to",
             y={"label": "agents (0: died out)", "min": 0},
             guides=[{"axis": "y", "at": 100, "label": "100 agents"}]),
    ], columns=1, title="Who is left to play",
        caption=("**Who is left to play.** Top: one dot per world, the median over iterations 300 "
                 "to 1,000 of the share of its agents holding exactly one token after the game. "
                 "Bottom: the fewest agents it fell to after the game of any iteration from 50 on, "
                 "0 if it died out. Bars are the median worlds."),
        recipe=recipe(E15_RUNS, "each run's frames and `stats.jsonl`", [
            "For every 50th iteration from 300, read the frame after the game "
            "(`book_data.frame_at(run, t, 2)`) and the share of its `tokens` equal to 1; take each "
            "world's median.",
            "Take the smallest `nodes` of the rows with `phase` = 2 and `iteration` ≥ 50; 0 for a "
            "run that ended extinct."]))


# ---------------------------------------------------------------------------
# Chapter 53 · Why evolution stops (Experiment 16)
# ---------------------------------------------------------------------------

E16_ERAS = (0, 100, 200, 500, 1000, 1500, 2000)
E16_CONDITIONS = ["B2", "no random inputs"]
E16_RUNS = ("The 20 runs of Experiment 16 — B2 at 5,000 tokens, seeds 1 to 10, 2,000 iterations, "
            "a whole world archived every 100 — with and without the random inputs. They are made "
            "with `python3 gol_lab.py run E16`.")


def _want_long(run_id: str) -> Dict[str, Any]:
    have = set(store.archived(run_id))
    return {"eras": [e for e in E16_ERAS if e in have], "T": T, "coins": 1}


@D.measure("tournament-long", field="key", want=_want_long)
def tournament_long(run_id: str) -> Dict[str, Any]:
    """The tournament of a world that lived 2,000 iterations, at fewer eras."""
    eras = _want_long(run_id)["eras"]
    if len(eras) < 2:
        return {"eras": eras, "elo": [0.0] * len(eras), "residual": None, "contests": []}
    played = tour.play(run_id, eras, T=T, coins=1, against="all", jobs=_jobs())
    return {"eras": played["eras"], "elo": played["elo"], "residual": played["residual"],
            "contests": played["contests"]}


@chapter
def stopping(ch: F.Chapter) -> None:
    specs = {c: [s for s in runs_of("E16", c) if store.load_meta(s.run_id).get("iteration", 0) >= 2000]
             for c in E16_CONDITIONS}
    if not all(specs.values()):
        return
    played = {c: [tournament_long(s.run_id) for s in specs[c]] for c in E16_CONDITIONS}
    series_ = []
    for k, c in enumerate(E16_CONDITIONS):
        table = [dict(zip(p["eras"], p["elo"])) for p in played[c]]
        for row in table:
            series_.append({"x": list(row), "y": list(row.values()), "colour": COLOURS[k], "alpha": 0.3,
                            "width": 1})
        med = [float(np.median([row[e] for row in table if e in row])) for e in E16_ERAS]
        series_.append({"x": list(E16_ERAS), "y": med, "colour": COLOURS[k], "width": 3, "label": c})
    ch.figure("elo", title="How much better than the founders, over 2,000 iterations",
              x={"label": "iteration the brains are taken from"}, y={"label": "Elo rating against the founders"},
              series=series_, guides=[{"axis": "y", "at": 0, "label": "the founders"}],
              caption=("**How much better than the founders, over 2,000 iterations.** Thin lines are "
                       "single worlds, thick ones the median world of each condition."),
              recipe=recipe(E16_RUNS, ARCHIVE.replace("every 50", "every 100"), [
                  f"For every pair of the eras {', '.join(str(e) for e in E16_ERAS)} play two contests, "
                  "one in each era's world (`gol_tournament.contest`), and fit one rating per era "
                  "(`gol_tournament.ratings`)."]))
    late = {c: [score(p, 500, 2000) for p in played[c]] for c in E16_CONDITIONS}
    ch.number("late_vs_500", {c: describe(late[c]) for c in E16_CONDITIONS})
    ch.number("champions", {c: [len(tour.dominance(p["eras"], p["contests"])) for p in played[c]]
                            for c in E16_CONDITIONS})
