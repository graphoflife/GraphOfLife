# -*- coding: utf-8 -*-
"""
the study of the baseline, the ages of a world: how old its agents are, how long they live, and
how both change over a world's life.

A frame lists every living agent's age — the iterations since it was born —
so reading a world's frames after every game gives its whole age structure,
iteration by iteration, and every agent's life: born at the iteration of its
first frame minus its age there, alive until the last game after which it is
still listed. The measure keeps both beside the runs.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

import book_data as D
import book_figures as F
from book_figures import Classes, band_series, chapter, describe, line, quantile_band, recipe, stacked
from book_chapters.common import (BAND_STEPS, BAND_WORDS, BLUE, CYAN, GREEN, GREY, ORANGE, RED, VIOLET,
                                  WORLD_SETS, WORLDS as W, YELLOW, count_word, listed)

#: Ages, in classes doubling in width from 4 on.
AGE_CLASSES = Classes((0, 0, "0"), (1, 3, "1–3"), (4, 15, "4–15"), (16, 63, "16–63"), (64, 255, "64–255"),
                      (256, 10 ** 9, "256+"))
#: How often the wealth and connections of each age class are read.
SNAPSHOT_EVERY = 25
#: The columns of `by_iteration`.
AGE_COLUMNS = ("t", "agents", "mean", "q25", "median", "q75", "q90", "q99", "oldest", "newborn", "from_start")


@D.measure("lifelines", file="{run}.lifelines.npz")
def lifelines(run_id: str) -> Dict[str, Any]:
    """
    Every agent's life, and the ages of the living after every game.

    by_iteration  one row per iteration, the columns of AGE_COLUMNS: the agents
                  alive after its game; their mean age, quartiles, 90th and 99th
                  percentiles and the oldest; the share born in that iteration
                  (age 0); and how many were born in iteration 0 — the founders
                  and their first children — and are alive still.
    birth, last   for every agent alive after at least one game: the iteration
                  it was born and the last iteration after whose game it was
                  alive; `censored` marks those alive after the run's last game.
    by_class      every SNAPSHOT_EVERY iterations, for each class of
                  AGE_CLASSES: t, then agents, mean tokens and mean connections
                  of each class.
    """
    last = D.last_iteration(run_id)
    born: Dict[int, int] = {}
    seen: Dict[int, int] = {}
    rows, classes = [], []
    for t in range(last + 1):
        frame = D.frame_at(run_id, t, 2)
        ids = frame["ids"]
        ages = np.asarray(frame["ages"], dtype=np.int64)
        for u, a in zip(ids, ages.tolist()):
            if u not in born:
                born[u] = t - a
            seen[u] = t
        q = np.quantile(ages, [0.25, 0.5, 0.75, 0.9, 0.99])
        rows.append([t, len(ids), float(ages.mean()), *q.tolist(), int(ages.max()), float(np.mean(ages == 0)),
                     int(np.sum(ages == t))])
        if t % SNAPSHOT_EVERY == 0:
            index = {u: i for i, u in enumerate(ids)}
            edges = np.asarray(frame["edges"], dtype=np.int64).reshape(-1, 2)
            ends = np.array([index[u] for u in edges.ravel()], dtype=np.int64)
            degree = np.bincount(ends, minlength=len(ids))
            tokens = np.asarray(frame["tokens"], dtype=float)
            row = [t]
            for m in AGE_CLASSES.masks(ages):
                row += [int(m.sum()), float(tokens[m].mean()) if m.any() else np.nan,
                        float(degree[m].mean()) if m.any() else np.nan]
            classes.append(row)
    agents = np.fromiter(born, dtype=np.int64, count=len(born))
    birth = np.array([born[u] for u in agents.tolist()], dtype=np.int32)
    end = np.array([seen[u] for u in agents.tolist()], dtype=np.int32)
    return {"by_iteration": np.array(rows, dtype=float), "birth": birth, "last": end,
            "censored": end == last, "by_class": np.array(classes, dtype=float)}


def survival(birth: np.ndarray, last: np.ndarray, censored: np.ndarray, horizon: int) -> np.ndarray:
    """
    S[L], the share of lives that last at least L iterations, L = 0 … horizon, by
    Kaplan and Meier's estimate (as book_figures.kaplan_meier, for many lives at
    once): a life lasts last − birth + 1; one still going is at risk until then.
    """
    durations = (last - birth + 1).astype(np.int64)
    ended = ~censored.astype(bool)
    size = max(horizon + 2, int(durations.max()) + 2) if durations.size else horizon + 2
    deaths = np.bincount(durations[ended], minlength=size).astype(float)
    leaving = np.bincount(durations, minlength=size).astype(float)
    at_risk = durations.size - np.concatenate([[0.0], np.cumsum(leaving)[:-1]])
    step = np.where(at_risk > 0, 1 - deaths / np.maximum(at_risk, 1), 1.0)
    S = np.concatenate([[1.0], np.cumprod(step)])        # S[L] = Π over l < L of the step at l
    return S[:horizon + 1]


def hazard(birth: np.ndarray, last: np.ndarray, censored: np.ndarray, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """For each age a = 1 … horizon: the lives still going at a, and the share of them that end there."""
    durations = (last - birth + 1).astype(np.int64)
    ended = ~censored.astype(bool)
    size = max(horizon + 2, int(durations.max()) + 2) if durations.size else horizon + 2
    deaths = np.bincount(durations[ended], minlength=size).astype(float)
    at_risk = durations.size - np.concatenate([[0.0], np.cumsum(np.bincount(durations, minlength=size))[:-1]])
    a = np.arange(1, horizon + 1)
    return at_risk[a], np.where(at_risk[a] > 0, deaths[a] / np.maximum(at_risk[a], 1), np.nan)


# ---------------------------------------------------------------------------
# The page: how ages and lifespans change over a world's life
# ---------------------------------------------------------------------------

LIFELINES = ("each run's frames after every game (frame `2·t + 1`, `gol_store.read_frame`), read in order "
             "by `book_chapters.ages.lifelines`: every agent's `ages` entry gives its birth iteration "
             "(the frame's iteration minus its age), and the last frame it is in its last iteration alive")
#: Birth periods whose lives are followed: in iterations, from–to.
COHORTS = {"E02": ((1, 49), (50, 99), (100, 249), (250, 499), (500, 999), (1000, 1999), (2000, 2899)),
           "E10": ((1, 49), (50, 99), (100, 249), (250, 499), (500, 749), (750, 999), (1000, 1399))}
#: Lives are compared up to this age, so that every cohort born before last − RESTRICTED counts in full.
RESTRICTED = 100


def median_life(S: np.ndarray) -> Optional[int]:
    """The median lifetime: the longest L that more than half the lives reach, S(L) > ½ ≥ S(L + 1)."""
    below = S <= 0.5
    return int(np.argmax(below)) - 1 if below.any() else None


def stationary(S: np.ndarray) -> Dict[str, float]:
    """
    The ages the living would have in a world whose births never change and
    whose lives all follow S: as many agents of age a as lives lasting at least
    a + 1 iterations, so a share S(a + 1) / Σ S of the living has age a — their
    median and 90th percentile, the mean life Σ S, and the newborns' share
    1 / Σ S.
    """
    weights = S[1:] / S[1:].sum()
    cum = np.cumsum(weights)
    return {"median_age": int(np.argmax(cum >= 0.5)), "q90_age": int(np.argmax(cum >= 0.9)),
            "mean_life": float(S[1:].sum()), "newborn_share": float(weights[0])}


#: Lives are followed this far when a world's ages are rebuilt from its survival curve.
STATIONARY = 999


def settled_lives(worlds: Any, kept: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The lives begun in the settled stretch of a set of worlds, pooled — born from
    its first iteration to RESTRICTED iterations before the end, so that every
    one of them can be followed at least that far.
    """
    kept = list(kept)
    b = np.concatenate([m["birth"] for m in kept])
    d = np.concatenate([m["last"] for m in kept])
    c = np.concatenate([m["censored"] for m in kept])
    keep = (b >= worlds.settled) & (b <= worlds.last - RESTRICTED)
    return b[keep], d[keep], c[keep]


def _column(m: Dict[str, Any], name: str) -> Tuple[np.ndarray, np.ndarray]:
    table = m["by_iteration"]
    return table[:, 0], table[:, AGE_COLUMNS.index(name)]


def _lives(m: Dict[str, Any], lo: int, hi: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep = (m["birth"] >= lo) & (m["birth"] <= hi)
    return m["birth"][keep], m["last"][keep], m["censored"][keep]


@chapter
def ages(ch: F.Chapter) -> None:
    specs, alive = W.specs, W.alive
    kept = {s.run_id: lifelines(s.run_id) for s in specs}
    n = len(specs)

    # ---- the age structure, iteration by iteration ----
    panels = []
    for title, parts in (("Median age, and the age nine in ten are below", [("median", "median age", BLUE),
                                                                            ("q90", "90th percentile", CYAN)]),
                         ("The oldest agent", [("oldest", "the oldest agent", ORANGE)])):
        series_ = [band_series([_column(m, c) for m in kept.values()], label, colour) for c, label, colour in parts]
        if title.startswith("The oldest"):
            series_.append(line([0, W.last], [0, W.last], "an agent alive since iteration 0", GREY, width=1,
                                dash=[4, 4]))
        panels.append(dict(title=title, x={"label": "iteration"}, y={"label": "age, iterations", "min": 0},
                           series=series_))
    ch.grid(
        "over-time", panels, columns=2, title="How old the living are",
        caption=f"After every game of the {n} worlds. Left: the median age of the agents alive, and the age that "
                "nine in ten of them are below (the 90th percentile). Right: the age of the oldest agent alive; "
                "the dashed diagonal is the age of an agent alive since iteration 0, which only the oldest can "
                f"follow. For each, {BAND_WORDS[0].lower()}{BAND_WORDS[1:]}.",
        recipe=recipe(W.runs_text(), LIFELINES,
                      ["After every game, the median, the 90th percentile and the largest of `ages`.", *BAND_STEPS]))
    for c in ("mean", "median", "q90", "q99", "oldest"):
        ch.number(f"{c}_settled", describe([float(np.mean(_column(m, c)[1][W.settled:])) for m in kept.values()]))
        ch.number(f"{c}_end", describe([float(_column(m, c)[1][W.last]) for m in kept.values()]))
    at = [t for t in (50, 100, 200, 300, 500, 750, 1000, 1250, 2000) if t < W.last] + [W.last]
    for c in ("median", "q90", "oldest"):
        ch.number(f"{c}_at", {str(t): describe([float(_column(m, c)[1][t]) for m in kept.values()]) for t in at})
    for t in (25, 50, 100, 200, 300, 400, 500):
        ch.number(f"from_start_{t}", describe([float(_column(m, "from_start")[1][t]) for m in kept.values()]))
    ch.number("last_from_start", describe([float(max((t for t, v in zip(*_column(m, "from_start")) if v > 0),
                                                     default=0)) for m in kept.values()]))

    # ---- the age pyramid at a few moments ----
    snaps = np.array(list(kept.values())[0]["by_class"][:, 0], int)
    moments = [t for t in (100, 250, 500, 750, 1000, 2000) if t < W.last and t in snaps] + [int(snaps[-1])]
    cats = [f"iteration {t:,}" for t in moments]
    labels = AGE_CLASSES.labels
    shares = []
    for t in moments:
        counts = np.zeros(len(labels))
        for m in kept.values():
            row = m["by_class"][m["by_class"][:, 0] == t]
            if len(row):
                counts += row[0, 1::3]
        shares.append(counts / counts.sum())
    shares = np.array(shares)
    ch.figure(
        "pyramids", title=f"The living, by age, at {count_word(len(moments))} moments",
        x={"label": "", "categories": cats}, y={"label": "share of the living", "min": 0, "max": 1},
        series=stacked(cats, [(f"age {lab}", shares[:, i], c) for i, (lab, c) in
                              enumerate(zip(labels, [RED, ORANGE, YELLOW, GREEN, CYAN, BLUE]))]),
        caption=f"All agents alive after the games of {listed(moments)}, pooled over the {n} worlds and sorted "
                "by age in classes that widen fourfold: born in that very iteration (age 0), 1 to 3, 4 to 15, "
                "16 to 63, 64 to 255, and 256 iterations or more. Each column adds up to 1.",
        recipe=recipe(W.runs_text(), LIFELINES,
                      [f"Every {SNAPSHOT_EVERY} iterations, count the agents of each age class; pool the worlds.",
                       "Divide each class by the agents of the moment and stack."]))
    ch.number("pyramids", {str(t): dict(zip(labels, row.tolist())) for t, row in zip(moments, shares)})

    # ---- lives by the period they began in ----
    horizon = W.last
    cohorts = COHORTS[W.experiment]
    pooled = {k: np.concatenate([m[k] for m in kept.values()]) for k in ("birth", "last", "censored")}
    curves, said = [], {}
    colours = [RED, ORANGE, YELLOW, GREEN, CYAN, BLUE, VIOLET]
    for (lo, hi), colour in zip(cohorts, colours):
        b, d, c = _lives(pooled, lo, hi)
        S = survival(b, d, c, horizon)
        reach = W.last - lo + 1
        L = np.arange(1, reach + 1)
        keep = S[1:reach + 1] > 0
        curves.append(line(L[keep], S[1:reach + 1][keep], f"born in iterations {lo:,}–{hi:,} ({b.size:,})",
                           colour, width=2))
        said[f"{lo}-{hi}"] = {"lives": int(b.size), "median": median_life(S),
                              **{f"at_least_{x}": float(S[x]) for x in (2, 5, 10, 50, 100, 500) if x <= reach}}
    ch.figure(
        "cohorts", title="How long agents live, by when they were born",
        x={"label": "lifetime L, in iterations (logarithmic)", "log": True, "min": 1},
        y={"label": "share living at least L (logarithmic)", "log": True, "min": 1e-5, "max": 1},
        series=curves,
        caption=f"Of all agents born in each period of the {n} worlds' lives that were alive after at least one "
                "game, the share that lived at least L iterations. A life is counted from the iteration of birth to the "
                "last iteration after whose game the agent was alive, both included; lives still going at the end "
                "of a run are counted as far as they went (Kaplan–Meier, as in [Births, deaths and ages](births-deaths-and-ages.md)). A period's line stops "
                "at the longest life its agents could have had before the run ended. Both axes logarithmic.",
        recipe=recipe(W.runs_text(), LIFELINES,
                      ["For every agent, its birth iteration b and its last iteration alive d; d − b + 1 iterations "
                       "lived; alive after the run's last game: still going.",
                       "Sort the agents by b into the periods shown, pool the worlds, and estimate S(L) by "
                       "Kaplan–Meier (`book_chapters.ages.survival`)."]))
    ch.number("cohorts", said)

    # ---- life expectancy, birth period by birth period ----
    width = 25
    starts = list(range(0, W.until - RESTRICTED, width))
    table = np.full((len(kept), len(starts)), np.nan)
    medians = np.full((len(kept), len(starts)), np.nan)
    for w, m in enumerate(kept.values()):
        for j, lo in enumerate(starts):
            b, d, c = _lives(m, max(lo, 1), lo + width - 1)
            if b.size >= 20:
                S = survival(b, d, c, RESTRICTED)
                table[w, j] = float(S[1:RESTRICTED + 1].sum())
                medians[w, j] = np.nan if median_life(S) is None else median_life(S)
    mids = [lo + width / 2 for lo in starts]
    ch.grid(
        "expectancy", [
            dict(title=f"Iterations lived, of the first {RESTRICTED}", x={"label": "birth iteration"},
                 y={"label": "iterations", "min": 0},
                 series=[{"label": None, **quantile_band(table, mids), "colour": GREEN}], legend=False),
            dict(title="Median lifetime", x={"label": "birth iteration"}, y={"label": "iterations", "min": 0},
                 series=[{"label": None, **quantile_band(medians, mids), "colour": BLUE}], legend=False)],
        columns=2, title="Life expectancy over a world's life",
        caption=f"Agents born in each stretch of {width} iterations (those born at iteration 0, the founders and "
                "their first children, left out). Left: how many of their first "
                f"{RESTRICTED} iterations they live on average — the sum of the Kaplan–Meier curve up to "
                f"L = {RESTRICTED}, a mean that every stretch born at least {RESTRICTED} iterations before the end "
                "can be measured by in full. Right: the median lifetime, the longest L that more than half of them live. "
                f"For each world separately; the line is the median of the {n} worlds, the darker band the middle "
                "half and the paler band nine in ten.",
        recipe=recipe(W.runs_text(), LIFELINES,
                      [f"For each world and stretch of {width} birth iterations with at least 20 births, the "
                       "Kaplan–Meier curve S(L) of their lives.",
                       f"Left: Σ S(L) for L = 1 … {RESTRICTED}. Right: the largest L with S(L) > ½."]))
    ch.number("expectancy", {"starts": starts, "restricted": [describe(table[:, j]) for j in range(len(starts))],
                             "median": [describe(medians[:, j]) for j in range(len(starts))]})

    # ---- the risk of dying, by age ----
    eras = [(1, 249, "born in iterations 1–249", ORANGE), (W.settled, W.last - 100, f"born from {W.settled} on", BLUE)]
    hz, out = [], {}
    for lo, hi, label, colour in eras:
        b, d, c = _lives(pooled, lo, hi)
        risk, h = hazard(b, d, c, 200)
        ages_ = np.arange(1, 201)
        keep = (risk >= 200) & (h > 0)
        hz.append(line(ages_[keep], h[keep], label, colour, width=2))
        out[label] = {str(a): float(h[a - 1]) for a in (1, 2, 3, 5, 10, 20, 50, 100, 200) if risk[a - 1] >= 200}
    ch.figure(
        "hazard", title="The risk of dying, by age",
        x={"label": "age, iterations (logarithmic)", "log": True, "min": 1},
        y={"label": "share of those alive at that age who die in the next game (logarithmic)", "log": True},
        series=hz,
        caption="Of the agents alive after the game at which they were a given age, the share that is gone after "
                "the next game — the agent's chance of dying at that age, pooled over the worlds, for agents born "
                f"early in a world's life (orange) and in its settled life (blue). An age is drawn only where at "
                "least 200 agents reached it.",
        recipe=recipe(W.runs_text(), LIFELINES,
                      ["For agents born in the period, with d − b + 1 iterations lived: at each L, the lives still "
                       "going at L (lasting at least L) and those ending at L (lasting exactly L, not still going "
                       "at the end).",
                       "Divide the second by the first (`book_chapters.ages.hazard`)."]))
    ch.number("hazard", out)

    # ---- who the old are: tokens and connections by age ----
    panels = []
    for k, (what, title) in enumerate((("tokens", "Mean tokens"), ("degree", "Mean connections"))):
        series_ = []
        for i, (lab, colour) in enumerate(zip(labels, [RED, ORANGE, YELLOW, GREEN, CYAN, BLUE])):
            col = 2 + 3 * i + k
            runs = [(m["by_class"][:, 0], m["by_class"][:, col]) for m in kept.values()]
            b = F.A.bands(runs)
            series_.append(line(b["x"], b["y"], f"age {lab}", colour, width=1.8))
        panels.append(dict(title=title + " of each age", x={"label": "iteration"},
                           y={"label": what if what == "tokens" else "connections", "min": 0, "log": False},
                           series=series_))
    ch.grid(
        "wealth-age", panels, columns=2, title="Who the old are",
        caption=f"Every {SNAPSHOT_EVERY} iterations, the agents alive after the game in each age class: their mean "
                f"tokens (left) and mean number of connections (right); each line is the median over the {n} "
                "worlds of the class's mean.",
        recipe=recipe(W.runs_text(), LIFELINES,
                      [f"Every {SNAPSHOT_EVERY} iterations, for each age class, the mean of `tokens` and of the "
                       "connections each agent has in `edges`.", "Median over the worlds."]))
    ch.number("by_class_end", {lab: {"tokens": describe([float(m["by_class"][-1, 2 + 3 * i]) for m in kept.values()]),
                                     "degree": describe([float(m["by_class"][-1, 3 + 3 * i]) for m in kept.values()])}
                               for i, lab in enumerate(labels)})

    # ---- small worlds, long lives: the worlds of 10,000 tokens beside these ----
    other = WORLD_SETS["E02"] if W.experiment != "E02" else None
    if other is not None:
        small = {s.run_id: lifelines(s.run_id) for s in other.specs}
        panels = []
        for c, title in (("median", "Median age"), ("oldest", "The oldest agent")):
            panels.append(dict(
                title=title, x={"label": "iteration"}, y={"label": "age, iterations", "min": 0},
                series=[band_series([_column(m, c) for m in kept.values()], f"{W.tokens:,} tokens ({n} worlds)", BLUE),
                        band_series([_column(m, c) for m in small.values()],
                                    f"{other.tokens:,} tokens ({len(other.specs)} worlds)", YELLOW)]))
        ch.grid(
            "two-sizes", panels, columns=2, title="Ages in worlds of two sizes",
            caption=f"The median age and the oldest agent after every game, in the {n} worlds of {W.tokens:,} tokens "
                    f"(blue, {W.until:,} iterations) and in the {len(other.specs)} worlds of {other.tokens:,} tokens "
                    f"of the first edition, Experiment 2 (yellow, {other.until:,} iterations). {BAND_WORDS}; a world that "
                    "died is no longer counted.",
            recipe=recipe(W.runs_text() + " " + other.runs_text(), LIFELINES,
                          ["As in the first figure, for both sets of worlds.", *BAND_STEPS[1:]]))
        for c in ("median", "oldest"):
            ch.number(f"small_{c}", {str(t): describe([float(_column(m, c)[1][t]) for m in small.values()
                                                       if len(m["by_iteration"]) > t])
                                     for t in (500, 1000, 1499, 2000, 2999)})
        ch.number("small_median_settled", describe([float(np.mean(_column(m, "median")[1][other.settled:]))
                                                    for s, m in small.items() if s in {x.run_id for x in other.alive}]))
        b, d, c = settled_lives(other, small.values())
        S = survival(b, d, c, 1000)
        ch.number("small_survival", {f"at_least_{x}": float(S[x]) for x in (2, 5, 10, 50, 100, 500, 1000)})
        ch.number("small_stationary", stationary(S[:STATIONARY + 1]))
        risk, h = hazard(b, d, c, 500)
        ch.number("small_hazard", {str(a): float(h[a - 1]) for a in (1, 2, 3, 5, 10, 20, 50, 100, 200, 300, 500)
                                   if risk[a - 1] >= 200})
    # Whether births hold steady, which the ages of a steady world assume: births per agent, stretch by stretch.
    for worlds, key in ((W, "births_rate"), (other, "small_births_rate")):
        if worlds is not None:
            stretches = [(a, min(a + 249, worlds.last)) for a in range(0, worlds.until, 250)]
            ch.number(key, {f"{a}-{b}": describe([F.mean_over(s.run_id, "births/nodes_before", a, b, 1)
                                                   for s in worlds.alive]) for a, b in stretches})
    b, d, c = settled_lives(W, kept.values())
    S = survival(b, d, c, STATIONARY)
    ch.number("survival", {"lives": int(b.size), **{f"at_least_{x}": float(S[x]) for x in (2, 5, 10, 50, 100, 500)}})
    ch.number("stationary", stationary(S))
    risk, h = hazard(b, d, c, 500)
    ch.number("settled_hazard", {str(a): float(h[a - 1]) for a in (1, 2, 3, 5, 10, 20, 50, 100, 200, 300, 500)
                                 if risk[a - 1] >= 200})
