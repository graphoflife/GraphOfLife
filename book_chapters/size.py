# -*- coding: utf-8 -*-
"""
Part IV: Chapter 31 — how does a world's size follow its tokens?
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np

import book_figures as F
import gol_analysis as A
from book_figures import STATS_FILE, chapter, describe, line, lived, mean_over, per_agent, recipe, runs_of, series
from book_chapters.common import BLUE, CYAN, GREEN, GREY, ORANGE, RED, VIOLET, YELLOW

# ---------------------------------------------------------------------------
# Chapter 31 · How does a world's size follow its tokens?
# ---------------------------------------------------------------------------

E08_RUNS = ("The 42 runs of Experiment 8 — B1 at 800 to 409,600 tokens, each size double the one before, "
            "seeds 1 to 3, 600 iterations — in two conditions: the baseline, `B1-800-s001` … "
            "`B1-409600-s003`; and *stopped only when empty* (`extinction_threshold` 0) at 800 to 6,400 "
            "tokens, `B1-800-57b358-s001` … `B1-6400-57b358-s003`. They are made with "
            "`python3 gol_lab.py run E08`; every setting is listed in the box at the top of the chapter.")
WITH_E02 = (" As a check at 10,000 tokens, the 28 baseline runs of Experiment 2 that reached iteration 600 "
            "(`B1-10000-s001` … `-s030`, Chapter 8).")
WINDOW = (500, 599)          # the plan's window: a settled world
LATE = (300, 599)            # the last 300 iterations, for how a world moves


def tokens(spec) -> int:
    return int(spec.lab["world"]["total_tokens"])


def settled(spec, stat: str, phase: int = 2) -> float:
    return mean_over(spec.run_id, stat, *WINDOW, phase)


def per_size(specs, value) -> Dict[int, List[float]]:
    """Every run's value, gathered by the tokens of its world."""
    out: Dict[int, List[float]] = {}
    for s in specs:
        v = value(s)
        if v is not None and np.isfinite(v):
            out.setdefault(tokens(s), []).append(float(v))
    return out


def fit(points: List[Tuple[float, float, float]], seed: int) -> Dict[str, Any]:
    """A power law through (x, y) points grouped by world size, as the experiment's own fit does."""
    x, y, g = (np.array(c, float) for c in zip(*points))
    return A.power_fit(x, y, g, seed=seed)


@chapter
def size(ch: F.Chapter) -> None:
    results = json.load(open(os.path.join(F.BOOK, "results", "E08.json")))
    fits = {name: per["baseline"]["fit"] for name, per in results["scaling"].items()}
    base = runs_of("E08", "baseline")
    empty = runs_of("E08", "stopped only when empty")
    alive = [s for s in base if lived(s) >= 600]
    small = [s for s in empty if tokens(s) <= 1600]
    e02 = [s for s in runs_of("E02") if lived(s) >= 600]
    big = [s for s in alive if tokens(s) >= 3200]
    sizes = sorted({tokens(s) for s in big})

    # ---- agents and connections against tokens -------------------------
    def dots_of(specs, value, colour, label, size_=7):
        pts = [(tokens(s), value(s)) for s in specs]
        pts = [(t, v) for t, v in pts if v is not None and np.isfinite(v)]
        return {"label": label, "x": [p[0] for p in pts], "y": [p[1] for p in pts], "points": True,
                "size": size_, "colour": colour, "alpha": 0.85}

    f = fits["agents"]
    ends = np.array([3200, 409600], float)
    pts = [(tokens(s), settled(s, "nodes")) for s in alive]
    middle = 10 ** np.mean([np.log10(v) - np.log10(t) for t, v in pts if t >= 3200])
    ch.figure(
        "agents", title="Agents against the tokens of the world",
        x={"label": "tokens in the world (logarithmic)", "log": True, "min": 600, "max": 600000},
        y={"label": "agents, mean over iterations 500–599 (logarithmic)", "log": True},
        series=[dots_of(alive, lambda s: settled(s, "nodes"), BLUE, "baseline, one dot per run"),
                dots_of(small, lambda s: settled(s, "nodes"), ORANGE,
                        "stopped only when empty, 800 and 1,600 tokens"),
                dots_of(e02, lambda s: settled(s, "nodes"), GREY, "the 28 worlds of Chapter 8, 10,000 tokens",
                        size_=5),
                line(ends, f["prefactor"] * ends ** f["exponent"],
                     f"power law fitted from 3,200 tokens on: exponent {f['exponent']:.2f}", RED, width=2),
                line(ends, middle * ends, "in proportion (exponent 1)", "#eef4fa", width=1, dash=[5, 4])],
        caption="Every run that lived to iteration 600 is a dot: the tokens of its world, and the number of agents "
                "alive after each game, averaged over iterations 500 to 599 — both on logarithmic axes, on which "
                "a power law, agents = a · tokens^b, is a straight line of slope b "
                "([Logarithmic axes](../notes/logarithmic-axes.md)). Blue: the baseline, three seeds per size "
                "(at 1,600 tokens only one of three lived through its first iteration, and its dot lies under the "
                "orange one of the same world; at 800 none did). Orange: "
                "the same worlds of 800 and 1,600 tokens with the stopping rule switched off. Grey: the "
                "baseline worlds of Chapter 8, a check from another experiment. Red: the straight line "
                f"fitted to the blue dots from 3,200 tokens on, exponent {f['exponent']:.2f} (95% interval "
                f"{f['interval'][0]:.2f} to {f['interval'][1]:.2f}). Dashed: exponent exactly 1.",
        recipe=recipe(E08_RUNS + WITH_E02, STATS_FILE,
                      ["For every run, the mean of `nodes` over the rows with `phase` = 2 and 500 ≤ "
                       "`iteration` ≤ 599; leave out a run that stopped before iteration 600.",
                       "Fit log₁₀(agents) = log₁₀ a + b · log₁₀(tokens) by least squares over the baseline "
                       "runs of 3,200 tokens and more; the interval of b from 10,000 resamplings of the runs "
                       "within each size (`gol_analysis.power_fit`, run by `python3 gol_lab.py analyse E08`)."]))

    e02_agents = [settled(s, "nodes") / 10000 for s in e02]
    predicted = f["prefactor"] * 10000 ** f["exponent"] / 10000
    ch.number("e02_check", {"agents_per_token": describe(e02_agents), "predicted": predicted})

    # ---- per token: the magnifying glass --------------------------------
    panels = []
    for title, value, fname, ylab in (
            ("Agents per token", lambda s: settled(s, "nodes") / tokens(s), "agents", "agents per token"),
            ("Connections per token", lambda s: settled(s, "edges") / tokens(s), "connections",
             "connections per token"),
            ("Connections per agent", lambda s: settled(s, "meanDegree"), "degree", "mean connections per agent")):
        fp = fits[fname]
        by = per_size(big, value)
        xs = np.array(sorted(by), float)
        law = (fp["prefactor"] * xs ** fp["exponent"] / (xs if fname != "degree" else 1))
        panels.append(dict(
            title=title, x={"label": "tokens (logarithmic)", "log": True, "min": 2000, "max": 600000},
            y={"label": ylab, "min": 0},
            series=[dots_of(big, value, BLUE, None),
                    dots_of(e02, value, GREY, None, size_=4),
                    line(xs, [float(np.median(by[t])) for t in sorted(by)], None, "#eef4fa", width=1.2),
                    line(xs, law, None, RED, width=1.5, dash=[5, 4])],
            legend=False))
    ch.grid(
        "per-token", panels, columns=3, title="The same law, seen through a magnifying glass",
        caption="The quantities of the figure above divided by what proportion would scale them by, so that "
                "growth in proportion is a flat line. Blue: every baseline run of 3,200 tokens and more that "
                "lived to iteration 600, its mean over iterations 500 to 599; grey: the 28 worlds of Chapter 8 "
                "at 10,000 tokens; white: the median of the three runs of each size; dashed red: the power law "
                "fitted to the runs, divided by the tokens (for connections per agent, the law fitted to the "
                "connections per agent themselves, whose exponent says by how much they grow with size).",
        recipe=recipe(E08_RUNS + WITH_E02, STATS_FILE,
                      ["For every run, the means of `nodes`, `edges` and `meanDegree` over the rows with "
                       "`phase` = 2 and 500 ≤ `iteration` ≤ 599; divide the first two by the tokens.",
                       "The red lines are the fits of `book/results/E08.json`, made by "
                       "`python3 gol_lab.py analyse E08`."]))
    ch.number("per_token", {
        name: {str(t): describe(v) for t, v in sorted(per_size(big, value).items())}
        for name, value in (("agents", lambda s: settled(s, "nodes") / tokens(s)),
                            ("connections", lambda s: settled(s, "edges") / tokens(s)),
                            ("degree", lambda s: settled(s, "meanDegree")))})

    # ---- a world's life at every size ------------------------------------
    order = sorted(sizes + [10000])
    panels = []
    for k, t in enumerate(order):
        runs = ([s for s in e02 if s.lab["seed"] <= 3] if t == 10000 else [s for s in big if tokens(s) == t])
        lines_ = []
        for s in sorted(runs, key=lambda s: s.lab["seed"]):
            its, n = series(s.run_id, "nodes")
            m = its <= 599
            lines_.append(line(its[m], n[m] / t, f"seed {s.lab['seed']}", [BLUE, YELLOW, GREEN][s.lab["seed"] - 1],
                               width=1))
        panels.append(dict(title=f"{t:,} tokens" + (" (Chapter 8)" if t == 10000 else ""),
                           x={"label": "iteration", "min": 0, "max": 600},
                           y={"label": "agents per token", "min": 0, "max": 0.35},
                           series=lines_, legend=k == 0))
    ch.grid(
        "lives", panels, columns=3, title="Six hundred iterations, at nine sizes",
        caption="The number of agents after every game, divided by the tokens of the world, over the 600 "
                "iterations of every baseline run of 3,200 tokens and more — one panel per size, one line per "
                "seed — and, in the third panel, the first three worlds of Chapter 8 at 10,000 tokens, over "
                "their first 600 iterations. The same vertical scale in every panel.",
        recipe=recipe(E08_RUNS + " The runs of Chapter 8 with seeds 1, 2 and 3.", STATS_FILE,
                      ["Take `nodes` of every row with `phase` = 2 and `iteration` ≤ 599, divided by the "
                       "tokens of the world."]))

    # ---- how a world moves, at every size --------------------------------
    moves, cuts_mean, cuts_max = [], [], []
    for s in big + e02:
        its, n = series(s.run_id, "nodes")
        m = (its >= LATE[0]) & (its <= LATE[1])
        x = n[m]
        moves.append((float(x.mean()), float(np.std(np.diff(x)) / x.mean()), tokens(s)))
        io, o = series(s.run_id, "orphaned")
        _, before = series(s.run_id, "nodes_before")
        mo = (io >= LATE[0]) & (io <= LATE[1]) & (before > 0)
        share = o[mo] / before[mo]
        cuts_mean.append((tokens(s), float(share.mean())))
        cuts_max.append((tokens(s), float(share.max())))
    move_fit = fit([(n, r, t) for n, r, t in moves], seed=31)
    small_n = [m for m in moves if m[2] == 3200]
    n0 = float(np.exp(np.mean([np.log(m[0]) for m in small_n])))
    r0 = float(np.exp(np.mean([np.log(m[1]) for m in small_n])))
    ns = np.array([n0, 70000.0])
    ch.grid(
        "fluctuations", [
            dict(title="Change from one game to the next",
                 x={"label": "agents (logarithmic)", "log": True},
                 y={"label": "sd of the change ÷ agents (logarithmic)", "log": True},
                 series=[{"label": "one run", "x": [m[0] for m in moves], "y": [m[1] for m in moves],
                          "points": True, "size": 6, "colour": BLUE, "alpha": 0.85},
                         line(ns, move_fit["prefactor"] * ns ** move_fit["exponent"],
                              f"fitted: slope {move_fit['exponent']:.2f}", RED, width=1.5),
                         line(ns, r0 * (ns / n0) ** -0.5, "if a world were independent parts: slope −0.5",
                              "#eef4fa", width=1, dash=[5, 4])]),
            dict(title="Share cut off in one game",
                 x={"label": "tokens (logarithmic)", "log": True, "min": 2000, "max": 600000},
                 y={"label": "share of the agents (logarithmic)", "log": True, "min": 0.003, "max": 1},
                 series=[{"label": "mean over the games", "x": [c[0] for c in cuts_mean],
                          "y": [c[1] for c in cuts_mean], "points": True, "size": 6, "colour": YELLOW},
                         {"label": "the largest single game", "x": [c[0] for c in cuts_max],
                          "y": [c[1] for c in cuts_max], "points": True, "size": 6, "colour": RED}])],
        columns=2, title="Big worlds do not average out",
        caption="Over the last 300 iterations (300 to 599) of every baseline run of 3,200 tokens and more, and "
                "of the 28 worlds of Chapter 8. Left: for each run, the standard deviation of the change in the "
                "number of agents from one game to the next, divided by the mean number of agents — against "
                "that mean, on logarithmic axes. If a world were made of independent parts, its relative "
                "changes would shrink as one over the square root of its size (dashed, drawn through the runs "
                "of 3,200 tokens); red is the line fitted to the runs. Right: for each run, the share of the "
                "agents alive at the start of a game that the game cut off from the network — the mean over "
                "the games (yellow) and the largest in any one game (red).",
        recipe=recipe(E08_RUNS + WITH_E02, STATS_FILE,
                      ["Take `nodes` of the rows with `phase` = 2 and 300 ≤ `iteration` ≤ 599; the standard "
                       "deviation of the differences between consecutive rows, divided by the mean.",
                       "Fit the logarithm of that against the logarithm of the mean by least squares; the "
                       "interval from resampling the runs within each size (`gol_analysis.power_fit`).",
                       "For the right panel, `orphaned` / `nodes_before` of the same rows: the mean and the "
                       "largest."]))
    by_mean = {}
    for t, v in cuts_mean:
        by_mean.setdefault(t, []).append(v)
    by_max = {}
    for t, v in cuts_max:
        by_max.setdefault(t, []).append(v)
    ch.number("fluctuations", {"fit": move_fit,
                               "relative_change": {str(t): describe([m[1] for m in moves if m[2] == t])
                                                   for t in sorted({m[2] for m in moves})}})
    ch.number("cuts", {"mean_share": {str(t): describe(v) for t, v in sorted(by_mean.items())},
                       "largest_share": {str(t): describe(v) for t, v in sorted(by_max.items())}})

    # ---- the same world at every size? -----------------------------------
    def ratio(stat, per_stat):
        return lambda s: (settled(s, stat) / settled(s, per_stat)) if settled(s, per_stat) else None

    def rate(stat, phase):
        def value(s):
            its, r = per_agent(s.run_id, stat, phase)
            m = (its >= WINDOW[0]) & (its <= WINDOW[1])
            return float(np.nanmean(r[m]))
        return value

    intensive = [
        ("Inequality (Gini)", lambda s: settled(s, "gini"), "gini"),
        ("Share of stakes at home", lambda s: settled(s, "selfAllocationShare"), "home"),
        ("Nodes kept by their own agent", lambda s: settled(s, "heldHomeShare"), "kept"),
        ("Genotypes per agent", lambda s: settled(s, "brainDiversity"), "genotypes"),
        ("Births per 100 agents", rate("births", 1), "births"),
        ("Cut off per 100 agents", rate("orphaned", 2), "cut"),
        ("Leaves, share of agents", ratio("leaves", "nodes"), "leaves"),
        ("Loops per connection", lambda s: settled(s, "loopDensity"), "loops"),
        ("Dimension from ball growth", lambda s: settled(s, "dimension"), "dimension")]
    panels, flat = [], {}
    for k, (title, value, key) in enumerate(intensive):
        by = per_size(big, value)
        pts = [(t, v, t) for t, vs in by.items() for v in vs if v > 0]
        flat[key] = {"fit": fit(pts, seed=310 + k), "bySize": {str(t): describe(v) for t, v in sorted(by.items())},
                     "chapter8": describe([value(s) for s in e02])}
        panels.append(dict(
            title=title, x={"label": "tokens (logarithmic)", "log": True, "min": 2000, "max": 600000},
            y={"label": "", "min": 0},
            series=[dots_of(big, value, BLUE, None, size_=6), dots_of(e02, value, GREY, None, size_=4),
                    line(np.array(sorted(by), float), [float(np.median(by[t])) for t in sorted(by)], None,
                         "#eef4fa", width=1.2)],
            legend=False))
    ch.grid(
        "same-world", panels, columns=3, title="Is a big world the same kind of world?",
        caption="Nine properties of a world that do not grow with its size by definition — shares, rates per "
                "agent, a dimension. Blue: every baseline run of 3,200 tokens and more that lived to iteration "
                "600, its mean over iterations 500 to 599; grey: the 28 worlds of Chapter 8 at 10,000 tokens; "
                "white: the median of each size. A world that is the same kind of world at every size gives a "
                "flat cloud.",
        recipe=recipe(E08_RUNS + WITH_E02, STATS_FILE,
                      ["For every run, the mean over the rows with `phase` = 2 and 500 ≤ `iteration` ≤ 599 of "
                       "`gini`, `selfAllocationShare`, `heldHomeShare`, `brainDiversity`, `leaves` / `nodes`, "
                       "`loopDensity` and `dimension` (the last two every 25 iterations).",
                       "Births: `births` of the rows with `phase` = 1, per 100 `nodes_before`; cut off: "
                       "`orphaned` of the rows with `phase` = 2, per 100 `nodes_before`; each averaged over "
                       "iterations 500 to 599.",
                       "For the numbers in the text: a power law of each property against the tokens, fitted "
                       "as for the agents — an exponent near 0 is a property that does not change with size."]))
    ch.number("same_world", flat)

    # ---- distances --------------------------------------------------------
    dist = {}
    panels = []
    for k, (stat, title, ylab) in enumerate((("meanPathLength", "Mean distance between two agents", "steps"),
                                             ("diameter", "The two agents furthest apart", "steps"))):
        pts = [(settled(s, "nodes"), settled(s, stat), tokens(s)) for s in big + e02]
        pts = [p for p in pts if p[1]]
        fd = fit(pts, seed=320 + k)
        dist[stat] = {"fit": fd, "bySize": {str(t): describe([p[1] for p in pts if p[2] == t])
                                           for t in sorted({p[2] for p in pts})}}
        x0 = [p for p in pts if p[2] == 3200]
        n0 = float(np.exp(np.mean([np.log(p[0]) for p in x0])))
        d0 = float(np.exp(np.mean([np.log(p[1]) for p in x0])))
        ns = np.geomspace(n0, 70000, 30)
        panels.append(dict(
            title=title, x={"label": "agents (logarithmic)", "log": True},
            y={"label": f"{ylab} (logarithmic)", "log": True},
            series=[{"label": "one run", "x": [p[0] for p in pts], "y": [p[1] for p in pts], "points": True,
                     "size": 6, "colour": BLUE, "alpha": 0.85},
                    line(ns, fd["prefactor"] * ns ** fd["exponent"], f"fitted: agents^{fd['exponent']:.2f}", RED,
                         width=1.5),
                    line(ns, d0 * np.log(ns) / np.log(n0), "growing as ln(agents), a small world", "#eef4fa",
                         width=1, dash=[5, 4])],
            legend=True))
    ch.grid(
        "distances", panels, columns=2, title="How far apart agents are, as worlds grow",
        caption="Every baseline run of 3,200 tokens and more that lived to iteration 600, and the 28 worlds of "
                "Chapter 8: the mean number of agents over iterations 500 to 599, and over the same iterations "
                "the mean distance between two agents (left) and the largest distance found (right), both "
                "estimated by the viewer from breadth-first searches out of 8 to 16 agents every 25 iterations "
                "([Radius and diameter](../notes/radius-and-diameter.md)). Red: a power law fitted to the "
                "dots. Dashed: what a small world would do — distances growing in proportion to the logarithm "
                "of the number of agents — drawn through the runs of 3,200 tokens.",
        recipe=recipe(E08_RUNS + WITH_E02, STATS_FILE,
                      ["For every run, the means of `nodes`, `meanPathLength` and `diameter` over the rows with "
                       "`phase` = 2 and 500 ≤ `iteration` ≤ 599 (the last two are filled every 25 iterations).",
                       "Fit log(distance) against log(agents) by least squares; the interval from resampling "
                       "the runs within each size."]))
    ch.number("distances", dist)

    # ---- below twenty agents ---------------------------------------------
    lines_, below = [], {}
    shades = {800: [ORANGE, RED, YELLOW], 1600: [CYAN, VIOLET, GREEN]}
    for s in sorted(small, key=lambda s: (tokens(s), s.lab["seed"])):
        its, n = series(s.run_id, "nodes")
        lines_.append(line(its, n, f"{tokens(s):,} tokens, seed {s.lab['seed']}",
                           shades[tokens(s)][s.lab["seed"] - 1], width=1.2))
        below[f"{tokens(s)}-{s.lab['seed']}"] = {
            "least": int(n.min()), "at": int(its[np.argmin(n)]), "iterations_at_most_20": int((n <= 20).sum()),
            "settled": float(settled(s, "nodes")), "baseline_stopped": lived(next(
                b for b in base if tokens(b) == tokens(s) and b.lab["seed"] == s.lab["seed"])) < 600}
    ch.figure(
        "small", title="Worlds below twenty agents",
        x={"label": "iteration", "min": 0, "max": 600},
        y={"label": "agents (logarithmic)", "log": True, "min": 2, "max": 1000},
        series=lines_, guides=[{"axis": "y", "at": 20, "label": "where the baseline stops a world"}],
        caption="The six worlds of 800 and 1,600 tokens with the stopping rule switched off "
                "(`extinction_threshold` 0): the number of agents after every game, on a logarithmic axis. "
                "Five of these six worlds — all three of 800 tokens, and seeds 1 and 3 of 1,600 — were stopped as "
                "extinct by the baseline after their first game, at the dashed line or below it.",
        recipe=recipe("The runs `B1-800-57b358-s001` … `-s003` and `B1-1600-57b358-s001` … `-s003` of "
                      "Experiment 8 (`python3 gol_lab.py run E08`).", STATS_FILE,
                      ["Take `nodes` of every row with `phase` = 2."]))
    ch.number("small", below)
    ch.number("fits", fits)
