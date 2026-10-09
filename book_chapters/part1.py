# -*- coding: utf-8 -*-
"""
Part I: the illustrations of Chapters 2 to 5 — the starting ring, what one
iteration does, one decision of the brain, and how worlds are measured.
"""
from __future__ import annotations

import numpy as np

import book_figures as F
from book_figures import (FRAMES, STATS_FILE, band_series, chapter, describe, line, recipe, rows, runs_of,
                          runs_text, series, surviving_text, survivors)
from book_chapters.common import BAND_STEPS, BAND_WORDS, BLUE, GREY, YELLOW, baseline


# ---------------------------------------------------------------------------
# Part I · the illustrations
# ---------------------------------------------------------------------------

@chapter
def world(ch: F.Chapter) -> None:
    """Chapter 2: the founders' starting ring of one real run."""
    import networkx as nx
    from GraphOfLifeSimple import new_world
    from gol_config import SimConfig
    spec = runs_of("E02")[0]
    cfg = SimConfig.from_dict(spec.config)
    w = new_world(cfg)
    ids = sorted(w.G.nodes())
    n = len(ids)
    pos = {u: (np.sin(2 * np.pi * i / n), -np.cos(2 * np.pi * i / n)) for i, u in enumerate(ids)}
    index = {u: i for i, u in enumerate(ids)}
    edges = [[index[a], index[b]] for a, b in w.G.edges()]
    shortcuts = sum(1 for a, b in edges if min((a - b) % n, (b - a) % n) > 2)
    ch.network(
        "ring", title="The 100 founders of the world with seed 1",
        nodes={"x": [pos[u][0] for u in ids], "y": [pos[u][1] for u in ids], "group": [0] * n},
        edges=edges, colour={"by": "group", "labels": ["a founder, holding 100 tokens"]}, height=420,
        caption=f"The starting graph of `B1-10000-s001`: 100 founders placed around a circle in the "
                f"order of the ring, each joined to its two nearest neighbours on either side; "
                f"{shortcuts} of the 200 connections were moved to a founder chosen at random, and "
                "cross the circle.",
        recipe=recipe("`B1-10000-s001`.", "its settings",
                      ["Build the graph exactly as the engine does: "
                       "`networkx.watts_strogatz_graph(n=100, k=5, p=0.2, seed=1)` — with k = 5 "
                       "networkx joins every node to ⌊5/2⌋ = 2 neighbours on either side.",
                       "Place node i at angle 2πi / 100 on a circle and draw every connection."]))
    ch.number("ring", {"founders": n, "connections": len(edges), "shortcuts": shortcuts,
                       "clustering": float(nx.transitivity(w.G)),
                       "mean_distance": float(nx.average_shortest_path_length(w.G))})


@chapter
def iteration(ch: F.Chapter) -> None:
    """Chapter 3: what one iteration does, on average, in a settled world."""
    alive = survivors(baseline())
    sums = {k: 0.0 for k in ("births", "orphaned1", "starved1", "starved2", "orphaned2", "start")}
    for s in alive:
        r = rows(s.run_id)
        p1 = {x["iteration"]: x for x in r if x["phase"] == 1}
        p2 = {x["iteration"]: x for x in r if x["phase"] == 2}
        for t in range(500, 3000):
            a, b = p1[t], p2[t]
            sums["start"] += a["nodes_before"]
            sums["births"] += a["births"]
            sums["orphaned1"] += a.get("orphaned") or 0
            sums["starved1"] += a.get("starved") or 0
            sums["starved2"] += b.get("starved") or 0
            sums["orphaned2"] += b.get("orphaned") or 0
    per100 = {k: 100 * v / sums["start"] for k, v in sums.items() if k != "start"}
    cats = ["born", "cut off in reproduction", "starved in reproduction", "starved in the game",
            "cut off in the game"]
    vals = [per100["births"], -per100["orphaned1"], -per100["starved1"], -per100["starved2"], -per100["orphaned2"]]
    k = np.arange(len(cats))
    ch.figure(
        "flows", title="What one iteration does to a settled world", x={"label": "", "categories": cats},
        y={"label": "agents per 100 at the start of the iteration", "min": -4, "max": 4},
        series=[{"label": None, "kind": "bars", "x0": (k - 0.35).tolist(), "x1": (k + 0.35).tolist(),
                 "y": vals, "colour": BLUE}], legend=False, guides=[{"axis": "y", "at": 0}],
        caption="Agents added (above zero) and removed (below), per 100 agents alive at the start "
                "of an iteration, averaged over iterations 500 to 2,999 of the 26 worlds that lived "
                f"to the end, pooled: every iteration of every world counts once. The bars add up to "
                f"{abs(sum(vals)):.2f}: the population hardly changes from one iteration to the next, "
                "while about three in a hundred agents are replaced.",
        recipe=recipe(surviving_text(), STATS_FILE,
                      ["For every iteration 500 ≤ t ≤ 2,999 of every run, take from the row with "
                       "`phase` = 1: `nodes_before`, `births`, `orphaned`, `starved`; from the row "
                       "with `phase` = 2: `starved`, `orphaned`.",
                       "Sum each over all iterations and runs; divide by the sum of `nodes_before` "
                       "of the rows with `phase` = 1, and multiply by 100."]))
    ch.number("flows_per_100", per100)


@chapter
def brain(ch: F.Chapter) -> None:
    """Chapter 4: one decision a brain makes, before and after selection."""
    specs = baseline()
    first, last = [], []
    for s in specs:
        first += F.world_pass(s.run_id)["shares"]["first"]
    for s in survivors(specs):
        last += F.world_pass(s.run_id)["shares"]["last"]
    edges = np.linspace(0, 1, 11)
    h0 = np.histogram(first, bins=edges)[0] / len(first)
    h1 = np.histogram(last, bins=edges)[0] / len(last)
    x0 = edges[:-1]
    ch.figure(
        "child-share", title="What share of its tokens a parent gives its child",
        x={"label": "share of the parent's tokens given to the child", "min": 0, "max": 1},
        y={"label": "share of births", "min": 0},
        series=[{"label": f"founders, iteration 0 ({len(first):,} births, 30 worlds)", "kind": "bars",
                 "x0": x0.tolist(), "x1": (x0 + 0.05).tolist(), "y": h0.tolist(), "colour": GREY},
                {"label": f"iterations 2,950–2,999 ({len(last):,} births, 26 worlds)", "kind": "bars",
                 "x0": (x0 + 0.05).tolist(), "x1": (x0 + 0.1).tolist(), "y": h1.tolist(), "colour": BLUE}],
        caption="For every birth: the tokens the child received divided by the tokens its parent "
                "held just before, in bins of 0.1. Grey: the founders' births in the very first "
                "reproduction phase, decided by brains of random weights. Blue: the births of the "
                "last 50 iterations, decided by brains descended from them through 2,950 "
                "iterations of copying, changing and conquering. Only parents that gave at least "
                "one whole token had a child and are counted.",
        recipe=recipe(runs_text(), FRAMES,
                      ["In the frames with `phase` 1 of iteration 0, and of iterations 2,950 to 2,999, "
                       "read `decisions.births`: each entry has the parent's `tokens_before` and the "
                       "child's `invested`.",
                       "Put `invested / tokens_before` into ten bins of width 0.1 and divide each "
                       "count by the number of births."]))
    ch.number("child_share", {"first": describe(first), "last": describe(last)})


@chapter
def measure(ch: F.Chapter) -> None:
    """Chapter 5: how a band is made from the worlds."""
    specs = baseline()
    thin_lines = []
    for s in specs:
        its, n = F.thin(*series(s.run_id, "nodes"), 25)
        thin_lines.append(line(its, n, None, BLUE, width=0.8, alpha=0.45))
    ch.figure(
        "thirty", title="30 worlds, each a line", x={"label": "iteration"}, y={"label": "agents", "min": 0},
        series=thin_lines, legend=False,
        caption="The number of agents of each of the 30 baseline worlds, every line one world, "
                "averaged over stretches of 25 iterations so that the lines stay legible.",
        recipe=recipe(runs_text(), STATS_FILE,
                      ["Take every run's rows with `phase` = 2 and `nodes`; average them in stretches "
                       "of 25 iterations; draw one line per run."]))
    ch.figure(
        "bands", title="The same 30 worlds, as a band", x={"label": "iteration"}, y={"label": "agents", "min": 0},
        series=[band_series([series(s.run_id, "nodes") for s in specs], "median, middle half, nine in ten", BLUE)],
        caption=f"{BAND_WORDS}.",
        recipe=recipe(runs_text(), STATS_FILE, ["Take every run's rows with `phase` = 2 and `nodes`.",
                                                  *BAND_STEPS]))
    s = specs[0]
    its, n = series(s.run_id, "nodes")
    means = [(a, F.A.window(its, n, a, a + 99)) for a in range(100, 3000, 100)]
    steps_x, steps_y = [], []
    for a, m in means:
        steps_x += [a, a + 100, None]
        steps_y += [m, m, None]
    ch.figure(
        "stretches", title="One world, and its stretches of 100 iterations", x={"label": "iteration"},
        y={"label": "agents", "min": 0},
        series=[line(its, n, "every iteration", BLUE, width=0.7, alpha=0.6),
                {"label": "mean of each stretch of 100", "x": steps_x, "y": steps_y, "colour": YELLOW, "width": 2.5}],
        caption="The world with seed 1: its number of agents after every game (blue), and its mean "
                "over each of the 29 stretches of 100 iterations from iteration 100 to 2,999 "
                "(yellow). Many measurements in this book are means over such stretches.",
        recipe=recipe("`B1-10000-s001`.", STATS_FILE,
                      ["Plot `nodes` of the rows with `phase` = 2.",
                       "For a = 100, 200, …, 2,900, average `nodes` over a ≤ `iteration` ≤ a + 99."]))
