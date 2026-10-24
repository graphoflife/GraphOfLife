# -*- coding: utf-8 -*-
"""
the physics of space, worlds at scale: Worlds of 100,000 tokens — how a line takes over a world ten
times the size of the baseline, the thesis of Experiment 10.

When does every agent alive descend from one and the same founder? The lineage
analysis (gol_analysis.lineage) follows every genotype's parent through every
frame, and every 25 iterations climbs from the living to their newest common
ancestor. Its first check at which they share one is the moment a single
founder's line has taken the world over.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

import book_data as D
import book_figures as F
from book_figures import FRAMES, STATS_FILE, chapter, describe, line, mean_over, recipe
from book_chapters.common import BLUE, GREEN, GREY, ORANGE, RED, VIOLET, WORLD_SETS, YELLOW

E08_WORDS = ("the baseline worlds of Experiment 8, three seeds at each of ten sizes from 800 to 409,600 tokens, "
             "600 iterations each ([How does a world's size follow its tokens?](how-does-a-worlds-size-follow-its-tokens.md); `python3 gol_lab.py run E08`)")


@D.measure("lineage")
def lineage_of(run_id: str) -> Dict[str, Any]:
    """gol_analysis.lineage of a run, kept beside it: genotypes and common ancestors through every frame."""
    import gol_analysis
    found = gol_analysis.lineage(run_id)
    return {"ancestor": found["ancestor"], "ancestry": found["ancestry"], "end": found["end"]}


def founder_times(experiment: str, condition: str = "baseline") -> Dict[str, Optional[int]]:
    """Every run's first check at which all its living descend from one founder: from the
    experiment's own analysis where it has one, otherwise from the kept measure."""
    path = os.path.join(F.BOOK, "results", f"{experiment}.json")
    if os.path.exists(path):
        found = json.load(open(path)).get("lineage", {}).get(condition)
        if found:
            return {k: v["ancestor"]["oneFounder"] for k, v in found["runs"].items()}
    return {s.run_id: lineage_of(s.run_id)["ancestor"]["oneFounder"] for s in F.runs_of(experiment, condition)
            if F.lived(s) >= s.until}


def censored_median(values: List[Optional[int]]) -> Optional[float]:
    """The median, a world that never got there counted as later than all; None if half never did."""
    if not values:
        return None
    v = sorted(np.inf if x is None else x for x in values)
    m = float(np.median(v))
    return None if np.isinf(m) else m


def not_yet(values: List[Optional[int]], horizon: int) -> Dict[str, List[float]]:
    """For every iteration t, the share of worlds whose living still descend from more than one founder."""
    t = np.arange(0, horizon + 1, 25)
    v = np.array([np.inf if x is None else x for x in values], float)
    return {"x": t.tolist(), "y": [float(np.mean(v > x)) for x in t]}


@chapter
def takeover(ch: F.Chapter) -> None:
    large, small = WORLD_SETS["E10"], WORLD_SETS["E02"]
    big = founder_times(large.lineage)
    little = founder_times(small.lineage)
    b, s = list(big.values()), list(little.values())
    mb, ms = censored_median(b), censored_median(s)
    ratio = mb / ms if mb and ms else None
    ch.number("one_founder", {"large": {"values": sorted(b, key=lambda x: (x is None, x)), "median": mb,
                                        "reached": sum(x is not None for x in b), "worlds": len(b)},
                              "small": {"values": sorted(s, key=lambda x: (x is None, x)), "median": ms,
                                        "reached": sum(x is not None for x in s), "worlds": len(s)},
                              "ratio": ratio, "holds": bool(mb is not None and 350 <= mb <= 875)})
    curves = [line(**not_yet(s, small.until - 1), label=f"{small.tokens:,} tokens ({len(s)} worlds)", colour=YELLOW,
                   width=2),
              line(**not_yet(b, large.until - 1), label=f"{large.tokens:,} tokens ({len(b)} worlds)", colour=BLUE,
                   width=2)]
    ch.figure(
        "not-yet", title="Worlds not yet taken over by one founder's line",
        x={"label": "iteration", "min": 0, "max": large.until}, y={"label": "share of the worlds", "min": 0, "max": 1},
        series=curves,
        guides=[{"axis": "x", "at": 350, "label": "thesis: from"}, {"axis": "x", "at": 875, "label": "to"},
                {"axis": "y", "at": 0.5, "label": ""}],
        caption="For every iteration, the share of the worlds whose living, at the check after that iteration's "
                "game (every 25 iterations), still descend from more than one founder. Yellow: the 30 worlds of "
                f"{small.tokens:,} tokens; blue: the 30 of {large.tokens:,}. A world that died, or never got there, "
                "stays in the share. Where a line crosses one half is its median; the thesis of Experiment 10 "
                "expected the blue one to cross between the two grey lines.",
        recipe=recipe(f"{large.runs_text()} {small.runs_text()}",
                      f"the lineage analyses of Experiments {int(large.lineage[1:])} and {int(small.lineage[1:])} "
                      f"(`book/results/{large.lineage}.json`, `book/results/{small.lineage}.json`, `oneFounder` of every run)",
                      ["For every run, the first check at which every living agent descends from one founder.",
                       "For every t, the share of runs whose first such check is later than t, or that have none."]))

    # Across sizes: the worlds of Experiment 8, then the two sets of thirty.
    by_size: Dict[int, List[Optional[int]]] = {}
    for s_ in F.runs_of("E08", "baseline"):
        if F.lived(s_) >= s_.until:
            by_size.setdefault(int(s_.config["total_tokens"]), []).append(lineage_of(s_.run_id)["ancestor"]["oneFounder"])
    sizes = sorted(by_size)
    xs, ys, cens = [], [], []
    for t in sizes:
        for v in by_size[t]:
            xs.append(t)
            ys.append(600.0 if v is None else float(v))
            cens.append(v is None)
    xs, ys, cens = np.array(xs, float), np.array(ys), np.array(cens)
    reached = ~cens
    width_line = 175 * (np.array(sizes, float) / small.tokens) ** 0.30
    count_line = 175 * (np.array(sizes, float) / small.tokens)
    ch.figure(
        "sizes", title="How long a takeover takes, by the size of the world",
        x={"label": "tokens (logarithmic)", "log": True}, y={"label": "iteration (logarithmic)", "log": True, "min": 10},
        series=[{"label": "a world of Experiment 8", "x": xs[reached].tolist(), "y": ys[reached].tolist(),
                 "points": True, "size": 7, "colour": GREEN},
                {"label": "… not taken over by its iteration 600", "x": xs[cens].tolist(), "y": ys[cens].tolist(),
                 "points": True, "size": 7, "colour": GREY},
                {"label": f"median of the 30 worlds of {small.tokens:,} and of {large.tokens:,} tokens",
                 "x": [small.tokens, large.tokens], "y": [ms, mb], "points": True, "size": 11, "colour": BLUE},
                line(sizes, width_line, "growing with the width of the world (tokens^0.30)", ORANGE, width=1.5,
                     dash=[5, 4]),
                line(sizes, count_line, "growing with its number of agents (tokens^1)", RED, width=1.5, dash=[2, 3])],
        caption="For every world of Experiment 8 that lived to its end (green, or grey if all its living still came "
                "from more than one founder at its last check, at iteration 600, drawn there), and for the median "
                "of each set of 30 worlds (blue): the first iteration at which every living agent descends from a "
                "single founder. Both axes logarithmic. Dashed: what the two paces of the thesis predict, drawn "
                f"through the median of the worlds of {small.tokens:,} tokens, 175 — a front, whose time grows with "
                "the width of the world (orange), and chance, whose time grows with the number of agents (red).",
        recipe=recipe(f"{E08_WORDS}; and the two sets of 30 worlds above.", FRAMES,
                      ["For the runs of Experiment 8, `gol_analysis.lineage` of every run (kept as the notebook measure "
                       "`lineage`): its `ancestor.oneFounder`.",
                       "Plot it against the run's tokens; the two sets of 30 as in the figure above."]))
    slope = None
    if reached.sum() >= 3:
        slope = float(np.polyfit(np.log(xs[reached]), np.log(ys[reached]), 1)[0])
    ch.number("sizes", {"by_tokens": {str(t): by_size[t] for t in sizes}, "slope_reached": slope,
                        "exponent_between_sets": float(np.log(mb / ms) / np.log(large.tokens / small.tokens))
                        if mb and ms else None})
    domain_figures(ch)
    ten_small(ch)


#: How far back a family is founded, in iterations, and how far out it is looked for.
LAGS = (25, 50, 100, 200)
DOMAIN_R = 30
DOMAIN_SOURCES = 100


@D.measure("domains")
def domains(run_id: str) -> Dict[str, Any]:
    """
    Do lines of descent hold regions of a world? At five moments of its settled
    life, every agent's family at each lag k — the genotype alive k iterations
    earlier that its own genotype descends from — and, from DOMAIN_SOURCES agents
    drawn at random, for every distance r: the pairs (source, agent r steps away)
    and how many of them share a family; with the share any two agents would.
    """
    from book_graph import bfs, graph
    from book_chapters.space import moments
    last = D.last_iteration(run_id)
    rng = np.random.default_rng(51)
    out = []
    for t in moments(last):
        if t - max(LAGS) < 0:
            continue
        parent: Dict[int, int] = {}
        for u in range(t - max(LAGS), t + 1):
            for phase in (1, 2):
                f = D.frame_at(run_id, u, phase)
                for g, p in zip(f["brain_ids"], f["parent_brain_ids"]):
                    parent.setdefault(g, p)
        anchors = {k: set(D.frame_at(run_id, t - k, 2)["brain_ids"]) for k in LAGS}
        frame = D.frame_at(run_id, t, 2)
        ids = list(frame["ids"])
        geno = dict(zip(ids, frame["brain_ids"]))

        def family(g: int, anchor: set) -> Optional[int]:
            steps = 0
            while g not in anchor and g in parent and steps < 100000:
                g, steps = parent[g], steps + 1
            return g if g in anchor else None
        fam = {k: {u: family(geno[u], anchors[k]) for u in ids} for k in LAGS}
        adj = graph(frame)
        pairs = np.zeros(DOMAIN_R + 1)
        same = {k: np.zeros(DOMAIN_R + 1) for k in LAGS}
        for i in rng.choice(len(ids), min(DOMAIN_SOURCES, len(ids)), replace=False):
            source = ids[i]
            for v, r in bfs(adj, source, DOMAIN_R).items():
                if r == 0:
                    continue
                pairs[r] += 1
                for k in LAGS:
                    a, b = fam[k][source], fam[k][v]
                    same[k][r] += a is not None and a == b
        chance, largest, families = {}, {}, {}
        for k in LAGS:
            counts = np.array(list(Counter(x for x in fam[k].values() if x is not None).values()), float)
            n = len(ids)
            chance[str(k)] = float((counts * (counts - 1)).sum() / (n * (n - 1))) if n > 1 else None
            largest[str(k)] = float(counts.max() / n) if counts.size else None
            families[str(k)] = int(counts.size)
        out.append({"t": t, "agents": len(ids), "pairs": pairs.tolist(),
                    "same": {str(k): v.tolist() for k, v in same.items()},
                    "chance": chance, "largest": largest, "genotype": families})
    return {"moments": out}


def enrichment(kept: List[Dict[str, Any]], lag: int) -> np.ndarray:
    """For every world (a row) and distance r = 1 … DOMAIN_R: how many times more often two agents r steps apart
    share their family of `lag` iterations back than any two agents do, pooled over the world's moments."""
    rows = []
    for m in kept:
        pairs = np.zeros(DOMAIN_R + 1)
        same = np.zeros(DOMAIN_R + 1)
        chance = []
        for x in m["moments"]:
            pairs += np.array(x["pairs"])
            same += np.array(x["same"][str(lag)])
            chance.append(x["chance"][str(lag)])
        with np.errstate(invalid="ignore", divide="ignore"):
            rows.append(np.where(pairs >= 200, same / pairs, np.nan)[1:] / np.mean(chance))
    return np.array(rows)


def reach(row: np.ndarray) -> float:
    """The first distance at which two agents are no more often of one family than any two."""
    below = np.flatnonzero(row <= 1)
    return float(below[0] + 1) if below.size else float("nan")


def domain_figures(ch: F.Chapter) -> None:
    """Worlds of 100,000 tokens's figure of domains: how far a line of descent reaches in a world."""
    large, small = WORLD_SETS["E10"], WORLD_SETS["E02"]
    sets = {"large": (large, [domains(s.run_id) for s in large.alive]),
            "small": (small, [domains(s.run_id) for s in small.alive])}
    r = np.arange(1, DOMAIN_R + 1)
    panels = []
    for key, (worlds, kept) in sets.items():
        series_ = []
        for lag, colour in zip(LAGS, (YELLOW, ORANGE, RED, VIOLET)):
            table = enrichment(kept, lag)
            series_.append(line(r, np.nanmedian(table, axis=0), f"family of {lag} iterations back", colour, width=2))
            ch.number(f"{key}_lag{lag}", {"median": np.nanmedian(table, axis=0)[:15].tolist(),
                                          "reach": describe([reach(row) for row in table])})
        panels.append(dict(title=f"Worlds of {worlds.tokens:,} tokens", x={"label": "steps apart", "min": 0, "max": 20},
                           y={"label": "same family, times as often as for any two (logarithmic)", "log": True,
                              "min": 0.05, "max": 50},
                           series=series_, guides=[{"axis": "y", "at": 1.0, "label": ""}]))
    ch.grid(
        "domains", panels, columns=2, title="Do lines of descent hold regions?",
        caption="An agent's family at lag k is the genotype alive k iterations earlier from which its own genotype "
                "descends. For pairs of agents r steps apart — from breadth-first searches out of 100 agents drawn at "
                "random, at five moments of each world's settled life — the share that belong to the same family, "
                "divided by the share among any two agents of the world, for four lags; each line is the median "
                f"over the worlds. Left: the {len(sets['large'][1])} worlds of {large.tokens:,} tokens; right: the "
                f"{len(sets['small'][1])} of {small.tokens:,}. Above 1, neighbours are kin more often than chance; "
                "where a line falls to 1 is how far a family reaches.",
        recipe=recipe(f"{large.runs_text()} {small.surviving_text()}", FRAMES,
                      ["At five moments t of each run (`book_chapters.space.moments`), read the frames from t − 200 "
                       "to t and note every genotype's parent; for each lag k, climb from each living agent's "
                       "genotype to one alive after the game of t − k.",
                       "Breadth-first searches out of 100 agents drawn at random (seed 51), to 30 steps; per r, the "
                       "pairs and the pairs in one family; chance: Σ c(c − 1) / (n(n − 1)) over the families' counts "
                       "c (`book_chapters.scale.domains`)."]))


# ---------------------------------------------------------------------------
# Is a large world ten small ones?
# ---------------------------------------------------------------------------

#: The viewer's statistics compared across the two sizes, in three panels: shares and rates per agent; the
#: network's shape and distances; its largest single things and its totals. (stat, phase, label, colour)
COMPARED = (
    ("Shares and rates per agent", (0.5, 2), (
        ("edges/nodes", 2, "connections", BLUE), ("births/nodes_before", 1, "births", BLUE),
        ("starved/nodes_before", 2, "starved", BLUE), ("orphaned/nodes_before", 2, "cut off", BLUE),
        ("leaves/nodes", 2, "leaves", BLUE), ("distinctBrains/nodes", 2, "genotypes", BLUE),
        ("revolutions/nodes_before", 2, "nodes taken", BLUE), ("heldHomeShare", 2, "nodes kept", BLUE),
        ("gini", 2, "Gini", BLUE), ("topDecileShare", 2, "richest tenth", BLUE))),
    ("Shape and distances", (0.3, 3), (
        ("coreShare", 2, "core", GREEN), ("cycleRank/nodes", 2, "bridges", GREEN),
        ("bridges/edges", 2, "bridges", GREEN), ("transitivity", 2, "clustering", GREEN),
        ("assortativity", 2, "assortativity", GREEN), ("spectralGap", 2, "spectral gap", GREEN),
        ("dimension", 2, "dimension", ORANGE), ("meanPathLength", 2, "mean distance", ORANGE),
        ("diameter", 2, "diameter", ORANGE))),
    ("The largest, and totals", (1, 20), (
        ("maxTokenAdded", 2, "largest gain", RED), ("maxTokens", 2, "richest", RED),
        ("maxDegree", 2, "best-connected", RED), ("nodes", 2, "agents", GREY),
        ("cladesInWindow", 2, "genotype", GREY))),
)
#: Both sets of worlds are compared over the same iterations.
COMPARED_WINDOW = (500, 1499)


def ten_small(ch: F.Chapter) -> None:
    """Every compared statistic's median over the large worlds, divided by its median over the small ones."""
    large, small = WORLD_SETS["E10"], WORLD_SETS["E02"]
    said, panels = {}, []
    for title, (low, high), stats in COMPARED:
        for stat, phase, label, colour in stats:
            values = {w.experiment: float(np.median([mean_over(s.run_id, stat, *COMPARED_WINDOW, phase)
                                                     for s in w.alive])) for w in (large, small)}
            said[stat] = {"large": values[large.experiment], "small": values[small.experiment],
                          "ratio": values[large.experiment] / values[small.experiment]}
        labels = [label for _, _, label, _ in stats]
        panels.append(dict(
            title=title, x={"label": "", "categories": labels},
            y={"label": "large ÷ small (logarithmic)", "log": True, "min": low, "max": high},
            series=[{"label": None, "kind": "bars", "x0": [k - 0.36], "x1": [k + 0.36], "y0": [1.0],
                     "y": [said[stat]["ratio"]], "colour": colour} for k, (stat, _, _, colour) in enumerate(stats)],
            legend=False))
    width = (large.tokens / small.tokens) ** 0.30
    agents = said["nodes"]["ratio"]
    for panel in panels:
        panel["guides"] = [{"axis": "y", "at": at, "label": text} for at, text in
                           ((1, "the same"), (width, f"×{width:.1f}"), (agents, f"×{agents:.0f}"))
                           if panel["y"]["min"] <= at <= panel["y"]["max"]]
    ch.grid(
        "ten-small", panels, columns=1, title="One world of 100,000 tokens beside one of 10,000",
        caption=f"For each of {len(said)} of the viewer's statistics, its mean over iterations "
                f"{COMPARED_WINDOW[0]}–{COMPARED_WINDOW[1]:,} in each world that lived to the end; the median of "
                f"the {len(large.alive)} worlds of {large.tokens:,} tokens divided by the median of the "
                f"{len(small.alive)} worlds of {small.tokens:,} — over the same iterations, though the small worlds "
                "ran on to 2,999. On logarithmic axes, so that half as much and twice as much lie equally far from "
                "1; each bar rises from 1 or falls from it. Blue: shares and rates per agent; green: the shape of the "
                "network; orange: its distances; red: its largest single things; grey: totals. The dashed guides: no change; "
                f"×{width:.1f}, the factor by which the width of a world grows (tokens^0.30, [How does a world's size follow its tokens?](how-does-a-worlds-size-follow-its-tokens.md)); and "
                f"×{agents:.0f}, the ratio of the agents.",
        recipe=recipe(f"{large.runs_text()} {small.surviving_text()}", STATS_FILE,
                      [f"For every run and statistic, the mean over the rows of iterations {COMPARED_WINDOW[0]} to "
                       f"{COMPARED_WINDOW[1]:,} (`phase` 1 for births, 2 for the rest; a ratio such as "
                       "`edges/nodes` is taken row by row).",
                       "The median over each set's runs; the large set's divided by the small set's."]))
    ch.number("ten_small", said)
