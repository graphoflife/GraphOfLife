# -*- coding: utf-8 -*-
"""
Part II, agents and connections over time: Chapter 20 — do the rich stay rich? —
and Chapter 21 — how the network grows.

Both follow single agents and single connections from frame to frame, so they read
the frames themselves, at regular moments of every world's settled life, and keep
what they found beside the runs (GraphOfLifeRuns/.book/<run>.<kind>.json).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List

import numpy as np

import book_data as D
import book_figures as F
from book_figures import FRAMES, band_series, chapter, describe, dots, line, recipe, survivors
from book_chapters.common import BLUE, CYAN, GREEN, GREY, ORANGE, RED, VIOLET, YELLOW, baseline

LAGS = (1, 2, 5, 10, 20, 50, 100, 200)
STARTS = range(500, 2800, 100)
QUINTILES = 5
SURVIVORS = ("The 26 baseline worlds that lived to the end (`B1-10000-s001` … `-s030`, Chapter 9; "
             "`python3 gol_lab.py run E02`).")


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks (1 … n), ties sharing the mean of their places."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values))
    sorted_v = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _ranks(a), _ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float("nan")


@D.measure("mobility")
def mobility(run_id: str) -> Dict[str, Any]:
    """Tokens of every agent after the game of each start iteration, and of the same agents later."""
    rho = {k: [] for k in LAGS}
    alive = {k: [] for k in LAGS}
    moves = np.zeros((QUINTILES, QUINTILES + 1))
    top = {k: {"dead": 0, "top10": 0, "ratio": []} for k in LAGS}
    tops = 0
    for t0 in STARTS:
        a = D.frame_at(run_id, t0, 2)
        tok0 = dict(zip(a["ids"], a["tokens"]))
        ids0 = np.array(a["ids"])
        values0 = np.array(a["tokens"], float)
        rank0 = _ranks(values0) / len(values0)
        richest = set(ids0[values0 >= np.quantile(values0, 0.99)].tolist())
        tops += len(richest)
        for k in LAGS:
            b = D.frame_at(run_id, t0 + k, 2)
            tok1 = dict(zip(b["ids"], b["tokens"]))
            both = [i for i in a["ids"] if i in tok1]
            alive[k].append(len(both) / len(a["ids"]))
            x = np.array([tok0[i] for i in both], float)
            y = np.array([tok1[i] for i in both], float)
            rho[k].append(spearman(x, y))
            v1 = np.array(b["tokens"], float)
            cut10 = np.quantile(v1, 0.9)
            for i in richest:
                if i not in tok1:
                    top[k]["dead"] += 1
                else:
                    top[k]["top10"] += tok1[i] >= cut10
                    top[k]["ratio"].append(tok1[i] / tok0[i])
            if k == 10:
                rank1 = dict(zip(b["ids"], _ranks(v1) / len(v1)))
                for i, r in zip(a["ids"], rank0):
                    q0 = min(QUINTILES - 1, int(r * QUINTILES - 1e-9))
                    if i in rank1:
                        q1 = min(QUINTILES - 1, int(rank1[i] * QUINTILES - 1e-9))
                        moves[q0, q1] += 1
                    else:
                        moves[q0, QUINTILES] += 1
    return {"rho": {str(k): v for k, v in rho.items()}, "alive": {str(k): v for k, v in alive.items()},
            "moves": moves.tolist(), "tops": tops,
            "top": {str(k): {"dead": v["dead"], "top10": int(v["top10"]),
                             "ratio": float(np.median(v["ratio"])) if v["ratio"] else None}
                    for k, v in top.items()}}


DEGREE_CLASSES = ((1, 1, "1"), (2, 2, "2"), (3, 9, "3–9"), (10, 49, "10–49"), (50, 10 ** 9, "50+"))


@D.measure("fragile")
def fragile(run_id: str) -> Dict[str, Any]:
    """Who dies within ten games, by connections: the richest hundredth, and every agent."""
    from book_chapters.structure import graph
    out = {label: {"richest": [0, 0], "all": [0, 0]} for _, _, label in DEGREE_CLASSES}
    for t0 in STARTS:
        a = D.frame_at(run_id, t0, 2)
        later = set(D.frame_at(run_id, t0 + 10, 2)["ids"])
        adj = graph(a)
        tokens = np.array(a["tokens"], float)
        cut = np.quantile(tokens, 0.99)
        for u, tok in zip(a["ids"], tokens):
            k = len(adj[u])
            label = next(lab for lo, hi, lab in DEGREE_CLASSES if lo <= k <= hi) if k else None
            if label is None:
                continue
            for group in ("all", "richest") if tok >= cut else ("all",):
                out[label][group][0] += 1
                out[label][group][1] += u not in later
    return out


@chapter
def rich(ch: F.Chapter) -> None:
    alive = survivors(baseline())
    mob = {s.run_id: mobility(s.run_id) for s in alive}
    lags = np.array(LAGS, float)
    # A moment after which fewer than three of its agents were still alive has no correlation.
    rho = np.array([[np.nanmean(np.array(m["rho"][str(k)], float)) for k in LAGS] for m in mob.values()])
    live = np.array([[np.mean(m["alive"][str(k)]) for k in LAGS] for m in mob.values()])

    def bands(table):
        q = np.quantile(table, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
        return {"x": lags.tolist(), "y": q[2].tolist(), "lo": q[1].tolist(), "hi": q[3].tolist(),
                "outerLo": q[0].tolist(), "outerHi": q[4].tolist()}
    ch.grid(
        "memory", [
            dict(title="Rank of wealth, then and later", x={"label": "games later (logarithmic)", "log": True},
                 y={"label": "rank correlation of tokens", "min": 0, "max": 1},
                 series=[{**bands(rho), "label": None, "colour": YELLOW}], legend=False),
            dict(title="Still alive", x={"label": "games later (logarithmic)", "log": True},
                 y={"label": "share of the agents", "min": 0, "max": 1},
                 series=[{**bands(live), "label": None, "colour": GREEN}], legend=False)],
        columns=2, title="How long wealth lasts",
        caption="Left: the agents alive after the game of an iteration t, and the same agents k games later, if "
                "they are still alive: Spearman's rank correlation of their tokens then and later "
                "([Correlation](../notes/correlation.md)) — 1 if every agent kept its place in the order of "
                "wealth, 0 if the later order has nothing to do with the earlier. Right: the share of the "
                "agents still alive k games later. For each world, the mean over 23 moments t = 500, 600, …, "
                "2,700 (leaving out, for the correlation, the few moments after which fewer than three of the "
                "agents were alive); the line is the median of the 26 worlds, the darker band the middle half "
                "of them and the paler band nine in ten.",
        recipe=recipe(SURVIVORS, FRAMES,
                      ["For t = 500, 600, …, 2,700 and k = 1, 2, 5, 10, 20, 50, 100, 200: read frames 2·t + 1 and "
                       "2·(t + k) + 1; the agents (`ids`) in both, and their `tokens` in each.",
                       "Spearman's ρ: the correlation of the ranks of the two token counts (ties share their "
                       "mean rank); alive: agents in both over agents in the first.",
                       "Mean over t for each world; median and quantiles over worlds."]))
    ch.number("memory", {"lags": list(LAGS), "rho": [describe(rho[:, i]) for i in range(len(LAGS))],
                         "alive": [describe(live[:, i]) for i in range(len(LAGS))]})

    # From one fifth to another, ten games later.
    moves = np.sum([np.array(m["moves"]) for m in mob.values()], axis=0)
    shares = moves / moves.sum(axis=1, keepdims=True)
    names = ["poorest fifth", "second", "third", "fourth", "richest fifth"]
    cells = {"kind": "cells", "x0": [], "x1": [], "y0": [], "y1": [], "value": []}
    for i in range(QUINTILES):
        for j in range(QUINTILES + 1):
            cells["x0"].append(j - 0.5)
            cells["x1"].append(j + 0.5)
            cells["y0"].append(i - 0.5)
            cells["y1"].append(i + 0.5)
            cells["value"].append(float(shares[i, j]))
    stay = shares[:, :QUINTILES]
    shorrocks = float((QUINTILES - np.trace(stay / stay.sum(axis=1, keepdims=True))) / (QUINTILES - 1))
    ch.figure(
        "moves", title="From one fifth to another, ten games later",
        x={"label": "fifth of wealth ten games later", "categories": names + ["dead"]},
        y={"label": "fifth of wealth now", "categories": names},
        series=[cells], colourbar={"map": "viridis", "min": 0, "max": float(shares.max()), "label": "share of the row"},
        caption="All agents alive after the game of t = 500, 600, …, 2,700 in the 26 worlds that lived to the "
                "end, sorted into fifths of the world by their tokens (ranks, ties shared), and where each one "
                "was ten games later: in which fifth, or dead. Each row adds up to 1; a cell's colour is the "
                "share of its row.",
        recipe=recipe(SURVIVORS, FRAMES,
                      ["As in the figure above, with k = 10: each agent's fifth is ⌈rank / n · 5⌉ among the n "
                       "agents of its frame.",
                       "Count the agents in every pair of fifths, and those not in the later frame as dead; divide "
                       "each row by its total."]))
    ch.number("moves", {"shares": shares.tolist(), "shorrocks": shorrocks})

    # The fate of the richest hundredth.
    tops = sum(m["tops"] for m in mob.values())
    dead = np.array([sum(m["top"][str(k)]["dead"] for m in mob.values()) / tops for k in LAGS])
    top10 = np.array([sum(m["top"][str(k)]["top10"] for m in mob.values()) / tops for k in LAGS])
    ratio = np.array([np.median([m["top"][str(k)]["ratio"] for m in mob.values() if m["top"][str(k)]["ratio"]])
                      for k in LAGS])
    ch.grid(
        "richest", [
            dict(title="Where the richest hundredth are", x={"label": "games later (logarithmic)", "log": True},
                 y={"label": "share of them", "min": 0, "max": 1},
                 series=[line(lags, top10, "still in the richest tenth", YELLOW, width=2),
                         line(lags, dead, "dead", RED, width=2)]),
            dict(title="What they still hold", x={"label": "games later (logarithmic)", "log": True},
                 y={"label": "tokens later ÷ tokens then (logarithmic)", "log": True, "min": 0.05, "max": 2},
                 series=[line(lags, ratio, "median of the survivors, median over worlds", ORANGE, width=2)],
                 guides=[{"axis": "y", "at": 1.0, "label": "as much as then"}])],
        columns=2, title="The fate of the richest",
        caption="The agents in the richest hundredth of their world after the game of t = 500, 600, …, 2,700 "
                f"({tops:,} in all, pooled over the 26 worlds that lived to the end), followed k games on. Left: "
                "the share of them still in the richest tenth of their world, and the share dead. Right: for "
                "those alive, their tokens then divided by their tokens at t — the median in each world, and the "
                "median of the worlds.",
        recipe=recipe(SURVIVORS, FRAMES,
                      ["As in the first figure: the richest hundredth are the agents with tokens at or above the "
                       "99th percentile of their frame; the richest tenth later, at or above the 90th."]))
    ch.number("richest", {"lags": list(LAGS), "dead": dead.tolist(), "top10": top10.tolist(),
                          "ratio": ratio.tolist(), "count": tops})

    # Which of the richest die: wealth on a dead end.
    from book_chapters.measures import bars
    frag = [fragile(s.run_id) for s in alive]
    labels = [label for _, _, label in DEGREE_CLASSES]
    counts = {g: np.array([[sum(f[lab][g][i] for f in frag) for lab in labels] for i in (0, 1)], float)
              for g in ("richest", "all")}
    share = {g: c[1] / np.maximum(c[0], 1) for g, c in counts.items()}
    ch.figure(
        "fragile", title="Which of the richest die",
        x={"label": "connections", "categories": labels}, y={"label": "share dead ten games later", "min": 0, "max": 0.6},
        series=[bars(labels, share["richest"], RED, "the richest hundredth", slot=(-0.38, 0)),
                bars(labels, share["all"], GREY, "every agent", slot=(0, 0.38))],
        caption="The agents alive after the game of t = 500, 600, …, 2,700 in the 26 worlds that lived to the end, "
                "in classes of their connections at t: the share no longer alive ten games later, for the richest "
                "hundredth of their world (red) and for every agent (grey). Pooled over moments and worlds: "
                + ", ".join(f"{int(counts['richest'][0][i]):,} of the richest and {int(counts['all'][0][i]):,} in "
                            f"all with {lab}" for i, lab in enumerate(labels)) + " connections.",
        recipe=recipe(SURVIVORS, FRAMES,
                      ["For t = 500, 600, …, 2,700: read frames 2·t + 1 and 2·(t + 10) + 1; each agent's "
                       "connections at t from `edges`; the richest hundredth as above.",
                       "Per class of connections, the agents of the first frame missing from the second, over all "
                       "agents of the class (`book_chapters.dynamics.fragile`)."]))
    ch.number("fragile", {"classes": labels, "richest": share["richest"].tolist(), "all": share["all"].tolist(),
                          "richest_n": counts["richest"][0].tolist(), "all_n": counts["all"][0].tolist()})


# ---------------------------------------------------------------------------
# Chapter 21 · How the network grows
# ---------------------------------------------------------------------------

GROWTH_STARTS = range(500, 3000, 25)
CLASSES = [(1, 1), (2, 2), (3, 4), (5, 9), (10, 19), (20, 49), (50, 99), (100, 10 ** 6)]


@D.measure("growth")
def growth(run_id: str) -> Dict[str, Any]:
    """Connections gained in a reproduction phase and lost in the game after it, agent by agent."""
    gained = defaultdict(lambda: [0, 0])     # class -> [agents, connections gained]
    lost = defaultdict(lambda: [0, 0])
    links = Counter()
    births = parent_linked = 0
    for t in GROWTH_STARTS:
        a = D.frame_at(run_id, t - 1, 2)       # after the game of t − 1
        b = D.frame_at(run_id, t, 1)           # after the reproduction phase of t
        c = D.frame_at(run_id, t, 2)           # after the game of t
        ea = {tuple(sorted(e)) for e in a["edges"]}
        eb = {tuple(sorted(e)) for e in b["edges"]}
        ec = {tuple(sorted(e)) for e in c["edges"]}
        deg_a, deg_b = Counter(), Counter()
        for u, v in ea:
            deg_a[u] += 1
            deg_a[v] += 1
        for u, v in eb:
            deg_b[u] += 1
            deg_b[v] += 1
        new_b = Counter()
        for u, v in eb - ea:
            new_b[u] += 1
            new_b[v] += 1
        gone_c = Counter()
        for u, v in eb - ec:
            gone_c[u] += 1
            gone_c[v] += 1
        alive_b = set(b["ids"])
        for u in a["ids"]:
            if u not in alive_b:
                continue
            k = deg_a[u]
            cls = next(i for i, (lo, hi) in enumerate(CLASSES) if lo <= k <= hi) if k else None
            if cls is not None:
                gained[cls][0] += 1
                gained[cls][1] += new_b[u]
        alive_c = set(c["ids"])
        for u in b["ids"]:
            k = deg_b[u]
            if not k:
                continue
            cls = next(i for i, (lo, hi) in enumerate(CLASSES) if lo <= k <= hi)
            lost[cls][0] += 1
            lost[cls][1] += k if u not in alive_c else gone_c[u]
        for birth in (b.get("decisions") or {}).get("births") or []:
            births += 1
            joined = birth.get("links") or []
            parent_linked += birth.get("agent") in joined
            links[min(len(joined), 6)] += 1
    return {"gained": {str(c): v for c, v in gained.items()}, "lost": {str(c): v for c, v in lost.items()},
            "births": births, "parent_linked": parent_linked, "links": {str(k): v for k, v in links.items()}}


@chapter
def grows(ch: F.Chapter) -> None:
    alive = survivors(baseline())
    g = {s.run_id: growth(s.run_id) for s in alive}
    labels = ["1", "2", "3–4", "5–9", "10–19", "20–49", "50–99", "100+"]
    mids = np.array([1, 2, 3.5, 7, 14.5, 34.5, 74.5, 150.0])

    def rate(kind):
        table = np.full((len(g), len(CLASSES)), np.nan)
        for w, m in enumerate(g.values()):
            for c in range(len(CLASSES)):
                n, total = m[kind].get(str(c), [0, 0])
                if n >= 20:
                    table[w, c] = total / n
        return table
    gain_t, loss_t = rate("gained"), rate("lost")
    pooled = {kind: np.array([sum(m[kind].get(str(c), [0, 0])[1] for m in g.values()) /
                              max(1, sum(m[kind].get(str(c), [0, 0])[0] for m in g.values()))
                              for c in range(len(CLASSES))]) for kind in ("gained", "lost")}
    fit = {}
    for kind, values in pooled.items():
        keep = values > 0
        slope, icpt = np.polyfit(np.log(mids[keep]), np.log(values[keep]), 1)
        fit[kind] = (float(slope), float(np.exp(icpt)))
    xs = np.array([1.0, 200.0])
    ch.figure(
        "kernel", title="Who gains and who loses connections",
        x={"label": "connections k at the start (logarithmic)", "log": True, "min": 0.8, "max": 250},
        y={"label": "connections per agent (logarithmic)", "log": True, "min": 0.005, "max": 200},
        series=[{"label": "gained in a reproduction phase", "x": mids.tolist(), "y": pooled["gained"].tolist(),
                 "points": True, "size": 8, "colour": GREEN},
                line(xs, fit["gained"][1] * xs ** fit["gained"][0], f"fitted: k^{fit['gained'][0]:.2f}", GREEN,
                     width=1.2, dash=[5, 4]),
                {"label": "lost in the game after it", "x": mids.tolist(), "y": pooled["lost"].tolist(),
                 "points": True, "size": 8, "colour": RED},
                line(xs, fit["lost"][1] * xs ** fit["lost"][0], f"fitted: k^{fit['lost'][0]:.2f}", RED,
                     width=1.2, dash=[5, 4]),
                line(xs, 0.05 * xs, "in proportion to k", "#eef4fa", width=1, dash=[2, 3])],
        caption="Agents sorted by their number of connections k at the start of a phase, in classes of growing "
                "width (dots at the middle of each class). Green: the new connections an agent alive before and "
                "after a reproduction phase gained in it, per agent — to its own newborn child, to a neighbour's "
                "child, or handed over. Red: the connections an agent had after the reproduction phase and no "
                "longer had after the game that followed — cut for carrying nothing, or gone with it if it died "
                "— per agent. Pooled over the reproduction phases and games of every 25th iteration from 500 on, "
                "in the 26 worlds that lived to the end. Dashed: straight lines fitted on these logarithmic axes; "
                "dotted: a line of slope 1, in proportion to k.",
        recipe=recipe(SURVIVORS, FRAMES,
                      ["For t = 500, 525, …, 2,975 read frames 2·t − 1 (before the reproduction phase of t), 2·t "
                       "(after it) and 2·t + 1 (after the game).",
                       "Gained: the edges in 2·t not in 2·t − 1, counted at both ends, for agents in both frames, "
                       "by their degree in 2·t − 1. Lost: the edges in 2·t not in 2·t + 1, counted for agents of "
                       "2·t by their degree there (all of them if the agent is gone).",
                       "Sum over t and worlds per class, divide by the agents; least squares of ln(rate) on "
                       "ln(class middle)."]))
    ch.number("kernel", {"classes": labels, "gained": pooled["gained"].tolist(), "lost": pooled["lost"].tolist(),
                         "gain_slope": fit["gained"][0], "loss_slope": fit["lost"][0],
                         "gained_by_world": [describe(gain_t[:, c]) for c in range(len(CLASSES))],
                         "lost_by_world": [describe(loss_t[:, c]) for c in range(len(CLASSES))]})

    births = sum(m["births"] for m in g.values())
    linked = sum(m["parent_linked"] for m in g.values())
    counts = Counter()
    for m in g.values():
        for k, v in m["links"].items():
            counts[int(k)] += v
    cats = ["0", "1", "2", "3", "4", "5", "6+"]
    ch.number("births", {"births": births, "parent_linked": linked / births if births else None,
                         "links": {c: counts.get(i, 0) / births for i, c in enumerate(cats)}})
    _age_degree(ch, alive)


def _age_degree(ch: F.Chapter, alive) -> None:
    """Connections against age, at the end of every world: does a head start pay?"""
    ages, degrees = [], []
    for s in alive:
        last = F.world_pass(s.run_id)["last"]
        ages.extend(last["ages"])
        degrees.extend(last["degrees"])
    ages, degrees = np.array(ages, float), np.array(degrees, float)
    edges = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 3000]
    mids, med, q25, q75, top = [], [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (ages >= a) & (ages < b)
        if m.sum() >= 30:
            mids.append(np.sqrt(max(a, 0.5) * b))
            med.append(float(np.median(degrees[m])))
            q25.append(float(np.quantile(degrees[m], 0.25)))
            q75.append(float(np.quantile(degrees[m], 0.75)))
            top.append(float(np.quantile(degrees[m], 0.99)))
    ch.figure(
        "age", title="Connections and age",
        x={"label": "age of the agent, iterations (logarithmic)", "log": True, "min": 0.5, "max": 3000},
        y={"label": "connections (logarithmic)", "log": True, "min": 0.8, "max": 500},
        series=[{"label": "median, with the middle half", "x": mids, "y": med, "lo": q25, "hi": q75,
                 "colour": BLUE, "width": 2},
                line(mids, top, "the best-connected hundredth", ORANGE, width=1.5, dash=[5, 4])],
        caption="The agents alive after the last game of the 26 worlds that lived to the end, sorted by age — "
                "the iterations since their node was born — in classes doubling in width: the median number "
                "of connections of each class, with the band from its 25th to its 75th percentile, and the 99th "
                "percentile (dashed).",
        recipe=recipe(SURVIVORS, FRAMES,
                      ["Read frame 5,999 of each run: `ages`, and each agent's connections in `edges`.",
                       "Classes of age [0, 1), [1, 2), [2, 4), …, [1,024, 3,000); in each class with at least 30 "
                       "agents, the median, quartiles and 99th percentile of the connections."]))
    ch.number("age", {"mids": mids, "median": med, "top": top,
                      "corr_log": float(np.corrcoef(np.log1p(ages), np.log(degrees))[0, 1])})
