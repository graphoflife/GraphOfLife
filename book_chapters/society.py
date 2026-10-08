# -*- coding: utf-8 -*-
"""
Part II, how a world is organised: Chapters 13 to 16 — wealth, the game, the
shape of the network, genotypes and lineages.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np

import book_figures as F
from book_figures import (BASELINE_RUNS, FRAMES, STATS_FILE, band_series, chapter, describe, dots,
                          line, mean_over, recipe, series, survivors)
from book_chapters.common import (BAND_STEPS, BAND_WORDS, BLUE, CYAN, GREEN, GREY, ORANGE, RED,
                                  VIOLET, YELLOW, baseline)

# ---------------------------------------------------------------------------
# Chapter 13 · Where do the tokens go?
# ---------------------------------------------------------------------------

def lorenz(tokens: List[int], points: int = 101) -> np.ndarray:
    """The Lorenz curve at agent shares 0, 1/(points−1), …, 1, by linear interpolation."""
    t = np.sort(np.asarray(tokens, dtype=float))
    cum = np.concatenate([[0.0], np.cumsum(t)]) / t.sum()
    share = np.linspace(0, 1, len(cum))
    return np.interp(np.linspace(0, 1, points), share, cum)


@chapter
def tokens(ch: F.Chapter) -> None:
    specs = baseline()
    alive = survivors(specs)
    grid = np.linspace(0, 1, 101)
    curves, all_curves = [], []
    for s in alive:
        L = lorenz(F.world_pass(s.run_id)["last"]["tokens"])
        all_curves.append(L)
        curves.append(line(grid, L, None, BLUE, width=0.8, alpha=0.5))
    curves.append(line(grid, np.median(all_curves, axis=0), "median of the 26 worlds", YELLOW, width=3))
    curves.append(line([0, 1], [0, 1], "everyone holding the same", GREY, width=1, dash=[4, 4]))
    ch.figure(
        "lorenz", title="Lorenz curves at iteration 2,999",
        x={"label": "share of agents, poorest first", "min": 0, "max": 1},
        y={"label": "share of all tokens they hold", "min": 0, "max": 1},
        series=curves,
        caption="For each of the 26 surviving worlds (thin blue lines), after its last game: the "
                "poorest share x of its agents holds the share y of all tokens. The yellow line "
                "is the median over the worlds at each x; the dashed diagonal is a world where "
                "everyone holds the same. The further a curve sags below the diagonal, the more "
                "unequal the world (Chapter 5, and the diagram there).",
        recipe=recipe(BASELINE_RUNS.replace("The 30", "The 26 surviving ones of the 30"), FRAMES,
                      ["Read each run's last frame (iteration 2,999, `phase` 2) and its `tokens`.",
                       "Sort the tokens from smallest to largest; the curve passes through the "
                       "points (i / n, (t₁ + … + tᵢ) / T) for i = 0 … n.",
                       "Interpolate each curve at x = 0, 0.01, …, 1 and take the median there."]))
    med = np.median(all_curves, axis=0)
    ch.number("lorenz_median", {f"{int(round(100 * x))}%": float(y) for x, y in zip(grid, med) if round(100 * x) % 10 == 0})

    pooled = np.concatenate([F.world_pass(s.run_id)["last"]["tokens"] for s in alive])
    xs = np.unique(pooled)
    ccdf = np.array([(pooled >= x).mean() for x in xs])
    ch.figure(
        "distribution", title="How many tokens agents hold",
        x={"label": "tokens x (logarithmic)", "log": True, "min": 1},
        y={"label": "share of agents holding at least x (logarithmic)", "log": True, "max": 1},
        series=[line(xs, ccdf, f"{len(pooled):,} agents of 26 worlds, after the last game", BLUE, width=2)],
        caption="Of all agents alive after the last game of the 26 surviving worlds, the share that "
                "hold at least x tokens, for every x. Both axes are logarithmic: a straight falling "
                "line here would be a power law.",
        recipe=recipe(BASELINE_RUNS.replace("The 30", "The 26 surviving ones of the 30"), FRAMES,
                      ["Pool the `tokens` of the last frame of every run.",
                       "For every value x that occurs, the share of agents with at least x tokens."]))
    ch.number("tokens_end", {"agents": int(len(pooled)), "median": float(np.median(pooled)),
                             "mean": float(pooled.mean()), "share_le_5": float(np.mean(pooled <= 5)),
                             "share_ge_100": float(np.mean(pooled >= 100)), "max": float(pooled.max()),
                             "share_1": float(np.mean(pooled == 1))})

    for name, stat, title, ylabel, colour in (
            ("gini", "gini", "Inequality of wealth (Gini)", "Gini coefficient", YELLOW),
            ("rich-tenth", "topDecileShare", "Share of all tokens held by the richest tenth", "share", ORANGE)):
        b = band_series([series(s.run_id, stat) for s in specs], "30 worlds: median, middle half, nine in ten", colour)
        ch.figure(name, title=title, x={"label": "iteration"}, y={"label": ylabel, "min": 0},
                  series=[b],
                  caption=f"After every game. {BAND_WORDS}.",
                  recipe=recipe(BASELINE_RUNS, STATS_FILE,
                                [f"Take every run's rows with `phase` = 2 and the statistic `{stat}` ([What a run records](../notes/frames-and-stats.md)).",
                                 *BAND_STEPS]))
        ch.number(f"{name}_settled", describe([mean_over(s.run_id, stat, 500, 2999) for s in alive]))
        ch.number(f"{name}_first", F.A.bands([series(s.run_id, stat) for s in specs])["y"][0])

    richest = band_series([(series(s.run_id, "maxTokens")[0], series(s.run_id, "maxTokens")[1] / 10000) for s in specs],
                          "the richest single agent's share", RED)
    ch.figure("richest", title="The richest agent's share of all tokens", x={"label": "iteration"},
              y={"label": "share of all tokens", "min": 0}, series=[richest],
              caption=f"After every game, the tokens of the richest agent divided by all 10,000. {BAND_WORDS}.",
              recipe=recipe(BASELINE_RUNS, STATS_FILE,
                            ["Take the rows with `phase` = 2; divide `maxTokens` by 10,000.", *BAND_STEPS]))
    ch.number("richest_settled", describe([mean_over(s.run_id, "maxTokens", 500, 2999) / 10000 for s in alive]))

    med = band_series([series(s.run_id, "medianTokens") for s in specs], "median agent's tokens", GREEN)
    mean = band_series([series(s.run_id, "meanTokens") for s in specs], "mean tokens per agent (10,000 ÷ agents)", BLUE)
    ch.figure("typical", title="What a typical agent holds", x={"label": "iteration"},
              y={"label": "tokens", "min": 0, "max": 40}, series=[mean, med],
              caption="After every game: the mean number of tokens per agent, which is 10,000 divided "
                      "by the number of agents (blue), and the median agent's tokens (green). The "
                      f"median lies below the mean because a few agents hold a lot. {BAND_WORDS}.",
              recipe=recipe(BASELINE_RUNS, STATS_FILE,
                            ["Take the rows with `phase` = 2 and the statistics `meanTokens` and `medianTokens`.",
                             *BAND_STEPS]))
    ch.number("median_tokens_settled", describe([mean_over(s.run_id, "medianTokens", 500, 2999) for s in alive]))
    ch.number("mean_tokens_settled", describe([mean_over(s.run_id, "meanTokens", 500, 2999) for s in alive]))

    w = F.world_pass("B1-10000-s001")
    t, d = np.array(w["last"]["tokens"], float), np.array(w["last"]["degrees"], float)
    rng = np.random.default_rng(1)
    ch.figure(
        "tokens-degree", title="Tokens against connections, world of seed 1 at the end",
        x={"label": "connections (degree)", "min": 0}, y={"label": "tokens (logarithmic)", "log": True, "min": 1},
        series=[{"label": None, "x": (d + rng.uniform(-0.25, 0.25, d.size)).tolist(), "y": t.tolist(),
                 "points": True, "size": 4, "colour": BLUE, "alpha": 0.5}], legend=False,
        caption=f"Each dot is one of the {len(t):,} agents alive after the last game of the world "
                "with seed 1: how many connections it has (spread sideways a little so that agents "
                "with the same number do not hide each other) and how many tokens it holds.",
        recipe=recipe("`B1-10000-s001`.", FRAMES,
                      ["Read the last frame (iteration 2,999, `phase` 2); count each agent's connections in `edges`.",
                       "Plot `tokens` against that count, one dot per agent."]))
    ch.number("tokens_degree_corr", float(np.corrcoef(d, np.log(t))[0, 1]))
    by_degree = {int(k): float(np.median(t[d == k])) for k in sorted(set(d.astype(int))) if np.sum(d == k) >= 10}
    ch.number("tokens_by_degree_median", by_degree)

    red = band_series([series(s.run_id, "redistributed") for s in specs], "tokens shared out", VIOLET)
    ch.figure("shared-out", title="Tokens of the dead, shared out in each game", x={"label": "iteration"},
              y={"label": "tokens", "min": 0}, series=[red],
              caption="In the cleanup of every game, the tokens of the agents removed — those cut off "
                      "from the largest piece; agents that starved hold none — which are shared out at "
                      f"random among the survivors. {BAND_WORDS}.",
              recipe=recipe(BASELINE_RUNS, STATS_FILE,
                            ["Take the rows with `phase` = 2 and the statistic `redistributed`.", *BAND_STEPS]))
    ch.number("shared_out_settled", describe([mean_over(s.run_id, "redistributed", 500, 2999) for s in alive]))


# ---------------------------------------------------------------------------
# Chapter 14 · How the game is played
# ---------------------------------------------------------------------------

@chapter
def game(ch: F.Chapter) -> None:
    specs = baseline()
    alive = survivors(specs)
    first, last = [], []
    for s in specs:
        first += F.world_pass(s.run_id)["stakes"].get("first", [])
    for s in alive:
        last += F.world_pass(s.run_id)["stakes"].get("last", [])
    edges = np.array([0, 1e-9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1 - 1e-9, 1.0 + 1e-9])
    labels = ["0", "0–0.1", "0.1–0.2", "0.2–0.3", "0.3–0.4", "0.4–0.5", "0.5–0.6", "0.6–0.7",
              "0.7–0.8", "0.8–0.9", "0.9–1", "1"]
    def hist(values):
        v = np.array(values)
        counts = [np.mean(v == 0)] + [np.mean((v > lo) & (v <= hi)) for lo, hi in
                                      zip([0] + list(np.arange(0.1, 1.0, 0.1)), list(np.arange(0.1, 1.0, 0.1)) + [1 - 1e-12])]
        counts[-1] = np.mean((v > 0.9) & (v < 1))
        return counts + [np.mean(v == 1)]
    h0, h1 = hist(first), hist(last)
    k = np.arange(len(labels))
    ch.figure(
        "home-stake", title="How much of its stake an agent puts on its own node",
        x={"label": "share of the agent's tokens staked on its own node", "categories": labels},
        y={"label": "share of agents", "min": 0},
        series=[{"label": f"the first game: {len(first):,} agents of 30 worlds", "kind": "bars",
                 "x0": (k - 0.4).tolist(), "x1": k.tolist(), "y": h0, "colour": GREY},
                {"label": f"the last game: {len(last):,} agents of 26 worlds", "kind": "bars",
                 "x0": k.tolist(), "x1": (k + 0.4).tolist(), "y": h1, "colour": BLUE}],
        caption="For every agent that staked in a game: the share of its tokens it put on its own "
                "node. '0' and '1' are exact: nothing at home, everything at home. Grey: the "
                "game of iteration 0, played by the founders and their first children, whose "
                "brains no selection has touched yet. Blue: the game of iteration 2,999.",
        recipe=recipe(BASELINE_RUNS, FRAMES,
                      ["In the frames of iteration 0 and 2,999 with `phase` 2, read "
                       "`decisions.allocations`: for each agent, `alloc[0]` is what it staked on its own "
                       "node (its first target is itself) and `tokens` all it staked.",
                       "Put `alloc[0] / tokens` into the bins; divide each count by the number of agents."]))
    ch.number("home_stake", {"first_mean": float(np.mean(first)), "last_mean": float(np.mean(last)),
                             "first_all_home": float(np.mean(np.array(first) == 1)),
                             "last_all_home": float(np.mean(np.array(last) == 1)),
                             "first_none_home": float(np.mean(np.array(first) == 0)),
                             "last_none_home": float(np.mean(np.array(last) == 0))})

    # Who won each node, in the first game and the last.
    kinds = ["its own agent, as the largest staker", "its own agent, with a coalition",
             "a neighbour, as the largest staker", "a neighbour, with a coalition"]

    def winners(run_ids, iteration):
        counts = Counter()
        for run_id in run_ids:
            frame = F.store.read_frame(run_id, 2 * iteration + 1)
            for w in (frame.get("decisions") or {}).get("winners") or []:
                counts[(w["winner"] != w["node"]) * 2 + bool(w.get("revolt"))] += 1
        total = sum(counts.values())
        return [counts[i] / total for i in range(4)], total

    w0, n0 = winners([s.run_id for s in specs], 0)
    w1, n1 = winners([s.run_id for s in alive], 2999)
    stack = []
    below = np.zeros(2)
    for i, label in enumerate(kinds):
        top = below + np.array([w0[i], w1[i]])
        stack.append({"label": label, "kind": "bars", "x0": [-0.3, 0.7], "x1": [0.3, 1.3],
                      "y0": below.tolist(), "y": top.tolist(), "colour": [GREEN, CYAN, ORANGE, VIOLET][i],
                      "alpha": 0.9})
        below = top
    ch.figure(
        "winners", title="Who wins a node",
        x={"label": "", "categories": [f"the first game ({n0:,} nodes, 30 worlds)",
                                       f"the last game ({n1:,} nodes, 26 worlds)"]},
        y={"label": "share of the nodes staked on", "min": 0, "max": 1}, series=stack,
        caption="Every node on which anyone staked in the game, by who won it: the agent living on "
                "it (green and cyan) or one of its neighbours (orange and violet), and whether the "
                "winner was the largest single staker (green, orange) or the member of a coalition "
                "that outweighed the largest staker (cyan, violet; see [How a coalition takes a "
                "node](../notes/revolution.md)). Each column adds up to 1. Left: the game of "
                "iteration 0; right: the game of iteration 2,999.",
        recipe=recipe(BASELINE_RUNS, FRAMES,
                      ["Read frame 1 of each of the 30 runs (the game of iteration 0) and frame 5,999 of "
                       "the 26 that reached it (the game of iteration 2,999).",
                       "For every entry of `decisions.winners`, note whether `winner` is the `node` itself, "
                       "and whether `revolt` is true (the node was won by a coalition).",
                       "Count the four kinds over all runs of a column, divide by the number of entries, "
                       "and stack them."]))
    ch.number("winners", {"first": dict(zip(kinds, w0)), "last": dict(zip(kinds, w1)),
                          "first_nodes": n0, "last_nodes": n1})

    named = {"heldHomeShare": "nodes won by their own agent",
             "revolutions/nodes_before": "nodes won by a coalition, per agent",
             "spreadShare": "agents who spread their stake",
             "selfAllocationShare": "staked tokens put at home",
             "revoltShare": "staked tokens marked revolutionary"}
    for name, stats, title, ylabel, words in (
            ("kept-taken", [("heldHomeShare", GREEN), ("revolutions/nodes_before", VIOLET)],
             "Who wins the nodes, over a world's life", "share",
             "Green: of all nodes on which anyone staked, the share won by the agent living on it "
             "(`heldHomeShare`). Violet: the number of nodes won by a coalition, divided by the "
             "number of agents at the start of the game (`revolutions/nodes_before`)."),
            ("doctrine", [("spreadShare", BLUE), ("selfAllocationShare", YELLOW), ("revoltShare", RED)],
             "How agents stake, over a world's life", "share",
             "Blue: the share of the agents that staked who spread their tokens over several "
             "candidates rather than putting all on one (`spreadShare`). Yellow: of all tokens staked, "
             "the share staked by agents on their own node (`selfAllocationShare`). Red: of all "
             "tokens staked, the share marked revolutionary (`revoltShare`).")):
        out = []
        for stat, colour in stats:
            out.append(band_series([series(s.run_id, stat) for s in specs], named[stat], colour))
            ch.number(f"{stat}_settled", describe([mean_over(s.run_id, stat, 500, 2999) for s in alive]))
            ch.number(f"{stat}_first", F.A.bands([series(s.run_id, stat) for s in specs])["y"][0])
        ch.figure(name, title=title, x={"label": "iteration"}, y={"label": ylabel, "min": 0, "max": 1},
                  series=out,
                  caption=f"After every game of the 30 baseline worlds. {words} For each, {BAND_WORDS[0].lower()}"
                          f"{BAND_WORDS[1:]} ([Bands](../notes/bands.md)).",
                  recipe=recipe(BASELINE_RUNS, STATS_FILE,
                                ["Take the rows with `phase` = 2 and the statistics "
                                 + ", ".join(f"`{s}`" for s, _ in stats)
                                 + " (`a/b` is the row's `a` divided by its `b`; "
                                 "[Staking at home](../notes/home-stake.md) defines each).", *BAND_STEPS]))

    # Where each world settles, statistic by statistic.
    order = ["spreadShare", "revoltShare", "heldHomeShare", "revolutions/nodes_before", "selfAllocationShare"]
    short = ["spread their stake", "tokens revolutionary", "nodes kept", "nodes to coalitions",
             "tokens at home"]
    groups = [[mean_over(s.run_id, stat, 500, 2999) for s in alive] for stat in order]
    ch.figure(
        "settled", title="How the game is played in each settled world",
        x={"label": "", "categories": short}, y={"label": "share (mean over iterations 500–2,999)",
                                                 "min": 0, "max": 1},
        series=dots(short, groups, [BLUE, RED, GREEN, VIOLET, YELLOW]), legend=False,
        caption="One dot per world that lived to the end: its mean over iterations 500 to 2,999 of "
                "each statistic of the two figures above; the bar is the median of the 26. From left: "
                "the share of stakers who spread their stake, the share of staked tokens marked "
                "revolutionary, the share of nodes kept by their own agent, nodes won by a coalition "
                "per agent, and the share of staked tokens put at home. The dots are spread "
                "sideways only so that they do not hide each other.",
        recipe=recipe(BASELINE_RUNS.replace("The 30", "The 26 surviving ones of the 30"), STATS_FILE,
                      ["For each run, average each statistic over the rows with `phase` = 2 and "
                       "500 ≤ `iteration` ≤ 2,999.",
                       "Draw one dot per run and statistic, and a bar at the median."]))


# ---------------------------------------------------------------------------
# Chapter 15 · What shape does the network take?
# ---------------------------------------------------------------------------

def roles(ids: List[int], edges: List[List[int]]) -> Dict[int, int]:
    """0 for an agent in the 2-core, 1 for one on a hanging tree, 2 for a leaf."""
    adj = {i: set() for i in ids}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    degree = {u: len(adj[u]) for u in ids}
    peeled, queue = set(), [u for u in ids if degree[u] <= 1]
    while queue:
        u = queue.pop()
        if u in peeled:
            continue
        peeled.add(u)
        for v in adj[u]:
            if v not in peeled:
                degree[v] -= 1
                if degree[v] == 1:
                    queue.append(v)
    return {u: 0 if u not in peeled else 2 if len(adj[u]) == 1 else 1 for u in ids}


ROLE_WORDS = ("Blue: agents in the core, which is what is left when agents with one connection or "
              "none are removed again and again until none remains ([Core, trees and leaves]"
              "(../notes/core-trees-leaves.md)); yellow: the rest of the hanging trees; red: leaves, "
              "agents with a single connection. Where a dot is drawn means nothing in itself: a "
              "layout (networkx's ForceAtlas2, seed 1) pulls joined agents together and pushes the "
              "rest apart.")
ROLE_STEPS = ("Peel: remove every agent with one connection or none, again and again, until none is "
              "left; the agents never removed are the core. Leaves are agents with exactly one connection.",
              "Lay the graph out with `networkx.forceatlas2_layout(G, max_iter=200, seed=1)`.")


def picture(run_id: str, iteration: int) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """A world after the game of `iteration`, as a network chart, and the shares of its roles."""
    import networkx as nx
    frame = F.store.read_frame(run_id, 2 * iteration + 1)
    ids, edges = frame["ids"], frame["edges"]
    G = nx.Graph()
    G.add_nodes_from(ids)
    G.add_edges_from(edges)
    pos = nx.forceatlas2_layout(G, max_iter=200, seed=1)
    index = {u: i for i, u in enumerate(ids)}
    role = roles(ids, edges)
    chart = dict(kind="network", title=f"Iteration {iteration:,}: {len(ids):,} agents",
                 nodes={"x": [float(pos[u][0]) for u in ids], "y": [float(pos[u][1]) for u in ids],
                        "group": [role[u] for u in ids]},
                 edges=[[index[a], index[b]] for a, b in edges],
                 colour={"labels": ["core: on a loop or between loops", "on a hanging tree",
                                    "leaf: a single connection"]})
    n = len(ids)
    return chart, {"agents": n, "edges": len(edges), "core": sum(1 for v in role.values() if v == 0) / n,
                   "tree": sum(1 for v in role.values() if v == 1) / n,
                   "leaves": sum(1 for v in role.values() if v == 2) / n}


@chapter
def shape(ch: F.Chapter) -> None:
    specs = baseline()
    alive = survivors(specs)
    charts = []
    for t in (0, 100, 1000, 2999):
        chart, counts = picture("B1-10000-s001", t)
        charts.append(chart)
        ch.number(f"picture_{t}", counts)
    ch.grid(
        "worlds", [dict(c, legend=False) for c in charts[:3]] + [charts[3]], columns=2,
        title="One world at four moments",
        caption="The world with seed 1 after the games of iterations 0, 100, 1,000 and 2,999. Every "
                f"agent alive is a dot, every connection a line. {ROLE_WORDS}",
        recipe=recipe("`B1-10000-s001`, one of the 30 baseline runs.", FRAMES,
                      ["Read frames `1`, `201`, `2001` and `5999` (iterations 0, 100, 1,000 and 2,999, "
                       "after the game): `ids` and `edges`.", *ROLE_STEPS]))
    ch.figure(
        "world-2999", **dict(charts[3], title="World of seed 1 after the game of iteration 2,999"),
        caption="Every agent alive after the game of iteration 2,999 is a dot, every connection a "
                f"line. {ROLE_WORDS}",
        recipe=recipe("`B1-10000-s001`.", FRAMES,
                      ["Read frame `5999` (iteration 2,999, after the game): `ids` and `edges`.",
                       *ROLE_STEPS]))

    pooled = np.concatenate([F.world_pass(s.run_id)["last"]["degrees"] for s in alive])
    ds = np.arange(1, pooled.max() + 1)
    ch.figure(
        "degree", title="How many connections agents have",
        x={"label": "connections d (logarithmic)", "log": True, "min": 1},
        y={"label": "share of agents with at least d (logarithmic)", "log": True, "max": 1},
        series=[line(ds, [(pooled >= d).mean() for d in ds], f"{len(pooled):,} agents of 26 worlds, after the last game",
                     BLUE, width=2)],
        guides=[{"axis": "x", "at": 4, "label": "every founder at the start"}],
        caption="Of all agents alive after the last game of the 26 surviving worlds, the share "
                "that have at least d connections, for every d. At the start every founder had "
                "exactly four (the grey line).",
        recipe=recipe(BASELINE_RUNS.replace("The 30", "The 26 surviving ones of the 30"), FRAMES,
                      ["Count every agent's connections in the last frame's `edges`; pool the 26 runs.",
                       "For d = 1, 2, …, the share of agents with at least d."]))
    ch.number("degree_end", {"agents": int(len(pooled)), "mean": float(pooled.mean()),
                             "share_1": float(np.mean(pooled == 1)), "share_2": float(np.mean(pooled == 2)),
                             "share_ge_10": float(np.mean(pooled >= 10)), "max": int(pooled.max())})

    panels = []
    for stat, label, colour in (("coreShare", "Agents in the core", BLUE),
                                ("leaves/nodes", "Agents that are leaves", RED),
                                ("bridges/edges", "Bridges", YELLOW)):
        panels.append(dict(title=label, x={"label": "iteration"}, y={"label": "share", "min": 0, "max": 1},
                           series=[band_series([series(s.run_id, stat) for s in specs], None, colour)],
                           legend=False))
        ch.number(f"{stat}_settled", describe([mean_over(s.run_id, stat, 500, 2999) for s in alive]))
        ch.number(f"{stat}_first", F.A.bands([series(s.run_id, stat) for s in specs])["y"][0])
    ch.grid(
        "structure", panels, columns=3, title="Core, leaves and bridges",
        caption="Three shares, after every game of the 30 baseline worlds. Left: the share of agents "
                "in the core ([Core, trees and leaves](../notes/core-trees-leaves.md)), measured every "
                "25 iterations. Middle: the share of agents with exactly one connection, after every "
                "game. Right: the share of connections that are bridges ([Bridges](../notes/bridges.md)), "
                f"every 25 iterations. In each, {BAND_WORDS[0].lower()}{BAND_WORDS[1:]}.",
        recipe=recipe(BASELINE_RUNS, STATS_FILE,
                      ["Take the rows with `phase` = 2 and the statistics `coreShare`, `leaves/nodes` and "
                       "`bridges/edges` (`a/b` is the row's `a` divided by its `b`).",
                       "For `leaves/nodes`, make the band as for every other band "
                       "([Bands](../notes/bands.md)): stretches of 5 iterations.",
                       "For `coreShare` and `bridges/edges`, which are measured every 25 iterations, "
                       "use stretches of 25."]))
    clus = band_series([series(s.run_id, "transitivity") for s in specs], "clustering (transitivity)", GREEN)
    ch.figure("clustering", title="Clustering", x={"label": "iteration"}, y={"label": "transitivity", "min": 0},
              series=[clus],
              caption="Three times the number of triangles divided by the number of connected triples "
                      "([Clustering](../notes/clustering.md)), every 25 iterations. The founders' ring starts near 0.24. "
                      f"{BAND_WORDS}.",
              recipe=recipe(BASELINE_RUNS, STATS_FILE,
                            ["Take the rows with `phase` = 2 and the statistic `transitivity`.",
                             "Bands as above, in stretches of 25."]))
    ch.number("transitivity_settled", describe([mean_over(s.run_id, "transitivity", 500, 2999) for s in alive]))

    n = np.array([mean_over(s.run_id, "nodes", 500, 2999) for s in alive])
    k = np.array([mean_over(s.run_id, "meanDegree", 500, 2999) for s in alive])
    L = np.array([mean_over(s.run_id, "meanPathLength", 500, 2999) for s in alive])
    rnd = np.log(n) / np.log(k)
    ch.figure(
        "distance", title="Steps between two agents",
        x={"label": "agents (mean over iterations 500–2,999)", "min": 0},
        y={"label": "mean steps between two agents", "min": 0},
        series=[{"label": "each world: measured", "x": n.tolist(), "y": L.tolist(), "points": True, "size": 7, "colour": BLUE},
                {"label": "each world: a random network of the same size, ln N ÷ ln k", "x": n.tolist(),
                 "y": rnd.tolist(), "points": True, "size": 5, "colour": GREY}],
        caption="For each surviving world (blue): its mean number of agents and the mean number "
                "of steps along connections from one agent to another, both averaged over "
                "iterations 500–2,999. Grey: the rough distance in a random network with the "
                "same number of agents N and connections per agent k, ln N / ln k.",
        recipe=recipe(BASELINE_RUNS.replace("The 30", "The 26 surviving ones of the 30"), STATS_FILE,
                      ["Average `nodes`, `meanDegree` and `meanPathLength` over the rows with "
                       "`phase` = 2 and 500 ≤ `iteration` ≤ 2,999 (`meanPathLength` is measured every "
                       "25 iterations, from breadth-first searches out of 8 to 16 evenly spread agents; [Path length](../notes/path-length.md)).",
                       "Plot one dot per run, and ln N / ln k for the same run."]))
    ch.number("distance", {"measured": describe(L), "random": describe(rnd), "ratio": describe(L / rnd)})


# ---------------------------------------------------------------------------
# Chapter 16 · Genotypes and lineages
# ---------------------------------------------------------------------------

@chapter
def lineage(ch: F.Chapter) -> None:
    specs = baseline()
    alive = survivors(specs)
    g_lives, a_lives = [], []
    for s in alive:
        w = F.world_pass(s.run_id)
        g_lives += w["genotypes"]
        a_lives += w["agents"]
    out = []
    for lives, label, colour in ((g_lives, f"genotypes ({len(g_lives):,})", VIOLET),
                                 (a_lives, f"agents ({len(a_lives):,})", BLUE)):
        at, surv = F.kaplan_meier(lives)
        keep = at <= 3000
        out.append(line(at[keep], surv[keep], label, colour, width=2))
        ch.number(f"km_{label.split()[0]}", {f"at_least_{L}": float(surv[np.searchsorted(at, L, side='right') - 1])
                                            for L in (2, 3, 5, 10, 20, 50, 100)})
    ch.figure(
        "lives", title="How long genotypes and agents last",
        x={"label": "lifetime L, in iterations (logarithmic)", "log": True, "min": 1},
        y={"label": "share lasting at least L (logarithmic)", "log": True, "min": 1e-6, "max": 1},
        series=out,
        caption="Of all genotypes that appeared, and all agents born, at iteration 500 or later in "
                "the 26 worlds that lived to the end, the share that lasted at least L iterations. "
                "A genotype lasts from the first game after which some agent carries it to the last; "
                "an agent from its birth to the last game it is alive after. Lives still going at the "
                "end are counted as far as they went (Kaplan–Meier).",
        recipe=recipe(BASELINE_RUNS.replace("The 30", "The 26 surviving ones of the 30"), FRAMES,
                      ["Read every frame with `phase` = 2; for every genotype (brain id in `brain_ids`) "
                       "and every agent (id in `ids`) note the first and last iteration it appears.",
                       "Keep those first seen at iteration 500 or later; one still present in the "
                       "last frame is censored.",
                       "Kaplan–Meier as in Chapter 11."]))

    s12 = F.world_pass("B1-10000-s012", (2550, 2999))
    top = band_series([(np.array([t for t, _ in F.world_pass(s.run_id)["top"]], float),
                        np.array([v for _, v in F.world_pass(s.run_id)["top"]], float)) for s in specs],
                      "30 worlds: median, middle half, nine in ten", VIOLET)
    single = line([t for t, _ in s12["top"]], [v for _, v in s12["top"]], "the world of seed 12", YELLOW,
                  width=0.8, alpha=0.9)
    ch.figure("top-share", title="The most common genotype's share of the world", x={"label": "iteration"},
              y={"label": "share of agents", "min": 0, "max": 0.7}, series=[top, single],
              caption="After every game, the share of the living that carry the most common genotype. "
                      f"{BAND_WORDS}. The yellow line is the single world with seed 12.",
              recipe=recipe(BASELINE_RUNS, FRAMES,
                            ["After every game, count the agents per genotype in `brain_ids`; the "
                             "largest count divided by the number of agents.", *BAND_STEPS]))

    depth = []
    for k, seed in enumerate((1, 12, 17)):
        w = F.world_pass(f"B1-10000-s{seed:03d}")
        xs = [t for t, *_ in w["ancestry"]]
        ys = [t if d is None else d for t, d, *_ in w["ancestry"]]
        depth.append(line(xs, ys, f"seed {seed}", [BLUE, YELLOW, GREEN][k], width=2))
    ch.figure("ancestor", title="How far back everyone shares one ancestor", x={"label": "iteration"},
              y={"label": "iterations back", "min": 0}, series=depth,
              guides=[{"axis": "y", "at": 0}],
              caption="Every 25 iterations, for three worlds (seeds 1, 12 and 17, one colour each): how "
                      "many iterations earlier the newest genotype lived from which every living agent "
                      "descends ([The common ancestor](../notes/common-ancestor.md)). While a line rises "
                      "by 25 every 25 iterations, that ancestor stays the same and simply grows older; "
                      "each drop is a moment when it moves forward to a younger genotype, because all "
                      "the other branches of the family tree have died out. Before the first drop the "
                      "living descend from more than one founder, and the line is drawn at the age of "
                      "the world.",
              recipe=recipe("`B1-10000-s001`, `-s012` and `-s017`.", FRAMES,
                            ["Read every frame in order; remember each genotype's parent and the "
                             "iteration it first appeared.",
                             "Every 25 iterations, after the game, count the agents per living genotype "
                             "and climb the family tree from them, newest genotype first, adding each "
                             "genotype's count to its parent's ([The common ancestor](../notes/common-ancestor.md)).",
                             "The first genotype reached that carries all agents is their newest common "
                             "ancestor; plot the iterations since it first appeared."]))

    for seed, window in ((12, (2550, 2999)), (1, (1000, 1499))):
        w = F.world_pass(f"B1-10000-s{seed:03d}", window)
        m = w["muller"]
        xs = [t for t, n, _ in m["series"]]
        shares = np.array([[c / n for c in per] for t, n, per in m["series"]])
        bottom = np.zeros(len(xs))
        areas = []
        for rank in range(min(7, shares.shape[1])):
            hi = bottom + shares[:, rank]
            areas.append({"label": f"family {rank + 1}", "kind": "area", "x": xs, "lo": bottom.tolist(),
                          "hi": hi.tolist(), "colour": rank, "alpha": 0.9})
            bottom = hi
        areas.append({"label": "all other families", "kind": "area", "x": xs, "lo": bottom.tolist(),
                      "hi": [1.0] * len(xs), "colour": "#5b6b7c", "alpha": 0.9})
        ch.figure(
            f"families-{seed}", title=f"Families of iteration {window[0]:,}, world of seed {seed}",
            x={"label": "iteration", "min": window[0], "max": window[1]},
            y={"label": "share of the living", "min": 0, "max": 1}, series=areas,
            caption=f"Every genotype alive after the game of iteration {window[0]:,} founds a family: "
                    "itself and every genotype descended from it. After every later game, the share of "
                    "the living in each family, stacked; the seven families that ever held the largest "
                    "share in colour, all others together in grey.",
            recipe=recipe(f"`B1-10000-s{seed:03d}`.", FRAMES,
                          [f"Note the genotypes alive after the game of iteration {window[0]:,}.",
                           "After every later game, follow each living agent's genotype up its parents "
                           "until one of those is reached; count the agents per family and divide by "
                           "the number of agents.",
                           "Stack the shares, the families with the largest share ever at the bottom."]))
        top_family = shares[:, 0] if shares.size else np.zeros(1)
        ch.number(f"families_{seed}", {"families_alive_at_start": None, "largest_final_share": float(top_family[-1]),
                                       "largest_max_share": float(top_family.max()),
                                       "from_half": next((int(t) for t, v in zip(xs, top_family) if v >= 0.5), None),
                                       "to_all": next((int(t) for t, v in zip(xs, top_family) if v >= 0.999), None)})

    import json
    runs = json.load(open(F.os.path.join(F.BOOK, "results", "E06.json")))["lineage"]["baseline"]["runs"]
    alive_ids = {s.run_id for s in alive}
    moves = [v["ancestor"]["moves"] for k, v in runs.items() if k in alive_ids]
    ch.figure("moves", title="How often one branch replaces all others", x={"label": "", "categories": ["26 worlds"]},
              y={"label": "moves of the common ancestor after iteration 100", "min": 0},
              series=dots(["26 worlds"], [moves], [GREEN]), legend=False,
              caption="One dot per surviving world: how many times, between iteration 100 and the "
                      "end, the newest common ancestor of all the living moved forward to a younger "
                      "genotype (one of the drops in the figure above). The bar is the median.",
              recipe=recipe(BASELINE_RUNS.replace("The 30", "The 26 surviving ones of the 30"),
                            "the lineage analysis of Experiment 6 (`python3 gol_lab.py analyse E06`), "
                            "in `book/results/E06.json` (`ancestor.moves` of every run)",
                            ["Take the common-ancestor depth every 25 iterations as above.",
                             "Count the checks after iteration 100 at which the depth is smaller than "
                             "the previous depth plus 25."]))
    ch.number("moves", describe(moves))
