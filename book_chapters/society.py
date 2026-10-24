# -*- coding: utf-8 -*-
"""
the study of the baseline, how a world is organised: Pages 15 to 18 — wealth, the game, the
shape of the network, genotypes and lineages.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np

import book_data as D
import book_figures as F
from book_figures import (FRAMES, STATS_FILE, band_series, chapter, describe, dots, line, mean_over,
                          recipe, series, stacked_areas)
from book_graph import ball
from book_chapters.common import (BAND_STEPS, BAND_WORDS, BLUE, CYAN, GREEN, GREY, ORANGE, RED, STAR,
                                  VIOLET, WORLDS as W, YELLOW, at_start)

#: The single worlds the lineage figures follow, chosen for what each shows: the world whose most
#: common genotype held the largest share, a world one family took over within the window and one no
#: family took over, and three for the common ancestor.
EXAMPLES = {
    "E02": {"top": 12, "families": ((12, (2550, 2999)), (1, (1000, 1499))), "ancestry": (1, 12, 17)},
    "E10": {"top": 28, "families": ((12, (550, 1499)), (1, (1000, 1499))), "ancestry": (1, 12, 17)},
}[W.experiment]

# ---------------------------------------------------------------------------
# Where do the tokens go?
# ---------------------------------------------------------------------------

def lorenz(tokens: List[int], points: int = 101) -> np.ndarray:
    """The Lorenz curve at agent shares 0, 1/(points−1), …, 1, by linear interpolation."""
    t = np.sort(np.asarray(tokens, dtype=float))
    cum = np.concatenate([[0.0], np.cumsum(t)]) / t.sum()
    share = np.linspace(0, 1, len(cum))
    return np.interp(np.linspace(0, 1, points), share, cum)


@chapter
def tokens(ch: F.Chapter) -> None:
    specs = W.specs
    alive = W.alive
    n = len(alive)
    grid = np.linspace(0, 1, 101)
    curves, all_curves = [], []
    for s in alive:
        L = lorenz(F.world_pass(s.run_id)["last"]["tokens"])
        all_curves.append(L)
        curves.append(line(grid, L, None, BLUE, width=0.8, alpha=0.5))
    curves.append(line(grid, np.median(all_curves, axis=0), f"median of the {n} worlds", YELLOW, width=3))
    curves.append(line([0, 1], [0, 1], "everyone holding the same", GREY, width=1, dash=[4, 4]))
    ch.figure(
        "lorenz", title=f"Lorenz curves at iteration {W.last:,}",
        x={"label": "share of agents, poorest first", "min": 0, "max": 1},
        y={"label": "share of all tokens they hold", "min": 0, "max": 1},
        series=curves,
        caption=f"For each of {W.lived} (thin blue lines), after its last game: the "
                "poorest share x of its agents holds the share y of all tokens. The yellow line "
                "is the median over the worlds at each x; the dashed diagonal is a world where "
                "everyone holds the same. The further a curve sags below the diagonal, the more "
                "unequal the world (the diagram above; [The Gini coefficient]"
                "(gini-coefficient.md)).",
        recipe=recipe(W.surviving_text(), FRAMES,
                      [f"Read each run's last frame (iteration {W.last:,}, `phase` 2) and its `tokens`.",
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
        series=[line(xs, ccdf, f"{len(pooled):,} agents of {n} worlds, after the last game", BLUE, width=2)],
        caption=f"Of all agents alive after the last game of {W.lived}, the share that "
                "hold at least x tokens, for every x. Both axes are logarithmic: a straight falling "
                "line here would be a power law.",
        recipe=recipe(W.surviving_text(), FRAMES,
                      ["Pool the `tokens` of the last frame of every run.",
                       "For every value x that occurs, the share of agents with at least x."]))
    ch.number("tokens_end", {"agents": int(len(pooled)), "median": float(np.median(pooled)),
                             "mean": float(pooled.mean()), "share_le_5": float(np.mean(pooled <= 5)),
                             "share_ge_100": float(np.mean(pooled >= 100)), "max": float(pooled.max()),
                             "share_1": float(np.mean(pooled == 1))})

    for name, stat, title, ylabel, colour in (
            ("gini", "gini", "Inequality of wealth (Gini)", "Gini coefficient", YELLOW),
            ("rich-tenth", "topDecileShare", "Share of all tokens held by the richest tenth", "share", ORANGE)):
        b = band_series([series(s.run_id, stat) for s in specs],
                        f"{len(specs)} worlds: median, middle half, nine in ten", colour)
        ch.figure(name, title=title, x={"label": "iteration"}, y={"label": ylabel, "min": 0},
                  series=[b],
                  caption=f"After every game. {BAND_WORDS}.",
                  recipe=recipe(W.runs_text(), STATS_FILE,
                                [f"Take every run's rows with `phase` = 2 and the statistic `{stat}` ([What a run records](frames-and-stats.md)).",
                                 *BAND_STEPS]))
        ch.number(f"{name}_settled", describe([mean_over(s.run_id, stat, *W.window) for s in alive]))
        # The first five iterations, in the median world.
        ch.number(f"{name}_first", float(np.median([mean_over(s.run_id, stat, 0, 4) for s in specs])))

    richest = band_series([(series(s.run_id, "maxTokens")[0], series(s.run_id, "maxTokens")[1] / W.tokens)
                           for s in specs], "the richest single agent's share", RED)
    ch.figure("richest", title="The richest agent's share of all tokens", x={"label": "iteration"},
              y={"label": "share of all tokens", "min": 0}, series=[richest],
              caption=f"After every game, the tokens of the richest agent divided by all {W.tokens:,}. {BAND_WORDS}.",
              recipe=recipe(W.runs_text(), STATS_FILE,
                            [f"Take the rows with `phase` = 2; divide `maxTokens` by {W.tokens:,}.", *BAND_STEPS]))
    ch.number("richest_settled", describe([mean_over(s.run_id, "maxTokens", *W.window) / W.tokens for s in alive]))

    med = band_series([series(s.run_id, "medianTokens") for s in specs], "median agent's tokens", GREEN)
    mean = band_series([series(s.run_id, "meanTokens") for s in specs],
                       f"mean tokens per agent ({W.tokens:,} ÷ agents)", BLUE)
    ch.figure("typical", title="What a typical agent holds", x={"label": "iteration"},
              y={"label": "tokens", "min": 0, "max": 40}, series=[mean, med],
              caption=f"After every game: the mean number of tokens per agent, which is {W.tokens:,} divided "
                      "by the number of agents (blue), and the median agent's tokens (green). The "
                      f"median lies below the mean because a few agents hold a lot. {BAND_WORDS}.",
              recipe=recipe(W.runs_text(), STATS_FILE,
                            ["Take the rows with `phase` = 2 and the statistics `meanTokens` and `medianTokens`.",
                             *BAND_STEPS]))
    ch.number("median_tokens_settled", describe([mean_over(s.run_id, "medianTokens", *W.window) for s in alive]))
    ch.number("mean_tokens_settled", describe([mean_over(s.run_id, "meanTokens", *W.window) for s in alive]))

    w = F.world_pass(W.run(1))
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
        recipe=recipe(f"`{W.run(1)}`.", FRAMES,
                      [f"Read the last frame (iteration {W.last:,}, `phase` 2); count each agent's connections in `edges`.",
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
              recipe=recipe(W.runs_text(), STATS_FILE,
                            ["Take the rows with `phase` = 2 and the statistic `redistributed`.", *BAND_STEPS]))
    ch.number("shared_out_settled", describe([mean_over(s.run_id, "redistributed", *W.window) for s in alive]))

    # The thesis of Experiment 4 checked on these worlds, as its plan checked it: from iteration 100 on.
    thesis = {}
    for stat, above in (("gini", 0.5), ("topDecileShare", 0.4), ("selfAllocationShare", 0.5)):
        values = [mean_over(s.run_id, stat, 100, W.last) for s in alive]
        thesis[stat] = {**describe(values), "above": int(sum(v > above for v in values)), "threshold": above}
    ch.number("thesis_E04", thesis)


# ---------------------------------------------------------------------------
# How the game is played
# ---------------------------------------------------------------------------

@chapter
def game(ch: F.Chapter) -> None:
    specs = W.specs
    alive = W.alive
    first, last = [], []
    for s in specs:
        first += F.world_pass(s.run_id)["stakes"].get("first", [])
    for s in alive:
        last += F.world_pass(s.run_id)["stakes"].get("last", [])
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
        series=[{"label": f"the first game: {len(first):,} agents of {len(specs)} worlds", "kind": "bars",
                 "x0": (k - 0.4).tolist(), "x1": k.tolist(), "y": h0, "colour": GREY},
                {"label": f"the last game: {len(last):,} agents of {len(alive)} worlds", "kind": "bars",
                 "x0": k.tolist(), "x1": (k + 0.4).tolist(), "y": h1, "colour": BLUE}],
        caption="For every agent that staked in a game: the share of its tokens it put on its own "
                "node. '0' and '1' are exact: nothing at home, everything at home. Grey: the "
                "game of iteration 0, played by the founders and their first children, whose "
                f"brains no selection has touched yet. Blue: the game of iteration {W.last:,}.",
        recipe=recipe(W.runs_text(), FRAMES,
                      [f"In the frames of iteration 0 and {W.last:,} with `phase` 2, read "
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
            frame = D.frame_at(run_id, iteration, 2)
            for w in (frame.get("decisions") or {}).get("winners") or []:
                counts[(w["winner"] != w["node"]) * 2 + bool(w.get("revolt"))] += 1
        total = sum(counts.values())
        return [counts[i] / total for i in range(4)], total

    w0, n0 = winners([s.run_id for s in specs], 0)
    w1, n1 = winners([s.run_id for s in alive], W.last)
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
        x={"label": "", "categories": [f"the first game ({n0:,} nodes, {len(specs)} worlds)",
                                       f"the last game ({n1:,} nodes, {len(alive)} worlds)"]},
        y={"label": "share of the nodes staked on", "min": 0, "max": 1}, series=stack,
        caption="Every node on which anyone staked in the game, by who won it: the agent living on "
                "it (green and cyan) or one of its neighbours (orange and violet), and whether the "
                "winner was the largest single staker (green, orange) or the member of a coalition "
                "that outweighed the largest staker (cyan, violet; see [How a coalition takes a "
                "node](revolution.md)). Each column adds up to 1. Left: the game of "
                f"iteration 0; right: the game of iteration {W.last:,}.",
        recipe=recipe(W.runs_text(), FRAMES,
                      [f"Read frame 1 of each of the {len(specs)} runs (the game of iteration 0) and frame "
                       f"{W.frame(W.last):,} of the {len(alive)} that reached it (the game of iteration {W.last:,}).",
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
            ch.number(f"{stat}_settled", describe([mean_over(s.run_id, stat, *W.window) for s in alive]))
            ch.number(f"{stat}_first", at_start(specs, stat))
        ch.figure(name, title=title, x={"label": "iteration"}, y={"label": ylabel, "min": 0, "max": 1},
                  series=out,
                  caption=f"After every game of the {len(specs)} baseline worlds. {words} For each, "
                          f"{BAND_WORDS[0].lower()}{BAND_WORDS[1:]} ([Bands](median-and-quantiles.md#bands-many-worlds-in-one-figure)).",
                  recipe=recipe(W.runs_text(), STATS_FILE,
                                ["Take the rows with `phase` = 2 and the statistics "
                                 + ", ".join(f"`{s}`" for s, _ in stats)
                                 + " (`a/b` is the row's `a` divided by its `b`; "
                                 "[Staking at home](how-the-game-is-played.md#staking-at-home-and-keeping-ones-node) defines each).", *BAND_STEPS]))

    # Where each world settles, statistic by statistic.
    order = ["spreadShare", "revoltShare", "heldHomeShare", "revolutions/nodes_before", "selfAllocationShare"]
    short = ["spread their stake", "tokens revolutionary", "nodes kept", "nodes to coalitions",
             "tokens at home"]
    groups = [[mean_over(s.run_id, stat, *W.window) for s in alive] for stat in order]
    ch.figure(
        "settled", title="How the game is played in each settled world",
        x={"label": "", "categories": short}, y={"label": f"share (mean over iterations {W.span})",
                                                 "min": 0, "max": 1},
        series=dots(short, groups, [BLUE, RED, GREEN, VIOLET, YELLOW]), legend=False,
        caption=f"One dot per {W.per_lived}: its mean over iterations {W.stretch} of "
                f"each statistic of the two figures above; the bar is the median of the {len(alive)}. From left: "
                "the share of stakers who spread their stake, the share of staked tokens marked "
                "revolutionary, the share of nodes kept by their own agent, nodes won by a coalition "
                "per agent, and the share of staked tokens put at home. The dots are spread "
                "sideways only so that they do not hide each other.",
        recipe=recipe(W.surviving_text(), STATS_FILE,
                      ["For each run, average each statistic over the rows with `phase` = 2 and "
                       f"{W.settled} ≤ `iteration` ≤ {W.last:,}.",
                       "Draw one dot per run and statistic, and a bar at the median."]))


# ---------------------------------------------------------------------------
# What shape does the network take?
# ---------------------------------------------------------------------------

#: A world with more agents than this is drawn a piece at a time: a layout of every agent
#: costs the square of their number, and fifteen thousand dots are a tangle.
PICTURE_LIMIT = 2500


def starting_ring(run_id: str) -> "nx.Graph":
    """A run's founders as the engine wired them, before anything happened: its own Watts–Strogatz ring."""
    from GraphOfLifeSimple import stream_seed
    cfg = F.store.load_config(run_id)
    return nx.watts_strogatz_graph(n=cfg.resolved_n(), k=cfg.resolved_k(), p=cfg.rewire_p, seed=stream_seed(cfg))


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
              "(bridges.md#core-trees-and-leaves)); yellow: the rest of the hanging trees; red: leaves, "
              "agents with a single connection. Where a dot is drawn means nothing in itself: a "
              "layout (networkx's ForceAtlas2, seed 1) pulls joined agents together and pushes the "
              "rest apart.")
ROLE_STEPS = ("Peel: remove every agent with one connection or none, again and again, until none is "
              "left; the agents never removed are the core. Leaves are agents with exactly one connection.",
              f"In a world of more than {PICTURE_LIMIT:,} agents, keep the agents within r steps of the oldest "
              f"agent, for the largest r that keeps at most {PICTURE_LIMIT:,} (`book_chapters.society.shown`).",
              "Lay the graph out with `networkx.forceatlas2_layout(G, max_iter=200, seed=1)`.")


def shown(frame: Dict[str, Any]) -> Tuple[List[int], Dict[str, Any]]:
    """
    The agents of a frame a picture draws: all of them, or, in a world too large
    to draw, every agent within r steps of its oldest agent (the lowest id among
    the oldest), for the largest r that holds at most PICTURE_LIMIT.
    """
    ids = list(frame["ids"])
    if len(ids) <= PICTURE_LIMIT:
        return ids, {"whole": True}
    adj = {u: set() for u in ids}
    for a, b in frame["edges"]:
        if a != b:
            adj[a].add(b)
            adj[b].add(a)
    centre = max(ids, key=lambda u: (frame["ages"][ids.index(u)], -u))
    members, radius = ball(adj, centre, PICTURE_LIMIT)
    return [u for u in ids if u in members], {"whole": False, "centre": centre, "radius": radius,
                                              "age": int(frame["ages"][ids.index(centre)])}


def piece_words(info: Dict[str, Any], drawn: int, total: int) -> str:
    """How a caption says which agents a picture holds."""
    if info["whole"]:
        return f"every one of its {total:,} agents"
    return (f"the {drawn:,} of its {total:,} agents within {info['radius']} steps of agent {info['centre']:,}, the "
            f"oldest, {info['age']:,} iterations old — a piece, since a world this large is a tangle when drawn whole")


def picture(run_id: str, iteration: int) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """A world after the game of `iteration` — or a piece of it — as a network chart, and the shares of its roles."""
    import networkx as nx
    frame = D.frame_at(run_id, iteration, 2)
    ids_all, edges_all = frame["ids"], frame["edges"]
    role = roles(ids_all, edges_all)
    ids, info = shown(frame)
    inside = set(ids)
    edges = [(a, b) for a, b in edges_all if a in inside and b in inside]
    G = nx.Graph()
    G.add_nodes_from(ids)
    G.add_edges_from(edges)
    pos = nx.forceatlas2_layout(G, max_iter=200, seed=1)
    index = {u: i for i, u in enumerate(ids)}
    title = (f"Iteration {iteration:,}: {len(ids):,} agents" if info["whole"] else
             f"Iteration {iteration:,}: {len(ids):,} of {len(ids_all):,} agents")
    chart = dict(kind="network", title=title,
                 nodes={"x": [float(pos[u][0]) for u in ids], "y": [float(pos[u][1]) for u in ids],
                        "group": [role[u] for u in ids]},
                 edges=[[index[a], index[b]] for a, b in edges],
                 colour={"labels": ["core: on a loop or between loops", "on a hanging tree",
                                    "leaf: a single connection"]})
    n = len(ids_all)
    return chart, {"agents": n, "edges": len(edges_all), "core": sum(1 for v in role.values() if v == 0) / n,
                   "tree": sum(1 for v in role.values() if v == 1) / n,
                   "leaves": sum(1 for v in role.values() if v == 2) / n, "drawn": len(ids), "piece": info}


@chapter
def shape(ch: F.Chapter) -> None:
    specs = W.specs
    alive = W.alive
    n = len(alive)
    moments = (0, 100, W.until // 3, W.last)
    charts, said = [], []
    for t in moments:
        chart, counts = picture(W.run(1), t)
        charts.append(chart)
        ch.number(f"picture_{t}", counts)
        said.append(f"at iteration {t:,}, " + piece_words(counts["piece"], counts["drawn"], counts["agents"]))
    whole = all(ch.numbers[f"picture_{t}"]["piece"]["whole"] for t in moments)
    ch.grid(
        "worlds", [dict(c, legend=False) for c in charts[:3]] + [charts[3]], columns=2,
        title="One world at four moments",
        caption=f"The world with seed 1 after the games of iterations {moments[0]}, {moments[1]}, "
                f"{moments[2]:,} and {moments[3]:,}. Every agent alive is a dot, every connection a line"
                + (". " if whole else f" ({'; '.join(said)}). ") + ROLE_WORDS,
        recipe=recipe(f"`{W.run(1)}`, one of the {len(specs)} baseline runs.", FRAMES,
                      ["Read frames " + ", ".join(f"`{W.frame(t)}`" for t in moments[:3])
                       + f" and `{W.frame(moments[3])}` (iterations {moments[0]}, {moments[1]}, "
                       f"{moments[2]:,} and {moments[3]:,}, after the game): `ids` and `edges`.", *ROLE_STEPS]))
    end = ch.numbers[f"picture_{W.last}"]
    ch.figure(
        "world-end", **dict(charts[3], title=f"World of seed 1 after the game of iteration {W.last:,}"),
        caption=f"The world with seed 1 after the game of iteration {W.last:,}, "
                f"{piece_words(end['piece'], end['drawn'], end['agents'])}: every agent drawn is a dot, every "
                f"connection between two of them a line. {ROLE_WORDS}",
        recipe=recipe(f"`{W.run(1)}`.", FRAMES,
                      [f"Read frame `{W.frame(W.last)}` (iteration {W.last:,}, after the game): `ids` and `edges`.",
                       *ROLE_STEPS]))

    pooled = np.concatenate([F.world_pass(s.run_id)["last"]["degrees"] for s in alive])
    ds = np.arange(1, pooled.max() + 1)
    rings = [starting_ring(s.run_id) for s in specs]
    founders_degree = int(round(np.mean([2 * g.number_of_edges() / g.number_of_nodes() for g in rings])))
    ch.number("ring", {"founders": rings[0].number_of_nodes(), "degree": founders_degree,
                       "transitivity": describe([nx.transitivity(g) for g in rings])})
    ch.figure(
        "degree", title="How many connections agents have",
        x={"label": "connections d (logarithmic)", "log": True, "min": 1},
        y={"label": "share of agents with at least d (logarithmic)", "log": True, "max": 1},
        series=[line(ds, [(pooled >= d).mean() for d in ds], f"{len(pooled):,} agents of {n} worlds, after the last game",
                     BLUE, width=2)],
        guides=[{"axis": "x", "at": founders_degree, "label": "a founder at the start"}],
        caption=f"Of all agents alive after the last game of {W.lived}, the share "
                f"that have at least d connections, for every d. At the start a founder had {founders_degree} "
                "on average (the grey line) — every one of them exactly that many before the ring's rewiring "
                "([The starting ring](starting-ring.md)).",
        recipe=recipe(W.surviving_text(), FRAMES,
                      [f"Count every agent's connections in the last frame's `edges`; pool the {n} runs.",
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
        ch.number(f"{stat}_settled", describe([mean_over(s.run_id, stat, *W.window) for s in alive]))
        ch.number(f"{stat}_first", at_start(specs, stat))
    ch.grid(
        "structure", panels, columns=3, title="Core, leaves and bridges",
        caption=f"Three shares, after every game of the {len(specs)} baseline worlds. Left: the share of agents "
                "in the core ([Core, trees and leaves](bridges.md#core-trees-and-leaves)), measured every "
                "25 iterations. Middle: the share of agents with exactly one connection, after every "
                "game. Right: the share of connections that are bridges ([Bridges](bridges.md)), "
                f"every 25 iterations. In each, {BAND_WORDS[0].lower()}{BAND_WORDS[1:]}.",
        recipe=recipe(W.runs_text(), STATS_FILE,
                      ["Take the rows with `phase` = 2 and the statistics `coreShare`, `leaves/nodes` and "
                       "`bridges/edges` (`a/b` is the row's `a` divided by its `b`).",
                       "For `leaves/nodes`, make the band as for every other band "
                       f"([Bands](median-and-quantiles.md#bands-many-worlds-in-one-figure)): {BAND_STEPS[0][0].lower()}{BAND_STEPS[0][1:]}",
                       "For `coreShare` and `bridges/edges`, which are measured every 25 iterations, "
                       "use stretches of 25."]))
    clus = band_series([series(s.run_id, "transitivity") for s in specs], "clustering (transitivity)", GREEN)
    ch.figure("clustering", title="Clustering", x={"label": "iteration"}, y={"label": "transitivity", "min": 0},
              series=[clus],
              caption="Three times the number of triangles divided by the number of connected triples "
                      "([Clustering](degree.md#clustering)), every 25 iterations. The founders' ring starts at "
                      f"{ch.numbers['ring']['transitivity']['median']:.2f} in the median world. {BAND_WORDS}.",
              recipe=recipe(W.runs_text(), STATS_FILE,
                            ["Take the rows with `phase` = 2 and the statistic `transitivity`.",
                             "Bands as above, in stretches of 25."]))
    ch.number("transitivity_settled", describe([mean_over(s.run_id, "transitivity", *W.window) for s in alive]))

    # The thesis of Experiment 5 checked on these worlds, as its plan checked it: from iteration 100 on.
    thesis = {}
    for stat, bound, side in (("bridges/edges", 0.2, "above"), ("transitivity", 0.1, "below"),
                              ("coreShare", 0.5, "below")):
        values = [mean_over(s.run_id, stat, 100, W.last) for s in alive]
        thesis[stat] = {**describe(values), "bound": bound, "side": side,
                        "worlds_on_side": int(sum((v > bound) if side == "above" else (v < bound) for v in values))}
    ch.number("thesis_E05", thesis)

    agents = np.array([mean_over(s.run_id, "nodes", *W.window) for s in alive])
    k = np.array([mean_over(s.run_id, "meanDegree", *W.window) for s in alive])
    L = np.array([mean_over(s.run_id, "meanPathLength", *W.window) for s in alive])
    rnd = np.log(agents) / np.log(k)
    ch.figure(
        "distance", title="Steps between two agents",
        x={"label": f"agents (mean over iterations {W.span})", "min": 0},
        y={"label": "mean steps between two agents", "min": 0},
        series=[{"label": "each world: measured", "x": agents.tolist(), "y": L.tolist(), "points": True, "size": 7,
                 "colour": BLUE},
                {"label": "each world: a random network of the same size, ln N ÷ ln k", "x": agents.tolist(),
                 "y": rnd.tolist(), "points": True, "size": 5, "colour": GREY}],
        caption=f"For each {W.per_lived} (blue): its mean number of agents and the mean number "
                "of steps along connections from one agent to another, both averaged over "
                f"iterations {W.span}. Grey: the rough distance in a random network with the "
                "same number of agents N and connections per agent k, ln N / ln k.",
        recipe=recipe(W.surviving_text(), STATS_FILE,
                      ["Average `nodes`, `meanDegree` and `meanPathLength` over the rows with "
                       f"`phase` = 2 and {W.settled} ≤ `iteration` ≤ {W.last:,} (`meanPathLength` is measured every "
                       "25 iterations, from breadth-first searches out of 8 to 16 evenly spread agents; [Path length](path-length.md)).",
                       "Plot one dot per run, and ln N / ln k for the same run."]))
    hubs = np.array([mean_over(s.run_id, "maxDegree/nodes", *W.window) for s in alive])
    plain = hubs < STAR
    ch.number("distance", {"measured": describe(L), "random": describe(rnd), "ratio": describe(L / rnd),
                           "agents": describe(agents), "size_corr": float(np.corrcoef(agents, L)[0, 1]),
                           "size_corr_without_stars": float(np.corrcoef(agents[plain], L[plain])[0, 1])
                           if plain.sum() > 2 else None,
                           "measured_without_stars": describe(L[plain])})


# ---------------------------------------------------------------------------
# Genotypes and lineages
# ---------------------------------------------------------------------------

@chapter
def lineage(ch: F.Chapter) -> None:
    specs = W.specs
    alive = W.alive
    g_lives, a_lives = [], []
    for s in alive:
        w = F.world_pass(s.run_id)
        g_lives += w["genotypes"]
        a_lives += w["agents"]
    out = []
    for lives, label, colour in ((g_lives, f"genotypes ({len(g_lives):,})", VIOLET),
                                 (a_lives, f"agents ({len(a_lives):,})", BLUE)):
        at, surv = F.kaplan_meier(lives)
        keep = at <= W.until
        out.append(line(at[keep], surv[keep], label, colour, width=2))
        ch.number(f"km_{label.split()[0]}", {f"at_least_{L}": float(surv[np.searchsorted(at, L, side='right') - 1])
                                            for L in (2, 3, 5, 10, 20, 50, 100)})
    ch.figure(
        "lives", title="How long genotypes and agents last",
        x={"label": "lifetime L, in iterations (logarithmic)", "log": True, "min": 1},
        y={"label": "share lasting at least L (logarithmic)", "log": True, "min": 1e-6, "max": 1},
        series=out,
        caption=f"Of all genotypes that appeared, and all agents born, at iteration {W.settled} or later in "
                f"{W.lived} and present after at least one game, the share that lasted at least L iterations. "
                "A genotype lasts from the first game after which some agent carries it to the last; "
                "an agent from its birth to the last game it is alive after. Lives still going at the "
                "end are counted as far as they went (Kaplan–Meier).",
        recipe=recipe(W.surviving_text(), FRAMES,
                      ["Read every frame with `phase` = 2; for every genotype (brain id in `brain_ids`) "
                       "and every agent (id in `ids`) note the first and last iteration it appears.",
                       f"Keep those first seen at iteration {W.settled} or later; one still present in the "
                       "last frame is censored.",
                       "Kaplan–Meier as in [Births, deaths and ages](births-deaths-and-ages.md)."]))

    top_seed = EXAMPLES["top"]
    example = F.world_pass(W.run(top_seed))
    top = band_series([(np.array([t for t, _ in F.world_pass(s.run_id)["top"]], float),
                        np.array([v for _, v in F.world_pass(s.run_id)["top"]], float)) for s in specs],
                      f"{len(specs)} worlds: median, middle half, nine in ten", VIOLET)
    single = line([t for t, _ in example["top"]], [v for _, v in example["top"]], f"the world of seed {top_seed}",
                  YELLOW, width=0.8, alpha=0.9)
    ch.figure("top-share", title="The most common genotype's share of the world", x={"label": "iteration"},
              y={"label": "share of agents", "min": 0, "max": 0.7}, series=[top, single],
              caption="After every game, the share of the living that carry the most common genotype. "
                      f"{BAND_WORDS}. The yellow line is the single world with seed {top_seed}.",
              recipe=recipe(W.runs_text(), FRAMES,
                            ["After every game, count the agents per genotype in `brain_ids`; the "
                             "largest count divided by the number of agents.", *BAND_STEPS]))
    peaks = {s.run_id: max(((t, v) for t, v in F.world_pass(s.run_id)["top"] if t >= 100), key=lambda x: x[1])
             for s in specs}
    highest = max(peaks, key=lambda r: peaks[r][1])
    ch.number("top_share", {
        "median": float(np.median([np.median([v for t, v in F.world_pass(s.run_id)["top"] if t >= 100])
                                   for s in specs])),
        "worlds_above": {str(x): int(sum(v > x for _, v in peaks.values())) for x in (0.1, 0.2, 0.33, 0.5)},
        "highest": {"seed": int(highest.rsplit("s", 1)[1]), "share": float(peaks[highest][1]),
                    "iteration": int(peaks[highest][0])}})

    # The thesis of Experiment 6 checked on these worlds, as its plan checked it: from iteration 100 on.
    lowest = []
    for s in alive:
        its, v = series(s.run_id, "cladesInWindow")
        lowest.append(float(np.nanmin(v[its >= 100])))
    ch.number("thesis_E06", {
        "distinct_per_agent": describe([mean_over(s.run_id, "distinctBrains/nodes", 100, W.last) for s in alive]),
        "families": describe([mean_over(s.run_id, "cladesInWindow", 100, W.last) for s in alive]),
        "families_lowest": describe(lowest), "worlds_ever_ten_or_fewer": int(sum(x <= 10 for x in lowest)),
        "worlds_with_a_tenth": int(sum(v > 0.1 for _, v in peaks.values()))})

    depth = []
    seeds = EXAMPLES["ancestry"]
    for k, seed in enumerate(seeds):
        w = F.world_pass(W.run(seed))
        xs = [t for t, *_ in w["ancestry"]]
        ys = [t if d is None else d for t, d, *_ in w["ancestry"]]
        depth.append(line(xs, ys, f"seed {seed}", [BLUE, YELLOW, GREEN][k], width=2))
    ch.figure("ancestor", title="How far back everyone shares one ancestor", x={"label": "iteration"},
              y={"label": "iterations back", "min": 0}, series=depth,
              guides=[{"axis": "y", "at": 0}],
              caption=f"Every 25 iterations, for three worlds (seeds {seeds[0]}, {seeds[1]} and {seeds[2]}, one "
                      "colour each): how "
                      "many iterations earlier the newest genotype lived from which every living agent "
                      "descends ([The common ancestor](genotype.md#the-common-ancestor-of-the-living)). While a line rises "
                      "by 25 every 25 iterations, that ancestor stays the same and simply grows older; "
                      "each drop is a moment when it moves forward to a younger genotype, because all "
                      "the other branches of the family tree have died out. Before the first drop the "
                      "living descend from more than one founder, and the line is drawn at the age of "
                      "the world.",
              recipe=recipe(f"`{W.run(seeds[0])}`, `-s{seeds[1]:03d}` and `-s{seeds[2]:03d}`.", FRAMES,
                            ["Read every frame in order; remember each genotype's parent and the "
                             "iteration it first appeared.",
                             "Every 25 iterations, after the game, count the agents per living genotype "
                             "and climb the family tree from them, newest genotype first, adding each "
                             "genotype's count to its parent's ([The common ancestor](genotype.md#the-common-ancestor-of-the-living)).",
                             "The first genotype reached that carries all agents is their newest common "
                             "ancestor; plot the iterations since it first appeared."]))

    for seed, window in EXAMPLES["families"]:
        w = F.world_pass(W.run(seed), window)
        m = w["muller"]
        xs = [t for t, n, _ in m["series"]]
        shares = np.array([[c / n for c in per] for t, n, per in m["series"]])
        areas = stacked_areas(xs, [(f"family {rank + 1}", shares[:, rank], rank)
                                   for rank in range(min(7, shares.shape[1]))], "all other families")
        ch.figure(
            f"families-{seed}", title=f"Families of iteration {window[0]:,}, world of seed {seed}",
            x={"label": "iteration", "min": window[0], "max": window[1]},
            y={"label": "share of the living", "min": 0, "max": 1}, series=areas,
            caption=f"Every genotype alive after the game of iteration {window[0]:,} founds a family: "
                    "itself and every genotype descended from it. After every later game, the share of "
                    "the living in each family, stacked; the seven families that ever held the largest "
                    "share in colour, all others together in grey.",
            recipe=recipe(f"`{W.run(seed)}`.", FRAMES,
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

    runs = json.load(open(F.os.path.join(F.BOOK, "results", f"{W.lineage}.json")))["lineage"][W.condition]["runs"]
    alive_ids = {s.run_id for s in alive}
    moves = [v["ancestor"]["moves"] for k, v in runs.items() if k in alive_ids]
    cat = f"{len(alive)} worlds"
    ch.figure("moves", title="How often one branch replaces all others", x={"label": "", "categories": [cat]},
              y={"label": "moves of the common ancestor after iteration 100", "min": 0},
              series=dots([cat], [moves], [GREEN]), legend=False,
              caption=f"One dot per {W.per_lived}: how many times, between iteration 100 and the "
                      "end, the newest common ancestor of all the living moved forward to a younger "
                      "genotype (one of the drops in the figure above). The bar is the median.",
              recipe=recipe(W.surviving_text(),
                            f"the lineage analysis of Experiment {int(W.lineage[1:])} (`python3 gol_lab.py analyse "
                            f"{W.lineage}`), in `book/results/{W.lineage}.json` (`ancestor.moves` of every run)",
                            ["Take the common-ancestor depth every 25 iterations as above.",
                             "Count the checks after iteration 100 at which the depth is smaller than "
                             "the previous depth plus 25."]))
    ch.number("moves", describe(moves))
