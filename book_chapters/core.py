# -*- coding: utf-8 -*-
"""
Part II, the core of a world: what is left when its hanging trees are peeled
away, how many dimensions that core has, and how it holds a world's loops.

Peeling removes every agent with one connection or none, again and again, until
none is left ([Core, trees and leaves](core-trees-leaves.md)). What
remains is the 2-core: every agent on it lies on a loop or on a path between
loops. Everything peeled hangs off the core in trees, each joined to it at one
agent. Peeling the core again, now removing agents with fewer than three
connections inside it, leaves the 3-core, and so on: the world as an onion.
"""
from __future__ import annotations

from collections import Counter, deque
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np

import book_data as D
import book_figures as F
from book_graph import graph
from book_figures import FRAMES, bars, chapter, describe, dots, line, quantile_band, recipe
from book_chapters.common import (BAND_WORDS, BLUE, CYAN, GREEN, ORANGE, RED, VIOLET, WORLD_SETS,
                                  WORLDS as W, YELLOW, listed)
from book_chapters.space import BIG, PLATEAUS, RULERS, band, measure, moments, plateau, world_dimensions

#: How often a world's core is taken apart.
CORE_EVERY = 25
#: Shells of the onion counted one by one; deeper ones are counted together.
SHELLS = 8


def peel(adj: Dict[int, set]) -> Dict[int, int]:
    """0 for an agent of the 2-core, 1 for one on a hanging tree, 2 for a leaf (as Chapter 17 colours them)."""
    degree = {u: len(v) for u, v in adj.items()}
    peeled, queue = set(), [u for u, k in degree.items() if k <= 1]
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
    return {u: 0 if u not in peeled else 2 if len(adj[u]) == 1 else 1 for u in adj}


def hanging(adj: Dict[int, set], role: Dict[int, int]) -> Tuple[Counter, Dict[int, int]]:
    """
    How the trees hang off the core: for each core agent, how many agents hang
    from it, and for every agent off the core, how many steps it is from the core.
    """
    depth = {u: 0 for u, r in role.items() if r == 0}
    root = {u: u for u in depth}
    queue = deque(depth)
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v not in depth:
                depth[v] = depth[u] + 1
                root[v] = root[u]
                queue.append(v)
    trees = Counter(root[u] for u, r in role.items() if r != 0 and u in root)
    return trees, {u: d for u, d in depth.items() if role[u] != 0}


def onion(adj: Dict[int, set]) -> Dict[int, int]:
    """Every agent's core number: the deepest k-core it belongs to."""
    g = nx.Graph()
    g.add_nodes_from(adj)
    g.add_edges_from((u, v) for u in adj for v in adj[u] if u < v)
    return nx.core_number(g)


def sub(adj: Dict[int, set], keep) -> Dict[int, set]:
    keep = set(keep)
    return {u: adj[u] & keep for u in keep}


@D.measure("coreshape")
def core_shape(run_id: str) -> Dict[str, Any]:
    """
    A world's core every CORE_EVERY iterations, at the start of a game and what
    the game does to it; and the dimensions of its 2-core and 3-core after the
    games of five moments of its settled life (book_chapters.space.moments).

    looks       one per look t: the agents, connections and loops (cycle rank)
                of the world at the start of the game of t; for the core, the
                trees and the leaves, how many agents, their mean tokens, age
                and connections, and how many die in the game; the trees'
                sizes and depths; and the agents in each shell of the onion.
    dimensions  for each of those moments: both rulers (book_chapters.space.measure)
                on the world, its 2-core and its 3-core after that game.
    """
    last = D.last_iteration(run_id)
    looks = []
    for t in range(CORE_EVERY, last + 1, CORE_EVERY):
        start, after = D.frame_at(run_id, t, 1), D.frame_at(run_id, t, 2)
        adj = graph(start)
        role = peel(adj)
        tokens = dict(zip(start["ids"], start["tokens"]))
        ages = dict(zip(start["ids"], start["ages"]))
        alive = set(after["ids"])
        parts = {}
        for r, name in enumerate(("core", "tree", "leaf")):
            members = [u for u, x in role.items() if x == r]
            parts[name] = {"agents": len(members),
                           "tokens": float(np.mean([tokens[u] for u in members])) if members else None,
                           "age": float(np.mean([ages[u] for u in members])) if members else None,
                           "degree": float(np.mean([len(adj[u]) for u in members])) if members else None,
                           "died": sum(1 for u in members if u not in alive),
                           "token_share": float(sum(tokens[u] for u in members) / max(1, sum(tokens.values())))}
        trees, depth = hanging(adj, role)
        sizes = np.array(list(trees.values()), dtype=float)
        depths = np.array(list(depth.values()), dtype=float)
        number = onion(adj)
        shells = Counter(min(k, SHELLS) for k in number.values())
        edges = sum(len(v) for v in adj.values()) // 2
        core_edges = sum(1 for u in adj for v in adj[u] if u < v and role[u] == 0 and role[v] == 0)
        looks.append({"t": t, "agents": len(adj), "edges": edges, "loops": edges - len(adj) + 1,
                      "core_edges": core_edges, "parts": parts,
                      "trees": {"count": int(len(sizes)), "mean": float(sizes.mean()) if sizes.size else None,
                                "median": float(np.median(sizes)) if sizes.size else None,
                                "largest": int(sizes.max()) if sizes.size else 0,
                                "share_of_one": float(np.mean(sizes == 1)) if sizes.size else None},
                      "depth": {"mean": float(depths.mean()) if depths.size else None,
                                "deepest": int(depths.max()) if depths.size else 0,
                                "counts": np.bincount(depths.astype(int)).tolist() if depths.size else []},
                      "shells": {str(k): shells.get(k, 0) for k in range(1, SHELLS + 1)},
                      "deepest_shell": int(max(number.values())) if number else 0})
    dimensions = []
    for t in moments(last):
        adj = graph(D.frame_at(run_id, t, 2))
        role = peel(adj)
        core2 = sub(adj, (u for u, r in role.items() if r == 0))
        number = onion(adj)
        core3 = sub(adj, (u for u, k in number.items() if k >= 3))
        dimensions.append({"t": t, "world": measure(adj, 23), "core": measure(core2, 23) if len(core2) > 50 else None,
                           "core3": measure(core3, 23) if len(core3) > 50 else None})
    return {"looks": looks, "dimensions": dimensions}


# ---------------------------------------------------------------------------
# The chapter: the core of a world
# ---------------------------------------------------------------------------

CORE_DATA = ("each run's frames, read every 25 iterations by `book_chapters.core.core_shape`: frame `2·t` (the "
             "start of the game of t) and frame `2·t + 1` (after it), `gol_store.read_frame`")
PARTS = (("core", "the core", BLUE), ("tree", "on a hanging tree", YELLOW), ("leaf", "leaves", RED))


def _looks(kept, key):
    """(t, value) per run of one number of the looks: a function of a look."""
    return [(np.array([x["t"] for x in m["looks"]], float), np.array([key(x) for x in m["looks"]], float))
            for m in kept.values()]


@chapter
def core(ch: F.Chapter) -> None:
    specs, alive = W.specs, W.alive
    kept = {s.run_id: core_shape(s.run_id) for s in specs}
    n = len(specs)

    # ---- how much of a world is core ----
    panels = []
    for title, field in (("Share of the agents", "agents"), ("Share of the tokens", "token_share")):
        series_ = []
        for part, label, colour in PARTS:
            if field == "agents":
                key = (lambda x, p=part: x["parts"][p]["agents"] / x["agents"])
            else:
                key = (lambda x, p=part: x["parts"][p]["token_share"])
            series_.append(F.band_series(_looks(kept, key), label, colour))
            ch.number(f"{part}_{field}_settled", describe([float(np.mean([key(x) for x in m["looks"]
                                                                          if x["t"] >= W.settled]))
                                                           for m in kept.values()]))
        panels.append(dict(title=title, x={"label": "iteration"}, y={"label": "share", "min": 0, "max": 1},
                           series=series_))
    ch.grid(
        "parts", panels, columns=2, title="Core, trees and leaves",
        caption=f"At the start of the game of every 25th iteration of the {n} worlds: the share of the agents (left) "
                "and of the tokens (right) in the core, on the hanging trees (agents with two connections or more "
                "that the peeling removes), and on leaves (agents with one connection). "
                f"{BAND_WORDS}.",
        recipe=recipe(W.runs_text(), CORE_DATA,
                      ["Peel every agent with one connection or none, again and again; what is never peeled is the "
                       "core; leaves are agents with one connection; the rest of the peeled are on trees.",
                       "Count the agents and add up the tokens of each part; divide by the world's."]))

    # ---- how many dimensions the core has ----
    dims = [world_dimensions(s.run_id) for s in alive]
    small = WORLD_SETS["E02"]
    small_dims = [world_dimensions(s.run_id) for s in small.alive] if W.experiment != "E02" else []
    big = [s for s in F.runs_of("E08", "baseline") if int(s.lab["world"]["total_tokens"]) == 409600
           and F.lived(s) >= s.until]
    bigs = [world_dimensions(s.run_id) for s in big]
    panels = []
    for xk, yk, title, xl in RULERS:
        series_ = [{**band([d["world"] for d in dims], xk, yk), "label": f"the {len(alive)} worlds, whole",
                    "colour": BLUE},
                   {**band([d["core"] for d in dims], xk, yk), "label": "their cores", "colour": GREEN},
                   {**band([d["twin"] for d in dims], xk, yk), "label": "their random twins", "colour": RED}]
        for i, d in enumerate(bigs):
            series_.append(line(d["core"][xk], d["core"][yk], "cores of worlds of 409,600 tokens" if i == 0 else None,
                                YELLOW, width=2))
        panels.append(dict(title=title, x={"label": xl, "log": True}, y={"label": "dimension", "min": 0, "max": 8},
                           series=series_, guides=[{"axis": "y", "at": d, "label": ""} for d in (1, 2, 3)]))
    ch.grid(
        "dimension", panels, columns=2, title="How many dimensions the core has",
        caption="The two rulers of Chapter 24 — the local dimension from how balls grow, and the spectral dimension "
                f"from how a random walk spreads — on {W.lived} after their last game (blue), on their cores alone "
                "(green), and on their random twins (red); each line is the median of the worlds, the darker band "
                "the middle half of them and the paler band nine in ten. Yellow: the cores of the three worlds of "
                "409,600 tokens of Chapter 42.",
        recipe=recipe(W.runs_text() + " " + BIG, FRAMES,
                      ["Read each run's last frame; build its network from `edges`; its core by peeling; its random "
                       "twin by `book_graph.degree_preserving` (seed 23).",
                       "Each network's largest piece, measured as in Chapter 24 (`book_chapters.space.measure`)."]))

    def plateaus(rows, ball, walk):
        return {"ball": describe([plateau(d["ball_r"], d["ball_d"], *ball) for d in rows if d]),
                "walk": describe([plateau(d["spec_t"], d["spec_d"], *walk) for d in rows if d]),
                "agents": describe([d["n"] for d in rows if d])}
    ch.number("dimension", {kind: plateaus([d[kind] for d in dims], *PLATEAUS) for kind in ("world", "core", "twin")})
    ch.number("dimension_big", {kind: plateaus([d[kind] for d in bigs], (3, 14), (4, 400))
                                for kind in ("world", "core", "twin")})
    if small_dims:
        ch.number("dimension_small", {kind: plateaus([d[kind] for d in small_dims], (2, 6), (4, 64))
                                      for kind in ("world", "core", "twin")})

    # ---- the core's dimension over a world's life ----
    times = [d["t"] for d in next(iter(kept.values()))["dimensions"]]
    panels = []
    for key, title, reader in (("ball", "From balls", lambda d: plateau(d["ball_r"], d["ball_d"], *PLATEAUS[0])),
                               ("walk", "From a random walk", lambda d: plateau(d["spec_t"], d["spec_d"], *PLATEAUS[1]))):
        series_ = []
        for part, label, colour in (("world", "the whole world", BLUE), ("core", "its 2-core", GREEN),
                                    ("core3", "its 3-core", VIOLET)):
            table = np.array([[reader(x[part]) if x[part] else np.nan for x in m["dimensions"]]
                              for m in kept.values()], float)
            series_.append({"label": label, **quantile_band(table, times), "colour": colour})
            ch.number(f"life_{key}_{part}", {str(t): describe(table[:, i]) for i, t in enumerate(times)})
        panels.append(dict(title=title, x={"label": "iteration"}, y={"label": "dimension", "min": 0, "max": 6},
                           series=series_, guides=[{"axis": "y", "at": d, "label": ""} for d in (1, 2, 3)]))
    ch.grid(
        "over-life", panels, columns=2, title="The core's dimension over a world's life",
        caption=f"After the games of iterations {listed(times)} of the {n} worlds: each ruler's typical reading — the "
                "median of its values over the scales of Chapter 24's plateau — on the whole world, on its 2-core "
                "and on its 3-core (what is left when agents with fewer than three connections inside the core are "
                f"peeled away too). {BAND_WORDS}.",
        recipe=recipe(W.runs_text(), CORE_DATA,
                      ["At each moment, the network after the game, its 2-core, and its 3-core "
                       "(`networkx.core_number` ≥ 3).",
                       "Both rulers on the largest piece of each (`book_chapters.space.measure`); the median of the "
                       "readings over the plateau's scales."]))

    # ---- the trees ----
    sizes, depth_counts = [], np.zeros(64)
    for m in kept.values():
        for x in m["looks"]:
            if x["t"] < W.settled:
                continue
            counts = np.array(x["depth"]["counts"], float)
            depth_counts[:len(counts)] += counts[:64]
    largest = _looks(kept, lambda x: x["trees"]["largest"])
    deepest = _looks(kept, lambda x: x["depth"]["deepest"])
    mean_size = _looks(kept, lambda x: x["trees"]["mean"] or np.nan)
    depth_share = depth_counts[1:] / depth_counts[1:].sum()
    k = np.arange(1, len(depth_share) + 1)
    keep = depth_share > 0
    ch.grid(
        "trees", [
            dict(title="How far agents hang from the core", x={"label": "steps from the core", "min": 0},
                 y={"label": "share of the agents off the core (logarithmic)", "log": True},
                 series=[{"label": None, "x": k[keep].tolist(), "y": depth_share[keep].tolist(), "points": True,
                          "size": 8, "colour": YELLOW}], legend=False),
            dict(title="The largest tree, and the deepest", x={"label": "iteration"},
                 y={"label": "agents, or steps (logarithmic)", "log": True, "min": 1},
                 series=[F.band_series(largest, "agents on the largest tree", ORANGE),
                         F.band_series(deepest, "steps to the deepest leaf", RED),
                         F.band_series(mean_size, "agents on a typical tree (mean)", YELLOW)])],
        columns=2, title="The trees that hang off the core",
        caption="Left: of all agents off the core at the start of the games of every 25th iteration from "
                f"{W.settled} on, pooled over the worlds, the share at each number of steps from the core, on a "
                "logarithmic axis: a straight falling line is a share that shrinks by the same factor with every "
                "step. Right: every 25 iterations, the number of agents hanging from the core agent that holds "
                "the most, the steps from the core to the farthest agent, and the mean number hanging from a core "
                f"agent that holds any. {BAND_WORDS}.",
        recipe=recipe(W.runs_text(), CORE_DATA,
                      ["Peel as above; a breadth-first search out of every core agent at once gives every other agent "
                       "its steps from the core and the core agent its tree hangs from "
                       "(`book_chapters.core.hanging`).",
                       "Count the agents per step and per core agent."]))
    ch.number("depth_share", depth_share[:12].tolist())
    for name, rows in (("largest_tree", largest), ("deepest", deepest), ("mean_tree", mean_size)):
        ch.number(f"{name}_settled", describe([float(np.nanmean(v[t >= W.settled])) for t, v in rows]))

    # ---- the onion ----
    shells = np.zeros(SHELLS)
    for m in kept.values():
        for x in m["looks"]:
            if x["t"] >= W.settled:
                shells += np.array([x["shells"][str(k)] for k in range(1, SHELLS + 1)], float)
    shells /= shells.sum()
    labels = [str(k) for k in range(1, SHELLS)] + [f"{SHELLS}+"]
    deepest_shell = _looks(kept, lambda x: x["deepest_shell"])
    ch.grid(
        "onion", [
            dict(title="Agents by the deepest core they are in", x={"label": "core number", "categories": labels},
                 y={"label": "share of the agents", "min": 0}, series=[bars(labels, shells.tolist(), VIOLET)],
                 legend=False),
            dict(title="The deepest core", x={"label": "iteration"}, y={"label": "core number", "min": 0},
                 series=[F.band_series(deepest_shell, None, VIOLET)], legend=False)],
        columns=2, title="A world as an onion",
        caption="An agent's core number is the largest k such that it belongs to the k-core: what is left when every "
                "agent with fewer than k connections is removed, again and again. Core number 1 is everything "
                "peeled to reach the 2-core (trees and leaves), 2 the agents of the 2-core that the 3-core no longer "
                "holds, and so on. Left: the share of agents with each core number, at the start of every 25th "
                f"game from {W.settled} on, pooled over the worlds. Right: the deepest core number in a world "
                f"every 25 iterations; {BAND_WORDS[0].lower()}{BAND_WORDS[1:]}.",
        recipe=recipe(W.runs_text(), CORE_DATA,
                      ["`networkx.core_number` of the network at the start of the game.",
                       "Count the agents of each core number (8 and more together); and the largest."]))
    ch.number("shells", dict(zip(labels, shells.tolist())))
    ch.number("deepest_shell_settled", describe([float(np.mean(v[t >= W.settled])) for t, v in deepest_shell]))

    # ---- the loops live in the core ----
    loops = _looks(kept, lambda x: x["loops"] / x["agents"])
    per_core = _looks(kept, lambda x: x["loops"] / max(1, x["parts"]["core"]["agents"]))
    core_degree = _looks(kept, lambda x: 2 * x["core_edges"] / max(1, x["parts"]["core"]["agents"]))
    ch.figure(
        "loops", title="Loops, per agent and per core agent", x={"label": "iteration"},
        y={"label": "per agent", "min": 0},
        series=[F.band_series(loops, "independent loops per agent", GREEN),
                F.band_series(per_core, "independent loops per core agent", BLUE),
                F.band_series(core_degree, "connections inside the core, per core agent", CYAN)],
        caption="Every 25 iterations at the start of the game: the number of independent loops of a world — "
                "connections minus agents plus one, its cycle rank ([Loops](loops.md)) — divided by its "
                "agents and by the agents of its core, which holds every loop; and the mean number of connections "
                f"a core agent has to other core agents. {BAND_WORDS}.",
        recipe=recipe(W.runs_text(), CORE_DATA,
                      ["Cycle rank = connections − agents + 1 (the world is one piece after every cleanup).",
                       "Divide by the agents, and by the core's agents; connections with both ends in the core, "
                       "twice, over the core's agents."]))
    for name, rows in (("loops_per_agent", loops), ("loops_per_core_agent", per_core),
                       ("core_degree", core_degree)):
        ch.number(f"{name}_settled", describe([float(np.mean(v[t >= W.settled])) for t, v in rows]))

    # ---- who lives where ----
    stats = {part: {"died": [], "tokens": [], "age": [], "degree": []} for part, _, _ in PARTS}
    for m in kept.values():
        looks = [x for x in m["looks"] if x["t"] >= W.settled]
        for part, _, _ in PARTS:
            agents = sum(x["parts"][part]["agents"] for x in looks)
            stats[part]["died"].append(sum(x["parts"][part]["died"] for x in looks) / max(1, agents))
            for f in ("tokens", "age", "degree"):
                stats[part][f].append(float(np.nanmean([x["parts"][part][f] if x["parts"][part][f] is not None
                                                        else np.nan for x in looks])))
    cats = [label for _, label, _ in PARTS]
    ch.grid(
        "where", [
            dict(title="Dead after the game", x={"label": "", "categories": cats}, y={"label": "share", "min": 0},
                 series=dots(cats, [stats[p]["died"] for p, _, _ in PARTS], [c for _, _, c in PARTS]), legend=False),
            dict(title="Tokens", x={"label": "", "categories": cats}, y={"label": "mean tokens", "min": 0},
                 series=dots(cats, [stats[p]["tokens"] for p, _, _ in PARTS], [c for _, _, c in PARTS]), legend=False),
            dict(title="Age", x={"label": "", "categories": cats}, y={"label": "mean age, iterations", "min": 0},
                 series=dots(cats, [stats[p]["age"] for p, _, _ in PARTS], [c for _, _, c in PARTS]), legend=False)],
        columns=3, title="Who lives where",
        caption="At the start of the games of every 25th iteration from "
                f"{W.settled} on: for the agents of the core, of the trees and the leaves, the share no longer alive "
                "after that game, their mean tokens, and their mean age. One dot per world; the bar is the median.",
        recipe=recipe(W.runs_text(), CORE_DATA,
                      ["Peel as above; for each part, the agents of frame `2·t` missing from frame `2·t + 1`, and the "
                       "means of `tokens` and `ages`.", "Pool the looks of a world; one dot per world."]))
    ch.number("where", {p: {f: describe(v) for f, v in stats[p].items()} for p, _, _ in PARTS})
