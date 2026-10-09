# -*- coding: utf-8 -*-
"""
Part II, the world as a space: Chapter 22 — like next to like — and Chapter 23 —
how many dimensions does a world have?

Both measure how things change with distance in the network: how alike agents are
at a given number of steps apart, and how much of a world lies within a given
number of steps, or how a random walk spreads through it. The dimension chapter
first calibrates its rulers on spaces whose dimension is known.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Tuple

import numpy as np

import book_data as D
import book_figures as F
from book_figures import FRAMES, chapter, describe, dots, line, lived_text, recipe, runs_of, survivors
from book_chapters.common import BLUE, GREEN, GREY, ORANGE, RED, VIOLET, YELLOW, baseline
from book_chapters.structure import graph

MOMENTS = (1000, 1500, 2000, 2500, 2999)
MAX_R = 15


def bfs(adj: Dict[int, set], source: int, limit: int = 10 ** 9) -> Dict[int, int]:
    dist = {source: 0}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        if dist[u] >= limit:
            continue
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


# ---------------------------------------------------------------------------
# Chapter 22 · Like next to like
# ---------------------------------------------------------------------------

VALUES = ("tokens", "age", "degree")
#: The values also shuffled among agents with the same number of connections — what position alone explains.
HELD = ("tokens", "age")
DEGREE_CUTS = [2, 3, 5, 10, 50]          # classes of connections: 1, 2, 3–4, 5–9, 10–49, 50 and more


#: Agents a likeness is measured out from, at each moment.
LIKENESS_SOURCES = 150


@D.measure("alike3")
def likeness(run_id: str) -> Dict[str, Any]:
    """
    How alike agents are at each distance, at five moments of a world's settled
    life, from LIKENESS_SOURCES agents drawn at random at each moment.
    """
    rng = np.random.default_rng(22)
    width = MAX_R + 1
    names = (*VALUES, *(f"{n}|k" for n in HELD))
    sums = {name: np.zeros((6, width)) for name in names}
    kin = np.zeros((2, width))
    neighbour = {name: [] for name in names}
    shuffled = {name: [] for name in VALUES}
    random_kin = []
    for t in MOMENTS:
        frame = D.frame_at(run_id, t, 2)
        adj = graph(frame)
        ids = list(frame["ids"])
        index = {u: i for i, u in enumerate(ids)}
        k = np.array([len(adj[u]) for u in ids])
        values = {"tokens": np.log1p(np.asarray(frame["tokens"], float)),
                  "age": np.log1p(np.asarray(frame["ages"], float)),
                  "degree": np.log(np.maximum(k, 1))}
        # Standardised within the moment, so that pooling five moments adds no likeness of its own.
        values = {name: (v - v.mean()) / v.std() for name, v in values.items()}
        classes = np.digitize(k, DEGREE_CUTS)
        for name in HELD:
            held = values[name].copy()
            for c in np.unique(classes):
                members = np.flatnonzero(classes == c)
                held[members] = rng.permutation(held[members])
            values[f"{name}|k"] = held
        geno = np.asarray(frame["brain_ids"])
        counts = np.unique(geno, return_counts=True)[1].astype(float)
        n = len(ids)
        random_kin.append(float((counts * (counts - 1)).sum() / (n * (n - 1))))
        pairs = np.array([(index[a], index[b]) for a, b in frame["edges"]
                          if a != b and a in index and b in index])
        src, dst = np.concatenate([pairs[:, 0], pairs[:, 1]]), np.concatenate([pairs[:, 1], pairs[:, 0]])
        for name, v in values.items():
            neighbour[name].append(float(np.corrcoef(v[src], v[dst])[0, 1]))
        for name in VALUES:
            v = rng.permutation(values[name])
            shuffled[name].append(float(np.corrcoef(v[src], v[dst])[0, 1]))
        for s in rng.choice(n, min(LIKENESS_SOURCES, n), replace=False):
            dist = bfs(adj, ids[s], MAX_R)
            reached = np.fromiter((index[u] for u in dist), int, len(dist))
            r = np.fromiter(dist.values(), int, len(dist))
            reached, r = reached[r > 0], r[r > 0]
            count = np.bincount(r, minlength=width)
            kin[0] += count
            kin[1] += np.bincount(r, weights=(geno[reached] == geno[s]).astype(float), minlength=width)
            for name, v in values.items():
                x, y = v[s], v[reached]
                by_r = np.bincount(r, weights=y, minlength=width)
                acc = sums[name]
                acc[0] += count
                acc[1] += count * x
                acc[2] += by_r
                acc[3] += count * x * x
                acc[4] += np.bincount(r, weights=y * y, minlength=width)
                acc[5] += x * by_r
    corr = {}
    for name, (m, sx, sy, sxx, syy, sxy) in sums.items():
        corr[name] = {}
        for r in range(1, width):
            if m[r] < 50:
                continue
            cov = sxy[r] / m[r] - sx[r] / m[r] * sy[r] / m[r]
            vx, vy = sxx[r] / m[r] - (sx[r] / m[r]) ** 2, syy[r] / m[r] - (sy[r] / m[r]) ** 2
            corr[name][str(r)] = float(cov / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else None
    return {"neighbour": neighbour, "shuffled": shuffled, "corr": corr,
            "kin": {str(r): (float(kin[1, r] / kin[0, r]) if kin[0, r] else None) for r in range(1, width)},
            "kin_pairs": {str(r): int(kin[0, r]) for r in range(1, width)},
            "random_kin": float(np.mean(random_kin))}


@chapter
def alike(ch: F.Chapter) -> None:
    alive = survivors(baseline())
    data = {s.run_id: likeness(s.run_id) for s in alive}
    names = [("tokens", "tokens", YELLOW), ("age", "age", GREEN), ("degree", "connections", BLUE)]
    cats = [label for _, label, _ in names]
    ch.figure(
        "neighbours", title="Are neighbours alike?",
        x={"label": "", "categories": cats}, y={"label": "correlation between neighbours", "min": -0.5, "max": 0.5},
        series=dots(cats, [[float(np.mean(d["neighbour"][n])) for d in data.values()] for n, _, _ in names],
                    [c for _, _, c in names]),
        guides=[{"axis": "y", "at": 0.0, "label": "no relation"}], legend=False,
        caption="For every connection of a world, the two agents at its ends: the correlation of their tokens, of "
                "their ages and of their numbers of connections (tokens and age as ln(1 + x), connections as "
                "ln k), counting every connection both ways round. One dot per world that lived to the end, the "
                "mean over five moments (after the games of iterations 1,000, 1,500, 2,000, 2,500 and 2,999); the "
                "bar is the median. The same values shuffled among a world's agents give correlations within "
                "0.01 of zero; the last column is the assortativity of Chapter 30, taken of ln k instead of k.",
        recipe=recipe(lived_text(), FRAMES,
                      ["Read frames 2·t + 1 for t = 1,000, 1,500, 2,000, 2,500, 2,999: `ids`, `tokens`, `ages`, "
                       "`brain_ids` and `edges`.",
                       "Pearson's correlation of the value at one end of a connection with the value at the other, "
                       "over every connection in both directions; the null shuffles the values among agents first "
                       "(`book_chapters.space.likeness`)."]))
    ch.number("neighbours", {n: {"world": describe([float(np.mean(d["neighbour"][n])) for d in data.values()]),
                                 "shuffled": describe([float(np.mean(d["shuffled"][n])) for d in data.values()])}
                             for n, _, _ in names})

    rs = np.arange(1, MAX_R + 1)

    def table(name: str) -> np.ndarray:
        return np.array([[d["corr"][name].get(str(r)) if d["corr"][name].get(str(r)) is not None else np.nan
                          for r in rs] for d in data.values()], float)
    panels = []
    for n, label, colour in names:
        q = np.nanquantile(table(n), [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
        series_ = [{"label": "the worlds", "x": rs.tolist(), "y": q[2].tolist(), "lo": q[1].tolist(),
                    "hi": q[3].tolist(), "outerLo": q[0].tolist(), "outerHi": q[4].tolist(), "colour": colour}]
        if n in HELD:
            series_.append(line(rs, np.nanmedian(table(f"{n}|k"), axis=0),
                                "shuffled among agents with as many connections", GREY, width=2, dash=[5, 4]))
        panels.append(dict(title=f"Of {label}", x={"label": "steps apart", "min": 0, "max": MAX_R},
                           y={"label": "correlation", "min": -0.3, "max": 0.5}, series=series_,
                           guides=[{"axis": "y", "at": 0.0, "label": ""}], legend=False))
        ch.number(f"distance_{n}", [describe(table(n)[:, i]) for i in range(len(rs))])
        if n in HELD:
            ch.number(f"distance_{n}_held", [describe(table(f"{n}|k")[:, i]) for i in range(len(rs))])
    ch.grid(
        "distance", panels, columns=3, title="How alike, how far apart",
        caption="For pairs of agents r steps apart, the correlation of their tokens, ages and connections (as in "
                "the figure above), for r = 1 to 15. For each world, pairs from breadth-first searches out of 150 "
                "agents drawn at random at each of the five moments; the coloured line is the median of the 26 "
                "worlds, the darker band the middle half of them and the paler band nine in ten. Grey dashes: the "
                "median when each agent's tokens (or age) are first shuffled among the agents with about as many "
                "connections (1, 2, 3–4, 5–9, 10–49, 50 and more) — what an agent's place in the network explains "
                "without any closeness of its own.",
        recipe=recipe(lived_text(), FRAMES,
                      ["At each moment, a breadth-first search out of 150 agents drawn at random (generator seeded "
                       "with 22), to 15 steps; every agent reached at r steps gives the pair (source, agent).",
                       "Standardise each value within its moment (subtract the mean, divide by the standard "
                       "deviation); per r, pooled over sources and moments, Pearson's correlation of the two values.",
                       "The grey null: the same pairs, with tokens and ages permuted at random within each class "
                       "of connections (`book_chapters.space.likeness`)."]))
    ch.number("neighbours_held", {n: describe([float(np.mean(d["neighbour"][f"{n}|k"])) for d in data.values()])
                                  for n in HELD})

    kin = np.array([[d["kin"].get(str(r)) if d["kin"].get(str(r)) is not None else np.nan for r in rs]
                    for d in data.values()], float)
    base = np.array([d["random_kin"] for d in data.values()])
    enrich = kin / base[:, None]
    near = rs <= 8
    shown = np.where(enrich > 0, enrich, np.nan)[:, near]
    q = np.nanquantile(shown, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    ch.figure(
        "kin", title="Kin, near and far",
        x={"label": "steps apart", "min": 0, "max": 8},
        y={"label": "same genotype, times as often as for any two (logarithmic)", "log": True, "min": 0.01, "max": 200},
        series=[{"label": None, "x": rs[near].tolist(), "y": q[2].tolist(), "lo": q[1].tolist(), "hi": q[3].tolist(),
                 "outerLo": q[0].tolist(), "outerHi": q[4].tolist(), "colour": VIOLET}],
        guides=[{"axis": "y", "at": 1.0, "label": "as for any two agents"}], legend=False,
        caption="For pairs of agents r steps apart (as in the figure above), the share that carry the same "
                "genotype, divided by the share among all pairs of agents of the world, on a logarithmic axis: "
                "a straight falling line would be a relatedness that halves over a fixed number of steps. The "
                "line is the median of the 26 worlds, the darker band the middle half of them and the paler band "
                "nine in ten. From 9 steps on, most worlds have no such pair at all, and none has a twentieth of what "
                "chance would give.",
        recipe=recipe(lived_text(), FRAMES,
                      ["As above; for every pair, whether the two `brain_ids` are equal.",
                       "Any two agents: Σ c(c − 1) / (n(n − 1)) over the genotypes' counts c among the n agents."]))
    ch.number("kin", {"enrichment": [describe(enrich[:, i]) for i in range(len(rs))],
                      "share": [describe(kin[:, i]) for i in range(len(rs))], "random": describe(base)})


# ---------------------------------------------------------------------------
# Chapter 23 · How many dimensions does a world have?
# ---------------------------------------------------------------------------

def ring(n: int) -> Dict[int, set]:
    return {i: {(i - 1) % n, (i + 1) % n} for i in range(n)}


def torus(side: int, dim: int) -> Dict[int, set]:
    import itertools
    coords = list(itertools.product(range(side), repeat=dim))
    index = {c: i for i, c in enumerate(coords)}
    adj = {}
    for c, i in index.items():
        nb = set()
        for d in range(dim):
            for step in (-1, 1):
                c2 = list(c)
                c2[d] = (c2[d] + step) % side
                nb.add(index[tuple(c2)])
        adj[i] = nb
    return adj


def regular(n: int, k: int, seed: int) -> Dict[int, set]:
    import networkx as nx
    g = nx.random_regular_graph(k, n, seed=seed)
    return {u: set(g[u]) for u in g}


def tree(n: int, seed: int) -> Dict[int, set]:
    """A tree drawn uniformly at random among all trees on n points."""
    import networkx as nx
    g = nx.random_labeled_tree(n, seed=seed)
    return {u: set(g[u]) for u in g}


def core(adj: Dict[int, set]) -> Dict[int, set]:
    """What is left when dead ends are pruned until none is left: every agent with one connection, again and again."""
    import networkx as nx
    g = nx.k_core(nx.Graph([(u, v) for u in adj for v in adj[u]]), 2)
    return {u: set(g[u]) for u in g}


def ball_growth(adj: Dict[int, set], sources: int, seed: int, limit: int = 40) -> np.ndarray:
    """
    The typical number of agents within r steps, r = 0 … limit: the geometric mean over
    sources drawn at random, so that the few sources beside a hub do not outweigh the rest.
    """
    rng = np.random.default_rng(seed)
    nodes = list(adj)
    logs = []
    for i in rng.choice(len(nodes), min(sources, len(nodes)), replace=False):
        dist = bfs(adj, nodes[i], limit)
        counts = np.bincount(np.fromiter(dist.values(), int), minlength=limit + 1)[:limit + 1]
        logs.append(np.log(np.cumsum(counts)))
    return np.exp(np.mean(logs, axis=0))


def local_dimension(balls: np.ndarray, n: int) -> Tuple[List[float], List[float]]:
    """d(r) = ln(N(r+1) / N(r)) / ln((r+1) / r), while the ball holds less than a quarter of the world."""
    xs, ds = [], []
    for r in range(1, len(balls) - 1):
        if balls[r + 1] > n / 4:
            break
        xs.append(float(np.sqrt(r * (r + 1))))
        ds.append(float(np.log(balls[r + 1] / balls[r]) / np.log((r + 1) / r)))
    return xs, ds


def returns(adj: Dict[int, set], sources: int, seed: int, steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    How often a lazy random walk (stay with probability ½, else step to a neighbour drawn
    at random) is back where it started after t steps, averaged over sources — with the
    probability it would have there at rest subtracted, so a finite world's floor is gone.
    """
    nodes = list(adj)
    at = {u: i for i, u in enumerate(nodes)}
    src = np.array([at[u] for u in nodes for v in adj[u]])
    dst = np.array([at[v] for u in nodes for v in adj[u]])
    degree = np.bincount(src, minlength=len(nodes)).astype(float)
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(nodes), min(sources, len(nodes)), replace=False)
    p = np.zeros((len(nodes), len(picked)))
    p[picked, np.arange(len(picked))] = 1.0
    rest = degree[picked] / degree.sum()
    out = np.zeros(steps)
    order = np.argsort(dst, kind="stable")
    src_o, dst_o = src[order], dst[order]
    starts = np.searchsorted(dst_o, np.arange(len(nodes)))
    has = np.diff(np.append(starts, len(dst_o))) > 0
    for t in range(steps):
        q = p / degree[:, None]
        moved = np.zeros_like(p)
        sums = np.add.reduceat(q[src_o], starts[has], axis=0)
        moved[np.arange(len(nodes))[has]] = sums
        p = 0.5 * p + 0.5 * moved
        out[t] = float(np.mean(p[picked, np.arange(len(picked))] - rest))
    return np.arange(1, steps + 1, dtype=float), out


def spectral_dimension(ts: np.ndarray, ps: np.ndarray, floor: float) -> Tuple[List[float], List[float]]:
    """
    d_s(t) = −2 · ln(P(2t) / P(t)) / ln 2, for t on a grid of ratio √2, while P(2t)
    stays above a floor — below it the walk has felt the edge of a finite world.
    """
    xs, ds = [], []
    for t in sorted({int(round(2 ** (k / 2))) for k in range(40)}):
        if 2 * t > len(ts):
            break
        a, b = ps[t - 1], ps[2 * t - 1]
        if b <= floor:
            break
        xs.append(float(np.sqrt(t * 2 * t)))
        ds.append(float(-2 * np.log(b / a) / np.log(2)))
    return xs, ds


def largest_piece(adj: Dict[int, set]) -> Dict[int, set]:
    import gol_spectral
    piece = set(gol_spectral._largest_component(sorted(adj), adj))
    return {u: adj[u] & piece for u in piece}


def last_world(run_id: str) -> Dict[int, set]:
    return graph(D.frame_at(run_id, D.last_iteration(run_id), 2))


def measure(adj: Dict[int, set], seed: int, steps: int = 512, balls: int = 40, walks: int = 24) -> Dict[str, Any]:
    """Both rulers on the largest piece of a network."""
    adj = largest_piece(adj)
    n = len(adj)
    volumes = ball_growth(adj, balls, seed)
    bx, bd = local_dimension(volumes, n)
    ts, ps = returns(adj, walks, seed, steps)
    sx, sd = spectral_dimension(ts, ps, 5 / n)
    return {"n": n, "balls": volumes.tolist(), "ball_r": bx, "ball_d": bd, "spec_t": sx, "spec_d": sd}


#: Agents above which a world's dimensions are read with more balls and longer walks:
#: the worlds of 409,600 tokens (52,000 to 70,000), and none smaller (at most 30,000).
LARGE_WORLD = 40_000


@D.measure("dims2")
def world_dimensions(run_id: str) -> Dict[str, Any]:
    """
    Both rulers on a world after its last game, on its random twin and on its
    core. A world of more than LARGE_WORLD agents is read with more balls and
    longer walks, which its size allows and its wider range of scales needs.
    """
    from book_chapters.structure import degree_preserving
    adj = last_world(run_id)
    size = dict(steps=1024, balls=100, walks=48) if len(adj) > LARGE_WORLD else {}
    return {"world": measure(adj, 23, **size), "twin": measure(degree_preserving(adj, 23), 23, **size),
            "core": measure(core(adj), 23, **size)}


LIFE = (10, 25, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 2999)


def at(xs: List[float], ds: List[float], x: float) -> float:
    """A ruler's reading at one scale: the value at the grid point nearest x (logarithmically)."""
    if not xs:
        return float("nan")
    k = int(np.argmin(np.abs(np.log(np.array(xs)) - np.log(x))))
    return float(ds[k]) if abs(np.log(xs[k] / x)) < 0.2 else float("nan")


@D.measure("lifedims2")
def life_dimensions(run_id: str) -> Dict[str, Any]:
    """Both rulers, read at one scale each, after the games of a few iterations of a world's life."""
    out = {"t": [], "agents": [], "ball": [], "spectral": []}
    for t in LIFE:
        m = measure(graph(D.frame_at(run_id, t, 2)), 23, 128)
        out["t"].append(t)
        out["agents"].append(m["n"])
        out["ball"].append(at(m["ball_r"], m["ball_d"], np.sqrt(6)))
        out["spectral"].append(at(m["spec_t"], m["spec_d"], np.sqrt(32)))
    return out


def plateau(xs: List[float], ds: List[float], lo: float, hi: float) -> float:
    """A ruler's typical reading over a range of scales: the median of its values there."""
    inside = [d for x, d in zip(xs, ds) if lo <= x <= hi]
    return float(np.median(inside)) if inside else float("nan")


def band(rows: List[Dict[str, Any]], xkey: str, ykey: str) -> Dict[str, Any]:
    """The median and the bands over networks of one ruler's readings, where at least half of them read."""
    grid = sorted({round(x, 6) for d in rows for x in d[xkey]})
    table = np.full((len(rows), len(grid)), np.nan)
    for i, d in enumerate(rows):
        for x, y in zip(d[xkey], d[ykey]):
            table[i, grid.index(round(x, 6))] = y
    ok = np.sum(np.isfinite(table), axis=0) >= max(3, len(rows) // 2)
    q = np.nanquantile(table[:, ok], [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    return {"x": np.array(grid)[ok].tolist(), "y": q[2].tolist(), "lo": q[1].tolist(), "hi": q[3].tolist(),
            "outerLo": q[0].tolist(), "outerHi": q[4].tolist()}


RULERS = (("ball_r", "ball_d", "From balls", "radius r, steps (logarithmic)"),
          ("spec_t", "spec_d", "From a random walk", "steps of the walk t (logarithmic)"))
BIG = ("The three worlds of 409,600 tokens of Experiment 8 (`B1-409600-s001` … `-s003`, Chapter 38; "
       "`python3 gol_lab.py run E08`), after their last game.")


@chapter
def dimensions(ch: F.Chapter) -> None:
    known = [("a ring (1)", ring(2000), GREY), ("a square grid (2)", torus(45, 2), GREEN),
             ("a cubic grid (3)", torus(13, 3), BLUE), ("a random tree", tree(2000, 23), ORANGE),
             ("a random network", regular(2000, 3, 23), RED)]
    measured = {name: measure(adj, 23) for name, adj, _ in known}
    ch.grid(
        "rulers", [dict(title=title, x={"label": xl, "log": True}, y={"label": "dimension", "min": 0, "max": 6},
                        series=[line(measured[name][xk], measured[name][yk], name, colour, width=2)
                                for name, _, colour in known],
                        guides=[{"axis": "y", "at": d, "label": ""} for d in (1, 2, 3)])
                   for xk, yk, title, xl in RULERS],
        columns=2, title="Two rulers, tried on spaces whose dimension is known",
        caption="Five networks of about 2,000 points each: a ring, a square grid and a cubic grid, each wrapped "
                "round so that it has no edge (dimension 1, 2 and 3); a tree drawn at random among all trees on "
                "2,000 points; and a random network in which every point has three neighbours. Left: the local "
                "dimension from how balls grow, d(r) = ln(N(r+1)/N(r)) / ln((r+1)/r), where N(r) is the typical "
                "number of points within r steps (the geometric mean over 40 points drawn at random), until a ball "
                "holds a quarter of the network. Right: the spectral dimension from a random walk, "
                "d(t) = −2 · ln(P(2t)/P(t)) / ln 2, where P(t) is how much more often than at rest a walk is back "
                "at its start after t steps (averaged over 24 starts), until P(2t) falls below 5/n.",
        recipe=recipe("No runs: the five networks are built by `book_chapters.space` (`ring`, `torus`, `tree` and "
                      "`regular`, the last two with networkx, seed 23).", "the networks themselves",
                      ["Balls: a breadth-first search out of each of 40 points drawn at random (seed 23); N(r) the "
                       "geometric mean, over the 40, of the number within r steps.",
                       "Walk: a lazy random walk — stay with probability ½, else step to a neighbour drawn at "
                       "random — started at each of 24 points, its distribution propagated exactly step by step; "
                       "P(t) is the probability of being at the start minus the probability of being there at "
                       "rest (the start's connections over twice all connections).",
                       "Read d(r) and d(t) as in the caption; t runs over a grid of ratio √2."]))
    ch.number("rulers", measured)
    ch.number("rulers_plateau", {name: {"ball": plateau(m["ball_r"], m["ball_d"], 3, 40),
                                        "walk": plateau(m["spec_t"], m["spec_d"], 4, 64)}
                                 for name, m in measured.items()})

    alive = survivors(baseline())
    dims = [world_dimensions(s.run_id) for s in alive]
    big = [s for s in runs_of("E08", "baseline") if int(s.lab["world"]["total_tokens"]) == 409600
           and F.lived(s) >= s.until]
    bigs = [world_dimensions(s.run_id) for s in big]
    panels = []
    for xk, yk, title, xl in RULERS:
        series_ = [{**band([d["world"] for d in dims], xk, yk), "label": "the 26 worlds", "colour": BLUE},
                   {**band([d["twin"] for d in dims], xk, yk), "label": "their random twins", "colour": RED}]
        for i, d in enumerate(bigs):
            series_.append(line(d["world"][xk], d["world"][yk], "worlds of 409,600 tokens" if i == 0 else None,
                                YELLOW, width=2))
            series_.append(line(d["twin"][xk], d["twin"][yk], "the large worlds' twins" if i == 0 else None,
                                ORANGE, width=1.5, dash=[5, 4]))
        panels.append(dict(title=title, x={"label": xl, "log": True}, y={"label": "dimension", "min": 0, "max": 8},
                           series=series_, guides=[{"axis": "y", "at": d, "label": ""} for d in (1, 2, 3)]))
    ch.grid(
        "worlds", panels, columns=2, title="The same rulers, on the worlds",
        caption="The two rulers of the figure above, on the largest piece of each world after its last game. "
                "Blue: the 26 baseline worlds of 10,000 tokens that lived to the end (about 1,300 agents each) — "
                "the line is the median of the worlds, the darker band the middle half and the paler band nine in "
                "ten. Red: for each, its random twin, a network with exactly as many connections at every agent "
                "but wired at random. Yellow: the three worlds of 409,600 tokens of Chapter 38 (about 60,000 "
                "agents each), measured from 100 balls and 48 walks of up to 1,024 steps; dashed orange, their "
                "random twins.",
        recipe=recipe(lived_text() + " " + BIG, FRAMES,
                      ["Read each run's last frame; build its network from `edges` and keep its largest piece.",
                       "The rulers as in the figure above (`book_chapters.space.measure`); the twin by "
                       "`book_chapters.structure.degree_preserving` (seed 23)."]))

    def plateaus(rows, ball, walk):
        return {"ball": describe([plateau(d["ball_r"], d["ball_d"], *ball) for d in rows]),
                "walk": describe([plateau(d["spec_t"], d["spec_d"], *walk) for d in rows]),
                "agents": describe([d["n"] for d in rows]),
                "reach": describe([max(d["ball_r"]) if d["ball_r"] else np.nan for d in rows])}
    ch.number("worlds", {kind: plateaus([d[kind] for d in dims], (2, 6), (4, 64)) for kind in ("world", "twin", "core")})
    ch.number("big", {kind: plateaus([d[kind] for d in bigs], (3, 14), (4, 400)) for kind in ("world", "twin", "core")})
    ch.number("big_curves", bigs)

    r = np.arange(1, 41)
    anchor = float(np.median([d["world"]["balls"][4] for d in bigs]))
    series_ = []
    for i, d in enumerate(bigs):
        for kind, label, colour, dash in (("world", "worlds of 409,600 tokens", YELLOW, None),
                                          ("twin", "their random twins", ORANGE, [5, 4])):
            v = np.array(d[kind]["balls"][1:], float)
            keep = v < d[kind]["n"] * 0.999
            series_.append(line(r[keep], v[keep], label if i == 0 else None, colour, width=2,
                                **({"dash": dash} if dash else {})))
    for power, label, dash in ((2, "r²", [2, 3]), (3, "r³", [8, 4])):
        series_.append(line(r, anchor * (r / 4) ** power, f"growth like {label}", GREY, width=1, dash=dash))
    ch.figure(
        "growth", title="How a large world fills out",
        x={"label": "radius r, steps (logarithmic)", "log": True, "min": 1, "max": 40},
        y={"label": "agents within r steps (logarithmic)", "log": True, "min": 1, "max": 1e5},
        series=series_,
        caption="The typical number of agents within r steps (the geometric mean over 100 agents drawn at "
                "random) in the three worlds of 409,600 tokens after their last game (yellow), and in their "
                "random twins (dashed orange), on logarithmic axes, until the ball holds the whole world. In a "
                "space of dimension d, the number grows like r^d, a straight line of slope d: in grey, r² "
                "(dotted, the shallower) and r³ (dashed, the steeper), drawn through the worlds' typical ball of "
                "radius 4.",
        recipe=recipe(BIG, FRAMES,
                      ["Read each run's last frame; build its network from `edges` and keep its largest piece; "
                       "make its random twin as above.",
                       "From each of 100 agents drawn at random (seed 23), a breadth-first search; N(r) the "
                       "geometric mean of the number within r steps."]))

    lives = {s.run_id: life_dimensions(s.run_id) for s in alive}
    panels = []
    for key, title, colour in (("ball", "From balls, at r = 2 to 3", BLUE),
                               ("spectral", "From a walk, at t = 4 to 8", VIOLET)):
        table = np.array([d[key] for d in lives.values()], float)
        q = np.nanquantile(table, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
        panels.append(dict(title=title, x={"label": "iteration (logarithmic)", "log": True},
                           y={"label": "dimension", "min": 0, "max": 6},
                           series=[{"label": None, "x": list(LIFE), "y": q[2].tolist(), "lo": q[1].tolist(),
                                    "hi": q[3].tolist(), "outerLo": q[0].tolist(), "outerHi": q[4].tolist(),
                                    "colour": colour}],
                           guides=[{"axis": "y", "at": d, "label": ""} for d in (1, 2, 3)], legend=False))
        ch.number(f"life_{key}", {str(t): describe(table[:, i]) for i, t in enumerate(LIFE)})
    ch.grid(
        "life", panels, columns=2, title="Dimension over a world's life",
        caption="The two rulers, each read at one scale, after the games of iterations 10, 25, 50, 100, 200, 300, "
                "500, 750, 1,000, 1,500, 2,000, 2,500 and 2,999 of the 26 baseline worlds that lived to the end, on "
                "a logarithmic axis of time. Left: the local dimension from balls between radius 2 and 3; right: "
                "the spectral dimension from walks of 4 to 8 steps. The line is the median of the worlds, the "
                "darker band the middle half of them and the paler band nine in ten. The viewer's own dimensions, "
                "every 25 iterations, are drawn in Chapter 31.",
        recipe=recipe(lived_text(), FRAMES,
                      ["Read frame 2·t + 1 for each t; take the largest piece of its network.",
                       "The rulers as in the first figure, the walk run for 128 steps "
                       "(`book_chapters.space.life_dimensions`); read the ball ruler at r = √6 and the walk at "
                       "t = √32."]))
    ch.number("life_agents", {str(t): describe([d["agents"][i] for d in lives.values()]) for i, t in enumerate(LIFE)})
