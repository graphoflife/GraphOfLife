# -*- coding: utf-8 -*-
"""
Part III, the second half: Chapters 23 to 26 — how agents' properties scale
together, the geometry of a world, how it breaks, and one world drawn in the
colours the viewer offers.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

import book_figures as F
from book_figures import (BASELINE_RUNS, FRAMES, STATS_FILE, band_series, chapter, describe, dots,
                          line, mean_over, recipe, series, survivors)
from book_chapters.common import (BAND_STEPS, BAND_WORDS, BLUE, CYAN, GREEN, GREY, ORANGE, RED,
                                  VIOLET, YELLOW, baseline)
from book_chapters.measures import DEGREE_CLASSES, SURVIVORS, bars, in_class, settled_rows

LAST = 2999


def graph(frame: Dict[str, Any]) -> Dict[int, set]:
    adj: Dict[int, set] = {a: set() for a in frame["ids"]}
    for a, b in frame["edges"]:
        if a != b and a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)
    return adj


def triangles(adj: Dict[int, set]) -> Dict[int, int]:
    """How many triangles each agent is a corner of."""
    out = Counter()
    for a, na in adj.items():
        for b in na:
            if b <= a:
                continue
            for c in na & adj[b]:
                if c > b:
                    out[a] += 1
                    out[b] += 1
                    out[c] += 1
    return out


def log_classes(values: np.ndarray, edges: Sequence[float]) -> List[Tuple[float, float, np.ndarray]]:
    """(geometric middle, width bounds, mask) for each class [edges[i], edges[i+1])."""
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (values >= a) & (values < b)
        if m.any():
            out.append((float(np.sqrt(a * (b - 1 if b - 1 >= a else b))), (a, b), m))
    return out


DEGREE_EDGES = [1, 2, 3, 4, 6, 9, 13, 20, 30, 50, 100, 200, 10 ** 5]


def last_frames() -> List[Tuple[Any, Dict[str, Any]]]:
    return [(s, F.store.read_frame(s.run_id, 2 * LAST + 1)) for s in survivors(baseline())]


# ---------------------------------------------------------------------------
# Chapter 23 · How properties scale together
# ---------------------------------------------------------------------------

@chapter
def scaling(ch: F.Chapter) -> None:
    alive = survivors(baseline())
    deg, tok, tri, clu_k, clu, knn_k, knn = [], [], [], [], [], [], []
    for s, frame in last_frames():
        adj = graph(frame)
        t = triangles(adj)
        tokens = dict(zip(frame["ids"], frame["tokens"]))
        for a, na in adj.items():
            k = len(na)
            if k == 0:
                continue
            deg.append(k)
            tok.append(tokens[a])
            tri.append(t.get(a, 0))
            if k >= 2:
                clu_k.append(k)
                clu.append(2 * t.get(a, 0) / (k * (k - 1)))
            knn_k.append(k)
            knn.append(np.mean([len(adj[b]) for b in na]))
    deg, tok, tri = np.array(deg, float), np.array(tok, float), np.array(tri, float)
    clu_k, clu, knn_k, knn = map(lambda v: np.array(v, float), (clu_k, clu, knn_k, knn))

    # Tokens against degree: a two-dimensional histogram.
    tedges = [1, 2, 3, 4, 6, 9, 13, 20, 30, 50, 100, 200, 500, 1000, 5000]
    cells = {"kind": "cells", "x0": [], "x1": [], "y0": [], "y1": [], "value": []}
    for _, (a, b), m in log_classes(deg, DEGREE_EDGES):
        for _, (c, d), m2 in log_classes(tok[m], tedges):
            cells["x0"].append(a)
            cells["x1"].append(b)
            cells["y0"].append(c)
            cells["y1"].append(d)
            cells["value"].append(int(m2.sum()))
    slope, intercept = np.polyfit(np.log(deg), np.log(tok), 1)
    ks = np.array([1, 400])
    mid = [(np.sqrt(a * b), float(np.median(tok[m]))) for _, (a, b), m in log_classes(deg, DEGREE_EDGES)]
    ch.figure(
        "tokens-degree", title="Tokens against connections",
        x={"label": "connections k (logarithmic)", "log": True, "min": 1, "max": 400},
        y={"label": "tokens (logarithmic)", "log": True, "min": 1, "max": 5000},
        series=[cells,
                line([m[0] for m in mid], [m[1] for m in mid], "median tokens of each class of k", "#eef4fa",
                     width=2),
                line(ks, np.exp(intercept) * ks ** slope, f"least squares: tokens ∝ k^{slope:.2f}", RED,
                     width=1.5, dash=[5, 4]),
                line(ks, mid[0][1] * (ks + 1) / 2, "tokens ∝ k + 1 (an even split's resting state)", CYAN,
                     width=1.5, dash=[2, 3])],
        colourbar={"map": "viridis", "min": 1, "max": max(cells["value"]), "label": "agents", "log": True},
        caption="Every agent alive after the last game of the 26 worlds that lived to the end "
                f"({len(deg):,} agents), counted in cells of connections k and tokens, both in classes of "
                "growing width; a cell's colour is how many agents it holds, on a logarithmic scale. White: the "
                "median tokens in each class of k. Red: the straight line least squares fits through ln(tokens) "
                "against ln(k) over all agents ([Scaling relations](../notes/scaling-relations.md)). Cyan: "
                "tokens proportional to k + 1, through the median at k = 1 — where an even split of every "
                "stake would leave the tokens on a network that held still "
                "([Chapter 20](20-where-the-tokens-flow.md)).",
        recipe=recipe(SURVIVORS, FRAMES,
                      ["Read frame `5999` of every run: each agent's tokens and its connections in `edges`.",
                       "Count the agents in each cell of the classes shown; colour by the count.",
                       "Fit ln(tokens) = a + b·ln(k) by least squares over all agents (the viewer's "
                       "`tokensVsDegree` does this for one frame)."]))
    ch.number("tokens_degree", {"slope": float(slope), "r2": float(np.corrcoef(np.log(deg), np.log(tok))[0, 1] ** 2),
                                "agents": int(len(deg))})
    # Tokens per candidate: constant across classes at the even split's resting state.
    per = tok / (deg + 1)
    ch.number("tokens_per_candidate", {
        "resting": float(tok.sum() / (deg + 1).sum()),
        # The exponent least squares would report if every agent held exactly its resting share.
        "slope_if_resting": float(np.polyfit(np.log(deg), np.log(deg + 1), 1)[0]),
        "classes": {lab: {"median": float(np.median(per[m])), "mean": float(per[m].mean())}
                    for lab, m in zip([c[2] for c in DEGREE_CLASSES], in_class(deg, DEGREE_CLASSES))}})

    def classes_mean(x, y, positive=True):
        out = []
        for _, (a, b), m in log_classes(x, DEGREE_EDGES):
            v = y[m]
            if positive:
                v = v[v > 0]
            if v.size >= 10:
                # The geometric mean, the mean on the logarithmic axis the line is fitted on.
                out.append((float(np.sqrt(a * max(a, b - 1))),
                            float(np.exp(np.log(v).mean())) if positive else float(v.mean())))
        return np.array(out)

    c = classes_mean(clu_k, clu)
    s1, i1 = np.polyfit(np.log(clu_k[clu > 0]), np.log(clu[clu > 0]), 1)
    tcls = classes_mean(deg, tri)
    s2, i2 = np.polyfit(np.log(deg[tri > 0]), np.log(tri[tri > 0]), 1)
    kk = np.array([2, 300])
    ch.grid(
        "clustering-triangles", [
            dict(title="Clustering against connections", x={"label": "connections k (logarithmic)", "log": True},
                 y={"label": "clustering (logarithmic)", "log": True},
                 series=[{"label": "geometric mean of each class of k", "x": c[:, 0].tolist(), "y": c[:, 1].tolist(),
                          "points": True, "size": 8, "colour": GREEN},
                         line(kk, np.exp(i1) * kk ** s1, f"least squares: slope {s1:.2f}", RED, width=1.5, dash=[5, 4]),
                         line(kk, c[0, 1] * (kk / kk[0]) ** -1.0, "slope −1", GREY, width=1, dash=[2, 3])]),
            dict(title="Triangles against connections", x={"label": "connections k (logarithmic)", "log": True},
                 y={"label": "triangles (logarithmic)", "log": True},
                 series=[{"label": "geometric mean of each class of k", "x": tcls[:, 0].tolist(), "y": tcls[:, 1].tolist(),
                          "points": True, "size": 8, "colour": VIOLET},
                         line(kk, np.exp(i2) * kk ** s2, f"least squares: slope {s2:.2f}", RED, width=1.5, dash=[5, 4]),
                         line(kk, tcls[0, 1] * (kk / kk[0]) ** 2, "slope 2", GREY, width=1, dash=[2, 3])])],
        columns=2, title="How an agent's neighbourhood closes, by its connections",
        caption="Agents alive after the last game of the 26 worlds that lived to the end, in classes of their "
                "connections k. Left: the geometric mean (the mean of the logarithms, turned back) of the clustering "
                "coefficient — the share of pairs of an agent's "
                "neighbours that are joined to each other — of the agents with k ≥ 2 and clustering above 0. "
                "Right: the geometric mean of the number of triangles an agent is a corner of, over the agents "
                "in at least one. "
                "Red: least squares on the logarithms of all those agents (not of the class means); grey: the "
                "slopes −1 and 2 for comparison.",
        recipe=recipe(SURVIVORS, FRAMES,
                      ["Read frame `5999` of every run; for every agent count its triangles: pairs of its "
                       "neighbours that are joined.",
                       "Clustering of an agent with k ≥ 2 neighbours = triangles / (k(k − 1)/2).",
                       "Average ln(y) by class of k and turn it back for the dots; fit ln(y) on ln(k) over the agents with y > 0 for the "
                       "red lines (`clusteringVsDegree`, `trianglesVsDegree` in the viewer)."]))
    ch.number("clustering_slope", float(s1))
    ch.number("triangles_slope", float(s2))

    n = classes_mean(knn_k, knn, positive=False)
    assort = [mean_over(s.run_id, "assortativity", 500, 2999) for s in alive]
    ch.figure(
        "neighbours", title="Whom the well-connected are joined to",
        x={"label": "connections k of an agent (logarithmic)", "log": True},
        y={"label": "mean connections of its neighbours (logarithmic)", "log": True},
        series=[{"label": "mean of each class of k", "x": n[:, 0].tolist(), "y": n[:, 1].tolist(),
                 "points": True, "size": 8, "colour": ORANGE},
                line(n[:, 0], n[:, 1], None, ORANGE, width=1)],
        caption="For the agents alive after the last game of the 26 worlds that lived to the end, in classes of "
                "their connections k: the mean, over the class, of the average number of connections of an "
                "agent's neighbours. A falling line means the well-connected are joined mostly to the poorly "
                "connected ([Assortativity](../notes/assortativity.md)).",
        recipe=recipe(SURVIVORS, FRAMES,
                      ["Read frame `5999` of every run; for every agent, the mean degree of its neighbours.",
                       "Average by class of the agent's own degree."]))
    ch.number("assortativity_settled", describe(assort))
    ch.number("neighbours", [{"k": float(k), "mean": float(v)} for k, v in n])

    names = ["tokensVsDegree", "trianglesVsDegree", "clusteringVsDegree", "changeVsTokens"]
    labels = ["tokens ~ k", "triangles ~ k", "clustering ~ k", "|change| ~ tokens"]
    ex = [[mean_over(s.run_id, st, 500, 2999) for s in alive] for st in names]
    r2 = [[mean_over(s.run_id, st + "R2", 500, 2999) for s in alive] for st in names]
    ch.grid(
        "exponents", [
            dict(title="Exponents", x={"label": "", "categories": labels}, y={"label": "slope on log–log axes"},
                 series=dots(labels, ex, [YELLOW, VIOLET, GREEN, CYAN]), legend=False,
                 guides=[{"axis": "y", "at": 0}]),
            dict(title="How tightly the agents follow them", x={"label": "", "categories": labels},
                 y={"label": "R²", "min": 0, "max": 1}, series=dots(labels, r2, [YELLOW, VIOLET, GREEN, CYAN]),
                 legend=False)],
        columns=2, title="Four scaling relations in the 26 worlds",
        caption="One dot per world that lived to the end: its mean, over the measurements every 25 iterations "
                "from 500 on, of four slopes fitted by least squares on the logarithms of the agents of a frame, "
                "and of their R². Tokens and triangles against connections, clustering against connections, and "
                "the size of an agent's change in tokens over the game against the tokens it holds "
                "([Scaling relations](../notes/scaling-relations.md)).",
        recipe=recipe(SURVIVORS, STATS_FILE,
                      ["Average `tokensVsDegree`, `trianglesVsDegree`, `clusteringVsDegree` and "
                       "`changeVsTokens`, and the same with `R2` appended, over the rows with `phase` = 2 and "
                       "500 ≤ `iteration` ≤ 2,999.", "One dot per run, a bar at the median."]))
    for st, a, b in zip(names, ex, r2):
        ch.number(f"{st}_settled", {"slope": describe(a), "r2": describe(b)})


# ---------------------------------------------------------------------------
# Chapter 24 · The geometry of a world
# ---------------------------------------------------------------------------

def exact_geometry(adj: Dict[int, set]) -> Dict[str, float]:
    """Exact distances, the spectral gap and the loops of the largest piece of a network."""
    import gol_spectral
    piece = gol_spectral._largest_component(sorted(adj), adj)
    inside = set(piece)
    sub = {a: {b for b in adj[a] if b in inside} for a in piece}
    total = pairs = 0
    eccentricity = []
    for s in piece:
        dist = {s: 0}
        queue = deque([s])
        while queue:
            u = queue.popleft()
            for w in sub[u]:
                if w not in dist:
                    dist[w] = dist[u] + 1
                    queue.append(w)
        total += sum(dist.values())
        pairs += len(dist) - 1
        eccentricity.append(max(dist.values()))
    edges = sum(len(v) for v in sub.values()) // 2
    return {"agents": len(piece), "mean_path": total / pairs, "diameter": max(eccentricity),
            "radius": min(eccentricity), "gap": gol_spectral.spectral_gap(piece, sub),
            "cycle_rank": edges - len(piece) + 1}


def degree_preserving(adj: Dict[int, set], seed: int) -> Dict[int, set]:
    """A random network with the same numbers of connections (a configuration model, made simple)."""
    import networkx as nx
    ids = sorted(adj)
    g = nx.Graph(nx.configuration_model([len(adj[a]) for a in ids], seed=seed))
    g.remove_edges_from(list(nx.selfloop_edges(g)))
    return {u: set(g[u]) for u in g}


def balls(adj: Dict[int, set], sources: int = 24, radius: int = 40) -> np.ndarray:
    """The mean number of agents within r steps of a source, for r = 0 … radius."""
    ids = sorted(adj)
    step = max(1, len(ids) // sources)
    volumes = np.zeros(radius + 1)
    used = 0
    for s in ids[::step][:sources]:
        dist = {s: 0}
        queue = deque([s])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        counts = np.bincount(np.minimum(list(dist.values()), radius + 1), minlength=radius + 2)[:radius + 1]
        volumes += np.cumsum(counts)
        used += 1
    return volumes / used


def box_counts(adj: Dict[int, set], sizes=(1, 3, 5, 9, 17, 33)) -> List[Tuple[int, int]]:
    """How many boxes of each size cover the graph, greedily from the best-connected (as gol_series does)."""
    position = {node: k for k, node in enumerate(adj)}
    order = sorted(adj, key=lambda i: (-len(adj[i]), position[i]))
    out = []
    for size in sizes:
        radius = (size - 1) // 2
        covered, boxes = set(), 0
        for seed in order:
            if seed in covered:
                continue
            boxes += 1
            covered.add(seed)
            seen, frontier = {seed}, [seed]
            for _ in range(radius):
                nxt = []
                for u in frontier:
                    for v in adj[u]:
                        if v not in seen:
                            seen.add(v)
                            covered.add(v)
                            nxt.append(v)
                frontier = nxt
                if not frontier:
                    break
        out.append((size, boxes))
        if boxes <= 1:
            break
    return out


def fundamental_cycles(adj: Dict[int, set]) -> Dict[int, int]:
    """How many cycles of a breadth-first cycle basis pass through each agent (as the viewer counts)."""
    parent, depth = {}, {}
    for root in adj:
        if root in parent:
            continue
        parent[root], depth[root] = None, 0
        queue = deque([root])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v not in parent:
                    parent[v], depth[v] = u, depth[u] + 1
                    queue.append(v)
    count = Counter()
    for a in adj:
        for b in adj[a]:
            if b <= a or parent.get(a) == b or parent.get(b) == a:
                continue
            x, y, nodes = a, b, []
            while depth[x] > depth[y]:
                nodes.append(x)
                x = parent[x]
            while depth[y] > depth[x]:
                nodes.append(y)
                y = parent[y]
            while x != y:
                nodes += [x, y]
                x, y = parent[x], parent[y]
            nodes.append(x)
            for n in nodes:
                count[n] += 1
    return count


@chapter
def geometry(ch: F.Chapter) -> None:
    specs = baseline()
    alive = survivors(specs)
    frame = F.store.read_frame("B1-10000-s001", 2 * LAST + 1)
    adj = graph(frame)
    v = balls(adj)
    r = np.arange(len(v))
    n = len(adj)
    keep = (r >= 1) & (v <= n / 2)
    slope, icpt = np.polyfit(np.log(r[keep]), np.log(v[keep]), 1)
    ch.grid(
        "ball", [
            dict(title="Logarithmic in both", x={"label": "steps r (logarithmic)", "log": True, "min": 1, "max": 40},
                 y={"label": "agents within r steps (logarithmic)", "log": True},
                 series=[{"label": None, "x": r[1:].tolist(), "y": v[1:].tolist(), "points": True, "size": 7,
                          "colour": BLUE},
                         line(r[keep], np.exp(icpt) * r[keep] ** slope, f"slope {slope:.2f}", RED, width=1.5,
                              dash=[5, 4])],
                 guides=[{"axis": "y", "at": n / 2, "label": "half the world"}]),
            dict(title="Logarithmic in the number only", x={"label": "steps r", "min": 0, "max": 40},
                 y={"label": "agents within r steps (logarithmic)", "log": True},
                 series=[{"label": None, "x": r[1:].tolist(), "y": v[1:].tolist(), "points": True, "size": 7,
                          "colour": BLUE}],
                 guides=[{"axis": "y", "at": n / 2, "label": "half the world"}], legend=False)],
        columns=2, title="How many agents are within r steps",
        caption=f"The world with seed 1 after its last game ({n:,} agents). From 24 agents spread evenly through "
                "the list of ids, a breadth-first search counts how many agents lie within r steps; the dots are "
                "the means over the 24. Left, both axes logarithmic: a power law r^d — growth like a "
                "d-dimensional space — is a straight line of slope d; red is the line least squares fits up to "
                "the radius where half the world is reached. Right, only the vertical axis logarithmic: "
                "exponential growth, as in a tree or a random network, would be a straight line there "
                "([Dimension and curvature](../notes/ball-dimension-and-curvature.md)).",
        recipe=recipe("`B1-10000-s001`.", FRAMES,
                      ["Read frame `5999`; build the network from `edges`.",
                       "From every ⌊n/24⌋-th agent of the id-sorted list, breadth-first search; count the agents "
                       "at distance ≤ r for r = 1 … 40; average over the 24 sources.",
                       "Fit ln V(r) = a + d·ln r by least squares for r with V(r) ≤ n/2."]))
    ch.number("ball_seed1", {"slope": float(slope), "agents": n, "half_radius": int(r[v >= n / 2][0])})

    panels = []
    for title, stat, colour, log in (("Dimension from ball growth", "dimension", BLUE, False),
                                     ("Curvature (Ricci scalar)", "ricciCurvature", ORANGE, False),
                                     ("Box dimension", "boxDimension", GREEN, False),
                                     ("Spectral gap λ₂ (logarithmic)", "spectralGap", VIOLET, True)):
        y = {"label": title.split(" (")[0].lower(), "log": log}
        if not log:
            y["min"] = 0 if stat != "ricciCurvature" else None
        panels.append(dict(title=title, x={"label": "iteration"}, y=y,
                           series=[band_series([series(s.run_id, stat) for s in specs], None, colour)],
                           legend=False, guides=[{"axis": "y", "at": 0}] if stat == "ricciCurvature" else []))
        ch.number(f"{stat}_settled", describe([mean_over(s.run_id, stat, 500, 2999) for s in alive]))
        ch.number(f"{stat}_first", F.A.bands([series(s.run_id, stat) for s in specs])["y"][0])
    ch.grid(
        "measures", panels, columns=2, title="Four measures of a world's geometry",
        caption="Every 25 iterations, in the 30 baseline worlds: the dimension and the curvature read off how "
                "balls grow ([Dimension and curvature](../notes/ball-dimension-and-curvature.md)), the box "
                "dimension ([Box dimension](../notes/box-dimension.md)), and the spectral gap, on a logarithmic "
                f"axis ([The spectral gap](../notes/spectral-gap.md)). In each, {BAND_WORDS[0].lower()}"
                f"{BAND_WORDS[1:]}.",
        recipe=recipe(BASELINE_RUNS, STATS_FILE,
                      ["Take `dimension`, `ricciCurvature`, `boxDimension` and `spectralGap` of the rows with "
                       "`phase` = 2 (measured every 25 iterations).", *BAND_STEPS[1:]]))
    for stat in ("boxDimensionR2", "radius", "diameter", "meanPathLength", "cycleRank", "loopDensity"):
        ch.number(f"{stat}_settled", describe([mean_over(s.run_id, stat, 500, 2999) for s in alive]))

    counts = box_counts(adj)
    sz = np.array([c[0] for c in counts], float)
    nb = np.array([c[1] for c in counts], float)
    bs, bi = np.polyfit(np.log(sz), np.log(nb), 1)
    ch.figure(
        "boxes", title="Covering one world with boxes",
        x={"label": "box size ℓ (logarithmic)", "log": True}, y={"label": "boxes needed (logarithmic)", "log": True},
        series=[{"label": "boxes needed", "x": sz.tolist(), "y": nb.tolist(), "points": True, "size": 9,
                 "colour": GREEN},
                line(sz, np.exp(bi) * sz ** bs, f"least squares: slope {bs:.2f}, box dimension {-bs:.2f}", RED,
                     width=1.5, dash=[5, 4])],
        caption="The world with seed 1 after its last game, covered by boxes: a box of size ℓ is every agent "
                "within (ℓ − 1)/2 steps of a centre, and centres are taken greedily, best-connected first, among "
                "the agents no box holds yet. Dots: how many boxes it takes for ℓ = 1, 3, 5, 9, 17 and 33; red: "
                "the least-squares line on logarithmic axes, whose slope is minus the box dimension "
                "([Box dimension](../notes/box-dimension.md)).",
        recipe=recipe("`B1-10000-s001`.", FRAMES,
                      ["Read frame `5999`; order the agents by degree, highest first.",
                       "For each ℓ: go down the order; every agent not yet in a box starts a new box, which takes "
                       "every agent within (ℓ − 1)/2 steps of it; count the boxes.",
                       "Fit ln(boxes) on ln ℓ by least squares."]))
    ch.number("boxes_seed1", {"counts": counts, "dimension": float(-bs)})

    lines_ = [band_series([series(s.run_id, st) for s in specs], lab, c)
              for st, lab, c in (("diameter", "diameter: the longest shortest path", RED),
                                 ("radius", "radius: the shortest reach that covers everyone", YELLOW),
                                 ("meanPathLength", "mean path: the average distance", BLUE))]
    ch.figure(
        "distances", title="Three distances", x={"label": "iteration"}, y={"label": "steps", "min": 0},
        series=lines_,
        caption="Every 25 iterations, in the 30 baseline worlds: the diameter, the radius and the mean path "
                "length, all estimated from breadth-first searches out of 8 to 16 spread agents "
                f"([Radius and diameter](../notes/radius-and-diameter.md)). For each, {BAND_WORDS[0].lower()}"
                f"{BAND_WORDS[1:]}.",
        recipe=recipe(BASELINE_RUNS, STATS_FILE,
                      ["Take `diameter`, `radius` and `meanPathLength` of the rows with `phase` = 2.",
                       *BAND_STEPS[1:]]))

    loops = fundamental_cycles(adj)
    through = np.array([loops.get(a, 0) for a in adj], float)
    lx = np.unique(through[through > 0])
    ly = [(through >= x).mean() for x in lx]
    ch.figure(
        "loops", title="How many loops pass through an agent",
        x={"label": "loops of the cycle basis through the agent (logarithmic)", "log": True},
        y={"label": "share of agents with at least that many (logarithmic)", "log": True, "max": 1},
        series=[line(lx, ly, "the world of seed 1 after its last game", GREEN, width=2)],
        caption="A breadth-first spanning tree of the world with seed 1 after its last game leaves "
                f"{int(len(frame['edges']) - n + 1):,} connections over, each closing one loop of a cycle basis "
                "([Loops](../notes/loops.md)). For every count x, the share of agents that at least x of those "
                f"loops pass through. {100 * np.mean(through == 0):.0f}% of the agents lie on none of them.",
        recipe=recipe("`B1-10000-s001`.", FRAMES,
                      ["Read frame `5999`; build a breadth-first spanning tree.",
                       "For every connection not in the tree, walk both ends up the tree to where they meet; "
                       "every agent on the way, and the meeting point, lies on that loop.",
                       "Count the loops per agent; draw the share with at least x."]))
    ch.number("loops_seed1", {"basis": int(len(frame["edges"]) - n + 1), "on_none": float(np.mean(through == 0)),
                              "max": float(through.max())})
    # Against a null: the same connections per agent, wired at random.
    nulls = [exact_geometry(degree_preserving(adj, seed)) for seed in range(1, 6)]
    ch.number("null_seed1", {"world": exact_geometry(adj),
                             "random": {k: float(np.mean([g[k] for g in nulls])) for k in nulls[0]}})


# ---------------------------------------------------------------------------
# Chapter 25 · How a world breaks
# ---------------------------------------------------------------------------

def cut_anatomy(run_id: str, t: int) -> Dict[str, float]:
    """
    How the game of iteration t cut agents off: which connections between the lost
    and the kept were pruned, and what the lost agents had staked.

    A cut-off agent that had a neighbour among the survivors lost that connection
    to pruning (nothing crossed it), since otherwise it would still be joined to
    them; so the "attachments" are exactly the pruned connections across the cut.
    """
    a, b = F.store.read_frame(run_id, 2 * t), F.store.read_frame(run_id, 2 * t + 1)
    after = set(b["ids"])
    staked, home = Counter(), {}
    for r in b["decisions"]["allocations"]:
        total = sum(r["alloc"])
        home[r["agent"]] = sum(x for g, x in zip(r["targets"], r["alloc"]) if g == r["agent"]) / total \
            if total else np.nan
        for g, x in zip(r["targets"], r["alloc"]):
            if x:
                staked[g] += x
    died = set(a["ids"]) - after
    lost = {u for u in died if staked[u]}           # cut off, not starved
    adj = graph(a)
    border = [(c, v) for c in lost for v in adj[c] if v in after]
    genotype = dict(zip(a["ids"], a["brain_ids"]))
    tokens = dict(zip(a["ids"], a["tokens"]))
    top = Counter(genotype[c] for c in lost).most_common(1)[0][1] / len(lost) if lost else np.nan

    def side(agents):
        """Tokens and connections of the agents on one side of the border."""
        return [(tokens[u], len(adj[u])) for u in agents]
    return {"iteration": t, "cut": len(lost), "attachments": len(border), "top_genotype": float(top),
            "home": float(np.nanmedian([home.get(c, np.nan) for c in lost])) if lost else np.nan,
            "lost_side": side({c for c, _ in border}), "kept_side": side({v for _, v in border}),
            "everyone": side(a["ids"])}


@chapter
def breaking(ch: F.Chapter) -> None:
    alive = survivors(baseline())
    risk, culled = [], []
    for s in alive:
        for row in settled_rows(s.run_id):
            if row.get("cutRiskBefore") is not None and row["nodes_before"]:
                risk.append(row["cutRiskBefore"])
                culled.append((row["orphaned"] or 0) / row["nodes_before"])
    risk, culled = np.array(risk), np.array(culled)
    edges_ = np.linspace(0, 0.5, 11)
    bx, by_ = [], []
    for a, b in zip(edges_[:-1], edges_[1:]):
        m = (risk >= a) & (risk < b)
        if m.sum() >= 20:
            bx.append((a + b) / 2)
            by_.append(float(culled[m].mean()))
    rng = np.random.default_rng(3)
    pick = rng.choice(len(risk), size=min(6000, len(risk)), replace=False)
    ch.figure(
        "risk", title="Does a fragile world lose more?",
        x={"label": "worst single cut before the game: share of agents behind one bridge", "min": 0, "max": 0.5},
        y={"label": "share of agents cut off in the game", "min": 0, "max": 0.5},
        series=[{"label": "one game (6,000 drawn at random)", "x": risk[pick].tolist(), "y": culled[pick].tolist(),
                 "points": True, "size": 3, "colour": ORANGE, "alpha": 0.4},
                line(bx, by_, "mean in classes of 0.05", "#eef4fa", width=2),
                line([0, 0.5], [0, 0.5], "the whole worst cut lost", BLUE, width=1, dash=[4, 4])],
        caption="Every game from iteration 500 on of the 26 worlds that lived to the end. x: before the game, "
                "the largest share of the world that one bridge held to the rest — if that one connection were "
                "cut, that many would be cut off ([Cut risk](../notes/cut-risk.md)). y: the share of agents the "
                "game really cut off. White: the mean of y in classes of x 0.05 wide; dashed: y = x.",
        recipe=recipe(SURVIVORS, STATS_FILE,
                      ["Take the rows with `phase` = 2 and 500 ≤ `iteration` ≤ 2,999: `cutRiskBefore`, and "
                       "`orphaned` / `nodes_before`.", "Plot one against the other; average y in classes of x."]))
    ch.number("risk", {"corr": float(np.corrcoef(risk, culled)[0, 1]), "games": int(len(risk)),
                       "mean_risk": float(risk.mean()), "mean_culled": float(culled.mean()),
                       "share_culled_over_10pc": float(np.mean(culled > 0.1)),
                       "share_risk_over_10pc": float(np.mean(risk > 0.1))})

    cut = np.concatenate([[r["orphaned"] or 0 for r in settled_rows(s.run_id)] for s in alive]).astype(float)
    order = np.sort(cut)[::-1]
    xs = np.unique(cut[cut > 0])
    weighted = [cut[cut >= x].sum() / cut.sum() for x in xs]
    ch.figure(
        "deaths-by-size", title="Where the cut-off deaths happen",
        x={"label": "agents cut off in one game, n (logarithmic)", "log": True, "min": 1},
        y={"label": "share of all cut-off deaths", "min": 0, "max": 1},
        series=[line(xs, weighted, "in games cutting off at least n", ORANGE, width=2)],
        caption="Of all agents cut off in games from iteration 500 on of the 26 worlds that lived to the end, the "
                "share that died in games which cut off at least n agents, for every n. Where the line is at "
                "0.5, half of all such deaths happened in games at least that large.",
        recipe=recipe(SURVIVORS, STATS_FILE,
                      ["Take `orphaned` of the rows with `phase` = 2 and 500 ≤ `iteration` ≤ 2,999.",
                       "For every n, add up `orphaned` over the rows with `orphaned` ≥ n and divide by the total."]))
    half = float(xs[np.searchsorted(-np.array(weighted), -0.5)]) if len(xs) else None
    ch.number("deaths_by_size", {"half_in_games_of_at_least": half, "total": float(cut.sum()),
                                 "games": int(len(cut)), "largest": float(order[0])})

    # The anatomy of the big cuts: one unused bridge, or a whole border gone quiet?
    big = [dict(run=s.run_id, **cut_anatomy(s.run_id, r["iteration"])) for s in alive
           for r in settled_rows(s.run_id) if (r["orphaned"] or 0) >= 100]
    kinds = {}
    for label, lo, hi in (("1–2", 1, 2), ("3–9", 3, 9), ("10+", 10, 10 ** 9)):
        g = [x for x in big if lo <= x["attachments"] <= hi]
        kinds[label] = {"games": len(g), "agents": int(sum(x["cut"] for x in g)),
                        "top_genotype": float(np.median([x["top_genotype"] for x in g])) if g else None,
                        "home": float(np.nanmedian([x["home"] for x in g])) if g else None}
    def poverty(key):
        """On the quiet borders: the agents' tokens and connections, and how many hold fewer tokens than links."""
        pairs = np.array([p for x in big if x["attachments"] >= 10 for p in x[key]], float)
        return {"agents": len(pairs), "median_tokens": float(np.median(pairs[:, 0])),
                "median_degree": float(np.median(pairs[:, 1])), "fewer_tokens_than_links": float(np.mean(pairs[:, 0] < pairs[:, 1]))}
    seed1 = [{k: v for k, v in x.items() if not k.endswith("side") and k != "everyone"}
             for x in sorted((x for x in big if x["run"] == "B1-10000-s001"), key=lambda x: -x["cut"])[:6]]
    ch.number("anatomy", {"games": len(big), "kinds": kinds, "seed1_largest": seed1,
                          "border": {side: poverty(side) for side in ("lost_side", "kept_side", "everyone")}})

    its, n = series("B1-10000-s001", "nodes")
    oits, o = series("B1-10000-s001", "orphaned")
    rits, rk = series("B1-10000-s001", "cutRiskBefore")
    window = (1000, 1600)
    m = (its >= window[0]) & (its <= window[1])
    mo = (oits >= window[0]) & (oits <= window[1])
    mr = (rits >= window[0]) & (rits <= window[1]) & np.isfinite(rk)
    ch.grid(
        "one-world", [
            dict(title="Agents after every game", x={"label": "iteration", "min": window[0], "max": window[1]},
                 y={"label": "agents", "min": 0}, series=[line(its[m], n[m], None, BLUE, width=1.2)], legend=False),
            dict(title="Agents cut off in every game", x={"label": "iteration", "min": window[0], "max": window[1]},
                 y={"label": "cut off", "min": 0},
                 series=[line(oits[mo], o[mo], None, ORANGE, width=1)], legend=False),
            dict(title="Worst single cut before the game", x={"label": "iteration", "min": window[0], "max": window[1]},
                 y={"label": "share of agents", "min": 0, "max": 0.5},
                 series=[line(rits[mr], rk[mr], None, RED, width=1.2)], legend=False)],
        columns=1, title="Six hundred iterations of one world",
        caption="The world with seed 1, iterations 1,000 to 1,600. Top: its number of agents after every game. "
                "Middle: how many agents each game cut off. Bottom: before each game, the largest share of the "
                "world held to the rest by a single bridge, measured on the network as it stood before the cleanup.",
        recipe=recipe("`B1-10000-s001`.", STATS_FILE,
                      ["Take `nodes`, `orphaned` and `cutRiskBefore` of the rows with `phase` = 2 and 1,000 ≤ "
                       "`iteration` ≤ 1,600."]))
    for stat in ("bridges", "prunedEdges", "cutRisk", "cutRiskBefore", "orphaned", "starved"):
        ch.number(f"{stat}_settled", describe([mean_over(s.run_id, stat, 500, 2999) for s in alive]))


# ---------------------------------------------------------------------------
# Chapter 26 · One world, many colours
# ---------------------------------------------------------------------------

@chapter
def colours(ch: F.Chapter) -> None:
    import networkx as nx
    run = "B1-10000-s001"
    game = F.store.read_frame(run, 2 * LAST + 1)
    start = F.store.read_frame(run, 2 * LAST)
    ids, edges = game["ids"], game["edges"]
    G = nx.Graph()
    G.add_nodes_from(ids)
    G.add_edges_from(edges)
    pos = nx.forceatlas2_layout(G, max_iter=200, seed=1)
    index = {u: i for i, u in enumerate(ids)}
    xy = {"x": [float(pos[u][0]) for u in ids], "y": [float(pos[u][1]) for u in ids]}
    E = [[index[a], index[b]] for a, b in edges]
    adj = graph(game)

    tokens = np.array(game["tokens"], float)
    ages = np.array(game["ages"], float) + 1
    change = np.array(game["delta"], float)
    t0 = dict(zip(start["ids"], start["tokens"]))
    sadj = graph(start)
    curvature = np.array([sum(t0[b] - t0[a] for b in sadj.get(a, ())) if a in t0 else 0 for a in ids], float)
    loops = fundamental_cycles(adj)
    through = np.array([loops.get(a, 0) for a in ids], float) + 1

    # Families: the genotypes alive 100 iterations before, followed down to the living.
    parent = {}
    for index_ in range(2 * (LAST - 100), 2 * LAST + 2):
        f = F.store.read_frame(run, index_)
        for b, p in zip(f["brain_ids"], f["parent_brain_ids"]):
            parent.setdefault(b, p)
    anchor = set(F.store.read_frame(run, 2 * (LAST - 100) + 1)["brain_ids"])

    def family(g):
        seen = 0
        while g not in anchor and g in parent and seen < 10000:
            g = parent[g]
            seen += 1
        return g if g in anchor else None
    fam = [family(g) for g in game["brain_ids"]]
    top = [f for f, _ in Counter(f for f in fam if f is not None).most_common(7)]
    group = [top.index(f) if f in top else 7 for f in fam]
    shares = Counter(group)

    def panel(title, colour_spec, values=None, groups=None):
        nodes = dict(xy)
        if values is not None:
            nodes["value"] = [float(v) for v in values]
        if groups is not None:
            nodes["group"] = groups
        return dict(kind="network", title=title, nodes=nodes, edges=E, colour=colour_spec, nodeSize=2.0,
                    edgeAlpha=0.25)
    sym = max(abs(change).max(), 1)
    csym = float(np.quantile(abs(curvature), 0.99))
    panels = [
        panel("Tokens", {"by": "value", "map": "viridis", "min": 1, "max": float(tokens.max()), "log": True,
                         "label": "tokens"}, tokens),
        panel("Age", {"by": "value", "map": "viridis", "min": 1, "max": float(ages.max()), "log": True,
                      "label": "iterations lived + 1"}, ages),
        panel("Change in the last game", {"by": "value", "map": "signed", "min": -sym, "max": sym, "symlog": True,
                                          "label": "tokens"}, change),
        panel("Token curvature before it", {"by": "value", "map": "signed", "min": -csym, "max": csym,
                                            "symlog": True, "label": "tokens"}, np.clip(curvature, -csym, csym)),
        panel("Families of 100 iterations before", {"labels": [f"family {i + 1}" for i in range(7)] + ["all others"]},
              groups=group),
        panel("Loops through each agent", {"by": "value", "map": "viridis", "min": 1, "max": float(through.max()),
                                           "log": True, "label": "loops + 1"}, through)]
    panels[4]["legend"] = True
    ch.grid(
        "six-views", panels, columns=2, title="The world of seed 1 after its last game, six ways",
        caption="Every agent alive after the game of iteration 2,999 of the world with seed 1 is a dot, every "
                "connection a faint line, and the dots sit in the same places in all six pictures (a "
                "ForceAtlas2 layout, seed 1, which pulls joined agents together). Each picture colours the "
                "dots by one of the quantities the viewer offers ([What the viewer can colour by]"
                "(../notes/viewer-colours.md)): tokens and age on logarithmic scales; the change of tokens in "
                "the last game and the token curvature at its start on a signed scale, blue below zero, red "
                "above, dark at zero (both stretched logarithmically, the curvature cut at its 1st and 99th "
                "percentile); the family — which of the genotypes alive 100 iterations earlier each agent's "
                "genotype descends from, the seven largest in colour; and how many loops of a breadth-first "
                "cycle basis pass through each agent.",
        recipe=recipe(f"`{run}`.", FRAMES,
                      ["Read frames `5998` (start of the last game) and `5999` (after it): `ids`, `tokens`, "
                       "`ages`, `delta`, `brain_ids`, `edges`.",
                       "Curvature: in frame 5998, Σ over neighbours of (their tokens − own tokens).",
                       "Families: read frames 5799 to 5999, note each genotype's parent, and climb from each "
                       "living genotype to one alive after the game of iteration 2,899.",
                       "Loops: as in [Chapter 24](24-the-geometry-of-a-world.md).",
                       "Lay out with `networkx.forceatlas2_layout(G, max_iter=200, seed=1)` and colour."]))
    ch.number("six_views", {"agents": len(ids), "families": {str(k): v for k, v in shares.items()},
                            "max_tokens": float(tokens.max()), "max_age": float(ages.max() - 1)})

    flow = Counter()
    for r in (game.get("decisions") or {}).get("allocations") or []:
        for target, amount in zip(r["targets"][1:], r["alloc"][1:]):
            if amount > 0:
                flow[(min(r["agent"], target), max(r["agent"], target))] += amount
    classes = [(1, 2), (3, 9), (10, 49), (50, 10 ** 9)]
    edge_group = []
    for a, b in edges:
        f = flow.get((min(a, b), max(a, b)), 0)
        edge_group.append(next((k + 1 for k, (lo, hi) in enumerate(classes) if lo <= f <= hi), 0))
    ch.network(
        "flows", title="Where the tokens went in the last game",
        nodes={**xy, "group": [0] * len(ids)}, edges=E, colour={"labels": []}, nodeSize=1.4, edgeAlpha=0.15,
        edgeGroup=edge_group, edgeLabels=["1–2 tokens", "3–9 tokens", "10–49 tokens", "50 or more"],
        caption="The same world and layout; every connection is coloured by how many tokens crossed it in the "
                "last game, in both directions together, and drawn the wider the more crossed. Every connection "
                "that is left carried at least one token — the others were cut at the end of the game. Agents "
                "are small dots.",
        recipe=recipe(f"`{run}`.", FRAMES,
                      ["In frame `5999`, read `decisions.allocations`: for every agent and target ≠ itself, add "
                       "the `alloc` to the connection between them.", "Colour each connection of `edges` by its class."]))
    ch.number("flows", {"classes": {str(k): v for k, v in Counter(edge_group).items()}})
