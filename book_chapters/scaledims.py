# -*- coding: utf-8 -*-
"""
Dimensions at scale: does a world's dimension depend on the scale at which it
is read, and do the largest worlds hint at what larger ones would show?

The two rulers of Is a world a space? (book_chapters.space) — how balls grow,
and how often a random walker is back where it started — and a third, how a
world's width grows with its agents, read on B1's worlds of every size:
Experiment 8's (1,600 to 409,600 tokens), Experiment 2's (10,000) and
Experiment 10's (100,000), each beside its random twin. Balls and walks are
read after a world's last game (the walks of the largest worlds for eight
times as long), and balls again at five moments of its settled life, since a
world's large-scale shape changes from one to the next. Beside the worlds,
bodies of known dimension with an edge — a disc cut out of a square grid and
a ball cut out of a cubic grid — at as many points as the worlds have agents
and read by the same code, show what a finite body's edge does to the rulers;
the endless grids say what the rulers read with no edge at all. Last, round
3's worlds of the screen are set beside B1's width at their size.

    scaledims   Dimensions at scale
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Tuple

import numpy as np

import book_data as D
import book_figures as F
from book_figures import FRAMES, chapter, describe, line, recipe
from book_graph import bfs, degree_preserving
from book_chapters.common import BLUE, GREEN, GREY, ORANGE, RED, STAR, VIOLET, YELLOW

#: The seed of every random choice here: the agents the balls and the walks start from, and the twin.
SEED = 23
#: Agents each world's balls are grown from (all the way out), and agents its walks start from.
BALLS, WALKS = 100, 48
#: Steps of the walks; a walk ends sooner once it is back no more often than at rest (see walk).
STEPS = 1024


def reach(adj: Dict[int, set], sources: int, seed: int) -> Dict[str, Any]:
    """
    Breadth-first searches all the way out from `sources` agents drawn at random:
    for each, how many agents lie within r steps for every r up to the farthest,
    how far it is from the others on average, and how far the farthest is. The
    agents are drawn as book_chapters.space.ball_growth draws them, so the first
    40 steps of the typical ball are its. Two more searches, from the farthest
    agent found and then from the farthest from it, bound the diameter from below.
    """
    rng = np.random.default_rng(seed)
    nodes = list(adj)
    n = len(nodes)
    counts, mean = [], []
    far, far_d = nodes[0], -1
    for i in rng.choice(n, min(sources, n), replace=False):
        dist = bfs(adj, nodes[i])
        d = np.fromiter(dist.values(), int, len(dist))
        counts.append(np.bincount(d))
        mean.append(float(d.sum() / max(n - 1, 1)))
        if d.max() > far_d:
            far_d = int(d.max())
            far = max(dist, key=dist.get)
    sweep = []
    for _ in range(2):
        dist = bfs(adj, far)
        sweep.append(max(dist.values()))
        far = max(dist, key=dist.get)
    width = max(len(c) for c in counts)
    cum = np.array([np.cumsum(np.pad(c, (0, width - len(c)))) for c in counts], dtype=np.int64)
    return {"cum": cum.tolist(), "mean_distance": mean, "eccentricity": [len(c) - 1 for c in counts],
            "diameter": int(max(far_d, *sweep))}


def walk(adj: Dict[int, set], sources: int, seed: int, steps: int, floor: float) -> List[float]:
    """
    What book_chapters.space.returns reads, from the same starts: how much more
    often than at rest a lazy random walk (stay with probability ½, else step to
    a neighbour drawn at random) is back at its start after t steps, averaged
    over `sources` starts. Two shortcuts, neither of which changes a reading:

    - Half the steps. A lazy walk is reversible, at rest in proportion to the
      agents' links (π), so the chance to be back after 2m or 2m + 1 steps
      follows from where the walk is after m and m + 1 steps:
      P₂ₘ(s, s) = π(s) Σ_y Pₘ(s, y)² / π(y) and
      P₂ₘ₊₁(s, s) = π(s) Σ_y Pₘ(s, y) Pₘ₊₁(s, y) / π(y).
    - It ends once the chance falls below half the floor below which the
      spectral dimension is no longer read: for a lazy walk it never rises again.
    """
    nodes = list(adj)
    n = len(nodes)
    at = {u: i for i, u in enumerate(nodes)}
    src = np.fromiter((at[u] for u in nodes for v in adj[u]), int)
    dst = np.fromiter((at[v] for u in nodes for v in adj[u]), int)
    degree = np.bincount(src, minlength=n).astype(float)
    rest_all = degree / degree.sum()
    rng = np.random.default_rng(seed)
    picked = rng.choice(n, min(sources, n), replace=False)
    columns = np.arange(len(picked))
    rest = rest_all[picked]
    order = np.argsort(dst, kind="stable")
    src_o, dst_o = src[order], dst[order]
    starts = np.searchsorted(dst_o, np.arange(n))
    has = np.diff(np.append(starts, len(dst_o))) > 0
    into = np.arange(n)[has]

    def step(p: np.ndarray) -> np.ndarray:
        moved = np.zeros_like(p)
        moved[into] = np.add.reduceat((p / degree[:, None])[src_o], starts[has], axis=0)
        return 0.5 * p + 0.5 * moved

    def back(weighted: np.ndarray, later: np.ndarray) -> float:
        return float(np.mean(rest * np.einsum("ij,ij->j", weighted, later) - rest))

    p = np.zeros((n, len(picked)))
    p[picked, columns] = 1.0
    out: List[float] = []
    for m in range(steps // 2 + 1):
        q = step(p)
        weighted = p / rest_all[:, None]
        if m > 0:
            out.append(back(weighted, p))       # back after 2m steps
        out.append(back(weighted, q))           # and after 2m + 1
        p = q
        if out[-1] < floor / 2 or len(out) >= steps:
            break
    return out[:steps]


def read(adj: Dict[int, set], seed: int = SEED, balls: int = BALLS, walks: int = WALKS,
         steps: int = STEPS) -> Dict[str, Any]:
    """Everything this page reads of one connected network: its size, its links, its balls, its walks."""
    n = len(adj)
    k = np.array([len(v) for v in adj.values()])
    return {"n": n, "links": float(k.mean()), "hub": int(k.max()), **reach(adj, balls, seed),
            "returns": walk(adj, walks, seed, steps, 5 / n)}


@D.measure("scaledims")
def reading(run_id: str) -> Dict[str, Any]:
    """
    The largest piece of a world after its last game, and its random twin's
    largest piece, read all the way out by balls and up to STEPS steps by walks.
    """
    from book_chapters.space import largest_piece, last_world
    adj = largest_piece(last_world(run_id))
    return {"world": read(adj), "twin": read(largest_piece(degree_preserving(adj, SEED)))}


#: Steps of the long walks on the largest worlds: eight times as many, so that a walker gets about as far as the
#: balls' plateau reaches; and the sizes of Experiment 8 walked so far: its largest.
LONG_STEPS = 8192
LONG_SIZES = (409600,)


@D.measure("scaledims-long")
def long_reading(run_id: str) -> Dict[str, Any]:
    """A world's largest piece after its last game, walked for up to LONG_STEPS steps from reading's 48 starts."""
    from book_chapters.space import largest_piece, last_world
    adj = largest_piece(last_world(run_id))
    return {"n": len(adj), "returns": walk(adj, WALKS, SEED, LONG_STEPS, 5 / len(adj))}


#: The radii out to which hub_reach counts the agents near a world's best-linked agent.
HUB_RADII = (2, 4, 6, 8, 10, 12, 15, 20)


@D.measure("scaledims-hub")
def hub_reach(run_id: str) -> Dict[str, Any]:
    """
    How far the agents of a world's largest piece are from its best-linked agent
    after its last game: its links, the median distance to it, and the share of
    agents within each of HUB_RADII steps of it.
    """
    from book_chapters.space import largest_piece, last_world
    adj = largest_piece(last_world(run_id))
    hub = max(adj, key=lambda u: len(adj[u]))
    d = np.fromiter(bfs(adj, hub).values(), int)
    return {"links": len(adj[hub]), "median": float(np.median(d)),
            "within": {str(r): float(np.mean(d <= r)) for r in HUB_RADII}}


#: How many moments of a world's settled life its balls are read at, evenly spaced from iteration 500 to its
#: last: between them, the median world's typical distance changes by a factor of about 1.8.
MOMENTS = 5


def moments(run_id: str) -> List[int]:
    """MOMENTS iterations evenly spaced over a world's settled life, its last among them."""
    last = D.last_iteration(run_id)
    first = min(F.SETTLED, last)
    return sorted({int(round(first + k * (last - first) / (MOMENTS - 1))) for k in range(MOMENTS)})


def _settled_key(run_id: str) -> Dict[str, Any]:
    return {"at": moments(run_id), "balls": BALLS, "seed": SEED}


@D.measure("scaledims-settled", field="key", want=_settled_key)
def settled(run_id: str) -> Dict[str, Any]:
    """
    A world's largest piece after the games of MOMENTS iterations of its settled
    life, and its random twin's, read all the way out by balls only.
    """
    from book_chapters.space import largest_piece
    from book_graph import graph
    out: Dict[str, Any] = {"at": [], "world": [], "twin": []}
    for t in moments(run_id):
        adj = largest_piece(graph(D.frame_at(run_id, t, 2)))
        out["at"].append(t)
        for kind, a in (("world", adj), ("twin", largest_piece(degree_preserving(adj, SEED)))):
            k = np.array([len(v) for v in a.values()])
            out[kind].append({"n": len(a), "links": float(k.mean()), "hub": int(k.max()), **reach(a, BALLS, SEED)})
    return out


# ---------------------------------------------------------------------------
# Bodies of known dimension, with an edge
# ---------------------------------------------------------------------------

#: The numbers of points of the bodies: 250 to 64,000, each twice the last, as the worlds of 1,600 to
#: 409,600 tokens hold about 0.15 agents a token.
BODY_SIZES = tuple(250 * 2 ** k for k in range(9))


def body(dim: int, n: int) -> Dict[int, set]:
    """
    A disc (dim 2) or a ball (dim 3) cut out of a square or cubic grid: the n
    points of the grid nearest its centre — and any as near as the last — each
    linked to the 2·dim points beside it that are inside too.
    """
    m = int(math.ceil((n / (math.pi if dim == 2 else 4 * math.pi / 3)) ** (1 / dim))) + 2
    coords = np.indices((2 * m + 1,) * dim).reshape(dim, -1).T - m
    norm = (coords ** 2).sum(1)
    inside = norm <= np.sort(norm)[n - 1]
    index = np.full((2 * m + 1,) * dim, -1)
    index[tuple((coords[inside] + m).T)] = np.arange(inside.sum())
    adj: Dict[int, set] = {i: set() for i in range(int(inside.sum()))}
    for axis in range(dim):
        step = np.zeros(dim, int)
        step[axis] = 1
        a = coords[inside]
        b = a + step
        ok = np.all(np.abs(b) <= m, axis=1)
        i, j = index[tuple((a[ok] + m).T)], index[tuple((b[ok] + m).T)]
        for u, v in zip(i[j >= 0].tolist(), j[j >= 0].tolist()):
            adj[u].add(v)
            adj[v].add(u)
    return adj


def grid_balls(dim: int, radius: int) -> np.ndarray:
    """The points of an endless grid within r steps of one, r = 0 … radius: Σₖ 2ᵏ·C(dim, k)·C(r, k)."""
    return np.array([sum(2 ** k * math.comb(dim, k) * math.comb(r, k) for k in range(dim + 1))
                     for r in range(radius + 1)], float)


def _body_key(name: str) -> Dict[str, Any]:
    return {"balls": BALLS, "walks": WALKS, "steps": STEPS, "seed": SEED, "version": 1}


def _body_reading(name: str) -> Dict[str, Any]:
    dim, n = (int(v) for v in name.split("d-"))
    return read(body(dim, n))


#: A body's reading, kept in GraphOfLifeRuns/.book/ as a run's measures are, under a name such as "3d-64000"
#: (three dimensions, 64,000 points). Not one of the run measures (D.MEASURES): it reads no run.
BODY = D.Measure("scalebody", _body_reading, "bodies/{run}.scalebody.json", field="key", want=_body_key)


def body_reading(dim: int, n: int) -> Dict[str, Any]:
    return D.get(BODY, f"{dim}d-{n}")


# ---------------------------------------------------------------------------
# What is read off a network's balls and walks
# ---------------------------------------------------------------------------

def typical(m: Dict[str, Any]) -> np.ndarray:
    """The typical ball, N(r) for r = 0 … the farthest: the geometric mean over the agents searched from."""
    return np.exp(np.log(np.array(m["cum"], float)).mean(axis=0))


def ball_curve(m: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    """The local dimension d(r) at r = √(r(r+1)), while the typical ball holds less than a quarter."""
    from book_chapters.space import local_dimension
    return local_dimension(typical(m), m["n"])


def walk_curve(m: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    """The spectral dimension d(t) at t = √(t·2t), while the walk is back more often than 5/n above rest."""
    from book_chapters.space import spectral_dimension
    ps = np.array(m["returns"], float)
    return spectral_dimension(np.arange(1, len(ps) + 1, dtype=float), ps, 5 / m["n"])


def half_reach(m: Dict[str, Any]) -> float:
    """
    The typical distance: the radius at which the typical ball first holds half
    the network, between whole steps by the logarithm of the ball (as
    research/atlas/space.py reads it).
    """
    b, n = typical(m), m["n"]
    k = int(np.argmax(b >= n / 2))
    if b[k] < n / 2 or k == 0:
        return float("nan")
    return (k - 1) + (np.log(n / 2) - np.log(b[k - 1])) / (np.log(b[k]) - np.log(b[k - 1]))


def at_radius(m: Dict[str, Any], r: int) -> float:
    """The local dimension from radius r to r + 1, or nothing where the ball has passed a quarter."""
    xs, ds = ball_curve(m)
    x = math.sqrt(r * (r + 1))
    return next((d for xx, d in zip(xs, ds) if abs(xx - x) < 1e-6), float("nan"))


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

#: Where each ruler is read for one number per world, as Is a world a space? compared its three sizes:
#: the balls from radius 3 to 6, the walks from 4 to 64 steps.
BALL_RANGE, WALK_RANGE = (3, 6), (4, 64)
#: The radii at which the balls are read at every size, each from r to r + 1.
RADII = (2, 4, 8)


def ball_summary(m: Dict[str, Any]) -> Dict[str, Any]:
    """What a network's balls say, as the page quotes it."""
    from book_chapters.space import plateau
    bx, bd = ball_curve(m)
    return {"agents": m["n"], "links": m["links"], "hub": m["hub"], "half": half_reach(m),
            "mean_distance": float(np.mean(m["mean_distance"])), "diameter": m["diameter"],
            "balls": plateau(bx, bd, *BALL_RANGE), "peak": max(bd) if bd else float("nan"),
            "reach": max(bx) if bx else float("nan"), **{f"r{r}": at_radius(m, r) for r in RADII}}


def summary(m: Dict[str, Any]) -> Dict[str, Any]:
    """One network's readings, by balls and by walks, as the page quotes them."""
    from book_chapters.space import plateau
    sx, sd = walk_curve(m)
    return {**ball_summary(m), "walks": plateau(sx, sd, *WALK_RANGE),
            "walk_reach": max(sx) if sx else float("nan")}


#: What a world's balls are summed up by over the moments of its settled life.
SETTLED_KEYS = ("agents", "links", "hub", "half", "mean_distance", "diameter", "balls", "peak", "reach",
                *(f"r{r}" for r in RADII))


def settled_summary(run_id: str) -> Dict[str, Any]:
    """
    A world's balls over the moments of its settled life: for itself and its
    twin, the median over the moments of each reading (of those that read it,
    if at least half do), its width (typical distance over the twin's) at each
    moment and their median, its typical distance at each moment, and the
    balls themselves.
    """
    r = settled(run_id)
    each = {kind: [ball_summary(m) for m in r[kind]] for kind in ("world", "twin")}

    def med(values: List[float]) -> float:
        v = [x for x in values if np.isfinite(x)]
        return float(np.median(v)) if v and len(v) >= len(values) / 2 else float("nan")
    out: Dict[str, Any] = {kind: {k: med([e[k] for e in each[kind]]) for k in SETTLED_KEYS} for kind in each}
    widths = [w["half"] / t["half"] for w, t in zip(each["world"], each["twin"])]
    out["world"]["width"] = med(widths)
    out["moments"] = {"at": r["at"], "half": [e["half"] for e in each["world"]], "width": widths,
                      "agents": [e["agents"] for e in each["world"]], "balls": [e["balls"] for e in each["world"]],
                      "hub": [e["hub"] for e in each["world"]]}
    out["readings"] = {kind: r[kind] for kind in ("world", "twin")}
    return out


def worlds() -> List[Dict[str, Any]]:
    """
    Every world of the page that lived to its end, read after its last game: Experiment 8's baseline
    worlds, Experiment 2's and Experiment 10's; each marked a star or not, as Stars and hubs marks them,
    by the share of its agents its best-linked agent was joined to over its settled life.
    """
    out = []
    for experiment in ("E08", "E02", "E10"):
        for s in F.runs_of(experiment, "baseline"):
            if F.lived(s) < s.until:
                continue
            hub = F.mean_over(s.run_id, "maxDegree/nodes", F.SETTLED, D.last_iteration(s.run_id))
            r = reading(s.run_id)
            row = {"run": s.run_id, "experiment": experiment, "tokens": int(s.config["total_tokens"]),
                   "iteration": D.last_iteration(s.run_id), "hub_share": float(hub), "star": bool(hub >= STAR),
                   "world": summary(r["world"]), "twin": summary(r["twin"]), "reading": r,
                   "settled": settled_summary(s.run_id)}
            if row["tokens"] in LONG_SIZES and experiment == "E08":
                row["long"] = long_reading(s.run_id)
            out.append(row)
    return out


#: Round 3 of the screen of random worlds, as research/screen/health.py scored it.
HEALTH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research", "screen", "health.json")


def round_three() -> List[Dict[str, Any]]:
    """
    Round 3's worlds that lived to the end of their runs (Experiments 40 to 42): their agents, their width
    (typical distance over the random twin's) and the viewer's dimension, each averaged from iteration 100 to
    the end, and whether their dead are mended. Nothing is measured here; health.json holds it all.
    """
    if not os.path.exists(HEALTH):
        return []
    with open(HEALTH) as f:
        d = json.load(f)
    return [{"id": w["id"], "round": w["round"], "seed": w["seed"], "agents": w["averages"]["agents"],
             "width": w["averages"]["width"], "dimension": w["averages"].get("dimension"),
             "mended": bool(w["set"].get("mend_on_death"))}
            for w in d["worlds"] if w.get("alive") and (w.get("averages") or {}).get("width")]


def bodies() -> Dict[int, List[Dict[str, Any]]]:
    """Every body's reading, by its dimension, smallest first."""
    out: Dict[int, List[Dict[str, Any]]] = {}
    for dim in (2, 3):
        out[dim] = []
        for n in BODY_SIZES:
            m = body_reading(dim, n)
            out[dim].append({"points": n, **summary(m), "reading": m})
    return out


def endless(dim: int, radius: int = 60) -> Tuple[List[float], List[float]]:
    """The ball ruler on an endless grid: its local dimension at r = √(r(r+1))."""
    from book_chapters.space import local_dimension
    return local_dimension(grid_balls(dim, radius), float("inf"))


def median_curve(curves: List[Tuple[List[float], List[float]]]) -> Tuple[List[float], List[float]]:
    """The median of a few networks' readings at every scale at least half of them read."""
    grid = sorted({round(x, 6) for xs, _ in curves for x in xs})
    table = np.full((len(curves), len(grid)), np.nan)
    for i, (xs, ds) in enumerate(curves):
        for x, d in zip(xs, ds):
            table[i, grid.index(round(x, 6))] = d
    keep = np.sum(np.isfinite(table), axis=0) >= max(1, math.ceil(len(curves) / 2))
    with np.errstate(all="ignore"):
        return [x for x, k in zip(grid, keep) if k], [float(v) for v, k in zip(np.nanmedian(table, axis=0), keep) if k]


def ramp(f: float) -> str:
    """A colour from dark (f = 0) to bright (f = 1), for sizes from the smallest to the largest."""
    from book_svg import cmap
    return cmap("viridis", 0.3 + 0.7 * f)


def size_colour(tokens: int) -> str:
    """Experiment 8's sizes, from 12,800 to 409,600 tokens, from dark to bright."""
    return ramp(math.log(tokens / 12800) / math.log(32))


#: Experiment 8's sizes drawn one by one: below 12,800 tokens most of its worlds are stars.
DRAWN = (12800, 25600, 51200, 102400, 204800, 409600)


def e08_text(rows: List[Dict[str, Any]]) -> str:
    """How a recipe names Experiment 8's worlds that lived to their end."""
    runs = [w["run"] for w in rows if w["experiment"] == "E08"]
    return (f"The {len(runs)} baseline worlds of Experiment 8 that lived to iteration 600 — B1 at 1,600 to 409,600 "
            f"tokens, seeds 1 to 3, of which one world of 1,600 tokens and all of 3,200 tokens and more (`{runs[0]}`, "
            f"`{runs[1]}` … `{runs[-1]}`; [Worlds of many sizes](b1-the-network-size-and-scaling.md#worlds-of-many-sizes); "
            "`python3 gol_lab.py run E08`), after the game of iteration 599.")


STEPS_READ = [
    "Read the frame after each world's last game; build its network from `edges` and keep its largest piece "
    "(`book_chapters.space.largest_piece`). Its random twin: the same numbers of links per agent wired at random "
    "(`book_graph.degree_preserving`, seed 23), its largest piece.",
    "Balls: a breadth-first search all the way out from each of 100 agents drawn at random (seed 23); N(r) the "
    "geometric mean of the numbers within r steps; d(r) = ln(N(r+1)/N(r)) / ln((r+1)/r), read while N(r+1) is less "
    "than a quarter of the piece (`book_chapters.scaledims.reach` and `ball_curve`).",
    "Walks: a lazy random walk (stay with probability ½, else step to a neighbour drawn at random) from each of 48 "
    "agents drawn at random, its distribution propagated exactly, for up to 1,024 steps; P(t) the probability of being "
    "back at the start minus that of being there at rest (the walk is reversible, so the chance to be back after 2m "
    "steps follows from where it is after m, and half the steps are propagated); d(t) = −2 · ln(P(2t)/P(t)) / ln 2 on "
    "a grid of t of ratio √2, read while P(2t) is above 5/n (`walk` and `walk_curve`).",
    "A world whose best-linked agent was joined, on average over iterations 500 to its end, to a tenth of its agents "
    "or more is a star ([Stars and hubs](measuring-networks.md#stars-and-hubs); `maxDegree`/`nodes` of its rows in `stats.jsonl`).",
]
BODY_STEPS = (
    "Bodies: the n points of a square or cubic grid nearest its centre (and any as near as the last), each linked to "
    "the points beside it that are inside too (`book_chapters.scaledims.body`), for n = 250, 500, … 64,000; read as "
    "the worlds are. The endless grids' balls are counted exactly: Σₖ 2ᵏ·C(d, k)·C(r, k) points within r steps.")
SETTLED_STEP = ("The same balls, of each world and of its twin, after the games of five iterations evenly spaced over "
                "its settled life, from 500 to its last (500, 525, 550, 574 and 599 in Experiment 8; 500, 750, 1,000, "
                "1,249 and 1,499 in Experiment 10; 500, 1,125, 1,750, 2,374 and 2,999 in Experiment 2); each reading "
                "the median of the five (`book_chapters.scaledims.settled`).")
#: How the figures name and colour the three experiments' worlds.
KINDS = (("E08", "Experiment 8, 600 iterations", BLUE), ("E02", "Experiment 2, 3,000 iterations", GREEN),
         ("E10", "Experiment 10, 1,500 iterations", YELLOW))


def walked(w: Dict[str, Any]) -> Dict[str, Any]:
    """A world's walk: the long one where it has one (its first 1,024 steps are those of the short one)."""
    return w.get("long") or w["reading"]["world"]


def fit(group: List[Dict[str, Any]], key: str, kind: str = "world") -> Dict[str, Any]:
    """A power law of a reading over the settled life against the agents, its interval from resampling within sizes."""
    import gol_analysis as A
    x = np.array([w["settled"][kind]["agents"] for w in group], float)
    y = np.array([w["settled"][kind][key] for w in group], float)
    g = np.array([f"{w['experiment']}-{w['tokens']}" for w in group])
    ok = np.isfinite(x) & np.isfinite(y)
    return A.power_fit(x[ok], y[ok], g[ok], seed=SEED)


def ranked(a: List[float], b: List[float]) -> float:
    """Spearman's rank correlation of two lists, over the pairs where both are numbers."""
    pairs = [(x, y) for x, y in zip(a, b) if np.isfinite(x) and np.isfinite(y)]
    if len(pairs) < 4:
        return float("nan")
    u, v = (np.argsort(np.argsort(np.array(c))) for c in zip(*pairs))
    return float(np.corrcoef(u, v)[0, 1])


@chapter
def scaledims(ch: F.Chapter) -> None:
    rows = worlds()
    shapes = bodies()
    plain = [w for w in rows if not w["star"]]
    sets = {name: [w for w in plain if w["experiment"] == name] for name in ("E02", "E10")}
    by_size = {t: [w for w in plain if w["experiment"] == "E08" and w["tokens"] == t]
               for t in sorted({w["tokens"] for w in rows if w["experiment"] == "E08"})}
    runs_all = " ".join([e08_text(rows), F.lived_text("E02"), F.lived_text("E10")])

    ch.number("worlds", {
        **{f"E08-{t}": {"worlds": sum(w["experiment"] == "E08" and w["tokens"] == t for w in rows),
                        "stars": sum(w["experiment"] == "E08" and w["tokens"] == t and w["star"] for w in rows)}
           for t in by_size},
        **{name: {"worlds": sum(w["experiment"] == name for w in rows),
                  "stars": sum(w["experiment"] == name and w["star"] for w in rows)} for name in ("E02", "E10")}})

    def described(group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Every reading over a group of worlds: after the last game, and over the settled life."""
        last = ("agents", "links", "hub", "half", "balls", "walks", "peak", "reach", "walk_reach",
                *(f"r{r}" for r in RADII))
        return {"last": {kind: {k: describe([w[kind][k] for w in group]) for k in last} for kind in ("world", "twin")},
                "settled": {kind: {k: describe([w["settled"][kind][k] for w in group])
                                   for k in SETTLED_KEYS + (("width",) if kind == "world" else ())}
                            for kind in ("world", "twin")}}
    ch.number("by_size", {str(t): described(g) for t, g in by_size.items() if g})
    ch.number("sets", {name: described(g) for name, g in sets.items()})
    ch.number("each", [{"run": w["run"], "star": w["star"], "hub_share": w["hub_share"],
                        "last": {"world": w["world"], "twin": w["twin"]},
                        "settled": {k: w["settled"][k] for k in ("world", "twin", "moments")}} for w in rows])

    # ---- the rulers at every size ----
    def ball_curves(group: List[Dict[str, Any]]) -> List[Tuple[List[float], List[float]]]:
        """The balls of a group of worlds at every moment of their settled lives."""
        return [ball_curve(m) for w in group for m in w["settled"]["readings"]["world"]]

    def walk_curves(group: List[Dict[str, Any]]) -> List[Tuple[List[float], List[float]]]:
        """The walks of a group of worlds after their last game."""
        return [walk_curve(walked(w)) for w in group]
    panels = []
    for curves, title, xlabel in ((ball_curves, "From balls", "radius r, steps (logarithmic)"),
                                  (walk_curves, "From a random walk", "steps of the walk t (logarithmic)")):
        series = []
        for t in DRAWN:
            if by_size.get(t):
                series.append(line(*median_curve(curves(by_size[t])), f"{t:,}", size_colour(t), width=2))
        for name, label, dash in (("E02", "10,000 (Experiment 2)", [2, 3]),
                                  ("E10", "100,000 (Experiment 10)", [7, 4])):
            series.append(line(*median_curve(curves(sets[name])), label, GREY, width=1.5, dash=dash))
        panels.append(dict(title=title, x={"label": xlabel, "log": True}, y={"label": "dimension", "min": 0, "max": 4},
                           series=series, guides=[{"axis": "y", "at": g, "label": ""} for g in (2, 3)], height=260))
    ch.grid(
        "rulers", panels, columns=2, title="The two rulers, at every size",
        caption="The two rulers of [Is a world a space?](how-many-dimensions-does-a-world-have.md#two-rulers) on "
                "B1's worlds that are not stars, each line the median of the worlds of one size. From dark to "
                "bright: Experiment 8's worlds of 12,800 to 409,600 tokens (two or three of each size; below "
                "12,800 tokens most of its worlds are stars). Grey: the "
                f"{len(sets['E02'])} worlds of 10,000 tokens of Experiment 2 (dotted) and the "
                f"{len(sets['E10'])} of 100,000 tokens of Experiment 10 (dashed). Left: the local dimension from "
                "balls, until a typical ball holds a quarter of its world, over the worlds and five moments of "
                "their settled lives. Right: the spectral dimension from a lazy random walk after their last "
                "game, of up to 1,024 steps, and of up to 8,192 in the worlds of 409,600 tokens, until "
                "it is back no more than 5/n more often than at rest.",
        recipe=recipe(runs_all, FRAMES, STEPS_READ + [
            SETTLED_STEP,
            "In the three worlds of 409,600 tokens, the same 48 walks for up to 8,192 steps "
            "(`book_chapters.scaledims.long_reading`).",
            "Each line: at every radius (or walk), the median over the worlds of that size (and, for the balls, "
            "their five moments) that read there, where at least half of them do."]))
    ch.number("ball_curves", {**{str(t): dict(zip(("x", "y"), median_curve(ball_curves(g))))
                                 for t, g in by_size.items() if g},
                              **{s: dict(zip(("x", "y"), median_curve(ball_curves(sets[s])))) for s in sets}})
    ch.number("walk_curves", {**{str(t): dict(zip(("x", "y"), median_curve(walk_curves(g))))
                                 for t, g in by_size.items() if g},
                              **{s: dict(zip(("x", "y"), median_curve(walk_curves(sets[s])))) for s in sets}})
    ch.number("long_walks", {w["run"]: dict(zip(("x", "y"), walk_curve(w["long"]))) for w in rows if "long" in w})
    # The largest worlds one by one: each moment's balls, hub and width, and how far the agents are from the hub.
    ch.number("largest", {w["run"]: {"hub": hub_reach(w["run"]),
                                     "moments": {k: w["settled"]["moments"][k] for k in ("at", "agents", "hub", "half")},
                                     "curves": [dict(zip(("x", "y"), ball_curve(m)))
                                                for m in w["settled"]["readings"]["world"]]}
                          for w in by_size[max(by_size)]})

    # ---- bodies of known dimension, with an edge ----
    panels = []
    for curve, ruler, xlabel in ((ball_curve, "balls", "radius r, steps (logarithmic)"),
                                 (walk_curve, "a random walk", "steps of the walk t (logarithmic)")):
        for dim, shape in ((2, "a disc of a square grid"), (3, "a ball of a cubic grid")):
            series = []
            for k, b in enumerate(shapes[dim]):
                series.append(line(*curve(b["reading"]), f"{b['points']:,} points" if k in (0, 4, 8) else None,
                                   ramp(k / (len(shapes[dim]) - 1)), width=1.5))
            if curve is ball_curve:
                # The endless grid, as far out as the largest body reads.
                out = int(max(ball_curve(shapes[dim][-1]["reading"])[0])) + 2
                series.append(line(*endless(dim, out), "the endless grid", GREY, width=1.5, dash=[6, 4]))
            panels.append(dict(title=f"{shape[0].upper()}{shape[1:]}, by {ruler}", x={"label": xlabel, "log": True},
                               y={"label": "dimension", "min": 0, "max": 3.5}, series=series,
                               guides=[{"axis": "y", "at": dim, "label": ""}]))
    ch.grid(
        "bodies", panels, columns=2, title="What an edge does to the rulers",
        caption="Bodies whose dimension is known and which end at an edge, as a world does: a disc cut out of a "
                "square grid (dimension 2) and a ball cut out of a cubic grid (dimension 3), of 250, 500, 1,000 … "
                "64,000 points, each twice the last, from dark to bright, read by the same code as the worlds. Top: "
                "the local dimension from balls, until a typical ball holds a quarter of the body; dashed, the same "
                "grid with no edge at all, counted exactly. Bottom: the spectral dimension from a lazy random walk "
                "of up to 1,024 steps.",
        recipe=recipe("No runs: the bodies are built by `book_chapters.scaledims.body`.", "the bodies themselves",
                      [BODY_STEPS, "Balls and walks as for the worlds: 100 balls grown all the way out and 48 walks "
                                   "of up to 1,024 steps, from points drawn at random (seed 23)."]))
    ch.number("bodies", {str(dim): [{k: v for k, v in b.items() if k != "reading"} for b in shapes[dim]]
                         for dim in shapes})
    radii = (1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 20, 30)
    ch.number("endless", {str(dim): {str(r): float(endless(dim)[1][r - 1]) for r in radii} for dim in (2, 3)})
    ch.number("body_curves", {str(dim): {str(b["points"]): dict(zip(("x", "y"), ball_curve(b["reading"])))
                                         for b in shapes[dim]} for dim in shapes})
    ch.number("body_walks", {str(dim): {str(b["points"]): dict(zip(("x", "y"), walk_curve(b["reading"])))
                                        for b in shapes[dim]} for dim in shapes})

    # ---- one radius, every size ----
    def dots_of(key: str, kind: str = "world", stars: bool = False) -> List[Dict[str, Any]]:
        """Every world's reading over its settled life against its agents, a colour per experiment."""
        out = []
        for experiment, label, colour in KINDS:
            group = [w["settled"][kind] for w in rows if w["experiment"] == experiment and w["star"] == stars]
            out.append({"label": None if stars else label, "x": [g["agents"] for g in group],
                        "y": [g[key] for g in group], "points": True, "size": 5 if stars else 6,
                        "colour": GREY if stars else colour, "alpha": 0.5 if stars else 0.85})
        if stars:
            out[0]["label"] = "stars"
        return out

    def body_line(dim: int, key: str) -> Dict[str, Any]:
        """A reading of the bodies of one dimension against their points: red the ball, violet the disc."""
        name = "a ball of a cubic grid" if dim == 3 else "a disc of a square grid"
        return line([b["agents"] for b in shapes[dim]], [b[key] for b in shapes[dim]], name,
                    RED if dim == 3 else VIOLET, width=2)
    panels = []
    for r in RADII:
        guides = [{"axis": "y", "at": float(endless(dim)[1][r - 1]), "label": f"no edge, {g}"}
                  for dim, g in ((3, "3-D"), (2, "2-D"))]
        panels.append(dict(title=f"From radius {r} to {r + 1}", x={"label": "agents (logarithmic)", "log": True},
                           y={"label": "dimension", "min": 1, "max": 4.5}, height=240,
                           series=dots_of(f"r{r}") + dots_of(f"r{r}", stars=True) + [body_line(3, f"r{r}"),
                                                                                     body_line(2, f"r{r}")],
                           guides=guides, legend=False))
    ch.grid(
        "fixed", panels, columns=3, title="One radius, every size",
        caption="The local dimension from balls, read between the same two radii at every size, against the agents "
                "of a world's largest piece (logarithmic). Each dot is a world, its median over five moments of its "
                "settled life: blue, Experiment 8's worlds of 1,600 to 409,600 tokens; green, Experiment 2's worlds "
                "of 10,000 tokens; yellow, Experiment 10's of 100,000 tokens; grey, the stars among them. A world "
                "whose typical ball holds a quarter of it before the outer radius, at most of its moments, has no "
                "dot. Lines, read the same way at 250 to 64,000 points: red, the ball cut out of a cubic grid; "
                "violet, the disc cut out of a square grid. Dashed: the same grids with no edge.",
        recipe=recipe(runs_all, FRAMES, STEPS_READ[:2] + [STEPS_READ[3], SETTLED_STEP, BODY_STEPS]))
    ch.number("fixed", {f"r{r}": {"E08": {str(t): describe([w["settled"]["world"][f"r{r}"] for w in g])
                                          for t, g in by_size.items() if g},
                                  **{s: describe([w["settled"]["world"][f"r{r}"] for w in sets[s]]) for s in sets}}
                        for r in RADII})

    # ---- how wide a world is ----
    ladder = [w for w in plain if w["experiment"] == "E08" and w["tokens"] >= 3200]
    ch.number("fits", {name: {**{key: fit(group, key) for key in ("half", "mean_distance", "diameter")},
                              "twin_half": fit(group, "half", "twin")}
                       for name, group in (("ladder", ladder), ("all", plain))})
    law = ch.numbers["fits"]["ladder"]["half"]
    agents = [w["settled"]["world"]["agents"] for w in ladder]
    xs = np.exp(np.linspace(np.log(min(agents)), np.log(max(agents)), 40))
    series = dots_of("half") + dots_of("half", stars=True)
    series.append({"label": "their random twins", "x": [w["settled"]["twin"]["agents"] for w in plain],
                   "y": [w["settled"]["twin"]["half"] for w in plain], "points": True, "size": 4, "colour": ORANGE,
                   "alpha": 0.6})
    series.append(line(xs, law["prefactor"] * xs ** law["exponent"], f"agents^{law['exponent']:.2f}, fitted",
                       BLUE, width=1.5, dash=[6, 4]))
    series += [body_line(3, "half"), body_line(2, "half")]
    ch.figure(
        "width", title="How wide a world is, at every size",
        x={"label": "agents (logarithmic)", "log": True},
        y={"label": "typical distance, steps (logarithmic)", "log": True},
        series=series,
        caption="The typical distance of each world — the number of steps within which a typical agent's ball holds "
                "half of the world's largest piece — against the agents in that piece, both the medians over five "
                "moments of its settled life, on logarithmic axes. Blue: Experiment 8's worlds of 1,600 to 409,600 "
                "tokens, and dashed, the power law fitted to those of 3,200 tokens and more that are not stars, "
                f"agents^{law['exponent']:.2f} (95% interval {law['interval'][0]:.2f} to {law['interval'][1]:.2f}); "
                "green: Experiment 2's of 10,000 tokens; yellow: Experiment 10's of 100,000 tokens; grey: the stars "
                "among them; orange: the random twins of the worlds that are not stars. Lines: the disc and the ball "
                "cut out of a grid. On these axes a body of d dimensions is a line of slope 1/d, and a random "
                "network a curve that flattens.",
        recipe=recipe(runs_all, FRAMES, STEPS_READ[:2] + [STEPS_READ[3], SETTLED_STEP, BODY_STEPS,
                      "The typical distance: the radius at which N(r) first reaches half the piece, between whole "
                      "steps by the logarithm of N (`book_chapters.scaledims.half_reach`, as "
                      "`research/atlas/space.py` reads it).",
                      "Fit log₁₀(typical distance) = log₁₀ a + b · log₁₀(agents) by least squares over Experiment "
                      "8's worlds of 3,200 tokens and more that are not stars; the interval of b from 10,000 "
                      "resamplings of the worlds within each size (`gol_analysis.power_fit`)."]))

    # What a wider world would cost: the tokens that make the typical distance of the largest worlds half as
    # long again, or twice as long, by the fitted exponent, and the notebook's own estimate of a run of 600
    # iterations of B1 that large (gol_costs.predict).
    import gol_costs
    import gol_store
    exponent = ch.numbers["fits"]["ladder"]["half"]["exponent"]
    largest = max(by_size)
    config = dict(gol_store.load_meta(by_size[largest][0]["run"])["config"])
    larger = {}
    for factor in (1.5, 2):
        tokens = int(round(largest * factor ** (1 / exponent), -5))
        cost = gol_costs.predict({**config, "total_tokens": tokens}, 600)
        larger[str(factor)] = {"tokens": tokens, "agents": cost["agents"], "days": cost["seconds"] / 86400,
                               "memoryGB": cost["peakMB"] / 1024, "diskGB": cost["diskBytes"] / 2 ** 30}
    ch.number("larger", {"exponent": exponent, "from_tokens": largest, "times": larger})

    def slopes(dim: int) -> Dict[str, Any]:
        x = np.log([b["agents"] for b in shapes[dim]])
        y = np.log([b["half"] for b in shapes[dim]])
        return {"all": float(np.polyfit(x, y, 1)[0]), "from_2000": float(np.polyfit(x[3:], y[3:], 1)[0]),
                "local": [float((y[i + 1] - y[i]) / (x[i + 1] - x[i])) for i in range(len(x) - 1)]}
    ch.number("body_slopes", {str(dim): slopes(dim) for dim in shapes})
    ch.number("fluctuation", {
        "half": describe([max(h) / min(h) for w in plain for h in [w["settled"]["moments"]["half"]]
                          if all(np.isfinite(h))]),
        "balls": describe([float(np.nanmax(b) - np.nanmin(b)) for w in plain for b in [w["settled"]["moments"]["balls"]]
                           if np.sum(np.isfinite(b)) >= 3])})
    ch.number("within", {name: {key: ranked([w["settled"]["world"]["balls"] for w in group],
                                            [w["settled"]["world"][key] for w in group])
                                for key in ("links", "agents", "hub", "half")}
                         for name, group in (("E10", sets["E10"]), ("E02", sets["E02"]),
                                             ("E08", [w for w in plain if w["experiment"] == "E08"]))})
    # Among the worlds of 10,000 agents and more, do the larger still read more over their first steps?
    large = [w for w in plain if w["settled"]["world"]["agents"] >= 10000]
    ch.number("large", {"worlds": len(large),
                        **{key: ranked([w["settled"]["world"]["agents"] for w in large],
                                       [w["settled"]["world"][key] for w in large])
                           for key in ("r2", "r4", "balls", "hub")}})

    # ---- round 3's worlds, beside B1's width at their size ----
    three = round_three()
    if not three:
        return
    yardstick = fit(plain, "width")
    for w in three:
        w["against_b1"] = w["width"] / (yardstick["prefactor"] * w["agents"] ** yardstick["exponent"])
    span = [min(w["settled"]["world"]["agents"] for w in plain), max(w["settled"]["world"]["agents"] for w in plain)]
    xs = np.exp(np.linspace(np.log(span[0]), np.log(span[1]), 40))
    ch.figure(
        "round3", title="Round 3's worlds, beside B1's at their size",
        x={"label": "agents (logarithmic)", "log": True},
        y={"label": "typical distance, over the random twin's (logarithmic)", "log": True},
        series=[{"label": "B1's worlds (Experiments 2, 8 and 10)", "x": [w["settled"]["world"]["agents"] for w in plain],
                 "y": [w["settled"]["world"]["width"] for w in plain], "points": True, "size": 5, "colour": BLUE,
                 "alpha": 0.7},
                line(xs, yardstick["prefactor"] * xs ** yardstick["exponent"], "B1's width at each size, fitted",
                     BLUE, width=1.5, dash=[6, 4]),
                {"label": "round 3, the dead not mended", "x": [w["agents"] for w in three if not w["mended"]],
                 "y": [w["width"] for w in three if not w["mended"]], "points": True, "size": 6, "colour": GREEN,
                 "alpha": 0.85},
                {"label": "round 3, the dead mended", "x": [w["agents"] for w in three if w["mended"]],
                 "y": [w["width"] for w in three if w["mended"]], "points": True, "size": 6, "colour": ORANGE,
                 "alpha": 0.85}],
        caption="The width of a world — its typical distance over its random twin's — against its agents, on "
                "logarithmic axes. Blue: B1's worlds that are not stars, each the median over five moments of its "
                "settled life, and dashed, the power law fitted to them. Green and orange: the worlds of round 3 of "
                "the [screen of random worlds](a-screen-of-random-worlds.md) that lived to the end of their runs, "
                "each averaged from iteration 100 on as [Which worlds are healthy?](which-worlds-are-healthy.md) "
                "reads them, by whether their dead are mended (`mend_on_death`).",
        recipe=recipe(runs_all + " Round 3 of the screen: every world of Experiments 40 to 42 that lived to the end "
                      "of its run, 30,000 tokens, 3,000 iterations.",
                      "B1's frames, as above, and for round 3 `research/screen/health.json` (made by "
                      "`research/screen/health.py` from the screen's readings)",
                      [STEPS_READ[0], STEPS_READ[1], STEPS_READ[3], SETTLED_STEP,
                       "Width: the typical distance over the twin's, at each moment; its median over the five. Fit "
                       "log width against log agents over B1's worlds that are not stars (`gol_analysis.power_fit`).",
                       "Round 3: each world's `averages.width` and `averages.agents` in `health.json`, and "
                       "`set.mend_on_death`; nothing is measured again."]))
    ch.number("round3", {
        "yardstick": yardstick, "worlds": len(three), "mended": sum(w["mended"] for w in three),
        "wider_than_b1": sorted(([w["round"], w["seed"], round(w["agents"]), w["width"], w["against_b1"], w["mended"]]
                                 for w in three if w["against_b1"] >= 1), key=lambda r: -r[4]),
        "against_b1": {kind: describe([w["against_b1"] for w in three if w["mended"] == m])
                       for kind, m in (("mended", True), ("not_mended", False))}})


