# -*- coding: utf-8 -*-
"""
the physics of space: Which properties have no scale? — the same quantities of
a world measured at many sizes, and rescaled by a power of the size: do they
fall onto one curve, a data collapse, or does each keep a size of its own?

The worlds are those of Experiment 8 (B1 at eight sizes from 3,200 to 409,600
tokens, three seeds each) and, where about thirty seeds a size are needed, those
of Experiments 2 and 10 (28 at 10,000 and 30 at 100,000 tokens). The
positive control is percolation on a square lattice, a system known to have
no scale at one value of its parameter and a scale of its own at every other,
drawn by the same code as the worlds.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np

import book_data as D
import book_figures as F
import book_svg
import gol_analysis as A
from book_figures import FRAMES, STATS_FILE, chapter, describe, line, lived, recipe, runs_of
from book_graph import bfs, graph
from book_chapters.common import BLUE, GREY, ORANGE, VIOLET, YELLOW, listed

# ---------------------------------------------------------------------------
# What is read from a world's frames
# ---------------------------------------------------------------------------

#: The games a world is read over: its last LAST_GAMES, every one of them. For
#: the worlds of Experiment 8 that is iterations 500 to 599, their settled life.
LAST_GAMES = 100
#: Every how many of those games the network itself is read: its bridges, its tokens, its connections.
NET_EVERY = 10
#: Likeness at a distance: read at three moments, out of SOURCES agents drawn at random, out to MAX_R steps.
SOURCES = 150
MAX_R = 10
#: What the likeness is measured of, each as its logarithm: ln(1 + tokens), ln(1 + age), ln(connections).
ALIKE = ("tokens", "age", "degree")


def games_of(run_id: str) -> List[int]:
    """The iterations whose games are read: the last LAST_GAMES of the run."""
    last = D.last_iteration(run_id)
    return list(range(max(1, last - LAST_GAMES + 1), last + 1))


def components(nodes: set, edges: List[Tuple[int, int]]) -> List[int]:
    """The sizes of the connected pieces of the network on `nodes`, largest first."""
    adj: Dict[int, List[int]] = {u: [] for u in nodes}
    for u, v in edges:
        if u != v and u in adj and v in adj:
            adj[u].append(v)
            adj[v].append(u)
    seen: set = set()
    sizes = []
    for s in adj:
        if s in seen:
            continue
        seen.add(s)
        stack, size = [s], 0
        while stack:
            u = stack.pop()
            size += 1
            for w in adj[u]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def cut_pieces(start: Dict[str, Any], after: Dict[str, Any]) -> Tuple[int, List[int]]:
    """
    What one game removed, rebuilt from the world as the game began (`start`,
    the frame after reproduction) and what the game decided (`after`, the frame
    after the game): the agents that starved, and the size of every piece the
    cull cut off.

    A node on which nobody staked a token ends the game with none and starves.
    The connections across which no token was staked are cut (`pruned_edges`).
    Of what is left, every connected piece but the largest dies: those are the
    pieces cut off, as the engine's cleanup finds them.
    """
    staked: Counter = Counter()
    for r in after["decisions"]["allocations"]:
        for target, amount in zip(r["targets"], r["alloc"]):
            if amount:
                staked[target] += amount
    pruned = {(min(u, v), max(u, v)) for u, v in after["decisions"]["pruned_edges"]}
    living = {u for u in start["ids"] if staked[u] > 0}
    edges = [(u, v) for u, v in start["edges"] if (min(u, v), max(u, v)) not in pruned]
    sizes = components(living, edges)
    return len(start["ids"]) - len(living), sizes[1:]


def histogram(values: List[int]) -> List[List[int]]:
    """[[value, how many], …] in order of value: a distribution of whole numbers, kept small."""
    return [[int(v), int(c)] for v, c in sorted(Counter(values).items())]


def likeness_at(frame: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    """
    How alike agents r steps apart are, r = 1 … MAX_R, in one frame: for each
    value of ALIKE, the sums from which Pearson's correlation over all pairs
    (source, agent r steps away) is made, every value standardised within the
    frame; and how many of those pairs carry the same genotype.
    """
    adj = graph(frame)
    ids = list(frame["ids"])
    index = {u: i for i, u in enumerate(ids)}
    k = np.array([len(adj[u]) for u in ids], float)
    values = {"tokens": np.log1p(np.asarray(frame["tokens"], float)),
              "age": np.log1p(np.maximum(np.asarray(frame["ages"], float), 0)),
              "degree": np.log(np.maximum(k, 1))}
    values = {name: (v - v.mean()) / (v.std() or 1) for name, v in values.items()}
    geno = np.asarray(frame["brain_ids"])
    width = MAX_R + 1
    sums = {name: np.zeros((6, width)) for name in ALIKE}
    same = np.zeros(width)
    for s in rng.choice(len(ids), min(SOURCES, len(ids)), replace=False):
        dist = bfs(adj, ids[s], MAX_R)
        reached = np.fromiter((index[u] for u in dist), int, len(dist))
        r = np.fromiter(dist.values(), int, len(dist))
        reached, r = reached[r > 0], r[r > 0]
        count = np.bincount(r, minlength=width)
        same += np.bincount(r, weights=(geno[reached] == geno[s]).astype(float), minlength=width)
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
    counts = np.unique(geno, return_counts=True)[1].astype(float)
    n = len(ids)
    return {"agents": n, "sums": {name: acc.tolist() for name, acc in sums.items()}, "same": same.tolist(),
            "chance": float((counts * (counts - 1)).sum() / (n * (n - 1))) if n > 1 else None}


@D.measure("noscale")
def noscale_pass(run_id: str) -> Dict[str, Any]:
    """
    What Which properties have no scale? reads of a world, over its last
    LAST_GAMES games (games_of):

      games    one row per game: the iteration, the agents at its start, how many
               starved, how many single agents the cull cut off, the sizes of the
               larger pieces it cut off, and whether those add up to the run's own
               count of agents cut off (`cleanup.orphaned` of the frame)
      looks    every NET_EVERY games, the network after the game: its agents, and
               the distributions of the pieces behind its bridges (the smaller
               side of each), of tokens and of connections
      alike    at the first, middle and last of those games: likeness_at
    """
    from GraphOfLife import bridge_splits
    D.needs_decisions(run_id, "the pieces a game cuts off")
    rng = np.random.default_rng(62)
    its = games_of(run_id)
    games, looks, alike = [], [], []
    moments = {its[0], its[len(its) // 2], its[-1]}
    for t in its:
        start, after = D.frame_at(run_id, t, 1), D.frame_at(run_id, t, 2)
        starved, pieces = cut_pieces(start, after)
        games.append({"t": t, "agents": len(start["ids"]), "starved": starved,
                      "single": sum(1 for p in pieces if p == 1), "pieces": [p for p in pieces if p > 1],
                      "matches": sum(pieces) == (after.get("cleanup") or {}).get("orphaned")})
        if (its[-1] - t) % NET_EVERY == 0:
            adj = graph(after)
            n = len(after["ids"])
            behind = [min(below, n - below) for _, _, below in bridge_splits(after["ids"], adj)]
            looks.append({"t": t, "agents": n, "branches": histogram(behind),
                          "tokens": histogram(after["tokens"]),
                          "degrees": histogram([len(adj[u]) for u in after["ids"]])})
        if t in moments:
            alike.append({"t": t, **likeness_at(after, rng)})
    return {"games": games, "looks": looks, "alike": alike}


# ---------------------------------------------------------------------------
# The control: percolation on a square lattice
# ---------------------------------------------------------------------------

#: Site percolation on a square lattice: its threshold, at which its clusters
#: have no scale (Newman and Ziff 2000), and two exponents known exactly in two
#: dimensions (Stauffer and Aharony 1994) — the fractal dimension D of a
#: cluster there, and τ of the number of clusters of s sites, n(s) ∝ s^−τ.
P_C = 0.592746
FRACTAL = 91 / 48
TAU = 187 / 91
#: Below the threshold clusters have a size of their own; above it one cluster holds a fixed share of a lattice.
BELOW, ABOVE = 0.50, 0.70
#: The sides of the lattices, and how many lattices of each side are drawn.
SIDES = (32, 64, 128, 256, 512)
LATTICES = 200
#: At most this many sites are labelled at once.
BATCH_SITES = 1 << 22


def lattice_clusters(occupied: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    The clusters of a stack of square lattices — an array (lattices, side,
    side) of occupied and empty sites, with open edges — as the size of every
    cluster and the lattice it lies in. Two occupied sites side by side, across
    or down, belong to one cluster. Every site points at a root; each round the
    larger of two joined roots is hung under the smaller, and every pointer is
    followed to its root, until no two joined sites have different roots.
    """
    k, L, _ = occupied.shape
    index = np.arange(k * L * L).reshape(k, L, L)
    across = occupied[:, :, :-1] & occupied[:, :, 1:]
    down = occupied[:, :-1, :] & occupied[:, 1:, :]
    a = np.concatenate([index[:, :, :-1][across], index[:, :-1, :][down]])
    b = np.concatenate([index[:, :, 1:][across], index[:, 1:, :][down]])
    root = np.arange(k * L * L)
    while True:
        ra, rb = root[a], root[b]
        apart = ra != rb
        if not apart.any():
            break
        np.minimum.at(root, np.maximum(ra, rb)[apart], np.minimum(ra, rb)[apart])
        while True:
            up = root[root]
            if np.array_equal(up, root):
                break
            root = up
    sizes = np.bincount(root[index[occupied]])
    roots = np.nonzero(sizes)[0]
    return sizes[roots], roots // (L * L)


def percolation(p: float, seed: int) -> Dict[int, Dict[str, np.ndarray]]:
    """
    For every side of SIDES, LATTICES lattices whose sites are each occupied
    with probability p, drawn from one seed: the size of every cluster, and
    the largest cluster of each lattice.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for L in SIDES:
        sizes, largest, done = [], [], 0
        while done < LATTICES:
            k = min(max(1, BATCH_SITES // (L * L)), LATTICES - done)
            s, which = lattice_clusters(rng.random((k, L, L)) < p)
            top = np.zeros(k, dtype=np.int64)
            np.maximum.at(top, which, s)
            sizes.append(s)
            largest.append(top)
            done += k
        out[L] = {"sizes": np.concatenate(sizes), "largest": np.concatenate(largest)}
    return out


# ---------------------------------------------------------------------------
# Distributions, and how they are compared across sizes
# ---------------------------------------------------------------------------

def merged(histograms: List[List[List[int]]]) -> Tuple[np.ndarray, np.ndarray]:
    """Several histograms [[value, how many], …] added up: the values, in order, and how many of each."""
    total: Counter = Counter()
    for h in histograms:
        for v, c in h:
            total[v] += c
    values = sorted(total)
    return np.array(values, float), np.array([total[v] for v in values], float)


def counted(values: Any) -> Tuple[np.ndarray, np.ndarray]:
    """A list of whole numbers as a histogram: the distinct values, and how many of each."""
    v, c = np.unique(np.asarray(values), return_counts=True)
    return v.astype(float), c.astype(float)


def at_least(values: np.ndarray, counts: np.ndarray, per: float,
             points: int = 44) -> Tuple[np.ndarray, np.ndarray]:
    """
    At about `points` whole numbers s spread evenly on a logarithmic axis, from
    the smallest value to the largest: how many of the values are at least s,
    divided by `per` — per agent, say, or per site of a lattice.
    """
    order = np.argsort(values)
    v, c = values[order], counts[order]
    above = np.cumsum(c[::-1])[::-1]
    grid = np.unique(np.ceil(np.geomspace(v[0], v[-1], points)))
    return grid, above[np.searchsorted(v, grid)] / per


def tail_fit(values: np.ndarray, counts: np.ndarray, candidates: int = 24) -> Dict[str, float]:
    """
    The power law that fits the tail of a distribution of whole numbers best,
    as Power laws, real and apparent fits one (gol_series._scale_free), done on
    a histogram: for each candidate start s_min of the tail, the
    maximum-likelihood exponent γ of P(s) ∝ s^−γ; and the start whose fitted law
    lies closest to the data, by the Kolmogorov–Smirnov distance.
    """
    order = np.argsort(values)
    v, c = values[order], counts[order]
    usable = v[:-1]
    if len(usable) > candidates:
        usable = usable[(np.arange(candidates) * (len(usable) / candidates)).astype(int)]
    best: Dict[str, float] = {}
    for s_min in usable:
        keep = v >= s_min
        tv, tc = v[keep], c[keep]
        n = tc.sum()
        if n < 25:
            continue
        shift = s_min - 0.5
        total = float((tc * np.log(tv / shift)).sum())
        if total <= 1e-12:
            continue
        gamma = 1.0 + n / total
        model = 1.0 - (tv / shift) ** (1.0 - gamma)
        upto = np.cumsum(tc)
        ks = float(np.max(np.maximum(np.abs(model - (upto - tc) / n), np.abs(model - upto / n))))
        if not best or ks < best["ks"]:
            best = {"exponent": float(gamma), "start": float(s_min), "tail": float(n),
                    "share": float(n / c.sum()), "ks": ks}
    return best


def spread(x: np.ndarray) -> float:
    """How widely values vary, relative to their middle: (90th − 10th percentile) ÷ median."""
    q10, q50, q90 = np.quantile(x, [0.1, 0.5, 0.9])
    return float((q90 - q10) / q50) if q50 > 0 else float("nan")


def growth(points: List[Tuple[float, float, Any]], seed: int) -> Dict[str, Any]:
    """How y grows with x, as a power: a straight line through log y against log x, resampled within groups."""
    x, y, g = (np.array(c) for c in zip(*points))
    return A.power_fit(x.astype(float), y.astype(float), g, seed=seed)


# ---------------------------------------------------------------------------
# The worlds
# ---------------------------------------------------------------------------

#: The settled life of the worlds of Experiment 8; and the stretch over which the
#: worlds of Experiments 2 and 10 are set side by side. Both ends included.
SETTLED_E08 = (500, 599)
SETTLED_PAIR = (500, 1499)
#: A big cut: a tenth of a world or more in one game.
BIG = 0.10


def tokens_of(spec: Any) -> int:
    return int(spec.config["total_tokens"])


def e08_worlds() -> List[Any]:
    """The baseline worlds of Experiment 8 from 3,200 tokens up: every one of them lived to its end."""
    return [s for s in runs_of("E08", "baseline") if tokens_of(s) >= 3200 and lived(s) >= s.until]


def pair_worlds(experiment: str) -> List[Any]:
    """The baseline worlds of Experiment 2 or 10 that lived through the stretch they are compared over."""
    return [s for s in runs_of(experiment) if lived(s) > SETTLED_PAIR[1]]


def by_size(specs: List[Any]) -> Dict[int, List[Any]]:
    out: Dict[int, List[Any]] = {}
    for s in specs:
        out.setdefault(tokens_of(s), []).append(s)
    return dict(sorted(out.items()))


def size_colour(value: float, smallest: float, largest: float) -> str:
    """A size's colour: from blue-green for the smallest to yellow for the largest, evenly on a logarithmic scale."""
    t = np.log(value / smallest) / np.log(largest / smallest)
    return book_svg.cmap("viridis", 0.3 + 0.7 * float(t))


def e08_text(specs: List[Any]) -> str:
    sizes = sorted({tokens_of(s) for s in specs})
    return (f"The {len(specs)} baseline runs of Experiment 8 from {sizes[0]:,} tokens up, `{specs[0].run_id}` … "
            f"`{specs[-1].run_id}`: B1 at {listed(sizes)} tokens, seeds 1 to 3, 600 iterations each, every one "
            "alive at the end (every setting is listed in "
            "[Worlds of many sizes](b1-the-network-size-and-scaling.md#worlds-of-many-sizes)). They are made with "
            "`python3 gol_lab.py run E08`.")


def pair_text(small: List[Any], large: List[Any]) -> str:
    return (f"The {len(small)} baseline runs of Experiment 2 that lived past iteration {SETTLED_PAIR[1]:,} "
            f"(`B1-10000-s001` … `-s030`: B1 at 10,000 tokens, 3,000 iterations each, [The life of a B1 world]"
            f"(b1-lives-and-wealth.md#the-life-of-a-b1-world)) and the {len(large)} of Experiment 10 (`B1-100000-s001` … `-s030`: B1 at 100,000 "
            "tokens, 1,500 iterations each, [Worlds of many sizes](b1-the-network-size-and-scaling.md#worlds-of-many-sizes)). "
            "They are made with `python3 gol_lab.py run E02` and `python3 gol_lab.py run E10`.")


PERCOLATION_TEXT = (f"No runs: site percolation on square lattices, drawn by this page's code "
                    f"(`book_chapters.noscale.percolation`) — {LATTICES} lattices of each side "
                    f"{listed(list(SIDES))}, every site occupied with probability p, independently, by numpy's "
                    "generator seeded 1 for the threshold p = 0.592746, 2 for p = 0.50 and 3 for p = 0.70.")


def cut_shares(run_id: str, window: Tuple[int, int]) -> Dict[str, np.ndarray]:
    """Every game of the window: the agents at its start, and the shares of them it cut off and starved."""
    rows = [r for r in F.rows(run_id)
            if r["phase"] == 2 and window[0] <= r["iteration"] <= window[1] and r["nodes_before"]]
    before = np.array([r["nodes_before"] for r in rows], float)
    return {"agents": before, "cut": np.array([r["orphaned"] or 0 for r in rows], float) / before,
            "starved": np.array([r["starved"] or 0 for r in rows], float) / before}


def share_at_least(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """For every value of the grid, the share of x at least that large."""
    s = np.sort(x)
    return 1.0 - np.searchsorted(s, grid, side="left") / len(s)


#: Where the share of a world cut off in one game is read: from a thousandth to the whole.
SHARE_GRID = np.geomspace(0.001, 1.0, 46)


def unpaired(a: List[float], b: List[float], seed: int) -> Dict[str, Any]:
    """Two sets of worlds compared, never paired: their seeds are numbered alike but start nothing alike."""
    return A.compare_conditions({i: v for i, v in enumerate(a)}, {10 ** 6 + i: v for i, v in enumerate(b)},
                                seed=seed)


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

#: Counts too small for an axis to name (the figures' axes write a millionth at the least) are counted per thousand.
PER = 1000


def lattice_lines(found: Dict[int, Dict[str, np.ndarray]], rescaled: bool, legend: bool) -> List[Dict[str, Any]]:
    """
    Clusters of at least s sites per thousand sites, one line per side; rescaled,
    s ÷ L^D against s^(τ−1) times the clusters per site.
    """
    out = []
    for L, got in found.items():
        s, q = at_least(*counted(got["sizes"]), per=LATTICES * L * L)
        x, y = (s / L ** FRACTAL, q * s ** (TAU - 1)) if rescaled else (s, q * PER)
        out.append(line(x, y, f"{L * L:,} sites" if legend else None,
                        size_colour(L * L, SIDES[0] ** 2, SIDES[-1] ** 2), width=1.8))
    return out


def control(ch: F.Chapter) -> Dict[str, Any]:
    """Percolation, where no scale is known to be: the clusters, their collapse, and the largest cluster."""
    found = {p: percolation(p, seed) for p, seed in ((P_C, 1), (BELOW, 2), (ABOVE, 3))}
    s = np.geomspace(1, 3e4, 20)
    raw_axis = {"x": {"label": "cluster size s, in sites (logarithmic)", "log": True, "min": 1, "max": 3e5},
                "y": {"label": "clusters of s sites or more, per 1,000 sites", "log": True, "min": 1e-5,
                      "max": 1000}}
    scaled_axis = {"x": {"label": "s ÷ L^D (logarithmic)", "log": True, "min": 1e-4, "max": 3},
                   "y": {"label": "s^(τ−1) × clusters per site", "log": True, "min": 0.003, "max": 0.3}}
    reference = 0.1 * PER * s ** (1 - TAU)
    ch.grid(
        "control", [
            dict(title="At the threshold: as counted", **raw_axis,
                 series=lattice_lines(found[P_C], False, True)
                 + [line(s, reference, "a power law, s^−(τ−1)", "#eef4fa", width=1, dash=[5, 4])]),
            dict(title="At the threshold: rescaled", **scaled_axis,
                 series=lattice_lines(found[P_C], True, False), legend=False),
            dict(title=f"Below it, p = {BELOW:.2f}: as counted", **raw_axis,
                 series=lattice_lines(found[BELOW], False, False), legend=False),
            dict(title=f"Below it, p = {BELOW:.2f}: rescaled", **scaled_axis,
                 series=lattice_lines(found[BELOW], True, False), legend=False)],
        columns=2, title="What a collapse looks like, and what its failure looks like",
        caption="Site percolation on square lattices of side L = 32 to 512, from 1,024 to 262,144 sites: every site "
                "filled with probability p, and neighbouring filled sites joined into clusters. Top: at the "
                f"threshold p = {P_C}, where clusters have no scale; bottom: below it, at p = {BELOW:.2f}. Left: "
                "for every cluster size s, how many clusters of s sites or more a lattice holds, per 1,000 sites; "
                "one line per size of lattice, from blue-green (the smallest) to yellow (the largest). Right: the "
                f"same lines rescaled with the exponents known for two dimensions — s divided by L^D, D = 91/48 "
                f"≈ {FRACTAL:.3f}, and the clusters per site multiplied by s^(τ−1), τ = 187/91 ≈ {TAU:.3f}. "
                "Dashed: a power law of the exponent τ − 1, for comparison.",
        recipe=recipe(PERCOLATION_TEXT, "the lattices themselves",
                      ["Label the clusters of each lattice: filled sites side by side, across or down, belong to "
                       "one cluster; the edges of a lattice are open (`book_chapters.noscale.lattice_clusters`).",
                       "For each side L, pool the clusters of its 200 lattices; for every s, the clusters of s "
                       "sites or more, divided by the 200 · L² sites.",
                       "Rescaled: s ÷ L^D on the x axis, and the clusters per site times s^(τ−1) on the y axis.",
                       "The fits of the text, as for the worlds: a straight line through the logarithm of each "
                       "lattice's largest cluster against that of its sites (`gol_analysis.power_fit`), and the "
                       "maximum-likelihood exponent of the cluster sizes of the largest lattices "
                       "(`book_chapters.noscale.tail_fit`)."]))

    # The same fits as the worlds get, made where the answer is known.
    numbers: Dict[str, Any] = {}
    for p, name in ((P_C, "threshold"), (BELOW, "below"), (ABOVE, "above")):
        got = found[p]
        largest = growth([(L * L, m, L) for L, g in got.items() for m in g["largest"]], seed=621)
        tail = tail_fit(*counted(got[SIDES[-1]]["sizes"]))
        numbers[name] = {
            "largest": largest, "tail": tail,
            "largest_share": {str(L * L): describe(g["largest"] / (L * L)) for L, g in got.items()},
            "spread": {str(L * L): spread(g["largest"] / (L * L)) for L, g in got.items()}}
    ch.number("percolation", numbers)
    return {p: {L * L: spread(g["largest"] / (L * L)) for L, g in got.items()} for p, got in found.items()}


def cut_figures(ch: F.Chapter, lattice_spread: Dict[float, Dict[int, float]]) -> None:
    """The share of a world one game cuts off: does it average out, and does its tail?"""
    e08 = e08_worlds()
    sizes = by_size(e08)
    small, large = pair_worlds("E02"), pair_worlds("E10")
    shares = {s.run_id: cut_shares(s.run_id, SETTLED_E08) for s in e08}
    pair = {s.run_id: cut_shares(s.run_id, SETTLED_PAIR) for s in small + large}
    smallest, largest = min(sizes), max(sizes)

    # ---- does it average out? ----
    def points(table, specs, key):
        return [(float(table[s.run_id]["agents"].mean()), spread(table[s.run_id][key]), tokens_of(s)) for s in specs]

    fits = {key: growth(points(shares, e08, key), seed=622 + k) for k, key in enumerate(("cut", "starved"))}
    ns = np.geomspace(300, 1e5, 20)
    lattice_ns = np.array(sorted(lattice_spread[ABOVE]), float)

    def dots(table, specs, key, colour, label, size=6, alpha=0.85):
        pts = points(table, specs, key)
        return {"label": label, "x": [p[0] for p in pts], "y": [p[1] for p in pts], "points": True,
                "size": size, "colour": colour, "alpha": alpha}

    first = lattice_spread[ABOVE][int(lattice_ns[0])]
    world0 = float(np.median([p[1] for p in points(shares, sizes[smallest], "cut")]))
    n0 = float(np.median([p[0] for p in points(shares, sizes[smallest], "cut")]))
    ch.grid(
        "spread", [
            dict(title="Percolation: the largest cluster's share",
                 x={"label": "sites (logarithmic)", "log": True, "min": 500, "max": 5e5},
                 y={"label": "spread, (90th − 10th percentile) ÷ median", "log": True, "min": 0.005, "max": 20},
                 series=[line(lattice_ns, [lattice_spread[P_C][int(n)] for n in lattice_ns],
                              "at the threshold", YELLOW, width=2),
                         line(lattice_ns, [lattice_spread[ABOVE][int(n)] for n in lattice_ns],
                              f"above it, p = {ABOVE:.2f}", BLUE, width=2),
                         line(lattice_ns, first * (lattice_ns / lattice_ns[0]) ** -0.5,
                              "independent parts: slope −1/2", "#eef4fa", width=1, dash=[5, 4])]),
            dict(title="The worlds: the share one game removes",
                 x={"label": "agents at the start of a game (logarithmic)", "log": True, "min": 300, "max": 1e5},
                 y={"label": "spread, (90th − 10th percentile) ÷ median", "log": True, "min": 0.05, "max": 20},
                 series=[dots(pair, small + large, "cut", GREY, "cut off, 10,000 and 100,000 tokens", size=4,
                              alpha=0.45),
                         dots(shares, e08, "cut", ORANGE, "cut off, Experiment 8"),
                         dots(shares, e08, "starved", VIOLET, "starved, Experiment 8"),
                         line(ns, world0 * (ns / n0) ** -0.5, "independent parts: slope −1/2", "#eef4fa",
                              width=1, dash=[5, 4])])],
        columns=2, title="Does it average out?",
        caption="How widely a share varies, relative to its middle: the 90th percentile minus the 10th, divided by "
                "the median. Left: the share of a lattice held by its largest cluster, over the 200 lattices of each "
                f"size, at the threshold of percolation (yellow) and above it, at p = {ABOVE:.2f} (blue). Right: for "
                "every world, over the games of its settled life, the share of the agents at the start of a game "
                "that the game cut off (orange) and that starved in it (violet), in the worlds of Experiment 8 "
                f"over iterations {SETTLED_E08[0]} to {SETTLED_E08[1]}; small grey dots, the share cut off in the "
                f"worlds of 10,000 and 100,000 tokens, over iterations {SETTLED_PAIR[0]} to {SETTLED_PAIR[1]:,}. "
                "Dashed: what a sum of independent parts does — its spread falls as one over the square root of "
                "its size — drawn through the smallest lattices above the threshold (left) and through the worlds "
                "of 3,200 tokens (right).",
        recipe=recipe(e08_text(e08) + " " + pair_text(small, large) + " And the lattices of the figure above.",
                      STATS_FILE,
                      [f"For each world, the rows with `phase` = 2 in its stretch: `orphaned` / `nodes_before` and "
                       "`starved` / `nodes_before`, one value per game; their 10th, 50th and 90th percentiles; "
                       "x is the mean of `nodes_before`.",
                       "For each lattice, its largest cluster divided by its sites; the percentiles over the "
                       "200 lattices of each size.",
                       "The slopes in the text: least squares through log spread against log agents, the interval "
                       "from resampling the worlds within each size (`gol_analysis.power_fit`)."]))

    # ---- the tail ----
    lines_ = []
    for t, specs in sizes.items():
        x = np.concatenate([shares[s.run_id]["cut"] for s in specs])
        lines_.append(line(SHARE_GRID, share_at_least(x, SHARE_GRID), f"{t:,}",
                           size_colour(t, smallest, largest), width=1.8))
    pair_lines = []
    for specs, label, colour in ((small, "10,000 tokens", VIOLET), (large, "100,000 tokens", YELLOW)):
        x = np.concatenate([pair[s.run_id]["cut"] for s in specs])
        pair_lines.append(line(SHARE_GRID, share_at_least(x, SHARE_GRID), f"{label}, {len(specs)} worlds",
                               colour, width=2))
    axis = {"x": {"label": "share of the world cut off in one game, x (logarithmic)", "log": True,
                  "min": 0.001, "max": 1},
            "y": {"label": "share of games cutting off x or more", "log": True, "min": 1e-4, "max": 1}}
    guide = [{"axis": "x", "at": BIG, "label": "a tenth"}]
    ch.grid(
        "cuts", [
            dict(title="Eight sizes, three worlds each", **axis, series=lines_, guides=guide),
            dict(title=f"Two sizes, {len(small)} and {len(large)} worlds", **axis, series=pair_lines,
                 guides=guide)],
        columns=2, title="Small cuts average out, big cuts keep their share",
        caption="For every share x of a world, the share of games that cut off at least that share of the agents "
                "alive at their start; both axes logarithmic. Left: the worlds of Experiment 8, the three of each "
                f"size pooled, over the games of iterations {SETTLED_E08[0]} to {SETTLED_E08[1]}, one line per "
                "size from blue-green (3,200 tokens) to yellow (409,600). Right: the worlds of 10,000 and of "
                f"100,000 tokens, pooled, over iterations {SETTLED_PAIR[0]} to {SETTLED_PAIR[1]:,}. The dashed "
                "line marks a tenth of a world.",
        recipe=recipe(e08_text(e08) + " " + pair_text(small, large), STATS_FILE,
                      ["Take `orphaned` / `nodes_before` of every row with `phase` = 2 in the stretch, and pool "
                       "the worlds of each size.",
                       "For every x, the share of those games with a value of x or more."]))

    # ---- the numbers ----
    per_size = {}
    for t, specs in sizes.items():
        x = np.concatenate([shares[s.run_id]["cut"] for s in specs])
        y = np.concatenate([shares[s.run_id]["starved"] for s in specs])
        per_size[str(t)] = {
            "agents": float(np.mean([shares[s.run_id]["agents"].mean() for s in specs])), "games": int(x.size),
            "cut": {"mean": float(x.mean()), "median": float(np.median(x)), "q10": float(np.quantile(x, 0.1)),
                    "q90": float(np.quantile(x, 0.9)), "q99": float(np.quantile(x, 0.99)),
                    "largest": float(x.max()), "big": float(np.mean(x >= BIG)), "big_games": int(np.sum(x >= BIG)),
                    "none": float(np.mean(x == 0)),
                    "spread": describe([spread(shares[s.run_id]["cut"]) for s in specs])},
            "starved": {"mean": float(y.mean()), "median": float(np.median(y)), "largest": float(y.max()),
                        "spread": describe([spread(shares[s.run_id]["starved"]) for s in specs])}}

    def world_stats(table, specs, key):
        out = {"spread": [], "median": [], "mean": [], "big": [], "huge": [], "q99": [], "largest": [], "agents": []}
        for s in specs:
            x = table[s.run_id][key]
            out["spread"].append(spread(x))
            out["median"].append(float(np.median(x)))
            out["mean"].append(float(x.mean()))
            out["big"].append(float(np.mean(x >= BIG)))
            out["huge"].append(float(np.mean(x >= 2 * BIG)))
            out["q99"].append(float(np.quantile(x, 0.99)))
            out["largest"].append(float(x.max()))
            out["agents"].append(float(table[s.run_id]["agents"].mean()))
        return out

    compared = {}
    for k, key in enumerate(("cut", "starved")):
        a, b = world_stats(pair, small, key), world_stats(pair, large, key)
        compared[key] = {name: {"small": describe(a[name]), "large": describe(b[name]),
                                "test": unpaired(a[name], b[name], seed=630 + 10 * k + j)}
                         for j, name in enumerate(a)}
        grow = np.log(np.mean(b["agents"]) / np.mean(a["agents"]))
        compared[key]["spread_exponent"] = float(np.log(np.mean(b["spread"]) / np.mean(a["spread"])) / grow)
        # Pieces of a share x of the world or more number about N · (xN)^−(γ−1) a game, which grows as N^(2−γ):
        # the interval of the big games' rate between the two sizes bounds 2 − γ, taking each as one piece.
        big = compared[key]["big"]
        if big["small"].get("mean"):
            rate = [1 + d / big["small"]["mean"] for d in big["test"]["interval"]]
            compared[key]["big_rate_exponent"] = [float(np.log(r) / grow) if r > 0 else None for r in rate]
    ch.number("cuts", {"fits": fits, "by_size": per_size, "pair": compared})


# ---------------------------------------------------------------------------
# What the frames say: pieces, branches, riches, likeness
# ---------------------------------------------------------------------------

class Kept:
    """
    What noscale_pass kept of the worlds of one size, pooled: every distribution
    of whole numbers as a histogram with what it is counted per (agents times
    games, or agents times looks), and the largest value of each game or look
    as a share of its world.
    """

    def __init__(self, specs: List[Any]) -> None:
        self.specs = specs
        self.runs = [noscale_pass(s.run_id) for s in specs]

    @property
    def games(self) -> List[Dict[str, Any]]:
        return [g for r in self.runs for g in r["games"]]

    @property
    def looks(self) -> List[Dict[str, Any]]:
        return [x for r in self.runs for x in r["looks"]]

    @property
    def agents(self) -> float:
        """The typical number of agents at the start of a game."""
        return float(np.mean([g["agents"] for g in self.games]))

    def run_agents(self) -> List[float]:
        """Each world's typical number of agents at the start of a game."""
        return [float(np.mean([g["agents"] for g in r["games"]])) for r in self.runs]

    def pieces(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """The pieces the games cut off, and the agents at the games' starts they are counted per."""
        hist: Counter = Counter()
        for g in self.games:
            hist[1] += g["single"]
            hist.update(g["pieces"])
        values = sorted(v for v in hist if hist[v])
        return (np.array(values, float), np.array([hist[v] for v in values], float),
                float(sum(g["agents"] for g in self.games)))

    def piece_largest(self, share: bool = True) -> List[np.ndarray]:
        """
        For every world, each game's largest piece cut off (0 if it cut none off),
        as a share of the agents at the game's start or in agents.
        """
        return [np.array([max(g["pieces"] or [1 if g["single"] else 0]) / (g["agents"] if share else 1)
                          for g in r["games"]]) for r in self.runs]

    def network(self, key: str) -> Tuple[np.ndarray, np.ndarray, float]:
        """A distribution read off the network every NET_EVERY games, and the agents it is counted per."""
        v, c = merged([x[key] for x in self.looks])
        return v, c, float(sum(x["agents"] for x in self.looks))

    def network_largest(self, key: str, share: bool = True) -> List[np.ndarray]:
        """For every world, the largest value of each look, as a share of the world's agents or as it is."""
        return [np.array([max(v for v, _ in x[key]) / (x["agents"] if share else 1) for x in r["looks"] if x[key]])
                for r in self.runs]

    def at_least_share(self, key: str, share: float) -> np.ndarray:
        """
        For every game (pieces) or look (branches), how many pieces or branches
        hold `share` of the world's agents or more.
        """
        if key == "pieces":
            return np.array([sum(1 for p in g["pieces"] if p >= share * g["agents"]) for g in self.games])
        return np.array([sum(c for v, c in x[key] if v >= share * x["agents"]) for x in self.looks])

    def likeness(self) -> Dict[str, np.ndarray]:
        """For r = 1 … MAX_R: the correlation of each value of ALIKE between agents r steps apart, and kin."""
        out = {}
        for name in ALIKE:
            m, sx, sy, sxx, syy, sxy = np.sum([np.array(a["sums"][name]) for r in self.runs for a in r["alike"]],
                                              axis=0)[:, 1:]
            with np.errstate(invalid="ignore", divide="ignore"):
                cov = sxy / m - sx / m * sy / m
                out[name] = cov / np.sqrt((sxx / m - (sx / m) ** 2) * (syy / m - (sy / m) ** 2))
        pairs = np.sum([np.array(a["sums"]["age"][0]) for r in self.runs for a in r["alike"]], axis=0)[1:]
        same = np.sum([np.array(a["same"]) for r in self.runs for a in r["alike"]], axis=0)[1:]
        out["kin"] = same / pairs
        out["chance"] = float(np.mean([a["chance"] for r in self.runs for a in r["alike"]]))
        return out


def fading(c: np.ndarray) -> float:
    """
    How many steps beyond the first a likeness takes to fall to 1/e of what it is
    at one step: read between the two distances it falls between, on a
    logarithmic scale. NaN if it never falls that far within MAX_R steps.
    """
    target = c[0] / np.e
    for r in range(1, len(c)):
        if c[r] <= target:
            if c[r] <= 0:
                return float(r - 1 + (c[r - 1] - target) / (c[r - 1] - c[r]))
            return float(r - 1 + np.log(c[r - 1] / target) / np.log(c[r - 1] / c[r]))
    return float("nan")


def size_lines(found: Dict[int, Tuple[np.ndarray, np.ndarray, float]], agents: Dict[int, float],
               rescaled: bool = False, legend: bool = False) -> List[Dict[str, Any]]:
    """
    One line per size of world: for every value s, how many values are at least
    s per thousand agents; rescaled, s as a share x of the world's agents against
    x times how many values of that share or more a world holds — which is s times
    the count per agent, the collapse of a law P(s) ∝ s^−2.
    """
    smallest, largest = min(found), max(found)
    out = []
    for t, (v, c, per) in found.items():
        s, q = at_least(v, c, per)
        x, y = (s / agents[t], s * q) if rescaled else (s, q * PER)
        out.append(line(x, y, f"{t:,}" if legend else None, size_colour(t, smallest, largest), width=1.8))
    return out


#: The exponent of a law P(s) ∝ s^−2, which a tree grown at random has for what hangs below each of its links:
#: the share of values of at least s then falls as s^−1. The pieces and the branches are rescaled with it.
EVERY_SIZE = 2.0
#: The tree grown at random that the branches are set beside: its members, and the seed it is drawn with.
TREE = 200_000
TREE_SEED = 4


def random_tree_branches(n: int, seed: int) -> np.ndarray:
    """
    A tree grown by joining each newcomer to a member drawn at random, n members
    in all: for every link, the members behind it on its smaller side.
    """
    rng = np.random.default_rng(seed)
    parent = (rng.random(n) * np.arange(n)).astype(np.int64)    # member i joins one of 0 … i − 1
    below = np.ones(n, dtype=np.int64)
    for i in range(n - 1, 0, -1):          # every member comes after its parent, so its count is whole by now
        below[parent[i]] += below[i]
    return np.minimum(below[1:], n - below[1:])


def frames_text(e08: List[Any], large: List[Any]) -> str:
    return (e08_text(e08) + f" For the numbers at 100,000 tokens, the {len(large)} runs of Experiment 10 "
            "(`B1-100000-s001` … `-s030`, `python3 gol_lab.py run E10`), over their last 100 games, "
            "iterations 1,400 to 1,499.")


def piece_figures(ch: F.Chapter, kept: Dict[int, Kept], large: Kept) -> None:
    """The pieces one game cuts off, and what hangs behind a bridge: their sizes at every size of world."""
    e08 = [s for k in kept.values() for s in k.specs]
    agents = {t: k.agents for t, k in kept.items()}
    numbers: Dict[str, Any] = {}
    tree = random_tree_branches(TREE, TREE_SEED)
    tree_s, tree_q = at_least(*counted(tree), per=TREE)
    numbers["random_tree"] = {"members": TREE, "seed": TREE_SEED,
                              "at_least": {str(x): float(np.mean(tree >= x)) for x in (2, 10, 100, 1000, 10000)}}
    for key, title, what, per_what, unit in (
            ("pieces", "The pieces one game cuts off", "pieces", "per 1,000 agents a game", "per game"),
            ("branches", "What hangs behind a bridge", "branches", "per 1,000 agents", "per world")):
        found = {t: (k.pieces() if key == "pieces" else k.network(key)) for t, k in kept.items()}
        s = np.geomspace(1, 1e5, 20)
        level = PER * float(np.median([at_least(*f[:2], per=f[2])[1][0] for f in found.values()]))
        reference = (line(s, level * s ** (1 - EVERY_SIZE), "a power law, P(s) ∝ s^−2", "#eef4fa", width=1,
                          dash=[5, 4]) if key == "pieces" else
                     line(tree_s, tree_q * PER, f"a tree grown at random, {TREE:,} members", "#eef4fa", width=1,
                          dash=[5, 4]))
        raw = dict(title="As counted", x={"label": "size s, in agents (logarithmic)", "log": True, "min": 1, "max": 1e5},
                   y={"label": f"{what} of s or more, {per_what}", "log": True, "min": 1e-5, "max": 1000},
                   series=size_lines(found, agents, legend=True) + [reference])
        scaled = dict(title="Rescaled by the size of the world",
                      x={"label": "x = s ÷ agents in the world (logarithmic)", "log": True, "min": 1e-5, "max": 1},
                      y={"label": f"x × {key} of x or more, {unit}", "log": True, "min": 1e-4, "max": 1},
                      series=size_lines(found, agents, rescaled=True), legend=False)
        if key == "pieces":
            caption = (f"Every piece the cull removed in the games of iterations {SETTLED_E08[0]} to "
                       f"{SETTLED_E08[1]} of the worlds of Experiment 8 — the connected pieces of the network, other "
                       "than the largest, left when the game's unused connections are cut and the agents with no "
                       "tokens removed. Left: for every size s, how many pieces of s agents or more a game cut off, "
                       "per 1,000 agents at its start; one line per size of world, the three worlds of a size "
                       "pooled, from blue-green (3,200 tokens) to yellow (409,600). Dashed: the law P(s) ∝ s^−2, for "
                       "comparison. Right: the same lines with s as a share x of the agents of the world, and the "
                       "number of pieces of that share or more one game cuts off, times x.")
            steps = ["For every game, read frame 2·t (the world after reproduction) and frame 2·t + 1 (after the "
                     "game). A node on which no agent staked a token in the game's `allocations` starves; take it "
                     "away, and every connection in the game's `pruned_edges`.",
                     "The connected pieces of what is left; all but the largest are the pieces cut off. In every "
                     "game read, they add up to the run's own count (`cleanup.orphaned`) "
                     "(`book_chapters.noscale.cut_pieces`).",
                     "For each size, pool the pieces of its three worlds; for every s, the pieces of s or more "
                     "divided by the agents at the start of all the games, times 1,000."]
        else:
            caption = (f"Every bridge of the network after the games of iterations {SETTLED_E08[0] + 9}, "
                       f"{SETTLED_E08[0] + 19}, … {SETTLED_E08[1]} of the worlds of Experiment 8, and the smaller of the "
                       "two pieces it holds together: the branch behind it, which cutting it would cut off "
                       "([Cut risk](measuring-networks.md#cut-risk)). Left: for every size s, how many branches of s agents or "
                       "more a world has, per 1,000 agents; one line per size of world, from blue-green (3,200 "
                       "tokens) to yellow (409,600). Dashed: the same for a tree grown by joining each newcomer to a "
                       f"member drawn at random, {TREE:,} members, per 1,000 members — exactly 1,000/s. Right: the "
                       "worlds' lines with s as a share x of the agents of the world, and the number of branches of "
                       "that share or more a world has, times x.")
            steps = ["Read frames 2·t + 1 for t = 509, 519, … 599: `ids` and `edges`.",
                     "Every bridge, and the agents behind it on each side (`GraphOfLife.bridge_splits`); the "
                     "branch is the smaller side.",
                     "For each size, pool the branches of its three worlds' ten looks; for every s, the branches of "
                     "s or more divided by the agents of all the looks, times 1,000.",
                     f"The tree: member i joins a member drawn at random from 0 … i − 1, by numpy's generator "
                     f"seeded {TREE_SEED}; for every link, the smaller side (`book_chapters.noscale."
                     "random_tree_branches`)."]
        ch.grid(key, [raw, scaled], columns=2, title=title, caption=caption,
                recipe=recipe(e08_text(e08), FRAMES, steps + [
                    "Rescaled: x = s ÷ the mean number of agents of the size; on the y axis, s times the count per "
                    "agent, which is x times the count per game (or per world)."]))

        def largest_of(k: Kept, share: bool = True) -> List[np.ndarray]:
            return k.piece_largest(share) if key == "pieces" else k.network_largest(key, share)

        def facts(k: Kept, f: Tuple[np.ndarray, np.ndarray, float]) -> Dict[str, Any]:
            v, c, per = f
            tenth = k.at_least_share(key, BIG)
            return {"agents": k.agents, "count_per_agent": float(c.sum() / per),
                    "single": float(c[v == 1].sum() / c.sum()), "tail": tail_fit(v, c),
                    "largest_share": describe(np.concatenate(largest_of(k))),
                    "largest_agents": describe(np.concatenate(largest_of(k, share=False))),
                    "at_least": {str(x): float(c[v >= x].sum() / per) for x in (2, 10, 100, 1000, 10000)},
                    "tenth": {"mean": float(tenth.mean()), "with_any": float(np.mean(tenth > 0)),
                              "of": int(tenth.size)}}
        numbers[key] = {str(t): facts(k, found[t]) for t, k in kept.items()}
        numbers[key]["E10"] = facts(large, large.pieces() if key == "pieces" else large.network(key))
        # How the typical largest grows with the world: each world's median, in agents, against its agents.
        numbers[key]["largest_growth"] = growth(
            [(n, float(np.median(m)), t) for t, k in kept.items()
             for n, m in zip(k.run_agents(), largest_of(k, share=False))], seed=641 if key == "pieces" else 642)
    # The landslides: how much of what a big game cut off was one piece.
    landslides = {}
    for t, k in list(kept.items()) + [("E10", large)]:
        big = [(max(g["pieces"] or [1]), g["single"] + sum(g["pieces"]), g["agents"]) for g in k.games
               if g["single"] + sum(g["pieces"]) >= BIG * g["agents"]]
        landslides[str(t)] = {"games": len(big), "of": len(k.games),
                              "in_largest": float(np.sum([b[0] for b in big]) / np.sum([b[1] for b in big])) if big else None}
    numbers["landslides"] = landslides
    numbers["checked"] = {"games": int(sum(len(k.games) for k in kept.values()) + len(large.games)),
                          "matching": int(sum(g["matches"] for k in list(kept.values()) + [large] for g in k.games))}
    ch.number("pieces", numbers)


def riches_figures(ch: F.Chapter, kept: Dict[int, Kept], large: Kept) -> None:
    """Tokens and connections at every size: do their tails move out with the world?"""
    e08 = [s for k in kept.values() for s in k.specs]
    agents = {t: k.agents for t, k in kept.items()}
    panels, numbers = [], {}
    for key, what, held in (("tokens", "agents holding x tokens or more", "tokens held, x"),
                            ("degrees", "agents with x connections or more", "connections, x")):
        found = {t: k.network(key) for t, k in kept.items()}
        panels.append(dict(title="Tokens" if key == "tokens" else "Connections",
                           x={"label": f"{held} (logarithmic)", "log": True, "min": 1, "max": 1e4},
                           y={"label": f"{what}, per 1,000 agents", "log": True, "min": 1e-4, "max": 1000},
                           series=size_lines(found, agents, legend=key == "tokens"), legend=key == "tokens"))
        numbers[key] = {str(t): {"tail": tail_fit(*found[t][:2]),
                                 "at_least": {str(x): float(found[t][1][found[t][0] >= x].sum() / found[t][2])
                                              for x in (10, 100, 1000, 3000)}} for t in kept}
        numbers[key]["E10"] = {"tail": tail_fit(*large.network(key)[:2])}
    for key, title, label in (("tokens", "The richest agent", "tokens of the richest"),
                              ("degrees", "The best-connected agent", "connections of the best-connected")):
        pts = [(n, float(np.median(m)), t) for t, k in kept.items()
               for n, m in zip(k.run_agents(), k.network_largest(key, share=False))]
        e10 = [(n, float(np.median(m))) for n, m in zip(large.run_agents(), large.network_largest(key, share=False))]
        fit = growth(pts, seed=651 if key == "tokens" else 652)
        top = [p for p in pts if p[2] >= 102400]
        numbers[f"{key}_largest"] = {"growth": fit, "from_102400": growth(top, seed=653) if len({p[2] for p in top}) >= 3
                                     else None,
                                     "by_size": {str(t): describe([p[1] for p in pts if p[2] == t]) for t in kept},
                                     "E10": describe([p[1] for p in e10])}
        ns = np.geomspace(300, 1e5, 20)
        smallest = [p for p in pts if p[2] == min(kept)]
        n0 = float(np.exp(np.mean([np.log(p[0]) for p in smallest])))
        y0 = float(np.exp(np.mean([np.log(p[1]) for p in smallest])))
        gamma = numbers[key][str(min(kept))]["tail"]["exponent"]
        panels.append(dict(title=title, x={"label": "agents in the world (logarithmic)", "log": True,
                                           "min": 300, "max": 1e5},
                           y={"label": f"{label} (logarithmic)", "log": True, "min": 30, "max": 3e4},
                           series=[{"label": "one world of Experiment 8", "x": [p[0] for p in pts],
                                    "y": [p[1] for p in pts], "points": True, "size": 6, "colour": ORANGE,
                                    "alpha": 0.9},
                                   {"label": "the 30 of 100,000 tokens", "x": [p[0] for p in e10],
                                    "y": [p[1] for p in e10], "points": True, "size": 4, "colour": GREY,
                                    "alpha": 0.6},
                                   line(ns, y0 * (ns / n0) ** (1 / (gamma - 1)),
                                        f"the largest of n draws from a pure power law", "#eef4fa", width=1,
                                        dash=[5, 4])],
                           legend=key == "tokens"))
    panels = [panels[0], panels[2], panels[1], panels[3]]
    ch.grid(
        "riches", panels, columns=2, title="Tokens and connections at every size",
        caption="Left: for every x, how many agents of every 1,000 hold x tokens or more (top) or have x "
                "connections or more (bottom), in the worlds of Experiment 8 after the games of iterations 509, "
                "519, … 599, the three worlds of a size pooled; one line per size, from blue-green (3,200 tokens) to "
                "yellow (409,600). Right: for every world, the median over those ten looks of the most tokens one agent "
                "held (top) and of the most connections (bottom), against the agents of the world: orange, the "
                f"worlds of Experiment 8; grey, the {len(large.specs)} worlds of Experiment 10, over their last 100 "
                "games. Dashed: how the largest of n values would grow if they were drawn from the power law "
                "fitted to the tail of the smallest worlds — as n^(1/(γ−1)) — drawn through the smallest worlds.",
        recipe=recipe(frames_text(e08, large.specs), FRAMES,
                      ["Read frames 2·t + 1 for the ten looks: `tokens`, and every agent's connections counted "
                       "from `edges`.",
                       "Left: pool each size's looks; for every x, the agents with x or more divided by all agents, "
                       "times 1,000.",
                       "Right: the largest value of each look, and its median over the looks of each world.",
                       "The tail exponent γ: the maximum-likelihood fit of [Power laws, real and apparent]"
                       "(measures-of-the-archived-studies.md#fitting-a-power-law-properly), done on the pooled "
                       "values of the smallest worlds (`book_chapters.noscale.tail_fit`)."]))
    ch.number("riches", numbers)


def likeness_figure(ch: F.Chapter, kept: Dict[int, Kept], large: Kept) -> None:
    """How far likeness reaches, at every size."""
    e08 = [s for k in kept.values() for s in k.specs]
    rs = np.arange(1, MAX_R + 1)
    found = {t: k.likeness() for t, k in kept.items()}
    smallest, largest = min(kept), max(kept)

    def lines_of(name: str, legend: bool) -> List[Dict[str, Any]]:
        return [line(rs, f[name], f"{t:,}" if legend else None, size_colour(t, smallest, largest), width=1.8)
                for t, f in found.items()]
    ch.grid(
        "alike", [
            dict(title="Kin: the same genotype", x={"label": "steps apart", "min": 1, "max": MAX_R},
                 y={"label": "share of pairs with one genotype (logarithmic)", "log": True, "min": 1e-5, "max": 0.3},
                 series=lines_of("kin", True)),
            dict(title="Age", x={"label": "steps apart", "min": 1, "max": MAX_R},
                 y={"label": "correlation of ln(1 + age)", "min": -0.1, "max": 0.3},
                 series=lines_of("age", False), guides=[{"axis": "y", "at": 0.0, "label": ""}], legend=False)],
        columns=2, title="How far likeness reaches, at every size",
        caption="For pairs of agents r steps apart in the worlds of Experiment 8: left, the share of the pairs whose "
                "two agents carry the same genotype (logarithmic); right, the correlation of their ages, as "
                "ln(1 + age). Pairs from breadth-first searches out of 150 agents drawn at random, after the games "
                f"of iterations {SETTLED_E08[0]}, 549 and {SETTLED_E08[1]}, the three worlds of a size pooled; one "
                "line per size, from blue-green (3,200 tokens) to yellow (409,600).",
        recipe=recipe(e08_text(e08), FRAMES,
                      [f"Read frames 2·t + 1 for t = {SETTLED_E08[0]}, 549 and {SETTLED_E08[1]}; draw 150 agents at "
                       "random (numpy's generator seeded 62, once per run) and search out from each to "
                       f"{MAX_R} steps.",
                       "Every agent reached at r steps makes a pair with the agent searched from. Kin: the share of "
                       "the pairs at r that carry the same `brain_ids`. Age: Pearson's correlation over the pairs at "
                       "r of ln(1 + `ages`), standardised within each frame, as in [Like next to like]"
                       "(b1-the-network-size-and-scaling.md#the-correlation-function) (`book_chapters.noscale.likeness_at`).",
                       "Pool the sums of each size's three worlds and three moments."]))
    numbers = {}
    for t, f in list(found.items()) + [("E10", large.likeness())]:
        excess = f["kin"] - f["chance"]
        numbers[str(t)] = {"kin": f["kin"].tolist(), "chance": f["chance"], "kin_fading": fading(excess),
                           "age": f["age"].tolist(), "age_fading": fading(f["age"]),
                           "tokens": f["tokens"].tolist(), "degree": f["degree"].tolist()}
    ch.number("alike", numbers)


@chapter
def noscale(ch: F.Chapter) -> None:
    lattice_spread = control(ch)
    cut_figures(ch, lattice_spread)
    kept = {t: Kept(specs) for t, specs in by_size(e08_worlds()).items()}
    large = Kept(pair_worlds("E10"))
    piece_figures(ch, kept, large)
    riches_figures(ch, kept, large)
    likeness_figure(ch, kept, large)

