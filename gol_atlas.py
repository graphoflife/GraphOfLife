#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What goes with what: correlations among the columns of one table.

A table is rows and named columns, and each column has a kind that says what
its logarithmic axis is. Four tables are made here, all of them for the
Research tab's Correlations view and the first also for the book (Appendix B):

- `agents`: one row for every agent in a game of a window of frames — its
  state, its neighbourhood, its place in the network, what it staked, and what
  then happened to it (VARIABLES);
- `stats`: one row for every game of the window, the world's statistics after
  it (gol_framestats.strip, the statistics a run records);
- `changes`: the same, as the change from one game to the next. Two statistics
  that both drift over a run correlate whatever they have to do with each
  other; their changes from game to game do not share the drift;
- `worlds`: one row for every simulation in book/results/worlds.json, its
  statistics over a stretch of iterations.

`summary` correlates every pair of columns on the four combinations of linear
and logarithmic axes, and by rank. `detail` gives one pair's points, the mean
of y in bins of x and the ticks, all in the coordinates of the axes asked for,
so a page draws them without transforming anything itself. The server and the
browser's worker both answer with these, so the two never disagree.

Not part of the engine, and free of the store, so the browser can run it.
"""
from __future__ import annotations

import math
from collections import Counter, deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

import gol_series
from gol_stats import finite

#: name: (what it is, kind). The kind decides the logarithmic axis: "pos" is
#: positive, and its log is the logarithm; "count" can be zero, and its log is
#: log(1 + x); "signed" takes either sign, and its log is sign(x) log(1 + |x|);
#: "share" is a fraction and "bool" a yes or no, which keep the linear axis.
VARIABLES: Dict[str, Tuple[str, str]] = {
    # the agent
    "tokens": ("tokens it held at the start of the game", "pos"),
    "age": ("iterations since it was born", "count"),
    "clones": ("agents carrying its brain, itself included", "pos"),
    # its neighbourhood
    "degree": ("its connections", "pos"),
    "curvature": ("token curvature: the sum over its neighbours of their tokens minus its own", "signed"),
    "nbr_tokens": ("its neighbours' mean tokens", "pos"),
    "nbr_max": ("its richest neighbour's tokens", "pos"),
    "poorer_nbrs": ("the share of its neighbours holding fewer tokens than it", "share"),
    "nbr_degree": ("its neighbours' mean connections", "pos"),
    "kin_nbrs": ("the share of its neighbours carrying its brain", "share"),
    "tokens_per_link": ("its tokens divided by its connections", "pos"),
    # its place in the network
    "clustering": ("the share of its neighbours' pairs that are linked", "share"),
    "triangles": ("triangles it belongs to", "count"),
    "core": ("its k-core number", "pos"),
    "forman": ("Forman-Ricci curvature of the network at it, the mean over its links", "signed"),
    "pagerank": ("PageRank", "pos"),
    "betweenness": ("betweenness, estimated from 64 sampled sources", "count"),
    # what it did in the game
    "home_share": ("the share of its tokens staked on its own node", "share"),
    "targets": ("neighbours it staked on", "count"),
    "max_share": ("its largest stake on a neighbour, as a share of its tokens", "share"),
    "rev_share": ("the share of its tokens marked revolutionary", "share"),
    # what happened
    "change": ("its node's change of tokens over the game", "signed"),
    "rel_change": ("that change divided by the tokens it held", "signed"),
    "kept": ("it kept its own node", "bool"),
    "took": ("nodes it took from others in the game", "count"),
    "alive_10": ("its node still exists ten iterations later", "bool"),
    "change_10": ("its node's change of tokens over ten iterations", "signed"),
    "degree_change_10": ("its node's change of connections over ten iterations", "signed"),
    "child_next": ("it had a child in the next reproduction phase", "bool"),
}
COLUMNS = list(VARIABLES)
#: The columns read from a game's decisions, missing when a run did not record them.
FROM_DECISIONS = ("home_share", "targets", "max_share", "rev_share", "kept", "took")
#: Pairs whose correlation says how the quantities are made, not how the world
#: behaves: one is computed from the other (curvature contains minus degree times
#: tokens; a change over the game contains minus the tokens held), or both from
#: the same numbers.
DEFINITIONAL = {frozenset(p) for p in [
    *[("curvature", x) for x in ("tokens", "degree", "nbr_tokens", "nbr_max", "poorer_nbrs", "tokens_per_link")],
    *[("poorer_nbrs", x) for x in ("tokens", "nbr_tokens", "nbr_max", "tokens_per_link")],
    ("tokens_per_link", "tokens"), ("tokens_per_link", "degree"), ("nbr_tokens", "nbr_max"),
    *[("forman", x) for x in ("degree", "nbr_degree", "triangles", "clustering")],
    ("clustering", "triangles"), ("triangles", "degree"), ("clustering", "degree"), ("core", "degree"),
    ("pagerank", "degree"),
    ("home_share", "targets"), ("home_share", "max_share"), ("targets", "max_share"),
    ("change", "rel_change"), ("rel_change", "tokens"), ("change", "tokens"), ("change_10", "tokens"),
    ("change_10", "alive_10"), ("degree_change_10", "alive_10"), ("degree_change_10", "degree"),
]}

#: What the agents table reads of a frame, so a caller can hand it no more.
FIELDS = ("iteration", "phase", "ids", "tokens", "brain_ids", "parent_ids", "ages",
          "edges", "delta", "decisions")
#: How far ahead the outcomes look, in iterations.
HORIZON = 10
#: Sources the betweenness is estimated from.
SOURCES = 64
#: Rows a summary correlates at most, drawn evenly from a larger table: the
#: error of a correlation over twenty thousand rows is under 0.01.
SUMMARY_ROWS = 20000
#: Points a scatter is drawn with at most.
POINTS = 4000
#: The fewest rows a correlation is reported for.
MIN_ROWS = 30
#: Not statistics of the world: bookkeeping, or the same in every frame.
NOT_STATISTICS = {"iteration", "phase", "tokens", "nodes_before", "cladesInWindow"}

LOG_KINDS = ("pos", "count", "signed")


#: World statistics made from the same numbers, so that any two in one group
#: move together by how they are computed: the tokens per agent and the
#: agents; the links, the mean degree and the density; flow and what was kept
#: at home; four measures of one inequality; births and what they cost;
#: gainers and losers among a population; and so on.
WORLD_GROUPS = [
    {"meanTokens", "nodes", "density"},
    {"edges", "meanDegree", "density", "nodes"},
    {"totalFlow", "selfAllocationShare", "meanEdgeFlow", "maxEdgeFlow"},
    {"gini", "tokenEvenness", "tokenEntropy", "topDecileShare"},
    {"degreeEntropy", "degreeEvenness"},
    {"births", "reproTokenShare", "meanInvestedShare", "distinctParents"},
    {"gainers", "losers", "nodes"},
    {"brainDiversity", "distinctBrains", "nodes"},
    {"cutRiskBefore", "minDegree", "leaves"},
    {"orphaned", "redistributed", "starved"},
]


#: World statistics that count things, and so grow with the size of a world.
#: Across worlds of different sizes any two of them go together for that reason
#: alone, so in the table of every world a pair of them counts as made from the
#: same numbers.
EXTENSIVE = {"nodes", "edges", "totalFlow", "maxEdgeFlow", "giftTokens", "redistributed", "prunedEdges",
             "maxDegree", "maxTokens", "maxTokenAdded", "maxTokenLost", "triangles", "bridges",
             "components", "distinctBrains", "distinctParents", "cycleRank", "diameter", "radius",
             "lightningLongest", "lightningScore", "netLightningLongest", "netLightningScore",
             "meanPathLength", "cutRisk", "cutRiskBefore", "tokens", "births", "gainers", "losers",
             "starved", "orphaned", "revolutions", "leaves", "handovers", "gifts", "genotypes", "roots",
             "conquests", "kin_conquests", "deaths"}


def definitional(a: str, b: str) -> bool:
    """Whether two quantities are related by how they are made rather than by how the world behaves."""
    return frozenset((a, b)) in DEFINITIONAL or any(a in g and b in g for g in WORLD_GROUPS)


# ---------------------------------------------------------------------------
# A table
# ---------------------------------------------------------------------------

class Table:
    """Rows and named columns, each with a kind and a line saying what it is."""

    def __init__(self, level: str, columns: Sequence[str], kinds: Dict[str, str],
                 values: np.ndarray, about: Optional[Dict[str, str]] = None,
                 labels: Optional[List[str]] = None, groups: Optional[List[str]] = None,
                 note: str = "") -> None:
        self.level = level
        self.columns = list(columns)
        self.kinds = kinds
        self.values = np.asarray(values, dtype=float).reshape(-1, len(self.columns))
        self.about = about or {}
        self.labels = labels
        self.groups = groups
        self.note = note
        #: For a table of every world, the stretches of iterations it can be taken over.
        self.periods: List[str] = []
        #: Whether its rows are worlds of different sizes, where two counts go together by size alone.
        self.sized = False

    def made_alike(self, a: str, b: str) -> bool:
        """Whether two columns are related by how they are made, in this table."""
        return definitional(a, b) or (self.sized and a in EXTENSIVE and b in EXTENSIVE)

    @property
    def rows(self) -> int:
        return int(self.values.shape[0])

    def column(self, name: str) -> np.ndarray:
        return self.values[:, self.columns.index(name)]


def kind_of(values: np.ndarray) -> str:
    """What a column's logarithmic axis is, read off its values."""
    v = values[np.isfinite(values)]
    if not len(v):
        return "share"
    integral = bool(np.all(v == np.round(v)))
    if v.min() >= 0 and v.max() <= 1 and not integral:
        return "share"
    if v.min() > 0:
        return "pos"
    if v.min() >= 0:
        return "count"
    return "signed"


# ---------------------------------------------------------------------------
# The network measures the agents table needs that networkx would need scipy
# for, or answers differently from one version to the next
# ---------------------------------------------------------------------------

def forman(G: nx.Graph) -> Dict[int, float]:
    """Forman-Ricci curvature of every link, 4 - deg(u) - deg(v) + 3 triangles(u, v), averaged at each node."""
    deg = dict(G.degree())
    total, count = Counter(), Counter()
    adj = {u: set(G[u]) for u in G}
    for u, v in G.edges():
        f = 4 - deg[u] - deg[v] + 3 * len(adj[u] & adj[v])
        total[u] += f
        total[v] += f
        count[u] += 1
        count[v] += 1
    return {u: total[u] / count[u] for u in count}


def pagerank(G: nx.Graph, damping: float = 0.85, tol: float = 1e-10, iters: int = 500) -> Dict[int, float]:
    """PageRank by power iteration (networkx's needs scipy): the share of a random surfer's time at each node."""
    nodes = list(G)
    index = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)
    if not n:
        return {}
    src = np.array([index[u] for u, v in G.edges()] + [index[v] for u, v in G.edges()], dtype=int)
    dst = np.array([index[v] for u, v in G.edges()] + [index[u] for u, v in G.edges()], dtype=int)
    deg = np.bincount(src, minlength=n).astype(float)
    p = np.full(n, 1.0 / n)
    for _ in range(iters):
        share = np.where(deg > 0, p / np.maximum(deg, 1), 0.0)
        new = np.bincount(dst, weights=share[src], minlength=n)
        new = damping * (new + p[deg == 0].sum() / n) + (1 - damping) / n
        if np.abs(new - p).sum() < tol:
            p = new
            break
        p = new
    return {u: float(p[index[u]]) for u in nodes}


def betweenness(G: nx.Graph, sources: int = SOURCES, seed: int = 1) -> Dict[int, float]:
    """
    Betweenness estimated from a sample of sources (Brandes): for every node,
    the shortest paths from the sampled sources that pass through it, scaled
    up to all sources. Written here rather than taken from networkx, whose
    sampled estimate is scaled differently from one version to the next, so
    the browser's copy and the server's would not agree.
    """
    nodes = sorted(G)
    n = len(nodes)
    out = {u: 0.0 for u in nodes}
    if n < 3:
        return out
    k = min(sources, n)
    picked = [nodes[i] for i in np.random.RandomState(seed).choice(n, size=k, replace=False)]
    adj = {u: list(G[u]) for u in nodes}
    for s in picked:
        order, preds = [], {s: []}
        sigma, dist = {s: 1.0}, {s: 0}
        queue = deque([s])
        while queue:
            v = queue.popleft()
            order.append(v)
            for w in adj[v]:
                if w not in dist:
                    dist[w] = dist[v] + 1
                    sigma[w] = 0.0
                    preds[w] = []
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    preds[w].append(v)
        delta = dict.fromkeys(order, 0.0)
        for w in reversed(order):
            for v in preds[w]:
                delta[v] += sigma[v] / sigma[w] * (1.0 + delta[w])
            if w != s:
                out[w] += delta[w]
    # Every path is found from both of its ends; and the sample stands for all.
    scale = n / k / 2.0
    return {u: b * scale for u, b in out.items()}


# ---------------------------------------------------------------------------
# The agents of a game
# ---------------------------------------------------------------------------

def rows(repro: Dict[str, Any], game: Dict[str, Any], nxt: Optional[Dict[str, Any]],
         later: Optional[Dict[str, Any]]) -> List[Dict[str, float]]:
    """
    One row for every agent at the start of a game: `repro` is the frame after
    the iteration's reproduction, which is the world the game starts from;
    `game` the frame after the game; `nxt` the next iteration's reproduction
    frame, for who had a child; `later` the game frame HORIZON iterations on.

    The agent of a row is the agent at that node when the game began. A node
    keeps its id when it is taken, so what is measured after the game is of the
    node, held by whoever holds it then; `kept` says whether that was still the
    same agent. Without the game's decisions the six columns read from them
    are missing (NaN), and the rest stay.
    """
    ids = repro.get("ids") or []
    if not ids:
        return []
    tokens = dict(zip(ids, repro["tokens"]))
    brain = dict(zip(ids, repro.get("brain_ids") or ids))
    ages = repro.get("ages")
    age = dict(zip(ids, ages)) if ages else {}
    G = nx.Graph()
    G.add_nodes_from(ids)
    G.add_edges_from((a, b) for a, b in repro.get("edges") or [] if a in tokens and b in tokens)
    deg = dict(G.degree())
    clones = Counter(brain.values())
    clust = nx.clustering(G)
    tri = nx.triangles(G)
    core = nx.core_number(G)
    form = forman(G)
    pr = pagerank(G)
    btw = betweenness(G)

    after = dict(zip(game.get("ids") or [], game.get("tokens") or []))
    if game.get("delta"):
        delta = dict(zip(game["ids"], game["delta"]))
    else:
        delta = {a: after[a] - tokens[a] for a in after if a in tokens}
    dec = game.get("decisions") or {}
    decided = bool(dec.get("allocations"))
    won = {w["node"]: w["winner"] for w in dec.get("winners") or []}
    took = Counter(w for v, w in won.items() if w != v)
    alloc = {r["agent"]: r for r in dec.get("allocations") or []}

    parents = None
    if nxt:
        parents = {p for p, a in zip(nxt.get("parent_ids") or [], nxt.get("ages") or []) if a == 0}
    later_tokens = dict(zip(later["ids"], later["tokens"])) if later else {}
    later_deg: Counter = Counter()
    for a, b in (later.get("edges") or []) if later else []:
        later_deg[a] += 1
        later_deg[b] += 1

    nan = float("nan")
    out = []
    for a in ids:
        x = tokens[a]
        nb = list(G[a])
        nt = [tokens[b] for b in nb]
        r = alloc.get(a)
        spent = r["tokens"] if r else 0
        stakes = r["alloc"][1:] if r else []
        revolt = r.get("revolt") if r else None
        row = {
            "tokens": x, "age": age.get(a, nan), "clones": clones[brain[a]],
            "degree": deg[a],
            "curvature": sum(nt) - deg[a] * x if nb else nan,
            "nbr_tokens": float(np.mean(nt)) if nb else nan,
            "nbr_max": max(nt) if nb else nan,
            "poorer_nbrs": float(np.mean([y < x for y in nt])) if nb else nan,
            "nbr_degree": float(np.mean([deg[b] for b in nb])) if nb else nan,
            "kin_nbrs": float(np.mean([brain[b] == brain[a] for b in nb])) if nb else nan,
            "tokens_per_link": x / deg[a] if deg[a] else nan,
            "clustering": clust[a], "triangles": tri[a], "core": core[a],
            "forman": form.get(a, nan), "pagerank": pr[a], "betweenness": btw[a],
            "home_share": (r["alloc"][0] / spent) if r and spent else nan,
            "targets": sum(1 for s in stakes if s > 0) if r else nan,
            "max_share": ((max(stakes) / spent) if stakes else 0.0) if r and spent else nan,
            "rev_share": (sum(revolt) / spent) if revolt is not None and spent else nan,
            "change": delta.get(a, nan),
            "rel_change": (delta[a] / x) if a in delta and x else nan,
            "kept": float(won.get(a) == a) if a in won else nan,
            "took": took.get(a, 0) if decided else nan,
            "alive_10": float(a in later_tokens) if later else nan,
            "change_10": (later_tokens[a] - x) if a in later_tokens else nan,
            "degree_change_10": (later_deg[a] - deg[a]) if a in later_tokens else nan,
            "child_next": float(a in parents) if parents is not None else nan,
        }
        out.append(row)
    return out


def agent_table(frames: Iterable[Dict[str, Any]], first: int, last: int, step: int = 1,
                horizon: int = HORIZON) -> Table:
    """
    The agents of every `step`-th game whose iteration is from `first` to `last`,
    from frames that reach `horizon` iterations further where they can.
    """
    at = {(f.get("iteration"), f.get("phase")): f for f in frames}
    games = sorted(t for (t, p) in at if p == 2 and first <= t <= last and (t - first) % max(1, step) == 0)
    table, without = [], 0
    for t in games:
        repro, game = at.get((t, 1)), at[(t, 2)]
        if repro is None:
            continue
        if not (game.get("decisions") or {}).get("allocations"):
            without += 1
        for row in rows(repro, game, at.get((t + 1, 1)), at.get((t + horizon, 2))):
            table.append([row[c] for c in COLUMNS])
    note = f"{len(games)} games"
    if without:
        note += (f"; {without} of them without the agents' decisions recorded, so what they staked, "
                 f"whether they kept their node and what they took are missing there")
    values = np.array(table, dtype=float) if table else np.zeros((0, len(COLUMNS)))
    return Table("agents", COLUMNS, {c: VARIABLES[c][1] for c in COLUMNS}, values,
                 about={c: VARIABLES[c][0] for c in COLUMNS}, note=note)


def lookahead() -> int:
    """Frames read past a window so its last games have their outcomes: the next reproduction and HORIZON games on."""
    return 2 * HORIZON + 2


def window_agents(frames: List[Dict[str, Any]], count: int, step: int = 1) -> Table:
    """
    The agents table of a window of `count` frames, read with lookahead() more
    after it: every `step`-th game whose game frame lies inside the window.
    """
    inside = [f for f in frames[:count] if f.get("phase") == 2 and f.get("iteration") is not None]
    if not inside:
        return agent_table([], 0, -1)
    return agent_table(frames, int(inside[0]["iteration"]), int(inside[-1]["iteration"]), step)


# ---------------------------------------------------------------------------
# The world's statistics, game by game
# ---------------------------------------------------------------------------

def is_statistic(name: str, value: Any) -> bool:
    """
    Whether a recorded row's entry is a statistic of the world that every
    frame has: not bookkeeping, and not one of the graph statistics a run
    records only every so many iterations, which would leave most rows empty
    and differ between a run's record and a frame worked out afresh.
    """
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and not name.startswith("_") and name not in NOT_STATISTICS
            and name not in gol_series.HEAVY_KEYS)


def stat_table(frame_rows: Iterable[Dict[str, Any]], changes: bool = False) -> Table:
    """
    One row per game: the statistics of the iteration's reproduction frame
    (births, gifts, what a child was given) and of its game frame, the game's
    where both have a value. With `changes`, each row is the change from the
    game before.
    """
    by_iteration: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for r in frame_rows:
        if r and r.get("iteration") is not None:
            by_iteration.setdefault(int(r["iteration"]), {})[int(r.get("phase", 0))] = r
    merged, its = [], []
    for t in sorted(by_iteration):
        phases = by_iteration[t]
        if 2 not in phases:
            continue
        row = {k: v for k, v in (phases.get(1) or {}).items() if is_statistic(k, v)}
        row.update({k: v for k, v in phases[2].items() if is_statistic(k, v)})
        merged.append(row)
        its.append(t)
    names = sorted({k for r in merged for k in r})
    values = np.array([[r.get(k, np.nan) for k in names] for r in merged], dtype=float) \
        if merged else np.zeros((0, len(names)))
    level = "changes" if changes else "stats"
    note = f"{len(merged)} games"
    if changes:
        values = values[1:] - values[:-1] if len(values) > 1 else np.zeros((0, len(names)))
        note = f"{len(values)} changes from one game to the next"
    keep = [j for j in range(len(names)) if np.isfinite(values[:, j]).sum() >= MIN_ROWS
            and np.nanstd(values[:, j]) > 0] if len(values) else []
    names = [names[j] for j in keep]
    values = values[:, keep] if len(values) else np.zeros((0, len(names)))
    return Table(level, names, {n: kind_of(values[:, j]) for j, n in enumerate(names)}, values,
                 labels=[str(t) for t in (its[1:] if changes else its)], note=note)


# ---------------------------------------------------------------------------
# Every world
# ---------------------------------------------------------------------------

def world_table(worlds: Dict[str, Any], window: str) -> Table:
    """
    One row per simulation of the table research/atlas/across.py builds: its
    statistics over the stretch `window` names, grouped by its baseline and
    whether it is a lab run or a pilot, and labelled with its id and the
    experiments it belongs to.
    """
    entries = [w for w in worlds.get("worlds", []) if (w.get("at") or {}).get(window)]
    names = sorted({k for w in entries for k, v in w["at"][window].items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)})
    values = np.array([[w["at"][window].get(k, np.nan) if w["at"][window].get(k) is not None else np.nan
                        for k in names] for w in entries], dtype=float) if entries else np.zeros((0, len(names)))
    keep = [j for j in range(len(names)) if len(values) and np.isfinite(values[:, j]).sum() >= MIN_ROWS
            and np.nanstd(values[:, j]) > 0]
    names = [names[j] for j in keep]
    values = values[:, keep] if len(values) else np.zeros((0, len(names)))
    table = Table("worlds", names, {n: kind_of(values[:, j]) for j, n in enumerate(names)}, values,
                  about=dict(worlds.get("explanations") or {}),
                  labels=[w.get("id", "") + (f" — {'; '.join(w['conditions'])}" if w.get("conditions") else "")
                          for w in entries],
                  groups=[f"{w.get('baseline') or '?'} {w.get('source', '')}".strip() for w in entries],
                  note=f"{len(entries)} worlds, {window}")
    table.periods = list(worlds.get("windows") or [])
    table.sized = True
    return table


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def axis(values: np.ndarray, kind: str, log: bool) -> Tuple[np.ndarray, np.ndarray]:
    """A column on its linear or logarithmic axis, and which of its values it can show."""
    if not log or kind not in LOG_KINDS:
        return values, np.isfinite(values)
    if kind == "pos":
        ok = np.isfinite(values) & (values > 0)
        return np.log(np.where(ok, values, 1.0)), ok
    if kind == "count":
        ok = np.isfinite(values) & (values >= 0)
        return np.log1p(np.where(ok, values, 0.0)), ok
    ok = np.isfinite(values)
    v = np.where(ok, values, 0.0)
    return np.sign(v) * np.log1p(np.abs(v)), ok


def rank(v: np.ndarray) -> np.ndarray:
    """Ranks from 0, ties sharing the mean of the ranks they span."""
    _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    stop = np.cumsum(counts)
    return ((stop - counts + stop - 1) / 2.0)[inv]


def pearson(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) < MIN_ROWS:
        return None
    # Sums of products rather than `@`: a dot product goes to the BLAS, whose
    # threads, on a machine busy with other work, made each of these tens of
    # milliseconds instead of tens of microseconds.
    xm, ym = x - x.mean(), y - y.mean()
    sxx, syy = float((xm * xm).sum()), float((ym * ym).sum())
    if sxx <= 0 or syy <= 0:
        return None
    return float(np.clip((xm * ym).sum() / math.sqrt(sxx * syy), -1.0, 1.0))


def _axes_of(kind: str) -> Tuple[bool, ...]:
    return (False, True) if kind in LOG_KINDS else (False,)


def _correlate(va: np.ndarray, ka: str, vb: np.ndarray, kb: str,
               ranks: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
    """
    Pearson's r on every combination of the two columns' axes, and Spearman's
    rho — from `ranks`, the columns' ranks worked out once, when neither has a
    missing value; otherwise ranked again over the rows both have.
    """
    out: Dict[str, Any] = {}
    for la in _axes_of(ka):
        xa, oka = axis(va, ka, la)
        for lb in _axes_of(kb):
            xb, okb = axis(vb, kb, lb)
            ok = oka & okb
            out[f"{'log' if la else 'lin'}-{'log' if lb else 'lin'}"] = pearson(xa[ok], xb[ok])
    ok = np.isfinite(va) & np.isfinite(vb)
    n = int(ok.sum())
    if ranks is not None and ok.all():
        out["rank"] = pearson(ranks[0], ranks[1]) if n >= MIN_ROWS else None
    else:
        out["rank"] = pearson(rank(va[ok]), rank(vb[ok])) if n >= MIN_ROWS else None
    out["n"] = n
    return out


def pair(table: np.ndarray, cols: Dict[str, int], a: str, b: str) -> Dict[str, Any]:
    """Pearson's r on the four combinations of linear and logarithmic axes, and Spearman's rho, of two VARIABLES."""
    return _correlate(table[:, cols[a]], VARIABLES[a][1], table[:, cols[b]], VARIABLES[b][1])


def _sample(n: int, m: int) -> np.ndarray:
    """m of n rows, the same ones every time."""
    if n <= m:
        return np.arange(n)
    return np.sort(np.random.default_rng(0).choice(n, size=m, replace=False))


def summary(table: Table) -> Dict[str, Any]:
    """
    Every pair of columns: r on linear axes, the combination of axes on which
    |r| is largest and r there, Spearman's rho, how many rows, and whether
    the pair is related by how it is made.
    """
    values = table.values[_sample(table.rows, SUMMARY_ROWS)]
    k = len(table.columns)
    finite_all = [bool(np.isfinite(values[:, j]).all()) for j in range(k)]
    ranks = [rank(values[:, j]) if finite_all[j] and len(values) else None for j in range(k)]
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            a, b = table.columns[i], table.columns[j]
            ranked = ranks[i] is not None and ranks[j] is not None
            got = _correlate(values[:, i], table.kinds[a], values[:, j], table.kinds[b],
                             (ranks[i], ranks[j]) if ranked else None)
            axes = [key for key in got if key not in ("rank", "n") and got[key] is not None]
            if not axes and got["rank"] is None:
                continue
            best = max(axes, key=lambda key: abs(got[key])) if axes else None
            pairs.append([i, j, got.get("lin-lin"), best, got.get(best) if best else None,
                          got["rank"], got["n"], table.made_alike(a, b)])
    ranked = sorted((p for p in pairs if not p[7] and p[4] is not None), key=lambda p: -abs(p[4]))
    return finite({
        "level": table.level,
        "rows": table.rows,
        "note": table.note,
        "variables": [{"name": c, "kind": table.kinds[c], "about": table.about.get(c, ""),
                       "finite": int(np.isfinite(table.values[:, j]).sum())}
                      for j, c in enumerate(table.columns)],
        "pairs": pairs,
        "strongest": [[p[0], p[1]] for p in ranked[:24]],
        "periods": table.periods,
    })


# ---------------------------------------------------------------------------
# One pair
# ---------------------------------------------------------------------------

def _label(v: float) -> str:
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 1 and v == round(v):
        return str(int(round(v)))
    return f"{v:.2g}"


def _nice(lo: float, hi: float, target: int = 6) -> List[float]:
    """Round numbers to rule an axis at: steps of 1, 2, 2.5 or 5 times a power of ten."""
    if not hi > lo:
        return [lo]
    rough = (hi - lo) / target
    mag = 10 ** math.floor(math.log10(rough))
    s = rough / mag
    step = mag * (1 if s <= 1 else 2 if s <= 2 else 2.5 if s <= 2.5 else 5 if s <= 5 else 10)
    out, v = [], math.ceil(lo / step) * step
    while v <= hi + step * 1e-9:
        out.append(0.0 if abs(v) < step * 1e-9 else v)
        v += step
    return out


def _decades(lo: float, hi: float) -> List[float]:
    """Round values from lo to hi (both positive) for a logarithmic axis: 1, 2, 5 or 1 a decade."""
    if not (hi > 0 and lo > 0) or hi < lo:
        return []
    first, last = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
    span = last - first
    steps = (1, 2, 5) if span <= 2 else (1,)
    every = 1 if span <= 8 else 2
    out = [m * 10.0 ** e for e in range(first, last + 1) if (e - first) % every == 0 for m in steps]
    return [v for v in out if lo * (1 - 1e-9) <= v <= hi * (1 + 1e-9)]


def ticks(lo: float, hi: float, kind: str, log: bool) -> List[List[Any]]:
    """Where to rule an axis and what to write there: positions on the axis, labels in the column's units."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return []
    if not log or kind not in LOG_KINDS:
        return [[v, _label(v)] for v in _nice(lo, hi)]
    if kind == "pos":
        values = _decades(math.exp(lo), math.exp(hi))
        if len(values) < 2:
            values = [v for v in _nice(math.exp(lo), math.exp(hi)) if v > 0]
        return [[math.log(v), _label(v)] for v in values]
    if kind == "count":
        top = math.expm1(hi)
        values = ([0.0] if lo <= 0 else []) + _decades(max(1.0, math.expm1(lo)), top)
        if len(values) < 2:
            values = [v for v in _nice(math.expm1(lo), top) if v >= 0]
        return [[math.log1p(v), _label(v)] for v in values]
    # signed: sign(x) log(1 + |x|), symmetric about zero. Below one the
    # logarithm is nearly linear, so fractions get ticks of their own there.
    def side(top: float) -> List[float]:
        if top <= 0:
            return []
        out = _decades(min(1.0, top), top) if top >= 1 else []
        return sorted(set([v for v in (0.1, 0.2, 0.5) if v <= top * (1 + 1e-9)] if len(out) < 3 else []) | set(out))
    values = ([-v for v in reversed(side(math.expm1(-lo)))] + ([0.0] if lo <= 0 <= hi else [])
              + side(math.expm1(hi)))
    return [[math.copysign(math.log1p(abs(v)), v), _label(v)] for v in values]


def bins(x: np.ndarray, y: np.ndarray) -> List[Dict[str, float]]:
    """
    The mean of y, and its quartiles, in bins of x: one bin per value when x
    takes few, otherwise bins of equal counts, as many as the Rice rule gives
    (twice the cube root of the number of points).
    """
    n = len(x)
    if not n:
        return []
    k = max(1, math.ceil(2 * n ** (1 / 3)))
    distinct = np.unique(x)
    if len(distinct) <= k:
        group = np.searchsorted(distinct, x)
        groups = len(distinct)
    else:
        edges = np.unique(np.quantile(x, np.linspace(0, 1, k + 1)))
        group = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
        groups = len(edges) - 1
    order = np.argsort(group, kind="stable")
    bounds = np.searchsorted(group[order], np.arange(groups + 1))
    out = []
    for g in range(groups):
        sel = order[bounds[g]:bounds[g + 1]]
        if not len(sel):
            continue
        ys = y[sel]
        q25, q75 = np.quantile(ys, [0.25, 0.75])
        out.append({"x": float(np.median(x[sel])), "mean": float(ys.mean()),
                    "q25": float(q25), "q75": float(q75), "n": int(len(sel))})
    return out


def detail(table: Table, a: str, b: str, xlog: bool = False, ylog: bool = False) -> Dict[str, Any]:
    """
    One pair, on the axes asked for: up to POINTS of its points, the binned
    means, the ticks of both axes, r on these axes and by rank, and how many
    rows there are. Coordinates are on the axes, so nothing is transformed again
    where they are drawn.
    """
    if a not in table.columns or b not in table.columns:
        raise ValueError(f"no column {a if a not in table.columns else b!r} in this table")
    ka, kb = table.kinds[a], table.kinds[b]
    xlog, ylog = bool(xlog) and ka in LOG_KINDS, bool(ylog) and kb in LOG_KINDS
    va, vb = table.column(a), table.column(b)
    x, okx = axis(va, ka, xlog)
    y, oky = axis(vb, kb, ylog)
    ok = okx & oky
    rows_ok = np.flatnonzero(ok)
    both = np.isfinite(va) & np.isfinite(vb)
    shown = rows_ok[_sample(len(rows_ok), POINTS)]
    xs, ys = x[ok], y[ok]
    reply: Dict[str, Any] = {
        "a": a, "b": b, "xlog": xlog, "ylog": ylog, "n": int(ok.sum()),
        "r": pearson(xs, ys),
        "lin": pearson(va[both], vb[both]),
        "rank": pearson(rank(va[both]), rank(vb[both])) if both.sum() >= MIN_ROWS else None,
        "definitional": table.made_alike(a, b),
        "points": {"x": x[shown].tolist(), "y": y[shown].tolist()},
        "bins": bins(xs, ys),
        "range": {"x": [float(xs.min()), float(xs.max())] if len(xs) else None,
                  "y": [float(ys.min()), float(ys.max())] if len(ys) else None},
    }
    if table.labels is not None:
        reply["points"]["label"] = [table.labels[i] for i in shown]
    if table.groups is not None:
        reply["points"]["group"] = [table.groups[i] for i in shown]
    if len(xs):
        reply["ticks"] = {"x": ticks(float(xs.min()), float(xs.max()), ka, xlog),
                          "y": ticks(float(ys.min()), float(ys.max()), kb, ylog)}
    return finite(reply)


def answer(table: Table, a: Optional[str] = None, b: Optional[str] = None,
           xlog: bool = False, ylog: bool = False) -> Dict[str, Any]:
    """The summary of a table, or one pair of it when a pair is named."""
    return detail(table, a, b, xlog, ylog) if a and b else summary(table)
