"""
What the screen's judge measures on a world from round 3 on (the author's
walkthrough of 2026-10-24): every number an average over iterations 100 to the
end of the world's run, read from what the world recorded. No world is played.

Each number is read where the run keeps it, as often as it is kept:
- every iteration, from the statistics the run recorded (stats.jsonl): how
  many agents there are, links per agent, the hub (the most links any agent
  has, over the others it could have), the largest holding, the Gini of the
  tokens, the share of agents that had a child, and the share that spread
  their stake rather than staking it all on one place;
- every 25th iteration, where the run recorded its heavy statistics: the
  dimension (those statistics'), and from the game's frame how alike the agents
  are in tokens, age and genotype, the 2-core and how alike its agents' numbers
  of links are, and the most one agent's or one link's loss cuts off;
- every kept world (every 250 iterations): the typical distance over the
  random twin's, the half-cut exponent of the world and of its twin, and the
  same-position share with the effective kinds of staking
  (research/oee/checks/diversity.py, with no game played).

"How alike" is the share of agents that share the most common value: 1 when
all are the same. For a yes-or-no choice it is the share making the more
common choice, at each iteration.

A world's readings are kept as they are made, one file a world in READINGS:
a reading is never made twice, and one that fails loses no other. A reading is
an item: the world's series to the iteration it ended at, or one kept world.
"""
from __future__ import annotations

import functools
import importlib.util
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import gol_record  # noqa: E402
import gol_store  # noqa: E402

#: A world is read in items: (run, "series", the iteration it ended at) and (run, "kept", an iteration kept).
Item = Tuple[str, str, int]

#: Fewer agents than this: no width is read (as research/atlas/space.py), and the world cannot count as a space.
WIDE_AT_LEAST = 100
#: Fewer agents than this: no kinds of staking are read (as the judge of rounds 1 and 2).
SMALL = 30
#: The seed of the random twin and of the walks, as everywhere the notebook reads a space.
SEED = 23
#: Lanczos steps. On round 3's worlds (3,000 to 6,000 agents) 400 give the ten smoothest ways to vary as
#: an exact eigensolver does, and 200 do not (2026-10-25); the walk ends sooner when it has found everything.
STEPS = 400
#: The smoothest ways to vary over a world whose median cuts are tried: the cheapest is its half cut. One
#: alone is not enough: their eigenvalues lie close together, and in a world of round 3 the median of the
#: smoothest cut 121 links where that of the third cut 62.
MODES = 10


@functools.lru_cache(maxsize=None)
def _module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def readings_dir() -> str:
    """Where each world's readings are kept: read from the store when asked, as tests move the store."""
    return os.path.join(gol_store.BASE_DIR, ".pilots", "screen", "worlds")


# ---------------------------------------------------------------------------
# What a number is made of
# ---------------------------------------------------------------------------

def alike(values) -> Optional[float]:
    """The share of the values equal to the most common one: 1 when all are the same."""
    _, counts = np.unique(np.asarray(values), return_counts=True)
    return float(counts.max() / counts.sum()) if counts.size else None


def larger_side(share: Optional[float]) -> Optional[float]:
    """For a yes-or-no choice made by a share of the agents: the share making the more common choice."""
    return None if share is None else max(share, 1.0 - share)


def pairs_of(n: int, pairs) -> np.ndarray:
    """Links as pairs of places in the list of agents, each once, none from an agent to itself."""
    p = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    p = p[(p[:, 0] != p[:, 1]) & (p >= 0).all(axis=1) & (p < n).all(axis=1)]
    return np.unique(np.sort(p, axis=1), axis=0)


def largest_piece(n: int, pairs: np.ndarray) -> np.ndarray:
    """Which agents belong to the largest piece of the network."""
    parent = np.arange(n)

    def root(u: int) -> int:
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    for a, b in pairs:
        ra, rb = root(int(a)), root(int(b))
        if ra != rb:
            parent[ra] = rb
    roots = np.array([root(u) for u in range(n)])
    values, counts = np.unique(roots, return_counts=True)
    return roots == values[np.argmax(counts)]


def smooth_modes(n: int, pairs: np.ndarray, seed: int = SEED) -> np.ndarray:
    """
    The MODES smoothest ways to vary over a network in one piece, short of being
    constant, one a column: the eigenvectors of its normalised Laplacian with the
    smallest eigenvalues after the first (the first of them is the Fiedler
    vector), by Lanczos with every vector kept orthogonal to the others and to
    the constant one (there is no scipy here), each returned as the random
    walk's (divided by the square root of each agent's links).
    """
    a, b = pairs[:, 0], pairs[:, 1]
    degree = np.bincount(pairs.ravel(), minlength=n).astype(float)
    inv = 1.0 / np.sqrt(degree)

    def walk(x: np.ndarray) -> np.ndarray:
        """x times D^-1/2 A D^-1/2: the matrix whose largest eigenvalues are the Laplacian's smallest."""
        y = x * inv
        return (np.bincount(a, weights=y[b], minlength=n) + np.bincount(b, weights=y[a], minlength=n)) * inv

    constant = np.sqrt(degree) / np.linalg.norm(np.sqrt(degree))
    q = np.random.RandomState(seed).standard_normal(n)
    q -= constant * (constant @ q)
    q /= np.linalg.norm(q)
    steps = min(STEPS, n - 1)
    Q = np.zeros((steps, n))
    alpha, beta = np.zeros(steps), np.zeros(steps)
    for j in range(steps):
        Q[j] = q
        w = walk(q)
        alpha[j] = q @ w
        for _ in range(2):          # twice is enough to keep the vectors orthogonal
            w -= Q[:j + 1].T @ (Q[:j + 1] @ w)
            w -= constant * (constant @ w)
        beta[j] = np.linalg.norm(w)
        if beta[j] < 1e-10:
            steps = j + 1
            break
        q = w / beta[j]
    T = np.diag(alpha[:steps]) + np.diag(beta[:steps - 1], 1) + np.diag(beta[:steps - 1], -1)
    _, vectors = np.linalg.eigh(T)
    smoothest = vectors[:, ::-1][:, :MODES]
    return (Q[:steps].T @ smoothest) * inv[:, None]


def half_cut(n: int, pairs: np.ndarray) -> Dict[str, Optional[float]]:
    """
    The cost of cutting a network in half: its largest piece split at the median
    of each of its smoothest ways to vary, and the links that cross the cheapest
    of these cuts (C) among its agents (N). Not the cheapest cut there is, which
    no one can find in a network this size, but the one this way finds. The
    exponent ln C / ln N is 2/3 for a body of three dimensions (a cube's halves
    meet on a square), near 0 for a bottleneck, near 1 for a folded small world.
    """
    pairs = pairs_of(n, pairs)
    piece = largest_piece(n, pairs)
    N = int(piece.sum())
    if N < 4:
        return {"agents": N, "crossing": None, "exponent": None}
    place = np.cumsum(piece) - 1
    sub = place[pairs[piece[pairs[:, 0]] & piece[pairs[:, 1]]]]
    crossing = None
    for mode in smooth_modes(N, sub).T:
        side = np.zeros(N, dtype=bool)
        side[np.argsort(mode, kind="stable")[:N // 2]] = True
        cut = int((side[sub[:, 0]] != side[sub[:, 1]]).sum())
        crossing = cut if crossing is None else min(crossing, cut)
    return {"agents": N, "crossing": crossing, "exponent": math.log(crossing) / math.log(N) if crossing else None}


# ---------------------------------------------------------------------------
# The three sources
# ---------------------------------------------------------------------------

def frame_row(frame: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """A game's frame: how alike the agents are, the 2-core, how alike its agents' links are, the worst cuts."""
    profile = _module("atlas_profile", os.path.join(ROOT, "research", "atlas", "profile.py"))
    ids = frame["ids"]
    n = len(ids)
    if n < 2:
        return {}
    place = {int(u): i for i, u in enumerate(ids)}
    pairs = pairs_of(n, [(place[int(u)], place[int(v)]) for u, v in frame["edges"]
                         if int(u) in place and int(v) in place])
    degree = np.bincount(pairs.ravel(), minlength=n)
    core = profile.two_core_members(n, pairs)
    cuts = profile.cuts(n, pairs)
    return {"tokens_alike": alike(frame["tokens"]), "ages_alike": alike(frame["ages"]),
            "genotypes_alike": alike(frame["brain_ids"]), "core": float(core.mean()),
            "core_links_alike": alike(degree[core]) if core.any() else None,
            "agent_cut": cuts["agent_cut"], "link_cut": cuts["link_cut"]}


def series(run_id: str, ended_at: int) -> Dict[str, Any]:
    """
    A world's iterations: from its statistics, one row an iteration; from the
    frames of the games whose statistics are heavy, one row each.
    """
    rows: Dict[int, Dict[str, Any]] = {}
    heavy: List[Tuple[int, int]] = []
    for r in gol_record.read_stats(run_id):
        t = int(r["iteration"])
        row = rows.setdefault(t, {"t": t})
        if r["phase"] == 1:
            if r.get("births") is not None and r.get("nodes_before"):
                row["children"] = r["births"] / r["nodes_before"]
                row["children_alike"] = larger_side(row["children"])
        elif r["phase"] == 2:
            n = int(r["nodes"])
            row.update({"agents": n, "links": r["meanDegree"], "max_tokens": r["maxTokens"], "gini": r["gini"],
                        "hub": r["maxDegree"] / (n - 1) if n > 1 else None,
                        "spread_alike": larger_side(r.get("spreadShare"))})
            if r.get("_heavy"):
                row["dimension"] = r.get("dimension")
                heavy.append((t, int(r["_frame"])))
    frames = [{"t": t, **frame_row(gol_store.read_frame(run_id, index))} for t, index in heavy]
    return {"ended_at": ended_at, "rows": [rows[t] for t in sorted(rows)], "frames": frames}


def kept_row(run_id: str, at: int) -> Dict[str, Any]:
    """A kept world: its width over its random twin's, both half cuts, and the same-position share."""
    from book_chapters.space import measure
    from book_graph import degree_preserving
    space = _module("atlas_space", os.path.join(ROOT, "research", "atlas", "space.py"))
    with np.load(gol_store.archive_path(run_id, at)) as z:
        n, edges = len(z["ids"]), np.array(z["edges"])
    row: Dict[str, Any] = {"t": at}
    if n < 2:
        return row
    pairs = pairs_of(n, edges)
    adj: Dict[int, set] = {u: set() for u in range(n)}
    for a, b in pairs:
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))
    twin = degree_preserving(adj, SEED)
    if n >= WIDE_AT_LEAST:
        width = space.typical_distance({"world": measure(adj, SEED), "twin": measure(twin, SEED)})
        row["width"] = float(width) if math.isfinite(width) else None
    row["half_cut"] = half_cut(n, pairs)["exponent"]
    row["half_cut_twin"] = half_cut(len(twin), np.array([(u, v) for u in twin for v in twin[u] if u < v]))["exponent"]
    if n >= SMALL:
        diversity = _module("diversity_checks", os.path.join(ROOT, "research", "oee", "checks", "diversity.py"))
        kinds = diversity.stake_kinds(f"{run_id}:{at}", game=False)
        row["apart"], row["kinds"] = float(kinds["pairs_apart"]), float(kinds["effective"])
    return row


# ---------------------------------------------------------------------------
# Items, kept as they are made
# ---------------------------------------------------------------------------

def items(run_id: str, ended_at: int, settle: int) -> List[Item]:
    """What a world that ended at `ended_at` is read in: its series, and every world it kept from `settle` on."""
    kept = [at for at in gol_store.archived(run_id) if settle <= at <= ended_at]
    return [(run_id, "series", ended_at)] + [(run_id, "kept", at) for at in kept]


def read(item: Item) -> Dict[str, Any]:
    run_id, kind, at = item
    return series(run_id, at) if kind == "series" else kept_row(run_id, at)


def _path(run_id: str) -> str:
    return os.path.join(readings_dir(), f"{run_id}.json")


def _key(item: Item) -> str:
    return f"{item[1]} {item[2]}"


def kept(runs: List[str]) -> Dict[Item, Dict[str, Any]]:
    """Every reading kept for these worlds."""
    have = {}
    for run_id in runs:
        if os.path.exists(_path(run_id)):
            for key, reading in json.load(open(_path(run_id))).items():
                kind, at = key.split()
                have[(run_id, kind, int(at))] = reading
    return have


def keep(item: Item, have: Dict[Item, Dict[str, Any]]) -> None:
    """Write the world of a reading just made, with every other reading of it, whole and then moved into place."""
    run_id = item[0]
    os.makedirs(readings_dir(), exist_ok=True)
    mine = {_key(i): r for i, r in have.items() if i[0] == run_id}
    with open(_path(run_id) + ".tmp", "w") as f:
        json.dump(mine, f)
    os.replace(_path(run_id) + ".tmp", _path(run_id))


def averages(have: Dict[Item, Dict[str, Any]], wanted: List[Item], settle: int) -> Dict[str, Optional[float]]:
    """Every number of a world, averaged over its readings from iteration `settle` on (None where it has none)."""
    rows: List[Dict[str, Any]] = []
    for item in wanted:
        reading = have[item]
        rows += reading["rows"] + reading["frames"] if item[1] == "series" else [reading]
    values: Dict[str, List[float]] = {}
    for row in rows:
        if row["t"] < settle:
            continue
        for key, v in row.items():
            if key != "t" and v is not None and math.isfinite(v):
                values.setdefault(key, []).append(float(v))
    return {key: float(np.mean(v)) for key, v in sorted(values.items())}
