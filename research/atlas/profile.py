"""
Healthy and rich: how close every world comes to the author's picture of a world
that might keep evolving (2026-10-16) — a network that is decentralised,
resilient and nearly three-dimensional, and agents that differ in almost
everything. A world whose agents all hold one to three tokens is not one.

Each world is read at one moment, from its checkpoint (a lab run's last; a
pilot's last; the long runs at their end), without its brains.

The network — the health bar of Chapter 46, whose numbers are the book's own:
  pieces              one piece (the share of agents in the largest)
  2-core              at least 0.8 of the agents in the 2-core
  worst agent cut     no single agent's loss cuts off more than 2% of the agents
  worst link cut      no single link's loss cuts off more than 1%
  hub                 the best-connected agent joined to at most 5% of the others
  links per agent     3 to 12
  dimension           2.3 to 4 by balls (gol_series's, on the checkpoint itself;
                      the author's "almost three" is the middle of the range)
Diversity — how widely the agents spread, with no line to draw:
  tokens, links, ages the effective number of doubling classes the agents fill
                      (1, 2-3, 4-7, ...; for ages, age + 1): exp of the Shannon
                      entropy of their shares. One class holding everyone gives 1;
                      k classes equally full give k. Agents holding one to three
                      tokens fill at most two.
  genotypes           distinct genotypes per agent, and the effective number
                      (1 / sum of squared shares) per agent
  kinds               where Chapter 70's measure was taken on the world: its
                      effective kinds of staking

    python3 research/atlas/profile.py [--jobs 2]     writes GraphOfLifeRuns/.pilots/atlas/profile.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = "/home/stefan/Documents/CodingProjects/GraphOfLife Claude"
sys.path.insert(0, ROOT)
RUNS = os.path.join(ROOT, "GraphOfLifeRuns")
PILOTS = os.path.join(RUNS, ".pilots")
DIVERSITY = os.path.join(PILOTS, "diversity")
OUT = os.path.join(PILOTS, "atlas", "profile.json")
#: The worlds a diversity score is ranked among, frozen so that a score does not move as worlds are added:
#: each spread's values over every world of profile.json on the day it was made (`--freeze`).
REFERENCE = os.path.join(ROOT, "book", "results", "diversity_reference.json")
#: The four spreads the diversity score is the mean percentile of.
SPREADS = ("spread_tokens", "spread_links", "spread_ages", "effective_genotypes_per_agent")
#: The book's health bar (Chapter 46).
BAR = {"core": 0.8, "agent_cut": 0.02, "link_cut": 0.01, "hub": 0.05, "links": (3.0, 12.0), "dimension": (2.3, 4.0)}
#: Before this a world has not settled (Chapter 10): its moment says little.
SETTLED = 500


def reference() -> Dict[str, np.ndarray]:
    """Each spread's values over the frozen reference worlds, sorted."""
    kept = json.load(open(REFERENCE))
    return {key: np.array(values, float) for key, values in kept["spreads"].items()}


def diversity(w: Dict[str, Any], ref: Dict[str, np.ndarray]) -> Tuple[Dict[str, float], Optional[float]]:
    """A world's percentile on each spread among the reference worlds, and their mean: its diversity score."""
    out = {}
    for key, values in ref.items():
        v = w.get(key)
        if v is not None and np.isfinite(v):
            out[key] = float(np.searchsorted(values, v) / max(1, len(values) - 1))
    return out, (float(np.mean(list(out.values()))) if out else None)


def freeze() -> None:
    """Make the reference from the worlds profile.json holds now."""
    worlds = [w for w in json.load(open(OUT))["worlds"] if "bar" in w]
    spreads = {}
    for key in SPREADS:
        values = np.sort(np.array([w.get(key, np.nan) for w in worlds], float))
        spreads[key] = [float(v) for v in values[np.isfinite(values)]]
    with open(REFERENCE, "w") as f:
        json.dump({"what": "The worlds a diversity score is ranked among (research/atlas/profile.py --freeze): each "
                           "spread's values, sorted, over every world of the profile on the day it was frozen.",
                   "frozen": time.strftime("%Y-%m-%d"), "worlds": len(worlds), "spreads": spreads}, f)
        f.write("\n")


def spread(values: np.ndarray) -> float:
    """The effective number of doubling classes the values fill: exp of the entropy of their shares."""
    v = np.asarray(values, dtype=float)
    v = v[v >= 1]
    if v.size == 0:
        return 0.0
    _, counts = np.unique(np.floor(np.log2(v)).astype(int), return_counts=True)
    p = counts / counts.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def cuts(n: int, edges: np.ndarray) -> Dict[str, float]:
    """
    One depth-first search per piece (Tarjan's, iterative), with subtree sizes:
    the most agents one agent's loss cuts off the largest remaining part, the
    most one link's loss cuts off, as shares of all agents; the pieces, and the
    share of agents in the largest.
    """
    adj: List[List[int]] = [[] for _ in range(n)]
    for a, b in edges:
        if a != b:
            adj[int(a)].append(int(b))
            adj[int(b)].append(int(a))
    disc = [-1] * n
    low = [0] * n
    size = [1] * n
    parent = [-1] * n
    clock = 0
    pieces: List[int] = []
    worst_agent = worst_link = 0
    for root in range(n):
        if disc[root] >= 0:
            continue
        disc[root] = low[root] = clock
        clock += 1
        order = [root]
        separated: Dict[int, List[int]] = {}
        stack = [(root, 0)]
        while stack:
            u, i = stack[-1]
            if i < len(adj[u]):
                stack[-1] = (u, i + 1)
                v = adj[u][i]
                if v == parent[u]:
                    continue
                if disc[v] < 0:
                    disc[v] = low[v] = clock
                    clock += 1
                    parent[v] = u
                    order.append(v)
                    stack.append((v, 0))
                else:
                    low[u] = min(low[u], disc[v])
            else:
                stack.pop()
                p = parent[u]
                if p >= 0:
                    size[p] += size[u]
                    low[p] = min(low[p], low[u])
                    if low[u] >= disc[p]:
                        separated.setdefault(p, []).append(size[u])
        piece = len(order)
        pieces.append(piece)
        for u in order[1:]:
            if low[u] > disc[parent[u]]:
                worst_link = max(worst_link, min(size[u], piece - size[u]))
        for agent, parts in separated.items():
            whole = parts if agent == root else parts + [piece - 1 - sum(parts)]
            worst_agent = max(worst_agent, piece - 1 - max(whole))
    total = max(1, sum(pieces))
    return {"agent_cut": worst_agent / total, "link_cut": worst_link / total, "pieces": len(pieces),
            "largest_piece": max(pieces) / total if pieces else 0.0}


def two_core(n: int, edges: np.ndarray) -> float:
    """The share of agents left when every agent with fewer than two links is peeled away, again and again."""
    degree = np.zeros(n, dtype=int)
    adj: List[List[int]] = [[] for _ in range(n)]
    for a, b in edges:
        if a != b:
            degree[a] += 1
            degree[b] += 1
            adj[int(a)].append(int(b))
            adj[int(b)].append(int(a))
    gone = np.zeros(n, dtype=bool)
    queue = [u for u in range(n) if degree[u] < 2]
    for u in queue:
        gone[u] = True
    while queue:
        u = queue.pop()
        for v in adj[u]:
            if not gone[v]:
                degree[v] -= 1
                if degree[v] < 2:
                    gone[v] = True
                    queue.append(v)
    return float((~gone).sum() / n) if n else 0.0


def worlds() -> List[Tuple[str, str, str, bool]]:
    """(world id, its checkpoint, its family, whether its brains are mixed at birth) for every world, each once."""
    from book_chapters.everyworld import replicas
    repeated = replicas()
    out = []
    for d in sorted(os.listdir(RUNS)):
        ckpt = os.path.join(RUNS, d, "checkpoint.npz")
        meta = os.path.join(RUNS, d, "meta.json")
        if not (os.path.exists(ckpt) and os.path.exists(meta)):
            continue
        if int(json.load(open(meta)).get("iteration") or 0) < SETTLED:
            continue
        cfg = json.load(open(meta)).get("config") or {}
        out.append((d, ckpt, re.sub(r"-s\d+$", "", d), bool(cfg.get("mix_at_birth"))))
    for config in sorted(glob.glob(os.path.join(PILOTS, "*", "*", "config.json"))):
        d = os.path.dirname(config)
        batch, name = d.split(os.sep)[-2:]
        pid = f"{batch}/{name}"
        # A replica is its lab world again; only the long runs, which go further, count on their own.
        if pid in repeated and batch != "L":
            continue
        have = sorted(glob.glob(os.path.join(d, "ckpt_*.npz")))
        if not have:
            continue
        last = have[-1]
        at = int(re.search(r"ckpt_(\d+)", last).group(1))
        if at < SETTLED:
            continue
        # A candidate's long run (research/oee/pilots/candidates.py) is a copy of
        # its world, LC/<batch>-<name>, until it has gone further than the original.
        if batch == "LC":
            original = glob.glob(os.path.join(PILOTS, *name.split("-", 1), "ckpt_*.npz"))
            if at <= max((int(re.search(r"ckpt_(\d+)", p).group(1)) for p in original), default=-1):
                continue
        meta = json.load(open(config))
        mixes = bool((meta.get("proto") or {}).get("mix") or (meta.get("cfg") or {}).get("mix_at_birth"))
        out.append((pid, last, f"{batch}/{re.sub(r'-s[0-9]+$', '', name)}", mixes))
    return out


def kinds_of() -> Dict[str, float]:
    """Effective kinds of staking, by world, from every measure of Chapter 70 (the latest moment measured)."""
    out: Dict[str, Tuple[int, float]] = {}
    for path in glob.glob(os.path.join(DIVERSITY, "*.json")):
        for w in json.load(open(path)).get("worlds") or []:
            spec = w["world"]
            at = re.search(r":(\d+)$", spec)
            moment = int(at.group(1)) if at else 10 ** 9
            key = spec.split(".pilots/")[-1].split(":")[0] if spec.startswith("pilot:") else spec.split(":")[0]
            if key not in out or moment >= out[key][0]:
                out[key] = (moment, float(w["stake_kinds"]["effective"]))
    return {k: v for k, (_, v) in out.items()}


def measure(item: Tuple[str, str, str, bool]) -> Optional[Dict[str, Any]]:
    from gol_series import _ball_growth
    wid, ckpt, family, mixes = item
    try:
        with np.load(ckpt, allow_pickle=False) as z:
            ids = z["ids"]
            tokens = z["tokens"]
            brains = z["brain_ids"]
            born = z["born_at"]
            edges = z["edges"]
            iteration = int(z["counters"][2])
            # A genotype is a set of weights. Its id says so everywhere but in the
            # worlds that mixed brains at birth with the prototype's rule (until
            # 2026-10-16), which gave a mixture of two clones — the parent's brain
            # again — an id of its own: there the weights are compared instead.
            if mixes and len(ids):
                layers = int(z["n_layers"][0])
                flat = np.concatenate([z[f"{k}{i}"].reshape(len(ids), -1).astype(np.float64)
                                       for i in range(layers) for k in ("W", "b")], axis=1)
                brains = np.unique(flat, axis=0, return_inverse=True)[1].ravel()
    except (OSError, KeyError, ValueError) as e:
        return {"id": wid, "error": f"{type(e).__name__}: {e}"}
    n = len(ids)
    if n < 2:
        return {"id": wid, "family": family, "at": iteration, "agents": n, "dead": True}
    # A checkpoint keeps its links as positions in its list of agents.
    pairs = np.asarray(edges, dtype=int).reshape(-1, 2)
    pairs = pairs[(pairs >= 0).all(axis=1) & (pairs < n).all(axis=1)]
    degree = np.bincount(pairs.ravel(), minlength=n)
    _, carried = np.unique(brains, return_counts=True)
    shares = carried / n
    out: Dict[str, Any] = {
        "id": wid, "family": family, "at": iteration, "agents": n, "tokens_per_agent": float(tokens.sum() / n),
        "median_tokens": float(np.median(tokens)), "share_at_most_three": float(np.mean(tokens <= 3)),
        "links_per_agent": float(2 * len(pairs) / n), "hub": float(degree.max() / max(1, n - 1)),
        "core": two_core(n, pairs),
        "spread_tokens": spread(tokens), "spread_links": spread(degree), "spread_ages": spread(iteration - born + 1),
        "genotypes_per_agent": float(len(carried) / n), "effective_genotypes_per_agent": float(1.0 / (shares ** 2).sum() / n),
    }
    out.update(cuts(n, pairs))
    adj: Dict[int, set] = {u: set() for u in range(n)}
    for a, b in pairs:
        if a != b:
            adj[int(a)].add(int(b))
            adj[int(b)].add(int(a))
    out["dimension"] = _ball_growth(list(range(n)), adj)[0]
    return out


def bar(w: Dict[str, Any]) -> Dict[str, bool]:
    """Which of the health bar's seven marks a world meets (None where it was not measured)."""
    dim = w.get("dimension")
    return {
        "one piece": w.get("pieces") == 1,
        "2-core": w.get("core", 0) >= BAR["core"],
        "agent cut": w.get("agent_cut", 1) <= BAR["agent_cut"],
        "link cut": w.get("link_cut", 1) <= BAR["link_cut"],
        "no hub": w.get("hub", 1) <= BAR["hub"],
        "links": BAR["links"][0] <= w.get("links_per_agent", 0) <= BAR["links"][1],
        "dimension": (BAR["dimension"][0] <= dim <= BAR["dimension"][1]) if dim is not None else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--freeze", action="store_true", help="make the diversity reference from profile.json, and stop")
    a = ap.parse_args()
    if a.freeze:
        return freeze()
    items = worlds()
    print(f"{len(items)} worlds", flush=True)
    kinds = kinds_of()
    results = []
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        for r in pool.map(measure, items, chunksize=4):
            if r is None:
                continue
            if "error" not in r and not r.get("dead"):
                r["kinds"] = kinds.get(r["id"])
                r["bar"] = bar(r)
            results.append(r)
    # Written whole and then moved into place: the search by elimination reads it while worlds run.
    with open(OUT + ".tmp", "w") as f:
        json.dump({"what": __doc__.strip().splitlines()[0], "bar": BAR, "worlds": results}, f, indent=1)
    os.replace(OUT + ".tmp", OUT)
    met = [sum(1 for v in r["bar"].values() if v) for r in results if "bar" in r]
    print(f"{len(results)} worlds measured; health marks met: "
          + ", ".join(f"{k}: {met.count(k)}" for k in range(8)), flush=True)


if __name__ == "__main__":
    main()
