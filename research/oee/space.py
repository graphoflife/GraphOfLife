#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Space alone: what network do births and deaths make, with no brains at all?

The question of Chapter 51. A world's network is made by three things —
births add an agent and its links, deaths take an agent and its links away,
and the cull keeps only the largest piece — and by whatever the rules say
about each. Take the brains away and these are the only forces left: every
agent has a child with the same chance, every agent dies with the same chance,
and the population hovers near a target size. What the network looks like
then is what the rules of space make by themselves.

The rules that can be switched:

  --links k        a newborn joins its parent and k of the parent's neighbours,
                   drawn at random (0: its parent alone)
  --mend           when an agent dies, the groups of its former neighbours that
                   are not linked to one another are joined, through their
                   least-linked members (the engine's mend_on_death)
  --ring           mend the groups into a ring rather than a chain, so no group
                   hangs on a single new link
  --mend-all       instead: join all its former neighbours in a chain, in a
                   random order (a stronger mending)
  --cap K          an agent with K links has no child, and a newborn's random
                   links go only to neighbours below K (a valence)
  --age A          every agent dies at age A, as in B2, instead of by chance
  --breeders who   which agents may have a child: any, the young (younger than
                   the median of the living) or the old (the median or older);
                   those that may breed do so the more often for being fewer

and the size and length of the run with --target, --deaths and --steps. It
prints, for the world at the end: agents, mean links, the share of agents in
the 2-core and the 3-core, the largest share one agent's loss or one link's
loss cuts off, the ball dimension, the diameter and the share of the others the
best-connected agent is joined to.

    python3 research/oee/space.py --links 2 --mend
    python3 research/oee/space.py --links 0
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import networkx as nx
import numpy as np

from GraphOfLifeSimple import worst_node_cut_share
from gol_series import _structure


def groups_of(G: nx.Graph, nbrs: List[int]) -> List[List[int]]:
    """The former neighbours of a dead agent, grouped by direct links among them."""
    groups: List[List[int]] = []
    where: Dict[int, int] = {}
    for x in sorted(nbrs):
        joined = sorted({where[y] for y in G[x] if y in where})
        if not joined:
            where[x] = len(groups)
            groups.append([x])
            continue
        keep = joined[0]
        groups[keep].append(x)
        where[x] = keep
        for g in joined[1:]:
            for y in groups[g]:
                where[y] = keep
            groups[keep].extend(groups[g])
            groups[g] = []
    return [g for g in groups if g]


def grow(links: int = 2, mend: bool = True, mend_all: bool = False, cap: int = 0, ring: bool = False,
         target: int = 1500, deaths: float = 0.02, steps: int = 300, seed: int = 0, age: int = 0,
         breeders: str = "any") -> nx.Graph:
    """The world's network after `steps` iterations of births, deaths and the cull."""
    rng = np.random.RandomState(seed)
    G = nx.watts_strogatz_graph(50, 6, 0.2, seed=seed)
    born = {u: -int(rng.randint(0, age)) if age else 0 for u in G.nodes()}
    nxt = 50
    for t in range(steps):
        n = G.number_of_nodes()
        # Births balance deaths at the target size, and fill it from below; when only some agents may breed,
        # each of them breeds the more often.
        died = 1.0 / age if age else deaths
        if breeders == "any":
            may = list(G.nodes())
        else:
            middle = np.median([t - born[u] for u in G.nodes()])
            may = [u for u in G.nodes() if (t - born[u] < middle) == (breeders == "young")]
        birth = (died * target / max(n, 1) + (0.05 if n < target else 0.0)) * n / max(1, len(may))
        for u in may:
            if cap and G.degree(u) >= cap:
                continue
            if rng.random_sample() >= birth:
                continue
            room = [v for v in G[u] if not cap or G.degree(v) < cap]
            child = nxt
            nxt += 1
            born[child] = t
            G.add_edge(child, u)
            for v in rng.permutation(room)[:links]:
                G.add_edge(child, int(v))
        for u in list(G.nodes()):
            if u not in G:
                continue
            if (t - born[u] < age) if age else (rng.random_sample() >= deaths):
                continue
            nbrs = list(G[u])
            G.remove_node(u)
            if mend_all and len(nbrs) > 1:
                order = [nbrs[i] for i in rng.permutation(len(nbrs))]
                for a, b in zip(order, order[1:]):
                    G.add_edge(a, b)
            elif mend:
                parts = groups_of(G, nbrs)
                pairs = list(zip(parts, parts[1:]))
                if ring and len(parts) >= 3:
                    pairs.append((parts[-1], parts[0]))
                for g1, g2 in pairs:
                    a = min(g1, key=lambda z: (G.degree(z), z))
                    b = min(g2, key=lambda z: (G.degree(z), z))
                    G.add_edge(a, b)
        if G.number_of_nodes():
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    # When each agent was born, and when the run ended, for whoever wants their ages.
    G.graph["born"] = {u: born.get(u, 0) for u in G.nodes()}
    G.graph["steps"] = steps
    return G


def measure(G: nx.Graph) -> Dict[str, float]:
    """What the health bar asks of a network, and a little more."""
    ids = list(G.nodes())
    s = _structure(ids, [list(e) for e in G.edges()])
    core = nx.core_number(G)
    deg = np.array([d for _, d in G.degree()])
    return {
        "agents": len(ids),
        "links": float(deg.mean()),
        "core2": float(s["coreShare"]),
        "core3": float(np.mean([c >= 3 for c in core.values()])),
        "nodeCut": float(worst_node_cut_share(ids, G.adj)),
        "edgeCut": float(s["cutRisk"]),
        "dimension": float(s["dimension"]) if s["dimension"] is not None else float("nan"),
        "diameter": float(s["diameter"]),
        "hub": float(deg.max() / max(1, len(ids) - 1)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--links", type=int, default=2)
    ap.add_argument("--mend", action="store_true")
    ap.add_argument("--mend-all", action="store_true")
    ap.add_argument("--ring", action="store_true")
    ap.add_argument("--cap", type=int, default=0)
    ap.add_argument("--target", type=int, default=1500)
    ap.add_argument("--deaths", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--age", type=int, default=0)
    ap.add_argument("--breeders", choices=("any", "young", "old"), default="any")
    a = ap.parse_args()
    out = measure(grow(a.links, a.mend, a.mend_all, a.cap, a.ring, a.target, a.deaths, a.steps, a.seed, a.age,
                       a.breeders))
    print("  ".join(f"{k}={v:.3g}" for k, v in out.items()))


if __name__ == "__main__":
    main()
