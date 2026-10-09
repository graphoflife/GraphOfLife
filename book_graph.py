#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The book's graph helpers, one copy of each.

A frame lists a world's agents and connections; the figures read it as an
adjacency, {agent: set of neighbours}, and walk it. Several chapters do, and
they used to carry their own breadth-first searches and degree counts.

These are the book's, not the engine's: the engine's statistics (gol_series)
build their own graphs, and a chapter that wants one of those reads it from
the run's recorded rows instead.
"""
from __future__ import annotations

from collections import Counter, deque
from typing import Any, Dict, Iterable


def graph(frame: Dict[str, Any]) -> Dict[int, set]:
    """A frame's agents and the connections between them, without self-loops."""
    adj: Dict[int, set] = {a: set() for a in frame["ids"]}
    for a, b in frame["edges"]:
        if a != b and a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)
    return adj


def degrees(edges: Iterable) -> Counter:
    """How many connections each agent has, counted from a frame's list of them."""
    degree: Counter = Counter()
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    return degree


def bfs(adj: Dict[int, set], source: int, limit: int = 10 ** 9) -> Dict[int, int]:
    """How many steps every agent within `limit` of `source` is from it."""
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


def core(adj: Dict[int, set]) -> Dict[int, set]:
    """What is left when dead ends are pruned until none is left: every agent with one connection, again and again."""
    import networkx as nx
    g = nx.k_core(nx.Graph([(u, v) for u in adj for v in adj[u]]), 2)
    return {u: set(g[u]) for u in g}


def degree_preserving(adj: Dict[int, set], seed: int) -> Dict[int, set]:
    """A random network with the same numbers of connections (a configuration model, made simple)."""
    import networkx as nx
    ids = sorted(adj)
    g = nx.Graph(nx.configuration_model([len(adj[a]) for a in ids], seed=seed))
    g.remove_edges_from(list(nx.selfloop_edges(g)))
    return {u: set(g[u]) for u in g}
