#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-frame summary statistics for a whole run.

The viewer draws the history of a single statistic — how the Gini coefficient
moved across every iteration, say. Fetching every frame in the browser to work
that out would mean pulling the full topology of thousands of frames, so the
numbers are reduced here instead and sent as flat arrays.

Results are cached in the run directory and extended incrementally: only frames
added since the last request are read. The cache is keyed by the statistics
version, so changing a formula invalidates it rather than silently mixing old
and new numbers.
"""
from __future__ import annotations

import json
import math
import os
import threading
from typing import Any, Dict, List

import gol_store as store

# Progress of in-flight builds, so the browser can show how far along a rebuild
# is instead of sitting on a blank wait. Reads happen on a different thread from
# the build, since the server handles each request in its own.
_PROGRESS: Dict[str, Dict[str, Any]] = {}
_PROGRESS_LOCK = threading.Lock()
_BUILD_LOCKS: Dict[str, threading.Lock] = {}


def _build_lock(run_id: str) -> threading.Lock:
    """One lock per run, so two callers do not rebuild the same series twice."""
    with _PROGRESS_LOCK:
        lock = _BUILD_LOCKS.get(run_id)
        if lock is None:
            lock = _BUILD_LOCKS[run_id] = threading.Lock()
        return lock


def _set_progress(run_id: str, done: int, total: int, building: bool = True) -> None:
    with _PROGRESS_LOCK:
        _PROGRESS[run_id] = {"building": building, "done": done, "total": total}


def progress(run_id: str) -> Dict[str, Any]:
    """How far a build has got, for the progress bar."""
    with _PROGRESS_LOCK:
        state = _PROGRESS.get(run_id)
    return dict(state) if state else {"building": False, "done": 0, "total": 0}


# Bump when a formula below changes, so stale caches are discarded.
SERIES_VERSION = 7

# Keys that count nodes, and are therefore also meaningful as a share of the
# population that entered the phase.
NODE_COUNT_KEYS = ("births", "revolutions", "starved", "orphaned", "leaves",
                   "gainers", "losers")


def _gini(values: List[int]) -> float:
    """
    Concentration of wealth. 0 is perfect equality, 1 is one agent holding all.

    Kept identical to the browser-side implementation in stats.js so the value
    shown on a frame matches the point plotted for it.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    cumulative = 0
    weighted = 0
    for v in ordered:
        cumulative += v
        weighted += cumulative
    if cumulative <= 0:
        return 0.0
    return (len(ordered) + 1 - 2 * weighted / cumulative) / len(ordered)


def _shannon(counts) -> float:
    """Shannon entropy of a set of counts, in bits."""
    total = sum(counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return h


def _structure(ids: List[int], edges: List[List[int]]) -> Dict[str, Any]:
    """
    Loops, triangles and dimension, mirroring web/js/graphstats.js.

    Kept in step with the browser implementation so a frame's number matches
    the point plotted for it.
    """
    adj: Dict[int, set] = {i: set() for i in ids}
    for a, b in edges:
        if a != b and a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)

    def component_labels(neighbours):
        label, count = {}, 0
        for start in ids:
            if start in label:
                continue
            stack = [start]
            label[start] = count
            while stack:
                u = stack.pop()
                for v in neighbours.get(u, ()):
                    if v not in label:
                        label[v] = count
                        stack.append(v)
            count += 1
        return label, count

    # --- bridges, by iterative depth-first search ---
    edge_index = {}
    for i, (a, b) in enumerate(edges):
        edge_index[(a, b) if a < b else (b, a)] = i

    disc: Dict[int, int] = {}
    low: Dict[int, int] = {}
    bridges: set = set()
    timer = 0

    for root in ids:
        if root in disc:
            continue
        stack = [[root, None, iter(adj[root])]]
        disc[root] = low[root] = timer
        timer += 1

        while stack:
            top = stack[-1]
            node, parent, it = top[0], top[1], top[2]
            nxt = next(it, None)

            if nxt is None:
                stack.pop()
                if stack:
                    u, v = stack[-1][0], node
                    low[u] = min(low[u], low[v])
                    if low[v] > disc[u]:
                        idx = edge_index.get((u, v) if u < v else (v, u))
                        if idx is not None:
                            bridges.add(idx)
                continue

            if nxt == parent:
                top[1] = None      # skip the edge we arrived on, once
                continue
            if nxt in disc:
                low[node] = min(low[node], disc[nxt])
                continue
            disc[nxt] = low[nxt] = timer
            timer += 1
            stack.append([nxt, node, iter(adj[nxt])])

    _, component_count = component_labels(adj)
    cycle_rank = max(0, len(edges) - len(ids) + component_count)

    # --- triangles ---
    triangle_total = 0
    for a, b in edges:
        na, nb = adj.get(a), adj.get(b)
        if not na or not nb:
            continue
        small, large = (na, nb) if len(na) <= len(nb) else (nb, na)
        triangle_total += sum(1 for w in small if w in large)
    triangle_total = round(triangle_total / 3)

    triples = sum(len(adj[i]) * (len(adj[i]) - 1) / 2 for i in ids)
    transitivity = (3 * triangle_total / triples) if triples > 0 else 0.0

    # --- ball-growth dimension ---
    dimension = None
    n = len(ids)
    if n >= 8:
        seeds, max_radius = 24, 5
        step = max(1, n // min(seeds, n))
        volume = [0.0] * (max_radius + 1)
        sampled = 0
        for s_i in range(0, n, step):
            seen = {ids[s_i]}
            shell = [ids[s_i]]
            volume[0] += 1
            for r in range(1, max_radius + 1):
                nxt_shell = []
                for u in shell:
                    for v in adj.get(u, ()):
                        if v not in seen:
                            seen.add(v)
                            nxt_shell.append(v)
                shell = nxt_shell
                volume[r] += len(seen)
                if not shell:
                    for rr in range(r + 1, max_radius + 1):
                        volume[rr] += len(seen)
                    break
            sampled += 1

        if sampled:
            volumes = [v / sampled for v in volume]
            xs, ys = [], []
            for r in range(1, max_radius + 1):
                if volumes[r] > n * 0.5:
                    continue
                shell_size = volumes[r] - volumes[r - 1]
                if shell_size <= 0:
                    continue
                xs.append(math.log(r))
                ys.append(math.log(shell_size))
            if len(xs) >= 2:
                mx = sum(xs) / len(xs)
                my = sum(ys) / len(ys)
                num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
                den = sum((x - mx) ** 2 for x in xs)
                # The shell exponent is d - 1.
                dimension = (num / den + 1) if den > 0 else None

    return {
        "cycleRank": cycle_rank,
        "loopDensity": (cycle_rank / len(edges)) if edges else 0.0,
        "bridges": len(bridges),
        "components": component_count,
        "triangles": triangle_total,
        "transitivity": transitivity,
        "dimension": dimension,
    }


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    return float(ordered[mid]) if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def _reconstruct_delta(frame: Dict[str, Any], previous: Dict[str, Any] | None) -> List[int] | None:
    """
    Per-node token change for frames recorded before the engine tracked it.

    A node's balance entering a phase is its balance at the end of the previous
    one, so the previous frame supplies exactly what the engine would have
    stored. A node missing from it did not exist yet and counts its whole
    balance as gained, matching how a newborn is treated.
    """
    if previous is None:
        return None
    before = dict(zip(previous.get("ids", []), previous.get("tokens", [])))
    return [t - before.get(i, 0) for i, t in zip(frame.get("ids", []), frame.get("tokens", []))]


def frame_stats(frame: Dict[str, Any], previous: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Reduce one frame to the scalars the viewer plots."""
    tokens = frame.get("tokens", [])
    ids = frame.get("ids", [])
    edges = frame.get("edges", [])
    n = len(ids)

    degree = {}
    for a, b in edges:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
    degrees = [degree.get(i, 0) for i in ids]

    ordered_tokens = sorted(tokens)
    median = 0.0
    if ordered_tokens:
        mid = len(ordered_tokens) // 2
        median = (float(ordered_tokens[mid]) if len(ordered_tokens) % 2
                  else (ordered_tokens[mid - 1] + ordered_tokens[mid]) / 2)

    distinct_brains = len(set(frame.get("brain_ids", [])))
    decisions = frame.get("decisions") or {}
    cleanup = frame.get("cleanup") or {}

    winners = decisions.get("winners")
    revolutions = sum(1 for w in winners if w.get("revolt")) if winners else None
    held_home = (sum(1 for w in winners if w.get("winner") == w.get("node")) / len(winners)
                 if winners else None)

    births = decisions.get("births")
    mean_invested = None
    mean_links = None
    repro_token_share = None
    if births is not None:
        repro_token_share = (sum(b["invested"] for b in births) / sum(tokens)) if sum(tokens) else 0.0
        if births:
            mean_invested = sum(
                (b["invested"] / b["tokens_before"]) if b.get("tokens_before") else 0.0
                for b in births) / len(births)
            mean_links = sum(len(b.get("links") or []) for b in births) / len(births)
        else:
            mean_invested = 0.0
            mean_links = 0.0

    # Edge traffic, rebuilt from the allocations rather than stored per edge.
    total_flow = mean_flow = max_flow = None
    self_share = revolt_share = spread_share = None
    allocations = decisions.get("allocations")
    if allocations is not None:
        flow: Dict[Any, int] = {}
        allocated = kept = revolted = 0
        spread_count = 0
        for record in allocations:
            agent = record["agent"]
            if record.get("spread"):
                spread_count += 1
            revolts = record.get("revolt") or []
            for i, target in enumerate(record["targets"]):
                amount = record["alloc"][i]
                if not amount:
                    continue
                allocated += amount
                if target == agent:
                    kept += amount
                else:
                    key = (agent, target) if agent < target else (target, agent)
                    flow[key] = flow.get(key, 0) + amount
                if i < len(revolts):
                    revolted += revolts[i]

        values = list(flow.values())
        total_flow = sum(values)
        mean_flow = (total_flow / len(values)) if values else 0.0
        max_flow = max(values) if values else 0
        self_share = (kept / allocated) if allocated else 0.0
        revolt_share = (revolted / allocated) if allocated else 0.0
        spread_share = (spread_count / len(allocations)) if allocations else 0.0

    # Per-node token change across the phase. Absent on runs recorded before
    # deltas were tracked, in which case the metrics stay None rather than
    # claiming everyone broke even.
    delta = frame.get("delta") or _reconstruct_delta(frame, previous)
    max_added = max_lost = gainers = losers = None
    if delta:
        max_added = max(0, max(delta))
        max_lost = max(0, -min(delta))
        gainers = sum(1 for v in delta if v > 0)
        losers = sum(1 for v in delta if v < 0)

    structure = _structure(ids, edges)

    degree_hist: Dict[int, int] = {}
    for d in degrees:
        degree_hist[d] = degree_hist.get(d, 0) + 1
    degree_entropy = _shannon(degree_hist.values())
    degree_classes = len(degree_hist)
    degree_evenness = (degree_entropy / math.log2(degree_classes)) if degree_classes > 1 else 0.0

    token_entropy = _shannon(tokens)
    token_evenness = (token_entropy / math.log2(n)) if n > 1 else 0.0

    top_count = max(1, round(n * 0.1))
    ordered_desc = sorted(tokens, reverse=True)
    total_tokens = sum(tokens)
    top_share = (sum(ordered_desc[:top_count]) / total_tokens) if total_tokens else 0.0

    return {
        "iteration": frame.get("iteration", 0),
        "phase": frame.get("phase", 0),
        "nodes_before": frame.get("nodes_before"),
        "nodes": n,
        "edges": len(edges),
        "tokens": sum(tokens),
        "meanDegree": (sum(degrees) / n) if n else 0.0,
        "maxDegree": max(degrees) if degrees else 0,
        "medianTokens": median,
        "maxTokens": max(tokens) if tokens else 0,
        "gini": _gini(tokens),
        "distinctBrains": distinct_brains,
        "brainDiversity": (distinct_brains / n) if n else 0.0,
        "distinctLineages": len(set(frame.get("parent_brain_ids", []))),
        "density": (2 * len(edges)) / (n * (n - 1)) if n > 1 else 0.0,
        "medianDegree": _median(degrees),
        "minDegree": min(degrees) if degrees else 0,
        "leaves": sum(1 for d in degrees if d == 1),
        "meanTokens": (total_tokens / n) if n else 0.0,
        "minTokens": min(tokens) if tokens else 0,
        "topDecileShare": top_share,
        "degreeEntropy": degree_entropy,
        "degreeEvenness": degree_evenness,
        "tokenEntropy": token_entropy,
        "tokenEvenness": token_evenness,
        **structure,
        "maxTokenAdded": max_added,
        "maxTokenLost": max_lost,
        "gainers": gainers,
        "losers": losers,
        "births": len(births) if births is not None else None,
        "meanInvestedShare": mean_invested,
        "reproTokenShare": repro_token_share,
        "meanChildLinks": mean_links,
        "revolutions": revolutions,
        "heldHomeShare": held_home,
        "totalFlow": total_flow,
        "meanEdgeFlow": mean_flow,
        "maxEdgeFlow": max_flow,
        "selfAllocationShare": self_share,
        "revoltShare": revolt_share,
        "spreadShare": spread_share,
        "prunedEdges": len(decisions["pruned_edges"]) if "pruned_edges" in decisions else None,
        "starved": cleanup.get("starved"),
        "orphaned": cleanup.get("orphaned"),
        "redistributed": cleanup.get("redistributed"),
    }


def _cache_path(run_id: str) -> str:
    return os.path.join(store.run_dir(run_id), "series.json")


def _load_cache(run_id: str) -> Dict[str, Any]:
    try:
        with open(_cache_path(run_id), "r") as f:
            cache = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"version": SERIES_VERSION, "rows": []}

    if cache.get("version") != SERIES_VERSION:
        return {"version": SERIES_VERSION, "rows": []}
    return cache


def _save_cache(run_id: str, cache: Dict[str, Any]) -> None:
    path = _cache_path(run_id)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(cache, f, separators=(",", ":"))
        os.replace(tmp, path)
    except OSError:
        pass  # a missing cache only costs time, never correctness


def build_series(run_id: str) -> Dict[str, Any]:
    """
    Statistics for every recorded frame, as parallel arrays.

    Frames already summarised are reused; only new ones are read. If the run was
    resumed and its history truncated, the cache is trimmed to match rather than
    describing frames that no longer exist.
    """
    with _build_lock(run_id):
        try:
            return _build_series_locked(run_id)
        finally:
            _set_progress(run_id, 0, 0, building=False)


def _build_series_locked(run_id: str) -> Dict[str, Any]:
    total = store.count_frames(run_id)
    cache = _load_cache(run_id)
    rows: List[Dict[str, Any]] = cache.get("rows", [])

    # A resumed run can be shorter than what we last saw.
    if len(rows) > total:
        rows = rows[:total]

    # Frames written before deltas were tracked need their predecessor to
    # reconstruct the change, so the previous frame is carried along.
    every = 1
    try:
        every = max(1, int(store.load_meta(run_id).get("config", {}).get("export_every", 1)))
    except (OSError, ValueError, json.JSONDecodeError):
        pass

    previous = None
    if every == 1 and len(rows) > 0:
        try:
            previous = store.read_frame(run_id, len(rows) - 1)
        except (OSError, json.JSONDecodeError):
            previous = None

    changed = len(rows) > len(cache.get("rows", []))
    remaining = total - len(rows)
    if remaining > 0:
        _set_progress(run_id, len(rows), total, building=True)

    for index in range(len(rows), total):
        try:
            frame = store.read_frame(run_id, index)
        except (OSError, json.JSONDecodeError, KeyError):
            break
        rows.append(frame_stats(frame, previous if every == 1 else None))
        previous = frame
        changed = True

        # Often enough to feel live, rarely enough not to thrash the lock.
        if index % 25 == 0:
            _set_progress(run_id, index + 1, total, building=True)

    if changed or len(rows) != len(cache.get("rows", [])):
        _save_cache(run_id, {"version": SERIES_VERSION, "rows": rows})

    if not rows:
        return {"count": 0, "keys": [], "series": {}, "nodeCountKeys": list(NODE_COUNT_KEYS)}

    keys = [k for k in rows[0].keys()]
    series = {k: [row.get(k) for row in rows] for k in keys}

    return {
        "count": len(rows),
        "keys": keys,
        "series": series,
        "nodeCountKeys": list(NODE_COUNT_KEYS),
    }
