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
from typing import Any, Dict, List, Optional, Tuple

import gol_store as store

#: What a brain with no recorded parent carries, matching the engine.
NO_PARENT = -1

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
# Bump this whenever frame_stats gains, loses or redefines a key. Caches
# stamped with an older number are discarded and rebuilt. Forgetting to bump it
# does not merely serve stale numbers: it leaves a cache holding two shapes of
# row at once, which is how the power-law statistics came to be computed and
# then dropped on the way out.
SERIES_VERSION = 17

# At most this many iterations are analysed for a run's history.
#
# The cost is not in reading the frames — decompressing and parsing one takes
# about 2ms, against 16ms to work out its loops, triangles and dimension. A run
# of fifty thousand frames would therefore take a quarter of an hour to
# summarise in full, for a chart a few hundred pixels wide that cannot show
# that detail anyway. Sampling evenly across the run keeps the shape of every
# curve while bounding the work.
MAX_SAMPLED_ITERATIONS = 1000

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
    # Per node as well as in total, since the power-law section asks how a
    # node's triangle count grows with its degree. A triangle is met once from
    # each of its three edges, and each meeting names all three corners, so
    # every node ends up counted three times over.
    per_node_triangles = {i: 0 for i in ids}
    for a, b in edges:
        na, nb = adj.get(a), adj.get(b)
        if not na or not nb:
            continue
        small, large = (na, nb) if len(na) <= len(nb) else (nb, na)
        for w in small:
            if w in large:
                triangle_total += 1
                per_node_triangles[a] = per_node_triangles.get(a, 0) + 1
                per_node_triangles[b] = per_node_triangles.get(b, 0) + 1
                per_node_triangles[w] = per_node_triangles.get(w, 0) + 1
    triangle_total = round(triangle_total / 3)
    for node in per_node_triangles:
        per_node_triangles[node] //= 3

    triples = sum(len(adj[i]) * (len(adj[i]) - 1) / 2 for i in ids)
    transitivity = (3 * triangle_total / triples) if triples > 0 else 0.0

    n = len(ids)

    # --- radius, diameter, mean path length ---
    #
    # Mirrors distances() in web/js/graphstats.js. Exact answers need every
    # pair's shortest path; a spread of sources is swept instead, plus a second
    # sweep from the furthest node found, which is what makes the diameter
    # estimate tight. The diameter is a lower bound and the radius an upper one.
    radius = diameter = mean_path = None
    if n >= 2:
        order = {node: i for i, node in enumerate(ids)}
        # Neighbours by position, built once, so the sweeps are integer work
        # rather than repeated dictionary lookups.
        neighbours: List[List[int]] = [[] for _ in range(n)]
        for a, b in edges:
            if a == b:
                continue
            ia, ib = order.get(a), order.get(b)
            if ia is None or ib is None:
                continue
            neighbours[ia].append(ib)
            neighbours[ib].append(ia)

        reached_sum = 0
        reached_count = 0
        state = {"farthest": 0}

        def sweep(start_index: int, count_toward_mean: bool = True) -> int:
            nonlocal reached_sum, reached_count
            dist = [-1] * n
            dist[start_index] = 0
            queue = [start_index]
            ecc = 0
            head = 0
            while head < len(queue):
                i = queue[head]
                head += 1
                d = dist[i]
                if d > ecc:
                    ecc = d
                    state["farthest"] = i
                if count_toward_mean:
                    reached_sum += d
                    reached_count += 1
                for j in neighbours[i]:
                    if dist[j] != -1:
                        continue
                    dist[j] = d + 1
                    queue.append(j)
            if count_toward_mean:
                reached_count -= 1      # do not count the source's own zero
            return ecc

        # Each sweep walks the whole graph, so the total is sources x (V + E).
        # Holding that near a fixed budget keeps the cost flat as the
        # population grows.
        per_sweep = max(1, n + 2 * len(edges))
        sources = max(8, min(16, round(250_000 / per_sweep)))

        step = max(1, n // min(sources, n))
        radius, diameter, deepest, deepest_from = None, 0, 0, 0
        for i in range(0, n, step):
            ecc = sweep(i)
            radius = ecc if radius is None else min(radius, ecc)
            diameter = max(diameter, ecc)
            if ecc > deepest:
                deepest, deepest_from = ecc, state["farthest"]

        # Excluded from the mean: this source is the most peripheral node found,
        # so its distances run long by construction. Folding them in dragged
        # the average up by a third of its value when few sources were swept.
        if deepest > 0:
            diameter = max(diameter, sweep(deepest_from, count_toward_mean=False))

        radius = radius or 0
        mean_path = (reached_sum / reached_count) if reached_count > 0 else 0.0

    # --- ball-growth dimension ---
    dimension = None
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
        "radius": radius,
        "diameter": diameter,
        "meanPathLength": mean_path,
        "_perNodeTriangles": per_node_triangles,
    }


# ---------------------------------------------------------------------------
# Power laws
# ---------------------------------------------------------------------------
#
# Mirrors the same section in web/js/graphstats.js. The two are compared value
# for value by tests/test_stats_parity.py, so they have to agree on the
# arithmetic and not merely on the idea.


def _power_fit(xs, ys, minimum_points: int = 8):
    """
    Slope and R-squared of ln(y) against ln(x), over the pairs where both are
    positive. A zero is a real observation rather than a small one, so those
    pairs are dropped rather than nudged into the logarithm.
    """
    n = 0
    sx = sy = 0.0
    for x, y in zip(xs, ys):
        if x > 0 and y > 0:
            n += 1
            sx += math.log(x)
            sy += math.log(y)
    if n < minimum_points:
        return None

    mx, my = sx / n, sy / n
    sxx = syy = sxy = 0.0
    for x, y in zip(xs, ys):
        if not (x > 0 and y > 0):
            continue
        dx = math.log(x) - mx
        dy = math.log(y) - my
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy

    # No spread on an axis means no line to fit.
    if sxx <= 1e-12 or syy <= 1e-12:
        return None
    return {"exponent": sxy / sxx, "r2": (sxy * sxy) / (sxx * syy)}


def _tail_exponent(values, minimum_distinct: int = 4):
    """
    The exponent of a distribution's tail, from its complementary CDF.

    Binned histograms are badly behaved out in the tail, where a bin holds one
    or two nodes. The fraction of nodes with at least k is smooth by
    construction, and for P(k) ~ k**-g it goes as k**-(g-1).
    """
    positive = sorted(v for v in values if v > 0)
    if len(positive) < 8:
        return None
    n = len(positive)

    xs, ys = [], []
    i = n - 1
    while i >= 0:
        value = positive[i]
        j = i
        while j >= 0 and positive[j] == value:
            j -= 1
        xs.append(value)
        ys.append((n - 1 - j) / n)
        i = j
    if len(xs) < minimum_distinct:
        return None

    fit = _power_fit(xs, ys, minimum_distinct)
    if fit is None:
        return None
    return {"exponent": 1 - fit["exponent"], "r2": fit["r2"]}


def _scale_free(values, max_candidates: int = 24):
    """
    A scale-free fit of P(k) ~ k**-gamma, done the way the literature does it
    rather than by drawing a line through a histogram.

    Two things separate this from _tail_exponent above. The exponent is a
    maximum-likelihood estimate instead of a least-squares slope, because
    log-log regression on a distribution is biased and the bias is worst
    exactly where the interesting nodes are. And the tail is found rather than
    assumed: real degree distributions are only straight above some k_min, so
    every candidate k_min is tried and the one whose fitted curve sits closest
    to the data — smallest Kolmogorov-Smirnov distance — wins.

    Returns the exponent, where the tail was judged to start, how many nodes
    are in it, the KS distance, and an R-squared over the tail alone so it can
    be read beside the other fits. Mirrors scaleFree() in graphstats.js.
    """
    positive = sorted(v for v in values if v > 0)
    if len(positive) < 16:
        return None
    n_all = len(positive)

    distinct = sorted(set(positive))
    if len(distinct) < 4:
        return None

    # Trying every distinct degree is wasted work on a broad distribution, and
    # a k_min out in the tail is fitted to a handful of nodes whatever it does
    # to the KS distance. Both ends are cut.
    usable = [k for k in distinct if k < distinct[-1]]
    if len(usable) > max_candidates:
        step = len(usable) / max_candidates
        usable = [usable[int(i * step)] for i in range(max_candidates)]

    best = None
    for k_min in usable:
        tail = [v for v in positive if v >= k_min]
        n = len(tail)
        # Below this an exponent is arithmetic rather than evidence.
        if n < 25:
            continue

        # Continuous MLE with the half-integer correction, which is the usual
        # estimator for integer data like a degree.
        shift = k_min - 0.5
        total = 0.0
        for v in tail:
            total += math.log(v / shift)
        if total <= 1e-12:
            continue
        gamma = 1.0 + n / total

        # KS distance between the empirical tail and the fitted power law.
        worst = 0.0
        for i, v in enumerate(tail):
            model = 1.0 - (v / shift) ** (1.0 - gamma)
            empirical_low = i / n
            empirical_high = (i + 1) / n
            worst = max(worst, abs(model - empirical_low), abs(model - empirical_high))

        if best is None or worst < best["ks"]:
            best = {"exponent": gamma, "kMin": k_min, "tailNodes": n, "ks": worst}

    if best is None:
        return None

    # R-squared of the CCDF over the fitted tail, for readers who want the
    # familiar number next to the exponent. It is a description of the fit, not
    # a test of it — see the note on degreeExponentR2.
    tail = [v for v in positive if v >= best["kMin"]]
    xs, ys = [], []
    i = len(tail) - 1
    while i >= 0:
        value = tail[i]
        j = i
        while j >= 0 and tail[j] == value:
            j -= 1
        xs.append(value)
        ys.append((len(tail) - 1 - j) / len(tail))
        i = j
    fit = _power_fit(xs, ys, 4)
    best["r2"] = fit["r2"] if fit else None
    best["coverage"] = best["tailNodes"] / n_all
    return best


def _box_dimension(ids, adj, sizes=(1, 3, 5, 9, 17, 33)):
    """
    The fractal dimension of the graph by box covering: N_B(l_B) ~ l_B**-d_B.

    How many boxes of size l_B does it take to cover the whole graph? A
    self-similar network needs a number that falls as a power of the size, and
    that power is the dimension. Sizes climb geometrically so the fit spans as
    many decades as the graph allows rather than crowding into the small end.

    Exact covering is NP-hard, so boxes are grown greedily around centres, in
    descending order of degree: hubs make the best centres, and taking them
    first is what stops the covering fragmenting into hundreds of boxes holding
    three nodes each. Each box is a true ball — the search runs over the whole
    graph and merely declines to claim nodes another box already holds. An
    earlier version searched only through unclaimed nodes, which is cheaper but
    walls a box in behind its neighbours: it reported nearly an order of
    magnitude too many boxes at the large sizes, and a dimension of 0.9 where
    the honest answer was 1.6.

    Mirrors boxDimension() in graphstats.js.
    """
    if len(ids) < 16:
        return None

    # Ties broken by position, which is stable and does not assume node ids
    # are comparable to one another.
    position = {node: k for k, node in enumerate(ids)}
    centres_first = sorted(ids, key=lambda i: (-len(adj.get(i, ())), position[i]))

    xs, ys = [], []
    for size in sizes:
        radius = (size - 1) // 2
        covered = set()
        boxes = 0
        for seed in centres_first:
            if seed in covered:
                continue
            boxes += 1
            covered.add(seed)
            # Breadth-first to the box radius. `seen` is this ball's own visit
            # set, so the walk can pass through nodes that belong to another
            # box on its way out to nodes that belong to none.
            seen = {seed}
            frontier = [seed]
            for _ in range(radius):
                nxt = []
                for node in frontier:
                    for other in adj.get(node, ()):  # noqa: E1133
                        if other in seen:
                            continue
                        seen.add(other)
                        covered.add(other)
                        nxt.append(other)
                if not nxt:
                    break
                frontier = nxt
        xs.append(size)
        ys.append(boxes)
        # Once the whole graph fits in one box, larger boxes say nothing.
        if boxes <= 1:
            break

    if len(xs) < 3:
        return None
    fit = _power_fit(xs, ys, 3)
    if fit is None:
        return None
    return {"exponent": -fit["exponent"], "r2": fit["r2"]}


def _assortativity(edges, degree):
    """
    Newman's degree correlation. Positive means like joins like; negative means
    hubs sit among the sparsely connected, which is what most grown networks do.
    """
    m = 0
    s1 = s2 = s3 = 0.0
    for a, b in edges:
        j, k = degree.get(a), degree.get(b)
        if j is None or k is None:
            continue
        m += 1
        s1 += j * k
        s2 += j + k
        s3 += j * j + k * k
    if m < 2:
        return None

    half = s2 / (2 * m)
    denominator = s3 / (2 * m) - half * half
    if abs(denominator) < 1e-12:
        return None
    return (s1 / m - half * half) / denominator


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
    # Absent when the run has revolutions off, rather than a misleading zero.
    revolutions = None
    if winners and any("revolt" in w for w in winners):
        revolutions = sum(1 for w in winners if w.get("revolt"))
    held_home = (sum(1 for w in winners if w.get("winner") == w.get("node")) / len(winners)
                 if winners else None)


    births = decisions.get("births")
    mean_invested = None
    mean_links = None
    repro_token_share = None
    handovers = None
    if births is not None:
        repro_token_share = (sum(b["invested"] for b in births) / sum(tokens)) if sum(tokens) else 0.0
        if any("handed_over" in b for b in births):
            handovers = sum(len(b.get("handed_over") or []) for b in births)
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
        spread_share = (spread_count / len(allocations)) if allocations else 0.0
        if any("revolt" in r for r in allocations):
            revolt_share = (revolted / allocated) if allocated else 0.0

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
    per_node_triangles = structure.pop("_perNodeTriangles", {})

    # ---- power laws ----
    #
    # How one quantity scales with another, as an exponent and how tightly the
    # points sit on that line. Everything here is over the nodes of this frame.
    adjacency: Dict[Any, set] = {i: set() for i in ids}
    for a, b in edges:
        if a == b:
            continue
        if a in adjacency:
            adjacency[a].add(b)
        if b in adjacency:
            adjacency[b].add(a)
    degree_of = {i: len(adjacency[i]) for i in ids}

    degree_list = [degree_of[i] for i in ids]
    triangle_list = [per_node_triangles.get(i, 0) for i in ids]

    # Clustering only means anything for a node with two neighbours to compare.
    clustering_degrees, clustering_values = [], []
    for i in ids:
        k = degree_of[i]
        if k < 2:
            continue
        clustering_degrees.append(k)
        clustering_values.append((2 * per_node_triangles.get(i, 0)) / (k * (k - 1)))

    delta_list = _reconstruct_delta(frame, previous) or frame.get("delta") or []
    change_tokens, change_sizes = [], []
    for idx in range(min(len(tokens), len(delta_list))):
        change_tokens.append(tokens[idx])
        change_sizes.append(abs(delta_list[idx]))

    def _pair(fit, key):
        return {key: fit["exponent"] if fit else None,
                key + "R2": fit["r2"] if fit else None}

    power_laws: Dict[str, Any] = {}
    power_laws.update(_pair(_tail_exponent(degree_list), "degreeExponent"))
    power_laws.update(_pair(_tail_exponent(list(tokens)), "tokenExponent"))
    power_laws.update(_pair(_power_fit(degree_list, list(tokens)), "tokensVsDegree"))
    power_laws.update(_pair(_power_fit(degree_list, triangle_list), "trianglesVsDegree"))
    power_laws.update(_pair(_power_fit(clustering_degrees, clustering_values),
                            "clusteringVsDegree"))
    power_laws.update(_pair(_power_fit(change_tokens, change_sizes), "changeVsTokens"))
    power_laws["assortativity"] = _assortativity(edges, degree_of)

    # Scale free: the degree distribution's tail, found rather than assumed.
    scale_free = _scale_free(degree_list)
    power_laws["degreeGamma"] = scale_free["exponent"] if scale_free else None
    power_laws["degreeGammaR2"] = scale_free["r2"] if scale_free else None
    power_laws["degreeKMin"] = scale_free["kMin"] if scale_free else None
    power_laws["degreeTailShare"] = scale_free["coverage"] if scale_free else None
    power_laws["degreeGammaKS"] = scale_free["ks"] if scale_free else None

    # Self-similar: how the number of boxes needed falls as boxes grow.
    boxes = _box_dimension(ids, adjacency)
    power_laws["boxDimension"] = boxes["exponent"] if boxes else None
    power_laws["boxDimensionR2"] = boxes["r2"] if boxes else None

    degree_hist: Dict[int, int] = {}
    for d in degrees:
        degree_hist[d] = degree_hist.get(d, 0) + 1
    degree_entropy = _shannon(degree_hist.values())
    degree_classes = len(degree_hist)
    degree_evenness = (degree_entropy / math.log2(degree_classes)) if degree_classes > 1 else 0.0

    token_entropy = _shannon(tokens)
    token_evenness = (token_entropy / math.log2(n)) if n > 1 else 0.0

    # floor(x + 0.5) rather than round(), which is not the same function here.
    # Python's round() breaks a tie towards the even number and JavaScript's
    # Math.round() breaks it upwards, so at 45 agents this side took the top 4
    # and stats.js took the top 5, and the same frame reported two different
    # shares. Only populations landing exactly on a half were affected, which is
    # why it sat here undetected until a run happened to pass through one.
    top_count = max(1, int(math.floor(n * 0.1 + 0.5)))
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
        # A brain id names a genotype, so these count genotypes rather than
        # agents. Before that change a copy was given a fresh id and every
        # agent carried its own, which made both of these close to the
        # population size and neither of them a diversity measure.
        "distinctBrains": distinct_brains,
        "brainDiversity": (distinct_brains / n) if n else 0.0,
        "distinctParents": len(set(frame.get("parent_brain_ids", []))),
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
        **power_laws,
        "maxTokenAdded": max_added,
        "maxTokenLost": max_lost,
        "gainers": gainers,
        "losers": losers,
        "births": len(births) if births is not None else None,
        "meanInvestedShare": mean_invested,
        "reproTokenShare": repro_token_share,
        "handovers": handovers,
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


def _sample_stride(total_iterations: int) -> int:
    """
    How many iterations to skip between samples.

    Kept to powers of two so that when a run grows past the limit the existing
    samples stay valid: every second one is dropped and the rest are reused,
    rather than the whole history being recomputed against a shifted grid.
    """
    stride = 1
    while total_iterations // stride > MAX_SAMPLED_ITERATIONS:
        stride *= 2
    return stride


# ----------------------------------------------------------------------------
# Families
# ----------------------------------------------------------------------------

#: How far back a family is counted from. A clade is a choice of anchor and
#: there is no single right one: anchored on the founders it collapses to a
#: single family within about seventeen iterations of a typical run and reads
#: one forever afterwards, which measures nothing. Anchored a few iterations
#: back it keeps saying how finely the population is currently divided.
CLADE_WINDOW = 8


class _CladeWindow:
    """
    Who each living agent descends from, a few iterations ago.

    Only the last `window` iterations of ancestry are held, which is all a
    windowed count needs and is what keeps this bounded on a long run — the
    whole forest of a five-thousand-iteration world is millions of nodes.

    It has to be fed every iteration in order. Ancestry is a chain: sample it
    and the links between the samples are gone, so a run recorded with
    `export_every > 1`, or sampled down because it grew long, cannot have this
    computed at all and the statistic is left absent rather than guessed at.
    """

    __slots__ = ("window", "parent", "born", "_seen")

    def __init__(self, window: int = CLADE_WINDOW) -> None:
        self.window = window
        self.parent: Dict[int, int] = {}
        self.born: Dict[int, int] = {}
        self._seen: List[Tuple[int, List[int]]] = []      # (iteration, ids added)

    def observe(self, iteration: int, brain_ids: Any, parent_brain_ids: Any) -> None:
        added: List[int] = []
        for brain, parent in zip(brain_ids, parent_brain_ids):
            brain = int(brain)
            if brain in self.born:
                continue
            self.born[brain] = iteration
            self.parent[brain] = int(parent)
            added.append(brain)
        self._seen.append((iteration, added))
        self._forget(iteration)

    def _forget(self, now: int) -> None:
        """Drop ancestry older than the window, so this stays bounded."""
        keep = now - self.window * 2
        while self._seen and self._seen[0][0] < keep:
            _at, ids = self._seen.pop(0)
            for brain in ids:
                self.parent.pop(brain, None)
                self.born.pop(brain, None)

    def families(self, brain_ids: Any, iteration: int) -> int:
        """How many distinct ancestors-of-`window`-ago the living share."""
        anchor = iteration - self.window
        memo: Dict[int, int] = {}
        roots = set()
        for brain in brain_ids:
            node = int(brain)
            path: List[int] = []
            while True:
                if node in memo:
                    answer = memo[node]
                    break
                when = self.born.get(node)
                # Off the end of what is held, or old enough: this is the one.
                if when is None or when <= anchor:
                    answer = node
                    break
                up = self.parent.get(node, NO_PARENT)
                if up == NO_PARENT:
                    answer = node
                    break
                path.append(node)
                node = up
            for step in path:
                memo[step] = answer
            memo[node] = answer
            roots.add(answer)
        return len(roots)


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


def _series_keys(rows: List[Dict[str, Any]]) -> List[str]:
    """
    Every key any row carries, not merely the ones the first row happens to
    have.

    A cache can hold rows from more than one version of this file. When a
    statistic is added, the rows already stored keep their old shape and only
    new frames arrive with the new key. Reading the shape off row zero dropped
    every such statistic from the whole run — the values were computed, stored,
    and then thrown away on the way out, so the chart for them reported no data
    while the numbers sat in the cache. The newest row is the richest, so its
    order leads and anything else is appended.
    """
    keys: List[str] = []
    seen = set()
    for row in reversed(rows):
        for key in row:
            if key != "_frame" and key not in seen:
                seen.add(key)
                keys.append(key)
    return keys


def _build_series_locked(run_id: str) -> Dict[str, Any]:
    total_frames = store.count_frames(run_id)
    # Frames come in pairs, one per phase, so an iteration is two of them.
    total_iterations = max(0, total_frames // 2)
    stride = _sample_stride(total_iterations)

    cache = _load_cache(run_id)
    rows: List[Dict[str, Any]] = cache.get("rows", [])
    cached_stride = int(cache.get("stride", 1) or 1)

    # A resumed run can be shorter than what we last saw.
    rows = [r for r in rows if r.get("_frame", 0) < total_frames]

    # The run has grown past the limit since last time: thin what we have
    # rather than starting over.
    if cached_stride < stride:
        rows = [r for r in rows if (r["_frame"] // 2) % stride == 0]

    done_iterations = {r["_frame"] // 2 for r in rows}

    # Both phases of an iteration are kept, so the phase filter still has game
    # frames to show; sampling only the even indices would drop them entirely.
    wanted: List[int] = []
    for it in range(0, total_iterations, stride):
        if it in done_iterations:
            continue
        wanted.extend([2 * it, 2 * it + 1])
    wanted = [i for i in wanted if i < total_frames]

    # Frames written before deltas were tracked need their predecessor to
    # reconstruct the change. That is only sound for genuinely consecutive
    # frames, which within a sampled iteration means its second phase.
    every = 1
    try:
        every = max(1, int(store.load_meta(run_id).get("config", {}).get("export_every", 1)))
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    can_reconstruct = (every == 1)

    changed = bool(wanted) or cached_stride != stride or len(rows) != len(cache.get("rows", []))
    if wanted:
        _set_progress(run_id, 0, len(wanted), building=True)

    # How many families the living divide into needs ancestry, and ancestry is
    # a chain: it cannot be read off one frame and it cannot be sampled. So it
    # is computed here rather than in frame_stats, and only where the chain is
    # whole — every iteration recorded, and none of them thinned away.
    families = _CladeWindow() if (can_reconstruct and stride == 1) else None
    if families is not None and wanted:
        # Resuming mid-run leaves the window empty, so the frames just before
        # the first new one are read to fill it. Their statistics are already
        # cached; only their ancestry is wanted.
        for index in range(max(0, wanted[0] - CLADE_WINDOW * 2), wanted[0]):
            try:
                warm = store.read_frame(run_id, index)
            except (OSError, json.JSONDecodeError, KeyError):
                break
            families.observe(int(warm.get("iteration", index // 2)),
                             warm.get("brain_ids", []), warm.get("parent_brain_ids", []))

    previous = None
    for step, index in enumerate(wanted):
        try:
            frame = store.read_frame(run_id, index)
        except (OSError, json.JSONDecodeError, KeyError):
            break

        prior = previous if (can_reconstruct and index % 2 == 1) else None
        row = frame_stats(frame, prior)
        row["_frame"] = index
        if families is not None:
            iteration = int(frame.get("iteration", index // 2))
            families.observe(iteration, frame.get("brain_ids", []),
                             frame.get("parent_brain_ids", []))
            row["cladesInWindow"] = families.families(frame.get("brain_ids", []), iteration)
        rows.append(row)
        previous = frame

        # Often enough to feel live, rarely enough to thrash the lock.
        if step % 10 == 0:
            _set_progress(run_id, step + 1, len(wanted), building=True)

    rows.sort(key=lambda r: r["_frame"])

    if changed:
        _save_cache(run_id, {"version": SERIES_VERSION, "stride": stride, "rows": rows})

    if not rows:
        return {"count": 0, "keys": [], "series": {}, "stride": stride,
                "sampled": False, "nodeCountKeys": list(NODE_COUNT_KEYS)}

    keys = _series_keys(rows)
    series = {k: [row.get(k) for row in rows] for k in keys}

    return {
        "count": len(rows),
        "keys": keys,
        "series": series,
        "stride": stride,
        "sampled": stride > 1,
        "totalIterations": total_iterations,
        "nodeCountKeys": list(NODE_COUNT_KEYS),
    }
