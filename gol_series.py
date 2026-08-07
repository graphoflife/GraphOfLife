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
import os
from typing import Any, Dict, List

import gol_store as store

# Bump when a formula below changes, so stale caches are discarded.
SERIES_VERSION = 1

# Keys that count nodes, and are therefore also meaningful as a share of the
# population that entered the phase.
NODE_COUNT_KEYS = ("births", "revolutions", "starved", "orphaned")


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


def frame_stats(frame: Dict[str, Any]) -> Dict[str, Any]:
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
        "births": len(decisions["births"]) if "births" in decisions else None,
        "revolutions": revolutions,
        "starved": cleanup.get("starved"),
        "orphaned": cleanup.get("orphaned"),
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
    total = store.count_frames(run_id)
    cache = _load_cache(run_id)
    rows: List[Dict[str, Any]] = cache.get("rows", [])

    # A resumed run can be shorter than what we last saw.
    if len(rows) > total:
        rows = rows[:total]

    changed = len(rows) > len(cache.get("rows", []))
    for index in range(len(rows), total):
        try:
            rows.append(frame_stats(store.read_frame(run_id, index)))
            changed = True
        except (OSError, json.JSONDecodeError, KeyError):
            break

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
