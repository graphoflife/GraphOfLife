#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The genotype forest of a window of frames, reduced to what can be drawn.

This used to happen in the browser, from whole frames. On a small run that was
fine. On a real one it was not, and the numbers say why: thirty-two iterations
of a forty-thousand-agent world hold **1.36 million distinct genotypes**, and
none of them lives longer than a single iteration — every brain mutates every
iteration, so a genotype is born, is seen once, and is gone. Sending that to a
page meant twenty-seven megabytes of brain ids over the wire, six hundred
milliseconds to aggregate on the main thread, and a canvas asked to draw more
lines than it has pixels. The tab locked up, and the picture it was locking up
to produce could not have been read.

So the aggregation happens where the frames already are, and only the part that
can be drawn travels. The count of what was left out travels with it, because a
view that quietly shows the largest two thousand of a million is lying by
omission.

The layout — who sits under whom, in what order, in what colour — stays in the
browser, where it runs on the couple of thousand nodes that arrive rather than
on all of them.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

#: What a genotype with no recorded parent carries.
NO_PARENT = -1

#: How many genotypes a reply carries at most.
#:
#: More than a canvas a thousand pixels tall can show as separate rows, and far
#: fewer than a real window holds. The cap is what keeps this bounded no matter
#: how large the world got.
DEFAULT_LIMIT = 2000


def forest(frames: Iterable[Dict[str, Any]], limit: int = DEFAULT_LIMIT) -> Dict[str, Any]:
    """
    Aggregate frames into genotypes, and keep the most prominent `limit`.

    A genotype is kept by how long it lasted first and how many agents carried
    it second — longevity is the scarcer thing here, and the one a lineage is
    actually about.

    Dropping a parent orphans its children, which the drawing already knows how
    to show: a node whose parent is not present is a root, described as an
    ancestor from outside the window. That is true of a dropped parent in
    exactly the same way it is true of one that lived before the window began.
    """
    nodes: Dict[int, Dict[str, Any]] = {}
    first = last = None

    for frame in frames:
        when = int(frame.get("iteration", 0))
        first = when if first is None else min(first, when)
        last = when if last is None else max(last, when)

        ids = frame.get("brain_ids") or []
        parents = frame.get("parent_brain_ids") or []

        here: Dict[int, int] = {}
        for position, raw in enumerate(ids):
            genotype = int(raw)
            here[genotype] = here.get(genotype, 0) + 1
            node = nodes.get(genotype)
            if node is None:
                parent = int(parents[position]) if position < len(parents) else NO_PARENT
                node = nodes[genotype] = {
                    "id": genotype, "parent": parent,
                    "born": when, "died": when, "peak": 0, "span": 0,
                }
            node["died"] = when

        for genotype, count in here.items():
            node = nodes[genotype]
            if count > node["peak"]:
                node["peak"] = count
            node["span"] += 1

    total = len(nodes)
    ranked = sorted(nodes.values(), key=lambda n: (n["span"], n["peak"]), reverse=True)
    kept = ranked[:max(1, limit)] if total else []

    return {
        "nodes": kept,
        "total": total,
        "shown": len(kept),
        "longestSpan": max((n["span"] for n in ranked), default=0),
        "firstIteration": first if first is not None else 0,
        "lastIteration": last if last is not None else 0,
    }
