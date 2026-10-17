#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What folds a world, with no brains at all (Chapter 87).

Chapter 87 found B2's worlds folded into near small worlds once their agents
die of age, each death mended with a ring of links, and two readings of why:
who breeds (where the old breed, they gather links till they die) and the ring
a death closes. Here the rules of space alone (research/oee/space.py, Chapter
48's model) are played with every agent dying at age 100 as in B2, and the
breeding left to any agent, to the young only or to the old only, with deaths
mended in a ring or not at all; and, beside them, Chapter 48's own model, whose
agents die by chance at any age. Each world is read at the end as Chapter 87
reads every world: its typical distance over its random twin's, and the groups
the neighbours of its oldest agents fall into.

    python3 research/oee/fold.py [--seeds 5] [--jobs 2]     writes GraphOfLifeRuns/.pilots/atlas/fold.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

#: Each condition: its name, and the model's settings (links at birth 2, a world of about 1,500 agents).
CONDITIONS = [
    ("die at 100, mended in a ring, anyone breeds", dict(age=100, mend=True, ring=True, breeders="any")),
    ("die at 100, mended in a ring, the young breed", dict(age=100, mend=True, ring=True, breeders="young")),
    ("die at 100, mended in a ring, the old breed", dict(age=100, mend=True, ring=True, breeders="old")),
    ("die at 100, not mended, anyone breeds", dict(age=100, mend=False, ring=False, breeders="any")),
    ("die by chance, mended in a ring (Chapter 48)", dict(age=0, mend=True, ring=True, breeders="any")),
]
STEPS = 600


def _load(name: str, path: str):
    """A module by its file: the model and the measure are both called space.py."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def one(item) -> Dict[str, Any]:
    name, settings, seed = item
    model = _load("space_model", os.path.join(HERE, "space.py"))
    atlas = _load("space_atlas", os.path.join(ROOT, "research", "atlas", "space.py"))
    from book_chapters.space import measure
    from book_graph import degree_preserving
    G = model.grow(links=2, steps=STEPS, seed=seed, **settings)
    adj = {u: set(G[u]) for u in G.nodes()}
    row = {"condition": name, "seed": seed, "agents": G.number_of_nodes(),
           "links_per_agent": 2 * G.number_of_edges() / max(1, G.number_of_nodes()),
           "typical_distance": atlas.typical_distance({"world": measure(adj, 23),
                                                       "twin": measure(degree_preserving(adj, 23), 23)}),
           "worst_cut": model.measure(G)["nodeCut"]}
    # The groups the neighbours of the oldest tenth fall into: the ring their deaths will close.
    ages = {u: G.graph["steps"] - b for u, b in G.graph["born"].items()}
    old = [u for u, a in ages.items() if a >= np.quantile(list(ages.values()), 0.9)]
    row["ring_of_the_oldest"] = float(np.mean([len(model.groups_of(G, list(G[u]))) for u in old]))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    todo = [(name, settings, seed) for name, settings in CONDITIONS for seed in range(a.seeds)]
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        rows = list(pool.map(one, todo))
    out = os.path.join(ROOT, "GraphOfLifeRuns", ".pilots", "atlas", "fold.json")
    with open(out + ".tmp", "w") as f:
        json.dump(rows, f, indent=1)
    os.replace(out + ".tmp", out)
    for name, _ in CONDITIONS:
        mine = [r for r in rows if r["condition"] == name]
        print(f"{name:48s} typical distance {np.median([r['typical_distance'] for r in mine]):.2f} "
              f"({min(r['typical_distance'] for r in mine):.2f}-{max(r['typical_distance'] for r in mine):.2f}), "
              f"links {np.median([r['links_per_agent'] for r in mine]):.2f}, "
              f"ring {np.median([r.get('ring_of_the_oldest', np.nan) for r in mine]):.1f}, "
              f"worst cut {np.median([r['worst_cut'] for r in mine]):.3f}", flush=True)


if __name__ == "__main__":
    main()
