"""
Is a world a space? Chapter 24's two rulers — the dimension from how balls grow,
and from how a random walk spreads — on worlds of every kind, each beside its
random twin (the same number of links at every agent, wired at random).

Chapter 24 read B1's worlds: a plateau by balls (2.3 at 10,000 tokens, 2.8 at
100,000, 3.3 at 409,600), 1.8 to 1.9 by walks, and twins that never settle.
Meta XI found that in B2's worlds the reading by balls (the health bar's, over
radii 1 to 5) rises with the number of agents, as a small world's would. This
reads B2's worlds, the mixed ones and B1's at 10,000 tokens with the full rulers.

Worlds are read from their whole worlds kept (a lab run's archive/world_<t>.npz,
a pilot's ckpt_<t>.npz); B1's lab runs of 10,000 tokens keep none, and are read
from their last frame, as Chapter 24 read them.

With `--at 1000`, every world Meta X's reading at that iteration holds
(GraphOfLifeRuns/.pilots/atlas/at1000.json, research/atlas/at.py) is read
instead, at that iteration, into space_at1000.json; worlds already read are
kept.

    python3 research/atlas/space.py [--jobs 2] [--at 1000]   writes GraphOfLifeRuns/.pilots/atlas/space.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
RUNS = os.path.join(ROOT, "GraphOfLifeRuns")
PILOTS = os.path.join(RUNS, ".pilots")
OUT = os.path.join(PILOTS, "atlas", "space.json")


def kept(family: str, at: int) -> List[Tuple[str, str]]:
    """A lab family's runs and their whole world kept at iteration `at`."""
    out = []
    for d in sorted(glob.glob(os.path.join(RUNS, f"{family}-s[0-9][0-9][0-9]"))):
        path = os.path.join(d, "archive", f"world_{at:06d}.npz")
        if os.path.exists(path):
            out.append((os.path.basename(d), path))
    return out


def pilots(pattern: str, at: int) -> List[Tuple[str, str]]:
    out = []
    for d in sorted(glob.glob(os.path.join(PILOTS, pattern))):
        path = os.path.join(d, f"ckpt_{at:06d}.npz")
        if os.path.exists(path):
            out.append((os.path.relpath(d, PILOTS), path))
    return out


#: Which worlds, as (group, base, [(id, source)]): a source is a whole world's file, or "frame" for a lab run
#: read from its last frame.
def worlds() -> List[Tuple[str, str, str, str]]:
    groups = [
        ("B2, 2,000 iterations (Experiment 16)", "B2", kept("B2-10000", 2000)),
        ("B2, 10,000 iterations (its worlds played on)", "B2", pilots("L/B2-long-s0[0-9][0-9]", 10000)),
        ("B2 at 5,000 tokens (Experiment 14)", "B2", kept("B2-5000", 1000)),
        ("mixed at birth, 10,000 iterations (Experiment 17)", "B2", kept("B2-10000-4af4a3", 10000)),
        ("mixed at birth, 50,000 iterations (Experiment 18)", "B2", kept("B2-10000-4af4a3", 50000)),
        ("B2, estates to the newborns or parents (batch R)", "B2",
         pilots("R/B2-newborns-s[12]", 2000) + pilots("R/B2-parents-s[12]", 2000)),
        ("B1 (Experiment 2)", "B1", [(os.path.basename(d), "frame")
                                     for d in sorted(glob.glob(os.path.join(RUNS, "B1-10000-s[0-9][0-9][0-9]")))]),
    ]
    return [(g, base, wid, src) for g, base, items in groups for wid, src in items]


def at_worlds(at: int) -> List[Tuple[str, str, str, str]]:
    """Every world of Meta X's reading at one iteration, with its whole world kept then."""
    out = []
    for w in json.load(open(os.path.join(PILOTS, "atlas", f"at{at}.json")))["worlds"]:
        if "bar" not in w:
            continue
        pilot = os.path.join(PILOTS, w["id"], f"ckpt_{at:06d}.npz")
        lab = os.path.join(RUNS, w["id"], "archive", f"world_{at:06d}.npz")
        source = pilot if os.path.exists(pilot) else lab
        if os.path.exists(source):
            out.append((w["family"], "", w["id"], source))
    return out


def network(wid: str, source: str) -> Optional[Dict[int, set]]:
    if source == "frame":
        import book_data as D
        from book_graph import graph
        if D.last_iteration(wid) < 2000:
            return None  # a world that died before its end
        return graph(D.frame_at(wid, D.last_iteration(wid), 2))
    with np.load(source) as z:
        n, edges = len(z["ids"]), z["edges"]
    # The edges name agents by their place in the list of agents.
    adj: Dict[int, set] = {u: set() for u in range(n)}
    for a, b in edges:
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))
    return adj


def one(item: Tuple[str, str, str, str]) -> Optional[Dict[str, Any]]:
    group, base, wid, source = item
    from book_chapters.space import measure
    from book_graph import degree_preserving
    adj = network(wid, source)
    if not adj or len(adj) < 100:
        return None
    return {"id": wid, "group": group, "base": base, "world": measure(adj, 23),
            "twin": measure(degree_preserving(adj, 23), 23)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--at", type=int, help="every world of atlas/at<AT>.json at that iteration")
    a = ap.parse_args()
    out = OUT if a.at is None else os.path.join(PILOTS, "atlas", f"space_at{a.at}.json")
    todo = worlds() if a.at is None else at_worlds(a.at)
    done = {}
    if a.at is not None and os.path.exists(out):
        done = {r["id"]: r for r in json.load(open(out))}
    print(f"{len(todo)} worlds, {sum(1 for t in todo if t[2] not in done)} to read", flush=True)
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        fresh = {r["id"]: r for r in pool.map(one, [t for t in todo if t[2] not in done]) if r}
    rows = [r for r in (done.get(t[2]) or fresh.get(t[2]) for t in todo) if r]
    with open(out + ".tmp", "w") as f:
        json.dump(rows, f)
    os.replace(out + ".tmp", out)
    print(f"{len(rows)} worlds measured", flush=True)


if __name__ == "__main__":
    main()
