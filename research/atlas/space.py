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

One number sums a world up: its **typical distance**, the steps within which
a typical agent's ball holds half the world, over its twin's. A space is far
wider than its twin (its distances grow as a power of its size, the twin's as
its logarithm); a small world is about as wide.

With `--at 1000`, every world Meta X's reading at that iteration holds
(GraphOfLifeRuns/.pilots/atlas/at1000.json, research/atlas/at.py) is read
instead, at that iteration, into space_at1000.json; worlds already read are
kept. With `--life`, a few worlds through their lives (LIFE), into
space_life.json.

    python3 research/atlas/space.py [--jobs 2] [--at 1000 | --life]   writes GraphOfLifeRuns/.pilots/atlas/space.json
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


#: A world to read: its group, its base (B1 or B2, where known), its id, the file of its whole world (or "frame"
#: for a lab run read from its last frame), and the iteration it is read at (None: its end).
Item = Tuple[str, str, str, str, Optional[int]]


def worlds() -> List[Item]:
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
    return [(g, base, wid, src, None) for g, base, items in groups for wid, src in items]


def at_worlds(at: int) -> List[Item]:
    """Every world of Meta X's reading at one iteration, with its whole world kept then."""
    out = []
    for w in json.load(open(os.path.join(PILOTS, "atlas", f"at{at}.json")))["worlds"]:
        if "bar" not in w:
            continue
        pilot = os.path.join(PILOTS, w["id"], f"ckpt_{at:06d}.npz")
        lab = os.path.join(RUNS, w["id"], "archive", f"world_{at:06d}.npz")
        source = pilot if os.path.exists(pilot) else lab
        if os.path.exists(source):
            out.append((w["family"], "", w["id"], source, at))
    return out


#: Worlds read through their lives, by group: lab families (whole worlds kept every 100 iterations) and pilots
#: (checkpoints every 100), seeds 1 to 3 where there are three.
LIFE_MOMENTS = (100, 200, 300, 500, 700, 1000, 1500, 2000)
LIFE = [
    ("B2", [f"B2-10000-s00{s}" for s in (1, 2, 3)]),
    ("B2, no old age", [f"B2-10000-3ec24f-s00{s}" for s in (1, 2, 3)]),
    ("B2, estates to the newborns", ["R/B2-newborns-s1", "R/B2-newborns-s2"]),
    ("B2, estates to the parents", ["R/B2-parents-s1", "R/B2-parents-s2"]),
    ("B2, every unused link lapses", ["S/S4-lapse-s1", "S/S4-lapse-s2"]),
    ("B2 at 5,000 tokens", [f"B2-5000-s00{s}" for s in (1, 2, 3)]),
    ("B2 at 5,000, no mending", [f"B2-5000-d6b607-s00{s}" for s in (1, 2, 3)]),
    ("B2 at 5,000, no links at birth", [f"B2-5000-28aef0-s00{s}" for s in (1, 2, 3)]),
    ("B2 at 5,000, no cap on links", [f"B2-5000-5c8c75-s00{s}" for s in (1, 2, 3)]),
    ("B2 at 5,000, handover", [f"B2-5000-894419-s00{s}" for s in (1, 2, 3)]),
]


def life_worlds() -> List[Item]:
    out = []
    for group, ids in LIFE:
        for wid in ids:
            for t in LIFE_MOMENTS:
                lab = os.path.join(RUNS, wid, "archive", f"world_{t:06d}.npz")
                pilot = os.path.join(PILOTS, wid, f"ckpt_{t:06d}.npz")
                source = lab if os.path.exists(lab) else pilot
                if os.path.exists(source):
                    out.append((group, "", wid, source, t))
    return out


def half_reach(balls: List[float], n: int) -> float:
    """The radius at which the typical ball first holds half the world, between whole steps by the
    logarithm of the ball."""
    b = np.asarray(balls, float)
    k = int(np.argmax(b >= n / 2))
    if b[k] < n / 2 or k == 0:
        return float("nan")
    return (k - 1) + (np.log(n / 2) - np.log(b[k - 1])) / (np.log(b[k]) - np.log(b[k - 1]))


def typical_distance(row: Dict[str, Any]) -> float:
    """A world's half-reach over its random twin's: about 1 for a small world, more for a space."""
    return half_reach(row["world"]["balls"], row["world"]["n"]) / half_reach(row["twin"]["balls"], row["twin"]["n"])


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


def one(item: Item) -> Optional[Dict[str, Any]]:
    group, base, wid, source, at = item
    from book_chapters.space import measure
    from book_graph import degree_preserving
    adj = network(wid, source)
    if not adj or len(adj) < 100:
        return None
    return {"id": wid, "group": group, "base": base, "at": at, "world": measure(adj, 23),
            "twin": measure(degree_preserving(adj, 23), 23)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--at", type=int, help="every world of atlas/at<AT>.json at that iteration")
    ap.add_argument("--life", action="store_true", help="the worlds of LIFE through their lives")
    a = ap.parse_args()
    if a.life:
        out, todo = os.path.join(PILOTS, "atlas", "space_life.json"), life_worlds()
    elif a.at is not None:
        out, todo = os.path.join(PILOTS, "atlas", f"space_at{a.at}.json"), at_worlds(a.at)
    else:
        out, todo = OUT, worlds()
    # A world read before at the same moment is not read again.
    done = {(r["id"], r.get("at", a.at)): r for r in json.load(open(out))} if out != OUT and os.path.exists(out) else {}
    wanted = [t for t in todo if (t[2], t[4]) not in done]
    print(f"{len(todo)} worlds, {len(wanted)} to read", flush=True)
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        fresh = {(r["id"], r["at"]): r for r in pool.map(one, wanted) if r}
    rows = [r for r in (done.get((t[2], t[4])) or fresh.get((t[2], t[4])) for t in todo) if r]
    with open(out + ".tmp", "w") as f:
        json.dump(rows, f)
    os.replace(out + ".tmp", out)
    print(f"{len(rows)} worlds measured", flush=True)


if __name__ == "__main__":
    main()
