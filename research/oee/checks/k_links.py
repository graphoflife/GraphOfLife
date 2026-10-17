"""
Links to strangers (batch K): how much the agents use the search, and what it does to their world.

For every world of batch K and for B2 itself with the same seeds (P/B2-s1, P/B2-s2):

- from 50 games played on from its checkpoint at 1,000 and recorded in the lab's
  form (pilot_frames.record_engine): the links proposed to strangers per hundred
  agents per iteration, the share of them that were made, and how far apart in
  the network the two ends were just before (in the previous game's graph);
- from its series (a row of network measures every 50 iterations, 500 to
  1,000): the dimension, the mean path length, the best-connected agent's share
  of the others, links per agent, the 2-core, the agents alive, the Gini of tokens;
- from its checkpoint at 1,000: the share of the agents holding three tokens or fewer.

Writes GraphOfLifeRuns/.pilots/k_links.json.

    python3 research/oee/checks/k_links.py [--jobs 2]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research", "oee", "checks"))
sys.path.insert(0, os.path.join(ROOT, "research", "oee", "pilots"))
PILOTS = os.path.join(ROOT, "GraphOfLifeRuns", ".pilots")
OUT = os.path.join(PILOTS, "k_links.json")
AT, GAMES, FAR = 1000, 50, 20
NETWORK = ("dimension", "meanPathLength", "maxDegreeShare", "meanDegree", "coreShare", "nodes", "gini")


def neighbours(frame: Dict[str, object]) -> Dict[int, List[int]]:
    nbrs: Dict[int, List[int]] = {u: [] for u in frame["ids"]}
    for u, v in frame["edges"]:
        nbrs[u].append(v)
        nbrs[v].append(u)
    return nbrs


def distance(nbrs: Dict[int, List[int]], a: int, b: int) -> Optional[int]:
    """Steps from a to b in a frame's graph (FAR or more counts as FAR); None if either is not in it."""
    if a not in nbrs or b not in nbrs:
        return None
    seen, queue = {a: 0}, deque([a])
    while queue:
        u = queue.popleft()
        if seen[u] >= FAR:
            break
        for v in nbrs[u]:
            if v not in seen:
                seen[v] = seen[u] + 1
                if v == b:
                    return seen[v]
                queue.append(v)
    return FAR


def one(pilot: str) -> Dict[str, object]:
    import pilot_frames
    name = os.path.relpath(pilot, PILOTS)
    out: Dict[str, object] = {"world": name}
    rows = [json.loads(line) for line in open(os.path.join(pilot, "series.jsonl"))]
    late = [r for r in rows if 500 <= r["it"] <= AT and "dimension" in r]
    for key in NETWORK:
        values = [r[key] for r in late if r.get(key) is not None]
        out[key] = float(np.mean(values)) if values else None
    with np.load(os.path.join(pilot, f"ckpt_{AT:06d}.npz")) as z:
        out["share_at_most_three"] = float(np.mean(z["tokens"] <= 3))
    world = pilot_frames.load(pilot, AT)
    _, _, frames = pilot_frames.record_engine(world, GAMES)
    proposed = made = agents = 0
    far: List[int] = []
    for (t, phase), frame in sorted(frames.items()):
        if phase != 1:
            continue
        proposals = (frame.get("decisions") or {}).get("proposals")
        if proposals is None:
            continue
        agents += frame.get("nodes_before") or len(frame["ids"])
        proposed += len(proposals)
        made += sum(1 for _, _, ok in proposals if ok)
        before = frames.get((t - 1, 2))
        if before and proposals:
            nbrs = neighbours(before)
            for u, w, ok in proposals:
                d = distance(nbrs, u, w)
                if d is not None:
                    far.append(d)
    out.update({"proposed_per_100": 100 * proposed / agents if agents else 0.0,
                "made_share": made / proposed if proposed else None,
                "made_per_100": 100 * made / agents if agents else 0.0,
                "distance_median": float(np.median(far)) if far else None,
                "distance_far_share": float(np.mean(np.array(far) >= FAR)) if far else None,
                "proposals": proposed})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    worlds = sorted(glob.glob(os.path.join(PILOTS, "K", "*-s[12]"))) + [os.path.join(PILOTS, "P", f"B2-s{s}")
                                                                       for s in (1, 2)]
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        rows = list(pool.map(one, worlds))
    with open(OUT + ".tmp", "w") as f:
        json.dump(rows, f, indent=1)
    os.replace(OUT + ".tmp", OUT)
    for r in rows:
        print(f"{r['world']:28s} proposed {r['proposed_per_100']:6.2f}/100/it made {r['made_per_100']:5.2f} "
              f"(share {r['made_share'] if r['made_share'] is None else round(r['made_share'], 2)}) "
              f"distance {r['distance_median']} far {r['distance_far_share']}  agents {r['nodes']:.0f} "
              f"dim {r['dimension']:.2f} path {r['meanPathLength']:.2f} hub {r['maxDegreeShare']:.3f} "
              f"deg {r['meanDegree']:.2f} poor {r['share_at_most_three']:.2f}", flush=True)


if __name__ == "__main__":
    main()
