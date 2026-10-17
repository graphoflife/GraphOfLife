"""
How large a ring a death closes: in every world read at one iteration whose agents die of age and are mended
round (B2's kind), the neighbours of the agents within ten iterations of their death, split into groups that are
linked to one another directly — the groups mending joins in a ring (Chapter 87). Set beside each world's typical
distance over its random twin's (research/atlas/space.py).

    python3 research/atlas/rings.py [--at 1000]     writes GraphOfLifeRuns/.pilots/atlas/rings_at1000.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import space as S  # noqa: E402  (research/atlas/space.py)


def groups(adj: List[set], u: int) -> int:
    """The neighbours of u, in groups linked to one another directly: how many links mending adds when u dies
    (one per group, in a ring, when there are three or more)."""
    seen, count = set(), 0
    for v in adj[u]:
        if v in seen:
            continue
        count += 1
        stack = [v]
        seen.add(v)
        while stack:
            w = stack.pop()
            for x in adj[w] & adj[u]:
                if x not in seen:
                    seen.add(x)
                    stack.append(x)
    return count


def one(wid: str, source: str, at: int) -> Optional[Dict[str, Any]]:
    cfg = S.settings(wid)
    max_age = cfg.get("max_age") or 0
    if not cfg.get("mend_on_death") or not max_age:
        return None
    with np.load(source) as z:
        n, e, born = len(z["ids"]), z["edges"], z["born_at"]
    adj = [set() for _ in range(n)]
    for a, b in e:
        adj[a].add(b)
        adj[b].add(a)
    dying = np.flatnonzero(at - born >= max_age - 10)
    if len(dying) < 10:
        return None
    rings = [groups(adj, u) for u in dying]
    return {"id": wid, "dying": int(len(dying)), "ring": float(np.mean(rings)),
            "links_of_dying": float(np.mean([len(adj[u]) for u in dying])), "links_per_agent": 2 * len(e) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--at", type=int, default=1000)
    a = ap.parse_args()
    rows = []
    for group, base, wid, source, at in S.at_worlds(a.at):
        r = one(wid, source, at)
        if r:
            rows.append(r)
    out = os.path.join(S.PILOTS, "atlas", f"rings_at{a.at}.json")
    with open(out + ".tmp", "w") as f:
        json.dump(rows, f, indent=1)
    os.replace(out + ".tmp", out)
    print(f"{len(rows)} worlds", flush=True)


if __name__ == "__main__":
    main()
