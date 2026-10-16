"""
How brains spread in Experiment 17's worlds and in B2's: by children, and by
nodes taken in the game.

A node taken in the game gets a copy of its winner's brain (the engine's
ALLOCATE_AND_CONQUER), so a brain spreads by every node it takes as well as by
every child it has. Where brains are mixed at birth, a child's brain is half a
neighbour's; a taken node's is the winner's own.

Every 10th game from iteration 500 to the end of Experiment 17 (10,000) in its
ten worlds, and from 500 to 2,000 in B2's ten (Experiment 16), read from the
game's frame: the nodes whose winner is not the agent itself, per hundred
agents. B2's continuations to 10,000 (L/B2-long-sNNN) count the same in their
series (`conquests`). Births per hundred agents from the same stretch of each
run's record. Writes GraphOfLifeRuns/.pilots/e17_conquests.json, in windows of
500 iterations.

    python3 research/oee/checks/e17_conquests.py
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
PILOTS = os.path.join(ROOT, "GraphOfLifeRuns", ".pilots")
OUT = os.path.join(PILOTS, "e17_conquests.json")
EVERY, WINDOW = 10, 500


def lab(args) -> List[Dict[str, float]]:
    """One lab run's windows: nodes taken and births, each per hundred agents per iteration."""
    run, start, end = args
    import gol_record
    import gol_store
    births = {r["iteration"]: (r["births"], r["nodes_before"]) for r in gol_record.read_stats(run)
              if r.get("phase") == 1}
    frames = gol_store.count_frames(run)
    out = []
    for w in range(start, end, WINDOW):
        taken = agents = 0
        # Two frames an iteration, the game's second: iteration t's game is frame 2t + 1.
        for t in range(w, min(w + WINDOW, end), EVERY):
            if 2 * t + 1 >= frames:
                break
            frame = gol_store.read_frame(run, 2 * t + 1)
            winners = (frame.get("decisions") or {}).get("winners") or []
            taken += sum(1 for x in winners if x["winner"] != x["node"])
            agents += len(winners)
        born = [births[t] for t in range(w, min(w + WINDOW, end)) if t in births]
        if agents and born:
            out.append({"from": w, "taken_per_100": 100 * taken / agents,
                        "births_per_100": 100 * sum(b for b, _ in born) / sum(n for _, n in born)})
    return out


def pilot(path: str, start: int, end: int) -> List[Dict[str, float]]:
    rows = [json.loads(line) for line in open(os.path.join(path, "series.jsonl"))]
    out = []
    for w in range(start, end, WINDOW):
        # A pilot's row `it` is the step that ends at it: the lab's iteration it - 1.
        mine = [r for r in rows if w < r["it"] <= w + WINDOW]
        n = sum(r["nodes"] for r in mine)
        if n:
            out.append({"from": w, "taken_per_100": 100 * sum(r.get("conquests", 0) for r in mine) / n,
                        "births_per_100": 100 * sum(r.get("births", 0) for r in mine) / n})
    return out


def main():
    mixed = [(f"B2-10000-4af4a3-s{s:03d}", 500, 10000) for s in range(1, 11)]
    b2 = [(f"B2-10000-s{s:03d}", 500, 2000) for s in range(1, 11)]
    with ProcessPoolExecutor(max_workers=2) as pool:
        found = dict(zip([r for r, _, _ in mixed + b2], pool.map(lab, mixed + b2)))
    out = {"mixed": {r: found[r] for r, _, _ in mixed},
           "B2": {r: found[r] + pilot(os.path.join(PILOTS, "L", f"B2-long-s{int(r[-3:]):03d}"), 2000, 10000)
                  for r, _, _ in b2}}
    with open(OUT + ".tmp", "w") as f:
        json.dump(out, f, indent=1)
    os.replace(OUT + ".tmp", OUT)
    for kind, worlds in out.items():
        taken = [w["taken_per_100"] for rows in worlds.values() for w in rows]
        born = [w["births_per_100"] for rows in worlds.values() for w in rows]
        print(f"{kind}: nodes taken per 100 agents per iteration, median {np.median(taken):.2f} "
              f"({min(taken):.2f}-{max(taken):.2f}); births {np.median(born):.2f}")


if __name__ == "__main__":
    main()
