"""
Why a mixed world of Experiment 17 is rich: who could have a child, and who has one.

The engine lets an agent have a child when it holds 11 tokens or more (a child
takes at least min_child = 10, and the parent keeps one), has fewer links than
its valence of 8, and its brain asks for 10 tokens or more. At every whole
world kept, this counts the agents that pass the first two — the agents that
*could* have a child — and the births of the next 50 reproduction phases:
births per iteration per such agent is how eagerly the able breed.

Worlds read: Experiment 17's ten (archives every 500 iterations); B2's ten
(Experiment 16's archives to 2,000, their continuations' checkpoints
L/B2-long-sNNN every 500 after); and the worlds of batch SW, where mixing was
switched off or on at a moment (checkpoints every 250), when they exist.
Writes GraphOfLifeRuns/.pilots/e17_breeding.json.

A lab run's stats row of iteration t is the step from the world kept at t to
the next; a pilot's series row `it` is the step that ends at it, so the same
step is `it - 1` there.
"""
from __future__ import annotations

import glob
import json
import os
import re
from typing import Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUNS = os.path.join(ROOT, "GraphOfLifeRuns")
PILOTS = os.path.join(RUNS, ".pilots")
OUT = os.path.join(PILOTS, "e17_breeding.json")
AHEAD = 50
MIXED = [f"B2-10000-4af4a3-s{s:03d}" for s in range(1, 11)]


def able(path: str) -> Dict[str, int]:
    """Agents, those with 11 tokens or more, and those that also have fewer than 8 links, in one whole world."""
    with np.load(path) as z:
        tokens, edges, n = z["tokens"], z["edges"], len(z["ids"])
    # The edges name agents by their place in the list of agents.
    degree = np.bincount(edges.ravel(), minlength=n) if len(edges) else np.zeros(n, int)
    return {"agents": int(n), "rich": int((tokens >= 11).sum()), "able": int(((tokens >= 11) & (degree < 8)).sum())}


def lab_births(run: str) -> Dict[int, int]:
    out = {}
    for line in open(os.path.join(RUNS, run, "stats.jsonl")):
        r = json.loads(line)
        if r.get("phase") == 1:
            out[r["iteration"]] = r["births"]
    return out


def pilot_births(d: str) -> Dict[int, int]:
    return {r["it"] - 1: r["births"] for r in map(json.loads, open(os.path.join(d, "series.jsonl")))}


def rows(worlds: Dict[int, str], births: Dict[int, int]) -> List[Dict[str, float]]:
    """One row per whole world: its agents, the able, and the births per iteration of the next AHEAD steps."""
    out = []
    for t in sorted(worlds):
        later = [births[s] for s in range(t, t + AHEAD) if s in births]
        if len(later) < AHEAD:
            continue
        r = {"at": t, **able(worlds[t]), "births": float(np.mean(later))}
        r["per_able"] = r["births"] / r["able"] if r["able"] else None
        out.append(r)
    return out


def _kept(pattern: str, every: int, start: int = 1) -> Dict[int, str]:
    found = {int(re.search(r"_(\d+)\.npz$", p).group(1)): p for p in glob.glob(pattern)}
    return {t: p for t, p in found.items() if t >= start and t % every == 0}


def mixed(run: str) -> List[Dict[str, float]]:
    """One of Experiment 17's worlds, to its end at 10,000 (a later experiment may carry the run further)."""
    worlds = _kept(os.path.join(RUNS, run, "archive", "world_*.npz"), 500)
    return rows({t: p for t, p in worlds.items() if t < 10000}, lab_births(run))


def b2(seed: int) -> List[Dict[str, float]]:
    """B2's world of this seed: the lab's to 2,000, its continuation after."""
    run, long = f"B2-10000-s{seed:03d}", os.path.join(PILOTS, "L", f"B2-long-s{seed:03d}")
    worlds = _kept(os.path.join(RUNS, run, "archive", "world_*.npz"), 500)
    worlds.update({t: p for t, p in _kept(os.path.join(long, "ckpt_*.npz"), 500, 2001).items() if t <= 10000})
    births = lab_births(run)
    births.update({t: b for t, b in pilot_births(long).items() if t >= 2000})
    return rows(worlds, births)


def switched(d: str) -> List[Dict[str, float]]:
    return rows(_kept(os.path.join(d, "ckpt_*.npz"), 250), pilot_births(d))


def main():
    out = {"mixed": {run: mixed(run) for run in MIXED}, "B2": {f"B2-10000-s{s:03d}": b2(s) for s in range(1, 11)},
           "switched": {os.path.basename(d): switched(d) for d in sorted(glob.glob(os.path.join(PILOTS, "SW", "*")))
                        if os.path.exists(os.path.join(d, "series.jsonl"))}}
    with open(OUT + ".tmp", "w") as f:
        json.dump(out, f, indent=1)
    os.replace(OUT + ".tmp", OUT)
    for kind in ("mixed", "B2", "switched"):
        print(kind, "— iteration/100: agents / share able / births per 100 able per iteration")
        for name, rs in out[kind].items():
            print(f"  {name[-12:]:>12}", " ".join(f"{r['at'] // 100:>3}:{r['agents']:5d}/{r['able'] / r['agents']:.2f}/"
                                                 f"{100 * (r['per_able'] or 0):5.2f}" for r in rs[::2]))


if __name__ == "__main__":
    main()
