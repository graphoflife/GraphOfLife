"""
What a crew of raiders meets in its first game (Chapter 102), read off the
worlds as Experiment 33 planted them; no world is played.

For every planting of 5% of a world (B2's five worlds played on, at 2,000,
5,000 and 10,000, places 1 to 3) the patch is drawn as gol_strategies.plant
draws it, and each patch agent's choice is worked out as the crew's rule
makes it: the weakest stranger next door, if it is no richer than the agent,
gets everything the agent may stake (all but its home share, B2's
keep_home). Raiders that chose the same stranger are one mob. Each mob is
then judged against the stranger's own stake at home, weighed as the game
weighs it (incumbency):

  sure      the mob outweighs the stranger even if it kept everything at home
  maybe     it outweighs only what the stranger must keep at home
  lost      it cannot outweigh even that; its tokens go to the stranger

leaving out what other neighbours stake on the stranger. Also counted: how
many patch agents the lone raider's rule would send out (it stakes only
where it alone outweighs everything the stranger holds), how many strangers
border each patch, how many patch agents each of them borders, and how many
patch agents hold one token (and so can stake nothing) or the 30 a crafted
raider needs before it has a child.

    python3 research/oee/strategies/mobs.py [OUT.json]
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

import GraphOfLifeSimple  # noqa: E402
import gol_store  # noqa: E402
import gol_strategies  # noqa: E402
from gol_config import SimConfig  # noqa: E402

OUT = os.path.join(ROOT, "GraphOfLifeRuns", ".pilots", "atlas", "e33_mobs.json")
WORLDS = [f"pilot:L/B2-long-s{s:03d}" for s in range(1, 6)]
ERAS = (2000, 5000, 10000)
PLACES = (1, 2, 3)
SIZE = 0.05
CHILD_AT = 30           # the crafted raiders' rule: a child only when holding this many


def kept(tokens: int, share: float) -> int:
    """What an agent must leave on its own node (the engine's keep_home)."""
    return max(1, int(math.ceil(share * tokens))) if share > 0 else 0


def planting(world, cfg, place: int) -> Dict[str, Any]:
    patch = set(gol_strategies.patch_of(world, SIZE, np.random.RandomState(1000 + place)))
    tok, G, h, share = world.tokens, world.G, cfg.incumbency, cfg.keep_home
    mobs: Dict[int, List[int]] = defaultdict(list)
    lone = frontier = 0
    borders: Counter = Counter()
    for u in patch:
        strangers = sorted(x for x in G[u] if x not in patch)
        borders.update(strangers)
        if not strangers:
            continue
        frontier += 1
        j = min(strangers, key=lambda x: (tok[x], x))       # the first weakest, as the brain's columns are ordered
        if 0.7 * tok[u] > h * tok[j]:
            lone += 1
        stake = tok[u] - kept(tok[u], share)
        if tok[j] <= tok[u] and stake > 0:
            mobs[j].append(stake)
    outcome = {"sure": [0, 0], "maybe": [0, 0], "lost": [0, 0]}         # strangers, tokens staked
    for j, stakes in mobs.items():
        total = sum(stakes)
        key = "sure" if total > h * tok[j] else "maybe" if total > h * kept(tok[j], share) else "lost"
        outcome[key][0] += 1
        outcome[key][1] += total
    held = np.array([tok[u] for u in patch])
    return {"patch": len(patch), "frontier": frontier, "wealth": float(held.mean()),
            "one_token": int((held <= 1).sum()), "can_breed": int((held >= CHILD_AT).sum()),
            "richest": int(held.max()), "lone_raiders_staking": lone,
            "crew_staking": sum(len(s) for s in mobs.values()), "targets": len(mobs),
            "mob_sizes": dict(Counter(len(s) for s in mobs.values())),
            "outcome": outcome, "strangers": len(borders),
            "patch_agents_per_stranger": float(np.mean(list(borders.values())))}


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else OUT
    rows = []
    for run in WORLDS:
        cfg = SimConfig.from_dict(gol_store.source_config(run))
        for era in ERAS:
            kept_world = gol_store.kept_world(run, era)
            for place in PLACES:
                world = GraphOfLifeSimple.GraphOfLife.from_checkpoint(kept_world, cfg)
                rows.append({"run": run, "era": era, "place": place, **planting(world, cfg, place)})
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    for era in ERAS:
        mine = [r for r in rows if r["era"] == era]
        print(era, {k: sum(r[k] for r in mine) for k in ("patch", "frontier", "one_token", "can_breed",
                                                         "crew_staking", "targets", "lone_raiders_staking")},
              {k: [sum(r["outcome"][k][i] for r in mine) for i in (0, 1)] for k in ("sure", "maybe", "lost")})


if __name__ == "__main__":
    main()
