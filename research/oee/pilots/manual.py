"""
Hand-written strategies in place of brains, and an invasion test.

A strategy reads exactly what a brain reads — the input rows X, one column per
candidate, column 0 the agent itself — and writes exactly what a brain writes:
the output rows Y that the engine turns into decisions. So it plugs into the
world where a brain does: it sees only what a brain sees, and acts only through
what a brain decides. But it may compute what no brain can. A B2 brain scores
each candidate's column on its own, so no B2 brain can write news of a third
agent into a message, as the factory does (book/pages/what-a-brain-can-compute.md).

Teammates recognise each other by what they write: the first four numbers of
every message a strategy sends are a fixed signature (an evolved brain could
write the same; it is a password, not a privilege). The fifth carries a number
a strategy wants its teammates to know, such as its distance from the frontier.

The invasion test: take an evolved world from a run's archive, give a
contiguous patch of agents (the nearest few per cent of a random agent) a
strategy, play on under the world's own rules, and follow the share of agents
the strategy holds. A control patch gets frozen copies of the world's own
brains (the same brains, marked and not mutated), so a strategy is judged
against what a patch of the residents would do from the same place.

    python3 manual.py RUN --era 2000 --strategies factory,turtle,raider,resident \\
        --seeds 1,2 --T 300 --patch 0.05 --jobs 3 --out out.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict

sys.path.insert(0, "/home/stefan/Documents/CodingProjects/GraphOfLife Claude")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# The strategies live in the engine now (gol_strategies.py), where the lab's trials plant them;
# these names keep this file's command line, and anything that imported them, as they were.
from gol_strategies import (CAP, SIGN, SIGNATURE, View, Writer, atanh, factory, lone,  # noqa: E402,F401
                            raider, turtle)
from gol_strategies import RULES as STRATEGIES                                        # noqa: E402
from gol_strategies import STRATEGY_BASE as MANUAL_ID                                 # noqa: E402
from gol_strategies import StrategyBrain as ManualBrain                               # noqa: E402,F401
from gol_strategies import patch_of, plant                                            # noqa: E402,F401


# ---------------------------------------------------------------------------
# The invasion test
# ---------------------------------------------------------------------------

def invade(task) -> Dict[str, Any]:
    run, era, strategy, seed, T, share = task
    import gol_store as store
    from gol_config import SimConfig
    from GraphOfLifeSimple import GraphOfLife
    cfg = SimConfig.from_dict(store.load_meta(run)["config"])
    cfg.extinction_threshold = 0
    world = GraphOfLife.from_checkpoint(store.load_archive(run, era), cfg)
    world.rng = np.random.RandomState(seed)
    patch = plant(world, strategy, share, seed)
    series = []
    for t in range(T + 1):
        if t:
            world.step(record_decisions=False)
        ids = list(world.G.nodes())
        mine = [u for u in ids if world.brains[u].brain_id >= MANUAL_ID]
        tok = sum(world.tokens[u] for u in mine)
        series.append({"t": t, "agents": len(ids), "mine": len(mine), "tokens": int(tok)})
        if not mine or not ids:
            break
    return {"run": run, "era": era, "strategy": strategy, "seed": seed, "patch": len(patch), "series": series}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--era", type=int, default=2000)
    ap.add_argument("--strategies", default="factory,turtle,raider,resident")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--patch", type=float, default=0.05)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    tasks = [(r, a.era, s, int(sd), a.T, a.patch) for r in a.runs for s in a.strategies.split(",")
             for sd in a.seeds.split(",")]
    with ProcessPoolExecutor(a.jobs) as ex:
        results = list(ex.map(invade, tasks))
    for r in results:
        s = r["series"]
        pts = " ".join(f"{x['t']}:{x['mine'] / max(1, x['agents']):.3f}" for x in s[:: max(1, len(s) // 6)])
        print(f"{r['run']} {r['strategy']:9s} seed {r['seed']} patch {r['patch']}: share {pts} | end {s[-1]['mine']}/{s[-1]['agents']}")
    if a.out:
        json.dump(results, open(a.out, "w"))


if __name__ == "__main__":
    main()
