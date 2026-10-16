"""
Experiment 15's thesis, checked as its plan wrote it down before any run:

    Median over iterations 300 to 1,000 of the share of agents holding one
    token (from the frames): below 0.5 in at least 7 of the 10 B2 worlds, above
    0.67 in at least 7 of the 10 worlds with keep_home 0.01; and at least 5 of
    the 10 worlds with keep_home 0 have fewer than 100 agents after some game.

The share is read from each game's frame (phase 2), every iteration from 300
to 1,000; the other conditions are measured the same way for the chapter.
"Fewer than 100 agents after some game" is the smallest count of agents in
any game's row of a world's stats.

    python3 research/oee/checks/e15_thesis.py     writes GraphOfLifeRuns/.pilots/e15_thesis.json
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "GraphOfLifeRuns", ".pilots", "e15_thesis.json")
CONDITIONS = {"B2": "B2-10000", "the last token kept": "B2-10000-054b50", "nothing kept": "B2-10000-575a1d",
              "half kept": "B2-10000-df2615", "children of any size": "B2-10000-185eb7",
              "no old age": "B2-10000-3ec24f"}
FROM, TO = 300, 1000


def measure(run: str):
    import gol_record
    import gol_store
    shares = []
    # Two frames an iteration, the game's second: iteration t's game is frame 2t + 1.
    for t in range(FROM, TO + 1):
        if 2 * t + 1 >= gol_store.count_frames(run):
            break
        frame = gol_store.read_frame(run, 2 * t + 1)
        assert frame["iteration"] == t and frame["phase"] == 2, (run, t)
        tokens = np.asarray(frame["tokens"])
        if tokens.size:
            shares.append(float(np.mean(tokens == 1)))
    agents = [r["nodes"] for r in gol_record.read_stats(run) if r.get("phase") == 2]
    return run, {"one_token": float(np.median(shares)) if shares else None, "games": len(shares),
                 "fewest_agents": int(min(agents)) if agents else 0, "games_run": len(agents)}


def main():
    runs = [f"{family}-s{seed:03d}" for family in CONDITIONS.values() for seed in range(1, 11)]
    with ProcessPoolExecutor(max_workers=2) as pool:
        found = dict(pool.map(measure, runs))
    out = {}
    for name, family in CONDITIONS.items():
        worlds = {r: found[r] for r in runs if r.startswith(family + "-s")}
        shares = [w["one_token"] for w in worlds.values() if w["one_token"] is not None]
        out[name] = {"worlds": worlds,
                     "one_token_median": float(np.median(shares)) if shares else None,
                     "below_half": sum(s < 0.5 for s in shares), "above_two_thirds": sum(s > 0.67 for s in shares),
                     "under_100": sum(w["fewest_agents"] < 100 for w in worlds.values())}
    verdict = {"B2 below 0.5 in at least 7": out["B2"]["below_half"] >= 7,
               "the last token kept above 0.67 in at least 7": out["the last token kept"]["above_two_thirds"] >= 7,
               "nothing kept under 100 agents in at least 5": out["nothing kept"]["under_100"] >= 5}
    with open(OUT, "w") as f:
        json.dump({"conditions": out, "verdict": verdict, "holds": all(verdict.values())}, f, indent=1)
    for name, c in out.items():
        print(f"{name:22s} one-token median {c['one_token_median']}  below 0.5: {c['below_half']}  "
              f"above 0.67: {c['above_two_thirds']}  under 100 agents: {c['under_100']}")
    print(verdict, "holds" if all(verdict.values()) else "fails")


if __name__ == "__main__":
    main()
