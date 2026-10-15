"""
Do later brains win when a contest lasts a whole life? E16's B2 worlds, eras played with T = 100.

A contest of the yardstick lasts 20 iterations; a B2 agent lives 100. What a
brain does for its children over a life barely shows in 20 iterations, so the
later eras of Experiment 16 are played against earlier ones again with T = 100
(gol_tournament.contest), once in each era's world.

    python3 research/oee/checks/long_contests.py --T 100 --pairs 500:2000,200:2000,100:200 --jobs 4
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "/home/stefan/Documents/CodingProjects/GraphOfLife Claude")

import numpy as np

import gol_store as store
import gol_tournament as tour


def one(task):
    run, home, a, b, T, seed = task
    config = store.load_meta(run)["config"]
    out = tour.contest(config, store.load_archive(run, home), store.load_archive(run, a),
                       store.load_archive(run, b), T, seed)
    return {"run": run, "home": home, "a": a, "b": b, "T": T, **out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=",".join(f"B2-10000-s{s:03d}" for s in range(1, 11)))
    ap.add_argument("--pairs", default="500:2000,200:2000,100:200")
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default="GraphOfLifeRuns/.pilots/long_contests.json")
    a = ap.parse_args()
    pairs = [tuple(int(x) for x in p.split(":")) for p in a.pairs.split(",")]
    tasks = [(run, home, x, y, a.T, 17) for run in a.runs.split(",") for x, y in pairs for home in (x, y)]
    with ProcessPoolExecutor(a.jobs) as pool:
        results = list(pool.map(one, tasks))
    json.dump(results, open(a.out, "w"))
    for x, y in pairs:
        per_world = []
        for run in a.runs.split(","):
            s = [r["share"] for r in results if r["run"] == run and r["a"] == x and r["b"] == y and r["share"] is not None]
            per_world.append(float(np.mean(s)))
        print(f"T={a.T}: {y} vs {x}: median {np.median(per_world):.3f}, worlds {min(per_world):.3f}-{max(per_world):.3f}, "
              f"above one half in {sum(1 for v in per_world if v > 0.5)} of {len(per_world)}", flush=True)


if __name__ == "__main__":
    main()
