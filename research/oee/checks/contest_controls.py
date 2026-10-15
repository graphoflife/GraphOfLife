"""
Controls for long contests: brains against themselves, and the eras' labels swapped.

    python3 research/oee/checks/contest_controls.py --T 100 --jobs 4

Null: era 2,000 against itself (both sides drawn from the same archive), in the
world of 2,000 - its share must be a tie whatever T is. Swap: era 2,000's brains
given the A label and era 500's the B label (gol_tournament.contest(state, A=2000,
B=500)) - if the later era really wins, B's share must fall below a half as far
as it rose above it when labelled the other way.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "/home/stefan/Documents/CodingProjects/GraphOfLife Claude")

import numpy as np

import gol_store as store
import gol_tournament as tour


def one(task):
    run, kind, T = task
    config = store.load_meta(run)["config"]
    home = store.load_archive(run, 2000)
    if kind == "null":
        out = tour.contest(config, home, home, home, T, 17)
    else:
        out = tour.contest(config, home, home, store.load_archive(run, 500), T, 17)
    return {"run": run, "kind": kind, "T": T, **out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    runs = [f"B2-10000-s{s:03d}" for s in range(1, 11)]
    tasks = [(r, k, a.T) for r in runs for k in ("null", "swap")]
    with ProcessPoolExecutor(a.jobs) as pool:
        res = list(pool.map(one, tasks))
    json.dump(res, open("GraphOfLifeRuns/.pilots/contest_controls.json", "w"))
    for k in ("null", "swap"):
        v = [r["share"] for r in res if r["kind"] == k]
        print(f"T={a.T} {k}: B's share median {np.median(v):.3f}, worlds {min(v):.3f}-{max(v):.3f}", flush=True)


if __name__ == "__main__":
    main()
