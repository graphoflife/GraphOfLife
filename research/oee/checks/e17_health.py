"""
Is a mixed world's health a state it keeps, or one it passes through?

Every whole world Experiment 17 kept (archive/world_<t>.npz, every 500
iterations, ten worlds) read as research/atlas/profile.py reads a world at its
end: the seven marks of the health bar, the share of agents holding three
tokens or fewer, the diversity spreads. Writes
GraphOfLifeRuns/.pilots/e17_health.json.

    python3 research/oee/checks/e17_health.py [--jobs 2]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research", "atlas"))

import numpy as np  # noqa: E402

import profile as P  # noqa: E402  (research/atlas/profile.py)

RUNS = os.path.join(ROOT, "GraphOfLifeRuns")
OUT = os.path.join(RUNS, ".pilots", "e17_health.json")
WORLDS = [f"B2-10000-4af4a3-s{s:03d}" for s in range(1, 11)]


def one(item):
    r = P.measure(item)
    if r and "error" not in r and not r.get("dead"):
        r["bar"] = P.bar(r)
        r["marks"] = sum(1 for v in r["bar"].values() if v)
        with np.load(item[1]) as z:
            r["share_at_most_three"] = float(np.mean(z["tokens"] <= 3))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    items = [(run, path, "B2-10000-4af4a3", True)
             for run in WORLDS for path in sorted(glob.glob(os.path.join(RUNS, run, "archive", "world_*.npz")))
             # Experiment 17's span: a later experiment may carry the runs further.
             if P.SETTLED <= int(os.path.basename(path)[6:12]) <= 10000]
    print(f"{len(items)} whole worlds", flush=True)
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        rows = [r for r in pool.map(one, items, chunksize=2) if r]
    with open(OUT + ".tmp", "w") as f:
        json.dump(rows, f, indent=1)
    os.replace(OUT + ".tmp", OUT)
    for run in WORLDS:
        mine = sorted((r for r in rows if r["id"] == run), key=lambda r: r["at"])
        print(run[-4:], " ".join(f"{r['at'] // 100:>3}:{r.get('marks', '-')}" for r in mine), flush=True)


if __name__ == "__main__":
    main()
