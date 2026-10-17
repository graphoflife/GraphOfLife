"""
Is a mixed world's health a state it keeps, or one it passes through?

Every whole world Experiment 17 kept (archive/world_<t>.npz, every 500
iterations, ten worlds) read as research/atlas/profile.py reads a world at its
end: the seven marks of the health bar, the share of agents holding three
tokens or fewer, the diversity spreads. Writes
GraphOfLifeRuns/.pilots/e17_health.json. Experiment 18 played five of the same
runs on to 50,000; `--seeds 2,4,5,6,10 --from 10500 --to 50000 --out
e18_health.json` reads that span. Another family of runs with whole worlds
kept, such as Experiment 19's, is read with `--family`.

    python3 research/oee/checks/e17_health.py [--jobs 2] [--family B2-10000-4af4a3]
        [--seeds 1..10 --from 500 --to 10000 --out e17_health.json]
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
FAMILY = "B2-10000-4af4a3"


def one(item):
    r = P.measure(item)
    if r and "error" not in r and not r.get("dead"):
        r["bar"] = P.bar(r)
        r["marks"] = sum(1 for v in r["bar"].values() if v)
        with np.load(item[1]) as z:
            r["share_at_most_three"] = float(np.mean(z["tokens"] <= 3))
    return r


def main():
    from gol_plan import parse_seeds
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--family", default=FAMILY)
    ap.add_argument("--seeds", default="1..10")
    ap.add_argument("--from", dest="start", type=int, default=P.SETTLED)
    ap.add_argument("--to", type=int, default=10000)
    ap.add_argument("--out", default="e17_health.json")
    a = ap.parse_args()
    worlds = [f"{a.family}-s{s:03d}" for s in parse_seeds(a.seeds)]
    out = os.path.join(RUNS, ".pilots", a.out)
    # One experiment's span of these runs: another experiment may carry them further.
    items = [(run, path, a.family, True)
             for run in worlds for path in sorted(glob.glob(os.path.join(RUNS, run, "archive", "world_*.npz")))
             if a.start <= int(os.path.basename(path)[6:12]) <= a.to]
    print(f"{len(items)} whole worlds", flush=True)
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        rows = [r for r in pool.map(one, items, chunksize=2) if r]
    with open(out + ".tmp", "w") as f:
        json.dump(rows, f, indent=1)
    os.replace(out + ".tmp", out)
    for run in worlds:
        mine = sorted((r for r in rows if r["id"] == run), key=lambda r: r["at"])
        print(run[-4:], " ".join(f"{r['at'] // 100:>3}:{r.get('marks', '-')}" for r in mine), flush=True)


if __name__ == "__main__":
    main()
