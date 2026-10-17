"""
Pilot worlds read as Meta X reads every world, at every checkpoint they kept.

For each named pilot world: at every checkpoint (from 500 on), the seven marks
of the health bar, the share of agents holding three tokens or fewer, the
agents alive and the dimension (research/atlas/profile.py's measure). Written
to GraphOfLifeRuns/.pilots/<out>, one row per world and checkpoint.

    python3 research/oee/checks/pilot_health.py M/mix-mateparents-s1 M/mix-buylife-s1 ... --out steadying.json
    python3 research/oee/checks/pilot_health.py --batch M --out steadying.json      every world of batch M
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research", "atlas"))

import profile as P  # noqa: E402  (research/atlas/profile.py)


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
    ap.add_argument("worlds", nargs="*")
    ap.add_argument("--batch", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    worlds = list(a.worlds) + ([os.path.relpath(d, P.PILOTS) for d in sorted(glob.glob(os.path.join(P.PILOTS, a.batch, "*")))
                                if os.path.isdir(d)] if a.batch else [])
    items = []
    for w in worlds:
        meta = json.load(open(os.path.join(P.PILOTS, w, "config.json")))
        mixes = bool((meta.get("proto") or {}).get("mix") or (meta.get("cfg") or {}).get("mix_at_birth"))
        for path in sorted(glob.glob(os.path.join(P.PILOTS, w, "ckpt_*.npz"))):
            if int(os.path.basename(path)[5:11]) >= P.SETTLED:
                items.append((w, path, w.rsplit("-s", 1)[0], mixes))
    print(f"{len(items)} checkpoints of {len(worlds)} worlds", flush=True)
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        rows = [r for r in pool.map(one, items, chunksize=2) if r]
    out = os.path.join(P.PILOTS, a.out)
    with open(out + ".tmp", "w") as f:
        json.dump(rows, f, indent=1)
    os.replace(out + ".tmp", out)
    for w in worlds:
        mine = sorted((r for r in rows if r["id"] == w), key=lambda r: r["at"])
        print(f"{w:28s}", " ".join(f"{r['at'] // 100}:{r.get('agents')}/{r.get('marks', '-')}/"
                                    f"{round(r.get('share_at_most_three', float('nan')), 2)}" for r in mine), flush=True)


if __name__ == "__main__":
    main()
