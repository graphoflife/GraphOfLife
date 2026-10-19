"""
Every world at one iteration: Meta X's profile (research/atlas/profile.py) read at the
same moment in every world, so that worlds played for different lengths compare.

Pilot worlds of the named batches are read at their checkpoint `ckpt_<AT>.npz`, lab
runs of the named families at their archive `archive/world_<AT>.npz`. Each world gets
the seven marks of the health bar, the share of its agents holding three tokens or
fewer, how evenly it shares its tokens (the Gini and the richest tenth's share,
profile.evenness), and a diversity score: the mean of its four spreads' percentiles among the
frozen reference worlds (book/results/diversity_reference.json, research/atlas/
profile.py --freeze), so that the scores read as Meta X's do and stay put as worlds
are added.

A world already in the file, whose checkpoint is older than the file, is not
read again (`--fresh` reads every world); the file holds the worlds named.

    python3 research/atlas/at.py --at 1000 --batches T U V W X Y K S P Q R \
        --lab B2-10000 B2-10000-4af4a3 [--jobs 2] [--fresh]   writes .pilots/atlas/at1000.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import profile as P  # noqa: E402  (research/atlas/profile.py)

RUNS = os.path.join(ROOT, "GraphOfLifeRuns")


def items(at: int, batches, families):
    out = []
    for batch in batches:
        for config in sorted(glob.glob(os.path.join(P.PILOTS, batch, "*", "config.json"))):
            d = os.path.dirname(config)
            ckpt = os.path.join(d, f"ckpt_{at:06d}.npz")
            if os.path.exists(ckpt):
                meta = json.load(open(config))
                mixes = bool((meta.get("proto") or {}).get("mix") or (meta.get("cfg") or {}).get("mix_at_birth"))
                out.append((os.path.relpath(d, P.PILOTS), ckpt, batch, mixes))
    for family in families:
        for d in sorted(glob.glob(os.path.join(RUNS, f"{family}-s[0-9][0-9][0-9]"))):
            path = os.path.join(d, "archive", f"world_{at:06d}.npz")
            if os.path.exists(path):
                cfg = json.load(open(os.path.join(d, "meta.json"))).get("config") or {}
                out.append((os.path.basename(d), path, family, bool(cfg.get("mix_at_birth"))))
    return out


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
    ap.add_argument("--at", type=int, required=True)
    ap.add_argument("--batches", nargs="*", default=[])
    ap.add_argument("--lab", nargs="*", default=[])
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--fresh", action="store_true", help="read every world again")
    a = ap.parse_args()
    out = os.path.join(P.PILOTS, "atlas", f"at{a.at}.json")
    todo = items(a.at, a.batches, a.lab)
    done = {}
    if os.path.exists(out) and not a.fresh:
        written = os.path.getmtime(out)
        done = {r["id"]: r for r in json.load(open(out))["worlds"]}
        done = {i[0]: done[i[0]] for i in todo if i[0] in done and os.path.getmtime(i[1]) < written}
    print(f"{len(todo)} worlds at {a.at}, {len(todo) - len(done)} to read", flush=True)
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        fresh = {r["id"]: r for r in pool.map(one, [i for i in todo if i[0] not in done], chunksize=2) if r}
    rows = [done.get(i[0]) or fresh.get(i[0]) for i in todo]
    rows = [r for r in rows if r]
    # Diversity on Meta X's scale: each spread's percentile among the frozen reference worlds.
    ref = P.reference()
    for r in rows:
        percentiles, score = P.diversity(r, ref)
        if percentiles:
            r["percentiles"], r["diversity"] = percentiles, score
    with open(out + ".tmp", "w") as f:
        json.dump({"at": a.at, "worlds": rows}, f, indent=1)
    os.replace(out + ".tmp", out)
    print(f"{len(rows)} worlds measured at {a.at}", flush=True)


if __name__ == "__main__":
    main()
