"""Which statistics keep rising? Window means over doubling windows, and the rank correlation with time.

    python3 trend.py RUN [RUN...] [--keys a,b,c] [--start 25]
"""
import argparse
import json
import os

import numpy as np

KEYS = ["nodes", "meanAge", "maxAge", "lifespan", "genotypes", "cloneLargest", "cloneMean", "kin4Largest",
        "kin4Mean", "cyclingShare", "lightningLongest", "lightningScore", "coreShare", "core3Share",
        "meanDegree", "transitivity", "dimension", "births%", "conquests%"]


def spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def values(rows, key):
    out = []
    for r in rows:
        if key == "lifespan":
            if r.get("deaths"):
                out.append((r["it"], r["deathAgeSum"] / r["deaths"]))
        elif key.endswith("%"):
            k = key[:-1]
            if r.get(k) is not None or k in ("births", "conquests"):
                out.append((r["it"], 100 * r.get(k, 0) / max(1, r["nodes"])))
        elif r.get(key) is not None:
            out.append((r["it"], r[key]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--keys", default=",".join(KEYS))
    ap.add_argument("--start", type=int, default=25)
    a = ap.parse_args()
    for run in a.runs:
        rows = [json.loads(l) for l in open(os.path.join(run, "series.jsonl"))]
        last = rows[-1]["it"]
        edges = [a.start]
        while edges[-1] * 2 <= last:
            edges.append(edges[-1] * 2)
        edges.append(last + 1)
        print(f"\n{run}  (windows {edges})")
        for key in a.keys.split(","):
            v = values(rows, key)
            if not v:
                continue
            means = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                w = [y for x, y in v if lo <= x < hi]
                means.append(np.mean(w) if w else np.nan)
            ok = [m for m in means if not np.isnan(m)]
            rho = spearman(np.arange(len(ok)), np.array(ok)) if len(ok) > 2 else 0
            print(f"  {key:18s} rho={rho:+.2f}  " + " ".join(f"{m:9.3g}" for m in means))


if __name__ == "__main__":
    main()
