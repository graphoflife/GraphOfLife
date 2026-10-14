"""Text summary of pilot runs: means over windows of iterations, per run.

    python3 summary.py r1 [--windows 0-100,100-500,500-1000,1000-1500] [--keys ...]
"""
import argparse
import glob
import json
import os

import numpy as np

KEYS = ["nodes", "edges", "births", "conquests", "kin_conquests", "revolutions",
        "starved_g", "orphaned_g", "meanAge", "maxAge", "roots", "genotypes", "gini",
        "bridges", "cutRisk", "coreShare", "core3Share", "articulationShare",
        "maxDegreeShare", "meanDegree", "dimension", "transitivity"]


def load(run):
    return [json.loads(l) for l in open(os.path.join(run, "series.jsonl"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--windows", default="0-100,100-500,500-1000,1000-1500")
    ap.add_argument("--keys", default=",".join(KEYS))
    a = ap.parse_args()
    windows = [tuple(int(x) for x in w.split("-")) for w in a.windows.split(",")]
    keys = a.keys.split(",")
    runs = []
    for d in a.dirs:
        runs += sorted(p for p in glob.glob(os.path.join(d, "*")) if os.path.isfile(os.path.join(p, "series.jsonl")))
    for key in keys:
        print(f"\n== {key}")
        print(f"{'run':22s}" + "".join(f"{f'{lo}-{hi}':>12s}" for lo, hi in windows))
        for run in runs:
            rows = load(run)
            cells = []
            for lo, hi in windows:
                vals = [r[key] for r in rows if lo <= r["it"] < hi and r.get(key) is not None]
                if key in ("births", "conquests", "kin_conquests", "revolutions", "starved_g", "orphaned_g"):
                    nodes = [r["nodes"] for r in rows if lo <= r["it"] < hi and r.get(key) is not None]
                    vals = [100 * v / max(1, n) for v, n in zip(vals, nodes)]
                cells.append(f"{np.mean(vals):12.3f}" if vals else f"{'':>12s}")
            print(f"{os.path.basename(run):22s}" + "".join(cells))


if __name__ == "__main__":
    main()
