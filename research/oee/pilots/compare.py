"""One line per run: the settled state over a window, and whether the network passes the health bar.

    python3 compare.py DIR [DIR...] [--from 200] [--to 600]

Health bar (the user's: neither tree nor complete, resilient, decentralised, ~3D):
  one piece (always, after the cull), 2-core >= 0.95, articulation points <= 0.05,
  largest single cut <= 0.01 of the agents, no agent joined to more than 5% of
  the others, mean degree 3..12, ball dimension 2.3..4.
"""
import argparse
import glob
import json
import os

import numpy as np

BAR = {"coreShare": (0.8, None), "nodeCutRisk": (None, 0.02), "cutRisk": (None, 0.01),
       "maxDegreeShare": (None, 0.05), "meanDegree": (3, 12), "dimension": (2.3, 4.0)}


def window_mean(rows, key, lo, hi):
    v = [r[key] for r in rows if lo <= r["it"] < hi and r.get(key) is not None]
    return float(np.mean(v)) if v else None


def per_agent(rows, key, lo, hi):
    v = [100 * r.get(key, 0) / max(1, r["nodes"]) for r in rows if lo <= r["it"] < hi]
    return float(np.mean(v)) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--from", dest="lo", type=int, default=200)
    ap.add_argument("--to", dest="hi", type=int, default=10 ** 9)
    a = ap.parse_args()
    runs = []
    for d in a.dirs:
        runs += sorted(p for p in glob.glob(os.path.join(d, "*")) if os.path.exists(os.path.join(p, "series.jsonl")))
    head = (f"{'run':26s} {'last':>5s} {'n':>5s} {'tok/n':>5s} {'birth%':>6s} {'conq%':>6s} {'age':>5s} "
            f"{'lifeD':>5s} {'geno':>5s} {'core':>5s} {'ncut':>5s} {'cut':>5s} {'maxd':>5s} {'deg':>5s} {'dim':>5s} {'kin4':>5s}  health")
    print(head)
    for run in runs:
        rows = [json.loads(l) for l in open(os.path.join(run, "series.jsonl"))]
        if not rows:
            continue
        cfg = json.load(open(os.path.join(run, "config.json")))
        T = cfg["cfg"]["total_tokens"]
        lo, hi = a.lo, a.hi
        n = window_mean(rows, "nodes", lo, hi)
        if n is None:
            print(f"{os.path.basename(run):26s} {rows[-1]['it']:5d}  (not there yet)")
            continue
        d = [r for r in rows if lo <= r["it"] < hi]
        deaths = sum(r.get("deaths", 0) for r in d)
        life = (sum(r.get("deathAgeSum", 0) for r in d) / deaths) if deaths else float("nan")
        vals = {k: window_mean(rows, k, lo, hi) for k in BAR}
        ok = []
        for k, (mn, mx) in BAR.items():
            v = vals[k]
            good = v is not None and (mn is None or v >= mn) and (mx is None or v <= mx)
            ok.append("+" if good else "-")
        fmt = lambda v, f="5.2f": (format(v, f) if v is not None else "  -  ")
        print(f"{os.path.basename(run):26s} {rows[-1]['it']:5d} {n:5.0f} {T / max(n, 1):5.1f} "
              f"{fmt(per_agent(rows, 'births', lo, hi), '6.1f')} {fmt(per_agent(rows, 'conquests', lo, hi), '6.1f')} "
              f"{fmt(window_mean(rows, 'meanAge', lo, hi), '5.0f')} {life:5.0f} {fmt(window_mean(rows, 'genotypes', lo, hi), '5.0f')} "
              f"{fmt(vals['coreShare'])} {fmt(vals['nodeCutRisk'])} {fmt(vals['cutRisk'])} {fmt(vals['maxDegreeShare'])} "
              f"{fmt(vals['meanDegree'], '5.1f')} {fmt(vals['dimension'])} {fmt(window_mean(rows, 'kin4Largest', lo, hi), '5.0f')}  {''.join(ok)}")


if __name__ == "__main__":
    main()
