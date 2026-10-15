"""
Read the correlations of research/atlas/correlations.py and say what they show.

    python3 research/atlas/report.py B2 B1

Prints, for each set of worlds: the strongest relations between quantities not
made from the same numbers; those the logarithm reveals (much stronger on some
logarithmic axes than on linear ones); for every outcome, what predicts it
best; and, between two sets, the relations whose sign differs.
"""
import importlib.util
import json
import os
import sys

_spec = importlib.util.spec_from_file_location("correlations", os.path.join(os.path.dirname(os.path.abspath(__file__)), "correlations.py"))
_c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_c)

OUT = "/home/stefan/Documents/CodingProjects/GraphOfLife Claude/GraphOfLifeRuns/.pilots/atlas"
OUTCOMES = ["change", "rel_change", "kept", "took", "alive_10", "change_10", "degree_change_10", "child_next"]
AXES = ("lin-lin", "log-lin", "lin-log", "log-log")


def load(label):
    d = json.load(open(os.path.join(OUT, f"correlations_{label}.json")))
    for p in d["pairs"]:                 # the current list of pairs that are so by definition
        p["definitional"] = _c.definitional(p["a"], p["b"])
    return d


def best(p):
    keys = [k for k in AXES if k in p]
    if not keys:
        return None, None
    k = max(keys, key=lambda k: abs(p[k]["median"]))
    return k, p[k]["median"]


def agree(p, k):
    return p[k]["same_sign"] == p[k]["worlds"]


def strongest(d, n=40):
    rows = []
    for p in d["pairs"]:
        k, r = best(p)
        if k and not p["definitional"] and agree(p, k):
            rows.append((abs(r), p, k, r))
    rows.sort(key=lambda z: -z[0])
    return rows[:n]


def revealed(d, gain=0.25, n=30):
    rows = []
    for p in d["pairs"]:
        k, r = best(p)
        if k and k != "lin-lin" and "lin-lin" in p and not p["definitional"] and agree(p, k):
            g = abs(r) - abs(p["lin-lin"]["median"])
            if g >= gain:
                rows.append((g, p, k, r))
    rows.sort(key=lambda z: -z[0])
    return rows[:n]


def predictors(d, outcome, n=6):
    rows = []
    for p in d["pairs"]:
        if outcome not in (p["a"], p["b"]) or p["definitional"]:
            continue
        other = p["b"] if p["a"] == outcome else p["a"]
        if other in OUTCOMES:
            continue
        k, r = best(p)
        if k and agree(p, k):
            rows.append((abs(r), other, k, r, p.get("rank", {}).get("median")))
    rows.sort(key=lambda z: -z[0])
    return rows[:n]


def flips(d1, d2, n=25):
    look = {(p["a"], p["b"]): p for p in d2["pairs"]}
    rows = []
    for p in d1["pairs"]:
        q = look.get((p["a"], p["b"]))
        if not q or p["definitional"] or "rank" not in p or "rank" not in q:
            continue
        r1, r2 = p["rank"]["median"], q["rank"]["median"]
        if r1 * r2 < 0 and agree(p, "rank") and agree(q, "rank") and min(abs(r1), abs(r2)) > 0.1:
            rows.append((abs(r1 - r2), p["a"], p["b"], r1, r2))
    rows.sort(key=lambda z: -z[0])
    return rows[:n]


def main():
    labels = sys.argv[1:] or ["B2", "B1"]
    data = {label: load(label) for label in labels}
    for label, d in data.items():
        print(f"===== {label}")
        print("-- strongest (best axes; every world the same sign):")
        for _, p, k, r in strongest(d):
            print(f"   {p['a']:16s} {p['b']:16s} {k:8s} {r:+.2f}  ({p[k]['min']:+.2f}..{p[k]['max']:+.2f})"
                  f"  linear {p.get('lin-lin', {}).get('median', float('nan')):+.2f}  rank {p.get('rank', {}).get('median', float('nan')):+.2f}")
        print("-- revealed by the logarithm (best minus linear |r| >= 0.25):")
        for g, p, k, r in revealed(d):
            print(f"   {p['a']:16s} {p['b']:16s} {k:8s} {r:+.2f}  linear {p['lin-lin']['median']:+.2f}  (+{g:.2f})")
        print("-- what predicts each outcome:")
        for o in OUTCOMES:
            got = predictors(d, o)
            print(f"   {o:16s} " + "; ".join(f"{other} {k} {r:+.2f}" for _, other, k, r, rk in got))
    if len(labels) == 2:
        print(f"===== signs that differ between {labels[0]} and {labels[1]} (rank correlations):")
        for _, a, b, r1, r2 in flips(data[labels[0]], data[labels[1]]):
            print(f"   {a:16s} {b:16s} {labels[0]} {r1:+.2f}  {labels[1]} {r2:+.2f}")


if __name__ == "__main__":
    main()
