"""
What goes with what in a whole world: the statistics a lab run records after every game.

Two statistics that both drift over a run correlate whether or not they have
anything to do with each other. So within a run the correlation is taken of
their *changes* from one recorded game to the next, which a shared trend does
not touch; and between runs, of their means over a stretch of iterations, one
pair of numbers per world.

    python3 research/atlas/world_series.py --runs B2-10000-s001 ... --label B2 --from 500 --to 2000

Writes GraphOfLifeRuns/.pilots/atlas/world_<label>.json.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, "/home/stefan/Documents/CodingProjects/GraphOfLife")

import numpy as np

import gol_record

OUT = "/home/stefan/Documents/CodingProjects/GraphOfLife/GraphOfLifeRuns/.pilots/atlas"
#: Not statistics of the world: bookkeeping, or the same in every frame.
SKIP = {"_frame", "_heavy", "iteration", "phase", "tokens", "nodes_before", "_seconds"}


def series(run, t0, t1):
    """Every numeric statistic after the game of each iteration from t0 to t1, as arrays by name."""
    rows = []
    with open(gol_record.stats_path(run)) as f:
        for line in f:
            r = json.loads(line)
            if r.get("phase") == 2 and t0 <= r.get("iteration", -1) < t1:
                rows.append(r)
    names = sorted({k for r in rows for k, v in r.items()
                    if k not in SKIP and isinstance(v, (int, float)) and not isinstance(v, bool)})
    its = np.array([r["iteration"] for r in rows])
    return its, {k: np.array([r.get(k, np.nan) if isinstance(r.get(k), (int, float)) else np.nan
                              for r in rows], float) for k in names}


def corr(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 30 or x[ok].std() == 0 or y[ok].std() == 0:
        return None
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def changes(its, v):
    """Change from one recorded game to the next where the statistic was measured at both."""
    ok = np.isfinite(v)
    d = np.full(len(v), np.nan)
    idx = np.flatnonzero(ok)
    d[idx[1:]] = v[idx[1:]] - v[idx[:-1]]
    return d


def analyse(runs, label, t0, t1):
    within, means = {}, {}
    for run in runs:
        its, s = series(run, t0, t1)
        d = {k: changes(its, v) for k, v in s.items()}
        names = sorted(s)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                r = corr(d[a], d[b])
                if r is not None:
                    within.setdefault((a, b), []).append(r)
        means[run] = {k: float(np.nanmean(v)) for k, v in s.items() if np.isfinite(v).any()}
    pairs = []
    for (a, b), vals in within.items():
        if len(vals) >= max(2, len(runs) // 2):
            pairs.append({"a": a, "b": b, "median": float(np.median(vals)), "min": min(vals), "max": max(vals),
                          "same_sign": int(max(sum(v > 0 for v in vals), sum(v < 0 for v in vals))),
                          "worlds": len(vals)})
    between = []
    names = sorted(set.intersection(*(set(m) for m in means.values())))
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            x = np.array([means[r][a] for r in runs])
            y = np.array([means[r][b] for r in runs])
            if len(runs) >= 8 and x.std() > 0 and y.std() > 0:
                between.append({"a": a, "b": b, "r": float(np.corrcoef(x, y)[0, 1]), "worlds": len(runs)})
    json.dump({"label": label, "runs": runs, "from": t0, "to": t1, "within": pairs, "between": between,
               "means": means}, open(os.path.join(OUT, f"world_{label}.json"), "w"))
    return pairs, between


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--from", dest="t0", type=int, default=500)
    ap.add_argument("--to", dest="t1", type=int, default=2000)
    a = ap.parse_args()
    pairs, between = analyse(a.runs, a.label, a.t0, a.t1)
    print(f"{len(pairs)} pairs within runs, {len(between)} between runs")


if __name__ == "__main__":
    main()
