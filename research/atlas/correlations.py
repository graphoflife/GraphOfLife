"""
What goes with what: a search for correlations among the statistics of agents.

For every agent at the start of a game, from a lab run's recorded frames, a row
of quantities: its state, its neighbourhood and its place in the network, what
it staked in that game, and what then happened to it. Every pair of them is
correlated in each world, on linear and on logarithmic axes, and by rank.

    python3 research/atlas/correlations.py extract RUN [...] --from 1000 --to 2000 --step 20
    python3 research/atlas/correlations.py analyse --runs B2-10000-s001 ... --label B2

`extract` writes one table per run to GraphOfLifeRuns/.pilots/atlas/<run>.npz;
`analyse` reads them and writes the correlations to
GraphOfLifeRuns/.pilots/atlas/correlations_<label>.json.

The agent of a row is the agent at that node when the game began. A node keeps
its id when it is taken, so what is measured after the game (its change of
tokens, ten games later) is of the node, held by whoever holds it then; the
column `kept` says whether that was still the same agent.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, "/home/stefan/Documents/CodingProjects/GraphOfLife Claude")

import numpy as np

import book_data as D

OUT = "/home/stefan/Documents/CodingProjects/GraphOfLife Claude/GraphOfLifeRuns/.pilots/atlas"

# The quantities, the pairs made from the same numbers, and the arithmetic are
# gol_atlas's, which the Research tab's Correlations view answers with too.
from gol_atlas import (COLUMNS, DEFINITIONAL, VARIABLES, axis, definitional,  # noqa: F401
                       forman, pagerank, pair, pearson, rank, rows)


def rows_at(run, t, horizon=10):
    """The agents of game t of a lab run, one row each (gol_atlas.rows), with t first."""
    repro, game = D.frame_at(run, t, 1), D.frame_at(run, t, 2)
    if not repro or not game or not repro["ids"] or "decisions" not in game:
        return []
    try:
        nxt = D.frame_at(run, t + 1, 1)
        later = D.frame_at(run, t + horizon, 2)
    except Exception:
        return []
    return [[t] + [row[c] for c in COLUMNS] for row in rows(repro, game, nxt, later)]


def extract(run, t0, t1, step):
    rows = []
    for t in range(t0, t1, step):
        rows += rows_at(run, t)
    os.makedirs(OUT, exist_ok=True)
    table = np.array(rows, dtype=float)
    np.savez_compressed(os.path.join(OUT, f"{run}.npz"), table=table, columns=np.array(["t"] + COLUMNS))
    return table


def analyse(runs, label):
    tables = {}
    for run in runs:
        z = np.load(os.path.join(OUT, f"{run}.npz"))
        tables[run] = z["table"]
        names = list(z["columns"])
    cols = {name: i for i, name in enumerate(names)}
    results = []
    for i, a in enumerate(COLUMNS):
        for b in COLUMNS[i + 1:]:
            per_world = {run: pair(t, cols, a, b) for run, t in tables.items()}
            entry = {"a": a, "b": b, "definitional": definitional(a, b)}
            for key in ("lin-lin", "log-lin", "lin-log", "log-log", "rank"):
                vals = [w.get(key) for w in per_world.values() if w.get(key) is not None]
                if len(vals) >= max(2, len(runs) // 2):
                    entry[key] = {"median": float(np.median(vals)), "min": float(min(vals)),
                                  "max": float(max(vals)),
                                  "same_sign": int(max(sum(v > 0 for v in vals), sum(v < 0 for v in vals))),
                                  "worlds": len(vals)}
            results.append(entry)
    json.dump({"label": label, "runs": runs, "variables": VARIABLES, "pairs": results},
              open(os.path.join(OUT, f"correlations_{label}.json"), "w"), indent=0)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("extract", "analyse"))
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--runs", dest="named", nargs="*", default=[])
    ap.add_argument("--label", default="")
    ap.add_argument("--from", dest="t0", type=int, default=1000)
    ap.add_argument("--to", dest="t1", type=int, default=2000)
    ap.add_argument("--step", type=int, default=20)
    a = ap.parse_args()
    if a.what == "extract":
        for run in a.runs:
            table = extract(run, a.t0, a.t1, a.step)
            print(run, table.shape, flush=True)
    else:
        analyse(a.runs or a.named, a.label)


if __name__ == "__main__":
    main()
