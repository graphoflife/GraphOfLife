"""
Is fitness passed on? Each world's family tree read for whether agents that
have many children have children that have many too (2026-11-07).

Natural selection changes a population only through differences in how many
offspring its members leave, and only those differences that are passed on
add up over generations (Fisher's fundamental theorem: a population adapts as
fast as its fitness varies heritably). In a world where who has children is
left to chance, the number of children a parent has says nothing about how
many its children will have: the correlation between the two is about zero.
Where it is clearly above zero, families differ in how fertile they are, and
selection has something to work with. The cause need not be the brain alone:
a rich parent may make rich children, and a place may favour its families.

For each world, from every birth of every iteration (the family tree of
research/screen/candidates.py):
  - every agent born in the window (from SETTLE to two thirds of the run, so
    that its children, and theirs, have time to be born before the run ends),
    and how many children it had by the end;
  - the correlation, across parent-child pairs both born in the window,
    between the parent's number of children and the child's;
  - beside it, the same correlation with the children's counts shuffled among
    the children born in the same iteration (which keeps any change of the
    world's birth rate over time, and only that): what chance and the times alone
    would give;
  - the share of the agents born in the window that had no child at all, and the
    share of all children the most fertile tenth of them had.

    python3 research/screen/families.py E40 [E41 ...] [--jobs 3]

Kept per world in GraphOfLifeRuns/.pilots/screen/families/<run>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import gol_plan  # noqa: E402
import gol_store  # noqa: E402

import candidates as K  # noqa: E402
import criteria as C  # noqa: E402

#: Shuffles for the chance correlation, and their seed.
SHUFFLES = 200
SEED = 5


def readings_dir() -> str:
    return os.path.join(gol_store.BASE_DIR, ".pilots", "screen", "families")


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def fertility(run_id: str, until: int) -> Dict[str, Any]:
    """How heritable having children is in one world (see the module's words)."""
    tree = K.Tree(run_id, until)
    known = np.flatnonzero(tree.parent != -2)
    born = tree.born[known]
    start, stop = C.SETTLE, (2 * until) // 3
    window = known[(born >= start) & (born < stop)]
    # Each agent's own children, counted from the tree (a founder's parent is -1, an unknown -2).
    kids = np.zeros(tree.parent.size, dtype=np.int64)
    parents = tree.parent[known]
    has = parents >= 0
    np.add.at(kids, parents[has], 1)
    n = kids[window]
    in_window = np.zeros(tree.parent.size, dtype=bool)
    in_window[window] = True
    pairs = window[in_window[np.maximum(tree.parent[window], 0)] & (tree.parent[window] >= 0)]
    p_kids = kids[tree.parent[pairs]]
    c_kids = kids[pairs]
    observed = _pearson(p_kids.astype(float), c_kids.astype(float))
    rng = np.random.RandomState(SEED)
    shuffled = []
    when = tree.born[pairs]
    by_time = np.argsort(when, kind="stable")
    for _ in range(SHUFFLES):
        # The children's counts shuffled among the children born in the same iteration: sorted by time, and
        # within each time in a random order, laid back on the children in time order.
        mixed = np.lexsort((rng.rand(when.size), when))
        c = np.empty_like(c_kids)
        c[by_time] = c_kids[mixed]
        shuffled.append(_pearson(p_kids.astype(float), c.astype(float)))
    shuffled = np.array([s for s in shuffled if np.isfinite(s)])
    top = np.sort(n)[::-1]
    tenth = max(1, n.size // 10)
    return {"born": int(n.size), "pairs": int(pairs.size), "window": [start, stop],
            "children_mean": float(n.mean()) if n.size else None,
            "childless": float((n == 0).mean()) if n.size else None,
            "top_tenth_share": float(top[:tenth].sum() / max(1, top.sum())) if n.size else None,
            "correlation": observed,
            "shuffled_mean": float(shuffled.mean()) if shuffled.size else None,
            "shuffled_sd": float(shuffled.std()) if shuffled.size else None,
            "above_shuffled": float((shuffled < observed).mean()) if shuffled.size and np.isfinite(observed)
            else None,
            # How many of chance's own spreads the correlation lies above chance: about 2 or more is clear.
            "z": float((observed - shuffled.mean()) / shuffled.std())
            if shuffled.size and shuffled.std() > 0 and np.isfinite(observed) else None}


def _path(run_id: str) -> str:
    return os.path.join(readings_dir(), f"{run_id}.json")


def read(item: Tuple[str, int]) -> str:
    run_id, until = item
    if not os.path.exists(_path(run_id)):
        out = fertility(run_id, until)
        os.makedirs(readings_dir(), exist_ok=True)
        with open(_path(run_id) + ".tmp", "w") as f:
            json.dump(out, f)
        os.replace(_path(run_id) + ".tmp", _path(run_id))
    return run_id


def kept(run_id: str) -> Dict[str, Any]:
    return json.load(open(_path(run_id))) if os.path.exists(_path(run_id)) else {}


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("rounds", nargs="+")
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    todo: List[Tuple[str, int]] = []
    for name in a.rounds:
        for spec in gol_plan.experiment_runs(name):
            if spec.lab["baseline"] != "B1" or not os.path.exists(os.path.join(gol_store.run_dir(spec.run_id),
                                                                              "meta.json")):
                continue
            meta = gol_store.load_meta(spec.run_id)
            if meta.get("status") != "extinct" and int(meta.get("iteration") or 0) >= spec.until:
                todo.append((spec.run_id, spec.until))
    print(f"{len(todo)} worlds lived to their end", flush=True)
    with ProcessPoolExecutor(a.jobs) as pool:
        futures = {pool.submit(read, item): item for item in todo}
        for future in as_completed(futures):
            try:
                print("read", future.result(), flush=True)
            except Exception as e:  # noqa: BLE001 - the others go on
                print("failed", futures[future][0], repr(e), flush=True)


if __name__ == "__main__":
    main()
