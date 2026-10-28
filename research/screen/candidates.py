"""
Round 3's worlds as candidates for open-ended evolution (the author,
2026-10-28): every world a candidate, looked at without prejudice, compared
with the others and ruled out one by one where it is least likely to evolve
open-endedly. No world is played: everything is read from what the worlds
recorded.

Evolution needs three things, and open-ended evolution needs them to go on:
variation among the agents, heredity (children like their parents) and
selection (some leave more descendants than others because of what they are).
A world's dossier asks of each what its records can say:

- **Its family tree**, from every birth of every iteration: how many
  generations it lived, and how fast the ancestry of the living narrows. In a
  population of N agents left to chance, the living descend from one ancestor
  of about 2N generations before; a world whose living descend from one
  ancestor of a few generations before has a tiny effective population, in
  which chance decides all but the largest differences.
- **Heredity**: in two kept worlds, agents and their parents alive together,
  played in the same situations (research/oee/checks/diversity.py). The
  heredity index is 1 when a child acts as its parent does (as the parent acts
  like itself between two tries) and 0 when it acts like a stranger.
- **The line of descent**: in every kept world, the ancestor of the most of
  the living at the end, and its brain then. The line's brains are played in
  the situations of every kept world: how far each era's behaviour lies from
  the one before, as a share of how far apart two strangers are, and whether
  the line keeps moving or comes to rest.
- **Its health**, by the screen's judge (research/screen/measures.py): every
  line of the author's criteria, averaged from iteration 100 on.

    python3 research/screen/candidates.py E40 [--jobs 3]    every finished world of the round

Each world's readings are kept as they are made, in
GraphOfLifeRuns/.pilots/screen/candidates/<run>.json.
"""
from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# One thread a process: the processes are the parallelism, beside the lab's workers.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

import gol_plan  # noqa: E402
import gol_store  # noqa: E402

#: The kept worlds heredity is read in.
HEREDITY_AT = (1500, 3000)
#: Parent-child pairs played, and situations they are played in.
PAIRS, SITUATIONS = 150, 100
#: Situations of each kept world the line of descent is played in.
LINE_SITUATIONS = 40
SEED = 7


@functools.lru_cache(maxsize=None)
def _module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _diversity():
    return _module("diversity_checks", os.path.join(ROOT, "research", "oee", "checks", "diversity.py"))


def _screen(name: str):
    sys.path.insert(0, HERE)
    return _module(f"screen_{name}", os.path.join(HERE, f"{name}.py"))


def readings_dir() -> str:
    return os.path.join(gol_store.BASE_DIR, ".pilots", "screen", "candidates")


# ---------------------------------------------------------------------------
# The family tree
# ---------------------------------------------------------------------------

class Tree:
    """Every agent a world ever had: its parent (-1 for a founder) and the iteration it was born in."""

    def __init__(self, run_id: str, until: int) -> None:
        first = gol_store.read_frame(run_id, 0)
        newborn = {int(b["child"]) for b in first["decisions"]["births"]}
        self.founders = np.array([int(u) for u in first["ids"] if int(u) not in newborn], dtype=np.int64)
        size = 1 << 16
        while int(self.founders.max()) >= size:
            size *= 2
        self.parent = np.full(size, -2, dtype=np.int64)
        self.born = np.full(size, -1, dtype=np.int32)
        self.generation = np.full(size, -1, dtype=np.int32)
        self.parent[self.founders] = -1
        self.born[self.founders] = 0
        self.generation[self.founders] = 0
        self.births = 0
        for t in range(until):
            made = gol_store.read_frame(run_id, 2 * t)["decisions"]["births"]
            if not made:
                continue
            kids = np.array([b["child"] for b in made], dtype=np.int64)
            parents = np.array([b["agent"] for b in made], dtype=np.int64)
            self._room(int(kids.max()))
            self.parent[kids] = parents
            self.born[kids] = t
            # A parent is always born before its child, so its generation is already known.
            self.generation[kids] = self.generation[parents] + 1
            self.births += int(kids.size)

    def _room(self, largest: int) -> None:
        while largest >= self.parent.size:
            more = self.parent.size
            self.parent = np.concatenate([self.parent, np.full(more, -2, dtype=np.int64)])
            self.born = np.concatenate([self.born, np.full(more, -1, dtype=np.int32)])
            self.generation = np.concatenate([self.generation, np.full(more, -1, dtype=np.int32)])

    def ancestors_at(self, ids: np.ndarray, at: int) -> np.ndarray:
        """Each agent's ancestor alive at iteration `at` (the agent itself if it was born by then)."""
        x = np.asarray(ids, dtype=np.int64).copy()
        young = self.born[x] > at
        while young.any():
            x[young] = self.parent[x[young]]
            young = self.born[x] > at
        return x


def _living(run_id: str, at: int) -> np.ndarray:
    """The agents alive after iteration at - 1's game, as its frame lists them."""
    frame = gol_store.read_frame(run_id, 2 * at - 1)
    assert (frame["iteration"], frame["phase"]) == (at - 1, 2), (run_id, at)
    return np.array(frame["ids"], dtype=np.int64)


def lineage(run_id: str, until: int, kept: List[int]) -> Dict[str, Any]:
    """
    A world's family tree read: the generations of the living at every kept world; for the living at
    the end, how many ancestors they have at earlier iterations, and how far back they all descend from
    one; and at every kept world, the ancestor of the most of them (the line of descent).
    """
    tree = Tree(run_id, until)
    end = _living(run_id, until)
    generations = {}
    for at in kept:
        alive = _living(run_id, at)
        g = tree.generation[alive]
        generations[str(at)] = {"agents": int(alive.size), "mean": float(g.mean()), "min": int(g.min()),
                                "max": int(g.max())}
    back = sorted({until - k for k in (1, 5, 10, 25, 50, 100, 250, 500, 1000, 1500, 2000, 2500)
                   if until - k >= 0} | set(kept), reverse=True)
    ancestors, x = {}, end.copy()
    for at in back:                        # going back, each step starts from the ancestors already found
        x = tree.ancestors_at(x, at - 1)
        ancestors[str(at)] = int(np.unique(x).size)
    # The latest iteration at which every agent alive at the end descends from one agent then alive.
    one = [at for at in back if ancestors[str(at)] == 1]
    common = max(one) if one else None
    if common is not None:
        lo, hi = common, min([at for at in back if at > common] or [until])
        while hi - lo > 1:                 # narrowed to the iteration
            mid = (lo + hi) // 2
            if np.unique(tree.ancestors_at(end, mid - 1)).size == 1:
                lo = mid
            else:
                hi = mid
        common = lo
    line = {}
    for at in kept:
        anc, counts = np.unique(tree.ancestors_at(end, at - 1), return_counts=True)
        lead = int(anc[np.argmax(counts)])
        line[str(at)] = {"agent": lead, "share": float(counts.max() / end.size),
                         "generation": int(tree.generation[lead])}
    return {"founders": int(tree.founders.size), "births": tree.births,
            "generations": generations, "ancestors": ancestors, "common_ancestor_at": common,
            "common_ancestor_generations": (float(tree.generation[end].mean()
                                                  - tree.generation[tree.ancestors_at(end[:1], common - 1)][0])
                                            if common is not None else None),
            "line": line}


# ---------------------------------------------------------------------------
# Behaviour: heredity, and the line of descent
# ---------------------------------------------------------------------------

def heredity(run_id: str, at: int) -> Dict[str, Any]:
    """
    Children against their parents, alive together in the world kept at `at`, played in the same situations:
    the median distance of a brain from itself (between two halves of the draws), of a child from its parent,
    and of a child from another agent's parent; and the heredity index, (stranger - parent) / (stranger - self).
    """
    D = _diversity()
    w = D.load(f"{run_id}:{at}", game=False)
    rng = np.random.RandomState(SEED)
    sits = D.situations(w, rng, SITUATIONS)
    living = {int(u) for u in w.G.nodes()}
    pairs = [(int(w.parent_of[c]), c) for c in sorted(living) if int(w.parent_of.get(c, -1)) in living]
    if len(pairs) < 10 or not sits:
        return {"pairs": len(pairs), "agents": len(living)}
    pick = sorted(rng.choice(len(pairs), size=min(PAIRS, len(pairs)), replace=False))
    pairs = [pairs[i] for i in pick]
    S = len(sits)
    pp, pq = D.profiles(*D.play(w, [w.brains[p] for p, _ in pairs], sits), S)
    _, cq = D.profiles(*D.play(w, [w.brains[c] for _, c in pairs], sits), S)
    cross = D.distance(pp, cq, S)
    itself = float(np.median(np.diag(D.distance(pp, pq, S))))
    child = float(np.median(np.diag(cross)))
    stranger = float(np.median(cross[~np.eye(len(pairs), dtype=bool)]))
    return {"pairs": len(pairs), "agents": len(living), "itself": itself, "child": child, "stranger": stranger,
            "index": (stranger - child) / (stranger - itself) if stranger > itself else None}


def line_of_descent(run_id: str, line: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    The line of descent's brains, one from each kept world, played together in the situations of every
    kept world: their distances, each pair's the mean over those worlds; and how far apart two strangers
    of each world are, to measure them by.
    """
    D = _diversity()
    eras = sorted(line, key=int)
    # One kept world at a time: a whole world's brains are too large to hold a dozen at once.
    brains = [D.load(f"{run_id}:{at}", game=False).brains[line[at]["agent"]] for at in eras]
    rng = np.random.RandomState(SEED)
    total = np.zeros((len(brains), len(brains)))
    strangers = []
    for at in eras:
        w = D.load(f"{run_id}:{at}", game=False)
        sits = D.situations(w, rng, LINE_SITUATIONS)
        S = len(sits)
        p, q = D.profiles(*D.play(w, brains, sits), S)
        cross = D.distance(p, q, S)
        total += (cross + cross.T) / 2
        living = sorted(int(u) for u in w.G.nodes())
        others = [w.brains[u] for u in rng.choice(living, size=min(40, len(living)), replace=False)]
        a, b = D.profiles(*D.play(w, others, sits), S)
        between = D.distance(a, b, S)
        strangers.append(float(np.median(between[~np.eye(len(others), dtype=bool)])))
    total /= len(eras)
    return {"eras": [int(at) for at in eras], "distance": total.round(5).tolist(),
            "strangers": float(np.mean(strangers)), "itself": float(np.mean(np.diag(total)))}


# ---------------------------------------------------------------------------
# A world's dossier, read in items and kept as they are made
# ---------------------------------------------------------------------------

def _path(run_id: str) -> str:
    return os.path.join(readings_dir(), f"{run_id}.json")


def kept(run_id: str) -> Dict[str, Any]:
    return json.load(open(_path(run_id))) if os.path.exists(_path(run_id)) else {}


def keep(run_id: str, key: str, value: Any) -> None:
    os.makedirs(readings_dir(), exist_ok=True)
    have = kept(run_id)
    have[key] = value
    with open(_path(run_id) + ".tmp", "w") as f:
        json.dump(have, f)
    os.replace(_path(run_id) + ".tmp", _path(run_id))


def health(run_id: str, until: int) -> Dict[str, Optional[float]]:
    """
    The averages of the screen's judge that a world's statistics and frames give (research/screen/measures.py),
    from iteration 100 on; the reading is kept where the judge keeps its own, so neither reads it twice.
    """
    M, C = _screen("measures"), _screen("criteria")
    item = (run_id, "series", until)
    have = M.kept([run_id])
    if item not in have:
        have[item] = M.read(item)
        M.keep(item, have)
    return M.averages(have, [item], C.SETTLE)


def dossier(item: Tuple[str, int]) -> str:
    """Read whatever of one finished world's dossier is not yet kept: its health, tree, heredity, line of descent."""
    run_id, until = item
    have = kept(run_id)
    if "health" not in have:
        keep(run_id, "health", health(run_id, until))
    archived = [at for at in gol_store.archived(run_id) if 0 < at <= until]
    if "lineage" not in have:
        keep(run_id, "lineage", lineage(run_id, until, archived))
        have = kept(run_id)
    for at in HEREDITY_AT:
        if at in archived and f"heredity {at}" not in have:
            keep(run_id, f"heredity {at}", heredity(run_id, at))
    if "line" not in have:
        keep(run_id, "line", line_of_descent(run_id, have["lineage"]["line"]))
    return run_id


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("round")
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    todo = []
    for spec in gol_plan.experiment_runs(a.round):
        if not os.path.exists(os.path.join(gol_store.run_dir(spec.run_id), "meta.json")):
            continue
        meta = gol_store.load_meta(spec.run_id)
        if meta.get("status") != "extinct" and int(meta.get("iteration") or 0) >= spec.until:
            todo.append((spec.run_id, spec.until))
    print(f"{len(todo)} worlds lived to their end", flush=True)
    failed = []
    with ProcessPoolExecutor(a.jobs) as pool:
        futures = {pool.submit(dossier, item): item for item in todo}
        for future in as_completed(futures):
            try:
                print("read", future.result(), flush=True)
            except Exception as e:  # noqa: BLE001 - the others go on; raised at the end
                failed.append((futures[future], e))
                print("failed", futures[future][0], repr(e), flush=True)
    if failed:
        raise RuntimeError(f"{len(failed)} worlds failed, the first {failed[0][0]}") from failed[0][1]


if __name__ == "__main__":
    main()
