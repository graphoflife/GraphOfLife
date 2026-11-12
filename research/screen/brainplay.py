"""
The brains of the brain-copy tree, played (2026-11-11): heredity read on the
tree's own parent-child pairs, and the line of descent against a line that only
mutates.

Nothing is simulated: brains are taken from the worlds the runs kept every 250
iterations and played in situations taken from those worlds
(research/oee/checks/diversity.py), as research/screen/candidates.py plays them.

- **Heredity on the brain tree.** In the worlds kept at the middle and the end
  of the run, parent and child instances alive together (research/screen/
  braintree.py, `pairs_at`), 150 drawn at random, played in the same 100
  situations. The heredity index is candidates.heredity's: 1 when a child acts
  as its parent does (as the parent acts like itself between two halves of
  the draws), 0 when it acts like a stranger. Read over all pairs, and for the
  pairs made by birth and by conquest apart.
- **The mutation-only null for novelty.** The line of descent's brain at the
  first kept era (250) is given, era by era, the changes the real line
  collected on its way (`braintree.genotype_events`): a mutation of the world's
  own kind (Brain.mutate with the world's mutation_noise_std and
  mutation_sparsity, made for sure), and where the world mixes at birth a mix
  with a contemporary drawn at random from the era the change was made in.
  A change at a birth that ids cannot tell from a mix or a mutation is a mix
  with the share the world's births give (braintree.copies). Each null line is
  played with the real line in the same situations of every kept world, and
  novelty, each era's distance from every earlier era, is counted only above
  what the null lines reach. NULLS seeded replicates.

    python3 research/screen/brainplay.py RUN     one world's readings (its tree read first by braintree.py)
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import gol_store  # noqa: E402
# The engine of this checkout, imported before diversity.py puts the main checkout's first on the path.
import GraphOfLife as engine  # noqa: E402

import braintree as BT  # noqa: E402
import candidates as K  # noqa: E402

#: Situations heredity is played in, and those of each kept world the lines are played in (as candidates.py).
SITUATIONS, LINE_SITUATIONS = K.SITUATIONS, K.LINE_SITUATIONS
#: Null lines a world, and the seed of everything drawn here.
NULLS, SEED = 6, 13
#: Pairs shuffled into strangers for heredity's null.
SHUFFLES = 200


def _is_layer(key: str) -> bool:
    return key[0] in "Wb" and key[1:].isdigit()


class _Unread:
    """Stands in for a stack of weights that is not read: it has the stack's shape and hands out empty arrays."""

    def __init__(self, shape: Tuple[int, ...], dtype: Any) -> None:
        self.shape, self.dtype = shape, dtype

    def __getitem__(self, i: int) -> np.ndarray:
        return np.empty((0, 0), dtype=self.dtype)


def kept_world(run_id: str, at: int, choose: Callable[[np.ndarray, np.ndarray], Sequence[int]]):
    """
    The world kept at iteration `at`, rebuilt with every agent's tokens, links and messages, and with the brains of
    the agents `choose(ids, tokens)` names only (every other brain empty): a world of tens of thousands of brains
    is too large to hold whole a dozen times over. Returns the world and the agents chosen.
    """
    cfg = BT.config_of(run_id)
    with np.load(gol_store.archive_path(run_id, at)) as z:
        blob = {k: z[k] for k in z.files if not _is_layer(k)}
        ids = blob["ids"]
        nodes = np.asarray(sorted(set(int(u) for u in choose(ids, blob["tokens"]))), np.int64)
        pos = np.searchsorted(ids, nodes)
        if nodes.size and not np.array_equal(ids[np.minimum(pos, ids.size - 1)], nodes):
            raise ValueError(f"{run_id} at {at}: agents asked for that the kept world does not hold")
        layers = int(blob["n_layers"][0])
        weights, biases = [], []
        for i in range(layers):
            W, b = z[f"W{i}"], z[f"b{i}"]
            weights.append(W[pos])
            biases.append(b[pos])
            blob[f"W{i}"] = _Unread((ids.size,) + W.shape[1:], W.dtype)
            blob[f"b{i}"] = _Unread((ids.size,) + b.shape[1:], b.dtype)
            del W, b
    world = engine.GraphOfLife.from_checkpoint(blob, cfg)
    for j, u in enumerate(nodes.tolist()):
        world.brains[u].weights = [w[j] for w in weights]
        world.brains[u].biases = [b[j] for b in biases]
    return world, nodes


def proxy(world: Any, sits: List[Dict[str, Any]]) -> Any:
    """
    A world of the situations' agents and their neighbours only, which is all the decisions read of a world
    (diversity.decide: the heads, the settings, and for a child the parent's links and its neighbours' tokens),
    so that a dozen kept worlds' situations can be held at once.
    """
    keep = sorted({int(v) for s in sits for v in s["cands"]})
    small = engine.GraphOfLife(None, world.cfg, _empty=True)
    small.G = world.G.subgraph(keep).copy()
    small.tokens = {v: int(world.tokens.get(v, 0)) for v in keep}
    return small


def _alone(brain: Any) -> Any:
    """A brain of its own, outside any world."""
    return brain.copy()


# ---------------------------------------------------------------------------
# Heredity on the brain tree
# ---------------------------------------------------------------------------

def heredity(run_id: str, at: int, pairs: Dict[str, Any]) -> Dict[str, Any]:
    """
    The drawn parent-child pairs of the world kept at `at`, played in the same situations: the median distance
    of a brain from itself, of a child from its parent, of a child from another pair's parent; the heredity
    index over all pairs, and for the pairs made by birth and by conquest apart (on the same scale).
    """
    D = K._diversity()
    drawn = pairs["drawn"]
    if len(drawn) < 10:
        return {"pairs": len(drawn)}
    w, _ = kept_world(run_id, at, lambda ids, tokens: [u for c, p, _ in drawn for u in (c, p)])
    rng = np.random.RandomState(SEED)
    sits = D.situations(w, rng, SITUATIONS)
    S = len(sits)
    pp, pq = D.profiles(*D.play(w, [w.brains[p] for _, p, _ in drawn], sits), S)
    _, cq = D.profiles(*D.play(w, [w.brains[c] for c, _, _ in drawn], sits), S)
    cross = D.distance(pp, cq, S)
    itself = float(np.median(np.diag(D.distance(pp, pq, S))))
    own = np.diag(cross)
    stranger = float(np.median(cross[~np.eye(len(drawn), dtype=bool)]))
    kinds = np.array([k for _, _, k in drawn])

    def index(child: float) -> Optional[float]:
        return (stranger - child) / (stranger - itself) if stranger > itself else None

    # Against its null: children given another pair's parent at random, SHUFFLES times; the median distance
    # those strangers lie at, and how many of its spreads the real median lies below it.
    shuffled = []
    order = np.arange(len(drawn))
    for _ in range(SHUFFLES):
        perm = rng.permutation(order)
        while np.any(perm == order):                 # a derangement: nobody keeps its own parent
            perm = rng.permutation(order)
        shuffled.append(float(np.median(cross[perm, order])))
    shuffled = np.array(shuffled)
    z = (float(shuffled.mean()) - float(np.median(own))) / float(shuffled.std(ddof=1)) if shuffled.std() > 0 else None
    out = {"pairs": len(drawn), "agents": pairs["agents"], "itself": itself, "child": float(np.median(own)),
           "stranger": stranger, "index": index(float(np.median(own))), "z": z}
    for k in ("birth", "conquest"):
        mine = own[kinds == k]
        out[k] = {"pairs": int(mine.size), "child": float(np.median(mine)) if mine.size else None,
                  "index": index(float(np.median(mine))) if mine.size >= 5 else None}
    return out


# ---------------------------------------------------------------------------
# The line of descent against lines that only mutate
# ---------------------------------------------------------------------------

def null_steps(line: List[Dict[str, Any]], mix_share: Optional[float], nulls: int, seed: int) -> List[List[str]]:
    """
    For each null line, the changes it makes on the way into each era after the first: 'm' a mutation, 'x' a
    mix with a contemporary. The real line's own changes, in their order; a change that was a mix or a mutation
    (e) is drawn a mix with the world's share of mixes among births with one change.
    """
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(nulls):
        steps = []
        for row in line[1:]:
            events = row.get("events") or ""
            steps.append("".join("x" if c == "x" or (c == "e" and rng.rand() < (mix_share or 0.0)) else "m"
                                 for c in events))
        out.append(steps)
    return out


def novelty(D: np.ndarray, eras: Sequence[int], until: int) -> Dict[str, Any]:
    """A line's novelty, as funnel.py reads it: each era's distance from the nearest earlier era, late eras' mean."""
    near = [float(min(D[i, j] for j in range(i))) for i in range(1, len(eras))]
    late = [v for e, v in zip(eras[1:], near) if e > until / 2]
    steps = [float(D[i, i + 1]) for i in range(len(eras) - 1)]
    return {"nearest": near, "late": float(np.mean(late)) if late else None, "steps": steps,
            "itself": float(np.mean(np.diag(D)))}


def line_and_null(run_id: str, line: List[Dict[str, Any]], mix_share: Optional[float], until: int,
                  nulls: int = NULLS, seed: int = SEED) -> Dict[str, Any]:
    """
    The line of descent's brains, one an era, and `nulls` lines that start from its first brain and make the
    same changes at random, played together in the situations of every kept world (diversity.py): each pair's
    distance the mean over those worlds. Read: the real line's late novelty, the null lines', and the real
    above the null.
    """
    D = K._diversity()
    cfg = BT.config_of(run_id)
    eras = [row["era"] for row in line]
    plans = null_steps(line, mix_share, nulls, seed)
    rng = np.random.RandomState(seed + 1)
    real: List[Any] = []
    mates: Dict[int, List[Any]] = {}
    places: List[Tuple[List[Dict[str, Any]], Any]] = []
    for i, row in enumerate(line):
        wanted = sum(plan[i].count("x") for plan in plans) if i < len(line) - 1 else 0

        def choose(ids: np.ndarray, tokens: np.ndarray) -> List[int]:
            holding = ids[tokens > 0]
            drawn = rng.choice(holding, size=wanted, replace=True).tolist() if wanted else []
            choose.drawn = drawn
            return [row["node"]] + drawn

        w, _ = kept_world(run_id, row["era"], choose)
        real.append(_alone(w.brains[row["node"]]))
        mates[i] = [_alone(w.brains[u]) for u in choose.drawn]
        sits = D.situations(w, rng, LINE_SITUATIONS)
        places.append((sits, proxy(w, sits)))
        del w
    # The null lines: the first era's real brain, changed as the real line was.
    forced = dataclasses.replace(cfg, mutation_probability=1.0)
    lines = []
    used = {i: 0 for i in mates}
    for plan in plans:
        brain = _alone(real[0])
        brain.cfg = forced
        made = [_alone(brain)]
        for i, steps in enumerate(plan):
            for c in steps:
                if c == "m":
                    brain.mutate(1, rng)
                else:
                    brain = brain.mixed_with(mates[i][used[i]], rng)
                    brain.cfg = forced
                    used[i] += 1
            made.append(_alone(brain))
        lines.append(made)
    brains = real + [b for made in lines for b in made]
    E = len(eras)
    total = np.zeros((len(brains), len(brains)))
    for sits, small in places:
        S = len(sits)
        p, q = D.profiles(*D.play(small, brains, sits), S)
        cross = D.distance(p, q, S)
        total += (cross + cross.T) / 2
    total /= len(places)
    real_read = novelty(total[:E, :E], eras, until)
    null_reads = [novelty(total[E * (1 + r):E * (2 + r), E * (1 + r):E * (2 + r)], eras, until) for r in range(nulls)]
    # Each line's novelty above its own noise, a brain's distance from itself between two halves of the draws:
    # a mutated brain may be noisier than an evolved one, and noise is not novelty.
    real_late = (real_read["late"] - real_read["itself"]) if real_read["late"] is not None else None
    null_late = np.array([n["late"] - n["itself"] for n in null_reads if n["late"] is not None], float)
    above = z = None
    if real_late is not None and null_late.size:
        above = real_late - float(null_late.mean())
        sd = float(null_late.std(ddof=1)) if null_late.size > 1 else 0.0
        z = above / sd if sd > 0 else None
    return {"eras": eras, "distance": total[:E, :E].round(5).tolist(), "real": real_read,
            "null": {"late": [n["late"] for n in null_reads], "late_above_noise": null_late.tolist(),
                     "nearest": np.mean([n["nearest"] for n in null_reads], axis=0).tolist(),
                     "steps": np.mean([n["steps"] for n in null_reads], axis=0).tolist(),
                     "itself": float(np.mean([n["itself"] for n in null_reads]))},
            "changes": [len(s) for s in plans[0]], "mixes": [s.count("x") for s in plans[0]],
            "broken": sum(1 for row in line[1:] if row.get("events") is None),
            "late_above_noise": real_late, "late_above_null": above, "z": z, "nulls": nulls, "seed": seed}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    a = ap.parse_args()
    have = BT.kept(a.run)
    tree = have["tree"]
    until = max(row["era"] for row in tree["line"])
    started = time.time()
    for at, pairs in tree["pairs"].items():
        print(at, json.dumps(BT._plain(heredity(a.run, int(at), pairs)))[:600], flush=True)
    out = line_and_null(a.run, tree["line"], tree["copies"].get("mix_share_of_one_change"), until)
    print(json.dumps(BT._plain({k: v for k, v in out.items() if k != "distance"}))[:3000])
    print("took", time.time() - started)


if __name__ == "__main__":
    main()
