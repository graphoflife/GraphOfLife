#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-time tournaments: are the brains of a later moment better than those of
an earlier one?

This is the yardstick of the book's Part V. A world that keeps evolving keeps
producing brains that beat their own ancestors; a world that has stopped
produces brains that only tie with them. Nothing inside one moment can tell
the two apart — a statistic that settles may be a world that is finished or a
world in a Red Queen race that keeps it in place — so the brains of two moments
are made to play each other.

**A contest.** The world as it stood at one archived moment — its graph, its
tokens, its ages — is given brains drawn from two eras, A and B: every agent
tosses a coin and gets a brain drawn at random from A's agents or from B's.
Messages in flight are cleared, so neither side is primed. The world then
plays `T` iterations under its own rules, but with every mutation switched
off, so no brain can change and every copy keeps its era's label. The score of
B is the share of agents carrying a B brain at the end: 0.5 is a tie, above it
B spreads better than A.

**A tournament** plays every pair of eras, on both eras' worlds (an era plays
at home once and away once), with a few coins each. **Ratings** read the
scores as Bradley–Terry odds — the score of B against A is taken as the chance
that a B brain beats an A brain, so logit(score) = r_B - r_A — and fit one
rating per era by least squares, the first era pinned at 0. Shown in Elo
points: +200 means the later era's brains win about three contests in four
against the first era's.

The eras come from the run's archive (gol_store.archived), which a run keeps
when it is made with `archive_every`.

    python3 gol_tournament.py RUN_ID [--eras 0,100,500] [--T 20] [--coins 2] [--jobs 4]
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Brain ids at or above this are B's; below, A's. Ids only ever name genotypes
#: and no mutation happens in a contest, so a copy keeps its era's id.
LABEL_B = 1_000_000_000


def mixed(state: Dict[str, np.ndarray], A: Dict[str, np.ndarray], B: Dict[str, np.ndarray],
          message_amount: int, rng: np.random.RandomState) -> Dict[str, np.ndarray]:
    """`state`'s world with every agent given a brain drawn from era A or era B."""
    n = len(state["ids"])
    layers = int(state["n_layers"][0])
    from_b = rng.random_sample(n) < 0.5
    pick_a = rng.randint(0, len(A["ids"]), size=n)
    pick_b = rng.randint(0, len(B["ids"]), size=n)
    blob = dict(state)
    for i in range(layers):
        for name in (f"W{i}", f"b{i}"):
            blob[name] = np.where(from_b.reshape((n,) + (1,) * (state[name].ndim - 1)),
                                  B[name][pick_b], A[name][pick_a])
    blob["brain_ids"] = np.where(from_b, LABEL_B + pick_b, pick_a).astype(np.int64)
    blob["parent_brain_ids"] = np.full(n, -1, dtype=np.int64)
    blob["msg_to"] = np.zeros(0, dtype=np.int64)
    blob["msg_from"] = np.zeros(0, dtype=np.int64)
    blob["msg_values"] = np.zeros((0, message_amount), dtype=np.float64)
    return blob


def contest(config: Dict[str, Any], state: Dict[str, np.ndarray], A: Dict[str, np.ndarray],
            B: Dict[str, np.ndarray], T: int, seed: int) -> Dict[str, Any]:
    """One contest: B's share of the agents (and of the tokens) after T iterations."""
    from gol_config import SimConfig
    from GraphOfLifeSimple import GraphOfLife

    cfg = SimConfig.from_dict(config)
    cfg.mutation_probability = 0.0
    cfg.extinction_threshold = 0
    rng = np.random.RandomState(seed)
    world = GraphOfLife.from_checkpoint(mixed(state, A, B, cfg.message_amount, rng), cfg)
    world.rng = np.random.RandomState(seed + 1)
    for _ in range(T):
        world.step(record_decisions=False)
        if world.G.number_of_nodes() == 0:
            break
    ids = list(world.G.nodes())
    if not ids:
        return {"share": None, "tokens": None, "agents": 0}
    is_b = np.array([world.brains[u].brain_id >= LABEL_B for u in ids])
    held = np.array([world.tokens[u] for u in ids], dtype=float)
    return {"share": float(is_b.mean()),
            "tokens": float(held[is_b].sum() / max(1.0, held.sum())),
            "agents": len(ids)}


def _play_one(task: Tuple[str, int, int, int, int, int]) -> Dict[str, Any]:
    import gol_store as store
    run_id, home, a, b, T, seed = task
    config = store.load_meta(run_id)["config"]
    result = contest(config, store.load_archive(run_id, home), store.load_archive(run_id, a),
                     store.load_archive(run_id, b), T, seed)
    return {"home": home, "a": a, "b": b, "seed": seed, **result}


def pairs(eras: Sequence[int], against: str = "all") -> List[Tuple[int, int]]:
    """Which eras play: every pair, each against the first, or each against the one before."""
    out = []
    for j, b in enumerate(eras):
        for i, a in enumerate(eras[:j]):
            if (against == "all" or (against == "first" and i == 0)
                    or (against == "previous" and i == j - 1)):
                out.append((a, b))
    return out


def play(run_id: str, eras: Optional[Sequence[int]] = None, T: int = 20, coins: int = 2,
         against: str = "all", jobs: int = 1) -> Dict[str, Any]:
    """A whole tournament on one run's archive: every contest, and the ratings."""
    import gol_store as store
    eras = sorted(eras if eras is not None else store.archived(run_id))
    tasks = [(run_id, home, a, b, T, 1000 * k + 17)
             for a, b in pairs(eras, against) for home in (a, b) for k in range(coins)]
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as pool:
            results = list(pool.map(_play_one, tasks))
    else:
        results = [_play_one(t) for t in tasks]
    elo, residual = ratings(eras, results)
    return {"run": run_id, "eras": eras, "T": T, "coins": coins, "against": against,
            "contests": results, "elo": elo, "residual": residual}


def ratings(eras: Sequence[int], results: Sequence[Dict[str, Any]],
            clip: float = 0.02) -> Tuple[List[float], float]:
    """
    One Elo rating per era, the first at 0, from every contest's score.

    A score is clipped away from 0 and 1 before its logit is taken: a contest
    one side won outright says the odds are long, not infinite.
    """
    index = {e: i for i, e in enumerate(eras)}
    rows, logits = [], []
    for r in results:
        if r.get("share") is None or r["a"] not in index or r["b"] not in index:
            continue
        p = min(1 - clip, max(clip, r["share"]))
        row = np.zeros(len(eras))
        row[index[r["b"]]] += 1
        row[index[r["a"]]] -= 1
        rows.append(row)
        logits.append(np.log(p / (1 - p)))
    if not rows:
        return [0.0] * len(eras), float("nan")
    X = np.array(rows)[:, 1:]
    y = np.array(logits)
    fit, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = np.concatenate([[0.0], fit])
    residual = float(np.sqrt(np.mean((y - X @ fit) ** 2)))
    return [float(v) for v in r * 400 / np.log(10)], residual


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--eras", default="")
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--coins", type=int, default=2)
    ap.add_argument("--against", default="all", choices=("all", "first", "previous"))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    a = ap.parse_args()
    eras = [int(x) for x in a.eras.split(",")] if a.eras else None
    out = play(a.run, eras, a.T, a.coins, a.against, a.jobs)
    for era, elo in zip(out["eras"], out["elo"]):
        print(f"{era:7d}  {elo:+7.0f}")
    print(f"(residual {out['residual']:.2f} in logits)")
    print(json.dumps({k: v for k, v in out.items() if k != "contests"}))


if __name__ == "__main__":
    main()
