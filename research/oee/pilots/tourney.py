"""
Cross-time tournament: do the brains of a later era beat those of an earlier one?

A contest takes the world as it stood at one checkpoint (its graph and tokens),
gives every agent a brain drawn at random from era A's population or from era
B's — a coin per agent — and plays T iterations with mutation switched off, so
no brain can change and every copy keeps its era's label. The score of A is the
share of agents carrying an A brain at the end. 0.5 means the two eras are
equally good at spreading; above 0.5, A is better.

Each pair is played on both eras' worlds (A's and B's) and with `reps` coins,
and the scores are averaged. With --layout halves each era gets a contiguous
half of the world instead of a coin per agent; with --layout turf, also the
era whose world it is keeps its own brains in place on its half, so that what
its brains are as a group (clones next to clones, allies next to allies) is
played too, once at home and once, drawn at random, away.

    python3 tourney.py RUNDIR --eras 0,250,500,... --T 20 --reps 2 --jobs 6
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "/home/stefan/Documents/CodingProjects/GraphOfLife Claude")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from gol_config import SimConfig
from proto import ProtoWorld, ProtoConfig, set_proto

LABEL_B = 1_000_000


def load_blob(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def era_path(run, it):
    return os.path.join(run, f"ckpt_{it:06d}.npz")


def halves(state, rng):
    """A coin per agent replaced by two contiguous regions: era B gets the half of
    the agents nearest (in steps) to a random agent, era A the rest."""
    from collections import deque
    # A checkpoint's edges are pairs of positions in its `ids`, not agent ids.
    n = len(state["ids"])
    adj = [[] for _ in range(n)]
    for a, b in state["edges"]:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    start = int(rng.randint(n))
    seen, order, queue = {start}, [], deque([start])
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                queue.append(v)
    order += [u for u in range(n) if u not in seen]
    from_b = np.zeros(n, dtype=bool)
    from_b[order[: n // 2]] = True
    return from_b


LAYOUT = "coins"


def contest(args):
    run, state_it, a_it, b_it, T, seed = args
    meta = json.load(open(os.path.join(run, "config.json")))
    set_proto(meta.get("proto", {}))
    if meta["cfg"].get("brain_kind") == "grow":
        import grow
        grow.install()
    cfg = ProtoConfig.from_dict(meta["cfg"], stored=False)
    cfg.mutation_probability = 0.0
    cfg.extinction_threshold = 0

    state = load_blob(era_path(run, state_it))
    A = load_blob(era_path(run, a_it))
    B = load_blob(era_path(run, b_it))
    rng = np.random.RandomState(seed)
    n = len(state["ids"])
    nl = int(state["n_layers"][0])
    from_b = halves(state, rng) if LAYOUT in ("halves", "turf") else rng.random_sample(n) < 0.5
    pick_a = rng.randint(0, len(A["ids"]), size=n)
    pick_b = rng.randint(0, len(B["ids"]), size=n)
    if LAYOUT == "turf":
        # The era whose world this is keeps its own brains where they are on its half,
        # clones next to clones; only the visiting era's brains are drawn at random.
        if state_it == b_it:
            pick_b = np.arange(n)
        if state_it == a_it:
            pick_a = np.arange(n)
    blob = dict(state)
    for i in range(nl):
        W = np.where(from_b[:, None, None], B[f"W{i}"][pick_b], A[f"W{i}"][pick_a])
        b = np.where(from_b[:, None, None], B[f"b{i}"][pick_b], A[f"b{i}"][pick_a])
        blob[f"W{i}"], blob[f"b{i}"] = W, b
    # A brain's label is its era: ids below LABEL_B are A's, at or above are B's. Within
    # an era a brain keeps its genotype's id, so clones are still clones (pool=genotype).
    blob["brain_ids"] = np.where(from_b, LABEL_B + B["brain_ids"][pick_b], A["brain_ids"][pick_a]).astype(np.int64)
    blob["parent_brain_ids"] = np.full(n, -1, dtype=np.int64)
    blob["msg_to"] = np.zeros(0, dtype=np.int64)
    blob["msg_from"] = np.zeros(0, dtype=np.int64)
    blob["msg_values"] = np.zeros((0, cfg.message_amount), dtype=np.float64)
    if "active" in A and "active" in B:
        blob["active"] = np.where(from_b, B["active"][pick_b], A["active"][pick_a])
    world = ProtoWorld.from_checkpoint(blob, cfg)
    import grow
    grow.restore_active(world, blob)
    world.rng = np.random.RandomState(seed + 7)
    for _ in range(T):
        world.step(record_decisions=False)
        if world.G.number_of_nodes() == 0:
            break
    ids = list(world.G.nodes())
    if not ids:
        return (state_it, a_it, b_it, seed, None, None, 0)
    is_a = np.array([world.brains[u].brain_id < LABEL_B for u in ids])
    tok = np.array([world.tokens[u] for u in ids], dtype=float)
    return (state_it, a_it, b_it, seed, float(is_a.mean()),
            float(tok[is_a].sum() / max(1.0, tok.sum())), len(ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--eras", default="")
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--pairs", default="all", help="all | base (each vs the first) | step (each vs the previous)")
    ap.add_argument("--out", default="")
    ap.add_argument("--layout", default="coins", choices=("coins", "halves", "turf"),
                    help="coins: a coin per agent; halves: each era gets a contiguous half of the world; "
                         "turf: halves, and the era whose world it is keeps its own brains in place")
    a = ap.parse_args()
    global LAYOUT
    LAYOUT = a.layout
    if a.eras:
        eras = [int(x) for x in a.eras.split(",")]
    else:
        eras = sorted(int(os.path.basename(p)[5:11]) for p in glob.glob(os.path.join(a.run, "ckpt_*.npz")))
    pairs = []
    for j, b_it in enumerate(eras):
        for i, a_it in enumerate(eras[:j]):
            if a.pairs == "base" and i != 0:
                continue
            if a.pairs == "step" and i != j - 1:
                continue
            pairs.append((a_it, b_it))
    tasks = []
    for a_it, b_it in pairs:
        for state_it in (a_it, b_it):
            for r in range(a.reps):
                tasks.append((a.run, state_it, a_it, b_it, a.T, 1000 + r))
    with ProcessPoolExecutor(a.jobs) as ex:
        results = list(ex.map(contest, tasks))
    out = a.out or os.path.join(a.run, f"tourney_T{a.T}_{a.pairs}{'' if a.layout == 'coins' else '_' + a.layout}.json")
    json.dump({"eras": eras, "T": a.T, "results": results}, open(out, "w"))
    # Later era's score against the earlier one, averaged over arenas and coins.
    by_pair = {}
    for s_it, a_it, b_it, seed, share_a, tok_a, n in results:
        if share_a is None:
            continue
        by_pair.setdefault((a_it, b_it), []).append(1 - share_a)
    for (a_it, b_it), v in sorted(by_pair.items()):
        print(f"{b_it:6d} vs {a_it:6d}: later era holds {np.mean(v):.3f} of agents  ({', '.join(f'{x:.2f}' for x in v)})")


if __name__ == "__main__":
    main()
