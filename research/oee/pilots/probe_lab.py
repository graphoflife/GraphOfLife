"""
Which inputs change an evolved agent's stakes, for lab runs (the archive of one iteration).

The same probe as sense.py, which reads pilot worlds: take the world as its
archive keeps it, play one game so messages are fresh, then for a sample of
agents with at least three tokens perturb one group of inputs at a time and
count the agents whose division of stakes changes. Own and neighbour state are
raised by log 2 (twice the tokens and links), messages are zeroed, the random
inputs redrawn.

    python3 probe_lab.py RUN:ITERATION [RUN:ITERATION ...] [--sample 400]
"""
import argparse
import sys

sys.path.insert(0, "/home/stefan/Documents/CodingProjects/GraphOfLife Claude")

import numpy as np

import gol_store as store
from gol_config import SimConfig
from GraphOfLifeSimple import GraphOfLife, _apportion

GROUPS = {"own state": ["own_tokens", "own_degree", "own_token_quantiles", "own_degree_quantiles"],
          "its state": ["its_tokens", "its_degree", "its_token_quantiles", "its_degree_quantiles"],
          "messages": ["message_to_self", "message_to_it", "message_from_it", "message_its_own"],
          "noise": ["noise"]}


def division(Y, heads, tokens, keep):
    scores = np.asarray(Y[heads["BLOTTO"], :], float)
    mode = np.mean(Y[heads["BLOTTO_MODE"], :], axis=1)
    kept = max(1, int(np.ceil(keep * tokens))) if keep > 0 else 0
    rest = tokens - kept
    if mode[0] > mode[1]:
        a = _apportion(scores, rest)
    else:
        a = np.zeros(Y.shape[1], dtype=int)
        a[int(np.argmax(scores))] = rest
    a[0] += kept
    return a


def probe(run, it, sample, seed=0):
    cfg = SimConfig.from_dict(store.load_meta(run)["config"])
    world = GraphOfLife.from_checkpoint(store.load_archive(run, it), cfg)
    world.step(record_decisions=False)
    log_deg, neighs, q_tok, q_deg, log_tok, at_risk = world._precompute_features()
    lay = cfg.input_layout()
    rng = np.random.RandomState(seed)
    nodes = sorted(world.G.nodes())
    nodes = [nodes[i] for i in rng.choice(len(nodes), min(sample, len(nodes)), replace=False)]
    counts = {g: 0 for g in GROUPS}
    used = 0
    for u in nodes:
        t = int(world.tokens[u])
        if t < 3:
            continue
        cands = [u] + list(neighs[u])
        X = world._inputs(u, cands, log_deg, q_tok, q_deg, log_tok, at_risk)
        base = division(world.brains[u].forward(X), world.heads, t, cfg.keep_home)
        used += 1
        for g, names in GROUPS.items():
            X2 = X.copy()
            for name in names:
                if name not in lay:
                    continue
                a, b, _ = lay[name]
                if name == "noise":
                    X2[a:b] = rng.uniform(-2, 2, size=X2[a:b].shape)
                elif name.startswith("message"):
                    X2[a:b] = 0
                else:
                    X2[a:b] = X2[a:b] + np.log(2)
            if not np.array_equal(division(world.brains[u].forward(X2), world.heads, t, cfg.keep_home), base):
                counts[g] += 1
    return used, {g: round(c / max(1, used), 3) for g, c in counts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+")
    ap.add_argument("--sample", type=int, default=400)
    a = ap.parse_args()
    for target in a.targets:
        run, it = target.rsplit(":", 1)
        used, shares = probe(run, int(it), a.sample)
        print(run, it, used, shares, flush=True)


if __name__ == "__main__":
    main()
