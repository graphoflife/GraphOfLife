"""Round robin between worlds of different sizes (same rules): who evolved the better brains?

    python3 cross2.py RUN:ERA RUN:ERA ... [--T 20] [--coins 2] [--jobs 3]

Every pair plays in both worlds' arenas, `coins` contests each. Prints each
world's mean score against all others and an Elo rating per world (the first at 0).
"""
import argparse, itertools, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from cross import one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("worlds", nargs="+")
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--coins", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    W = a.worlds
    tasks, keys = [], []
    for i, j in itertools.combinations(range(len(W)), 2):
        for arena in (W[i], W[j]):
            for k in range(a.coins):
                tasks.append((arena, W[i], W[j], a.T, 200 + k))
                keys.append((i, j))
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(one, tasks))
    rows, ys = [], []
    scores = {i: [] for i in range(len(W))}
    for (i, j), share_b in zip(keys, res):
        p = float(np.clip(share_b, 0.02, 0.98))
        r = np.zeros(len(W)); r[j] += 1; r[i] -= 1
        rows.append(r); ys.append(np.log(p / (1 - p)))
        scores[j].append(share_b); scores[i].append(1 - share_b)
    X = np.array(rows)[:, 1:]
    fit, *_ = np.linalg.lstsq(X, np.array(ys), rcond=None)
    elo = np.concatenate([[0.0], fit]) * 400 / np.log(10)
    for i, w in enumerate(W):
        print(f"{w:28s} mean score {np.mean(scores[i]):.3f}   Elo {elo[i]:+6.0f}")
    if a.out:
        json.dump({"worlds": W, "keys": keys, "shares": res, "elo": elo.tolist()}, open(a.out, "w"))


if __name__ == "__main__":
    main()
