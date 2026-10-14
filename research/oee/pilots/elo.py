"""
Ratings from a cross-time tournament: one number per era, the strength of its brains.

The score of a contest is the share of the world the later era holds at its end.
Read as the probability that a brain of that era beats one of the other, the
Bradley-Terry model says logit(score) = r_later - r_earlier. Least squares over
every contest gives the ratings, the first era pinned at 0. Shown in Elo points
(400 per factor of ten in the odds), so +200 means the later era's brains win
about three contests in four against the first era's.

    python3 elo.py RUN/tourney_*.json [...]
"""
import json
import sys

import numpy as np


def ratings(path, clip=0.02):
    data = json.load(open(path))
    eras = data["eras"]
    index = {e: i for i, e in enumerate(eras)}
    rows, ys = [], []
    for s_it, a_it, b_it, seed, share_a, tok_a, n in data["results"]:
        if share_a is None:
            continue
        p = float(np.clip(1 - share_a, clip, 1 - clip))   # the later era, B
        row = np.zeros(len(eras))
        row[index[b_it]] += 1
        row[index[a_it]] -= 1
        rows.append(row)
        ys.append(np.log(p / (1 - p)))
    X = np.array(rows)[:, 1:]          # era 0 pinned at 0
    y = np.array(ys)
    r, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = np.concatenate([[0.0], r])
    resid = y - X @ r[1:]
    return eras, r * 400 / np.log(10), float(np.sqrt(np.mean(resid ** 2)))


if __name__ == "__main__":
    for path in sys.argv[1:]:
        eras, elo, rmse = ratings(path)
        print(path, f"(residual {rmse:.2f} in logits)")
        print("   " + "  ".join(f"{e}:{v:+.0f}" for e, v in zip(eras, elo)))
