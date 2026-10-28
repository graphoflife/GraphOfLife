"""
Do families feed each other? Gifts and takeovers between clones, against chance.

A stake on a neighbour that does not take it is a gift to whoever holds the
node. If agents gave such gifts to neighbours carrying their own brain more
often than the share of such neighbours would make by chance, families would be
pumping tokens to their own: the factory of Chapter 51, found by evolution.
The same comparison for takeovers says whether agents spare their clones.

For every game of a window (brains read before the game, from the frame of the
reproduction phase of the same iteration):

  kin share of neighbours   for the agents that staked out, the share of their
                            neighbours that carry their brain (the chance level)
  kin share of gifts        of the tokens given as gifts, the share given to clones
  kin share of takeovers    of the nodes taken, the share taken from a clone

    python3 research/oee/checks/kin_flows.py RUN [...] --from 1000 --to 2000
"""
import argparse
import json
import sys

sys.path.insert(0, "/home/stefan/Documents/CodingProjects/GraphOfLife")

import numpy as np

import book_data as D


def analyse(run, t0, t1):
    chance_w, gifts_kin, gifts_all, taken_kin, taken_all, staked_kin, staked_all = [], 0, 0, 0, 0, 0, 0
    for t in range(t0, t1):
        try:
            before = D.frame_at(run, t, 1)
            f = D.frame_at(run, t, 2)
        except Exception:
            continue
        if not f or "decisions" not in f or not before:
            continue
        brain = dict(zip(before["ids"], before["brain_ids"]))
        winners = {w["node"]: w["winner"] for w in f["decisions"]["winners"]}
        for a in f["decisions"]["allocations"]:
            u = a["agent"]
            tg, al = a["targets"], a["alloc"]
            nbrs = tg[1:]
            out = [(v, x) for v, x in zip(nbrs, al[1:]) if x > 0]
            if not out or not nbrs:
                continue
            bu = brain.get(u)
            kin_nbrs = sum(1 for v in nbrs if brain.get(v) == bu)
            chance_w.append((kin_nbrs / len(nbrs), sum(x for _, x in out)))
            for v, x in out:
                kin = brain.get(v) == bu
                staked_all += x
                staked_kin += x * kin
                if winners.get(v) == u:
                    taken_all += 1
                    taken_kin += kin
                else:
                    gifts_all += x
                    gifts_kin += x * kin
    w = np.array([c[1] for c in chance_w], dtype=float)
    p = np.array([c[0] for c in chance_w], dtype=float)
    chance = float((p * w).sum() / w.sum()) if w.sum() else float("nan")
    return {"run": run, "from": t0, "to": t1, "kin_share_of_neighbours": chance,
            "kin_share_of_stakes": staked_kin / max(1, staked_all),
            "kin_share_of_gifts": gifts_kin / max(1, gifts_all),
            "kin_share_of_takeovers": taken_kin / max(1, taken_all),
            "gift_tokens": gifts_all, "takeovers": taken_all}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--from", dest="t0", type=int, default=1000)
    ap.add_argument("--to", dest="t1", type=int, default=2000)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = []
    for run in a.runs:
        r = analyse(run, a.t0, a.t1)
        res.append(r)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
