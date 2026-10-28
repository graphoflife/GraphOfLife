"""
Musical chairs: agents that take each other's nodes in the same game.

In one game every node goes to one winner. Draw an arrow from each node taken
to the node its winner came from: v <- u when u took v. Following the arrows
backwards, a node whose holder took a neighbour while its own node was taken
by a third agent is a link in a *shift*: everyone moves one step along a path.
If the path closes, the agents have moved round a loop: two agents that took
each other's nodes have *swapped*, three round a triangle have *rotated*.

Per game, counted over a window:

  takeovers     nodes taken by a neighbour
  shifts        takeovers whose taker lost its own node in the same game
  swaps         pairs that took each other's nodes
  rotations     loops of three or more takeovers
  chance        the swaps expected if every taker had picked the nodes it took
                at random among its neighbours, everything else as it was:
                for each takeover u -> v, the share of v's neighbours that v
                took, summed and halved

    python3 research/oee/checks/rotations.py RUN [...] --from 1000 --to 2000
"""
import argparse
import json
import sys
from collections import Counter

sys.path.insert(0, "/home/stefan/Documents/CodingProjects/GraphOfLife")

import book_data as D


def cycles(took):
    """Cycle lengths of the map node -> the node its winner came from (only takeovers)."""
    lengths, seen = [], set()
    for start in took:
        if start in seen:
            continue
        path, x = [], start
        while x in took and x not in seen and x not in path:
            path.append(x)
            x = took[x]
        if x in path:
            lengths.append(len(path) - path.index(x))
        seen.update(path)
    return lengths


def analyse(run, t0, t1):
    games = takeovers = shifts = 0
    loops = Counter()
    chance = 0.0
    for t in range(t0, t1):
        try:
            f = D.frame_at(run, t, 2)
        except Exception:
            continue
        if not f or "decisions" not in f:
            continue
        games += 1
        dec = f["decisions"]
        winners = {w["node"]: w["winner"] for w in dec["winners"]}
        took = {v: w for v, w in winners.items() if w != v}          # node -> its taker's own node
        deg = {a["agent"]: len(a["targets"]) - 1 for a in dec["allocations"]}
        took_by = Counter(took.values())                              # how many nodes each taker took
        takeovers += len(took)
        shifts += sum(1 for v, u in took.items() if winners.get(u, u) != u)
        loops.update(cycles(took))
        chance += sum(took_by.get(v, 0) / deg[v] for v, u in took.items() if deg.get(v)) / 2
    return {"run": run, "from": t0, "to": t1, "games": games, "takeovers": takeovers,
            "shift_share": shifts / max(1, takeovers),
            "swaps": loops.get(2, 0), "swaps_chance": chance,
            "rotations": sum(n for k, n in loops.items() if k >= 3),
            "rotation_lengths": {int(k): int(n) for k, n in sorted(loops.items()) if k >= 3}}


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
