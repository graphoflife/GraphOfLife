"""
Gliders: packets of wealth and brain that travel through the network, game after game.

A packet moves when an agent takes a neighbour's node with at least half of its
tokens: the tokens go to the node taken, and so does a copy of the brain. If in
the next game that node takes a neighbour's node the same way, the packet has
travelled on. This follows every chain of such moves through a lab run's
recorded games and describes the chains of at least `min_len` moves:

  length        how many moves in a row
  reach         how far, in steps through the network at the chain's last game,
                its last node is from its first: about the length if the packet
                travels outward, about its square root if it wanders
  packet        the tokens it carried at each move, first and last
  same brain    the share of moves after which the holder still carried the
                genotype the chain started with
  share         of all nodes taken in the window, the share taken by chains of
                at least `min_len` moves
  home kept     the share of moves after which the mover still held the node it
                came from: a gun leaves a copy behind at every step, a glider
                moves and leaves nothing
  end           how the chains ended: the node holding the packet taken by
                another agent, kept without moving on, or left to starve
  walk reach    how far a random walk of the same length gets from the same
                start on the same network: the chance level of reach
  guns          nodes that stay where they are and launch chain after chain: how
                many chains each launching node starts in the window, against
                the Poisson spread the same number of launches would have if
                every launching node launched at the same rate, and the gaps
                between one node's launches (a gun firing every game has gap 1)
  returns       how often a move goes back to the node the packet held two moves
                before (lag 2: back and forth, a blinker) or three moves before
                (lag 3: round a triangle), against the chance that a raider
                choosing among its neighbours at random would pick that node
                (1 / its neighbours, if the node is one of them)

    python3 research/oee/checks/gliders.py RUN --from 1000 --to 2000 [--min-len 3]
"""
import argparse
import json
import random
import sys
from collections import Counter, deque

sys.path.insert(0, "/home/stefan/Documents/CodingProjects/GraphOfLife Claude")

import numpy as np

import book_data as D


def adjacency(edges):
    adj = {}
    for x, y in edges:
        adj.setdefault(x, []).append(y)
        adj.setdefault(y, []).append(x)
    return adj


def distance(adj, a, b, limit=60):
    if a not in adj or b not in adj:
        return None
    seen = {a: 0}
    q = deque([a])
    while q:
        x = q.popleft()
        if x == b:
            return seen[x]
        if seen[x] >= limit:
            continue
        for y in adj[x]:
            if y not in seen:
                seen[y] = seen[x] + 1
                q.append(y)
    return None


def walk_reach(adj, start, steps, rng, draws=5):
    """Median distance a random walk of `steps` steps gets from `start`."""
    if start not in adj:
        return None
    reach = []
    for _ in range(draws):
        x = start
        for _ in range(steps):
            x = rng.choice(adj[x])
        reach.append(distance(adj, start, x))
    reach = [r for r in reach if r is not None]
    return float(np.median(reach)) if reach else None


def close(chain, adj, min_len, rng):
    if len(chain["packets"]) >= min_len:
        chain["reach"] = distance(adj, chain["nodes"][0], chain["nodes"][-1])
        chain["walk"] = walk_reach(adj, chain["nodes"][0], len(chain["packets"]), rng)


def analyse(run, t0, t1, min_len=3, seed=1):
    rng = random.Random(seed)
    open_chains = {}             # node holding a packet -> chain record
    done = []
    taken_total = 0
    for t in range(t0, t1):
        f = D.frame_at(run, t, 2)
        if not f or "decisions" not in f:
            continue
        dec = f["decisions"]
        winners = {w["node"]: w["winner"] for w in dec["winners"]}
        brain_after = dict(zip(f["ids"], f["brain_ids"]))
        taken_total += sum(1 for v, w in winners.items() if w != v)
        moved = {}
        for a in dec["allocations"]:
            u, tok = a["agent"], a["tokens"]
            for v, x in zip(a["targets"][1:], a["alloc"][1:]):
                if x > 0 and winners.get(v) == u and tok > 0 and x >= 0.5 * tok:
                    moved[v] = (u, x, a["targets"][1:])
        new_open = {}
        for v, (u, x, nbrs) in moved.items():
            chain = open_chains.pop(u, None)
            if chain is None:
                chain = {"start": u, "t0": t, "nodes": [u], "packets": [], "brains": [], "first_brain": None,
                         "home_kept": [], "returns": []}
            n = chain["nodes"]
            # Returns at lag 2 and 3 for this move, with the chance a random pick among u's
            # neighbours would make them.
            for lag in (2, 3):
                if len(n) >= lag:
                    back = n[-lag]
                    chain["returns"].append((lag, v == back, (1.0 / len(nbrs)) if back in nbrs and back != u else 0.0))
            chain["nodes"].append(v)
            chain["packets"].append(x)
            chain["home_kept"].append(winners.get(u) == u)
            chain.setdefault("home_starved", []).append(u not in winners)
            b = brain_after.get(v)
            if chain["first_brain"] is None:
                chain["first_brain"] = b
            chain["brains"].append(b)
            chain["t1"] = t
            new_open[v] = chain
        adj = None
        for node, chain in open_chains.items():     # chains that did not move on
            w = winners.get(node)
            chain["end"] = "starved" if w is None else ("kept" if w == node else "taken")
            if len(chain["packets"]) >= min_len:
                adj = adj or adjacency(f["edges"])
            close(chain, adj, min_len, rng)
            done.append(chain)
        open_chains = new_open
    adj = adjacency(D.frame_at(run, t1 - 1, 2)["edges"])
    for chain in open_chains.values():
        chain["end"] = "open"
        close(chain, adj, min_len, rng)
        done.append(chain)
    long = [c for c in done if len(c["packets"]) >= min_len]
    lengths = np.array([len(c["packets"]) for c in done])
    launches = {}
    for c in done:
        launches.setdefault(c["start"], []).append(c["t0"])
    per_node = np.array([len(v) for v in launches.values()])
    gaps = Counter(int(b - a) for v in launches.values() for a, b in zip(sorted(v), sorted(v)[1:]))
    out = {"run": run, "from": t0, "to": t1, "chains": len(done), "moves": int(lengths.sum()),
           "taken": taken_total, "min_len": min_len, "long_chains": len(long),
           "share_of_taken_in_long": float(sum(len(c["packets"]) for c in long) / max(1, taken_total)),
           "length_max": int(lengths.max()) if len(lengths) else 0,
           "length_counts": {int(k): int(v) for k, v in sorted(Counter(lengths.tolist()).items())},
           "home_kept_all": float(np.mean([k for c in done for k in c["home_kept"]])) if done else None,
           "home_kept_long": float(np.mean([k for c in long for k in c["home_kept"]])) if long else None,
           "home_starved_long": float(np.mean([k for c in long for k in c["home_starved"]])) if long else None,
           "end_long": dict(Counter(c["end"] for c in long)),
           **{f"lag{lag}_{what}": float(np.mean([r[i] for c in long for r in c["returns"] if r[0] == lag]))
              if any(r[0] == lag for c in long for r in c["returns"]) else None
              for lag in (2, 3) for what, i in (("return", 1), ("chance", 2))},
           "launching_nodes": len(launches),
           "launches_per_node_mean": float(per_node.mean()) if len(per_node) else None,
           "launches_per_node_dispersion": float(per_node.var() / per_node.mean()) if len(per_node) else None,
           "launches_per_node_max": int(per_node.max()) if len(per_node) else 0,
           "launch_gaps": {g: gaps[g] for g in sorted(gaps)[:12]},
           "distinct_over_length": float(np.median([len(set(c["nodes"])) / (len(c["packets"]) + 1) for c in long]))
           if long else None,
           "end_short": dict(Counter(c["end"] for c in done if len(c["packets"]) < min_len))}
    rows = []
    for c in sorted(long, key=lambda c: -len(c["packets"]))[:300]:
        rows.append({"length": len(c["packets"]), "reach": c.get("reach"), "walk": c.get("walk"),
                     "packet_first": c["packets"][0], "packet_last": c["packets"][-1],
                     "packet_max": max(c["packets"]),
                     "same_brain": float(np.mean([b == c["first_brain"] for b in c["brains"]])),
                     "home_kept": float(np.mean(c["home_kept"])), "end": c["end"],
                     "revisits": len(c["nodes"]) - len(set(c["nodes"]))})
    if rows:
        L = np.array([r["length"] for r in rows])
        reach = np.array([np.nan if r["reach"] is None else r["reach"] for r in rows], dtype=float)
        walk = np.array([np.nan if r["walk"] is None else r["walk"] for r in rows], dtype=float)
        both = ~np.isnan(reach) & ~np.isnan(walk) & (walk > 0)
        out.update({
            "examined": len(rows),
            "length_median": float(np.median(L)),
            "reach_over_length": float(np.nanmedian(reach / L)),
            "reach_over_sqrt_length": float(np.nanmedian(reach / np.sqrt(L))),
            "walk_over_sqrt_length": float(np.nanmedian(walk / np.sqrt(L))),
            "reach_over_walk": float(np.median(reach[both] / walk[both])) if both.any() else None,
            "packet_first_median": float(np.median([r["packet_first"] for r in rows])),
            "packet_last_median": float(np.median([r["packet_last"] for r in rows])),
            "packet_max_median": float(np.median([r["packet_max"] for r in rows])),
            "same_brain_median": float(np.median([r["same_brain"] for r in rows])),
            "revisits_median": float(np.median([r["revisits"] for r in rows])),
            "longest": rows[:5], "examined_rows": rows})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--from", dest="t0", type=int, default=1000)
    ap.add_argument("--to", dest="t1", type=int, default=2000)
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = []
    for run in a.runs:
        r = analyse(run, a.t0, a.t1, a.min_len)
        res.append(r)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                          if k not in ("longest", "examined_rows", "length_counts")}), flush=True)
        for row in r.get("longest", []):
            print("   ", row, flush=True)
        if a.out:
            json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
