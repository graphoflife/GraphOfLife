"""
Structures that outlast their parts: persistent flow channels, loops, and travelling packets.

In Conway's Game of Life a glider is five cells that rebuild themselves one
step further on: the pattern persists while the cells that carry it change. The
question here is whether the agents' game makes anything like that — patterns
of play that persist although agents die, brains are replaced and tokens move —
and whether taking part in one pays. Everything is read from a lab run's
recorded games (frames with decisions).

Measured per run over a window of games (the game phase of each iteration):

channels   A stake from u on v is a *channel*. How often does a channel that
           carried tokens in one game carry them again in the next? Compared
           with chance: if each agent staked on as many neighbours as it did,
           but drawn at random among its neighbours, a channel would recur
           with probability (targets of u next game) / (neighbours of u).
loops      Two-loops (u on v and v on u in the same game) and three-loops
           (u on v, v on w, w on u), how many, and how long the same loop
           (the same agents, the same direction) keeps recurring, against the
           same chance model.
gifts      A stake that does not take its target is a gift to whoever holds
           it. The share of staked tokens that are gifts, and the share of gift
           tokens an agent gives that it also receives back as gifts in the same
           game (an upper bound on the gifts that can go round).
packets    A *packet* moves when an agent takes a neighbour's node with at
           least half of its tokens: brain and wealth move together. If the
           node taken does the same in the next game, the packet travels on. A
           chain of such moves is a glider. How long do chains get?
carriers   For channels that recur for at least `long` games: how many of them
           saw the brain of one of their two agents replaced while they lasted
           — a structure outliving the brains that made it.

    python3 research/oee/checks/structures.py RUN --from 1000 --to 2000 [--long 10]
"""
import argparse
import json
import sys
from collections import defaultdict

sys.path.insert(0, "/home/stefan/Documents/CodingProjects/GraphOfLife")

import numpy as np

import book_data as D


def games(run, t0, t1):
    for t in range(t0, t1):
        try:
            f = D.frame_at(run, t, 2)
        except Exception:
            continue
        if not f or "decisions" not in f:
            continue
        yield t, f


def analyse(run, t0, t1, long=10):
    prev_edges = None
    prev_deg = None
    prev_targets = None
    repeat, chance = [], []
    run_len = {}                 # edge -> current run length
    finished_runs = []
    loops2, loops3, loops2_chance = [], [], []
    movers = []                  # share of agents that move a packet, per game (the chance a chain goes on)
    pair_run = {}                # unordered pair with stakes both ways -> run length
    pair_finished = []
    member_kept, other_kept = [], []
    loop3_run = {}
    loop3_finished = []
    gifts_share, gift_loop_share = [], []
    brain_at = {}                # node -> brain id now
    brain_changes = defaultdict(int)   # node -> number of brain changes seen
    edge_start_changes = {}      # edge -> brain changes of its two nodes when it started
    long_edges, long_edges_carried = 0, 0
    packet_chain = {}            # node holding a packet -> chain length so far
    chains = []
    n_games = 0
    for t, f in games(run, t0, t1):
        n_games += 1
        dec = f["decisions"]
        winners = {w["node"]: w["winner"] for w in dec["winners"]}
        edges = {}
        targets_of = {}
        deg = {}
        tokens_before = {}
        for a in dec["allocations"]:
            u = a["agent"]
            tokens_before[u] = a["tokens"]
            tg, al = a["targets"], a["alloc"]
            deg[u] = len(tg) - 1
            out = [(v, x) for v, x in zip(tg[1:], al[1:]) if x > 0]
            targets_of[u] = len(out)
            for v, x in out:
                edges[(u, v)] = x
        # Brains now, and how many times each node's brain has changed.
        for node, bid in zip(f["ids"], f["brain_ids"]):
            old = brain_at.get(node)
            if old is not None and old != bid:
                brain_changes[node] += 1
            brain_at[node] = bid
        # Channels: recurrence against chance.
        if prev_edges is not None:
            for (u, v) in prev_edges:
                if u in targets_of and deg.get(u):
                    repeat.append(1.0 if (u, v) in edges else 0.0)
                    chance.append(targets_of[u] / deg[u])
        # Run lengths of channels.
        new_run = {}
        for e in edges:
            new_run[e] = run_len.get(e, 0) + 1
            if e not in run_len:
                edge_start_changes[e] = brain_changes[e[0]] + brain_changes[e[1]]
        for e, L in run_len.items():
            if e not in edges:
                finished_runs.append(L)
                if L >= long:
                    long_edges += 1
                    if brain_changes[e[0]] + brain_changes[e[1]] > edge_start_changes.get(e, 0):
                        long_edges_carried += 1
                edge_start_changes.pop(e, None)
        run_len = new_run
        # Loops, and how many two-loops chance would make: v stakes on u with probability
        # (targets of v) / (neighbours of v), whatever u does.
        two = sum(1 for (u, v) in edges if u < v and (v, u) in edges)
        loops2.append(two / max(1, len(edges)))
        loops2_chance.append(sum(targets_of.get(v, 0) / deg[v] for (u, v) in edges if deg.get(v)) / 2
                             / max(1, len(edges)))
        out_adj = defaultdict(list)
        for (u, v) in edges:
            out_adj[u].append(v)
        three = set()
        for u, vs in out_adj.items():
            for v in vs:
                for w in out_adj.get(v, ()):
                    if w != u and w != v and (w, u) in edges:
                        cyc = (u, v, w)
                        i = cyc.index(min(cyc))
                        three.add(cyc[i:] + cyc[:i])     # one key per directed loop
        loops3.append(len(three) / max(1, len(edges)))
        new3 = {}
        for c in three:
            new3[c] = loop3_run.get(c, 0) + 1
        for c, L in loop3_run.items():
            if c not in three:
                loop3_finished.append(L)
        loop3_run = new3
        # Gifts: stakes that did not take their target.
        staked = sum(edges.values())
        gift_edges = {e: x for e, x in edges.items() if winners.get(e[1]) != e[0]}
        gifts_share.append(sum(gift_edges.values()) / max(1, staked))
        in_g = defaultdict(int)
        out_g = defaultdict(int)
        for (u, v), x in gift_edges.items():
            out_g[u] += x
            in_g[v] += x
        looped = sum(min(out_g[u], in_g[u]) for u in out_g)
        gift_loop_share.append(looped / max(1, sum(gift_edges.values())))
        # Packets: an agent that takes a node with at least half of its tokens.
        moved_to = {}
        for (u, v), x in edges.items():
            if winners.get(v) == u and tokens_before.get(u, 0) > 0 and x >= 0.5 * tokens_before[u]:
                moved_to[v] = u
        movers.append(len(set(moved_to.values())) / max(1, len(tokens_before)))
        # Partnerships: two agents staking on each other, game after game. Do members keep
        # their own node more often than agents with stakes out who are in no partnership?
        pairs_now = {(u, v) for (u, v) in edges if u < v and (v, u) in edges}
        new_pairs = {}
        for pr in pairs_now:
            new_pairs[pr] = pair_run.get(pr, 0) + 1
        for pr, L in pair_run.items():
            if pr not in pairs_now:
                pair_finished.append(L)
        pair_run = new_pairs
        in_long_pair = {x for pr, L in pair_run.items() if L >= 3 for x in pr}
        staking = {u for (u, v) in edges}
        for u in staking:
            kept = 1.0 if winners.get(u) == u else 0.0
            (member_kept if u in in_long_pair else other_kept).append(kept)
        new_chain = {}
        for v, u in moved_to.items():
            new_chain[v] = packet_chain.get(u, 0) + 1
        for node, L in packet_chain.items():
            if node not in moved_to.values():
                chains.append(L)
        packet_chain = new_chain
        prev_edges, prev_deg, prev_targets = edges, deg, targets_of
    chains.extend(packet_chain.values())
    finished_runs.extend(run_len.values())
    loop3_finished.extend(loop3_run.values())
    r = np.array(repeat)
    c = np.array(chance)
    runs = np.array(finished_runs)
    ch = np.array(chains) if chains else np.zeros(1)
    l3 = np.array(loop3_finished) if loop3_finished else np.zeros(1)
    pair_finished.extend(pair_run.values())
    pf = np.array(pair_finished) if pair_finished else np.zeros(1)
    p_on = float(np.mean(movers))          # chance that a chain goes on, if moves were independent
    n_chains = len(ch)
    return {
        "run": run, "from": t0, "to": t1, "games": n_games,
        "channel_repeat": float(r.mean()), "channel_repeat_chance": float(c.mean()),
        "channel_runs_mean": float(runs.mean()), "channel_runs_p99": float(np.percentile(runs, 99)),
        "channels_lasting_long": int(long_edges),
        "of_those_outliving_a_brain": float(long_edges_carried / max(1, long_edges)),
        "two_loops_per_channel": float(np.mean(loops2)), "three_loops_per_channel": float(np.mean(loops3)),
        "three_loop_runs_mean": float(l3.mean()), "three_loop_runs_max": int(l3.max()),
        "gift_share": float(np.mean(gifts_share)), "gifts_balanced": float(np.mean(gift_loop_share)),
        "packet_moves": int(ch.sum()), "packet_chain_mean": float(ch.mean()), "packet_chain_max": int(ch.max()),
        "packet_chains_ge3": int((ch >= 3).sum()),
        "two_loops_chance": float(np.mean(loops2_chance)),
        "packet_go_on_chance": p_on,
        "packet_chains_ge3_chance": float(n_chains * p_on ** 2),
        "partnership_runs_mean": float(pf.mean()), "partnership_runs_max": int(pf.max()),
        "partnerships_ge3": int((pf >= 3).sum()),
        "kept_in_partnership": float(np.mean(member_kept)) if member_kept else float("nan"),
        "kept_otherwise": float(np.mean(other_kept)) if other_kept else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--from", dest="t0", type=int, default=1000)
    ap.add_argument("--to", dest="t1", type=int, default=2000)
    ap.add_argument("--long", type=int, default=10)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    results = []
    for run in a.runs:
        res = analyse(run, a.t0, a.t1, a.long)
        results.append(res)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
    if a.out:
        json.dump(results, open(a.out, "w"))


if __name__ == "__main__":
    main()
