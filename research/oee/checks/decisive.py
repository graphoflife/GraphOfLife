"""
Help, defined by the rules: a stake without which its target's holder would have lost the node.

Every stake an agent puts on a neighbour's node changes the chance that the
node's holder keeps it. In the plain game a stake on someone else's node is a
bid against its holder and can only lower that chance. Two rules let a stake
raise it: revolutions (a crowd of small revolutionary stakes overthrows the
largest staker, and the winner is drawn from the crowd) and pooled alliances
(an ally's stake counts for the holder's side). Which of them a world has
decides whether one agent can help another at all.

For every stake u -> v (u not v) of every recorded game this computes, from the
stakes as they were and the world's own rule for deciding a node,

    effect = P(v keeps v | all stakes) - P(v keeps v | all stakes but u's)

exactly: ties and the revolution's draw are counted as the probabilities they
are, nothing is sampled. A positive effect is help, a negative one an attack.
Summed over a window: the nodes saved by others' stakes and the nodes cost by
them, per 100 agents a game; whether help is returned and whether it goes
round; and whether it goes to clones more often than clones are among the
neighbours.

Returned: of the pairs in which u helped v in a window of games, the share in
which v also helped u. Its chance level gives every pair the same chance per
game they spent as neighbours, so that pairs that stayed neighbours longer are
not mistaken for pairs that return help: P(v helps u at least once) =
1 - (1 - q)^n, n the games v had u among its neighbours, q the help events per
neighbour-game of the window.

Pooled over the whole run, the same comparison counts returned pairs and the
pairs chance would return, summed over the windows. Help can only pass where a
situation allows it (allies, or two neighbours facing the same attacker), and
such situations are symmetric, so help between them looks returned whether or
not anyone answers help with help: returned help above chance is not yet
reciprocity. (Timing cannot tell the two apart either: every pair of help
events, u to v and then v to u, is "after" seen from u and "before" seen from
v, so counted over both directions the two are always equal.)

Round: the net help between two agents in a window (u's help to v minus v's
to u) is a flow on the network, split like the token flow of circulation.py
into the part that runs one way and the part that goes round loops, against
the share that would go round if each pair's net help had a random direction.

    python3 research/oee/checks/decisive.py RUN [...] --from 1000 --to 2000
"""
import argparse
import json
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/home/stefan/Documents/CodingProjects/GraphOfLife Claude")

import numpy as np

import book_data as D


def p_plurality(offers, revs, who, h):
    """
    P(who wins), the engine's rule (GraphOfLife._weighed and _resolve_winner) counted exactly.

    The largest weighed offer is drawn among those tied for it; the crowd is every other
    revolutionary. Largest stakers with the same revolutionary amount leave the same crowd
    behind, so each such group is walked once: a hub whose hundred neighbours tie with one
    token each costs one walk, not a hundred.
    """
    if not offers:
        return 0.0
    weighed = dict(offers)
    if h != 1.0 and who in weighed:
        weighed[who] = weighed[who] * h
    mx = max(weighed.values())
    tops = [a for a, x in weighed.items() if x == mx]
    revs = {a: t for a, t in revs.items() if t > 0}
    amounts = sorted(revs.values())
    total = sum(amounts)
    count = Counter(amounts)
    r_who = revs.get(who, 0)
    who_top = who in weighed and weighed[who] == mx
    p = 0.0
    for r_h, n_h in Counter(revs.get(a, 0) for a in tops).items():
        # The crowd without one largest staker of revolutionary amount r_h.
        mob = list(amounts)
        if r_h:
            mob.remove(r_h)
        lower, i, crowd = 0, 0, None
        while i < len(mob):
            amount = mob[i]
            while i < len(mob) and mob[i] == amount:
                lower += amount
                i += 1
            if lower > total - r_h - lower + mx:
                crowd = amount
                break
        who_here = who_top and r_who == r_h          # who is one of this group's n_h
        if crowd is None:                            # the largest staker keeps it
            p += who_here / len(tops)
        else:                                        # drawn from the crowd's step
            size = count[crowd] - (r_h == crowd)
            p += (r_who == crowd) * (n_h - who_here) / size / len(tops)
    return p


def p_pooled(offers, who, h, allied):
    """P(the owner's side wins), the pilots' pooled rule (proto.ProtoWorld._resolve_sides) counted exactly."""
    weighed = dict(offers)
    if h != 1.0 and who in weighed:
        weighed[who] = weighed[who] * h
    owner = [who] + [b for b in weighed if b != who and (b, who) in allied]
    rest = [b for b in weighed if b not in owner]
    sides = [owner]
    while rest:
        group = [rest.pop(0)]
        frontier = list(group)
        while frontier:
            x = frontier.pop()
            joined = [y for y in rest if (x, y) in allied]
            for y in joined:
                rest.remove(y)
            group += joined
            frontier += joined
        sides.append(group)
    totals = [sum(weighed.get(b, 0) for b in s) for s in sides]
    best = max(totals)
    tied = [i for i, t in enumerate(totals) if t == best]
    return 1.0 / len(tied) if 0 in tied else 0.0


def effects(frame, rules):
    """(u, v, effect, tokens) for every stake on another agent's node in one game."""
    offers = defaultdict(dict)
    revs = defaultdict(dict)
    for a in frame["decisions"]["allocations"]:
        u = a["agent"]
        rv = a.get("revolt") or [0] * len(a["targets"])
        for v, x, r in zip(a["targets"], a["alloc"], rv):
            if x > 0:
                offers[v][u] = offers[v].get(u, 0) + x
            if r > 0:
                revs[v][u] = revs[v].get(u, 0) + r
    allied = None
    if rules.get("pooled"):
        allied = {(u, v) for u, v in frame.get("allies", [])} | {(v, u) for u, v in frame.get("allies", [])}
    h = rules.get("incumbency", 1.0)
    out = []
    for v, off in offers.items():
        others = [u for u in off if u != v]
        if not others:
            continue
        if allied is not None:
            keep = lambda o: p_pooled(o, v, h, allied)
            base = keep(off)
            for u in others:
                out.append((u, v, base - keep({b: x for b, x in off.items() if b != u}), off[u]))
        else:
            rv = revs.get(v, {})
            base = p_plurality(off, rv, v, h)
            for u in others:
                out.append((u, v, base - p_plurality({b: x for b, x in off.items() if b != u},
                                                     {b: x for b, x in rv.items() if b != u}, v, h), off[u]))
    return out


def lab_rules(run):
    import gol_store as store
    c = store.load_meta(run)["config"]
    return {"incumbency": float(c.get("incumbency") or 1.0), "revolutions": bool(c.get("allow_revolutions"))}


def returned(help_pairs, exposure):
    """Share of helping pairs whose reverse also helped, and its chance level matched to exposure."""
    events = sum(help_pairs.values())
    games = sum(exposure.values())
    if not help_pairs or not games:
        return None, None
    q = events / games
    seen = [(v, u) in help_pairs for u, v in help_pairs]
    chance = [1 - (1 - q) ** exposure.get((v, u), 0) for u, v in help_pairs]
    return float(np.mean(seen)), float(np.mean(chance))


def round_share(help_amount, rng, draws=4):
    """Circulation share of the net help flow, and the sign-flip chance level."""
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location(
        "circulation", os.path.join(os.path.dirname(os.path.abspath(__file__)), "circulation.py"))
    circ = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(circ)
    net = {}
    for (u, v), e in help_amount.items():
        key = (u, v) if u < v else (v, u)
        net[key] = net.get(key, 0.0) + (e if u < v else -e)
    if len(net) < 10:
        return None, None
    row = circ.hodge(net, draws, rng)[1]
    if not row["random_share"]:
        return None, None              # help that closes no loop can neither go round nor fail to
    return row["circulation_share"], row.get("flip_share")


def analyse(run, t0, t1, rules=None, window=50):
    rules = rules or lab_rules(run)
    games = agents = 0
    stakes = helps = harms = 0
    saved = cost = 0.0
    windows = []
    kin_help = kin_chance_w = kin_chance = 0.0
    exposure = defaultdict(int)          # (a, v): games in which v was among a's neighbours
    help_events = defaultdict(int)       # (u, v): games in which u's stake helped v
    help_amount = defaultdict(float)     # (u, v): summed effect of u's help to v
    rng = np.random.default_rng(1)
    for t in range(t0, t1):
        try:
            f = D.frame_at(run, t, 2)
            before = D.frame_at(run, t, 1)
        except Exception:
            continue
        if not f or "decisions" not in f:
            continue
        games += 1
        agents += len(f["decisions"]["allocations"])
        brain = dict(zip(before["ids"], before["brain_ids"])) if before else {}
        for a in f["decisions"]["allocations"]:
            for v in a["targets"][1:]:
                exposure[(a["agent"], v)] += 1
        nbrs = {a["agent"]: a["targets"][1:] for a in f["decisions"]["allocations"]}
        for u, v, e, x in effects(f, rules):
            stakes += 1
            if e > 0:
                helps += 1
                saved += e
                help_events[(u, v)] += 1
                help_amount[(u, v)] += e
                # Kin: the share of help given to clones, against the clone share of each
                # helper's neighbours, weighted by the help it gave.
                if brain and nbrs.get(u):
                    kin_help += e * (brain.get(u) == brain.get(v))
                    kin_chance += e * np.mean([brain.get(w) == brain.get(u) for w in nbrs[u]])
                    kin_chance_w += e
            elif e < 0:
                harms += 1
                cost -= e
        if (t - t0 + 1) % window == 0 or t == t1 - 1:
            if help_events:
                back, back_chance = returned(help_events, exposure)
                circ, circ_flip = round_share(help_amount, rng)
                windows.append({"from": t - (t - t0) % window, "returned": back, "returned_chance": back_chance,
                                "round": circ, "round_flip": circ_flip, "pairs": len(help_events)})
            exposure, help_events, help_amount = defaultdict(int), defaultdict(int), defaultdict(float)
    med = lambda k: (float(np.median([w[k] for w in windows if w[k] is not None]))
                     if any(w[k] is not None for w in windows) else None)
    pooled_back = sum(w["returned"] * w["pairs"] for w in windows if w["returned"] is not None)
    pooled_chance = sum(w["returned_chance"] * w["pairs"] for w in windows if w["returned_chance"] is not None)
    return {"run": run, "from": t0, "to": t1, "rules": rules, "games": games,
            "agents_mean": agents / max(1, games), "stakes_on_others": stakes,
            "helping_stakes": helps, "attacking_stakes": harms,
            "nodes_saved_per_100_agents_a_game": 100 * saved / max(1, agents),
            "nodes_cost_per_100_agents_a_game": 100 * cost / max(1, agents),
            "help_returned": med("returned"), "help_returned_chance": med("returned_chance"),
            "returned_pairs": pooled_back, "returned_pairs_chance": pooled_chance,
            "help_round": med("round"), "help_round_flip": med("round_flip"),
            "helping_pairs_per_window": med("pairs"), "windows": windows,
            "help_to_clones": kin_help / kin_chance_w if kin_chance_w else None,
            "help_to_clones_chance": kin_chance / kin_chance_w if kin_chance_w else None}


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
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "windows"}),
              flush=True)
        if a.out:
            json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
