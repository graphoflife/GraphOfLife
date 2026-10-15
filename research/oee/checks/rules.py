"""
The game's rule for deciding a node, counted exactly, and the effect of every stake on its target's holder.

Shared by research/oee/checks/decisive.py, which measures help in recorded
games, and by the pilots (research/oee/pilots/proto.py, option helped_by),
whose agents see what each neighbour's stake did to their node in the last game.

    effect(u -> v) = P(v keeps v | all stakes) - P(v keeps v | all stakes but u's)

Positive is help, negative an attack. Nothing is sampled: ties and the
revolution's draw are counted as the probabilities they are.
"""
from collections import Counter, defaultdict


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
