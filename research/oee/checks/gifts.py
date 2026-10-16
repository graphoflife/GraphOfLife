"""
Gifts: what agents hand to their neighbours, and to whom.

With allow_gifting, an agent may hand tokens to neighbours in the reproduction
phase, out of what its child left it (GraphOfLifeSimple._choose_gifts). The
reproduction frame records every gift paid as [giver, taker, tokens]. This reads
a world's recorded games (book_data.frame_at, as the other checks do) and asks:

- how much is given: tokens a game as a share of the world's, and the share of
  agents that give;
- which way it runs: from the richer of the two to the poorer, or the other way
  (tokens and links as they were after the game before);
- to whom: the share of gift tokens to a clone (the same genotype), against the
  share of the giver's neighbours that are clones — what giving at random
  among one's neighbours would give;
- whether it comes back: of the gifts from g to t, the share followed by a gift
  from t to g within RETURN games, against what t's gifts would bring back if
  each went to a neighbour drawn at random;
- what it keeps alive: the share of gift tokens on links that carried no stake
  either way in the game before — links a gift may be what keeps.

Nothing here has a tuned number; RETURN is the window of the question asked.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import book_data as D

RETURN = 10


def _frame(name: str, t: int, phase: int) -> Optional[Dict[str, Any]]:
    try:
        return D.frame_at(name, t, phase)
    except Exception:
        return None


def _state(frame: Dict[str, Any]) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, set], Dict[int, int]]:
    """Tokens, degree, neighbours and genotype of every agent of a frame."""
    ids = [int(u) for u in frame["ids"]]
    tokens = {u: int(x) for u, x in zip(ids, frame.get("tokens") or [])}
    brains = {u: int(b) for u, b in zip(ids, frame.get("brain_ids") or [])}
    near: Dict[int, set] = {u: set() for u in ids}
    for a, b in frame.get("edges") or []:
        a, b = int(a), int(b)
        if a in near and b in near:
            near[a].add(b)
            near[b].add(a)
    return tokens, {u: len(v) for u, v in near.items()}, near, brains


def _staked_pairs(frame: Dict[str, Any]) -> set:
    """The links on which some token was staked, either way, in a game."""
    out = set()
    for rec in (frame.get("decisions") or {}).get("allocations") or []:
        u = int(rec["agent"])
        for v, x in zip(rec.get("targets") or [], rec.get("alloc") or []):
            if x and int(v) != u:
                out.add((min(u, int(v)), max(u, int(v))))
    return out


def analyse(name: str, t0: int, t1: int) -> Dict[str, Any]:
    games = 0
    total_gift = total_tokens = givers = agents = 0
    down = up = level = 0
    to_richer_degree = to_poorer_degree = 0
    to_clone = clone_chance = 0.0
    on_unused = 0
    gifts_seen: List[Tuple[int, int, int, int]] = []          # (game, giver, taker, tokens)
    degree_at: Dict[Tuple[int, int], int] = {}
    for t in range(t0 + 1, t1):
        before = _frame(name, t - 1, 2)
        rep = _frame(name, t, 1)
        if not before or not rep:
            continue
        gifts = (rep.get("decisions") or {}).get("gifts")
        if gifts is None:
            continue
        games += 1
        tokens, degree, near, brains = _state(before)
        staked = _staked_pairs(before)
        total_tokens += sum(tokens.values())
        agents += len(tokens)
        givers += len({int(g) for g, _, _ in gifts})
        for g, r, x in gifts:
            g, r, x = int(g), int(r), int(x)
            total_gift += x
            gifts_seen.append((t, g, r, x))
            degree_at[(t, r)] = degree.get(r, 0)
            tg, tr = tokens.get(g, 0), tokens.get(r, 0)
            if tg > tr:
                down += x
            elif tg < tr:
                up += x
            else:
                level += x
            if degree.get(r, 0) > degree.get(g, 0):
                to_richer_degree += x
            elif degree.get(r, 0) < degree.get(g, 0):
                to_poorer_degree += x
            if g in brains and r in brains:
                to_clone += x * (brains[g] == brains[r])
                friends = near.get(g) or set()
                if friends:
                    clone_chance += x * sum(brains.get(v) == brains[g] for v in friends) / len(friends)
            if (min(g, r), max(g, r)) not in staked:
                on_unused += x
    # Returns: a gift from g to t answered by one from t to g within RETURN games.
    by_giver: Dict[int, List[Tuple[int, int]]] = {}
    for t, g, r, _ in gifts_seen:
        by_giver.setdefault(g, []).append((t, r))
    returned = expected = 0.0
    for t, g, r, _ in gifts_seen:
        later = [(s, to) for s, to in by_giver.get(r, []) if t < s <= t + RETURN]
        if any(to == g for _, to in later):
            returned += 1
        # Had each of r's later gifts gone to a neighbour drawn at random, the chance one reaches g.
        miss = 1.0
        for s, _ in later:
            deg = max(1, degree_at.get((s, r), 0) or 1)
            miss *= 1.0 - 1.0 / deg
        expected += 1.0 - miss
    n = len(gifts_seen)
    return {
        "games": games,
        "gifts": n,
        "gift_share": total_gift / total_tokens if total_tokens else None,
        "givers_share": givers / agents if agents else None,
        "tokens_per_gift": total_gift / n if n else None,
        "downhill": down / total_gift if total_gift else None,
        "uphill": up / total_gift if total_gift else None,
        "to_better_connected": to_richer_degree / total_gift if total_gift else None,
        "to_less_connected": to_poorer_degree / total_gift if total_gift else None,
        "to_clone": to_clone / total_gift if total_gift else None,
        "to_clone_chance": clone_chance / total_gift if total_gift else None,
        "returned": returned / n if n else None,
        "returned_chance": expected / n if n else None,
        "on_unused_links": on_unused / total_gift if total_gift else None,
    }
