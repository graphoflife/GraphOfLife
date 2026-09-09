#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightning: how much of a Blotto phase's token flow goes round in circles.

A **lightning** is a closed loop of token flow — tokens leaving an agent,
crossing links, and arriving back where they started, like current round a
circuit. One token sent round a triangle is a small thing; ten tokens carried
round a ten-edge loop is a large one, so a loop of `L` hops carries `L` tokens
(one per edge) and scores `L²`. Every token belongs to at most one lightning.

**The exact maximum is out of reach, and it is worth saying why.** Maximising
Σ L² rewards packing the flow into few long loops, so the optimum contains
"find the longest cycle" — which is the Hamiltonian cycle problem wearing a
hat. There is no efficient exact algorithm and there will not be one.

So this reports a bracket, and labels both ends:

  `lightningScore`   Σ L² from a greedy peel. Walk the flow until a node
                     repeats — which is exactly the rule "the circle is closed
                     as soon as a node is in it twice" — take that loop out,
                     and go again. A **lower bound**: the true maximum is at
                     least this.

  `cyclingShare`     the tokens those loops used, over all the tokens that
                     moved. Also a lower bound, and the more comparable of the
                     two because it does not grow with the size of the world.

  `flowImbalance`    the share of flow that conservation *forbids* from
                     circulating: an agent that receives more than it sends has
                     the difference stranded as one-way transport, and no
                     decomposition can loop it. Exact, no heuristic, and an
                     upper bound — cycling can never exceed `1 - imbalance`.

Between the floor and the ceiling sits the truth. If the two are close the
greedy peel has found nearly everything there is; if they are far apart the
number is a weak reading and should be treated as one.

Everything is reported twice, on the flow as sent and on the **net** flow.

  gross   every token as allocated. Two agents trading five one way and three
          the other are five and three, and three of those are a two-hop loop.

  net     the same pair is two tokens one way and nothing back. Every edge now
          carries flow in at most one direction, so no two-hop loop can exist
          at all and every lightning left is tokens genuinely going *round*
          rather than sloshing between neighbours.

The net figure is the stricter claim and the more interesting one. Mutual
exchange between neighbours is easy and everywhere; a circuit that survives
cancellation is not. Note that netting cannot change any agent's balance — it
removes equal and opposite amounts — so `flowImbalance` is the same ceiling for
both, and comparing the two shares against it says how much of the circulation
was only reciprocity.

This is the same quantity ecology calls a **cycling index** (Finn 1976) and
that Helmholtz–Hodge theory calls the **circulation** of a flow — the part that
no potential can explain. Both give exact answers on small networks; neither
scales to forty thousand agents, which is why the bracket is what travels.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

#: Loops to pull out before giving up, so one frame cannot run away with the
#: whole summary. Reached only on a flow with an enormous amount of circulation,
#: which is itself worth knowing and is reported.
MAX_LOOPS = 20000

#: How far a single walk may wander before being abandoned as unproductive.
MAX_WALK = 4096


def _network(frame: Dict[str, Any]) -> Tuple[Dict[int, Dict[int, int]], int]:
    """
    Who sent how many tokens to whom, and how many moved in total.

    Self-allocation is dropped: an agent staking on its own node has not sent
    anything anywhere, and a self-loop is not a circuit.
    """
    decisions = frame.get("decisions") or {}
    flow: Dict[int, Dict[int, int]] = {}
    total = 0
    for entry in decisions.get("allocations") or []:
        source = int(entry.get("agent"))
        targets = entry.get("targets") or []
        amounts = entry.get("alloc") or []
        for position, raw in enumerate(targets):
            target = int(raw)
            if target == source or position >= len(amounts):
                continue
            amount = int(amounts[position])
            if amount <= 0:
                continue
            flow.setdefault(source, {})
            flow[source][target] = flow[source].get(target, 0) + amount
            total += amount
    return flow, total


def _netted(flow: Dict[int, Dict[int, int]]) -> Tuple[Dict[int, Dict[int, int]], int]:
    """
    The same flow with reciprocal amounts cancelled off against each other.

    Five one way and three the other becomes two one way and nothing back. Every
    edge then carries flow in at most one direction, which is what makes a
    two-hop loop impossible and every remaining lightning a real circuit.
    """
    out: Dict[int, Dict[int, int]] = {}
    total = 0
    seen = set()
    for source, targets in flow.items():
        for target, amount in targets.items():
            pair = (source, target) if source < target else (target, source)
            if pair in seen:
                continue
            seen.add(pair)
            back = flow.get(target, {}).get(source, 0)
            net = amount - back
            if net > 0:
                out.setdefault(source, {})[target] = net
                total += net
            elif net < 0:
                out.setdefault(target, {})[source] = -net
                total += -net
    return out, total


def _imbalance(flow: Dict[int, Dict[int, int]], total: int) -> float:
    """
    The share of flow conservation forbids from circulating.

    Every token in a loop leaves its agent and returns to it, so a loop
    contributes nothing to any agent's net balance. Whatever net imbalance
    remains is therefore one-way transport that no decomposition can bend into
    a circle. Halved because each stranded token shows up twice — once as a
    surplus, once as a deficit.
    """
    if total <= 0:
        return 0.0
    net: Dict[int, int] = {}
    for source, targets in flow.items():
        for target, amount in targets.items():
            net[source] = net.get(source, 0) - amount
            net[target] = net.get(target, 0) + amount
    stranded = sum(abs(v) for v in net.values()) / 2
    return min(1.0, stranded / total)


def _peel(flow: Dict[int, Dict[int, int]], total: int) -> Tuple[int, int, int]:
    """
    Peel loops out of a flow, greedily and longest-first-ish.

    The walk prefers a step to an agent it has not been to yet, which is what
    makes it find long loops rather than closing on the first triangle it
    stumbles into. When there is nowhere new to go it takes any step it can,
    and the moment it lands on an agent already in the walk the loop is closed
    — the reader's own rule.

    Deterministic throughout: neighbours are taken in sorted order and starts
    in sorted order, so the same frame gives the same number every time and
    both implementations give the same number as each other.
    """
    # Residual flow, and a stable neighbour order for every agent.
    residual = {source: dict(targets) for source, targets in flow.items()}
    order = {source: sorted(targets) for source, targets in flow.items()}

    score = 0
    used = 0
    longest = 0
    loops = 0

    for start in sorted(residual):
        while loops < MAX_LOOPS:
            walk = [start]
            where = {start: 0}
            closed = None

            for _ in range(MAX_WALK):
                here = walk[-1]
                options = order.get(here) or []
                step = None
                fallback = None
                for candidate in options:
                    if residual.get(here, {}).get(candidate, 0) <= 0:
                        continue
                    if candidate not in where:
                        step = candidate
                        break
                    if fallback is None:
                        fallback = candidate
                # Somewhere new if there is anywhere new; otherwise close.
                step = step if step is not None else fallback
                if step is None:
                    break
                if step in where:
                    closed = where[step]
                    walk.append(step)
                    break
                where[step] = len(walk)
                walk.append(step)

            if closed is None:
                break

            # The loop is the tail of the walk from where it rejoined itself.
            hops = len(walk) - 1 - closed
            if hops < 2:
                # A pair sending to each other is a loop of two, which counts;
                # anything shorter is not a circuit at all.
                if hops < 2:
                    break
            for i in range(closed, len(walk) - 1):
                a, b = walk[i], walk[i + 1]
                residual[a][b] -= 1
                if residual[a][b] <= 0:
                    del residual[a][b]
                    order[a] = [n for n in order[a] if n != b]

            score += hops * hops
            used += hops
            longest = max(longest, hops)
            loops += 1

    return score, used, longest


def lightning(frame: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Both readings of one frame: the flow as sent, and the net flow."""
    flow, total = _network(frame)
    if total <= 0:
        return {"lightningScore": None, "cyclingShare": None,
                "lightningLongest": None, "flowImbalance": None,
                "netLightningScore": None, "netCyclingShare": None,
                "netLightningLongest": None, "netFlowShare": None}

    score, used, longest = _peel(flow, total)
    net, net_total = _netted(flow)
    net_score, net_used, net_longest = _peel(net, net_total)

    return {
        "lightningScore": float(score),
        "cyclingShare": min(1.0, used / total),
        "lightningLongest": float(longest),
        "flowImbalance": _imbalance(flow, total),
        "netLightningScore": float(net_score),
        # Against the net total, so it answers "of the tokens that actually went
        # somewhere, how many went round" rather than being diluted by the
        # reciprocity that netting just removed.
        "netCyclingShare": (min(1.0, net_used / net_total) if net_total else 0.0),
        "netLightningLongest": float(net_longest),
        # How much of the gross flow survived cancellation. Low means neighbours
        # are mostly trading back and forth.
        "netFlowShare": min(1.0, net_total / total),
    }
