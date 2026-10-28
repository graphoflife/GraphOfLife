#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The game's stakes placed a token at a time: allocation = weakest_first or
strongest_first, as the original engine placed them (GraphOfLife.py of
2026-08-01, BLOTTO_ALLOCATION_MODE), back at the author's request
(2026-10-28).

Under full allocation every agent looks once and places all its tokens at
once, before anyone has placed anything. Here the game is played in rounds.
In each round some agents place one token each: under weakest_first every
agent with a token left, so the poorest are done first and the richest places
its last tokens alone; under strongest_first only the agents with the most
tokens left, so the richest starts alone, the next richest join it as it comes
down to them, and everyone places a last token in the last round. The game has
as many rounds as its richest agent has tokens.

Before each round an agent that places a token looks again, with the board in
view: for each of its candidates, what that node has drawn so far and from
whom, who leads it and by how much, and what is still to be placed around it
(gol_config.BOARD). It puts its token where its brain scores highest, and
marks it revolutionary if its brain says so for that node. Everyone in a round
reads the board as it stood when the round began, and their tokens land
together after it, so the order of the ids decides nothing.

What an agent writes this phase comes from a first look, with the board still
empty, as under full allocation. Tokens kept home under keep_home sit on the
agent's own node before the first round; only the rest are placed in rounds.
The spread-or-all-in head is not read: every token is a choice of its own.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from gol_config import BOARD


def who_places(remaining: Dict[int, int], allocation: str) -> List[int]:
    """The agents that place a token in the next round, in the order of their ids; none when all are placed."""
    left = [u for u, r in remaining.items() if r > 0]
    if allocation == "strongest_first" and left:
        most = max(remaining[u] for u in left)
        left = [u for u in left if remaining[u] == most]
    return sorted(left)


def schedule(remaining: Dict[int, int], allocation: str) -> List[List[int]]:
    """
    Every round of a game at once, as who_places gives them one at a time: everyone in a round places
    exactly one token, so the agents with r tokens left are in the first r rounds under weakest_first and
    the last r under strongest_first, and nothing need be looked at again between rounds.
    """
    ranked = sorted((u for u, r in remaining.items() if r > 0), key=lambda u: -remaining[u])
    rounds = remaining[ranked[0]] if ranked else 0
    out, k = [], 0
    for r in range(1, rounds + 1):
        least = r if allocation == "weakest_first" else rounds - r + 1
        if allocation == "weakest_first":
            k = sum(1 for u in ranked if remaining[u] >= least) if r == 1 else k
            while k and remaining[ranked[k - 1]] < least:
                k -= 1
        else:
            while k < len(ranked) and remaining[ranked[k]] >= least:
                k += 1
        out.append(sorted(ranked[:k]))
    return out


def would_take(stakes: Dict[int, int], revolts: Dict[int, int]) -> Optional[int]:
    """
    Who takes a node if the game ended with these stakes: the engine's rule
    (GraphOfLife._resolve_winner), with ties broken to the largest id rather than
    at random, so that reading the board draws no chance. None when nothing is on it.
    """
    if not stakes:
        return None
    lead = max(stakes.values())
    hegemon = max(agent for agent, amount in stakes.items() if amount == lead)
    mob = sorted((amount, agent) for agent, amount in revolts.items() if agent != hegemon and amount > 0)
    total = sum(amount for amount, _ in mob)
    lower, i = 0, 0
    while i < len(mob):
        rung = []
        amount = mob[i][0]
        while i < len(mob) and mob[i][0] == amount:
            rung.append(mob[i][1])
            lower += amount
            i += 1
        if lower > total - lower + lead:
            return max(rung)
    return hegemon


class Board:
    """The game as it stands: every stake so far, and what each agent has left to place."""

    def __init__(self, world, neighs: Dict[int, List[int]], allocations_to: Dict[int, Dict[int, int]],
                 revolution_to: Dict[int, Dict[int, int]], incoming_totals: Dict[int, int],
                 edge_flow: Dict[Tuple[int, int], int], remaining: Dict[int, int]) -> None:
        self.world, self.neighs = world, neighs
        self.stakes, self.revolts = allocations_to, revolution_to
        self.staked, self.flow = incoming_totals, edge_flow
        self.lottery = world.cfg.contest == "lottery"
        # What each agent has left, and what everyone who may stake on each node has left (the node and its
        # neighbours), as arrays: a hub's token touches every neighbour's count in one step.
        self.at = {v: i for i, v in enumerate(neighs)}
        self.near = {v: np.array([self.at[w] for w in neighs[v]], dtype=np.int64) for v in neighs}
        self.left = np.array([remaining[v] for v in neighs], dtype=np.int64)
        self.around = self.left.copy()
        for v, i in self.at.items():
            self.around[i] += self.left[self.near[v]].sum()
        self._summary: Dict[int, Tuple[Any, ...]] = {}

    def new_round(self) -> None:
        self._summary = {}

    def place(self, u: int, v: int, revolutionary: bool) -> None:
        """One of u's tokens onto v."""
        self.staked[v] += 1
        self.stakes[v][u] = self.stakes[v].get(u, 0) + 1
        if revolutionary:
            self.revolts[v][u] = self.revolts[v].get(u, 0) + 1
        if u != v:
            edge = (u, v) if u < v else (v, u)
            if edge in self.flow:
                self.flow[edge] += 1
                self.world._mark_flow(u, v)
        i = self.at[u]
        self.left[i] -= 1
        self.around[i] -= 1
        self.around[self.near[u]] -= 1

    def _of(self, v: int) -> Tuple[Any, ...]:
        """What every observer reads alike of node v this round, worked out once."""
        got = self._summary.get(v)
        if got is None:
            stakes = self.stakes[v]
            lead = max(stakes.values()) if stakes else 0
            leaders = {s for s, a in stakes.items() if a == lead} if lead else set()
            below = max((a for a in stakes.values() if a < lead), default=0)
            revolts = self.revolts[v]
            top_revolt = max(revolts.values()) if revolts else 0
            lead_revolts = float(bool(top_revolt) and any(revolts.get(s, 0) == top_revolt for s in leaders))
            # The two who have most left among those who may stake on v, to find the most a rival has.
            near = self.near[v]
            values = np.concatenate([[self.left[self.at[v]]], self.left[near]])
            who = np.concatenate([[self.at[v]], near])
            best = np.argsort(-values, kind="stable")[:2]
            left = [(int(values[b]), int(who[b])) for b in best]
            taker = None if self.lottery else would_take(stakes, revolts)
            got = (lead, leaders, below, float(sum(revolts.values())), lead_revolts, left, taker)
            self._summary[v] = got
        return got

    def read(self, u: int, targets: List[int]) -> np.ndarray:
        """What u reads of each of its candidates, a column each, a row per name of BOARD, as magnitudes."""
        out = np.zeros((len(BOARD), len(targets)))
        me = self.at[u]
        mine_left = float(self.left[me])
        for j, v in enumerate(targets):
            lead, leaders, below, revolt_staked, lead_revolts, left, taker = self._of(v)
            stakes = self.stakes[v]
            mine = float(stakes.get(u, 0))
            staked = float(self.staked[v])
            leading = u in leaders
            others_most = float(below if (leading and len(leaders) == 1) else lead)
            rival_most = float(next((r for r, w in left if w != me), 0))
            around = float(self.around[self.at[v]])
            if u == v:
                crossed = 1.0
            else:
                crossed = float(self.flow.get((u, v) if u < v else (v, u), 0) > 0)
            if self.lottery:
                would = mine / staked if staked else 0.0
            elif taker is None:
                would = 0.0
            else:
                would = float(taker == u)
            out[:, j] = (
                staked, lead, crossed, mine, float(stakes.get(v, 0)),
                (1.0 / len(leaders)) if leading else 0.0,
                mine_left, float(self.left[self.at[v]]), around, (mine_left / around) if around else 0.0,
                float(len(self.neighs[v])), around - mine_left, rival_most,
                staked - mine, others_most, float(lead - below), (lead - below) / (1.0 + lead),
                float(len(leaders)), float(leading), max(0.0, others_most - mine + 1.0),
                revolt_staked, float(self.revolts[v].get(u, 0)), lead_revolts, would,
            )
        return np.log1p(out)


def stake_in_rounds(world, features, outbox: Dict[int, Dict[int, List[float]]],
                    allocations_to: Dict[int, Dict[int, int]], revolution_to: Dict[int, Dict[int, int]],
                    incoming_totals: Dict[int, int], edge_flow: Dict[Tuple[int, int], int],
                    record_decisions: bool) -> List[Dict[str, Any]]:
    """
    The game's stakes under a step allocation, filling the game's tallies in place as the one-shot
    allocation does, and each agent's record of what it placed where.
    """
    from GraphOfLifeSimple import _share_of_first
    cfg, heads = world.cfg, world.heads
    log_deg, neighs, q_tok, q_deg, log_tok, at_risk = features
    start, stop, _ = cfg.input_layout()["board"]
    targets_of = {u: [u] + list(neighs[u]) for u in world.G.nodes()}

    # A first look each, with the board still empty: what every agent writes this phase.
    for u in sorted(world.G.nodes()):
        Y = world._observe(u, targets_of[u], log_deg, q_tok, q_deg, log_tok, at_risk)
        world._emit_messages(u, targets_of[u], Y, outbox)

    remaining = {}
    for u in sorted(world.G.nodes()):
        tokens = int(world.tokens.get(u, 0))
        kept = max(1, int(np.ceil(cfg.keep_home * tokens))) if (cfg.keep_home > 0 and tokens > 0) else 0
        if kept:
            incoming_totals[u] += kept
            allocations_to[u][u] = allocations_to[u].get(u, 0) + kept
        remaining[u] = max(0, tokens - kept)
    board = Board(world, neighs, allocations_to, revolution_to, incoming_totals, edge_flow, remaining)
    placed = {u: np.zeros(len(targets_of[u]), dtype=int) for u in remaining}
    revolted = {u: np.zeros(len(targets_of[u]), dtype=int) for u in remaining}
    for u in remaining:
        placed[u][0] = allocations_to[u].get(u, 0)
    revolts_allowed = cfg.allow_revolutions

    for order in schedule(remaining, cfg.allocation):
        board.new_round()
        choices = []
        for u in order:
            targets = targets_of[u]
            if cfg.random_decisions:
                Y = world.rng.standard_normal((world.output_rows, len(targets)))
            else:
                X = world._inputs(u, targets, log_deg, q_tok, q_deg, log_tok, at_risk)
                X[start:stop] = board.read(u, targets)
                Y = world.brains[u].forward(X)
            idx = int(np.argmax(np.asarray(Y[heads["BLOTTO"], :], dtype=float)))
            revolutionary = revolts_allowed and _share_of_first(
                Y[heads["REV_FRACTION"], idx][0], Y[heads["REV_FRACTION"], idx][1]) > 0.5
            choices.append((u, idx, revolutionary))
        for u, idx, revolutionary in choices:
            board.place(u, targets_of[u][idx], revolutionary)
            placed[u][idx] += 1
            revolted[u][idx] += int(revolutionary)

    records: List[Dict[str, Any]] = []
    if record_decisions:
        for u in sorted(remaining):
            tokens = int(world.tokens.get(u, 0))
            if tokens <= 0:
                continue
            record = {"agent": int(u), "tokens": tokens,
                      # Whether its tokens went to more than one node: no head chose it here.
                      "spread": bool((placed[u] > 0).sum() > 1),
                      "targets": [int(v) for v in targets_of[u]], "alloc": [int(a) for a in placed[u]]}
            if revolts_allowed:
                record["revolt"] = [int(a) for a in revolted[u]]
            records.append(record)
    return records
