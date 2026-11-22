#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What a run's readout says so far: how its brains fare against their own past.

A plan's readout (gol_plan) has each of its runs, at every check, play the
world kept there against worlds kept before it, a few coins each, every coin
twice with the same deal: once with the later era in contest place B, once in
place A. This reads the contests played so far.

The score of a contest is the later era's share of the agents at its end. A
coin's score is the mean of its two contests, and a pairing's (one check
against one earlier era) is the mean of its coins: above 0.5, the later
brains spread better. A coin counts once both its contests are played: one
contest alone has nothing to cancel its place against, and a contest whose
world died out has no score, so its coin never counts. Coins are independent,
so the noise of a pairing's score is the spread of one coin over the root of
how many there are. A pairing has only a few coins, so one coin's spread is
read from all the run's pairings of its kind at once, each coin against its
own pairing's mean: the pairings one lag back together, and those against one
fixed era together. Pooled over every pairing of a run, as it was until 22
November 2026, the pairings against an early era, which the later brains win
outright with next to no spread, made the band of the pairings near 0.5
narrower than their own coins say. While no pairing of a kind has two coins,
the largest spread a number between 0 and 1 can have, 0.5, stands in. The
band is the score plus and minus two of its standard errors, kept between 0
and 1.

In round 3's worlds most contests end with one side holding every agent
(Experiment 45), so one contest says little more than who won: four coins
cannot place a score of 0.7 above 0.5. A readout is for reading pooled, over
checks or over runs; one pairing on its own rarely decides anything.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Iterable, List, Tuple

import gol_plan
import gol_store as store


def contests(name: str) -> List[Dict[str, Any]]:
    """Every readout contest of a plan played so far, with the later era's score; none where the world died out."""
    found = []
    for spec in gol_plan.readout_trials(name):
        try:
            with open(os.path.join(store.run_dir(spec.run_id), "result.json")) as f:
                share = json.load(f)["result"]["share"]
        except (OSError, ValueError, KeyError):
            continue
        if share is None:
            continue
        trial = spec.trial
        era, other = max(trial["a"], trial["b"]), min(trial["a"], trial["b"])
        found.append({"run": trial["run"], "era": era, "against": other, "coin": trial["seed"],
                      "home": trial["home"], "place": "b" if trial["b"] == era else "a",
                      "score": share if trial["b"] == era else 1.0 - share})
    return found


def spread(pairings: Iterable[List[float]]) -> float:
    """One coin's standard deviation, pooled over pairings, each coin against its own pairing's mean."""
    squares, freedom = 0.0, 0
    for scores in pairings:
        if len(scores) > 1:
            mean = sum(scores) / len(scores)
            squares += sum((s - mean) ** 2 for s in scores)
            freedom += len(scores) - 1
    return math.sqrt(squares / freedom) if freedom else 0.5


def kind(readout: Dict[str, Any], era: int, other: int) -> Tuple[str, int]:
    """Which pairings share a spread: those one lag back, or those against one fixed era."""
    return ("lag", era - other) if era - other in readout["lags"] else ("era", other)


def summary(name: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Each run's readout so far, by check and earlier era: its whole coins and
    the contests played, the score, and its band. A pairing is listed once it
    has a whole coin.
    """
    readout = gol_plan.readout_of(name)
    coins: Dict[str, Dict[Tuple[int, int], Dict[int, List[float]]]] = {}
    for c in contests(name):
        coins.setdefault(c["run"], {}).setdefault((c["era"], c["against"]), {}) \
            .setdefault(c["coin"], []).append(c["score"])
    out: Dict[str, List[Dict[str, Any]]] = {}
    for run, pairings in coins.items():
        means = {pair: [sum(s) / 2 for s in per_coin.values() if len(s) == 2] for pair, per_coin in pairings.items()}
        one: Dict[Tuple[str, int], float] = {}
        for k in {kind(readout, *pair) for pair in means}:
            one[k] = spread(scores for pair, scores in means.items() if kind(readout, *pair) == k)
        rows = []
        for (era, other), scores in sorted(means.items()):
            if not scores:
                continue
            score = sum(scores) / len(scores)
            half = 2 * one[kind(readout, era, other)] / math.sqrt(len(scores))
            rows.append({"era": era, "against": other, "coins": len(scores),
                         "contests": sum(len(s) for s in pairings[(era, other)].values()),
                         "score": round(score, 4), "low": round(max(0.0, score - half), 4),
                         "high": round(min(1.0, score + half), 4)})
        out[run] = rows
    return out
