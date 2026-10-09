# -*- coding: utf-8 -*-
"""
Part III, the end: Chapter 34 — do agents cooperate? — and Chapter 35 —
questioning the mechanics.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

import book_data as D
import book_figures as F
from book_figures import FRAMES, chapter, describe, dots, line, recipe, survivors
from book_chapters.common import BLUE, CYAN, GREEN, GREY, ORANGE, RED, VIOLET, YELLOW, baseline
from book_chapters.measures import (DEGREE_CLASSES, FR, PA, SAMPLE, SURVIVORS, bars, in_class, sampled,
                                    stacked)

EVERY = 100


@D.measure("kin")
def kinship(run_id: str) -> Dict[str, Any]:
    """
    Who is related to whom, and whether it shows in the game: from the starts and
    ends of the games of every EVERY-th iteration from SETTLED on.
    """
    D.needs_decisions(run_id, "who staked on whom")
    rng = np.random.default_rng(11)
    edges_same = edges_close = edges_all = 0
    pairs_same = pairs_close = pairs_all = 0
    kin_stake, other_stake, kin_more, both = [], [], 0, 0
    kin_revolt, other_revolt = [], []
    taken_same, taken_all, expect_same = 0, 0, 0.0
    returned = []
    by_degree = defaultdict(lambda: [0.0, 0.0, 0])
    last = (F.store.count_frames(run_id) - 1) // 2
    for t in range(F.SETTLED, last + 1, EVERY):
        start = F.store.read_frame(run_id, 2 * t)
        game = F.store.read_frame(run_id, 2 * t + 1)
        g = dict(zip(start["ids"], start["brain_ids"]))
        p = dict(zip(start["ids"], start["parent_brain_ids"]))

        def same(a, b):
            return g[a] == g[b]

        def close(a, b):
            return g[a] == g[b] or g[a] == p[b] or p[a] == g[b] or (p[a] == p[b] and p[a] != -1)
        for a, b in start["edges"]:
            edges_all += 1
            edges_same += same(a, b)
            edges_close += close(a, b)
        ids = start["ids"]
        for _ in range(2000):
            a, b = rng.choice(len(ids), 2, replace=False)
            pairs_all += 1
            pairs_same += same(ids[a], ids[b])
            pairs_close += close(ids[a], ids[b])
        degree = defaultdict(int)
        for a, b in start["edges"]:
            degree[a] += 1
            degree[b] += 1
        stakes: Dict[Tuple[int, int], int] = {}
        stakers = defaultdict(list)
        for r in (game.get("decisions") or {}).get("allocations") or []:
            u, tokens = r["agent"], r["tokens"]
            kin, other, kin_r, other_r = [], [], [], []
            revolt = r.get("revolt") or [0] * len(r["targets"])
            for v, amount, rev in zip(r["targets"][1:], r["alloc"][1:], revolt[1:]):
                if amount > 0:
                    stakes[(u, v)] = amount
                    stakers[v].append(u)
                (kin if same(u, v) else other).append(amount / tokens)
                if amount > 0:
                    (kin_r if same(u, v) else other_r).append(rev / amount)
            if kin and other:
                both += 1
                kin_stake.append(float(np.mean(kin)))
                other_stake.append(float(np.mean(other)))
                kin_more += np.mean(kin) > np.mean(other)
                d = degree[u]
                cls = next(i for i, (lo, hi, _) in enumerate(DEGREE_CLASSES) if lo <= d <= hi)
                by_degree[cls][0] += np.mean(kin)
                by_degree[cls][1] += np.mean(other)
                by_degree[cls][2] += 1
            if kin_r and other_r:
                kin_revolt.append(float(np.mean(kin_r)))
                other_revolt.append(float(np.mean(other_r)))
        for (u, v), x in stakes.items():
            back = stakes.get((v, u))
            if back:
                returned.append((x, back))
        for w in (game.get("decisions") or {}).get("winners") or []:
            node, winner = w["node"], w["winner"]
            if winner == node or node not in g or winner not in g:
                continue
            others = [s for s in stakers[node] if s != node and s in g]
            if not others:
                continue
            taken_all += 1
            taken_same += same(node, winner)
            expect_same += np.mean([same(node, s) for s in others])
    returned_arr = np.array(returned, float) if returned else np.zeros((0, 2))
    return {"edges": [edges_all, edges_same, edges_close], "pairs": [pairs_all, pairs_same, pairs_close],
            "agents_with_both": both, "kin_more": int(kin_more),
            "kin_stake": float(np.mean(kin_stake)) if kin_stake else None,
            "other_stake": float(np.mean(other_stake)) if other_stake else None,
            "kin_revolt": float(np.mean(kin_revolt)) if kin_revolt else None,
            "other_revolt": float(np.mean(other_revolt)) if other_revolt else None,
            "by_degree": {str(k): v for k, v in by_degree.items()},
            "taken": [taken_all, taken_same, expect_same],
            "returned_equal": float(np.mean(returned_arr[:, 0] == returned_arr[:, 1])) if len(returned_arr) else None,
            "returned_corr": float(np.corrcoef(np.log(returned_arr[:, 0]), np.log(returned_arr[:, 1]))[0, 1])
            if len(returned_arr) > 2 else None,
            "returned_sample": returned_arr[rng.choice(len(returned_arr), min(400, len(returned_arr)),
                                                       replace=False)].tolist() if len(returned_arr) else []}


# ---------------------------------------------------------------------------
# Chapter 34 · Do agents cooperate?
# ---------------------------------------------------------------------------

LAGS = (1, 2, 4, 8, 16)


@D.measure("partners")
def partners(run_id: str) -> Dict[str, Any]:
    """
    How long two brains stay face to face across one connection.

    For the connections at the start of the games of every 200th iteration from
    SETTLED on: the share still there k games later; the share still there with
    neither end taken over by a neighbour in those k games (the same two brains,
    give or take the changes every brain is offered after a game); and the share
    still there with exactly the same two genotypes.
    """
    D.needs_decisions(run_id, "which nodes were taken")
    last = (F.store.count_frames(run_id) - 1) // 2
    there = {k: [0, 0, 0] for k in LAGS}
    total = 0
    for t in range(F.SETTLED, last - max(LAGS) + 1, 2 * EVERY):
        start = F.store.read_frame(run_id, 2 * t)
        g = dict(zip(start["ids"], start["brain_ids"]))
        pairs = {(min(a, b), max(a, b)) for a, b in start["edges"] if a != b}
        total += len(pairs)
        taken: set = set()
        for j in range(max(LAGS)):
            game = F.store.read_frame(run_id, 2 * (t + j) + 1)
            taken |= {w["node"] for w in game["decisions"]["winners"] if w["winner"] != w["node"]}
            k = j + 1
            if k not in there:
                continue
            later = F.store.read_frame(run_id, 2 * (t + k))
            g2 = dict(zip(later["ids"], later["brain_ids"]))
            now = {(min(a, b), max(a, b)) for a, b in later["edges"]}
            for a, b in pairs & now:
                there[k][0] += 1
                there[k][1] += a not in taken and b not in taken
                there[k][2] += g2.get(a) == g[a] and g2.get(b) == g[b]
    return {"lags": list(LAGS), "connections": total,
            "share": {name: [there[k][i] / total for k in LAGS]
                      for i, name in enumerate(("connection", "brains", "genotypes"))}}


@chapter
def cooperation(ch: F.Chapter) -> None:
    alive = survivors(baseline())
    kin = [kinship(s.run_id) for s in alive]

    # Are neighbours related?
    neighbour_same = [k["edges"][1] / k["edges"][0] for k in kin]
    random_same = [k["pairs"][1] / k["pairs"][0] for k in kin]
    neighbour_close = [k["edges"][2] / k["edges"][0] for k in kin]
    random_close = [k["pairs"][2] / k["pairs"][0] for k in kin]
    cats = ["neighbours: same genotype", "any two agents: same genotype", "neighbours: close kin",
            "any two agents: close kin"]
    ch.figure(
        "related", title="Are neighbours related?",
        x={"label": "", "categories": cats}, y={"label": "share of pairs (mean over the looks)", "min": 0},
        series=dots(cats, [neighbour_same, random_same, neighbour_close, random_close], [GREEN, GREY, CYAN, GREY]),
        legend=False,
        caption="One dot per world that lived to the end. For the pairs of agents joined by a connection "
                "(\"neighbours\"), and for pairs of agents drawn at random from the whole world (2,000 per look), "
                "the share that carry the same genotype, and the share that are close kin: the same genotype, "
                "or one's genotype the parent of the other's, or both with the same parent genotype. At the start "
                "of the games of every 100th iteration from 500 on.",
        recipe=recipe(SURVIVORS, FRAMES,
                      ["For every 100th iteration t from 500 on, read frame `2·t`: each agent's genotype "
                       "(`brain_ids`) and its parent genotype (`parent_brain_ids`).",
                       "For every connection of `edges`, test the two relations; also for 2,000 pairs of agents "
                       "drawn at random (generator seeded with 11).",
                       "Divide the counts by the pairs, per run."]))
    ch.number("related", {"neighbour_same": describe(neighbour_same), "random_same": describe(random_same),
                          "neighbour_close": describe(neighbour_close), "random_close": describe(random_close)})

    # Do agents treat kin differently?
    kin_s = [k["kin_stake"] for k in kin if k["kin_stake"] is not None]
    oth_s = [k["other_stake"] for k in kin if k["other_stake"] is not None]
    kin_r = [k["kin_revolt"] for k in kin if k["kin_revolt"] is not None]
    oth_r = [k["other_revolt"] for k in kin if k["other_revolt"] is not None]
    cats2 = ["on a neighbour of its own genotype", "on any other neighbour"]
    ch.grid(
        "kin-stakes", [
            dict(title="Share of its tokens staked on one neighbour", x={"label": "", "categories": cats2},
                 y={"label": "mean share per neighbour", "min": 0}, series=dots(cats2, [kin_s, oth_s], [GREEN, GREY]),
                 legend=False),
            dict(title="Share of a stake marked revolutionary", x={"label": "", "categories": cats2},
                 y={"label": "mean share", "min": 0, "max": 1}, series=dots(cats2, [kin_r, oth_r], [GREEN, GREY]),
                 legend=False)],
        columns=2, title="Does an agent treat its own kind differently?",
        caption="Only agents that had at least one neighbour of their own genotype and at least one of another, "
                "in the games of every 100th iteration from 500 on; one dot per world that lived to the end. "
                "Left: the share of its tokens an agent staked on each neighbour, averaged over the neighbours "
                "of its own genotype (green) and over the others (grey), then over the agents. Right: of what it "
                "staked on a neighbour, the share it marked revolutionary.",
        recipe=recipe(SURVIVORS, FRAMES,
                      ["For every 100th iteration t from 500 on, take the genotypes from frame `2·t` and the "
                       "stakes from `decisions.allocations` of frame `2·t + 1`.",
                       "For each agent with neighbours of both kinds: the mean of `alloc` / `tokens` over each "
                       "kind, and the mean of `revolt` / `alloc` over the neighbours it staked on.",
                       "Average over the agents of a run; one dot per run."]))
    both = sum(k["agents_with_both"] for k in kin)
    more = sum(k["kin_more"] for k in kin)
    ch.number("kin_stakes", {"kin": describe(kin_s), "other": describe(oth_s), "agents": both,
                             "share_kin_more": more / both if both else None,
                             "kin_revolt": describe(kin_r), "other_revolt": describe(oth_r)})

    # Who takes a node: its own kind, or another?
    taken = np.array([k["taken"] for k in kin], float)
    observed = taken[:, 1] / taken[:, 0]
    expected = taken[:, 2] / taken[:, 0]
    cats3 = ["observed", "if the winner were any of the node's stakers"]
    ch.figure(
        "succession", title="When a neighbour takes a node, is it the same kind?",
        x={"label": "", "categories": cats3},
        y={"label": "share of the nodes taken over whose winner has the node's genotype", "min": 0},
        series=dots(cats3, [observed.tolist(), expected.tolist()], [GREEN, GREY]), legend=False,
        caption="Nodes won by a neighbour in the games of every 100th iteration from 500 on; one dot per world "
                "that lived to the end. Left: the share in which the winner carried the same genotype as the "
                "agent whose node it took — a takeover that changes nothing in the brain. Right: the same share "
                "if the winner had been drawn at random from the neighbours that staked on the node.",
        recipe=recipe(SURVIVORS, FRAMES,
                      ["For every 100th iteration t from 500 on, read the genotypes in frame `2·t` and "
                       "`decisions.winners` and `decisions.allocations` in frame `2·t + 1`.",
                       "For each node whose `winner` is not the node: is the winner's genotype the node's? "
                       "And: of the agents other than the node that staked on it, the share with the node's genotype.",
                       "Per run: the first count, and the sum of the second, over the nodes taken."]))
    ch.number("succession", {"observed": describe(observed.tolist()), "expected": describe(expected.tolist()),
                             "taken": float(taken[:, 0].sum())})

    # How exactly stakes are returned.
    sample = np.array([pair for k in kin for pair in k["returned_sample"]], float)
    cells = {"kind": "cells", "x0": [], "x1": [], "y0": [], "y1": [], "value": []}
    edges_ = [1, 2, 3, 4, 6, 9, 15, 30, 60, 120, 250, 500, 1000]
    for a, b in zip(edges_[:-1], edges_[1:]):
        for c, d in zip(edges_[:-1], edges_[1:]):
            m = (sample[:, 0] >= a) & (sample[:, 0] < b) & (sample[:, 1] >= c) & (sample[:, 1] < d)
            if m.any():
                cells["x0"].append(a)
                cells["x1"].append(b)
                cells["y0"].append(c)
                cells["y1"].append(d)
                cells["value"].append(int(m.sum()))
    equal = [k["returned_equal"] for k in kin]
    corr = [k["returned_corr"] for k in kin]
    ch.figure(
        "returned", title="What comes back for what is given",
        x={"label": "tokens A staked on B (logarithmic)", "log": True, "min": 1, "max": 1000},
        y={"label": "tokens B staked on A (logarithmic)", "log": True, "min": 1, "max": 1000},
        series=[cells], colourbar={"map": "viridis", "min": 1, "max": max(cells["value"]), "label": "pairs",
                                   "log": True},
        caption="Pairs of neighbours that both staked on each other in a game: how many tokens each staked on the "
                f"other ({len(sample):,} pairs, up to 400 drawn at random per world that lived to the end, from "
                "the games of every 100th iteration from 500 on). A cell's colour is how many pairs fall in it; "
                f"the diagonal is an even exchange. In the median world {100 * float(np.median(equal)):.0f}% of "
                "the pairs staked exactly the same on each other.",
        recipe=recipe(SURVIVORS, FRAMES,
                      ["From `decisions.allocations` of frame `2·t + 1`, for every 100th iteration t from 500 on, "
                       "collect every pair (A, B) with a stake of A on B and of B on A.",
                       "Draw up to 400 pairs per run (generator seeded with 11) and count them in cells."]))
    ch.number("returned", {"equal": describe(equal), "log_corr": describe(corr), "pairs_drawn": int(len(sample)),
                           "one_for_one": float(np.mean((sample[:, 0] == 1) & (sample[:, 1] == 1)))})

    # The shadow of the future: how long the same two brains face each other.
    pp = [partners(s.run_id) for s in alive]
    lags = np.array(LAGS, float)
    total = sum(x["connections"] for x in pp)
    pooled = {name: [sum(x["share"][name][i] * x["connections"] for x in pp) / total for i in range(len(LAGS))]
              for name in ("connection", "brains", "genotypes")}
    ch.figure(
        "partners", title="How long partners last",
        x={"label": "games later (logarithmic)", "log": True, "min": 1, "max": 16},
        y={"label": "share of the connections of a game (logarithmic)", "log": True, "min": 1e-4, "max": 1},
        series=[line(lags, pooled["connection"], "the connection is still there", BLUE, width=2),
                line(lags, pooled["brains"], "… and neither end was taken over", GREEN, width=2),
                line(lags, pooled["genotypes"], "… and both ends carry the same genotypes", YELLOW, width=2)],
        caption="Of the connections at the start of a game, the share still there k games later (blue); still "
                "there with neither of its two nodes won by a neighbour in any of the k games in between, so that "
                "the same two brains face each other, changed at most by the small changes offered after every "
                "game (green); and still there with exactly the same two genotypes (yellow). Pooled over the "
                f"{total:,} connections at the start of the games of every 200th iteration from 500 on, in the 26 "
                "worlds that lived to the end. Both axes are logarithmic.",
        recipe=recipe(SURVIVORS, FRAMES,
                      ["For every 200th iteration t from 500 on, take the connections of frame `2·t` and the "
                       "genotypes of their ends.",
                       "For k = 1, 2, 4, 8, 16: read frame `2·(t + k)` — is the connection still in `edges`, and are "
                       "the ends' genotypes the same? — and the `decisions.winners` of every game frame "
                       "`2·(t + j) + 1`, j < k: was either end won by a neighbour?",
                       "Pool the counts over the runs and divide by the connections."]))
    ch.number("partners", {"lags": list(LAGS), **pooled, "connections": total})


# ---------------------------------------------------------------------------
# Chapter 35 · Questioning the mechanics
# ---------------------------------------------------------------------------

@chapter
def mechanics(ch: F.Chapter) -> None:
    rows = sampled("frames")
    births, takeovers, new_repro, new_game = [], [], [], []
    for _, fr in rows:
        births.append(float(np.mean(fr[:, FR["births"]])))
        pairs = fr[:, 10:22].reshape(len(fr), 6, 2)
        takeovers.append(float(np.mean(pairs[:, :, 0].sum(axis=1) - pairs[:, :, 1].sum(axis=1))))
        new_repro.append(float(np.mean(fr[:, FR["new_repro"]])))
        new_game.append(float(np.mean(fr[:, FR["new_game"]])))
    cats = ["children born, per iteration", "nodes taken by a neighbour, per game",
            "new genotypes in a reproduction phase", "new genotypes in a game"]
    ch.figure(
        "where-brains-go", title="How brains spread, and where new ones come from",
        x={"label": "", "categories": cats}, y={"label": "per iteration (logarithmic)", "log": True, "min": 1},
        series=dots(cats, [births, takeovers, new_repro, new_game], [GREEN, ORANGE, CYAN, VIOLET]), legend=False,
        caption="One dot per world that lived to the end: means over the iterations sampled every 25 from 500 on. "
                "From the left: children born in a reproduction phase; nodes won in a game by a neighbour, each "
                "of which takes on a copy of the winner's brain; genotypes that appear for the first time in the "
                "frame after a reproduction phase (a newborn's brain that changed); and genotypes that appear for "
                "the first time after a game (every brain is offered a change at its end). On a logarithmic axis.",
        recipe=recipe(SURVIVORS, SAMPLE,
                      ["For every 25th iteration t from 500 on: births = the length of `decisions.births` in frame "
                       "`2·t`; takeovers = the entries of `decisions.winners` in frame `2·t + 1` with `winner` ≠ "
                       "`node`.",
                       "New genotypes: the `brain_ids` of frame `2·t` not in frame `2·t − 1`, and those of frame "
                       "`2·t + 1` not in frame `2·t`.", "Average per run."]))
    ch.number("where_brains_go", {"births": describe(births), "takeovers": describe(takeovers),
                                  "new_repro": describe(new_repro), "new_game": describe(new_game),
                                  "share_new_in_game": describe([g / (g + r) for g, r in zip(new_game, new_repro)])})

    founders = np.concatenate([F.sample_pass(s.run_id)["parents"] for s in baseline()])
    founders = founders[(founders[:, PA["t"]] == 0) & (founders[:, PA["invested"]] > 0)]
    first = founders[:, PA["invested"]] / founders[:, PA["tokens"]]
    edges_ = np.linspace(0, 1, 51)
    counts, _ = np.histogram(first, bins=edges_)
    settled_ = np.concatenate([t for _, t in sampled("parents")])
    settled_ = settled_[(settled_[:, PA["t"]] >= F.SETTLED) & (settled_[:, PA["invested"]] > 0)]
    later = settled_[:, PA["invested"]] / settled_[:, PA["tokens"]]
    counts2, _ = np.histogram(later, bins=edges_)
    ch.grid(
        "child-shares", [
            dict(title="The founders' first children", x={"label": "share of its tokens a parent gives",
                                                          "min": 0, "max": 1},
                 y={"label": "share of the births", "min": 0},
                 series=[{"label": None, "kind": "bars", "x0": edges_[:-1].tolist(), "x1": edges_[1:].tolist(),
                          "y": (counts / counts.sum()).tolist(), "colour": GREY}], legend=False),
            dict(title="Births from iteration 500 on", x={"label": "share of its tokens a parent gives",
                                                          "min": 0, "max": 1},
                 y={"label": "share of the births", "min": 0},
                 series=[{"label": None, "kind": "bars", "x0": edges_[:-1].tolist(), "x1": edges_[1:].tolist(),
                          "y": (counts2 / counts2.sum()).tolist(), "colour": BLUE}], legend=False)],
        columns=2, title="Where the share function jumps",
        caption="The share of its tokens a parent gave its child, in 50 classes of width 0.02. Left: the "
                f"{len(first):,} founders that had a child in the first reproduction phase of the 30 worlds, each "
                "holding 100 tokens, so the share is exact to a hundredth. Right: the births in the reproduction "
                "phases of every 100th iteration from 500 on, in the 26 worlds that lived to the end — parents "
                "with few tokens, whose shares can only be simple fractions.",
        recipe=recipe("The 30 baseline runs (left) and the 26 that lived to the end (right).", SAMPLE,
                      ["Read `decisions.births` of frame 0 (left) and of frame `2·t` for every 100th iteration t "
                       "from 500 on (right): `invested` / `tokens_before` for every birth.",
                       "Count in classes of 0.02 and divide by the births."]))
    ch.number("child_shares", {"first_exactly_half": float(np.mean(first == 0.5)),
                               "first_exactly_all": float(np.mean(first == 1.0)),
                               "first_births": int(len(first)),
                               "later_exactly_half": float(np.mean(later == 0.5)),
                               "later_exactly_all": float(np.mean(later == 1.0))})
