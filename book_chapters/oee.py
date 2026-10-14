# -*- coding: utf-8 -*-
"""
Part V: engineering open-ended evolution — Chapters 46 to 50, Experiments 12 to 16.

The yardstick is a cross-time tournament (gol_tournament): the brains of two
moments of one world play each other in that world, with mutation switched off,
and the share of the world each era ends up holding is its score. A world that
keeps evolving keeps beating its own past; one that has stopped only ties with
it.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

import book_data as D
import book_figures as F
from book_figures import chapter, describe, recipe, runs_of
from book_chapters.common import BLUE, GREY
import gol_store as store
import gol_tournament as tour

#: The moments a world's brains are played from: the founders, then ever
#: further apart. Every one of them is in the archive of a run made with
#: archive_every = 50; a run without one has none of them.
ERAS = (0, 50, 100, 200, 300, 500, 700, 1000)

#: Iterations a contest lasts. Long enough for most of a contested world to
#: change hands at least once (Chapter 16), short enough to play many.
T = 20


def _jobs() -> int:
    """Contests are played in parallel, on this many processes."""
    return int(os.environ.get("GOL_BOOK_JOBS", max(1, (os.cpu_count() or 2) // 2)))


def _lived(run_id: str) -> int:
    """How far a run has gone: 0 for one that does not exist yet."""
    return store.load_meta(run_id).get("iteration", 0) if os.path.exists(store.meta_path(run_id)) else 0


def _eras(run_id: str) -> List[int]:
    have = set(store.archived(run_id))
    return [e for e in ERAS if e in have]


def _want(run_id: str) -> Dict[str, Any]:
    """A kept tournament holds as long as the same eras were played the same way."""
    return {"eras": _eras(run_id), "T": T, "coins": 1}


@D.measure("tournament", field="key", want=_want)
def tournament(run_id: str) -> Dict[str, Any]:
    """Every pair of eras, at home and away, and one rating per era."""
    eras = _eras(run_id)
    if len(eras) < 2:
        return {"eras": eras, "elo": [0.0] * len(eras), "residual": None, "contests": []}
    played = tour.play(run_id, eras, T=T, coins=1, against="all", jobs=_jobs())
    return {"eras": played["eras"], "elo": played["elo"], "residual": played["residual"],
            "contests": played["contests"]}


def score(played: Dict[str, Any], earlier: int, later: int) -> float:
    """The later era's share against the earlier one, averaged over both worlds it was played in."""
    shares = [c["share"] for c in played["contests"]
              if c["a"] == earlier and c["b"] == later and c["share"] is not None]
    return float(np.mean(shares)) if shares else float("nan")


E12_RUNS = ("The 10 runs of Experiment 12 — B1 at 5,000 tokens, seeds 1 to 10, 1,000 "
            "iterations, a whole world archived every 50 — `B1-5000-s001` … `-s010`. They are made "
            "with `python3 gol_lab.py run E12`; every setting is listed in the box under the "
            "thesis.")
ARCHIVE = ("each run's archive, `GraphOfLifeRuns/<run>/archive/world_<iteration>.npz`: the whole "
           "world as a checkpoint writes it, kept every 50 iterations")
HOW = (f"For every pair of the eras {', '.join(str(e) for e in ERAS)} play two contests, one in "
       f"each era's world: give every agent a brain drawn at random from one era or the other, a "
       f"coin each, clear the messages, switch mutation off and run {T} iterations "
       f"(`gol_tournament.contest`). The later era's score is the share of agents carrying one "
       f"of its brains at the end.")


@chapter
def yardstick(ch: F.Chapter) -> None:
    specs = [s for s in runs_of("E12", "baseline")
             if _lived(s.run_id) >= ERAS[-1]]
    if not specs:
        return
    played = {s.lab["seed"]: tournament(s.run_id) for s in specs}

    # Ratings against the founders, one world at a time.
    curves = []
    table = []
    for seed, p in sorted(played.items()):
        curves.append({"x": p["eras"], "y": p["elo"], "colour": GREY, "alpha": 0.45, "width": 1})
        table.append(dict(zip(p["eras"], p["elo"])))
    median = [float(np.median([row[e] for row in table if e in row])) for e in ERAS]
    curves.append({"x": list(ERAS), "y": median, "colour": BLUE, "width": 3,
                   "label": "the median world"})
    ch.number("elo_median", dict(zip(ERAS, median)))
    ch.figure(
        "elo",
        title="How much better than the founders",
        x={"label": "iteration the brains are taken from"},
        y={"label": "Elo rating against the founders"},
        series=curves,
        guides=[{"axis": "y", "at": 0, "label": "the founders"}],
        caption=("Each grey line is one world; the blue "
                 "line is the median of the ten at each era. A rating says how often an era's "
                 "brains beat the founders': +200 is about three contests in four, +400 ten in "
                 "eleven. The ratings are fitted to every contest between every pair of eras, "
                 "so each point leans on the others."),
        recipe=recipe(E12_RUNS, ARCHIVE, [
            HOW,
            "Read every score *p* as odds, log(*p* / (1 − *p*)), after keeping it between 0.02 and "
            "0.98, and find one rating per era — the founders' at 0 — that fits them best by least "
            "squares, the later era's rating minus the earlier's for each contest "
            "(`gol_tournament.ratings`). Multiply by 400 / ln 10 for Elo points.",
            "Draw each world's ratings, and the median world at each era.",
        ]))

    # Every pair of eras, the median world's score.
    cells = {"kind": "cells", "x0": [], "x1": [], "y0": [], "y1": [], "value": []}
    for i, earlier in enumerate(ERAS):
        for later in ERAS[i + 1:]:
            values = [score(p, earlier, later) for p in played.values()]
            values = [v for v in values if np.isfinite(v)]
            if not values:
                continue
            cells["x0"].append(i - 0.5)
            cells["x1"].append(i + 0.5)
            j = ERAS.index(later)
            cells["y0"].append(j - 0.5)
            cells["y1"].append(j + 0.5)
            cells["value"].append(float(np.median(values)))
    ch.figure(
        "pairs",
        title="Every era against every earlier one",
        x={"label": "the earlier era", "categories": [str(e) for e in ERAS]},
        y={"label": "the later era", "categories": [str(e) for e in ERAS]},
        series=[cells],
        colourbar={"label": "the later era's share", "min": 0.3, "max": 0.9},
        caption=("Each square is one pair of eras: "
                 "the share of a contested world the later era's brains held, in the median "
                 "world. 0.5 is a tie. The left column is every era against the founders; in a "
                 "world still improving the shares would grow the further a square is from the "
                 "diagonal, in every column."),
        recipe=recipe(E12_RUNS, ARCHIVE, [
            HOW,
            "For each world and each pair, average its two contests.",
            "Take the median over the ten worlds, and colour the pair's square by it.",
        ]))

    late = [score(p, 300, 1000) for p in played.values()]
    early = [score(p, 0, 300) for p in played.values()]
    ch.number("late_vs_300", describe(late))
    ch.number("300_vs_founders", describe(early))
    ch.number("late_vs_300_worlds", {str(seed): score(p, 300, 1000) for seed, p in sorted(played.items())})
    ch.number("300_vs_founders_worlds", {str(seed): score(p, 0, 300) for seed, p in sorted(played.items())})
    # Each era against the one before it, and the champions of a dominance tournament.
    steps = {f"{a}-{b}": describe([score(p, a, b) for p in played.values()]) for a, b in zip(ERAS, ERAS[1:])}
    ch.number("steps", steps)
    ch.number("champions", {str(seed): tour.dominance(p["eras"], p["contests"])
                            for seed, p in sorted(played.items())})
    ch.number("elo_worlds", {str(seed): dict(zip(p["eras"], p["elo"])) for seed, p in sorted(played.items())})


# ---------------------------------------------------------------------------
# What every chapter of the part reads off a world
# ---------------------------------------------------------------------------

from book_figures import band_series, line, series  # noqa: E402
from book_chapters.common import GREEN, ORANGE, RED, VIOLET, YELLOW  # noqa: E402

COLOURS = (BLUE, RED, GREEN, VIOLET, ORANGE, YELLOW)

#: The health bar of Chapter 46: statistic of the recorded rows, its words,
#: and the range it has to lie in.
BAR = (("coreShare", "agents in the 2-core", 0.8, None),
       ("cutRisk", "largest share one connection cuts off", None, 0.01),
       ("maxDegree/nodes", "best-connected agent's links per agent", None, 0.05),
       ("meanDegree", "connections per agent", 3.0, 12.0),
       ("dimension", "ball dimension", 2.3, 4.0))


def settled_median(run_id: str, stat: str, start: int, stop: int, phase: int = 2) -> float:
    its, values = series(run_id, stat, phase)
    pick = (its >= start) & (its <= stop) & np.isfinite(values)
    return float(np.median(values[pick])) if pick.any() else float("nan")


def lowest(run_id: str, start: int = 50) -> float:
    """The fewest agents after the game of any iteration from `start` on (0 for a world that died out)."""
    its, n = series(run_id, "nodes", 2)
    pick = its >= start
    low = float(np.min(n[pick])) if pick.any() else float("nan")
    return 0.0 if store.load_meta(run_id).get("status") == "extinct" else low


def dots(name: str, conditions: List[str], values: Dict[str, List[float]], *, title: str, y: Dict[str, Any],
         guides: List[Dict[str, Any]] = ()) -> Dict[str, Any]:
    """One column of dots per condition, with its median as a bar."""
    out = []
    for k, cond in enumerate(conditions):
        v = np.array([x for x in values.get(cond, []) if x is not None and np.isfinite(x)], dtype=float)
        if not v.size:
            continue
        out.append({"label": None, "x": F.jitter(k, v.size, 11 + k).tolist(), "y": v.tolist(),
                    "points": True, "size": 6, "colour": COLOURS[k % len(COLOURS)], "alpha": 0.85})
        out.append(line([k - 0.38, k + 0.38], [float(np.median(v))] * 2, None, COLOURS[k % len(COLOURS)], width=3))
    return {"title": title, "x": {"categories": conditions}, "y": y, "series": out,
            "guides": list(guides), "legend": False}


# ---------------------------------------------------------------------------
# Chapter 47 · Brains that see, and worlds that break (Experiment 13)
# ---------------------------------------------------------------------------

E13_CONDITIONS = ["baseline", "one hidden layer", "one hidden layer, change only when copied"]
E13_RUNS = ("The 30 runs of Experiment 13 — at 5,000 tokens, seeds 1 to 10, 1,500 iterations — "
            "in three conditions: the baseline B1 (`B1-5000-s001` … `-s010`, shared with "
            "Experiment 12), its brains with one hidden layer of 16, and those brains changed only "
            "when copied. They are made with `python3 gol_lab.py run E13`.")


@chapter
def seeing(ch: F.Chapter) -> None:
    specs = {c: [s for s in runs_of("E13", c) if _lived(s.run_id) > 0]
             for c in E13_CONDITIONS}
    if not all(specs.values()):
        return
    low = {c: [lowest(s.run_id) for s in specs[c]] for c in E13_CONDITIONS}
    broke = {c: int(sum(1 for v in low[c] if v < 50)) for c in E13_CONDITIONS}
    ch.number("lowest", {c: describe(low[c]) for c in E13_CONDITIONS})
    ch.number("broke", broke)

    bands = [band_series([series(s.run_id, "nodes", 2) for s in specs[c]], c, COLOURS[k])
             for k, c in enumerate(E13_CONDITIONS)]
    ch.figure("agents", title="Agents alive", x={"label": "iteration"}, y={"label": "agents", "min": 0},
              series=bands,
              caption=("For each condition the median world at each iteration, "
                       "with the middle half of the worlds (darker) and nine in ten (paler)."),
              recipe=recipe(E13_RUNS, F.STATS_FILE, [
                  "Take `nodes` of the rows with `phase` = 2, one line per run.",
                  "For each condition, the median, quartiles and 5th and 95th percentiles of the runs "
                  "at each iteration (`book_figures.band_series`)."]))
    ch.figure("lowest", **dots("lowest", E13_CONDITIONS, low, title="The fewest agents a world fell to",
                               y={"label": "agents (0: died out)", "min": 0},
                               guides=[{"axis": "y", "at": 50, "label": "50 agents"}]),
              caption=("One dot per world: its fewest agents "
                       "after the game of any iteration from 50 on, 0 for a world that died out; the "
                       "bar is the median world. The dashed line is the thesis's 50 agents."),
              recipe=recipe(E13_RUNS, F.STATS_FILE, [
                  "Take the smallest `nodes` of the rows with `phase` = 2 and `iteration` ≥ 50; a run "
                  "that ended extinct counts 0.",
                  "Count the worlds below 50 in each condition."]))
    panels = []
    for stat, words, lo, hi in BAR[:3]:
        values = {c: [settled_median(s.run_id, stat, 200, 1500) for s in specs[c]] for c in E13_CONDITIONS}
        ch.number(f"health_{stat}", {c: describe(values[c]) for c in E13_CONDITIONS})
        guides = [{"axis": "y", "at": v, "label": "the bar"} for v in (lo, hi) if v is not None]
        y = {"label": words, "min": 0, **({"max": 1} if stat == "coreShare" else {})}
        panels.append(dots(stat, E13_CONDITIONS, values, title=words, y=y, guides=guides))
    ch.grid("health", panels, columns=1, title="How the network holds",
            caption=("One dot per world, at its median over iterations "
                     "200 to 1,500 while it lived; the bar is the median world and the dashed line the "
                     "health bar of Chapter 46."),
            recipe=recipe(E13_RUNS, F.STATS_FILE, [
                "Take `coreShare`, `cutRisk` and `maxDegree` / `nodes` of the rows with `phase` = 2 "
                "and 200 ≤ `iteration` ≤ 1,500, and each world's median of each."]))

    ch.number("worst_cull", {c: [worst_cull(s.run_id) for s in specs[c]] for c in E13_CONDITIONS})

    # How the worlds that died, died: births against the deaths of each kind.
    rates: Dict[str, Dict[str, Any]] = {}
    for c in E13_CONDITIONS:
        for a, b in ((50, 200), (200, 500), (500, 1500)):
            got = [r for r in (death_rates(s.run_id, a, b) for s in specs[c]) if r]
            if got:
                rates.setdefault(c, {})[f"{a}-{b}"] = {k: describe([r[k] for r in got])
                                                       for k in ("births", "cutOff", "starved")}
    ch.number("rates", rates)
    ch.figure("births", title="Births per agent",
              x={"label": "iteration"}, y={"label": "births per agent in a reproduction phase", "log": True},
              series=[band_series([series(s.run_id, "births/nodes_before", 1) for s in specs[c]], c, COLOURS[k])
                      for k, c in enumerate(E13_CONDITIONS)],
              caption=("For each condition the median world at each iteration, with the middle half "
                       "of the worlds (darker) and nine in ten (paler), on a logarithmic axis. A world "
                       "that died out leaves the band when it does."),
              recipe=recipe(E13_RUNS, F.STATS_FILE, [
                  "Take `births` / `nodes_before` of the rows with `phase` = 1, one line per run.",
                  "For each condition, the median, quartiles and 5th and 95th percentiles of the runs "
                  "at each iteration (`book_figures.band_series`)."]))

    # Do the brains of each era still have children? Their own worlds cannot say —
    # a rich agent in a small world lives differently — so every era plays the same
    # world: the founders'.
    probed = {c: [children(s.run_id) for s in specs[c]] for c in E13_CONDITIONS}
    curves = []
    for k, c in enumerate(E13_CONDITIONS):
        for p in probed[c]:
            curves.append({"x": p["eras"], "y": [max(v, 1e-4) for v in p["births"]],
                           "colour": COLOURS[k], "alpha": 0.35, "width": 1})
    for k, c in enumerate(E13_CONDITIONS):
        eras = sorted({e for p in probed[c] for e in p["eras"]})
        median = [float(np.median([dict(zip(p["eras"], p["births"]))[e] for p in probed[c] if e in p["eras"]]))
                  for e in eras]
        curves.append({"x": eras, "y": [max(v, 1e-4) for v in median], "colour": COLOURS[k], "width": 3,
                       "label": c})
    ch.number("children", {c: {str(e): describe([dict(zip(p["eras"], p["births"]))[e]
                                                 for p in probed[c] if e in p["eras"]])
                               for e in (0, 100, 200, 300, 500, 1000, 1500)
                               if any(e in p["eras"] for p in probed[c])}
                           for c in E13_CONDITIONS})
    ch.number("childless", {c: int(sum(1 for p in probed[c] if p["eras"][-1] >= 300 and p["births"][-1] == 0))
                            for c in E13_CONDITIONS})
    ch.figure("children", title="Do the brains still have children?",
              x={"label": "iteration the brains are taken from"},
              y={"label": "births per agent and iteration, in the founders' world", "log": True},
              series=curves,
              caption=(f"Every era's brains play the same world — the founders' world of their own "
                       f"run, 50 agents of 100 tokens on the starting ring — for {CHILD_T} iterations "
                       f"with mutation off. Thin lines: one world each, its births per agent and "
                       f"iteration, the mean of {len(CHILD_COINS)} draws; thick lines: the median "
                       f"world of each condition at each era, over the worlds that still lived. No "
                       f"birth at all is drawn at 0.0001."),
              recipe=recipe(E13_RUNS, ("each run's archive, `GraphOfLifeRuns/<run>/archive/"
                                       "world_<iteration>.npz`, kept every 100 iterations (every 50 for "
                                       "the baseline, of which every 100th is used)"), [
                  "For every era *e* that is a multiple of 100: take the archived world of iteration 0, "
                  "give every agent a brain drawn at random from the agents of era *e* "
                  "(`gol_tournament.mixed` with both eras *e*), clear the messages, switch mutation off "
                  f"and run {CHILD_T} iterations; count the births of the reproduction phases over the "
                  "agents alive at the start of each iteration (`oee.children`). Draws "
                  f"{', '.join(str(c) for c in CHILD_COINS)}.",
                  "Draw each world's rate by era, and each condition's median."]))


def worst_cull(run_id: str) -> Dict[str, Any]:
    """The most agents one cull cut off, in either phase, and when."""
    size, at = 0.0, None
    for phase in (1, 2):
        its, cut = series(run_id, "orphaned", phase)
        if len(cut) and np.nanmax(cut) > size:
            k = int(np.nanargmax(cut))
            size, at = float(cut[k]), int(its[k])
    return {"agents": size, "iteration": at}


def death_rates(run_id: str, start: int, stop: int) -> Optional[Dict[str, float]]:
    """Births, agents cut off and agents starved, each per agent alive and iteration, in [start, stop)."""
    alive = {}
    count = {"births": 0.0, "cutOff": 0.0, "starved": 0.0}
    for r in F.rows(run_id):
        if not start <= r["iteration"] < stop:
            continue
        count["births"] += r.get("births") or 0
        count["cutOff"] += r.get("orphaned") or 0
        count["starved"] += r.get("starved") or 0
        if r.get("phase") == 2:
            alive[r["iteration"]] = r["nodes"]
    total = sum(alive.values())
    return {k: v / total for k, v in count.items()} if total else None


#: The births probe: iterations a probe lasts, and its draws.
CHILD_T = 10
CHILD_COINS = (5, 6)


def _child_eras(run_id: str) -> List[int]:
    return [e for e in store.archived(run_id) if e % 100 == 0]


def _want_children(run_id: str) -> Dict[str, Any]:
    return {"eras": _child_eras(run_id), "T": CHILD_T, "coins": list(CHILD_COINS)}


def _children_in_founders_world(task) -> float:
    """Births per agent and iteration when the brains of one era play the founders' world."""
    run_id, era, coin = task
    from gol_config import SimConfig
    from GraphOfLifeSimple import GraphOfLife
    cfg = SimConfig.from_dict(store.load_meta(run_id)["config"])
    cfg.mutation_probability = 0.0
    cfg.extinction_threshold = 0
    brains = store.load_archive(run_id, era)
    planted = tour.mixed(store.load_archive(run_id, 0), brains, brains, cfg.message_amount,
                         np.random.RandomState(coin))
    world = GraphOfLife.from_checkpoint(planted, cfg)
    world.rng = np.random.RandomState(coin + 1)
    births = alive = 0
    for _ in range(CHILD_T):
        if world.G.number_of_nodes() == 0:
            break
        alive += world.G.number_of_nodes()
        for frame in world.step(record_decisions=True):
            births += len((frame.get("decisions") or {}).get("births") or [])
    return births / max(1, alive)


@D.measure("children", field="key", want=_want_children)
def children(run_id: str) -> Dict[str, Any]:
    """For every hundredth archived era: how often its brains have children in the founders' world."""
    from concurrent.futures import ProcessPoolExecutor
    eras = _child_eras(run_id)
    tasks = [(run_id, e, c) for e in eras for c in CHILD_COINS]
    with ProcessPoolExecutor(_jobs()) as pool:
        rates = list(pool.map(_children_in_founders_world, tasks))
    k = len(CHILD_COINS)
    return {"eras": eras, "births": [float(np.mean(rates[i * k:(i + 1) * k])) for i in range(len(eras))]}


# ---------------------------------------------------------------------------
# Chapter 48 · A world that cannot break (Experiment 14)
# ---------------------------------------------------------------------------

E14_CONDITIONS = ["all the rules", "no mending", "any unused link lapses", "no links at birth",
                  "no valence", "handover"]
E14_RUNS = ("The 60 runs of Experiment 14 — B2 at 5,000 tokens, seeds 1 to 10, 1,000 iterations — "
            "in six conditions: all the rules of B2, and B2 with one of them taken away or "
            "changed. They are made with `python3 gol_lab.py run E14`.")


#: The line of the health bar no run records: the largest share of the others
#: one agent's loss would cut off from the largest piece. Read off the frames.
NODE_CUT = ("nodeCut", "largest share one agent's loss cuts off", None, 0.02)
NODE_CUT_EVERY = 10


def _want_node_cut(run_id: str) -> Dict[str, Any]:
    return {"every": NODE_CUT_EVERY, "upTo": min(_lived(run_id), 1000)}


@D.measure("nodecut", field="key", want=_want_node_cut)
def node_cut(run_id: str) -> Dict[str, Any]:
    """The worst single-agent cut after the game, every NODE_CUT_EVERY iterations."""
    from GraphOfLifeSimple import worst_node_cut_share
    its, shares = [], []
    for t in range(0, min(_lived(run_id), 1000) + 1, NODE_CUT_EVERY):
        try:
            frame = D.frame_at(run_id, t, 2)
        except (KeyError, ValueError, FileNotFoundError):
            continue
        adj = {u: set() for u in frame["ids"]}
        for a, b in frame["edges"]:
            adj[a].add(b)
            adj[b].add(a)
        its.append(t)
        shares.append(float(worst_node_cut_share(frame["ids"], adj)) if adj else float("nan"))
    return {"iterations": its, "share": shares}


def health(run_id: str, start: int = 200, stop: int = 1000) -> Dict[str, Any]:
    """
    A world against the health bar: its medians, whether each passes, how many
    agents were cut off, and whether it held — every median inside the bar,
    nobody cut off, and the world still alive at the end. The worst agent cut
    comes from the frames (node_cut), every other line from the recorded rows.
    """
    medians = {stat: settled_median(run_id, stat, start, stop) for stat, *_ in BAR}
    cut = node_cut(run_id)
    picked = [v for t, v in zip(cut["iterations"], cut["share"]) if start <= t <= stop and np.isfinite(v)]
    medians[NODE_CUT[0]] = float(np.median(picked)) if picked else float("nan")
    passes = {stat: bool(np.isfinite(medians[stat]) and (lo is None or medians[stat] >= lo)
                         and (hi is None or medians[stat] <= hi)) for stat, _, lo, hi in BAR + (NODE_CUT,)}
    cut_off = 0.0
    for phase in (1, 2):
        its, orphaned = series(run_id, "orphaned", phase)
        pick = (its >= start) & (its <= stop) & np.isfinite(orphaned)
        cut_off += float(np.sum(orphaned[pick]))
    extinct = store.load_meta(run_id).get("status") == "extinct"
    return {"medians": medians, "passes": passes, "cutOff": cut_off, "extinct": extinct,
            "holds": all(passes.values()) and cut_off == 0 and not extinct}


def ended(spec) -> bool:
    """A run is over: it reached its last iteration, or its world died out."""
    if not os.path.exists(store.meta_path(spec.run_id)):
        return False
    meta = store.load_meta(spec.run_id)
    return meta.get("status") == "extinct" or meta.get("iteration", 0) >= spec.until


# Space alone: the network that births, deaths and the cull make with no brains
# at all (research/oee/space.py). Each rule is a set of keyword arguments of
# its `grow`; the label is what the figure's column says.
SPACE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "research", "oee", "space.py")
SPACE_RULES = (
    ("parent", {"links": 0, "mend": False}),
    ("+2", {"links": 2, "mend": False}),
    ("chain", {"links": 2, "mend": True}),
    ("ring", {"links": 2, "mend": True, "ring": True}),
    ("ring, 1", {"links": 1, "mend": True, "ring": True}),
    ("ring, 3", {"links": 3, "mend": True, "ring": True}),
    ("join all", {"links": 2, "mend": False, "mend_all": True}),
    ("ring, ≤ 8", {"links": 2, "mend": True, "ring": True, "cap": 8}),
)
SPACE_SEEDS = range(1, 11)
#: What the space figure shows of each world, its words, and the health bar's range.
SPACE_BAR = (("nodeCut", "largest share one agent's loss cuts off", None, 0.02, True),
             ("edgeCut", "largest share one link's loss cuts off", None, 0.01, True),
             ("core2", "agents in the 2-core", 0.8, None, False),
             ("links", "links per agent", 3.0, 12.0, False),
             ("dimension", "ball dimension", 2.3, 4.0, False),
             ("hub", "best-connected agent's links per agent", None, 0.05, True))
SPACE_HOW = ("A world of no brains (`research/oee/space.py`): it starts as a ring of 50 agents, each "
             "joined to its 6 nearest (Watts–Strogatz, rewiring 0.2). Every iteration each agent has "
             "a child with probability 0.02 × 1,500 / *n*, plus 0.05 while there are fewer than "
             "1,500 agents, and then dies with probability 0.02; after both, only the largest piece "
             "is kept. 300 iterations.")
_SPACE = None


def _space_model():
    global _SPACE
    if _SPACE is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("oee_space", SPACE_FILE)
        _SPACE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_SPACE)
    return _SPACE


def _space_world(task):
    k, seed = task
    model = _space_model()
    return k, seed, model.measure(model.grow(seed=seed, **SPACE_RULES[k][1]))


def space_alone(ch: F.Chapter) -> None:
    """Chapter 48's model: every rule of space on its own, ten worlds each, and two pictures."""
    from concurrent.futures import ProcessPoolExecutor
    tasks = [(k, seed) for k in range(len(SPACE_RULES)) for seed in SPACE_SEEDS]
    with ProcessPoolExecutor(_jobs()) as pool:
        done = list(pool.map(_space_world, tasks))
    labels = [label for label, _ in SPACE_RULES]
    got = {label: [] for label in labels}
    for k, seed, out in done:
        got[labels[k]].append(out)
    ch.number("space", {label: {stat: describe([w[stat] for w in got[label]])
                                for stat in ("agents", "links", "core2", "core3", "nodeCut",
                                             "edgeCut", "dimension", "diameter", "hub")}
                        for label in labels})
    panels = []
    for stat, words, lo, hi, log in SPACE_BAR:
        values = {label: [max(w[stat], 1e-4) if log else w[stat] for w in got[label]] for label in labels}
        guides = [{"axis": "y", "at": v, "label": "the bar"} for v in (lo, hi) if v is not None]
        panels.append(dots(stat, labels, values, title=words,
                           y={"label": words, **({"log": True} if log else {"min": 0})}, guides=guides))
    ch.grid("space", panels, columns=2, title="What the rules of space make, with no brains",
            caption=("Each column is one set of "
                     "rules, each dot one world of about 1,500 agents after 300 iterations, the bar "
                     "the median of ten. *parent*: a newborn is joined to its parent alone; *+2*: "
                     "also to two of the parent's neighbours, drawn at random; *chain*: also, when an "
                     "agent dies, the groups of its former neighbours that are not linked to one "
                     "another are joined in a chain; *ring*: joined in a ring — B2's rules; "
                     "*ring, 1* and *ring, 3*: the ring, with one or three of the parent's neighbours "
                     "instead of two; *join all*: every former neighbour of the dead joined in a "
                     "chain, linked or not; *ring, ≤ 8*: B2's rules, and no child for an agent of 8 "
                     "links or more nor a newborn's random link to one (mending still links past the "
                     "cap). The "
                     "dashed lines are the health bar of Chapter 46. Cut shares and the "
                     "best-connected agent's share are drawn on logarithmic axes, a cut of nothing "
                     "at 0.0001."),
            recipe=recipe("No runs: the worlds are made by the model itself, seeds 1 to 10 for every "
                          "set of rules.", "the networks the model returns",
                          [SPACE_HOW,
                           "Measure the last network (`space.measure`): the largest share of the "
                           "others one agent's loss would cut off from the largest piece "
                           "(`GraphOfLifeSimple.worst_node_cut_share`), the same for one link "
                           "(`cutRisk`), the 2-core, links per agent, the ball dimension, and the "
                           "best-connected agent's links over the agents less one."]))

    # Two worlds drawn: what births alone make, and what mending makes of it.
    import networkx as nx
    model = _space_model()
    for name, label, rules in (("blobs", "A tree of blobs", SPACE_RULES[1][1]),
                               ("mended", "Mended in a ring", SPACE_RULES[3][1])):
        # Of five seeds, the world that ended largest: the cull can leave a tree of
        # blobs with few agents, and a picture needs some to show.
        G = max((model.grow(seed=seed, target=600, **rules) for seed in range(1, 6)),
                key=lambda g: (g.number_of_nodes(), -min(g.nodes())))
        nodes = sorted(G.nodes())
        index = {u: i for i, u in enumerate(nodes)}
        pos = nx.forceatlas2_layout(G, max_iter=300, seed=1)
        bridges = set(map(frozenset, nx.bridges(G)))
        cuts = set(nx.articulation_points(G))
        edges = [(index[a], index[b]) for a, b in G.edges()]
        worst = model.measure(G)
        ch.network(
            name, title=label,
            nodes={"x": [float(pos[u][0]) for u in nodes], "y": [float(pos[u][1]) for u in nodes],
                   "group": [1 if u in cuts else 0 for u in nodes]},
            edges=edges, edgeGroup=[1 if frozenset((nodes[a], nodes[b])) in bridges else 0 for a, b in edges],
            colour={"labels": ["an agent", "one whose loss alone splits the world"]},
            edgeLabels=["a link whose loss alone splits it"], nodeSize=2.0, edgeAlpha=0.3,
            caption=(f"A world of the model aiming at 600 agents "
                     f"({len(nodes)} at the end): {'newborns joined to their parent and two of its neighbours, nothing mended' if name == 'blobs' else 'the same, and every death mended in a ring'}. "
                     "Of seeds 1 to 5, the world that ended largest. "
                     "Agents and links whose loss alone would split the world are drawn in the "
                     f"second colour, the links wider; the worst of them cuts off "
                     f"{100 * max(worst['nodeCut'], worst['edgeCut']):.1f}% of the world."),
            recipe=recipe("No runs: the model's worlds, seeds 1 to 5.", "the network the model returns",
                          [SPACE_HOW.replace("1,500", "600"),
                           "Keep the world that ended with the most agents.",
                           "Find the bridges and the cut vertices (`networkx.bridges`, "
                           "`networkx.articulation_points`), lay the network out with "
                           "`networkx.forceatlas2_layout(G, max_iter=300, seed=1)` and draw it."]))
        ch.number(name, {"agents": len(nodes), "bridges": len(bridges), "cutVertices": len(cuts),
                         "nodeCut": worst["nodeCut"], "edgeCut": worst["edgeCut"]})


@chapter
def unbreakable(ch: F.Chapter) -> None:
    space_alone(ch)
    specs = {c: runs_of("E14", c) for c in E14_CONDITIONS}
    if not all(specs.values()) or not all(ended(s) for c in specs for s in specs[c]):
        return
    checked = {c: [health(s.run_id) for s in specs[c]] for c in E14_CONDITIONS}
    ch.number("hold", {c: int(sum(h["holds"] for h in checked[c])) for c in E14_CONDITIONS})
    ch.number("worlds", {c: len(checked[c]) for c in E14_CONDITIONS})
    ch.number("extinct", {c: [s.lab["seed"] for s, h in zip(specs[c], checked[c]) if h["extinct"]]
                          for c in E14_CONDITIONS})
    ch.number("cutOff", {c: {str(s.lab["seed"]): h["cutOff"] for s, h in zip(specs[c], checked[c])}
                         for c in E14_CONDITIONS})
    ch.number("fails", {c: {stat: int(sum(1 for h in checked[c] if not h["passes"][stat]))
                            for stat, *_ in BAR + (NODE_CUT,)} for c in E14_CONDITIONS})
    ch.number("dimension_worlds", {c: {str(s.lab["seed"]): h["medians"]["dimension"]
                                       for s, h in zip(specs[c], checked[c])} for c in E14_CONDITIONS})

    ch.figure("agents", title="Agents alive", x={"label": "iteration"}, y={"label": "agents", "min": 0},
              series=[band_series([series(s.run_id, "nodes", 2) for s in specs[c]], c, COLOURS[k % len(COLOURS)])
                      for k, c in enumerate(E14_CONDITIONS)],
              caption=("For each condition the median world at each iteration, with the middle half "
                       "of the worlds (darker) and nine in ten (paler). A world that died out leaves the "
                       "band when it does."),
              recipe=recipe(E14_RUNS, F.STATS_FILE, [
                  "Take `nodes` of the rows with `phase` = 2, one line per run.",
                  "For each condition, the median, quartiles and 5th and 95th percentiles of the runs "
                  "at each iteration (`book_figures.band_series`)."]))
    ch.figure("lost", **dots("lost", E14_CONDITIONS,
                             {c: [h["cutOff"] + 1 for h in checked[c]] for c in E14_CONDITIONS},
                             title="Agents cut off from iteration 200 to 1,000",
                             y={"label": "agents cut off, plus one", "log": True}),
              caption=("One dot per world: every agent the cull removed because it was no longer "
                       "joined to the largest piece, in either phase, from iteration 200 to 1,000 — "
                       "plus one, so that a world that lost nobody sits at 1 on the logarithmic "
                       "axis. The bar is the median world."),
              recipe=recipe(E14_RUNS, F.STATS_FILE, [
                  "Sum `orphaned` over the rows of both phases with 200 ≤ `iteration` ≤ 1,000.",
                  "Add one, and draw a dot per world."]))
    ch.number("medians", {c: {stat: describe([h["medians"][stat] for h in checked[c]])
                              for stat, *_ in BAR + (NODE_CUT,)} for c in E14_CONDITIONS})
    panels = []
    for stat, words, lo, hi in BAR + (NODE_CUT,):
        values = {c: [h["medians"][stat] for h in checked[c]] for c in E14_CONDITIONS}
        guides = [{"axis": "y", "at": v, "label": "the bar"} for v in (lo, hi) if v is not None]
        panels.append(dots(stat, E14_CONDITIONS, values, title=words, y={"label": words, "min": 0},
                           guides=guides))
    ch.grid("bar", panels, columns=2, title="Each rule against the health bar",
            caption=("One dot per world, at its median over "
                     "iterations 200 to 1,000; the bar is the median world; dashed lines are the "
                     "health bar of Chapter 46. The first column has every rule of B2; each other "
                     "column takes one away."),
            recipe=recipe(E14_RUNS, F.STATS_FILE, [
                "For every run take the rows with `phase` = 2 and 200 ≤ `iteration` ≤ 1,000, and "
                "the median of `coreShare`, `cutRisk`, `maxDegree` / `nodes`, `meanDegree` and "
                "`dimension`.",
                f"No run records the worst cut by one agent, so read it off the frames: every "
                f"{NODE_CUT_EVERY}th iteration from 200 to 1,000, rebuild the network after the game "
                f"and take `GraphOfLifeSimple.worst_node_cut_share` (`oee.node_cut`); its median.",
                "A world holds if every median is inside the bar and `orphaned` is 0 in every row "
                "of either phase in the same iterations."]))


# ---------------------------------------------------------------------------
# Chapter 49 · Who dies, and why (Experiment 15)
# ---------------------------------------------------------------------------

E15_CONDITIONS = ["B2", "the last token kept", "nothing kept", "half kept", "children of any size",
                  "no old age"]
E15_RUNS = ("The 60 runs of Experiment 15 — B2 at 10,000 tokens, seeds 1 to 10, 1,000 iterations — "
            "in six conditions that change how much an agent keeps at home, how small a child may "
            "be, and whether agents die of age. They are made with `python3 gol_lab.py run E15`.")


@D.measure("onetoken")
def one_token(run_id: str) -> Dict[str, Any]:
    """The share of agents holding a single token after the game, every 50 iterations."""
    last = D.last_iteration(run_id)
    its, shares = [], []
    for t in range(0, last + 1, 50):
        try:
            frame = D.frame_at(run_id, t, 2)
        except (KeyError, ValueError, FileNotFoundError):
            continue
        tokens = np.array(frame["tokens"])
        if tokens.size:
            its.append(t)
            shares.append(float(np.mean(tokens == 1)))
    return {"iterations": its, "shares": shares}


@chapter
def deaths(ch: F.Chapter) -> None:
    specs = {c: [s for s in runs_of("E15", c) if _lived(s.run_id) > 0]
             for c in E15_CONDITIONS}
    if not all(specs.values()):
        return
    single = {}
    for c in E15_CONDITIONS:
        single[c] = []
        for s in specs[c]:
            m = one_token(s.run_id)
            v = [x for t, x in zip(m["iterations"], m["shares"]) if t >= 300]
            single[c].append(float(np.median(v)) if v else float("nan"))
    low = {c: [lowest(s.run_id) for s in specs[c]] for c in E15_CONDITIONS}
    ch.number("one_token", {c: describe(single[c]) for c in E15_CONDITIONS})
    ch.number("lowest", {c: describe(low[c]) for c in E15_CONDITIONS})
    ch.number("starved_worlds", {c: int(sum(1 for v in low[c] if v < 100)) for c in E15_CONDITIONS})
    ch.grid("who", [
        dots("single", E15_CONDITIONS, single, title="Agents holding a single token",
             y={"label": "share of agents", "min": 0, "max": 1},
             guides=[{"axis": "y", "at": 0.5, "label": "half"}]),
        dots("lowest", E15_CONDITIONS, low, title="The fewest agents a world fell to",
             y={"label": "agents (0: died out)", "min": 0},
             guides=[{"axis": "y", "at": 100, "label": "100 agents"}]),
    ], columns=1, title="Who is left to play",
        caption=("Top: one dot per world, the median over iterations 300 "
                 "to 1,000 of the share of its agents holding exactly one token after the game. "
                 "Bottom: the fewest agents it fell to after the game of any iteration from 50 on, "
                 "0 if it died out. Bars are the median worlds."),
        recipe=recipe(E15_RUNS, "each run's frames and `stats.jsonl`", [
            "For every 50th iteration from 300, read the frame after the game "
            "(`book_data.frame_at(run, t, 2)`) and the share of its `tokens` equal to 1; take each "
            "world's median.",
            "Take the smallest `nodes` of the rows with `phase` = 2 and `iteration` ≥ 50; 0 for a "
            "run that ended extinct."]))


# ---------------------------------------------------------------------------
# Chapter 50 · Why evolution stops (Experiment 16)
# ---------------------------------------------------------------------------

E16_ERAS = (0, 100, 200, 500, 1000, 1500, 2000)
E16_CONDITIONS = ["B2", "no random inputs"]
E16_RUNS = ("The 20 runs of Experiment 16 — B2 at 10,000 tokens, seeds 1 to 10, 2,000 iterations, "
            "a whole world archived every 100 — with and without the random inputs. They are made "
            "with `python3 gol_lab.py run E16`.")


def _want_long(run_id: str) -> Dict[str, Any]:
    have = set(store.archived(run_id))
    return {"eras": [e for e in E16_ERAS if e in have], "T": T, "coins": 1}


@D.measure("tournament-long", field="key", want=_want_long)
def tournament_long(run_id: str) -> Dict[str, Any]:
    """The tournament of a world that lived 2,000 iterations, at fewer eras."""
    eras = _want_long(run_id)["eras"]
    if len(eras) < 2:
        return {"eras": eras, "elo": [0.0] * len(eras), "residual": None, "contests": []}
    played = tour.play(run_id, eras, T=T, coins=1, against="all", jobs=_jobs())
    return {"eras": played["eras"], "elo": played["elo"], "residual": played["residual"],
            "contests": played["contests"]}


@chapter
def stopping(ch: F.Chapter) -> None:
    specs = {c: [s for s in runs_of("E16", c) if _lived(s.run_id) >= 2000]
             for c in E16_CONDITIONS}
    if not all(specs.values()):
        return
    played = {c: [tournament_long(s.run_id) for s in specs[c]] for c in E16_CONDITIONS}
    series_ = []
    for k, c in enumerate(E16_CONDITIONS):
        table = [dict(zip(p["eras"], p["elo"])) for p in played[c]]
        for row in table:
            series_.append({"x": list(row), "y": list(row.values()), "colour": COLOURS[k], "alpha": 0.3,
                            "width": 1})
        med = [float(np.median([row[e] for row in table if e in row])) for e in E16_ERAS]
        series_.append({"x": list(E16_ERAS), "y": med, "colour": COLOURS[k], "width": 3, "label": c})
    ch.figure("elo", title="How much better than the founders, over 2,000 iterations",
              x={"label": "iteration the brains are taken from"}, y={"label": "Elo rating against the founders"},
              series=series_, guides=[{"axis": "y", "at": 0, "label": "the founders"}],
              caption=("Thin lines are "
                       "single worlds, thick ones the median world of each condition."),
              recipe=recipe(E16_RUNS, ARCHIVE.replace("every 50", "every 100"), [
                  f"For every pair of the eras {', '.join(str(e) for e in E16_ERAS)} play two contests, "
                  "one in each era's world (`gol_tournament.contest`), and fit one rating per era "
                  "(`gol_tournament.ratings`)."]))
    late = {c: [score(p, 500, 2000) for p in played[c]] for c in E16_CONDITIONS}
    ch.number("late_vs_500", {c: describe(late[c]) for c in E16_CONDITIONS})
    ch.number("champions", {c: [len(tour.dominance(p["eras"], p["contests"])) for p in played[c]]
                            for c in E16_CONDITIONS})
