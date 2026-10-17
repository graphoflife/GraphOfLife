# -*- coding: utf-8 -*-
"""
Is a world a space? Chapter 24's rulers — how balls grow, against a random
twin with the same links at every agent — on worlds of every kind
(research/atlas/space.py): B1 and B2 side by side, every world read at
iteration 1,000, and a few worlds through their lives.

    spaces   Chapter 87 · What folds a world
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import numpy as np

import book_figures as F
from book_figures import chapter, recipe
from book_chapters.common import BLUE, GREEN, GREY, ORANGE, RED, VIOLET, YELLOW

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "GraphOfLifeRuns")
PILOTS = os.path.join(RUNS, ".pilots")
ATLAS = os.path.join(PILOTS, "atlas")
sys.path.insert(0, os.path.join(ROOT, "research", "atlas"))

#: The groups of research/atlas/space.py's worlds drawn side by side, and their colours.
SIDE = [("B1 (Experiment 2)", "B1", BLUE),
        ("B2, 2,000 iterations (Experiment 16)", "B2", GREY),
        ("mixed at birth, 10,000 iterations (Experiment 17)", "B2 mixed at birth", ORANGE),
        ("B2, estates to the newborns or parents (batch R)", "B2, estates to newborns or parents", VIOLET)]
#: The worlds read through their lives (research/atlas/space.py LIFE), in two panels.
LIFE = [("At 10,000 tokens", [("B2", GREY), ("B2, no old age", YELLOW), ("B2, estates to the newborns", VIOLET),
                              ("B2, estates to the parents", RED), ("B2, every unused link lapses", GREEN)]),
        ("At 5,000 tokens (Experiment 14)", [("B2 at 5,000 tokens", GREY), ("B2 at 5,000, no mending", BLUE),
                                             ("B2 at 5,000, no links at birth", YELLOW),
                                             ("B2 at 5,000, no cap on links", RED),
                                             ("B2 at 5,000, handover", GREEN)])]
RUNS_TEXT = ("B1: the 26 worlds of 10,000 tokens of Experiment 2 that lived to the end, after their last game; B2: "
             "Experiment 16's ten worlds at iteration 2,000; mixed at birth: Experiment 17's worlds at 10,000; the "
             "estates to the newborns or the parents: batch R's B2 worlds (`R/B2-newborns-s1`, `-s2`, "
             "`R/B2-parents-s1`, `-s2`) at 2,000.")


def _load(name: str) -> List[Dict[str, Any]]:
    path = os.path.join(ATLAS, name)
    return json.load(open(path)) if os.path.exists(path) else []


def _kind(wid: str) -> str:
    """What makes a world of 1,000 iterations stand out, for its colour: its estates, its links, its brains."""
    import space as S  # research/atlas/space.py
    cfg = S.settings(wid)
    if not cfg.get("mend_on_death"):
        return "B1"
    if cfg.get("redistribution") in ("newborns", "parents"):
        return "estates"
    if cfg.get("prune_needs") == 0:
        return "lapse"
    if cfg.get("mix") or cfg.get("mix_at_birth"):
        return "mixed"
    return "B2"


KINDS = [("B2", "B2 with other rules", GREY), ("mixed", "brains mixed at birth", ORANGE),
         ("estates", "estates to the newborns or parents", VIOLET), ("lapse", "every unused link lapses", GREEN),
         ("B1", "B1's rules (no mending)", BLUE)]


@chapter
def spaces(ch: F.Chapter) -> None:
    import space as S  # research/atlas/space.py
    side = _load("space.json")
    every = _load("space_at1000.json")
    life = _load("space_life.json")
    if not side or not every or not life:
        return

    # B1 and B2 side by side: how a typical agent's ball fills its world, and the local dimension.
    share_series, dim_series, side_numbers = [], [], {}
    for group, name, colour in SIDE:
        rows = [r for r in side if r["group"] == group]
        if not rows:
            continue
        for which, dash in (("world", None), ("twin", [5, 4])):
            shares = [np.array(r[which]["balls"]) / r[which]["n"] for r in rows]
            width = min(len(s) for s in shares)
            med = np.median(np.array([s[:width] for s in shares]), axis=0)
            stop = int(np.argmax(med >= 0.999)) or width - 1
            label = name if which == "world" else (None if group != SIDE[0][0] else "random twins (dashed)")
            share_series.append(F.line(range(1, stop + 1), med[1:stop + 1], label, colour, width=2,
                                       **({"dash": dash} if dash else {})))
            grid = sorted({round(x, 6) for r in rows for x in r[which]["ball_r"]})
            table = np.full((len(rows), len(grid)), np.nan)
            for i, r in enumerate(rows):
                for x, y in zip(r[which]["ball_r"], r[which]["ball_d"]):
                    table[i, grid.index(round(x, 6))] = y
            keep = np.sum(np.isfinite(table), axis=0) >= max(2, len(rows) // 2)
            dim_series.append(F.line(np.array(grid)[keep], np.nanmedian(table[:, keep], axis=0), None, colour,
                                     width=2, **({"dash": dash} if dash else {})))
        distance = [S.typical_distance(r) for r in rows]
        side_numbers[name] = {"worlds": len(rows), "agents": float(np.median([r["world"]["n"] for r in rows])),
                              "typical_distance": float(np.nanmedian(distance)),
                              "half_reach": float(np.nanmedian([S.half_reach(r["world"]["balls"], r["world"]["n"])
                                                                for r in rows])),
                              "twin_half_reach": float(np.nanmedian([S.half_reach(r["twin"]["balls"], r["twin"]["n"])
                                                                     for r in rows]))}
    ch.number("side", side_numbers)
    ch.grid("balls", [
        {"title": "How a typical agent's ball fills its world", "x": {"label": "radius r, steps", "log": True},
         "y": {"label": "share of the world within r steps", "log": True}, "series": share_series},
        {"title": "The dimension the balls show", "x": {"label": "radius r, steps (logarithmic)", "log": True},
         "y": {"label": "local dimension d(r)", "min": 0}, "series": dim_series}],
        columns=2, title="B1 and B2 side by side",
        caption=("Left: the share of its world a typical agent's ball holds within r steps (the geometric mean over "
                 "40 agents drawn at random), the median of each kind of world, on logarithmic axes; dashed, the same "
                 "for the worlds' random twins, which have exactly the links of their world at every agent, wired at "
                 "random. Right: the local dimension the balls show, d(r) = ln(N(r+1)/N(r)) / ln((r+1)/r), until a "
                 "ball holds a quarter of the world, the median of each kind; dashed, the twins."),
        recipe=recipe(RUNS_TEXT, "each world's whole world kept then (`research/atlas/space.py`, "
                                 "`GraphOfLifeRuns/.pilots/atlas/space.json`)",
                      ["Build the network of the world's largest piece; make its random twin.",
                       "From 40 agents drawn at random, a breadth-first search; N(r) the geometric mean of the "
                       "number within r steps.", "Read the share N(r)/n and the local dimension d(r)."]))

    # A world through its life: its typical distance, over its twin's.
    charts, life_numbers = [], {}
    for title, groups in LIFE:
        series = []
        for group, colour in groups:
            rows = [r for r in life if r["group"] == group]
            moments = sorted({r["at"] for r in rows})
            meds = [float(np.nanmedian([S.typical_distance(r) for r in rows if r["at"] == t])) for t in moments]
            if moments:
                series.append(F.line(moments, meds, group.replace("B2 at 5,000, ", "").replace("B2, ", ""),
                                     colour, width=2))
                series.append({"label": None, "x": moments, "y": meds, "points": True, "size": 5,
                               "colour": colour})
                life_numbers[group] = dict(zip(map(str, moments), meds))
        charts.append({"title": title, "x": {"label": "iteration"},
                       "y": {"label": "typical distance, over the twin's", "min": 0.8},
                       "series": series, "guides": [{"axis": "y", "at": 1.0, "label": "as the twin"}]})
    ch.number("life", life_numbers)
    ch.grid("life", charts, columns=1, title="A world through its life",
            caption=("The typical distance of a world — the steps within which a typical agent's ball holds half "
                     "of it — over the same for its random twin, at moments of its life; the median of two or three "
                     "seeds of each kind. About 1: a small world, as near as its twin. Above: a space, wider than "
                     "its twin. Above, worlds of 10,000 tokens: B2 itself (Experiment 16), without old age, with "
                     "the estate of the dead to the newborns or to the parents (batch R), and with every unused "
                     "link lapsing (batch S). Below, worlds of 5,000 tokens from Experiment 14, each without one of "
                     "B2's rules, or with handover."),
            recipe=recipe("Experiment 16's B2 worlds and their sister experiment without old age (`B2-10000-s001` "
                          "… `s003`, `B2-10000-3ec24f-s001` … `s003`), batch R's and batch S's pilot worlds, and "
                          "Experiment 14's worlds at 5,000 tokens (`B2-5000-…-s001` … `s003`).",
                          "the whole worlds kept every 100 iterations (`research/atlas/space.py --life`, "
                          "`GraphOfLifeRuns/.pilots/atlas/space_life.json`)",
                          ["Measure each world as in the figure above, at each moment.",
                           "Divide the radius at which its typical ball holds half the world by its twin's."]))

    # Every world at iteration 1,000.
    at = {w["id"]: w for w in json.load(open(os.path.join(ATLAS, "at1000.json")))["worlds"]}
    rows = [r for r in every if r["id"] in at]
    kinds = {r["id"]: _kind(r["id"]) for r in rows}
    rings = {r["id"]: r for r in _load("rings_at1000.json")}
    charts = []
    # The ring a death closes, in the worlds whose agents die of age and are mended round (research/atlas/rings.py).
    series = []
    for kind, name, colour in KINDS:
        mine = [r for r in rows if kinds[r["id"]] == kind and r["id"] in rings]
        if mine:
            series.append({"label": None, "x": [rings[r["id"]]["ring"] for r in mine],
                           "y": [S.typical_distance(r) for r in mine], "points": True, "size": 5, "colour": colour,
                           "alpha": 0.8})
    charts.append({"title": "Against the ring a death closes",
                   "x": {"label": "groups of neighbours of the agents about to die"},
                   "y": {"label": "typical distance, over the twin's", "log": True}, "series": series,
                   "guides": [{"axis": "y", "at": 1.0, "label": "as the twin"}]})
    for key, label, title in (("links_per_agent", "links per agent", "Against the links per agent"),
                              ("share_at_most_three", "share of the agents with three tokens or fewer",
                               "Against the share of the poor")):
        series = []
        for kind, name, colour in KINDS:
            mine = [r for r in rows if kinds[r["id"]] == kind]
            if mine:
                series.append({"label": name if key == "links_per_agent" else None,
                               "x": [float(at[r["id"]][key]) for r in mine],
                               "y": [S.typical_distance(r) for r in mine], "points": True, "size": 5,
                               "colour": colour, "alpha": 0.8})
        charts.append({"title": title, "x": {"label": label},
                       "y": {"label": "typical distance, over the twin's", "log": True}, "series": series,
                       "guides": [{"axis": "y", "at": 1.0, "label": "as the twin"}]})
    distance = np.array([S.typical_distance(r) for r in rows])

    def rank_corr(key: str) -> float:
        v = np.array([float(at[r["id"]][key]) for r in rows])
        ok = np.isfinite(v) & np.isfinite(distance)
        a, b = np.argsort(np.argsort(v[ok])), np.argsort(np.argsort(distance[ok]))
        return float(np.corrcoef(a, b)[0, 1])

    ringed = [r for r in rows if r["id"] in rings and np.isfinite(S.typical_distance(r))]
    ring_v = np.array([rings[r["id"]]["ring"] for r in ringed])
    ring_d = np.array([S.typical_distance(r) for r in ringed])
    bins = {}
    for lo, hi in ((0, 5), (5, 6), (6, 7), (7, 9), (9, 99)):
        m = (ring_v >= lo) & (ring_v < hi)
        if m.any():
            bins[f"{lo}-{hi}"] = {"worlds": int(m.sum()), "median": float(np.median(ring_d[m]))}
    ch.number("rings", {"worlds": len(ringed), "bins": bins,
                        "rank_correlation": float(np.corrcoef(np.argsort(np.argsort(ring_v)),
                                                              np.argsort(np.argsort(ring_d)))[0, 1])})
    ch.number("every", {"worlds": len(rows), "by_kind": {k: {"worlds": sum(1 for r in rows if kinds[r["id"]] == k),
                                                              "median": float(np.nanmedian([d for r, d in
                                                                                            zip(rows, distance)
                                                                                            if kinds[r["id"]] == k]))}
                                                          for k, _, _ in KINDS},
                        "rank_correlation": {k: rank_corr(k) for k in ("links_per_agent", "share_at_most_three",
                                                                       "agents", "tokens_per_agent")}})
    ch.grid("every", charts, columns=1, title="Every world at iteration 1,000",
            caption=("Each dot is a world of the book read at iteration 1,000 — the pilot batches and the lab's "
                     "B2 families at 10,000 tokens — its typical distance over its random twin's (logarithmic). "
                     "Top, for the worlds whose agents die of age and are mended round: against the number of "
                     "groups the neighbours of its agents within ten iterations of their death fall into — groups "
                     "linked to one another directly, which mending joins in a ring. Middle: against its links per "
                     "agent; bottom: against the share of its agents holding three tokens or fewer. Colours: what "
                     "stands out in its rules."),
            recipe=recipe("Every world of `GraphOfLifeRuns/.pilots/atlas/at1000.json` (`research/atlas/at.py`): the "
                          "pilot batches K, M, P, Q, R, S, T, U, W, X and Y, and the lab's B2 families of Experiments "
                          "15 to 17.",
                          "each world's whole world kept at iteration 1,000 (`research/atlas/space.py --at 1000`, "
                          "`GraphOfLifeRuns/.pilots/atlas/space_at1000.json`)",
                          ["Measure each world and its twin as above.", "Split the neighbours of each agent within ten "
                           "iterations of its death into groups linked directly (`research/atlas/rings.py`).",
                           "Set its typical distance over the twin's against the mean number of groups, its links "
                           "per agent and its share of poor agents."]))
