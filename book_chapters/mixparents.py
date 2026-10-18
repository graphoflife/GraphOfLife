# -*- coding: utf-8 -*-
"""
Chapter 85 · Mixed at birth, with parents as mates — what Experiment 19's worlds did.

Experiment 19 ran B2 with brains mixed at birth, the mate drawn only among
neighbours with a living child (mix_mates = parents), for 20,000 iterations, ten
seeds (runs B2-10000-bcc1bc-s001 … s010). Beside it: Experiment 17's mixed worlds
with any neighbour as mate, to 10,000 iterations (B2-10000-4af4a3-s001 … s010),
and the five of them Experiment 18 played on to 50,000, read here to 20,000.

    sizes      every world's agents through time
    poor       the share of agents holding three tokens or fewer, every 500 iterations
    verdict    each world at iteration 20,000: that share (the thesis), and the marks of the health bar
    breeding   the agents that could have a child, and how often they have one (e17_breeding.py)
    evolution  lifetime contests (T = 100) between eras
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

import book_figures as F
from book_figures import chapter, recipe
from book_chapters.common import GREY, ORANGE, VIOLET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(ROOT, "GraphOfLifeRuns")
PILOTS = os.path.join(RUNS_DIR, ".pilots")
PARENTS = [f"B2-10000-bcc1bc-s{s:03d}" for s in range(1, 11)]
ANY = [f"B2-10000-4af4a3-s{s:03d}" for s in range(1, 11)]
END = 20000
#: The thesis: below this median share of agents with three tokens or fewer at 20,000 (B2's least, at 10,000).
LINE = 0.66
RUNS = ("The ten runs of Experiment 19 — B2 at 10,000 tokens with brains mixed at birth and the mate drawn only among "
        "neighbours with a living child (`mix_mates` = `parents`), seeds 1 to 10, 20,000 iterations (`python3 "
        "gol_lab.py run E19`) — beside Experiment 17's ten runs with any neighbour as mate, to 10,000 iterations, and "
        "the five of them Experiment 18 played on, read to 20,000.")


def _poor_share(path: str) -> Tuple[int, float, int]:
    """(iteration, share of agents with three tokens or fewer, agents) of a whole world."""
    with np.load(path, allow_pickle=False) as z:
        tokens = z["tokens"]
        iteration = int(z["counters"][2])
    return iteration, float(np.mean(tokens <= 3)) if tokens.size else float("nan"), int(tokens.size)


def _through_time(run: str) -> Tuple[np.ndarray, np.ndarray]:
    points = sorted(_poor_share(p) for p in glob.glob(os.path.join(RUNS_DIR, run, "archive", "world_*.npz")))
    points = [p for p in points if p[0] % 500 == 0 and 0 < p[0] <= END]
    return np.array([p[0] for p in points]), np.array([p[1] for p in points])


def _contests(path: str) -> Dict[Tuple[int, int], Dict[str, float]]:
    """Each world's later-era share for each pair of eras: the mean of its contests."""
    by: Dict[Tuple[int, int], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in json.load(open(path)) if os.path.exists(path) else []:
        if r.get("share") is not None:
            by[(r["a"], r["b"])][r["world"]].append(r["share"])
    return {pair: {w: float(np.mean(v)) for w, v in worlds.items()} for pair, worlds in by.items()}


def _ended(run: str) -> int:
    meta = os.path.join(RUNS_DIR, run, "meta.json")
    return int(json.load(open(meta)).get("iteration", 0)) if os.path.exists(meta) else 0


@chapter
def mixparents(ch: F.Chapter) -> None:
    if not all(os.path.exists(os.path.join(RUNS_DIR, run, "meta.json")) for run in PARENTS):
        return
    if any(json.load(open(os.path.join(RUNS_DIR, run, "meta.json"))).get("status") == "running" for run in PARENTS):
        return

    # 1. Every world's agents through time.
    series = []
    for k, run in enumerate(ANY):
        its, nodes = F.series(run, "nodes", 2)
        keep = its <= END
        its, nodes = F.thin(its[keep], nodes[keep], 50)
        series.append(F.line(its, nodes, "any neighbour as mate (Experiments 17 and 18)" if k == 0 else None, GREY,
                             width=1.0))
    for k, run in enumerate(PARENTS):
        its, nodes = F.series(run, "nodes", 2)
        its, nodes = F.thin(its, nodes, 50)
        series.append(F.line(its, nodes, "only parents as mates (Experiment 19)" if k == 0 else None, ORANGE,
                             width=1.3))
    ch.figure("sizes", title="Agents alive, world by world", x={"label": "iteration"},
              y={"label": "agents", "min": 0}, series=series,
              caption=("Each line is one world, its agents after each game (means over 50 iterations). Orange: the "
                       "ten worlds of Experiment 19, whose mates are drawn only among neighbours with a living child; "
                       "grey: Experiment 17's ten, any neighbour a mate, to 10,000 iterations, and the five of them "
                       "Experiment 18 played on."),
              recipe=recipe(RUNS, "each run's `stats.jsonl`", ["Read the agents after each game (`nodes`, phase 2).",
                                                               "Average over stretches of 50 iterations."]))

    # 2. Agents with three tokens or fewer, every 500 iterations.
    parents_poor = [_through_time(run) for run in PARENTS]
    any_poor = [_through_time(run) for run in ANY]
    ch.figure("poor", title="Agents holding three tokens or fewer", x={"label": "iteration"},
              y={"label": "share of the agents", "min": 0, "max": 1},
              series=[F.band_series(any_poor, "any neighbour as mate", GREY),
                      F.band_series(parents_poor, "only parents as mates", ORANGE)],
              guides=[{"axis": "y", "at": LINE, "label": "the thesis's line"}],
              caption=("Each band is the worlds of one experiment — the median, the middle half and nine in ten — "
                       "every 500 iterations: the share of their agents holding one, two or three tokens. Grey: "
                       "Experiment 17's ten worlds to 10,000, and the five Experiment 18 played on; a world that "
                       "died out leaves the band when it does."),
              recipe=recipe(RUNS, "each run's whole worlds kept every 500 iterations (`archive/world_<iteration>.npz`)",
                            ["Read every agent's tokens in each whole world.",
                             "Count those holding three or fewer, as a share of all."]))

    # 3. The verdict: each world at the end.
    ends = {run: _ended(run) for run in PARENTS}
    shares = {}
    for run in PARENTS:
        path = os.path.join(RUNS_DIR, run, "archive", f"world_{END:06d}.npz")
        if os.path.exists(path):
            shares[run] = _poor_share(path)[1]
    health_path = os.path.join(PILOTS, "e19_health.json")
    marks = {}
    if os.path.exists(health_path):
        for r in json.load(open(health_path)):
            if r.get("at") == END and "marks" in r:
                marks[r["id"]] = r["marks"]
    lived = [run for run in PARENTS if ends[run] >= END]
    ch.number("verdict", {"lived": len(lived), "ended": ends, "shares": shares,
                          "median_share": float(np.median(list(shares.values()))) if shares else None,
                          "line": LINE, "marks": marks})
    ch.grid("verdict", [
        {"title": "Agents with three tokens or fewer, at 20,000", "x": {"categories": ["only parents as mates"]},
         "y": {"label": "share of the agents", "min": 0, "max": 1},
         "series": F.dots(["only parents as mates"], [list(shares.values())], [ORANGE]),
         "guides": [{"axis": "y", "at": LINE, "label": "the thesis: the median below this"}]},
        {"title": "Marks of the health bar met", "x": {"categories": ["only parents as mates"]},
         "y": {"label": "marks (of 7)", "min": 0, "max": 7.5},
         "series": F.dots(["only parents as mates"], [list(marks.values())], [ORANGE]),
         "guides": [{"axis": "y", "at": 7, "label": "every mark"}]}],
        columns=2, title="Experiment 19 at iteration 20,000",
        caption=("Each dot is a world at iteration 20,000, read from the whole world kept then. Left: the share of "
                 "its agents holding three tokens or fewer; the thesis asked that the median of the ten be below "
                 "0.66, the least of B2's ten worlds at 10,000 iterations. Right: how many of the seven marks of "
                 "Chapter 46's health bar each world meets."),
        recipe=recipe(RUNS, "each run's whole world kept at 20,000 (`archive/world_020000.npz`) and "
                            "`GraphOfLifeRuns/.pilots/e19_health.json` (`research/oee/checks/e17_health.py --family "
                            "B2-10000-bcc1bc`)",
                      ["Count the agents holding three tokens or fewer, as a share of all.",
                       "Measure the seven marks as `research/atlas/profile.py` does."]))

    # 4. Who could have a child, and who has one: any mate against only parents.
    any_path, parents_path = (os.path.join(PILOTS, "e17_breeding.json"), os.path.join(PILOTS, "e19_breeding.json"))
    if os.path.exists(any_path) and os.path.exists(parents_path):
        kinds = [("any neighbour as mate", GREY, json.load(open(any_path))["mixed"]),
                 ("only parents as mates", ORANGE, json.load(open(parents_path))["mixed"])]

        def band(worlds, value):
            out = []
            for rows in worlds.values():
                rows = [r for r in rows if value(r) is not None]
                if rows:
                    out.append((np.array([r["at"] for r in rows]), np.array([value(r) for r in rows], float)))
            return out

        def able(r):
            return r["able"] / r["agents"] if r["agents"] else None

        def eager(r):
            return 100 * r["per_able"] if r["per_able"] else None

        ch.grid("breeding", [
            {"title": "Agents that could have a child", "x": {"label": "iteration"},
             "y": {"label": "share of the agents", "min": 0, "max": 1},
             "series": [F.band_series(band(w, able), name, colour) for name, colour, w in kinds]},
            {"title": "How often they have one", "x": {"label": "iteration"},
             "y": {"label": "births per iteration per 100 of them", "log": True},
             "series": [F.band_series(band(w, eager), name, colour) for name, colour, w in kinds]}],
            columns=2, title="Who could have a child, and who has one",
            caption=("Left: the share of the agents that could have a child — 11 tokens or more and fewer than 8 "
                     "links. Right: the births of the next 50 iterations, per iteration, per hundred such agents, on "
                     "a logarithmic axis. Each band is an experiment's worlds every 500 iterations — the median, "
                     "the middle half and nine in ten."),
            recipe=recipe(RUNS, "each world kept whole every 500 iterations, and each run's births "
                                "(`research/oee/checks/e17_breeding.py`, `GraphOfLifeRuns/.pilots/e17_breeding.json` "
                                "and `e19_breeding.json`)",
                          ["In each whole world, count the agents with 11 tokens or more and fewer than 8 links.",
                           "Read the births of the 50 reproduction phases that follow.",
                           "Divide their mean per iteration by the agents counted."]))
        ch.number("breeding", {name: {"able": float(np.median([able(r) for rows in w.values() for r in rows
                                                               if able(r) is not None])),
                                      "per_100_able": float(np.median([eager(r) for rows in w.values() for r in rows
                                                                       if eager(r) is not None]))}
                               for name, _, w in kinds})

    # 5. Does evolution go on?
    parents_contests = _contests(os.path.join(PILOTS, "e19_T100.json"))
    any_contests = _contests(os.path.join(PILOTS, "e17_T100.json"))
    pairs = [((500, 1000), "1,000/500"), ((2000, 5000), "5,000/2,000"), ((5000, 10000), "10,000/5,000"),
             ((10000, 20000), "20,000/10,000")]
    cats, groups, colours = [], [], []
    for pair, label in pairs:
        for who, contests, colour in (("any mate", any_contests, GREY), ("parents", parents_contests, ORANGE)):
            values = list(contests.get(pair, {}).values())
            if values:
                cats.append(f"{who} · {label}")
                groups.append(values)
                colours.append(colour)
    if cats:
        ch.figure("evolution", title="Do later brains beat earlier ones?", x={"categories": cats},
                  y={"label": "the later brains' share of a contested world", "min": 0.0, "max": 1}, series=F.dots(
                      cats, groups, colours), guides=[{"axis": "y", "at": 0.5, "label": "no better"}],
                  caption=("Each dot is a world: the share of a world the later era's brains hold after 100 "
                           "iterations against the earlier era's, the mean of two contests, one in each era's world; "
                           "the bar is the median world. 1,000/500: the brains of iteration 1,000 against those of "
                           "500. Orange: Experiment 19; grey: Experiment 17, any neighbour a mate."),
                  recipe=recipe(RUNS, "each run's whole worlds (`GraphOfLifeRuns/.pilots/e19_T100.json` and "
                                      "`e17_T100.json`)",
                                ["Seat the brains of the two eras on alternate agents of each era's world and play "
                                 "it on for 100 iterations (`research/oee/checks/rejudge.py`).",
                                 "Count the agents the later era's brains hold at the end, as a share of all."]))
        ch.number("evolution", {label: {"parents": [round(v, 3) for v in parents_contests.get(pair, {}).values()],
                                        "any": [round(v, 3) for v in any_contests.get(pair, {}).values()]}
                                for pair, label in pairs})
