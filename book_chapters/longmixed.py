# -*- coding: utf-8 -*-
"""
Five mixed worlds, fifty thousand iterations — what Experiment 18's worlds did.

Experiment 18 played five worlds of Experiment 17 (B2 with brains mixed at
birth, seeds 2, 4, 5, 6 and 10) on from iteration 10,000 to 50,000, or until
they died out (runs B2-10000-4af4a3-s002 …). The checks it reads:
research/oee/checks/e17_health.py and e17_breeding.py over this span
(GraphOfLifeRuns/.pilots/e18_health.json, e18_breeding.json) and the lifetime
contests of research/oee/checks/rejudge.py (e18_T100.json).

    sizes      every world's agents through all 50,000 iterations
    health     each world's marks of the health bar and its poor, every 500 iterations
    collapse   the last thousands of iterations of each world that died: who could have a child, who had one
    contests   later brains against earlier ones, and the brains before each collapse against their forebears
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

import book_figures as F
from book_figures import chapter, recipe
from book_chapters.common import BLUE, GREEN, ORANGE, RED, VIOLET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(ROOT, "GraphOfLifeRuns")
PILOTS = os.path.join(RUNS_DIR, ".pilots")
SEEDS = (2, 4, 5, 6, 10)
COLOUR = {2: BLUE, 4: GREEN, 5: ORANGE, 6: VIOLET, 10: RED}
START, END = 10000, 50000
RUNS = ("The five runs of Experiment 18 — Experiment 17's worlds of seeds 2, 4, 5, 6 and 10 (B2 at 10,000 tokens "
        "with brains mixed at birth), played on by the lab from iteration 10,000 to 50,000 or until they died "
        "out (`python3 gol_lab.py run E18`); their first 10,000 iterations are Experiment 17's.")


def _run(seed: int) -> str:
    return f"B2-10000-4af4a3-s{seed:03d}"


def _status(seed: int) -> Tuple[str, int]:
    meta = json.load(open(os.path.join(RUNS_DIR, _run(seed), "meta.json")))
    return meta.get("status") or "", int(meta.get("iteration") or 0)


def _agents(seed: int) -> Tuple[np.ndarray, np.ndarray]:
    its, nodes = F.series(_run(seed), "nodes", 2)
    keep = its < END
    return its[keep], nodes[keep]


def _by_world(rows: List[Dict[str, Any]], key: str) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[int(r[key][-3:])].append(r)
    return out


def _contests() -> Dict[Tuple[int, int, int], List[float]]:
    path = os.path.join(PILOTS, "e18_T100.json")
    out: Dict[Tuple[int, int, int], List[float]] = defaultdict(list)
    for r in json.load(open(path)) if os.path.exists(path) else []:
        if r.get("share") is not None:
            out[(int(r["world"][-3:]), r["a"], r["b"])].append(r["share"])
    return out


@chapter
def longmixed(ch: F.Chapter) -> None:
    states = {s: _status(s) for s in SEEDS}
    if any(state == "running" or (state != "extinct" and at < END) for state, at in states.values()):
        return
    dead = {s: at for s, (state, at) in states.items() if state == "extinct"}

    # 1. Every world through all its iterations.
    series = []
    for s in SEEDS:
        its, nodes = _agents(s)
        its, nodes = F.thin(its, nodes, 100)
        label = f"seed {s}" + (f", died out at {dead[s]:,}" if s in dead else "")
        series.append(F.line(its, nodes, label, COLOUR[s], width=1.3))
    ch.figure("sizes", title="Agents alive, world by world", x={"label": "iteration"},
              y={"label": "agents", "min": 0}, series=series,
              guides=[{"axis": "x", "at": START, "label": "Experiment 17 ends"}],
              caption=("Each line is one world, its agents after each game (means over 100 iterations), from its "
                       "birth in Experiment 17 to iteration 50,000 or to the game after which 20 agents or fewer "
                       "were left, where the lab stops a world."),
              recipe=recipe(RUNS, "Each run's `stats.jsonl`",
                            ["Read the agents after each game (`nodes`, phase 2).",
                             "Average over stretches of 100 iterations."]))
    ch.number("dead", {f"seed {s}": at for s, at in dead.items()})

    # 2. Health and riches every 500 iterations.
    path = os.path.join(PILOTS, "e18_health.json")
    if os.path.exists(path):
        moments = _by_world([r for r in json.load(open(path)) if "bar" in r], "id")
        marks, poor = [], []
        for s in SEEDS:
            rows = sorted(moments.get(s, []), key=lambda r: r["at"])
            x = [r["at"] for r in rows]
            marks.append(F.line(x, [r["marks"] + (SEEDS.index(s) - 2) * 0.06 for r in rows], f"seed {s}",
                                COLOUR[s], points=True, size=3))
            poor.append(F.line(x, [r["share_at_most_three"] for r in rows], f"seed {s}", COLOUR[s], width=1.2))
        ch.grid("health", [
            {"title": "Marks of the health bar met", "x": {"label": "iteration"},
             "y": {"label": "marks (of 7)", "min": 3.5, "max": 7.5}, "series": marks,
             "guides": [{"axis": "y", "at": 7, "label": "every mark"}]},
            {"title": "Agents holding three tokens or fewer", "x": {"label": "iteration"},
             "y": {"label": "share of the agents", "min": 0, "max": 1}, "series": poor,
             "guides": [{"axis": "y", "at": 0.5, "label": "half"}]}],
            columns=2, title="Health and riches, every 500 iterations",
            caption=("Each world read as [Healthy and rich](healthy-and-rich.md) reads every world, from the whole worlds the lab kept every 500 "
                     "iterations from 10,500 on, while it lived. Left: how many of the seven marks of [How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md)'s "
                     "health bar it meets (the dots of each world are set a little apart so that they do not hide each other). "
                     "Right: the share of its agents holding three tokens or fewer."),
            recipe=recipe(RUNS, "Each run's `archive/world_<iteration>.npz` "
                                "(`research/oee/checks/e17_health.py --seeds 2,4,5,6,10 --from 10500 --to 50000`)",
                          ["Read each whole world as `research/atlas/profile.py` reads every world.",
                           "Count the marks it meets and its agents at three tokens or fewer."]))
        total = 0
        healthy = 0
        for s in SEEDS:
            rows = moments.get(s, [])
            alive = {r["at"]: r["marks"] == 7 for r in rows}
            for t in range(START + 500, END + 1, 500):
                total += 1
                healthy += bool(alive.get(t, False))
        ends = {}
        for s in SEEDS:
            rows = sorted(moments.get(s, []), key=lambda r: r["at"])
            ends[f"seed {s}"] = ("died out" if s in dead else
                                 round(rows[-1]["share_at_most_three"], 3) if rows and rows[-1]["at"] == END else None)
        ch.number("thesis", {"moments": total, "healthy": healthy, "share": healthy / total,
                             "at_50000": ends,
                             "dead_or_poor": sum(1 for v in ends.values()
                                                 if v == "died out" or (isinstance(v, float) and v >= 0.5))})

    # 3. Each collapse: who could have a child, and who had one.
    path = os.path.join(PILOTS, "e18_breeding.json")
    if os.path.exists(path) and dead:
        rows = json.load(open(path))["mixed"]
        able, eager = [], []
        for s in sorted(dead):
            mine = rows.get(_run(s), [])
            x = [r["at"] - dead[s] for r in mine]
            able.append(F.line(x, [r["able"] / r["agents"] for r in mine], f"seed {s}", COLOUR[s], width=1.4))
            eager.append(F.line(x, [100 * r["per_able"] if r["per_able"] is not None else None for r in mine],
                                f"seed {s}", COLOUR[s], width=1.4))
        ch.grid("collapse", [
            {"title": "Agents that could have a child", "x": {"label": "iterations before the world died out",
                                                              "min": -12000, "max": 0},
             "y": {"label": "share of the agents", "min": 0, "max": 1}, "series": able},
            {"title": "How often they had one", "x": {"label": "iterations before the world died out",
                                                      "min": -12000, "max": 0},
             "y": {"label": "births per iteration per 100 of them", "min": 0},
             "series": eager, "guides": [{"axis": "y", "at": 1, "label": "one per hundred"}]}],
            columns=2, title="The worlds that died out",
            caption=("Each line is a world that died out, read every 500 iterations up to its end: left, the share "
                     "of its agents that could have a child (11 tokens or more and fewer than 8 links); right, how "
                     "often they had one in the next 50 iterations, per hundred of them. Every agent dies at 100, "
                     "so a world whose agents all could have a child dies out once they have fewer than one per "
                     "hundred an iteration."),
            recipe=recipe(RUNS, "Each run's whole worlds and births (`research/oee/checks/e17_breeding.py --seeds "
                                "2,4,10 --from 10000 --to 50000`)",
                          ["In each whole world, count the agents with 11 tokens or more and fewer than 8 links.",
                           "Divide the births per iteration of the next 50 iterations by them."]))

    # 4. Contests: the pairs the plan named, and the last brains before each collapse.
    contests = _contests()
    if contests:
        planned = sorted(k for k in contests if (k[1], k[2]) in ((10000, 25000), (25000, 50000)))
        before = sorted(k for k in contests if k not in planned)
        charts = []
        for title, keys in (("The pairs the plan named", planned), ("The last brains before a collapse", before)):
            cats = [f"seed {s}, {b:,} vs {a:,}" for s, a, b in keys]
            charts.append({"title": title, "x": {"categories": cats},
                           "y": {"label": "the later brains' share", "min": 0, "max": 1},
                           "series": F.dots(cats, [contests[k] for k in keys], [COLOUR[k[0]] for k in keys]),
                           "guides": [{"axis": "y", "at": 0.5, "label": "no better"}]})
            for k in keys:
                ch.number(f"contest:{k[0]}:{k[2]}v{k[1]}", {"mean": float(np.mean(contests[k])),
                                                           "contests": [round(x, 3) for x in contests[k]]})
        ch.grid("contests", charts, columns=1, title="Later brains against earlier ones",
                caption=("Each dot is a contest between two of a world's moments: the brains of the later moment "
                         "against those of the earlier, half and half on the same world, for 100 iterations, in "
                         "each moment's world with two seeds; the bar is the median of the four. Above: the pairs "
                         "the plan named, for the worlds alive then. Below: the brains of the last world kept "
                         "before each collapse that still had a few hundred agents, against those of 5,000 "
                         "iterations before."),
                recipe=recipe(RUNS, "`GraphOfLifeRuns/.pilots/e18_T100.json` "
                                    "(`research/oee/checks/rejudge.py --seeds 17,18`)",
                              ["Seat the brains of the two moments on alternate agents of each moment's world "
                               "and play 100 iterations, an agent's lifetime.",
                               "Count the agents the later moment's brains hold at the end, as a share of all."]))
