"""
Round 3 of the screen of random worlds (Experiments 40 to 42), as it ended:
every world's number of agents over its run, coloured by how it ended, and the
worlds that died, with when and how. Read from the screen's readings of every
world (research/screen/measures.py keeps a row per iteration) and the judge's
judgement (research/screen/E40.json). Nothing is played.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

import book_figures as F
from book_chapters.common import BLUE, RED
from book_figures import chapter, recipe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCREEN = os.path.join(ROOT, "research", "screen")
JUDGED = os.path.join(SCREEN, "E40.json")
EVERY = 10

RUNS = ("every world of round 3: Experiment 40's 84 worlds, and the eight draws of Experiments 41 and 42 played a "
        "token at a time, weakest first and strongest first (30,000 tokens, 3,000 iterations)")
DATA = ("the screen's readings of every world, `GraphOfLifeRuns/.pilots/screen/worlds/<run>.json`, whose series "
        "holds each iteration's number of agents from the run's `stats.jsonl` (research/screen/measures.py), and "
        "the judge's `research/screen/E40.json` for how each world ended")


def _measures():
    sys.path.insert(0, SCREEN)
    import measures  # noqa: E402
    return measures


def label(w: Dict[str, Any]) -> str:
    tag = {"E41": " w", "E42": " s"}.get(w["round"], "")
    return f"s{w['seed']}{tag}"


@chapter
def roundthree(ch: F.Chapter) -> None:
    if not os.path.exists(JUDGED):
        return
    d = json.load(open(JUDGED))
    worlds = [w for w in d["worlds"] if w["fate"] in ("lived", "died")]
    M = _measures()
    have = M.kept([w["id"] for w in worlds])
    series = {}
    for w in worlds:
        item = (w["id"], "series", w["ended_at"])
        if item not in have:
            continue
        rows = [r for r in have[item]["rows"] if r.get("agents")]
        series[w["id"]] = ([r["t"] for r in rows][::EVERY], [r["agents"] for r in rows][::EVERY])
    lived = [w for w in worlds if w["fate"] == "lived" and w["id"] in series]
    died = [w for w in worlds if w["fate"] == "died" and w["id"] in series]
    lines = [{"x": series[w["id"]][0], "y": series[w["id"]][1], "colour": BLUE, "width": 1, "alpha": 0.35}
             for w in lived]
    lines += [{"x": series[w["id"]][0], "y": series[w["id"]][1], "colour": RED, "width": 1.5, "alpha": 0.8}
              for w in died]
    if lived:
        lines[0]["label"] = f"lived to 3,000 ({len(lived)})"
    if died:
        lines[len(lived)]["label"] = f"died ({len(died)})"
    ch.figure("fates", title="A hundred worlds, iteration by iteration",
              x={"label": "iteration", "min": 0, "max": 3000},
              y={"label": "agents (logarithmic)", "log": True, "min": 10},
              series=lines,
              caption=f"Each line is a world of round 3: its number of agents after every {EVERY}th game. Blue: "
                      f"the {len(lived)} worlds that lived through their 3,000 iterations; red: the {len(died)} that "
                      "fell to 20 agents and died. Every world started with 150, 300 or 600 founders and 30,000 "
                      "tokens.",
              recipe=recipe(RUNS, DATA, [f"Read each world's agents per iteration; keep every {EVERY}th.",
                                         "Draw the living in blue, the dead in red, on a logarithmic scale."]))
    ch.number("fates", {"worlds": len(worlds), "lived": len(lived), "died": len(died),
                        "died_at": F.describe([w["ended_at"] for w in died]),
                        "died_by_250": sum(w["ended_at"] <= 250 for w in died)})
    peak = {w["id"]: max(series[w["id"]][1]) for w in worlds if w["id"] in series}
    ch.number("table:died", {
        "head": ["world", "died at iteration", "most agents it had"],
        "rows": [[{"text": label(w), "href": f"run:{w['id']}"}, f"{w['ended_at']:,}", f"{peak[w['id']]:,}"]
                 for w in sorted(died, key=lambda w: w["ended_at"])]})
