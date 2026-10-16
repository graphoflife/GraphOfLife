# -*- coding: utf-8 -*-
"""
Chapter 78 · Three more ideas: brains mixed at birth, units that hold a token,
an eye for raids on kin — and links that cost a token in a world without old
age. Pilot batch Q (research/oee/pilots/jobs_Q.txt), B2's settings at 10,000
tokens, two seeds each, 2,000 iterations; the control is batch P's B2 worlds of
the same settings and seeds, and for units that hold a token, the same
growable brains with units for free.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np

import book_figures as F
from book_figures import chapter, recipe
from book_chapters.common import BLUE, GREEN, GREY, ORANGE, RED, VIOLET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILOTS = os.path.join(ROOT, "GraphOfLifeRuns", ".pilots")
VARIANTS = [("B2 (control)", ["P/B2-s1", "P/B2-s2"], GREY),
            ("mixed at birth", ["Q/mix-s1", "Q/mix-s2"], RED),
            ("eye for raids on kin", ["Q/tookkin-s1", "Q/tookkin-s2"], VIOLET),
            ("units free", ["Q/grow-s1", "Q/grow-s2"], BLUE),
            ("units hold a token", ["Q/unittoken-s1", "Q/unittoken-s2"], GREEN),
            ("links cost, no old age", ["Q/linknoage-s1", "Q/linknoage-s2"], ORANGE)]
RUNS = ("Pilot worlds of 10,000 tokens with B2's settings, two seeds each, 2,000 iterations "
        "(`GraphOfLifeRuns/.pilots/Q/`, `research/oee/pilots/jobs_Q.txt`): `mix`, `took_kin`, growable brains "
        "(`brain_kind = grow`, up to 32 units) with units free and with `unit_token`, and `link_token` with "
        "`valence = 0` and `max_age = 0`; the control is batch P's B2 worlds of the same settings and seeds.")


def _contests() -> Dict[str, Dict[Tuple[int, int], List[float]]]:
    """
    Each world's contests, by pair of moments. The mixed-brain worlds' come from
    their replay (batch_Q_mix_T100.json): in the first, every child mixed during a
    contest was counted with the earlier moment, whoever its parent was.
    """
    out: Dict[str, Dict[Tuple[int, int], List[float]]] = {}
    for name in ("batch_Q_T100.json", "batch_P_T100.json", "batch_Q_mix_T100.json"):
        path = os.path.join(PILOTS, name)
        rows = json.load(open(path)) if os.path.exists(path) else []
        if name == "batch_Q_mix_T100.json":
            for c in rows:
                out.pop(c["world"].split(".pilots/")[-1], None)
        for c in rows:
            if c.get("share") is not None:
                out.setdefault(c["world"].split(".pilots/")[-1], {}).setdefault((c["a"], c["b"]), []).append(c["share"])
    return out


def _series(world: str) -> List[Dict[str, Any]]:
    path = os.path.join(PILOTS, world, "series.jsonl")
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


@chapter
def ideas(ch: F.Chapter) -> None:
    contests = _contests()
    charts = []
    for pair, title in (((500, 1000), "1,000 against 500"), ((1000, 2000), "2,000 against 1,000")):
        cats, groups, colours = [], [], []
        for name, worlds, colour in VARIANTS:
            vals = [float(np.mean(contests[w][pair])) for w in worlds if w in contests and pair in contests[w]]
            if vals:
                cats.append(name)
                groups.append(vals)
                colours.append(colour)
                ch.number(f"contest:{name}:{pair[1]}v{pair[0]}", [round(v, 3) for v in vals])
        charts.append({"title": title, "x": {"categories": cats},
                       "y": {"label": "share of nodes the later brains hold", "min": 0, "max": 1},
                       "series": F.dots(cats, groups, colours), "guides": [{"axis": "y", "at": 0.5, "label": "a tie"}]})
    ch.grid("contests", charts, columns=1, title="Does evolution go on?",
            caption=("Lifetime contests (T = 100 iterations): the brains of a later moment against those of an "
                     "earlier one, half and half on the same world, in each moment's world; the share of nodes the "
                     "later brains hold, averaged over the two. Above a half the later brains are better. Each dot "
                     "is a world."),
            recipe=recipe(RUNS, "`GraphOfLifeRuns/.pilots/batch_Q_T100.json`, `batch_Q_mix_T100.json` (the mixed-brain worlds, judged again), `batch_P_T100.json` "
                          "(`research/oee/checks/rejudge.py --T 100`)",
                          ["Play each pair of moments for 100 iterations in each moment's world; average."]))

    # Brain size where units are free and where each holds a token.
    series = []
    for name, worlds, colour in VARIANTS[3:5]:
        for k, w in enumerate(worlds):
            rows = [r for r in _series(w) if "activeMean" in r]
            if not rows:
                continue
            its = [r["it"] for r in rows]
            series.append(F.line(its, [r["activeMean"] for r in rows], name if k == 0 else None, colour, width=2))
            series.append(F.line(its, [r.get("shadowMean") for r in rows], "neutral shadow" if (k == 0 and colour == BLUE) else None,
                                 colour, width=1, dash=[4, 4]))
            ch.number(f"units:{w}", {"start": rows[0]["activeMean"], "end": rows[-1]["activeMean"],
                                     "shadow_end": rows[-1].get("shadowMean")})
    ch.figure("units", title="How many units brains use",
              x={"label": "iteration"}, y={"label": "active units per brain (mean over agents)", "min": 0},
              series=series,
              caption=("Growable brains of up to 32 hidden units, where switching a unit on is free (blue) and where "
                       "every active unit holds one of its agent's tokens (green). The dashed lines are each world's "
                       "neutral shadow: the same count for a unit that mutates but does nothing, which is where the "
                       "number would sit if selection did not care."),
              recipe=recipe(RUNS, "each world's `series.jsonl` (`activeMean`, `shadowMean`)",
                            ["Read the mean number of active units, and of the shadow's, every 50 iterations."]))

    # The worlds themselves.
    for name, worlds, _ in VARIANTS:
        for w in worlds:
            rows = [r for r in _series(w) if 1500 <= r.get("it", 0) <= 2000]
            if rows:
                ch.number(f"world:{w}", {"agents": float(np.mean([r["nodes"] for r in rows])),
                                         "births_per_agent": float(np.mean([r.get("births", 0) / r["nodes"] for r in rows])),
                                         "deaths": float(np.mean([r.get("deaths", 0) for r in rows])),
                                         "gini": float(np.mean([r.get("gini", np.nan) for r in rows]))})
