# -*- coding: utf-8 -*-
"""
One rule combined with every other (pilot batches T, W, X and Y, 2026-10-16).

Each world of a batch is the fourth corner of a two-by-two: B2, B2 with the
batch's rule, B2 with another rule, and B2 with both, at 10,000 tokens, seeds 1
and 2. Every corner is read at iteration 1,000, as Meta X reads every world
(research/atlas/at.py → GraphOfLifeRuns/.pilots/atlas/at1000.json, diversity on
Meta X's scale), so that worlds played for different lengths compare; and each
world's brains of 1,000 are played against those of 500 over a lifetime (T = 100;
research/oee/checks/rejudge.py → GraphOfLifeRuns/.pilots/batch_TUV_T100.json).

    mixcombos   Chapter 82 · Brains mixed at birth, with every other rule (batch Y)
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

import book_figures as F
from book_figures import chapter, recipe
from book_chapters.common import GREY, ORANGE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILOTS = os.path.join(ROOT, "GraphOfLifeRuns", ".pilots")
AT = os.path.join(PILOTS, "atlas", "at1000.json")
sys.path.insert(0, os.path.join(ROOT, "research", "oee", "pilots"))

#: The rules, in the words of the chapters, by the names the batches give them.
WORDS = {"bytokens": "estates by wealth", "poorest": "estates to the poorest", "newborns": "estates to newborns",
         "parents": "estates to parents", "linktoken": "links hold a token", "helped": "input: who helped",
         "came": "input: came from", "handover": "handover", "nodice": "no random inputs", "noage": "no old age",
         "anychild": "children of any size", "keephalf": "half kept at home", "keepone": "last token kept",
         "grow": "brains that grow", "unittoken": "units hold a token", "mut05": "slow brain change",
         "mut80": "fast brain change", "lapse": "unused links lapse", "lottery": "nodes by lottery",
         "pool": "pooled by genotype", "kinsense": "a sense of kin", "familysense": "a sense of family",
         "familypool": "pooled by family"}


def _measured() -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(AT):
        return {}
    return {w["id"]: w for w in json.load(open(AT))["worlds"] if "bar" in w}


def _alone(control: str, seed: int) -> str:
    """A control world's id: a lab family's run of this seed, or a pilot world of this seed."""
    return f"{control[4:]}-s{seed:03d}" if control.startswith("lab ") else f"{control}-s{seed}"


def _rules() -> List[Tuple[str, str]]:
    """Every other rule and the world that has it alone, as batch Y pairs them."""
    from make_jobs_TU import W_NOREV, W_REV
    return ([(name, control) for name, _, _, control in W_NOREV if name != "mix"]
            + [(name, f"W/rev-{name}") for name, _, _ in W_REV])


def _pairs(categories: List[str], alone: List[List[float]], both: List[List[float]],
           colours: Tuple[int, int]) -> List[Dict[str, Any]]:
    """Two columns of dots in each category, the rule alone on the left and with the batch's rule on the right."""
    out = []
    for k, _ in enumerate(categories):
        for values, side, colour in ((alone[k], -0.17, colours[0]), (both[k], 0.17, colours[1])):
            v = [x for x in values if x is not None and np.isfinite(x)]
            if not v:
                continue
            out.append({"label": None, "x": [k + side] * len(v), "y": v, "points": True, "size": 6,
                        "colour": colour, "alpha": 0.85})
            m = float(np.mean(v))
            out.append({"label": None, "x": [k + side - 0.13, k + side + 0.13], "y": [m, m], "colour": colour,
                        "width": 3})
    return out


def _contests() -> Dict[str, float]:
    """Each pilot world's later-era share, 1,000 against 500: the mean of its contests."""
    path = os.path.join(PILOTS, "batch_TUV_T100.json")
    by: Dict[str, List[float]] = {}
    for r in json.load(open(path)) if os.path.exists(path) else []:
        if (r["a"], r["b"]) == (500, 1000) and r.get("share") is not None:
            by.setdefault(r["world"].split(".pilots/")[-1], []).append(r["share"])
    return {w: float(np.mean(v)) for w, v in by.items()}


def _lab_contests(family: str, name: str = "long_steps.json") -> List[float]:
    """A lab family's worlds' later-era shares, 1,000 against 500, each the mean of its contests."""
    by: Dict[str, List[float]] = {}
    for r in json.load(open(os.path.join(PILOTS, name))):
        world = r.get("run") or r.get("world")
        if world.startswith(family) and world[len(family):].isdigit() and (r["a"], r["b"]) == (500, 1000):
            by.setdefault(world, []).append(r["share"])
    return [float(np.mean(v)) for v in by.values()]


@chapter
def mixcombos(ch: F.Chapter) -> None:
    measured = _measured()
    rows = []
    for name, control in _rules():
        alone = [measured.get(_alone(control, s)) for s in (1, 2)]
        both = [measured.get(f"Y/mix-{name}-s{s}") for s in (1, 2)]
        if any(both):
            rows.append((name, [w for w in alone if w], [w for w in both if w]))
    if len(rows) < 20:
        return
    b2 = [measured[f"B2-10000-s00{s}"] for s in (1, 2)]
    mix = [measured[f"B2-10000-4af4a3-s00{s}"] for s in (1, 2)]

    def value(ws: List[Dict[str, Any]], key: str) -> List[float]:
        return [float(w[key]) for w in ws if w.get(key) is not None]

    # The rules whose riches mixing keeps first: sorted by the share of poor agents with mixing.
    rows.sort(key=lambda r: np.mean(value(r[2], "share_at_most_three")))
    third = -(-len(rows) // 3)
    parts = [rows[i:i + third] for i in range(0, len(rows), third)]
    for key, label, fig, title in (
            ("share_at_most_three", "share of the agents with three tokens or fewer", "riches",
             "Agents holding three tokens or fewer"),
            ("diversity", "diversity score (Meta X)", "diversity", "How much the agents differ")):
        charts = []
        for part in parts:
            cats = [WORDS[name] for name, _, _ in part]
            charts.append({"x": {"categories": cats}, "y": {"label": label, "min": 0, "max": 1},
                           "series": _pairs(cats, [value(a, key) for _, a, _ in part],
                                            [value(b, key) for _, _, b in part], (GREY, ORANGE)),
                           "guides": [{"axis": "y", "at": float(np.mean(value(b2, key))), "label": "B2"},
                                      {"axis": "y", "at": float(np.mean(value(mix, key))), "label": "mixing alone"}]})
        ch.grid(fig, charts, columns=1, title=title,
                caption=("Each grey dot is a world of B2 with one other rule, each orange dot the same rule with "
                         "brains mixed at birth as well (batch Y), seeds 1 and 2, read at iteration 1,000; the short "
                         "bars are the means. The dashed lines are B2 itself and B2 with mixing alone, the means of "
                         "the same seeds. The rules are in the order of the orange means."),
                recipe=recipe("Batch Y's 46 pilot worlds (B2 at 10,000 tokens, brains mixed at birth and one other "
                              "rule, seeds 1 and 2, `GraphOfLifeRuns/.pilots/Y/`) and their controls: the same rule "
                              "alone, from the batch or lab experiment that made it, and the lab's B2 and mixed "
                              "worlds of the same seeds (Experiments 16 and 17).",
                              "Each world's checkpoint at iteration 1,000, or the lab's whole world kept then "
                              "(`research/atlas/at.py --at 1000`, `GraphOfLifeRuns/.pilots/atlas/at1000.json`)",
                              ["Read each world as `research/atlas/profile.py` reads every world.",
                               "Rank its four spreads among Meta X's worlds; the diversity score is the mean."]))
    # Does evolution go on? The brains of 1,000 against those of 500, a lifetime long.
    contests = _contests()
    if sum(1 for name, _, _ in rows if contests.get(f"Y/mix-{name}-s1") or contests.get(f"Y/mix-{name}-s2")) >= 20:
        b2_med = float(np.median(_lab_contests("B2-10000-s")))
        mix_med = float(np.median(_lab_contests("B2-10000-4af4a3-s", "e17_T100.json")))
        charts = []
        for part in parts:
            cats = [WORDS[name] for name, _, _ in part]
            groups = [[contests[w] for w in (f"Y/mix-{name}-s1", f"Y/mix-{name}-s2") if w in contests]
                      for name, _, _ in part]
            charts.append({"x": {"categories": cats},
                           "y": {"label": "the later brains' share", "min": 0.2, "max": 1},
                           "series": F.dots(cats, groups, [ORANGE] * len(cats)),
                           "guides": [{"axis": "y", "at": 0.5, "label": "no better"},
                                      {"axis": "y", "at": b2_med, "label": "B2"},
                                      {"axis": "y", "at": mix_med, "label": "mixing alone"}]})
        ch.grid("evolution", charts, columns=1, title="Do the brains of 1,000 beat those of 500?",
                caption=("Each dot is a world of batch Y: the share of a contested world its brains of iteration "
                         "1,000 hold after 100 iterations against those of 500, the mean of two contests, one in "
                         "each era's world; the bars are the medians, the rules in the order of the figures above. "
                         "The dashed lines are the medians of B2's ten lab worlds and of the ten mixed worlds of "
                         "Experiment 17, played the same way."),
                recipe=recipe("Batch Y's 46 pilot worlds (`GraphOfLifeRuns/.pilots/Y/`); B2: Experiment 16's ten "
                              "runs; mixing alone: Experiment 17's ten runs.",
                              "`GraphOfLifeRuns/.pilots/batch_TUV_T100.json`, `long_steps.json` and `e17_T100.json` "
                              "(`research/oee/checks/rejudge.py`)",
                              ["Seat the brains of the two eras on alternate agents of each era's world and play "
                               "100 iterations.", "Count the agents the later era's brains hold, as a share of all."]))
        ch.number("evolution", {"worlds": contests, "b2_median": b2_med, "mixing_median": mix_med})
    ch.number("rows", [{"rule": name, "alone": {k: value(a, k) for k in ("share_at_most_three", "diversity",
                                                                          "marks", "agents")},
                        "with_mixing": {k: value(b, k) for k in ("share_at_most_three", "diversity", "marks",
                                                                 "agents")}}
                       for name, a, b in rows])
    ch.number("references", {"B2": {k: value(b2, k) for k in ("share_at_most_three", "diversity", "marks", "agents")},
                             "mixing": {k: value(mix, k) for k in ("share_at_most_three", "diversity", "marks",
                                                                   "agents")}})
