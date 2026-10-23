# -*- coding: utf-8 -*-
"""
How alike the agents are: the measures of How different are the agents?
(research/oee/checks/diversity.py) in every kind of world — three seeds of
every lab condition at the end of its run, and both seeds of every pilot
condition (research/oee/checks/diversity_across.sh, diversity_more.sh; results
in GraphOfLifeRuns/.pilots/diversity/).

A world without random inputs has no noise to set the line between kinds: one
brain never differs from itself, so any difference at all makes two kinds. Its
count of kinds is drawn apart and not compared with the others.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import book_figures as F
from book_figures import chapter, recipe
from book_chapters.common import BLUE, GREEN, GREY, ORANGE
from book_chapters.diversity import DIVERSITY, METHOD, CHECK

#: Lab conditions, by family: what to call them, and which rule set they belong to.
LAB: List[Tuple[str, str, str]] = [
    ("B1-10000", "B1", "B1"),
    ("B1-10000-a8525d", "fixed", "B1"),
    ("B1-5000", "B1 5k", "B1"),
    ("B1-100000", "B1 100k", "B1"),
    ("B1-sizes", "B1 sizes", "B1"),
    ("B2-10000", "B2", "B2"),
    ("B2-10000-bf74a9", "m .05", "B2"),
    ("B2-10000-87318c", "m .1", "B2"),
    ("B2-10000-56c74d", "m .4", "B2"),
    ("B2-10000-fe3009", "m .8", "B2"),
    ("B2-10000-054b50", "keep 1", "B2"),
    ("B2-10000-df2615", "keep ½", "B2"),
    ("B2-10000-185eb7", "child", "B2"),
    ("B2-10000-3ec24f", "no age", "B2"),
    ("B2-10000-bad270", "no dice", "B2"),
    ("B2-5000", "B2 5k", "B2"),
    ("B2-5000-28aef0", "no link", "B2"),
    ("B2-5000-5c8c75", "no val.", "B2"),
    ("B2-5000-894419", "hand", "B2"),
    ("B2-5000-d6b607", "no mend", "B2"),
    ("B2-5000-71fc89", "lapse", "lapse"),
]
#: What the short names stand for, said under the figure.
NAMES = ("B1: B1 at 10,000 tokens (Experiment 2); fixed: brains that never change (Experiment 7); B1 5k, B1 100k, "
         "B1 sizes: B1 at 5,000 (Experiment 12), 100,000 (Experiment 10) and 3,200 to 204,800 tokens (Experiment 8). B2: B2 "
         "at 10,000 tokens (Experiment 16); m .05 to m .8: the chance that a copied brain changes, 0.05 to 0.8 "
         "(B2's own is 0.2; Experiment 9); keep 1, keep ½: the last token kept, half the tokens kept; child: "
         "children of any size; no age: no old age (Experiment 15); B2 5k: B2 at 5,000 tokens, and with one of its "
         "rules taken away — no link: no links at birth, no val.: no valence, hand: handover allowed, no mend: no "
         "mending, lapse: any unused link lapses (Experiment 14).")
SIZES = re.compile(r"^B1-(800|1600|3200|6400|12800|25600|51200|102400|204800|409600)(-57b358)?$")
COLOUR = {"B1": ORANGE, "B2": BLUE, "lapse": GREEN}
MUTATION = [("B2-10000-bf74a9", 0.05), ("B2-10000-87318c", 0.1), ("B2-10000", 0.2), ("B2-10000-56c74d", 0.4),
            ("B2-10000-fe3009", 0.8)]
NO_DICE = "B2-10000-bad270"
#: Batch S's longer condition names, shortened so that the labels do not run into each other.
SHORT = {"kinsense": "kin", "familysense": "family", "familypool": "fam. pool", "giftslapse": "g.+lapse",
         "bytokens": "by tok."}
#: The pilot batches measured, in the order of the panels, and what each tried.
PILOT_BATCHES = [("A3k", "Alliances at 3,000 tokens (A3k)"), ("A10k", "Alliances at 10,000 tokens (A10k)"),
                 ("N", "Rules without numbers (N)"),
                 ("S", "The switches and contests (S; B1 and B2 underneath)"),
                 ("P", "Two inputs, links that hold a token (P)"),
                 ("Q", "Mixed brains, units that cost, an eye for raids on kin (Q)")]
RUNS = ("Three seeds of every condition of the lab's experiments at the end of their runs (Experiments 7 to 16; "
        "B2 itself from Experiment 16 at 2,000 iterations and B1 from Experiment 2 at 3,000), and both seeds of "
        "every pilot condition at its checkpoint of 1,000 or 2,000 iterations.")


def _load(label: str) -> List[Dict[str, Any]]:
    path = os.path.join(DIVERSITY, f"{label}.json")
    return json.load(open(path))["worlds"] if os.path.exists(path) else []


def _family(world: Dict[str, Any]) -> str:
    """The lab family or pilot condition a measured world belongs to."""
    spec = world["world"]
    if spec.startswith("pilot:"):
        path = spec.split(":")[1]
        batch, name = path.rstrip("/").split("/")[-2:]
        return f"{batch}/{re.sub(r'-s[0-9]+$', '', name)}"
    family = re.sub(r"-s\d+$", "", spec.split(":")[0])
    return "B1-sizes" if SIZES.match(family) else family


def _median(values: List[Optional[float]]) -> Optional[float]:
    v = [x for x in values if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def _relative(w: Dict[str, Any]) -> Optional[float]:
    """How far apart two agents stake, against two fresh founders in the same situations (1: as random brains)."""
    founders = w["stake_founders"]["between"]
    return w["stake"]["between"] / founders if founders > 0 else None


def _worlds() -> Dict[str, List[Dict[str, Any]]]:
    """Every measured world by family: B1 and B2 themselves from How different are the agents?'s measures, the rest from the sweep."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for w in _load("B1"):
        out.setdefault("B1-10000", []).append(w)
    for w in _load("B2"):
        if w["world"].endswith(":2000"):
            out.setdefault("B2-10000", []).append(w)
    for label in ("across_lab", "across_pilots"):
        for w in _load(label):
            out.setdefault(_family(w), []).append(w)
    return out


@chapter
def how_alike(ch: F.Chapter) -> None:
    worlds = _worlds()
    if not any(f in worlds for f, _, _ in LAB[1:]):
        return

    # 1. Kinds of staking in every lab condition.
    cats, groups, colours, included = [], [], [], []
    for family, label, base in LAB:
        ws = worlds.get(family) or []
        if not ws or family == NO_DICE:
            continue
        included.append(family)
        cats.append(label)
        groups.append([w["stake_kinds"]["effective"] for w in ws])
        colours.append(COLOUR[base])
        ch.number(f"kinds:{label}", {"staking": _median([w["stake_kinds"]["effective"] for w in ws]),
                                     "every decision": _median([w["kinds"]["effective"] for w in ws]),
                                     "genotypes per agent": _median([w["genotypes"] / w["agents"] for w in ws]),
                                     "apart": _median([w["stake_kinds"]["pairs_apart"] for w in ws]),
                                     "relative": _median([_relative(w) for w in ws]),
                                     "worlds": len(ws)})
    charts = [{"title": "Kinds of staking", "x": {"categories": cats},
               "y": {"label": "effective number of kinds", "log": True, "min": 1},
               "series": F.dots(cats, groups, colours)},
              {"title": "How far apart two agents stake, against random brains", "x": {"categories": cats},
               "y": {"label": "agents' distance / founders' distance", "min": 0},
               "series": F.dots(cats, [[_relative(w) for w in worlds[f]] for f in included], colours),
               "guides": [{"axis": "y", "at": 1, "label": "as different as random brains"}]}]
    ch.grid("lab", charts, columns=1, title="How alike the agents are, in every lab condition",
            caption=("Each dot is a world. Top: how many kinds of staking its agents fall into, counted by the "
                     "agents in each (an effective number; logarithmic axis). Bottom: how far apart two agents' "
                     "stakes are in the same situation and with the same random inputs, divided by how far apart "
                     "two fresh founder brains are in the same situations: 1 means as different as brains drawn "
                     "at random, less means more alike. Orange: B1 and its variants; blue: B2 and its variants; "
                     "green: B2 with lapsing links. B2 without random inputs is left out of both: without noise "
                     "every difference counts, and its kinds cannot be compared. " + NAMES),
            recipe=recipe(RUNS, f"`GraphOfLifeRuns/.pilots/diversity/across_lab.json`, `B1.json`, `B2.json` (`{CHECK}`)",
                          METHOD + ["Count the kinds of staking as in [How different are the agents?](how-different-are-the-agents.md); divide the agents' total "
                                    "variation by the founders'."]))

    # 2. The rate of change: more mutation, more genotypes, more kinds.
    rates, kinds, genotypes = [], [], []
    b2_1500 = _load("B2_1500")
    for family, rate in MUTATION:
        ws = b2_1500 if family == "B2-10000" else worlds.get(family) or []
        if not ws:
            continue
        rates.append(rate)
        kinds.append([w["stake_kinds"]["effective"] for w in ws])
        genotypes.append([w["genotypes"] / w["agents"] for w in ws])
    if len(rates) >= 3:
        def dots_at(xs, groups, colour):
            pts = {"label": None, "x": [x for x, g in zip(xs, groups) for _ in g], "y": [v for g in groups for v in g],
                   "points": True, "size": 6, "colour": colour, "alpha": 0.8}
            med = F.line(xs, [float(np.median(g)) for g in groups], None, colour, width=2)
            return [pts, med]
        ch.grid("mutation", [
            {"title": "Kinds of staking", "x": {"label": "chance that a copied brain changes", "log": True, "min": 0.04, "max": 1},
             "y": {"label": "effective number of kinds", "log": True}, "series": dots_at(rates, kinds, BLUE)},
            {"title": "Genotypes", "x": {"label": "chance that a copied brain changes", "log": True, "min": 0.04, "max": 1},
             "y": {"label": "distinct genotypes per agent", "min": 0, "max": 1},
             "series": dots_at(rates, genotypes, GREY)}],
            columns=2, title="The more brains change, the more kinds",
            caption=("B2's worlds of Experiment 9 at 1,500 iterations, at five chances that a brain changes when it "
                     "is copied (B2's own is 0.2). Each dot is a world; the line joins the medians. Left: kinds of "
                     "staking, counted by their carriers; right: distinct genotypes per agent."),
            recipe=recipe("Three seeds of each of Experiment 9's five mutation rates, at iteration 1,500.",
                          f"`GraphOfLifeRuns/.pilots/diversity/across_lab.json` and `B2_1500.json` (`{CHECK}`)",
                          METHOD))
        ch.number("mutation", {str(r): {"kinds": float(np.median(k)), "genotypes per agent": float(np.median(g))}
                               for r, k, g in zip(rates, kinds, genotypes)})

    # 3. Every pilot condition.
    pilots = sorted(f for f in worlds if "/" in f)
    if pilots:
        def base_of(f: str) -> str:
            w = worlds[f][0]
            path = w["world"].split(":")[1]
            cfg = json.load(open(os.path.join(path, "config.json"))).get("cfg", {})
            return "B2" if cfg.get("mend_on_death") else "B1"
        # One panel per batch, the conditions by their own short names.
        charts = []
        for batch, title in PILOT_BATCHES:
            fs = [f for f in pilots if f.split("/")[0] == batch]
            if not fs:
                continue
            cats = [SHORT.get(c, c) for c in (re.sub(r"^[GS]\d-", "", f.split("/")[1]) for f in fs)]
            charts.append({"title": title, "x": {"categories": cats},
                           "y": {"label": "kinds of staking", "log": True, "min": 1, "max": 100}, "height": 170,
                           "series": F.dots(cats, [[w["stake_kinds"]["effective"] for w in worlds[f]] for f in fs],
                                            [COLOUR[base_of(f)] for f in fs])})
        ch.grid("pilots", charts, columns=1, title="Kinds of staking in the pilot worlds",
                caption=("Each dot is a pilot world at its checkpoint of 1,000 iterations (batches A, N and S) or "
                         "2,000 (batches P and Q), the bar the median; orange: B1's rules underneath, blue: B2's. "
                         "The pages that ran each batch say what its conditions change. Batch S: lottery, genotype, "
                         "kin, family, fam. pool — contests by lottery, sides pooled by genotype, an eye for kin, an "
                         "eye for a family tag, sides pooled by family (all without revolutions); B2, norev, gifts, "
                         "g.+lapse, lapse, by tok. — B2 itself, without revolutions, with gifts, with gifts and "
                         "every unused link lapsing, with every unused link lapsing, estates shared by wealth; B1, "
                         "B1norev — B1, and B1 without revolutions."),
                recipe=recipe("Both seeds of every pilot condition that has a checkpoint there.",
                              f"`GraphOfLifeRuns/.pilots/diversity/across_pilots.json` (`{CHECK}`)", METHOD))
        for f in pilots:
            ws = worlds[f]
            ch.number(f"pilot:{f}", {"staking": _median([w["stake_kinds"]["effective"] for w in ws]),
                                     "every decision": _median([w["kinds"]["effective"] for w in ws]),
                                     "relative": _median([_relative(w) for w in ws]), "worlds": len(ws)})

    # The world without random inputs, apart.
    ws = worlds.get(NO_DICE) or []
    if ws:
        ch.number("no random inputs", {"staking": _median([w["stake_kinds"]["effective"] for w in ws]),
                                       "chance share of home": _median([w["decisions"]["home"]["chance"] for w in ws]),
                                       "relative": _median([_relative(w) for w in ws])})
