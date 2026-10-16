# -*- coding: utf-8 -*-
"""
Meta X · Healthy and rich: every world against the author's picture of one that
might keep evolving (2026-10-16) — a decentralised, resilient, nearly
three-dimensional network, and agents that differ in almost every statistic.

The measures are research/atlas/profile.py's (GraphOfLifeRuns/.pilots/atlas/
profile.json): the health bar of Chapter 46 read from each world's checkpoint,
and how widely the agents spread over doubling classes of tokens, links and
ages, and over genotypes. A world's diversity score is the mean of its
percentiles among all worlds on those four: no line is drawn anywhere.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np

import book_figures as F
from book_figures import chapter, recipe
from book_chapters.common import BLUE, CYAN, GREEN, GREY, ORANGE, RED, VIOLET, YELLOW

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILOTS = os.path.join(ROOT, "GraphOfLifeRuns", ".pilots")
PROFILE = os.path.join(PILOTS, "atlas", "profile.json")
SPREADS = [("spread_tokens", "tokens"), ("spread_links", "links"), ("spread_ages", "ages"),
           ("effective_genotypes_per_agent", "genotypes")]
MARKS = ["one piece", "2-core", "agent cut", "link cut", "no hub", "links", "dimension"]
#: Groups of worlds, as the chapters on every world draw them.
GROUPS = [("B1 10k", ORANGE), ("B1 100k", RED), ("B1 sizes", GREY), ("B1 5k", YELLOW), ("B1 fixed", YELLOW),
          ("B2 10k", BLUE), ("B2 var", CYAN), ("B2 5k", VIOLET), ("lapse", GREEN), ("long runs", BLUE),
          ("B1 pil.", ORANGE), ("B2 pil.", CYAN)]
#: The map draws fewer groups, each in its own colour.
MAP = [("B1", ["B1 10k", "B1 100k", "B1 sizes", "B1 5k"], ORANGE), ("B1, brains fixed", ["B1 fixed"], YELLOW),
       ("B2", ["B2 10k", "B2 var", "B2 5k"], BLUE), ("lapse", ["lapse"], GREEN), ("long runs", ["long runs"], VIOLET),
       ("B1 pilots", ["B1 pil."], RED), ("B2 pilots", ["B2 pil."], CYAN)]
SIZES = re.compile(r"^B1-(800|1600|3200|6400|12800|25600|51200|102400|204800|409600)(-57b358)?$")
RUNS = ("Every world the book has made that settled (500 iterations or more), each once, at its last checkpoint: "
        "the lab's runs at their end, the pilot worlds at theirs, the long runs at 10,000 iterations; pilots "
        "that replay a lab world are counted as that world (`python3 research/atlas/profile.py`).")


def _group(w: Dict[str, Any]) -> Optional[str]:
    family = w["family"]
    if "/" in family:
        batch = family.split("/")[0]
        if batch == "L":
            return "long runs"
        config = os.path.join(PILOTS, w["id"], "config.json")
        cfg = json.load(open(config))["cfg"] if os.path.exists(config) else {}
        return "B2 pil." if cfg.get("mend_on_death") else "B1 pil."
    if family == "B1-10000":
        return "B1 10k"
    if family == "B1-100000":
        return "B1 100k"
    if family.startswith("B1-10000-"):
        return "B1 fixed"
    if family.startswith("B1-5000"):
        return "B1 5k"
    if SIZES.match(family):
        return "B1 sizes"
    if family == "B2-10000":
        return "B2 10k"
    if family.startswith("B2-10000-"):
        return "B2 var"
    if family == "B2-5000-71fc89":
        return "lapse"
    if family.startswith("B2-5000"):
        return "B2 5k"
    return None


def _load() -> List[Dict[str, Any]]:
    if not os.path.exists(PROFILE):
        return []
    worlds = [w for w in json.load(open(PROFILE))["worlds"] if "bar" in w]
    # Each world's percentile among all worlds on each spread, and their mean.
    for key, _ in SPREADS:
        values = np.array([w.get(key, np.nan) for w in worlds], float)
        order = values.argsort().argsort()
        for w, rank in zip(worlds, order):
            w.setdefault("percentiles", {})[key] = float(rank / max(1, len(worlds) - 1))
    for w in worlds:
        w["diversity"] = float(np.mean(list(w["percentiles"].values())))
        w["marks"] = sum(1 for v in w["bar"].values() if v)
        w["group"] = _group(w)
    return [w for w in worlds if w["group"]]


@chapter
def healthy(ch: F.Chapter) -> None:
    worlds = _load()
    if not worlds:
        return
    colour = dict(GROUPS)
    ch.number("worlds", len(worlds))

    # 1. Every world on both halves.
    rng = np.random.default_rng(7)
    series = []
    for name, members, c in MAP:
        ws = [w for w in worlds if w["group"] in members]
        if ws:
            series.append({"label": name, "x": [w["marks"] + rng.uniform(-0.3, 0.3) for w in ws],
                           "y": [w["diversity"] for w in ws], "points": True, "size": 5, "colour": c,
                           "alpha": 0.75})
    ch.figure("map", title="Healthy, and diverse?",
              x={"label": "marks of the health bar met (of 7)", "min": -0.5, "max": 7.5},
              y={"label": "diversity score (mean percentile of the four spreads)", "min": 0, "max": 1},
              series=series,
              caption=("Each dot is a world at its last checkpoint. Across: how many of the seven marks of "
                       "Chapter 46's health bar it meets — one piece, a 2-core of at least 0.8, no agent whose "
                       "loss cuts off more than 2%, no link more than 1%, no agent joined to more than 5% of the "
                       "others, 3 to 12 links per agent, a dimension of 2.3 to 4 (the dots are spread sideways "
                       "within each count only so that they do not hide each other). Up: how widely its agents "
                       "spread over tokens, links, ages and genotypes, as the mean of its percentiles among all "
                       "worlds. The author's picture of a world that might keep evolving is the top right. B1: "
                       "B1 at every size and its variants (Experiments 2, 8, 10, 12, 13); brains fixed: Experiment "
                       "7; B2: B2 and its variants (Experiments 9, 14, 15, 16), except lapse, where any unused link "
                       "lapses; long runs: the pilot worlds played to 10,000 iterations; pilots: the other pilot "
                       "worlds, by the rule set they start from."),
              recipe=recipe(RUNS, "`GraphOfLifeRuns/.pilots/atlas/profile.json`",
                            ["Read each world's checkpoint: its agents' tokens, links, birth times and genotypes.",
                             "The network: pieces, the 2-core, the worst cut by one agent and by one link (one "
                             "depth-first search), the best-connected agent's share, links per agent; the "
                             "dimension from the table of every world.",
                             "The spreads: the effective number of doubling classes the agents fill (exp of the "
                             "entropy), for tokens, links and ages; genotypes per agent (1 / sum of squared "
                             "shares, over the agents).",
                             "Rank every world on each spread; the score is the mean of its four percentiles."]))

    # 2. Poor and alike: the share of agents with three tokens or fewer.
    groups = {name: [w["share_at_most_three"] for w in worlds if w["group"] == name] for name, _ in GROUPS}
    cats = [g for g in groups if groups[g]]
    ch.figure("poor", title="Agents with three tokens or fewer",
              x={"categories": cats}, y={"label": "share of the agents", "min": 0, "max": 1},
              series=F.dots(cats, [groups[c] for c in cats], [colour[c] for c in cats]),
              caption=("Each dot is a world: the share of its agents holding one, two or three tokens. B1 10k: B1 "
                       "at 10,000 tokens (Experiment 2); 100k: Experiment 10; sizes: Experiment 8; 5k: "
                       "Experiments 12 and 13; fixed: brains that never change (Experiment 7); B2 10k: Experiment "
                       "16; B2 var: B2 with one rule or rate changed (Experiments 9, 15, 16); B2 5k: Experiment 14, "
                       "except lapse, where any unused link lapses; long runs: the pilot worlds played to 10,000 "
                       "iterations; pil.: the other pilot worlds, by the rule set they start from."),
              recipe=recipe(RUNS, "`GraphOfLifeRuns/.pilots/atlas/profile.json`",
                            ["Count the agents of each world's checkpoint holding at most three tokens."]))
    for name in cats:
        ch.number(f"poor:{name}", float(np.median(groups[name])))

    # 3. The four spreads.
    charts = []
    for key, label in SPREADS:
        g = {name: [w[key] for w in worlds if w["group"] == name and w.get(key) is not None] for name, _ in GROUPS}
        cs = [c for c in g if g[c]]
        charts.append({"title": label.capitalize(), "x": {"categories": cs},
                       "y": {"label": "effective classes" if key.startswith("spread") else "per agent", "min": 0},
                       "series": F.dots(cs, [g[c] for c in cs], [colour[c] for c in cs])})
        for name in cs:
            ch.number(f"{key}:{name}", float(np.median(g[name])))
    ch.grid("spreads", charts, columns=2, title="How widely the agents spread",
            caption=("Each dot is a world. Tokens, links, ages: the effective number of doubling classes its agents "
                     "fill (1, 2–3, 4–7, …; for ages, age + 1) — 1 if every agent is in one class, k if they are "
                     "spread evenly over k. Genotypes: the effective number of genotypes per agent (1 for every "
                     "agent its own genotype, near 0 for one genotype everywhere)."),
            recipe=recipe(RUNS, "`GraphOfLifeRuns/.pilots/atlas/profile.json`",
                          ["For each world, exp of the Shannon entropy of its agents' shares over the doubling "
                           "classes of each quantity; the effective genotypes, 1 / sum of squared shares, per agent."]))

    # The marks each group meets, and the worlds that come closest.
    for name, _ in GROUPS:
        ws = [w for w in worlds if w["group"] == name]
        if ws:
            ch.number(f"marks:{name}", {m: float(np.mean([bool(w["bar"].get(m)) for w in ws])) for m in MARKS})
    best = sorted(worlds, key=lambda w: (-w["marks"], -w["diversity"]))[:12]
    ch.number("table:closest", {
        "head": ["world", "group", "marks (of 7)", "diversity score", "median tokens", "agents with 3 tokens or fewer",
                 "dimension", "agents"],
        "rows": [[f"`{w['id']}`", w["group"], str(w["marks"]), f"{w['diversity']:.2f}", f"{w['median_tokens']:.0f}",
                  f"{w['share_at_most_three']:.0%}", f"{w['dimension']:.1f}" if w.get("dimension") else "—",
                  f"{w['agents']:,}"] for w in best]})
    ch.number("seven_marks", {name: sum(1 for w in worlds if w["group"] in members and w["marks"] == 7)
                              for name, members, _ in MAP})
    ch.number("closest", [{"id": w["id"], "group": w["group"], "marks": w["marks"], "diversity": round(w["diversity"], 3),
                           "median_tokens": w["median_tokens"], "share_at_most_three": round(w["share_at_most_three"], 3),
                           "dimension": w.get("dimension"), "agents": w["agents"]} for w in best])
