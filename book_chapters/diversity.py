# -*- coding: utf-8 -*-
"""
How different are the agents? Every brain of a world put into the same
situations (research/oee/checks/diversity.py; results in
GraphOfLifeRuns/.pilots/diversity/<label>.json). This module draws them.

The worlds: the ten B2 worlds of Experiment 16 at 0, 100, 500, 1,000 and 2,000
iterations (from their archives), the first ten B1 worlds of Experiment 2 at the
end of their 3,000 iterations, and the long pilot worlds (B1 from its start to
10,000, B2 at 5,000 and 10,000); and, for kinds over time, pairs of moments of
the same worlds.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import book_figures as F
from book_figures import chapter, recipe
from book_chapters.common import BLUE, CYAN, GREEN, GREY, ORANGE, RED, VIOLET, YELLOW

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIVERSITY = os.path.join(ROOT, "GraphOfLifeRuns", ".pilots", "diversity")
DECISIONS = [("home", "kept at home"), ("child", "given to a child"), ("spread", "spread or all in"),
             ("link", "a newborn's links"), ("revolt", "marked revolutionary")]
PARTS = [("situation", "where it is", BLUE), ("brain", "who it is", ORANGE),
         ("interaction", "who it is, there", VIOLET), ("chance", "chance", F.REST)]
CHECK = "research/oee/checks/diversity.py"
METHOD = [f"Load the world and play one game, so its messages are fresh (`{CHECK}`).",
          "Take what 200 living agents read, and draw their random input rows 8 times, the same draws "
          "for every brain.",
          "Put the brains of 400 agents drawn at random, and 400 fresh founder brains, into every "
          "situation and draw, and turn their outputs into decisions by the engine's rules."]


def _load(label: str) -> Dict[str, Any]:
    path = os.path.join(DIVERSITY, f"{label}.json")
    return json.load(open(path)) if os.path.exists(path) else {"worlds": [], "persist": []}


def _at(world: Dict[str, Any]) -> Optional[int]:
    """The iteration a measured world was taken at, from its name ('RUN:IT', 'pilot:DIR:IT'), or None for a final state."""
    m = re.search(r":(\d+)$", world["world"])
    return int(m.group(1)) if m else None


def _median(values: List[float]) -> Optional[float]:
    v = [x for x in values if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def _shares(worlds: List[Dict[str, Any]], key: str = "decisions") -> Dict[str, Dict[str, float]]:
    """The median over worlds of each part of each decision's variance."""
    return {d: {p: _median([w[key][d][p] for w in worlds]) for p, _, _ in PARTS} for d, _ in DECISIONS}


@chapter
def diversity(ch: F.Chapter) -> None:
    b2, b1, long_runs, persist = _load("B2"), _load("B1"), _load("L"), _load("persist")
    b2_end = [w for w in b2["worlds"] if _at(w) == 2000]
    b1_end = b1["worlds"]
    if not b2_end:
        return

    # 1. Where, who and chance.
    charts = []
    for title, worlds, key in (("B2 after 2,000 iterations", b2_end, "decisions"),
                               ("B1 after 3,000 iterations", b1_end, "decisions"),
                               ("Founder brains, in B2's situations", b2_end, "founders")):
        if not worlds:
            continue
        shares = _shares(worlds, key)
        cats = [label for _, label in DECISIONS]
        charts.append({"title": title, "x": {"categories": cats}, "y": {"label": "share of the variance", "min": 0, "max": 1},
                       "series": F.stacked(cats, [(name, [shares[d][p] for d, _ in DECISIONS], colour)
                                                  for p, name, colour in PARTS])})
        for d, _ in DECISIONS:
            for p, _, _ in PARTS:
                ch.number(f"{title}:{d}:{p}", round(shares[d][p], 3))
    ch.grid("split", charts, columns=1, title="Where, who and chance",
            caption=("How the variance of each decision splits: into where the agent is (the situation), who it "
                     "is (its brain, wherever it is), who it is in that situation (how its brain answers that "
                     "situation), and chance (its random inputs). The median over ten worlds of each part. A "
                     "decision driven by where agents are is one they all take alike; one driven by who they "
                     "are is a trait that differs between them."),
            recipe=recipe("The ten B2 worlds of Experiment 16 at iteration 2,000 (their archives) and the first ten "
                          "B1 worlds of Experiment 2 at the end of their 3,000 iterations.",
                          f"`GraphOfLifeRuns/.pilots/diversity/B2.json` and `B1.json` (`{CHECK}`)",
                          METHOD + ["Split each decision's variance over brains, situations and draws into the "
                                    "four parts (the note *Where, who and chance*); take the median over worlds."]))

    # 2. Two agents in the same spot, over a world's life.
    times = sorted({_at(w) for w in b2["worlds"] if _at(w) is not None})
    by_time = {t: [w for w in b2["worlds"] if _at(w) == t] for t in times}
    series = []
    for key, label, colour in (("between", "two agents, same situation and draw", BLUE),
                               ("within", "one brain, two draws (its own noise)", GREY)):
        series.append(F.line(times, [_median([w["stake"][key] for w in by_time[t]]) for t in times], label, colour, width=2))
    series.append(F.line(times, [_median([w["stake_founders"]["between"] for w in by_time[t]]) for t in times],
                         "two founder brains", ORANGE, width=2, dash=[5, 4]))
    kinds = [F.line(times, [_median([w["stake_kinds"]["effective"] for w in by_time[t]]) for t in times],
                    "kinds of staking (effective number)", BLUE, width=2),
             F.line(times, [_median([w["founder_stake_kinds"]["effective"] for w in by_time[t]]) for t in times],
                    "the same among founders", ORANGE, width=2, dash=[5, 4])]
    ch.number("b2_stake_tv", {str(t): _median([w["stake"]["between"] for w in by_time[t]]) for t in times})
    ch.number("b2_stake_noise", {str(t): _median([w["stake"]["within"] for w in by_time[t]]) for t in times})
    ch.number("b2_stake_kinds", {str(t): _median([w["stake_kinds"]["effective"] for w in by_time[t]]) for t in times})
    ch.number("b2_pairs_apart", {str(t): _median([w["stake_kinds"]["pairs_apart"] for w in by_time[t]]) for t in times})
    ch.number("b2_genotypes_effective", {str(t): _median([w["genotypes_effective"] for w in by_time[t]]) for t in times})
    ch.grid("life", [
        {"title": "How far apart two agents stake", "x": {"label": "iteration"},
         "y": {"label": "total variation between stakes", "min": 0}, "series": series},
        {"title": "How many kinds of staking", "x": {"label": "iteration"},
         "y": {"label": "effective number of kinds", "min": 0}, "series": kinds}],
        columns=2, title="Through a B2 world's life",
        caption=("Left: in the same situation and with the same random inputs, how far apart two agents' "
                 "stakes are (the total variation: the share of tokens one would have to move to stake like "
                 "the other), against how far one brain is from itself with other random inputs, and against "
                 "two fresh founders. Right: how many kinds of staking the agents fall into, counted by the "
                 "agents in each (an effective number), beside the same for founders. Medians of ten worlds."),
        recipe=recipe("The ten B2 worlds of Experiment 16, from their archives at iterations 0, 100, 500, 1,000 "
                      "and 2,000.", f"`GraphOfLifeRuns/.pilots/diversity/B2.json` (`{CHECK}`)",
                      METHOD + ["Two brains are of one kind while their stakes differ by no more than 95% of "
                                "brains differ from themselves between two halves of the draws; count the kinds "
                                "by their carriers (1 / sum of squared shares)."]))

    # 3. Genotypes against kinds of behaviour, in every world measured.
    pts = []
    for worlds, label, colour in ((b2["worlds"], "B2", BLUE), (b1_end, "B1", ORANGE),
                                  (long_runs["worlds"], "long runs", GREEN)):
        xs = [w["genotypes_effective"] for w in worlds]
        ys = [w["stake_kinds"]["effective"] for w in worlds]
        if xs:
            pts.append({"label": label, "x": xs, "y": ys, "points": True, "size": 6, "colour": colour, "alpha": 0.8})
    ch.figure("kinds", title="Genotypes and kinds of behaviour",
              x={"label": "effective number of genotypes in the world", "log": True},
              y={"label": "effective number of kinds of staking", "log": True}, series=pts,
              caption=("Each dot is a world at one moment: across, how many genotypes it holds, counted by "
                       "their carriers; up, how many kinds of staking its agents fall into, counted the same way. "
                       "Far more genotypes than kinds means most genetic differences change nothing an agent "
                       "does."),
              recipe=recipe("Every world measured: B2 over its life, B1 at its end, the long pilot worlds.",
                            f"`GraphOfLifeRuns/.pilots/diversity/*.json` (`{CHECK}`)", METHOD))

    # 4. Do kinds last? Against the same moment sampled twice, which is all a sample of 200 can find.
    rows = (persist.get("persist") or []) + (_load("persist_gaps").get("persist") or [])
    if rows:
        # A ceiling belongs to the moment it was taken at: the lab's worlds at 1,000, the long runs at 5,000.
        groups: Dict[Tuple[str, int, int], List[float]] = {}
        for r in rows:
            base = "B1" if "B1-long" in r["from"] else "B2"
            start = _at({"world": r["from"]})
            gap = _at({"world": r["to"]}) - start
            groups.setdefault((base, gap, start if gap == 0 else 0), []).append(r["a_agents_whose_kind_lives_on"])
        keys = sorted(groups, key=lambda k: (k[0] != "B2", k[1], k[2]))
        cats = [f"{base} ceiling {start:,}" if gap == 0 else f"{base} +{gap:,}" for base, gap, start in keys]
        ch.figure("persist", title="Do kinds of behaviour last?",
                  x={"categories": cats}, y={"label": "share of agents whose kind is found again", "min": 0, "max": 1},
                  series=F.dots(cats, [groups[k] for k in keys], [ORANGE if k[0] == "B1" else BLUE for k in keys]),
                  caption=("For two moments of the same world: 200 agents drawn from each, their brains put into the "
                           "situations of the first and grouped into kinds together, over the stake and every other "
                           "decision. The share of the first moment's agents whose kind is found among the second's, "
                           "+N iterations later. "
                           "The *ceiling* draws the second 200 from the same world at the same moment (at iteration "
                           "1,000 in Experiment 16's worlds, 5,000 in the long runs): it is as much as a sample of 200 "
                           "can find, since a rare kind can be missed. Each dot is a world."),
                  recipe=recipe("The ten B2 worlds of Experiment 16 at 1,000 (against themselves, and 100 iterations "
                                "later), 500→1,000 and 1,000→2,000; three of them played on as pilots, at 5,000 against "
                                "themselves, 2,000→5,000 and 5,000→10,000; and B1's two long pilot worlds, at 5,000 "
                                "against themselves, 2,000→5,000 and 5,000→10,000.",
                                f"`GraphOfLifeRuns/.pilots/diversity/persist.json` and `persist_gaps.json` "
                                f"(`{CHECK} --persist`)",
                                METHOD[:2] + ["Draw 200 agents from each moment and put their brains into the first "
                                              "moment's situations and draws.",
                                              "Group the brains of both moments together, over the stake and the "
                                              "five decisions; ask which kinds of the first are found in the second."]))
        for k, c in zip(keys, cats):
            ch.number(f"persist:{c}", {"median": _median(groups[k]), "worlds": len(groups[k]),
                                       "range": [min(groups[k]), max(groups[k])]})

    # Niches.
    for worlds, name in ((b2_end, "B2"), (b1_end, "B1")):
        for key in ("niche_tokens", "niche_degree"):
            vals = [w[key] for w in worlds if w.get(key)]
            if vals:
                ch.number(f"{name}:{key}", {"eta2": _median([v["eta2"] for v in vals]),
                                            "shuffled": _median([v["shuffled"] for v in vals]),
                                            "above": _median([v["above_shuffled"] for v in vals])})
