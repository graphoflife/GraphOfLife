"""
What makes a world interesting, as the screen of random worlds judges it from
round 3 on: the author's criteria as they revised them in their walkthrough of
2026-10-24 (research/screen/walkthrough.md). One list, read by the judge
(research/screen/judge.py) and by the notebook's page about it
(book_chapters/criteria.py), so the two cannot say different things. Rounds 1
and 2 were judged by the criteria before these, kept in criteria_1.py.

Every number is an average over iterations 100 to the end of a world's run,
never a reading at one moment (research/screen/measures.py says where each one
comes from). Every line is a number the author controls: each world's value is
shown beside it, and the sensitivity table moves each line alone to show how
many worlds would pass.

A world is **healthy** when it meets every criterion. What makes a world
**interesting** is to be defined with the author once round 3 has run; until
then a world counts as interesting, provisionally, when it is alive and misses
at most two of the other criteria, the author's first answer. A drawn value is
**marked bad** only once it has been tried in at least ten worlds where it
could matter, and every one of them died, or none of them was interesting.

No heavy imports here: the notebook reads this file too.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

#: The iterations every world begins with are ignored: every number is averaged from this one on.
SETTLE = 100

#: Every line: its key, what it measures in words, the criterion it belongs to, the measure (an average)
#: it is drawn on, which side of it passes, its value, and the values the sensitivity table moves it to.
LINES: List[Dict[str, Any]] = [
    {"key": "hub", "name": "links of the best-linked agent, as a share of the others", "criterion": "decentralised", "measure": "hub", "passes": "at most", "value": 0.05,
     "tries": (0.02, 0.05, 0.1, 0.2)},
    {"key": "links_min", "name": "links per agent", "criterion": "decentralised", "measure": "links", "passes": "at least", "value": 3,
     "tries": (2, 3, 4)},
    {"key": "links_max", "name": "links per agent", "criterion": "decentralised", "measure": "links", "passes": "at most", "value": 12,
     "tries": (8, 12, 20)},
    {"key": "core", "name": "share of the agents in the 2-core", "criterion": "resilient", "measure": "core", "passes": "at least", "value": 0.8,
     "tries": (0.6, 0.7, 0.8, 0.9)},
    {"key": "agent_cut", "name": "most of the world one agent's loss cuts off", "criterion": "resilient", "measure": "agent_cut", "passes": "at most", "value": 0.15,
     "tries": (0.02, 0.05, 0.1, 0.15, 0.2, 0.3)},
    {"key": "link_cut", "name": "most of the world one link's loss cuts off", "criterion": "resilient", "measure": "link_cut", "passes": "at most", "value": 0.15,
     "tries": (0.01, 0.05, 0.1, 0.15, 0.2, 0.3)},
    {"key": "width", "name": "typical distance, over the random twin's", "criterion": "space", "measure": "width", "passes": "at least", "value": 1.3,
     "tries": (1.0, 1.1, 1.2, 1.3, 1.5, 2.0)},
    {"key": "dimension_min", "name": "dimension, by how balls grow", "criterion": "space", "measure": "dimension", "passes": "at least", "value": 2.0,
     "tries": (1.5, 2.0, 2.3)},
    {"key": "dimension_max", "name": "dimension, by how balls grow", "criterion": "space", "measure": "dimension", "passes": "at most", "value": 5.0,
     "tries": (4.0, 5.0, 6.0)},
    {"key": "tokens_alike", "name": "share of the agents holding the most common number of tokens", "criterion": "diverse", "measure": "tokens_alike", "passes": "below", "value": 0.8,
     "tries": (0.6, 0.7, 0.8, 0.9, 0.95)},
    {"key": "max_tokens", "name": "the largest holding", "criterion": "diverse", "measure": "max_tokens", "passes": "above", "value": 3,
     "tries": (2, 3, 5, 10)},
    {"key": "core_links_alike", "name": "share of the 2-core's agents with its most common number of links", "criterion": "diverse", "measure": "core_links_alike", "passes": "below",
     "value": 0.8, "tries": (0.6, 0.7, 0.8, 0.9, 0.95)},
    {"key": "ages_alike", "name": "share of the agents of the most common age", "criterion": "diverse", "measure": "ages_alike", "passes": "below", "value": 0.8,
     "tries": (0.6, 0.7, 0.8, 0.9, 0.95)},
    {"key": "genotypes_alike", "name": "share of the agents carrying the most common genotype", "criterion": "diverse", "measure": "genotypes_alike", "passes": "below",
     "value": 0.8, "tries": (0.6, 0.7, 0.8, 0.9, 0.95)},
    {"key": "children_alike", "name": "share of the agents making the more common choice, to have a child or not", "criterion": "diverse", "measure": "children_alike", "passes": "below",
     "value": 0.95, "tries": (0.7, 0.8, 0.9, 0.95, 0.99, 1.0)},
    {"key": "gini", "name": "Gini of the tokens", "criterion": "uneven", "measure": "gini", "passes": "at least", "value": 0.46,
     "tries": (0.3, 0.4, 0.46, 0.5, 0.6)},
]
#: Every line where the judge draws it unless told otherwise.
VALUES = {line["key"]: line["value"] for line in LINES}

#: The criteria, in the order the page lists them: a key, a name, and what it asks, in words made from the lines.
CRITERIA: List[Dict[str, str]] = [
    {"key": "alive", "name": "alive",
     "asks": "played to the end of its run (3,000 iterations in round 3) and never down to 20 agents"},
    {"key": "decentralised", "name": "decentralised",
     "asks": "no agent linked to more than {hub:.0%} of the others, and {links_min:g} to {links_max:g} links per "
             "agent"},
    {"key": "resilient", "name": "resilient",
     "asks": "at least {core:.0%} of the agents in the 2-core, and no single agent's loss cuts off more than "
             "{agent_cut:.0%} of the world, nor a single link's more than {link_cut:.0%} (the rules keep a world in "
             "one piece)"},
    {"key": "space", "name": "almost three-dimensional",
     "asks": "a typical distance at least {width:g} times its random twin's, and a dimension of "
             "{dimension_min:g} to {dimension_max:g}"},
    {"key": "diverse", "name": "diverse",
     "asks": "statistic by statistic: fewer than {tokens_alike:.0%} of the agents hold the same number of tokens, "
             "and some agent holds more than {max_tokens:g}; fewer than {core_links_alike:.0%} of the 2-core's "
             "agents have the same number of links; fewer than {ages_alike:.0%} have the same age, and fewer than "
             "{genotypes_alike:.0%} the same genotype; and fewer than {children_alike:.0%} make the same choice, to "
             "have a child or not"},
    {"key": "uneven", "name": "not too even",
     "asks": "a Gini of the tokens of at least {gini:g} (B2's most even world in Experiment 16)"},
]
for _c in CRITERIA:
    _c["asks"] = _c["asks"].format(**VALUES)
KEYS = [c["key"] for c in CRITERIA]

#: Numbers shown for every world with no line drawn: the author's two new measures of health (2026-10-24),
#: whose lines are to be set together once round 3 has run, and what helps to read the others.
REPORTED: List[Dict[str, str]] = [
    {"measure": "half_cut", "name": "half-cut exponent",
     "says": "the links crossing the cheapest cut into halves found, as a power of the number of agents: 2/3 "
             "for a body of three dimensions, near 0 for a bottleneck, near 1 for a folded small world"},
    {"measure": "half_cut_twin", "name": "half-cut exponent of the random twin",
     "says": "the same for the world's random twin, which is no space at all"},
    {"measure": "apart", "name": "same-position share",
     "says": "of two agents put in the same situation, how often they stake measurably differently"},
    {"measure": "kinds", "name": "effective kinds of staking",
     "says": "how many kinds of staking the agents fall into, counted by their shares"},
    {"measure": "spread_alike", "name": "doctrine alike",
     "says": "the share of agents making the more common choice between spreading their stake and staking it "
             "all on one place"},
    {"measure": "children", "name": "having a child",
     "says": "the share of agents that have a child in an iteration"},
    {"measure": "agents", "name": "agents", "says": "how many agents the world holds"},
]

#: How many of the criteria after "alive" a world may miss and still count as interesting: provisional, the
#: author's first answer, until interesting is defined with them after round 3.
MISSES_ALLOWED = 2
#: How many worlds a value must have been tried in before it can be marked bad.
TRIED_AT_LEAST = 10


def passes(line: Dict[str, Any], value: Optional[float], at: Optional[float] = None) -> bool:
    """Whether an average lies on the passing side of a line, drawn at `at` or else where it is. None never does."""
    if value is None or not math.isfinite(value):
        return False
    bound = line["value"] if at is None else at
    side = line["passes"]
    if side == "at most":
        return value <= bound
    if side == "at least":
        return value >= bound
    if side == "below":
        return value < bound
    if side == "above":
        return value > bound
    raise ValueError(f"a line passes {side!r}?")


def judged(averages: Dict[str, Optional[float]], alive: bool,
           moved: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    A world's verdict from its averages: every line, every criterion, how many of the criteria after
    "alive" it misses, whether it is healthy and whether it is interesting. `moved` draws some lines
    elsewhere, as the sensitivity table does.
    """
    at = {**VALUES, **(moved or {})}
    lines = {line["key"]: passes(line, averages.get(line["measure"]), at[line["key"]]) for line in LINES}
    criteria = {"alive": bool(alive)}
    for c in CRITERIA[1:]:
        criteria[c["key"]] = bool(alive) and all(lines[line["key"]] for line in LINES
                                                 if line["criterion"] == c["key"])
    misses = sum(not criteria[key] for key in KEYS[1:])
    return {"lines": lines, "criteria": criteria, "misses": misses,
            "healthy": all(criteria.values()), "interesting": bool(alive) and misses <= MISSES_ALLOWED}


def sensitivity(worlds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Every line moved alone over the values it tries, the others where they are: how many of the living
    worlds pass it, meet its criterion, are healthy and are interesting. A world is a dict with its
    "averages" and whether it is "alive".
    """
    living = [w for w in worlds if w["alive"]]
    rows = []
    for line in LINES:
        for value in line["tries"]:
            got = [judged(w["averages"], True, {line["key"]: value}) for w in living]
            rows.append({"line": line["key"], "value": value, "drawn": value == line["value"],
                         "pass": sum(g["lines"][line["key"]] for g in got),
                         "criterion": sum(g["criteria"][line["criterion"]] for g in got),
                         "healthy": sum(g["healthy"] for g in got),
                         "interesting": sum(g["interesting"] for g in got)})
    return rows
