"""
What makes a world interesting, as the screen of random worlds judges it (the
author's criteria of 2026-10-22 and 2026-10-23). One list, read by the judge
(research/screen/judge.py) and by the notebook's page about it
(book_chapters/criteria.py), so the two cannot say different things.

A world is read at iteration 2,000 from its archive. It is **healthy** when it
meets every criterion, and **interesting** when it is alive and misses at most
one of the others: the room for variability the author asked for. A drawn value
is **marked bad** only once it has been tried in at least ten worlds where it
could matter, and every one of them died, or none of them was interesting.

No heavy imports here: the notebook reads this file too.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

#: The criteria, in the order the page lists them: a key, a name, and what it asks.
CRITERIA = [
    {"key": "alive", "name": "alive",
     "asks": "more than 20 agents at iteration 2,000, and not stopped as too slow"},
    {"key": "decentralised", "name": "decentralised",
     "asks": "no agent linked to more than 5% of the others, and 3 to 12 links per agent"},
    {"key": "resilient", "name": "resilient",
     "asks": "one piece, with at least 80% of the agents in the 2-core; no single agent's loss cuts off "
             "more than 2% of the world, and no single link's more than 1%"},
    {"key": "space", "name": "almost three-dimensional",
     "asks": "a dimension of 2 to 4, and a typical distance at least 1.5 times its random twin's: a true "
             "space, not a folded small world"},
    {"key": "diverse", "name": "diverse",
     "asks": "tokens, links, ages and genotypes spread more widely than in B2's median world, and its "
             "agents' decisions vary more (effective kinds of staking)"},
    {"key": "uneven", "name": "not too even",
     "asks": "a Gini of the tokens no lower than B2's least even world"},
]
KEYS = [c["key"] for c in CRITERIA]
#: How many of the criteria after "alive" an interesting world may miss.
MISSES_ALLOWED = 1
#: How many worlds a value must have been tried in before it can be marked bad.
TRIED_AT_LEAST = 10
#: The health bar's marks that make up "decentralised" and "resilient" (research/atlas/profile.py).
DECENTRALISED = ("no hub", "links")
RESILIENT = ("one piece", "2-core", "agent cut", "link cut")


def meets(reading: Optional[Dict[str, Any]], kinds: Optional[float], alive: bool,
          ref: Dict[str, float]) -> Dict[str, bool]:
    """Which criteria a world meets at one moment, from its reading (profile.py) and B2's lines there."""
    if not alive or reading is None or reading.get("dead") or reading.get("error"):
        return {key: False for key in KEYS}
    bar = reading.get("bar") or {}
    dim = reading.get("dimension")
    return {
        "alive": True,
        "decentralised": all(bool(bar.get(mark)) for mark in DECENTRALISED),
        "resilient": all(bool(bar.get(mark)) for mark in RESILIENT),
        "space": dim is not None and 2.0 <= dim <= 4.0 and (reading.get("wide") or 0.0) >= 1.5,
        "diverse": (reading.get("diversity") or 0.0) > ref["diversity"] and kinds is not None
        and kinds > ref["kinds"],
        "uneven": (reading.get("gini_tokens") or 0.0) >= ref["gini"],
    }


def healthy(met: Dict[str, bool]) -> bool:
    return all(met.get(key, False) for key in KEYS)


def interesting(met: Dict[str, bool]) -> bool:
    """Alive, and missing at most MISSES_ALLOWED of the other criteria."""
    return bool(met.get("alive")) and sum(not met.get(key, False) for key in KEYS[1:]) <= MISSES_ALLOWED
