"""
A screen of random worlds, round by round (Experiment 38 is round 1).

Every round's judgement as research/screen/judge.py wrote it
(research/screen/<round>.json): how many drawn worlds died, were too slow or
lived; how many met each criterion at iteration 2,000; the worlds that missed
at most two criteria; and the values the author's rule sets aside, each with
the chance that it would be set aside by luck alone. Nothing is played or read
here but the judgement.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from statistics import median

import book_figures as F
from book_figures import chapter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUNDS = ["E38"]
AT = "2000"


def _judgement(name: str):
    path = os.path.join(ROOT, "research", "screen", f"{name}.json")
    return json.load(open(path)) if os.path.exists(path) else None


def _setting(key: str) -> dict:
    return {"text": f"`{key}`", "href": f"pages/settings.md#{key}"}


@chapter
def screenround(ch: F.Chapter) -> None:
    for name in ROUNDS:
        d = _judgement(name)
        if d is None:
            return
        keys = [c["key"] for c in d["criteria"]]
        drawn = [w for w in d["worlds"] if w["baseline"] != "B2"]
        b2 = [w for w in d["worlds"] if w["baseline"] == "B2"]
        crit = lambda w: w["moments"][AT]["criteria"]
        fates = Counter(w["fate"] for w in drawn)
        alive = [w for w in drawn if crit(w)["alive"]]
        ch.number(f"{name}_fates", {"drawn": len(drawn), **fates,
                                    "died_at_median": median(w["ended_at"] for w in drawn if w["fate"] == "died")})
        ch.number(f"table:{name}_met", {
            "head": ["criterion", "what it asks", f"drawn worlds that meet it (of {len(drawn)})",
                     f"B2's {len(b2)} worlds"],
            "rows": [[c["name"], c["asks"], sum(crit(w)[c["key"]] for w in drawn),
                      sum(crit(w)[c["key"]] for w in b2)] for c in d["criteria"]]})
        misses = lambda w: [c["name"] for c in d["criteria"][1:] if not crit(w)[c["key"]]]
        near = sorted((w for w in alive + b2 if len(misses(w)) <= 2), key=lambda w: (len(misses(w)), w["seed"]))
        ch.number(f"table:{name}_near", {
            "head": ["world", "misses", "agents", "width", "dimension", "diversity score", "Gini",
                     "kinds of staking"],
            "rows": [[(f"B2, seed {w['seed']}" if w["baseline"] == "B2" else f"{w['seed'] - 1000} (seed {w['seed']})"),
                      ", ".join(misses(w)) or "none",
                      f"{w['moments'][AT]['reading']['agents']:,}",
                      f"{w['moments'][AT]['reading']['wide']:.2f}",
                      f"{w['moments'][AT]['reading']['dimension']:.2f}",
                      f"{w['moments'][AT]['reading']['diversity']:.2f}",
                      f"{w['moments'][AT]['reading']['gini_tokens']:.2f}",
                      f"{w['moments'][AT]['kinds']:.1f}"] for w in near]})
        ch.number(f"{name}_misses", dict(Counter(len(misses(w)) for w in alive)))
        ch.number(f"{name}_resilient_space", {f"{r}/{s}": sum(1 for w in alive if crit(w)["resilient"] == r
                                                              and crit(w)["space"] == s)
                                              for r in (True, False) for s in (True, False)})
        marked = [v for v in d["values"] if v["bad"]]
        chance = lambda v: re.search(r"chance ([0-9.]+)", " ".join(v["bad"])).group(1)
        ch.number(f"table:{name}_aside", {
            "head": ["setting", "value set aside", "worlds where it could matter", "of them died",
                     "chance of this by luck"],
            "rows": [[_setting(v["setting"]), str(v["value"]), v["worlds"], v["died"], chance(v)]
                     for v in sorted(marked, key=lambda v: (v["setting"], str(v["value"])))]})
        ch.number(f"{name}_interesting", d["interesting"])
        ch.number(f"{name}_forced_next", d["forced_next"])
