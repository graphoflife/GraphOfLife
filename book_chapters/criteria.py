"""
What makes a world interesting: the criteria of the screen of random worlds
(research/screen/criteria.py from round 3 on, criteria_1.py for rounds 1 and
2), the lines they are drawn at, and the lines of B2 the first criteria were
measured against (research/screen/reference.json), as tables for the
notebook's page about them. Read straight from the files the judge reads, so
the page and the judge cannot disagree.
"""
from __future__ import annotations

import importlib.util
import json
import os

import book_figures as F
from book_figures import chapter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCREEN = os.path.join(ROOT, "research", "screen")


def _criteria(name: str):
    spec = importlib.util.spec_from_file_location(f"screen_{name}", os.path.join(SCREEN, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@chapter
def criteria(ch: F.Chapter) -> None:
    C, C1 = _criteria("criteria"), _criteria("criteria_1")
    named = {c["key"]: c["name"] for c in C.CRITERIA}
    ch.number("table:criteria", {"head": ["criterion", "what it asks"],
                                 "rows": [[c["name"], c["asks"]] for c in C.CRITERIA]})
    ch.number("table:lines", {"head": ["what is measured", "criterion", "passes", "the sensitivity table tries"],
                              "rows": [[line["name"], named[line["criterion"]], f"{line['passes']} {line['value']:g}",
                                        ", ".join(f"{v:g}" for v in line["tries"])] for line in C.LINES]})
    ch.number("table:reported", {"head": ["number", "what it says"],
                                 "rows": [[r["name"], r["says"]] for r in C.REPORTED]})
    ch.number("table:criteria-1", {"head": ["criterion", "what it asked"],
                                   "rows": [[c["name"], c["asks"]] for c in C1.CRITERIA]})
    with open(os.path.join(SCREEN, "reference.json")) as f:
        ref = json.load(f)
    rows = [("diversity score of B2's median world", "diversity", "{:.2f}"),
            ("effective kinds of staking of B2's median world", "kinds", "{:.1f}"),
            ("Gini of the tokens of B2's most even world", "gini", "{:.2f}")]
    ch.number("table:references", {"head": ["the line", "at iteration 1,000", "at iteration 2,000"],
                                   "rows": [[text, form.format(ref["1000"][key]), form.format(ref["2000"][key])]
                                            for text, key, form in rows]})
    ch.number("rule", {"misses_allowed": C.MISSES_ALLOWED, "misses_allowed_1": C1.MISSES_ALLOWED,
                       "tried_at_least": C.TRIED_AT_LEAST, "settle": C.SETTLE})
