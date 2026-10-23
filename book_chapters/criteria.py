"""
What makes a world interesting: the criteria of the screen of random worlds
(research/screen/criteria.py) and the lines of B2 they are measured against
(research/screen/reference.json), as tables for the notebook's page about them.
Read straight from the files the judge reads, so the page and the judge cannot
disagree.
"""
from __future__ import annotations

import importlib.util
import json
import os

import book_figures as F
from book_figures import chapter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCREEN = os.path.join(ROOT, "research", "screen")


def _criteria():
    spec = importlib.util.spec_from_file_location("screen_criteria", os.path.join(SCREEN, "criteria.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@chapter
def criteria(ch: F.Chapter) -> None:
    C = _criteria()
    ch.number("table:criteria", {"head": ["criterion", "what it asks"],
                                 "rows": [[c["name"], c["asks"]] for c in C.CRITERIA]})
    with open(os.path.join(SCREEN, "reference.json")) as f:
        ref = json.load(f)
    rows = [("diversity score of B2's median world", "diversity", "{:.2f}"),
            ("effective kinds of staking of B2's median world", "kinds", "{:.1f}"),
            ("Gini of the tokens of B2's least even world", "gini", "{:.2f}")]
    ch.number("table:references", {"head": ["the line", "at iteration 1,000", "at iteration 2,000"],
                                   "rows": [[text, form.format(ref["1000"][key]), form.format(ref["2000"][key])]
                                            for text, key, form in rows]})
    ch.number("rule", {"misses_allowed": C.MISSES_ALLOWED, "tried_at_least": C.TRIED_AT_LEAST})
