"""
Round 3 ranked by evolution on the brain-copy tree (research/screen/evolution.py over the readings of
research/screen/braintree.py, selection.py and brainplay.py): the first ten, every world's readings against
their nulls, and what the brain tree changes in the numbers the births-only tree gave. Nothing is read here
but research/screen/evolution.json.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

import book_figures as F
from book_chapters.common import BLUE, GREY, ORANGE, YELLOW
from book_figures import chapter, recipe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVOLUTION = os.path.join(ROOT, "research", "screen", "evolution.json")
ROUND = {"E40": "round 3", "E41": "round 3, weakest first", "E42": "round 3, strongest first"}

RUNS = ("every world of round 3 of the screen of random worlds that lived to the end of its run (Experiments 40, 41 "
        "and 42; 30,000 tokens, 3,000 iterations)")
DATA = ("each world's readings on its brain-copy tree, `GraphOfLifeRuns/.pilots/screen/braintree/<run>.json` "
        "(research/screen/braintree.py: one pass over every frame, births and conquests both copying brains, and "
        "100 shadow worlds of pure luck; selection.py; brainplay.py, brains of the kept worlds played), gathered by "
        "`research/screen/evolution.py` into `research/screen/evolution.json`, with the births-only numbers of "
        "`research/screen/candidates.json` beside them")


def label(w: Dict[str, Any]) -> str:
    return f"s{w['seed']}" + {"E41": " w", "E42": " s"}.get(w["round"], "")


def link(w: Dict[str, Any]) -> Dict[str, str]:
    return {"text": label(w), "href": f"run:{w['id']}"}


def _f(v: Any, form: str = "{:.2f}") -> str:
    return "—" if v is None else form.format(v)


def _median(values: List[Optional[float]]) -> Optional[float]:
    have = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(have)) if have else None


@chapter
def evolutionrank(ch: F.Chapter) -> None:
    if not os.path.exists(EVOLUTION):
        return
    d = json.load(open(EVOLUTION))
    worlds = d["worlds"]
    if not worlds:
        return
    b = lambda w, k: w["beside"].get(k)
    v = lambda w, k: w["values"].get(k)
    ch.number("counts", {"read": d["complete"], "of": d["of"], "provisional": d["provisional"],
                         "shortlisted_in_top": sum(1 for w in worlds[:d["top"]] if w["shortlisted"]),
                         "union": len(d["union"])})
    among = lambda w: ", ".join(o for o, ids in d["bests"].items() if w["id"] in ids) or "—"
    filtered = lambda w: (w.get("filter") or "").split(":")[0] or "—"
    ch.number("table:top", {
        "head": ["rank", "world", "round", "score", "among the first 10 by", "lines beyond luck (z)",
                 "novelty above the null", "selection strength", "heredity", "copies by conquest",
                 "health's rank", "on the shortlist", "health filters it out"],
        "rows": [[str(w["rank"]), link(w), ROUND[w["round"]], f"{w['score']:.3f}", among(w),
                  _f(v(w, "shadow"), "{:.1f}"), _f(v(w, "novelty"), "{:+.3f}"), _f(v(w, "selection"), "{:.3f}"),
                  _f(v(w, "heredity")), _f(w["conquest_share"], "{:.0%}"),
                  str((w["health"].get("ranks") or {}).get("product", "—")), "yes" if w["shortlisted"] else "no",
                  filtered(w)]
                 for w in worlds[:d["top"]]]})
    ch.number("table:all", {
        "head": ["rank", "world", "score", "ranks by product, sum, worst", "lines beyond luck (z)",
                 "novelty above the null", "selection strength", "heredity", "copies by conquest",
                 "generations (births only)", "effective population (births only, shadow)", "on the shortlist",
                 "health filters it out"],
        "rows": [[str(w["rank"]), link(w), f"{w['score']:.3f}",
                  f"{w['ranks']['product']}, {w['ranks']['sum']}, {w['ranks']['worst']}",
                  _f(v(w, "shadow"), "{:.1f}"), _f(v(w, "novelty"), "{:+.3f}"), _f(v(w, "selection"), "{:.3f}"),
                  _f(v(w, "heredity")), _f(w["conquest_share"], "{:.0%}"),
                  f"{_f(b(w, 'generations'), '{:,.0f}')} ({_f(b(w, 'generations_births_only'), '{:,.0f}')})",
                  f"{_f(b(w, 'effective_population'), '{:,.0f}')} ({_f(b(w, 'effective_population_births_only'), '{:,.0f}')}, "
                  f"{_f(b(w, 'effective_population_shadow'), '{:,.0f}')})",
                  "yes" if w["shortlisted"] else "no", filtered(w)] for w in worlds]})
    rows = [("generations by the end", "generations_births_only", "generations", "{:,.0f}"),
            ("births on the way", None, "births_on_path", "{:,.0f}"),
            ("effective population", "effective_population_births_only", "effective_population", "{:,.0f}"),
            ("heredity index", "heredity_births_only", None, "{:.2f}"),
            ("late novelty above noise", "novelty_births_only", "novelty_above_noise", "{:.2f}")]
    table = []
    for name, old, new, form in rows:
        before = _median([b(w, old) for w in worlds]) if old else None
        after = _median([v(w, "heredity") for w in worlds]) if name == "heredity index" else \
            _median([b(w, new) for w in worlds])
        table.append([name, _f(before, form), _f(after, form)])
    table.append(["shadow worlds' effective population", "—",
                  _f(_median([b(w, "effective_population_shadow") for w in worlds]), "{:,.0f}")])
    table.append(["copies made by conquest", "—", _f(_median([w["conquest_share"] for w in worlds]), "{:.0%}")])
    ch.number("table:changed", {"head": ["in the median world", "births-only tree", "brain-copy tree"],
                                "rows": table})
    ch.number("medians", {k: _median([b(w, k) for w in worlds]) for k in worlds[0]["beside"]
                          if not isinstance(worlds[0]["beside"][k], dict)})

    have = [w for w in worlds if b(w, "heredity_births_only") is not None and v(w, "heredity") is not None]
    ch.figure("heredity", title="Is a child's brain like its parent's?",
              x={"label": "heredity index on the births-only tree", "min": 0, "max": 1.2},
              y={"label": "heredity index on the brain-copy tree", "min": 0, "max": 1.2},
              series=[{"x": [0, 1.2], "y": [0, 1.2], "colour": GREY, "dash": [4, 3], "label": "the same on both"},
                      {"x": [b(w, "heredity_births_only") for w in have], "y": [v(w, "heredity") for w in have],
                       "points": True, "size": 6, "colour": BLUE, "alpha": 0.85, "label": "a world"}],
              caption="Each dot is a world of round 3 that lived. Across: the heredity index as the births-only tree "
                      "read it, playing a child against the agent that bore it. Up: the same index on the brain-copy "
                      "tree, playing each brain against the brain it was copied from, by birth or by conquest. 1 "
                      "means a child acts as its parent does, 0 as a stranger does.",
              recipe=recipe(RUNS, DATA, ["Read each world's `heredity_births_only` (`beside`) and its brain-tree "
                                         "heredity (`values`) from `evolution.json`."]))

    have = [w for w in worlds if b(w, "common_ancestor_age") is not None]
    run_length = 3000.0
    ch.figure("ancestry", title="How far back do the living meet one ancestor?",
              x={"label": "iterations back to the common ancestor, on the brain-copy tree", "log": True},
              y={"label": "the same in the shadow worlds of pure luck (median)", "log": True},
              series=[{"x": [10, run_length], "y": [10, run_length], "colour": GREY, "dash": [4, 3],
                       "label": "as far back as luck"},
                      {"x": [b(w, "common_ancestor_age") for w in have],
                       "y": [b(w, "common_ancestor_age_shadow") or run_length for w in have],
                       "points": True, "size": 6, "colour": YELLOW, "alpha": 0.85, "label": "a world"}],
              caption="Each dot is a world of round 3 that lived. Across: how many iterations before the end all its "
                      "last agents' brains descend from one brain. Up: the same in 100 shadow worlds, whose copies "
                      "are made at the same moments into the same living but from parents drawn at random (the "
                      "median; a world whose shadows mostly never meet one ancestor within the run is drawn at "
                      "3,000). A dot above the line: the real ancestry narrows faster than luck would narrow it.",
              recipe=recipe(RUNS, DATA, ["Read each world's `common_ancestor_age` and `common_ancestor_age_shadow` "
                                         "(`beside`) from `evolution.json`."]))

    have = [w for w in worlds if b(w, "novelty_above_noise") is not None
            and b(w, "novelty_null_above_noise") is not None]
    top = max([0.3] + [b(w, "novelty_above_noise") for w in have] + [b(w, "novelty_null_above_noise") for w in have])
    ch.figure("novelty", title="Does the line of descent find more than mutation alone?",
              x={"label": "late novelty of the line of descent, above its noise", "min": 0, "max": top},
              y={"label": "the same for lines that only mutate (mean of six)", "min": 0, "max": top},
              series=[{"x": [0, top], "y": [0, top], "colour": GREY, "dash": [4, 3], "label": "as far as mutation"},
                      {"x": [b(w, "novelty_above_noise") for w in have],
                       "y": [b(w, "novelty_null_above_noise") for w in have],
                       "points": True, "size": 6, "colour": ORANGE, "alpha": 0.85, "label": "a world"}],
              caption="Each dot is a world of round 3 that lived. Across: how far the later eras of its line of "
                      "descent, on the brain-copy tree, lie from every earlier era in the same situations, above the "
                      "noise of playing one brain twice. Up: the same for six lines that start from the line's brain "
                      "of iteration 250 and make as many changes as it made, at random. A dot below the line: the "
                      "real line moved further than mutation alone would take it; above it, less far.",
              recipe=recipe(RUNS, DATA, ["Read each world's `novelty_above_noise` and `novelty_null_above_noise` "
                                         "(`beside`) from `evolution.json`."]))
