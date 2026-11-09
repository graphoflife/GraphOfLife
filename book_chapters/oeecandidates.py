"""
Candidates for open-ended evolution: every world drawn at random that lived,
read for what evolution needs (research/screen/funnel.py over the dossiers of
research/screen/candidates.py and the families of research/screen/families.py):
the worlds ruled out, each with its reason; the rest ranked in every category;
and the numbers of evolution against each other. Nothing is read here but
research/screen/candidates.json.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict

import book_figures as F
from book_chapters.common import BLUE, YELLOW
from book_figures import chapter, recipe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCREEN = os.path.join(ROOT, "research", "screen")
CANDIDATES = os.path.join(SCREEN, "candidates.json")
ROUND = {"E40": "round 3", "E41": "round 3, weakest first", "E42": "round 3, strongest first"}

RUNS = ("every world of round 3 of the screen of random worlds that lived to the end of its run (Experiments 40, 41 "
        "and 42; 30,000 tokens, 3,000 iterations)")
DATA = ("each world's dossier, `GraphOfLifeRuns/.pilots/screen/candidates/<run>.json` (research/screen/candidates.py: "
        "the family tree from every birth, heredity in two kept worlds, the line of descent played in every kept "
        "world), its families, `.pilots/screen/families/<run>.json` (research/screen/families.py), and its health "
        "(research/screen/health.json), gathered by `research/screen/funnel.py` into `research/screen/candidates.json`")


def label(w: Dict[str, Any]) -> str:
    tag = {"E41": " w", "E42": " s"}.get(w["round"], "")
    return f"s{w['seed']}{tag}"


def link(w: Dict[str, Any]) -> Dict[str, str]:
    return {"text": label(w), "href": f"run:{w['id']}"}


def _f(v: Any, form: str = "{:.2f}") -> str:
    return "—" if v is None else form.format(v)


@chapter
def oeecandidates(ch: F.Chapter) -> None:
    if not os.path.exists(CANDIDATES):
        return
    d = json.load(open(CANDIDATES))
    worlds = d["worlds"]
    gone = [w for w in worlds if w["out"]]
    left = [w for w in worlds if not w["out"]]
    keys = [c["key"] for c in d["categories"]]

    ch.number("counts", {"worlds": len(worlds), "out": len(gone), "left": len(left),
                         "reasons": {r: sum(1 for w in gone if w["out"].startswith(r.split()[0]))
                                     for r in ("died", "a star", "one holding", "no heredity", "standing still")}})
    ch.number("table:out", {
        "head": ["world", "round", "why it is ruled out"],
        "rows": [[link(w), ROUND[w["round"]], w["out"]] for w in sorted(gone, key=lambda w: (w["round"], w["seed"]))]})

    mean_rank = lambda w: sum(w.get("ranks", {}).get(k, len(left)) for k in keys) / len(keys)
    ranked = sorted(left, key=mean_rank)
    ch.number("table:ranks", {
        "head": ["world", "round"] + [c["key"] for c in d["categories"]] + ["mean rank", "channels open"],
        "rows": [[link(w), ROUND[w["round"]]] + [w.get("ranks", {}).get(k, "—") for k in keys]
                 + [f"{mean_rank(w):.1f}", ", ".join(w["capability"]["channels"]) or "none"] for w in ranked]})
    ch.number("table:numbers", {
        "head": ["world", "agents", "generations", "effective population", "heredity", "fertility passed on",
                 "novelty", "contemporaries", "health"],
        "rows": [[link(w), _f(w["numbers"].get("agents"), "{:,.0f}"), _f(w["numbers"].get("generations"), "{:.0f}"),
                  _f(w["numbers"].get("effective_population"), "{:,.0f}"), _f(w["numbers"].get("heredity")),
                  _f(w["numbers"].get("fertility")), _f(w["numbers"].get("late_novelty")),
                  _f(w["numbers"].get("contemporaries")), _f(w["numbers"].get("health"))] for w in ranked]})

    # The shortlist: every candidate among the first of any of health's three orders.
    short = sorted((w for w in left if w.get("shortlisted")), key=lambda w: -w["numbers"]["health"])
    among = {"product": "product", "sum": "sum", "worst": "worst"}
    ch.number("table:shortlist", {
        "head": ["world", "round", f"among the first {d['top']} by", "health (product)", "evolution's mean rank",
                 "heredity", "novelty", "generations", "effective population", "fertility passed on", "channels open"],
        "rows": [[link(w), ROUND[w["round"]], ", ".join(among[o] for o in w["top_of"]),
                  _f(w["numbers"].get("health"), "{:.3f}"), f"{w['evolution_rank']:.1f}",
                  _f(w["numbers"].get("heredity")), _f(w["numbers"].get("late_novelty")),
                  _f(w["numbers"].get("generations"), "{:,.0f}"), _f(w["numbers"].get("effective_population"), "{:,.0f}"),
                  _f(w["numbers"].get("fertility")), ", ".join(w["capability"]["channels"]) or "none"] for w in short]})

    # Heredity against novelty: evolution needs children like their parents, and a line that keeps changing.
    have = [w for w in left
            if w["numbers"].get("heredity") is not None and w["numbers"].get("late_novelty") is not None]
    ch.figure("heredity_novelty", title="Like their parents, yet still changing?",
              x={"label": "heredity index (0: children act as strangers do, 1: as their parent)", "min": 0, "max": 1},
              y={"label": "late novelty of the line of descent", "min": 0},
              series=[{"x": [w["numbers"]["heredity"] for w in have],
                       "y": [w["numbers"]["late_novelty"] for w in have], "points": True, "size": 6,
                       "colour": BLUE, "alpha": 0.85}],
              legend=False,
              caption="Each dot is a candidate world. Across: how much more a child acts like its parent than like a "
                      "stranger, in the same situations. Up: how far the later eras of its line of descent lie from "
                      "every earlier era, above the noise of playing one brain twice. A world in the top right passes "
                      "on what its agents do and keeps changing it.",
              recipe=recipe(RUNS, DATA, ["Read each candidate's heredity index (the mean of its kept worlds read) and "
                                         "late novelty from `candidates.json`."]))

    have = [w for w in left if w["numbers"].get("generations") and w["numbers"].get("effective_population")]
    ch.figure("generations_population", title="How many generations, and how many lines?",
              x={"label": "generations lived by the end", "log": True},
              y={"label": "effective population (logarithmic)", "log": True},
              series=[{"x": [w["numbers"]["generations"] for w in have],
                       "y": [w["numbers"]["effective_population"] for w in have], "points": True, "size": 6,
                       "colour": YELLOW, "alpha": 0.85}],
              legend=False,
              caption="Each dot is a candidate world. Across: the mean generation of the agents alive at the end (how "
                      "many births separate them from the founders). Up: a rough effective population, from how fast "
                      "the ancestry of the living narrows: the smaller it is, the more chance decides which line "
                      "survives, and the larger a difference selection needs before it can see it.",
              recipe=recipe(RUNS, DATA, ["Read each candidate's generations and effective population from "
                                         "`candidates.json` (`numbers`)."]))

    have = [w for w in left if w["numbers"].get("fertility") is not None]
    ch.figure("fertility", title="Is having children passed on?",
              x={"label": "", "categories": ["candidate worlds"]},
              y={"label": "correlation of a parent's and its child's numbers of children, over chance"},
              series=F.dots(["candidate worlds"], [[w["numbers"]["fertility"] for w in have]], [BLUE]),
              guides=[{"axis": "y", "at": 0, "label": "chance"}], legend=False,
              caption="Each dot is a candidate world: across every parent and child both born from iteration 100 to "
                      "two thirds of the run, the correlation between how many children the parent had and how many "
                      "the child had, less the same correlation with the children's counts shuffled among the "
                      "children born in the same iteration. Above zero, families differ in how fertile they are, and "
                      "selection has something to work on.",
              recipe=recipe(RUNS, DATA, ["Read each candidate's `fertility` from `candidates.json` "
                                         "(families.py: correlation less the shuffled mean)."]))
