"""
Which worlds are healthy? The three orders of research/screen/health.py, for
the author to check (2026-11-07; 2026-11-09: no lines, every measure a score
that is better or worse, and health by the sum, the product and the worst of the
criteria): every world drawn at random in round 3 (Experiments 40 to 42; the
author, 2026-11-08: round 3 only), as a heat map of its criteria, a table with
every number and a link that opens the world in the viewer, the first of each
order, every measure against the worlds, what the worlds lose in their culls,
and the health of each run's two halves. Nothing is read here but health.json.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import book_figures as F
from book_chapters.common import BLUE, GREY, ORANGE
from book_figures import chapter, dots, recipe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCREEN = os.path.join(ROOT, "research", "screen")
HEALTH = os.path.join(SCREEN, "health.json")
#: The criteria as the page names them.
NAMES = {"decentralised": "decentralised", "resilient": "resilient",
         "space": "almost 3-D", "diverse": "diverse", "uneven": "not too even", "culls": "few tokens culled"}
#: The orders as the page names them.
ORDER_NAMES = {"product": "by the product", "sum": "by the sum", "worst": "by the worst criterion"}
#: How far down each order the page lists, as the candidates are drawn from the first 13 of each.
TOP = 13
#: How the rounds' worlds are told apart where their seeds are the same.
TAG = {"E41": "w", "E42": "s"}
#: The page's worked example: first by the sum, far down by the worst criterion.
EXAMPLE = ("E40", 3077)
ROUND = {"E40": "round 3", "E41": "round 3, weakest first", "E42": "round 3, strongest first"}

RUNS = ("every world of round 3 of the screen of random worlds (Experiments 40, 41 and 42: 100 worlds of 30,000 "
        "tokens, played for 3,000 iterations)")
DATA = ("the screen's readings of every world, `GraphOfLifeRuns/.pilots/screen/worlds/<run>.json` "
        "(research/screen/measures.py: every iteration's statistics, every 25th iteration's frame, every kept "
        "world), and each run's own statistics for its culls, scored by `research/screen/health.py` into "
        "`research/screen/health.json`")
SCORING = ["Average every measure of a world over iterations 100 to the end of its run (the judge's averages, "
           "`research/screen/measures.py`).",
           "Score each measure by the share of the living worlds the world does better than (`shares` in "
           "`health.py`): the best 1, the worst 1/82, ties sharing their place.",
           "Each criterion: the geometric mean of its measures' scores. Health: the six criteria multiplied (as "
           "their geometric average), added (as their average), or the worst of them."]


def label(w: Dict[str, Any]) -> str:
    """A world's short name: its seed, and where seeds repeat, its round."""
    tag = TAG.get(w["round"])
    return f"s{w['seed']}" + (f" {tag}" if tag else "")


def world_link(w: Dict[str, Any]) -> Dict[str, str]:
    return {"text": label(w), "href": f"run:{w['id']}"}


def weakest(w: Dict[str, Any]) -> str:
    """The criterion a world scores least in, with its score."""
    c = min(w["criteria"], key=lambda k: w["criteria"][k])
    return f"{NAMES[c]} ({w['criteria'][c]:.2f})"


def _room(values: List[float], log: bool) -> Dict[str, float]:
    """An axis's range holding every value, with a little room around them."""
    lo, hi = min(values), max(values)
    if log:
        return {"min": lo / 1.5, "max": hi * 1.5}
    pad = 0.06 * (hi - lo or abs(hi) or 1.0)
    return {"min": lo - pad, "max": hi + pad}


@chapter
def healthorder(ch: F.Chapter) -> None:
    if not os.path.exists(HEALTH):
        return
    d = json.load(open(HEALTH))
    criteria, measures = d["criteria"], d["measures"]
    worlds = d["worlds"]
    living = [w for w in worlds if w["alive"] and w.get("health")]
    stopped = [w for w in worlds if w.get("stopped")]
    dead = [w for w in worlds if not w["alive"] and not w.get("stopped")]
    rounds = sorted({w["round"] for w in worlds})
    by = {key: sorted(living, key=lambda w: w["ranks"][key]) for key in d["orders"]}

    ch.number("counts", {
        "worlds": len(worlds), "living": len(living), "dead": len(dead), "stopped": len(stopped),
        "by_round": {r: {"worlds": sum(w["round"] == r for w in worlds),
                         "living": sum(w["round"] == r and w["alive"] for w in worlds)} for r in rounds},
        "health": {key: F.describe([w["health"][key] for w in living]) for key in d["orders"]},
        "criteria": {c: F.describe([w["criteria"][c] for w in living]) for c in criteria},
        # The criterion each world scores least in, how often.
        "weakest": {c: sum(min(w["criteria"], key=w["criteria"].get) == c for w in living) for c in criteria},
        "first": {key: [label(w) for w in ws[:TOP]] for key, ws in by.items()},
    })

    # The order by the product, as a heat map: one row a world, one column a criterion; the best at the top.
    rows = list(reversed(by["product"]))
    cells = [(j - 0.5, j + 0.5, i - 0.5, i + 0.5, float(w["criteria"][c]))
             for i, w in enumerate(rows) for j, c in enumerate(criteria)]
    charts = [{"title": "the living worlds, the healthiest by the product at the top",
               "x": {"label": "", "categories": [NAMES[c] for c in criteria]},
               "y": {"label": "", "categories": [f"{w['ranks']['product']}. {label(w)}" for w in rows]},
               "series": [F.cells(cells)], "height": 12 * len(rows), "legend": False,
               "colourbar": {"map": "viridis", "min": 0, "max": 1,
                             "label": "criterion's score (1: better than every other world in it)"}}]
    ch.grid("order", charts, columns=1, title="The living worlds, healthiest first",
            caption=f"Each of the {len(living)} worlds that lived to the end of its run is a row, in the order of "
                    "its health by the product of its six criteria; each criterion is a column, coloured by the "
                    "world's score in it, from dark (worse than nearly every world) to bright yellow (better than "
                    "every other). A world named \"w\" or \"s\" is one of the eight draws played a token at a time, "
                    f"weakest first or strongest first. The {len(dead)} worlds that died are not shown: they "
                    "come last in every order.",
            recipe=recipe(RUNS, DATA, SCORING + ["Rank the living worlds by the product, ties by their weakest "
                                                 "criterion."]))

    # The three orders, as a table: every number, and a link that opens each world.
    ch.number("table:order", {
        "head": ["rank by the product", "world", "round", "by the product", "by the sum", "by the worst",
                 "rank by the sum", "rank by the worst"]
                + [NAMES[c] for c in criteria] + ["agents", "first half → second", "weakest criterion"],
        "rows": [[w["ranks"]["product"], world_link(w), ROUND[w["round"]]]
                 + [f"{w['health'][key]:.3f}" for key in ("product", "sum", "worst")]
                 + [w["ranks"]["sum"], w["ranks"]["worst"]]
                 + [f"{w['criteria'][c]:.2f}" for c in criteria]
                 + [f"{w['averages']['median_agents']:,.0f}",
                    f"{w['halves'][0]['product']:.3f} → {w['halves'][1]['product']:.3f}", weakest(w)]
                 for w in by["product"]]})
    ch.number("table:dead", {
        "head": ["world", "round", "died at iteration"],
        "rows": [[world_link(w), ROUND[w["round"]], f"{w['ended_at']:,}"] for w in dead]
                + [[world_link(w), ROUND[w["round"]], f"stopped as too slow at {w['ended_at']:,}"] for w in stopped]})
    # The first of each order, side by side.
    ch.number("table:first", {
        "head": ["place"] + [ORDER_NAMES[key] for key in d["orders"]],
        "rows": [[i + 1] + [world_link(by[key][i]) for key in d["orders"]] for i in range(TOP)]})

    # Every measure against the living worlds: one chart a measure.
    charts = []
    for m in measures:
        have = [v for v in (w["averages"].get(m["key"]) for w in living) if v is not None]
        if not have:
            continue
        log = m["key"] in ("hub", "richest", "agent_cut", "link_cut", "culled_tokens") and min(have) > 0
        better = {"more": "more is better", "less": "less is better", "near 3": "nearer 3 is better"}[m["better"]]
        charts.append({"title": f"{m['key']} ({NAMES[m['criterion']]}): {better}",
                       "x": {"label": "", "categories": [""]},
                       "y": {"label": m["key"], "log": log, **_room(have, log)},
                       "series": dots([""], [have], [BLUE]), "legend": False, "height": 120})
    ch.grid("measures", charts, columns=3, title="Every measure, against the living worlds",
            caption="One chart a measure: each dot is a living world's average of it, from iteration 100 to the end "
                    "of its run; the short bar is the median. The title says which criterion the measure belongs "
                    "to and which way is better. A world's score in a measure is the share of the other worlds it "
                    "does better than, so only the order of the dots counts. The dots are spread sideways only so "
                    "that they do not hide each other.",
            recipe=recipe(RUNS, DATA, ["Read each living world's averages from `health.json`.",
                                       "One dot per world and measure."]))

    # What the worlds lose in their culls: agents that hold little are leaves, a share of the tokens a branch.
    shares = [v for w in living for v in (w["averages"]["culled_agents"], w["averages"]["culled_tokens"]) if v]
    span = _room(shares, True)
    ch.figure("culls", title="Leaves or branches: what the worlds lose in their culls",
              x={"label": "share of the agents culled in a game", "log": True, **span},
              y={"label": "share of all tokens culled in a game", "log": True, **span},
              series=[{"x": [span["min"], span["max"]], "y": [span["min"], span["max"]], "colour": GREY,
                       "dash": [4, 3], "width": 1},
                      {"x": [w["averages"]["culled_agents"] for w in living],
                       "y": [w["averages"]["culled_tokens"] for w in living], "points": True, "size": 6,
                       "colour": BLUE, "alpha": 0.85}],
              legend=False,
              caption="Each dot is a living world, averaged over the games from iteration 100 to the end of its run. "
                      "Across, the share of its agents a game culls (cuts off from the largest piece of the network "
                      "and removes); up, the share of all the world's tokens those agents held. On the dashed "
                      "diagonal the culled hold as much as the average agent does. Below it they hold less: leaves, "
                      "poor agents cut off at the edge. On it and above, they are whole branches, with all they "
                      "held. Agents that starved are not counted: they hold nothing.",
              recipe=recipe(RUNS, "each run's own statistics, `GraphOfLifeRuns/<run>/stats.jsonl` (the game phase's "
                                  "cleanup: `orphaned`, `redistributed`, `nodes`, `starved`), read by "
                                  "`research/screen/health.py` into `research/screen/health.json`", [
                  "For every game: the agents culled over the agents just before the cleanup (those left, the "
                  "starved and the culled), and the tokens shared out by the cleanup over all the world's tokens "
                  "(the starved hold none, so every token shared out was a culled agent's).",
                  "Average both over the games from iteration 100 to the end of the run."]))
    cut = sorted(living, key=lambda w: w["averages"]["culled_tokens"])
    ch.number("culls", {
        "tokens": F.describe([w["averages"]["culled_tokens"] for w in living]),
        "agents": F.describe([w["averages"]["culled_agents"] for w in living]),
        "worlds": {label(w): {"tokens": w["averages"]["culled_tokens"], "agents": w["averages"]["culled_agents"],
                              "score": w["criteria"]["culls"]} for w in living},
        "least": [label(w) for w in cut[:5]], "most": [label(w) for w in cut[-5:]]})

    # The two halves of each run: did a world grow healthier or sicker?
    first = [w["halves"][0]["product"] for w in living]
    second = [w["halves"][1]["product"] for w in living]
    ch.figure("halves", title="Healthier or sicker with time?",
              x={"label": "health over the first half of the run", "min": 0, "max": 1},
              y={"label": "health over the second half", "min": 0, "max": 1},
              series=[{"x": [0, 1], "y": [0, 1], "colour": GREY, "dash": [4, 3], "width": 1},
                      {"x": first, "y": second, "points": True, "size": 6, "colour": ORANGE, "alpha": 0.8}],
              legend=False,
              caption="Each dot is a living world: its health by the product, scored over the first half of its run "
                      "(iterations 100 to the middle) and over the second (the middle to the end), every half "
                      "against every half of every world. Above the dashed diagonal a world grew healthier; below "
                      "it, sicker.",
              recipe=recipe(RUNS, DATA, ["Average every measure of each half of every living world's run.",
                                         "Score all the halves together, as the whole runs are scored."]))
    ch.number("halves", {"better": sum(b > a + 0.02 for a, b in zip(first, second)),
                         "worse": sum(b < a - 0.02 for a, b in zip(first, second)),
                         "change": F.describe([b - a for a, b in zip(first, second)])})

    # What the half-cut exponent reads for random networks of known shape, at the worlds' densities.
    ch.number("table:calibration", {
        "head": ["random network in", "links an agent", "agents in its largest piece", "links cut to halve it",
                 "half-cut exponent", "a lattice's"],
        "rows": [["a plane" if c["dimensions"] == 2 else "a cube", f"{c['links']:.1f}", f"{c['agents']:,}",
                  c["crossing"], f"{c['exponent']:.2f}", f"{c['lattice']:.2f}"] for c in d["calibration"]]})

    # The measures of each criterion, and those left out.
    better = {"more": "more", "less": "less", "near 3": "nearer three"}
    ch.number("table:measures", {
        "head": ["criterion", "measure", "better"],
        "rows": [[NAMES[m["criterion"]], m["name"], better[m["better"]]] for m in measures]})
    ch.number("table:dropped", {
        "head": ["left out", "why"], "rows": [[k, why] for k, why in d["dropped"].items()]})

    # One world's scores, as a worked example: first by the sum, far down by its worst criterion.
    example = next(w for w in living if w["round"] == EXAMPLE[0] and w["seed"] == EXAMPLE[1])
    ch.number("example", {"world": label(example), "id": example["id"], "ranks": example["ranks"],
                          "health": example["health"], "criteria": example["criteria"],
                          "measures": {m["key"]: {"value": example["averages"].get(m["key"]),
                                                  "score": example["measures"][m["key"]]} for m in measures}})
