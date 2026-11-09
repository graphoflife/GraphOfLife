"""
Which worlds are healthy? The proposed order of research/screen/health.py, for
the author to check (2026-11-07): every world drawn at random in round 3
(Experiments 40 to 42; the author, 2026-11-08: round 3 only), most healthy first, as a
heat map of its criteria, a table with every number and a link that opens the
world in the viewer, every line of the proposal against the worlds, the pieces
the worlds lose at once, and the health of each run's two halves. Nothing is
read here but health.json.
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
         "space": "almost 3-D", "diverse": "diverse", "uneven": "not too even", "culls": "no big culls"}
#: How often, every hundred iterations, a big piece might be allowed, for the table that moves the culls line.
CULL_LINES = (0.5, 1.0, 2.0, 5.0)
#: How the rounds' worlds are told apart where their seeds are the same.
TAG = {"E41": "w", "E42": "s"}
#: The page's worked example: decentralised, resilient, diverse and uneven, and folded.
EXAMPLE = ("E40", 3077)
ROUND = {"E40": "round 3", "E41": "round 3, weakest first", "E42": "round 3, strongest first"}

RUNS = ("every world of round 3 of the screen of random worlds (Experiments 40, 41 and 42: 100 worlds of 30,000 "
        "tokens, played for 3,000 iterations)")
DATA = ("the screen's readings of every world, `GraphOfLifeRuns/.pilots/screen/worlds/<run>.json` "
        "(research/screen/measures.py: every iteration's statistics, every 25th iteration's frame, every kept "
        "world), scored by `research/screen/health.py` into `research/screen/health.json`")


def label(w: Dict[str, Any]) -> str:
    """A world's short name: its seed, and where seeds repeat, its round."""
    tag = TAG.get(w["round"])
    return f"s{w['seed']}" + (f" {tag}" if tag else "")


def world_link(w: Dict[str, Any]) -> Dict[str, str]:
    return {"text": label(w), "href": f"run:{w['id']}"}


def weakest(w: Dict[str, Any], lines: List[Dict[str, Any]]) -> str:
    """The line a world is furthest from meeting, in words, or none."""
    got = w["proposal"]["lines"]
    key = min(got, key=lambda k: got[k])
    if got[key] >= 1.0:
        return "none"
    line = next(line for line in lines if line["key"] == key)
    return f"{line['key']} ({got[key]:.2f})"


def _room(values: List[float], log: bool) -> Dict[str, float]:
    """An axis's range holding every value and the line, with a little room around them."""
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
    criteria, lines = d["criteria"], d["lines"]
    worlds = d["worlds"]
    living = [w for w in worlds if w["alive"] and w.get("proposal")]
    stopped = [w for w in worlds if w.get("stopped")]
    dead = [w for w in worlds if not w["alive"] and not w.get("stopped")]
    rounds = sorted({w["round"] for w in worlds})

    ch.number("counts", {
        "worlds": len(worlds), "living": len(living), "dead": len(dead), "stopped": len(stopped),
        "by_round": {r: {"worlds": sum(w["round"] == r for w in worlds),
                         "living": sum(w["round"] == r and w["alive"] for w in worlds)} for r in rounds},
        "healthy": sum(w["proposal"]["met"] == w["proposal"]["of"] for w in living),
        "healthy_author": sum(w["author"]["met"] == w["author"]["of"] for w in living),
        "met": {str(k): sum(w["proposal"]["met"] == k for w in living) for k in range(len(criteria) + 1)},
        "met_author": {str(k): sum(w["author"]["met"] == k for w in living) for k in range(6)},
        "criterion_met": {c: sum(w["proposal"]["criteria"][c] >= 1.0 for w in living) for c in criteria},
        "health": F.describe([w["proposal"]["health"] for w in living]),
    })

    # The order, as a heat map: one row a world, one column a criterion; the best at the top.
    rows = list(reversed(living))
    cells = [(j - 0.5, j + 0.5, i - 0.5, i + 0.5, float(w["proposal"]["criteria"][c]))
             for i, w in enumerate(rows) for j, c in enumerate(criteria)]
    charts = [{"title": "the living worlds, most healthy at the top",
               "x": {"label": "", "categories": [NAMES[c] for c in criteria]},
               "y": {"label": "", "categories": [f"{w['rank']}. {label(w)}" for w in rows]},
               "series": [F.cells(cells)], "height": 12 * len(rows), "legend": False,
               "colourbar": {"map": "viridis", "min": 0, "max": 1,
                             "label": "share of what the criterion asks (1: met)"}}]
    ch.grid("order", charts, columns=1, title="The living worlds, most healthy first",
            caption=f"Each of the {len(living)} worlds that lived to the end of its run is a row, ranked by its "
                    "health, the sum of its criteria's scores; each criterion is a column, coloured by how "
                    "much of what it asks the world has, from none (dark) to all of it (bright yellow): a criterion "
                    "is met as far as its weakest line. A world named \"w\" or \"s\" is one of the eight draws played "
                    "a token at a time, weakest first or strongest first. The "
                    f"{len(dead)} worlds that died are not shown: they come last in the order.",
            recipe=recipe(RUNS, DATA, [
                "Average every number of a world over iterations 100 to the end of its run (the judge's averages, "
                "`research/screen/measures.py`).",
                "For each line, the share of what it asks that the world has, counted from the line's floor "
                "(`progress` in `health.py`), at most 1.",
                "Each criterion: its weakest line's share. Health: the sum over the criteria.",
                "Rank the living worlds by health, ties by their weakest criterion."]))

    # The order, as a table: every number, and a link that opens each world.
    ch.number("table:order", {
        "head": ["rank", "world", "round", f"health (of {len(criteria)})", "rank by the author's lines",
                 "rank if the weakest criterion decides"]
                + [NAMES[c] for c in criteria] + ["agents", "first half → second", "weakest line"],
        "rows": [[w["rank"], world_link(w), ROUND[w["round"]], f"{w['proposal']['health']:.2f}",
                  w["rank_author"], w["rank_weakest"]]
                 + [f"{w['proposal']['criteria'][c]:.2f}" for c in criteria]
                 + [f"{w['averages']['median_agents']:,.0f}", f"{w['halves'][0]:.2f} → {w['halves'][1]:.2f}",
                    weakest(w, lines)] for w in living]})
    ch.number("table:dead", {
        "head": ["world", "round", "died at iteration"],
        "rows": [[world_link(w), ROUND[w["round"]], f"{w['ended_at']:,}"] for w in dead]
                + [[world_link(w), ROUND[w["round"]], f"stopped as too slow at {w['ended_at']:,}"] for w in stopped]})

    # Every line of the proposal against the living worlds: one chart a measure, with each of its lines.
    charts = []
    measures = list(dict.fromkeys(line["measure"] for line in lines))
    for measure in measures:
        mine = [line for line in lines if line["measure"] == measure]
        have = [v for v in (w["averages"].get(measure) for w in living) if v is not None]
        if not have:
            continue
        log = measure in ("hub", "agent_cut", "link_cut", "max_tokens", "richest") and min(have) > 0
        met = sum(all(w["proposal"]["lines"][line["key"]] >= 1.0 for line in mine) for w in living)
        sign = {"at most": "≤", "at least": "≥", "below": "<", "above": ">"}
        rule = ", ".join(f"{sign[line['passes']]} {line['value']:g}" for line in mine)
        charts.append({"title": f"{measure} {rule}: {met} of {len(living)}",
                       "x": {"label": "", "categories": [""]},
                       "y": {"label": measure, "log": log, **_room(have + [line["value"] for line in mine], log)},
                       "series": dots([""], [have], [BLUE]), "legend": False, "height": 120,
                       "guides": [{"axis": "y", "at": line["value"], "label": "line"} for line in mine]})
    ch.grid("lines", charts, columns=3, title="Every line of the proposal, against the living worlds",
            caption="One chart a measure: each dot is a living world's average of it, from iteration 100 to the end "
                    "of its run, and each dashed line is a line drawn on it; the short bar is the median. The title "
                    "says which side passes and how many of the living worlds meet every line on the chart. The "
                    "dots are spread sideways only so that they do not hide each other.",
            recipe=recipe(RUNS, DATA, ["Read each living world's averages from `health.json`.",
                                       "One dot per world and measure; each line's value as a guide."]))

    # What the worlds lose at once: leaves one by one, or whole branches.
    line, big = next(line for line in lines if line["key"] == "big_culls"), d["big_cull"]
    read = [w for w in living if w.get("culls")]
    met = [w for w in read if w["proposal"]["criteria"]["culls"] >= 1.0]
    missed = [w for w in read if w["proposal"]["criteria"]["culls"] < 1.0]
    series = []
    for group, colour, name in ((met, BLUE, "no big culls"), (missed, ORANGE, "big culls")):
        series.append({"label": f"{name} ({len(group)})", "x": [w["culls"]["cut_off"] for w in group],
                       "y": [w["averages"]["big_culls"] for w in group], "points": True, "size": 6,
                       "colour": colour, "alpha": 0.85})
    cut_off = [w["culls"]["cut_off"] for w in read]
    ch.figure("culls", title="Leaves or branches: what the worlds lose at once",
              x={"label": "share of the network a phase culls, all pieces together", "log": True,
                 **_room(cut_off, True)},
              y={"label": f"pieces of more than {big:.0%} culled at once, every 100 iterations", "min": 0},
              series=series, guides=[{"axis": "y", "at": line["value"], "label": "line"}],
              caption="Each dot is a living world. Across, how much of its network an average phase culls (cuts "
                      "off from the largest piece and removes), every piece counted, from leaves culled one by one "
                      f"to whole branches; up, how often a single piece of more than {big:.0%} of the network was "
                      "culled at once, every 100 iterations (iterations 100 to the end of the run). Agents that "
                      "starved are not counted. A world far to the right and low loses much, but in small pieces; "
                      "a world high up loses whole branches. The dashed line is the criterion's line.",
              recipe=recipe(RUNS, "every phase of every living world's recorded frames, read by "
                                  "`research/screen/culls.py` into `GraphOfLifeRuns/.pilots/screen/culls/<run>.json`, "
                                  "and scored by `research/screen/health.py` into `research/screen/health.json`", [
                  "Rebuild the network as each phase left it, just before its cleanup: the frame before, changed as "
                  "the phase's record says (newborns and their links, links handed over, searches, lapsed links).",
                  "Of the agents the cleanup removed, tell those that starved from those culled: an agent linked "
                  "to a survivor, or that held nothing (in a game nothing staked on it; in reproduction what it "
                  "had, less its child, its gifts, its mates' dues and its upkeep), starved; the run's counts "
                  "of the starved and the culled settle the agents the record leaves open, and any still open "
                  "count as culled.",
                  "Group the culled by their links (under mend_on_death, by the piece of the network they were "
                  "cut off in): the pieces culled; the largest counts.",
                  "Across: the share of the network culled in all, averaged over the phases. Up: how many phases "
                  f"culled a piece of more than {big:.0%}, every 100 iterations."]))
    ch.number("culls", {
        "read": len(read), "met": len(met), "none": sum(w["averages"]["big_culls"] == 0 for w in read),
        "big": big, "line": line["value"],
        "cut_off": F.describe(cut_off),
        "big_culls": F.describe([w["averages"]["big_culls"] for w in read]),
        # Counting only the agents surely culled: how many worlds then meet the line, and the most it moves a world.
        "met_surely": sum(w["averages"]["big_culls_surely"] <= line["value"] for w in read),
        "moved": max((w["averages"]["big_culls"] - w["averages"]["big_culls_surely"] for w in read), default=0.0),
        # The phases whose largest piece the record leaves open, as a share of a world's phases, and how wide open.
        "open": F.describe([w["culls"]["open"] / w["culls"]["phases"] for w in read]),
        "widest": max((w["culls"]["widest"] for w in read), default=0.0),
        "contradicted": sum(w["culls"]["contradicted"] for w in read),
        "worlds": {label(w): {"cut_off": w["culls"]["cut_off"], "largest": w["culls"]["largest"],
                              "big_culls": w["averages"]["big_culls"],
                              "big_culls_surely": w["averages"]["big_culls_surely"], "per_100": w["culls"]["per_100"],
                              "score": w["proposal"]["criteria"]["culls"], "rank": w["rank"]} for w in read}})
    # How many worlds would meet the culls criterion for other sizes of a big piece and other lines.
    sizes = sorted(read[0]["culls"]["per_100"], key=float) if read else []
    ch.number("table:culls", {
        "head": ["a big piece is more than"] + [f"at most {n:g} every 100 iterations" for n in CULL_LINES],
        "rows": [[f"{float(size):.0%}" + (" (the line)" if abs(float(size) - big) < 1e-9 else "")]
                 + [sum(w["culls"]["per_100"][size][1] <= n for w in read) for n in CULL_LINES] for size in sizes]})

    # The two halves of each run: did a world grow healthier or sicker?
    first = [w["halves"][0] for w in living]
    second = [w["halves"][1] for w in living]
    ch.figure("halves", title="Healthier or sicker with time?",
              x={"label": "health over the first half of the run", "min": 0, "max": len(criteria)},
              y={"label": "health over the second half", "min": 0, "max": len(criteria)},
              series=[{"x": [0, len(criteria)], "y": [0, len(criteria)], "colour": GREY, "dash": [4, 3], "width": 1},
                      {"x": first, "y": second, "points": True, "size": 6, "colour": BLUE, "alpha": 0.8}],
              legend=False,
              caption="Each dot is a living world: its health scored over the first half of its run (iterations "
                      "100 to the middle) and over the second (the middle to the end). Above the dashed diagonal a "
                      "world grew healthier; below it, sicker.",
              recipe=recipe(RUNS, DATA, ["Score each half of every living world's run as the whole is scored."]))
    ch.number("halves", {"better": sum(b > a + 0.05 for a, b in zip(first, second)),
                         "worse": sum(b < a - 0.05 for a, b in zip(first, second)),
                         "change": F.describe([b - a for a, b in zip(first, second)])})

    # What the half-cut exponent reads for random networks of known shape, at the worlds' densities.
    ch.number("table:calibration", {
        "head": ["random network in", "links an agent", "agents in its largest piece", "links cut to halve it",
                 "half-cut exponent", "a lattice's"],
        "rows": [["a plane" if c["dimensions"] == 2 else "a cube", f"{c['links']:.1f}", f"{c['agents']:,}",
                  c["crossing"], f"{c['exponent']:.2f}", f"{c['lattice']:.2f}"] for c in d["calibration"]]})

    # The lines of the proposal, and what it changes.
    named = {c: NAMES[c] for c in criteria}
    ch.number("table:proposal", {
        "head": ["line", "criterion", "passes", "counted from"],
        "rows": [[line["name"], named[line["criterion"]], f"{line['passes']} {line['value']:g}",
                  f"{d['floor'][line['key']]:g}"] for line in lines]})
    ch.number("table:changes", {
        "head": ["what the proposal changes", "why"],
        "rows": [[f"adds: {p['name']} ({p['passes']} {p['value']:g})", p["why"]] for p in d["proposed"]]
                + [[f"drops: {k}", why] for k, why in d["dropped"].items()]})

    # One world's lines, as a worked example: a world that meets four criteria and has nothing of the fifth.
    example = next(w for w in living if w["round"] == EXAMPLE[0] and w["seed"] == EXAMPLE[1])
    ch.number("example", {"world": label(example), "rank": example["rank"], "id": example["id"],
                          "health": example["proposal"]["health"],
                          "lines": {line["key"]: {"value": example["averages"].get(line["measure"]),
                                                  "line": line["value"], "passes": line["passes"],
                                                  "progress": example["proposal"]["lines"][line["key"]]}
                                    for line in lines}})
