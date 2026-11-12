"""
What keeps wealth spread? Planned on 3 October 2026 as an experiment, B1 without
coalitions against B1, and written on 9 November 2026 from worlds already
played: B1 and B2 with and without coalitions (the pilots of batch S, 10,000
tokens, iterations 500 to 1,000), and round 3 of the screen of random worlds
(Experiment 40), where the game's contest, its coalitions and the share-out of
the estates of the dead were drawn at random. Experiments 41 and 42 play eight
of the same draws twice, so their settings are not independent and are left
out of the comparisons, as research/screen/effects.py leaves them out.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import numpy as np

import book_figures as F
import gol_store as store
from book_chapters.common import BLUE, GREEN, GREY, ORANGE, RED, VIOLET, YELLOW
from book_figures import chapter, dots, recipe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCREEN = os.path.join(ROOT, "research", "screen")
HEALTH = os.path.join(SCREEN, "health.json")
EFFECTS = os.path.join(SCREEN, "effects.json")

#: The pilot worlds of batch S read here, as the page names them, and their folders.
SWITCHES = [("B1", "S6-B1"), ("B1, no coalitions", "S7-B1norev"), ("B2", "S0-B2"),
            ("B2, no coalitions", "S1-norev"), ("B2, estates by wealth", "S5-bytokens")]
SEEDS = (1, 2)
#: The settled stretch the pilots are read over, as on Revolutions, handover and gifts.
SETTLED = (500, 1000)
#: Round 3's share-out rules, in the order of the figure, with the page's words and colours.
RULES = [("uniform", "evenly over all", BLUE), ("poorest", "to the poorest", GREEN),
         ("newborns", "to the newborns", YELLOW), ("parents", "to the parents", ORANGE),
         ("children", "to the dead's children", VIOLET), ("local", "to the nearest living", RED)]
#: Round 3's game: the contest, and under the largest stake, coalitions or none.
GAMES = [("largest stake, with coalitions", ORANGE), ("largest stake, no coalitions", RED), ("lottery", BLUE)]
FROM = 100  # round 3 is read from iteration 100 on, as its health is

PILOT_RUNS = ("the pilot worlds of batch S ([Revolutions, handover and gifts](what-the-switches-did-in-pilots.md#revolutions-handover-and-gifts-the-switches-the-world-was-built-with)): "
              "B1 and B2 at 10,000 tokens, each with and without coalitions (`allow_revolutions` off), and B2 with "
              "the estates of the dead shared out by wealth (`redistribution` `by_tokens`), seeds 1 and 2, played "
              "to iteration 1,000")
PILOT_DATA = ("each pilot world's series, `GraphOfLifeRuns/.pilots/S/<world>-s<seed>/series.jsonl`, one row an "
              "iteration after its game (`gini`, `births`, `nodes`, `meanAge`)")
ROUND_RUNS = ("the worlds of round 3 of the screen of random worlds that lived to the end of their 3,000 iterations "
              "(Experiment 40: 30,000 tokens, every setting drawn at random)")
ROUND_DATA = ("each world's averages over iterations 100 to the end, `research/screen/health.json` (from the "
              "screen's readings, `research/screen/measures.py`), and each run's own statistics, "
              "`GraphOfLifeRuns/<run>/stats.jsonl` (`redistributed`, `tokens`)")


def random_gini(mean: float) -> float:
    """The Gini of whole tokens dealt out at random, one at least, at a mean of `mean` an agent (as health.py)."""
    return (mean - 1.0) / (2.0 * mean - 1.0) if mean > 1.0 else 0.0


def _pilot(folder: str, seed: int) -> Dict[str, float]:
    """A pilot world's medians over the settled stretch: inequality, births, age and agents."""
    path = os.path.join(store.BASE_DIR, ".pilots", "S", f"{folder}-s{seed}", "series.jsonl")
    rows = [json.loads(line) for line in open(path)]
    rows = [r for r in rows if SETTLED[0] <= r["it"] <= SETTLED[1]]
    tokens = 10_000
    return {"gini": float(np.median([r["gini"] for r in rows])),
            "relative": float(np.median([r["gini"] / random_gini(tokens / r["nodes"]) for r in rows])),
            "births": float(np.median([100 * r["births"] / max(1, r["nodes"]) for r in rows])),
            "age": float(np.median([r["meanAge"] for r in rows])),
            "agents": float(np.median([r["nodes"] for r in rows]))}


def _shared(run_id: str) -> float:
    """The share of all a world's tokens its cleanups shared out, averaged over its phases from FROM on."""
    path = os.path.join(store.BASE_DIR, run_id, "stats.jsonl")
    shares = []
    for line in open(path):
        r = json.loads(line)
        if r["iteration"] >= FROM and r.get("tokens"):
            shares.append((r.get("redistributed") or 0) / r["tokens"])
    return float(np.mean(shares))


def _game(s: Dict[str, Any]) -> str:
    if s.get("contest", "largest") == "lottery":
        return "lottery"
    return "largest stake, with coalitions" if s.get("allow_revolutions", True) else "largest stake, no coalitions"


def _rank(x: List[float]) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    return ranks


def spearman(a: List[float], b: List[float]) -> float:
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])


def _test(effects: Dict[str, Any], outcome: str, setting: str) -> Dict[str, Any]:
    """research/screen/effects.py's permutation test of one setting on one outcome, among the living."""
    test = next(t for t in effects["tests"] if t["outcome"] == outcome and t["among"] == "living")
    row = next(r for r in test["rows"] if r["setting"] == setting)
    return {"worlds": row["worlds"], "chance": row["chance"], "fdr": row["fdr"],
            "by": {json.loads(k) if k.startswith('"') else k: v for k, v in row["by"].items()}}


@chapter
def wealthspread(ch: F.Chapter) -> None:
    if not (os.path.exists(HEALTH) and os.path.exists(EFFECTS)):
        return

    # B1 and B2, with coalitions and without: the pilots of batch S.
    pilots = {name: [_pilot(folder, s) for s in SEEDS] for name, folder in SWITCHES}
    names = [name for name, _ in SWITCHES]
    colours = [ORANGE, RED, BLUE, VIOLET, GREY]
    panels = [("gini", "Gini coefficient of the tokens", {"min": 0.25, "max": 0.65}),
              ("births", "births per hundred agents an iteration", {"min": 0.5, "max": 40, "log": True}),
              ("age", "mean age of the agents (iterations)", {"min": 4, "max": 200, "log": True})]
    charts = [{"title": title, "x": {"label": "", "categories": names},
               "y": {"label": "", **axis}, "series": dots(names, [[p[key] for p in pilots[n]] for n in names], colours),
               "legend": False, "height": 190}
              for key, title, axis in panels]
    ch.grid("coalitions", charts, columns=1, title="Without coalitions, B1 grows more even, not less",
            caption="One dot per world (seeds 1 and 2), its median over iterations 500 to 1,000; the short bar is "
                    "the middle of the two. Top: the Gini coefficient of the tokens (0 if every agent held the same, "
                    "near 1 if one held all). Middle and bottom, on logarithmic axes: how many agents are born in "
                    "an iteration, per hundred living, and how old the average agent is. \"No coalitions\": "
                    "`allow_revolutions` off, so a node always goes to its largest staker. \"Estates by wealth\": "
                    "the tokens of the dead shared out in proportion to what each survivor holds.",
            recipe=recipe(PILOT_RUNS, PILOT_DATA, [
                "Keep each world's rows from iteration 500 to 1,000.",
                "Births per hundred: 100 · births / agents, row by row. Take the median of each quantity.",
                "One dot per world; the bar is the middle of a pair."]))
    ch.number("pilots", pilots)

    # Round 3: what was drawn, and how unequal each world became.
    d = json.load(open(HEALTH))
    living = [w for w in d["worlds"] if w["round"] == "E40" and w["alive"] and w.get("read")]
    rows = []
    for w in living:
        s = w["set"]
        rows.append({"id": w["id"], "seed": w["seed"], "gini": w["averages"]["gini"],
                     "relative": w["averages"]["gini_random"],
                     "rule": s.get("redistribution", "uniform"), "game": _game(s),
                     "shared": _shared(w["id"]), "agents": w["averages"]["agents"],
                     "mend": bool(s.get("mend_on_death", False)),
                     "culled": w["averages"]["culled_tokens"]})
    effects = json.load(open(EFFECTS))
    tests = {setting: _test(effects, "gini", setting)
             for setting in ("redistribution", "contest", "allow_revolutions", "mend_on_death", "valence",
                             "link_token")}

    rule_names = [words for _, words, _ in RULES]
    by_rule = [[r["gini"] for r in rows if r["rule"] == key] for key, _, _ in RULES]
    by_game = [[r["gini"] for r in rows if r["game"] == g] for g, _ in GAMES]
    by_rule_relative = [[r["relative"] for r in rows if r["rule"] == key] for key, _, _ in RULES]
    axis = {"min": 0.15, "max": 0.8}
    charts = [{"title": "where the estates of the dead go", "x": {"label": "", "categories": rule_names},
               "y": {"label": "Gini coefficient", **axis},
               "series": dots(rule_names, by_rule, [c for _, _, c in RULES]), "legend": False, "height": 220},
              {"title": "the same, against chance", "x": {"label": "", "categories": rule_names},
               "y": {"label": "Gini over a random deal's", "min": 0.3, "max": 2.6},
               "series": dots(rule_names, by_rule_relative, [c for _, _, c in RULES]) + [
                   {"x": [-0.5, len(RULES) - 0.5], "y": [1, 1], "colour": GREY, "dash": [4, 3], "width": 1}],
               "legend": False, "height": 220},
              {"title": "how a node is won", "x": {"label": "", "categories": [g for g, _ in GAMES]},
               "y": {"label": "Gini coefficient", **axis},
               "series": dots([g for g, _ in GAMES], by_game, [c for _, c in GAMES]), "legend": False,
               "height": 220}]
    ch.grid("rules", charts, columns=1, title="In round 3, the share-out of estates sets inequality; coalitions do not",
            caption=f"Each dot is one of the {len(rows)} worlds of Experiment 40 that lived to the end of its run: "
                    "the Gini coefficient of its tokens, averaged from iteration 100 to the end. The short bar is "
                    "the median of a column. Top: the worlds by where their rule sends the tokens of agents that "
                    "die. Middle: the same Gini coefficient over the Gini of the same tokens dealt out at random over "
                    "the same agents (each holding at least one), so that a world of paupers, which cannot be very "
                    "unequal, is not called even for that alone; on the dashed line a world is as unequal as chance "
                    "would make it. Bottom: the worlds' Gini by how their game gives a node away: to the largest stake, with "
                    "or without coalitions of smaller stakers that can outweigh it, or by a lottery weighted by the "
                    "stakes (where coalitions are not drawn). Every other setting of each world was drawn at "
                    "random too, so each column mixes many kinds of world.",
            recipe=recipe(ROUND_RUNS, ROUND_DATA, [
                "Read each living world's average Gini coefficient from `health.json` (`averages.gini`), and its "
                "Gini over a random deal's (`averages.gini_random`: each iteration's Gini over (m − 1)/(2m − 1), "
                "m the mean holding, the Gini of a random deal that leaves every agent at least one token).",
                "Group the worlds by their drawn `redistribution`, and by `contest` and `allow_revolutions`.",
                "One dot per world; the bar is the median."]))

    # How much passes through the share-out, against inequality.
    series = []
    for key, words, colour in RULES:
        mine = [r for r in rows if r["rule"] == key]
        series.append({"label": words, "x": [r["shared"] for r in mine], "y": [r["gini"] for r in mine],
                       "points": True, "size": 6, "colour": colour, "alpha": 0.9})
    shared = [r["shared"] for r in rows]
    ch.figure("shared", title="Where more tokens pass from the dead to the living, worlds tend to be more even",
              x={"label": "share of all tokens shared out by a cleanup (mean over phases)", "log": True,
                 "min": min(shared) / 1.5, "max": max(shared) * 1.5},
              y={"label": "Gini coefficient", **axis}, series=series, legend=True,
              caption="Each dot is a world of Experiment 40 that lived, coloured by its share-out rule. Across, on a "
                      "logarithmic axis: the share of all the world's tokens that a cleanup hands on from the dead to "
                      "the living, averaged over every phase from iteration 100 to the end; it counts the estates of "
                      "every agent that died, culled or starved (the starved hold nothing). Up: the world's Gini "
                      "coefficient, averaged over the same stretch.",
            recipe=recipe(ROUND_RUNS, ROUND_DATA, [
                "For every phase from iteration 100 on: `redistributed` over `tokens` (the cleanup's handed-on "
                "tokens over all of them). Average over the phases.",
                "Plot against the world's average Gini coefficient; colour by `redistribution`."]))

    # The numbers the page quotes.
    wide = [r for r in rows if r["rule"] in ("uniform", "poorest", "newborns", "parents")]
    ch.number("round", {
        "worlds": len(rows),
        "gini": F.describe([r["gini"] for r in rows]),
        "by_rule": {key: {"worlds": len(v), "gini": float(np.median(v)) if v else None}
                    for (key, _, _), v in zip(RULES, by_rule)},
        "relative_by_rule": {key: float(np.median(v)) for (key, _, _), v in zip(RULES, by_rule_relative) if v},
        "relative": F.describe([r["relative"] for r in rows]),
        "r_shared_relative": spearman(shared, [r["relative"] for r in rows]),
        "relative_tests": {setting: _test(effects, "c_uneven", setting)
                           for setting in ("redistribution", "contest", "allow_revolutions")},
        "by_game": {g: {"worlds": len(v), "gini": float(np.median(v)) if v else None}
                    for (g, _), v in zip(GAMES, by_game)},
        "tests": tests,
        "shared": F.describe(shared),
        "shared_by_rule": {key: float(np.median([r["shared"] for r in rows if r["rule"] == key]))
                           for key, _, _ in RULES if any(r["rule"] == key for r in rows)},
        # Rank correlations of the Gini coefficient over the worlds: with the share handed on, over all and among
        # the rules that hand it on widely; with the share culled; with the agents.
        "r_shared": spearman(shared, [r["gini"] for r in rows]),
        "r_shared_wide": spearman([r["shared"] for r in wide], [r["gini"] for r in wide]),
        "wide_worlds": len(wide),
        "r_culled": spearman([r["culled"] for r in rows], [r["gini"] for r in rows]),
        "r_agents": spearman([r["agents"] for r in rows], [r["gini"] for r in rows]),
        "mend": {str(m): {"worlds": sum(r["mend"] == m for r in rows),
                          "gini": float(np.median([r["gini"] for r in rows if r["mend"] == m])),
                          "shared": float(np.median([r["shared"] for r in rows if r["mend"] == m]))}
                 for m in (False, True)},
    })
