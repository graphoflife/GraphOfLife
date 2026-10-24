"""
Every era against three (Experiment 28).

Before the contests: how much an agent could gain by its best reply to the rest, era by era, in every
long world (research/oee/checks/exploitability.py): whether evolution closes in on an equilibrium.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

import book_figures as F
import gol_plan
import gol_store
import gol_tournament
from book_figures import chapter, recipe
from book_chapters.common import BLUE, GREEN, GREY, ORANGE, RED, VIOLET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOM = os.path.join(ROOT, "GraphOfLifeRuns", ".pilots", "exploitability_eras.json")
#: Each family of long worlds: its label, how its runs are named, its colour.
FAMILIES = [("Experiment 18, seed 5 (mixed at birth, to 50,000)", lambda r: r == "B2-10000-4af4a3-s005", ORANGE),
            ("Experiment 18, seed 6", lambda r: r == "B2-10000-4af4a3-s006", VIOLET),
            ("Experiment 19 (parents as mates), ten worlds", lambda r: "bcc1bc" in r, BLUE),
            ("Experiment 17 (mixed at birth), eight worlds", lambda r: "4af4a3" in r, GREEN),
            ("B2, Experiment 16, ten worlds", lambda r: r.startswith("B2-10000-s"), GREY)]
RUNS = ("The lab's long worlds: Experiment 18's two worlds played to 50,000 iterations "
        "(`B2-10000-4af4a3-s005`, `-s006`, every 2,500), Experiment 19's ten to 20,000 "
        "(`B2-10000-bcc1bc-s001` … `-s010`), Experiment 17's to 10,000 (`B2-10000-4af4a3-s001` … `-s010` but "
        "the two above) and Experiment 16's B2 worlds to 2,000 (`B2-10000-s001` … `-s010`), at the iterations "
        "their archives keep.")


WORLDS = [("B2-10000-4af4a3-s005", "Experiment 18, seed 5"), ("B2-10000-4af4a3-s006", "Experiment 18, seed 6")]
REFS = [(2500, GREY), (10000, BLUE), (25000, ORANGE)]
CONTESTS = ("Experiment 28's 240 contests (`python3 gol_lab.py run E28`): every 2,500th era of Experiment 18's "
            "two worlds played to 50,000 iterations (`B2-10000-4af4a3-s005`, `-s006`) against the eras of 2,500, "
            "10,000 and 25,000, each pair once on each era's world (coin 17), and each of those three eras "
            "against itself four times (coins 17 to 20).")


def _contests() -> List[Dict[str, Any]]:
    """Every contest of Experiment 28 that has a result, with its settings."""
    out = []
    for found in gol_plan.trials_of("E28"):
        path = os.path.join(gol_store.run_dir(gol_plan.trial_id_for(found["trial"])), "result.json")
        if os.path.exists(path):
            with open(path) as f:
                out.append({**found["trial"], **json.load(f)["result"]})
    return out


def _score(m: Dict[Any, float], x: int, y: int) -> float:
    """Era x's share against era y, from the mean share of the later era in each pair."""
    return m[(y, x)] if (y, x) in m else 1.0 - m[(x, y)]


def contests(ch: F.Chapter) -> None:
    """Experiment 28: every era against three, its noise band, the ratings and the chain's test."""
    found = _contests()
    if len(found) < 240:
        return
    logit = lambda p: float(np.log(p / (1 - p)))
    panels, ratings, summary = [], [], {}
    for run, label in WORLDS:
        mine = [r for r in found if r["run"] == run]
        self_ = [r["share"] for r in mine if r["a"] == r["b"]]
        sd = float(np.std(self_, ddof=1))
        band = (0.5 - 2 * sd, 0.5 + 2 * sd)
        pairs: Dict[Any, List[float]] = defaultdict(list)
        for r in mine:
            if r["a"] != r["b"]:
                pairs[(r["a"], r["b"])].append(r["share"])
        m = {k: float(np.mean(v)) for k, v in pairs.items()}
        eras = sorted({e for k in m for e in k})
        series = [{"label": "noise: 0.5 ± twice an era against itself" if not panels else None, "kind": "area",
                   "x": [eras[0], eras[-1]], "lo": [band[0]] * 2, "hi": [band[1]] * 2, "colour": GREY, "alpha": 0.15}]
        against = {}
        for ref, colour in REFS:
            xs = [e for e in eras if e != ref]
            ys = [_score(m, e, ref) for e in xs]
            against[ref] = dict(zip(xs, ys))
            series.append({"label": f"against the brains of {ref:,}" if not panels else None, "x": xs, "y": ys,
                           "colour": colour, "width": 1.6})
        panels.append({"title": label, "x": {"label": "the era's iteration"},
                       "y": {"label": "its share against the fixed era", "min": 0.15, "max": 0.85}, "series": series})
        elo, residual = gol_tournament.ratings(eras, [{"a": a, "b": b, "share": v} for (a, b), v in m.items()])
        ratings.append({"label": label, "x": eras, "y": elo, "colour": ORANGE if run.endswith("5") else VIOLET,
                        "width": 1.8})
        later = [e for e in eras if e > 25000]
        chain = [against[2500][e] - 1 / (1 + np.exp(-(logit(against[25000][e]) + logit(_score(m, 25000, 2500)))))
                 for e in later]
        summary[run] = {
            "self_sd": sd, "band": band,
            "later_vs_25000_median": float(np.median([against[25000][e] for e in later])),
            "beat_2500_beyond_band": sum(against[2500][e] > band[1] for e in eras if e >= 5000),
            "eras_from_5000": sum(1 for e in eras if e >= 5000),
            "chain_deficit_mean": float(np.mean(chain)), "chain_deficit_below": int(np.sum(np.array(chain) < 0)),
            "elo": dict(zip(eras, elo)), "residual": residual,
            "against": {str(ref): {str(e): v for e, v in a.items()} for ref, a in against.items()}}
    ch.number("contests", summary)
    ch.grid("scores", panels, title="Every era against three fixed eras", columns=1,
            caption=("Each line is one fixed era: the share every other era's brains held against its brains, the "
                     "mean of two contests of 100 iterations, one on each era's world. The grey band is 0.5 plus or "
                     "minus twice the spread of the fixed eras played against themselves: a score inside it could "
                     "be noise."),
            recipe=recipe(CONTESTS, "each trial's `result.json`",
                          ["For each pair of eras, the later era's share of the agents after 100 iterations, the mean "
                           "of its two contests; an earlier era's score against a later one is 1 less that.",
                           "The band: the standard deviation of the twelve contests of an era against itself, twice "
                           "on either side of 0.5."]))
    ch.figure("ratings", title="Each era's rating, from every contest",
              x={"label": "the era's iteration"}, y={"label": "rating (Elo points, the era of 2,500 at 0)"},
              series=ratings,
              caption=("One rating per era and world, fitted to every contest of that world by least squares on "
                       "the log-odds of the scores (Bradley–Terry): +100 means the era's brains beat the brains of "
                       "2,500 about 64 times in a hundred."),
              recipe=recipe(CONTESTS, "each trial's `result.json`",
                            ["For each pair of eras, the mean share of the later era over its two contests.",
                             "Fit one rating per era so that the log-odds of every score is the difference of the "
                             "two ratings, the era of 2,500 at 0 (`gol_tournament.ratings`); shown in Elo points."]))


@chapter
def eras(ch: F.Chapter) -> None:
    contests(ch)
    if not os.path.exists(ROOM):
        return
    rows = json.load(open(ROOM))
    series: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}
    for label, member, colour in FAMILIES:
        mine = [r for r in rows if member(r["run"])]
        rows = [r for r in rows if not member(r["run"])]
        by_it: Dict[int, List[float]] = defaultdict(list)
        for r in mine:
            by_it[int(r["iteration"])].append(r["eras-and-contests"])
        its = sorted(by_it)
        med = [float(np.median(by_it[i])) for i in its]
        lo = [float(min(by_it[i])) for i in its]
        hi = [float(max(by_it[i])) for i in its]
        if len({r["run"] for r in mine}) > 1:
            series.append({"label": None, "kind": "area", "x": its, "lo": lo, "hi": hi, "colour": colour,
                           "alpha": 0.18})
        series.append({"label": label, "x": its, "y": med, "colour": colour, "width": 1.6})
        summary[label] = {"iterations": its, "median": med, "lowest": lo, "highest": hi}
    ch.number("room", summary)
    ch.figure("room", title="What a best reply would gain, era by era",
              x={"label": "iteration"}, y={"label": "nodes a game, best reply over own play", "min": 0},
              series=series,
              caption=("Each line is a family of long worlds: for a sample of up to 300 agents of each world "
                       "holding three tokens or more, the nodes it would hold after one game by its best reply "
                       "to the others' play, less what its own play holds, averaged. A shaded band, where a "
                       "family has several worlds, runs from its lowest world to its highest; the line is the "
                       "median. A population at an equilibrium would score 0."),
              recipe=recipe(RUNS, "the worlds the archives keep",
                            ["Load the world at the iteration and play one game, so its messages are fresh.",
                             "Draw the random inputs 8 times and turn every brain's outputs into stakes as the "
                             "engine does: the others' play, as 8 draws.",
                             "For up to 300 agents with three tokens or more, score its own play and the best "
                             "of a set of replies (everything home; the cheapest way to keep its node and take "
                             "the cheapest neighbours; taking without defending) by the nodes it holds after the "
                             "game, averaged over the 8 draws (`python3 research/oee/checks/exploitability.py "
                             "RUN:ITERATION … --draws 8 --sample 300`).",
                             "The gain is the best reply's score less the own play's, averaged over the agents."]))
