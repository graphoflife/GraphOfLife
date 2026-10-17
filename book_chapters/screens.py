# -*- coding: utf-8 -*-
"""
Eight switches at once (pilot batch U, 2026-10-16): a two-level fractional
factorial design of resolution IV over eight of the book's switches on B2,
sixteen worlds of 10,000 tokens, seed 1, 1,000 iterations
(research/oee/pilots/make_jobs_TU.py; the design in book/notes/
fractional-factorial.md).

Every world is read at iteration 1,000 as Meta X reads every world
(research/atlas/at.py → GraphOfLifeRuns/.pilots/atlas/at1000.json); its brains
of 1,000 are played against those of 500 over a lifetime (T = 100;
research/oee/checks/rejudge.py → GraphOfLifeRuns/.pilots/batch_TUV_T100.json).
Each switch's main effect is set beside the same switch thrown alone in B2
(the batch or lab worlds that made it, seeds 1 and 2), and beside the seed's
own noise: the spread of the lab's ten B2 worlds at 1,000.

    eightswitches   Chapter 86 · Eight switches at once (batch U)
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

import book_figures as F
from book_figures import chapter, recipe
from book_chapters.common import GREY, ORANGE, VIOLET
from book_chapters.combos import _contests, _lab_contests, _measured

#: The switches of batch U, by letter: their words, and the worlds that have each alone in B2 and their twins
#: without it, seeds 1 and 2 (ids of at1000.json).
U = {
    "A": ("gifts", ("S/S2-gifts-s{s}", "S/S0-B2-s{s}")),
    "B": ("no revolutions", ("S/S1-norev-s{s}", "S/S0-B2-s{s}")),
    "C": ("unused links lapse", ("S/S4-lapse-s{s}", "S/S0-B2-s{s}")),
    "D": ("no random inputs", ("B2-10000-bad270-s00{s}", "B2-10000-s00{s}")),
    "E": ("estates to newborns", ("R/B2-newborns-s{s}", "P/B2-s{s}")),
    "F": ("links hold a token", ("P/linktoken-s{s}", "P/B2-s{s}")),
    "G": ("brains mixed at birth", ("Q/mix-s{s}", "P/B2-s{s}")),
    "H": ("handover", ("T/handover-s{s}", "P/B2-s{s}")),
}
#: The seven chains of pairs the sixteen worlds cannot tell apart (book/notes/fractional-factorial.md).
CHAINS = [("AB", "CG", "DH", "EF"), ("AC", "BG", "DF", "EH"), ("AD", "BH", "CF", "EG"), ("BC", "AG", "DE", "FH"),
          ("BD", "AH", "CE", "FG"), ("CD", "AF", "BE", "GH"), ("AE", "BF", "CH", "DG")]
#: What is read, its axis label, and how many decimals the chapter quotes.
MEASURES = [("agents", "agents"), ("share_at_most_three", "share with three tokens or fewer"),
            ("diversity", "diversity score (Meta X)"), ("dimension", "dimension by ball growth"),
            ("links_per_agent", "links per agent"), ("marks", "marks of the health bar (of 7)")]
RUNS = ("Batch U's sixteen pilot worlds (`GraphOfLifeRuns/.pilots/U/`): B2 at 10,000 tokens, seed 1, 1,000 "
        "iterations, with each of eight switches on in eight of them (`research/oee/pilots/make_jobs_TU.py`); each "
        "switch alone in B2, seeds 1 and 2, from the batch or lab experiment that made it; and the lab's ten B2 "
        "worlds (Experiment 16) for the seed's own noise.")


def _on(world: str) -> Dict[str, int]:
    """+1 or -1 for each switch, from a world's name (U-ABDH-s1: A, B, D and H on)."""
    letters = world.split("/")[-1].split("-")[1]
    return {k: (1 if letters != "none" and k in letters else -1) for k in "ABCDEFGH"}


def _contrast(ys: np.ndarray, column: np.ndarray) -> float:
    return float(ys[column > 0].mean() - ys[column < 0].mean())


@chapter
def eightswitches(ch: F.Chapter) -> None:
    measured = _measured()
    worlds = sorted(w for w in measured if w.startswith("U/U-"))
    if len(worlds) < 16:
        return
    on = {w: _on(w) for w in worlds}
    cols = {k: np.array([on[w][k] for w in worlds]) for k in "ABCDEFGH"}
    chains = {"+".join(c): cols[c[0][0]] * cols[c[0][1]] for c in CHAINS}
    b2 = [measured[f"B2-10000-s{s:03d}"] for s in range(1, 11) if f"B2-10000-s{s:03d}" in measured]
    letters = list("ABCDEFGH")
    cats = [U[k][0] for k in letters]
    effects: Dict[str, Any] = {}
    charts, pair_charts = [], []
    for key, label in MEASURES:
        ys = np.array([float(measured[w][key]) for w in worlds])
        noise = float(np.std([float(w[key]) for w in b2], ddof=1))
        main = {k: _contrast(ys, cols[k]) for k in letters}
        alone = {}
        for k in letters:
            with_it, without = U[k][1]
            alone[k] = [float(measured[with_it.format(s=s)][key]) - float(measured[without.format(s=s)][key])
                        for s in (1, 2) if with_it.format(s=s) in measured and without.format(s=s) in measured]
        pairs = {name: _contrast(ys, col) for name, col in chains.items()}
        effects[key] = {"main": main, "alone": alone, "pairs": pairs, "seed_sd": noise}
        # The seed's own noise, as a band round zero: one world's spread over the lab's ten B2 worlds.
        band = {"kind": "area", "label": "seed noise", "x": [-0.5, 7.5], "lo": [-noise, -noise],
                "hi": [noise, noise], "colour": GREY, "alpha": 0.25}
        series = [band]
        for i, k in enumerate(letters):
            if alone[k]:
                series.append({"label": "thrown alone in B2 (seeds 1, 2)" if i == 0 else None,
                               "x": [i - 0.15] * len(alone[k]), "y": alone[k], "points": True, "size": 6,
                               "colour": GREY, "alpha": 0.9})
        series.append({"label": "among the sixteen (main effect)", "x": [i + 0.15 for i in range(8)],
                       "y": [main[k] for k in letters], "points": True, "size": 8, "colour": ORANGE})
        guides = [{"axis": "y", "at": 0.0, "label": ""}]
        charts.append({"title": label[0].upper() + label[1:], "x": {"categories": cats},
                       "y": {"label": f"effect on {label}"}, "series": series, "guides": guides})
        names = list(pairs)
        pair_charts.append({"title": label[0].upper() + label[1:], "x": {"categories": names},
                            "y": {"label": f"effect on {label}"},
                            "series": [dict(band, x=[-0.5, len(names) - 0.5]),
                                       {"label": None, "x": list(range(len(names))), "y": [pairs[n] for n in names],
                                        "points": True, "size": 8, "colour": VIOLET}],
                            "guides": guides})
    ch.number("effects", effects)
    ch.number("worlds", {w: {key: measured[w][key] for key, _ in MEASURES} for w in worlds})
    steps = ["Read each world as `research/atlas/profile.py` reads every world.",
             "A switch's main effect: the mean of the eight worlds with it on, minus the mean of the eight with it "
             "off.", "Thrown alone: the world with the switch minus its twin without it, each seed.",
             "The seed noise: the standard deviation of the lab's ten B2 worlds at iteration 1,000."]
    data = ("each world's checkpoint at iteration 1,000, or the lab's whole world kept then "
            "(`research/atlas/at.py --at 1000`, `GraphOfLifeRuns/.pilots/atlas/at1000.json`)")
    ch.grid("effects", charts, columns=1, title="What each switch does, among the sixteen and alone",
            caption=("Orange: each switch's main effect among the sixteen worlds of batch U — the mean of the eight "
                     "worlds with it on, minus the mean of the eight with it off — at iteration 1,000. Grey: the "
                     "same switch thrown alone in B2, the world with it minus its twin without it, seeds 1 and 2. "
                     "The grey band round zero is the seed's own noise: one standard deviation of the measure over "
                     "the lab's ten B2 worlds, twice the noise of a main effect."),
            recipe=recipe(RUNS, data, steps))
    ch.grid("pairs", pair_charts, columns=1, title="Switches that act together",
            caption=("The seven chains of pairs of batch U (book/notes/fractional-factorial.md): for each, the mean "
                     "of the eight worlds where the product of the pair's two switches is +1, minus the mean of the "
                     "other eight — the sum of the four pairs' effects that one chain cannot tell apart. The grey "
                     "band: the seed's own noise, as in the figure above."),
            recipe=recipe(RUNS, data, steps[:1] + ["A chain's effect: the mean of the eight worlds where the "
                                                   "product of a pair's switches is +1, minus the mean of the "
                                                   "other eight."]))

    # Does evolution go on? The brains of 1,000 against those of 500, by switch.
    contests = _contests()
    judged = [w for w in worlds if w in contests]
    if len(judged) == 16:
        ys = np.array([contests[w] for w in worlds])
        main = {k: _contrast(ys, cols[k]) for k in letters}
        pairs = {name: _contrast(ys, col) for name, col in chains.items()}
        b2 = _lab_contests("B2-10000-s")
        noise = float(np.std(b2, ddof=1))
        band = {"kind": "area", "label": "seed noise", "x": [-0.5, 7.5], "lo": [-noise, -noise], "hi": [noise, noise],
                "colour": GREY, "alpha": 0.25}
        ch.number("evolution", {"worlds": {w: contests[w] for w in worlds}, "main": main, "pairs": pairs,
                                "b2_median": float(np.median(b2)), "seed_sd": noise})
        ch.grid("evolution", [
            {"title": "Each world", "x": {"categories": ["sixteen worlds"]},
             "y": {"label": "the later brains' share", "min": 0.2, "max": 1},
             "series": F.dots(["sixteen worlds"], [list(ys)], [ORANGE]),
             "guides": [{"axis": "y", "at": 0.5, "label": "no better"},
                        {"axis": "y", "at": float(np.median(b2)), "label": "B2"}]},
            {"title": "Main effects", "x": {"categories": cats}, "y": {"label": "effect on the later brains' share"},
             "series": [band, {"label": None, "x": list(range(8)), "y": [main[k] for k in letters], "points": True,
                                "size": 8, "colour": ORANGE}],
             "guides": [{"axis": "y", "at": 0.0, "label": ""}]}],
            columns=1, title="Do the brains of 1,000 beat those of 500?",
            caption=("Above: each world of batch U, the share of a contested world its brains of iteration 1,000 "
                     "hold after 100 iterations against those of 500, the mean of two contests, one in each era's "
                     "world; the dashed line marked B2 is the median of the lab's ten B2 worlds, played the same "
                     "way. Below: each switch's main effect on that share; the grey band is the spread of the ten "
                     "B2 worlds (one standard deviation), twice the noise of a main effect."),
            recipe=recipe(RUNS, "`GraphOfLifeRuns/.pilots/batch_TUV_T100.json` and, for B2, `long_steps.json` "
                                "(`research/oee/checks/rejudge.py`)",
                          ["Seat the brains of the two eras on alternate agents of each era's world and play 100 "
                           "iterations.", "Count the agents the later era's brains hold, as a share of all.",
                           "A switch's main effect: the mean of the eight worlds with it on, minus the other eight."]))
