# -*- coding: utf-8 -*-
"""
Part III: Chapter 18 — do the brains matter?
"""
from __future__ import annotations

from collections import Counter

import numpy as np

import book_figures as F
from book_figures import (STATS_FILE, chapter, describe, line, lived, mean_over, per_agent, recipe,
                          rows, runs_of, series, survivors)
from book_chapters.common import BLUE, GREEN, GREY, ORANGE, RED, VIOLET, YELLOW

# ---------------------------------------------------------------------------
# Chapter 18 · Do the brains matter at all?
# ---------------------------------------------------------------------------

E07_RUNS = ("The 90 runs of Experiment 7 — B1 at 10,000 tokens, seeds 1 to 30, 3,000 "
            "iterations — in three conditions: the 30 baseline runs `B1-10000-s001` … `-s030`; "
            "*brains never change*, `B1-10000-a8525d-s001` … `-s030` (`mutation_sparsity` 0); and "
            "*decisions by chance*, `B1-10000-e7f450-s001` … `-s030` (`random_decisions` on). "
            "They are made with `python3 gol_lab.py run E07`; every setting is listed in the box at the "
            "top of the chapter.")


def kind_of(births_per_100: float) -> str:
    """The kinds of world a changeless world falls into, by its births per 100 agents per iteration."""
    if births_per_100 == 0:
        return "frozen"
    if births_per_100 < 0.5:
        return "nearly frozen"
    if births_per_100 > 50:
        return "teeming"
    return "breeding"


@chapter
def brains(ch: F.Chapter) -> None:
    base = survivors(runs_of("E07", "baseline"))
    fixed = runs_of("E07", "brains never change")
    chance = runs_of("E07", "decisions by chance")
    conds = ["brains change (baseline)", "brains never change"]

    def settled(specs, stat, phase=2):
        return [mean_over(s.run_id, stat, 500, 2999, phase) for s in survivors(specs)]

    def births100(s):
        its, rate = per_agent(s.run_id, "births", 1)
        return float(np.nanmean(rate[its >= 500]))

    rates = {s.lab["seed"]: births100(s) for s in survivors(fixed)}
    base_rates = [births100(s) for s in base]
    kinds = {seed: kind_of(r) for seed, r in rates.items()}
    ch.number("kinds", dict(sorted(kinds.items())))
    ch.number("births_per_100", {"baseline": describe(base_rates), "fixed": describe(list(rates.values()))})

    order = ["frozen", "nearly frozen", "breeding", "teeming"]
    kind_colour = {"frozen": VIOLET, "nearly frozen": GREEN, "breeding": YELLOW, "teeming": RED}

    def two_columns(base_values, fixed_by_seed, legend):
        """The baseline's worlds in one column, the changeless worlds in the other, coloured by kind."""
        out = []
        v = np.array([x for x in base_values if x is not None and np.isfinite(x)])
        out.append({"label": "brains change" if legend else None, "x": F.jitter(0, v.size, 7).tolist(),
                    "y": v.tolist(), "points": True, "size": 6, "colour": BLUE, "alpha": 0.85})
        out.append(line([-0.38, 0.38], [float(np.median(v))] * 2, None, BLUE, width=3))
        seeds_ = sorted(fixed_by_seed)
        xs = dict(zip(seeds_, F.jitter(1, len(seeds_), 8).tolist()))
        for kind in order:
            pick = [s_ for s_ in seeds_ if kinds.get(s_) == kind]
            out.append({"label": f"never change: {kind}" if legend else None, "x": [xs[s_] for s_ in pick],
                        "y": [fixed_by_seed[s_] for s_ in pick], "points": True, "size": 6,
                        "colour": kind_colour[kind], "alpha": 0.9})
        out.append(line([0.62, 1.38], [float(np.median(list(fixed_by_seed.values())))] * 2, None, GREY, width=3))
        return out

    panels = []
    for name, stat, phase, title, ylabel, fn, log in (
            ("agents", "nodes", 2, "Agents alive", "agents", None, False),
            ("births", None, 1, "Births per 100 agents", "per iteration (logarithmic)", births100, True),
            ("gini", "gini", 2, "Inequality (Gini)", "Gini coefficient", None, False),
            ("kept", "heldHomeShare", 2, "Nodes kept by their own agent", "share of nodes", None, False)):
        base_values = [fn(s) for s in base] if fn else settled(runs_of("E07", "baseline"), stat, phase)
        fixed_values = {s.lab["seed"]: (fn(s) if fn else mean_over(s.run_id, stat, 500, 2999, phase))
                        for s in survivors(fixed)}
        if log:
            # A frozen world has no births at all, which a logarithmic axis cannot show: it is
            # drawn at the bottom of the axis, 0.001, and the caption says so.
            fixed_values = {k: max(v, 1e-3) for k, v in fixed_values.items()}
        panels.append(dict(title=title, x={"label": "", "categories": ["brains change", "brains never change"]},
                           y={"label": ylabel, "min": 1e-3 if log else 0, "log": log},
                           series=two_columns(base_values, fixed_values, name == "kept"),
                           legend=name == "kept"))
        ch.number(f"dots_{name}", {conds[0]: describe(base_values), conds[1]: describe(list(fixed_values.values()))})
    ch.grid(
        "settled", panels, columns=2, title="Where the worlds settle, with and without change",
        caption="One dot per world that lived to iteration 3,000, at its mean over iterations 500 to "
                "2,999. In each panel, the left column holds the 26 baseline worlds, whose brains "
                "change (blue); the right column the 30 worlds whose brains never change, coloured by "
                "the kind of world each became (see the figure of the four kinds below): violet frozen, "
                "green nearly frozen, yellow breeding, red teeming. The bars are the medians. The "
                "panel of births has a logarithmic axis; the 15 frozen worlds, which had no births at "
                "all, are drawn at its bottom, 0.001. No world deciding by chance lived more than 3 "
                "iterations, so that condition has no dots. The dots are spread sideways only so "
                "that they do not hide each other.",
        recipe=recipe(E07_RUNS, STATS_FILE,
                      ["For agents, the Gini coefficient and nodes kept: average `nodes`, `gini` and "
                       "`heldHomeShare` over the rows with `phase` = 2 and 500 ≤ `iteration` ≤ 2,999.",
                       "For births: divide `births` in each row with `phase` = 1 by the row's "
                       "`nodes_before`, multiply by 100, and average from iteration 500 to 2,999.",
                       "Sort the worlds whose brains never change into the four kinds by their births "
                       "(frozen: none; nearly frozen: below 0.5; teeming: above 50; breeding: between).",
                       "Plot one dot per run that reached iteration 3,000, one column per condition."]))

    k = np.arange(4)
    ch.figure(
        "kinds", title="Four kinds of world", x={"label": "", "categories": order},
        y={"label": "worlds", "min": 0, "max": 30},
        series=[{"label": "brains change (26 worlds)", "kind": "bars", "x0": (k - 0.4).tolist(), "x1": k.tolist(),
                 "y": [sum(1 for r in base_rates if kind_of(r) == o) for o in order], "colour": BLUE},
                {"label": "brains never change (30 worlds)", "kind": "bars", "x0": k.tolist(), "x1": (k + 0.4).tolist(),
                 "y": [sum(1 for v in kinds.values() if v == o) for o in order], "colour": ORANGE}],
        caption="Every world that lived to iteration 3,000, sorted by its mean births per 100 agents "
                "per iteration from iteration 500 on: frozen — none at all; nearly frozen — fewer "
                "than 0.5; breeding — 0.5 to 50; teeming — more than 50, about one child per agent "
                "per iteration.",
        recipe=recipe(E07_RUNS, STATS_FILE,
                      ["Compute each run's births per 100 agents as for the figure of births above.",
                       "Sort the runs into the four kinds by that number and count them."]))

    by_seed = {s.lab["seed"]: s for s in fixed}
    show = [("B1-10000-s001", "Brains change: seed 1"),
            (by_seed[2].run_id, "Never change: seed 2 (frozen)"),
            (by_seed[1].run_id, "Never change: seed 1 (breeding)"),
            (by_seed[25].run_id, "Never change: seed 25 (teeming)"),
            (by_seed[12].run_id, "Never change: seed 12 (teeming)")]
    panels = []
    for i, (run_id, title) in enumerate(show):
        its, n = series(run_id, "nodes")
        bits, b = series(run_id, "births", 1)
        panels.append(dict(title=title, x={"label": "iteration", "min": 0, "max": 3000},
                           y={"label": "per iteration", "min": 0, "max": 4500},
                           series=[line(its, n, "agents after the game", BLUE, width=1),
                                   line(bits, b, "children born", YELLOW, width=1, alpha=0.8)],
                           legend=i == 0))
    ch.grid(
        "traces", panels, columns=2, title="Five single worlds",
        caption="Each panel is one world: the number of agents alive after every game (blue) and of "
                "children born in every reproduction phase (yellow), on the same scale in every panel. "
                "Top left: a baseline world, whose brains change (seed 1). The others are worlds whose "
                "brains never change: seed 2, frozen — no child born after iteration 95, the same "
                "agents game after game; seed 1, breeding; seeds 25 and 12, teeming — about one child "
                "for every agent in every iteration. In seed 12 every child is lost again at once, so "
                "the yellow line lies above the blue.",
        recipe=recipe("`B1-10000-s001` (baseline), and `B1-10000-a8525d-s002`, `-s001`, `-s025` and "
                      "`-s012` (brains never change), all from Experiment 7.", STATS_FILE,
                      ["Plot `nodes` of the rows with `phase` = 2 and `births` of the rows with "
                       "`phase` = 1 against `iteration`."]))

    lines_ = []
    for specs, colour, label in ((runs_of("E07", "baseline"), BLUE, "brains decide (30 worlds)"),
                                 (chance, RED, "decisions by chance (30 worlds)")):
        for j, s in enumerate(specs):
            its, n = series(s.run_id, "nodes")
            keep = its <= 4
            lines_.append(line(its[keep], n[keep], label if j == 0 else None, colour, width=1, alpha=0.6))
    ch.figure(
        "chance", title="The first iterations, with and without the brains", x={"label": "iteration", "min": 0, "max": 4},
        y={"label": "agents after the game (logarithmic)", "log": True, "min": 1},
        series=lines_, guides=[{"axis": "y", "at": 20, "label": "extinct at 20 or fewer"}],
        caption="Each line is one world: the agents alive after the game of iterations 0 to 4. "
                "Blue: the baseline, whose agents decide with their brains; red: the same 30 seeds "
                "with every decision drawn at random. A red line ends where its world died.",
        recipe=recipe(E07_RUNS, STATS_FILE,
                      ["Plot `nodes` of the rows with `phase` = 2 and `iteration` ≤ 4, one line per run."]))
    deaths = [lived(s) for s in chance]
    ch.number("chance_died_at", dict(Counter(deaths)))
    last = []
    for s in chance:
        r = [x for x in rows(s.run_id) if x["phase"] == 2][-1]
        last.append((r.get("orphaned", 0) / r["nodes_before"], r.get("starved", 0) / r["nodes_before"]))
    ch.number("chance_last_game", {"cut_off_median": float(np.median([a for a, _ in last])),
                                   "starved_median": float(np.median([b for _, b in last]))})

    import json
    res = json.load(open(F.os.path.join(F.BOOK, "results", "E07.json")))
    w = res["wandering"]["nodes"]
    seed_of = lambda run_id: int(run_id.split("-s")[-1])
    base_w = list(w["baseline"]["highOverLowBySeed"].values())
    fixed_w = {int(k): v for k, v in w["brains never change"]["highOverLowBySeed"].items()}
    groups = [base_w, list(fixed_w.values())]
    ch.figure(
        "wander-dots", title="How far a world wanders", x={"label": "", "categories": ["brains change", "brains never change"]},
        y={"label": "most crowded ÷ emptiest 100 iterations (logarithmic)", "log": True, "min": 1},
        series=two_columns(base_w, fixed_w, True),
        caption="For each world that lived to the end, coloured as in the figure above: cut its life "
                "from iteration 100 to 2,999 into 29 stretches of 100 iterations, take its mean number "
                "of agents in each, and divide the largest of the 29 by the smallest. 1 means a world "
                "that never moved; 4 that its most crowded stretch held four times as many agents as "
                "its emptiest.",
        recipe=recipe(E07_RUNS, "the analysis of Experiment 7 (`python3 gol_lab.py analyse E07`), whose "
                      "results file holds every run's ratio under `wandering.nodes`; or from " + STATS_FILE,
                      ["Cut iterations 100–2,999 into 29 stretches of 100 and average `nodes` in each.",
                       "Divide the largest of the 29 by the smallest."]))
    ch.number("wander", {c: describe(g) for c, g in zip(conds, groups)})

    lin = res["lineage"]
    base_m = [v["ancestor"]["moves"] for k, v in lin["baseline"]["runs"].items()
              if k in {s.run_id for s in base}]
    fixed_m = {seed_of(k): v["ancestor"]["moves"] for k, v in lin["brains never change"]["runs"].items()}
    groups = [base_m, list(fixed_m.values())]
    ch.figure(
        "moves-dots", title="How often one branch replaces all others",
        x={"label": "", "categories": ["brains change", "brains never change"]},
        y={"label": "moves of the common ancestor after iteration 100", "min": 0},
        series=two_columns(base_m, fixed_m, True),
        caption="For each world that lived to the end, coloured as above: how many times after "
                "iteration 100 the newest common ancestor of all the living moved forward to a younger "
                "genotype ([The common ancestor](../notes/common-ancestor.md)). In a world whose "
                "brains never change, every genotype is a renamed copy of one founder's brain once "
                "that founder's line fills the world, so which branch wins is chance.",
        recipe=recipe(E07_RUNS, "the lineage analysis of Experiment 7, `book/results/E07.json` "
                      "(`lineage.<condition>.runs.<run>.ancestor.moves`)",
                      ["As in Chapter 16, for every run."]))
    ch.number("moves", {c: describe(g) for c, g in zip(conds, groups)})
    by_kind = {}
    for run_id, v in lin["brains never change"]["runs"].items():
        seed_ = int(run_id.split("-s")[-1])
        by_kind.setdefault(kinds.get(seed_, "?"), []).append(v["ancestor"]["moves"])
    ch.number("moves_by_kind", {k: describe(v) for k, v in by_kind.items()})
