# -*- coding: utf-8 -*-
"""
Part II, the life of a world: Chapters 10 to 13 — a world's life, its youth,
births and deaths, and how much the seed decides.
"""
from __future__ import annotations

from typing import List

import numpy as np

import book_figures as F
from book_figures import (FRAMES, STATS_FILE, band_series, chapter, describe, dots, line, lived, mean_over,
                          per_agent, recipe, runs_text, series, surviving_text, survivors)
from book_chapters.common import (BAND_STEPS, BAND_WORDS, BLUE, CYAN, GREEN, GREY, ORANGE, RED,
                                  VIOLET, YELLOW, baseline)

# ---------------------------------------------------------------------------
# Chapter 10 · A world's life
# ---------------------------------------------------------------------------

@chapter
def life(ch: F.Chapter) -> None:
    specs = baseline()
    by_seed = {s.lab["seed"]: s for s in specs}

    # Six single worlds, on one scale, side by side.
    panels = []
    for seed in range(1, 7):
        s = by_seed[seed]
        its, n = series(s.run_id, "nodes")
        died = lived(s) < s.until
        panels.append(dict(
            title=f"Seed {seed}" + (f" — died after {lived(s)} iterations" if died else ""),
            x={"label": "iteration", "min": 0, "max": 3000}, y={"label": "agents", "min": 0, "max": 4000},
            series=[line(its, n, None, BLUE, width=1)], legend=False))
    ch.grid(
        "worlds", panels, columns=2, title="Six worlds, one at a time",
        caption="The number of agents alive after every game in the worlds with seeds 1 to 6, "
                "each in a chart of its own and all on the same scale (0 to 4,000 agents, "
                "iterations 0 to 3,000). Each line has one point per iteration.",
        recipe=recipe("`B1-10000-s001` … `B1-10000-s006`, six of the 30 baseline runs (their "
                      "settings are listed in [Chapter 9](09-thirty-worlds.md)).", STATS_FILE,
                      ["Take the rows with `phase` = 2.",
                       "Plot `nodes` against `iteration`, one point per iteration, joined."]))

    # All thirty: the bands.
    agents = [series(s.run_id, "nodes") for s in specs]
    ch.figure(
        "agents", title="Agents alive, 30 worlds",
        x={"label": "iteration"}, y={"label": "agents", "min": 0},
        series=[band_series(agents, "30 worlds: median, middle half, nine in ten", BLUE)],
        caption=f"{BAND_WORDS}. Four worlds die out; after a world's death it is no longer "
                "counted, so the bands of the last thousand iterations are those of 26 worlds.",
        recipe=recipe(runs_text(), STATS_FILE,
                      ["Take every run's rows with `phase` = 2 and the statistic `nodes`.",
                       *BAND_STEPS]))

    alive = F.A.bands(agents)
    ch.figure(
        "alive", title="Worlds still alive",
        x={"label": "iteration"}, y={"label": "worlds", "min": 0, "max": 31},
        series=[line(alive["x"], alive["alive"], None, GREY, width=2)], legend=False,
        caption="How many of the 30 worlds are still alive. A world dies when an iteration ends "
                "with 20 agents or fewer; four did, after 4, 313, 1,547 and 2,184 iterations.",
        recipe=recipe(runs_text(), STATS_FILE,
                      ["For every stretch of 5 iterations, count the worlds that have a row "
                       "with `phase` = 2 in it."]))

    # Early against late, world by world.
    early = {s.lab["seed"]: mean_over(s.run_id, "nodes", 100, 200) for s in specs if lived(s) >= 200}
    late = {s.lab["seed"]: mean_over(s.run_id, "nodes", 2800, 2999) for s in survivors(specs)}
    both = sorted(set(early) & set(late))
    pairs = []
    for seed in both:
        ratio = late[seed] / early[seed]
        colour = GREEN if ratio > 1.2 else RED if ratio < 0.8 else GREY
        pairs.append(line([0, 1], [early[seed], late[seed]], None, colour, width=1.2, alpha=0.8))
    pairs.append(line([0, 1], [np.median([early[s] for s in both]), np.median([late[s] for s in both])],
                      "median of the 26 worlds", BLUE, width=3.5))
    ch.figure(
        "early-late", title="Each world early and late",
        x={"label": "", "categories": ["iterations 100–200", "iterations 2,800–2,999"]},
        y={"label": "mean number of agents", "min": 0},
        series=pairs,
        caption="One line per surviving world, from its mean number of agents over iterations "
                "100–200 (left) to its mean over iterations 2,800–2,999 (right). Green: the "
                "late mean is more than 1.2 times the early one; red: less than 0.8 times; grey: "
                "in between. The thick blue line joins the medians of the 26 worlds.",
        recipe=recipe(surviving_text(), STATS_FILE,
                      ["For each run that reached iteration 3,000, take the rows with `phase` = 2.",
                       "Average `nodes` over the rows with 100 ≤ `iteration` ≤ 200, and again "
                       "over 2,800 ≤ `iteration` ≤ 2,999.",
                       "Draw one line from the first mean to the second; colour it by their ratio.",
                       "Join the two medians over the 26 runs."]))
    ratios = np.array([late[s] / early[s] for s in both])
    ch.number("early_late", {
        "worlds": len(both),
        "median_early": float(np.median([early[s] for s in both])),
        "median_late": float(np.median([late[s] for s in both])),
        "median_early_all_alive_at_200": float(np.median(list(early.values()))),
        "within_20pc": int(np.sum(np.abs(ratios - 1) <= 0.2)),
        "up_more_than_20pc": int(np.sum(ratios > 1.2)),
        "down_more_than_20pc": int(np.sum(ratios < 0.8)),
        "ratio_min": float(ratios.min()), "ratio_max": float(ratios.max())})

    # Connections, and connections per agent.
    ch.figure(
        "connections", title="Connections, 30 worlds",
        x={"label": "iteration"}, y={"label": "connections", "min": 0},
        series=[band_series([series(s.run_id, "edges") for s in specs],
                            "30 worlds: median, middle half, nine in ten", ORANGE)],
        caption=f"The number of connections after every game. {BAND_WORDS}.",
        recipe=recipe(runs_text(), STATS_FILE,
                      ["Take every run's rows with `phase` = 2 and the statistic `edges`.",
                       *BAND_STEPS]))

    # The dead.
    dead = [s for s in specs if lived(s) < s.until]
    traces = []
    for k, s in enumerate(dead):
        its, n = series(s.run_id, "nodes")
        traces.append(line(its, n, f"seed {s.lab['seed']} (died after {lived(s)} iterations)", [RED, YELLOW, VIOLET, CYAN][k],
                           width=1.2))
    ch.figure(
        "dead", title="The four worlds that died",
        x={"label": "iteration"}, y={"label": "agents (logarithmic)", "log": True, "min": 10},
        series=traces,
        guides=[{"axis": "y", "at": 20, "label": "extinct at 20 or fewer"}],
        caption="The number of agents after every game in the four worlds that died, on a "
                "logarithmic axis, so that a fall from 1,000 to 100 looks as long as one from "
                "100 to 10. A world stops when an iteration ends with 20 agents or fewer.",
        recipe=recipe("`B1-10000-s014`, `-s020`, `-s023` and `-s024`, the four of the 30 "
                      "baseline runs that died out.", STATS_FILE,
                      ["Take the rows with `phase` = 2 and plot `nodes` against `iteration` "
                       "on a logarithmic axis."]))
    for s in dead:
        its, n = series(s.run_id, "nodes")
        below100 = next((int(t) for t, v in zip(its, n) if v < 100 and np.all(n[its >= t] < 100)), None)
        ch.number(f"dead_s{s.lab['seed']:02d}", {"died": lived(s), "below_100_for_good_from": below100,
                                                 "last_five": n[-5:].tolist()})

    # Window numbers for the text.
    for stat, phase in (("nodes", 2), ("edges", 2), ("births", 1), ("starved", 2), ("orphaned", 2)):
        e = describe([mean_over(s.run_id, stat, 100, 200, phase) for s in specs if lived(s) >= 200])
        l = describe([mean_over(s.run_id, stat, 2800, 2999, phase) for s in survivors(specs)])
        ch.number(f"window_{stat}", {"early": e, "late": l})


# ---------------------------------------------------------------------------
# Chapter 11 · The first hundred iterations
# ---------------------------------------------------------------------------

@chapter
def youth(ch: F.Chapter) -> None:
    specs = baseline()
    early = lambda its, v, stop=150: (its[its <= stop], v[its <= stop])

    traces = []
    for s in specs:
        its, n = early(*series(s.run_id, "nodes"))
        traces.append(line(its, n, None, BLUE, width=0.8, alpha=0.45))
    med = F.A.bands([early(*series(s.run_id, "nodes")) for s in specs], points=151)
    traces.append(line(med["x"], med["y"], "median of the worlds alive", YELLOW, width=3))
    ch.figure(
        "boom", title="Agents in the first 150 iterations",
        x={"label": "iteration", "min": 0, "max": 150}, y={"label": "agents (logarithmic)", "log": True, "min": 5},
        series=traces,
        caption="Each thin blue line is one of the 30 worlds: the number of agents alive after "
                "every game. The yellow line is their median. The axis is logarithmic, so equal "
                "heights are equal factors: the step from 100 to 1,000 is as tall as the one from "
                "1,000 to 10,000.",
        recipe=recipe(runs_text(), STATS_FILE,
                      ["Take every run's rows with `phase` = 2 and 0 ≤ `iteration` ≤ 150.",
                       "Draw `nodes` against `iteration` for each run.",
                       "At each iteration, take the median of `nodes` over the runs that have a row there."]))
    peak_iter = int(med["x"][int(np.nanargmax(med["y"]))])
    ch.number("median_peak", {"agents": float(np.nanmax(med["y"])), "at": peak_iter})
    for t in (0, 5, 10, 20, 50, 100, 150):
        k = int(np.argmin(np.abs(np.array(med["x"]) - t)))
        ch.number(f"median_agents_{t}", float(med["y"][k]))

    flows = [("births", 1, "born", YELLOW), ("orphaned", 1, "cut off in reproduction", ORANGE),
             ("starved", 2, "starved in the game", RED), ("orphaned", 2, "cut off in the game", VIOLET)]
    out = []
    for stat, phase, label, colour in flows:
        b = F.A.bands([early(*series(s.run_id, stat, phase)) for s in specs], points=151)
        out.append(line(b["x"], b["y"], label, colour, width=2))
        ch.number(f"flow_{stat}_{phase}_peak", {"median": float(np.nanmax(b["y"])),
                                                "at": float(b["x"][int(np.nanargmax(b["y"]))])})
    ch.figure(
        "flows", title="Births and deaths in the first 150 iterations",
        x={"label": "iteration", "min": 0, "max": 150}, y={"label": "agents per iteration", "min": 0},
        series=out,
        caption="For each iteration, the median over the worlds of: the children born in the "
                "reproduction phase (yellow); the agents removed by the cleanup of that phase "
                "because they were joined to no one or to a piece smaller than the largest "
                "(orange — mostly newborns joined to no one); the agents removed by the cleanup "
                "of the game because nobody, not even themselves, staked a token on their node "
                "(red); and those removed in the game's cleanup because they were cut off (violet).",
        recipe=recipe(runs_text(), STATS_FILE,
                      ["For iterations 0 to 150, take `births` and `orphaned` from the rows with "
                       "`phase` = 1, and `starved` and `orphaned` from the rows with `phase` = 2.",
                       "At each iteration, take the median of each over the runs that have a row there."]))

    share = F.A.bands([early(*series(s.run_id, "reproTokenShare", 1)) for s in specs], points=151)
    share.update(label="30 worlds: median, middle half, nine in ten", colour=YELLOW)
    ch.figure(
        "children-share", title="Share of all tokens given to children",
        x={"label": "iteration", "min": 0, "max": 150}, y={"label": "share of all tokens", "min": 0},
        series=[share],
        caption="In each reproduction phase, the tokens all parents together gave their children, "
                f"as a share of all 10,000 tokens. {BAND_WORDS}, here at every single iteration.",
        recipe=recipe(runs_text(), STATS_FILE,
                      ["Take the rows with `phase` = 1 and 0 ≤ `iteration` ≤ 150, and the "
                       "statistic `reproTokenShare` — the sum of `invested` over the births of "
                       "that phase, divided by the 10,000 tokens.",
                       "At each iteration take the median, quartiles and 5th and 95th percentiles over the runs."]))
    for t in (0, 10, 20, 50, 100):
        k = int(np.argmin(np.abs(np.array(share["x"]) - t)))
        ch.number(f"children_share_{t}", float(share["y"][k]))

    fam = F.A.bands([early(*series(s.run_id, "cladesInWindow")) for s in specs], points=151)
    fam.update(label="30 worlds: median, middle half, nine in ten", colour=GREEN)
    ch.figure(
        "families", title="Families: distinct ancestors eight iterations back",
        x={"label": "iteration", "min": 0, "max": 150}, y={"label": "families", "min": 0},
        series=[fam],
        caption="After every game, how many different genotypes of eight iterations earlier the "
                "living descend from (before iteration 8: how many founders). "
                f"{BAND_WORDS}, at every iteration.",
        recipe=recipe(runs_text(), STATS_FILE,
                      ["Take the rows with `phase` = 2 and 0 ≤ `iteration` ≤ 150, and the statistic "
                       "`cladesInWindow` ([Families](../notes/families.md): for every living agent, follow its genotype's "
                       "parents back to the newest one born at or before iteration t − 8, or to a "
                       "founder's; count the distinct ones).",
                       "At each iteration take the median, quartiles and 5th and 95th percentiles over the runs."]))
    k = int(np.nanargmin(np.array(fam["y"][:15], dtype=float)))
    ch.number("families_low", {"median": float(fam["y"][k]), "at": float(fam["x"][k])})

    # When does everyone descend from one founder? From the lineage analysis of Experiment 6.
    import json
    lineage = json.load(open(F.os.path.join(F.BOOK, "results", "E06.json")))["lineage"]["baseline"]["runs"]
    when = [v["ancestor"]["oneFounder"] for v in lineage.values() if v["ancestor"]["oneFounder"] is not None]
    edges = np.arange(0, 800, 25)
    counts = [sum(1 for w in when if w == e) for e in edges]
    ch.figure(
        "one-founder", title="When everyone descends from one founder",
        x={"label": "iteration of the first check at which they do", "min": 0, "max": 775},
        y={"label": "worlds", "min": 0},
        series=[{"label": None, "kind": "bars", "x0": (edges - 10).tolist(), "x1": (edges + 10).tolist(),
                 "y": counts, "colour": GREEN}], legend=False,
        caption="For each world, the first of the checks — made every 25 iterations — at which "
                f"every living agent descends from one and the same founder. {len(when)} of the "
                "30 worlds got there; the world with seed 23 died after 4 iterations, first.",
        recipe=recipe(runs_text(), "the runs' frames, through the lineage analysis of "
                      "Experiment 6 (`python3 gol_lab.py analyse E06`), whose results file "
                      "`book/results/E06.json` holds every run's `oneFounder`",
                      ["Read every frame of a run in order and remember, for every genotype "
                       "(brain id), the genotype it was copied from (`parent_brain_ids`).",
                       "After the game of every 25th iteration, follow every living agent's genotype "
                       "back through its parents to a founder's genotype (one with no parent).",
                       "The first iteration at which they all lead to the same founder is the world's value.",
                       "Count the worlds at each value."]))
    ch.number("one_founder", describe(when))
    ch.number("one_founder_by_100", int(sum(1 for w in when if w <= 100)))

    # Which founders' lines survive: shares in world 1.
    w = F.world_pass("B1-10000-s001", (1000, 1499))
    founders = w["founders"]
    xs = [t for t, n, _ in founders["series"]]
    shares = np.array([[c / n for c in per] for t, n, per in founders["series"]])
    order = np.argsort(-shares.max(axis=0))
    top = order[:7]
    bottom = np.zeros(len(xs))
    areas = []
    for rank, j in enumerate(top):
        hi = bottom + shares[:, j]
        areas.append({"label": f"founder line {rank + 1}", "kind": "area", "x": xs,
                      "lo": bottom.tolist(), "hi": hi.tolist(), "colour": rank, "alpha": 0.9})
        bottom = hi
    areas.append({"label": "all other founders", "kind": "area", "x": xs, "lo": bottom.tolist(),
                  "hi": [1.0] * len(xs), "colour": "#5b6b7c", "alpha": 0.9})
    ch.figure(
        "founder-lines", title="Whose descendants? World of seed 1, first 300 iterations",
        x={"label": "iteration", "min": 0, "max": 300}, y={"label": "share of the living", "min": 0, "max": 1},
        series=areas,
        caption="After every game, the share of the living agents that descend from each of the "
                "100 founders, stacked. Each coloured layer is the line of one founder; the seven "
                "founders whose lines ever held the most are drawn in colour, the other 93 together "
                "in grey. Where a layer reaches the full height, everyone alive descends from that founder.",
        recipe=recipe("`B1-10000-s001`, one of the 30 baseline runs.", FRAMES,
                      ["Read every frame in order and remember each genotype's parent genotype "
                       "(`brain_ids`, `parent_brain_ids`).",
                       "After each game up to iteration 300, follow every living agent's genotype "
                       "back to its founder and count the agents per founder; divide by the number "
                       "of agents.",
                       "Stack the shares, the founders with the largest share ever at the bottom."]))
    one = next((t for t, row in zip(xs, shares) if row.max() > 0.999), None)
    ch.number("seed1_one_founder_from", one)


# ---------------------------------------------------------------------------
# Chapter 12 · Births, deaths and ages
# ---------------------------------------------------------------------------

@chapter
def births(ch: F.Chapter) -> None:
    specs = baseline()
    alive = survivors(specs)
    kinds = [("births", 1, "born, per 100 agents that faced reproduction", YELLOW),
             ("orphaned", 1, "cut off in reproduction, per 100 that faced it", ORANGE),
             ("starved", 2, "starved in the game, per 100 that faced it", RED),
             ("orphaned", 2, "cut off in the game, per 100 that faced it", VIOLET)]
    out = []
    for stat, phase, label, colour in kinds:
        b = F.A.bands([per_agent(s.run_id, stat, phase) for s in specs], points=120)
        out.append(line(b["x"], b["y"], label, colour, width=2))
        settled = [np.nanmean(per_agent(s.run_id, stat, phase)[1][500:]) for s in alive]
        ch.number(f"rate_{stat}_{phase}", describe(settled))
    ch.figure(
        "rates", title="Births and deaths per 100 agents",
        x={"label": "iteration"}, y={"label": "per 100 agents", "min": 0, "max": 12},
        series=out,
        caption="Each line is the median over the worlds, in stretches of 25 iterations, of a "
                "count divided by the agents present when the phase began, times 100. The first "
                "iterations run far above the top of the axis (Chapter 11).",
        recipe=recipe(runs_text(), STATS_FILE,
                      ["For every row, divide the count (`births` or `orphaned` in rows with "
                       "`phase` = 1; `starved` or `orphaned` in rows with `phase` = 2) by the "
                       "row's `nodes_before`, the agents present when that phase began, and multiply by 100.",
                       "Cut the 3,000 iterations into 120 stretches of 25; take each run's mean in each.",
                       "Draw the median over the runs that have a value in the stretch."]))

    lives = []
    for s in alive:
        lives += F.world_pass(s.run_id)["agents"]
    at, surv = F.kaplan_meier(lives)
    keep = at <= 3000
    ch.figure(
        "lifetimes", title="How long agents live",
        x={"label": "lifetime L, in iterations (logarithmic)", "log": True, "min": 1},
        y={"label": "share living at least L (logarithmic)", "log": True, "min": 1e-5, "max": 1},
        series=[line(at[keep], surv[keep], f"{len(lives):,} agents born from iteration 500 on", BLUE, width=2)],
        caption="Of all agents born at iteration 500 or later in the 26 worlds that lived to the "
                "end and that took part in at least one game, the share that lived at least L "
                "iterations, for every L. A life is counted from the iteration of birth to the last "
                "iteration after whose game the agent was alive, both included. Lives still going "
                "when a run ended are counted as far as they went (Kaplan–Meier). Both axes are "
                "logarithmic.",
        recipe=recipe(surviving_text(), FRAMES,
                      ["Read every frame with `phase` = 2. For every agent id note its birth "
                       "iteration (the frame's `iteration` minus its `ages` entry) and the last "
                       "iteration it appears.",
                       "Keep agents born at iteration 500 or later. An agent still present in the "
                       "last frame is censored: its life is known to be at least that long.",
                       "Compute the Kaplan–Meier estimate S(L) = Π over lifetimes ℓ < L of (1 − d(ℓ) / n(ℓ)), "
                       "with d(ℓ) the lives that ended at ℓ and n(ℓ) those still at risk."]))
    def s_at(L):
        return float(surv[np.searchsorted(at, L, side="right") - 1])
    ch.number("lives", {"agents": len(lives), "censored": int(sum(1 for b, d, a in lives if a)),
                        **{f"at_least_{L}": s_at(L) for L in (2, 5, 10, 50, 100, 500, 1000)}})
    median_life = float(at[np.argmax(surv <= 0.5)]) - 1
    ch.number("median_life", median_life)

    ages = []
    for s in alive:
        ages += [a for a in F.world_pass(s.run_id)["last"]["ages"] if a >= 0]
    ages = np.array(ages)
    a_edges = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3001])
    counts = np.histogram(ages, bins=a_edges)[0] / len(ages)
    labels = ["0", "1", "2–3", "4–7", "8–15", "16–31", "32–63", "64–127", "128–255",
              "256–511", "512–1,023", "1,024–2,047", "2,048+"]
    ch.figure(
        "ages", title="Ages of the living at the end",
        x={"label": "age in iterations", "categories": labels}, y={"label": "share of the living", "min": 0},
        series=[{"label": None, "kind": "bars", "x0": [i - 0.4 for i in range(len(labels))],
                 "x1": [i + 0.4 for i in range(len(labels))], "y": counts.tolist(), "colour": BLUE}],
        legend=False,
        caption=f"The ages of all {len(ages):,} agents alive after the last game of the 26 worlds "
                "that lived to the end, in bins that double in width: age 0 is born in this "
                "iteration, age 1 in the one before, and so on.",
        recipe=recipe(surviving_text(), FRAMES,
                      ["Read each run's last frame (iteration 2,999, `phase` 2) and its `ages`.",
                       "Count the ages in the bins 0, 1, 2–3, 4–7, …, and divide by the number of agents."]))
    ch.number("ages_end", {"n": int(len(ages)), "median": float(np.median(ages)),
                           "share_0": float(np.mean(ages == 0)), "share_ge_100": float(np.mean(ages >= 100)),
                           "share_ge_1000": float(np.mean(ages >= 1000))})

    med_age = []
    for s in specs:
        a = F.world_pass(s.run_id)["age"]
        med_age.append((np.array([x for x, _ in a], float), np.array([y for _, y in a], float)))
    band = band_series(med_age, "30 worlds: median, middle half, nine in ten", BLUE)
    ch.figure(
        "median-age", title="Median age of the living",
        x={"label": "iteration"}, y={"label": "iterations", "min": 0},
        series=[band],
        caption=f"After every game, the median age of the agents alive. {BAND_WORDS}.",
        recipe=recipe(runs_text(), FRAMES,
                      ["After every game (frames with `phase` = 2) take the median of `ages`.",
                       *BAND_STEPS]))
    for t in (100, 500, 1000, 2000, 2999):
        k = int(np.argmin(np.abs(np.array(band["x"]) - t)))
        ch.number(f"median_age_{t}", float(band["y"][k]))


# ---------------------------------------------------------------------------
# Chapter 13 · How much does the seed decide?
# ---------------------------------------------------------------------------

SEED_STATS = [("nodes", 2, "agents"), ("edges", 2, "connections"), (("births"), 1, "births"),
              ("cladesInWindow", 2, "families"), ("gini", 2, "Gini"), ("topDecileShare", 2, "richest tenth"),
              ("meanDegree", 2, "connections per agent"), ("leaves/nodes", 2, "leaf share"),
              ("bridges/edges", 2, "bridge share"), ("coreShare", 2, "core share"),
              ("transitivity", 2, "clustering"), ("distinctBrains/nodes", 2, "genotypes per agent")]


def stretches(run_id: str, stat: str, phase: int, start: int = 100, stop: int = 3000,
              width: int = 100) -> List[float]:
    its, v = series(run_id, stat, phase)
    return [F.A.window(its, v, a, a + width - 1) for a in range(start, stop, width)]


@chapter
def seed(ch: F.Chapter) -> None:
    alive = survivors(baseline())
    groups = [SEED_STATS[:6], SEED_STATS[6:]]
    for g, (name, title) in enumerate((("dots-1", "Where 26 worlds settle: people, wealth, families"),
                                       ("dots-2", "Where 26 worlds settle: the shape of the network, genotypes"))):
        cats, values = [], []
        for stat, phase, label in groups[g]:
            v = np.array([mean_over(s.run_id, stat, 500, 2999, phase) for s in alive], float)
            cats.append(label)
            values.append((v / v.mean()).tolist())
            ch.number(f"settled_{label}", {**describe(v), "cv": float(v.std(ddof=1) / v.mean())})
        ch.figure(
            name, title=title,
            x={"label": "", "categories": cats}, y={"label": "level ÷ mean of the 26 worlds", "min": 0},
            series=dots(cats, values, [BLUE] * 6), legend=False,
            guides=[{"axis": "y", "at": 1.0}],
            caption="One dot per world: its mean over iterations 500 to 2,999, divided by the mean "
                    "of that value over the 26 worlds, so that every statistic is on the same "
                    "scale and 1 is the average world. The bar is the median. A column of dots "
                    "close to 1 means the worlds settle alike; a tall column, that they do not. "
                    "The dots are spread sideways only so they do not hide one another.",
            recipe=recipe(surviving_text(), STATS_FILE,
                          ["For each run and statistic, average the statistic over the rows with "
                           "500 ≤ `iteration` ≤ 2,999 (rows with `phase` 1 for births, 2 for the rest; "
                           "`a/b` means the row's `a` divided by its `b`).",
                           "Divide each run's average by the mean of the 26 averages.",
                           "Plot one dot per run, with a bar at the median."]))

    lags = list(range(1, 11))
    memory, between = [], {}
    for k, (stat, phase, label) in enumerate([SEED_STATS[i] for i in (0, 1, 2, 3, 4, 9)]):
        levels = np.array([stretches(s.run_id, stat, phase) for s in alive], float)
        centred = levels - levels.mean(axis=1, keepdims=True)
        corr = [float(np.corrcoef(centred[:, :-L].ravel(), centred[:, L:].ravel())[0, 1]) for L in lags]
        memory.append(line([100 * L for L in lags], corr, label, k, width=2))
        b = float(levels.mean(axis=1).var(ddof=1))
        w = float(levels.var(axis=1, ddof=1).mean())
        between[label] = b / (b + w)
        ch.number(f"memory_{label}", dict(zip([100 * L for L in lags], corr)))
    ch.figure(
        "memory", title="How long a world remembers its level",
        x={"label": "iterations apart", "min": 0, "max": 1000}, y={"label": "correlation", "min": -0.4, "max": 1},
        series=memory, guides=[{"axis": "y", "at": 0}],
        caption="For each statistic, the correlation between a world's level in one stretch of "
                "100 iterations and its level a given number of iterations later, over all pairs "
                "of stretches in the 26 worlds. 1 would mean a world stays exactly where it was; "
                "0 that where it was says nothing about where it will be.",
        recipe=recipe(surviving_text(), STATS_FILE,
                      ["For each run, cut iterations 100 to 2,999 into 29 stretches of 100 and take "
                       "the mean of the statistic in each.",
                       "Subtract from each run's 29 values their own mean, so that only the run's "
                       "movement remains.",
                       "For a lag of L stretches, pair every stretch with the one L later, in every "
                       "run, and take the Pearson correlation of all the pairs together."]))

    cats = list(between)
    ch.figure(
        "between", title="Variation between worlds, and within each world",
        x={"label": "", "categories": cats}, y={"label": "share of all variation", "min": 0, "max": 1},
        series=[{"label": "between the worlds' own averages", "kind": "bars", "x0": [i - 0.35 for i in range(len(cats))],
                 "x1": [i + 0.35 for i in range(len(cats))], "y": [between[c] for c in cats], "colour": BLUE},
                {"label": "within each world, over time", "kind": "bars", "x0": [i - 0.35 for i in range(len(cats))],
                 "x1": [i + 0.35 for i in range(len(cats))], "y": [1.0] * len(cats), "colour": GREY, "alpha": 0.25}],
        caption="All the variation of a statistic's 100-iteration levels — 29 stretches in each "
                "of 26 worlds — split into the part that lies between the worlds' own averages "
                "(blue) and the part that is each world moving around its own average (the rest "
                "of the column, grey).",
        recipe=recipe(surviving_text(), STATS_FILE,
                      ["Take each run's 29 stretch means as for the figure above.",
                       "Between: the variance (n − 1 in the denominator) of the 26 runs' own averages.",
                       "Within: the mean over runs of the variance of each run's 29 values.",
                       "Draw between ÷ (between + within)."]))
    ch.number("between_share", between)

    panels, said = [], []
    for stat, label, colour in (("nodes", "agents", BLUE), ("edges", "connections", ORANGE)):
        a = np.array([mean_over(s.run_id, stat, 100, 1499) for s in alive])
        b = np.array([mean_over(s.run_id, stat, 1500, 2999) for s in alive])
        r = float(np.corrcoef(a, b)[0, 1])
        lo, hi = float(min(a.min(), b.min()) * 0.9), float(max(a.max(), b.max()) * 1.05)
        panels.append(dict(
            title=f"{label.capitalize()}: r = {r:.2f}",
            x={"label": "mean over iterations 100–1,499", "min": lo, "max": hi},
            y={"label": "mean over 1,500–2,999", "min": lo, "max": hi},
            series=[{"label": None, "x": a.tolist(), "y": b.tolist(), "points": True, "size": 7, "colour": colour},
                    line([lo, hi], [lo, hi], "the same in both halves", GREY, width=1, dash=[4, 4])]))
        said.append(f"`{stat}`")
        ch.number(f"halves_{label}", r)
    ch.grid(
        "halves", panels, columns=2, title="First half against second half",
        caption="One dot per surviving world. Left: its mean number of agents over iterations "
                "100–1,499 (x) against its mean over iterations 1,500–2,999 (y). Right: the same "
                "for connections. A dot on the dashed diagonal had the same mean in both halves. "
                "r is the correlation over the 26 dots.",
        recipe=recipe(surviving_text(), STATS_FILE,
                      [f"For {' and '.join(said)}, average over the rows with `phase` = 2 and "
                       "100 ≤ `iteration` ≤ 1,499, and over 1,500 ≤ `iteration` ≤ 2,999.",
                       "Plot the second against the first, one dot per run; r is their Pearson "
                       "correlation ([correlation](../notes/correlation.md))."]))

    lengths = list(range(100, 2901, 100))
    curves = []
    for k, (stat, phase, label) in enumerate([SEED_STATS[i] for i in (0, 1, 2, 3, 4)]):
        cv = []
        for L in lengths:
            v = np.array([mean_over(s.run_id, stat, 3000 - L, 2999, phase) for s in alive], float)
            cv.append(float(v.std(ddof=1) / v.mean()))
        curves.append(line(lengths, cv, label, k, width=2))
        ch.number(f"cv_by_length_{label}", dict(zip(lengths, cv)))
    ch.figure(
        "stretch", title="Longer looks, smaller differences",
        x={"label": "iterations averaged, counting back from iteration 2,999"},
        y={"label": "spread between worlds (sd ÷ mean)", "min": 0},
        series=curves,
        caption="For each statistic, how much the 26 worlds differ — the standard deviation of "
                "their averages divided by the mean — when each world is measured by its average "
                "over its last L iterations, for L from 100 to 2,900.",
        recipe=recipe(surviving_text(), STATS_FILE,
                      ["For each length L, average the statistic over the rows with "
                       "3,000 − L ≤ `iteration` ≤ 2,999 in each run.",
                       "Divide the standard deviation of the 26 averages (n − 1 in the denominator) by their mean."]))

    z = F.A.Z_ALPHA + F.A.Z_POWER
    changes = np.linspace(0.05, 0.5, 46)
    needed = []
    for k, (stat, phase, label) in enumerate([SEED_STATS[0], SEED_STATS[4]]):
        for start, how, dash in ((2400, "last fifth", [5, 4]), (500, "from 500", None)):
            v = np.array([mean_over(s.run_id, stat, start, 2999, phase) for s in alive], float)
            cv = v.std(ddof=1) / v.mean()
            n = 2 * (z * cv / changes) ** 2
            needed.append(line(100 * changes, np.ceil(n), f"{label}, {how} (spread {cv:.2f})", k * 2,
                               width=2, **({"dash": dash} if dash else {})))
    ch.figure(
        "seeds-needed", title="Seeds a condition needs",
        x={"label": "the change to be seen, % of the mean"},
        y={"label": "seeds per condition (logarithmic)", "log": True, "min": 1},
        series=needed, guides=[{"axis": "y", "at": 30}],
        caption="How many worlds each of two conditions needs for a change of a given size in "
                "the average to be found four times in five at the 5% level, if worlds vary as "
                "much as these do. Dashed: worlds measured over their last fifth (iterations "
                "2,400–2,999); solid: over iterations 500–2,999. The grey line is 30 seeds.",
        recipe=recipe(surviving_text(), STATS_FILE,
                      ["Measure each run by its average over the stretch, and compute the spread "
                       "c = sd ÷ mean of the 26 averages.",
                       "For a change Δ (as a share of the mean), n = 2 (z₁ + z₂)² c² / Δ², with "
                       "z₁ = 1.960 (5%, two-sided) and z₂ = 0.842 (power 80%); round up."]))
