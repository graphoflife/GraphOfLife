# -*- coding: utf-8 -*-
"""
Part II, the life of a world: Chapters 10 to 14 — a world's life, its youth,
births and deaths, and how much the seed decides.
"""
from __future__ import annotations

import json
from typing import List

import numpy as np

import book_figures as F
from book_figures import (FRAMES, STATS_FILE, band_series, chapter, describe, dots, line, lived,
                          mean_over, per_agent, recipe, series, stacked_areas)
from book_chapters.common import (BAND_STEPS, BAND_WORDS, BLUE, CYAN, GREEN, GREY, ORANGE, RED,
                                  VIOLET, WORLDS as W, YELLOW, ceiling, count_word, listed)

# ---------------------------------------------------------------------------
# Chapter 10 · A world's life
# ---------------------------------------------------------------------------

@chapter
def life(ch: F.Chapter) -> None:
    specs = W.specs
    by_seed = {s.lab["seed"]: s for s in specs}
    dead = W.dead

    # Six single worlds, on one scale, side by side.
    six = [series(by_seed[seed].run_id, "nodes") for seed in range(1, 7)]
    top = ceiling(max(float(np.nanmax(n)) for _, n in six))
    panels = []
    for seed, (its, n) in zip(range(1, 7), six):
        s = by_seed[seed]
        died = lived(s) < s.until
        panels.append(dict(
            title=f"Seed {seed}" + (f" — died after {lived(s)} iterations" if died else ""),
            x={"label": "iteration", "min": 0, "max": W.until}, y={"label": "agents", "min": 0, "max": top},
            series=[line(its, n, None, BLUE, width=1)], legend=False))
    ch.grid(
        "worlds", panels, columns=2, title="Six worlds, one at a time",
        caption="The number of agents alive after every game in the worlds with seeds 1 to 6, "
                f"each in a chart of its own and all on the same scale (0 to {top:,.0f} agents, "
                f"iterations 0 to {W.until:,}). Each line has one point per iteration.",
        recipe=recipe(f"`{W.run(1)}` … `{W.run(6)}`, six of the {len(specs)} baseline runs (their "
                      f"settings are listed in {F._listed_in(W.experiment)}).", STATS_FILE,
                      ["Take the rows with `phase` = 2.",
                       "Plot `nodes` against `iteration`, one point per iteration, joined."]))

    # All of them: the bands.
    agents = [series(s.run_id, "nodes") for s in specs]
    died_words = (f"{count_word(len(dead)).capitalize()} worlds die out; after a world's death it is no "
                  "longer counted." if dead else f"All {len(specs)} live to the end.")
    ch.figure(
        "agents", title=f"Agents alive, {len(specs)} worlds",
        x={"label": "iteration"}, y={"label": "agents", "min": 0},
        series=[band_series(agents, f"{len(specs)} worlds: median, middle half, nine in ten", BLUE)],
        caption=f"{BAND_WORDS}. {died_words}",
        recipe=recipe(W.runs_text(), STATS_FILE,
                      ["Take every run's rows with `phase` = 2 and the statistic `nodes`.",
                       *BAND_STEPS]))

    if dead:
        alive = F.A.bands(agents)
        ch.figure(
            "alive", title="Worlds still alive",
            x={"label": "iteration"}, y={"label": "worlds", "min": 0, "max": len(specs) + 1},
            series=[line(alive["x"], alive["alive"], None, GREY, width=2)], legend=False,
            caption=f"How many of the {len(specs)} worlds are still alive. A world dies when an iteration ends "
                    f"with 20 agents or fewer; {count_word(len(dead))} did, after "
                    f"{listed(sorted(lived(s) for s in dead))} iterations.",
            recipe=recipe(W.runs_text(), STATS_FILE,
                          ["For every stretch of the bands above, count the worlds that have a row "
                           "with `phase` = 2 in it."]))

    # Early against late, world by world.
    late_from, late_to = W.tail(200)
    early = {s.lab["seed"]: mean_over(s.run_id, "nodes", 100, 200) for s in specs if lived(s) >= 200}
    late = {s.lab["seed"]: mean_over(s.run_id, "nodes", late_from, late_to) for s in W.alive}
    both = sorted(set(early) & set(late))
    pairs = []
    for seed in both:
        ratio = late[seed] / early[seed]
        colour = GREEN if ratio > 1.2 else RED if ratio < 0.8 else GREY
        pairs.append(line([0, 1], [early[seed], late[seed]], None, colour, width=1.2, alpha=0.8))
    pairs.append(line([0, 1], [np.median([early[s] for s in both]), np.median([late[s] for s in both])],
                      f"median of the {len(both)} worlds", BLUE, width=3.5))
    ch.figure(
        "early-late", title="Each world early and late",
        x={"label": "", "categories": ["iterations 100–200", f"iterations {late_from:,}–{late_to:,}"]},
        y={"label": "mean number of agents", "min": 0},
        series=pairs,
        caption=f"One line per {W.per_lived}, from its mean number of agents over iterations "
                f"100–200 (left) to its mean over iterations {late_from:,}–{late_to:,} (right). Green: the "
                "late mean is more than 1.2 times the early one; red: less than 0.8 times; grey: "
                f"in between. The thick blue line joins the medians of the {len(both)} worlds.",
        recipe=recipe(W.surviving_text(), STATS_FILE,
                      [f"For each run that reached iteration {W.until:,}, take the rows with `phase` = 2.",
                       "Average `nodes` over the rows with 100 ≤ `iteration` ≤ 200, and again "
                       f"over {late_from:,} ≤ `iteration` ≤ {late_to:,}.",
                       "Draw one line from the first mean to the second; colour it by their ratio.",
                       f"Join the two medians over the {len(both)} runs."]))
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
        "connections", title=f"Connections, {len(specs)} worlds",
        x={"label": "iteration"}, y={"label": "connections", "min": 0},
        series=[band_series([series(s.run_id, "edges") for s in specs],
                            f"{len(specs)} worlds: median, middle half, nine in ten", ORANGE)],
        caption=f"The number of connections after every game. {BAND_WORDS}.",
        recipe=recipe(W.runs_text(), STATS_FILE,
                      ["Take every run's rows with `phase` = 2 and the statistic `edges`.",
                       *BAND_STEPS]))

    # The dead.
    if dead:
        traces = []
        for k, s in enumerate(dead):
            its, n = series(s.run_id, "nodes")
            traces.append(line(its, n, f"seed {s.lab['seed']} (died after {lived(s)} iterations)",
                               [RED, YELLOW, VIOLET, CYAN][k % 4], width=1.2))
        names = [f"`{s.run_id}`" for s in dead]
        ch.figure(
            "dead", title=f"The {count_word(len(dead))} worlds that died",
            x={"label": "iteration"}, y={"label": "agents (logarithmic)", "log": True, "min": 10},
            series=traces,
            guides=[{"axis": "y", "at": 20, "label": "extinct at 20 or fewer"}],
            caption=f"The number of agents after every game in the {count_word(len(dead))} worlds that died, on a "
                    "logarithmic axis, so that a fall from 1,000 to 100 looks as long as one from "
                    "100 to 10. A world stops when an iteration ends with 20 agents or fewer.",
            recipe=recipe(f"{listed(names)}, the {count_word(len(dead))} of the {len(specs)} "
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
        l = describe([mean_over(s.run_id, stat, late_from, late_to, phase) for s in W.alive])
        ch.number(f"window_{stat}", {"early": e, "late": l})


# ---------------------------------------------------------------------------
# Chapter 11 · The first hundred iterations
# ---------------------------------------------------------------------------

@chapter
def youth(ch: F.Chapter) -> None:
    specs = W.specs
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
        caption=f"Each thin blue line is one of the {len(specs)} worlds: the number of agents alive after "
                "every game. The yellow line is their median. The axis is logarithmic, so equal "
                "heights are equal factors: the step from 100 to 1,000 is as tall as the one from "
                "1,000 to 10,000.",
        recipe=recipe(W.runs_text(), STATS_FILE,
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
        recipe=recipe(W.runs_text(), STATS_FILE,
                      ["For iterations 0 to 150, take `births` and `orphaned` from the rows with "
                       "`phase` = 1, and `starved` and `orphaned` from the rows with `phase` = 2.",
                       "At each iteration, take the median of each over the runs that have a row there."]))

    share = F.A.bands([early(*series(s.run_id, "reproTokenShare", 1)) for s in specs], points=151)
    share.update(label=f"{len(specs)} worlds: median, middle half, nine in ten", colour=YELLOW)
    ch.figure(
        "children-share", title="Share of all tokens given to children",
        x={"label": "iteration", "min": 0, "max": 150}, y={"label": "share of all tokens", "min": 0},
        series=[share],
        caption="In each reproduction phase, the tokens all parents together gave their children, "
                f"as a share of all {W.tokens:,} tokens. {BAND_WORDS}, here at every single iteration.",
        recipe=recipe(W.runs_text(), STATS_FILE,
                      ["Take the rows with `phase` = 1 and 0 ≤ `iteration` ≤ 150, and the "
                       "statistic `reproTokenShare` — the sum of `invested` over the births of "
                       f"that phase, divided by the {W.tokens:,} tokens.",
                       "At each iteration take the median, quartiles and 5th and 95th percentiles over the runs."]))
    for t in (0, 10, 20, 50, 100):
        k = int(np.argmin(np.abs(np.array(share["x"]) - t)))
        ch.number(f"children_share_{t}", float(share["y"][k]))

    fam = F.A.bands([early(*series(s.run_id, "cladesInWindow")) for s in specs], points=151)
    fam.update(label=f"{len(specs)} worlds: median, middle half, nine in ten", colour=GREEN)
    ch.figure(
        "families", title="Families: distinct ancestors eight iterations back",
        x={"label": "iteration", "min": 0, "max": 150}, y={"label": "families", "min": 0},
        series=[fam],
        caption="After every game, how many different genotypes of eight iterations earlier the "
                "living descend from (before iteration 8: how many founders). "
                f"{BAND_WORDS}, at every iteration.",
        recipe=recipe(W.runs_text(), STATS_FILE,
                      ["Take the rows with `phase` = 2 and 0 ≤ `iteration` ≤ 150, and the statistic "
                       "`cladesInWindow` ([Families](families.md): for every living agent, follow its genotype's "
                       "parents back to the newest one born at or before iteration t − 8, or to a "
                       "founder's; count the distinct ones).",
                       "At each iteration take the median, quartiles and 5th and 95th percentiles over the runs."]))
    k = int(np.nanargmin(np.array(fam["y"][:15], dtype=float)))
    ch.number("families_low", {"median": float(fam["y"][k]), "at": float(fam["x"][k])})

    # When does everyone descend from one founder? From the lineage analysis of the worlds' experiment.
    lineage = json.load(open(F.os.path.join(F.BOOK, "results", f"{W.lineage}.json")))["lineage"][W.condition]["runs"]
    when = [v["ancestor"]["oneFounder"] for v in lineage.values() if v["ancestor"]["oneFounder"] is not None]
    never = [run for run, v in lineage.items() if v["ancestor"]["oneFounder"] is None]
    stop = int(25 * np.ceil((max(when) + 1) / 25)) if when else 25
    edges = np.arange(0, stop + 25, 25)
    counts = [sum(1 for w in when if w == e) for e in edges]
    early_dead = [r for r in never if lived(next(s for s in specs if s.run_id == r)) < 25]
    if not never:
        missing = f"All {len(specs)} worlds got there."
    else:
        missing = (f"{len(when)} of the {len(specs)} worlds got there; "
                   + "; ".join(f"the world with seed {int(r.rsplit('s', 1)[1])} "
                               + (f"died after {lived(next(s for s in specs if s.run_id == r))} iterations, first"
                                  if r in early_dead else "never did")
                               for r in never) + ".")
    ch.figure(
        "one-founder", title="When everyone descends from one founder",
        x={"label": "iteration of the first check at which they do", "min": 0, "max": float(edges[-1])},
        y={"label": "worlds", "min": 0},
        series=[{"label": None, "kind": "bars", "x0": (edges - 10).tolist(), "x1": (edges + 10).tolist(),
                 "y": counts, "colour": GREEN}], legend=False,
        caption="For each world, the first of the checks — made every 25 iterations — at which "
                f"every living agent descends from one and the same founder. {missing}",
        recipe=recipe(W.runs_text(), "the runs' frames, through the lineage analysis of "
                      f"Experiment {int(W.lineage[1:])} (`python3 gol_lab.py analyse {W.lineage}`), whose results "
                      f"file `book/results/{W.lineage}.json` holds every run's `oneFounder`",
                      ["Read every frame of a run in order and remember, for every genotype "
                       "(brain id), the genotype it was copied from (`parent_brain_ids`).",
                       "After the game of every 25th iteration, follow every living agent's genotype "
                       "back through its parents to a founder's genotype (one with no parent).",
                       "The first iteration at which they all lead to the same founder is the world's value.",
                       "Count the worlds at each value."]))
    ch.number("one_founder", describe(when))
    ch.number("one_founder_by_100", int(sum(1 for w in when if w <= 100)))
    ch.number("one_founder_never", sorted(never))

    # Which founders' lines survive: shares in world 1.
    w = F.world_pass(W.run(1), (1000, 1499))
    founders = w["founders"]
    xs = [t for t, n, _ in founders["series"]]
    shares = np.array([[c / n for c in per] for t, n, per in founders["series"]])
    order = np.argsort(-shares.max(axis=0))
    top = order[:7]
    areas = stacked_areas(xs, [(f"founder line {rank + 1}", shares[:, j], rank) for rank, j in enumerate(top)],
                          "all other founders")
    count = F.store.load_config(W.run(1)).resolved_n()
    ch.figure(
        "founder-lines", title="Whose descendants? World of seed 1, first 300 iterations",
        x={"label": "iteration", "min": 0, "max": 300}, y={"label": "share of the living", "min": 0, "max": 1},
        series=areas,
        caption="After every game, the share of the living agents that descend from each of the "
                f"{count:,} founders, stacked. Each coloured layer is the line of one founder; the seven "
                f"founders whose lines ever held the most are drawn in colour, the other {count - 7:,} together "
                "in grey (most of them with no descendants alive at all). Where a layer reaches the full height, "
                "everyone alive descends from that founder.",
        recipe=recipe(f"`{W.run(1)}`, one of the {len(specs)} baseline runs.", FRAMES,
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
    specs = W.specs
    alive = W.alive
    stretches = W.until // 25
    kinds = [("births", 1, "born, per 100 agents that faced reproduction", YELLOW),
             ("orphaned", 1, "cut off in reproduction, per 100 that faced it", ORANGE),
             ("starved", 2, "starved in the game, per 100 that faced it", RED),
             ("orphaned", 2, "cut off in the game, per 100 that faced it", VIOLET)]
    out = []
    for stat, phase, label, colour in kinds:
        b = F.A.bands([per_agent(s.run_id, stat, phase) for s in specs], points=stretches)
        out.append(line(b["x"], b["y"], label, colour, width=2))
        settled = [np.nanmean(per_agent(s.run_id, stat, phase)[1][W.settled:]) for s in alive]
        ch.number(f"rate_{stat}_{phase}", describe(settled))
    # Starving in reproduction is too rare to draw beside the others; the text gives its rate.
    ch.number("rate_starved_1", describe([np.nanmean(per_agent(s.run_id, "starved", 1)[1][W.settled:])
                                          for s in alive]))
    ch.figure(
        "rates", title="Births and deaths per 100 agents",
        x={"label": "iteration"}, y={"label": "per 100 agents", "min": 0, "max": 12},
        series=out,
        caption="Each line is the median over the worlds, in stretches of 25 iterations, of a "
                "count divided by the agents present when the phase began, times 100. The first "
                "iterations run far above the top of the axis (Chapter 11).",
        recipe=recipe(W.runs_text(), STATS_FILE,
                      ["For every row, divide the count (`births` or `orphaned` in rows with "
                       "`phase` = 1; `starved` or `orphaned` in rows with `phase` = 2) by the "
                       "row's `nodes_before`, the agents present when that phase began, and multiply by 100.",
                       f"Cut the {W.until:,} iterations into {stretches} stretches of 25; take each run's mean in each.",
                       "Draw the median over the runs that have a value in the stretch."]))

    lives = []
    for s in alive:
        lives += F.world_pass(s.run_id)["agents"]
    at, surv = F.kaplan_meier(lives)
    keep = at <= W.until
    ch.figure(
        "lifetimes", title="How long agents live",
        x={"label": "lifetime L, in iterations (logarithmic)", "log": True, "min": 1},
        y={"label": "share living at least L (logarithmic)", "log": True, "min": 1e-5, "max": 1},
        series=[line(at[keep], surv[keep], f"{len(lives):,} agents born from iteration {W.settled} on, alive after a game", BLUE,
                     width=2)],
        caption=f"Of all agents born at iteration {W.settled} or later in {W.lived} and alive after at "
                "least one game — the newborns cut off in the reproduction phase of their birth, or dead "
                "in their first game, are left out — the share that lived at least L "
                "iterations, for every L. A life is counted from the iteration of birth to the last "
                "iteration after whose game the agent was alive, both included. Lives still going "
                "when a run ended are counted as far as they went (Kaplan–Meier). Both axes are "
                "logarithmic.",
        recipe=recipe(W.surviving_text(), FRAMES,
                      ["Read every frame with `phase` = 2. For every agent id note its birth "
                       "iteration (the frame's `iteration` minus its `ages` entry) and the last "
                       "iteration it appears.",
                       f"Keep agents born at iteration {W.settled} or later. An agent still present in the "
                       "last frame is censored: its life is known to be at least that long.",
                       "Compute the Kaplan–Meier estimate S(L) = Π over lifetimes ℓ < L of (1 − d(ℓ) / n(ℓ)), "
                       "with d(ℓ) the lives that ended at ℓ and n(ℓ) those still at risk."]))
    def s_at(L):
        return float(surv[np.searchsorted(at, L, side="right") - 1])
    ch.number("lives", {"agents": len(lives), "censored": int(sum(1 for b, d, a in lives if a)),
                        **{f"at_least_{L}": s_at(L) for L in (2, 5, 10, 50, 100, 500, 1000) if L < W.until}})
    median_life = float(at[np.argmax(surv <= 0.5)]) - 1
    ch.number("median_life", median_life)

    ages = []
    for s in alive:
        ages += [a for a in F.world_pass(s.run_id)["last"]["ages"] if a >= 0]
    ages = np.array(ages)
    doubling = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    a_edges = np.array([e for e in doubling if e < W.until] + [W.until + 1])
    labels = [f"{a:,}" if b - a == 1 else (f"{a:,}+" if b > W.until else f"{a:,}–{b - 1:,}")
              for a, b in zip(a_edges[:-1], a_edges[1:])]
    counts = np.histogram(ages, bins=a_edges)[0] / len(ages)
    ch.figure(
        "ages", title="Ages of the living at the end",
        x={"label": "age in iterations", "categories": labels}, y={"label": "share of the living", "min": 0},
        series=[{"label": None, "kind": "bars", "x0": [i - 0.4 for i in range(len(labels))],
                 "x1": [i + 0.4 for i in range(len(labels))], "y": counts.tolist(), "colour": BLUE}],
        legend=False,
        caption=f"The ages of all {len(ages):,} agents alive after the last game of {W.lived}, in bins that "
                "double in width: age 0 is born in this iteration, age 1 in the one before, and so on.",
        recipe=recipe(W.surviving_text(), FRAMES,
                      [f"Read each run's last frame (iteration {W.last:,}, `phase` 2) and its `ages`.",
                       "Count the ages in the bins 0, 1, 2–3, 4–7, …, and divide by the number of agents."]))
    ch.number("ages_end", {"n": int(len(ages)), "median": float(np.median(ages)),
                           "share_0": float(np.mean(ages == 0)), "share_ge_100": float(np.mean(ages >= 100)),
                           "share_ge_1000": float(np.mean(ages >= 1000))})


# ---------------------------------------------------------------------------
# Chapter 14 · How much does the seed decide?
# ---------------------------------------------------------------------------

SEED_STATS = [("nodes", 2, "agents"), ("edges", 2, "connections"), (("births"), 1, "births"),
              ("cladesInWindow", 2, "families"), ("gini", 2, "Gini"), ("topDecileShare", 2, "richest tenth"),
              ("meanDegree", 2, "connections per agent"), ("leaves/nodes", 2, "leaf share"),
              ("bridges/edges", 2, "bridge share"), ("coreShare", 2, "core share"),
              ("transitivity", 2, "clustering"), ("distinctBrains/nodes", 2, "genotypes per agent")]


def stretches(run_id: str, stat: str, phase: int, start: int = 100, stop: int = None,
              width: int = 100) -> List[float]:
    its, v = series(run_id, stat, phase)
    return [F.A.window(its, v, a, a + width - 1) for a in range(start, stop or W.until, width)]


@chapter
def seed(ch: F.Chapter) -> None:
    alive = W.alive
    n = len(alive)
    groups = [SEED_STATS[:6], SEED_STATS[6:]]
    for g, (name, title) in enumerate((("dots-1", f"Where {n} worlds settle: people, wealth, families"),
                                       ("dots-2", f"Where {n} worlds settle: the shape of the network, genotypes"))):
        cats, values = [], []
        for stat, phase, label in groups[g]:
            v = np.array([mean_over(s.run_id, stat, *W.window, phase) for s in alive], float)
            cats.append(label)
            values.append((v / v.mean()).tolist())
            ch.number(f"settled_{label}", {**describe(v), "cv": float(v.std(ddof=1) / v.mean())})
        ch.figure(
            name, title=title,
            x={"label": "", "categories": cats}, y={"label": f"level ÷ mean of the {n} worlds", "min": 0},
            series=dots(cats, values, [BLUE] * 6), legend=False,
            guides=[{"axis": "y", "at": 1.0}],
            caption=f"One dot per world: its mean over iterations {W.stretch}, divided by the mean "
                    f"of that value over the {n} worlds, so that every statistic is on the same "
                    "scale and 1 is the average world. The bar is the median. A column of dots "
                    "close to 1 means the worlds settle alike; a tall column, that they do not. "
                    "The dots are spread sideways only so they do not hide one another.",
            recipe=recipe(W.surviving_text(), STATS_FILE,
                          ["For each run and statistic, average the statistic over the rows with "
                           f"{W.settled} ≤ `iteration` ≤ {W.last:,} (rows with `phase` 1 for births, 2 for the rest; "
                           "`a/b` means the row's `a` divided by its `b`).",
                           f"Divide each run's average by the mean of the {n} averages.",
                           "Plot one dot per run, with a bar at the median."]))

    pieces = len(range(100, W.until, 100))
    lags = [L for L in range(1, 11) if L < pieces]
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
    # What the same cutting and centring give a level that wanders without limit, and one with no memory.
    rng = np.random.default_rng(13)

    def reference(walk: bool) -> List[float]:
        total = np.zeros(len(lags))
        for _ in range(200):
            steps = rng.normal(size=(n, pieces * 100))
            x = np.cumsum(steps, axis=1) if walk else steps
            levels = x.reshape(n, pieces, 100).mean(axis=2)
            centred = levels - levels.mean(axis=1, keepdims=True)
            total += [np.corrcoef(centred[:, :-L].ravel(), centred[:, L:].ravel())[0, 1] for L in lags]
        return (total / 200).tolist()
    walk, noise = reference(True), reference(False)
    memory += [line([100 * L for L in lags], walk, "a random walk, cut and centred the same way", GREY, width=1.5,
                    dash=[5, 4]),
               line([100 * L for L in lags], noise, "levels with no memory, the same way", GREY, width=1.5,
                    dash=[2, 3])]
    ch.number("memory_reference", {"walk": dict(zip([100 * L for L in lags], walk)),
                                   "noise": dict(zip([100 * L for L in lags], noise))})
    ch.figure(
        "memory", title="How long a world remembers its level",
        x={"label": "iterations apart", "min": 0, "max": 100 * max(lags)},
        y={"label": "correlation", "min": -0.8, "max": 1},
        series=memory, guides=[{"axis": "y", "at": 0}],
        caption="For each statistic, the correlation between a world's level in one stretch of "
                "100 iterations and its level a given number of iterations later, over all pairs "
                f"of stretches in the {n} worlds. 1 would mean a world stays exactly where it was; "
                "0 that where it was says nothing about where it will be. Because each world's own mean is "
                f"taken away first, a world cut into only {pieces} stretches shows negative correlations at long "
                "lags even if it simply wandered: the grey dashes are what a random walk, cut and centred the "
                "same way, gives (averaged over 200 sets of as many walks), and the grey dots what levels with no "
                "memory at all give.",
        recipe=recipe(W.surviving_text(), STATS_FILE,
                      [f"For each run, cut iterations 100 to {W.last:,} into {pieces} stretches of 100 and take "
                       "the mean of the statistic in each.",
                       f"Subtract from each run's {pieces} values their own mean, so that only the run's "
                       "movement remains.",
                       "For a lag of L stretches, pair every stretch with the one L later, in every "
                       "run, and take the Pearson correlation of all the pairs together.",
                       f"References: {n} series of {pieces * 100:,} steps, each step normal with mean 0 — summed "
                       "(a random walk) or not (no memory) — cut, centred and correlated the same way; the mean over "
                       "200 such sets (generator seeded with 13)."]))

    cats = list(between)
    ch.figure(
        "between", title="Variation between worlds, and within each world",
        x={"label": "", "categories": cats}, y={"label": "share of all variation", "min": 0, "max": 1},
        series=[{"label": "between the worlds' own averages", "kind": "bars", "x0": [i - 0.35 for i in range(len(cats))],
                 "x1": [i + 0.35 for i in range(len(cats))], "y": [between[c] for c in cats], "colour": BLUE},
                {"label": "within each world, over time", "kind": "bars", "x0": [i - 0.35 for i in range(len(cats))],
                 "x1": [i + 0.35 for i in range(len(cats))], "y": [1.0] * len(cats), "colour": GREY, "alpha": 0.25}],
        caption=f"All the variation of a statistic's 100-iteration levels — {pieces} stretches in each "
                f"of {n} worlds — split into the part that lies between the worlds' own averages "
                "(blue) and the part that is each world moving around its own average (the rest "
                "of the column, grey).",
        recipe=recipe(W.surviving_text(), STATS_FILE,
                      [f"Take each run's {pieces} stretch means as for the figure above.",
                       f"Between: the variance (n − 1 in the denominator) of the {n} runs' own averages.",
                       f"Within: the mean over runs of the variance of each run's {pieces} values.",
                       "Draw between ÷ (between + within)."]))
    ch.number("between_share", between)

    half = W.until // 2
    first_half, second_half = (100, half - 1), (half, W.last)
    panels, said = [], []
    for stat, label, colour in (("nodes", "agents", BLUE), ("edges", "connections", ORANGE)):
        a = np.array([mean_over(s.run_id, stat, *first_half) for s in alive])
        b = np.array([mean_over(s.run_id, stat, *second_half) for s in alive])
        r = float(np.corrcoef(a, b)[0, 1])
        lo, hi = float(min(a.min(), b.min()) * 0.9), float(max(a.max(), b.max()) * 1.05)
        panels.append(dict(
            title=f"{label.capitalize()}: r = {r:.2f}",
            x={"label": f"mean over iterations 100–{half - 1:,}", "min": lo, "max": hi},
            y={"label": f"mean over {half:,}–{W.last:,}", "min": lo, "max": hi},
            series=[{"label": None, "x": a.tolist(), "y": b.tolist(), "points": True, "size": 7, "colour": colour},
                    line([lo, hi], [lo, hi], "the same in both halves", GREY, width=1, dash=[4, 4])]))
        said.append(f"`{stat}`")
        ch.number(f"halves_{label}", r)
    ch.grid(
        "halves", panels, columns=2, title="First half against second half",
        caption=f"One dot per {W.per_lived}. Left: its mean number of agents over iterations "
                f"100–{half - 1:,} (x) against its mean over iterations {half:,}–{W.last:,} (y). Right: the same "
                "for connections. A dot on the dashed diagonal had the same mean in both halves. "
                f"r is the correlation over the {n} dots.",
        recipe=recipe(W.surviving_text(), STATS_FILE,
                      [f"For {' and '.join(said)}, average over the rows with `phase` = 2 and "
                       f"100 ≤ `iteration` ≤ {half - 1:,}, and over {half:,} ≤ `iteration` ≤ {W.last:,}.",
                       "Plot the second against the first, one dot per run; r is their Pearson "
                       "correlation ([correlation](correlation.md))."]))

    lengths = list(range(100, W.until - 100 + 1, 100))
    curves = []
    for k, (stat, phase, label) in enumerate([SEED_STATS[i] for i in (0, 1, 2, 3, 4)]):
        cv = []
        for L in lengths:
            v = np.array([mean_over(s.run_id, stat, *W.tail(L), phase) for s in alive], float)
            cv.append(float(v.std(ddof=1) / v.mean()))
        curves.append(line(lengths, cv, label, k, width=2))
        ch.number(f"cv_by_length_{label}", dict(zip(lengths, cv)))
    ch.figure(
        "stretch", title="Longer looks, smaller differences",
        x={"label": f"iterations averaged, counting back from iteration {W.last:,}"},
        y={"label": "spread between worlds (sd ÷ mean)", "min": 0},
        series=curves,
        caption=f"For each statistic, how much the {n} worlds differ — the standard deviation of "
                "their averages divided by the mean — when each world is measured by its average "
                f"over its last L iterations, for L from 100 to {lengths[-1]:,}.",
        recipe=recipe(W.surviving_text(), STATS_FILE,
                      ["For each length L, average the statistic over the rows with "
                       f"{W.until:,} − L ≤ `iteration` ≤ {W.last:,} in each run.",
                       f"Divide the standard deviation of the {n} averages (n − 1 in the denominator) by their mean."]))

    z = F.A.Z_ALPHA + F.A.Z_POWER
    changes = np.linspace(0.05, 0.5, 46)
    fifth = W.tail(W.until // 5)
    needed = []
    for k, (stat, phase, label) in enumerate([SEED_STATS[0], SEED_STATS[4]]):
        for start, how, dash in ((fifth[0], "last fifth", [5, 4]), (W.settled, f"from {W.settled}", None)):
            v = np.array([mean_over(s.run_id, stat, start, W.last, phase) for s in alive], float)
            cv = v.std(ddof=1) / v.mean()
            n_needed = 2 * (z * cv / changes) ** 2
            needed.append(line(100 * changes, np.ceil(n_needed), f"{label}, {how} (spread {cv:.2f})", k * 2,
                               width=2, **({"dash": dash} if dash else {})))
    ch.figure(
        "seeds-needed", title="Seeds a condition needs",
        x={"label": "the change to be seen, % of the mean"},
        y={"label": "seeds per condition (logarithmic)", "log": True, "min": 1},
        series=needed, guides=[{"axis": "y", "at": 30}],
        caption="How many worlds each of two conditions needs for a change of a given size in "
                "the average to be found four times in five at the 5% level, if worlds vary as "
                "much as these do. Dashed: worlds measured over their last fifth (iterations "
                f"{fifth[0]:,}–{fifth[1]:,}); solid: over iterations {W.span}. The grey line is 30 seeds.",
        recipe=recipe(W.surviving_text(), STATS_FILE,
                      ["Measure each run by its average over the stretch, and compute the spread "
                       f"c = sd ÷ mean of the {n} averages.",
                       "For a change Δ (as a share of the mean), n = 2 (z₁ + z₂)² c² / Δ², with "
                       "z₁ = 1.960 (5%, two-sided) and z₂ = 0.842 (power 80%); round up."]))
