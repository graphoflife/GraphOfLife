# -*- coding: utf-8 -*-
"""
the work on open-ended evolution, the pages written from pilot worlds: worlds made by
research/oee/pilots/run.py rather than by the lab, kept in
GraphOfLifeRuns/.pilots — Strategies written by hand (strategies written by hand), 54 (the
original algorithm's switches), 55 (another game), 56 (alliances) and 57
(rules without numbers).

A pilot world writes one row per iteration to `series.jsonl` (the network's
health every tenth), its settings to `config.json` and its whole world every
100 iterations (`ckpt_<iteration>.npz`); a tournament between those is written
beside them by research/oee/pilots/tourney.py. Pilots are fewer and shorter than
an experiment's runs — two seeds where an experiment has ten — so what they
show is a lead, and the pages say so.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import book_figures as F
from book_figures import chapter, recipe
from book_chapters.common import BLUE, GREEN, GREY, ORANGE, RED, VIOLET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILOTS = os.path.join(ROOT, "GraphOfLifeRuns", ".pilots")
TOOLS = os.path.join(ROOT, "research", "oee", "pilots")


def rows_of(world: str) -> List[Dict[str, Any]]:
    """One row per iteration of a pilot world (`A10k/allies-s1`), empty if it does not exist."""
    path = os.path.join(PILOTS, world, "series.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def lived(world: str) -> int:
    rows = rows_of(world)
    return rows[-1]["it"] if rows else 0


def per_agent(rows: List[Dict[str, Any]], key: str, lo: int, hi: int) -> Optional[float]:
    """A count per 100 agents per iteration, averaged over iterations lo to hi."""
    v = [100 * r.get(key, 0) / max(1, r["nodes"]) for r in rows if lo <= r["it"] < hi]
    return float(np.mean(v)) if v else None


def health_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if "coreShare" in r]


def share_at(rows: List[Dict[str, Any]], key: str, t: int) -> Optional[float]:
    """`key` (a count of agents) as a share of the agents, at the first measured iteration from t."""
    later = [r for r in rows if r["it"] >= t and key in r]
    return later[0][key] / later[0]["nodes"] if later else None


def ratings(world: str, name: str = "tourney_T20_all.json") -> Optional[Tuple[List[int], List[float]]]:
    """The Elo rating of every era of a pilot tournament, the first era at 0 (research/oee/pilots/elo.py)."""
    path = os.path.join(PILOTS, world, name)
    if not os.path.exists(path):
        return None
    sys.path.insert(0, TOOLS)
    from elo import ratings as fit
    eras, elo, _ = fit(path)
    return list(eras), [float(e) for e in elo]


def pair_score(world: str, earlier: int, later: int, name: str = "tourney_T20_all.json") -> Optional[float]:
    """The later era's share of the agents against the earlier one, over every contest the two played."""
    path = os.path.join(PILOTS, world, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    s = [1 - share_a for _, a, b, _, share_a, _, _ in data["results"]
         if a == earlier and b == later and share_a is not None]
    return float(np.mean(s)) if s else None


PROTO_B2 = ("B2's rules as the pilots' prototype states them (`research/oee/pilots/proto.py`: mor=all "
            "incumbency=1.5 birth_links=2 repair=true safe_prune=2 max_age=100 valence=8 "
            "valence_mode=optional min_child=10 keep_frac=0.3), with one hidden layer of 16, messages of "
            "8 numbers and no pre-pass, revolutions off, and a memory of the last game's stakes "
            "(memory=true)")


# ---------------------------------------------------------------------------
# Alliances
# ---------------------------------------------------------------------------

ALLY_WORLDS = {
    "with a home advantage, 10,000 tokens": (["A10k/allies-s1", "A10k/allies-s2", "AL/allies10k-s3",
                                              "AL/allies10k-s4"], BLUE),
    "with a home advantage, 20,000 tokens": (["AL/allies20k-s1", "AL/allies20k-s2"], RED),
    "without one, 10,000 tokens": (["N/N0-s1", "N/N0-s2"], GREEN),
}
FIGHT_WORLDS = {
    "no alliances": (["A10k/ctrl-s1", "A10k/ctrl-s2"], GREY),
    "alliances": (["A10k/allies-s1", "A10k/allies-s2", "AL/allies10k-s3", "AL/allies10k-s4"], BLUE),
    "kin pooled": (["A10k/kin-s1", "A10k/kin-s2"], VIOLET),
    "alliances, no home advantage": (["N/N0-s1", "N/N0-s2"], GREEN),
}
TOURNEYS = {"no alliances": "A10k/ctrl-s1", "alliances": "A10k/allies-s1", "kin pooled": "A10k/kin-s1",
            "support counted twice": "A10k/gain-s1", "alliances, 3,000 iterations": "AL/allies10k-s3"}
ALLY_RUNS = ("Pilot worlds of `research/oee/pilots/run.py`, two seeds each, kept in "
             "`GraphOfLifeRuns/.pilots`: at 10,000 tokens for 1,000 iterations without alliances "
             "(`A10k/ctrl-s1`, `-s2`), with them (`A10k/allies-s1`, `-s2`), with kin pooled "
             "(`A10k/kin-s1`, `-s2`) and with support counted twice (`A10k/gain-s1`, `-s2`); with "
             "alliances for 3,000 iterations (`AL/allies10k-s3`, `-s4`), at 20,000 tokens for 2,000 "
             "(`AL/allies20k-s1`, `-s2`), and without the home advantage, incumbency 1, for 1,000 "
             f"(`N/N0-s1`, `-s2`). Every one under {PROTO_B2}.")


@chapter
def allies(ch: F.Chapter) -> None:
    if not all(lived(w) for group, _ in ALLY_WORLDS.values() for w in group):
        return
    series_ = []
    for label, (worlds, colour) in ALLY_WORLDS.items():
        for k, w in enumerate(worlds):
            h = [r for r in health_rows(rows_of(w)) if "sideLargest" in r]
            series_.append({"label": label if k == 0 else None, "x": [r["it"] for r in h],
                            "y": [r["sideLargest"] / r["nodes"] for r in h], "colour": colour, "width": 1.5})
    ch.figure("bloc", title="The largest alliance's share of the world",
              x={"label": "iteration"}, y={"label": "share of the agents in the largest alliance", "min": 0, "max": 1},
              series=series_,
              caption=("One line per world. An alliance is a group of agents joined by connections "
                       "between allies: two neighbours that both chose, in the last game, to be allied."),
              recipe=recipe(ALLY_RUNS, "each world's `series.jsonl`", [
                  "Every 10th iteration, join every two neighbours that were allied in the game just "
                  "played (`ProtoWorld.same_side`) and find the largest connected group "
                  "(`research/oee/pilots/run.py`, `organisation`, `sideLargest`).",
                  "Divide by the agents."]))

    fights = []
    for label, (worlds, colour) in FIGHT_WORLDS.items():
        for k, w in enumerate(worlds):
            rows = rows_of(w)
            its = np.array([r["it"] for r in rows])
            vals = np.array([100 * r.get("conquests", 0) / max(1, r["nodes"]) for r in rows])
            x, y = F.thin(its, vals, 50)
            fights.append({"label": label if k == 0 else None, "x": x.tolist(), "y": y.tolist(),
                           "colour": colour, "width": 1.5})
    ch.figure("fighting", title="How much changes hands",
              x={"label": "iteration"}, y={"label": "nodes taken per 100 agents per game"},
              series=fights,
              caption="One line per world, each a mean over 50 iterations.",
              recipe=recipe(ALLY_RUNS, "each world's `series.jsonl`", [
                  "Count the nodes whose winner was not their owner in each game (`conquests`), per 100 "
                  "agents, and average over consecutive stretches of 50 iterations."]))

    elo_series, table = [], {}
    for k, (label, w) in enumerate(TOURNEYS.items()):
        got = ratings(w)
        if got is None:
            continue
        eras, elo = got
        colour = [GREY, BLUE, VIOLET, RED, BLUE][k]
        elo_series.append({"label": label, "x": eras, "y": elo, "colour": colour, "width": 2,
                           **({"dash": [6, 4]} if k == 4 else {})})
        table[label] = {"eras": eras, "elo": [round(e) for e in elo],
                        "steps": [pair_score(w, a, b) for a, b in zip(eras, eras[1:])]}
    ch.figure("elo", title="How much better than the founders",
              x={"label": "iteration the brains are taken from"}, y={"label": "Elo rating against the founders"},
              series=elo_series, guides=[{"axis": "y", "at": 0, "label": "the founders"}],
              caption=("One world of each kind, seed 1 (seed 3 for the long one), its eras played "
                       "against each other two contests per pair in each era's world."),
              recipe=recipe(ALLY_RUNS, "each world's checkpoints, `ckpt_<iteration>.npz`, every 100 iterations", [
                  "For every pair of eras play four contests, two in each era's world: every agent gets "
                  "a brain drawn from one era or the other, a coin each, mutation is switched off, and "
                  "20 iterations are played (`research/oee/pilots/tourney.py --T 20 --reps 2`).",
                  "Fit one rating per era to the later era's shares (Bradley–Terry, "
                  "`research/oee/pilots/elo.py`), the founders at 0, in Elo points."]))
    ch.number("tournaments", table)
    ch.number("halves_1000_vs_500", {w: pair_score(w, 500, 1000, "tourney_T50_all_halves.json")
                                     for w in ("A10k/allies-s1", "A10k/ctrl-s1")})
    ch.number("turf_1000_vs_500", {w: pair_score(w, 500, 1000, "tourney_T50_all_turf.json")
                                   for w in ("A10k/allies-s1", "A10k/ctrl-s1")})

    worlds = sorted({w for g, _ in ALLY_WORLDS.values() for w in g})
    ch.number("bloc", {w: {t: share_at(rows_of(w), "sideLargest", t) for t in (60, 300, 500, 1000, 2000, 3000)
                           if lived(w) >= t} for w in worlds})
    ch.number("wish", {w: {t: (lambda r: r["ally_wishes"] / (2 * r["edges"]))([x for x in rows_of(w) if x["it"] >= t][0])
                           for t in (1, 300, 1000, 2000) if lived(w) >= t} for w in worlds})
    ch.number("conquest", {w: {"0-100": per_agent(rows_of(w), "conquests", 0, 100),
                               "500-1000": per_agent(rows_of(w), "conquests", 500, 1000),
                               "last 500": per_agent(rows_of(w), "conquests", lived(w) - 500, lived(w) + 1)}
                           for g, _ in FIGHT_WORLDS.values() for w in g})
    bar = {}
    for w in worlds:
        h = [r for r in health_rows(rows_of(w)) if r["it"] > lived(w) - 500]
        bar[w] = {k: float(np.median([r[k] for r in h])) for k in
                  ("coreShare", "nodeCutRisk", "cutRisk", "maxDegreeShare", "meanDegree", "dimension")}
        bar[w]["agents"] = float(np.median([r["nodes"] for r in h]))
    ch.number("health_last_500", bar)


# ---------------------------------------------------------------------------
# Strategies written by hand
# ---------------------------------------------------------------------------

INVADE = {"first": "invade_B2_2000.json", "ten": "invade_B2_2000_ten.json",
          "lone": "invade_B2_2000_lone.json", "single": "invade_B2_2000_single.json"}
STRATEGY_COLOURS = {"resident": GREY, "turtle": GREEN, "raider": RED, "lone": VIOLET, "factory": BLUE}
MANUAL = "python3 research/oee/pilots/manual.py"
INVASION_RUNS = ("The B2 worlds of Experiment 16, `B2-10000-s001` … `-s010`, as their archives keep them "
                 "at iteration 2,000 (`GraphOfLifeRuns/<run>/archive/`). Strategies planted with "
                 f"`{MANUAL}`: in `-s001` and `-s002` every strategy and the control on seeds 1 and 2 "
                 "(`--strategies factory,turtle,raider,resident --seeds 1,2 --T 300 --patch 0.05`); in all "
                 "ten worlds the raider, the factory and the control on seed 1; in `-s001` and `-s002` the "
                 "lone raider on seeds 1 and 2; in `-s002` a single raider, and a single resident, on seeds "
                 "1 to 20 (`--patch 0.0002 --T 200`).")
INVASION_STEPS = [
    "Load the world of iteration 2,000 (`GraphOfLife.from_checkpoint`). On seed *s*, draw an "
    "agent at random and take the agents nearest to it, step by step, until the patch holds the "
    "share asked for (5%, or one agent).",
    "Give each agent of that patch the strategy (`manual.py`), or, for the control, a frozen copy "
    "of the brain of one agent of the patch drawn at random; neither ever mutates.",
    "Play on under the world's own rules and count, after each iteration, the agents carrying "
    "the strategy, as a share of all agents."]


def invasions(which: str) -> List[Dict[str, Any]]:
    path = os.path.join(PILOTS, INVADE[which])
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def end_share(r: Dict[str, Any], T: int) -> float:
    """A planting's share of the agents after T iterations (0 if it died out before)."""
    s = r["series"]
    at = [p for p in s if p["t"] == T]
    return at[0]["mine"] / max(1, at[0]["agents"]) if at else 0.0


@chapter
def handmade(ch: F.Chapter) -> None:
    first = invasions("first")
    if not first:
        return
    series_ = []
    for strategy, colour in STRATEGY_COLOURS.items():
        mine = [r for r in first + invasions("lone") if r["strategy"] == strategy]
        for k, r in enumerate(mine):
            s = r["series"]
            series_.append({"label": strategy if k == 0 else None, "x": [p["t"] for p in s],
                            "y": [p["mine"] / max(1, p["agents"]) for p in s], "colour": colour, "width": 1.5,
                            **({"dash": [6, 3]} if r["run"].endswith("s001") else {})})
    ch.figure("shares", title="The share of two evolved worlds held by a strategy",
              x={"label": "iterations after planting"}, y={"label": "share of the agents", "min": 0},
              series=series_,
              caption=("One line per planting: two seeds in each of two worlds, the world of seed 1 "
                       "dashed, of seed 2 solid. Grey is the control, a patch of the residents' own "
                       "brains planted in the same place."),
              recipe=recipe(INVASION_RUNS, "the worlds the archives keep", INVASION_STEPS))
    ch.number("first", [{"run": r["run"], "seed": r["seed"], "strategy": r["strategy"], "patch": r["patch"],
                         "end": end_share(r, 300), "last": r["series"][-1]["t"]} for r in first])
    ten = invasions("ten")
    if ten:
        worlds = sorted({r["run"] for r in ten})
        cats = ["resident", "raider", "factory"]
        groups = [[end_share(r, 300) for r in ten if r["strategy"] == c] for c in cats]
        ch.figure("ten", title="Ten evolved worlds, a patch of 5% each",
                  x={"categories": ["residents (control)", "raider", "factory"]},
                  y={"label": "share of the agents after 300 iterations", "min": 0},
                  series=F.dots(cats, groups, [STRATEGY_COLOURS[c] for c in cats]),
                  caption="One dot per world, the bar the median of ten.",
                  recipe=recipe(INVASION_RUNS, "the worlds the archives keep", INVASION_STEPS))
        ch.number("ten", {w: {c: end_share(next(r for r in ten if r["run"] == w and r["strategy"] == c), 300)
                              for c in cats} for w in worlds})
    lone = invasions("lone")
    if lone:
        ch.number("lone", [{"run": r["run"], "seed": r["seed"], "end": end_share(r, 300),
                            "last": r["series"][-1]["t"]} for r in lone])
    single = invasions("single")
    if single:
        out = {}
        for c in ("raider", "resident"):
            mine = [r for r in single if r["strategy"] == c]
            out[c] = {"plantings": len(mine),
                      "alive_at_200": sum(1 for r in mine if r["series"][-1]["t"] == 200 and r["series"][-1]["mine"] > 0),
                      "agents_at_200": [r["series"][-1]["mine"] if r["series"][-1]["t"] == 200 else 0 for r in mine],
                      "lived": [r["series"][-1]["t"] for r in mine]}
        ch.number("single", out)


# ---------------------------------------------------------------------------
# Rules without numbers
# ---------------------------------------------------------------------------

B3_WORLDS = {"B3a, alliances": (["N/B3a-s1", "N/B3a-s2"], BLUE),
             "B3b, alliances and births among allies": (["N/B3b-s1", "N/B3b-s2"], RED),
             "B3c, no alliances": (["N/B3c-s1", "N/B3c-s2"], GREY)}
B3_RUNS = ("Pilot worlds of `research/oee/pilots/run.py`, kept in `GraphOfLifeRuns/.pilots/N`: "
           "`B3a-s1`, `-s2`, `B3b-s1`, `-s2` and `B3c-s1`, `-s2`, at 10,000 tokens for 1,000 "
           "iterations. Settings: one hidden layer of 16, messages of 8 numbers, no pre-pass, no "
           "handover, revolutions off, and every number of B2 at its original value; prototype "
           "switches `mor=all repair=true memory=true`, with `allies=true` (B3a), `allies=true "
           "birth_allies=true` (B3b), or neither (B3c).")


def _thinned(world: str, key: str, every: int = 10) -> Tuple[List[float], List[float]]:
    rows = rows_of(world)
    its = np.array([r["it"] for r in rows], dtype=float)
    vals = np.array([float(r[key]) for r in rows])
    x, y = F.thin(its, vals, every)
    return x.tolist(), y.tolist()


@chapter
def numberless(ch: F.Chapter) -> None:
    worlds = [w for g, _ in B3_WORLDS.values() for w in g]
    if not all(lived(w) >= 1000 or (rows_of(w) and rows_of(w)[-1]["nodes"] <= 1) for w in worlds):
        return
    agents, ages = [], []
    for label, (group, colour) in B3_WORLDS.items():
        for k, w in enumerate(group):
            x, y = _thinned(w, "nodes")
            agents.append({"label": label if k == 0 else None, "x": x, "y": [max(1.0, v) for v in y],
                           "colour": colour, "width": 1.5})
            alive = [r for r in rows_of(w) if r["nodes"] > 20]
            its = np.array([r["it"] for r in alive], dtype=float)
            x, y = F.thin(its, np.array([r["meanAge"] for r in alive]), 10)
            ages.append({"label": label if k == 0 else None, "x": x.tolist(), "y": y.tolist(),
                         "colour": colour, "width": 1.5})
    ch.figure("agents", title="Worlds without numbers",
              x={"label": "iteration"}, y={"label": "agents", "log": True, "min": 1},
              series=agents,
              caption="One line per world, each a mean over 10 iterations; a world that died out ends at one agent.",
              recipe=recipe(B3_RUNS, "each world's `series.jsonl`", [
                  "Take the agents after each iteration's game (`nodes`) and average over consecutive "
                  "stretches of 10 iterations."]))
    ch.figure("ages", title="How old the agents are",
              x={"label": "iteration"}, y={"label": "mean age of the agents", "log": True, "min": 1},
              series=ages + [{"label": "B2: everyone dies at 100", "x": [0, 1000], "y": [50, 50],
                              "colour": GREEN, "width": 1, "dash": [4, 4]}],
              caption=("One line per world while it had more than 20 agents, each a mean over 10 "
                       "iterations. In B2 every agent dies at 100, so the mean age settles near 50."),
              recipe=recipe(B3_RUNS, "each world's `series.jsonl`", [
                  "Take the mean age of the agents after each game (`meanAge`) while the world has more "
                  "than 20 agents, and average over stretches of 10 iterations."]))
    out = {}
    for w in worlds:
        rows = rows_of(w)
        last = rows[-1]["it"]
        died = next((r["it"] for r in rows if r["nodes"] <= 20), None)
        tail = [r for r in rows if r["it"] > last - 300]
        h = [r for r in health_rows(tail) if r["nodes"] > 20]
        entry = {"died_at": died, "agents": rows[-1]["nodes"]}
        if h and died is None:
            entry.update({k: float(np.median([r[k] for r in h])) for k in
                          ("meanDegree", "coreShare", "nodeCutRisk", "cutRisk", "maxDegreeShare", "dimension",
                           "leafShare")})
            entry["side"] = float(np.median([r.get("sideLargest", 0) / r["nodes"] for r in h]))
            entry.update({k: float(np.mean([100 * sum(r.get(c, 0) for c in cs) / r["nodes"] for r in tail]))
                          for k, cs in (("births", ("births",)), ("conquests", ("conquests",)),
                                        ("starved", ("starved_r", "starved_g")),
                                        ("cut_off", ("orphaned_r", "orphaned_g")))})
            entry["mean_age"] = float(np.mean([r["meanAge"] for r in tail]))
        out[w] = entry
    ch.number("worlds", out)


# ---------------------------------------------------------------------------
# The switches the world was built with and Another game · The original switches, and other contests
# ---------------------------------------------------------------------------

SWITCHES = {"B1": "S6-B1", "B1, no revolutions": "S7-B1norev", "B2": "S0-B2", "B2, no revolutions": "S1-norev",
            "gifts": "S2-gifts", "gifts, all lapse": "S3-giftslapse", "all lapse": "S4-lapse",
            "estates by wealth": "S5-bytokens"}
GAMES = {"Blotto": "S1-norev", "lottery": "G1-lottery", "by genotype": "G2-genotype", "kin sense": "G3-kinsense",
         "family sense": "G4-familysense", "pooled by family": "G5-familypool"}
SEEDS = (1, 2)
B2_SET = ("hidden_layers=16 message_amount=8 message_prepass=false allow_handover=false "
          "mutate_on_replication=true incumbency=1.5 birth_links=2 mend_on_death=true prune_needs=2 "
          "valence=8 keep_home=0.3 max_age=100 min_child=10")
SWITCH_RUNS = ("Pilot worlds of `research/oee/pilots/run.py` in the engine's own code (`--proto "
               "engine=true`), kept in `GraphOfLifeRuns/.pilots/S`, two seeds each (`-s1`, `-s2`), at "
               "10,000 tokens for 1,000 iterations with a whole world every 100 (`--ckpt 100`): B1 "
               "(`S6-B1`, the settings of the study of the baseline), B1 without revolutions (`S7-B1norev`), "
               f"B2 (`S0-B2`: `--set {B2_SET}`), B2 without revolutions (`S1-norev`), B2 with gifts "
               "(`S2-gifts`: `allow_gifting=true prune_after=reproduction inactive_window=iteration`), "
               "the same with every unused link lapsing (`S3-giftslapse`: also `prune_needs=0`), B2 "
               "with every unused link lapsing (`S4-lapse`) and B2 with estates shared by wealth "
               "(`S5-bytokens`: `redistribution=by_tokens`). The jobs are listed in "
               "`research/oee/pilots/jobs_S.txt`.")
GAME_RUNS = ("Pilot worlds of `research/oee/pilots/run.py` in the engine's own code, kept in "
             "`GraphOfLifeRuns/.pilots/S`, two seeds each, at 10,000 tokens for 1,000 iterations with a "
             f"whole world every 100: B2 without revolutions (`--set {B2_SET} allow_revolutions=false "
             "--proto engine=true`, worlds `S1-norev`), the same with a lottery contest (`contest=lottery`, "
             "`G1-lottery`), with stakes pooled by genotype (`pool=genotype`, `G2-genotype`), with "
             "only the sense of which neighbours carry one's own brain (`kin_sense=true`, `G3-kinsense`), "
             "and with family tags (`tag_rate=0.05`) sensed (`kin_sense=true`, `G4-familysense`) or pooled "
             "(`pool=family`, `G5-familypool`).")
TOURNEY_STEPS = [
    "For every pair of the eras 0, 100, 200, 500 and 1,000 play four contests, two in each era's world: "
    "every agent gets a brain drawn at random from one era or the other, mutation is switched off, and "
    "20 iterations are played under the world's own rules (`research/oee/pilots/tourney.py --T 20 --reps 2`).",
    "Fit one rating per era (Bradley–Terry, `research/oee/pilots/elo.py`), the founders at 0."]


def _set_worlds(table: Dict[str, str]) -> Dict[str, List[str]]:
    return {label: [f"S/{name}-s{s}" for s in SEEDS] for label, name in table.items()}


def _finished(world: str) -> bool:
    rows = rows_of(world)
    return bool(rows) and (rows[-1]["it"] >= 1000 or rows[-1]["nodes"] <= 1)


def _window(world: str, lo: int = 500, hi: int = 1001) -> List[Dict[str, Any]]:
    return [r for r in rows_of(world) if lo <= r["it"] < hi]


def _health(world: str) -> Dict[str, float]:
    h = [r for r in health_rows(_window(world)) if r["nodes"] > 20]
    if not h:
        return {}
    med = lambda key: float(np.median([r[key] for r in h if r.get(key) is not None] or [np.nan]))
    return {"nodeCut": med("nodeCutRisk"), "edgeCut": med("cutRisk"), "core2": med("coreShare"),
            "links": med("meanDegree"), "dimension": med("dimension"), "hub": med("maxDegreeShare")}


def _life(world: str) -> Dict[str, float]:
    rows = [r for r in _window(world) if r["nodes"] > 20]
    if not rows:
        return {}
    rate = lambda cs: float(np.mean([100 * sum(r.get(c, 0) for c in cs) / r["nodes"] for r in rows]))
    deaths = sum(r.get("deaths", 0) for r in rows)
    return {"agents": float(np.median([r["nodes"] for r in rows])), "births": rate(("births",)),
            "conquests": rate(("conquests",)), "revolutions": rate(("revolutions",)),
            "starved": rate(("starved_r", "starved_g")), "cut_off": rate(("orphaned_r", "orphaned_g")),
            "mean_age": float(np.mean([r["meanAge"] for r in rows])),
            "lifespan": float(sum(r.get("deathAgeSum", 0) for r in rows) / deaths) if deaths else float("nan"),
            "clone_mean": float(np.mean([r["cloneMean"] for r in rows if "cloneMean" in r])),
            "genotypes": float(np.mean([r["genotypes"] for r in rows]))}


def _tourney(world: str) -> Dict[str, Any]:
    got = ratings(world)
    if got is None:
        return {}
    eras, elo = got
    return {"eras": eras, "elo": elo, "1000_vs_500": pair_score(world, 500, 1000),
            "1000_vs_100": pair_score(world, 100, 1000),
            "steps": [pair_score(world, a, b) for a, b in zip(eras, eras[1:])]}


def _bar_grid(ch: F.Chapter, name: str, labels: List[str], worlds: Dict[str, List[str]], title: str,
              caption: str, runs: str) -> None:
    from book_chapters.oee import SPACE_BAR, dots
    measured = {label: [_health(w) for w in worlds[label]] for label in labels}
    panels = []
    for stat, words, lo, hi, log in SPACE_BAR:
        values = {label: [max(m[stat], 1e-4) if log else m[stat] for m in measured[label] if m]
                  for label in labels}
        guides = [{"axis": "y", "at": v, "label": "the bar"} for v in (lo, hi) if v is not None]
        panels.append(dots(stat, labels, values, title=words,
                           y={"label": words, **({"log": True} if log else {"min": 0})}, guides=guides))
    ch.grid(name, panels, columns=2, title=title, caption=caption,
            recipe=recipe(runs, "each world's `series.jsonl`", [
                "Of the rows from iteration 500 to 1,000 that measured the network (every 10th), take "
                "the median of the worst share of the others one agent's loss cuts off (`nodeCutRisk`), "
                "the same for one link (`cutRisk`), the 2-core (`coreShare`), links per agent "
                "(`meanDegree`), the ball dimension and the best-connected agent's share "
                "(`maxDegreeShare`)."]))


@chapter
def switches(ch: F.Chapter) -> None:
    worlds = _set_worlds(SWITCHES)
    every = [w for ws in worlds.values() for w in ws]
    if not all(_finished(w) for w in every):
        return
    labels = list(SWITCHES)
    short = dict(zip(labels, ("B1", "B1 −rev", "B2", "B2 −rev", "gifts", "gifts, lapse", "lapse", "by wealth")))
    worlds_short = {short[l]: worlds[l] for l in labels}
    _bar_grid(ch, "bar", list(short.values()), worlds_short, "Each switch against the health bar",
              ("One dot per world, at its median over iterations 500 to 1,000; the bar is the median of "
               "the two; dashed lines are the health bar of [How to tell a world that keeps evolving](how-to-tell-a-world-that-keeps-evolving.md). *−rev*: no revolutions; "
               "*gifts, lapse*: gifts, and every unused link lapses; *lapse*: every unused link lapses; "
               "*by wealth*: estates shared by wealth. Cut shares and the best-connected agent's share "
               "are drawn on logarithmic axes, a cut of nothing at 0.0001."),
              SWITCH_RUNS)
    from book_chapters.oee import dots
    life = {label: [_life(w) for w in worlds[label]] for label in labels}
    sl = list(short.values())
    lf = {short[l]: life[l] for l in labels}
    panels = [dots("conquests", sl, {l: [m["conquests"] for m in lf[l] if m] for l in sl},
                   title="nodes taken per 100 agents per game", y={"label": "per 100 agents", "min": 0}),
              dots("births", sl, {l: [m["births"] for m in lf[l] if m] for l in sl},
                   title="births per 100 agents per iteration", y={"label": "per 100 agents", "min": 0}),
              dots("age", sl, {l: [m["mean_age"] for m in lf[l] if m] for l in sl},
                   title="mean age of the agents", y={"label": "iterations", "log": True}),
              dots("agents", sl, {l: [m["agents"] for m in lf[l] if m] for l in sl},
                   title="agents", y={"label": "agents", "min": 0})]
    ch.grid("life", panels, columns=2, title="How each switch changes a world's life",
            caption=("One dot per world, the mean or median over iterations 500 to 1,000; the bar is the "
                     "median of the two. Labels as in the figure below."),
            recipe=recipe(SWITCH_RUNS, "each world's `series.jsonl`", [
                "Of the rows from iteration 500 to 1,000: the nodes taken (`conquests`) and the births per "
                "100 agents, averaged; the mean age (`meanAge`), averaged; the agents (`nodes`), median."]))
    played = {label: [_tourney(w) for w in worlds[label]] for label in labels}
    if all(p for ps in played.values() for p in ps):
        panels = [dots("elo1000", labels, {l: [p["elo"][-1] for p in played[l]] for l in labels},
                       title="rating of iteration 1,000 against the founders", y={"label": "Elo"},
                       guides=[{"axis": "y", "at": 0, "label": "the founders"}]),
                  dots("late", labels, {l: [p["1000_vs_500"] for p in played[l]] for l in labels},
                       title="iteration 1,000 against iteration 500", y={"label": "share of the world"},
                       guides=[{"axis": "y", "at": 0.5, "label": "a tie"}])]
        ch.grid("elo", panels, columns=1, title="Does any switch keep evolution going?",
                caption=("One dot per world. Top: the rating of the brains of iteration 1,000 against the "
                         "founders'. Bottom: their share of a world contested with the brains of "
                         "iteration 500 of the same world."),
                recipe=recipe(SWITCH_RUNS, "each world's checkpoints, `ckpt_<iteration>.npz`", TOURNEY_STEPS))
        ch.number("tournaments", played)
    ch.number("life", life)
    ch.number("health", {label: [_health(w) for w in worlds[label]] for label in labels})


@chapter
def games(ch: F.Chapter) -> None:
    worlds = _set_worlds(GAMES)
    every = [w for ws in worlds.values() for w in ws]
    if not all(_finished(w) for w in every):
        return
    labels = list(GAMES)
    colours = dict(zip(labels, (GREY, RED, BLUE, GREEN, VIOLET, ORANGE)))
    for name, key, title, ylabel, log in (("fighting", "conquests", "How much changes hands",
                                           "nodes taken per 100 agents per game", False),
                                          ("groups", "cloneMean", "How large a group of clones is",
                                           "agents in an agent's group of clones, on average", True)):
        series_ = []
        for label in labels:
            for k, w in enumerate(worlds[label]):
                rows = [r for r in rows_of(w) if key in r]
                its = np.array([r["it"] for r in rows], dtype=float)
                vals = np.array([(100 * r[key] / r["nodes"]) if key == "conquests" else r[key] for r in rows])
                x, y = F.thin(its, vals, 50 if key == "conquests" else 10)
                series_.append({"label": label if k == 0 else None, "x": x.tolist(), "y": y.tolist(),
                                "colour": colours[label], "width": 1.5})
        ch.figure(name, title=title, x={"label": "iteration"}, y={"label": ylabel, **({"log": True} if log else {"min": 0})},
                  series=series_,
                  caption=("One line per world, means over 50 iterations." if key == "conquests" else
                           "One line per world, every 10th iteration: the agents of an agent's connected "
                           "group of neighbours with the same brain, averaged over agents."),
                  recipe=recipe(GAME_RUNS, "each world's `series.jsonl`", [
                      "Count the nodes whose winner was not their owner in each game, per 100 agents."
                      if key == "conquests" else
                      "Every 10th iteration, join neighbours with the same brain and take the size of each "
                      "agent's group, averaged over agents (`organisation`, `cloneMean`)."]))
    played = {label: [_tourney(w) for w in worlds[label]] for label in labels}
    if all(p for ps in played.values() for p in ps):
        series_ = []
        for label in labels:
            for k, p in enumerate(played[label]):
                series_.append({"label": label if k == 0 else None, "x": p["eras"], "y": p["elo"],
                                "colour": colours[label], "width": 2})
        ch.figure("elo", title="How much better than the founders",
                  x={"label": "iteration the brains are taken from"}, y={"label": "Elo rating against the founders"},
                  series=series_, guides=[{"axis": "y", "at": 0, "label": "the founders"}],
                  caption="One line per world, its eras played against each other, four contests per pair.",
                  recipe=recipe(GAME_RUNS, "each world's checkpoints, `ckpt_<iteration>.npz`", TOURNEY_STEPS))
        ch.number("tournaments", played)
    ch.number("life", {label: [_life(w) for w in worlds[label]] for label in labels})
    ch.number("health", {label: [_health(w) for w in worlds[label]] for label in labels})
