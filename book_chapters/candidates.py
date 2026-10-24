"""
The candidates, played on (the long runs).

Batch LC: every candidate of Healthy and rich — a world that met every mark of the health bar with a
diversity score above the median of B2's worlds at iteration 1,000 — played on to 10,000
iterations in a copy of its own (GraphOfLifeRuns/.pilots/LC/). Each is read at 1,000 and at
10,000 as Healthy and rich reads every world (research/atlas/at.py), its space as What folds a world reads it
(research/atlas/space.py), and its brains of three moments against those of earlier ones, a
lifetime long (research/oee/checks/rejudge.py). Experiment 17's ten mixed worlds and B2's ten
worlds played on to 10,000 are the comparison. The two mixed worlds of batch Q are Experiment
17's worlds of seeds 1 and 2, agent for agent, and are counted there (Healthy and rich).

The computer was restarted on 18 October 2026 with six long runs unfinished. Two use engine settings
only and were played by the lab from iteration 0 (Experiment 20); they are the pilots' worlds game
for game (the twin check below), and are read from the lab's runs. The other four use rules that so
far exist only in the pilots' prototype and wait for Experiment 23.
"""
from __future__ import annotations

import importlib.util
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import book_figures as F
from book_figures import chapter, recipe
from book_chapters.combos import WORDS, _contests
from book_chapters.common import BLUE, GREEN, GREY, ORANGE, RED, VIOLET, YELLOW

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILOTS = os.path.join(ROOT, "GraphOfLifeRuns", ".pilots")
ATLAS = os.path.join(PILOTS, "atlas")
LC = os.path.join(PILOTS, "LC")
#: Experiment 17's worlds, and the long runs that are the same worlds.
E17 = [f"B2-10000-4af4a3-s{s:03d}" for s in range(1, 11)]
SAME_AS_E17 = {"Q-mix-s1", "Q-mix-s2"}
#: B2's own ten worlds, played on to 10,000 as pilots (batch L).
B2_LONG = [f"L/B2-long-s{s:03d}" for s in range(1, 11)]
#: The lab stops a world as extinct when a game leaves this many agents or fewer; a pilot plays on.
EXTINCT = 20
#: Long runs the lab finished from iteration 0 (Experiment 20): the pilot's name and the lab's run.
REPLAYED = {"K-reach-random-forced-s1": "B2-10000-e10e0d-s001", "Y-mix-keepone-s1": "B2-10000-82bd1e-s001"}
#: Long runs whose rules are not yet the engine's: Experiment 23 is to play them.
PENDING = {"Y-mix-lottery-s2", "Y-mix-unittoken-s2", "Y-mix-came-s1", "Y-mix-familysense-s1"}
#: The steps judged: the brains of the later moment against those of the earlier.
STEPS = [((500, 1000), "1,000/500"), ((2000, 5000), "5,000/2,000"), ((5000, 10000), "10,000/5,000")]
RUNS = ("Batch LC's long runs (`GraphOfLifeRuns/.pilots/LC/`): each candidate of [Healthy and rich](healthy-and-rich.md), B2 at 10,000 tokens with "
        "the rules of the batch it came from, played on in a copy of its own from its checkpoint to 10,000 "
        "iterations (`research/oee/pilots/run.py --resume`), two of them finished by the lab from iteration 0 "
        "(Experiment 20: `B2-10000-e10e0d-s001`, `B2-10000-82bd1e-s001`); Experiment 17's ten mixed worlds "
        "(`B2-10000-4af4a3-s001` … `s010`) and B2's ten worlds played on to 10,000 "
        "(`GraphOfLifeRuns/.pilots/L/B2-long-s001` … `s010`).")


def _module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source(name: str) -> str:
    """The world a long run went on from: LC/Y-mix-poorest-s1 is Y/mix-poorest-s1."""
    batch, rest = name.split("-", 1)
    return f"{batch}/{rest}"


def words(name: str) -> str:
    """A long run's rules, in the pages' words."""
    rule = name.rsplit("-s", 1)[0]
    fixed = {"W-norev-mix": "mixing, no revolutions", "W-norev-linktoken": "links hold a token, no revolutions",
             "S-G4-familysense": "a sense of family, no revolutions",
             "U-U-BDFG": "no revolutions, no random inputs, links hold a token, mixing",
             "M-mix-children": "mixing, estates to own children",
             "K-reach-random-forced": "links to strangers"}
    if rule in fixed:
        return fixed[rule]
    if rule.startswith("Y-mix-"):
        return "mixing, " + WORDS.get(rule[len("Y-mix-"):], rule[len("Y-mix-"):])
    return rule


def _readings(at: int) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
    """Every world read at one iteration, and its typical distance over its random twin's."""
    profile = _module("profile", os.path.join(ROOT, "research", "atlas", "profile.py"))
    space = _module("space_atlas", os.path.join(ROOT, "research", "atlas", "space.py"))
    ref = profile.reference()
    worlds = {}
    for w in json.load(open(os.path.join(ATLAS, f"at{at}.json")))["worlds"]:
        if "bar" in w:
            worlds[w["id"]] = {**w, "diversity": profile.diversity(w, ref)[1]}
    wide = {}
    path = os.path.join(ATLAS, f"space_at{at}.json")
    for row in json.load(open(path)) if os.path.exists(path) else []:
        try:
            wide[row["id"]] = float(space.typical_distance(row))
        except (KeyError, ValueError, ZeroDivisionError):
            pass
    return worlds, wide


def at_end(name: str) -> str:
    """Where a long run is read at 10,000: the lab's run for a replayed one, else its pilot copy."""
    return REPLAYED.get(name, f"LC/{name}")


def _pilot_rows(name: str) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in open(os.path.join(LC, name, "series.jsonl"))]


def _lab_meta(run: str) -> Dict[str, Any]:
    return json.load(open(os.path.join(ROOT, "GraphOfLifeRuns", run, "meta.json")))


def died_at(name: str) -> Optional[int]:
    """The first iteration a long run (with the batch it came from) ended with EXTINCT agents or fewer."""
    if name in REPLAYED:
        meta = _lab_meta(REPLAYED[name])
        return int(meta["iteration"]) if meta.get("status") == "extinct" else None
    for r in _pilot_rows(name):
        if r.get("nodes") is not None and r["nodes"] <= EXTINCT:
            return int(r["it"])
    return None


def sizes_of(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """A long run's agents after each game, iteration by iteration."""
    if name in REPLAYED:
        its, nodes = F.series(REPLAYED[name], "nodes", 2)
        return its + 1, nodes                       # the lab counts the game of iteration i; a pilot's row is i + 1
    rows = _pilot_rows(name)
    return (np.array([r["it"] for r in rows], float), np.array([r.get("nodes", np.nan) for r in rows], float))


def twin(name: str) -> Dict[str, Any]:
    """How far a replayed world's agents and links after each game match its pilot's."""
    lab = {}
    with open(os.path.join(ROOT, "GraphOfLifeRuns", REPLAYED[name], "stats.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("phase") == 2:
                lab[r["iteration"] + 1] = (r["nodes"], r["edges"])
    pilot = {r["it"]: (r["nodes"], r["edges"]) for r in _pilot_rows(name)}
    common = sorted(set(lab) & set(pilot))
    first = next((it for it in common if lab[it] != pilot[it]), None)
    return {"compared": len(common), "pilot_end": max(pilot), "first_difference": first}


def held(w: Dict[str, Any], b2: List[Dict[str, Any]], died: Optional[int]) -> bool:
    """Healthy and rich's test of a long run: it lived, meets every mark, is more diverse than B2's median world and
    poorer than none of B2's. A world that fell to the lab's extinction line has died, though a pilot plays on
    from its last agent."""
    return (died is None and w["marks"] == 7
            and w["diversity"] > float(np.median([x["diversity"] for x in b2]))
            and w["share_at_most_three"] < min(x["share_at_most_three"] for x in b2))


def _lc_contests() -> Dict[Tuple[str, Tuple[int, int]], float]:
    """Each long run's later-era share for each step: the mean of its contests (pilots by name, lab runs by id)."""
    by: Dict[Tuple[str, Tuple[int, int]], List[float]] = {}
    for name in ("lc_T100.json", "e21_T100.json"):
        path = os.path.join(PILOTS, name)
        for r in json.load(open(path)) if os.path.exists(path) else []:
            if r.get("share") is not None:
                by.setdefault((r["world"].split("/")[-1], (r["a"], r["b"])), []).append(r["share"])
    return {k: float(np.mean(v)) for k, v in by.items()}


@chapter
def candidates(ch: F.Chapter) -> None:
    if not os.path.isdir(LC) or not os.path.exists(os.path.join(ATLAS, "at10000.json")):
        return
    at1, wide1 = _readings(1000)
    at10, wide10 = _readings(10000)
    names = sorted(n for n in os.listdir(LC) if os.path.exists(os.path.join(LC, n, "config.json"))
                   and n not in SAME_AS_E17 and n not in PENDING)
    done = [n for n in names if at_end(n) in at10 and source(n) in at1]
    if len(done) < len(names):
        return
    b2 = [at10[w] for w in B2_LONG if w in at10]
    e17 = [at10[w] for w in E17 if w in at10]
    died = {n: died_at(n) for n in done}
    keep = {n: held(at10[at_end(n)], b2, died[n]) for n in done}
    fate = {n: "held" if keep[n] else "died" if died[n] is not None else "fell back" for n in done}
    colour_of = {"held": GREEN, "fell back": VIOLET, "died": RED}

    # 1. Where each candidate went, on the map of riches and diversity.
    series = [{"label": "B2, played on to 10,000", "x": [w["share_at_most_three"] for w in b2],
               "y": [w["diversity"] for w in b2], "points": True, "size": 6, "colour": YELLOW, "alpha": 0.8},
              {"label": "Experiment 17 (mixing) at 10,000", "x": [w["share_at_most_three"] for w in e17],
               "y": [w["diversity"] for w in e17], "points": True, "size": 6, "colour": BLUE, "alpha": 0.8}]
    for k, n in enumerate(done):
        a, b = at1[source(n)], at10[at_end(n)]
        series.append({"label": None, "x": [a["share_at_most_three"], b["share_at_most_three"]],
                       "y": [a["diversity"], b["diversity"]], "colour": GREY, "width": 1.0, "alpha": 0.7})
    series.append({"label": "a candidate at 1,000", "x": [at1[source(n)]["share_at_most_three"] for n in done],
                   "y": [at1[source(n)]["diversity"] for n in done], "points": True, "size": 4, "colour": GREY,
                   "alpha": 0.9})
    for kind, label in (("held", "at 10,000, held"), ("fell back", "at 10,000, fell back"),
                        ("died", "at 10,000, after dying out")):
        mine = [n for n in done if fate[n] == kind]
        if mine:
            series.append({"label": label, "x": [at10[at_end(n)]["share_at_most_three"] for n in mine],
                           "y": [at10[at_end(n)]["diversity"] for n in mine], "points": True, "size": 7,
                           "colour": colour_of[kind], "alpha": 0.9})
    ch.figure("paths", title="From 1,000 iterations to 10,000",
              x={"label": "agents with three tokens or fewer (share)", "min": 0, "max": 1},
              y={"label": "diversity score", "min": 0, "max": 1}, series=series,
              caption=("Each line is a candidate, from where it stood at iteration 1,000 (small grey dot) to where "
                       "it stood at 10,000 (large dot): green if it still met every mark of the health bar, was "
                       "more diverse than B2's median world and poorer than none of B2's worlds — [Healthy and rich](healthy-and-rich.md)'s test "
                       "of a long run — violet if it fell back, red if it had died out on the way (20 agents or "
                       "fewer, the lab's line) and was refilled from its last agents. Yellow: B2's ten worlds "
                       "played on to 10,000; blue: Experiment 17's mixed worlds that lived to 10,000. The "
                       "author's picture is the top left: agents that differ, and few of them poor."),
              recipe=recipe(RUNS, "each world's checkpoint at 1,000 and at 10,000 (`research/atlas/at.py --at 1000` "
                                  "and `--at 10000`, `GraphOfLifeRuns/.pilots/atlas/at1000.json` and "
                                  "`at10000.json`)",
                            ["Read each world as `research/atlas/profile.py` reads every world.",
                             "Rank its four spreads among the frozen reference worlds; the diversity score is "
                             "their mean."]))
    ch.number("fate", fate)
    ch.number("twins", {n: twin(n) for n in REPLAYED if n in done})
    ch.number("pending", sorted(PENDING))
    ch.number("died_at", {n: died[n] for n in done if died[n] is not None})
    ch.number("worlds", {n: {"rules": words(n),
                             "at_1000": {k: at1[source(n)][k] for k in ("agents", "share_at_most_three", "marks",
                                                                        "diversity", "dimension")},
                             "at_10000": {k: at10[at_end(n)][k] for k in ("agents", "share_at_most_three", "marks",
                                                                          "diversity", "dimension")},
                             "wide": [wide1.get(source(n)), wide10.get(at_end(n))]} for n in done})

    # 2. How large each world was, iteration by iteration.
    lines = []
    for n in done:
        its, nodes = F.thin(*sizes_of(n), 50)
        lines.append({"label": None, "x": its.tolist(), "y": nodes.tolist(),
                      "colour": colour_of[fate[n]], "width": 1.1, "alpha": 0.85})
    ch.figure("sizes", title="How large each candidate grew", x={"label": "iteration"},
              y={"label": "agents", "log": True}, series=lines,
              caption=("Each line is a candidate: its agents after each game, means over 50 iterations, on a "
                       "logarithmic axis. Green: the candidates that held at 10,000 iterations; violet: those that "
                       "fell back; red: those that died out on the way and were refilled from their last agents."),
              recipe=recipe(RUNS, "each long run's `series.jsonl`", ["Read the agents after each game.",
                                                                    "Average over stretches of 50 iterations."]))
    ch.number("sizes", {n: [at1[source(n)]["agents"], at10[at_end(n)]["agents"]] for n in done})

    # 3. Space: the typical distance over the random twin's, at 1,000 and at 10,000.
    pairs = [n for n in done if wide1.get(source(n)) is not None and wide10.get(at_end(n)) is not None]
    space = [{"label": None, "x": [0, 1], "y": [wide1[source(n)], wide10[at_end(n)]],
              "colour": colour_of[fate[n]], "width": 1.1, "alpha": 0.8} for n in pairs]
    space += [{"label": "B2, played on to 10,000", "x": [1.08] * len([w for w in B2_LONG if w in wide10]),
               "y": [wide10[w] for w in B2_LONG if w in wide10], "points": True, "size": 5, "colour": YELLOW},
              {"label": "Experiment 17 at 10,000", "x": [1.12] * len([w for w in E17 if w in wide10]),
               "y": [wide10[w] for w in E17 if w in wide10], "points": True, "size": 5, "colour": BLUE}]
    ch.figure("space", title="Room, at 1,000 iterations and at 10,000",
              x={"categories": ["iteration 1,000", "iteration 10,000"]},
              y={"label": "typical distance over the random twin's", "min": 0.9}, series=space,
              guides=[{"axis": "y", "at": 1.0, "label": "as near as a random network"}],
              caption=("Each line is a candidate: the steps within which a typical agent's ball holds half its "
                       "world, over the same for its random twin ([What folds a world](how-many-dimensions-does-a-world-have.md#what-folds-a-world)), at iteration 1,000 and at 10,000. "
                       "Green: the candidates that held; violet: those that fell back; red: those that died out on "
                       "the way. Beside the right end: B2's ten worlds played on to 10,000 (yellow) and Experiment "
                       "17's (blue)."),
              recipe=recipe(RUNS, "each world's checkpoint at 1,000 and 10,000 (`research/atlas/space.py --at 1000` "
                                  "and `--at 10000`)",
                            ["Count, from a sample of agents, how many others lie within each number of steps.",
                             "Find the steps within which the typical ball holds half the world, between whole "
                             "steps by the logarithm of the ball.",
                             "Do the same on a random network with every agent's links kept; divide."]))
    ch.number("space", {n: [wide1[source(n)], wide10[at_end(n)]] for n in pairs})

    # 4. Does evolution go on? The brains of three moments against those of earlier ones.
    lc = _lc_contests()
    first = _contests()
    e17_path = os.path.join(PILOTS, "e17_T100.json")
    e17_c: Dict[Tuple[int, int], List[float]] = {}
    by: Dict[Tuple[str, Tuple[int, int]], List[float]] = {}
    for r in json.load(open(e17_path)) if os.path.exists(e17_path) else []:
        if r.get("share") is not None:
            by.setdefault((r["world"], (r["a"], r["b"])), []).append(r["share"])
    for (w, pair), v in by.items():
        e17_c.setdefault(pair, []).append(float(np.mean(v)))
    cats, groups, colours, steps = [], [], [], {}
    for pair, label in STEPS:
        # A world that died out and refilled from its last agents has no eras after its death to judge.
        mine = [first.get(source(n)) if pair == (500, 1000) else lc.get((at_end(n).split("/")[-1], pair))
                for n in done if died[n] is None or pair[1] < died[n]]
        mine = [v for v in mine if v is not None]
        steps[label] = {"candidates": mine, "E17": e17_c.get(pair, [])}
        for who, values, colour in (("Experiment 17", e17_c.get(pair, []), GREY), ("candidates", mine, ORANGE)):
            if values:
                cats.append(f"{who} · {label}")
                groups.append(values)
                colours.append(colour)
    ch.figure("evolution", title="Do later brains beat earlier ones?", x={"categories": cats},
              y={"label": "the later brains' share of a contested world", "min": 0, "max": 1},
              series=F.dots(cats, groups, colours), guides=[{"axis": "y", "at": 0.5, "label": "no better"}],
              caption=("Each dot is a world: the share of a contested world the later moment's brains hold after "
                       "100 iterations against the earlier moment's, the mean of two contests, one in each moment's "
                       "world; the bar is the median world. 5,000/2,000: the brains of iteration 5,000 against "
                       "those of 2,000. Orange: the candidates (the first step judged on the world each came from; a "
                       "world that died out on the way has the steps before its death only); grey: Experiment 17's "
                       "mixed worlds."),
              recipe=recipe(RUNS, "`GraphOfLifeRuns/.pilots/lc_T100.json` (`research/oee/checks/lc_rejudge_worlds.txt`), "
                                  "the batches' own contests for 1,000/500, and `e17_T100.json`",
                            ["Seat the brains of the two moments on alternate agents of each moment's world and "
                             "play 100 iterations (`research/oee/checks/rejudge.py`).",
                             "Count the agents the later moment's brains hold, as a share of all."]))
    ch.number("evolution", steps)
