"""
Chapter 93 · Five more seeds for the rules that held (Experiment 21).

Meta X's second step for the long runs: each rule whose candidate held over 10,000 iterations —
mixing with a fast change of brains, with no random inputs, with no revolutions — in five new
worlds (seeds 3 to 7), beside Experiment 17's mixed worlds of the same seeds. Each world is read at
10,000 as Meta X reads every world (research/atlas/at.py) and judged by Meta X's test of a long
run: it lives, meets every mark of the health bar, is more diverse than B2's median world at
10,000 and poorer than none of B2's ten worlds there (B2 played on to 10,000, batch L).
"""
from __future__ import annotations

import importlib.util
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import book_figures as F
from book_figures import chapter, recipe
from book_chapters.common import BLUE, GREEN, GREY, ORANGE, RED, VIOLET, YELLOW

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(ROOT, "GraphOfLifeRuns")
PILOTS = os.path.join(RUNS_DIR, ".pilots")
ATLAS = os.path.join(PILOTS, "atlas")
SEEDS = list(range(3, 8))
END = 10000
#: Each rule of the experiment: its words, its runs' family, its colour. The first is the control.
RULES = [("mixing alone (Experiment 17)", "B2-10000-4af4a3", GREY),
         ("fast brain change", "B2-10000-0df8be", ORANGE),
         ("no random inputs", "B2-10000-1109f3", BLUE),
         ("no revolutions", "B2-10000-1f9b27", YELLOW)]
B2_LONG = [f"L/B2-long-s{s:03d}" for s in range(1, 11)]
STEPS = [((500, 1000), "1,000/500"), ((2000, 5000), "5,000/2,000"), ((5000, 10000), "10,000/5,000")]
FATE = {"passed": GREEN, "failed": VIOLET, "died": RED}
RUNS = ("The twenty runs of Experiment 21 — B2 at 10,000 tokens with brains mixed at birth, alone (Experiment 17's "
        "runs `B2-10000-4af4a3-s003` … `s007`) or with a mutation probability of 0.8 (`B2-10000-0df8be-…`), no random "
        "inputs (`B2-10000-1109f3-…`) or no revolutions (`B2-10000-1f9b27-…`), seeds 3 to 7, 10,000 iterations "
        "(`python3 gol_lab.py run E21`) — and B2's ten worlds played on to 10,000 "
        "(`GraphOfLifeRuns/.pilots/L/B2-long-s001` … `s010`).")


def _module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _meta(run: str) -> Dict[str, Any]:
    return json.load(open(os.path.join(RUNS_DIR, run, "meta.json")))


def died_at(run: str) -> Optional[int]:
    """The iteration a run died out at, if it did by END (the lab stops a world at 20 agents or fewer)."""
    m = _meta(run)
    return int(m["iteration"]) if m.get("status") == "extinct" and m.get("iteration", 0) <= END else None


def _contests(name: str) -> Dict[Tuple[str, Tuple[int, int]], float]:
    """Each world's later-era share for each pair of eras: the mean of its contests."""
    by: Dict[Tuple[str, Tuple[int, int]], List[float]] = {}
    path = os.path.join(PILOTS, name)
    for r in json.load(open(path)) if os.path.exists(path) else []:
        if r.get("share") is not None:
            by.setdefault((r["world"], (r["a"], r["b"])), []).append(r["share"])
    return {k: float(np.mean(v)) for k, v in by.items()}


@chapter
def fiveseeds(ch: F.Chapter) -> None:
    runs = [f"{fam}-s{s:03d}" for _, fam, _ in RULES for s in SEEDS]
    if not all(os.path.exists(os.path.join(RUNS_DIR, r, "meta.json")) for r in runs):
        return
    if any(_meta(r).get("status") == "running" for r in runs):
        return
    profile = _module("profile", os.path.join(ROOT, "research", "atlas", "profile.py"))
    space = _module("space_atlas", os.path.join(ROOT, "research", "atlas", "space.py"))
    ref = profile.reference()
    at = {w["id"]: w for w in json.load(open(os.path.join(ATLAS, "at10000.json")))["worlds"] if "bar" in w}
    wide = {}
    for row in json.load(open(os.path.join(ATLAS, "space_at10000.json"))):
        try:
            wide[row["id"]] = float(space.typical_distance(row))
        except (KeyError, ValueError, ZeroDivisionError):
            pass
    b2 = [at[w] for w in B2_LONG]
    div_line = float(np.median([profile.diversity(w, ref)[1] for w in b2]))
    poor_line = min(w["share_at_most_three"] for w in b2)

    worlds: Dict[str, Dict[str, Any]] = {}
    for rule, fam, _ in RULES:
        for s in SEEDS:
            run = f"{fam}-s{s:03d}"
            died = died_at(run)
            w = at.get(run) if died is None else None
            row = {"rule": rule, "seed": s, "died": died}
            if w is not None:
                d = profile.diversity(w, ref)[1]
                row.update(agents=w["agents"], poor=w["share_at_most_three"], marks=w["marks"], diversity=d,
                           dimension=w["dimension"], missing=[k for k, v in w["bar"].items() if not v],
                           wide=wide.get(run))
                row["fate"] = ("passed" if w["marks"] == 7 and d > div_line and w["share_at_most_three"] < poor_line
                               else "failed")
            else:
                row["fate"] = "died"
            worlds[run] = row
    passed = {rule: sum(1 for w in worlds.values() if w["rule"] == rule and w["fate"] == "passed")
              for rule, _, _ in RULES}
    ch.number("lines", {"diversity": div_line, "poor": poor_line})
    ch.number("worlds", worlds)
    ch.number("passed", passed)

    # 1. The verdict: each world at 10,000 on the test's three counts.
    cats = [rule for rule, _, _ in RULES]

    def panel(title, key, label, guide, lo=0.0, hi=1.0, guide_label=""):
        series = []
        for fate, colour in FATE.items():
            xs, ys = [], []
            for k, rule in enumerate(cats):
                mine = [w for w in worlds.values() if w["rule"] == rule and w["fate"] == fate]
                for w in mine:
                    xs.append(k - 0.25 + 0.5 * (w["seed"] - SEEDS[0]) / (len(SEEDS) - 1))   # seeds left to right
                    ys.append(hi if fate == "died" else w[key])
            if xs:
                series.append({"label": {"passed": "passed the test", "failed": "failed it",
                                         "died": "died out (drawn at the top)"}[fate] if key == "poor" else None,
                               "x": xs, "y": ys, "points": True, "size": 7, "colour": colour, "alpha": 0.9})
        return {"title": title, "x": {"categories": cats}, "y": {"label": label, "min": lo, "max": hi},
                "series": series, "guides": [{"axis": "y", "at": guide, "label": guide_label}]}

    ch.grid("verdict", [
        panel("Agents with three tokens or fewer, at 10,000", "poor", "share of the agents", poor_line,
              guide_label="B2's least poor world"),
        panel("Diversity score, at 10,000", "diversity", "diversity score", div_line,
              guide_label="B2's median world"),
        panel("Marks of the health bar met", "marks", "marks (of 7)", 7, lo=0, hi=7.5, guide_label="every mark")],
        columns=1, title="Experiment 21 at iteration 10,000",
        caption=("Each dot is a world at iteration 10,000, seeds 3 to 7 from left to right within each rule: green if "
                 "it passed Meta X's test of a long run — alive, every mark of the health bar met, more diverse than "
                 "B2's median world and poorer than none of B2's ten worlds played on to 10,000 — violet if it failed "
                 "it, red, at the top, if it died out before 10,000. Grey column: plain mixing, Experiment 17's worlds "
                 "of the same seeds."),
        recipe=recipe(RUNS, "each run's whole world kept at iteration 10,000 (`archive/world_010000.npz`, read by "
                            "`research/atlas/at.py --at 10000`, `GraphOfLifeRuns/.pilots/atlas/at10000.json`)",
                      ["Read each world as `research/atlas/profile.py` reads every world.",
                       "Rank its four spreads among the frozen reference worlds; the diversity score is their mean.",
                       "A world that died out (20 agents or fewer) fails."]))

    # 2. Each world through its life.
    lines = []
    for rule, fam, colour in RULES:
        for k, s in enumerate(SEEDS):
            its, nodes = F.series(f"{fam}-s{s:03d}", "nodes", 2)
            keep = its <= END                      # the control's worlds were played on to 50,000 (Experiment 18)
            its, nodes = F.thin(its[keep], nodes[keep], 50)
            lines.append(F.line(its, nodes, rule if k == 0 else None, colour, width=1.1))
    ch.figure("sizes", title="Agents alive, world by world", x={"label": "iteration"},
              y={"label": "agents", "log": True}, series=lines,
              caption=("Each line is one world, its agents after each game (means over 50 iterations), on a "
                       "logarithmic axis; a line that stops early is a world that died out."),
              recipe=recipe(RUNS, "each run's `stats.jsonl`", ["Read the agents after each game (`nodes`, phase 2).",
                                                               "Average over stretches of 50 iterations."]))

    # 3. Does evolution go on? One panel per step, the four rules side by side.
    contests = {**_contests("e17_T100.json"), **_contests("e21_T100.json")}
    short = ["mixing alone", "fast brain change", "no random inputs", "no revolutions"]
    panels, steps = [], {}
    for pair, label in STEPS:
        groups = [[contests[(f"{fam}-s{s:03d}", pair)] for s in SEEDS if (f"{fam}-s{s:03d}", pair) in contests]
                  for _, fam, _ in RULES]
        steps[label] = {rule: values for (rule, _, _), values in zip(RULES, groups)}
        panels.append({"title": f"The brains of {label.split('/')[0]} against those of {label.split('/')[1]}",
                       "x": {"categories": short},
                       "y": {"label": "the later brains' share", "min": 0, "max": 1},
                       "series": F.dots(short, groups, [colour for _, _, colour in RULES]),
                       "guides": [{"axis": "y", "at": 0.5, "label": "no better"}]})
    ch.grid("evolution", panels, columns=1, title="Do later brains beat earlier ones?",
            caption=("Each dot is a world: the share of a contested world the later moment's brains hold after 100 "
                     "iterations against the earlier moment's, the mean of two contests, one in each moment's world; "
                     "the bar is the median world. Mixing alone: Experiment 17's worlds of the same seeds. A world "
                     "that died out has the steps before its death only."),
            recipe=recipe(RUNS, "`GraphOfLifeRuns/.pilots/e21_T100.json` and `e17_T100.json` "
                                "(`research/oee/checks/rejudge.py`, `research/oee/checks/e20_e21_rejudge_worlds.txt`)",
                          ["Seat the brains of the two moments on alternate agents of each moment's world and play "
                           "100 iterations.", "Count the agents the later moment's brains hold, as a share of all."]))
    ch.number("evolution", steps)
