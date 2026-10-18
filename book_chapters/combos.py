# -*- coding: utf-8 -*-
"""
One rule combined with every other (pilot batches T, W, X and Y, 2026-10-16).

Each world of a batch is the fourth corner of a two-by-two: B2, B2 with the
batch's rule, B2 with another rule, and B2 with both, at 10,000 tokens, seeds 1
and 2. Every corner is read at iteration 1,000, as Meta X reads every world
(research/atlas/at.py → GraphOfLifeRuns/.pilots/atlas/at1000.json, diversity on
Meta X's scale), so that worlds played for different lengths compare; and each
world's brains of 1,000 are played against those of 500 over a lifetime (T = 100;
research/oee/checks/rejudge.py → GraphOfLifeRuns/.pilots/batch_TUV_T100.json,
and rejudge_T100.json for batch S's worlds).

    mixcombos   Chapter 82 · Brains mixed at birth, with every other rule (batch Y)
    norevcombos Chapter 88 · Revolutions, with every other rule (batch W)
    handovercombos  Chapter 89 · Handover, with every other rule (batch X)
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

import book_figures as F
from book_figures import chapter, recipe
from book_chapters.common import GREY, ORANGE, VIOLET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILOTS = os.path.join(ROOT, "GraphOfLifeRuns", ".pilots")
AT = os.path.join(PILOTS, "atlas", "at1000.json")
sys.path.insert(0, os.path.join(ROOT, "research", "oee", "pilots"))

#: The rules, in the words of the chapters, by the names the batches give them.
WORDS = {"bytokens": "estates by wealth", "poorest": "estates to the poorest", "newborns": "estates to newborns",
         "parents": "estates to parents", "linktoken": "links hold a token", "helped": "input: who helped",
         "came": "input: came from", "handover": "handover", "nodice": "no random inputs", "noage": "no old age",
         "anychild": "children of any size", "keephalf": "half kept at home", "keepone": "last token kept",
         "grow": "brains that grow", "unittoken": "units hold a token", "mut05": "slow brain change",
         "mut80": "fast brain change", "lapse": "unused links lapse", "lottery": "nodes by lottery",
         "pool": "pooled by genotype", "kinsense": "a sense of kin", "familysense": "a sense of family",
         "familypool": "pooled by family", "mix": "brains mixed at birth", "norev": "no revolutions"}
KEYS = ("share_at_most_three", "diversity", "marks", "agents")


class Batch(NamedTuple):
    """What a chapter on one rule in combination needs to know about its batch."""
    letter: str                              # the batch (its folder in GraphOfLifeRuns/.pilots)
    words: str                               # its rule, as the captions name it: "brains mixed at birth"
    alone: str                               # the reference line of the rule alone: "mixing alone"
    alone_worlds: Tuple[str, str]            # the worlds of the rule alone, seeds 1 and 2 (ids of at1000.json)
    rows: Callable[[], List[Tuple[str, str, str]]]  # every other rule: its name, the world with it alone (a
    #                                          control as make_jobs_TU names it), and the world with both
    alone_contests: Callable[[], List[float]]  # the rule alone's later-era shares, 1,000 against 500
    alone_where: str                         # where its worlds of seeds 1 and 2 come from: "Experiment 17's
    #                                          lab worlds"
    alone_judged: str                        # the worlds whose contests give its median: "the ten mixed worlds
    #                                          of Experiment 17"


def _measured() -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(AT):
        return {}
    return {w["id"]: w for w in json.load(open(AT))["worlds"] if "bar" in w}


def _alone(control: str, seed: int) -> str:
    """A control world's id: a lab family's run of this seed, or a pilot world of this seed."""
    return f"{control[4:]}-s{seed:03d}" if control.startswith("lab ") else f"{control}-s{seed}"


def _pairs(categories: List[str], alone: List[List[float]], both: List[List[float]],
           colours: Tuple[int, int]) -> List[Dict[str, Any]]:
    """Two columns of dots in each category, the rule alone on the left and with the batch's rule on the right."""
    out = []
    for k, _ in enumerate(categories):
        for values, side, colour in ((alone[k], -0.17, colours[0]), (both[k], 0.17, colours[1])):
            v = [x for x in values if x is not None and np.isfinite(x)]
            if not v:
                continue
            out.append({"label": None, "x": [k + side] * len(v), "y": v, "points": True, "size": 6,
                        "colour": colour, "alpha": 0.85})
            m = float(np.mean(v))
            out.append({"label": None, "x": [k + side - 0.13, k + side + 0.13], "y": [m, m], "colour": colour,
                        "width": 3})
    return out


def _contests() -> Dict[str, float]:
    """Each pilot world's later-era share, 1,000 against 500: the mean of its contests, from every batch's file
    of lifetime contests (a batch's later worlds may be judged into a file of their own)."""
    import glob
    by: Dict[str, List[float]] = {}
    for path in sorted(glob.glob(os.path.join(PILOTS, "batch_*_T100.json"))) + [os.path.join(PILOTS,
                                                                                         "rejudge_T100.json")]:
        for r in json.load(open(path)) if os.path.exists(path) else []:
            if (r["a"], r["b"]) == (500, 1000) and r.get("share") is not None:
                by.setdefault(r["world"].split(".pilots/")[-1], []).append(r["share"])
    return {w: float(np.mean(v)) for w, v in by.items()}


def _lab_contests(family: str, name: str = "long_steps.json") -> List[float]:
    """A lab family's worlds' later-era shares, 1,000 against 500, each the mean of its contests."""
    by: Dict[str, List[float]] = {}
    for r in json.load(open(os.path.join(PILOTS, name))):
        world = r.get("run") or r.get("world")
        if world.startswith(family) and world[len(family):].isdigit() and (r["a"], r["b"]) == (500, 1000):
            by.setdefault(world, []).append(r["share"])
    return [float(np.mean(v)) for v in by.values()]


def _combos(ch: F.Chapter, b: Batch) -> None:
    """The batch's rule beside every other: riches, diversity and evolution, each rule alone and with it."""
    measured = _measured()
    rows = []
    for name, control, with_both in b.rows():
        alone = [measured.get(_alone(control, s)) for s in (1, 2)]
        both = [measured.get(f"{with_both}-s{s}") for s in (1, 2)]
        if any(both):
            rows.append((name, [w for w in alone if w], [w for w in both if w], with_both))
    if len(rows) < 20:
        return
    b2 = [measured[f"B2-10000-s00{s}"] for s in (1, 2)]
    ref = [measured[w] for w in b.alone_worlds if w in measured]

    def value(ws: List[Dict[str, Any]], key: str) -> List[float]:
        return [float(w[key]) for w in ws if w.get(key) is not None]

    # The rules whose riches the batch's rule keeps first: sorted by the share of poor agents with it.
    rows.sort(key=lambda r: np.mean(value(r[2], "share_at_most_three")))
    third = -(-len(rows) // 3)
    parts = [rows[i:i + third] for i in range(0, len(rows), third)]
    for key, label, fig, title in (
            ("share_at_most_three", "share of the agents with three tokens or fewer", "riches",
             "Agents holding three tokens or fewer"),
            ("diversity", "diversity score (Meta X)", "diversity", "How much the agents differ")):
        charts = []
        for part in parts:
            cats = [WORDS[name] for name, _, _, _ in part]
            charts.append({"x": {"categories": cats}, "y": {"label": label, "min": 0, "max": 1},
                           "series": _pairs(cats, [value(a, key) for _, a, _, _ in part],
                                            [value(w, key) for _, _, w, _ in part], (GREY, ORANGE)),
                           "guides": [{"axis": "y", "at": float(np.mean(value(b2, key))), "label": "B2"},
                                      {"axis": "y", "at": float(np.mean(value(ref, key))), "label": b.alone}]})
        ch.grid(fig, charts, columns=1, title=title,
                caption=(f"Each grey dot is a world of B2 with one other rule, each orange dot the same rule with "
                         f"{b.words} as well (batch {b.letter}), seeds 1 and 2, read at iteration 1,000; the short "
                         f"bars are the means. The dashed lines are B2 itself and B2 with "
                         f"{b.alone.replace(' alone', '')} alone, the means of the same seeds. The rules are in the "
                         f"order of the orange means."),
                recipe=recipe(f"Batch {b.letter}'s {sum(len(r[2]) for r in rows)} pilot worlds (B2 at 10,000 tokens, "
                              f"{b.words} and one other rule, seeds 1 and 2, `GraphOfLifeRuns/.pilots/{b.letter}/`) "
                              f"and their controls: the same rule alone, from the batch or lab experiment that made "
                              f"it, and, of the same seeds, B2 itself (Experiment 16's lab worlds) and B2 with "
                              f"{b.words} alone ({b.alone_where}).",
                              "each world's checkpoint at iteration 1,000, or the lab's whole world kept then "
                              "(`research/atlas/at.py --at 1000`, `GraphOfLifeRuns/.pilots/atlas/at1000.json`)",
                              ["Read each world as `research/atlas/profile.py` reads every world.",
                               "Rank its four spreads among Meta X's worlds; the diversity score is the mean."]))
    # Does evolution go on? The brains of 1,000 against those of 500, a lifetime long.
    contests = _contests()
    judged = [r for r in rows if any(f"{r[3]}-s{s}" in contests for s in (1, 2))]
    if len(judged) >= 20:
        b2_med = float(np.median(_lab_contests("B2-10000-s")))
        alone_med = float(np.median(b.alone_contests()))
        charts = []
        for part in parts:
            cats = [WORDS[name] for name, _, _, _ in part]
            groups = [[contests[w] for w in (f"{both}-s1", f"{both}-s2") if w in contests]
                      for _, _, _, both in part]
            charts.append({"x": {"categories": cats},
                           "y": {"label": "the later brains' share", "min": 0.2, "max": 1},
                           "series": F.dots(cats, groups, [ORANGE] * len(cats)),
                           "guides": [{"axis": "y", "at": 0.5, "label": "no better"},
                                      {"axis": "y", "at": b2_med, "label": "B2"},
                                      {"axis": "y", "at": alone_med, "label": b.alone}]})
        ch.grid("evolution", charts, columns=1, title="Do the brains of 1,000 beat those of 500?",
                caption=(f"Each dot is a world of batch {b.letter}: the share of a contested world its brains of "
                         f"iteration 1,000 hold after 100 iterations against those of 500, the mean of two contests, "
                         f"one in each era's world; the bars are the medians, the rules in the order of the figures "
                         f"above. The dashed lines are the medians of B2's ten lab worlds and of "
                         f"{b.alone_judged}, played the same way."),
                recipe=recipe(f"Batch {b.letter}'s pilot worlds (`GraphOfLifeRuns/.pilots/{b.letter}/`); B2: "
                              f"Experiment 16's ten runs; {b.alone}: {b.alone_judged}.",
                              "`GraphOfLifeRuns/.pilots/batch_TUV_T100.json`, `rejudge_T100.json`, "
                              "`long_steps.json` and `e17_T100.json` (`research/oee/checks/rejudge.py`)",
                              ["Seat the brains of the two eras on alternate agents of each era's world and play "
                               "100 iterations.", "Count the agents the later era's brains hold, as a share of all."]))
        ch.number("evolution", {"worlds": {f"{both}-s{s}": contests[f"{both}-s{s}"] for _, _, _, both in rows
                                           for s in (1, 2) if f"{both}-s{s}" in contests},
                                "b2_median": b2_med, "alone_median": alone_med})
    ch.number("rows", [{"rule": name, "alone": {k: value(a, k) for k in KEYS},
                        "with_rule": {k: value(w, k) for k in KEYS}} for name, a, w, _ in rows])
    ch.number("references", {"B2": {k: value(b2, k) for k in KEYS}, "alone": {k: value(ref, k) for k in KEYS}})


def _changes(ch: F.Chapter, b: Batch, fig: str, marks: bool = False) -> Optional[Dict[str, Any]]:
    """A compact view of a batch: for each other rule, what adding the batch's rule changes in the share of poor
    agents and in diversity (each seed a dot), with `marks` also in the marks of the health bar, and whether its
    brains of 1,000 beat those of 500 — one row per rule, the rules in the order of the change in the poor
    share."""
    measured, contests = _measured(), _contests()
    rows = []
    for name, control, with_both in b.rows():
        pairs = [(measured.get(_alone(control, s)), measured.get(f"{with_both}-s{s}")) for s in (1, 2)]
        pairs = [(a, w) for a, w in pairs if a and w]
        if pairs:
            judged = [contests[f"{with_both}-s{s}"] for s in (1, 2) if f"{with_both}-s{s}" in contests]
            rows.append((name, pairs, judged))
    if len(rows) < 20:
        return None

    def change(pairs, key):
        return [float(w[key]) - float(a[key]) for a, w in pairs if a.get(key) is not None and w.get(key) is not None]

    rows.sort(key=lambda r: np.mean(change(r[1], "share_at_most_three")))
    cats = [WORDS[name] for name, _, _ in rows]
    series = []
    for key, colour, side, label in (("share_at_most_three", ORANGE, -0.15, "share with three tokens or fewer"),
                                     ("diversity", VIOLET, 0.15, "diversity score")):
        xs, ys = [], []
        for k, (_, pairs, _) in enumerate(rows):
            for v in change(pairs, key):
                xs.append(v)
                ys.append(k + side)
        series.append({"label": label, "x": xs, "y": ys, "points": True, "size": 6, "colour": colour, "alpha": 0.85})
    evolution = [{"label": None, "x": [v for _, _, c in rows for v in c],
                  "y": [k for k, (_, _, c) in enumerate(rows) for _ in c], "points": True, "size": 6,
                  "colour": ORANGE, "alpha": 0.85}]
    b2_med = float(np.median(_lab_contests("B2-10000-s")))
    height = 18 * len(rows)
    health = []
    if marks:
        xs, ys = [], []
        for k, (_, pairs, _) in enumerate(rows):
            for v in change(pairs, "marks"):
                xs.append(v)
                ys.append(k)
        health = [{"title": f"What {b.words} changes in health", "x": {"label": "marks of the health bar (of 7), "
                                                                               "with it minus without it",
                                                                      "min": min(xs + [0]) - 0.4,
                                                                      "max": max(xs + [0]) + 0.4},
                   "y": {"categories": cats}, "height": height,
                   "series": [{"label": None, "x": xs, "y": ys, "points": True, "size": 6, "colour": GREY,
                               "alpha": 0.85}],
                   "guides": [{"axis": "x", "at": 0.0, "label": ""}]}]
    spread = [v for sr in series for v in sr["x"]]
    room = 0.05 * (max(spread) - min(spread)) if spread else 0.05
    ch.grid(fig, [
        {"title": f"What {b.words} changes", "x": {"label": "with it, minus without it",
                                                    "min": min(spread) - room, "max": max(spread) + room},
         "y": {"categories": cats}, "series": series, "height": height,
         "guides": [{"axis": "x", "at": 0.0, "label": ""}]}] + health + [
        {"title": "Do the brains of 1,000 beat those of 500?", "x": {"label": "the later brains' share", "min": 0.2,
                                                                    "max": 1.0},
         "y": {"categories": cats}, "series": evolution, "height": height,
         "guides": [{"axis": "x", "at": 0.5, "label": "no better"}, {"axis": "x", "at": b2_med, "label": "B2"}]}],
        columns=1, title=f"Each rule, with {b.words} and without",
        caption=(f"Each row is one other rule. First: what adding {b.words} to a world of B2 with that rule changes "
                 f"(batch {b.letter}), at iteration 1,000 — in orange, the share of the agents holding three tokens "
                 f"or fewer; in violet, the diversity score; a dot for each seed, 1 and 2. "
                 + ("Then: what it changes in the marks of the health bar the world meets. " if marks else "")
                 + f"Last: the share of a contested world the brains of 1,000 of the world with both rules hold "
                 f"after 100 iterations against those of 500, the mean of two contests, a dot for each seed; the "
                 f"dashed line marked B2 is the median of the lab's ten B2 worlds, played the same way."),
        recipe=recipe(f"Batch {b.letter}'s pilot worlds (B2 at 10,000 tokens, {b.words} and one other rule, seeds "
                      f"1 and 2, `GraphOfLifeRuns/.pilots/{b.letter}/`) and their controls: the same rule alone, "
                      f"from the batch or lab experiment that made it.",
                      "each world's checkpoint at iteration 1,000 (`research/atlas/at.py --at 1000`, "
                      "`GraphOfLifeRuns/.pilots/atlas/at1000.json`) and the contests in "
                      "`GraphOfLifeRuns/.pilots/batch_TUV_T100.json` and `rejudge_T100.json`",
                      ["Read each world as `research/atlas/profile.py` reads every world.",
                       "Subtract the world with the rule alone from the world with both, seed by seed.",
                       "Seat the brains of the two eras on alternate agents of each era's world, play 100 "
                       "iterations, and count the agents the later brains hold."]))
    out = {"rows": {name: {"poor": change(pairs, "share_at_most_three"), "diversity": change(pairs, "diversity"),
                           "marks": change(pairs, "marks"), "agents": change(pairs, "agents"), "evolution": c}
                    for name, pairs, c in rows}, "b2_median": b2_med}
    ch.number(fig, out)
    return out


def _other_rules(prefix: str, skip: str) -> List[Tuple[str, str, str]]:
    """Every switch of batch W and every other contest of batch S, alone and with the batch's rule, as batches
    X and Y pair them: the contests of batch S alone are the worlds batch W made with revolutions."""
    from make_jobs_TU import W_NOREV, W_REV
    return ([(name, control, f"{prefix}-{name}") for name, _, _, control in W_NOREV if name != skip]
            + [(name, f"W/rev-{name}", f"{prefix}-{name}") for name, _, _ in W_REV])


def _norev_rules() -> List[Tuple[str, str, str]]:
    """Batch W: every switch with and without revolutions. For batch S's contests, which were only ever played
    without revolutions, the world with the rule alone is the one batch W made with revolutions."""
    from make_jobs_TU import W_NOREV, W_REV
    return ([(name, control, f"W/norev-{name}") for name, _, _, control in W_NOREV]
            + [(name, f"W/rev-{name}", control) for name, _, control in W_REV])


NOREV = Batch("W", "no revolutions", "no revolutions alone", ("S/S1-norev-s1", "S/S1-norev-s2"), _norev_rules,
              lambda: [v for w, v in _contests().items() if w.startswith("S/S1-norev-")],
              "batch S's `S/S1-norev`", "batch S's two worlds without revolutions")
HANDOVER = Batch("X", "handover", "handover alone", ("T/handover-s1", "T/handover-s2"),
                 lambda: _other_rules("X/handover", "handover"),
                 lambda: [v for w, v in _contests().items() if w.startswith("T/handover-")],
                 "batch T's `T/handover`", "the two worlds of batch T with handover alone")


def _revolution_rates() -> Dict[str, Dict[str, float]]:
    """Revolutions per hundred agents an iteration, and their share of the nodes that changed hands, over
    iterations 500 to 1,000 of each world of batch W's rules with revolutions on (pilot series, or a lab run's
    rows of the game)."""
    from make_jobs_TU import W_NOREV, W_REV
    out: Dict[str, Dict[str, float]] = {}
    worlds = [(name, _alone(c, s)) for name, _, _, c in W_NOREV for s in (1, 2)]
    worlds += [(name, f"W/rev-{name}-s{s}") for name, _, _ in W_REV for s in (1, 2)]
    for name, wid in worlds:
        pilot = os.path.join(PILOTS, wid, "series.jsonl")
        lab = os.path.join(ROOT, "GraphOfLifeRuns", wid, "stats.jsonl")
        if os.path.exists(pilot):
            # A pilot's row counts revolutions and conquests only in an iteration that had some.
            rows = [r for r in map(json.loads, open(pilot)) if 500 <= r["it"] <= 1000]
            counted = True
        elif os.path.exists(lab):
            rows = [r for r in map(json.loads, open(lab)) if r.get("phase") == 2 and 500 <= r["iteration"] <= 1000]
            counted = bool(rows) and "revolutions" in rows[0]
        else:
            continue
        if not rows or not counted:
            continue
        rev = float(np.mean([r.get("revolutions", 0) for r in rows]))
        agents = float(np.mean([r["nodes"] for r in rows]))
        conquests = (float(np.mean([r.get("conquests", 0) for r in rows]))
                     if any("conquests" in r for r in rows) else float("nan"))
        out[wid] = {"rule": name, "per_hundred_agents": 100 * rev / agents,
                    "share_of_conquests": rev / conquests if conquests else float("nan")}
    return out


@chapter
def norevcombos(ch: F.Chapter) -> None:
    if _changes(ch, NOREV, "changes") is not None:
        ch.number("revolution_rates", _revolution_rates())


@chapter
def handovercombos(ch: F.Chapter) -> None:
    _changes(ch, HANDOVER, "changes", marks=True)


@chapter
def mixcombos(ch: F.Chapter) -> None:
    _combos(ch, Batch("Y", "brains mixed at birth", "mixing alone", ("B2-10000-4af4a3-s001", "B2-10000-4af4a3-s002"),
                      lambda: _other_rules("Y/mix", "mix"),
                      lambda: _lab_contests("B2-10000-4af4a3-s", "e17_T100.json"),
                      "Experiment 17's lab worlds", "the ten mixed worlds of Experiment 17"))
