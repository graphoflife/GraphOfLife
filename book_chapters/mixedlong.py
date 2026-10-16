# -*- coding: utf-8 -*-
"""
Chapter 80 · Mixed at birth: ten long worlds — what Experiment 17's worlds did.

Experiment 17 ran B2 with brains mixed at birth for 10,000 iterations, ten seeds
(runs B2-10000-4af4a3-s001 … s010). B2 itself beside it: Experiment 16's ten
worlds (B2-10000-s001 … s010) to 2,000 iterations and their continuations to
10,000 (the pilots L/B2-long-s001 … s010), which are the same worlds.

    sizes      every world's agents through time
    poor       the share of agents holding three tokens or fewer, every 500 iterations
    verdict    each world at iteration 10,000: that share, and the marks of the health bar it meets
    breeding   the agents that could have a child, and how often they have one
               (research/oee/checks/e17_breeding.py)
    switch     mixing switched off in five of these worlds at 5,000 and on in five of B2's at 2,000
               (batch SW), each beside its twin played on unchanged
    size       the two marks the mixed worlds miss, against their size, at every whole world kept
               (research/oee/checks/e17_health.py), beside B2's worlds at 10,000 and 5,000 tokens
    evolution  lifetime contests (T = 100) between eras: 1,000 against 500, 5,000 against 2,000,
               10,000 against 5,000
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

import book_figures as F
from book_figures import chapter, recipe
from book_chapters.common import BLUE, GREY, ORANGE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(ROOT, "GraphOfLifeRuns")
PILOTS = os.path.join(RUNS_DIR, ".pilots")
MIXED = [f"B2-10000-4af4a3-s{s:03d}" for s in range(1, 11)]
B2 = [f"B2-10000-s{s:03d}" for s in range(1, 11)]
#: Experiment 17's span. Its runs may be carried further by a later experiment; this chapter reads them to here.
END = 10000
RUNS = ("The ten runs of Experiment 17 — B2 at 10,000 tokens with brains mixed at birth, seeds 1 to 10, 10,000 "
        "iterations (`python3 gol_lab.py run E17`) — and B2 itself: Experiment 16's ten worlds to 2,000 iterations "
        "and the same worlds played on to 10,000 as pilots (`GraphOfLifeRuns/.pilots/L/B2-long-s001` … `s010`).")


def _b2_agents(seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """One B2 world's agents after every game: the lab's run to 2,000, its pilot continuation after."""
    its, nodes = F.series(f"B2-10000-s{seed:03d}", "nodes", 2)
    keep = its < 2000
    its, nodes = list(its[keep]), list(nodes[keep])
    series = os.path.join(PILOTS, "L", f"B2-long-s{seed:03d}", "series.jsonl")
    for line in open(series):
        r = json.loads(line)
        if 2000 <= r["it"] <= 10000:
            its.append(r["it"])
            nodes.append(r["nodes"])
    return np.array(its), np.array(nodes, dtype=float)


def _poor_share(path: str) -> Tuple[int, float, int]:
    """(iteration, share of agents with three tokens or fewer, agents) of a whole world."""
    with np.load(path, allow_pickle=False) as z:
        tokens = z["tokens"]
        iteration = int(z["counters"][2])
    return iteration, float(np.mean(tokens <= 3)) if tokens.size else float("nan"), int(tokens.size)


def _poor_through_time() -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
    out: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {"mixed": [], "B2": []}
    for run in MIXED:
        points = sorted(_poor_share(p) for p in glob.glob(os.path.join(RUNS_DIR, run, "archive", "world_*.npz")))
        points = [p for p in points if p[0] % 500 == 0 and 0 < p[0] <= END]
        out["mixed"].append((np.array([p[0] for p in points]), np.array([p[1] for p in points])))
    for seed in range(1, 11):
        points = [_poor_share(p) for p in glob.glob(os.path.join(RUNS_DIR, f"B2-10000-s{seed:03d}", "archive",
                                                                "world_*.npz"))]
        points += [_poor_share(p) for p in glob.glob(os.path.join(PILOTS, "L", f"B2-long-s{seed:03d}", "ckpt_*.npz"))]
        points = sorted({p[0]: p for p in points if p[0] % 500 == 0 and 0 < p[0] <= 10000}.values())
        out["B2"].append((np.array([p[0] for p in points]), np.array([p[1] for p in points])))
    return out


def _contests() -> Dict[Tuple[int, int], Dict[str, Dict[str, float]]]:
    """Each world's later-era share for each pair of eras, the mean of its contests, for the mixed worlds and B2's."""
    by: Dict[Tuple[int, int], Dict[str, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for path, key, kind in ((os.path.join(PILOTS, "e17_T100.json"), "world", "mixed"),
                            (os.path.join(PILOTS, "long_steps.json"), "run", "B2"),
                            (os.path.join(PILOTS, "long_rejudge_T100.json"), "world", "B2")):
        if not os.path.exists(path):
            continue
        for r in json.load(open(path)):
            name = r.get(key) or r.get("run") or r.get("world")
            by[(r["a"], r["b"])][kind][name].append(r["share"])
    return {pair: {kind: {w: float(np.mean(v)) for w, v in worlds.items()} for kind, worlds in kinds.items()}
            for pair, kinds in by.items()}


@chapter
def mixedlong(ch: F.Chapter) -> None:
    profile = os.path.join(PILOTS, "e17_profile.json")
    if not all(os.path.exists(os.path.join(RUNS_DIR, run, "checkpoint.npz")) for run in MIXED) \
            or not os.path.exists(profile):
        return

    # 1. Every world's agents through time.
    series = [F.band_series([_b2_agents(s) for s in range(1, 11)], "B2 (ten worlds)", GREY)]
    for k, run in enumerate(MIXED):
        its, nodes = F.series(run, "nodes", 2)
        its, nodes = F.thin(its[its < END], nodes[its < END], 25)
        series.append(F.line(its, nodes, "brains mixed at birth, one line a world" if k == 0 else None, ORANGE,
                             width=1.2))
    ch.figure("sizes", title="Agents alive, world by world", x={"label": "iteration"},
              y={"label": "agents", "min": 0}, series=series,
              caption=("Each orange line is one world of Experiment 17, its agents after each game (means over 25 "
                       "iterations); the grey band is B2's ten worlds — their median, middle half and nine in ten. "
                       "B2's worlds keep their size; the mixed ones swing between a few hundred and two thousand "
                       "agents, and one died out at iteration 5,376."),
              recipe=recipe(RUNS, "Each run's `stats.jsonl` (the lab's runs) and `series.jsonl` (B2's continuations)",
                            ["Read the agents after each game (`nodes`, phase 2).",
                             "For B2, join each lab world (to 2,000) to its continuation (2,000 to 10,000)."]))

    # 2. Agents with three tokens or fewer, every 500 iterations.
    poor = _poor_through_time()
    ch.figure("poor", title="Agents holding three tokens or fewer", x={"label": "iteration"},
              y={"label": "share of the agents", "min": 0, "max": 1},
              series=[F.band_series(poor["B2"], "B2 (ten worlds)", GREY),
                      F.band_series(poor["mixed"], "brains mixed at birth (ten worlds)", ORANGE)],
              guides=[{"axis": "y", "at": 0.5, "label": "half"}],
              caption=("Each band is ten worlds — the median, the middle half and nine in ten — every 500 "
                       "iterations: the share of their agents holding one, two or three tokens. A world that died "
                       "out leaves the band when it does."),
              recipe=recipe(RUNS, "Each run's whole worlds: the lab's archives (`archive/world_<iteration>.npz`) "
                                  "and the continuations' checkpoints (`ckpt_<iteration>.npz`)",
                            ["Read every agent's tokens in each whole world kept at a multiple of 500 iterations.",
                             "Count those holding three or fewer, as a share of all."]))

    # 3. The verdict: each world at iteration 10,000.
    rows = json.load(open(profile))
    b2_end = [_poor_share(sorted(glob.glob(os.path.join(PILOTS, "L", f"B2-long-s{s:03d}", "ckpt_010000.npz")))[0])[1]
              for s in range(1, 11) if glob.glob(os.path.join(PILOTS, "L", f"B2-long-s{s:03d}", "ckpt_010000.npz"))]
    shares = [r["share_at_most_three"] for r in rows]
    marks = [r["marks"] for r in rows]
    ch.grid("verdict", [
        {"title": "Agents with three tokens or fewer, at the end", "x": {"categories": ["B2", "mixed at birth"]},
         "y": {"label": "share of the agents", "min": 0, "max": 1},
         "series": F.dots(["B2", "mixed at birth"], [b2_end, shares], [GREY, ORANGE]),
         "guides": [{"axis": "y", "at": 0.5, "label": "the thesis: below a half"}]},
        {"title": "Marks of the health bar met", "x": {"categories": ["mixed at birth"]},
         "y": {"label": "marks (of 7)", "min": 0, "max": 7.5},
         "series": F.dots(["mixed at birth"], [marks], [ORANGE]),
         "guides": [{"axis": "y", "at": 7, "label": "every mark"}]}],
        columns=2, title="Experiment 17 at iteration 10,000",
        caption=("Each dot is a world at its last checkpoint, read as Meta X reads every world. Left: the share of "
                 "its agents holding three tokens or fewer, B2's ten worlds at 10,000 iterations beside the mixed "
                 "ones. Right: how many of the seven marks of Chapter 46's health bar each mixed world meets; the "
                 "world that died out is read at its last checkpoint, with 20 agents."),
        recipe=recipe(RUNS, "Each run's last checkpoint (`checkpoint.npz`; B2's continuations' `ckpt_010000.npz`) "
                            "(`GraphOfLifeRuns/.pilots/e17_profile.json`)",
                      ["Count the agents holding three tokens or fewer, as a share of all.",
                       "Measure the seven marks as `research/atlas/profile.py` does."]))
    ch.number("verdict", {"median_share": float(np.median(shares)), "seven_marks": sum(m == 7 for m in marks),
                          "shares": shares, "marks": marks, "b2_end": b2_end})

    # 4. Who could have a child, and who has one.
    breeding_path = os.path.join(PILOTS, "e17_breeding.json")
    if os.path.exists(breeding_path):
        breeding = json.load(open(breeding_path))

        def band(kind: str, value) -> List[Tuple[np.ndarray, np.ndarray]]:
            out = []
            for rows in breeding[kind].values():
                rows = [r for r in rows if value(r) is not None]
                if rows:
                    out.append((np.array([r["at"] for r in rows]), np.array([value(r) for r in rows], float)))
            return out

        def able(r):
            return r["able"] / r["agents"]

        def eager(r):
            return 100 * r["per_able"] if r["per_able"] else None

        ch.grid("breeding", [
            {"title": "Agents that could have a child", "x": {"label": "iteration"},
             "y": {"label": "share of the agents", "min": 0, "max": 1},
             "series": [F.band_series(band("B2", able), "B2 (ten worlds)", GREY),
                        F.band_series(band("mixed", able), "brains mixed at birth (ten worlds)", ORANGE)]},
            {"title": "How often they have one", "x": {"label": "iteration"},
             "y": {"label": "births per iteration per 100 of them", "log": True},
             "series": [F.band_series(band("B2", eager), "B2 (ten worlds)", GREY),
                        F.band_series(band("mixed", eager), "brains mixed at birth (ten worlds)", ORANGE)]}],
            columns=2, title="Who could have a child, and who has one",
            caption=("Left: the share of the agents that could have a child — 11 tokens or more, since a child takes "
                     "at least 10 and the parent keeps one, and fewer than 8 links. Right: the births of the next 50 "
                     "iterations, per iteration, per hundred such agents, on a logarithmic axis. Each band is ten "
                     "worlds every 500 iterations — the median, the middle half and nine in ten; B2's ten worlds are "
                     "Experiment 16's to 2,000 iterations and the same worlds played on after."),
            recipe=recipe(RUNS, "Each world kept whole every 500 iterations, and each run's births "
                                "(`research/oee/checks/e17_breeding.py`, `GraphOfLifeRuns/.pilots/e17_breeding.json`)",
                          ["In each whole world, count the agents with 11 tokens or more and fewer than 8 links.",
                           "Read the births of the 50 reproduction phases that follow, from the run's record.",
                           "Divide their mean per iteration by the agents counted."]))
        ch.number("breeding", {kind: {"able": [float(np.median([able(r) for rows in breeding[kind].values()
                                                                for r in rows]))],
                                      "per_100_able": float(np.median([eager(r) for rows in breeding[kind].values()
                                                                       for r in rows if eager(r) is not None]))}
                               for kind in ("B2", "mixed")})

    # 5. Mixing switched at a moment (batch SW): each switched world beside its twin played on unchanged.
    def pilot_agents(path: str) -> Tuple[np.ndarray, np.ndarray]:
        rows = [json.loads(line) for line in open(path)]
        # A pilot's row `it` is the step that ends at it: the lab's iteration it - 1.
        return np.array([r["it"] - 1 for r in rows]), np.array([r["nodes"] for r in rows], float)

    panels, switched = [], {}
    for prefix, at, title, mine, twin in (
            ("mixoff", 5000, "Mixing switched off at iteration 5,000",
             ("the same world without mixing from 5,000", GREY), ("as Experiment 17 played it on", ORANGE)),
            ("mixon", 2000, "Mixing switched on at iteration 2,000",
             ("the same world with mixing from 2,000", ORANGE), ("as B2 played it on", GREY))):
        series, worlds = [], sorted(glob.glob(os.path.join(PILOTS, "SW", f"{prefix}-s*", "series.jsonl")))
        for k, path in enumerate(worlds):
            seed = int(os.path.basename(os.path.dirname(path))[-3:])
            its, n = pilot_agents(path)
            t_its, t_n = (F.series(f"B2-10000-4af4a3-s{seed:03d}", "nodes", 2) if prefix == "mixoff"
                          else _b2_agents(seed))
            keep = (t_its >= at - 500) & (t_its < min(END, at + 2000))
            series.append(F.line(*F.thin(t_its[keep], t_n[keep], 25), twin[0] if k == 0 else None, twin[1],
                                 width=1.0, dash=[4, 3]))
            series.append(F.line(*F.thin(its, n, 25), mine[0] if k == 0 else None, mine[1], width=1.4))
            switched[f"{prefix}-s{seed:03d}"] = {"at": int(its.max()), "agents": float(n[-1]),
                                                 "twin": float(t_n[keep][np.argmin(abs(t_its[keep] - its.max()))])}
        if series:
            panels.append({"title": title, "x": {"label": "iteration"}, "y": {"label": "agents", "min": 0},
                           "series": series, "guides": [{"axis": "x", "at": at, "label": "the switch"}]})
    if panels:
        ch.grid("switch", panels, columns=len(panels), title="Mixing switched at a moment",
                caption=("Each solid line is a world played on as a pilot from the moment the lab had kept it whole, "
                         "with mixing at birth switched; each dashed line is the same world played on unchanged — "
                         "the lab's own run of Experiment 17, or B2's continuation. Agents after each game, means "
                         "over 25 iterations. Batch SW: mixing off in the five worlds of Experiment 17 that met every "
                         "mark, at iteration 5,000; on in five of B2's worlds, at 2,000."),
                recipe=recipe(RUNS + " Batch SW: `GraphOfLifeRuns/.pilots/SW/mixoff-sNNN` and `mixon-sNNN` "
                                     "(`research/oee/pilots/make_jobs_TU.py`).",
                              "Each pilot's `series.jsonl`; the lab runs' `stats.jsonl`",
                              ["Start from the lab's whole world at the moment (`run.py --from RUN:ITERATION --set "
                               "mix_at_birth=…`) and play 2,000 iterations.",
                               "Read the agents after each game in the pilot and in the run it was taken from."]))
        ch.number("switch", switched)

    # 6. The two marks the mixed worlds miss, against their size.
    health_path = os.path.join(PILOTS, "e17_health.json")
    if os.path.exists(health_path):
        moments = [r for r in json.load(open(health_path)) if "bar" in r]
        from book_chapters.healthy import _load
        # B2 itself at both sizes: Experiment 16's worlds and their continuations, Experiment 14's.
        loaded = [w for w in _load() if w.get("dimension") is not None]
        others = [("B2 at 10,000 tokens", GREY, [w for w in loaded if w["id"] in B2 or w["id"].startswith("L/B2-long")]),
                  ("B2 at 5,000 tokens", BLUE, [w for w in loaded if w["family"] == "B2-5000"])]

        def dots(key: str) -> List[Dict[str, Any]]:
            out = []
            for label, colour, ws in others:
                out.append({"label": label, "x": [w["agents"] for w in ws], "y": [w[key] for w in ws],
                            "points": True, "size": 4, "colour": colour, "alpha": 0.8})
            out.append({"label": "brains mixed at birth, every 500 iterations", "x": [r["agents"] for r in moments],
                        "y": [r[key] for r in moments], "points": True, "size": 4, "colour": ORANGE, "alpha": 0.7})
            return out

        ch.grid("size", [
            {"title": "Dimension", "x": {"label": "agents", "log": True}, "y": {"label": "dimension (ball growth)"},
             "series": dots("dimension"), "guides": [{"axis": "y", "at": 4, "label": "the bar: 4 at most"}]},
            {"title": "The best-connected agent", "x": {"label": "agents", "log": True},
             "y": {"label": "share of the others it is joined to", "min": 0},
             "series": dots("hub"), "guides": [{"axis": "y", "at": 0.05, "label": "the bar: 5% at most"}]}],
            columns=2, title="The marks a world misses, against its size",
            caption=("Each orange dot is a world of Experiment 17 at one of the moments it was kept whole, every 500 "
                     "iterations; each grey dot a world of B2 at 10,000 tokens — Experiment 16's ten at 2,000 "
                     "iterations, and the same worlds played on, at 10,000 or more — and each blue one a world of B2 "
                     "at 5,000 tokens, Experiment 14's ten at 1,000 iterations. Across, "
                     "on a logarithmic axis: the agents alive. Left: the dimension the network shows by ball growth; "
                     "right: the share of the others the best-connected agent is joined to."),
            recipe=recipe(RUNS + " B2 at 5,000 tokens: Experiment 14's ten runs (`B2-5000-s001` … `s010`).",
                          "`GraphOfLifeRuns/.pilots/e17_health.json` (`research/oee/checks/e17_health.py`) and "
                          "`GraphOfLifeRuns/.pilots/atlas/profile.json`",
                          ["Read each whole world as `research/atlas/profile.py` reads every world: the dimension "
                           "by ball growth, the best-connected agent's links over the others.",
                           "Set each against the number of agents alive."]))
        n = np.array([r["agents"] for r in moments], float)
        ch.number("size", {"moments": len(moments), "seven": sum(r["marks"] == 7 for r in moments),
                           "corr_dimension_log_agents": float(np.corrcoef(np.log(n), [r["dimension"] for r in moments])[0, 1]),
                           "by_size": [{"from": lo, "to": hi, "moments": int(((n >= lo) & (n < hi)).sum()),
                                        "seven": float(np.mean([r["marks"] == 7 for r in moments
                                                                if lo <= r["agents"] < hi]))}
                                       for lo, hi in ((0, 300), (300, 600), (600, 1000), (1000, 1500), (1500, 3000))
                                       if ((n >= lo) & (n < hi)).any()]})

    # 7. Does evolution go on?
    contests = _contests()
    pairs = [((500, 1000), "1,000 vs 500"), ((2000, 5000), "5,000 vs 2,000"), ((5000, 10000), "10,000 vs 5,000")]
    mixed = [list(contests.get(pair, {}).get("mixed", {}).values()) for pair, _ in pairs]
    b2 = [list(contests.get(pair, {}).get("B2", {}).values()) for pair, _ in pairs]
    # B2 and the mixed worlds side by side, a column each for every pair of eras.
    cats = [f"{who} · {label}" for _, label in pairs for who in ("B2", "mixed")]
    groups = [g for pair in zip(b2, mixed) for g in pair]
    series = F.dots(cats, groups, [GREY, ORANGE] * 3)
    ch.figure("evolution", title="Do later brains beat earlier ones?", x={"categories": cats},
              y={"label": "the later brains' share of a contested world", "min": 0.2, "max": 1}, series=series,
              guides=[{"axis": "y", "at": 0.5, "label": "no better"}],
              caption=("Each dot is a world: the share of a world the later era's brains hold after 100 iterations "
                       "against the earlier era's, the mean of two contests, one in each era's world; the bar is the "
                       "median world. Orange: the mixed worlds of Experiment 17; grey: B2 — its ten worlds for 1,000 "
                       "against 500, three of their continuations for the later pairs."),
              recipe=recipe(RUNS, "Each run's whole worlds every 500 iterations (`GraphOfLifeRuns/.pilots/"
                                  "e17_T100.json`, `long_steps.json`, `long_rejudge_T100.json`)",
                            ["Seat the brains of the two eras on alternate agents of each era's world and play it on "
                             "for 100 iterations, an agent's lifetime (`research/oee/checks/rejudge.py`).",
                             "Count the agents the later era's brains hold at the end, as a share of all."]))
    ch.number("evolution", {label: {"mixed": float(np.median(m)) if m else None, "mixed_worlds": len(m),
                                    "b2": float(np.median(b)) if b else None, "b2_worlds": len(b)}
                            for (pair, label), m, b in zip(pairs, mixed, b2)})
