# -*- coding: utf-8 -*-
"""
Flow modules: groups that outlast their members, measured in every
world by the page's own arithmetic (web/js/flowmodules.js through
tools/flowmodules_cli.js).

Two passes:
- research/atlas/flowmodules.py — every world, fifty consecutive games: the
  modules, followed as the Research view follows them, and the same games'
  flow rewired at random (GraphOfLifeRuns/.pilots/atlas/flowmodules/);
- research/atlas/flowpersist.py — eighteen lab worlds of every kind: how long
  modules live against the same modules with their agents dealt at random,
  and how alike two games' partitions are, lag games apart
  (GraphOfLifeRuns/.pilots/atlas/flowpersist/).
"""
from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

import book_figures as F
from book_figures import chapter, recipe
from book_chapters.common import BLUE, GREEN, ORANGE, YELLOW
from book_chapters.everyworld import GROUP_WORDS, GROUPS, PILOTS, _dots, _median, _worlds, replicas

FLOWMODULES = os.path.join(PILOTS, "atlas", "flowmodules")
FLOWPERSIST = os.path.join(PILOTS, "atlas", "flowpersist")
FLOW_TEXT = ("Fifty consecutive games of each world: a lab run's from its record, from iteration 950 when it "
             "lived that long (up to five seeds of every condition); a pilot's played on from its checkpoint at "
             "1,000 (`python3 research/atlas/flowmodules.py lab`, `pilots`)")
FLOW_STEPS = ["For every game, the flow between each pair of neighbours: the tokens staked either way.",
              "Group the agents by the map equation (web/js/flowmodules.js, through tools/flowmodules_cli.js).",
              "Follow the groups from game to game as the Research view does: best overlap first, above its "
              "floor of 0.3 of the members two modules share.",
              "Do the same on each game's flow rewired at random, every agent's amounts kept: the null."]
PERSIST_TEXT = ("The same fifty games of eighteen lab worlds: B1 at 10,000 tokens (three seeds), B2 at 10,000 "
                "(three), B1 at 100,000 (two), B1 whose brains never change (two), B2 without random inputs (two), "
                "B2 at mutation rates 0.05 and 0.8 (one each), B2 at 5,000 tokens (two) and with lapsing links "
                "(two) (`python3 research/atlas/flowpersist.py`)")
#: The second pass's worlds, by what they are.
PERSIST_GROUPS = [("B1 10k", ("B1-10000-s001", "B1-10000-s002", "B1-10000-s003"), ORANGE),
                  ("B1 100k", ("B1-100000-s001", "B1-100000-s002"), ORANGE),
                  ("B1 fixed", ("B1-10000-a8525d-s001", "B1-10000-a8525d-s002"), YELLOW),
                  ("B2 10k", ("B2-10000-s001", "B2-10000-s002", "B2-10000-s003"), BLUE),
                  ("B2 var", ("B2-10000-bad270-s001", "B2-10000-bad270-s002", "B2-10000-bf74a9-s001",
                              "B2-10000-fe3009-s001", "B2-5000-s001", "B2-5000-s002"), BLUE),
                  ("lapse", ("B2-5000-71fc89-s002", "B2-5000-71fc89-s003"), GREEN)]


def _flow_results() -> List[Dict[str, Any]]:
    """
    Every world's first-pass result, each world once. A pilot that replays a lab
    world (its settings and seed) is that world again and is left out; and two
    worlds that give the same answer to the last digit are one world — a
    condition that never came into play, like Experiment 8's "stopped only when
    empty" in worlds that never neared extinction — and are counted once.
    """
    repeated = replicas()
    out, seen = [], set()
    for path in sorted(glob.glob(os.path.join(FLOWMODULES, "*.json"))):
        r = json.load(open(path))
        floor = r.get("floor") or {}
        if not (floor.get("summary") and "lives" in floor and floor.get("agentsPerFrame")):
            continue
        if r.get("source") == "pilot" and r["id"] in repeated:
            continue
        same = json.dumps(floor["summary"], sort_keys=True)
        if same in seen:
            continue
        seen.add(same)
        out.append(r)
    return out


def _persist() -> Dict[str, Dict[str, Any]]:
    return {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(os.path.join(FLOWPERSIST, "*.json"))}


def _group_of(world: Optional[Dict[str, Any]]) -> Optional[str]:
    return next((g for g, test, _ in GROUPS if world and test(world)), None)


def _lived_at_least(lives: Dict[str, int], games: int) -> Optional[float]:
    total = sum(lives.values())
    return sum(n for life, n in lives.items() if int(life) >= games) / total if total else None


def _per_world(r: Dict[str, Any], mode: str, what: str) -> Optional[float]:
    m = r.get(mode) or {}
    s = m.get("summary") or {}
    if not s:
        return None
    agents = float(np.mean(m.get("agentsPerFrame") or [np.nan]))
    sizes = m.get("sizes") or {}
    appearances = sum(sizes.values())
    if what == "compression":
        return s.get("compression")
    if what == "size":
        return sum(int(k) * v for k, v in sizes.items()) / appearances if appearances else None
    if what == "in_modules":
        # The share of the agents that belong to a module in a game: the rest sent and received nothing.
        held = sum(int(k) * v for k, v in sizes.items()) / max(1, len(m.get("modulesPerFrame") or []))
        return held / agents if agents and np.isfinite(agents) else None
    if what == "lasting":
        return _lived_at_least(m.get("lives") or {}, 10)
    if what == "turnover":
        return s.get("meanTurnover")
    if what == "born":
        return s.get("regenerationShare")
    return None


@chapter
def flowmodules(ch: F.Chapter) -> None:
    results = _flow_results()
    if not results:
        return
    by_id = {w["id"]: w for w in _worlds()}
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        g = _group_of(by_id.get(r["id"]))
        if g:
            grouped.setdefault(g, []).append(r)
    ch.number("worlds", sum(len(v) for v in grouped.values()))

    # 1. Modules in every world.
    charts = []
    for what, title, label, null in (("compression", "How much the modules compress the flow", "compression", True),
                                     ("size", "How large a module is", "agents per module (mean)", False),
                                     ("in_modules", "Who is in a module", "share of the agents in a module", False),
                                     ("turnover", "How fast members change", "new members per game (share)", False)):
        real = {g: [v for v in (_per_world(r, "floor", what) for r in rs) if v is not None] for g, rs in grouped.items()}
        chart = _dots(title, label, {g: v for g, v in real.items() if v}, min=0)
        if null:
            cats = [c for c in chart["x"]["categories"]]
            for k, g in enumerate(cats):
                nv = [v for v in (_per_world(r, "null", what) for r in grouped[g]) if v is not None]
                if nv:
                    med = float(np.median(nv))
                    chart["series"].append({"label": "rewired flow (median)" if k == 0 else None,
                                            "x": [k - 0.38, k + 0.38], "y": [med, med],
                                            "colour": F.REST, "width": 2, "dash": [4, 3]})
                    ch.number(f"{what}:null:{g}", round(med, 3))
        charts.append(chart)
        for g, v in real.items():
            ch.number(f"{what}:{g}", _median(v))
    ch.grid("every_world", charts, columns=2, title="Flow modules in every world",
            caption=("Each dot is a world, over fifty games. Compression: how much shorter the modules make a "
                     "description of a token wandering the flow (0: no group structure); the dashed bars are the "
                     "same games' flow rewired at random, each agent's amounts kept. Size: agents per module. "
                     "Who is in a module: the share of the agents that send or receive any flow, which is what puts "
                     "an agent in a module. How fast members change: of a module followed from one game to the "
                     "next, the share of its members that are new. " + GROUP_WORDS),
            recipe=recipe(f"{FLOW_TEXT}.", "`GraphOfLifeRuns/.pilots/atlas/flowmodules/*.json`", FLOW_STEPS))

    # 2–3. How long they live and who stays together: the real flow, the same flow with every agent's
    # choice of neighbours undone, and the same modules with their agents dealt at random.
    persist = _persist()
    if persist:
        def groups_present():
            for name, worlds, colour in PERSIST_GROUPS:
                have = [persist[w] for w in worlds if w in persist and "neighbours" in persist[w]]
                if have:
                    yield name, have, colour

        charts = []
        for name, have, colour in groups_present():
            xs = sorted(int(L) for L in have[0]["lives"])
            floor = lambda ys: [max(y, 1e-4) if y is not None else None for y in ys]
            real = [_median([p["lives"][str(L)]["real"] for p in have]) for L in xs]
            chosen = [_median([p["neighbours"]["lives"][str(L)] for p in have]) for L in xs]
            dealt = [_median([p["lives"][str(L)]["shuffled"] for p in have]) for L in xs]
            charts.append({"title": name, "x": {"label": "games", "log": True},
                           "y": {"label": "share living at least this long", "log": True, "min": 1e-4, "max": 1},
                           "series": [F.line(xs, floor(real), "the flow", colour, width=2),
                                      F.line(xs, floor(chosen), "partners dealt",
                                             colour, width=1.5, dash=[5, 4]),
                                      F.line(xs, floor(dealt), "members dealt",
                                             F.REST, width=1.5, dash=[2, 3])]})
            for L, r, c, d in zip(xs, real, chosen, dealt):
                ch.number(f"lives:{name}:{L}", {"flow": r, "neighbours": c, "dealt": d})
        ch.grid("lives", charts, columns=3, title="How long modules live",
                caption=("Of all the modules fifty games hold, the share followed for at least so many games, as "
                         "the Research view follows them (the median of each group's worlds; logarithmic axes, a "
                         "share of 0 drawn at the bottom). The flow: as it was. Partners dealt: the same games with "
                         "every agent's stakes on its neighbours dealt among the same neighbours at random — the "
                         "network and every agent's amounts kept, its choice of whom to stake on undone. Members "
                         "dealt: the real modules with their agents dealt among them at random, game by game — only "
                         "the sizes kept. B1 fixed: two worlds whose brains never change, one frozen (every module "
                         "lasts) and one teeming with births (almost none does), so their median lies between. B2 "
                         "var: B2 without random inputs, at mutation rates 0.05 and 0.8, and at 5,000 tokens; lapse: "
                         "B2 at 5,000 tokens where any unused link lapses."),
                recipe=recipe(f"{PERSIST_TEXT}.", "`GraphOfLifeRuns/.pilots/atlas/flowpersist/*.json`",
                              FLOW_STEPS[:3] + ["Deal every agent's stakes on its neighbours among the same "
                                                "neighbours at random and do the same "
                                                "(`tools/flowmodules_cli.js --neighbours 1`).",
                                                "Deal each game's agents at random among modules of the sizes "
                                                "that game had, and follow those (`--shuffle 1`)."]))

        charts = []
        for name, have, colour in groups_present():
            lags = sorted(int(k) for k in have[0]["together"])
            pick = lambda get: [_median([v for v in (get(p, str(L)) for p in have) if v is not None]) for L in lags]
            real = pick(lambda p, L: p["together"].get(L, {}).get("real"))
            chosen = pick(lambda p, L: p["neighbours"]["together"].get(L, {}).get("real"))
            dealt = pick(lambda p, L: p["together"].get(L, {}).get("null"))
            charts.append({"title": name, "x": {"label": "games later", "log": True},
                           "y": {"label": "share of pairs still in one module", "min": 0, "max": 1},
                           "series": [F.line(lags, real, "the flow", colour, width=2),
                                      F.line(lags, chosen, "partners dealt",
                                             colour, width=1.5, dash=[5, 4]),
                                      F.line(lags, dealt, "members dealt",
                                             F.REST, width=1.5, dash=[2, 3])]})
            for L, r, c, d in zip(lags, real, chosen, dealt):
                ch.number(f"together:{name}:{L}", {"flow": r, "neighbours": c, "dealt": d})
        ch.grid("together", charts, columns=3, title="Two agents of one module, games later",
                caption=("Of two agents in one module, the share still in one module so many games later, over "
                         "the pairs whose agents are both in a module then (the median of each group's worlds). "
                         "The flow: as it was; partners dealt: every agent's stakes dealt among its own neighbours "
                         "at random; members dealt: the later game's agents dealt at random among modules of its "
                         "sizes. No module is followed, so no rule for following one is needed."),
                recipe=recipe(f"{PERSIST_TEXT}.", "`GraphOfLifeRuns/.pilots/atlas/flowpersist/*.json`",
                              FLOW_STEPS[:2] + ["For every two games lag apart, the pairs of agents sharing a "
                                                "module in the first that are both in a module in the second; "
                                                "the share that share one there.",
                                                "The same on the flow with every agent's stakes dealt among its "
                                                "neighbours (`--neighbours 1`), and with the later game's agents "
                                                "dealt among modules of its sizes (three deals)."]))

        # Whether long-lived modules outlast the agents they began with.
        for name, have, _ in groups_present():
            for L in ("10", "20"):
                def of(p, key, side):
                    o = (p["outlast"]["real"] if side == "flow" else p["neighbours"]["outlast"]).get(L) or {}
                    return o.get(key)
                ch.number(f"outlast:{name}:{L}", {
                    side: {key: _median([v for v in (of(p, key, side) for p in have) if v is not None])
                           for key in ("modules", "renewed", "stayed")}
                    for side in ("flow", "neighbours")})

    # 4. Where new members come from.
    born = {g: [v for v in (_per_world(r, "floor", "born") for r in rs) if v is not None] for g, rs in grouped.items()}
    ch.figure("renewal", title="Where a module's new members come from",
              x={"categories": [g for g in born if born[g]]},
              y={"label": "share of new members born, not arrived", "min": 0, "max": 1},
              series=_dots("", "", {g: v for g, v in born.items() if v})["series"],
              caption=("Of the new members of modules followed from one game to the next, the share that had been "
                       "born since (the rest were alive before, in another module or none). A module that renews "
                       "itself from births regenerates; one that takes in agents from around it exchanges them. "
                       "Each dot is a world. " + GROUP_WORDS),
              recipe=recipe(f"{FLOW_TEXT}.", "`GraphOfLifeRuns/.pilots/atlas/flowmodules/*.json`",
                            FLOW_STEPS[:3] + ["A new member is born if its id was not alive the game before."]))
    for g, v in born.items():
        ch.number(f"born:{g}", _median(v))
