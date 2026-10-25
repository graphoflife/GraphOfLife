"""
Births as rich as the neighbours (Experiment 32).

Experiment 19's world (brains mixed at birth, parents as mates) with the
estate of the dead paid to the newborns or to the parents, and a child that
must start with at least the median holding of its parent's neighbours; and
the floor alone. Each world read at iteration 3,000 by the four measures of
the author's picture beside the same worlds without the floor (Experiments 31
and 19); then where its poor agents come from: what agents hold at each age,
and how many births keep a world its size.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np

import book_figures as F
import gol_store
from book_figures import chapter, recipe
from book_chapters.common import BLUE, GREY, ORANGE, VIOLET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATLAS = os.path.join(ROOT, "GraphOfLifeRuns", ".pilots", "atlas")
#: B2's ten worlds played on, read at 3,000: the lines of the author's picture there.
LINES = {"marks": 7, "diversity": 0.543, "wide": 1.5, "gini_tokens": 0.466}
#: (the rule, the runs without the floor, the runs with it); seeds 1 and 2 of each.
RULES = [("parents as mates", "B2-10000-bcc1bc", "B2-10000-b289dd"),
         ("estate, newborns", "B2-10000-06b876", "B2-10000-12215f"),
         ("estate, parents", "B2-10000-b8e9fc", "B2-10000-86df9e")]
GROUPS = [("parents as mates, no floor", GREY), ("parents as mates, the floor", BLUE),
          ("an estate, no floor", VIOLET), ("an estate, the floor", ORANGE)]
AT = 3000
AGES = [(a, a + 9) for a in range(1, 100, 10)]
RUNS = ("Experiment 32's six runs (`python3 gol_lab.py run E32`): B2 at 10,000 tokens with brains mixed at birth, "
        "parents as mates and a child never poorer than the median holding of its parent's neighbours, with the "
        "estate of the dead paid to the newborns (`B2-10000-12215f-…`), to the parents (`-86df9e-…`) or neither "
        "(`-b289dd-…`), seeds 1 and 2, to 3,000 iterations; against Experiment 31's estate worlds of the same "
        "seeds (`B2-10000-06b876-…`, `-b8e9fc-…`) and Experiment 19's ten worlds (`B2-10000-bcc1bc-s001` … "
        "`-s010`), which have no floor.")
STEPS = ["Read each world as `research/atlas/profile.py` reads every world: the seven marks, the four spreads "
         "ranked among the frozen reference worlds (their mean is the diversity score), the Gini of the tokens; "
         "and its typical distance over its random twin's, as `research/atlas/space.py` measures it "
         "(`research/atlas/readings.py` does both).",
         "The lines: every mark; the median diversity, and the least Gini, of B2's ten worlds played on and read "
         "at 3,000; a typical distance of 1.5 times the twin's."]


def _group(run_id: str) -> str:
    base = run_id.rsplit("-s", 1)[0]
    for rule, without, with_floor in RULES:
        if base in (without, with_floor):
            kind = "parents as mates" if rule == "parents as mates" else "an estate"
            return f"{kind}, {'the floor' if base == with_floor else 'no floor'}"
    raise KeyError(run_id)


def _readings() -> List[Dict[str, Any]]:
    rows = []
    for name in ("e31_readings.json", "e32_readings.json"):
        path = os.path.join(ATLAS, name)
        if not os.path.exists(path):
            return []
        rows += [r for r in json.load(open(path)) if r.get("at") == AT]
    keep = {b for _, without, with_floor in RULES for b in (without, with_floor)}
    return [r for r in rows if r["id"].rsplit("-s", 1)[0] in keep]


def _ages(run_id: str) -> Tuple[np.ndarray, np.ndarray]:
    z = gol_store.load_archive(run_id, AT)
    return int(z["counters"][2]) - z["born_at"], z["tokens"].astype(float)


def _ledger(run_id: str) -> Tuple[float, float]:
    """Births an iteration and agents alive, means over iterations 2,000 to 3,000."""
    births, agents = [], []
    with open(os.path.join(gol_store.run_dir(run_id), "stats.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("phase") == 1 and 2000 <= r["iteration"] < AT:
                births.append(r["births"])
                agents.append(r["nodes"])
    return float(np.mean(births)), float(np.mean(agents))


def _floor(run_id: str, at: int = AT) -> Dict[str, Any]:
    """How many agents could have a child at `at`, and for how many the floor asks more than min_child."""
    z = gol_store.load_archive(run_id, at)
    tokens = z["tokens"].astype(float)
    n = len(tokens)
    nb: List[List[int]] = [[] for _ in range(n)]
    for a, b in np.asarray(z["edges"]).reshape(-1, 2):
        if a != b and 0 <= a < n and 0 <= b < n:
            nb[a].append(int(b))
            nb[b].append(int(a))
    median = np.array([np.median(tokens[x]) if x else 0.0 for x in nb])
    able = tokens >= 11                  # a child of 10 (min_child) and one token kept
    age = int(z["counters"][2]) - z["born_at"][able]
    return {"able": int(able.sum()), "floor_above_min_child": int((able & (median > 10)).sum()),
            "median_neighbours": float(np.median(median)),
            "able_median_age": float(np.median(age)), "able_aged_10_or_less": float(np.mean(age <= 10))}


@chapter
def floor(ch: F.Chapter) -> None:
    rows = _readings()
    if len(rows) < 18:
        return
    cats = [rule for rule, _, _ in RULES]

    def panel(title: str, ylabel: str, key: str) -> Dict[str, Any]:
        series = []
        for side, colour, off, label in ((0, GREY, -0.17, "without the floor"), (1, ORANGE, 0.17, "with the floor")):
            xs, ys = [], []
            for k, rule in enumerate(RULES):
                mine = [r[key] for r in rows if r["id"].rsplit("-s", 1)[0] == rule[1 + side]]
                xs += list(F.jitter(k, len(mine), 3 + k + 7 * side, width=0.12) + off)
                ys += mine
            series.append({"label": label if key == "marks" else None, "x": xs, "y": ys, "points": True,
                           "size": 6, "colour": colour, "alpha": 0.85})
        series.append({"label": None, "x": [-0.5, len(cats) - 0.5], "y": [LINES[key]] * 2, "colour": GREY,
                       "dash": [5, 4], "width": 1.2})
        return {"title": title, "x": {"categories": cats}, "y": {"label": ylabel, "min": 0}, "series": series}

    ch.grid("measures", [panel("Marks of the health bar", "of 7", "marks"),
                         panel("Diversity score", "mean percentile", "diversity"),
                         panel("Width over the random twin's", "typical distance ratio", "wide"),
                         panel("Gini of the agents' tokens", "Gini", "gini_tokens")],
            title="The floor, with each estate and without", columns=2,
            caption=("Each dot is one world at iteration 3,000, without the floor (grey: Experiment 19's ten worlds "
                     "for parents as mates alone, Experiment 31's two for each estate) and with it (orange: "
                     "Experiment 32's two for each). The dashed line is what the author's picture asks: every mark, "
                     "a diversity score above B2's median world at 3,000 (0.54), a typical distance at least 1.5 "
                     "times the random twin's, a Gini no lower than B2's most even world at 3,000 (0.47)."),
            recipe=recipe(RUNS, "each world's archive at iteration 3,000", STEPS))

    worlds = [f"{base}-s{s:03d}" for _, without, with_floor in RULES for base in (without, with_floor)
              for s in (1, 2)]
    worlds += [f"B2-10000-bcc1bc-s{s:03d}" for s in range(3, 11)]
    age_series, by_age = [], {}
    for group, colour in GROUPS:
        mine = [w for w in worlds if _group(w) == group]
        held = []
        for w in mine:
            age, tokens = _ages(w)
            held.append([float(tokens[(age >= a) & (age <= b)].mean()) for a, b in AGES])
        by_age[group] = np.mean(held, axis=0).tolist()
        age_series.append({"label": group, "x": [(a + b) / 2 for a, b in AGES], "y": by_age[group],
                           "colour": colour, "width": 1.6})
    ledger = {w: _ledger(w) for w in worlds}
    top = max(b for b, _ in ledger.values()) * 1.1
    led_series = [{"label": None, "x": [ledger[w][0] for w in worlds if _group(w) == group],
                   "y": [ledger[w][1] for w in worlds if _group(w) == group], "points": True, "size": 6,
                   "colour": colour, "alpha": 0.85} for group, colour in GROUPS]
    led_series.append({"label": None, "x": [0, top], "y": [0, 100 * top], "colour": GREY, "dash": [5, 4],
                       "width": 1.2})
    ch.grid("paupers", [
        {"title": "What agents hold at each age", "x": {"label": "age (iterations)", "min": 0, "max": 100},
         "y": {"label": "mean tokens of the agents of that age", "log": True}, "series": age_series},
        {"title": "Births and agents", "x": {"label": "births an iteration", "min": 0},
         "y": {"label": "agents alive", "min": 0}, "series": led_series}],
        title="Where the poor agents come from", columns=2,
        caption=("Left: the mean tokens of the agents of each age, in tens of iterations, at iteration 3,000, "
                 "averaged over the worlds of each kind: Experiment 19's ten, and seeds 1 and 2 of the others. "
                 "Right: each world's births an iteration and agents alive, both averaged over iterations 2,000 to 3,000; the dashed line is a hundred agents "
                 "for every birth an iteration, which a world keeps when everyone lives out a hundred iterations."),
        recipe=recipe(RUNS, "each world's archive at iteration 3,000 and its `stats.jsonl`",
                      ["Left: each agent's age is the iteration less the one it was born in; average the tokens "
                       "of the agents aged 1 to 10, 11 to 20, and so on.",
                       "Right: the births and the agents of each iteration's first phase, averaged over the "
                       "iterations 2,000 to 2,999."]))
    ch.number("worlds", [{**{k: r.get(k) for k in ("id", "agents", "marks", "diversity", "wide", "gini_tokens",
                                                    "tokens_per_agent", "dimension")}, "group": _group(r["id"])}
                         for r in rows])
    ch.number("by_age", by_age)
    ch.number("ledger", {w: {"births": round(b, 1), "agents": round(a), "agents_per_birth": round(a / b, 1)}
                         for w, (b, a) in ledger.items()})
    ch.number("floor_binds", {w: _floor(w) for w in worlds})


def _children(run_id: str, start: int, end: int) -> Dict[str, float]:
    """The tokens a child starts with on average, and the agents alive, over iterations start to end - 1."""
    given = births = 0.0
    agents = []
    with open(os.path.join(gol_store.run_dir(run_id), "stats.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("phase") == 1 and start <= r["iteration"] < end and r.get("births") is not None:
                given += (r.get("reproTokenShare") or 0.0) * r["tokens"]
                births += r["births"]
                agents.append(r["nodes"])
    return {"child_tokens": given / births if births else float("nan"), "agents": float(np.mean(agents))}


#: Experiment 2's thirty worlds of plain B1 at 10,000 tokens: those that lived to 3,000 are B1's line.
PLAIN_B1 = [f"B1-10000-s{s:03d}" for s in range(1, 31)]
#: Iterations a line of Experiment 37's figure averages over.
STRETCH = 50
#: The two floor worlds' colours.
SEEDS = (ORANGE, VIOLET)
B1_RUNS = ("Experiment 37's two runs (`python3 gol_lab.py run E37`): B1 at 10,000 tokens with a child never "
           "poorer than the median holding of its parent's neighbours (`B1-10000-60e522-s001`, `-s002`), to 3,000 "
           "iterations; beside the 26 of Experiment 2's 30 worlds of plain B1 (`B1-10000-s001` … `-s030`) that "
           "lived to 3,000.")


def _stats(run_id: str) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """A run's statistics by iteration: its reproduction phase's rows and its game's."""
    birth, game = {}, {}
    with open(os.path.join(gol_store.run_dir(run_id), "stats.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            (birth if r.get("phase") == 1 else game)[r["iteration"]] = r
    return birth, game


def _stretches(run_id: str, end: int = AT) -> Dict[str, List[float]]:
    """
    A world through time, in stretches of STRETCH iterations: its agents after the game, the tokens a child
    starts with (all given to children over all births), births per hundred agents, and the share of its
    holding a parent gives its child.
    """
    birth, game = _stats(run_id)
    out: Dict[str, List[float]] = {"agents": [], "child": [], "births": [], "given": []}
    for a in range(0, end, STRETCH):
        its = range(a, a + STRETCH)
        given = sum((birth[t].get("reproTokenShare") or 0.0) * birth[t]["tokens"] for t in its)
        born = sum(birth[t]["births"] for t in its)
        out["agents"].append(float(np.mean([game[t]["nodes"] for t in its])))
        out["child"].append(given / born if born else float("nan"))
        out["births"].append(float(np.mean([100 * birth[t]["births"] / birth[t]["nodes_before"] for t in its])))
        shares = [birth[t]["meanInvestedShare"] for t in its if birth[t].get("meanInvestedShare") is not None
                  and birth[t]["births"]]
        out["given"].append(float(np.mean(shares)) if shares else float("nan"))
    return out


def _window(run_id: str, start: int = 2000, end: int = AT) -> Dict[str, float]:
    """Iterations start to end - 1: what a child starts with, the agents, births per agent and how long agents live."""
    birth, game = _stats(run_id)
    its = range(start, end)
    born = sum(birth[t]["births"] for t in its)
    agents = float(np.mean([birth[t]["nodes"] for t in its]))
    return {"child_tokens": sum((birth[t].get("reproTokenShare") or 0.0) * birth[t]["tokens"] for t in its) / born,
            "agents": agents,
            "births_per_agent": float(np.mean([birth[t]["births"] / birth[t]["nodes_before"] for t in its])),
            "life": agents / (born / len(its)),
            "given": float(np.mean([birth[t]["meanInvestedShare"] for t in its if birth[t]["births"]])),
            "hub": float(np.mean([game[t]["maxDegree"] / max(1, game[t]["nodes"] - 1) for t in its]))}


def _barred(run_id: str, at: int) -> Dict[str, float]:
    """
    How hard the floor binds where a child may start with a single token, at the end of iteration at - 1:
    the share of agents holding less than the median holding of their neighbours, whom the floor bars from
    any child even if they gave it all they hold; and that median, over the agents.
    """
    frame = gol_store.read_frame(run_id, 2 * at - 1)
    assert (frame["iteration"], frame["phase"]) == (at - 1, 2), (run_id, at)
    place = {int(u): i for i, u in enumerate(frame["ids"])}
    nb: List[List[int]] = [[] for _ in place]
    for u, v in frame["edges"]:
        nb[place[int(u)]].append(place[int(v)])
        nb[place[int(v)]].append(place[int(u)])
    tokens = np.asarray(frame["tokens"], float)
    median = np.array([np.median(tokens[x]) if x else 0.0 for x in nb])
    return {"agents": len(tokens), "barred": float(np.mean(tokens < median)),
            "neighbours_median": float(np.median(median))}


def _whole(run_id: str, start: int, end: int, every: int = 10) -> float:
    """The share of births in which the parent gives its child everything it holds, every tenth iteration."""
    whole = births = 0
    for t in range(start, end, every):
        for b in gol_store.read_frame(run_id, 2 * t)["decisions"]["births"]:
            births += 1
            whole += b["invested"] == b["tokens_before"]
    return whole / births if births else float("nan")


def _end(run_id: str) -> Dict[str, Any]:
    """A world after its last game: its agents, its richest agent's tokens, links and age, and its last births."""
    frame = gol_store.read_frame(run_id, 2 * AT - 1)
    place = {int(u): i for i, u in enumerate(frame["ids"])}
    links = np.zeros(len(place), dtype=int)
    for u, v in frame["edges"]:
        links[place[int(u)]] += 1
        links[place[int(v)]] += 1
    tokens = np.asarray(frame["tokens"])
    ages = np.asarray(frame["ages"])
    richest = int(np.argmax(tokens))
    others = np.arange(len(tokens)) != richest
    hubs = np.argsort(-links)[:3]
    born = gol_store.read_frame(run_id, 2 * AT - 2)["decisions"]["births"]
    birth, game = _stats(run_id)
    return {"agents": len(tokens), "richest_tokens": int(tokens[richest]), "richest_links": int(links[richest]),
            "richest_age": int(ages[richest]), "links": len(frame["edges"]),
            "one_link": int((links == 1).sum()), "median_tokens": float(np.median(tokens)),
            "others_ages": [int(ages[others].min()), int(ages[others].max())],
            "others_median_tokens": float(np.median(tokens[others])),
            "most_linked": [{"links": int(links[h]), "tokens": int(tokens[h]), "age": int(ages[h])} for h in hubs],
            "last_births": [[int(b["tokens_before"]), int(b["invested"])] for b in born],
            "below_50_since": min((t for t in range(2000, AT) if all(game[u]["nodes"] < 50 for u in range(t, AT))),
                                  default=None)}


def _swing(run_id: str, start: int = 500, end: int = 2250) -> float:
    """How widely a world's size swings: the standard deviation of the logarithm of its agents after the game."""
    _, game = _stats(run_id)
    return float(np.std(np.log([game[t]["nodes"] for t in range(start, end)])))


def _hubs(run_id: str) -> List[float]:
    """The most links any agent has, as a share of the others, averaged over each stretch of 250 iterations."""
    _, game = _stats(run_id)
    return [float(np.mean([game[t]["maxDegree"] / max(1, game[t]["nodes"] - 1) for t in range(a, a + 250)]))
            for a in range(0, AT, 250)]


def _b1_figure(ch: F.Chapter, floors: List[str], plain: List[str]) -> None:
    mids = [a + STRETCH / 2 for a in range(0, AT, STRETCH)]
    worlds = {r: _stretches(r) for r in floors + plain}
    panels = [("agents", "Agents", "agents after the game", False),
              ("child", "What a child starts with", "tokens per child", True),
              ("births", "Births", "births per hundred agents", False),
              ("given", "What a parent gives", "share of its tokens given to the child", False)]
    charts = []
    for key, title, ylabel, log in panels:
        series = [{"label": "plain B1, 26 worlds", **F.quantile_band([worlds[r][key] for r in plain], mids),
                   "colour": GREY}]
        series += [F.line(mids, worlds[r][key], f"with the floor, seed {k + 1}", colour, width=1.6)
                   for k, (r, colour) in enumerate(zip(floors, SEEDS))]
        y = {"label": ylabel, "log": True} if log else {"label": ylabel, "min": 0}
        if key == "given":
            y["max"] = 1
        charts.append({"title": title, "x": {"label": "iteration", "min": 0, "max": AT}, "y": y,
                       "series": series, "legend": key == "agents"})
    ch.grid("b1", charts, title="The floor in B1, through time", columns=2,
            caption=("Each line is a mean over stretches of 50 iterations: Experiment 37's two worlds with the floor "
                     "(orange and violet), beside the 26 worlds of plain B1 that lived to 3,000 (grey: the median "
                     "line, the middle half darker, nine in ten paler). Top left: the agents. Top right: the tokens "
                     "a child starts with, all the tokens given to children over all the births (a logarithmic "
                     "scale). Bottom left: births in an iteration per hundred agents. Bottom right: the share of "
                     "its tokens a parent gives its child."),
            recipe=recipe(B1_RUNS, "each run's `stats.jsonl`",
                          ["Agents: the agents after each iteration's game.",
                           "What a child starts with: the tokens given to children in a stretch (each birth "
                           "phase's `reproTokenShare` times the tokens) over its births.",
                           "Births: each birth phase's births over the agents it began with, times 100.",
                           "What a parent gives: each birth phase's `meanInvestedShare`, the mean over its births "
                           "of the tokens given over the tokens the parent held.",
                           "Means over the stretches 0 to 49, 50 to 99, and so on; the band over plain B1's 26 "
                           "worlds stretch by stretch."]))


@chapter
def floorplain(ch: F.Chapter) -> None:
    """
    The floor in plain B2 (Experiment 36) and in B1 (Experiment 37): how often
    it asks more than the least child, and what children start with, beside the
    worlds of the same rules without it (Experiments 16 and 2). In B1, where the
    floor binds, also how the two worlds lived and ended, and a figure of them
    through time beside plain B1's.
    """
    import gol_plan
    b2 = [s.run_id for s in gol_plan.experiment_runs("E36")]
    if all(os.path.exists(os.path.join(gol_store.run_dir(r), "archive", "world_003000.npz")) for r in b2):
        ch.number("E36", {r: {"floor": {str(at): _floor(r, at) for at in (1000, 2000, 3000)},
                              "children": {f"{a}-{b}": _children(r, a, b) for a, b in ((1000, 2000), (2000, 3000))}}
                          for r in b2})
        ch.number("E36_without", {r: _children(r, 1000, 2000) for r in [f"B2-10000-s{s:03d}" for s in range(1, 11)]})
    floors = [s.run_id for s in gol_plan.experiment_runs("E37")]
    if not all(int(gol_store.load_meta(r).get("iteration") or 0) >= AT for r in floors):
        return
    plain = [r for r in PLAIN_B1 if int(gol_store.load_meta(r).get("iteration") or 0) >= AT]
    late = {r: _window(r) for r in floors + plain}
    early = {r: _window(r, 1000, 2000) for r in floors + plain}
    swing = {r: _swing(r) for r in floors + plain}
    ch.number("E37", {r: {"late": late[r], "early": early[r], "swing": swing[r],
                          "barred": {str(at): _barred(r, at) for at in (1000, 2000, 3000)},
                          "whole": {f"{a}-{a + 249}": _whole(r, a, a + 250) for a in range(0, AT, 250)},
                          "end": _end(r)}
                      for r in floors})

    def spread(values: List[float]) -> List[float]:
        return [float(min(values)), float(np.median(values)), float(max(values))]

    hubs = {r: _hubs(r) for r in plain}
    ch.number("E37_plain", {"worlds": len(plain), "died": [r for r in PLAIN_B1 if r not in plain],
                            **{key: spread([late[r][key] for r in plain]) for key in late[plain[0]]},
                            "early": {key: spread([early[r][key] for r in plain]) for key in early[plain[0]]},
                            "swing": spread([swing[r] for r in plain]),
                            "swing_above_the_floor_worlds": sum(swing[r] > min(swing[f] for f in floors)
                                                                for r in plain),
                            "stars": {r: [round(h, 3) for h in hubs[r]] for r in plain if max(hubs[r]) > 0.25},
                            "barred": {str(at): spread([_barred(r, at)["barred"] for r in plain])
                                       for at in (1000, 2000, 3000)},
                            "whole": spread([_whole(r, 2000, AT) for r in plain])})
    _b1_figure(ch, floors, plain)
