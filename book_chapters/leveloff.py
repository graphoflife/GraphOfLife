"""
What Experiment 46's runs can still say, though its thesis cannot be judged: which of them lived to their end, how
the others died, and how the one arm whose worlds all lived differs from the rest. Nothing is read but each run's
own statistics (`GraphOfLifeRuns/<run>/stats.jsonl`) and record (`meta.json`), and, for the world the arms
started from, its numbers in round 3 (`research/screen/candidates.json`).
"""
from __future__ import annotations

import json
import os
from math import comb
from typing import Any, Dict, List

import numpy as np

import book_figures as F
import gol_plan
import gol_store as store
from book_chapters.common import BLUE, GREEN, GREY, ORANGE, VIOLET, YELLOW
from book_figures import chapter, dots, line, recipe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENT = "E46"
WORLD = "B1-30000-4108dd-s3058"
#: Where the notebook's settled life starts.
SETTLED = 500
#: Each arm's short name in the figures, and its colour.
SHORT = {"control": ("control", GREY),
         "4 times the tokens": ("4× tokens", BLUE),
         "faithful copies": ("faithful copies", VIOLET),
         "message to oneself: largest": ("other message", YELLOW),
         "revolutionary tokens for the coalition alone": ("costly flag", ORANGE),
         "stakes pooled by family": ("family pools", GREEN)}
#: A thinned line keeps every this many iterations, and the last.
STRIDE = 25
#: How far back the figure of the ends looks.
BEFORE_END = 150


def arms() -> List[str]:
    return [c["name"] for c in gol_plan.read_plan(EXPERIMENT)["runs"]["conditions"]]


def reading(spec: gol_plan.RunSpec) -> Dict[str, Any]:
    """One run: its course of agents, how it ended, and the medians of its settled life."""
    rows = F.rows(spec.run_id)
    meta = store.load_meta(spec.run_id)
    tokens = int(spec.config["total_tokens"])
    threshold = int(spec.config["extinction_threshold"])
    game = [r for r in rows if r["phase"] == 2]
    its = np.array([r["iteration"] for r in game])
    agents = np.array([r["nodes"] for r in game], float)
    out: Dict[str, Any] = {"run": spec.run_id, "seed": int(spec.run_id.rsplit("-s", 1)[1]), "tokens": tokens,
                           "end": int(meta["iteration"]), "died": meta["status"] == "extinct",
                           "its": its, "agents": agents, "threshold": threshold}
    if out["died"]:
        many = np.nonzero(agents >= 10 * threshold)[0]
        out["many_until"] = int(its[many[-1]]) if many.size else None
        out["last_before"], out["last_after"] = int(agents[-2]), int(agents[-1])
        out["richest_at_end"] = game[-1]["maxTokens"] / tokens
    settled = [r for r in game if r["iteration"] >= SETTLED]
    if not settled:
        return out
    deaths = [r for r in rows if r["iteration"] >= SETTLED]
    starved = sum(r.get("starved") or 0 for r in deaths)
    culled = sum(r.get("orphaned") or 0 for r in deaths)
    lived = sum(r["nodes"] for r in settled)
    out.update(
        per_thousand=float(np.median([1000 * r["nodes"] / tokens for r in settled])),
        richest=float(np.median([100 * r["maxTokens"] / tokens for r in settled])),
        gini=float(np.median([r["gini"] for r in settled if r.get("gini") is not None])),
        cull_share=100 * culled / (starved + culled) if starved + culled else None,
        deaths=(starved + culled) / lived,
        cut=float(np.median([(r.get("prunedEdges") or 0) / r["nodes"] for r in settled])),
        degree=float(np.median([r["meanDegree"] for r in settled])),
        brains=float(np.median([r["distinctBrains"] / r["nodes"] for r in settled])))
    return out


def thinned(its: np.ndarray, values: np.ndarray) -> tuple:
    keep = (its % STRIDE == 0)
    keep[-1] = True
    return its[keep], values[keep]


def _median(values: List[Any]) -> Any:
    v = [x for x in values if x is not None]
    return float(np.median(v)) if v else None


def _spearman(a: List[float], b: List[float]) -> float:
    rank = lambda v: np.argsort(np.argsort(v, kind="stable"), kind="stable").astype(float)
    return float(np.corrcoef(rank(np.asarray(a)), rank(np.asarray(b)))[0, 1])


@chapter
def leveloff(ch: F.Chapter) -> None:
    names = arms()
    runs = {arm: [reading(s) for s in F.runs_of(EXPERIMENT, arm)] for arm in names}
    every = [r for arm in names for r in runs[arm]]
    dead = [r for r in every if r["died"]]
    small = [r for r in every if r["tokens"] == min(x["tokens"] for x in every)]
    alive_small = [r for r in small if not r["died"]]
    arms_small = [arm for arm in names if all(r["tokens"] == small[0]["tokens"] for r in runs[arm])]

    ch.number("lived", {arm: [{"seed": r["seed"], "end": r["end"], "died": r["died"]} for r in runs[arm]]
                        for arm in names})
    # If an arm had nothing to do with which runs live, every way of choosing the survivors among the runs at
    # that size would be as likely; this is how often all of them would then fall in one arm.
    per_arm = len(small) // len(arms_small)
    ways_one_arm = len(arms_small) * comb(per_arm, len(alive_small)) if len(alive_small) <= per_arm else 0
    ch.number("one_arm_chance", {"runs": len(small), "arms": len(arms_small), "lived": len(alive_small),
                                 "ways_in_one_arm": ways_one_arm, "ways": comb(len(small), len(alive_small))})
    ch.number("ends", {"many_until_before_end": _median([r["end"] - r["many_until"] for r in dead
                                                         if r.get("many_until") is not None]),
                       "many_until_range": [min(r["end"] - r["many_until"] for r in dead),
                                            max(r["end"] - r["many_until"] for r in dead)],
                       "last_game": [[r["last_before"], r["last_after"]] for r in dead],
                       "richest_at_end": _median([r["richest_at_end"] for r in dead]),
                       "richest_at_end_range": [min(r["richest_at_end"] for r in dead),
                                                max(r["richest_at_end"] for r in dead)],
                       "died": len(dead), "runs": len(every)})
    keys = ("per_thousand", "richest", "gini", "cull_share", "deaths", "cut", "degree", "brains")
    ch.number("settled", {arm: {k: [r.get(k) for r in runs[arm]] for k in keys} for arm in names})
    with open(os.path.join(ROOT, "research", "screen", "candidates.json"), encoding="utf-8") as f:
        world = next(w for w in json.load(f)["worlds"] if w["id"] == WORLD)
    ch.number("round3", {"agents": world["numbers"]["agents"], "links": world["numbers"]["links"],
                         "tokens": world["tokens"], "ended_at": world["ended_at"]})

    RUNS = (f"The {len(every)} runs of Experiment 46, three seeds (4601 to 4603) in each of its arms, for 20,000 "
            "iterations or until the world died out; 10,000 tokens, and 40,000 in the arm of four times the tokens "
            "(every setting is listed in [Why does evolution level off?](why-does-evolution-level-off.md)). They are "
            "made with `python3 gol_lab.py run E46`.")
    DATA = "each run's `GraphOfLifeRuns/<run>/stats.jsonl`, and its `meta.json` for where it ended"

    # ---- Every run's agents through its life
    panels = []
    for k, arm in enumerate(names):
        title, colour = SHORT[arm]
        panels.append({
            "title": f"{title}: {sum(not r['died'] for r in runs[arm])} of 3 lived",
            "x": {"label": "iteration", "min": 0, "max": max(r["end"] for r in every) + 1},
            "y": {"label": "agents", "log": True, "min": 10, "max": 10000},
            "series": [line(*thinned(r["its"], r["agents"]), None, colour, alpha=0.9) for r in runs[arm]],
            "guides": [{"axis": "y", "at": every[0]["threshold"], "label": "a world ends" if k == 0 else ""}],
            "legend": False, "height": 170})
    ch.grid("lives", panels, columns=3, title="Which runs lived",
            caption="Every run of Experiment 46, one panel for each arm, one line for each of its three seeds: "
                    "how many agents the world held after each game, on a logarithmic scale (each line marks "
                    "every 25th iteration, and the last). A world ends when a game leaves it 20 agents or "
                    "fewer (the dashed line). The arm of four times the tokens plays 40,000 tokens; all the "
                    "others play 10,000.",
            recipe=recipe(RUNS, DATA, [
                "Keep the rows after the game (`phase` 2).",
                "Plot `nodes` against `iteration`, every 25th iteration and the last row."]))

    # ---- How the runs that died ended, and how long the runs lived against how large they were
    last, named = [], set()
    for arm in names:
        title, colour = SHORT[arm]
        for r in runs[arm]:
            if not r["died"]:
                continue
            near = r["its"] >= r["end"] - BEFORE_END
            last.append(line(r["its"][near] - r["end"], r["agents"][near],
                             title if arm not in named else None, colour, alpha=0.85))
            named.add(arm)
    sized = [r for r in every if "per_thousand" in r]
    size = {r["run"]: float(np.median(r["agents"][r["its"] >= SETTLED])) for r in sized}
    gone = [r for r in sized if r["died"]]
    ch.number("size_and_life", {"rho_died": _spearman([size[r["run"]] for r in gone], [r["end"] for r in gone]),
                                "died": len(gone),
                                "agents": {r["run"]: size[r["run"]] for r in sized}})
    points = []
    # The last arm first, so that the dots of the other arms are drawn over the family pools' where they meet.
    for arm in reversed(names):
        title, colour = SHORT[arm]
        for died, shape in ((True, 6), (False, 9)):
            pick = [r for r in runs[arm] if "per_thousand" in r and r["died"] == died]
            if pick:
                points.append(line([size[r["run"]] for r in pick], [r["end"] for r in pick], None, colour,
                                   points=True, size=shape, alpha=0.9))
    ch.grid("ends", [
        {"title": f"the last {BEFORE_END} iterations of the {len(dead)} that died",
         "x": {"label": "iterations before the end", "min": -BEFORE_END, "max": 0},
         "y": {"label": "agents", "log": True, "min": 10, "max": 3000},
         "series": last, "guides": [{"axis": "y", "at": every[0]["threshold"], "label": ""}], "height": 260},
        {"title": "how long a run lived, and how large it was",
         "x": {"label": f"agents, the median from iteration {SETTLED} on", "log": True, "min": 50, "max": 5000},
         "y": {"label": "iteration it ended", "log": True, "min": 300, "max": 30000},
         "series": points, "guides": [{"axis": "y", "at": max(r["end"] for r in every), "label": "the end"}],
         "legend": False, "height": 260}],
        columns=2, title="How the worlds died",
        caption=f"Left: each line is one of the {len(dead)} runs that died, coloured by its arm: its agents after "
                f"each game in its last {BEFORE_END} iterations, the end at 0, on a logarithmic scale; the "
                "dashed line is the 20 agents at which a world ends. The steps down are culls: a game cuts the "
                "links nobody staked a token on, and every piece of the network cut off from the largest one "
                f"dies at once. Right: each dot is one run that lived past iteration {SETTLED}, in its arm's "
                "colour: the median of its agents after each game from then on, and the iteration it ended, "
                "both on logarithmic scales. The larger dots on the line at 20,000 lived to the end; the three family-pool "
                "runs lie there under the run of four times the tokens that lived, and beside the one that died at "
                "19,667.",
        recipe=recipe(RUNS, DATA, [
            f"Left: the rows after the game (`phase` 2) of the last {BEFORE_END} iterations of each run that "
            "died; `nodes` against the iteration minus the run's last one.",
            f"Right: for each run with rows from iteration {SETTLED} on, the median of `nodes` over them, against "
            "the run's last iteration (`meta.json`, `iteration`).",
            "The rank correlation of the two over the runs that died: Spearman's rho."]))

    # ---- What the family-pool arm does differently
    columns = [SHORT[arm][0] for arm in names]
    colours = [SHORT[arm][1] for arm in names]
    def col(key: str) -> List[List[Any]]:
        return [[r.get(key) for r in runs[arm]] for arm in names]
    ch.grid("spread", [
        {"title": "agents per 1,000 tokens", "x": {"label": "", "categories": columns},
         "y": {"label": "agents per 1,000 tokens", "log": True, "min": 5, "max": 300},
         "series": dots(columns, col("per_thousand"), colours), "legend": False, "height": 200},
        {"title": "the richest agent's share", "x": {"label": "", "categories": columns},
         "y": {"label": "% of the world's tokens", "log": True, "min": 0.1, "max": 30},
         "series": dots(columns, col("richest"), colours), "legend": False, "height": 200},
        {"title": "deaths that were culls", "x": {"label": "", "categories": columns},
         "y": {"label": "% of the deaths", "min": 0, "max": 100},
         "series": dots(columns, col("cull_share"), colours), "legend": False, "height": 200},
        {"title": "links cut for want of a stake", "x": {"label": "", "categories": columns},
         "y": {"label": "per agent and game", "min": 0, "max": 0.5},
         "series": dots(columns, col("cut"), colours), "legend": False, "height": 200}],
        columns=2, title="What the arm of family pools does differently",
        caption=f"Each dot is one run of Experiment 46 that lived past iteration {SETTLED} ("
                f"{sum(1 for r in every if 'per_thousand' in r)} of the {len(every)}); each bar the median "
                f"of its arm. All are read over the run's settled life, from iteration {SETTLED} to its end. "
                "Top left: the median of its agents per 1,000 tokens (logarithmic). Top right: the median "
                "share of all the world's tokens held by its richest agent (logarithmic). Bottom left: of all "
                "its deaths, the share that were culls, agents in a piece cut off from the largest piece of "
                "the network, against starving. Bottom right: the median of the links cut after a game "
                "because neither end staked a token on the other, per agent.",
        recipe=recipe(RUNS, DATA, [
            f"Keep the rows from iteration {SETTLED} on.",
            "Agents per 1,000 tokens: the median over the rows after the game of 1,000 · `nodes` / the run's "
            "tokens; the richest agent's share: the same of `maxTokens`.",
            "Culls: the sum of `orphaned` over every row (both phases) over the sum of `orphaned` and "
            "`starved`.",
            "Links cut: the median over the rows after the game of `prunedEdges` / `nodes`."]))
