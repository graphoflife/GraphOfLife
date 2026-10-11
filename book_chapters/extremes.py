# -*- coding: utf-8 -*-
"""
Part III, the largest and the smallest: the richest agent and the
best-connected, the poorest and the least connected, and the largest gain and
loss of a game — how they move over a world's life, how long anyone holds
first place, and how they grow with the size of a world.

The viewer reports each of these after every game (`maxTokens`, `minTokens`,
`maxDegree`, `medianDegree`, `minDegree`, `maxTokenAdded`, `maxTokenLost`).
Who holds them, and for how long, is read from the frames.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

import book_data as D
import book_figures as F
from book_figures import (FRAMES, STATS_FILE, band_series, chapter, describe, dots, line, mean_over, recipe,
                          series)
from book_chapters.common import (BAND_STEPS, BAND_WORDS, BLUE, CYAN, GREEN, GREY, ORANGE, RED, VIOLET,
                                  WORLD_SETS, WORLDS as W, YELLOW, listed)

#: The columns of a reigns table: after the game of iteration t, the richest agent and the
#: best-connected, and the largest gain and loss of that game.
REIGN_COLUMNS = ("t", "rich_id", "rich_tokens", "rich_degree", "rich_age", "hub_id", "hub_degree", "hub_tokens",
                 "hub_age", "gain", "loss", "gainer_id", "loser_id", "pair")


@D.measure("reigns", file="{run}.reigns.npz")
def reigns(run_id: str) -> Dict[str, Any]:
    """
    After every game: who is richest and who has the most connections, with
    their tokens, connections and age; the largest gain and loss of the game, by
    whom; and whether the two were joined at the start of the game (`pair` 1),
    so that one pile may simply have crossed one connection.
    """
    last = D.last_iteration(run_id)
    rows = []
    for t in range(last + 1):
        start, after = D.frame_at(run_id, t, 1), D.frame_at(run_id, t, 2)
        ids = np.asarray(after["ids"], dtype=np.int64)
        index = {u: i for i, u in enumerate(after["ids"])}
        edges = np.asarray(after["edges"], dtype=np.int64).reshape(-1, 2)
        degree = np.bincount(np.array([index[u] for u in edges.ravel()], dtype=np.int64), minlength=len(ids))
        tokens = np.asarray(after["tokens"], dtype=float)
        ages = np.asarray(after["ages"], dtype=float)
        delta = np.asarray(after["delta"], dtype=float)
        r, h, g, l = int(np.argmax(tokens)), int(np.argmax(degree)), int(np.argmax(delta)), int(np.argmin(delta))
        joined = {(min(a, b), max(a, b)) for a, b in start["edges"]}
        pair = (min(ids[g], ids[l]), max(ids[g], ids[l])) in joined
        rows.append([t, ids[r], tokens[r], degree[r], ages[r], ids[h], degree[h], tokens[h], ages[h],
                     delta[g], delta[l], ids[g], ids[l], float(pair)])
    return {"table": np.array(rows, dtype=float)}


def runs_of_reign(ids: np.ndarray) -> List[int]:
    """The lengths of the stretches of iterations in which the same agent holds first place."""
    if not len(ids):
        return []
    change = np.flatnonzero(np.diff(ids) != 0)
    edges = np.concatenate([[0], change + 1, [len(ids)]])
    return np.diff(edges).tolist()


#: The worlds of Experiment 8, from 800 to 409,600 tokens, run for 600 iterations each.
SIZES = ("The baseline worlds of Experiment 8, from 800 to 409,600 tokens, three seeds each, 600 iterations "
         "(Chapter 38; `python3 gol_lab.py run E08`)")
SIZE_WINDOW = (300, 599)


@chapter
def extremes(ch: F.Chapter) -> None:
    specs, alive = W.specs, W.alive
    n = len(specs)
    tok = W.tokens

    # ---- over a world's life ----
    def share(stat):
        return [(series(s.run_id, stat)[0], series(s.run_id, stat)[1] / tok) for s in specs]
    panels = [
        dict(title="Tokens, as a share of all", x={"label": "iteration"},
             y={"label": "share of all tokens (logarithmic)", "log": True},
             series=[band_series(share("maxTokens"), "the richest agent holds", RED),
                     band_series(share("maxTokenAdded"), "the largest gain of a game", GREEN),
                     band_series(share("maxTokenLost"), "the largest loss of a game", VIOLET),
                     band_series(share("meanTokens"), "the mean agent holds", GREY)]),
        dict(title="Connections", x={"label": "iteration"}, y={"label": "connections (logarithmic)", "log": True},
             series=[band_series([series(s.run_id, "maxDegree") for s in specs], "the best-connected agent", BLUE),
                     band_series([series(s.run_id, "meanDegree") for s in specs], "the mean agent", GREY),
                     band_series([series(s.run_id, "medianDegree") for s in specs], "the median agent", CYAN)])]
    ch.grid(
        "over-time", panels, columns=2, title="The largest, over a world's life",
        caption=f"After every game of the {n} worlds. Left: the tokens of the richest agent, the largest gain and "
                "the largest loss of one agent in that game, and the tokens of the mean agent, each as a share of all "
                f"{tok:,} tokens. Right: the connections of the best-connected agent, of the mean agent and of the "
                f"median one. Both on logarithmic axes; {BAND_WORDS[0].lower()}{BAND_WORDS[1:]}.",
        recipe=recipe(W.runs_text(), STATS_FILE,
                      ["Take the rows with `phase` = 2: `maxTokens`, `maxTokenAdded`, `maxTokenLost` and "
                       f"`meanTokens`, divided by {tok:,}; `maxDegree`, `meanDegree` and `medianDegree`.",
                       *BAND_STEPS]))
    for stat in ("maxTokens", "maxTokenAdded", "maxTokenLost", "minTokens"):
        ch.number(f"{stat}_settled", describe([mean_over(s.run_id, stat, *W.window) / tok for s in alive]))
    for stat in ("maxDegree", "medianDegree", "minDegree", "meanDegree"):
        ch.number(f"{stat}_settled", describe([mean_over(s.run_id, stat, *W.window) for s in alive]))
    ch.number("hub_share_settled", describe([mean_over(s.run_id, "maxDegree/nodes", *W.window) for s in alive]))
    ch.number("minTokens_ever", float(max(np.nanmax(series(s.run_id, "minTokens")[1]) for s in specs)))
    ch.number("minDegree_ever", float(max(np.nanmax(series(s.run_id, "minDegree")[1]) for s in specs)))

    # ---- who holds first place, and for how long ----
    kept = {s.run_id: reigns(s.run_id)["table"] for s in alive}
    col = {name: i for i, name in enumerate(REIGN_COLUMNS)}
    rich_runs, hub_runs, same, pairs = [], [], [], []
    for table in kept.values():
        settled = table[table[:, 0] >= W.settled]
        rich_runs += runs_of_reign(settled[:, col["rich_id"]])
        hub_runs += runs_of_reign(settled[:, col["hub_id"]])
        same.append(float(np.mean(settled[:, col["rich_id"]] == settled[:, col["hub_id"]])))
        pairs.append(float(np.mean(settled[:, col["pair"]])))
    out = []
    for runs, label, colour in ((rich_runs, "the richest agent", RED), (hub_runs, "the best-connected agent", BLUE)):
        v = np.asarray(runs, float)
        xs = np.unique(v)
        out.append(line(xs, [(v >= x).mean() for x in xs], f"{label} ({len(v):,} reigns)", colour, width=2))
    ch.figure(
        "reigns", title="How long first place is held",
        x={"label": "iterations in a row, L (logarithmic)", "log": True, "min": 1},
        y={"label": "share of the reigns lasting at least L (logarithmic)", "log": True, "max": 1},
        series=out,
        caption=f"After every game from iteration {W.settled} on, in {W.lived}, which agent holds the most tokens "
                "and which the most connections. A reign is a stretch of consecutive iterations with the same agent "
                "first; for every L, the share of the reigns that lasted at least L iterations, pooled over the "
                "worlds. A reign still going at the end of a run is counted as far as it went.",
        recipe=recipe(W.surviving_text(), FRAMES,
                      ["Read the frame after every game; the agent with the most `tokens`, and the one with the most "
                       "connections in `edges` (`book_chapters.extremes.reigns`).",
                       "Cut each run's sequence of first agents into stretches of the same agent; count lengths."]))
    for name, runs in (("rich", rich_runs), ("hub", hub_runs)):
        v = np.asarray(runs, float)
        ch.number(f"reigns_{name}", {"reigns": int(v.size), "median": float(np.median(v)), "mean": float(v.mean()),
                                     "share_1": float(np.mean(v == 1)), "share_ge_10": float(np.mean(v >= 10)),
                                     "share_ge_100": float(np.mean(v >= 100)), "longest": float(v.max())})
    ch.number("rich_is_hub", describe(same))
    ch.number("gain_loss_joined", describe(pairs))

    # ---- what kind of agent comes first ----
    fields = [("rich_degree", "connections of the richest"), ("hub_tokens", "tokens of the best-connected"),
              ("rich_age", "age of the richest"), ("hub_age", "age of the best-connected")]
    ch.number("first", {name: describe([float(np.median(t[t[:, 0] >= W.settled, col[name]])) for t in kept.values()])
                        for name, _ in fields})

    # ---- with the size of a world ----
    import gol_plan
    by_size: Dict[int, List[Dict[str, float]]] = {}
    for s in F.runs_of("E08", "baseline"):
        if F.lived(s) < s.until:
            continue
        t = int(s.config["total_tokens"])
        by_size.setdefault(t, []).append({
            "agents": mean_over(s.run_id, "nodes", *SIZE_WINDOW),
            "rich": mean_over(s.run_id, "maxTokens", *SIZE_WINDOW) / t,
            "hub": mean_over(s.run_id, "maxDegree", *SIZE_WINDOW),
            "gain": mean_over(s.run_id, "maxTokenAdded", *SIZE_WINDOW) / t})
    sizes = sorted(by_size)
    xs = np.array([w["agents"] for t in sizes for w in by_size[t]], float)
    fits = {}
    panels = []
    for key, title, ylabel, colour in (("rich", "The richest agent's share", "share of all tokens", RED),
                                       ("hub", "The best-connected agent", "connections", BLUE)):
        ys = np.array([w[key] for t in sizes for w in by_size[t]], float)
        slope, icpt = np.polyfit(np.log(xs), np.log(ys), 1)
        fits[key] = float(slope)
        grid = np.geomspace(xs.min(), xs.max(), 20)
        here = [mean_over(s.run_id, "nodes", *W.window) for s in alive]
        mine = [mean_over(s.run_id, "maxTokens" if key == "rich" else "maxDegree", *W.window) / (tok if key == "rich" else 1)
                for s in alive]
        panels.append(dict(title=title, x={"label": "agents (logarithmic)", "log": True},
                           y={"label": f"{ylabel} (logarithmic)", "log": True},
                           series=[{"label": "worlds of Experiment 8", "x": xs.tolist(), "y": ys.tolist(),
                                    "points": True, "size": 7, "colour": colour},
                                   line(grid, np.exp(icpt) * grid ** slope, f"least squares: slope {slope:.2f}", GREY,
                                        width=1.5, dash=[5, 4]),
                                   {"label": f"the {len(alive)} worlds of {tok:,} tokens", "x": here, "y": mine,
                                    "points": True, "size": 5, "colour": YELLOW}]))
    ch.grid(
        "with-size", panels, columns=2, title="The largest, by the size of a world",
        caption="For every world of Experiment 8 that lived to its end, from 800 to 409,600 tokens, its mean over "
                f"iterations {SIZE_WINDOW[0]}–{SIZE_WINDOW[1]} of the number of agents (x), and of the richest "
                "agent's share of all tokens (left) or the best-connected agent's connections (right); and the "
                f"same for the {len(alive)} worlds of {tok:,} tokens over iterations {W.span} (yellow). Both axes "
                "logarithmic; grey: the straight line least squares fits through the worlds of Experiment 8.",
        recipe=recipe(SIZES + "; " + W.runs_text(), STATS_FILE,
                      ["Average `nodes`, `maxTokens` / total tokens and `maxDegree` over the rows with `phase` = 2 in "
                       "the stretch.", "Fit ln y = a + b·ln x by least squares."]))
    ch.number("with_size", {"slopes": fits, "sizes": {str(t): {k: describe([w[k] for w in by_size[t]])
                                                               for k in ("agents", "rich", "hub", "gain")}
                                                      for t in sizes}})
