# -*- coding: utf-8 -*-
"""
the measurements of the baseline, the largest and the smallest: the richest agent and the
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
from book_figures import FRAMES, STATS_FILE, band_series, chapter, describe, line, mean_over, recipe, series
from book_chapters.common import (BAND_STEPS, BAND_WORDS, BLUE, CYAN, GREEN, GREY, ORANGE, RED, STAR, VIOLET,
                                  WORLD_SETS, WORLDS as W, YELLOW)

#: The columns of a reigns table: after the game of iteration t, the richest agent and the
#: best-connected, and the largest gain and loss of that game.
REIGN_COLUMNS = ("t", "rich_id", "rich_tokens", "rich_degree", "rich_age", "hub_id", "hub_degree", "hub_tokens",
                 "hub_age", "gain", "loss", "gainer_id", "loser_id", "pair", "rich_parent", "hub_parent")


@D.measure("reigns", file="{run}.reigns.npz")
def reigns(run_id: str) -> Dict[str, Any]:
    """
    After every game: who is richest and who has the most connections, with
    their tokens, connections and age; the largest gain and loss of the game, by
    whom; whether the two were joined at the start of the game (`pair` 1),
    so that one pile may simply have crossed one connection; and the parent
    agent of the richest and of the best-connected (−1 for a founder), so that
    a first place handed from parent to child can be told apart.
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
        parents = np.asarray(after["parent_ids"], dtype=np.int64)
        r, h, g, l = int(np.argmax(tokens)), int(np.argmax(degree)), int(np.argmax(delta)), int(np.argmin(delta))
        joined = {(min(a, b), max(a, b)) for a, b in start["edges"]}
        pair = (min(ids[g], ids[l]), max(ids[g], ids[l])) in joined
        rows.append([t, ids[r], tokens[r], degree[r], ages[r], ids[h], degree[h], tokens[h], ages[h],
                     delta[g], delta[l], ids[g], ids[l], float(pair), parents[r], parents[h]])
    return {"table": np.array(rows, dtype=float)}


#: How many games a star is followed for.
STAR_GAMES = 100


@D.measure("star")
def star(run_id: str) -> Dict[str, Any]:
    """
    The anatomy of a world's biggest star: at the game of its settled life whose
    best-connected agent held the largest share of the world, and before which
    STAR_GAMES more games are left —

    at        that game: the agents at its start, the hub's connections and tokens,
              its share of all tokens; how many of its neighbours are leaves and
              what they hold; what the hub staked on its neighbours and on its
              own node, and what its leaves staked on the hub's node;
    games     the STAR_GAMES games from there: whether the best-connected agent's
              node was taken by a coalition, and how many different brains the
              best-connected agent carried after them;
    handover  the first time in those games that first place passed to another
              agent: the old hub's connections before and after, the new one's,
              its age, and whether it is the old hub's child.
    """
    until = D.last_iteration(run_id) - STAR_GAMES
    settled = [r for r in F.rows(run_id) if r.get("phase") == 2 and F.SETTLED <= r["iteration"] <= until]
    t0 = int(max(settled, key=lambda r: r["maxDegree"] / r["nodes"])["iteration"])

    def hub_of(frame):
        degree: Dict[int, int] = {}
        for a, b in frame["edges"]:
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1
        return max(degree, key=degree.get), degree

    start, after = D.frame_at(run_id, t0, 1), D.frame_at(run_id, t0, 2)
    hub, degree = hub_of(start)
    tokens = dict(zip(start["ids"], start["tokens"]))
    neighbours = {b if a == hub else a for a, b in start["edges"] if hub in (a, b)}
    leaves = [u for u in neighbours if degree[u] == 1]
    allocations = {r["agent"]: r for r in after["decisions"]["allocations"]}
    mine = allocations[hub]
    on = dict(zip(mine["targets"], mine["alloc"]))
    leaf_share = [allocations[u]["alloc"][allocations[u]["targets"].index(hub)] / max(1, tokens[u])
                  for u in leaves if u in allocations and hub in allocations[u]["targets"]]
    at = {"t": t0, "agents": len(start["ids"]), "hub_degree": degree[hub], "hub_tokens": tokens[hub],
          "hub_token_share": tokens[hub] / sum(start["tokens"]), "leaf_neighbours": len(leaves),
          "leaf_tokens": float(np.mean([tokens[u] for u in leaves])) if leaves else None,
          "staked_per_neighbour": float(np.mean([on.get(u, 0) for u in neighbours])),
          "neighbours_staked_on": float(np.mean([on.get(u, 0) > 0 for u in neighbours])),
          "staked_most": float(max(on.get(u, 0) for u in neighbours)), "staked_home": float(on.get(hub, 0)),
          "leaf_share_on_hub": float(np.mean(leaf_share)) if leaf_share else None}

    taken, brains, handover = 0, set(), None
    for t in range(t0, t0 + STAR_GAMES):
        start, after = D.frame_at(run_id, t, 1), D.frame_at(run_id, t, 2)
        h, deg = hub_of(start)
        won = next(w for w in after["decisions"]["winners"] if w["node"] == h)
        taken += bool(won["revolt"])
        if h in after["ids"]:
            brains.add(after["brain_ids"][after["ids"].index(h)])
        if handover is None and h != hub:
            parents = dict(zip(start["ids"], start["parent_ids"]))
            before = D.frame_at(run_id, t - 1, 2)
            _, old_degree = hub_of(before)
            handover = {"t": t, "old_before": old_degree.get(hub, 0), "old_after": deg.get(hub, 0),
                        "new": deg[h], "new_age": int(dict(zip(start["ids"], start["ages"]))[h]),
                        "child": parents.get(h) == hub}
    return {"at": at, "games": {"count": STAR_GAMES, "taken_by_coalition": taken, "brains": len(brains)},
            "handover": handover}


def runs_of_reign(ids: np.ndarray) -> List[int]:
    """The lengths of the stretches of iterations in which the same agent holds first place."""
    if not len(ids):
        return []
    change = np.flatnonzero(np.diff(ids) != 0)
    edges = np.concatenate([[0], change + 1, [len(ids)]])
    return np.diff(edges).tolist()


#: The worlds of Experiment 8, from 800 to 409,600 tokens, run for 600 iterations each.
SIZES = ("The baseline worlds of Experiment 8, from 800 to 409,600 tokens, three seeds each, 600 iterations "
         "([How does a world's size follow its tokens?](how-does-a-worlds-size-follow-its-tokens.md); `python3 gol_lab.py run E08`)")
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

    # ---- stars: worlds held together by one hub ----
    hub_mean = {s.run_id: mean_over(s.run_id, "maxDegree/nodes", *W.window) for s in alive}
    stars = sorted((r for r in hub_mean if hub_mean[r] >= STAR), key=lambda r: -hub_mean[r])
    rest = [s for s in alive if s.run_id not in stars]
    series_ = [band_series([series(s.run_id, "maxDegree/nodes") for s in rest],
                           f"the other {len(rest)} worlds", GREY)]
    for k, run in enumerate(stars):
        its, v = series(run, "maxDegree/nodes")
        series_.append(line(its, v, f"seed {int(run.rsplit('s', 1)[1])}", [RED, ORANGE, YELLOW, GREEN, CYAN, BLUE,
                                                                             VIOLET][k % 7], width=1.2, alpha=0.9))
    scatter = []
    small = WORLD_SETS["E02"] if W.experiment != "E02" else None
    for worlds, colour, label in ((W, BLUE, f"{W.tokens:,} tokens"),) + (((small, YELLOW, f"{small.tokens:,} tokens"),)
                                                                         if small else ()):
        hx = [mean_over(s.run_id, "maxDegree/nodes", *worlds.window) for s in worlds.alive]
        cy = [mean_over(s.run_id, "coreShare", *worlds.window) for s in worlds.alive]
        scatter.append({"label": f"a world of {label}", "x": hx, "y": cy, "points": True, "size": 7, "colour": colour})
        ch.number(f"stars_{worlds.experiment}", {"worlds": len(hx), "star_worlds": int(sum(x >= STAR for x in hx)),
                                                  "corr_core_hub": float(np.corrcoef(hx, cy)[0, 1]),
                                                  "largest": float(max(hx))})
    ch.grid(
        "stars", [
            dict(title="The largest hub's share of the world", x={"label": "iteration"},
                 y={"label": "agents joined to the best-connected agent, share", "min": 0, "max": 1}, series=series_),
            dict(title="Hub and core", x={"label": "agents joined to the largest hub, share (mean)", "min": 0},
                 y={"label": "agents in the core, share (mean)", "min": 0, "max": 1}, series=scatter)],
        columns=2, title="Worlds held together by one agent",
        caption=f"Left: after every game, the connections of the best-connected agent divided by the agents alive, in "
                f"the worlds whose best-connected agent held, on average over iterations {W.span}, a tenth of the "
                f"world or more (one line each), and in the other worlds ({BAND_WORDS[0].lower()}{BAND_WORDS[1:]}). "
                f"Right: one dot "
                f"per world, its mean over the settled life of that share and of the share of agents in the core "
                f"([Core, trees and leaves](core-trees-leaves.md)), for the worlds of {W.tokens:,} tokens"
                + (f" (blue) and of {small.tokens:,} (yellow, over iterations {small.span})." if small else "."),
        recipe=recipe(W.runs_text() + (" " + small.surviving_text() if small else ""), STATS_FILE,
                      ["Take the rows with `phase` = 2: `maxDegree` / `nodes`, and `coreShare` (every 25 iterations).",
                       "Left: one line per star world; the band over the others as every band. Right: each world's means "
                       "over its settled life."]))

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
    # ---- succession: is first place handed down? ----
    succession, dynasty_runs = {}, {}
    for who in ("rich", "hub"):
        changes = to_child = newborn = 0
        dynasties = []
        for table in kept.values():
            settled = table[table[:, 0] >= W.settled]
            ids, parents, ages = (settled[:, col[f"{who}_id"]], settled[:, col[f"{who}_parent"]],
                                  settled[:, col[f"{who}_age"]])
            start = 0
            for i in range(1, len(ids)):
                if ids[i] == ids[i - 1]:
                    continue
                changes += 1
                handed = parents[i] == ids[i - 1]
                to_child += handed
                newborn += ages[i] == 0
                if not handed:
                    dynasties.append(i - start)
                    start = i
            dynasties.append(len(ids) - start)
        v = np.asarray(dynasties, float)
        dynasty_runs[who] = v
        succession[who] = {"changes": changes, "to_child": to_child / max(1, changes),
                           "to_newborn": newborn / max(1, changes), "dynasties": int(v.size),
                           "dynasty_median": float(np.median(v)), "dynasty_longest": float(v.max()),
                           "dynasty_share_ge_100": float(np.mean(v >= 100))}
    ch.number("succession", succession)

    # ---- the biggest star, from inside ----
    if stars:
        biggest = stars[0]
        table = reigns(biggest)["table"]
        settled = table[table[:, 0] >= W.settled]
        ids, parents, ages = (settled[:, col["hub_id"]], settled[:, col["hub_parent"]], settled[:, col["hub_age"]])
        changes = np.flatnonzero(np.diff(ids) != 0) + 1
        anatomy = star(biggest)
        ch.number("star", {"run": biggest, "seed": int(biggest.rsplit("s", 1)[1]),
                           **{k: anatomy[k] for k in ("at", "games", "handover")},
                           "settled_changes": int(changes.size),
                           "to_newborn_child": float(np.mean([parents[i] == ids[i - 1] and ages[i] == 0
                                                              for i in changes])) if changes.size else None})

    out = []
    for runs, label, colour in ((rich_runs, "the richest agent", RED), (hub_runs, "the best-connected agent", BLUE)):
        v = np.asarray(runs, float)
        xs = np.unique(v)
        out.append(line(xs, [(v >= x).mean() for x in xs], f"{label} ({len(v):,} reigns)", colour, width=2))
    for who, label, colour in (("rich", "the richest, handed down", ORANGE), ("hub", "the best-connected, handed down",
                                                                              CYAN)):
        v = dynasty_runs[who]
        xs = np.unique(v)
        out.append(line(xs, [(v >= x).mean() for x in xs], f"{label} ({len(v):,} dynasties)", colour, width=1.5,
                        dash=[5, 4]))
    ch.figure(
        "reigns", title="How long first place is held",
        x={"label": "iterations in a row, L (logarithmic)", "log": True, "min": 1},
        y={"label": "share of the reigns lasting at least L (logarithmic)", "log": True, "max": 1},
        series=out,
        caption=f"After every game from iteration {W.settled} on, in {W.lived}, which agent holds the most tokens "
                "and which the most connections. A reign is a stretch of consecutive iterations with the same agent "
                "first; for every L, the share of the reigns that lasted at least L iterations, pooled over the "
                "worlds. Dashed: the same for dynasties — a dynasty goes on while every new first agent is a child "
                "of the one before it, and ends when first place passes to anyone else — pooled over the "
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
