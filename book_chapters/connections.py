# -*- coding: utf-8 -*-
"""
the measurements of the baseline, what crosses a connection: how many tokens cross each connection in
a game, how much of that is cancelled by what comes back, how much goes round
in loops — the viewer's lightning — and which connections carry nothing and
are cut.

In the baseline every connection that no token crossed in a game is cut at the
end of that game (`prune_after` = `blotto`, `inactive_window` = `phase`;
[Every setting](settings.md)): the viewer counts them as
`prunedEdges`. So a connection lives exactly as long as tokens keep crossing
it, and the flow on the connections is also their survival.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict

import numpy as np

import book_data as D
import book_figures as F
from book_figures import (DEGREE_CLASSES, STATS_FILE, band_series, bars, chapter, describe, mean_over,
                          recipe, series)
from book_chapters.common import (BAND_STEPS, BAND_WORDS, BLUE, CYAN, GREEN, ORANGE, RED, WORLD_SETS,
                                  WORLDS as W)

#: How often the connections are looked at one by one.
FLOW_EVERY = 25
#: Tokens across one connection in one game, in classes of growing width.
FLOW_EDGES = (0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 5000, 10 ** 7)


@D.measure("edgeflow")
def edge_flow(run_id: str) -> Dict[str, Any]:
    """
    Every FLOW_EVERY iterations from the settled start on, every connection at
    the start of the game, with what was staked across it in each direction, and
    whether it was still there after the game.

    gross     how many connections carried a total (both directions) in each
              class of FLOW_EDGES — the first class is nothing at all;
    net       the same for what was left after cancelling one direction
              against the other;
    fate      for connections whose two ends both survived the game: carried
              nothing and cut, nothing and kept, something and cut, something
              and kept;
    by_degree for each class of DEGREE_CLASSES of the less connected end: the
              connections, and those that carried nothing.
    """
    last = D.last_iteration(run_id)
    gross = np.zeros(len(FLOW_EDGES) - 1)
    net = np.zeros(len(FLOW_EDGES) - 1)
    fate = np.zeros(4)
    by_degree = np.zeros((len(DEGREE_CLASSES), 2))
    looks = []
    for t in range(500, last + 1, FLOW_EVERY):
        start, after = D.frame_at(run_id, t, 1), D.frame_at(run_id, t, 2)
        pairs = {(min(a, b), max(a, b)) for a, b in start["edges"] if a != b}
        staked: Dict[Any, int] = defaultdict(int)
        for r in (after.get("decisions") or {}).get("allocations") or []:
            u = r["agent"]
            for v, amount in zip(r["targets"], r["alloc"]):
                if v != u and amount > 0:
                    staked[(u, v)] += int(amount)
        kept = {(min(a, b), max(a, b)) for a, b in after["edges"]}
        alive = set(after["ids"])
        degree = defaultdict(int)
        for a, b in pairs:
            degree[a] += 1
            degree[b] += 1
        totals, nets = [], []
        empty = 0
        for a, b in pairs:
            ab, ba = staked.get((a, b), 0), staked.get((b, a), 0)
            totals.append(ab + ba)
            nets.append(abs(ab - ba))
            carried = ab + ba > 0
            empty += not carried
            k = DEGREE_CLASSES.which(min(degree[a], degree[b]))
            if k is not None:
                by_degree[k, 0] += 1
                by_degree[k, 1] += not carried
            if a in alive and b in alive:
                fate[(0 if (a, b) not in kept else 1) + (0 if not carried else 2)] += 1
        gross += np.histogram(totals, bins=FLOW_EDGES)[0]
        net += np.histogram(nets, bins=FLOW_EDGES)[0]
        looks.append([t, len(pairs), empty, float(np.mean(totals)) if totals else 0.0,
                      float(np.max(totals)) if totals else 0.0])
    return {"gross": gross.tolist(), "net": net.tolist(), "fate": fate.tolist(), "by_degree": by_degree.tolist(),
            "looks": looks}


FLOW_DATA = ("each run's frames every 25 iterations from 500 on, read by `book_chapters.connections.edge_flow`: the "
             "connections of frame `2·t` (the start of the game of t), `decisions.allocations` and the connections "
             "of frame `2·t + 1` (after it)")


@chapter
def connections(ch: F.Chapter) -> None:
    specs, alive = W.specs, W.alive
    n = len(specs)
    kept = {s.run_id: edge_flow(s.run_id) for s in alive}

    # ---- how many tokens cross a connection ----
    labels = ["0", "1", "2", "3–4", "5–9", "10–19", "20–49", "50–99", "100–199", "200–499", "500–999",
              "1,000–4,999", "5,000+"]
    gross = np.sum([m["gross"] for m in kept.values()], axis=0)
    net = np.sum([m["net"] for m in kept.values()], axis=0)
    ch.figure(
        "per-connection", title="Tokens across one connection in one game",
        x={"label": "tokens, both directions together", "categories": labels},
        y={"label": "share of the connections (logarithmic)", "log": True},
        series=[bars(labels, (gross / gross.sum()).tolist(), BLUE, "staked across it", (-0.38, 0)),
                bars(labels, (net / net.sum()).tolist(), CYAN, "left after cancelling the two directions", (0, 0.38))],
        caption="Every connection at the start of the games of every 25th iteration from 500 on, in "
                f"{W.lived}, pooled: how many tokens its two agents staked on each other in that game, both directions "
                "added (blue), and what is left when the smaller direction is cancelled against the larger (cyan; "
                "[Token flow](token-flow.md)). A connection in the first class carried nothing and is cut "
                "at the end of the game.",
        recipe=recipe(W.surviving_text(), FLOW_DATA,
                      ["For every connection (a, b) of frame `2·t`: the tokens a staked on b and b on a, from "
                       "`decisions.allocations` of frame `2·t + 1`.",
                       "Count the connections by the sum, and by the difference, in the classes shown."]))
    ch.number("per_connection", {"gross": dict(zip(labels, (gross / gross.sum()).tolist())),
                                 "net": dict(zip(labels, (net / net.sum()).tolist())),
                                 "connections": float(gross.sum())})

    # ---- which connections carry nothing, and are cut ----
    by_degree = np.sum([m["by_degree"] for m in kept.values()], axis=0)
    dl = DEGREE_CLASSES.labels
    share_empty = (by_degree[:, 1] / np.maximum(by_degree[:, 0], 1)).tolist()
    looks = {s: np.array(m["looks"], float) for s, m in kept.items()}
    cut = [(x[:, 0], x[:, 2] / x[:, 1]) for x in looks.values()]
    ch.grid(
        "cut", [
            dict(title="Connections that carried nothing", x={"label": "iteration"},
                 y={"label": "share of the connections", "min": 0},
                 series=[band_series(cut, None, RED)], legend=False),
            dict(title="By the less connected end", x={"label": "connections of the less connected end",
                                                       "categories": dl},
                 y={"label": "share that carried nothing", "min": 0, "max": 1},
                 series=[bars(dl, share_empty, RED)], legend=False)],
        columns=2, title="The connections a game cuts",
        caption="Left: at the start of the game of every 25th iteration from 500 on, the share of the connections "
                "across which no token was staked in that game — the ones the game cuts; "
                f"{BAND_WORDS[0].lower()}{BAND_WORDS[1:]}. Right: the same share, pooled over the games and worlds, "
                "for connections sorted by the number of connections of their less connected end.",
        recipe=recipe(W.surviving_text(), FLOW_DATA,
                      ["As above; a connection carried nothing if neither end staked a token on the other.",
                       "Sort the connections by the smaller of their two ends' degrees in frame `2·t`."]))
    fate = np.sum([m["fate"] for m in kept.values()], axis=0)
    ch.number("cut", {"share_empty": describe([float(np.mean(x[:, 2] / x[:, 1])) for x in looks.values()]),
                      "by_degree": dict(zip(dl, share_empty)),
                      "fate": dict(zip(("empty cut", "empty kept", "carried cut", "carried kept"), fate.tolist()))})
    ch.number("prunedEdges_settled", describe([mean_over(s.run_id, "prunedEdges/edges", *W.window) for s in alive]))

    # ---- the flow, in round figures ----
    panels = []
    for title, parts in (("Tokens on a connection", [("meanEdgeFlow", "the mean over the connections that carried tokens", BLUE),
                                                     ("maxEdgeFlow", "the most on one connection", RED)]),
                         ("Loops of flow", [("lightningLongest", "the longest loop found, as staked", ORANGE),
                                            ("netLightningLongest", "the longest after cancelling", GREEN)])):
        panels.append(dict(title=title, x={"label": "iteration"}, y={"label": "tokens" if "Tokens" in title else "agents in the loop",
                                                                     "log": "Tokens" in title},
                           series=[band_series([series(s.run_id, stat) for s in specs], label, colour)
                                   for stat, label, colour in parts]))
    ch.grid(
        "round-figures", panels, columns=2, title="Flow on the connections, and the loops it makes",
        caption=f"After every game of the {n} worlds. Left: the viewer's `meanEdgeFlow` — the tokens staked across "
                "a connection, over the connections something crossed — and `maxEdgeFlow`, the most across one, on a "
                "logarithmic axis. Right, every 25 iterations: the longest closed loop of flow the greedy peel of "
                "[Lightning](lightning.md) found in the flow as staked (`lightningLongest`) and in what is "
                f"left after cancelling (`netLightningLongest`). {BAND_WORDS}.",
        recipe=recipe(W.runs_text(), STATS_FILE,
                      ["Take the rows with `phase` = 2: `meanEdgeFlow`, `maxEdgeFlow`, `lightningLongest` and "
                       "`netLightningLongest`.", *BAND_STEPS]))
    for stat in ("meanEdgeFlow", "maxEdgeFlow", "totalFlow", "lightningScore", "lightningLongest",
                 "netLightningScore", "netLightningLongest", "cyclingShare", "netCyclingShare"):
        ch.number(f"{stat}_settled", describe([mean_over(s.run_id, stat, *W.window) for s in alive]))
    ch.number("maxEdgeFlow_share_settled", describe([mean_over(s.run_id, "maxEdgeFlow", *W.window) / W.tokens
                                                    for s in alive]))

    # ---- the worlds of 10,000 tokens beside these ----
    other = WORLD_SETS["E02"] if W.experiment != "E02" else None
    if other is not None:
        ch.number("small", {stat: describe([mean_over(s.run_id, stat, *other.window) for s in other.alive])
                            for stat in ("meanEdgeFlow", "maxEdgeFlow", "lightningScore", "lightningLongest",
                                         "netLightningScore", "netLightningLongest", "prunedEdges/edges")})
