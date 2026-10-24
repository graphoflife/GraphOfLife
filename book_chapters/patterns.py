# -*- coding: utf-8 -*-
"""
Gliders, and who helps whom: the patterns in the record of play,
and help as the rules define it.

The worlds are the ten B2 worlds of Experiment 16 and the first ten B1 worlds
of the baseline set, read over their recorded games 1,000 to 2,000, and pilot
worlds with and without alliances (Alliances), whose games are recorded by
going on from a checkpoint for 200 iterations. The scripts of
research/oee/checks compute everything and keep it in GraphOfLifeRuns/.pilots,
one JSON file per check and set of worlds; this module draws it.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

import book_figures as F
from book_figures import chapter, recipe
from book_chapters.common import BLUE, GREY, ORANGE, VIOLET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILOTS = os.path.join(ROOT, "GraphOfLifeRuns", ".pilots")
CHECKS = "research/oee/checks"

B2 = [f"B2-10000-s{i:03d}" for i in range(1, 11)]
B1 = [f"B1-10000-s{i:03d}" for i in range(1, 11)]
ALLIES = ["A10k_allies-s1_500", "A10k_allies-s2_500", "AL_allies10k-s3_2000", "AL_allies10k-s4_2000"]
CONTROLS = ["A10k_ctrl-s1_500", "A10k_ctrl-s2_500"]
COLOURS = {"B2": BLUE, "B1": ORANGE, "another-game": VIOLET, "no alliances": GREY}
#: Helping pairs a stretch of 50 games needs before the share of them returned, going round or
#: kept in the family means anything.
ENOUGH_PAIRS = 100

LAB_RUNS = ("The ten B2 worlds of Experiment 16, `B2-10000-s001` … `-s010`, and the first ten worlds of the "
            "baseline set, `B1-10000-s001` … `-s010` (B1 at 10,000 tokens, Experiment 2). Both record every "
            "game with every agent's stakes; read are the games of iterations 1,000 to 2,000.")
PILOT_RUNS = ("Pilot worlds of [Alliances](another-game.md#alliances) (`GraphOfLifeRuns/.pilots`): with alliances `A10k/allies-s1` and "
              "`-s2` from iteration 500 and `AL/allies10k-s3` and `-s4` from iteration 2,000; without, "
              "`A10k/ctrl-s1` and `-s2` from iteration 500. Each goes on from its checkpoint for 200 "
              "iterations with every game recorded in the lab's form "
              f"(`python3 {CHECKS}/pilot_frames.py <world> --at <iteration> --games 200`).")


def _load(name: str) -> Optional[Any]:
    path = os.path.join(PILOTS, name)
    return json.load(open(path)) if os.path.exists(path) else None


def _by_run(name: str) -> Dict[str, Dict[str, Any]]:
    return {r["run"]: r for r in (_load(name) or [])}


def _pilots(names: List[str]) -> List[Dict[str, Any]]:
    return [r for r in (_load(f"frames_{n}.json") for n in names) if r]


def _med(values: List[Optional[float]]) -> Optional[float]:
    v = [x for x in values if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def _summary(values: List[Optional[float]]) -> Dict[str, Optional[float]]:
    v = [x for x in values if x is not None and np.isfinite(x)]
    return {"median": float(np.median(v)) if v else None, "min": min(v) if v else None,
            "max": max(v) if v else None, "worlds": len(v)}


def _panel(title: str, ylabel: str, groups: Dict[str, List[Optional[float]]], guide: Optional[float] = None,
           **y: Any) -> Dict[str, Any]:
    cats = [c for c, v in groups.items() if any(x is not None for x in v)]
    chart = {"title": title, "x": {"categories": cats}, "y": {"label": ylabel, **y},
             "series": F.dots(cats, [groups[c] for c in cats], [COLOURS[c] for c in cats])}
    if guide is not None:
        chart["guides"] = [{"axis": "y", "at": guide, "label": ""}]
    return chart


def _chain_picture(run: str, row: Dict[str, Any], title: str) -> Optional[Dict[str, Any]]:
    """A chain on the network of its last game: the nodes it visited and their neighbours."""
    import networkx as nx
    import book_data as D
    frame = D.frame_at(run, row["t1"], 2)
    G = nx.Graph()
    G.add_edges_from(map(tuple, frame["edges"]))
    path = row["nodes"]
    keep = {u for u in path if u in G}
    for u in list(keep):
        keep.update(G.neighbors(u))
    sub = G.subgraph(keep).copy()
    moves = {frozenset(m) for m in zip(path, path[1:]) if m[0] in keep and m[1] in keep and m[0] != m[1]}
    sub.add_edges_from(tuple(m) for m in moves)
    if not sub.number_of_nodes():
        return None
    # The piece of the network that holds most of the chain; stray pieces of neighbours go.
    on_path = set(path)
    sub = sub.subgraph(max(nx.connected_components(sub), key=lambda c: len(c & on_path))).copy()
    nodes = sorted(sub.nodes())
    index = {u: i for i, u in enumerate(nodes)}
    pos = nx.forceatlas2_layout(sub, max_iter=300, seed=1)
    group = [3 if u == path[0] else 2 if u == path[-1] else 1 if u in path else 0 for u in nodes]
    edges = [(index[a], index[b]) for a, b in sub.edges()]
    return {"kind": "network", "title": title,
            "nodes": {"x": [float(pos[u][0]) for u in nodes], "y": [float(pos[u][1]) for u in nodes],
                      "group": group},
            "edges": edges, "edgeGroup": [1 if frozenset((nodes[a], nodes[b])) in moves else 0 for a, b in edges],
            "colour": {"labels": ["a neighbour of the chain", "a node it took", "where it ended",
                                  "where it began"][:4 if 3 in group else 3]},
            "edgeLabels": ["a move"], "nodeSize": 3.0, "edgeAlpha": 0.35}


def _examples(ch: F.Chapter) -> None:
    """One chain of each rule set drawn: the one that visited the most different nodes."""
    rows = _by_run("gliders_examples.json")
    picked, panels = {}, []
    for base, run in (("B2", B2[0]), ("B1", B1[0])):
        mine = [r for r in rows.get(run, {}).get("examined_rows", []) if r.get("nodes")]
        if not mine:
            return
        best = max(mine, key=lambda r: (len(set(r["nodes"])), r["length"]))
        picked[base] = {"run": run, "moves": best["length"], "nodes": len(set(best["nodes"])),
                        "reach": best["reach"], "walk": best["walk"], "games": [best["t0"], best["t1"]],
                        "home_kept": best["home_kept"], "packet_first": best["packet_first"],
                        "packet_last": best["packet_last"], "end": best["end"]}
        kind = "puffer" if best["home_kept"] >= 0.5 else "glider"
        panel = _chain_picture(run, best, f"A {kind} in {base}: {best['length']} moves")
        if panel:
            panels.append(panel)
    if len(panels) < 2:
        return
    ch.grid("example", panels, title="Two chains on their networks",
            caption=(f"Left, in `{B2[0]}`, the chain of its 300 longest that took the most different "
                     f"nodes: {picked['B2']['moves']} moves over {picked['B2']['nodes']} nodes in games "
                     f"{picked['B2']['games'][0]:,} to {picked['B2']['games'][1]:,}, its packet "
                     f"{picked['B2']['packet_first']} tokens at the first move and "
                     f"{picked['B2']['packet_last']} at the last. Right, the same for `{B1[0]}`: "
                     f"{picked['B1']['moves']} moves over {picked['B1']['nodes']} nodes, games "
                     f"{picked['B1']['games'][0]:,} to {picked['B1']['games'][1]:,}, its packet "
                     f"{picked['B1']['packet_first']} tokens, then {picked['B1']['packet_last']}. Drawn is "
                     "the network of each chain's last game around the nodes it took, with their "
                     "neighbours; the chain's moves are the coloured links, drawn even where the "
                     "connection a move crossed has gone since. A node the chain took that no longer "
                     "exists at its last game is not drawn, the node where it began among them."),
            recipe=recipe(LAB_RUNS, "the two runs' frames of the game phase, with their agents' decisions", [
                "Track the chains as for the figures below and keep the 300 longest of each world, with "
                "the nodes each took (`research/oee/checks/gliders.py`).",
                "Of those, take the chain that took the most different nodes.",
                "Take the network of its last game, the nodes it took that are still there and their "
                "neighbours; add the moves as links; lay it out with "
                "`networkx.forceatlas2_layout(G, max_iter=300, seed=1)`."]),
            columns=2)
    ch.number("examples", picked)


@chapter
def patterns(ch: F.Chapter) -> None:
    gl = _by_run("gliders_20.json")
    st = {**_by_run("structures_B2.json"), **_by_run("structures_B1.json")}
    if not all(w in gl and w in st for w in B2 + B1):
        return
    allies, controls = _pilots(ALLIES), _pilots(CONTROLS)

    # --- How long chains get, against chance --------------------------------
    series = []
    for base, worlds in (("B2", B2), ("B1", B1)):
        for k, w in enumerate(worlds):
            counts = {int(L): n for L, n in gl[w]["length_counts"].items()}
            Ls = sorted(counts)
            series.append({"label": f"{base}, seen" if k == 0 else None, "x": Ls, "y": [counts[L] for L in Ls],
                           "colour": COLOURS[base], "width": 1.2, "alpha": 0.8})
            p, n = st[w]["packet_go_on_chance"], gl[w]["chains"]
            Lc = [L for L in range(1, max(Ls) + 1) if n * (1 - p) * p ** (L - 1) >= 0.1]
            series.append({"label": f"{base}, chance" if k == 0 else None, "x": Lc,
                           "y": [n * (1 - p) * p ** (L - 1) for L in Lc],
                           "colour": COLOURS[base], "width": 1, "dash": [4, 4], "alpha": 0.6})
    ch.figure("chains", title="How long the chains of takeovers get",
              x={"label": "moves in a row", "log": True, "min": 1, "max": 200},
              y={"label": "chains in 1,000 games (logarithmic)", "log": True, "min": 0.1},
              series=series,
              caption=("One line per world. A move is a takeover made with at least half of the taker's "
                       "tokens; a chain is a run of moves, each made from the node the last one took. "
                       "Dashed: the chains there would be if every node holding a packet moved it on with "
                       "the same chance as any agent moves one in a game, whatever came before."),
              recipe=recipe(LAB_RUNS, "each run's frames of the game phase, with its agents' decisions", [
                  "In every game, find the moves: an agent's stake on a neighbour's node that took it and "
                  "was at least half of the agent's tokens (`research/oee/checks/gliders.py`).",
                  "A move made from a node that a move took in the game before continues that chain; any "
                  "other starts one. Count the chains by their number of moves.",
                  "Chance: with p the share of agents that make a move in a game, averaged over the games "
                  "(`structures.py`, `packet_go_on_chance`), and N the chains, N (1 − p) p^(L−1) chains "
                  "would have L moves."]))

    _examples(ch)

    # --- What the chains are --------------------------------------------------
    def both(f):
        return {"B2": [f(gl[w], st[w]) for w in B2], "B1": [f(gl[w], st[w]) for w in B1]}

    over = both(lambda g, s: s["packet_chains_ge3"] / s["packet_chains_ge3_chance"])
    home = both(lambda g, s: g["home_kept_long"])
    lag2 = both(lambda g, s: g["lag2_return"] / g["lag2_chance"])
    walk = both(lambda g, s: g["reach_over_walk"])
    ch.grid("zoo", [
        _panel("Chains of three moves or more", "times as many as chance (logarithmic)", over, 1,
               log=True, min=1, max=100),
        _panel("The mover keeps the node it left", "share of moves", home, None, min=0, max=1),
        _panel("Back to the node before", "returns at lag 2 ÷ chance", lag2, 1, min=0.8, max=1.6),
        _panel("How far chains get", "distance ÷ a random walk's", walk, 1, min=0, max=1.5)],
        title="What the chains are",
        caption=("One dot per world, the bar the median of ten. Chains of at least three moves, over games "
                 "1,000 to 2,000. Top right: after how many of their moves the mover still held its own "
                 "node, a puffer leaving copies behind (high) or a glider leaving nothing (low). Bottom "
                 "left: how often a move went back to the node the packet held two moves before, divided "
                 "by the chance of picking that node among the mover's neighbours. Bottom right: the "
                 "distance from a chain's first node to its last, divided by a random walk's of the same "
                 "length from the same node, for the 300 longest chains of each world."),
        recipe=recipe(LAB_RUNS, "each run's frames of the game phase, with its agents' decisions", [
            "Find the chains as for the figure above (`research/oee/checks/gliders.py`).",
            "Chance of a chain of at least three moves: N p² for N chains and p as above "
            "(`structures.py`).",
            "For every move, note whether the mover won its own node in the same game.",
            "For every move from the third on, note whether it went to the node the packet held two "
            "moves before, and the chance 1/k of that node among the mover's k neighbours.",
            "Distance: steps through the network at the chain's end between its first and last node; "
            "the walk's: the median of five random walks with as many steps from the first node."]))
    for name, groups in (("chains_over_chance", over), ("home_kept", home), ("lag2_over_chance", lag2),
                         ("reach_over_walk", walk)):
        ch.number(name, {c: _summary(v) for c, v in groups.items()})
    for key in ("moves", "taken", "length_max", "home_starved_long", "packet_first_median", "packet_last_median",
                "lag3_return", "lag3_chance", "launches_per_node_dispersion", "launches_per_node_max",
                "share_of_taken_in_long", "same_brain_median", "long_chains"):
        ch.number(f"gl_{key}", {c: _summary([gl[w][key] for w in ws]) for c, ws in (("B2", B2), ("B1", B1))})
    ch.number("all_in", {c: _summary([gl[w]["moves"] / gl[w]["taken"] for w in ws])
                         for c, ws in (("B2", B2), ("B1", B1))})
    ch.number("ends_long", {c: {k: _med([gl[w]["end_long"].get(k, 0) / sum(gl[w]["end_long"].values()) for w in ws])
                                for k in ("kept", "taken", "starved")} for c, ws in (("B2", B2), ("B1", B1))})
    ch.number("partnerships", {c: {"kept_in": _summary([st[w]["kept_in_partnership"] for w in ws]),
                                   "kept_otherwise": _summary([st[w]["kept_otherwise"] for w in ws]),
                                   "longest": _summary([st[w]["partnership_runs_max"] for w in ws]),
                                   "two_loops": _summary([st[w]["two_loops_per_channel"] for w in ws]),
                                   "two_loops_chance": _summary([st[w]["two_loops_chance"] for w in ws]),
                                   "channel_repeat": _summary([st[w]["channel_repeat"] for w in ws]),
                                   "channel_repeat_chance": _summary([st[w]["channel_repeat_chance"] for w in ws])}
                               for c, ws in (("B2", B2), ("B1", B1))})
    rot = _by_run("rotations_20.json")
    if all(w in rot for w in B2 + B1):
        ch.number("swaps", {c: {"swaps_over_chance": _summary([rot[w]["swaps"] / rot[w]["swaps_chance"] for w in ws]),
                                "shift_share": _summary([rot[w]["shift_share"] for w in ws]),
                                "rotations": _summary([rot[w]["rotations"] for w in ws])}
                            for c, ws in (("B2", B2), ("B1", B1))})

    # --- Where the tokens go -------------------------------------------------
    circ = _by_run("circulation_20.json")
    if all(w in circ for w in B2 + B1):
        def flows(kind, key):
            g = {"B2": [circ[w][kind][key] for w in B2], "B1": [circ[w][kind][key] for w in B1]}
            if allies:
                g["another-game"] = [p["token-flow"][kind][key] for p in allies]
            if controls:
                g["no alliances"] = [p["token-flow"][kind][key] for p in controls]
            return g

        def excess(kind):
            share, flip = flows(kind, "circulation_share"), flows(kind, "flip_share")
            return {c: [a - b for a, b in zip(share[c], flip[c])] for c in share}

        ch.grid("flows", [
            _panel("Gifts", "circulation share − its chance level", excess("gift"), 0, min=-0.3, max=0.3),
            _panel("Takeovers", "circulation share − its chance level", excess("take"), 0, min=-0.3, max=0.3),
            _panel("Gifts, the next 50 games", "correlation of the downhill part", flows("gift", "persistence_gradient"),
                   0, min=-0.2, max=0.6),
            _panel("Gifts, the next 50 games", "correlation of the circulation", flows("gift", "persistence_circulation"),
                   0, min=-0.2, max=0.6)],
            title="Where the tokens go",
            caption=("One dot per world, the bar the median. The tokens each connection carried, net, over "
                     "50 games, split into the part that runs downhill and the part that goes round loops "
                     "(the note *Flows that go round*). Top: how much more of the flow goes round than if "
                     "each connection's net flow had a random direction; above zero it goes round, below "
                     "it runs downhill. Gifts are stakes that did not take their target, takeovers stakes "
                     "that did. Bottom: how much of each part comes back on the same connections in the "
                     "next 50 games."),
            recipe=recipe(LAB_RUNS + " " + PILOT_RUNS, "the recorded games", [
                "Add up, over each stretch of 50 games, the tokens staked from each node on each "
                "neighbour's node, one way minus the other, separately for gifts and takeovers "
                "(`research/oee/checks/circulation.py`).",
                "Find the pressure that explains as much of that net flow as possible (least squares, "
                "by conjugate gradients on the network's Laplacian); the circulation is the rest. Its "
                "share is its sum of squares over the flow's.",
                "Chance level: the same with each connection's direction thrown at random, four times, "
                "averaged.",
                "Persistence: over the connections two consecutive stretches both used, the correlation "
                "of their downhill parts, and of their circulations.",
                "Each world's number is the median over its stretches."]))
        for kind in ("all", "gift", "take"):
            for key in ("circulation_share", "flip_share", "random_share", "persistence_flow",
                        "persistence_gradient", "persistence_circulation"):
                ch.number(f"flow_{kind}_{key}", {c: _summary(v) for c, v in flows(kind, key).items()})
            ch.number(f"flow_{kind}_excess", {c: _summary(v) for c, v in excess(kind).items()})

    # --- Who helps whom ------------------------------------------------------
    dec = _by_run("decisive_20.json")
    if all(w in dec for w in B2 + B1):
        def helped(f):
            g = {"B2": [f(dec[w]) for w in B2], "B1": [f(dec[w]) for w in B1]}
            if allies and all("decisive" in p for p in allies):
                g["another-game"] = [f(p["decisive"]) for p in allies]
            if controls and all("decisive" in p for p in controls):
                g["no alliances"] = [f(p["decisive"]) for p in controls]
            return g

        # Returned help, pooled over a run's stretches: the pairs that returned it against
        # the pairs chance would, shown where chance expects at least one.
        def pooled(d):
            ws = [w for w in d.get("windows", []) if w["returned"] is not None]
            seen = sum(w["returned"] * w["pairs"] for w in ws)
            expected = sum(w["returned_chance"] * w["pairs"] for w in ws)
            return seen, expected

        enough = lambda d: (d.get("helping_pairs_per_window") or 0) >= ENOUGH_PAIRS
        saved = helped(lambda d: d["nodes_saved_per_100_agents_a_game"])
        back = helped(lambda d: pooled(d)[0] / pooled(d)[1] if pooled(d)[1] >= 1 else None)
        kin = helped(lambda d: d["help_to_clones"] / d["help_to_clones_chance"]
                     if d.get("helping_stakes", 0) >= ENOUGH_PAIRS and d["help_to_clones_chance"] else None)
        # Help that closes no loop (both shares zero, up to rounding) can neither go round nor
        # fail to.
        rounds = helped(lambda d: d["help_round"] - d["help_round_flip"]
                        if enough(d) and max(abs(d["help_round"] or 0), abs(d["help_round_flip"] or 0)) > 1e-9
                        else None)
        ch.grid("help", [
            _panel("Nodes saved by others' stakes", "per 100 agents a game (logarithmic)",
                   {c: [max(x, 1e-3) for x in v] for c, v in saved.items()}, None, log=True, min=1e-3, max=100),
            _panel("Help returned", "returned pairs ÷ chance (logarithmic)", back, 1, log=True, min=0.3, max=30),
            _panel("Help that goes round", "circulation share − its chance level", rounds, 0, min=-0.3, max=0.3),
            _panel("Help to clones", "share ÷ the clones' share (logarithmic)", kin, 1, log=True, min=0.3, max=10)],
            title="Who helps whom",
            caption=("One dot per world, the bar the median. A stake on a neighbour's node helps if, without "
                     "it, the neighbour would have been less likely to keep its node (the note *Help*). Top "
                     "left: the nodes so saved, counted as the sum of those effects, per 100 agents and "
                     "game; the pilots without alliances, in which no stake can help, are drawn at 0.001. "
                     "Top right: the pairs in which help went both ways within a stretch of 50 games, "
                     "summed over the stretches, divided by the number expected from how long each pair "
                     "were neighbours; worlds in which chance expects fewer than one have no dot. Bottom: "
                     "whether the net help between agents goes round loops more than chance (only where "
                     f"stretches have at least {ENOUGH_PAIRS} helping pairs and their help closes loops), and "
                     "whether help goes to agents carrying the helper's brain more than such agents are "
                     f"among its neighbours (only where at least {ENOUGH_PAIRS} stakes helped)."),
            recipe=recipe(LAB_RUNS + " " + PILOT_RUNS, "the recorded games", [
                "For every stake u → v on another agent's node, the chance that v keeps its node with all "
                "stakes and without u's, from the world's rule for deciding a node counted exactly: the "
                "largest stake, the owner's counting 1.5 times in B2 and the pilots, ties drawn, and the "
                "revolution's crowd (B1, B2), or the sides of allies (the pilots with alliances) "
                "(`research/oee/checks/decisive.py`). The difference is the stake's effect; above zero "
                "it is help.",
                "Saved: the sum of the positive effects over the games, per 100 agents and game.",
                "Returned: in each stretch of 50 games, the pairs with help from u to v that also had "
                "help from v to u; chance, for each such pair, 1 − (1 − q)^n for the n games in which v "
                "had u for a neighbour and q the help per neighbour and game of the stretch; both summed "
                "over the stretches.",
                "Round: the net help between two agents in a stretch, split as the token flow above, "
                "minus the same for random directions.",
                "Clones: the share of the help that went to an agent with the helper's brain id, divided "
                "by the share of such agents among the helper's neighbours, both weighted by the help."]))
        for name, groups in (("saved", saved), ("returned_over_chance", back), ("round_excess", rounds),
                             ("clones_over_chance", kin)):
            ch.number(f"help_{name}", {c: _summary(v) for c, v in groups.items()})
        ch.number("help_returned_pairs", {c: v for c, v in helped(lambda d: {
            "seen": pooled(d)[0], "chance": pooled(d)[1]}).items()})
        ch.number("help_cost", {c: _summary(v) for c, v in
                                helped(lambda d: d["nodes_cost_per_100_agents_a_game"]).items()})
        ch.number("help_stakes", {c: _summary(v) for c, v in
                                  helped(lambda d: d["helping_stakes"] / max(1, d["stakes_on_others"])).items()})

    # --- The alliance pilots, what their labels say ---------------------------
    if allies:
        ch.number("alliance_labels", {p["pilot"]: p["another-game"] for p in allies})
        ch.number("alliance_partnerships", {p["pilot"]: {"kept_in": p["structures"]["kept_in_partnership"],
                                                         "kept_otherwise": p["structures"]["kept_otherwise"]}
                                            for p in allies + controls})
