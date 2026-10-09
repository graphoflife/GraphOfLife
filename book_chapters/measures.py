# -*- coding: utf-8 -*-
"""
Part III, the first half: Chapters 19 to 23 — entropy, gains and losses, the
flow of tokens, how agents have children, and power laws.

Most of what these chapters show is read from the statistics every run
records; what needs single agents comes from book_figures.sample_pass, which
reads the frames every 25 iterations and keeps what it found beside the runs.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

import book_figures as F
from book_figures import (BASELINE_RUNS, FRAMES, STATS_FILE, band_series, chapter, describe, dots,
                          line, mean_over, recipe, series, survivors)
from book_chapters.common import (BAND_STEPS, BAND_WORDS, BLUE, CYAN, GREEN, GREY, ORANGE, RED,
                                  VIOLET, YELLOW, baseline)

SURVIVORS = BASELINE_RUNS.replace("The 30", "The 26 surviving ones of the 30")
SAMPLE = ("each run's frames, read every 25 iterations by `book_figures.sample_pass` (frame "
          "`2·t` after the reproduction phase of iteration *t*, frame `2·t + 1` after its game; "
          "`gol_store.read_frame(run, index)`)")

# The columns of sample_pass's tables.
FR = dict(t=0, agents=1, genotypes=2, entropy=3, new_repro=4, new_game=5, births=6, home=7, others=8,
          mutual=9)
AG = dict(t=0, tokens=1, degree=2, curvature=3, change=4, kept=5, home=6, candidates=7, staked=8, age=9)
PA = dict(t=0, tokens=1, degree=2, invested=3, links=4, handed=5)

TOKEN_CLASSES = [(1, 1, "1"), (2, 2, "2"), (3, 4, "3–4"), (5, 9, "5–9"), (10, 19, "10–19"),
                 (20, 49, "20–49"), (50, 99, "50–99"), (100, 10 ** 9, "100+")]
DEGREE_CLASSES = [(1, 1, "1"), (2, 2, "2"), (3, 4, "3–4"), (5, 9, "5–9"), (10, 49, "10–49"),
                  (50, 10 ** 9, "50+")]


def sampled(which: str, settled: bool = True) -> List[Tuple[Any, np.ndarray]]:
    """(run, table) for every surviving baseline run, the frame rows from SETTLED on if asked."""
    out = []
    for s in survivors(baseline()):
        table = F.sample_pass(s.run_id)[which]
        if which == "frames" and settled:
            table = table[table[:, 0] >= F.SETTLED]
        out.append((s, table))
    return out


def pooled(which: str) -> np.ndarray:
    return np.concatenate([table for _, table in sampled(which)])


def in_class(values: np.ndarray, classes) -> List[np.ndarray]:
    return [(values >= lo) & (values <= hi) for lo, hi, _ in classes]


def bars(categories: Sequence[str], values: Sequence[float], colour: int, label: str = None,
         slot: Tuple[float, float] = (-0.38, 0.38), **extra: Any) -> Dict[str, Any]:
    """One bar per category, filling `slot` of the space each category has."""
    k = np.arange(len(categories))
    return {"label": label, "kind": "bars", "x0": (k + slot[0]).tolist(), "x1": (k + slot[1]).tolist(),
            "y": [float(v) for v in values], "colour": colour, **extra}


def stacked(categories: Sequence[str], parts: Sequence[Tuple[str, Sequence[float], int]],
            width: float = 0.36) -> List[Dict[str, Any]]:
    """Bars stacked one on another: parts are (label, a value per category, colour)."""
    below = np.zeros(len(categories))
    out = []
    for label, values, colour in parts:
        top = below + np.asarray(values, float)
        out.append(bars(categories, top, colour, label, (-width, width), y0=below.tolist(), alpha=0.9))
        below = top
    return out


def settled_rows(run_id: str, phase: int = 2) -> List[Dict[str, Any]]:
    return [r for r in F.rows(run_id) if r["phase"] == phase and F.SETTLED <= r["iteration"]]


# ---------------------------------------------------------------------------
# Chapter 19 · How even is a world?
# ---------------------------------------------------------------------------

@chapter
def entropy(ch: F.Chapter) -> None:
    specs = baseline()
    alive = survivors(specs)

    def genotype_evenness(run_id: str) -> Tuple[np.ndarray, np.ndarray]:
        fr = F.sample_pass(run_id)["frames"]
        n = fr[:, FR["agents"]]
        ok = n > 1
        return fr[ok, 0], fr[ok, FR["entropy"]] / np.log2(n[ok])

    panels = []
    for title, data, colour in (
            ("Tokens", [series(s.run_id, "tokenEvenness") for s in specs], YELLOW),
            ("Connections per agent", [series(s.run_id, "degreeEvenness") for s in specs], ORANGE),
            ("Genotypes", [genotype_evenness(s.run_id) for s in specs], VIOLET)):
        panels.append(dict(title=title, x={"label": "iteration"}, y={"label": "evenness", "min": 0, "max": 1},
                           series=[band_series(data, None, colour)], legend=False))
    ch.grid(
        "evenness", panels, columns=3, title="How even a world is, in three respects",
        caption="Evenness after every game of the 30 baseline worlds, from 0 (everything in one hand) "
                "to 1 (as even as it can be; [Entropy and evenness](../notes/entropy-and-evenness.md)). "
                "Left: of the tokens over the agents (`tokenEvenness`). Middle: of the agents over the "
                "numbers of connections that occur (`degreeEvenness`). Right: of the agents over the "
                "genotypes they carry — the entropy of the genotypes' shares divided by log₂ of the "
                f"number of agents, measured every 25 iterations. In each, {BAND_WORDS[0].lower()}"
                f"{BAND_WORDS[1:]}.",
        recipe=recipe(BASELINE_RUNS, STATS_FILE + "; for genotypes, " + SAMPLE,
                      ["Take `tokenEvenness` and `degreeEvenness` of the rows with `phase` = 2.",
                       "For genotypes: in the frame after the game of every 25th iteration, count the "
                       "agents per genotype (`brain_ids`), turn the counts into shares pᵢ, and compute "
                       "H = −Σ pᵢ log₂ pᵢ; divide by log₂ of the number of agents.",
                       *BAND_STEPS]))
    for stat in ("tokenEvenness", "degreeEvenness", "tokenEntropy", "degreeEntropy", "gini"):
        ch.number(f"{stat}_settled", describe([mean_over(s.run_id, stat, 500, 2999) for s in alive]))
    ch.number("tokenEvenness_first", F.A.bands([series(s.run_id, "tokenEvenness") for s in specs])["y"][0])

    # Effective numbers: how many equal holders, or equal genotypes, the world is worth.
    holders, distinct, effective = [], [], []
    for s, fr in sampled("frames"):
        rows_ = settled_rows(s.run_id)
        holders.append(float(np.mean([2 ** r["tokenEntropy"] / r["nodes"] for r in rows_])))
        n = fr[:, FR["agents"]]
        distinct.append(float(np.mean(fr[:, FR["genotypes"]] / n)))
        effective.append(float(np.mean(2 ** fr[:, FR["entropy"]] / n)))
    cats = ["effective holders of the tokens", "distinct genotypes", "effective genotypes"]
    ch.figure(
        "effective", title="Effective numbers, as a share of the agents",
        x={"label": "", "categories": cats}, y={"label": "per agent (mean over iterations 500–2,999)",
                                                "min": 0, "max": 1},
        series=dots(cats, [holders, distinct, effective], [YELLOW, VIOLET, VIOLET]), legend=False,
        caption="One dot per world that lived to the end; the bar is the median. Left: 2^H of the tokens "
                "— the number of agents that, holding equal shares, would make the tokens as even as "
                "they are — divided by the number of agents. Middle: the number of different genotypes "
                "among the living, per agent. Right: 2^H of the genotypes — how many equally common "
                "genotypes would be as diverse as the ones there are — per agent. The dots are spread "
                "sideways only so that they do not hide each other.",
        recipe=recipe(SURVIVORS, STATS_FILE + "; for genotypes, " + SAMPLE,
                      ["Tokens: for each row with `phase` = 2 and 500 ≤ `iteration` ≤ 2,999, compute "
                       "2^`tokenEntropy` / `nodes`, and average over the rows.",
                       "Genotypes: in the frames after the game of every 25th iteration from 500 on, the "
                       "number of distinct `brain_ids` per agent, and 2^H per agent with H the entropy of "
                       "the genotypes' shares; average over the frames.",
                       "One dot per run, a bar at the median."]))
    ch.number("effective", {"holders": describe(holders), "distinct": describe(distinct),
                            "effective_genotypes": describe(effective)})

    # Gini against evenness: two measures of one thing.
    g, e = [], []
    for s in alive:
        for r in F.rows(s.run_id):
            if r["phase"] == 2 and r["iteration"] >= 500 and r["iteration"] % 25 == 0:
                g.append(r["gini"])
                e.append(r["tokenEvenness"])
    g, e = np.array(g), np.array(e)
    ch.figure(
        "gini-evenness", title="Two measures of the same inequality",
        x={"label": "Gini coefficient"}, y={"label": "token evenness"},
        series=[{"label": None, "x": g.tolist(), "y": e.tolist(), "points": True, "size": 3,
                 "colour": YELLOW, "alpha": 0.45}], legend=False,
        caption=f"One dot per world and look: every 25th iteration from 500 on of the 26 worlds that "
                f"lived to the end ({len(g):,} dots). Across the dots the two measures move against each "
                f"other — the correlation is {np.corrcoef(g, e)[0, 1]:.2f} — but not along one curve: the "
                "same Gini comes with different evenness, because the two weigh the poor and the rich "
                "differently.",
        recipe=recipe(SURVIVORS, STATS_FILE,
                      ["Take the rows with `phase` = 2 and `iteration` a multiple of 25 from 500 on.",
                       "Plot `tokenEvenness` against `gini`, one dot per row."]))
    ch.number("gini_evenness_corr", float(np.corrcoef(g, e)[0, 1]))


# ---------------------------------------------------------------------------
# Chapter 20 · Gains and losses
# ---------------------------------------------------------------------------

@chapter
def gains(ch: F.Chapter) -> None:
    specs = baseline()
    alive = survivors(specs)

    def shares(rows_):
        start = sum(r["nodes_before"] for r in rows_)
        starved = sum(r["starved"] or 0 for r in rows_) / start
        cut = sum(r["orphaned"] or 0 for r in rows_) / start
        gained = sum(r["gainers"] for r in rows_) / start
        lost = sum(r["losers"] for r in rows_) / start
        same = sum(r["nodes"] - r["gainers"] - r["losers"] for r in rows_) / start
        return [starved, cut, lost, same, gained]

    first = shares([r for s in specs for r in F.rows(s.run_id) if r["phase"] == 2 and r["iteration"] == 0])
    late = shares([r for s in alive for r in settled_rows(s.run_id)])
    cats = ["the first game (30 worlds)", "games from iteration 500 on (26 worlds)"]
    names = ["starved", "cut off", "lost tokens", "kept exactly what it had", "gained tokens"]
    ch.figure(
        "outcome", title="What one game does to the agents",
        x={"label": "", "categories": cats}, y={"label": "share of the agents at the start of the game",
                                                "min": 0, "max": 1},
        series=stacked(cats, [(nm, [first[i], late[i]], c) for i, (nm, c) in
                              enumerate(zip(names, [RED, ORANGE, VIOLET, GREY, GREEN]))]),
        caption="Every agent alive at the start of a game ends it in one of five ways: it starved (no one "
                "staked on its node), it was cut off from the largest piece of the network, or it "
                "survived holding fewer tokens, exactly as many, or more than before. Left: the game of "
                "iteration 0 in all 30 worlds. Right: every game from iteration 500 on in the 26 worlds "
                "that lived to the end, pooled.",
        recipe=recipe(BASELINE_RUNS, STATS_FILE,
                      ["Take the rows with `phase` = 2 (iteration 0, or 500 to 2,999).",
                       "Sum over them: `nodes_before` (agents at the start), `starved`, `orphaned`, `gainers`, "
                       "`losers`, and `nodes` − `gainers` − `losers` (survivors whose tokens did not change).",
                       "Divide each sum by the sum of `nodes_before` and stack the shares."]))
    ch.number("outcome", {"first": dict(zip(names, first)), "settled": dict(zip(names, late))})

    ag = pooled("agents")
    tokens, change = ag[:, AG["tokens"]], ag[:, AG["change"]]
    survived = ~np.isnan(change)
    labels = [c[2] for c in TOKEN_CLASSES]
    died, lost, same, gained, rel = [], [], [], [], []
    for m in in_class(tokens, TOKEN_CLASSES):
        died.append(np.mean(~survived[m]))
        sm = m & survived
        lost.append(np.sum(change[sm] < 0) / m.sum())
        same.append(np.sum(change[sm] == 0) / m.sum())
        gained.append(np.sum(change[sm] > 0) / m.sum())
        rel.append(float(np.mean(change[sm] / tokens[sm])))
    ch.figure(
        "by-tokens", title="What a game does, by how rich an agent was",
        x={"label": "tokens at the start of the game", "categories": labels},
        y={"label": "share of the agents", "min": 0, "max": 1},
        series=stacked(labels, [("died in the game", died, RED), ("lost tokens", lost, VIOLET),
                                ("kept exactly what it had", same, GREY), ("gained tokens", gained, GREEN)]),
        caption="Agents alive at the start of a game, sorted by the tokens they held then; for each class, "
                "the share that died in the game (starved or cut off), and of the rest the shares that "
                "lost tokens, kept exactly as many, or gained. From the games of every 100th iteration "
                f"from 500 on, in the 26 worlds that lived to the end ({len(tokens):,} agents in all).",
        recipe=recipe(SURVIVORS, SAMPLE,
                      ["For every 100th iteration t from 500 on, take each agent in frame `2·t` (the start "
                       "of the game) with its `tokens`.",
                       "Find it in frame `2·t + 1`: if it is not there, it died in the game; if it is, its "
                       "`delta` there is its change over the game.",
                       "Sort the agents by their tokens at the start into the classes shown, and count the "
                       "four outcomes in each class."]))
    ch.number("by_tokens", {lab: {"died": d, "lost": l, "same": s_, "gained": g_, "mean_relative_change": r}
                            for lab, d, l, s_, g_, r in zip(labels, died, lost, same, gained, rel)})
    # Which of the rich die: the few-connected ones, by far.
    rich, degree = tokens >= 100, ag[:, AG["degree"]]
    ch.number("rich_deaths", {lab: {"agents": int((rich & m).sum()), "died": float(np.mean(~survived[rich & m]))}
                              for lab, m in (("1–2", degree <= 2), ("3–9", (degree >= 3) & (degree <= 9)),
                                             ("10+", degree >= 10))})

    # Downhill: the change against the token curvature, for all agents and for agents of one wealth.
    curvature = ag[:, AG["curvature"]]
    cclasses = [(-1e12, -50, "below −50"), (-50, -10, "−50 to −10"), (-10, -3, "−10 to −3"),
                (-3, 3, "−3 to 3"), (3, 10, "3 to 10"), (10, 50, "10 to 50"), (50, 1e12, "above 50")]
    clabels = [c[2] for c in cclasses]

    def downhill(mask):
        out = []
        for lo, hi, _ in cclasses:
            m = mask & survived & (curvature > lo) & (curvature <= hi)
            out.append(float(np.mean(change[m] / tokens[m])) if m.sum() >= 1000 else None)
        return out
    everyone = downhill(np.ones(len(tokens), bool))
    middle = downhill((tokens >= 3) & (tokens <= 9))
    ch.figure(
        "downhill", title="Tokens run downhill",
        x={"label": "token curvature at the start of the game: neighbours' tokens minus own, summed",
           "categories": clabels},
        y={"label": "mean change over the game, per token held"},
        series=[bars(clabels, [v if v is not None else 0 for v in everyone], YELLOW, "all agents", (-0.38, 0)),
                bars(clabels, [v if v is not None else 0 for v in middle], CYAN, "agents holding 3 to 9 tokens",
                     (0, 0.38))],
        guides=[{"axis": "y", "at": 0}],
        caption="The token curvature of an agent is the sum, over its neighbours, of how many more tokens "
                "the neighbour holds than it does ([Token curvature](../notes/token-curvature.md)): "
                "negative for an agent richer than its neighbourhood, positive for one poorer. For each "
                "class of curvature, the mean change of an agent's tokens over the game divided by the "
                "tokens it held at the start, over the agents that survived the game. Yellow: all agents; "
                "cyan: only agents that held 3 to 9 tokens, so that wealth itself cannot explain the "
                "pattern. A class with fewer than 1,000 agents is left out (drawn at 0).",
        recipe=recipe(SURVIVORS, SAMPLE,
                      ["For every 100th iteration t from 500 on, read frame `2·t`: for every agent u, "
                       "κ(u) = Σ over its neighbours v of (τ(v) − τ(u)).",
                       "Find each agent in frame `2·t + 1` and take its `delta` (agents not there died and "
                       "are left out).",
                       "Sort by κ into the classes shown and average `delta` / τ(u) in each."]))
    ch.number("downhill", {"all": dict(zip(clabels, everyone)), "tokens_3_9": dict(zip(clabels, middle))})

    # The share-out of the dead: a lottery.
    lines_ = [band_series([series(s.run_id, "redistributed", 2) for s in specs], "after the game", GREEN),
              band_series([series(s.run_id, "redistributed", 1) for s in specs], "after the reproduction phase",
                          CYAN)]
    ch.figure(
        "lottery", title="Tokens shared out among the survivors",
        x={"label": "iteration"}, y={"label": "tokens per cleanup (logarithmic)", "log": True, "min": 1},
        series=lines_,
        caption="How many tokens the cleanup dealt out at random among the survivors — the tokens of the "
                "agents that were cut off — after each game (green) and after each reproduction phase "
                f"(cyan), on a logarithmic axis. For each, {BAND_WORDS[0].lower()}{BAND_WORDS[1:]}.",
        recipe=recipe(BASELINE_RUNS, STATS_FILE,
                      ["Take `redistributed` of the rows with `phase` = 2 and of those with `phase` = 1.",
                       *BAND_STEPS]))
    lot = []
    for s in alive:
        rows_ = settled_rows(s.run_id)
        lot.append(float(np.mean([r["redistributed"] / r["nodes"] for r in rows_])))
    ch.number("lottery_per_agent", describe(lot))
    ch.number("redistributed_settled", {
        "game": describe([mean_over(s.run_id, "redistributed", 500, 2999, 2) for s in alive]),
        "repro": describe([mean_over(s.run_id, "redistributed", 500, 2999, 1) for s in alive])})
    for stat in ("maxTokenAdded", "maxTokenLost", "gainers", "losers"):
        ch.number(f"{stat}_settled", {ph: describe([mean_over(s.run_id, stat, 500, 2999, ph) for s in alive])
                                      for ph in (1, 2)})


# ---------------------------------------------------------------------------
# Chapter 21 · Where the tokens flow
# ---------------------------------------------------------------------------

def even_split(run_id: str, iterations) -> Tuple[np.ndarray, np.ndarray]:
    """
    For every agent at the start of the games of `iterations`: what it would
    receive, minus what it holds, if every agent split its tokens evenly over
    itself and its neighbours — and what it really received, minus what it held.
    """
    predicted, real = [], []
    for t in iterations:
        repro = F.store.read_frame(run_id, 2 * t)
        game = F.store.read_frame(run_id, 2 * t + 1)
        tokens = dict(zip(repro["ids"], repro["tokens"]))
        neighbours = defaultdict(list)
        for a, b in repro["edges"]:
            neighbours[a].append(b)
            neighbours[b].append(a)
        expect = defaultdict(float)
        for u in repro["ids"]:
            share = tokens[u] / (len(neighbours[u]) + 1)
            expect[u] += share
            for v in neighbours[u]:
                expect[v] += share
        received = defaultdict(int)
        for r in (game.get("decisions") or {}).get("allocations") or []:
            for target, amount in zip(r["targets"], r["alloc"]):
                received[target] += amount
        for u in repro["ids"]:
            predicted.append(expect[u] - tokens[u])
            real.append(received[u] - tokens[u])
    return np.array(predicted), np.array(real)


@chapter
def flow(ch: F.Chapter) -> None:
    specs = baseline()
    alive = survivors(specs)

    def budget(rows_):
        total = sum(r["tokens"] for r in rows_)
        moved = sum(r["totalFlow"] for r in rows_)
        net = sum(r["totalFlow"] * r["netFlowShare"] for r in rows_)
        return [(total - moved) / total, (moved - net) / total, net / total]

    first = budget([r for s in specs for r in F.rows(s.run_id) if r["phase"] == 2 and r["iteration"] == 0])
    late = budget([r for s in alive for r in settled_rows(s.run_id) if r.get("netFlowShare") is not None])
    cats = ["the first game (30 worlds)", "games from iteration 500 on (26 worlds)"]
    names = ["staked on the agent's own node", "staked on a neighbour, and matched by a stake back",
             "staked on a neighbour, one way"]
    ch.figure(
        "budget", title="Where the tokens of a game go",
        x={"label": "", "categories": cats}, y={"label": "share of all tokens", "min": 0, "max": 1},
        series=stacked(cats, [(nm, [first[i], late[i]], c) for i, (nm, c) in
                              enumerate(zip(names, [YELLOW, CYAN, RED]))]),
        caption="Every token is staked in every game. Yellow: the share staked by agents on their own node. "
                "Cyan: staked on a neighbour's node, but cancelled by what that neighbour staked back — if A "
                "puts 5 on B's node and B puts 3 on A's, 3 tokens each way cancel. Red: what is left after "
                "cancelling, flow with a direction ([Token flow](../notes/token-flow.md)). Left: the game of "
                "iteration 0, pooled over the 30 worlds; right: every game from iteration 500 on of the 26 "
                "worlds that lived to the end, pooled. Exactly 10,000 tokens are staked in each game.",
        recipe=recipe(BASELINE_RUNS, STATS_FILE,
                      ["Take the rows with `phase` = 2: `tokens` (all staked), `totalFlow` (staked on "
                       "others) and `netFlowShare` (the share of that flow left after cancelling).",
                       "Sum over the rows: at home = Σ `tokens` − Σ `totalFlow`; cancelled = Σ `totalFlow` × "
                       "(1 − `netFlowShare`); one way = Σ `totalFlow` × `netFlowShare`.",
                       "Divide each by Σ `tokens`."]))
    ch.number("budget", {"first": dict(zip(names, first)), "settled": dict(zip(names, late))})

    def returned(run_id):
        fr = F.sample_pass(run_id)["frames"]
        ok = ~np.isnan(fr[:, FR["mutual"]])
        return fr[ok, 0], fr[ok, FR["mutual"]]
    ch.figure(
        "reciprocity", title="Stakes that are returned",
        x={"label": "iteration"}, y={"label": "share of stakes returned", "min": 0, "max": 1},
        series=[band_series([returned(s.run_id) for s in specs], None, CYAN)], legend=False,
        caption="Of all pairs (A, B) in a game where agent A staked at least one token on neighbour B's node, "
                "the share in which B also staked at least one token on A's — every 25 iterations, in the 30 "
                f"baseline worlds. {BAND_WORDS}.",
        recipe=recipe(BASELINE_RUNS, SAMPLE,
                      ["In the frame after the game of every 25th iteration, read `decisions.allocations`: "
                       "for every agent A and every target B ≠ A with a stake > 0, note the pair (A, B).",
                       "Count the pairs whose reverse (B, A) is also there, divided by all pairs.",
                       *BAND_STEPS[1:]]))
    mutual_settled = [float(np.mean(fr[:, FR["mutual"]])) for _, fr in sampled("frames")]
    ch.number("reciprocity_settled", describe(mutual_settled))

    ag = pooled("agents")
    degree = ag[:, AG["degree"]]
    labels = [c[2] for c in DEGREE_CLASSES]
    staked_share, home_share, even = [], [], []
    for m in in_class(degree, DEGREE_CLASSES):
        ok = m & (ag[:, AG["candidates"]] > 0)
        staked_share.append(float(np.mean(ag[ok, AG["staked"]] / ag[ok, AG["candidates"]])))
        home_share.append(float(np.nanmean(ag[ok, AG["home"]])))
        even.append(float(np.mean(1 / (degree[ok] + 1))))
    k = np.arange(len(labels))
    ch.figure(
        "breadth", title="How widely agents stake",
        x={"label": "connections of the agent", "categories": labels}, y={"label": "share", "min": 0, "max": 1},
        series=[bars(labels, staked_share, BLUE, "share of its candidates it staked on", (-0.38, 0)),
                bars(labels, home_share, YELLOW, "share of its tokens it staked at home", (0, 0.38)),
                {"label": "1 / (connections + 1): its home share if it split evenly", "x": (k + 0.19).tolist(),
                 "y": even, "points": True, "size": 9, "colour": "#eef4fa"}],
        caption="Agents at the start of a game, sorted by their number of connections. Blue: the share of "
                "their candidates (themselves and their neighbours) on which they staked at least one token. "
                "Yellow: the share of their tokens they staked on their own node. White dots: the home share "
                "an agent would have if it split its tokens evenly over all its candidates, 1/(d + 1), "
                "averaged over the class. From the games of every 100th iteration from 500 on, in the 26 "
                "worlds that lived to the end.",
        recipe=recipe(SURVIVORS, SAMPLE,
                      ["For every 100th iteration t from 500 on, take each agent's degree d in frame `2·t`.",
                       "In frame `2·t + 1`, read its entry of `decisions.allocations`: the number of `targets` "
                       "(its candidates), how many of them got `alloc` > 0, and `alloc[0]` / `tokens` (its "
                       "home share; its first target is itself).",
                       "Average by class of d; the white dots are the mean of 1/(d + 1) in each class."]))
    ch.number("breadth", {lab: {"staked": a, "home": b, "even": c} for lab, a, b, c in
                          zip(labels, staked_share, home_share, even)})

    predicted, real = [], []
    for s in alive:
        p, r = even_split(s.run_id, range(500, 3000, 250))
        predicted.append(p)
        real.append(r)
    predicted, real = np.concatenate(predicted), np.concatenate(real)
    slope, intercept = np.polyfit(predicted, real, 1)
    r2 = float(np.corrcoef(predicted, real)[0, 1] ** 2)
    edges = np.arange(-20, 22, 2)
    xs, mean, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (predicted >= a) & (predicted < b)
        if m.sum() >= 200:
            xs.append((a + b) / 2)
            mean.append(float(real[m].mean()))
            lo.append(float(np.quantile(real[m], 0.25)))
            hi.append(float(np.quantile(real[m], 0.75)))
    ch.figure(
        "random-walk", title="The game as a random walk of tokens",
        x={"label": "what the agent would gain or lose if everyone split their stakes evenly", "min": -20, "max": 20},
        y={"label": "what it really gained or lost in the stakes", "min": -25, "max": 25},
        series=[{"label": "mean (band: middle half)", "x": xs, "y": mean, "lo": lo, "hi": hi, "colour": BLUE,
                 "width": 2},
                line([-20, 20], [-20, 20], "the same", GREY, width=1, dash=[4, 4])],
        caption="For every agent at the start of a game: x is the balance it would end the stakes with — the "
                "tokens staked on its node minus the tokens it held — if every agent split its tokens evenly "
                "over itself and its neighbours; y is the balance it really ended them with. Agents are "
                "binned by x in steps of 2; the line is the mean of y in each bin and the band its middle "
                "half. The dashed line is y = x. Over all "
                f"{len(predicted):,} agents (the games of every 250th iteration from 500 on, 26 worlds), "
                f"the straight line fitted to the points has slope {slope:.2f}, and the even split accounts "
                f"for {100 * r2:.0f}% of the variance (R², [Fitting a straight line](../notes/least-squares.md)).",
        recipe=recipe(SURVIVORS, SAMPLE.replace("every 25 iterations by `book_figures.sample_pass` ", ""),
                      ["For every 250th iteration t from 500 on, read frame `2·t` (the start of the game): each "
                       "agent's tokens τ and neighbours.",
                       "Even split: every agent u gives τ(u)/(d(u) + 1) to itself and to each neighbour; x(u) is "
                       "what u receives minus τ(u).",
                       "Real: in frame `2·t + 1`, add up every `alloc` whose target is u; y(u) is that minus τ(u).",
                       "Bin by x and average y; fit y = a·x + b by least squares over all agents."]))
    ch.number("random_walk", {"slope": float(slope), "intercept": float(intercept), "r2": r2,
                              "agents": int(len(predicted))})

    kept_nodes, kept = np.zeros(6), np.zeros(6)
    for _, fr in sampled("frames"):
        pairs = fr[:, 10:22].reshape(len(fr), 6, 2).sum(axis=0)
        kept_nodes += pairs[:, 0]
        kept += pairs[:, 1]
    share_kept = kept / kept_nodes
    ch.figure(
        "kept", title="Who keeps their node, by connections",
        x={"label": "connections of the node", "categories": labels},
        y={"label": "share of nodes won by their own agent", "min": 0, "max": 1},
        series=[bars(labels, share_kept, GREEN)], legend=False,
        caption="Every node on which anyone staked in a game, sorted by its number of connections at the start "
                "of the game; the share that was won by the agent living on it — whether as the largest "
                "staker or through a coalition. From the games of every 25th iteration from 500 on, in the "
                "26 worlds that lived to the end.",
        recipe=recipe(SURVIVORS, SAMPLE,
                      ["For every 25th iteration t from 500 on, count each node's connections in frame `2·t`.",
                       "In frame `2·t + 1`, read `decisions.winners`: for each node, whether `winner` = `node`.",
                       "Pool over runs and frames, and divide kept by all, per class."]))
    ch.number("kept_by_degree", {lab: {"nodes": float(n), "kept": float(k_)} for lab, n, k_ in
                                 zip(labels, kept_nodes, share_kept)})

    panels = []
    for title, stats_ in (("As staked", [("cyclingShare", "share found going round in loops", BLUE),
                                         ("flowImbalance", "share that must go one way (ceiling: 1 − this)", RED)]),
                          ("After cancelling", [("netFlowShare", "share of the flow left after cancelling", CYAN),
                                                ("netCyclingShare", "share of that found going round", GREEN)])):
        panels.append(dict(title=title, x={"label": "iteration"}, y={"label": "share", "min": 0, "max": 1},
                           series=[band_series([series(s.run_id, st) for s in specs], lab, c)
                                   for st, lab, c in stats_]))
        for st, _, _ in stats_:
            ch.number(f"{st}_settled", describe([mean_over(s.run_id, st, 500, 2999) for s in alive]))
    ch.grid(
        "circulation", panels, columns=2, title="Do tokens go round?",
        caption="Left, on the flow as staked: the share of all tokens staked on neighbours that a greedy search "
                "could place in closed loops (blue; [Lightning](../notes/lightning.md)), and the share that "
                "conservation forbids from ever going round, because some agents receive more than they send "
                "(red) — so at most 1 − red can circulate. Right, after cancelling every stake against the one "
                "coming back: the share of the flow left (cyan), and of that, the share found in loops "
                f"(green). Every 25 iterations, 30 worlds; {BAND_WORDS[0].lower()}{BAND_WORDS[1:]}.",
        recipe=recipe(BASELINE_RUNS, STATS_FILE,
                      ["Take `cyclingShare`, `flowImbalance`, `netFlowShare` and `netCyclingShare` of the rows "
                       "with `phase` = 2 (measured every 25 iterations).", *BAND_STEPS[1:]]))
    for stat in ("totalFlow", "meanEdgeFlow", "maxEdgeFlow", "prunedEdges", "lightningLongest",
                 "netLightningLongest", "lightningScore", "netLightningScore"):
        ch.number(f"{stat}_settled", describe([mean_over(s.run_id, stat, 500, 2999) for s in alive]))


# ---------------------------------------------------------------------------
# Chapter 22 · How agents have children
# ---------------------------------------------------------------------------

@chapter
def children(ch: F.Chapter) -> None:
    specs = baseline()
    alive = survivors(specs)
    pa = pooled("parents")
    settled_ = pa[pa[:, PA["t"]] >= F.SETTLED]
    founders = np.concatenate([F.sample_pass(s.run_id)["parents"] for s in specs])
    founders = founders[founders[:, PA["t"]] == 0]
    tokens, invested = settled_[:, PA["tokens"]], settled_[:, PA["invested"]]
    had = invested > 0
    labels = [c[2] for c in TOKEN_CLASSES]
    p_child, med, q25, q75, mean_share = [], [], [], [], []
    for m in in_class(tokens, TOKEN_CLASSES):
        p_child.append(float(had[m].mean()))
        share = invested[m & had] / tokens[m & had]
        med.append(float(np.median(share)) if share.size else None)
        q25.append(float(np.quantile(share, 0.25)) if share.size else None)
        q75.append(float(np.quantile(share, 0.75)) if share.size else None)
        mean_share.append(float(share.mean()) if share.size else None)
    k = np.arange(len(labels))
    whisk_x, whisk_y = [], []
    for i, (a, b) in enumerate(zip(q25, q75)):
        whisk_x += [i, i, None]
        whisk_y += [a, b, None]
    ch.grid(
        "by-tokens", [
            dict(title="Who has a child", x={"label": "tokens of the agent", "categories": labels},
                 y={"label": "share that had a child", "min": 0}, series=[bars(labels, p_child, GREEN)],
                 legend=False),
            dict(title="How much a parent gives", x={"label": "tokens of the parent", "categories": labels},
                 y={"label": "share of its tokens given to the child", "min": 0, "max": 1},
                 series=[{"label": "middle half of the parents", "x": whisk_x, "y": whisk_y, "colour": CYAN,
                          "width": 4, "alpha": 0.7},
                         {"label": "median", "x": k.tolist(), "y": med, "points": True, "size": 9,
                          "colour": "#eef4fa"}])],
        columns=2, title="Children, by the wealth of the parent",
        caption="Agents alive at the start of the reproduction phases of every 100th iteration from 500 on, in "
                "the 26 worlds that lived to the end, sorted by the tokens they held. Left: the share of them "
                "that had a child. Right: for those that did, the share of their tokens they gave it — the "
                "white dot is the median, the cyan bar runs from the 25th to the 75th percentile.",
        recipe=recipe(SURVIVORS, SAMPLE,
                      ["For every 100th iteration t from 500 on, take the agents of frame `2·t − 1` (after the "
                       "game before) with their `tokens`.",
                       "In frame `2·t`, `decisions.births` lists every parent (`agent`) with `tokens_before` and "
                       "`invested`, the tokens its child got.",
                       "Per class of tokens: the share of agents that are parents, and the quantiles of "
                       "`invested` / `tokens_before` over the parents."]))
    ch.number("by_tokens", {lab: {"p_child": p, "median_share": m_, "q25": a, "q75": b, "mean_share": ms}
                            for lab, p, m_, a, b, ms in zip(labels, p_child, med, q25, q75, mean_share)})
    ch.number("founders", {"p_child": float((founders[:, PA["invested"]] > 0).mean())})

    links = settled_[had, PA["links"]]
    handed = settled_[had, PA["handed"]]
    lcats = ["0", "1", "2", "3", "4", "5+"]
    lshare = [float(np.mean(links == v)) for v in range(5)] + [float(np.mean(links >= 5))]
    hcats = ["0", "1", "2", "3+"]
    hshare = [float(np.mean(handed == v)) for v in range(3)] + [float(np.mean(handed >= 3))]
    ch.grid(
        "links", [
            dict(title="Connections a child is born with", x={"label": "connections", "categories": lcats},
                 y={"label": "share of the children", "min": 0}, series=[bars(lcats, lshare, BLUE)], legend=False),
            dict(title="Connections handed over by the parent", x={"label": "connections", "categories": hcats},
                 y={"label": "share of the births", "min": 0}, series=[bars(hcats, hshare, VIOLET)],
                 legend=False)],
        columns=2, title="How a child is joined to the world",
        caption="Every birth in the reproduction phases of every 100th iteration from 500 on, in the 26 worlds "
                "that lived to the end. Left: how many of its parent's candidates — the parent itself and its "
                "neighbours — the child was joined to at birth. Right: how many of its own connections the "
                "parent handed to the child. A child with no connection at all is cut off at once.",
        recipe=recipe(SURVIVORS, SAMPLE,
                      ["In frame `2·t` of every 100th iteration t from 500 on, read `decisions.births`: the "
                       "length of `links` and of `handed_over` for every birth.",
                       "Count the births by these lengths and divide by all births."]))
    ch.number("links", {"share": dict(zip(lcats, lshare)), "mean": float(links.mean()),
                        "handed": dict(zip(hcats, hshare)), "handed_mean": float(handed.mean())})

    p_by_degree = [float(had[m].mean()) for m in in_class(settled_[:, PA["degree"]], DEGREE_CLASSES)]
    dl = [c[2] for c in DEGREE_CLASSES]
    ch.figure(
        "by-degree", title="Who has a child, by connections",
        x={"label": "connections of the agent", "categories": dl}, y={"label": "share that had a child", "min": 0},
        series=[bars(dl, p_by_degree, GREEN)], legend=False,
        caption="As the left panel of the figure above, with the agents sorted by their number of connections "
                "at the start of the reproduction phase instead of their tokens.",
        recipe=recipe(SURVIVORS, SAMPLE,
                      ["As for the figure above, counting each agent's connections in frame `2·t − 1`."]))
    ch.number("by_degree", dict(zip(dl, p_by_degree)))

    panels = []
    for title, stat, ylabel, colour, per_birth in (
            ("Share of its tokens a parent gives", "meanInvestedShare", "mean share", CYAN, False),
            ("Share of all tokens given to children", "reproTokenShare", "share of all tokens", YELLOW, False),
            ("Connections a child is born with", "meanChildLinks", "mean per child", BLUE, False),
            ("Connections handed over per birth", "handovers", "mean per birth", VIOLET, True)):
        data = []
        for s in specs:
            its, v = series(s.run_id, stat, 1)
            if per_birth:
                _, b = series(s.run_id, "births", 1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    v = np.where(b > 0, v / b, np.nan)
            data.append((its, v))
        panels.append(dict(title=title, x={"label": "iteration"}, y={"label": ylabel, "min": 0},
                           series=[band_series(data, None, colour)], legend=False))
    ch.grid(
        "over-time", panels, columns=2, title="Reproduction over a world's life",
        caption="After every reproduction phase of the 30 baseline worlds ([Reproduction statistics]"
                "(../notes/reproduction-statistics.md)): the mean share of its tokens a parent gave its child "
                "(`meanInvestedShare`); the tokens given to all children as a share of all tokens "
                "(`reproTokenShare`); the mean number of connections a child was born with (`meanChildLinks`); "
                f"and the connections handed over per birth (`handovers` / `births`). {BAND_WORDS}.",
        recipe=recipe(BASELINE_RUNS, STATS_FILE,
                      ["Take the rows with `phase` = 1: `meanInvestedShare`, `reproTokenShare`, "
                       "`meanChildLinks`, and `handovers` divided by `births`.", *BAND_STEPS]))
    for stat in ("meanInvestedShare", "reproTokenShare", "meanChildLinks"):
        ch.number(f"{stat}_settled", describe([mean_over(s.run_id, stat, 500, 2999, 1) for s in alive]))


# ---------------------------------------------------------------------------
# Chapter 23 · Power laws, real and apparent
# ---------------------------------------------------------------------------

def ccdf(values: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Every distinct value x, and the share of the values at least x."""
    v = np.sort(np.asarray([x for x in values if x > 0], float))
    xs, first = np.unique(v, return_index=True)
    return xs, 1.0 - first / v.size


def tail_pvalue(values: Sequence[float], reps: int, seed: int = 1) -> float:
    """
    How often a sample drawn from the fitted power law fits it worse than the data do.

    The semi-parametric bootstrap of Clauset, Shalizi and Newman (2009): keep the
    values below k_min as they are, draw the tail from the fitted law, fit each
    sample the same way, and count the samples whose Kolmogorov–Smirnov distance
    is at least the data's. A share of 0.1 or less rules the power law out.
    """
    import gol_series
    v = np.asarray([x for x in values if x > 0], float)
    fit = gol_series._scale_free(list(v))
    k_min, gamma = fit["kMin"], fit["exponent"]
    body, share = v[v < k_min], float(np.mean(v >= k_min))
    rng = np.random.default_rng(seed)
    worse = 0
    for _ in range(reps):
        m = rng.binomial(v.size, share)
        tail = np.floor((k_min - 0.5) * (1 - rng.random(m)) ** (-1 / (gamma - 1)) + 0.5)
        sample = np.concatenate([rng.choice(body, v.size - m) if body.size else np.empty(0), tail])
        again = gol_series._scale_free(list(sample))
        worse += again is not None and again["ks"] >= fit["ks"]
    return worse / reps


def lognormal_ccdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    from math import erf, log, sqrt
    return np.array([0.5 * (1 - erf((log(v) - mu) / (sigma * sqrt(2)))) for v in x])


@chapter
def powerlaws(ch: F.Chapter) -> None:
    import gol_series
    specs = baseline()
    alive = survivors(specs)

    # One world's degrees, the two ways of fitting them.
    w = F.world_pass("B1-10000-s001")
    degrees = np.array(w["last"]["degrees"], float)
    xs, ys = ccdf(degrees)
    mle = gol_series._scale_free(list(degrees))
    tail = mle["coverage"]
    kfit = np.geomspace(mle["kMin"], xs.max(), 40)
    model = tail * ((kfit - 0.5) / (mle["kMin"] - 0.5)) ** (1 - mle["exponent"])
    slope, intercept = np.polyfit(np.log(xs), np.log(ys), 1)
    ch.figure(
        "degree-fit", title="Fitting the degrees of one world",
        x={"label": "connections k (logarithmic)", "log": True, "min": 1},
        y={"label": "share of agents with at least k (logarithmic)", "log": True, "max": 1},
        series=[{"label": "the world of seed 1 after its last game", "x": xs.tolist(), "y": ys.tolist(),
                 "points": True, "size": 6, "colour": BLUE},
                line(xs, np.exp(intercept) * xs ** slope, f"a line through all of it: exponent {1 - slope:.2f}",
                     GREY, width=1.5, dash=[5, 4]),
                line(kfit, model, f"maximum likelihood from k = {mle['kMin']:.0f}: exponent {mle['exponent']:.2f}",
                     RED, width=2)],
        caption=f"Dots: for every number of connections k that occurs among the {len(degrees):,} agents of the "
                "world with seed 1 after its last game, the share of agents with at least k — the "
                "complementary cumulative distribution, on logarithmic axes. Dashed: the straight line "
                "least squares fits through all the dots, whose slope is 1 − γ for a power law of exponent γ "
                "(the viewer's `degreeExponent`). Red: the power law fitted by maximum likelihood to the "
                f"tail from the k that fits best, k = {mle['kMin']:.0f} (`degreeGamma`, `degreeKMin`), which "
                f"holds {100 * tail:.1f}% of the agents ([Fitting a power law](../notes/power-law-fit.md)).",
        recipe=recipe("`B1-10000-s001`.", FRAMES,
                      ["Read the last frame after a game (`5999`) and count every agent's connections in `edges`.",
                       "Dots: for each distinct count k, the share of agents with at least k.",
                       "Dashed: least squares of ln(share) on ln(k) over the dots.",
                       "Red: `gol_series._scale_free(degrees)` — for each candidate k_min, γ = 1 + n / Σ ln(k / "
                       "(k_min − ½)) over the n agents with k ≥ k_min, and the Kolmogorov–Smirnov distance; the "
                       "k_min with the smallest distance wins."]))
    ch.number("degree_fit_seed1", {"mle": mle, "regression_exponent": float(1 - slope),
                                   "p": tail_pvalue(degrees, 200)})
    p_worlds = [tail_pvalue(F.world_pass(s.run_id)["last"]["degrees"], 100) for s in alive]
    ch.number("degree_p_per_world", {"values": p_worlds, "ruled_out": int(sum(p <= 0.1 for p in p_worlds))})

    groups = [[mean_over(s.run_id, st, 500, 2999) for s in alive]
              for st in ("degreeGamma", "degreeExponent", "tokenExponent")]
    groups2 = [[mean_over(s.run_id, st, 500, 2999) for s in alive]
               for st in ("degreeTailShare", "degreeGammaKS")]
    c1 = ["degrees, maximum likelihood", "degrees, least squares", "tokens, least squares"]
    c2 = ["share of agents in the fitted tail", "Kolmogorov–Smirnov distance"]
    ch.grid(
        "exponents", [
            dict(title="Exponents γ", x={"label": "", "categories": c1}, y={"label": "γ", "min": 1, "max": 4.5},
                 series=dots(c1, groups, [RED, GREY, YELLOW]), legend=False),
            dict(title="How much the tail is, and how well it fits", x={"label": "", "categories": c2},
                 y={"label": "share or distance", "min": 0, "max": 0.3},
                 series=dots(c2, groups2, [RED, RED]), legend=False)],
        columns=2, title="Power-law exponents of the 26 worlds",
        caption="One dot per world that lived to the end, at its mean over the measurements (every 25 "
                "iterations) from iteration 500 on; the bar is the median. Left: the exponent γ of the degree "
                "distribution by maximum likelihood on the tail (`degreeGamma`) and by least squares on the "
                "whole distribution (`degreeExponent`), and of the token distribution by least squares "
                "(`tokenExponent`). Right: the share of agents in the tail the maximum-likelihood fit chose "
                "(`degreeTailShare`) and the largest gap between that tail and the fitted law "
                "(`degreeGammaKS`).",
        recipe=recipe(SURVIVORS, STATS_FILE,
                      ["Average `degreeGamma`, `degreeExponent`, `tokenExponent`, `degreeTailShare` and "
                       "`degreeGammaKS` over the rows with `phase` = 2 and 500 ≤ `iteration` ≤ 2,999.",
                       "One dot per run, a bar at the median."]))
    for st, g in zip(("degreeGamma", "degreeExponent", "tokenExponent", "degreeTailShare", "degreeGammaKS"),
                     groups + groups2):
        ch.number(f"{st}_settled", describe(g))
    for st in ("degreeGammaR2", "degreeExponentR2", "tokenExponentR2", "degreeKMin"):
        ch.number(f"{st}_settled", describe([mean_over(s.run_id, st, 500, 2999) for s in alive]))

    # The tokens: a power law, or a log-normal?
    tokens = np.concatenate([np.array(F.world_pass(s.run_id)["last"]["tokens"], float) for s in alive])
    xs, ys = ccdf(tokens)
    fit = gol_series._scale_free(list(tokens))
    logs = np.log(tokens)
    mu, sigma = float(logs.mean()), float(logs.std())
    kfit = np.geomspace(fit["kMin"], xs.max(), 40)
    ch.figure(
        "tokens-fit", title="Two candidate laws for the tokens",
        x={"label": "tokens x (logarithmic)", "log": True, "min": 1},
        y={"label": "share of agents with at least x (logarithmic)", "log": True, "max": 1, "min": 1e-5},
        series=[{"label": f"{len(tokens):,} agents of 26 worlds, after the last game", "x": xs.tolist(),
                 "y": ys.tolist(), "points": True, "size": 5, "colour": YELLOW},
                line(kfit, fit["coverage"] * ((kfit - 0.5) / (fit["kMin"] - 0.5)) ** (1 - fit["exponent"]),
                     f"power law from x = {fit['kMin']:.0f}, exponent {fit['exponent']:.2f}", RED, width=2),
                line(xs, lognormal_ccdf(xs, mu, sigma), f"log-normal: ln x has mean {mu:.2f}, sd {sigma:.2f}",
                     CYAN, width=2, dash=[6, 4])],
        caption="Dots: for every token count x that occurs among the agents alive after the last game of the 26 "
                "worlds that lived to the end, the share holding at least x. Red: a power law fitted by maximum "
                "likelihood to the tail, from the x that fits it best. Cyan: a log-normal distribution with the "
                "mean and standard deviation of ln x over all agents — a distribution whose logarithm is "
                "normal. Both on logarithmic axes.",
        recipe=recipe(SURVIVORS, FRAMES,
                      ["Pool the `tokens` of the last frame of every run.",
                       "Dots: for each distinct x, the share of agents with at least x.",
                       "Red: `gol_series._scale_free(tokens)`, as for the degrees.",
                       "Cyan: μ and σ = the mean and standard deviation of ln x; the curve is 1 − Φ((ln x − μ)/σ), "
                       "Φ the standard normal distribution."]))
    degrees_all = np.concatenate([np.array(F.world_pass(s.run_id)["last"]["degrees"], float) for s in alive])
    rich = tokens >= fit["kMin"]
    ch.number("tokens_fit", {"power": fit, "lognormal": {"mu": mu, "sigma": sigma}, "agents": int(tokens.size),
                             "p": tail_pvalue(tokens, 100),
                             "tail_degrees": {"median": float(np.median(degrees_all[rich])),
                                              "share_1_2": float(np.mean(degrees_all[rich] <= 2)),
                                              "share_10_up": float(np.mean(degrees_all[rich] >= 10))}})

    # Avalanches: how many agents a game cuts off at once.
    cut = np.concatenate([[r["orphaned"] for r in settled_rows(s.run_id)] for s in alive]).astype(float)
    starved = np.concatenate([[r["starved"] for r in settled_rows(s.run_id)] for s in alive]).astype(float)
    cx, cy = ccdf(cut)
    sx, sy = ccdf(starved)
    ch.figure(
        "avalanches", title="How many agents one game removes",
        x={"label": "agents removed in one game, n (logarithmic)", "log": True, "min": 1},
        y={"label": "share of games removing at least n (logarithmic)", "log": True, "max": 1},
        series=[line(cx, cy * np.mean(cut > 0), "cut off", ORANGE, width=2),
                line(sx, sy * np.mean(starved > 0), "starved", RED, width=2)],
        caption="Of all games from iteration 500 on in the 26 worlds that lived to the end "
                f"({len(cut):,} games), the share that cut off at least n agents (orange) and the share in "
                "which at least n agents starved (red), for every n, on logarithmic axes. Games in which no "
                "one was removed count in the denominator, which is why the curves start below 1.",
        recipe=recipe(SURVIVORS, STATS_FILE,
                      ["Take `orphaned` and `starved` of every row with `phase` = 2 and 500 ≤ `iteration` ≤ 2,999.",
                       "For each n that occurs, the share of rows with at least n."]))
    ch.number("avalanches", {"games": int(len(cut)), "share_with_cut": float(np.mean(cut > 0)),
                             "max_cut": float(cut.max()), "share_cut_ge_10": float(np.mean(cut >= 10)),
                             "share_cut_ge_100": float(np.mean(cut >= 100)), "mean_cut": float(cut.mean()),
                             "deaths_in_cuts_ge_100": float(cut[cut >= 100].sum() / cut.sum()),
                             "deaths_in_cuts_ge_10": float(cut[cut >= 10].sum() / cut.sum()),
                             "max_starved": float(starved.max())})

    # The rhythm of the wander: a power spectrum.
    spectra = []
    for s in alive:
        its, n = series(s.run_id, "nodes")
        x = n[(its >= 500) & (its <= 2999)]
        x = x - np.polyval(np.polyfit(np.arange(x.size), x, 1), np.arange(x.size))
        power = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
        spectra.append(power / power[1:].sum())
    power = np.mean(spectra, axis=0)[1:]
    freq = np.arange(1, power.size + 1) / 2500
    # Averaged over bands of equal width on the log axis, so the line is legible.
    edges_ = np.geomspace(freq[0], freq[-1] * 1.0001, 40)
    fx, fy = [], []
    for a, b in zip(edges_[:-1], edges_[1:]):
        m = (freq >= a) & (freq < b)
        if m.any():
            fx.append(float(np.exp(np.log(freq[m]).mean())))
            fy.append(float(power[m].mean()))
    fx, fy = np.array(fx), np.array(fy)
    fitm = (fx >= 1 / 500) & (fx <= 1 / 4)
    b1, a1 = np.polyfit(np.log(fx[fitm]), np.log(fy[fitm]), 1)
    ch.figure(
        "spectrum", title="The rhythm of the wander",
        x={"label": "frequency, cycles per iteration (logarithmic)", "log": True},
        y={"label": "power (logarithmic)", "log": True},
        series=[line(fx, fy, "number of agents, 26 worlds", BLUE, width=2),
                line(fx[fitm], np.exp(a1) * fx[fitm] ** b1, f"a power law, slope {b1:.2f}", RED, width=1.5,
                     dash=[5, 4])],
        caption="The power spectrum of the number of agents ([The power spectrum](../notes/power-spectrum.md)): "
                "for each frequency, how much of a world's up-and-down motion happens at that rhythm — slow "
                "wanders on the left, iteration-to-iteration jitter on the right. Each world's series from "
                "iteration 500 to 2,999, with its straight-line trend removed and a Hann window applied, "
                "normalised to total power 1 and averaged over the 26 worlds; then averaged in 39 bands of equal "
                "width on the logarithmic axis. Red: the straight line least squares fits between periods of "
                "500 and 4 iterations.",
        recipe=recipe(SURVIVORS, STATS_FILE,
                      ["For each run, take `nodes` of the rows with `phase` = 2 and 500 ≤ `iteration` ≤ 2,999 "
                       "(2,500 values); subtract the least-squares straight line.",
                       "Multiply by a Hann window and take |FFT|², the power at frequencies k/2,500, "
                       "k = 1 … 1,250; divide by the total.",
                       "Average over runs; average within bands of equal width in log frequency; fit a line on "
                       "log–log axes between frequencies 1/500 and 1/4."]))
    ch.number("spectrum", {"slope": float(b1)})
