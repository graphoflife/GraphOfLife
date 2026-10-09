#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The book's figures, made from the runs.

    python3 book_figures.py                 every chapter's figures
    python3 book_figures.py life youth      the figures of these chapters

Every figure is drawn to book/figures/<chapter>/<name>.svg (book_svg.py), and
two texts are kept for under it, in book/results/<chapter>.json:

    caption   what is visible: what every line, band, bar and dot is
    recipe    how to make it again by hand: which runs, which file, which
              statistic, how it is averaged, and the command that redoes it

beside the numbers the chapter quotes, so the text can be checked against
them. Then book_fill.py writes each figure, its caption and its recipe into
the chapters that ask for it.

Everything is read from the runs folder (GraphOfLifeRuns/, or GOL_RUNS_DIR):
the runs' `stats.jsonl` — one row of statistics per recorded phase — and,
where a figure needs more than a statistic, their frames. Numpy and networkx
only, and every random choice here is seeded, so the same runs always give
the same figures.
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

import book_svg
import gol_analysis as A
import gol_lab
import gol_record
import gol_store as store
from gol_series import NO_PARENT

BOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "book")
COMMAND = "python3 book_figures.py"

#: The runs most of Part II reads, and how the recipes name them.
BASELINE_RUNS = ("The 30 baseline runs `B1-10000-s001` … `B1-10000-s030`: the baseline B1 at "
                 "10,000 tokens, seeds 1 to 30, 3,000 iterations each (every setting is listed in "
                 "Chapter 9). They are made with `python3 gol_lab.py run E02`.")
STATS_FILE = ("each run's `GraphOfLifeRuns/<run>/stats.jsonl`, which holds one row per "
              "recorded phase: `phase` 1 is the world just after reproduction, `phase` 2 just "
              "after the game, and every statistic is a field of the row (see [What a run records]"
              "(../notes/frames-and-stats.md))")
FRAMES = ("each run's frames, `GraphOfLifeRuns/<run>/frames/`, read with "
          "`gol_store.read_frame(run, index)`; frame `2·t` is iteration *t* after "
          "reproduction and frame `2·t + 1` after the game")


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

class Chapter:
    """The figures and numbers of one chapter, written as they are made."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.numbers: Dict[str, Any] = {}
        self.figures: Dict[str, Dict[str, str]] = {}

    def grid(self, name: str, charts: List[Dict[str, Any]], *, title: str, caption: str,
             recipe: str, columns: int = 2) -> None:
        """
        A figure of several charts side by side, drawn to
        book/figures/<chapter>/<name>.svg; its title, caption and recipe are
        kept for the chapter's text (book_fill.py puts them under it).
        """
        path = os.path.join(BOOK, "figures", self.name, f"{name}.svg")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(book_svg.render(charts, columns, f"{self.name}-{name}"))
        self.figures[name] = {"title": title, "caption": caption,
                              "recipe": recipe + f"\n\n**To make it again:** `{COMMAND} {self.name}`."}

    def figure(self, name: str, *, caption: str, recipe: str, **chart: Any) -> None:
        """One chart: its title, axes x and y, series and the rest (book_svg.py)."""
        self.grid(name, [chart], title=chart.get("title", name), caption=caption, recipe=recipe,
                  columns=1)

    def network(self, name: str, *, caption: str, recipe: str, **chart: Any) -> None:
        """A picture of a world: nodes {x, y, group}, edges, colour {labels}."""
        self.figure(name, caption=caption, recipe=recipe, kind="network", **chart)

    def number(self, key: str, value: Any) -> Any:
        self.numbers[key] = value
        return value

    def save(self) -> None:
        A._write(os.path.join(BOOK, "results", f"{self.name}.json"),
                 {"chapter": self.name, "numbers": self.numbers, "figures": self.figures})
        # A figure the chapter no longer draws would otherwise linger beside the ones it does.
        folder = os.path.join(BOOK, "figures", self.name)
        for name in os.listdir(folder) if os.path.isdir(folder) else []:
            if name.endswith(".svg") and name[:-4] not in self.figures:
                os.remove(os.path.join(folder, name))


def recipe(runs: str, data: str, steps: Iterable[str]) -> str:
    """How a figure is made, in the same three parts every time."""
    return (f"**Runs.** {runs}\n\n**Data.** From {data}.\n\n"
            + "\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1)))


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

_ROWS: Dict[str, List[Dict[str, Any]]] = {}


def rows(run_id: str) -> List[Dict[str, Any]]:
    if run_id not in _ROWS:
        _ROWS[run_id] = gol_record.read_stats(run_id)
    return _ROWS[run_id]


def runs_of(experiment: str, condition: Optional[str] = None) -> List[gol_lab.RunSpec]:
    return [s for s in gol_lab.experiment_runs(experiment)
            if condition is None or s.condition == condition]


def lived(spec: gol_lab.RunSpec) -> int:
    return store.load_meta(spec.run_id).get("iteration", 0)


def survivors(specs: List[gol_lab.RunSpec]) -> List[gol_lab.RunSpec]:
    return [s for s in specs if lived(s) >= s.until]


def series(run_id: str, stat: str, phase: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    return A.series(rows(run_id), stat, phase)


def mean_over(run_id: str, stat: str, start: int, stop: int, phase: int = 2) -> Optional[float]:
    """A run's mean of a statistic over iterations start..stop, both included."""
    its, values = series(run_id, stat, phase)
    return A.window(its, values, start, stop)


def per_agent(run_id: str, stat: str, phase: int, per: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    A count as a rate: the statistic of each row of `phase`, divided by the
    agents present when that phase began (the row's `nodes_before`), times
    `per`. Births per hundred agents, say, are the births of a reproduction
    phase per hundred agents that faced it.
    """
    its, values = series(run_id, stat, phase)
    _, before = series(run_id, "nodes_before", phase)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(before > 0, values / before * per, np.nan)
    return its, rate


def band_series(runs: List[Tuple[np.ndarray, np.ndarray]], label: str, colour: int,
                points: int = A.FIGURE_POINTS) -> Dict[str, Any]:
    band = A.bands(runs, points)
    band.update(label=label, colour=colour)
    return band


def line(x: Iterable[float], y: Iterable[float], label: Optional[str], colour: int,
         **extra: Any) -> Dict[str, Any]:
    return {"label": label, "x": [float(v) for v in x],
            "y": [None if v is None or not np.isfinite(v) else float(v) for v in y],
            "colour": colour, **extra}


def thin(its: np.ndarray, values: np.ndarray, every: int) -> Tuple[np.ndarray, np.ndarray]:
    """Means over consecutive stretches of `every` iterations, so a 3,000-point line stays legible."""
    if every <= 1:
        return its, values
    edges = np.arange(its.min(), its.max() + every, every)
    which = np.digitize(its, edges) - 1
    xs, ys = [], []
    for b in range(len(edges) - 1):
        inside = values[(which == b) & np.isfinite(values)]
        if inside.size:
            xs.append((edges[b] + edges[b + 1] - 1) / 2)
            ys.append(inside.mean())
    return np.array(xs), np.array(ys)


def jitter(k: int, n: int, seed: int, width: float = 0.28) -> np.ndarray:
    """Positions spread across a category, so dots of equal value stay apart."""
    rng = np.random.default_rng(seed)
    return k + rng.uniform(-width, width, n)


def dots(categories: List[str], groups: List[List[float]], colour_of: List[int],
         seed: int = 7) -> List[Dict[str, Any]]:
    """
    One dot per value, a column per category, with a short bar at the median.
    The dots are spread sideways at random only so that they do not hide each
    other; their sideways position means nothing.
    """
    out = []
    for k, (name, values) in enumerate(zip(categories, groups)):
        v = np.array([x for x in values if x is not None and np.isfinite(x)], dtype=float)
        if not v.size:
            continue
        out.append({"label": None, "x": jitter(k, v.size, seed + k).tolist(), "y": v.tolist(),
                    "points": True, "size": 6, "colour": colour_of[k], "alpha": 0.8})
        med = float(np.median(v))
        out.append({"label": None, "x": [k - 0.38, k + 0.38], "y": [med, med],
                    "colour": colour_of[k], "width": 3})
    return out


def describe(values: Iterable[float]) -> Dict[str, Any]:
    v = np.array([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if not v.size:
        return {"n": 0}
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "sd": float(v.std(ddof=1)) if v.size > 1 else None,
            "min": float(v.min()), "max": float(v.max()),
            "q25": float(np.quantile(v, 0.25)), "q75": float(np.quantile(v, 0.75))}


# ---------------------------------------------------------------------------
# One pass over a world's frames
# ---------------------------------------------------------------------------

#: From which iteration on a life counts as one of the settled world's (Chapter 10).
SETTLED = 500
#: Iterations between two looks at the common ancestor of the living.
ANCESTRY_EVERY = 25


def world_pass(run_id: str, muller: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
    """
    Everything the frame-based figures need from one world, in one reading of
    its frames, kept in a cache beside the runs (`<runs>/.book/<run>.json`) so
    that it is read once. The cache is keyed by the run's last iteration and
    the Muller window asked for; anything else asked of it is recomputed.

      agents     every agent born in a reproduction phase at iteration
                 b ≥ SETTLED that took part in at least one game: (b, d,
                 alive) — d the last iteration after whose game it was alive,
                 alive whether it still was at the end of the run
      genotypes  the same for genotypes: first and last iteration after a game
                 in which some agent carried it
      top        after every game, the share of agents carrying the most
                 common genotype
      age        after every game, the median age of the living
      ancestry   every ANCESTRY_EVERY iterations, how many iterations back all,
                 nine in ten and half of the living share one ancestor
                 (None while they descend from more than one founder)
      founders   after every game up to iteration 300, how many agents descend
                 from each founder (the founders whose line ever held 2% or more)
      muller     for (t0, t1): after every game from t0 to t1, how many agents
                 descend from each genotype alive at t0 (those that ever held 2%)
      last       at the last game frame: tokens, degrees and ages of the living
      stakes     at iteration 0 and at the last iteration, the share of its
                 own tokens each agent staked on its own node
      shares     in reproduction phases at iteration 0 and in the last 50
                 iterations, the share of its tokens each parent gave its child
    """
    cache_dir = os.path.join(store.BASE_DIR, ".book")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{run_id}.json")
    key = {"iteration": store.load_meta(run_id).get("iteration"), "muller": list(muller) if muller else None,
           "version": 3}
    if os.path.exists(path):
        with open(path) as f:
            cached = json.load(f)
        have = cached.get("key", {})
        # A pass made with a Muller window answers a question asked without one.
        if (have.get("iteration"), have.get("version")) == (key["iteration"], key["version"]) \
                and (muller is None or have.get("muller") == key["muller"]):
            return cached

    parent: Dict[int, int] = {}
    born_g: Dict[int, int] = {}
    first_g: Dict[int, int] = {}
    last_g: Dict[int, int] = {}
    birth: Dict[int, int] = {}
    last_a: Dict[int, int] = {}
    top, age, ancestry, founders_t, muller_t = [], [], [], [], []
    founder_of: Dict[int, int] = {}
    family_of: Dict[int, int] = {}
    alive_at_t0: set = set()
    stakes: Dict[str, List[float]] = {}
    shares: Dict[str, List[float]] = {"first": [], "last": []}
    end = None
    last_frame = None
    count = store.count_frames(run_id)
    last_iteration = (count - 1) // 2

    def climb(g: int, memo: Dict[int, int], stop) -> int:
        path, node = [], g
        while node not in memo:
            if stop(node):
                memo[node] = node
                break
            up = parent.get(node, NO_PARENT)
            if up == NO_PARENT or up not in born_g:
                memo[node] = node
                break
            path.append(node)
            node = up
        for step in path:
            memo[step] = memo[node]
        return memo[node]

    for index in range(count):
        frame = store.read_frame(run_id, index)
        it, phase = frame["iteration"], frame.get("phase")
        for b, p in zip(frame["brain_ids"], frame["parent_brain_ids"]):
            if b not in born_g:
                born_g[b], parent[b] = it, p
        decisions = frame.get("decisions") or {}
        if phase == 1:
            if it == 0 or it > last_iteration - 50:
                which = "first" if it == 0 else "last"
                shares[which] += [d["invested"] / d["tokens_before"]
                                  for d in decisions.get("births") or [] if d.get("tokens_before")]
            continue
        ids = frame["ids"]
        if not ids:
            continue
        end = it
        last_frame = frame
        counts = Counter(frame["brain_ids"])
        top.append([it, max(counts.values()) / len(ids)])
        known = [a for a in frame["ages"] if a >= 0]
        if known:
            age.append([it, float(np.median(known))])
        for a, g_age in zip(ids, frame["ages"]):
            if a not in birth:
                birth[a] = it - g_age if g_age >= 0 else it
            last_a[a] = it
        for g in counts:
            first_g.setdefault(g, it)
            last_g[g] = it
        if it % ANCESTRY_EVERY == 0:
            ancestry.append([it, *(None if g is None else it - born_g[g]
                                   for g in A.shared_ancestors(parent, counts))])
        if it <= 300:
            per = Counter()
            for g, c in counts.items():
                per[climb(g, founder_of, lambda node: parent.get(node, NO_PARENT) == NO_PARENT)] += c
            founders_t.append([it, len(ids), dict(per)])
        if muller and muller[0] <= it <= muller[1]:
            if it == muller[0]:
                alive_at_t0 = set(counts)
            per = Counter()
            for g, c in counts.items():
                per[climb(g, family_of, lambda node: node in alive_at_t0)] += c
            muller_t.append([it, len(ids), dict(per)])
        if it in (0, last_iteration):
            allocs = decisions.get("allocations") or []
            stakes["first" if it == 0 else "last"] = [
                r["alloc"][0] / r["tokens"] for r in allocs if r.get("tokens")]

    def keep_big(table):
        if not table:
            return {"lines": [], "series": []}
        peak: Dict[int, float] = {}
        for _, n, per in table:
            for g, c in per.items():
                peak[g] = max(peak.get(g, 0), c / n)
        big = sorted((g for g, v in peak.items() if v >= 0.02), key=lambda g: -peak[g])
        return {"lines": big,
                "series": [[t, n, [per.get(g, 0) for g in big]] for t, n, per in table]}

    agents = [[birth[a], last_a[a], last_a[a] == end] for a in birth if birth[a] >= SETTLED]
    genotypes = [[first_g[g], last_g[g], last_g[g] == end] for g in first_g if first_g[g] >= SETTLED]
    degree = Counter()
    for a, b in last_frame["edges"]:
        degree[a] += 1
        degree[b] += 1
    out = {"key": key, "end": end, "agents": agents, "genotypes": genotypes, "top": top,
           "age": age, "ancestry": ancestry, "founders": keep_big(founders_t),
           "muller": keep_big(muller_t),
           "last": {"tokens": last_frame["tokens"],
                    "degrees": [degree.get(a, 0) for a in last_frame["ids"]],
                    "ages": last_frame["ages"]},
           "stakes": stakes, "shares": shares}
    with open(path, "w") as f:
        json.dump(out, f)
    return out


#: Iterations between two looks of sample_pass, and between two looks at single agents.
SAMPLE_EVERY = 25
AGENT_EVERY = 100


def sample_pass(run_id: str) -> Dict[str, Any]:
    """
    What Part III needs from a world's frames, read every SAMPLE_EVERY
    iterations and kept beside the runs (`<runs>/.book/<run>.sample.npz`).

      frames   one row per look, after the game of iteration t (t = 0, 25, …):
               t, agents, genotypes, genotype entropy (bits), new genotypes in
               the reproduction phase and in the game, births, tokens staked
               (at home, on others), the share of directed flows that are
               returned, and for nodes of degree 1, 2, 3–4, 5–9, 10–49 and 50+
               how many were staked on and how many kept by their own agent
      agents   one row per agent alive at the start of the game, for every
               AGENT_EVERY iterations from SETTLED on: t, tokens and degree at
               the start of the game, its token curvature then, its change over
               the game (NaN if it did not survive it), whether it kept its node,
               the share it staked at home, its candidates and how many it
               staked on, and its age
      parents  one row per agent alive at the start of a reproduction phase, at
               iteration 0 and every AGENT_EVERY iterations from SETTLED on: t,
               tokens, degree, the tokens it gave a child (0 for none), the
               child's links and the connections handed over
    """
    path = os.path.join(store.BASE_DIR, ".book", f"{run_id}.sample.npz")
    stamp = store.load_meta(run_id).get("iteration")
    if os.path.exists(path):
        cached = np.load(path)
        if int(cached["stamp"]) == stamp:
            return {k: cached[k] for k in cached.files}
    bins = [(1, 1), (2, 2), (3, 4), (5, 9), (10, 49), (50, 10 ** 9)]
    frames, agents, parents = [], [], []
    last = (store.count_frames(run_id) - 1) // 2
    previous_game = None
    for t in range(0, last + 1, SAMPLE_EVERY):
        repro = store.read_frame(run_id, 2 * t)
        game = store.read_frame(run_id, 2 * t + 1)
        before = store.read_frame(run_id, 2 * t - 1) if t > 0 else None
        if not game["ids"]:
            break
        counts = Counter(game["brain_ids"])
        p = np.array(list(counts.values()), float) / len(game["ids"])
        entropy = float(-(p * np.log2(p)).sum())
        new_repro = len(set(repro["brain_ids"]) - set(before["brain_ids"])) if before else 0
        new_game = len(set(game["brain_ids"]) - set(repro["brain_ids"]))
        degree = Counter()
        for a, b in repro["edges"]:
            degree[a] += 1
            degree[b] += 1
        tokens0 = dict(zip(repro["ids"], repro["tokens"]))
        flows: Dict[Tuple[int, int], int] = {}
        home = others = 0
        allocs = {}
        for r in (game.get("decisions") or {}).get("allocations") or []:
            allocs[r["agent"]] = r
            home += r["alloc"][0]
            for target, amount in zip(r["targets"][1:], r["alloc"][1:]):
                if amount > 0:
                    flows[(r["agent"], target)] = amount
                    others += amount
        mutual = (sum(1 for a, b in flows if (b, a) in flows) / len(flows)) if flows else np.nan
        won = {w["node"]: w["winner"] for w in (game.get("decisions") or {}).get("winners") or []}
        kept_by = []
        for lo, hi in bins:
            nodes = [v for v in won if lo <= degree.get(v, 0) <= hi]
            kept_by += [len(nodes), sum(1 for v in nodes if won[v] == v)]
        births = (repro.get("decisions") or {}).get("births") or []
        frames.append([t, len(game["ids"]), len(counts), entropy, new_repro, new_game, len(births),
                       home, others, mutual, *kept_by])

        if t >= SETTLED and t % AGENT_EVERY == 0:
            curvature = Counter()
            for a, b in repro["edges"]:
                curvature[a] += tokens0[b] - tokens0[a]
                curvature[b] += tokens0[a] - tokens0[b]
            after = dict(zip(game["ids"], game["delta"]))
            ages = dict(zip(repro["ids"], repro["ages"]))
            for a in repro["ids"]:
                r = allocs.get(a)
                agents.append([t, tokens0[a], degree.get(a, 0), curvature.get(a, 0),
                               after.get(a, np.nan), won.get(a) == a,
                               (r["alloc"][0] / r["tokens"]) if r and r["tokens"] else np.nan,
                               len(r["targets"]) if r else 0,
                               sum(1 for x in r["alloc"] if x > 0) if r else 0, ages.get(a, -1)])
        if (t == 0 or (t >= SETTLED and t % AGENT_EVERY == 0)) and (before is not None or t == 0):
            start = before if before is not None else None
            if start is None:
                # The founders, all of them — also those that gave everything and starved, and so are
                # missing from the frame. Ids are handed out in order, so they are every id below the
                # first child's; each held an equal share of the tokens.
                founders = min((d["child"] for d in births), default=len(repro["ids"]))
                ids0 = list(range(founders))
                share = store.load_meta(run_id)["config"]["total_tokens"] // max(1, founders)
                start_tokens = {a: share for a in ids0}
                start_degree = Counter()
            else:
                ids0 = start["ids"]
                start_tokens = dict(zip(start["ids"], start["tokens"]))
                start_degree = Counter()
                for a, b in start["edges"]:
                    start_degree[a] += 1
                    start_degree[b] += 1
            by_parent = {d["agent"]: d for d in births}
            for a in ids0:
                d = by_parent.get(a)
                tokens = d["tokens_before"] if d else start_tokens.get(a, 0)
                parents.append([t, tokens, start_degree.get(a, 0) if start is not None else 4,
                                d["invested"] if d else 0, len(d["links"]) if d else 0,
                                len(d.get("handed_over") or []) if d else 0])
        previous_game = game
    out = {"stamp": np.array(stamp), "frames": np.array(frames, float),
           "agents": np.array(agents, float), "parents": np.array(parents, float)}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def kaplan_meier(lives: List[List[Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    The share of lives that last at least L iterations, for every L, from
    lives (start, last, still going): a life of start b and last d lasted
    d − b + 1 iterations; one still going at the end of the run is known only
    to have lasted at least that long, and is counted as at risk up to there
    and no further (Kaplan and Meier, 1958).
    """
    durations = np.array([d - b + 1 for b, d, _ in lives], dtype=float)
    ended = np.array([not alive for _, _, alive in lives], dtype=bool)
    order = np.argsort(durations)
    durations, ended = durations[order], ended[order]
    at, survival = [1.0], [1.0]
    s, n = 1.0, len(durations)
    i = 0
    while i < len(durations):
        L = durations[i]
        j = i
        deaths = 0
        while j < len(durations) and durations[j] == L:
            deaths += ended[j]
            j += 1
        if deaths:
            s *= 1 - deaths / n
        # Lasting at least L + 1 means surviving the step at L.
        at.append(L + 1)
        survival.append(s)
        n -= j - i
        i = j
    return np.array(at), np.array(survival)


# ---------------------------------------------------------------------------
# Chapters
# ---------------------------------------------------------------------------

CHAPTERS: Dict[str, Callable[[], None]] = {}


def chapter(fn: Callable[[Chapter], None]) -> Callable[[Chapter], None]:
    """Register a chapter's figures under its function's name."""
    CHAPTERS[fn.__name__] = lambda: _run(fn)
    return fn


def _run(fn: Callable[[Chapter], None]) -> None:
    ch = Chapter(fn.__name__)
    fn(ch)
    ch.save()
    print(f"{fn.__name__}: written")


def main(argv: List[str]) -> int:
    """Make the figures of the chapters named, or of every chapter."""
    names = argv or list(CHAPTERS)
    unknown = [n for n in names if n not in CHAPTERS]
    if unknown:
        print(f"no chapter called {', '.join(unknown)}; there are {', '.join(CHAPTERS)}",
              file=sys.stderr)
        return 2
    for name in names:
        CHAPTERS[name]()
    import book_fill
    print(f"{book_fill.fill()} Markdown files filled in")
    return 0


if __name__ == "__main__":
    # The chapters live in the book_chapters package and register themselves
    # with this module as it is imported there — under its own name, book_figures,
    # not as __main__, so it is that module's main that runs them.
    import book_chapters  # noqa: F401
    import book_figures
    sys.exit(book_figures.main(sys.argv[1:]))
