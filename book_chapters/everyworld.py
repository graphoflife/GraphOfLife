# -*- coding: utf-8 -*-
"""
One statistic, every world: the pages that take a quantity and measure it in
every simulation made so far — lab runs of every experiment and pilot worlds.

The numbers come from research/atlas/across.py (book/results/worlds.json: every
world's statistics over the same stretches of iterations), from the runs'
records, and from the checks of research/oee/checks and research/atlas, which
keep their results in GraphOfLifeRuns/.pilots. This module draws them.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import book_figures as F
from book_figures import chapter, recipe
from book_chapters.common import BLUE, CYAN, GREEN, GREY, ORANGE, RED, VIOLET, YELLOW
from gol_atlas import EXTENSIVE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILOTS = os.path.join(ROOT, "GraphOfLifeRuns", ".pilots")
WORLDS = os.path.join(ROOT, "book", "results", "worlds.json")

TABLE_TEXT = ("every simulation made so far, as `book/results/worlds.json` holds it "
              "(`python3 research/atlas/across.py lab`, `pilots`, `write`): a lab run's statistics are the "
              "means of its record over a stretch of iterations; a pilot's, of 20 games recorded from its "
              "checkpoint and summarised the same way")


#: Settings that change how a world is kept, not what happens in it.
BOOKKEEPING = {"extinction_threshold", "archive_every", "checkpoint_every"}
#: Prototype switches the engine has since taken in as settings: a pilot run with
#: one is the engine's world with that setting, draw for draw (equiv.py checks it).
ENGINE_NOW = {"mix": {"mix_at_birth": True}}


def _settings(cfg: Dict[str, Any]) -> Optional[str]:
    """A world's settings in full, bookkeeping aside, as one string to compare; None if they cannot be read."""
    from gol_config import SimConfig
    try:
        full = SimConfig.from_dict(cfg).to_dict()
    except (ValueError, TypeError, KeyError):
        return None
    return json.dumps({k: v for k, v in full.items() if k not in BOOKKEEPING}, sort_keys=True)


def replicas() -> Dict[str, str]:
    """
    The pilot worlds that replay a lab world: the same settings and seed, and no
    prototype rule beyond the engine's own game. Such a pilot is that lab world
    again, game for game (a continuation from a lab archive is exact), so a
    measure of both counts one world twice. Pilot id ("L/B2-long-s001") -> lab run.
    """
    runs = os.path.join(ROOT, "GraphOfLifeRuns")
    lab: Dict[str, str] = {}
    for d in sorted(os.listdir(runs)):
        meta = os.path.join(runs, d, "meta.json")
        if os.path.exists(meta) and os.path.exists(os.path.join(runs, d, "stats.jsonl")):
            key = _settings(json.load(open(meta)).get("config") or {})
            if key:
                lab.setdefault(key, d)
    out = {}
    for batch in sorted(os.listdir(PILOTS)):
        if not os.path.isdir(os.path.join(PILOTS, batch)):
            continue
        for name in sorted(os.listdir(os.path.join(PILOTS, batch))):
            path = os.path.join(PILOTS, batch, name, "config.json")
            if not os.path.exists(path):
                continue
            meta = json.load(open(path))
            proto, cfg = dict(meta.get("proto") or {}), dict(meta.get("cfg") or {})
            for flag, settings in ENGINE_NOW.items():
                if proto.pop(flag, False):
                    cfg.update(settings)
            if proto != {"engine": True}:
                continue
            key = _settings(cfg)
            if key in lab:
                out[f"{batch}/{name}"] = lab[key]
    return out


def _worlds() -> List[Dict[str, Any]]:
    return json.load(open(WORLDS))["worlds"] if os.path.exists(WORLDS) else []


def _at(world: Dict[str, Any], stat: str, *windows: str) -> Optional[float]:
    """A world's value of a statistic over the first of `windows` it has."""
    for window in windows:
        v = (world.get("at") or {}).get(window, {}).get(stat)
        if v is not None and np.isfinite(v):
            return float(v)
    return None


def _has(world: Dict[str, Any], condition: str) -> bool:
    return any(c.endswith(condition) or c == condition for c in world.get("conditions") or [])


#: The groups of worlds the "every world" figures put side by side, each a short
#: label, a test on a world's table entry and a colour. A world falls in the first
#: group whose test it passes. GROUP_WORDS says in full what each label stands for.
GROUPS: List[Tuple[str, Any, int]] = [
    ("B1 10k", lambda w: w["source"] == "lab" and _has(w, "E02 · baseline"), ORANGE),
    ("B1 fixed", lambda w: _has(w, "E07 · brains never change"), YELLOW),
    ("B1 100k", lambda w: _has(w, "E10 · baseline"), RED),
    ("B1 sizes", lambda w: w["source"] == "lab" and w.get("baseline") == "B1"
     and (w["family"].startswith("B1-5000") or _has(w, "E08 · baseline") or _has(w, "E08 · stopped only when empty")),
     GREY),
    ("B2 10k", lambda w: _has(w, "E16 · B2"), BLUE),
    ("B2 var", lambda w: w["source"] == "lab" and w["family"].startswith("B2-10000"), CYAN),
    ("lapse", lambda w: _has(w, "E14 · any unused link lapses"), GREEN),
    ("B2 5k", lambda w: w["source"] == "lab" and w["family"].startswith("B2-5000"), VIOLET),
    # A pilot that replays a lab world (replica_of) is already counted as that world.
    ("B2 pil.", lambda w: w["source"] == "pilot" and w.get("baseline") == "B2" and w.get("engine")
     and not w.get("replica_of"), BLUE),
    ("B1 pil.", lambda w: w["source"] == "pilot" and w.get("baseline") == "B1" and w.get("engine")
     and not w.get("replica_of"), ORANGE),
]


GROUP_WORDS = ("B1 10k: the thirty B1 worlds of 10,000 tokens (Experiment 2); B1 fixed: the same with brains "
               "that never change (Experiment 7); B1 100k: B1 at 100,000 tokens (Experiment 10); B1 sizes: B1 at "
               "800 to 409,600 tokens (Experiment 8) and at 5,000 with and without a hidden layer "
               "(Experiments 12 and 13); B2 10k: the ten B2 worlds "
               "of Experiment 16; B2 var: B2 with one rule or rate changed (Experiments 9, 15 and 16); B2 5k: "
               "B2 at 5,000 tokens, with all its rules or with one taken away (Experiment 14) — except "
               "lapse: the condition in which any unused link lapses; B1 pil. and B2 pil.: the pilot worlds that play the engine's game, "
               "except those that replay a lab world with its settings and seed.")


def _grouped(worlds: List[Dict[str, Any]], stat: str, *windows: str) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {label: [] for label, _, _ in GROUPS}
    for w in worlds:
        for label, test, _ in GROUPS:
            if test(w):
                v = _at(w, stat, *windows)
                if v is not None:
                    out[label].append(v)
                break
    return {k: v for k, v in out.items() if v}


def _dots(title: str, ylabel: str, groups: Dict[str, List[float]], **y: Any) -> Dict[str, Any]:
    colour = {label: c for label, _, c in GROUPS}
    cats = list(groups)
    return {"title": title, "x": {"categories": cats}, "y": {"label": ylabel, **y},
            "series": F.dots(cats, [groups[c] for c in cats], [colour[c] for c in cats])}


def _median(values: List[float]) -> Optional[float]:
    v = [x for x in values if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


# ---------------------------------------------------------------------------
# Lightning
# ---------------------------------------------------------------------------

#: The B1 worlds of every size: Experiment 8's sizes, the 5,000 and 10,000 baselines and Experiment 10.
SIZES = [("E08 · baseline", "500"), ("E12 · baseline", "1000"), ("E02 · baseline", "1000"),
         ("E10 · baseline", "1000")]
B1_LIFE = [f"B1-10000-s{i:03d}" for i in range(1, 31)]
B2_LIFE = [f"B2-10000-s{i:03d}" for i in range(1, 11)]


def _fit(x: List[float], y: List[float]) -> Tuple[float, float]:
    """The slope and intercept of log y against log x, by least squares."""
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    slope, intercept = np.polyfit(lx, ly, 1)
    return float(slope), float(intercept)


@chapter
def lightning(ch: F.Chapter) -> None:
    worlds = _worlds()

    # Every world, side by side.
    charts = []
    for stat, title, label, kw in (
            ("cyclingShare", "Flow that goes round", "share of staked flow in loops (greedy, before cancelling)", {"min": 0}),
            ("netFlowShare", "Flow that is left after cancelling", "net flow ÷ staked flow", {"min": 0, "max": 1}),
            ("lightningLongest", "The longest loop", "agents on it (greedy, before cancelling)", {"min": 0}),
            ("netCyclingShare", "Net flow that goes round", "share of net flow in loops", {"min": 0})):
        groups = _grouped(worlds, stat, "1000", "500")
        charts.append(_dots(title, label, groups, **kw))
        for g, values in groups.items():
            ch.number(f"{stat}:{g}", _median(values))
    ch.grid("every_world", charts, columns=2,
            title="Loops of flow in every world",
            caption=("Each dot is one world, over iterations 900–999 (400–499 for a world that did not live "
                     "to 1,000), a column per kind of world; the bar is the median. Top left: how much of "
                     "what agents stake on their neighbours a greedy search can put on closed loops, before "
                     "a token one way and a token back cancel. Top right: what is left once they cancel. "
                     "Bottom: the longest loop found, and how much of the net flow goes round. " + GROUP_WORDS),
            recipe=recipe(f"{TABLE_TEXT}.", "each world's `cyclingShare`, `netFlowShare`, `lightningLongest` and "
                          "`netCyclingShare` (gol_lightning.py, on the heavy rows of a run's record)",
                          ["Take each world's mean over iterations 900–999, or 400–499 if it has no later one.",
                           "Put it in the first group of the list in book_chapters/everyworld.py it belongs to.",
                           "Draw one dot per world; the bar is each group's median."]))

    # The longest loop against the size of the world.
    sizes: Dict[int, List[float]] = {}
    net: Dict[int, List[float]] = {}
    for w in worlds:
        for condition, window in SIZES:
            if _has(w, condition) and w.get("baseline") == "B1":
                v, n = _at(w, "lightningLongest", window), _at(w, "netLightningLongest", window)
                if v:
                    sizes.setdefault(int(w["tokens"]), []).append(v)
                if n:
                    net.setdefault(int(w["tokens"]), []).append(n)
    tokens = sorted(sizes)
    med = [float(np.median(sizes[t])) for t in tokens]
    slope, intercept = _fit(tokens, med)
    ch.number("longest_exponent", round(slope, 3))
    ch.number("longest_at", {str(t): m for t, m in zip(tokens, med)})
    fit_x = [tokens[0], tokens[-1]]
    series = [{"label": "every world", "x": [t for t in tokens for _ in sizes[t]],
               "y": [v for t in tokens for v in sizes[t]], "points": True, "size": 5, "colour": ORANGE,
               "alpha": 0.6},
              F.line(tokens, med, "median at each size", ORANGE, width=2),
              F.line(fit_x, [float(np.exp(intercept) * t ** slope) for t in fit_x],
                     f"tokens^{slope:.2f}", GREY, dash=[5, 4])]
    net_tokens = sorted(t for t in net if net[t])
    if net_tokens:
        series.append(F.line(net_tokens, [float(np.median(net[t])) for t in net_tokens],
                             "after cancelling (median)", BLUE, width=2))
    ch.figure("size", title="The longest loop grows with the world",
              x={"label": "tokens in the world", "log": True}, y={"label": "agents on the longest loop", "log": True},
              series=series,
              caption=(f"The longest loop the greedy search finds in a B1 world, against the world's size, both "
                       f"axes logarithmic: the median grows as the tokens to the power {slope:.2f}, so a world "
                       f"twice the size has loops {2 ** slope:.2f} times as long. The blue line is the same after "
                       f"a token one way and a token back have cancelled."),
              recipe=recipe("The B1 worlds of Experiment 8 (800 to 409,600 tokens, three seeds each, over "
                            "iterations 400–499), of Experiments 12 and 2 (5,000 and 10,000 tokens) and of "
                            "Experiment 10 (100,000 tokens), over iterations 900–999.",
                            "each world's mean `lightningLongest` and `netLightningLongest` in "
                            "book/results/worlds.json",
                            ["Take each world's mean of the longest loop over its stretch.",
                             "Take the median at each size; fit a straight line to the logarithms."]))

    # Over a world's life.
    charts = []
    for stat, title, label in (("cyclingShare", "Flow that goes round", "share of staked flow in loops"),
                               ("netFlowShare", "Flow left after cancelling", "net ÷ staked flow")):
        bands = []
        for runs, name, colour in ((B1_LIFE, "B1", ORANGE), (B2_LIFE, "B2", BLUE)):
            data = [F.series(r, stat) for r in runs]
            data = [(its, v) for its, v in data if len(its)]
            if data:
                bands.append(F.band_series(data, name, colour))
        charts.append({"title": title, "x": {"label": "iteration"}, "y": {"label": label, "min": 0},
                       "series": bands})
    ch.grid("life", charts, columns=2, title="Through a world's life",
            caption=("The two statistics through the life of thirty B1 worlds (Experiment 2) and ten B2 "
                     "worlds (Experiment 16): the median at each recorded iteration, with the middle half "
                     "of the worlds as a band."),
            recipe=recipe("The thirty B1 worlds of Experiment 2 and the ten B2 worlds of Experiment 16, "
                          "10,000 tokens each.", "`stats.jsonl`, the heavy rows (every 25th iteration)",
                          ["Read each world's statistic at every heavy row.",
                           "Draw the median of the worlds and their middle half at each iteration."]))

    # The greedy count against the exact one.
    hodge = {r["run"]: r for r in json.load(open(os.path.join(PILOTS, "circulation_20.json")))} \
        if os.path.exists(os.path.join(PILOTS, "circulation_20.json")) else {}
    series = []
    pairs = []
    for runs, name, colour in ((B1_LIFE[:10], "B1", ORANGE), (B2_LIFE, "B2", BLUE)):
        xs, ys = [], []
        for r in runs:
            h = (hodge.get(r) or {}).get("all", {}).get("circulation_share")
            g = F.mean_over(r, "cyclingShare", 1000, 2000)
            if h is not None and g is not None:
                xs.append(h)
                ys.append(g)
                pairs.append((h, g))
        if xs:
            series.append({"label": name, "x": xs, "y": ys, "points": True, "size": 7, "colour": colour,
                           "alpha": 0.85})
    if pairs:
        ch.number("hodge_vs_greedy_r", round(float(np.corrcoef(*zip(*pairs))[0, 1]), 3))
    ch.figure("exact", title="A greedy count and an exact one",
              x={"label": "circulating share of the net flow over 50 games (Hodge)", "min": 0},
              y={"label": "flow in loops, greedy, per game", "min": 0}, series=series,
              caption=("Each dot is a world: across, the exact share of the flow over 50 games that circulates "
                       "(the Helmholtz–Hodge decomposition of [Gliders, and who helps whom](gliders-and-who-helps-whom.md)); up, the greedy loop count of a "
                       "single game, averaged over the same thousand iterations."),
              recipe=recipe("The ten B2 worlds of Experiment 16 and the first ten B1 worlds of Experiment 2, "
                            "iterations 1,000 to 2,000.",
                            "`GraphOfLifeRuns/.pilots/circulation_20.json` (research/oee/checks/circulation.py) "
                            "and each world's `cyclingShare`",
                            ["Take each world's circulating share, the median over its twenty windows of 50 games.",
                             "Take its mean greedy loop share over the same iterations.",
                             "Draw one against the other."]))


# ---------------------------------------------------------------------------
# Every world side by side (Every world side by side)
# ---------------------------------------------------------------------------

#: The counts of agents among the statistics that grow with a world, which the
#: comparison divides by the agents; the rest of gol_atlas.EXTENSIVE is left out.
PER_AGENT = {"births", "gainers", "losers", "starved", "orphaned", "revolutions", "leaves", "handovers", "gifts"}
WINDOW = "500"


def _intensive(world: Dict[str, Any], stat: str) -> Optional[float]:
    at = (world.get("at") or {}).get(WINDOW, {})
    v = at.get(stat)
    if v is None or not np.isfinite(v):
        return None
    if stat in PER_AGENT:
        return v / at["nodes"] if at.get("nodes") else None
    return None if stat in EXTENSIVE else float(v)


def comparable(worlds: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str], np.ndarray]:
    """The lab worlds over iterations 400–499 and the statistics that do not grow with a world, standardised."""
    lab = [w for w in worlds if w["source"] == "lab" and WINDOW in (w.get("at") or {})]
    stats = sorted({k for w in lab for k in w["at"][WINDOW]})
    X = np.array([[_intensive(w, k) if _intensive(w, k) is not None else np.nan for k in stats] for w in lab], float)
    keep = (np.isfinite(X).mean(axis=0) > 0.9) & (np.nanstd(X, axis=0) > 0)
    X, stats = X[:, keep], [k for k, ok in zip(stats, keep) if ok]
    X = np.where(np.isfinite(X), X, np.nanmedian(X, axis=0))
    return lab, stats, (X - X.mean(axis=0)) / X.std(axis=0)


def _family(world: Dict[str, Any]) -> str:
    """A world's condition: the family of its run ids, the replicates of Experiment 1 together."""
    return "E01" if any(c.startswith("E01") for c in world.get("conditions") or []) else world["family"]


def _eta2(values: np.ndarray, labels: List[str]) -> float:
    total = float(((values - values.mean()) ** 2).sum())
    if total <= 0:
        return float("nan")
    labels = np.array(labels)
    return float(sum(((values[labels == g].mean() - values.mean()) ** 2) * (labels == g).sum()
                     for g in set(labels.tolist())) / total)


@chapter
def side_by_side(ch: F.Chapter) -> None:
    worlds = _worlds()
    lab, stats, Z = comparable(worlds)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    share = S ** 2 / (S ** 2).sum()
    ch.number("worlds", len(lab))
    ch.number("statistics", len(stats))
    ch.number("component_shares", [round(float(x), 3) for x in share[:5]])
    for k in range(3):
        order = np.argsort(-np.abs(Vt[k]))[:10]
        ch.number(f"component_{k + 1}", [[stats[i], round(float(Vt[k][i]), 3)] for i in order])
    # Each component turned so that B1 sits to the right.
    b1 = np.array([w.get("baseline") == "B1" for w in lab])
    scores = U * S
    for k in range(2):
        if scores[b1, k].mean() < scores[~b1, k].mean():
            scores[:, k] *= -1
    series = []
    for label, test, colour in GROUPS:
        idx = [i for i, w in enumerate(lab) if next((g for g, t, _ in GROUPS if t(w)), None) == label]
        if idx:
            series.append({"label": label, "x": scores[idx, 0].tolist(), "y": scores[idx, 1].tolist(),
                           "points": True, "size": 5, "colour": colour, "alpha": 0.75})
    ch.figure("components", title="Every lab world on its first two components",
              x={"label": f"first component ({share[0] * 100:.0f}% of the differences)"},
              y={"label": f"second component ({share[1] * 100:.0f}%)"}, series=series,
              caption=(f"Each dot is one of the {len(lab)} lab worlds that lived to iteration 500, placed by the "
                       f"two directions along which the {len(stats)} statistics that do not grow with a world "
                       f"differ most (principal components of their standardised means over iterations "
                       f"400–499). " + GROUP_WORDS),
              recipe=recipe(f"{TABLE_TEXT}; the lab worlds that lived to iteration 500.",
                            "each world's means over iterations 400–499",
                            ["Keep the statistics that do not grow with the size of a world; divide the counts "
                             "of agents (births, deaths, revolutions, leaves) by the agents.",
                             "Fill a missing value with the median of the worlds; standardise each statistic.",
                             "Take the principal components (a singular value decomposition) and draw every "
                             "world on the first two."]))

    # How much of each statistic's spread the settings decide.
    labels = [_family(w) for w in lab]
    raw = np.array([[_intensive(w, k) if _intensive(w, k) is not None else np.nan for k in stats] for w in lab], float)
    decided = []
    for j, k in enumerate(stats):
        v = raw[:, j]
        ok = np.isfinite(v)
        e = _eta2(v[ok], [labels[i] for i in np.flatnonzero(ok)])
        if np.isfinite(e):
            decided.append((k, round(e, 3)))
    decided.sort(key=lambda r: -r[1])
    ch.number("decided_by_settings", decided)
    ch.number("families", len(set(labels)))

    # One statistic the settings decide, one the seed does.
    charts = []
    for stat, title, label in (("spreadShare", "Decided by the settings", "share of agents that spread their stake"),
                               ("births", "Decided by the seed", "births per agent and iteration")):
        groups: Dict[str, List[float]] = {}
        for w in lab:
            g = next((g for g, t, _ in GROUPS if t(w)), None)
            v = _intensive(w, stat)
            if g and v is not None:
                groups.setdefault(g, []).append(v)
        charts.append(_dots(title, label, {g: groups[g] for g, _, _ in GROUPS if g in groups}, min=0))
    ch.grid("decided", charts, columns=2, title="What the settings decide, and what the seed does",
            caption=("Left: the share of agents that spread their stake over several nodes rather than going "
                     "all in — 99% of its differences between worlds lie between conditions. Right: births per "
                     "agent — under a third do; worlds of one condition differ from each other as much as the "
                     "conditions do. Each dot is a world, over iterations 400–499. " + GROUP_WORDS),
            recipe=recipe(f"{TABLE_TEXT}.", "each world's `spreadShare` and `births` ÷ `nodes`",
                          ["Take each world's mean over iterations 400–499.",
                           "Group the worlds as in the other figures; draw a dot per world and the median."]))

