# -*- coding: utf-8 -*-
"""
Appendix B · What goes with what: correlations among the statistics of agents
and of whole worlds.

The numbers are computed by research/atlas/correlations.py (agents) and
research/atlas/world_series.py (worlds) and kept in GraphOfLifeRuns/.pilots/atlas;
this module draws them.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

import book_figures as F
from book_figures import chapter, recipe
from book_chapters.common import BLUE, GREY, ORANGE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATLAS = os.path.join(ROOT, "GraphOfLifeRuns", ".pilots", "atlas")
B2 = [f"B2-10000-s{i:03d}" for i in range(1, 11)]
B1 = [f"B1-10000-s{i:03d}" for i in range(1, 11)]
COLOURS = {"B2": BLUE, "B1": ORANGE}

#: The quantities known before the game, and what happened after it.
BEFORE = ["tokens", "age", "clones", "degree", "curvature", "nbr_tokens", "nbr_max", "poorer_nbrs", "nbr_degree",
          "kin_nbrs", "tokens_per_link", "clustering", "triangles", "core", "forman", "pagerank", "betweenness",
          "home_share", "targets", "max_share", "rev_share"]
AFTER = ["change", "rel_change", "kept", "took", "alive_10", "change_10", "degree_change_10", "child_next"]
#: The outcomes as the heatmap's columns name them.
AFTER_SHORT = ["change", "change ÷ tokens", "kept", "took", "alive in 10", "change in 10", "links change in 10",
               "child next"]

RUNS = ("The ten B2 worlds of Experiment 16, `B2-10000-s001` … `-s010`, and the first ten worlds of B1's "
        "baseline set, `B1-10000-s001` … `-s010` (Experiment 2): 10,000 tokens each, every game recorded "
        "with every agent's stakes.")
EXTRACT = ("For every 20th game from iteration 1,000 to 2,000, one row per agent alive at the start of the "
           "game, from the frames after the reproduction phase and after the game, of the next iteration's "
           "reproduction and of ten iterations later: the 29 quantities of the table in the text "
           "(`python3 research/atlas/correlations.py extract <run> --from 1000 --to 2000 --step 20`).")


def _load(name: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(ATLAS, name)
    return json.load(open(path)) if os.path.exists(path) else None


def _lookup(corr: Dict[str, Any]) -> Dict[tuple, Dict[str, Any]]:
    out = {}
    for p in corr["pairs"]:
        out[(p["a"], p["b"])] = p
        out[(p["b"], p["a"])] = p
    return out


def _value(entry: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if not entry or key not in entry:
        return None
    return entry[key]["median"]


def best(entry: Dict[str, Any]) -> tuple:
    """The axes (linear or logarithmic) on which a pair correlates most, and that correlation."""
    keys = [k for k in ("lin-lin", "log-lin", "lin-log", "log-log") if k in entry]
    if not keys:
        return None, None
    k = max(keys, key=lambda k: abs(entry[k]["median"]))
    return k, entry[k]["median"]


#: Classes of token curvature: whole numbers, so a class edge sits between two of them.
CURVATURE_EDGES = [-100.5, -9.5, -2.5, -0.5, 0.5, 3.5, 10.5, 100.5]
CURVATURE_LABELS = ["≤ −101", "−100 to −10", "−9 to −3", "−2 to −1", "0", "1 to 3", "4 to 10", "11 to 100", "> 100"]


def _table(run: str):
    z = np.load(os.path.join(ATLAS, f"{run}.npz"))
    return z["table"], {c: i for i, c in enumerate(z["columns"])}


def curvature_profile(runs: List[str]) -> Dict[str, Any]:
    """Per world and curvature class: mean relative change, share kept; and the heat-equation slope."""
    rel = np.full((len(runs), len(CURVATURE_LABELS)), np.nan)
    kept = np.full_like(rel, np.nan)
    net = np.full_like(rel, np.nan)
    slopes, r_all, r_kept = [], [], []
    for w, run in enumerate(runs):
        t, c = _table(run)
        k, d, x, ch, kp = (t[:, c[n]] for n in ("curvature", "degree", "tokens", "change", "kept"))
        ok = np.isfinite(k) & np.isfinite(ch) & (x > 0)
        cls = np.digitize(k, CURVATURE_EDGES)
        for j in range(len(CURVATURE_LABELS)):
            m = ok & (cls == j)
            if m.sum() >= 50:
                rel[w, j] = float(np.mean(ch[m] / x[m]))
                kept[w, j] = float(np.mean(kp[m]))
                net[w, j] = float(np.mean(ch[m] > 0) - np.mean(ch[m] < 0))
        pred = k / (d + 1)
        slopes.append(float((pred[ok] * ch[ok]).sum() / (pred[ok] ** 2).sum()))
        r_all.append(float(np.corrcoef(pred[ok], ch[ok])[0, 1]))
        kk = ok & (kp == 1)
        r_kept.append(float(np.corrcoef(pred[kk], ch[kk])[0, 1]))
    return {"relative": rel, "kept": kept, "net": net, "slope": slopes, "r": r_all, "r_kept": r_kept}


def world_components(label: str = "E10") -> Optional[Dict[str, Any]]:
    """Principal components of the worlds' mean statistics, each standardised: the share of their
    variance each component holds, the statistics that load on it most, and each world's score."""
    import sys
    sys.path.insert(0, os.path.join(ROOT, "research", "atlas"))
    from groups import GROUP_OF
    d = _load(f"world_{label}.json")
    if not d:
        return None
    runs = d["runs"]
    names = sorted(k for k in set.intersection(*(set(m) for m in d["means"].values())) if k in GROUP_OF)
    X = np.array([[d["means"][r][k] for k in names] for r in runs])
    keep = X.std(0) > 0
    X, names = X[:, keep], [n for n, k in zip(names, keep) if k]
    Z = (X - X.mean(0)) / X.std(0)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    share = S ** 2 / (S ** 2).sum()
    score = U[:, 0] * S[0]
    hub = X[:, names.index("maxDegree")] / X[:, names.index("nodes")]
    if np.corrcoef(score, hub)[0, 1] < 0:
        score, Vt = -score, -Vt
    return {"runs": runs, "share": share.tolist(), "score": score.tolist(), "hub": hub.tolist(),
            "statistics": len(names),
            "loadings": [[(names[i], float(Vt[c][i])) for i in np.argsort(-np.abs(Vt[c]))[:8]] for c in range(3)]}


@chapter
def atlas(ch: F.Chapter) -> None:
    corr = {label: _load(f"correlations_{label}.json") for label in ("B2", "B1")}
    if not all(corr.values()):
        return
    look = {label: _lookup(c) for label, c in corr.items()}

    # --- Before and after: rank correlations of every earlier quantity with every outcome ----
    charts = []
    for label in ("B2", "B1"):
        rects = []
        for i, a in enumerate(BEFORE):
            for j, b in enumerate(AFTER):
                v = _value(look[label].get((a, b)), "rank")
                if v is not None:
                    rects.append((j - 0.5, j + 0.5, i - 0.5, i + 0.5, v))
        charts.append({"title": label, "x": {"categories": AFTER_SHORT}, "y": {"categories": BEFORE[::-1]},
                       "series": [F.cells([(x0, x1, len(BEFORE) - 1 - y1, len(BEFORE) - 1 - y0, v)
                                           for x0, x1, y0, y1, v in rects])],
                       "colourbar": {"map": "signed", "min": -1, "max": 1, "label": "rank correlation"},
                       "height": 460, "legend": False})
    ch.grid("before_after", charts, title="What comes before, and what follows",
            caption=("Each cell is the rank correlation (Spearman's) between a quantity known at the start of "
                     "a game (rows) and what then happened (columns), the median over ten worlds. Red: the "
                     "more of the one, the more of the other; blue: the more of the one, the less of the "
                     "other. The quantities are those of the table in the text."),
            recipe=recipe(RUNS, "the runs' recorded frames", [
                EXTRACT,
                "In each world, rank every column (ties share their mean rank) and take Pearson's "
                "correlation of the ranks for every pair (`python3 research/atlas/correlations.py analyse "
                "--runs <runs> --label <B2|B1>`).",
                "Take the median over the ten worlds."]),
            columns=1)

    # --- Which relations the logarithm reveals ---------------------------------------------
    charts = []
    for label in ("B2", "B1"):
        dots = {"label": None, "x": [], "y": [], "points": True, "size": 4, "colour": COLOURS[label], "alpha": 0.7}
        for p in corr[label]["pairs"]:
            if p.get("definitional") or "lin-lin" not in p:
                continue
            k, r = best(p)
            dots["x"].append(abs(p["lin-lin"]["median"]))
            dots["y"].append(abs(r))
        charts.append({"title": label, "x": {"label": "|r| on linear axes", "min": 0, "max": 1},
                       "y": {"label": "|r| on the best axes", "min": 0, "max": 1},
                       "series": [{"label": None, "x": [0, 1], "y": [0, 1], "colour": GREY, "width": 1,
                                   "dash": [4, 4]}, dots], "legend": False})
    ch.grid("logarithm", charts, title="What the logarithm reveals",
            caption=("One dot per pair of quantities that are not made from the same numbers. Across, Pearson's "
                     "correlation on linear axes; up, on whichever axes give the strongest: linear or "
                     "logarithmic on either. Dots far above the dashed diagonal are relations a linear plot "
                     "hides: power laws, and quantities whose few large values swamp the rest."),
            recipe=recipe(RUNS, "the runs' recorded frames", [
                EXTRACT,
                "For every pair, Pearson's r with each quantity on a linear axis or a logarithmic one: the "
                "logarithm of a positive quantity, log(1 + x) of a count, sign(x) log(1 + |x|) of a signed "
                "one; shares and yes-or-no stay linear. The median over ten worlds for each choice of axes.",
                "Leave out the pairs made from the same numbers (tokens and tokens per link, say)."]),
            columns=2)
    ch.number("pairs", {label: len(c["pairs"]) for label, c in corr.items()})

    # --- Token curvature and token change ----------------------------------------------------
    profiles = {"B2": curvature_profile(B2), "B1": curvature_profile(B1)}
    charts = []
    for what, ylabel, extra in (("net", "share that gained − share that lost", {"min": -0.6, "max": 0.6}),
                                ("kept", "share that kept their node", {"min": 0, "max": 1})):
        series = []
        for k, label in enumerate(("B2", "B1")):
            table = profiles[label][what]
            rng = np.random.RandomState(3 + k)
            for j in range(len(CURVATURE_LABELS)):
                v = [x for x in table[:, j] if np.isfinite(x)]
                if not v:
                    continue
                x0 = j + (k - 0.5) * 0.32
                series.append({"label": label if j == 0 else None,
                               "x": (x0 + rng.uniform(-0.07, 0.07, len(v))).tolist(), "y": v,
                               "points": True, "size": 4, "colour": COLOURS[label], "alpha": 0.8})
                series.append({"label": None, "x": [x0 - 0.12, x0 + 0.12], "y": [float(np.median(v))] * 2,
                               "colour": COLOURS[label], "width": 3})
        charts.append({"title": "Tokens over the game" if what == "net" else "Node kept",
                       "x": {"label": "token curvature at the start of the game", "categories": CURVATURE_LABELS},
                       "y": {"label": ylabel, **extra}, "series": series,
                       "guides": [{"axis": "y", "at": 0, "label": ""}] if what == "net" else []})
    ch.grid("curvature", charts, title="Tokens run downhill",
            caption=("One dot per world, the bar the median of ten. Token curvature is the sum, over an agent's "
                     "neighbours, of their tokens minus its own (the note *Token curvature*): below zero an "
                     "agent is a peak, above zero a valley. Left: of the agents in a class, the share whose "
                     "node held more tokens after the game than before, minus the share whose node held "
                     "fewer; most hold the same. Right: the share of agents that kept their own node. A "
                     "class with fewer than 50 agents in a world has no dot for it."),
            recipe=recipe(RUNS, "the runs' recorded frames", [
                EXTRACT,
                "Sort the rows by curvature into the classes on the axis; in each, count the agents whose "
                "node gained tokens over the game and those whose node lost, and those that kept their own "
                "node."]),
            columns=2)
    ch.number("curvature", {label: {"slope": F.describe(p["slope"]), "r": F.describe(p["r"]),
                                    "r_kept": F.describe(p["r_kept"]),
                                    "relative": [float(np.nanmedian(p["relative"][:, j])) for j in range(len(CURVATURE_LABELS))],
                                    "kept": [float(np.nanmedian(p["kept"][:, j])) for j in range(len(CURVATURE_LABELS))],
                                    "net": [float(np.nanmedian(p["net"][:, j])) for j in range(len(CURVATURE_LABELS))]}
                            for label, p in profiles.items()})

    # --- Between worlds: one number ---------------------------------------------------------
    comp = world_components("E10")
    if comp:
        ch.figure("worlds", title="What tells thirty worlds apart",
                  x={"label": "the best-connected agent's share of the world (logarithmic)", "log": True,
                     "min": 0.005, "max": 1},
                  y={"label": "score on the first component"},
                  series=[{"label": None, "x": comp["hub"], "y": comp["score"], "points": True, "size": 6,
                           "colour": ORANGE, "alpha": 0.9}],
                  caption=(f"One dot per world of Experiment 10 (B1 at 100,000 tokens). Each world is described "
                           f"by the means of {comp['statistics']} statistics over iterations 500 to 1,500, each "
                           f"standardised over the thirty worlds; the first principal component of those "
                           f"descriptions holds {comp['share'][0]:.0%} of their variance. Across, how many of the "
                           f"other agents the best-connected agent is linked to."),
                  recipe=recipe("The thirty runs of Experiment 10, `B1-100000-s001` … `-s030`.",
                                "each run's statistics after every game (`stats.jsonl`)", [
                      "Average every statistic a run records after the game over iterations 500 to 1,500 "
                      "(`python3 research/atlas/world_series.py --runs <runs> --label E10 --from 500 --to 1500`).",
                      "Standardise each over the thirty worlds and take the principal components (a singular "
                      "value decomposition of the worlds-by-statistics table).",
                      "The best-connected agent's share: its connections divided by the agents, both "
                      "averaged over the same iterations."]))
        ch.number("world_components", {k: comp[k] for k in ("share", "loadings", "statistics")}
                  | {"r_score_hub": float(np.corrcoef(np.log(comp["hub"]), comp["score"])[0, 1])})
