#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Which worlds are spaces, and what they pay for it (Meta XII).

Every world read at iteration 1,000 (atlas/at1000.json, research/atlas/at.py) whose
space was measured then (atlas/space_at1000.json, research/atlas/space.py --at 1000),
each counted once: a world is a space here when its typical distance is at least half
again its random twin's (B2's own worlds read about 1.2, B1's 2.3). The spaces are
sorted by the rule that sets them apart, in this order: deaths not mended (B1, and
worlds built on it), every unused link lapsing, the estate paid to births, no old age,
handover, anything else; for each, the medians of the share of agents with three tokens
or fewer, the marks of the health bar and the dimension. The worlds of no such rule are
listed one by one.

    python3 research/atlas/census.py
"""
from __future__ import annotations

import importlib.util
import json
import os
from collections import Counter
from typing import Any, Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ATLAS = os.path.join(ROOT, "GraphOfLifeRuns", ".pilots", "atlas")
#: Half again as wide as its random twin: between B2's worlds (about 1.2) and B1's (2.3).
SPACE = 1.5
KINDS = ["births", "no old age", "handover", "lapse", "not mended", "other"]


def _load(name: str, path: str):
    """A module by its file (several of the atlas's scripts share names elsewhere)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def kind(settings: Dict[str, Any]) -> str:
    """The rule that sets a world apart from B2's way of keeping space, first match wins."""
    if settings.get("mend_on_death") is False:
        return "not mended"
    if settings.get("prune_needs") == 0:
        return "lapse"
    if settings.get("redistribution") in ("newborns", "parents"):
        return "births"
    if not settings.get("max_age"):
        return "no old age"
    if settings.get("allow_handover"):
        return "handover"
    return "other"


def worlds() -> List[Dict[str, Any]]:
    """Every world read at 1,000 with its space measured, each once (some batches share a world under two names)."""
    space = _load("space_atlas", os.path.join(HERE, "space.py"))
    profile = _load("profile", os.path.join(HERE, "profile.py"))
    ref = profile.reference()
    at = {w["id"]: w for w in json.load(open(os.path.join(ATLAS, "at1000.json")))["worlds"]}
    seen, out = set(), []
    for row in json.load(open(os.path.join(ATLAS, "space_at1000.json"))):
        w = at.get(row["id"])
        if w is None:
            continue
        try:
            wide = space.typical_distance(row)
        except (KeyError, ValueError, ZeroDivisionError):
            continue
        if not np.isfinite(wide) or (w["agents"], round(wide, 6)) in seen:
            continue
        seen.add((w["agents"], round(wide, 6)))
        out.append({"id": row["id"], "wide": wide, "kind": kind(space.settings(row["id"])), "agents": w["agents"],
                    "poor": w["share_at_most_three"], "marks": w["marks"], "dimension": w["dimension"],
                    "diversity": profile.diversity(w, ref)[1]})
    return out


def main() -> None:
    every = worlds()
    spaces = [w for w in every if w["wide"] >= SPACE]
    print(f"{len(every)} worlds, {len(spaces)} of them spaces: {dict(Counter(w['kind'] for w in spaces))}")
    for k in KINDS:
        mine = [w for w in spaces if w["kind"] == k]
        if mine:
            print(f"  {k:11s} {len(mine):3d}  poor {np.median([w['poor'] for w in mine]):.2f}  "
                  f"marks {np.median([w['marks'] for w in mine]):.1f}  dimension {np.median([w['dimension'] for w in mine]):.2f}  "
                  f"fewer than half poor: {sum(w['poor'] < 0.5 for w in mine)}")
    print("of no such rule:")
    for w in sorted((w for w in spaces if w["kind"] == "other"), key=lambda w: -w["wide"]):
        print(f"  {w['id']:26s} {w['wide']:.2f} times its twin's, {w['agents']} agents, {w['poor']:.0%} poor, "
              f"{w['marks']} marks, dimension {w['dimension']:.2f}, diversity {w['diversity']:.2f}")


if __name__ == "__main__":
    main()
