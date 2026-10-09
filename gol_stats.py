#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small numbers that several modules need to agree on.

A median, a description of a set of values, JSON with no NaN in it, and a
way to write that JSON whole: the lab, the analysis and the book each had a
copy of one or more of these, and a copy is a place for the two to start
disagreeing.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


def median(values: Iterable[Optional[float]]) -> Optional[float]:
    """
    The middle of the values that are there — the mean of the two in the
    middle when there is an even number of them — or None if none is.
    """
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def describe(values: Iterable[Optional[float]]) -> Dict[str, Any]:
    """Count, mean, median, spread and range of the finite values, or just {"n": 0}."""
    v = np.array([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if not v.size:
        return {"n": 0}
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "sd": float(v.std(ddof=1)) if v.size > 1 else None,
            "min": float(v.min()), "max": float(v.max()),
            "q25": float(np.quantile(v, 0.25)), "q75": float(np.quantile(v, 0.75))}


def finite(value: Any) -> Any:
    """`value` with every number that is not finite as None, and numpy's numbers as Python's."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return finite(value.item())
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    return value


def write_json(path: str, value: Any, compact: bool = False) -> None:
    """
    JSON any page can read — no NaN anywhere — written whole or not at all.
    Indented for people to read; compact for what is thousands of points for a
    page to draw.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part = f"{path}.{os.getpid()}.part"
    with open(part, "w") as f:
        json.dump(finite(value), f, allow_nan=False,
                  **({"separators": (",", ":")} if compact else {"indent": 1}))
        f.write("\n")
    os.replace(part, path)
